"""Custom dynamic attention for Gemma-4 packed training.

Gemma-4 mixes two attention layer types with *different* head dims:

  * FULL attention layers   -> head_dim 512  (FlashAttention-3 only supports <= 256,
                                               so these fall back to SDPA, whose math
                                               backend handles any head_dim)
  * SLIDING-window layers    -> head_dim 256  (handled by the FA3 varlen kernel)

`dynamic_attention` is registered with transformers' AttentionInterface and dispatches
per layer based on `query.shape[-1]`. It is kept dependency-light (torch only; FA3 is a
lazy import) so it can be unit-tested in isolation — see test_attention.py.
"""
import os
from typing import Optional

import torch
from torch.nn.functional import scaled_dot_product_attention  # sdpa
from torch.utils.checkpoint import checkpoint


# Query-block size for the head_dim-512 FULL-attention SDPA path. The math backend
# would otherwise materialize one O(H·S²) score tensor (78 GiB at 32k → OOM). Tiling
# the QUERY axis caps the live score to (H, SDPA_QUERY_BLOCK, S) and is *bit-exact*:
# each query row still softmaxes over the WHOLE key axis (an ordinary softmax — no
# online/streaming softmax, because the key axis is never tiled). 0 / >=S disables it
# (single SDPA call = legacy behaviour). Override per-run with env SDPA_QUERY_BLOCK.
SDPA_QUERY_BLOCK = int(os.environ.get("SDPA_QUERY_BLOCK", "2048"))


def block_diagonal_concat(*masks, dtype=torch.bfloat16):
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)
    current_pos = 0
    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size
    return combined_mask.unsqueeze(0)


def _doc_ids_from_cu(cu_seq_lens, seq_total, device):
    """Per-token document id from cu_seqlens, e.g. cu=[0,3,7] -> [0,0,0,1,1,1,1]."""
    cu = cu_seq_lens.to(device=device, dtype=torch.long)
    seq_len = cu[1:] - cu[:-1]
    doc_ids = torch.repeat_interleave(torch.arange(seq_len.numel(), device=device), seq_len)
    if doc_ids.numel() != seq_total:  # defensive: cu must cover the whole packed sequence
        raise ValueError(f"cu_seqlens cover {doc_ids.numel()} tokens but sequence is {seq_total}")
    return doc_ids


def _block_causal_mask(doc_ids, q_start, q_end, seq_total, device):
    """Boolean (1, q_end-q_start, S) keep-mask: True where a query attends a key.

    A query at global row i attends key column j iff (same document) AND (j <= i):
    per-document causal. This is the slice of block_diagonal_concat(tril blocks) for
    rows [q_start:q_end] — built directly so the full (S, S) mask is never allocated.
    """
    rows = torch.arange(q_start, q_end, device=device)        # (Bq,)  global query indices
    cols = torch.arange(seq_total, device=device)             # (S,)   key indices
    causal = cols[None, :] <= rows[:, None]                   # (Bq, S)  j <= i
    same_doc = doc_ids[q_start:q_end][:, None] == doc_ids[None, :]  # (Bq, S)
    return (causal & same_doc).unsqueeze(0)                   # (1, Bq, S) bool, True = attend


def _packed_sdpa_full(query, key, value, cu_seq_lens_q, scaling, enable_gqa, block=None):
    """Per-document causal FULL attention via SDPA, tiled over the QUERY axis.

    Mathematically identical to a single `scaled_dot_product_attention(q, k, v, mask)`
    call, but processes `block` queries at a time so the live score tensor is
    (H, block, S) instead of (H, S, S). Each block is wrapped in activation
    checkpointing during training so the BACKWARD recomputes one block's score at a
    time (peak ~ one block) rather than retaining every block's score (= O(S²) again).
    head_dim 512 has no flash kernel (FA3 caps at 256; FlexAttention's Triton kernel
    exceeds the 227 KB SRAM budget at 512), so this is the memory-bounded fallback.
    """
    B, H, S, D = query.shape
    if block is None:
        block = SDPA_QUERY_BLOCK
    doc_ids = _doc_ids_from_cu(cu_seq_lens_q, S, query.device)

    def _attend(q_blk, k, v, mask_blk):
        return scaled_dot_product_attention(
            q_blk, k, v, attn_mask=mask_blk, scale=scaling, enable_gqa=enable_gqa,
        )

    if block <= 0 or block >= S:  # single call (short sequences / blocking disabled)
        mask = _block_causal_mask(doc_ids, 0, S, S, query.device)
        return _attend(query, key, value, mask)

    use_ckpt = torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    )
    outs = []
    for q_start in range(0, S, block):
        q_end = min(q_start + block, S)
        q_blk = query[:, :, q_start:q_end, :]
        mask_blk = _block_causal_mask(doc_ids, q_start, q_end, S, query.device)
        if use_ckpt:
            o = checkpoint(_attend, q_blk, key, value, mask_blk, use_reentrant=False)
        else:
            o = _attend(q_blk, key, value, mask_blk)
        outs.append(o)
    return torch.cat(outs, dim=2)  # (B, H, S, D)


def dynamic_attention(
    module: torch.nn.Module,  # required
    query: torch.Tensor,  # required
    key: torch.Tensor,  # required
    value: torch.Tensor,  # required
    attention_mask: Optional[torch.Tensor],  # required
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_length_q=None,
    max_length_k=None,
    sliding_window=None,
    scaling=None,   # transformers passes the attn scale as `scaling` (gemma-4: self.scaling=1.0,
                    # since q_norm/k_norm absorb the 1/sqrt(d) factor). MUST be named `scaling` —
                    # naming it `scale` silently drops it into **kwargs and SDPA/FA3 fall back to
                    # 1/sqrt(head_dim), corrupting the attention (verified via compare_logits.py).
    **kwargs
):
    enable_gqa = False
    if query.shape[1] != key.shape[1]:
        enable_gqa = True

    if cu_seq_lens_q is None:
        raise ValueError(
            "dynamic_attention requires cu_seq_lens_q (packed varlen metadata). "
            "It was None — transformers did not propagate the FlashAttentionKwargs. "
            "Pass cu_seq_lens_q/cu_seq_lens_k/max_length_q/max_length_k through the model call."
        )

    # full attention head_dim is 512 (for gemma4 model), sliding-window layers use 256.
    # FlashAttention-3 only supports head_dim <= 256, so the 512-dim FULL-attention layers
    # fall back to SDPA (math backend handles head_dim > 256); the 256-dim SLIDING-window
    # layers go through the FA3 varlen path.
    fa3_supported = query.shape[-1] <= 256
    if not fa3_supported:  # fallback to sdpa, full attention
        # q.shape = (B, num_head, S, head_dim)
        if sliding_window is not None:
            raise ValueError("Expecting sliding window is None for full attention, but got sliding_window = {}".format(sliding_window))
        # Per-document causal attention via SDPA, tiled over the query axis so the math
        # backend never materializes the full O(H·S²) score (78 GiB at 32k -> OOM). The
        # mask is rebuilt per query-block from cu_seqlens: each packed document is a
        # lower-triangular block, off-block entries stay masked so attention never crosses
        # a document boundary (causality + no cross-contamination).
        # IMPORTANT: SDPA treats a *float* attn_mask as an ADDITIVE bias and a *bool* mask
        # as keep/drop. A 0/1 float mask masks NOTHING — the mask MUST be boolean (True =
        # attend). Tiling is bit-exact and uses no is_causal (causality is in the mask;
        # SDPA forbids is_causal + explicit mask and has no cu_seqlens concept).
        attn_output = _packed_sdpa_full(query, key, value, cu_seq_lens_q, scaling, enable_gqa)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, S, num_head, head_dim)
    else:
        from flash_attn_interface import flash_attn_varlen_func  # fa3 (lazy import: SDPA-only runs need no FA3)

        # FA3 varlen wants unbatched (total_tokens, num_head, head_dim) and int32 cu_seqlens on-device.
        query = query.permute(0, 2, 1, 3).squeeze(0).contiguous()
        key = key.permute(0, 2, 1, 3).squeeze(0).contiguous()
        value = value.permute(0, 2, 1, 3).squeeze(0).contiguous()
        cu_seq_lens_q = cu_seq_lens_q.to(device=query.device, dtype=torch.int32)
        cu_seq_lens_k = cu_seq_lens_k.to(device=query.device, dtype=torch.int32)
        # window_size (-1, -1) is non-sliding (full causal); (sliding_window, 0) is a left-only
        # causal sliding window. Guard against sliding_window=None.
        window_size = (sliding_window, 0) if sliding_window is not None else (-1, -1)
        attn_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            window_size=window_size,
            softmax_scale=scaling,
            causal=True,
        )
        # Some FA3 builds return (out, softmax_lse); keep only the output.
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_output = attn_output.unsqueeze(0)  # (B=1, S, num_head, head_dim)
        if attn_output.dim() != 4:
            raise ValueError(f"Expecting attn_output of shape (B=1, S, num_head, head_dim), but got {tuple(attn_output.shape)}")

    return attn_output, None

"""Gemma 4 attention backend that routes ALL layers through this repo's FA4 cute kernel
(flash_attn.cute.interface.flash_attn_varlen_func) — including the head_dim=512 global
layers (which the autotrain repo's dynamic_attention can only do via tiled SDPA, since FA3
caps at 256). Used by compare_logits_fa4.py to check that FA-512 gives the same model
logits as the default attention.

Drop-in for transformers' AttentionInterface, mirroring the signature of
GPUPlatform/autotrain/gemma4/attention.py::dynamic_attention.
"""
from typing import Optional
import torch


def fa4_attention(
    module: torch.nn.Module,
    query: torch.Tensor,            # (B, num_head, S, head_dim)
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_length_q=None,
    max_length_k=None,
    sliding_window=None,
    scaling=None,
    **kwargs,
):
    from flash_attn.cute.interface import flash_attn_varlen_func

    if cu_seq_lens_q is None:
        raise ValueError("fa4_attention requires cu_seq_lens_q (packed varlen metadata).")

    # (B, H, S, D) -> packed (total, H, D); B is always 1 in the packed pipeline.
    q = query.permute(0, 2, 1, 3).squeeze(0).contiguous().to(torch.bfloat16)
    k = key.permute(0, 2, 1, 3).squeeze(0).contiguous().to(torch.bfloat16)
    v = value.permute(0, 2, 1, 3).squeeze(0).contiguous().to(torch.bfloat16)
    cu_q = cu_seq_lens_q.to(device=q.device, dtype=torch.int32)
    cu_k = cu_seq_lens_k.to(device=q.device, dtype=torch.int32)
    window_size = (sliding_window, 0) if sliding_window is not None else (None, None)

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_length_q,
        max_seqlen_k=max_length_k,
        window_size=window_size,
        softmax_scale=scaling,
        causal=True,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.unsqueeze(0), None  # (B=1, S, num_head, head_dim)

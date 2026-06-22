# head_dim=512 on SM90 (Gemma 4 global layers)

Goal: support symmetric `head_dim=512` (q=k=v=512) attention on Hopper (SM90) for Gemma 4's
global-attention layers. Gemma 4 (transformers >=5.5.0, `models/gemma4`) uses `head_dim=256`
for sliding layers and `global_head_dim=512` for the ~1/6 global layers — symmetric, GQA
(8 query / 4 kv heads), causal, bf16. q_norm/k_norm/v_norm + RoPE are applied before attention,
so the kernel just sees plain q/k/v at head_dim=512.

Local dev GPUs are Ampere (RTX 3090) and cannot run FA4; all work was done on a RunPod H100
(`RUNPOD_API_KEY` in `.env`), torch 2.8.0+cu129, py3.12.

## Dependency pin (or nothing imports/runs)

The cute `pyproject.toml` uses `>=` bounds; latest deps break with API skew. Working combo:

```bash
pip install -e "flash_attn/cute[dev]"
pip install "nvidia-cutlass-dsl==4.4.2" "quack-kernels==0.3.10"
```

- quack 0.5.0 dropped the `arch` positional in `get_smem_store_C` → `TypeError` in `flash_bwd_sm90.py`.
- cutlass-dsl 4.5.x renamed `cutlass.base_dsl.Arch` → `arch` → quack 0.3.x `import Arch` fails.

## The constraints at head_dim=512 on SM90

- WGMMA N-mode is capped at 256 → any MMA whose N is the 512 head dim must split into ≤256 atoms.
- `O[tile_m, 512]` fp32 accumulator: with 1 warpgroup (128 threads) and tile_m=64 that's
  64*512/128 = 256 regs/thread → spills (255 reg limit) → ~12× slowdown.
- smem is 232448 B (227KB). A `[128,512]` bf16 tile = 128KB, so tile_m must be 64.

## Forward — DONE, fused, fast

Files: `flash_attn/cute/interface.py`, `flash_attn/cute/flash_fwd_sm90.py`.

1. `_validate_head_dims`: SM90 range = `8..256` **or exactly 512** (keeps 288/384 rejected; the
   `test_flash_attn_invalid_head_dim` test still passes).
2. `_tile_size_fwd_sm90`: `head_dim>256` → `FwdConfig(tile_m=64, tile_n=80)` (Q+K+V at num_stages=1
   ≈ 224KB, just under 227KB). `num_stages=1` for hd>256 in the SM90 dispatch.
3. `_get_tiled_mma` (the key change): `pv_n_split = 2 if tile_hdimv > 256 else 1`. PV MMA uses
   `atom_layout=(tile_m//64, pv_n_split, 1)` with `tiler_mn=(64, tile_hdimv // pv_n_split)` → atom N=256.
   `num_mma_threads = max(qk.size, pv.size)` (so PV's 2 warpgroups win), and when
   `qk.size < num_mma_threads` the QK gemm is **replicated** across both warpgroups via `tidx % qk.size`
   (+ `wg_mma_qk` slice 0). So each warpgroup recomputes the full `QK^T`/softmax and owns one 256-wide
   half of O → O accumulator 256→128 regs/thread, no spill. Everything downstream (acc_O, softmax
   row count, epilogue O/LSE store, masking) adapts automatically through the tiled-MMA partitioning.

Redundant QK costs ~1.5× total flops but removes the spill: **52 → 374 TFLOPS** at d=512 (H100,
b2 hq8 hkv4 causal s4096). d=256 unchanged at ~658 TFLOPS. No regression on d=64/96/128/256.

(A faster non-redundant variant would compute QK once on one warpgroup and share P via smem
between the two PV warpgroups — more control-flow complexity, not done.)

## Backward — DONE as a correct memory-efficient path; fused kernel is the TODO

The fused SM90 bwd (`flash_bwd_sm90.py`) **cannot fit** head_dim=512 in smem: four `[64,512]` tiles
(Q,K,V,dO) = 256KB **alone** exceed 227KB, before the `[64,512]` fp32 `sdQaccum` (128KB). Its MMAs
already N-split dK/dV/dQ across 2 warpgroups (atoms become 256, so it *compiles* at 512) — it fails
at *launch* with `cudaErrorInvalidValue` (smem over the limit). dK+dV register accumulators
(`[tile_n,512]` each) are also a pressure point.

Current solution (`interface.py`): both `FlashAttnFunc.backward` and
`FlashAttnVarlenFunc.backward` route `head_dim > 256` to `_flash_attn_bwd_large_headdim`
— an exact recompute backward (varlen builds a per-document block-causal mask from
cu_seqlens, so Gemma 4's packed training is covered):
- Blocks over the query dim (`q_block=2048`), so the `s_q×s_k` scores are never fully materialised.
- bf16 tensor-core matmuls (fp32 accumulation, matching the kernel) + fp32 softmax & cross-block
  gradient accumulation. Causal block-skipping (only keys up to the diagonal).
- Handles causal / sliding-window / GQA / fp16 / bf16. Asserts no softcap/return_lse-grad
  (Gemma 4 attention uses neither; sink/score_mod/mask_mod also unsupported on this path).

**torch.compile optimisation (2026-06, this fork — OPT-IN).** Profiling showed the recompute bwd is
**memory-bound, not compute-bound**: in eager PyTorch ~53% of the time is `.float()`/`.to(bf16)`
casts and ~21% is the softmax/dS elementwise ops, while the 5 matmuls are only ~17%. So instead of
a custom fused kernel, the per-block body (`_bwd_large_headdim_block`) can be run through
`torch.compile(dynamic=True)` — inductor fuses the softmax + cast + dS pointwise chain into the
matmul epilogues. Plus two cheap wins: **dQ stays bf16** (written once per query block, so no fp32
accumulator + cast — kept in the eager path too) and **q_block 1024→2048** (halves the cross-block
fp32 dK/dV accumulation passes; compile-path only). Net: **~2.0–2.5× faster AND lower peak memory**
(fewer live fp32 intermediates), e.g. `bench.py` bwd at b2/hq8/causal: s=4096 67→136, s=8192 83→194,
s=16384 95→243 TFLOPS — at s≥16k it matches the d=256 fused bwd (~241). Bit-identical math to the
eager path (verified: eager vs compiled give the same grads to the last digit at a fixed seed); all
36 `dev512` tests pass. q_block=2048 was the sweet spot: q_block=4096 is only ~4% faster but ~50%
more memory (would break the 128k-fits-in-80GB story).

**Why it's OFF by default** (`FLASH_ATTENTION_LARGE_HEADDIM_COMPILE=1` to enable). varlen *training*
feeds a different total-token count (and document layout) every step. `dynamic=True` is *meant* to
keep one reusable graph across seqlens — but if its symbolic-shape inference ever degrades to
per-shape recompilation, it blows past `torch._dynamo.config.cache_size_limit` (default 8) after 8
distinct shapes and then **silently falls back to eager permanently** (just a warning). That silent
perf cliff is unacceptable as a training default, and I have only verified the *fixed-shape* loop,
not a varying-shape stream — so the default is eager (original ~67–95 TFLOPS, recompile-free) and
compile is opt-in. `dev512/test_varlen_recompile.py` streams 30 differently-shaped varlen batches
and asserts the recompile count stays bounded + grads stay correct — run it before enabling compile
for a given training shape distribution. (TODO: run that test on H100 to confirm dynamic=True holds.)

### Fused chunked backward — plan (further perf follow-up)

To make the bwd a fused kernel, head-dim chunk `flash_bwd_sm90.py` by factor 2 (256-wide):
- Load Q/K/V/dO in 256-wide d-chunks (smem ≈ 4×32KB + sdQaccum 64KB + P/dS ≈ 209KB < 227KB).
- `S=Q@K^T` and `dP=dO@V^T` accumulate over the 2 d-chunks (MMA_K), softmax/dS computed once on the
  full `[tile_m,tile_n]`.
- `dV_c=P^T@dO_c`, `dK_c=dS^T@Q_c`, `dQ_c=dS@K_c` emitted per chunk; reload operands per chunk.
- Prefer "chunk outside the q-loop" (recompute S/dS per chunk) to keep dK/dV accumulators at
  `[tile_n,256]` (≈128 regs) instead of `[tile_n,512]`.
This touches `load()`, `mma()` (5 fragment sets), `mma_one_m_block()`, smem layout, the dQ TMA-reduce
and `epilogue_dKV` — large and pipeline-sensitive.

## Benchmark vs the SDPA-512 fallback (the thing this replaces)

Gemma 4's head_dim=512 full-attention currently uses query-tiled SDPA + activation
checkpointing (`GPUPlatform/autotrain/gemma4/attention.py::_packed_sdpa_full`).
`dev512/compare_attn.py` benchmarks FA-512 (fused fwd + recompute bwd, with
`FLASH_ATTENTION_LARGE_HEADDIM_COMPILE=1`) against it, combined fwd+bwd, B=1 H=8 Hkv=4 D=512 bf16
causal on one H100 80GB (outputs/grads match within bf16 tol; default eager bwd is ~2–2.5× slower):

| seqlen | FA-512 mem / fwd+bwd | SDPA-tiled mem / fwd+bwd | naive SDPA |
|-------:|---------------------:|-------------------------:|-----------:|
|   4096 |  1.0 GB /     3 ms   |   1.6 GB /    28 ms      | 2.8 GB / 20 ms |
|   8192 |  2.1 GB /     8 ms   |   3.2 GB /   109 ms      | 9.7 GB / 79 ms |
|  16384 |  4.2 GB /    28 ms   |   6.4 GB /   419 ms      | 36.6 GB / 314 ms |
|  32768 |  8.3 GB /   103 ms   |  13.2 GB /  1692 ms      | OOM |
|  65536 | 16.5 GB /   400 ms   |  28.4 GB /  6747 ms      | OOM |
| 131072 | 33.1 GB /  1560 ms   |  65.3 GB / 26881 ms      | OOM |

~10–17× faster fwd+bwd (~22× fwd-only), ~35–50% less memory, no O(H·S²) OOM. 128k context
fits on one 80GB H100 (~33 GB), leaving room for ~256k. (Was ~5–7× / ~25–40% with the old eager
recompute bwd — the torch.compile bwd roughly tripled the fwd+bwd advantage and cut memory.)
`gemma4_dynamic_attention.py` is copied from the autotrain repo so the comparison is
self-contained.

## End-to-end logits validation (real Gemma 4)

`dev512/compare_logits_fa4.py` loads `google/gemma-4-31B-it` twice: once with all attention
layers routed through this repo's FA4 varlen kernel (`gemma4_fa4_attention.py`, registered via
transformers `AttentionInterface`), once with the default attention. On a short single-document
prompt the logits match: argmax + top-5 identical, **cosine 0.99914** (bf16 noise). So FA-512 is a
verified drop-in for Gemma 4's attention end-to-end (head_dim=512 global + 256 sliding layers).
Needs `HF_TOKEN` + ~62 GB model on an 80 GB H100.

## Verification

- `dev512/check.py` / `dev512/check_varlen.py` — dense / packed-varlen fwd/bwd vs torch reference (args: `--d --dv --causal --hq --hkv --sq --sk --dtype --bwd`).
- `dev512/bench.py` — TFLOPS for fwd/bwd.
- `dev512/test_hdim512.py` — pytest, 36 cases (dtype × causal × GQA/MHA/MQA × seqlen), all pass.
  Run from `dev512/`: `cd dev512 && pytest test_hdim512.py -q` (so the installed `flash-attn-4`
  cute package is used instead of the FA2-importing top-level `flash_attn/__init__.py`).

Note: symmetric `head_dim=192` *backward* is broken in this tree+deps (`flash_bwd_postprocess.py`
builds an M=192 MMA → "M-mode must be 64") — pre-existing, unrelated to the 512 work.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashAttention-4 (FA4) — fast, memory-efficient exact attention kernels written in Python using CuTeDSL (NVIDIA CUTLASS DSL). Kernels are compiled to PTX/CUBIN at runtime. Targets Hopper (SM90) and Blackwell (SM100/SM110) GPUs. Package name: `flash-attn-4`.

The repository also contains older generations (FA2 in top-level `csrc/`, FA3 in `hopper/`) but active development is on FA4 in `flash_attn/cute/`.

**This fork adds symmetric `head_dim=512` support on Hopper (SM90)** for Gemma 4's global-attention layers (Gemma 4 uses `head_dim=256` for sliding layers and `global_head_dim=512` for the ~1/6 global layers; GQA, causal). See the "head_dim=512 (Gemma 4)" section below.

## Agent Scratch Space

Use `agent_space/` for project-local scratch work such as lab notes, profiling outputs, temporary repro scripts, and experiment artifacts. Treat it as disposable workspace rather than product code.

## Build & Install

```bash
pip install flash-attn-4
# or dev install:
pip install -e "flash_attn/cute[dev]"
```

Dependencies: `nvidia-cutlass-dsl>=4.4.1`, `torch`, `einops`, `apache-tvm-ffi`, `quack-kernels>=0.2.10`.

> **Version pin (important):** the `>=` bounds let `pip` pull *too-new* deps that break with API skew (quack 0.5.0 dropped the `arch` arg in `get_smem_store_C`; cutlass-dsl 4.5.x renamed `cutlass.base_dsl.Arch`). The combo that works with this tree (HEAD ~2026-04) on H100 / torch 2.8 / py3.12 is:
> ```bash
> pip install -e "flash_attn/cute[dev]"
> pip install "nvidia-cutlass-dsl==4.4.2" "quack-kernels==0.3.10"
> ```

## Running Tests

```bash
pytest tests/cute/test_flash_attn.py
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x  # single test
pytest tests/cute/test_flash_attn_varlen.py
pytest tests/cute/test_mask_mod.py
pytest tests/cute/test_score_mod.py
pytest tests/cute/test_block_sparsity.py
```

### Fast two-pass testing

Compilation dominates test time. The fast workflow separates compilation (parallel, no GPU needed) from execution (uses cached binaries):

```bash
# Pass 1: compile all kernels in parallel using FakeTensorMode (no GPU memory allocation)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: run tests using cached compiled kernels
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -x tests/cute/test_flash_attn.py
```

- `FLASH_ATTENTION_FAKE_TENSOR=1` — uses PyTorch FakeTensorMode to compile kernels without allocating GPU memory or running them.
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` — enables persistent disk cache at `/tmp/${USER}/flash_attention_cute_dsl_cache/`.
- `-n 256` — pytest-xdist parallel workers (only useful in the compilation pass).

Tests are parametrized over dtype (fp16/bf16), head dimension (64, 96, 128), sequence length, causal/non-causal, and MHA/GQA/MQA.

If you get OOM errors running tests or benchmarks, use `nvidia-smi` to find a free GPU and select it with `CUDA_VISIBLE_DEVICES=<id>`.

## Linting

Pre-commit uses ruff on `flash_attn/cute/` files. Large kernel files (`flash_bwd.py`, `flash_fwd.py`, `flash_fwd_sm100.py`, `interface.py`) are excluded from auto-formatting.

```bash
ruff check flash_attn/cute/ --fix
ruff format flash_attn/cute/
```

## Code Architecture

### Public API (`flash_attn/cute/interface.py`)

Two entry points exported from `flash_attn/cute/__init__.py`:
- `flash_attn_func(q, k, v, ...)` — standard attention
- `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)` — variable-length

Key parameters: `causal`, `window_size_left/right`, `softmax_scale`, `softcap`, `score_mod`, `mask_mod`, `block_sparse_tensors`, `num_splits`, `pack_gqa`, `m_block_size`, `n_block_size`, `num_threads`.

Tensor layout: `(batch, seqlen, num_heads, head_dim)`, last dim contiguous, 16-byte aligned.

### Forward Kernels

- `flash_fwd.py` — `FlashAttentionForwardSm90`: Hopper forward. No SplitKV or paged KV.
- `flash_fwd_sm100.py` — `FlashAttentionForwardSm100`: Blackwell forward. Full features including SplitKV, paged KV cache, persistent kernels, 2CTA instructions.
- `flash_fwd_combine.py` — `FlashAttentionForwardCombine`: merges SplitKV partial results.

### Backward Kernels

- `flash_bwd.py` — `FlashAttentionBackwardSm80`: Ampere backward (base).
- `flash_bwd_sm90.py` — `FlashAttentionBackwardSm90`: Hopper backward.
- `flash_bwd_sm100.py` — `FlashAttentionBackwardSm100`: Blackwell backward with 2CTA and block sparse support.
- `flash_bwd_preprocess.py` / `flash_bwd_postprocess.py` — auxiliary backward kernels.

### Core Abstractions

- `softmax.py` — Online softmax with row_max/row_sum tracking, score modifier support.
- `mask.py` — `AttentionMask`: causal, local/sliding window, block sparse, mask_mod application.
- `block_info.py` — `BlockInfo`: tile dimensions, n/m block range computation for causal/local masking.
- `seqlen_info.py` — `SeqlenInfoQK`: sequence length and offset tracking for varlen.
- `pipeline.py` — `PipelineStateSimple`: circular buffer index/phase management for pipelined loads.
- `tile_scheduler.py` — Tile scheduling strategies (single tile, varlen-aware, persistent).
- `copy_utils.py` — Type-converting copies, shared-to-register loads, TMA copy atoms.
- `named_barrier.py` — Named barrier enums for warp synchronization.

### Architecture-Specific Helpers

- `hopper_helpers.py` — SM90 warp-group GEMM, shared memory layout creation, fence/commit/wait.
- `blackwell_helpers.py` — SM100 UMMA-based GEMM, PTX-optimized paths, 2CTA support.
- `mma_sm100_desc.py` — Hardware MMA descriptor enums (formats, saturation, scaling).

### Other Components

- `pack_gqa.py` — Packs multiple Q heads per KV head for efficient GQA.
- `paged_kv.py` — `PagedKVManager`: paged KV cache with TMA support.
- `fast_math.py` — exp2 polynomial coefficients, softcap score_mod creation.
- `utils.py` — Hash functions for compile cache keys, warp reductions, predicates.
- `cache_utils.py` — JIT compilation cache management.
- `cute_dsl_utils.py` — Patched `cute.compile` that optionally dumps SASS.

### Compilation & Caching

Kernels are JIT-compiled. Cache key includes dtype, head_dim, causal, mask/score_mod hashes, architecture, block sizes. Caching levels: in-memory LRU + optional disk cache via `get_jit_cache()`.

Env vars: `CUTE_CUBIN_PATH` (dump CUBIN/SASS), `CUTE_DSL_KEEP_PTX=1` (inspect PTX), `CUTE_DSL_PTXAS_PATH` (custom ptxas).

## head_dim=512 (Gemma 4)

Symmetric `head_dim=512` (q=k=v=512) is enabled on **SM90 only** (Hopper). Gemma 4's global
layers use it (GQA 8/4, causal, bf16); local/sliding layers use `head_dim=256` (unchanged).
Use the normal entry points — `flash_attn_func` / `flash_attn_varlen_func` route automatically.

Why 512 is hard on SM90: WGMMA N-mode caps at 256, the `O[tile_m, 512]` fp32 accumulator is
huge, and a `[128, 512]` bf16 tile alone is 128KB of the 227KB smem budget.

**Forward — fused, fast** (`flash_fwd_sm90.py`, `interface.py`):
- `_validate_head_dims`: SM90 allows `8..256` **or exactly 512** (288/384/etc. still rejected).
- `_tile_size_fwd_sm90`: hdim-512 → `FwdConfig(tile_m=64, tile_n=80)`, `num_stages=1` (smem-bound).
- `_get_tiled_mma`: when `tile_hdimv > 256`, `pv_n_split=2` → PV MMA `atom_layout=(tile_m//64, 2, 1)`,
  atom N=256. Technique = **redundant-QK 2-warpgroup N-split**: `num_mma_threads=max(qk,pv)`, the
  QK gemm is replicated across both warpgroups (`tidx % qk_size`) so each redundantly computes the
  full `QK^T`/softmax and owns one 256-wide half of O → O accumulator drops 256→128 regs/thread.
- Result on H100: ~374 TFLOPS at d=512 (vs ~52 spilling before the N-split); ~658 TFLOPS at d=256 (unchanged).

**Backward — correct, memory-efficient (not yet a fused kernel)** (`interface.py`):
- The fused SM90 bwd cannot fit head_dim=512 in smem (four `[64,512]` tiles = 256KB **alone**
  exceed 227KB, before the `[64,512]` fp32 `sdQaccum`), so a fused bwd needs head-dim chunking
  (a large rewrite of `flash_bwd_sm90.py`'s 5-gemm core — **TODO / perf follow-up**).
- For now both `FlashAttnFunc.backward` **and `FlashAttnVarlenFunc.backward`** route
  `head_dim > 256` to `_flash_attn_bwd_large_headdim`: an exact recompute backward, blocked
  over the query dim (never materialises the full `s_q×s_k` scores), bf16 tensor-core matmuls
  + fp32 softmax/accumulate, with causal block-skipping. Varlen (packed) uses a per-document
  block-causal mask built from cu_seqlens — i.e. Gemma 4's packed training is covered.
  ~67–83 TFLOPS at d=512 on H100. No softcap/sink/score_mod/mask_mod (Gemma 4 needs none).
- **Validated end-to-end on the real `google/gemma-4-31B-it`**: FA-512 (all attention layers
  routed through this repo's varlen kernel) vs the default attention give matching logits —
  argmax + top-5 identical, cosine 0.99914 (bf16 noise). See `dev512/compare_logits_fa4.py`.
- Adequate for training since global (512) layers are only ~1/6 of Gemma 4; the fast kernels run everywhere else.

**Memory & speed vs the SDPA-512 fallback.** Gemma 4's `head_dim=512` full-attention
layers currently fall back to query-tiled SDPA + activation checkpointing
(`GPUPlatform/autotrain/gemma4/attention.py::_packed_sdpa_full`, since FA3 caps at 256).
FA-512 (fused fwd + recompute bwd) vs that fallback — combined forward+backward, B=1,
H=8/Hkv=4, D=512, bf16, causal, single H100 80GB (outputs/grads match within bf16 tol):

| seqlen | FA-512 mem / fwd+bwd | SDPA-tiled mem / fwd+bwd | naive SDPA |
|-------:|---------------------:|-------------------------:|-----------:|
|   4096 |   1.4 GB /     6 ms  |    1.6 GB /    28 ms     | 2.8 GB / 20 ms |
|   8192 |   2.7 GB /    18 ms  |    3.2 GB /   108 ms     | 9.7 GB / 79 ms |
|  16384 |   5.4 GB /    64 ms  |    6.4 GB /   420 ms     | 36.6 GB / 314 ms |
|  32768 |  10.2 GB /   241 ms  |   13.2 GB /  1692 ms     | OOM |
|  65536 |  21.7 GB /   952 ms  |   28.4 GB /  6747 ms     | OOM |
| 131072 |  41.0 GB /  3769 ms  |   65.3 GB / 26881 ms     | OOM |

→ **~5–7× faster fwd+bwd** (~20–90× forward-only), **~25–40% less memory** than the tiled
SDPA fallback, and it avoids the naive-SDPA O(H·S²) OOM entirely — 128k context fits on one
80GB H100 (~41 GB), leaving room for ~192k. The non-fused recompute backward is the current
bottleneck (a fused chunked bwd kernel would widen the gap further).

**Packed varlen** — this is what Gemma 4 training actually does: documents packed to `total`
tokens with per-document block-causal attention (cu_seqlens). FA-512 via
`flash_attn_varlen_func` vs the SDPA-512 fallback (`_packed_sdpa_full`, same cu_seqlens),
fwd+bwd, doc_len=2048, H=8/Hkv=4, D=512, bf16, H100 80GB (outputs match within bf16 tol):

| total tokens | FA-512 mem / fwd+bwd | SDPA-tiled mem / fwd+bwd | speedup |
|-------------:|---------------------:|-------------------------:|--------:|
|         8192 |   2.6 GB /    17 ms  |    3.0 GB /   108 ms     |  ~6.2×  |
|        16384 |   5.2 GB /    60 ms  |    6.1 GB /   421 ms     |  ~7.1×  |
|        32768 |  10.4 GB /   222 ms  |   12.7 GB /  1694 ms     |  ~7.6×  |
|        65536 |  20.8 GB /   869 ms  |   27.4 GB /  6714 ms     |  ~7.7×  |

Same ~7× fwd+bwd speedup and ~25% lower memory as the dense case — the packing (block-diagonal
mask) doesn't change the win. Reproduce: `cd dev512 && python compare_attn.py` (dense) and
`python compare_attn_varlen.py` (packed) — both copy `gemma4_dynamic_attention.py` from the autotrain repo.

**Testing / dev** (cannot run on local Ampere — needs SM90). All in `dev512/`:
- `check.py` / `check_varlen.py` — dense / packed-varlen fwd+bwd correctness vs torch ref.
- `test_hdim512.py` — pytest, 36 cases (run from `dev512/` so the installed `flash-attn-4`
  cute package shadows the FA2-importing top-level `flash_attn`: `cd dev512 && pytest test_hdim512.py -q`).
- `bench.py` — TFLOPS; `compare_attn.py` / `compare_attn_varlen.py` — vs SDPA-512 fallback.
- `compare_logits_fa4.py` + `gemma4_fa4_attention.py` — real-Gemma-4 logits match vs default attn
  (needs `HF_TOKEN`, ~62 GB model, 80 GB H100).
- This was developed on a RunPod H100 (`RUNPOD_API_KEY`, `HF_TOKEN` in `.env`); see `AI/HDIM512.md`.

## Key Patterns

- Compile-time constants use `cutlass.Constexpr[type]` for kernel specialization.
- Score/mask modifiers are user-defined `@cute.jit` callables injected into the kernel at compile time.
- Forward execution: load Q tile → loop over K/V blocks (pipelined) → online softmax accumulation → store O and LSE.
- 2CTA instructions (SM100, hdim=128): both CTAs in a cluster coordinate via shared mbarriers; tx_count must be multiplied by `cta_group_size`.

## Debugging GPU Kernels

See `AI/DEBUG_2CTA.md` for kernel hang/deadlock debugging (printf bisection, pipeline barrier analysis, 2CTA pitfalls). See `AI/RACECHECK_TMA_HAZARD.md` for `compute-sanitizer` false positives with `cp.async.bulk`.

Key tools:
- `cute.printf` with thread guards (`tidx % 32 == 0`, `elect_one()`) for targeted output
- `compute-sanitizer --tool=racecheck` (beware false positives with raw TMA)
- `CUTE_DSL_KEEP_PTX=1` and `CUTE_DSL_LINEINFO=1` for PTX inspection and sanitizer source mapping

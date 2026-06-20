"""Memory + speed comparison for Gemma 4 head_dim=512 full-attention (fwd+bwd):

  FA-512            : this repo's flash_attn_func (fused N-split fwd + recompute bwd)
  SDPA-tiled (ckpt) : the dynamic_attention head_dim=512 fallback from
                      GPUPlatform/autotrain/gemma4/attention.py (_packed_sdpa_full:
                      query-axis-tiled SDPA + activation checkpointing)
  SDPA-naive        : a single scaled_dot_product_attention call (materializes H*S^2)

Single-document causal (cu=[0,S]) so all three compute identical attention; verifies
outputs/grads match, then reports peak GPU memory and fwd/bwd wall-clock.
"""
import argparse, math, time, gc
import torch
from torch.nn.functional import scaled_dot_product_attention

from flash_attn.cute.interface import flash_attn_func
from gemma4_dynamic_attention import _packed_sdpa_full


def sync():
    torch.cuda.synchronize()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def timed(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(iters):
        fn()
    sync()
    return (time.time() - t0) / iters * 1e3  # ms


def run_one(method, S, B, H, Hkv, D, scale, iters):
    dev = "cuda"
    torch.manual_seed(0)
    q = (torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16) / 2).requires_grad_()
    k = (torch.randn(B, S, Hkv, D, device=dev, dtype=torch.bfloat16) / 2).requires_grad_()
    v = (torch.randn(B, S, Hkv, D, device=dev, dtype=torch.bfloat16) / 2).requires_grad_()
    cu = torch.tensor([0, S], device=dev, dtype=torch.int32)

    def fa_fwd():
        o = flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
        return o[0] if isinstance(o, (tuple, list)) else o

    def sdpa_tiled_fwd():
        o = _packed_sdpa_full(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                              cu, scale, enable_gqa=True, block=2048)
        return o.transpose(1, 2)  # -> (B,S,H,D)

    def sdpa_naive_fwd():
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        o = scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=scale, enable_gqa=True)
        return o.transpose(1, 2)

    fwd = {"fa": fa_fwd, "sdpa_tiled": sdpa_tiled_fwd, "sdpa_naive": sdpa_naive_fwd}[method]

    g = torch.randn(B, S, H, D, device=dev, dtype=torch.bfloat16)

    def fwd_bwd():
        for t in (q, k, v):
            t.grad = None
        out = fwd()
        out.backward(g)
        return out

    # correctness vs FA reference is checked by the caller; here measure mem+time.
    torch.cuda.reset_peak_memory_stats()
    sync()
    out = fwd_bwd()
    sync()
    mem = peak_gb()
    t_fb = timed(fwd_bwd, iters)
    # forward-only time
    t_f = timed(lambda: fwd(), iters)
    del out
    grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
    ofa = fwd().detach().clone()
    for t in (q, k, v):
        t.grad = None
    return mem, t_f, t_fb, ofa, grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--Hkv", type=int, default=4)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--seqlens", type=str, default="2048,4096,8192,16384,32768")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--methods", type=str, default="fa,sdpa_tiled,sdpa_naive")
    args = ap.parse_args()
    scale = 1.0 / math.sqrt(args.D)
    seqlens = [int(s) for s in args.seqlens.split(",")]
    methods = args.methods.split(",")
    print(f"# B={args.B} H={args.H} Hkv={args.Hkv} D={args.D} bf16 causal  GPU={torch.cuda.get_device_name()}")
    print(f"{'seqlen':>7} {'method':>11} {'peak_GB':>8} {'fwd_ms':>8} {'fwd+bwd_ms':>11} {'vs_FA_out':>10} {'vs_FA_dq':>9}")
    for S in seqlens:
        ref_out = None; ref_grads = None
        for m in methods:
            gc.collect(); torch.cuda.empty_cache()
            try:
                mem, t_f, t_fb, out, grads = run_one(m, S, args.B, args.H, args.Hkv, args.D, scale, args.iters)
            except torch.cuda.OutOfMemoryError:
                print(f"{S:>7} {m:>11} {'OOM':>8}")
                gc.collect(); torch.cuda.empty_cache()
                continue
            if m == "fa":
                ref_out, ref_grads = out, grads
                od = dqd = 0.0
            else:
                od = (out.float() - ref_out.float()).abs().max().item()
                dqd = (grads[0].float() - ref_grads[0].float()).abs().max().item()
            print(f"{S:>7} {m:>11} {mem:>8.2f} {t_f:>8.2f} {t_fb:>11.2f} {od:>10.1e} {dqd:>9.1e}")
        print()


if __name__ == "__main__":
    main()

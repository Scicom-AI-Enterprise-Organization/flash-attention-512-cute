"""Quick fwd/bwd timing for head_dim=512 vs a reference point."""
import argparse, math, time
import torch
from flash_attn.cute.interface import flash_attn_func


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--hq", type=int, default=8)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--s", type=int, default=4096)
    ap.add_argument("--bwd", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"; dt = torch.bfloat16
    d = args.d
    q = torch.randn(args.b, args.s, args.hq, d, device=dev, dtype=dt, requires_grad=args.bwd == 1) / 2
    k = torch.randn(args.b, args.s, args.hkv, d, device=dev, dtype=dt, requires_grad=args.bwd == 1) / 2
    v = torch.randn(args.b, args.s, args.hkv, d, device=dev, dtype=dt, requires_grad=args.bwd == 1) / 2
    scale = 1.0 / math.sqrt(d)
    causal = bool(args.causal)

    def fwd():
        o = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
        return o[0] if isinstance(o, (tuple, list)) else o

    t = bench(fwd)
    # flops: 2 matmuls (QK, PV), each 2*b*hq*s*s*d, causal ~ /2
    f = 2 * 2 * args.b * args.hq * args.s * args.s * d
    if causal:
        f = f / 2
    print(f"fwd  d={d} b={args.b} hq={args.hq} hkv={args.hkv} s={args.s} causal={causal}: "
          f"{t*1e3:.3f} ms  {f/t/1e12:.1f} TFLOPS")
    if args.bwd:
        o = fwd()
        g = torch.randn_like(o)

        def bwd():
            torch.autograd.grad(o, (q, k, v), g, retain_graph=True)
        tb = bench(bwd, iters=20, warmup=5)
        fb = f * 2.5  # bwd ~2.5x fwd flops
        print(f"bwd  : {tb*1e3:.3f} ms  {fb/tb/1e12:.1f} TFLOPS")


if __name__ == "__main__":
    main()

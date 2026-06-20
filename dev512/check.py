"""Correctness harness for flash_attn cute fwd+bwd at arbitrary head_dim.

Usage:
  python dev512/check.py --d 128 --dv 128 --causal 1 --hq 8 --hkv 4 --sq 1024 --sk 1024
  python dev512/check.py --d 512 --dv 512 --causal 1 --hq 8 --hkv 4 --sq 512 --sk 512
"""
import argparse, math, sys
import torch

from flash_attn.cute.interface import flash_attn_func


def ref_attn(q, k, v, causal, scale, gqa_groups):
    # q: (b, sq, hq, d), k/v: (b, sk, hkv, d/dv).  fp32 reference.
    b, sq, hq, d = q.shape
    sk = k.shape[1]
    hkv = k.shape[2]
    qf = q.float(); kf = k.float(); vf = v.float()
    if hkv != hq:
        rep = hq // hkv
        kf = kf.repeat_interleave(rep, dim=2)
        vf = vf.repeat_interleave(rep, dim=2)
    # -> (b, h, s, d)
    qf = qf.transpose(1, 2); kf = kf.transpose(1, 2); vf = vf.transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if causal:
        # align bottom-right (sq query rows vs sk keys)
        i = torch.arange(sq, device=q.device).view(sq, 1)
        j = torch.arange(sk, device=q.device).view(1, sk)
        mask = (j > (i + (sk - sq)))
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, vf)  # (b, h, sq, dv)
    return o.transpose(1, 2).contiguous()  # (b, sq, hq, dv)


def report(name, a, b):
    a = a.float(); b = b.float()
    diff = (a - b).abs()
    rel = diff.max() / (b.abs().max() + 1e-6)
    print(f"  {name:4s} max_abs={diff.max().item():.4e}  mean_abs={diff.mean().item():.4e}  rel={rel.item():.4e}")
    return diff.max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--dv", type=int, default=None)
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--hq", type=int, default=8)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--sq", type=int, default=1024)
    ap.add_argument("--sk", type=int, default=1024)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--bwd", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dv = args.dv if args.dv is not None else args.d
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(args.seed)
    dev = "cuda"
    causal = bool(args.causal)
    scale = 1.0 / math.sqrt(args.d)

    q = torch.randn(args.b, args.sq, args.hq, args.d, device=dev, dtype=dtype) / 2
    k = torch.randn(args.b, args.sk, args.hkv, args.d, device=dev, dtype=dtype) / 2
    v = torch.randn(args.b, args.sk, args.hkv, dv, device=dev, dtype=dtype) / 2
    q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)

    print(f"shape q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} causal={causal} dtype={args.dtype}")
    try:
        out = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
        if isinstance(out, (tuple, list)):
            out = out[0]
    except Exception as e:
        print("FWD RAISED:", repr(e))
        raise
    ref = ref_attn(q, k, v, causal, scale, args.hq // args.hkv)
    print("forward:")
    fmax = report("out", out, ref)

    ok = fmax < 3e-2
    if args.bwd:
        g = torch.randn_like(out)
        dq, dk, dv_ = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
        rq, rk, rv = torch.autograd.grad(ref, (q, k, v), g)
        print("backward:")
        bq = report("dq", dq, rq)
        bk = report("dk", dk, rk)
        bvv = report("dv", dv_, rv)
        ok = ok and max(bq, bk, bvv) < 5e-2
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

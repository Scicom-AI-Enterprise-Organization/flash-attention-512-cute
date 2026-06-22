"""Profile the head_dim=512 recompute backward: matmul vs softmax/elementwise split,
and test optimized variants (use saved LSE, torch.compile) for correctness + speed."""
import argparse, math, time
import torch
from flash_attn.cute.interface import flash_attn_func, _flash_attn_bwd_large_headdim


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def ref_bwd(q, k, v, dout, scale, causal):
    """fp32 autograd ground truth."""
    qf = q.float().detach().requires_grad_(True)
    kf = k.float().detach().requires_grad_(True)
    vf = v.float().detach().requires_grad_(True)
    b, sq, hq, d = qf.shape
    sk, hkv = kf.shape[1], kf.shape[2]
    g = hq // hkv
    kk = kf.repeat_interleave(g, dim=2) if g != 1 else kf
    vv = vf.repeat_interleave(g, dim=2) if g != 1 else vf
    qt, kt, vt = qf.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2)
    s = torch.matmul(qt, kt.transpose(-1, -2)) * scale
    if causal:
        i = torch.arange(sq, device=q.device).view(sq, 1)
        j = torch.arange(sk, device=q.device).view(1, sk)
        s = s.masked_fill(j > i + (sk - sq), float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, vt).transpose(1, 2)
    o.backward(dout.float())
    return qf.grad, kf.grad, vf.grad


def cmp(name, got, ref):
    got = got.float(); ref = ref.float()
    d = (got - ref).abs()
    rel = (d.max() / (ref.abs().max() + 1e-6)).item()
    print(f"    {name}: max_abs={d.max().item():.3e} rel={rel:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=4096)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--hq", type=int, default=8)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--check", type=int, default=1)
    args = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    b, s, hq, hkv, d = args.b, args.s, args.hq, args.hkv, args.d
    causal = bool(args.causal)
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)
    q = torch.randn(b, s, hq, d, device=dev, dtype=dt) / 2
    k = torch.randn(b, s, hkv, d, device=dev, dtype=dt) / 2
    v = torch.randn(b, s, hkv, d, device=dev, dtype=dt) / 2
    out, lse = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale, return_lse=True)
    dout = torch.randn_like(out)
    print(f"shapes: q{tuple(q.shape)} out{tuple(out.shape)} lse{tuple(lse.shape)}")

    # ---- baseline recompute bwd ----
    f = 2 * 2 * b * hq * s * s * d * (0.5 if causal else 1.0)
    fb = f * 2.5
    def base():
        return _flash_attn_bwd_large_headdim(q, k, v, out, dout, scale, causal=causal)
    tb = timed(base)
    print(f"baseline recompute bwd: {tb*1e3:.2f} ms  {fb/tb/1e12:.1f} TFLOPS")

    if args.check:
        dq, dk, dv = base()
        rq, rk, rv = ref_bwd(q, k, v, dout, scale, causal)
        print("  baseline vs fp32 ref:")
        cmp("dq", dq, rq); cmp("dk", dk, rk); cmp("dv", dv, rv)

    # ---- profiler breakdown ----
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3):
        base()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            base()
        torch.cuda.synchronize()
    evts = prof.key_averages()
    cats = {"matmul/gemm": 0.0, "softmax": 0.0, "elementwise": 0.0, "copy/cast": 0.0, "reduce": 0.0, "other": 0.0}
    total = 0.0
    for e in evts:
        t = e.self_device_time_total  # us
        total += t
        n = e.key.lower()
        if any(x in n for x in ["gemm", "cutlass", "ampere", "sm90", "matmul", "wgmma", "s16816", "bmm", "cublas", "h16816"]):
            cats["matmul/gemm"] += t
        elif "softmax" in n:
            cats["softmax"] += t
        elif any(x in n for x in ["copy", "cast", "convert", "to_copy"]):
            cats["copy/cast"] += t
        elif any(x in n for x in ["reduce", "sum"]):
            cats["reduce"] += t
        elif any(x in n for x in ["elementwise", "mul", "add", "sub", "fill", "where", "masked"]):
            cats["elementwise"] += t
        else:
            cats["other"] += t
    print(f"  --- CUDA self-time breakdown (total {total/5/1e3:.2f} ms/iter) ---")
    for kk, vv in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {kk:14s}: {vv/total*100:5.1f}%  ({vv/5/1e3:.2f} ms)")
    # top kernels
    print("  top kernels by self CUDA time:")
    for e in sorted(evts, key=lambda e: -e.self_device_time_total)[:8]:
        print(f"    {e.self_device_time_total/5/1e3:7.3f} ms  {e.key[:70]}")


if __name__ == "__main__":
    main()

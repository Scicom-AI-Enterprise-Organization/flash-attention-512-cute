"""Experiment: faster head_dim=512 recompute backward.
Baseline is memory-bound (53% copy/cast). Try: (1) use saved LSE to skip softmax
reductions, (2) torch.compile the per-block body to fuse elementwise+cast chains.
Check correctness vs fp32 autograd ref and time each."""
import argparse, math, time
import torch
from flash_attn.cute.interface import flash_attn_func, _flash_attn_bwd_large_headdim


def timed(fn, iters=20, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def ref_bwd(q, k, v, dout, scale, causal):
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


# ---------------- optimized variant: LSE + torch.compile per-block body ----------------
def _block_body(qb, kf_b, vf_b, dob, lse_b, delta_b, scale, mask):
    # qb (b,h,bm,d), kf_b (b,h,kmax,d), vf_b (b,h,kmax,dv), dob (b,h,bm,dv)
    # lse_b (b,h,bm,1), delta_b (b,h,bm,1)
    s = torch.matmul(qb, kf_b.transpose(-1, -2)).float() * scale  # (b,h,bm,kmax)
    if mask is not None:
        s = s.masked_fill(mask, float("-inf"))
    p = torch.exp(s - lse_b)                  # softmax via saved LSE, no max/sum
    p_cast = p.to(qb.dtype)
    dp = torch.matmul(dob, vf_b.transpose(-1, -2)).float()
    ds = (p * (dp - delta_b) * scale).to(qb.dtype)
    dv_c = torch.matmul(p_cast.transpose(-1, -2), dob).float()   # (b,h,kmax,dv) fp32
    dq_b = torch.matmul(ds, kf_b)                                # (b,h,bm,d) bf16 (written once)
    dk_c = torch.matmul(ds.transpose(-1, -2), qb).float()        # (b,h,kmax,d) fp32
    return dq_b, dk_c, dv_c


_block_compiled = torch.compile(_block_body, dynamic=True)


# variant that recomputes softmax in-graph (no LSE needed) — simplifies integration (esp. varlen)
def _block_body_softmax(qb, kf_b, vf_b, dob, lse_b, delta_b, scale, mask):
    s = torch.matmul(qb, kf_b.transpose(-1, -2)).float() * scale
    if mask is not None:
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    p_cast = p.to(qb.dtype)
    dp = torch.matmul(dob, vf_b.transpose(-1, -2)).float()
    ds = (p * (dp - delta_b) * scale).to(qb.dtype)
    dv_c = torch.matmul(p_cast.transpose(-1, -2), dob).float()
    dq_b = torch.matmul(ds, kf_b)
    dk_c = torch.matmul(ds.transpose(-1, -2), qb).float()
    return dq_b, dk_c, dv_c


_block_softmax_compiled = torch.compile(_block_body_softmax, dynamic=True)


# ---- v2: GQA-fold (broadcast K/V, no repeat_interleave), reduce-over-group in-graph ----
def _block_body_gqa(qb, kf_b, vf_b, dob, lse_b, delta_b, scale, mask):
    # qb (b,hkv,g,bm,d), kf_b (b,hkv,1,kmax,d), vf_b (b,hkv,1,kmax,dv), dob (b,hkv,g,bm,dv)
    # lse_b/delta_b (b,hkv,g,bm,1).  K/V broadcast over g (strided batched gemm, no copy).
    s = torch.matmul(qb, kf_b.transpose(-1, -2)).float() * scale   # (b,hkv,g,bm,kmax)
    if mask is not None:
        s = s.masked_fill(mask, float("-inf"))
    p = torch.exp(s - lse_b)
    p_cast = p.to(qb.dtype)
    dp = torch.matmul(dob, vf_b.transpose(-1, -2)).float()
    ds = (p * (dp - delta_b) * scale).to(qb.dtype)
    dq_b = torch.matmul(ds, kf_b)                                  # (b,hkv,g,bm,d)
    # reduce over group g in-graph so cross-block accumulation writes hkv-sized tensors
    dv_c = torch.matmul(p_cast.transpose(-1, -2), dob).sum(2).float()   # (b,hkv,kmax,dv)
    dk_c = torch.matmul(ds.transpose(-1, -2), qb).sum(2).float()        # (b,hkv,kmax,d)
    return dq_b, dk_c, dv_c


_block_gqa_compiled = torch.compile(_block_body_gqa, dynamic=True)


def opt_bwd_gqa(q, k, v, out, dout, lse, scale, causal, q_block=1024, compiled=True):
    b, sq, hq, d = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    dv = v.shape[-1]
    g = hq // hkv
    in_dtype = q.dtype
    dev = q.device
    body = _block_gqa_compiled if compiled else _block_body_gqa

    # (b,hkv,g,sq,d) views; K/V kept at hkv with a broadcast group axis
    qb_all = q.transpose(1, 2).reshape(b, hkv, g, sq, d)
    dob_all = dout.transpose(1, 2).reshape(b, hkv, g, sq, dv)
    of = out.transpose(1, 2).reshape(b, hkv, g, sq, dv)
    kf = k.transpose(1, 2).unsqueeze(2)      # (b,hkv,1,sk,d)
    vf = v.transpose(1, 2).unsqueeze(2)      # (b,hkv,1,sk,dv)
    delta = (dob_all.float() * of.float()).sum(-1)        # (b,hkv,g,sq)
    lse_e = lse.reshape(b, hkv, g, sq)

    dq = torch.empty((b, hkv, g, sq, d), dtype=in_dtype, device=dev)   # written once/block
    dk_e = torch.zeros((b, hkv, sk, d), dtype=torch.float32, device=dev)
    dv_e = torch.zeros((b, hkv, sk, dv), dtype=torch.float32, device=dev)

    offset = sk - sq
    j_full = torch.arange(sk, device=dev).view(1, sk)
    for q0 in range(0, sq, q_block):
        q1 = min(q0 + q_block, sq)
        kmax = min(sk, q1 + offset) if causal else sk
        qb = qb_all[:, :, :, q0:q1]
        dob = dob_all[:, :, :, q0:q1]
        lse_b = lse_e[:, :, :, q0:q1].unsqueeze(-1)
        delta_b = delta[:, :, :, q0:q1].unsqueeze(-1)
        kf_b = kf[:, :, :, :kmax]
        vf_b = vf[:, :, :, :kmax]
        mask = None
        if causal:
            i_idx = torch.arange(q0, q1, device=dev).view(-1, 1) + offset
            j_idx = j_full[:, :kmax]
            m = (j_idx > i_idx)
            if m.any():
                mask = m.view(1, 1, 1, q1 - q0, kmax)
        dq_b, dk_c, dv_c = body(qb, kf_b, vf_b, dob, lse_b, delta_b, scale, mask)
        dq[:, :, :, q0:q1] = dq_b
        dk_e[:, :, :kmax] += dk_c
        dv_e[:, :, :kmax] += dv_c

    dq_o = dq.reshape(b, hq, sq, d).transpose(1, 2).contiguous()
    dk_o = dk_e.transpose(1, 2).contiguous().to(in_dtype)
    dv_o = dv_e.transpose(1, 2).contiguous().to(in_dtype)
    return dq_o, dk_o, dv_o


def _pick_qblock(sq, sk, tile_budget):
    # bound the largest score tile [q_block x sk] to tile_budget elements; clamp to [256, 2048]
    qb = max(256, min(2048, tile_budget // max(sk, 1)))
    qb = min(qb, sq)
    return qb


def opt_bwd(q, k, v, out, dout, lse, scale, causal, q_block=1024, compiled=True, use_lse=True):
    b, sq, hq, d = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    dv = v.shape[-1]
    g = hq // hkv
    in_dtype = q.dtype
    dev = q.device
    if use_lse:
        body = _block_compiled if compiled else _block_body
    else:
        body = _block_softmax_compiled if compiled else _block_body_softmax

    qb_all = q.transpose(1, 2)               # (b,hq,sq,d)
    dob_all = dout.transpose(1, 2)           # (b,hq,sq,dv)
    of = out.transpose(1, 2)                 # (b,hq,sq,dv)
    kt = k.transpose(1, 2); vt = v.transpose(1, 2)
    kf_e = kt.repeat_interleave(g, dim=1) if g != 1 else kt
    vf_e = vt.repeat_interleave(g, dim=1) if g != 1 else vt
    delta = (dob_all.float() * of.float()).sum(-1)       # (b,hq,sq)
    lse_e = lse                                          # (b,hq,sq)

    dq = torch.empty((b, hq, sq, d), dtype=in_dtype, device=dev)   # written once/block
    dk_e = torch.zeros((b, hq, sk, d), dtype=torch.float32, device=dev)
    dv_e = torch.zeros((b, hq, sk, dv), dtype=torch.float32, device=dev)

    offset = sk - sq
    pure_causal = causal
    j_full = torch.arange(sk, device=dev).view(1, sk)
    for q0 in range(0, sq, q_block):
        q1 = min(q0 + q_block, sq)
        kmax = min(sk, q1 + offset) if pure_causal else sk
        qb = qb_all[:, :, q0:q1]
        dob = dob_all[:, :, q0:q1]
        lse_b = lse_e[:, :, q0:q1].unsqueeze(-1)
        delta_b = delta[:, :, q0:q1].unsqueeze(-1)
        kf_b = kf_e[:, :, :kmax]
        vf_b = vf_e[:, :, :kmax]
        mask = None
        if causal:
            i_idx = torch.arange(q0, q1, device=dev).view(-1, 1) + offset
            j_idx = j_full[:, :kmax]
            m = (j_idx > i_idx)
            if m.any():
                mask = m.view(1, 1, q1 - q0, kmax)
        dq_b, dk_c, dv_c = body(qb, kf_b, vf_b, dob, lse_b, delta_b, scale, mask)
        dq[:, :, q0:q1] = dq_b
        dk_e[:, :, :kmax] += dk_c
        dv_e[:, :, :kmax] += dv_c

    if g != 1:
        dk = dk_e.view(b, hkv, g, sk, d).sum(2)
        dvv = dv_e.view(b, hkv, g, sk, dv).sum(2)
    else:
        dk, dvv = dk_e, dv_e
    dq_o = dq.transpose(1, 2).contiguous().to(in_dtype)
    dk_o = dk.transpose(1, 2).contiguous().to(in_dtype)
    dv_o = dvv.transpose(1, 2).contiguous().to(in_dtype)
    return dq_o, dk_o, dv_o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=4096)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--hq", type=int, default=8)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--causal", type=int, default=1)
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
    f = 2 * 2 * b * hq * s * s * d * (0.5 if causal else 1.0)
    fb = f * 2.5
    print(f"s={s} causal={causal}  bwd flops={fb/1e12:.3f} TFLOP")

    do_ref = s <= 8192
    if do_ref:
        rq, rk, rv = ref_bwd(q, k, v, dout, scale, causal)

    def memrun(fn):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        out_ = fn(); torch.cuda.synchronize()
        return out_, torch.cuda.max_memory_allocated() / 1e9

    def base():
        return _flash_attn_bwd_large_headdim(q, k, v, out, dout, scale, causal=causal)
    if s <= 16384:
        tb = timed(base)
        (dq, dk, dv), mb = memrun(base)
        print(f"[baseline       ] {tb*1e3:8.2f} ms  {fb/tb/1e12:6.1f} TFLOPS  peak {mb:.2f} GB")
        if do_ref:
            cmp("dq", dq, rq); cmp("dk", dk, rk); cmp("dv", dv, rv)
    else:
        tb = float("nan")

    qb = min(2048, s)
    for tag, ulse in (("lse    ", True), ("softmax", False)):
        def vcomp(ulse=ulse):
            return opt_bwd(q, k, v, out, dout, lse, scale, causal, q_block=qb, compiled=True, use_lse=ulse)
        tc = timed(vcomp)
        (dq, dk, dv), mc = memrun(vcomp)
        sp = (tb / tc) if tb == tb else float("nan")
        print(f"[compile {tag} qb={qb}] {tc*1e3:8.2f} ms  {fb/tc/1e12:6.1f} TFLOPS  peak {mc:.2f} GB  ({sp:.2f}x)")
        if do_ref:
            cmp("dq", dq, rq); cmp("dk", dk, rk); cmp("dv", dv, rv)


if __name__ == "__main__":
    main()

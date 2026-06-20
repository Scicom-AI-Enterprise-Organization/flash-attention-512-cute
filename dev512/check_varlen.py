"""Varlen (packed, per-document block-causal) correctness for head_dim=512.

Mirrors Gemma 4 packed training: q/k/v are (total_tokens, h, d) with cu_seqlens marking
document boundaries; attention is per-document causal (no cross-document attention).
Compares flash_attn_varlen_func fwd+bwd against a per-document dense reference.
"""
import argparse, math, sys
import torch
from flash_attn.cute.interface import flash_attn_varlen_func


def ref_varlen(q, k, v, cu, scale, hq, hkv):
    # q:(total,hq,d) k/v:(total,hkv,d). per-doc causal. returns (total,hq,dv)
    out = torch.empty(q.shape[0], hq, v.shape[-1], device=q.device, dtype=torch.float32)
    g = hq // hkv
    for d in range(len(cu) - 1):
        s, e = cu[d].item(), cu[d + 1].item()
        if e <= s:
            continue
        qd = q[s:e].float().transpose(0, 1)          # (hq, L, d)
        kd = k[s:e].float().transpose(0, 1)          # (hkv, L, d)
        vd = v[s:e].float().transpose(0, 1)
        if g != 1:
            kd = kd.repeat_interleave(g, dim=0)
            vd = vd.repeat_interleave(g, dim=0)
        sc = torch.matmul(qd, kd.transpose(-1, -2)) * scale  # (hq, L, L)
        L = e - s
        i = torch.arange(L, device=q.device).view(L, 1)
        j = torch.arange(L, device=q.device).view(1, L)
        sc = sc.masked_fill(j > i, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        out[s:e] = torch.matmul(p, vd).transpose(0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--hq", type=int, default=8)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--docs", type=str, default="300,517,128,1024,77")
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--bwd", type=int, default=1)
    args = ap.parse_args()
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    dev = "cuda"
    torch.manual_seed(0)
    lens = [int(x) for x in args.docs.split(",")]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), device=dev, dtype=torch.int32)
    total = cu[-1].item()
    maxlen = max(lens)
    d, dv = args.d, args.d
    scale = 1.0 / math.sqrt(d)
    q = (torch.randn(total, args.hq, d, device=dev, dtype=dt) / 2).requires_grad_()
    k = (torch.randn(total, args.hkv, d, device=dev, dtype=dt) / 2).requires_grad_()
    v = (torch.randn(total, args.hkv, dv, device=dev, dtype=dt) / 2).requires_grad_()
    print(f"varlen total={total} docs={lens} hq={args.hq} hkv={args.hkv} d={d} {args.dtype}")
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                 max_seqlen_q=maxlen, max_seqlen_k=maxlen,
                                 softmax_scale=scale, causal=True)
    if isinstance(out, (tuple, list)):
        out = out[0]
    ref = ref_varlen(q, k, v, cu, scale, args.hq, args.hkv)
    fmax = (out.float() - ref).abs().max().item()
    print(f"forward: out max_abs={fmax:.3e}")
    ok = fmax < 3e-2
    if args.bwd:
        g = torch.randn_like(out)
        dq, dk, dv_ = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
        rq, rk, rv = torch.autograd.grad(ref, (q, k, v), g)
        for nm, a, b in [("dq", dq, rq), ("dk", dk, rk), ("dv", dv_, rv)]:
            m = (a.float() - b.float()).abs().max().item()
            print(f"  {nm} max_abs={m:.3e}")
            ok = ok and m < 5e-2
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

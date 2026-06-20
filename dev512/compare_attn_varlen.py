"""Varlen (packed, per-document block-causal) memory+speed: FA-512 vs SDPA-512 fallback.

Matches Gemma 4 packed training: documents of length `doc_len` packed to `total` tokens,
per-document causal. FA-512 -> flash_attn_varlen_func; SDPA -> the dynamic_attention
head_dim=512 fallback (_packed_sdpa_full, query-tiled SDPA + checkpointing) with the same
cu_seqlens. Combined forward+backward.
"""
import argparse, math, time, gc
import torch
from flash_attn.cute.interface import flash_attn_varlen_func
from gemma4_dynamic_attention import _packed_sdpa_full


def sync():
    torch.cuda.synchronize()


def timed(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(iters):
        fn()
    sync()
    return (time.time() - t0) / iters * 1e3


def run(method, total, doc_len, H, Hkv, D, scale, iters):
    dev = "cuda"
    torch.manual_seed(0)
    lens = [doc_len] * (total // doc_len)
    if sum(lens) < total:
        lens.append(total - sum(lens))
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=dev, dtype=torch.int32)
    maxlen = max(lens)
    q = (torch.randn(total, H, D, device=dev, dtype=torch.bfloat16) / 2).requires_grad_()
    k = (torch.randn(total, Hkv, D, device=dev, dtype=torch.bfloat16) / 2).requires_grad_()
    v = (torch.randn(total, Hkv, D, device=dev, dtype=torch.bfloat16) / 2).requires_grad_()
    g = torch.randn(total, H, D, device=dev, dtype=torch.bfloat16)

    def fa():
        o = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                   max_seqlen_q=maxlen, max_seqlen_k=maxlen,
                                   softmax_scale=scale, causal=True)
        return o[0] if isinstance(o, (tuple, list)) else o

    def sdpa():
        # (1,H,total,D) per-document block-causal tiled SDPA + checkpointing
        o = _packed_sdpa_full(q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2),
                              v.unsqueeze(0).transpose(1, 2), cu, scale, enable_gqa=True, block=2048)
        return o.squeeze(0).transpose(0, 1)  # (total,H,D)

    fwd = fa if method == "fa" else sdpa

    def fwd_bwd():
        for t in (q, k, v):
            t.grad = None
        fwd().backward(g)

    torch.cuda.reset_peak_memory_stats(); sync()
    fwd_bwd(); sync()
    mem = torch.cuda.max_memory_allocated() / 1e9
    t_fb = timed(fwd_bwd, iters)
    out = fwd().detach().clone()
    for t in (q, k, v):
        t.grad = None
    return mem, t_fb, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--Hkv", type=int, default=4)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--doc_len", type=int, default=2048)
    ap.add_argument("--totals", type=str, default="8192,16384,32768,65536")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    scale = 1.0 / math.sqrt(args.D)
    print(f"# varlen packed doc_len={args.doc_len} H={args.H} Hkv={args.Hkv} D={args.D} bf16 per-doc-causal  GPU={torch.cuda.get_device_name()}")
    print(f"{'total':>7} {'method':>8} {'peak_GB':>8} {'fwd+bwd_ms':>11} {'vs_FA_out':>10}")
    for total in [int(t) for t in args.totals.split(",")]:
        ref = None
        for m in ["fa", "sdpa"]:
            gc.collect(); torch.cuda.empty_cache()
            try:
                mem, t_fb, out = run(m, total, args.doc_len, args.H, args.Hkv, args.D, scale, args.iters)
            except torch.cuda.OutOfMemoryError:
                print(f"{total:>7} {m:>8} {'OOM':>8}"); continue
            od = 0.0 if m == "fa" else (out.float() - ref.float()).abs().max().item()
            if m == "fa":
                ref = out
            print(f"{total:>7} {m:>8} {mem:>8.2f} {t_fb:>11.2f} {od:>10.1e}")
        print()


if __name__ == "__main__":
    main()

"""Does the torch.compile'd head_dim=512 varlen backward survive *training* —
i.e. a stream of batches each with DIFFERENT cu_seqlens / total tokens, as opposed
to the fixed-shape bench loop?

Checks:
  1. Recompile count stays bounded (does NOT grow per distinct shape) — the real
     failure mode is dynamic-shape inference falling back to static + blowing past
     torch._dynamo cache_size_limit=8 -> silent permanent eager fallback.
  2. Per-step wall time stays flat after warmup (a recompile shows up as a spike).
  3. Grads stay correct vs a fp32 torch reference on every step.

Run from dev512/:  python test_varlen_recompile.py
"""
import math, time, random
import torch
import torch._dynamo as dynamo
from flash_attn.cute.interface import flash_attn_varlen_func


def make_batch(n_docs_range=(3, 12), doc_range=(128, 3072), H=8, Hkv=4, D=512,
               dtype=torch.bfloat16, dev="cuda", seed=0):
    g = random.Random(seed)
    docs = [g.randint(*doc_range) for _ in range(g.randint(*n_docs_range))]
    cu = torch.tensor([0] + list(torch.tensor(docs).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    T = int(cu[-1])
    q = (torch.randn(T, H, D, device=dev, dtype=dtype) / 2).requires_grad_(True)
    k = (torch.randn(T, Hkv, D, device=dev, dtype=dtype) / 2).requires_grad_(True)
    v = (torch.randn(T, Hkv, D, device=dev, dtype=dtype) / 2).requires_grad_(True)
    return q, k, v, cu, T, docs


def ref_bwd(q, k, v, cu, dout, scale, H, Hkv, D):
    g = H // Hkv
    qf = q.float().detach().requires_grad_(True)
    kf = k.float().detach().requires_grad_(True)
    vf = v.float().detach().requires_grad_(True)
    T = q.shape[0]
    ke = kf.repeat_interleave(g, dim=1); ve = vf.repeat_interleave(g, dim=1)
    out = torch.zeros(T, H, D, device=q.device, dtype=torch.float32)
    for d in range(cu.numel() - 1):
        a, b = int(cu[d]), int(cu[d + 1])
        qd = qf[a:b].transpose(0, 1)                 # (H, L, D)
        kd = ke[a:b].transpose(0, 1); vd = ve[a:b].transpose(0, 1)
        s = torch.matmul(qd, kd.transpose(-1, -2)) * scale
        L = b - a
        i = torch.arange(L, device=q.device).view(L, 1); j = torch.arange(L, device=q.device).view(1, L)
        s = s.masked_fill(j > i, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[a:b] = torch.matmul(p, vd).transpose(0, 1)
    out.backward(dout.float())
    return qf.grad, kf.grad, vf.grad


def main():
    dev = "cuda"; H, Hkv, D = 8, 4, 512
    scale = 1.0 / math.sqrt(D)
    dynamo.reset()
    n_steps = 30
    times = []
    max_err = 0.0
    print(f"{'step':>4} {'total_tok':>9} {'n_docs':>6} {'ms':>8} {'recompiles':>11}")
    for step in range(n_steps):
        q, k, v, cu, T, docs = make_batch(seed=1000 + step)  # DIFFERENT shape each step
        dout = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
        torch.cuda.synchronize(); t0 = time.time()
        out = flash_attn_varlen_func(q, k, v, cu, cu, max_seqlen_q=max(docs), max_seqlen_k=max(docs),
                                     causal=True, softmax_scale=scale)
        out = out[0] if isinstance(out, (tuple, list)) else out
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        torch.cuda.synchronize(); dt = time.time() - t0
        times.append(dt)
        # recompile counter from dynamo
        recompiles = dynamo.utils.counters["stats"].get("unique_graphs", 0)
        # spot-check correctness every few steps
        chk = ""
        if step % 6 == 0:
            rq, rk, rv = ref_bwd(q, k, v, cu, dout, scale, H, Hkv, D)
            e = max((dq.float() - rq).abs().max().item(),
                    (dk.float() - rk).abs().max().item(),
                    (dv.float() - rv).abs().max().item())
            max_err = max(max_err, e); chk = f"  err={e:.2e}"
        print(f"{step:>4} {T:>9} {len(docs):>6} {dt*1e3:>8.1f} {recompiles:>11}{chk}")

    warm = times[5:]
    print(f"\nsteady-state ms: min={min(warm)*1e3:.1f} max={max(warm)*1e3:.1f} "
          f"mean={sum(warm)/len(warm)*1e3:.1f}  (spikes => recompiles)")
    print(f"max grad err vs fp32 ref: {max_err:.2e}")
    print("dynamo counters:", dict(dynamo.utils.counters["stats"]))
    frames = dynamo.utils.counters["frames"]
    print("frames:", dict(frames))


if __name__ == "__main__":
    main()

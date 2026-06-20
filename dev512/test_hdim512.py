"""Regression tests for symmetric head_dim=512 (Gemma 4 global layers) on SM90.

Run from this directory so the installed `flash-attn-4` cute package is imported
without the top-level FA2 `flash_attn` package (which needs flash_attn_2_cuda):

    cd dev512 && pytest test_hdim512.py -q

Covers forward (fused N-split SM90 kernel) and backward (memory-efficient recompute
path) at head_dim=512 across causal/non-causal, GQA/MHA, fp16/bf16, and non-aligned
sequence lengths, comparing to a PyTorch fp32 reference.
"""
import math
import pytest
import torch

from flash_attn.cute.interface import flash_attn_func


def ref_attn(q, k, v, causal, scale):
    b, sq, hq, d = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    qf, kf, vf = q.float(), k.float(), v.float()
    if hkv != hq:
        rep = hq // hkv
        kf = kf.repeat_interleave(rep, dim=2)
        vf = vf.repeat_interleave(rep, dim=2)
    qf, kf, vf = qf.transpose(1, 2), kf.transpose(1, 2), vf.transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if causal:
        i = torch.arange(sq, device=q.device).view(sq, 1)
        j = torch.arange(sk, device=q.device).view(1, sk)
        scores = scores.masked_fill(j > (i + (sk - sq)), float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, vf).transpose(1, 2).contiguous()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "hq,hkv", [(8, 4), (8, 8), (4, 1)]  # GQA, MHA, MQA
)
@pytest.mark.parametrize("seqlen", [333, 512, 1024])
def test_hdim512_fwd_bwd(dtype, causal, hq, hkv, seqlen):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("head_dim=512 path is implemented for SM90 (Hopper)")
    torch.manual_seed(0)
    b, d = 2, 512
    dev = "cuda"
    scale = 1.0 / math.sqrt(d)
    q = (torch.randn(b, seqlen, hq, d, device=dev, dtype=dtype) / 2).requires_grad_()
    k = (torch.randn(b, seqlen, hkv, d, device=dev, dtype=dtype) / 2).requires_grad_()
    v = (torch.randn(b, seqlen, hkv, d, device=dev, dtype=dtype) / 2).requires_grad_()

    out = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
    if isinstance(out, (tuple, list)):
        out = out[0]
    ref = ref_attn(q, k, v, causal, scale)
    assert (out.float() - ref.float()).abs().max() < 3e-2

    g = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
    rq, rk, rv = torch.autograd.grad(ref, (q, k, v), g)
    for name, a, bb in [("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv)]:
        assert (a.float() - bb.float()).abs().max() < 5e-2, name

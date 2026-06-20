"""Assert Gemma 4 + FA-512 (this repo) gives the same logits as the default attention.

Adapted from GPUPlatform/autotrain/gemma4/compare_logits.py, but the custom attention
backend routes the head_dim=512 global layers (and 256 sliding layers) through this repo's
FA4 cute kernel via fa4_attention, instead of the tiled-SDPA fallback.

  (1) load gemma-4 with attn_implementation="fa4_attention", feed the packed single-doc input
  (2) load gemma-4 with default attention, feed the plain input
  (3) assert next-token argmax matches and logits agree up to bf16 noise.

For one document spanning the whole prompt, the block-diagonal causal mask reduces to ordinary
causal attention, so the two MUST agree. Prompt is short (< sliding_window) so sliding/global
layers are all effectively full-causal — this isolates the head_dim=512 attention math.

    HF_TOKEN=... python compare_logits_fa4.py
"""
import os, sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, AttentionInterface
from gemma4_fa4_attention import fa4_attention

AttentionInterface.register("fa4_attention", fa4_attention)

MODEL = os.environ.get("MODEL_ID", "google/gemma-4-31B-it")
TEXT = "hello world, how are you today?"

tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok(TEXT)["input_ids"]
L = len(ids)
input_ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

custom_batch = {
    "input_ids": input_ids_t,
    "position_ids": torch.arange(L, dtype=torch.long).unsqueeze(0),
    "attention_mask": None,
    "mm_token_type_ids": torch.zeros_like(input_ids_t),
    "cu_seq_lens_q": torch.tensor([0, L], dtype=torch.int32),
    "cu_seq_lens_k": torch.tensor([0, L], dtype=torch.int32),
    "max_length_q": L,
    "max_length_k": L,
}
print(f"prompt={TEXT!r}  L={L} tokens  MODEL={MODEL}", flush=True)


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def load(attn_impl):
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if attn_impl is not None:
        kw["attn_implementation"] = attn_impl
    return Gemma4ForConditionalGeneration.from_pretrained(MODEL, **kw).eval()


print(">> [1] fa4_attention (FA-512) + packed input", flush=True)
m = load("fa4_attention")
b = to_dev(custom_batch, m.device)
with torch.no_grad():
    logit_fa = m(**b, use_cache=False).logits[0, -1].float().cpu()
del m; torch.cuda.empty_cache()

print(">> [2] default attention + plain input", flush=True)
m = load(None)
with torch.no_grad():
    logit_def = m(input_ids=input_ids_t.to(m.device), use_cache=False).logits[0, -1].float().cpu()
del m; torch.cuda.empty_cache()

am_fa, am_def = logit_fa.argmax().item(), logit_def.argmax().item()
max_abs = (logit_fa - logit_def).abs().max().item()
cos = F.cosine_similarity(logit_fa, logit_def, dim=0).item()
print("\n================ result ================")
print(f"argmax  FA-512={am_fa} ({tok.decode([am_fa])!r})   default={am_def} ({tok.decode([am_def])!r})")
print(f"max_abs_diff={max_abs:.4f}   cosine={cos:.6f}")
print(f"top5 FA-512 ={logit_fa.topk(5).indices.tolist()}")
print(f"top5 default={logit_def.topk(5).indices.tolist()}")

ok = (am_fa == am_def) and (cos > 0.99)
print("\nPASS ✅  FA-512 logits match the default attention" if ok else "\nFAIL ❌")
sys.exit(0 if ok else 1)

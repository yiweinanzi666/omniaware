"""Targeted checks: single-modality routing (OK-VQA query image), capture vs bias in the captured layer,
replication ceil() vs EAB bias, Norm/Combiner behaviour near zero IQR."""
import sys, math, types
REPO = "C:/Users/22688/Desktop/www_code/omniaware"
sys.path[:0] = [REPO + "/src", REPO + "/tests"]
import torch
import numpy as np

from omniaware import routing, order as order_mod, pipeline, run as run_mod
from omniaware import qwen_eab as qe
from omniaware.qwen_eab import Encoded, STATE
from eab_fixtures import make_qwen, encode as text_encode

print("== 1. routing with only the OK-VQA query image as supplied context")
# OK-VQA: one query image (chunk '__query__'), candidates are text only.
attn = torch.tensor([0.0] * 3 + [0.2, 0.3, 0.1, 0.4] + [0.0] * 2)
enc = types.SimpleNamespace(spans={"__query__": (3, 7)})
r = routing.initial_route(attn, enc, {"query_images": ["x.jpg"]}, "mh")
print("initial_route ->", {k: r[k] for k in ("mode", "modalities")})
cands = [{"cid": f"k{i}", "modality": "text", "title": "", "text": "t", "gold": None} for i in range(1, 6)]
e = {"qid": "q", "candidates": cands, "fixed_evidence": False, "query_images": ["x.jpg"]}
rel = {c["cid"]: 0.5 for c in cands}
try:
    chosen = run_mod.select_chunks(e, rel, 5, "rank", 0, 0, r)
    print("select_chunks returned", len(chosen), "chunks")
except ValueError as ex:
    print("select_chunks raises ValueError:", ex)

print("\n== 2. replication copies vs the EAB bias they are meant to mirror (gamma=1.2)")
for U in (0.41, 0.6, 1.0):
    for rj in (0.05, 0.3, 0.6, 1.0):
        lam = 1.2 * U * rj
        c = order_mod.replication_counts([rj], U, True, 1.2)[0]
        print(f"U={U:.2f} r={rj:.2f}  EAB logit bias={lam:.3f}  copies={c}  log(copies)={math.log(c):.3f}")
print("gate off ->", order_mod.replication_counts([0.9, 0.2], 0.3, False, 1.2))

print("\n== 3. Norm with near-zero IQR and zsum standardisation")
n_tiny = pipeline.Norm(median=1.0, iqr=1e-9)
print("Norm(1e-9 IQR) of values 1.0,1.001,0.999:", [round(n_tiny(v), 3) for v in (1.0, 1.001, 0.999)])
comb = pipeline.Combiner.fit([{"h_grd": 0.5, "h_sem": 0.1}] * 5 + [{"h_grd": 0.5, "h_sem": 0.2}], 0.3, "zsum")
print("zsum: grd norm iqr=%s, returned term for h_grd=0.9 ->" % comb.grd.iqr, comb({"h_grd": 0.9, "h_sem": 0.1}))

print("\n== 4. captured attention vs the bias applied in the SAME layer (last layer, row t_end)")
q = make_qwen()
enc = text_encode(q)
spec_vals = {"c1": 3.0, "c2": 3.0}
last = q.layers[-1]
# Unbiased attention of the last layer, row t_end, from a clean prefill:
q._activate(enc, None)
cache = q._prefix(enc)
clean_logits = q._step(enc.input_ids[:, enc.split:], cache, capture=True)
clean_attn = STATE.captured.clone()
# Biased run with the same bias on every layer, capture on the last layer:
from omniaware.pipeline import BiasSpec
spec = BiasSpec(values=spec_vals, q_start=enc.length - 1, layers=None)
q._activate(enc, spec)
cache2 = q._prefix(enc)
biased_logits = q._step(enc.input_ids[:, enc.split:], cache2, capture=True)
biased_attn = STATE.captured.clone()
q._activate(enc, None)
print("captured(biased run) vs captured(clean):", float((biased_attn - clean_attn).abs().max()))
# Explicit last-layer attention of row t_end with the bias included:
mod = [m for m in q.model.modules() if getattr(m, "_erc_layer", None) == q.layers[-1]][0]
print("max |captured - clean| = bias effect in captured layer excluded from capture? ->",
      "YES, captured == clean attention" if float((biased_attn - clean_attn).abs().max()) < 1e-6 else "NO")

print("\n== 5. U with a flat calibration set (no probe, no evidence): IQR of Norm")
flat = pipeline.Norm.fit([0.3] * 10 + [0.300000001])
print("Norm.fit on near-constant H_pred:", flat, " U(0.31)=", round(flat(0.31), 3), " U(0.29)=", round(flat(0.29), 3))

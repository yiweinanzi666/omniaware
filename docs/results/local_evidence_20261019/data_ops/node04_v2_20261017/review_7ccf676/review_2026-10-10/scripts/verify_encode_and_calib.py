"""(a) QwenEAB.encode with a real Qwen2VL-family image processor (patch 16 / merge 2 / temporal 2, as in the
Qwen3-VL config) on real PNGs, tiny model: image-token expansion vs the vision output, span positions.
(b) capture-layer-only bias: captured attention must be unchanged in the captured layer.
(c) Calibrator: a mean of two per-channel percentiles is no longer a percentile (independent and correlated)."""
import sys, types, os
REPO = "C:/Users/22688/Desktop/www_code/omniaware"
sys.path[:0] = [REPO + "/src", REPO + "/tests"]
import numpy as np, torch
from PIL import Image

from omniaware import relevance
from omniaware.qwen_eab import QwenEAB, Segment, STATE
from omniaware.pipeline import BiasSpec
from eab_fixtures import make_qwen, CharTokenizer, encode as text_encode
from transformers import Qwen2VLImageProcessor

SCR = "C:/Users/22688/AppData/Local/Temp/claude/C--Users-22688-Desktop-www-code/8e3454d9-d7cd-4088-813b-e62b2bcd35b9/scratchpad/review"
torch.manual_seed(0)
for name, size in (("a.png", (320, 240)), ("b.png", (96, 400))):
    Image.new("RGB", size, (120, 80, 200)).save(os.path.join(SCR, name))

print("== (a) encode() with the image processor (patch 16, merge 2, temporal 2)")
q = make_qwen()
q.kind = "qwen3_vl"
ip = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
q.processor = types.SimpleNamespace(image_processor=ip, tokenizer=CharTokenizer())
q.merge = 2
q.size = {"shortest_edge": 64 * 32 * 32, "longest_edge": 256 * 32 * 32}
q.vision_start_id, q.vision_end_id, q.image_token_id = 152, 153, 150
q.eos = {155, 156, 157}
segs = [Segment(text="<|im_start|>user\n"),
        Segment(image=os.path.join(SCR, "a.png"), chunk="__query__"), Segment(text="\n"),
        Segment(text="Based on the following evidence.\n"),
        Segment(text="[E1] (image): t\n", chunk="c1"),
        Segment(image=os.path.join(SCR, "b.png"), chunk="c1", max_pixels=64 * 32 * 32),
        Segment(text="\nCaption: x\n\n", chunk="c1"),
        Segment(text="Question: q?<|im_end|>\n<|im_start|>assistant\n")]
enc = QwenEAB.encode(q, segs, split_marker=len(segs) - 1)
pad_pos = (enc.input_ids[0] == 150).nonzero().flatten().tolist()
n_q = int(enc.image_grid_thw[0].prod()) // 4
print("grid_thw:", enc.image_grid_thw.tolist(), " image_pad tokens:", len(pad_pos),
      " expected:", int(enc.image_grid_thw.prod(-1).sum()) // 4)
print("spans:", enc.spans, " length:", enc.length, " split:", enc.split)
print("query pads inside __query__ span:", all(enc.spans["__query__"][0] <= p < enc.spans["__query__"][1] for p in pad_pos[:n_q]))
print("c1 pads inside c1 span:", all(enc.spans["c1"][0] <= p < enc.spans["c1"][1] for p in pad_pos[n_q:]),
      " (query pads:", n_q, ")")
try:
    out = q.run(enc, None, 2, decode_static=True)[0]["static"].text
    print("forward + greedy decode OK, text:", repr(out))
except Exception as ex:
    print("forward FAILED:", type(ex).__name__, str(ex)[:300])

print("\n== (b) bias only in the captured (last) layer: captured attention must not change")
q2 = make_qwen()
enc2 = text_encode(q2)
q2._activate(enc2, None); cache = q2._prefix(enc2)
q2._step(enc2.input_ids[:, enc2.split:], cache, capture=True); clean = STATE.captured.clone()
spec = BiasSpec(values={"c1": 3.0, "c2": 3.0}, q_start=enc2.length - 1, layers={q2.layers[-1]})
cache = q2._prefix(enc2); q2._activate(enc2, spec)
l_b = q2._step(enc2.input_ids[:, enc2.split:], cache, capture=True); biased = STATE.captured.clone()
q2._activate(enc2, None)
print("captured-layer-only bias: max |captured_biased - captured_clean| =", float((biased - clean).abs().max()))

print("\n== (c) per-modality percentile averaged across two channels")
rng = np.random.default_rng(0)
cal = relevance.Calibrator({"image_clip": list(rng.normal(size=5000)), "image_cap": list(rng.normal(size=5000)),
                            "text": list(rng.normal(size=5000))})
vals_mix = [cal.relevance({"image_clip": float(a), "image_cap": float(b)})
            for a, b in zip(rng.normal(size=4000), rng.normal(size=4000))]
vals_text = [cal.relevance({"text": float(a)}) for a in rng.normal(size=4000)]
vals_clip = [cal.relevance({"image_clip": float(a)}) for a in rng.normal(size=4000)]
for nm, v in (("text (single ECDF)", vals_text), ("image clip-only (single ECDF)", vals_clip),
              ("image mix (mean of two ECDFs)", vals_mix)):
    v = np.asarray(v)
    print(f"{nm:34s} mean={v.mean():.3f} std={v.std():.3f} p05={np.quantile(v,.05):.3f} p95={np.quantile(v,.95):.3f}")

print("\n== (c2) correlated channels: does the compression survive?")
for rho in (0.0, 0.5, 0.9):
    a = rng.normal(size=6000)
    b = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(size=6000)
    cal2 = relevance.Calibrator({"image_clip": list(a), "image_cap": list(b)})
    xs = rng.normal(size=3000)
    ys = rho * xs + np.sqrt(1 - rho ** 2) * rng.normal(size=3000)
    v = np.asarray([cal2.relevance({"image_clip": float(x), "image_cap": float(y)}) for x, y in zip(xs, ys)])
    print(f"rho={rho:.1f}: mean={v.mean():.3f} std={v.std():.3f} p05={np.quantile(v,.05):.3f} p95={np.quantile(v,.95):.3f}")

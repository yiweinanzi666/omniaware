import sys, types, traceback
REPO = "C:/Users/22688/Desktop/www_code/omniaware"
sys.path[:0] = [f"{REPO}/src", f"{REPO}/tests", REPO]
import torch
from tiny_qwen import image_prompt, tiny_model
from test_fitm_qwen_adapter import model18, tokenizer64
from omniaware.baselines import arms
from omniaware.baselines.fitm_qwen_adapter import make_null_reference, document_indices, prepare_fitm_calibration

inputs = image_prompt()          # ids [2, 60(vision_start), 61(image), 62(vision_end), 6, 7]
print("input_ids:", inputs["input_ids"].tolist())
# Real layout from QwenEAB.encode: an image segment's span covers vision_start + image tokens + vision_end.
enc = types.SimpleNamespace(input_ids=inputs["input_ids"], spans={"i1": (1, 4)})
qwen = types.SimpleNamespace(image_token_id=61, spans=None)
sources = arms.fitm_sources(qwen, enc, ["i1"])
print("fitm_sources:", sources)
tok = tokenizer64()
ref, meta = make_null_reference(inputs, sources, 0, tok)
print("reference input_ids:", ref["input_ids"].tolist())
print("replacement tokens:", meta["replacement_token_ids"], "text idx:", meta["text_token_indices"])
model = model18()
with torch.inference_mode():
    try:
        out = model(**ref, use_cache=False)
        print("reference forward OK, logits finite:", torch.isfinite(out.logits).all().item())
    except Exception as e:
        print("reference forward FAILED:", type(e).__name__, str(e)[:300])
    try:
        alpha, rec = prepare_fitm_calibration(model, inputs, sources, tok)
        print("calibration OK")
    except Exception as e:
        print("calibration FAILED:", type(e).__name__, str(e)[:300])

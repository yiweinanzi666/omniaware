import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, "C:/Users/22688/Desktop/www_code/omniaware/src")
import numpy as np
from omniaware import relevance, run

calls = {"n": 0}
class FakeBGE:
    def __init__(self, *a, **k): pass
    def encode(self, texts, query=False, batch=32):
        calls["n"] += len(texts)
        # embedding depends on the text length so a changed text gives a different score
        return np.array([[len(t) / 100.0, 1.0] for t in texts], dtype=np.float32)
relevance.BGE = FakeBGE
relevance.CLIP = None

def ex(text_k1, extra=False):
    cands = [{"cid": "k1", "modality": "text", "title": "", "text": text_k1, "image": None, "gold": None}]
    if extra:
        cands.append({"cid": "k6", "modality": "text", "title": "", "text": "new passage", "image": None, "gold": None})
    return [{"qid": "q1", "question": "what?", "candidates": cands, "query_images": [], "options": None}]

tmp = Path(tempfile.mkdtemp()) / "relevance_okvqa_dense.json"
r1 = run.load_relevance(ex("short passage"), tmp)
print("first run raw k1:", r1["q1"]["k1"])
# corpus / candidate file changed: same qid and cid, different text (e.g. regenerated candidates or new env var path)
r2 = run.load_relevance(ex("a much much longer replacement passage text here"), tmp)
print("second run raw k1 (text changed):", r2["q1"]["k1"], "| BGE calls total:", calls["n"])
# candidate list grew: new cid k6 is not in the cache -> KeyError downstream (rel_of uses raw[qid][cid])
r3 = run.load_relevance(ex("a much much longer replacement passage text here", extra=True), tmp)
try:
    _ = r3["q1"]["k6"]
    print("k6 found")
except KeyError as err:
    print("KeyError for new candidate k6 (crash, not silent):", err)
print("cache file keys:", list(json.loads(tmp.read_text()).keys()))

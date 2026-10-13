import sys, json, hashlib, tempfile, types
from pathlib import Path
REPO = Path(r"C:\Users\22688\Desktop\www_code\omniaware")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))

# 1) download_models skip on a partially downloaded directory (config.json present, weights missing)
import importlib
dm = importlib.import_module("scripts.setup.download_models")
tmp = Path(tempfile.mkdtemp()) / "bge"; tmp.mkdir(parents=True)
(tmp / "config.json").write_text("{}")                        # first file of an interrupted snapshot
calls = []
fake = types.ModuleType("huggingface_hub")
fake.snapshot_download = lambda **kw: calls.append(kw)
sys.modules["huggingface_hub"] = fake
dm.hf_download("BAAI/bge-large-en-v1.5", "d4aa", tmp, ["model.safetensors"])
print("1) snapshot_download called on partial dir:", bool(calls), "(False = skipped, model.safetensors never fetched)")

# 2) pool key: current prepare_webqa_open vs legacy exact-body key
import scripts.setup.prepare_webqa_open as pw
from omniaware import data
out = Path(tempfile.mkdtemp()) / "open"; out.mkdir()
src = Path(tempfile.mkdtemp())
webqa = {
 "a": {"split": "train", "Q": "q", "Qcate": "text", "A": ["x"], "txt_posFacts": [
        {"snippet_id": "s1", "title": "Paris", "fact": "The  tower is   in Paris."},
        {"snippet_id": "s2", "title": "Eiffel", "fact": "The tower is in Paris."},      # same body, other title
        {"snippet_id": "s3", "title": "Paris", "fact": "The tower is in Paris."}],       # same title+body as s1 after clean
       "txt_negFacts": [], "txt_Facts": [], "img_posFacts": [], "img_negFacts": []}}
(src / "WebQA_train_val.json").write_text(json.dumps(webqa))
pw.OUT = out
import omniaware.paths as P; P.WEBQA_OPEN = out
pw.build_pool(src)
docs = [json.loads(l) for l in (out / "text_documents.jsonl").read_text().splitlines()]
print("2) current pool docs:", len(docs), [d["text"] for d in docs])
bodies = {}
for f in webqa["a"]["txt_posFacts"]:
    bodies.setdefault(f["fact"], set()).add(f["snippet_id"])
print("   legacy exact-body keying (no normalisation) would give:", len(bodies), "documents")
print("   (current key = _clean(title)+': '+_clean(body); legacy = exact body, title only as metadata)")

"""prepare_webqa_open builds exactly what data.load_webqa_open reads (stand-in encoders, CPU)."""
import base64
import io
import json

import numpy as np
from PIL import Image

from omniaware import data, paths, relevance
from scripts.setup import prepare_webqa_open as pw


class FakeBGE:
    def encode(self, texts, query=False, batch=32):
        out = np.array([[1.0 if "Paris" in t else 0.0, 1.0 if "cat" in t else 0.0, 0.1] for t in texts])
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class FakeCLIP:
    def encode_text(self, texts):
        return FakeBGE().encode(texts)

    def encode_images(self, image_paths, batch=16):
        return FakeBGE().encode(["Paris" if "300" in p else "cat" for p in image_paths])


def test_build_pool_images_encode_search_then_load(tmp_path, monkeypatch):
    webqa = {"g1": {"split": "val", "Q": "\"Where is the Eiffel tower, Paris?\"", "Qcate": "text", "A": ["\"Paris.\""],
                    "txt_posFacts": [{"snippet_id": "s1", "title": "Paris", "fact": "The tower is in Paris."}],
                    "txt_negFacts": [{"snippet_id": "s2", "title": "Cats", "fact": "A cat sleeps."}],
                    "img_posFacts": [{"image_id": 300, "caption": "Eiffel tower Paris"}],
                    "img_negFacts": [{"image_id": 301, "caption": "A cat"}]},
             "g2": {"split": "train", "Q": "What sleeps? cat", "Qcate": "text", "A": ["A cat."],
                    "txt_posFacts": [{"snippet_id": "s2", "title": "Cats", "fact": "A cat sleeps."}],
                    "txt_negFacts": [], "img_posFacts": [], "img_negFacts": []}}
    src = tmp_path / "webqa"
    src.mkdir()
    (src / "WebQA_train_val.json").write_text(json.dumps(webqa))
    lines = []
    for iid in (300, 301):
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (iid % 255, 0, 0)).save(buf, format="JPEG")
        lines.append(f"{iid}\t{base64.b64encode(buf.getvalue()).decode()}")
    (src / "imgs.tsv").write_text("\n".join(lines) + "\n")
    out = tmp_path / "open"
    monkeypatch.setattr(pw, "OUT", out)
    monkeypatch.setattr(paths, "WEBQA_OPEN", out)
    monkeypatch.setattr(paths, "WEBQA_JSON", src / "WebQA_train_val.json")
    monkeypatch.setattr(relevance, "BGE", FakeBGE)
    monkeypatch.setattr(relevance, "CLIP", FakeCLIP)

    pw.build_pool(src)
    docs = [json.loads(l) for l in (out / "text_documents.jsonl").read_text().splitlines()]
    assert len(docs) == 2 and sorted(d["source_snippet_ids"] for d in docs) == [["s1"], ["s2"]]
    pw.extract_images(src, workers=1)
    assert (out / "images" / "300.jpg").exists()
    for kind in ("text", "image"):
        for shard in range(2):
            pw.encode_shard(kind, shard, 2, batch=4)
    pw.search("val", top=2)
    pw.search("train", top=2)

    (e,) = data.load_webqa_open("all", split="val")
    by = {c["cid"]: c for c in e["candidates"]}
    assert by["t1"]["gold"] and by["t1"]["text"].startswith("Paris:")      # Paris passage ranked first
    assert by["i1"]["gold"] and by["i1"]["image"].endswith("300.jpg")
    assert e["golds"] == ["Paris."] and e["missing_images"] == 0
    (t,) = data.load_webqa_open("all", split="train")
    assert t["qid"] == "g2"

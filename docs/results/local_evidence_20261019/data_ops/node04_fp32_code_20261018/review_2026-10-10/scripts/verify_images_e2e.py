import sys, io, json, base64, tempfile, traceback
from pathlib import Path
REPO = Path(r"C:\Users\22688\Desktop\www_code\omniaware")
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
import numpy as np
from PIL import Image

def jpeg_b64(seed, truncate=False):
    buf = io.BytesIO(); Image.new("RGB", (64, 64), (seed % 255, 30, 30)).save(buf, format="JPEG", quality=95)
    raw = buf.getvalue()
    if truncate:
        raw = raw[: int(len(raw) * 0.6)]
    return base64.b64encode(raw).decode()

class FakeBGE:
    def encode(self, texts, query=False, batch=32):
        out = np.array([[1.0 if "Paris" in t else 0.0, 1.0 if "cat" in t else 0.0, 0.1] for t in texts])
        return out / np.linalg.norm(out, axis=1, keepdims=True)
class FakeCLIP:
    def encode_text(self, texts): return FakeBGE().encode(texts)
    def encode_images(self, image_paths, batch=16):
        return FakeBGE().encode(["Paris" if "300" in p else "cat" for p in image_paths])

def main():
    import scripts.setup.prepare_webqa_open as pw
    from omniaware import paths, relevance, data
    tmp = Path(tempfile.mkdtemp()); src = tmp / "webqa"; src.mkdir(); out = tmp / "open"
    webqa = {"g1": {"split": "val", "Q": "Where is Paris?", "Qcate": "text", "A": ["Paris"],
                    "txt_posFacts": [{"snippet_id": "s1", "title": "Paris", "fact": "The tower is in Paris."}],
                    "txt_negFacts": [], "img_posFacts": [{"image_id": 300, "caption": "Paris"}],
                    "img_negFacts": [{"image_id": 301, "caption": "A cat"}]}}
    (src / "WebQA_train_val.json").write_text(json.dumps(webqa))
    # 301 is in the pool but absent from imgs.tsv; 300 is fine
    (src / "imgs.tsv").write_text(f"300\t{jpeg_b64(300)}\n")
    pw.OUT = out; paths.WEBQA_OPEN = out; paths.WEBQA_JSON = src / "WebQA_train_val.json"
    relevance.BGE = FakeBGE; relevance.CLIP = FakeCLIP
    pw.build_pool(src)
    print("== A. pool absent-from-TSV case")
    pw.extract_images(src, workers=1)
    pw.encode_shard("text", 0, 1, batch=4); pw.encode_shard("image", 0, 1, batch=4)
    pw.search("val", top=1)
    rows = [json.loads(l) for l in (out / "dense_top5_full/webqa_val/clip/rankings.jsonl").read_text().splitlines()]
    print("   CLIP ranked ids:", [c["source_id"] for c in rows[0]["candidates"]])
    (e,) = data.load_webqa_open("all", split="val")
    print("   loader missing_images =", e["missing_images"], "; image candidates:", [c["cid"] for c in e["candidates"] if c["modality"]=="image"])

    print("== B. truncated JPEG in imgs.tsv")
    src2 = tmp / "webqa2"; src2.mkdir(); out2 = tmp / "open2"
    (src2 / "WebQA_train_val.json").write_text(json.dumps(webqa))
    (src2 / "imgs.tsv").write_text(f"300\t{jpeg_b64(300)}\n301\t{jpeg_b64(301, truncate=True)}\n")
    pw.OUT = out2; paths.WEBQA_OPEN = out2
    pw.build_pool(src2)
    try:
        pw.extract_images(src2, workers=1)
        print("   extraction completed")
    except Exception as ex:
        print("   extract_images aborted:", type(ex).__name__, str(ex)[:80])
    print("   files written:", sorted(p.name for p in (out2/"images").glob("*")))

if __name__ == "__main__":
    main()

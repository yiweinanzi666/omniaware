"""Build the WebQA open-retrieval artifacts that ``data.load_webqa_open`` reads, from the official release.

Inputs (WebQA README, https://github.com/WebQnA/WebQA): ``WebQA_train_val.json``, ``WebQA_test.json`` (optional,
adds its facts to the pool) and the image file ``imgs.tsv`` (``image_id<TAB>base64`` per line).

    python -m scripts.setup.prepare_webqa_open pool   --webqa-dir /data/webqa
    python -m scripts.setup.prepare_webqa_open images --webqa-dir /data/webqa [--workers 32]
    python -m scripts.setup.prepare_webqa_open encode --kind text  --gpus 0,1,2,3
    python -m scripts.setup.prepare_webqa_open encode --kind image --gpus 0,1,2,3
    python -m scripts.setup.prepare_webqa_open search --split val      # and --split train (calibration)

Output under paths.WEBQA_OPEN (env OMNIAWARE_WEBQA_OPEN_DIR):
  text_documents.jsonl   {"document_id": "text-sha256:<hex>", "text", "source_snippet_ids"}  (deduplicated)
  image_corpus.jsonl     {"fact_id": "image:<id>", "image_id", "captions"}
  images/<id>.jpg, images/assets.jsonl, faiss_clip_full/plan.json
  emb/{text,image}.shardXX-of-NN.{npy,ids.json}           (fp16, L2-normalised)
  dense_top5_full/webqa_<split>/{bge,clip}/rankings.jsonl  {"qid", "candidates": [{"rank", "source_id", "score"}]}
Search is exact inner product over all normalised embeddings (the result of a FAISS IndexFlatIP).
Text passages are "<title>: <fact>"; text queries use the BGE retrieval instruction, image queries CLIP text.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths  # noqa: E402
from omniaware.data import _clean  # noqa: E402  (the loader's own normalisation)

OUT = paths.WEBQA_OPEN


def text_key(title, fact):
    body = _clean(fact or "")
    return f"{_clean(title or '')}: {body}" if _clean(title or "") else body


# ------------------------------------------------------------------- pool --
def build_pool(webqa_dir: Path) -> None:
    files = [webqa_dir / "WebQA_train_val.json"] + ([webqa_dir / "WebQA_test.json"]
                                                     if (webqa_dir / "WebQA_test.json").exists() else [])
    texts, images = {}, {}
    for f in files:
        for v in json.loads(f.read_text(encoding="utf8")).values():
            for key in ("txt_posFacts", "txt_negFacts", "txt_Facts"):
                for fact in v.get(key, []):
                    text = text_key(fact.get("title"), fact.get("fact"))
                    doc = "text-sha256:" + hashlib.sha256(text.encode("utf8")).hexdigest()
                    entry = texts.setdefault(doc, {"document_id": doc, "text": text, "source_snippet_ids": []})
                    sid = str(fact.get("snippet_id"))
                    if sid not in entry["source_snippet_ids"]:
                        entry["source_snippet_ids"].append(sid)
            for key in ("img_posFacts", "img_negFacts", "img_Facts"):
                for fact in v.get(key, []):
                    iid = str(fact["image_id"])
                    entry = images.setdefault(iid, {"fact_id": f"image:{iid}", "image_id": iid, "captions": []})
                    cap = _clean(fact.get("caption") or fact.get("title") or "")
                    if cap and cap not in entry["captions"]:
                        entry["captions"].append(cap)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "text_documents.jsonl").open("w", encoding="utf8") as fh:
        for i, row in enumerate(texts.values()):
            fh.write(json.dumps({"row_index": i, **row}, ensure_ascii=False) + "\n")
    with (OUT / "image_corpus.jsonl").open("w", encoding="utf8") as fh:
        for row in images.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[pool] {len(texts)} text documents, {len(images)} images from {[f.name for f in files]} -> {OUT}")


# ----------------------------------------------------------------- images --
def _decode(args):
    iid, b64, out_dir = args
    from PIL import Image
    path = Path(out_dir) / f"{iid}.jpg"
    if not path.exists():
        Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").save(path, quality=95)
    return iid, str(path)


def extract_images(webqa_dir: Path, workers: int) -> None:
    wanted = {json.loads(l)["image_id"] for l in (OUT / "image_corpus.jsonl").open(encoding="utf8")}
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    assets = {}

    def jobs():
        with (webqa_dir / "imgs.tsv").open(encoding="utf8") as fh:
            for line in fh:
                iid, _, b64 = line.rstrip("\n").partition("\t")
                if iid in wanted:
                    yield iid, b64, str(img_dir)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for iid, path in pool.map(_decode, jobs(), chunksize=64):
            assets[iid] = path
    with (img_dir / "assets.jsonl").open("w") as fh:
        for iid, path in assets.items():
            fh.write(json.dumps({"image_id": iid, "path": path}) + "\n")
    (OUT / "faiss_clip_full").mkdir(exist_ok=True)
    (OUT / "faiss_clip_full" / "plan.json").write_text(json.dumps({"inputs": [{"directory": str(img_dir)}]}, indent=1))
    print(f"[images] {len(assets)} of {len(wanted)} pool images extracted -> {img_dir}")
    if len(assets) < len(wanted):
        print(f"[images] {len(wanted) - len(assets)} pool images are absent from imgs.tsv; the loader counts them")


# ----------------------------------------------------------------- encode --
def _items(kind):
    if kind == "text":
        return [(r["document_id"], r["text"]) for r in map(json.loads, (OUT / "text_documents.jsonl").open(encoding="utf8"))]
    return [("image:" + r["image_id"], r["path"]) for r in map(json.loads, (OUT / "images" / "assets.jsonl").open())]


def encode_shard(kind: str, shard: int, num_shards: int, batch: int) -> None:
    import numpy as np
    from omniaware import relevance
    items = _items(kind)[shard::num_shards]
    name = f"{kind}.shard{shard:02d}-of-{num_shards:02d}"
    npy, ids_path = OUT / "emb" / f"{name}.npy", OUT / "emb" / f"{name}.ids.json"
    if npy.exists():
        print(f"[encode] {name} exists", flush=True)
        return
    npy.parent.mkdir(parents=True, exist_ok=True)
    enc = relevance.BGE() if kind == "text" else relevance.CLIP()
    vecs = []
    for i in range(0, len(items), batch):
        chunk = [x for _, x in items[i:i + batch]]
        vecs.append((enc.encode(chunk, batch=batch) if kind == "text" else enc.encode_images(chunk, batch=batch))
                    .astype(np.float16))
        if (i // batch) % 50 == 0:
            print(f"[encode] {kind} shard {shard}: {i + len(chunk)}/{len(items)}", flush=True)
    ids_path.write_text(json.dumps([sid for sid, _ in items]))
    np.save(npy, np.concatenate(vecs) if vecs else np.zeros((0, 1), dtype=np.float16))


def encode(kind: str, gpus: list[str], batch: int) -> None:
    if len(gpus) <= 1:
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        encode_shard(kind, 0, 1, batch)
        return
    procs = [subprocess.Popen([sys.executable, "-m", "scripts.setup.prepare_webqa_open", "encode-shard", "--kind", kind,
                               "--shard", str(i), "--num-shards", str(len(gpus)), "--batch", str(batch)],
                              cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g)) for i, g in enumerate(gpus)]
    if any(p.wait() for p in procs):
        raise SystemExit("an encoding shard failed; rerun to resume (finished shards are kept)")


# ----------------------------------------------------------------- search --
def _load(kind):
    import numpy as np
    shards = sorted((OUT / "emb").glob(f"{kind}.shard*.npy"))
    if not shards:
        raise SystemExit(f"no {kind} embeddings under {OUT / 'emb'}; run encode first")
    totals = {s.name.split("-of-")[1] for s in shards}
    if len(totals) != 1:
        raise SystemExit(f"{kind} embeddings from different shard counts: {sorted(totals)}")
    ids = [i for s in shards for i in json.loads((s.parent / (s.name[:-4] + ".ids.json")).read_text())]
    arrays = [np.load(s) for s in shards if np.load(s, mmap_mode="r").shape[0]]
    return ids, np.concatenate(arrays)


def search(split: str, top: int, chunk: int = 4096) -> None:
    import torch
    from omniaware import relevance
    raw = json.loads(paths.WEBQA_JSON.read_text(encoding="utf8"))
    questions = [(guid, _clean(v["Q"])) for guid, v in raw.items() if v["split"] == split]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for kind, name in (("text", "bge"), ("image", "clip")):
        ids, emb = _load(kind)
        bank = torch.from_numpy(emb).to(device)
        enc = relevance.BGE() if kind == "text" else relevance.CLIP()
        texts = [q for _, q in questions]
        qv = enc.encode(texts, query=True) if kind == "text" else enc.encode_text(texts)
        out = OUT / "dense_top5_full" / f"webqa_{split}" / name / "rankings.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for s in range(0, len(questions), chunk):
                q = torch.from_numpy(qv[s:s + chunk]).to(device, bank.dtype)
                scores, idx = (q @ bank.T).float().topk(top, dim=1)
                for (guid, _), sc, ix in zip(questions[s:s + chunk], scores.tolist(), idx.tolist()):
                    fh.write(json.dumps({"qid": guid, "candidates": [
                        {"rank": r, "source_id": ids[i], "score": round(v, 6)}
                        for r, (v, i) in enumerate(zip(sc, ix), 1)]}) + "\n")
        print(f"[search] {split}/{name}: {len(questions)} questions x {len(ids)} items, top {top} -> {out}", flush=True)
        del bank


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pool", "images"):
        p = sub.add_parser(name)
        p.add_argument("--webqa-dir", type=Path, required=True)
        p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    e = sub.add_parser("encode")
    e.add_argument("--kind", choices=["text", "image"], required=True)
    e.add_argument("--gpus", default="auto")
    e.add_argument("--batch", type=int, default=256)
    es = sub.add_parser("encode-shard")
    es.add_argument("--kind", choices=["text", "image"], required=True)
    es.add_argument("--shard", type=int, required=True)
    es.add_argument("--num-shards", type=int, required=True)
    es.add_argument("--batch", type=int, default=256)
    s = sub.add_parser("search")
    s.add_argument("--split", choices=["val", "train"], required=True)
    s.add_argument("--top", type=int, default=5)
    args = ap.parse_args(argv)
    if args.cmd == "pool":
        build_pool(args.webqa_dir)
    elif args.cmd == "images":
        extract_images(args.webqa_dir, args.workers)
    elif args.cmd == "encode":
        from scripts.launch import detect_gpus
        gpus = detect_gpus() if args.gpus == "auto" else [g for g in args.gpus.split(",") if g]
        encode(args.kind, gpus, args.batch)
    elif args.cmd == "encode-shard":
        encode_shard(args.kind, args.shard, args.num_shards, args.batch)
    else:
        search(args.split, args.top)


if __name__ == "__main__":
    main()

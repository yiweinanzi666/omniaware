"""Build the WebQA open-retrieval artifacts that ``data.load_webqa_open`` reads, from the official release.

Inputs (WebQA README, https://github.com/WebQnA/WebQA): ``WebQA_train_val.json``, ``WebQA_test.json`` (optional,
adds its facts to the pool) and the image file ``imgs.tsv`` (``image_id<TAB>base64`` per line).

    python -m scripts.setup.prepare_webqa_open pool   --webqa-dir /data/webqa
    python -m scripts.setup.prepare_webqa_open images --webqa-dir /data/webqa [--workers 32] [--batch-size 2000]
    python -m scripts.setup.prepare_webqa_open encode --kind text  --gpus 0,1,2,3
    python -m scripts.setup.prepare_webqa_open encode --kind image --gpus 0,1,2,3
    python -m scripts.setup.prepare_webqa_open search --split val [--top 5] [--chunk 512]   # and --split train

Output under paths.WEBQA_OPEN (env OMNIAWARE_WEBQA_OPEN_DIR):
  text_documents.jsonl   {"document_id": "text-sha256:<hex>", "text", "source_snippet_ids"}  (deduplicated)
  image_corpus.jsonl     {"fact_id": "image:<id>", "image_id", "captions"}
  pool_stats.json        pool counts (texts, images) and the sha256 of the WebQA JSON files they came from;
                         the paper's pool numbers are read from here, and search refuses another WebQA JSON
  images/<id>.jpg, images/assets.jsonl                   (paths relative to OMNIAWARE_ASSETS)
  images/missing_images.jsonl, images/bad_images.jsonl   pool images not in imgs.tsv / not decodable (not indexed)
  faiss_clip_full/plan.json                              index inputs, relative to OMNIAWARE_ASSETS
  emb/{text,image}.shardXX-of-NN.{npy,ids.json,done}     fp16, L2-normalised; .done is written last
  emb/{text,image}.manifest.json                         shard layout, row counts and sha256 of every shard
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
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths  # noqa: E402
from omniaware.data import _clean  # noqa: E402  (the loader's own normalisation)

OUT = paths.WEBQA_OPEN


def text_key(title, fact):
    body = _clean(fact or "")
    return f"{_clean(title or '')}: {body}" if _clean(title or "") else body


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, write) -> None:
    """Write through ``<name>.tmp`` and rename, so an interrupted run never leaves a truncated file behind."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        write(fh)
    os.replace(tmp, path)


def _asset_rel(path) -> str:
    """Path relative to OMNIAWARE_ASSETS, which is what the index records (absolute only across drives)."""
    try:
        return Path(os.path.relpath(path, paths.ASSET_ROOT)).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def _asset_abs(recorded: str) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else paths.ASSET_ROOT / path


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
    stats = {"text_documents": len(texts), "image_corpus": len(images),
             "sources": {f.name: _sha256(f) for f in files},
             "text_key": "cleaned title + ': ' + cleaned fact (title omitted when empty), deduplicated on that string"}
    (OUT / "pool_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf8")
    print(f"[pool] {len(texts)} text documents, {len(images)} images from {[f.name for f in files]} -> {OUT}")
    print(f"[pool] statistics written to {OUT / 'pool_stats.json'} (the source of the pool numbers in the paper)")


def _pool_stats() -> dict:
    path = OUT / "pool_stats.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run the pool step first")
    return json.loads(path.read_text(encoding="utf8"))


# ----------------------------------------------------------------- images --
def _decode(args):
    """Decode one base64 image to ``<iid>.jpg``.  Returns (iid, path, error): a corrupt or truncated image is
    reported, never raised, so one bad file cannot abort the extraction.  Written via a temporary file."""
    iid, b64, out_dir = args
    from PIL import Image
    path = Path(out_dir) / f"{iid}.jpg"
    if path.exists():
        return iid, str(path), None
    tmp = path.with_name(path.name + ".tmp")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")   # convert() loads the pixels
        image.save(tmp, format="JPEG", quality=95)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - recorded as a bad image
        tmp.unlink(missing_ok=True)
        return iid, None, f"{type(exc).__name__}: {exc}"
    return iid, str(path), None


def _batched(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_jsonl(path: Path, rows) -> None:
    _write_atomic(path, lambda fh: fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                                            .encode("utf8")))


def extract_images(webqa_dir: Path, workers: int, batch_size: int = 2000) -> None:
    """Pool images from imgs.tsv.  Jobs are submitted in batches of ``batch_size`` lines, so only one batch of
    base64 strings is held by the parent at a time."""
    order = [json.loads(l)["image_id"] for l in (OUT / "image_corpus.jsonl").open(encoding="utf8")]
    wanted = set(order)
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    assets, bad, seen = {}, {}, set()

    def jobs():
        with (webqa_dir / "imgs.tsv").open(encoding="utf8") as fh:
            for line in fh:
                iid, _, b64 = line.rstrip("\n").partition("\t")
                if iid in wanted and iid not in seen:
                    seen.add(iid)
                    yield iid, b64, str(img_dir)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in _batched(jobs(), batch_size):
            for iid, path, error in pool.map(_decode, batch, chunksize=64):
                if error is None:
                    assets[iid] = path
                else:
                    bad[iid] = error
    missing = [iid for iid in order if iid not in assets and iid not in bad]
    _write_jsonl(img_dir / "assets.jsonl", [{"image_id": iid, "path": _asset_rel(assets[iid])}
                                            for iid in order if iid in assets])
    _write_jsonl(img_dir / "missing_images.jsonl", [{"image_id": iid} for iid in missing])
    _write_jsonl(img_dir / "bad_images.jsonl", [{"image_id": iid, "error": bad[iid]} for iid in order if iid in bad])
    (OUT / "faiss_clip_full").mkdir(exist_ok=True)
    _write_atomic(OUT / "faiss_clip_full" / "plan.json", lambda fh: fh.write(json.dumps(
        {"inputs": [{"directory": _asset_rel(img_dir)}]}, indent=1).encode("utf8")))
    print(f"[images] {len(assets)} of {len(order)} pool images extracted and indexed -> {img_dir}")
    print(f"[images] missing from imgs.tsv: {len(missing)} (not indexed; list in {img_dir / 'missing_images.jsonl'})")
    print(f"[images] undecodable (truncated or corrupt): {len(bad)} (not indexed; list in "
          f"{img_dir / 'bad_images.jsonl'})")
    if missing or bad:
        print("[images] the candidate image pool is smaller than the pool statistics; report the indexed count "
              "together with pool_stats.json", flush=True)


# ----------------------------------------------------------------- encode --
def _items(kind):
    if kind == "text":
        return [(r["document_id"], r["text"]) for r in map(json.loads, (OUT / "text_documents.jsonl").open(encoding="utf8"))]
    return [("image:" + r["image_id"], str(_asset_abs(r["path"])))
            for r in map(json.loads, (OUT / "images" / "assets.jsonl").open(encoding="utf8"))]


def _shard_name(kind, shard, num_shards):
    return f"{kind}.shard{shard:02d}-of-{num_shards:02d}"


def encode_shard(kind: str, shard: int, num_shards: int, batch: int) -> None:
    import numpy as np
    from omniaware import relevance
    items = _items(kind)[shard::num_shards]
    name = _shard_name(kind, shard, num_shards)
    npy, ids_path = OUT / "emb" / f"{name}.npy", OUT / "emb" / f"{name}.ids.json"
    done = OUT / "emb" / f"{name}.done"
    if done.exists():
        print(f"[encode] {name} done", flush=True)
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
    arr = np.concatenate(vecs) if vecs else np.zeros((0, 1), dtype=np.float16)
    _write_atomic(ids_path, lambda fh: fh.write(json.dumps([sid for sid, _ in items]).encode("utf8")))
    _write_atomic(npy, lambda fh: np.save(fh, arr))
    done.write_text(json.dumps({"rows": int(arr.shape[0]), "dim": int(arr.shape[1])}))   # last: marks the shard complete


def write_manifest(kind: str, num_shards: int) -> None:
    """After every shard is done: record the shard layout, the row counts and the sha256 of each shard.
    The item total must equal the number of items the step encodes (no silently dropped rows)."""
    total = len(_items(kind))
    shards, rows = [], 0
    for shard in range(num_shards):
        name = _shard_name(kind, shard, num_shards)
        done = OUT / "emb" / f"{name}.done"
        if not done.exists():
            raise SystemExit(f"{name} is not finished (no .done marker); rerun encode")
        info = json.loads(done.read_text())
        shards.append({"index": shard, "file": f"{name}.npy", "rows": info["rows"], "dim": info["dim"],
                       "sha256": _sha256(OUT / "emb" / f"{name}.npy")})
        rows += info["rows"]
    if rows != total:
        raise SystemExit(f"{kind} shards hold {rows} rows but the pool has {total} items; rerun encode")
    manifest = {"kind": kind, "num_shards": num_shards, "items": total, "shards": shards}
    _write_atomic(OUT / "emb" / f"{kind}.manifest.json",
                  lambda fh: fh.write(json.dumps(manifest, indent=1).encode("utf8")))
    print(f"[encode] {kind}: {total} items in {num_shards} shard(s) -> {OUT / 'emb' / f'{kind}.manifest.json'}")


def _run_shards(kind: str, gpus: list[str], batch: int) -> None:
    """One process per GPU.  Waits for every child; on the first failure the remaining children are killed and
    the command fails.  Finished shards keep their .done marker, so rerunning resumes."""
    procs = [subprocess.Popen([sys.executable, "-m", "scripts.setup.prepare_webqa_open", "encode-shard", "--kind", kind,
                               "--shard", str(i), "--num-shards", str(len(gpus)), "--batch", str(batch)],
                              cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g)) for i, g in enumerate(gpus)]
    codes: dict[int, int] = {}
    killed: set[int] = set()
    try:
        while len(codes) < len(procs):
            for i, p in enumerate(procs):
                if i not in codes and p.poll() is not None:
                    codes[i] = p.returncode
            if any(c != 0 for c in codes.values()):
                for i, p in enumerate(procs):
                    if i not in codes:
                        p.kill()
                        killed.add(i)
                for i, p in enumerate(procs):
                    if i not in codes:
                        codes[i] = p.wait()
                break
            if len(codes) < len(procs):
                time.sleep(1)
    except BaseException:
        for p in procs:
            if p.poll() is None:
                p.kill()
        raise
    failed = sorted(i for i, c in codes.items() if c != 0 and i not in killed)
    if failed:
        raise SystemExit(f"encoding shard(s) {failed} failed (exit codes {[codes[i] for i in failed]}); the other "
                         "shards were stopped. Rerun the same command: finished shards are kept.")


def encode(kind: str, gpus: list[str], batch: int) -> None:
    if len(gpus) <= 1:
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        encode_shard(kind, 0, 1, batch)
        write_manifest(kind, 1)
        return
    _run_shards(kind, gpus, batch)
    write_manifest(kind, len(gpus))


# ----------------------------------------------------------------- search --
def _load(kind):
    """Embeddings and ids of one kind, checked against the manifest: shard numbers 0..N-1 all present, every
    shard finished, sha256 and row counts equal to the manifest, and the total equal to the pool size."""
    import numpy as np
    manifest_path = OUT / "emb" / f"{kind}.manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no {kind} embeddings manifest at {manifest_path}; run encode first")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    n = manifest["num_shards"]
    pattern = re.compile(rf"^{kind}\.shard(\d{{2}})-of-(\d{{2}})\.npy$")
    found = sorted(int(m.group(1)) for p in (OUT / "emb").glob(f"{kind}.shard*.npy")
                   if (m := pattern.match(p.name)) and int(m.group(2)) == n)
    if found != list(range(n)):
        raise SystemExit(f"{kind} shards present {found} do not cover 0..{n - 1}; rerun encode")
    ids, arrays = [], []
    for s in sorted(manifest["shards"], key=lambda s: s["index"]):
        name = _shard_name(kind, s["index"], n)
        npy, ids_path = OUT / "emb" / f"{name}.npy", OUT / "emb" / f"{name}.ids.json"
        if not (OUT / "emb" / f"{name}.done").exists():
            raise SystemExit(f"{name} is incomplete (no .done marker); rerun encode")
        if _sha256(npy) != s["sha256"]:
            raise SystemExit(f"{name} differs from the manifest (sha256); rerun encode")
        shard_ids = json.loads(ids_path.read_text(encoding="utf8"))
        arr = np.load(npy)
        if arr.shape[0] != s["rows"] or len(shard_ids) != arr.shape[0]:
            raise SystemExit(f"{name}: {arr.shape[0]} vectors, {len(shard_ids)} ids, manifest {s['rows']}; rerun encode")
        ids.extend(shard_ids)
        if arr.shape[0]:
            arrays.append(arr)
    if len(ids) != manifest["items"]:
        raise SystemExit(f"{kind}: {len(ids)} ids but the manifest says {manifest['items']}")
    return ids, np.concatenate(arrays)


def _check_webqa_json(path: Path) -> None:
    """search must read the same WebQA JSON the pool was built from."""
    recorded = _pool_stats()["sources"].get("WebQA_train_val.json")
    if recorded is None:
        raise SystemExit("pool_stats.json has no WebQA_train_val.json record; rerun the pool step")
    if _sha256(path) != recorded:
        raise SystemExit(f"{path} differs from the WebQA JSON the pool was built from (sha256 mismatch); "
                         "use the same file for pool and search")


def search(split: str, top: int, chunk: int = 512, webqa_json: Path | None = None) -> None:
    import torch
    from omniaware import relevance
    webqa_json = Path(webqa_json) if webqa_json else paths.WEBQA_JSON
    _check_webqa_json(webqa_json)
    raw = json.loads(webqa_json.read_text(encoding="utf8"))
    questions = [(guid, _clean(v["Q"])) for guid, v in raw.items() if v["split"] == split]
    if not questions:
        raise SystemExit(f"no {split} questions in {webqa_json}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for kind, name in (("text", "bge"), ("image", "clip")):
        ids, emb = _load(kind)
        if top > len(ids):
            raise SystemExit(f"--top {top} exceeds the {kind} index ({len(ids)} items); use a smaller --top")
        bank = torch.from_numpy(emb).to(device)
        enc = relevance.BGE() if kind == "text" else relevance.CLIP()
        texts = [q for _, q in questions]
        qv = enc.encode(texts, query=True) if kind == "text" else enc.encode_text(texts)
        out = OUT / "dense_top5_full" / f"webqa_{split}" / name / "rankings.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[search] {kind}: a chunk of {chunk} queries x {len(ids)} items needs about "
              f"{chunk * len(ids) * 6 / 1e9:.1f} GB for fp16 scores plus their fp32 copy; lower --chunk on small GPUs",
              flush=True)
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
    p = sub.add_parser("pool")
    p.add_argument("--webqa-dir", type=Path, required=True)
    p = sub.add_parser("images")
    p.add_argument("--webqa-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    p.add_argument("--batch-size", type=int, default=2000, help="imgs.tsv lines submitted to the pool at a time")
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
    s.add_argument("--chunk", type=int, default=512, help="queries scored per matrix product (GPU memory)")
    s.add_argument("--webqa-dir", type=Path, default=None,
                   help="directory with the WebQA JSON the pool was built from (default: paths.WEBQA_JSON)")
    args = ap.parse_args(argv)
    if args.cmd == "pool":
        build_pool(args.webqa_dir)
    elif args.cmd == "images":
        extract_images(args.webqa_dir, args.workers, args.batch_size)
    elif args.cmd == "encode":
        from scripts.launch import detect_gpus
        gpus = detect_gpus() if args.gpus == "auto" else [g for g in args.gpus.split(",") if g]
        encode(args.kind, gpus, args.batch)
    elif args.cmd == "encode-shard":
        encode_shard(args.kind, args.shard, args.num_shards, args.batch)
    else:
        search(args.split, args.top, args.chunk,
               args.webqa_dir / "WebQA_train_val.json" if args.webqa_dir else None)


if __name__ == "__main__":
    main()

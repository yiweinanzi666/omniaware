"""Dense BGE retrieval for OK-VQA over a passage corpus.

Corpora: 'wiki3m' (the first 3M DPR Wikipedia passages, matching the original e5_Flat.index setup) and
'gs112k', the Google Search corpus of Luo et al. (EMNLP 2021) used across the OK-VQA retrieval literature
(full test corpus, 168,306 sentences; okvqa_full_corpus.csv of their release).

  python -m omniaware.okvqa_dense --corpus gs112k prepare --csv okvqa_full_corpus.csv   # CSV -> corpus.jsonl
  python -m omniaware.okvqa_dense --corpus gs112k encode [--limit N]     # passages -> fp16 memmap (resumable)
  python -m omniaware.okvqa_dense --corpus gs112k search --n-passages N  # top-k per question -> candidates.jsonl

Datasets okvqa_dense (wiki3m) and okvqa_gs (gs112k) read these candidates; every arm of a run shares them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import paths, relevance

CORPORA = {   # name -> (corpus.jsonl with {"id", "contents"}, output directory)
    "wiki3m": (paths.DATA / "okvqa_knowledge_dpr_first3m" / "corpus_wiki_first3m.jsonl", paths.LOCAL_CACHE / "okvqa_dense"),
    "gs112k": (paths.LOCAL_CACHE / "okvqa_gs112k" / "corpus.jsonl", paths.LOCAL_CACHE / "okvqa_gs112k"),
}
CORPUS, OUT = CORPORA["wiki3m"]


def use_corpus(name: str) -> None:
    global CORPUS, OUT
    CORPUS, OUT = CORPORA[name]


def prepare_gs112k(csv_path: Path) -> Path:
    """Luo et al.'s CSV (one knowledge sentence per row) -> corpus.jsonl; ids are the row order."""
    import csv
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(csv_path, newline="", encoding="utf8") as fh, CORPUS.open("w", encoding="utf8") as out:
        reader = csv.DictReader(fh)
        field = "text" if "text" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            text = " ".join((row[field] or "").split())
            if text:
                out.write(json.dumps({"id": n, "contents": text}, ensure_ascii=False) + "\n")
                n += 1
    print(f"[prepare] {n} passages from column '{field}' of {csv_path} -> {CORPUS}", flush=True)
    return CORPUS
DIM = 1024
# The original index (rebuild_index_wiki_3m.py, e5_Flat.index) used BGE-large weights with
# retrieval_method='e5': "passage: " / "query: " prefixes, CLS pooling, normalised, max_length 512.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def _offsets(limit: int | None) -> np.ndarray:
    path = OUT / "offsets.npy"
    if path.exists():
        off = np.load(path)
        if limit is None or len(off) >= limit:
            return off[:limit] if limit else off
    offs, pos = [], 0
    with open(CORPUS, "rb") as fh:
        for line in fh:
            offs.append(pos)
            pos += len(line)
            if limit and len(offs) >= limit:
                break
    off = np.asarray(offs, dtype=np.int64)
    if limit is None:
        np.save(path, off)
    return off


def _passage(fh, offset: int) -> dict:
    fh.seek(int(offset))
    return json.loads(fh.readline())


def encode(limit: int | None, batch: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    off = _offsets(limit)
    n = len(off)
    emb_path, done_path = OUT / f"emb_{n}.f16", OUT / f"emb_{n}.done"
    mode = "r+" if emb_path.exists() else "w+"
    emb = np.memmap(emb_path, dtype=np.float16, mode=mode, shape=(n, DIM))
    start = int(done_path.read_text()) if done_path.exists() else 0
    bge = relevance.BGE()
    if bge.device != "cpu":
        bge.model = bge.model.half()
    t0 = time.time()
    with open(CORPUS, "rb") as fh:
        fh.seek(int(off[start]) if start < n else 0)
        i = start
        while i < n:
            texts = []
            for _ in range(min(batch, n - i)):
                texts.append(PASSAGE_PREFIX + json.loads(fh.readline())["contents"])
            vec = bge.encode(texts, batch=batch)
            emb[i:i + len(texts)] = vec.astype(np.float16)
            i += len(texts)
            if (i // batch) % 40 == 0 or i == n:
                emb.flush()
                done_path.write_text(str(i))
                rate = (i - start) / max(time.time() - t0, 1e-6)
                print(f"[encode] {i}/{n} {rate:.0f} passages/s, eta {(n - i) / max(rate, 1e-6) / 3600:.2f} h", flush=True)
    emb.flush()
    done_path.write_text(str(n))


def search(n_passages: int, k: int, limit_queries: int | None, chunk: int, tag: str) -> Path:
    from .data import load_okvqa

    emb = np.memmap(OUT / f"emb_{n_passages}.f16", dtype=np.float16, mode="r", shape=(n_passages, DIM))
    done = int((OUT / f"emb_{n_passages}.done").read_text())
    assert done == n_passages, f"encoding incomplete: {done}/{n_passages}"
    examples = load_okvqa()[:limit_queries]
    bge = relevance.BGE()
    q = torch.from_numpy(bge.encode([QUERY_PREFIX + e["question"] for e in examples])).to(bge.device, torch.float16)
    best_s = torch.full((len(examples), k), -1e4, device=bge.device)
    best_i = torch.zeros((len(examples), k), dtype=torch.long, device=bge.device)
    for s in range(0, n_passages, chunk):
        block = torch.from_numpy(np.asarray(emb[s:s + chunk])).to(bge.device)
        scores = (q @ block.T).float()
        top_s, top_i = scores.topk(min(k, scores.shape[1]), dim=1)
        cat_s = torch.cat([best_s, top_s], dim=1)
        cat_i = torch.cat([best_i, top_i + s], dim=1)
        best_s, idx = cat_s.topk(k, dim=1)
        best_i = cat_i.gather(1, idx)
    off = _offsets(n_passages)
    out_path = OUT / f"candidates_{tag}.jsonl"
    with open(CORPUS, "rb") as fh, out_path.open("w") as out:
        for e, ss, ii in zip(examples, best_s.cpu().tolist(), best_i.cpu().tolist()):
            cands = []
            for rank, (score, doc) in enumerate(zip(ss, ii), 1):
                p = _passage(fh, off[doc])
                cands.append({"rank": rank, "doc_id": int(p["id"]), "fact_id": f"{OUT.name}:{p['id']}",
                              "text": p["contents"], "bge_score": round(float(score), 5)})
            out.write(json.dumps({"question_id": e["qid"], "candidates": cands}, ensure_ascii=False) + "\n")
    print(f"[search] {len(examples)} queries over {n_passages} passages -> {out_path}", flush=True)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="wiki3m")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("prepare")
    c.add_argument("--csv", type=Path, required=True)
    a = sub.add_parser("encode")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--batch", type=int, default=128)
    b = sub.add_parser("search")
    b.add_argument("--n-passages", type=int, default=3_000_000)
    b.add_argument("--k", type=int, default=20)
    b.add_argument("--limit-queries", type=int, default=None)
    b.add_argument("--chunk", type=int, default=200_000)
    b.add_argument("--tag", default="full")
    args = ap.parse_args(argv)
    use_corpus(args.corpus)
    if args.cmd == "prepare":
        if args.corpus != "gs112k":
            ap.error("prepare converts the GS-112K CSV")
        prepare_gs112k(args.csv)
    elif args.cmd == "encode":
        encode(args.limit, args.batch)
    else:
        search(args.n_passages, args.k, args.limit_queries, args.chunk, args.tag)


if __name__ == "__main__":
    main()

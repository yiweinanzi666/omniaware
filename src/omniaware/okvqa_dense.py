"""Dense BGE retrieval for OK-VQA over a passage corpus.

Corpora (``--corpus``):
  wiki3m       the first 3M DPR Wikipedia passages (psgs_w100 prefix); the original e5_Flat.index setup.
  gs112k       the Google Search corpus of Luo et al. (EMNLP 2021) collected from the TRAINING questions only:
               okvqa_train_corpus.csv, 112,724 sentences. Default corpus of the okvqa_gs dataset and the
               only GS corpus used for dev selection.
  gs168k_full  okvqa_full_corpus.csv, 168,306 sentences. Luo et al. collected it from the training AND the test
               questions, so it contains retrieval results for evaluation questions. Optional, appendix only;
               it does not take part in dev selection.

  python -m omniaware.okvqa_dense --corpus gs112k prepare --csv okvqa_train_corpus.csv   # CSV -> corpus.jsonl
  python -m omniaware.okvqa_dense --corpus gs112k encode [--limit N]      # passages -> fp16 memmap (resumable)
  python -m omniaware.okvqa_dense --corpus gs112k search                  # top-k per question -> candidates.jsonl

The number of passages is read from corpus.jsonl (``--n-passages`` overrides it only for a partial check).
Passages are encoded without a prefix and questions with the BGE query instruction, through the same
relevance.BGE.encode calls that the relevance re-scoring uses, so the index and the scores agree.
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
    "gs168k_full": (paths.LOCAL_CACHE / "okvqa_gs168k_full" / "corpus.jsonl", paths.LOCAL_CACHE / "okvqa_gs168k_full"),
}
GS_CORPORA = ("gs112k", "gs168k_full")
CORPUS, OUT = CORPORA["wiki3m"]
DIM = 1024


def use_corpus(name: str) -> None:
    global CORPUS, OUT
    CORPUS, OUT = CORPORA[name]


def prepare_gs_csv(csv_path: Path) -> Path:
    """Luo et al.'s CSV (one knowledge sentence per row, in column 'text') -> corpus.jsonl; ids are the row order
    of the non-empty sentences.  A CSV without a 'text' column is rejected: the sentence column is never guessed."""
    import csv
    with open(csv_path, newline="", encoding="utf8") as fh:
        reader = csv.DictReader(fh)
        if "text" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} has no 'text' column (header: {reader.fieldnames}); "
                             "the sentence column must be named 'text'")
        CORPUS.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with CORPUS.open("w", encoding="utf8") as out:
            for row in reader:
                text = " ".join((row["text"] or "").split())
                if text:
                    out.write(json.dumps({"id": n, "contents": text}, ensure_ascii=False) + "\n")
                    n += 1
    print(f"[prepare] {n} passages from column 'text' of {csv_path} -> {CORPUS}", flush=True)
    return CORPUS


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
                texts.append(json.loads(fh.readline())["contents"])   # BGE passages: no instruction
            vec = bge.encode(texts, query=False, batch=batch)
            emb[i:i + len(texts)] = vec.astype(np.float16)
            i += len(texts)
            if (i // batch) % 40 == 0 or i == n:
                emb.flush()
                done_path.write_text(str(i))
                rate = (i - start) / max(time.time() - t0, 1e-6)
                print(f"[encode] {i}/{n} {rate:.0f} passages/s, eta {(n - i) / max(rate, 1e-6) / 3600:.2f} h", flush=True)
    emb.flush()
    done_path.write_text(str(n))


def search(n_passages: int | None, k: int, limit_queries: int | None, chunk: int, tag: str) -> Path:
    from .data import load_okvqa

    if n_passages is None:   # the corpus itself decides the size; no hand-typed count to go stale
        n_passages = len(_offsets(None))
    emb = np.memmap(OUT / f"emb_{n_passages}.f16", dtype=np.float16, mode="r", shape=(n_passages, DIM))
    done = int((OUT / f"emb_{n_passages}.done").read_text())
    assert done == n_passages, f"encoding incomplete: {done}/{n_passages}"
    examples = load_okvqa(candidates_path=False)[:limit_queries]
    bge = relevance.BGE()
    q = torch.from_numpy(bge.encode([relevance.query_text(e) for e in examples], query=True)).to(bge.device, torch.float16)
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
    b.add_argument("--n-passages", type=int, default=None, help="default: the number of lines in corpus.jsonl")
    b.add_argument("--k", type=int, default=20)
    b.add_argument("--limit-queries", type=int, default=None)
    b.add_argument("--chunk", type=int, default=200_000)
    b.add_argument("--tag", default="full")
    args = ap.parse_args(argv)
    use_corpus(args.corpus)
    if args.cmd == "prepare":
        if args.corpus not in GS_CORPORA:
            ap.error("prepare converts the Luo et al. GS CSV (--corpus gs112k or gs168k_full)")
        prepare_gs_csv(args.csv)
    elif args.cmd == "encode":
        encode(args.limit, args.batch)
    else:
        search(args.n_passages, args.k, args.limit_queries, args.chunk, args.tag)


if __name__ == "__main__":
    main()

"""Dataset loaders producing one unified example format.

Example = {
    "dataset", "qid", "question", "qcate",
    "query_images": [path, ...],          # images that belong to the query itself (OK-VQA)
    "candidates": [{"cid", "modality", "title", "text", "image", "gold"}],
    "fixed_evidence": bool,               # True: all candidates are shown (video frames)
    "golds": [...], "options": [...] (MC only), "gold_letter" (MC only),
}
"""

from __future__ import annotations

import gzip
import html
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from . import paths


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip().strip('"').strip()


# ------------------------------------------------------------------ WebQA --
def load_webqa(subset: str = "img", split: str = "val") -> list[dict]:
    """WebQA questions with their own distractor pool (txt/img pos+neg facts).

    subset='img': image-based questions (gold facts include an image); 'all': every question.
    This is the WebQA distractor setting, not open retrieval over the full fact corpus.
    Facts whose image file is missing are dropped and counted in ``missing_images``.
    """
    raw = json.loads(paths.WEBQA_JSON.read_text())
    out = []
    missing_total = 0
    for guid, v in raw.items():
        if v["split"] != split:
            continue
        if subset == "img" and not v["img_posFacts"]:
            continue
        missing = 0
        cands = []
        for gold, key in ((True, "txt_posFacts"), (False, "txt_negFacts")):
            for i, f in enumerate(v[key]):
                cands.append({
                    "cid": f"t{'p' if gold else 'n'}{i}", "modality": "text",
                    "title": _clean(f.get("title")), "text": _clean(f.get("fact")),
                    "image": None, "gold": gold,
                })
        for gold, key in ((True, "img_posFacts"), (False, "img_negFacts")):
            for i, f in enumerate(v[key]):
                img = paths.WEBQA_IMAGES / f"{f['image_id']}.jpg"
                if not img.exists():
                    missing += 1
                    continue
                cands.append({
                    "cid": f"i{'p' if gold else 'n'}{i}", "modality": "image",
                    "title": _clean(f.get("title")), "text": _clean(f.get("caption")),
                    "image": str(img), "gold": gold,
                })
        missing_total += missing
        if not any(c["gold"] for c in cands):
            continue
        out.append({
            "dataset": "webqa", "qid": guid, "question": _clean(v["Q"]), "qcate": v["Qcate"],
            "query_images": [], "candidates": cands, "fixed_evidence": False,
            "golds": [_clean(a) for a in v["A"]], "missing_images": missing,
        })
    if missing_total:
        print(f"[data] webqa/{split}/{subset}: {missing_total} fact images missing under "
              f"{paths.WEBQA_IMAGES}; those facts were dropped", flush=True)
    return out


# ------------------------------------------------------------------- MMQA --
def _read_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as fh:
        for line in fh:
            yield json.loads(line)


def _table_chunks(table: dict, rows_per_chunk: int, answer_rows: set[int]) -> list[dict]:
    t = table["table"]
    header = [h["column_name"] for h in t["header"]]
    labels = [re.sub(r"\[\[([^\]]+)\]\]", lambda m: m.group(1).rsplit("|", 1)[-1], _clean(h)) for h in header]
    name = f"{_clean(table['title'])} - {_clean(t.get('table_name', ''))}"
    rows = [[_clean(c["text"]) for c in r] for r in t["table_rows"]]
    chunks = []
    for start in range(0, len(rows), rows_per_chunk):
        block = rows[start:start + rows_per_chunk]
        lines = [f"Table: {name}", "Columns: " + " | ".join(header)]
        lines += [" | ".join(r) for r in block]
        cell_lines = [f"Table: {name}"] + [
            f"Row {start + i + 1}: " + "; ".join(f"{h} = {v}" for h, v in zip(labels, row, strict=True))
            for i, row in enumerate(block)]
        idx = set(range(start, start + len(block)))
        chunks.append({
            "cid": f"tab{start // rows_per_chunk}", "modality": "table", "title": name,
            "text": "\n".join(lines), "cell_text": "\n".join(cell_lines),
            "image": None, "gold": bool(idx & answer_rows),
        })
    return chunks


def load_mmqa_table(rows_per_chunk: int = 8, include_images: bool = True, split: str = "dev",
                    scope: str = "tableq") -> list[dict]:
    """MMQA questions answered from a single table alone.

    scope='tableq': every question whose modalities are only tables (369 dev questions,
    including the 70 same-table comparison questions ['table', 'table']);
    scope='legacy299': the earlier literal filter modalities == ['table'] (299 dev questions).
    """
    if scope == "tableq":
        def keep(md):
            return set(md["modalities"]) == {"table"}
    elif scope == "legacy299":
        def keep(md):
            return md["modalities"] == ["table"]
    else:
        raise ValueError(scope)
    questions = [q for q in _read_jsonl_gz(paths.MMQA_DIR / f"MMQA_{split}.jsonl.gz") if keep(q["metadata"])]
    need_tables = {q["metadata"]["table_id"] for q in questions}
    need_texts = {t for q in questions for t in q["metadata"]["text_doc_ids"]}
    need_images = {t for q in questions for t in q["metadata"]["image_doc_ids"]}
    tables = {t["id"]: t for t in _read_jsonl_gz(paths.MMQA_DIR / "MMQA_tables.jsonl.gz") if t["id"] in need_tables}
    texts = {t["id"]: t for t in _read_jsonl_gz(paths.MMQA_DIR / "MMQA_texts.jsonl.gz") if t["id"] in need_texts}
    images = {t["id"]: t for t in _read_jsonl_gz(paths.MMQA_DIR / "MMQA_images.jsonl.gz") if t["id"] in need_images}
    out = []
    for q in questions:
        md = q["metadata"]
        answer_rows = {r for a in q["answers"] for r, _c in a.get("table_indices", [])}
        cands = _table_chunks(tables[md["table_id"]], rows_per_chunk, answer_rows)
        for i, tid in enumerate(md["text_doc_ids"]):
            if tid in texts:
                t = texts[tid]
                cands.append({"cid": f"txt{i}", "modality": "text", "title": _clean(t["title"]),
                              "text": _clean(t["text"]), "image": None, "gold": False})
        if include_images:
            for i, iid in enumerate(md["image_doc_ids"]):
                im = images.get(iid)
                if im is None:
                    continue
                p = paths.MMQA_IMAGES / im["path"]
                if p.exists():
                    cands.append({"cid": f"img{i}", "modality": "image", "title": _clean(im["title"]),
                                  "text": _clean(im["title"]), "image": str(p), "gold": False})
        golds = [_clean(a["answer"]) for a in q["answers"]]
        out.append({
            "dataset": "mmqa_table", "qid": q["qid"], "question": _clean(q["question"]),
            "qcate": md["type"], "query_images": [], "candidates": cands, "fixed_evidence": False,
            "golds": golds,
        })
    return out


# --------------------------------------------------------------- Video-MME --
_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def _parse_srt(path: Path) -> list[tuple[float, str]]:
    items, t0 = [], None
    for line in path.read_text(errors="ignore").splitlines():
        m = _SRT_TIME.search(line)
        if m:
            h, mi, s, ms = map(int, m.groups()[:4])
            t0 = h * 3600 + mi * 60 + s + ms / 1000
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()
        if text and not text.isdigit() and t0 is not None:
            items.append((t0, text))
    return items


def load_videomme_short(max_subtitle_words: int | None = 220, frames_dir: Path | None = None) -> list[dict]:
    """Video-MME Short; frames come from ``frames_dir`` (8-frame default, or the 32-frame set).
    ``max_subtitle_words=None`` keeps the full subtitles."""
    import pandas as pd

    frames_dir = Path(frames_dir) if frames_dir else paths.VIDEOMME_FRAMES
    df = pd.read_parquet(paths.VIDEOMME_DIR / "official_annotations.parquet")
    df = df[df["duration"] == "short"]
    out = []
    for row in df.itertuples():
        fdir = frames_dir / row.videoID
        meta_path = fdir / "frames.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        cands = []
        for f in meta["frames"]:
            sec = f["actual_seconds"]
            cands.append({"cid": f"f{f['index']}", "modality": "image",
                          "title": f"Frame at {int(sec) // 60:02d}:{int(sec) % 60:02d}",
                          "text": "", "image": str(frames_dir / f["path"]), "gold": None})
        srt = paths.VIDEOMME_DIR / "subtitles" / f"{row.videoID}.srt"
        if srt.exists():
            words = " ".join(t for _s, t in _parse_srt(srt)).split()
            if words:
                cands.append({"cid": "sub", "modality": "text", "title": "Subtitles",
                              "text": " ".join(words[:max_subtitle_words] if max_subtitle_words else words),
                              "image": None, "gold": None})
        options = [str(o) for o in row.options]
        out.append({
            "dataset": "videomme", "qid": row.question_id, "question": _clean(row.question),
            "qcate": row.task_type, "query_images": [], "candidates": cands, "fixed_evidence": True,
            "options": options, "gold_letter": row.answer, "golds": [row.answer],
            "video_id": row.videoID,
        })
    return out


# ------------------------------------------------------------------ OK-VQA --
def load_okvqa(candidates_path=None) -> list[dict]:
    with zipfile.ZipFile(paths.OKVQA_ANNOTATIONS) as zf:
        name = [n for n in zf.namelist() if n.endswith(".json")][0]
        ann = json.loads(zf.read(name))
    answers = {a["question_id"]: [x["answer"] for x in a["answers"]] for a in ann["annotations"]}
    cands_by_q = {}
    with open(candidates_path or paths.OKVQA_CANDIDATES) as fh:
        for line in fh:
            d = json.loads(line)
            cands_by_q[str(d["question_id"])] = d["candidates"]
    out = []
    with open(paths.OKVQA_INPUTS) as fh:
        for line in fh:
            d = json.loads(line)
            qid = str(d["question_id"])
            cands = []
            for c in cands_by_q.get(qid, []):
                title, _, body = c["text"].partition("\n")
                cands.append({"cid": f"k{c['rank']}", "modality": "text", "title": _clean(title),
                              "text": _clean(body or title), "image": None, "gold": None})
            out.append({
                "dataset": "okvqa", "qid": qid, "question": _clean(d["question"]), "qcate": "okvqa",
                "query_images": [d["image_path"]], "candidates": cands, "fixed_evidence": False,
                "golds": answers[int(qid)],
            })
    return out


def load_okvqa_dense() -> list[dict]:
    """OK-VQA questions covered by dense BGE retrieval; rows stay dataset 'okvqa'."""
    import os
    default = paths.LOCAL_CACHE / "okvqa_dense" / "candidates_full.jsonl"
    examples = load_okvqa(os.environ.get("ERC_OKVQA_DENSE_CANDIDATES", default))
    return [e for e in examples if e["candidates"]]


def _webqa_img(**kw):
    return load_webqa("img", **kw)


def _webqa_full(**kw):
    return load_webqa("all", **kw)


LOADERS = {
    "webqa_img": _webqa_img,
    "webqa_full": _webqa_full,
    "mmqa_table": load_mmqa_table,
    "videomme": load_videomme_short,
    "okvqa": load_okvqa,
    "okvqa_dense": load_okvqa_dense,
}

# Training-split loaders for unlabeled calibration (paper: calibration queries come from the
# training splits).  Datasets without an entry can only calibrate on a held-out slice.
TRAIN_LOADERS = {
    "webqa_img": lambda: load_webqa("img", split="train"),
    "webqa_full": lambda: load_webqa("all", split="train"),
    "mmqa_table": lambda: load_mmqa_table(split="train"),
}


def _shuffled(examples: list[dict], seed: int) -> list[dict]:
    order = sorted(examples, key=lambda e: e["qid"])
    random.Random(seed).shuffle(order)
    return order


def split_examples(examples: list[dict], n_calib: int, n_dev: int = 0, seed: int = 2027,
                   calib_pool: list[dict] | None = None) -> dict[str, list[dict]]:
    """Deterministic, disjoint calib / dev / test split.

    With ``calib_pool`` (e.g. the training split) the calibration questions come from it and
    every evaluation question is kept; otherwise the first ``n_calib`` shuffled questions are
    held out for calibration and never evaluated.  ``eval`` = dev + test (legacy key).
    """
    order = _shuffled(examples, seed)
    if calib_pool is not None:
        calib = _shuffled(calib_pool, seed)[:n_calib]
        rest = order
    else:
        calib, rest = order[:n_calib], order[n_calib:]
    dev, test = rest[:n_dev], rest[n_dev:]
    return {"calib": calib, "dev": dev, "test": test, "eval": rest}


def group_by(examples: list[dict], key: str) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for e in examples:
        g[e[key]].append(e)
    return g

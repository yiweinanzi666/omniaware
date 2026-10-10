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
            "gold_modalities": ["image"] if v["img_posFacts"] else ["text"],
        })
    if missing_total:
        print(f"[data] webqa/{split}/{subset}: {missing_total} fact images missing under "
              f"{paths.WEBQA_IMAGES}; those facts were dropped", flush=True)
    return out


def _asset_path(recorded: str) -> Path:
    """Map an absolute path recorded on another host into the local asset root."""
    path = Path(recorded)
    if path.exists():
        return path
    marker = "experiments/eab_first_token/"
    text = recorded.replace("\\", "/")
    return paths.ASSET_ROOT / text.split(marker, 1)[1] if marker in text else path


def load_webqa_open(subset: str = "img", split: str = "val", rankings: Path | None = None) -> list[dict]:
    """WebQA with open retrieval over the full public pool (544,489 texts, 389,750 images).

    Candidates are the BGE text top-k and the CLIP image top-k returned by FAISS for the question
    (scripts/webqa_open_retrieval: encode_webqa_queries --split <split>, then search_dense_queries
    into <rankings>/bge and <rankings>/clip).  ``gold`` marks a candidate whose snippet aliases or
    image id belong to the question's positive facts; it feeds recall diagnostics only.
    Ranked images without an asset record or file are dropped, counted per question in
    ``missing_images`` and reported in total (CONVENTIONS: no silent skips).
    """
    root = paths.WEBQA_OPEN
    rankings = Path(rankings) if rankings else root / "dense_top5_full" / f"webqa_{split}"
    tops: dict[str, dict[str, list[str]]] = {}
    for kind in ("bge", "clip"):
        for line in (rankings / kind / "rankings.jsonl").open():
            r = json.loads(line)
            tops.setdefault(r["qid"], {})[kind] = [c["source_id"] for c in r["candidates"]]
    wanted = {sid for per_q in tops.values() for ids in per_q.values() for sid in ids}
    texts = {}
    with (root / "text_documents.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["document_id"] in wanted:
                texts[row["document_id"]] = row
    captions = {}
    with (root / "image_corpus.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["fact_id"] in wanted:
                variants = sorted({" ".join(c.split()) for c in row["captions"]} - {""})
                captions[row["fact_id"]] = variants[0] if variants else ""
    images = {}
    plan = json.loads((root / "faiss_clip_full" / "plan.json").read_text())
    for encoding in plan["inputs"]:
        with (_asset_path(encoding["directory"]) / "assets.jsonl").open() as fh:
            for line in fh:
                row = json.loads(line)
                if "image:" + str(row["image_id"]) in wanted:
                    images["image:" + str(row["image_id"])] = str(_asset_path(row["path"]))
    raw = json.loads(paths.WEBQA_JSON.read_text())
    out, unranked, missing_total = [], 0, 0
    for guid, v in raw.items():
        if v["split"] != split or (subset == "img" and not v["img_posFacts"]):
            continue
        if guid not in tops:
            unranked += 1
            continue
        gold_text = {str(f["snippet_id"]) for f in v["txt_posFacts"]}
        gold_image = {"image:" + str(f["image_id"]) for f in v["img_posFacts"]}
        cands, missing = [], 0
        for rank, sid in enumerate(tops[guid].get("bge", []), 1):
            row = texts[sid]
            cands.append({"cid": f"t{rank}", "modality": "text", "title": "", "text": _clean(row["text"]),
                          "image": None, "gold": bool(gold_text & set(map(str, row["source_snippet_ids"])))})
        for rank, sid in enumerate(tops[guid].get("clip", []), 1):
            if sid not in images or not Path(images[sid]).exists():
                missing += 1
                continue
            caption = _clean(captions.get(sid, ""))
            cands.append({"cid": f"i{rank}", "modality": "image", "title": caption, "text": caption,
                          "image": images[sid], "gold": sid in gold_image})
        out.append({
            "dataset": "webqa", "qid": guid, "question": _clean(v["Q"]), "qcate": v["Qcate"],
            "query_images": [], "candidates": cands, "fixed_evidence": False,
            "golds": [_clean(a) for a in v["A"]],
            "gold_modalities": ["image"] if v["img_posFacts"] else ["text"],
            "missing_images": missing,
        })
        missing_total += missing
    if missing_total:
        print(f"[data] webqa_open/{split}/{subset}: {missing_total} ranked images missing, dropped from the "
              f"candidates of {sum(e['missing_images'] > 0 for e in out)} questions", flush=True)
    if unranked:
        print(f"[data] webqa_open/{split}/{subset}: {unranked} questions have no rankings in {rankings}", flush=True)
    return out


def load_okvqa(candidates_path=None) -> list[dict]:
    """OK-VQA val questions with retrieval candidates from ``candidates_path`` (default: the BM25 file);
    ``candidates_path=False`` loads the questions alone (split universe, dense search)."""
    with zipfile.ZipFile(paths.OKVQA_ANNOTATIONS) as zf:
        name = [n for n in zf.namelist() if n.endswith(".json")][0]
        ann = json.loads(zf.read(name))
    answers = {a["question_id"]: [x["answer"] for x in a["answers"]] for a in ann["annotations"]}
    cands_by_q = {}
    if candidates_path is not False:
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
    """OK-VQA questions covered by dense BGE retrieval over DPR Wikipedia; rows stay dataset 'okvqa'."""
    import os
    default = paths.LOCAL_CACHE / "okvqa_dense" / "candidates_full.jsonl"
    examples = load_okvqa(os.environ.get("ERC_OKVQA_DENSE_CANDIDATES", default))
    return [e for e in examples if e["candidates"]]


def load_okvqa_gs() -> list[dict]:
    """OK-VQA with dense BGE retrieval over the GS-112K Google Search corpus (okvqa_dense --corpus gs112k)."""
    examples = load_okvqa(paths.LOCAL_CACHE / "okvqa_gs112k" / "candidates_full.jsonl")
    return [e for e in examples if e["candidates"]]


def _webqa_img(**kw):
    return load_webqa("img", **kw)


def _webqa_full(**kw):
    return load_webqa("all", **kw)


def _webqa_open_img(**kw):
    return load_webqa_open("img", **kw)


def _webqa_open_full(**kw):
    return load_webqa_open("all", **kw)


# ------------------------------------------------------- UniversalRAG suite --
# Web-aligned benchmarks of the UniversalRAG suite (Yeo et al.), queries from its dataset/query/*.json.
UNIVERSALRAG_TARGETS = {          # loader name -> (query file, dataset family, default modalities)
    "hybridqa": ("hybridqa", "hybridqa", ("table", "paragraph")),
    "videorag_synth": ("videorag_synth", "videorag", ("video", "clip")),
}
_UR_MODALITY = {"paragraph": "text", "document": "text", "table": "table", "image": "image",
                "video": "video", "clip": "video"}
_UR_PREFIX = {"paragraph": "p", "document": "d", "table": "t", "image": "i", "video": "v", "clip": "c"}


_VIDEORAG_QID = re.compile(r"videorag_synth_(.+)_(\d+)")


def videorag_video_id(qid) -> str:
    """Source video of a VideoRAG-Synth query whose index is videorag_synth_<video>_<n>; other formats raise."""
    m = _VIDEORAG_QID.fullmatch(str(qid))
    if m is None:
        raise ValueError(f"VideoRAG-Synth query id {qid!r} is not videorag_synth_<video>_<n>")
    return m.group(1)


def frame_key(source_id: str) -> str:
    """Directory name of a ranked video or clip under paths.UNIVERSALRAG_FRAMES."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", source_id.replace("dataset/", "", 1))


def _video_script(root: Path, video_path: str) -> str:
    vid = Path(video_path).stem.split(".mp4")[0]
    scripts = root / Path(video_path).parent.parent / "scripts"
    for name in (f"{vid}.txt", f"{vid}.en.vtt"):
        if (scripts / name).exists():
            text = (scripts / name).read_text(encoding="utf8", errors="replace")
            if name.endswith(".vtt"):
                text = " ".join(l for l in text.splitlines() if l and "-->" not in l and not l.startswith("WEBVTT"))
            return _clean(text)
    return ""


def load_universalrag(target: str, modalities: tuple[str, ...] | None = None, top: int = 5,
                      rankings: Path | None = None, root: Path | None = None,
                      frames_dir: Path | None = None) -> list[dict]:
    """HybridQA / VideoRAG-Synth queries with candidates from UniversalRAG's own retrievers.

    Candidates are the top ``top`` of each modality list exported by
    scripts/universalrag_retrieval/export_rankings.py; their contents come from the suite's unified corpora
    (dataset/paragraph|document|table|image.parquet, videos with scripts, clip.parquet). Videos and clips
    become one 'video' chunk: sampled frames (prepare_frames.py) plus the transcript. ``gold`` marks the
    annotated table/videos of the question (HybridQA: the gt table and its linked passages; VideoRAG:
    gt_videos).
    """
    import pandas as pd
    query_file, family, default_modalities = UNIVERSALRAG_TARGETS[target]
    modalities = tuple(modalities or default_modalities)
    root = Path(root or paths.UNIVERSALRAG_ROOT)
    rankings = Path(rankings or paths.UNIVERSALRAG_RANKINGS) / query_file
    frames_dir = Path(frames_dir or paths.UNIVERSALRAG_FRAMES)
    queries = json.loads((root / "dataset" / "query" / f"{query_file}.json").read_text(encoding="utf8"))
    tops: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for m in modalities:
        for line in (rankings / f"{m}.jsonl").open():
            r = json.loads(line)
            tops[r["qid"]][m] = [c["source_id"] for c in r["candidates"]][:top]
    wanted = {m: {s for per_q in tops.values() for s in per_q.get(m, [])} for m in modalities}
    corpora = {}
    for m in ("paragraph", "document", "table", "image"):
        if wanted.get(m):
            df = pd.read_parquet(root / "dataset" / f"{m}.parquet")
            corpora[m] = df[df.index.isin(wanted[m])].to_dict("index")
    out, missing_frames = [], 0
    for q in queries:
        qid = q["index"]
        if qid not in tops:
            continue
        gt_table = q.get("gt_tables")
        gt_videos = set(q.get("gt_videos") or [])
        cands = []
        for m in modalities:
            for rank, sid in enumerate(tops[qid].get(m, []), 1):
                cid = f"{_UR_PREFIX[m]}{rank}"
                if m in ("paragraph", "document", "table"):
                    row = corpora[m][sid]
                    gold = (sid == f"hybridqa-{gt_table}" or sid.startswith(f"hybridqa-{gt_table}_")) if gt_table else None
                    cands.append({"cid": cid, "modality": _UR_MODALITY[m], "title": "", "text": _clean(row["text"]),
                                  "image": None, "gold": gold, "source_id": sid})
                elif m == "image":
                    row = corpora[m][sid]
                    caption = _clean(row.get("caption") or row.get("title") or "")
                    cands.append({"cid": cid, "modality": "image", "title": caption, "text": caption,
                                  "image": str(root / sid), "gold": None, "source_id": sid})
                else:
                    meta_path = frames_dir / frame_key(sid) / "frames.json"
                    if not meta_path.exists():
                        missing_frames += 1
                        continue
                    meta = json.loads(meta_path.read_text())
                    video = sid.rsplit("_", 1)[0] if m == "clip" else sid
                    cands.append({"cid": cid, "modality": "video", "title": "", "text": _video_script(root, video),
                                  "image": None, "frames": [str(meta_path.parent / f["path"]) for f in meta["frames"]],
                                  "gold": (video in gt_videos) if gt_videos else None, "source_id": sid})
        gold_modalities = sorted({_UR_MODALITY[g] for g in q.get("gt_retrieval", "").split("+") if g in _UR_MODALITY})
        out.append({
            "dataset": family, "qid": qid, "question": _clean(q["question"]), "qcate": q.get("source", family),
            "query_images": [str(root / q["query_image"])] if q.get("query_image") else [],
            "candidates": cands, "fixed_evidence": False,
            "golds": [a for a in [q["answer"], *(q.get("aliases") or [])] if a],
            "gold_modalities": gold_modalities,
            # VideoRAG-Synth clusters its bootstrap by source video (index videorag_synth_<video>_<n>)
            **({"cluster_id": videorag_video_id(qid)} if family == "videorag" else {}),
        })
    if missing_frames:
        print(f"[data] {target}: {missing_frames} ranked videos/clips have no frames under {frames_dir}", flush=True)
    return out


LOADERS = {
    "webqa_img": _webqa_img,
    "webqa_full": _webqa_full,
    "webqa_open_img": _webqa_open_img,
    "webqa_open_full": _webqa_open_full,
    "okvqa": load_okvqa,
    "okvqa_dense": load_okvqa_dense,
    "okvqa_gs": load_okvqa_gs,
    **{name: (lambda name: lambda **kw: load_universalrag(name, **kw))(name) for name in UNIVERSALRAG_TARGETS},
}

# Training-split loaders for unlabeled calibration (paper: calibration queries come from the
# training splits).  Datasets without an entry can only calibrate on a held-out slice.
TRAIN_LOADERS = {
    "webqa_img": lambda: load_webqa("img", split="train"),
    "webqa_full": lambda: load_webqa("all", split="train"),
    "webqa_open_img": lambda: load_webqa_open("img", split="train"),
    "webqa_open_full": lambda: load_webqa_open("all", split="train"),
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

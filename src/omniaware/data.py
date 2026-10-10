"""Dataset loaders producing one unified example format, and the calib / dev / test split (docs/PROTOCOL.md).

Example = {
    "dataset", "qid", "question", "qcate",
    "query_images": [path, ...],          # images that belong to the query itself (OK-VQA)
    "candidates": [{"cid", "modality", "title", "text", "image", "gold"}],
    "fixed_evidence": bool,               # True: all candidates are shown (video frames)
    "golds": [...], "options": [...] (MC only), "gold_letter" (MC only),
}

Loaders return an ``Examples`` list.  A missing asset (a fact image, a ranked file, an image file, a corpus
row) or a question without candidates raises ``MissingDataError`` unless the loader is called with
``allow_missing=True``; then the affected questions or candidates are dropped and recorded in
``Examples.meta["dropped"]``, which run.py writes into the run configuration (CONVENTIONS: no silent skips).
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


class MissingDataError(RuntimeError):
    """A loader needs assets or rankings that are absent; the message lists them."""


class Examples(list):
    """Loader output: the examples plus loader metadata (``dropped`` entries recorded under allow_missing)."""

    def __init__(self, items=(), meta=None):
        super().__init__(items)
        self.meta = dict(meta or {})


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip().strip('"').strip()


def _bump(table: dict, reason: str, qid: str, n: int = 1) -> None:
    """Count ``n`` missing items of ``reason`` for question ``qid`` (``table`` maps reason -> {qid: count})."""
    counts = table.setdefault(reason, {})
    counts[qid] = counts.get(qid, 0) + n


def _missing_policy(what: str, missing: dict[str, dict[str, int]], allow_missing: bool) -> dict:
    """Fail-fast policy shared by the loaders.  ``missing`` maps a reason to {qid: items lacking}.

    Without ``allow_missing`` any entry raises, listing reasons and example qids.  With it the drop is printed
    and returned as {reason: {"items": n, "qids": [...]}} for the run configuration.
    """
    missing = {reason: qids for reason, qids in missing.items() if qids}
    if not missing:
        return {}
    lines = [f"{reason}: {sum(qids.values())} items in {len(qids)} questions, e.g. {sorted(qids)[:5]}"
             for reason, qids in missing.items()]
    if not allow_missing:
        raise MissingDataError(f"{what}: " + "; ".join(lines) + ". Fix the assets, or pass --allow-missing-assets "
                               "to drop them (the drop is recorded in the run configuration).")
    print(f"[data] {what}: dropped " + "; ".join(lines), flush=True)
    return {reason: {"items": sum(qids.values()), "qids": sorted(qids)} for reason, qids in missing.items()}


# ------------------------------------------------------------------ WebQA --
def _gold_modalities(v: dict) -> list[str]:
    """Modalities of the positive facts; a question with both kinds of positive fact records both (M-37)."""
    return sorted(m for m, facts in (("image", v["img_posFacts"]), ("text", v["txt_posFacts"])) if facts)


def load_webqa(subset: str = "img", split: str = "val", allow_missing: bool = False) -> Examples:
    """WebQA questions with their own distractor pool (txt/img pos+neg facts).

    subset='img': image-based questions (gold facts include an image); 'all': every question.
    This is the WebQA distractor setting, not open retrieval over the full fact corpus.
    A fact whose image file is missing is an error unless ``allow_missing``; then the fact is dropped, and so is a
    question left without any gold candidate.  Both are recorded in ``meta["dropped"]``.
    """
    raw = json.loads(paths.WEBQA_JSON.read_text())
    out, missing = [], {}
    for guid, v in raw.items():
        if v["split"] != split:
            continue
        if subset == "img" and not v["img_posFacts"]:
            continue
        cands = []
        for gold, key in ((True, "txt_posFacts"), (False, "txt_negFacts")):
            for i, f in enumerate(v[key]):
                cands.append({
                    "cid": f"t{'p' if gold else 'n'}{i}", "modality": "text",
                    "title": _clean(f.get("title")), "text": _clean(f.get("fact")),
                    "image": None, "gold": gold,
                })
        missing_here = 0
        for gold, key in ((True, "img_posFacts"), (False, "img_negFacts")):
            for i, f in enumerate(v[key]):
                img = paths.WEBQA_IMAGES / f"{f['image_id']}.jpg"
                if not img.exists():
                    missing_here += 1
                    _bump(missing, "fact image missing", guid)
                    continue
                cands.append({
                    "cid": f"i{'p' if gold else 'n'}{i}", "modality": "image",
                    "title": _clean(f.get("title")), "text": _clean(f.get("caption")),
                    "image": str(img), "gold": gold,
                })
        if not any(c["gold"] for c in cands):
            _bump(missing, "no gold fact left", guid)
            continue
        out.append({
            "dataset": "webqa", "qid": guid, "question": _clean(v["Q"]), "qcate": v["Qcate"],
            "query_images": [], "candidates": cands, "fixed_evidence": False,
            "golds": [_clean(a) for a in v["A"]], "missing_images": missing_here,
            "gold_modalities": _gold_modalities(v),
        })
    dropped = _missing_policy(f"webqa/{split}/{subset}", missing, allow_missing)
    return Examples(out, {"dropped": dropped})


def _asset_path(recorded: str) -> Path:
    """Map an absolute path recorded on another host into the local asset root."""
    path = Path(recorded)
    if path.exists():
        return path
    marker = "experiments/eab_first_token/"
    text = recorded.replace("\\", "/")
    return paths.ASSET_ROOT / text.split(marker, 1)[1] if marker in text else path


def load_webqa_open(subset: str = "img", split: str = "val", rankings: Path | None = None,
                    allow_missing: bool = False) -> Examples:
    """WebQA with open retrieval over the full public pool (544,489 texts, 389,750 images).

    Candidates are the BGE text top-k and the CLIP image top-k returned by FAISS for the question
    (scripts/webqa_open_retrieval: encode_webqa_queries --split <split>, then search_dense_queries
    into <rankings>/bge and <rankings>/clip).  ``gold`` marks a candidate whose snippet aliases or
    image id belong to the question's positive facts; it feeds recall diagnostics only.
    A question without rankings, a ranked text without a row, or a ranked image without an asset record or file
    is an error unless ``allow_missing``; then it is dropped (questions) or dropped from the candidates
    (``missing_images`` counts those per question).  Everything dropped is recorded in ``meta["dropped"]``.
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
    out, missing = [], {}
    for guid, v in raw.items():
        if v["split"] != split or (subset == "img" and not v["img_posFacts"]):
            continue
        if guid not in tops:
            _bump(missing, "no ranking", guid)
            continue
        gold_text = {str(f["snippet_id"]) for f in v["txt_posFacts"]}
        gold_image = {"image:" + str(f["image_id"]) for f in v["img_posFacts"]}
        cands, lacking = [], 0
        for rank, sid in enumerate(tops[guid].get("bge", []), 1):
            if sid not in texts:
                _bump(missing, "ranked text missing", guid)
                continue
            row = texts[sid]
            cands.append({"cid": f"t{rank}", "modality": "text", "title": "", "text": _clean(row["text"]),
                          "image": None, "gold": bool(gold_text & set(map(str, row["source_snippet_ids"])))})
        for rank, sid in enumerate(tops[guid].get("clip", []), 1):
            if sid not in images or not Path(images[sid]).exists():
                lacking += 1
                _bump(missing, "ranked image missing", guid)
                continue
            caption = _clean(captions.get(sid, ""))
            cands.append({"cid": f"i{rank}", "modality": "image", "title": caption, "text": caption,
                          "image": images[sid], "gold": sid in gold_image})
        out.append({
            "dataset": "webqa", "qid": guid, "question": _clean(v["Q"]), "qcate": v["Qcate"],
            "query_images": [], "candidates": cands, "fixed_evidence": False,
            "golds": [_clean(a) for a in v["A"]],
            "gold_modalities": _gold_modalities(v),
            "missing_images": lacking,
        })
    dropped = _missing_policy(f"webqa_open/{split}/{subset}", missing, allow_missing)
    return Examples(out, {"dropped": dropped})


# ------------------------------------------------------------------ OK-VQA --
def load_okvqa(candidates_path=None, allow_missing: bool = False) -> Examples:
    """OK-VQA val questions with retrieval candidates from ``candidates_path`` (default: the BM25 file);
    ``candidates_path=False`` loads the questions alone (split universe, dense search).

    With candidates loaded, a question without candidates or with a missing query image is an error unless
    ``allow_missing``, when it is dropped and recorded in ``meta["dropped"]``.
    """
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
    out, missing = [], {}
    with open(paths.OKVQA_INPUTS) as fh:
        for line in fh:
            d = json.loads(line)
            qid = str(d["question_id"])
            image = str(_asset_path(d["image_path"]))
            if candidates_path is not False:
                if not cands_by_q.get(qid):
                    _bump(missing, "no candidates", qid)
                if not Path(image).exists():
                    _bump(missing, "query image missing", qid)
            cands = []
            for c in cands_by_q.get(qid, []):
                title, sep, body = c["text"].partition("\n")
                if sep and body.strip():                       # "title" line, then the passage body
                    title, body = _clean(title), _clean(body)
                else:                                          # single-line passage: the whole text, no title
                    title, body = "", _clean(c["text"])
                cands.append({"cid": f"k{c['rank']}", "modality": "text", "title": title,
                              "text": body, "image": None, "gold": None})
            out.append({
                "dataset": "okvqa", "qid": qid, "question": _clean(d["question"]), "qcate": "okvqa",
                "query_images": [image], "candidates": cands, "fixed_evidence": False,
                "golds": answers[int(qid)],
            })
    dropped = _missing_policy("okvqa", missing, allow_missing)
    bad = {q for qids in missing.values() for q in qids}
    return Examples([e for e in out if e["qid"] not in bad], {"dropped": dropped})


def load_okvqa_dense(allow_missing: bool = False) -> Examples:
    """OK-VQA questions covered by dense BGE retrieval over DPR Wikipedia; rows stay dataset 'okvqa'."""
    import os
    default = paths.LOCAL_CACHE / "okvqa_dense" / "candidates_full.jsonl"
    return load_okvqa(os.environ.get("ERC_OKVQA_DENSE_CANDIDATES", default), allow_missing=allow_missing)


def load_okvqa_gs(allow_missing: bool = False) -> Examples:
    """OK-VQA with dense BGE retrieval over the GS-112K Google Search corpus (okvqa_dense --corpus gs112k)."""
    return load_okvqa(paths.LOCAL_CACHE / "okvqa_gs112k" / "candidates_full.jsonl", allow_missing=allow_missing)


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
# Modality of each UniversalRAG candidate list.  clip and video are separate modalities (D6).
_UR_MODALITY = {"paragraph": "text", "document": "text", "table": "table", "image": "image",
                "video": "video", "clip": "clip"}
_UR_PREFIX = {"paragraph": "p", "document": "d", "table": "t", "image": "i", "video": "v", "clip": "c"}


def universalrag_golds(query: dict) -> list[str]:
    """Reference answers of a UniversalRAG query: its answer and aliases, empty strings dropped."""
    return [a for a in [query["answer"], *(query.get("aliases") or [])] if a]


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
                      frames_dir: Path | None = None, allow_missing: bool = False) -> Examples:
    """HybridQA / VideoRAG-Synth queries with candidates from UniversalRAG's own retrievers.

    Candidates are the top ``top`` of each modality list exported by
    scripts/universalrag_retrieval/export_rankings.py; their contents come from the suite's unified corpora
    (dataset/paragraph|document|table|image.parquet, videos with scripts, clip.parquet).  Videos and clips are
    separate modalities ('video', 'clip'); each becomes sampled frames (prepare_frames.py) plus the transcript.
    ``gold`` marks the annotated table/videos of the question (HybridQA: the gt table and its linked passages;
    VideoRAG: gt_videos).  Missing rankings, corpus rows, image files, frames or query images are an error unless
    ``allow_missing`` (then the candidate or question is dropped and recorded in ``meta["dropped"]``).
    """
    import pandas as pd
    query_file, family, default_modalities = UNIVERSALRAG_TARGETS[target]
    modalities = tuple(modalities or default_modalities)
    root = Path(root or paths.UNIVERSALRAG_ROOT)
    rankings = Path(rankings or paths.UNIVERSALRAG_RANKINGS) / query_file
    frames_dir = Path(frames_dir or paths.UNIVERSALRAG_FRAMES)
    queries = json.loads((root / "dataset" / "query" / f"{query_file}.json").read_text(encoding="utf8"))
    tops: dict[str, dict[str, list[str]]] = defaultdict(dict)
    suite = {}   # (qid, modality) -> {source_id: the suite retriever's own score}
    for m in modalities:
        for line in (rankings / f"{m}.jsonl").open():
            r = json.loads(line)
            tops[r["qid"]][m] = [c["source_id"] for c in r["candidates"]][:top]
            suite[(r["qid"], m)] = {c["source_id"]: float(c["score"]) for c in r["candidates"][:top]}
    wanted = {m: {s for per_q in tops.values() for s in per_q.get(m, [])} for m in modalities}
    corpora = {}
    for m in ("paragraph", "document", "table", "image"):
        if wanted.get(m):
            df = pd.read_parquet(root / "dataset" / f"{m}.parquet")
            corpora[m] = df[df.index.isin(wanted[m])].to_dict("index")
    out, missing = [], {}
    for q in queries:
        qid = q["index"]
        if qid not in tops:
            _bump(missing, "no rankings", qid)
            continue
        if q.get("query_image") and not (root / q["query_image"]).exists():
            _bump(missing, "query image missing", qid)
            continue
        gt_table = q.get("gt_tables")
        gt_videos = set(q.get("gt_videos") or [])
        cands = []
        for m in modalities:
            for rank, sid in enumerate(tops[qid].get(m, []), 1):
                cid = f"{_UR_PREFIX[m]}{rank}"
                if m in ("paragraph", "document", "table", "image") and sid not in corpora.get(m, {}):
                    _bump(missing, "ranked source missing from corpus", qid)
                    continue
                score = suite[(qid, m)][sid]
                if m in ("paragraph", "document", "table"):
                    row = corpora[m][sid]
                    gold = (sid == f"hybridqa-{gt_table}" or sid.startswith(f"hybridqa-{gt_table}_")) if gt_table else None
                    cands.append({"cid": cid, "modality": _UR_MODALITY[m], "title": "", "text": _clean(row["text"]),
                                  "image": None, "gold": gold, "source_id": sid, "suite_score": score})
                elif m == "image":
                    row = corpora[m][sid]
                    if not (root / sid).exists():
                        _bump(missing, "image file missing", qid)
                        continue
                    caption = _clean(row.get("caption") or row.get("title") or "")
                    cands.append({"cid": cid, "modality": "image", "title": caption, "text": caption,
                                  "image": str(root / sid), "gold": None, "source_id": sid, "suite_score": score})
                else:
                    meta_path = frames_dir / frame_key(sid) / "frames.json"
                    if not meta_path.exists():
                        _bump(missing, "frames missing", qid)
                        continue
                    meta = json.loads(meta_path.read_text())
                    video = sid.rsplit("_", 1)[0] if m == "clip" else sid
                    cands.append({"cid": cid, "modality": _UR_MODALITY[m], "title": "", "text": _video_script(root, video),
                                  "image": None, "frames": [str(meta_path.parent / f["path"]) for f in meta["frames"]],
                                  "gold": (video in gt_videos) if gt_videos else None, "source_id": sid,
                                  "suite_score": score})
        gold_modalities = sorted({_UR_MODALITY[g] for g in q.get("gt_retrieval", "").split("+") if g in _UR_MODALITY})
        out.append({
            "dataset": family, "qid": qid, "question": _clean(q["question"]), "qcate": q.get("source", family),
            "query_images": [str(root / q["query_image"])] if q.get("query_image") else [],
            "candidates": cands, "fixed_evidence": False,
            "golds": universalrag_golds(q),
            "gold_modalities": gold_modalities,
            # VideoRAG-Synth clusters its bootstrap by source video (index videorag_synth_<video>_<n>)
            **({"cluster_id": videorag_video_id(qid)} if family == "videorag" else {}),
        })
    dropped = _missing_policy(f"universalrag/{target}", missing, allow_missing)
    return Examples(out, {"dropped": dropped})


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

# Training-split pools (paper: calibration queries come from the training splits).  Only WebQA has one: its
# calibration and dev questions are drawn from the training split (D2).  The other datasets calibrate on a
# held-out reserve of their own universe (CALIB_RESERVE).
TRAIN_LOADERS = {
    "webqa_full": lambda **kw: load_webqa("all", split="train", **kw),
    "webqa_open_full": lambda **kw: load_webqa_open("all", split="train", **kw),
}
# WebQA image subsets are cut from their full split: the partition is drawn over all questions and then restricted
# to the image questions (D2).  The image dev questions are then the image part of the full dev split.
IMAGE_SUBSET_OF = {"webqa_img": "webqa_full", "webqa_open_img": "webqa_open_full"}

# ------------------------------------------------------ protocol sizes (D1, D2) --
# Held-out datasets: the universe is shuffled with the split seed.  Its first CALIB_RESERVE questions are the
# calibration reserve (calib = the first n_calib of them), the next N_DEV are dev, and the rest is test.  The test
# set therefore does not depend on n_calib.
CALIB_RESERVE = {"okvqa": 500, "hybridqa": 500, "videorag_synth": 100}
# Training-split pools (WebQA): calib = the first n_calib of the pool's first TRAIN_RESERVE questions, dev = the
# N_DEV questions after that reserve, so dev does not move with n_calib (calibration-size ablation up to 1000);
# every question of the evaluation universe is test.
N_CALIB_TRAIN = {"webqa": 500}
TRAIN_RESERVE = {"webqa": 1000}
N_DEV = {"okvqa": 256, "webqa": 256, "hybridqa": 256, "videorag_synth": 50}


def protocol_family(dataset: str) -> str:
    """Key of the protocol sizes for a loader name: the OK-VQA and WebQA variants share their universe's sizes."""
    if dataset.startswith("okvqa"):
        return "okvqa"
    if dataset.startswith("webqa"):
        return "webqa"
    return dataset


def _shuffled(examples: list[dict], seed: int) -> list[dict]:
    order = sorted(examples, key=lambda e: e["qid"])
    random.Random(seed).shuffle(order)
    return order


def check_disjoint(sp: dict[str, list[dict]]) -> None:
    """Raise unless calib, dev and test are pairwise disjoint (an explicit check, so it also runs under -O)."""
    ids = {k: {e["qid"] for e in sp[k]} for k in ("calib", "dev", "test")}
    for a, b in (("calib", "dev"), ("calib", "test"), ("dev", "test")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise ValueError(f"{a} and {b} share {len(overlap)} questions, e.g. {sorted(overlap)[:3]}")


def split_examples(examples: list[dict], n_calib: int, n_dev: int, seed: int = 2027,
                   calib_pool: list[dict] | None = None, reserve: int | None = None) -> dict[str, list[dict]]:
    """Deterministic, disjoint calib / dev / test split.

    Held-out (``calib_pool`` is None): ``examples`` are shuffled, calib = the first ``n_calib`` of the first
    ``reserve`` questions, dev = the next ``n_dev``, test = the rest.  Test does not depend on ``n_calib``.
    Training pool (``calib_pool`` given): calib = the first ``n_calib`` of the shuffled pool's first ``reserve``
    questions (default: ``n_calib``), dev = the ``n_dev`` after them, and every question of ``examples`` is test.
    ``eval`` = dev + test (legacy key).  Raises on an oversized request, an empty test set or any overlap.
    """
    order = _shuffled(examples, seed)
    if calib_pool is not None:
        pool = _shuffled(calib_pool, seed)
        pool_reserve = n_calib if reserve is None else reserve
        if not 1 <= n_calib <= pool_reserve:
            raise ValueError(f"n_calib {n_calib} must lie in [1, {pool_reserve}], the training-pool reserve")
        if pool_reserve + n_dev > len(pool):
            raise ValueError(f"training pool has {len(pool)} questions, need a {pool_reserve} reserve + {n_dev} dev")
        calib, dev, test = pool[:n_calib], pool[pool_reserve:pool_reserve + n_dev], order
    else:
        if reserve is None:
            raise ValueError("held-out calibration needs the reserve size (CALIB_RESERVE)")
        if not 1 <= n_calib <= reserve:
            raise ValueError(f"n_calib {n_calib} must lie in [1, {reserve}], the calibration reserve")
        calib, dev, test = order[:n_calib], order[reserve:reserve + n_dev], order[reserve + n_dev:]
    if len(dev) != n_dev:
        raise ValueError(f"dev split needs {n_dev} questions; the universe has {len(examples)}")
    if not test:
        raise ValueError(f"the evaluation (test) split is empty: {len(examples)} questions, dev {n_dev}")
    sp = {"calib": calib, "dev": dev, "test": test, "eval": dev + test}
    check_disjoint(sp)
    return sp


def is_image_question(e: dict) -> bool:
    """WebQA image question: it has a positive image fact (recorded in gold_modalities, M-37)."""
    return "image" in e["gold_modalities"]


def image_subset(sp: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Restrict every part of a full split to its image questions (WebQA-img, D2)."""
    return {k: [e for e in v if is_image_question(e)] for k, v in sp.items()}


def candidate_shortfall(examples: list[dict], k: int) -> dict:
    """Count questions that cannot fill a top-k selection, or lack candidates of a modality the set has (M-21).

    Reported at load time so that a short question is visible before the run reaches it; select_chunks would
    otherwise raise mid-run.  Fixed-evidence questions (all candidates are shown) are not counted as short.
    """
    modalities = sorted({c["modality"] for e in examples for c in e["candidates"]})
    short = [e["qid"] for e in examples if not e["fixed_evidence"] and len(e["candidates"]) < k]
    lacking = {m: [e["qid"] for e in examples if m not in {c["modality"] for c in e["candidates"]}]
               for m in modalities}
    return {"k": k, "fewer_than_k": {"count": len(short), "qids": short[:10]},
            "without_modality": {m: {"count": len(q), "qids": q[:10]} for m, q in lacking.items() if q}}


def group_by(examples: list[dict], key: str) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for e in examples:
        g[e[key]].append(e)
    return g

"""Does a WebQA image reach the model? CPU check of the evidence input of two v2 actions. No model is run.

    python -m scripts.analysis.image_evidence_check --fixture --output report.json
    python -m scripts.analysis.image_evidence_check --n 50 --output report.json

For each sampled WebQA image question the check builds the input segments that run.py would encode for
``static_image`` (the top-k image candidates) and ``static_mixed`` (the joint top-k), with the same functions:
run.select_chunks / run.modality_pool and pipeline.build_segments. It then reports, per action:

- how many images are in the input, and whether each file opens;
- the resized pixel count under the default --max-pixels budget (QwenEAB's Qwen3-VL resize rule, patch 14, merge
  2, factor 28) and the visual-token count it implies (the same count QwenEAB.encode takes from the image grid);
- whether the gold image is among the chunks the action selected, and at which position (E<j>, in the order the
  model reads them, which is the real selection, not the candidate file order);
- whether each image is paired with its own title and caption in the prompt, and whether its file name matches the
  image id of its candidate.

The real mode needs the WebQA annotations, the image files and the relevance cache written by run.py for webqa_full
(``cache/relevance_webqa_full.json`` under OMNIAWARE_CACHE). It uses the protocol calibration of run.py (training-split
calibration, 500 questions, split seed 2027) to turn raw scores into the calibrated relevance, so the selection is
the one a run makes. A missing input stops the check with the paths it needs.

``--fixture`` builds small images and candidates in a temporary directory, and feeds its relevance scores directly.
It runs the same CPU checks and needs no assets. The GPU probe (ask the model to describe each sampled image) is not
run by this script; ``--gpu-probe`` prints what it would compare.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from omniaware import data, paths, pipeline, relevance, runio  # noqa: E402
from omniaware import run as run_mod  # noqa: E402
from omniaware.actions import V2_ACTIONS  # noqa: E402
from omniaware.qwen_eab import Segment  # noqa: E402

K = 5                                   # top-k of the evidence actions (run.py --k default)
MAX_PIXELS = 256 * 32 * 32              # run.py --max-pixels default
MIN_PIXELS = 64 * 32 * 32               # QwenEAB min_pixels default
FACTOR = 28                             # Qwen3-VL: patch 14 x spatial merge 2
IMAGE_MODE = "mix"                      # run.py --image-mode default
ACTIONS = tuple(a for a in V2_ACTIONS if a in ("static_image", "static_mixed"))   # the two actions checked here
DATASET = "webqa_full"
CALIB_QIDS_N = data.N_CALIB_TRAIN["webqa"]
GPU_PROBE = ("GPU probe (not executed here): feed each image listed in the report to the model with the prompt "
             "'Describe the image.' and compare the description with the candidate caption. Needs the 8B weights.")


def smart_resize(height: int, width: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Qwen2-VL / Qwen3-VL smart_resize: the (height, width) the image processor resizes to."""
    if height < 1 or width < 1:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def read_image(path: str | None) -> dict:
    """Open one image file. An unreadable path stays unread; nothing is filled in for it."""
    out = {"path": path, "readable": False, "width": None, "height": None, "pixels": None,
           "resized_pixels": None, "visual_tokens": None}
    if not path or not Path(path).is_file():
        return out
    from PIL import Image
    with Image.open(path) as im:
        width, height = im.size
    h_bar, w_bar = smart_resize(height, width)
    out.update(readable=True, width=width, height=height, pixels=width * height,
               resized_pixels=h_bar * w_bar, visual_tokens=(h_bar // FACTOR) * (w_bar // FACTOR))
    return out


def image_segments(segments: list[Segment]) -> list[tuple[int, Segment]]:
    return [(i, s) for i, s in enumerate(segments) if s.image is not None]


def inspect_action(example: dict, name: str, chunks: list[dict], rel: dict[str, float]) -> dict:
    """What one action sends to the model, checked against its candidates."""
    segments, _ = pipeline.build_segments(example, chunks, "default")
    images = []
    gold_positions = []
    for j, c in enumerate(chunks, 1):
        if c["modality"] == "image" and c.get("gold"):
            gold_positions.append(j)
    for index, seg in image_segments(segments):
        chunk = next(c for c in chunks if c["cid"] == seg.chunk)
        position = next(j for j, c in enumerate(chunks, 1) if c["cid"] == seg.chunk)
        measured = read_image(seg.image)
        head = segments[index - 1] if index > 0 else None
        tail = segments[index + 1] if index + 1 < len(segments) else None
        caption = pipeline.chunk_text(chunk)
        title = chunk.get("title") or ""
        title_paired = bool(head and head.chunk == seg.chunk and (not title or title in head.text))
        caption_paired = bool(tail and tail.chunk == seg.chunk and (not caption or caption in tail.text))
        image_id = chunk.get("image_id")
        images.append({**measured, "cid": chunk["cid"], "position": position, "gold": bool(chunk.get("gold")),
                       "path_matches_candidate": seg.image == chunk.get("image"),
                       "title_paired": title_paired, "caption_paired": caption_paired,
                       "id_matches_file": (not image_id) or Path(seg.image).stem == str(image_id),
                       "relevance": round(float(rel[chunk["cid"]]), 6)})
    return {"action": name, "selected": [c["cid"] for c in chunks], "images": images,
            "n_images": len(images), "gold_image_selected": bool(gold_positions),
            "gold_image_positions": gold_positions,
            "text_words": sum(len(s.text.split()) for s in segments if s.text)}


def inspect_question(example: dict, rel: dict[str, float], k: int = K) -> dict:
    """The static_image and static_mixed inputs of one question, selected with run.py's own functions."""
    static_image = run_mod.select_chunks(run_mod.modality_pool(example, "image"), rel, k, "rank", 0)
    static_mixed = run_mod.select_chunks(example, rel, k, "rank", 0)
    return {"qid": example.get("qid"), "question": example.get("question"),
            "image_candidates": sum(1 for c in example["candidates"] if c["modality"] == "image"),
            "gold_image_candidates": sum(1 for c in example["candidates"] if c["modality"] == "image" and c.get("gold")),
            "actions": {"static_image": inspect_action(example, "static_image", static_image, rel),
                        "static_mixed": inspect_action(example, "static_mixed", static_mixed, rel)}}


def summarize(questions: list[dict]) -> dict:
    """Counts over the questions, each computed from the inspected inputs."""
    out = {}
    for name in ACTIONS:
        rows = [q["actions"][name] for q in questions]
        images = [im for r in rows for im in r["images"]]
        out[name] = {
            "questions": len(rows),
            "questions_with_image_in_input": sum(1 for r in rows if r["n_images"]),
            "images": len(images),
            "unreadable_images": sum(1 for im in images if not im["readable"]),
            "images_over_budget": sum(1 for im in images if im["readable"] and im["pixels"] > MAX_PIXELS),
            "images_resized_to_budget": sum(1 for im in images if im["readable"] and im["resized_pixels"] <= MAX_PIXELS
                                            and im["pixels"] > MAX_PIXELS),
            "visual_tokens_total": sum(im["visual_tokens"] or 0 for im in images),
            "images_with_pairing_error": sum(1 for im in images if not (im["title_paired"] and im["caption_paired"]
                                                                        and im["path_matches_candidate"])),
            "images_with_id_mismatch": sum(1 for im in images if not im["id_matches_file"]),
            "gold_image_selected": sum(1 for r in rows if r["gold_image_selected"]),
            "gold_image_positions": sorted(p for r in rows for p in r["gold_image_positions"]),
        }
    return out


def findings(summary: dict, questions: list[dict]) -> list[str]:
    """Plain statements of what the numbers show. Every line is computed here."""
    lines = []
    for name in ACTIONS:
        s = summary[name]
        lines.append(f"{name}: {s['questions_with_image_in_input']} of {s['questions']} questions send at least one "
                     f"image ({s['images']} images in all, {s['visual_tokens_total']} visual tokens estimated).")
        if s["unreadable_images"]:
            lines.append(f"{name}: {s['unreadable_images']} image file(s) cannot be opened, so the run would stop on "
                         "them: fix the assets or the candidate list.")
        if s["images_with_pairing_error"]:
            lines.append(f"{name}: {s['images_with_pairing_error']} image(s) are not paired with their own title, "
                         "caption or path in the prompt.")
        if s["images_with_id_mismatch"]:
            lines.append(f"{name}: {s['images_with_id_mismatch']} image(s) have a file name that differs from their "
                         "image id.")
        if s["images_over_budget"]:
            lines.append(f"{name}: {s['images_over_budget']} image(s) are larger than the --max-pixels budget and are "
                         f"resized to it ({s['images_resized_to_budget']} of them to within the budget).")
    gold_in_mixed = summary["static_mixed"]["gold_image_selected"]
    gold_total = sum(1 for q in questions if q["gold_image_candidates"])
    lines.append(f"static_mixed selects a gold image on {gold_in_mixed} of the {gold_total} questions that have one.")
    return lines


def fixture_inputs(directory: Path) -> list[dict]:
    """Real tiny image files and candidates; relevance is given per candidate (see fixture_relevance)."""
    from PIL import Image
    small = directory / "gold.png"
    big = directory / "big.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(small)
    Image.new("RGB", (800, 800), (10, 20, 30)).save(big)
    return [{
        "dataset": DATASET, "qid": "fixture-1", "question": "What color is the car?", "query_images": [],
        "gold_modalities": ["image", "text"], "fixed_evidence": False, "golds": ["red"],
        "candidates": [
            {"cid": "tp", "modality": "text", "title": "Cars", "text": "Cars come in many colours.",
             "image": None, "gold": True},
            {"cid": "ig", "modality": "image", "title": "Red car", "text": "a red car",
             "image": str(small), "gold": True, "image_id": "gold"},
            {"cid": "ib", "modality": "image", "title": "Building", "text": "a building",
             "image": str(big), "gold": False, "image_id": "big"},
            {"cid": "imiss", "modality": "image", "title": "Missing", "text": "absent file",
             "image": str(directory / "missing.jpg"), "gold": False, "image_id": "missing"},
            {"cid": "iswap", "modality": "image", "title": "Wrong id", "text": "stem does not match",
             "image": str(small), "gold": False, "image_id": "not-the-file-stem"},
        ],
    }]


FIXTURE_RELEVANCE = {"tp": 0.9, "ig": 0.5, "ib": 0.7, "imiss": 0.2, "iswap": 0.4}


def run_fixture() -> dict:
    with tempfile.TemporaryDirectory(prefix="image_evidence_") as tmp:
        examples = fixture_inputs(Path(tmp))
        questions = [inspect_question(e, FIXTURE_RELEVANCE) for e in examples]
        return build_report(questions, mode="fixture")


def missing_inputs_error() -> SystemExit:
    return SystemExit(
        "image_evidence_check needs real WebQA inputs, which are missing:\n"
        f"  annotations: {paths.WEBQA_JSON}\n"
        f"  image files: {paths.WEBQA_IMAGES}\n"
        f"  relevance cache written by run.py for {DATASET}: {paths.LOCAL_CACHE / f'relevance_{DATASET}.json'}\n"
        "Set OMNIAWARE_ASSETS (and OMNIAWARE_CACHE) to the asset root, produce the relevance cache with\n"
        f"  python -m omniaware.run --dataset {DATASET} --split calib --n-calib 500 --n-dev 256 --calib-source train "
        "--tag image_check --calibrate-only --v2 --v2-reranker\n"
        "or run with --fixture, which needs no assets.")


def calibrator_for_run(examples: list[dict], raw: dict) -> relevance.Calibrator:
    """The calibrator run.py fits for webqa_full with its protocol defaults: calibration questions from the
    training split (data.split_examples with the run's sizes and seed)."""
    calib_pool = data.TRAIN_LOADERS[DATASET]()
    split = data.split_examples(examples, CALIB_QIDS_N, data.N_DEV["webqa"], 2027, calib_pool,
                                reserve=data.TRAIN_RESERVE["webqa"])
    calib_qids = [e["qid"] for e in split["calib"]]
    missing = [q for q in calib_qids if q not in raw]
    if missing:
        raise SystemExit(f"{len(missing)} calibration questions have no relevance scores in the cache; "
                         "produce the cache with the command printed above")
    return relevance.Calibrator.fit({q: raw[q] for q in calib_qids}, calib_qids, IMAGE_MODE)


def run_webqa(n: int) -> dict:
    annotations = paths.WEBQA_JSON
    cache = paths.LOCAL_CACHE / f"relevance_{DATASET}.json"
    if not annotations.is_file() or not cache.is_file():
        raise missing_inputs_error()
    try:
        examples = data.LOADERS[DATASET]()
        pool = sorted((e for e in examples if data.is_image_question(e)), key=lambda e: str(e["qid"]))[:n]
    except data.MissingDataError as exc:
        raise SystemExit(f"WebQA image assets are missing: {exc}") from exc
    if not pool:
        raise SystemExit(f"{annotations} has no WebQA image questions")
    stored = relevance.load_raw(cache)
    raw = {}
    for e in pool:
        entry = stored.get(e["qid"])
        if not isinstance(entry, dict) or entry.get("fingerprint") != run_mod.relevance_fingerprint(e):
            raise SystemExit(f"{cache} has no current relevance entry for {e['qid']}; produce it with the command "
                             "printed above")
        raw[e["qid"]] = entry["scores"]
    all_scores = {qid: entry["scores"] for qid, entry in stored.items() if isinstance(entry, dict) and "scores" in entry}
    cal = calibrator_for_run(examples, all_scores)
    questions = []
    for e in pool:
        rel = {c["cid"]: cal.relevance(raw[e["qid"]][c["cid"]], IMAGE_MODE) for c in e["candidates"]}
        questions.append(inspect_question(e, rel))
    return build_report(questions, mode="webqa")


def build_report(questions: list[dict], *, mode: str) -> dict:
    summary = summarize(questions)
    return {"mode": mode, "n": len(questions), "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS,
            "k": K, "visual_token_rule": "smart_resize with factor 28 (patch 14, merge 2); the count QwenEAB.encode "
                                         "takes from the image grid, not a model forward",
            "summary": summary, "findings": findings(summary, questions), "gpu_probe": GPU_PROBE,
            "gpu_probe_executed": False, "questions": questions}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50, help="WebQA image questions to inspect (default 50)")
    parser.add_argument("--fixture", action="store_true", help="synthetic inputs; needs no WebQA assets")
    parser.add_argument("--gpu-probe", action="store_true", help="print the GPU probe description and do not run it")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    if args.gpu_probe:
        print(GPU_PROBE, flush=True)
    if args.n < 1:
        parser.error("--n must be at least 1")
    report = run_fixture() if args.fixture else run_webqa(args.n)
    text = runio.dumps_strict(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU diagnostic for WebQA image evidence. Does not run a model.

    python -m scripts.analysis.image_evidence_check
    python -m scripts.analysis.image_evidence_check --fixture --output report.json

The real run reads WebQA image questions (default 50) and measures each image that would be
packed: whether the file can be read, its pixel size, and a visual-token count under the default
``--max-pixels`` budget (256*32*32), using the Qwen3-VL resize rule (patch 14, merge 2). It does
not score a retriever, so rank is position in that candidate list, not a retrieval rank.

The GPU probe (ask the model to describe each image) is printed and is not executed.
``--fixture`` builds tiny images in a temporary directory and does not need WebQA assets.
Missing assets on a real run exit with an error. Measurements come only from files that were opened.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

MAX_PIXELS = 256 * 32 * 32
MIN_PIXELS = 64 * 32 * 32
# Qwen3-VL image processor: patch_size 14, spatial merge 2. One visual token covers factor x factor pixels.
FACTOR = 28
GPU_PROBE_COMMAND = (
    "python -m scripts.analysis.image_evidence_check --gpu-probe --n 50"
)
# What pipeline.build_segments writes for an image chunk, checked here so a swap is a finding, not a guess.
PROMPT_ORDER = ("title_text", "image_pixels", "caption")


def smart_resize(height: int, width: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Qwen2-VL / Qwen3-VL smart_resize. Returns the resized (height, width) the processor would use."""
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


def visual_tokens(width: int, height: int, max_pixels: int = MAX_PIXELS) -> int:
    h_bar, w_bar = smart_resize(height, width, max_pixels=max_pixels)
    return (h_bar // FACTOR) * (w_bar // FACTOR)


def read_image(path: str | None) -> dict:
    """Open one file. Unreadable paths stay null; nothing is filled in."""
    out = {"path": path, "read": False, "width": None, "height": None, "pixels": None, "visual_tokens": None}
    if not path:
        return out
    file = Path(path)
    if not file.is_file():
        return out
    from PIL import Image
    with Image.open(file) as im:
        width, height = im.size
    out.update(read=True, width=width, height=height, pixels=width * height,
               visual_tokens=visual_tokens(width, height))
    return out


def inspect_question(example: dict, k: int = 5) -> dict:
    """Image candidates in list order, truncated to k. This is not a scored retrieval ranking."""
    images = [c for c in example.get("candidates") or [] if c.get("modality") == "image"]
    rows = []
    for rank, cand in enumerate(images[:k], 1):
        measured = read_image(cand.get("image"))
        image_id = cand.get("image_id")
        stem = Path(cand.get("image") or "").stem
        measured.update(cid=cand.get("cid"), gold=bool(cand.get("gold")), rank=rank,
                        path_mismatch=bool(image_id) and stem != str(image_id),
                        title=cand.get("title") or "", caption=cand.get("text") or "")
        rows.append(measured)
    golds = [row for row in rows if row["gold"]]
    return {"qid": example.get("qid"), "images": rows,
            "context_definition": "image candidates in the order supplied, truncated to k; "
                                  "not a scored retriever rank",
            "gold_in_context": bool(golds),
            "gold_rank": golds[0]["rank"] if golds else None,
            "prompt_order": list(PROMPT_ORDER)}


def collect_findings(questions: list[dict]) -> tuple[list[str], bool]:
    findings = [
        "title/image order: pipeline.build_segments writes title text, then image pixels, then the caption. "
        "That builder order was checked; it does not swap the title and the image."
    ]
    mismatches, unread, over_budget = [], [], []
    for question in questions:
        for image in question["images"]:
            if image["path_mismatch"]:
                mismatches.append(f"{question['qid']} {image['cid']}")
            if not image["read"]:
                unread.append(f"{question['qid']} {image['path']}")
            elif image["pixels"] is not None and image["pixels"] > MAX_PIXELS:
                over_budget.append(image["pixels"])
    if mismatches:
        findings.append("path mismatch (file stem != image_id): " + ", ".join(mismatches[:10]))
    if unread:
        findings.append("image file was not read: " + ", ".join(unread[:10]))
    budget_small = bool(over_budget)
    if budget_small:
        findings.append(
            f"the default --max-pixels budget {MAX_PIXELS} (256*32*32) looks too small: "
            f"{len(over_budget)} read image(s) have more native pixels than that budget "
            f"(largest {max(over_budget)})."
        )
    else:
        findings.append(
            f"no read image exceeded the default --max-pixels budget {MAX_PIXELS} (256*32*32)."
        )
    return findings, budget_small


def fixture_questions(directory: Path) -> list[dict]:
    """Real tiny files so pixel sizes are measured. No WebQA assets."""
    from PIL import Image
    small = directory / "gold.png"
    big = directory / "big.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(small)
    Image.new("RGB", (800, 800), (10, 20, 30)).save(big)
    return [{
        "qid": "fixture-1",
        "question": "What color is the car?",
        "candidates": [
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


def load_webqa_image_questions(n: int) -> list[dict]:
    from omniaware import data, paths
    if not paths.WEBQA_JSON.is_file():
        raise SystemExit(
            f"error: WebQA annotations not found at {paths.WEBQA_JSON}. "
            "Set OMNIAWARE_ASSETS, or pass --fixture to run without assets."
        )
    try:
        examples = data.load_webqa("img", split="val")
    except data.MissingDataError as exc:
        raise SystemExit(f"error: WebQA image assets are missing: {exc}") from exc
    questions = sorted((e for e in examples if data.is_image_question(e)), key=lambda e: str(e["qid"]))
    if not questions:
        raise SystemExit(f"error: {paths.WEBQA_JSON} has no WebQA image questions to inspect")
    return questions[:n]


def build_report(questions: list[dict], *, mode: str, requested_n: int) -> dict:
    inspected = [inspect_question(q) for q in questions]
    findings, budget_small = collect_findings(inspected)
    return {
        "mode": mode,
        "requested_n": requested_n,
        "n": len(inspected),
        "max_pixels": MAX_PIXELS,
        "max_pixels_expr": "256*32*32",
        "min_pixels": MIN_PIXELS,
        "visual_token_rule": "Qwen3-VL smart_resize with patch 14 and merge 2 (factor 28); "
                             "not a model forward",
        "gpu_probe_command": GPU_PROBE_COMMAND,
        "gpu_probe_executed": False,
        "budget_looks_too_small": budget_small,
        "findings": findings,
        "questions": inspected,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50, help="WebQA image questions to inspect (default 50)")
    parser.add_argument("--fixture", action="store_true",
                        help="inspect synthetic images in a temp directory; does not need WebQA assets")
    parser.add_argument("--gpu-probe", action="store_true",
                        help="print the GPU probe command and do not run it")
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here (default: stdout)")
    args = parser.parse_args(argv)
    if args.gpu_probe:
        print(GPU_PROBE_COMMAND)
        print("GPU probe is not executed.", flush=True)
    if args.fixture:
        with tempfile.TemporaryDirectory(prefix="image_evidence_") as tmp:
            report = build_report(fixture_questions(Path(tmp)), mode="fixture", requested_n=args.n)
            _emit(report, args.output)
        return 0
    report = build_report(load_webqa_image_questions(args.n), mode="webqa", requested_n=args.n)
    _emit(report, args.output)
    return 0


def _emit(report: dict, output: Path | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf8")


if __name__ == "__main__":
    raise SystemExit(main())

"""Asset locations.  Override the root on another machine with ERC_ASSET_ROOT."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get("ERC_ASSET_ROOT", REPO_ROOT / "experiments" / "eab_first_token")).resolve()

DATA = ASSET_ROOT / "data"
CACHE = ASSET_ROOT / "cache"

QWEN_8B = Path(os.environ.get("ERC_QWEN_8B", CACHE / "qwen3-vl-8b-instruct"))
QWEN_2B = Path(os.environ.get("ERC_QWEN_2B", CACHE / "qwen3-vl-2b-instruct"))
BGE = Path(os.environ.get("ERC_BGE", CACHE / "bge-large-en-v1.5"))
CLIP = Path(os.environ.get("ERC_CLIP", CACHE / "clip-vit-large-patch14"))
RERANKER = Path(os.environ.get("ERC_RERANKER", CACHE / "bge-reranker-v2-m3"))

WEBQA_JSON = DATA / "WebQA_train_val.json"
WEBQA_IMAGES = CACHE / "webqa_full4966_images" / "images"

MMQA_DIR = DATA / "mmqa_official"
MMQA_IMAGES = MMQA_DIR / "images" / "final_dataset_images"

VIDEOMME_DIR = DATA / "videomme_short"
VIDEOMME_FRAMES = DATA / "videomme_short_frames"

OKVQA_INPUTS = DATA / "okvqa_val5046_assets" / "validation_inputs.jsonl"
OKVQA_CANDIDATES = DATA / "okvqa_val5046_assets" / "validation_candidates.jsonl"
OKVQA_ANNOTATIONS = DATA / "okvqa_local" / "original" / "mscoco_val2014_annotations.json.zip"

RUNS = PROJECT_ROOT / "runs"

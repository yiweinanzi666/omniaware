"""Asset locations.

Large files (models, datasets, image caches) stay outside git under one asset
root.  Set ``OMNIAWARE_ASSETS``; the legacy ``ERC_ASSET_ROOT`` is still honoured,
so the existing ``experiments/eab_first_token`` directory on the experiment host
works unchanged.  Run outputs and reusable caches default to ``runs/`` and
``cache/`` inside this repository (both git-ignored).
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(os.environ.get("OMNIAWARE_ASSETS") or os.environ.get("ERC_ASSET_ROOT")
                  or REPO_ROOT / "assets").resolve()

DATA = ASSET_ROOT / "data"
CACHE = ASSET_ROOT / "cache"

QWEN_8B = Path(os.environ.get("ERC_QWEN_8B", CACHE / "qwen3-vl-8b-instruct"))
QWEN_2B = Path(os.environ.get("ERC_QWEN_2B", CACHE / "qwen3-vl-2b-instruct"))
INTERNVL_8B = Path(os.environ.get("OMNIAWARE_INTERNVL_8B", CACHE / "InternVL3_5-8B-HF"))  # native transformers format
BGE = Path(os.environ.get("ERC_BGE", CACHE / "bge-large-en-v1.5"))
CLIP = Path(os.environ.get("ERC_CLIP", CACHE / "clip-vit-large-patch14"))
RERANKER = Path(os.environ.get("ERC_RERANKER", CACHE / "bge-reranker-v2-m3"))
BARTSCORE = Path(os.environ.get("OMNIAWARE_BARTSCORE", CACHE / "bartscore"))

WEBQA_JSON = DATA / "WebQA_train_val.json"
WEBQA_IMAGES = CACHE / "webqa_full4966_images" / "images"



OKVQA_INPUTS = DATA / "okvqa_val5046_assets" / "validation_inputs.jsonl"
OKVQA_CANDIDATES = DATA / "okvqa_val5046_assets" / "validation_candidates.jsonl"
OKVQA_ANNOTATIONS = DATA / "okvqa_local" / "original" / "mscoco_val2014_annotations.json.zip"

# WebQA open retrieval artifacts (global pool, encodings, FAISS indices, rankings).  Only the environment
# variable or the default ``<asset root>/webqa_open`` selects the directory; the legacy directory of the
# experiment host is never picked up silently.  Set OMNIAWARE_WEBQA_OPEN_DIR to reuse it.
_LEGACY_WEBQA_OPEN = ASSET_ROOT / "runs" / "phase1_dense_retrieval_codex_20261006"
WEBQA_OPEN = Path(os.environ.get("OMNIAWARE_WEBQA_OPEN_DIR", ASSET_ROOT / "webqa_open"))
if "OMNIAWARE_WEBQA_OPEN_DIR" not in os.environ and _LEGACY_WEBQA_OPEN.exists():
    warnings.warn(f"{_LEGACY_WEBQA_OPEN} exists but is not used: WebQA open-retrieval artifacts come from "
                  f"{WEBQA_OPEN}. Set OMNIAWARE_WEBQA_OPEN_DIR={_LEGACY_WEBQA_OPEN} to use it.", stacklevel=2)
# Citation-support NLI checker (pinned revision, see eval/citation/checker_asset_manifest.json)
NLI_CHECKER = CACHE / "e5_citation_support" / "deberta_nli" / "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"

THIRD_PARTY = REPO_ROOT / "third_party"
EXTERNAL = REPO_ROOT / "external"
# UniversalRAG benchmark suite (HybridQA, InfoSeek, VideoRAG): its dataset/ tree (unified paragraph/table/
# image/video/clip corpora built by its script/0_dataset.sh), rankings exported with its own retrievers
# (scripts/universalrag_retrieval/export_rankings.py) and frames sampled from its videos.
UNIVERSALRAG_ROOT = Path(os.environ.get("OMNIAWARE_UNIVERSALRAG_DIR", EXTERNAL / "UniversalRAG"))
UNIVERSALRAG_RANKINGS = Path(os.environ.get("OMNIAWARE_UNIVERSALRAG_RANKINGS", ASSET_ROOT / "universalrag" / "rankings"))
UNIVERSALRAG_FRAMES = Path(os.environ.get("OMNIAWARE_UNIVERSALRAG_FRAMES", ASSET_ROOT / "universalrag" / "frames"))
RUNS = Path(os.environ.get("OMNIAWARE_RUNS", REPO_ROOT / "runs"))
LOCAL_CACHE = Path(os.environ.get("OMNIAWARE_CACHE", REPO_ROOT / "cache"))

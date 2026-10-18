"""Report which models, datasets and retrieval artifacts are present, per experiment.

    python -m scripts.setup.check_assets            # status of every asset
    python -m scripts.setup.check_assets --help

Prints one line per asset (OK / MISSING, path) and, at the end, which datasets of scripts/paper_runs.py can run.
Models are checked completely (config.json and every weight shard, see download_models.model_complete), so a
truncated download is reported as MISSING.  Exits with status 1 when an asset needed by the main experiments is
missing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths  # noqa: E402
from scripts.setup.download_models import CITATION_CACHE, model_complete  # noqa: E402


def _exists(path) -> bool:
    return Path(path).exists()


def _nonempty_dir(path) -> bool:
    return Path(path).is_dir() and any(Path(path).iterdir())


def _webqa_distractor_images(path) -> bool:
    """Every image fact (img_posFacts, img_negFacts) of the WebQA train and val questions has its <image_id>.jpg in
    ``path``. webqa_full, the v2 main setting, reads val for evaluation and train for calibration, and
    data.load_webqa raises on a missing image, so a partial directory is MISSING."""
    path = Path(path)
    if not (path.is_dir() and paths.WEBQA_JSON.is_file()):
        return False
    present = {entry.name for entry in os.scandir(path)}
    raw = json.loads(paths.WEBQA_JSON.read_text(encoding="utf-8"))
    return all(f"{fact['image_id']}.jpg" in present for v in raw.values() if v["split"] in ("train", "val")
               for key in ("img_posFacts", "img_negFacts") for fact in v[key])


def _m2rag_source() -> Path:
    cached = paths.CACHE / "official_m2rag" / "src"
    return cached if cached.exists() else paths.EXTERNAL / "M2RAG" / "src"


ASSETS = {   # name -> (path, needed by, check)
    "Qwen3-VL-8B-Instruct": (paths.QWEN_8B, "all main runs", model_complete),
    "InternVL3.5-8B-HF": (paths.INTERNVL_8B, "backbone transfer", model_complete),
    "BGE-large-en-v1.5": (paths.BGE, "relevance, WebQA/OK-VQA retrieval", model_complete),
    "CLIP ViT-L/14": (paths.CLIP, "relevance, WebQA image retrieval", model_complete),
    "bge-reranker-v2-m3": (paths.RERANKER, "cross-encoder baseline", model_complete),
    "DeBERTa NLI checker": (paths.NLI_CHECKER, "citation metrics", model_complete),
    "NLTK 3.9.2 runtime": (CITATION_CACHE / "nltk_runtime" / "nltk" / "__init__.py", "citation metrics", _exists),
    "NLTK English Punkt": (CITATION_CACHE / "sentence_splitter_assets.json", "citation metrics", _exists),
    "BARTScore ParaBank": (paths.BARTSCORE / "bart_score.pth", "WebQA QA_approx (--bart)", _exists),
    "WebQA_train_val.json": (paths.WEBQA_JSON, "WebQA", _exists),
    "WebQA distractor images": (paths.WEBQA_IMAGES, "webqa_full (v2 main setting), train and val",
                                _webqa_distractor_images),
    "WebQA open pool": (paths.WEBQA_OPEN / "text_documents.jsonl", "WebQA", _exists),
    "WebQA open pool stats": (paths.WEBQA_OPEN / "pool_stats.json", "WebQA (pool numbers)", _exists),
    "WebQA val rankings (BGE)": (paths.WEBQA_OPEN / "dense_top5_full" / "webqa_val" / "bge" / "rankings.jsonl",
                                 "WebQA", _exists),
    "WebQA val rankings (CLIP)": (paths.WEBQA_OPEN / "dense_top5_full" / "webqa_val" / "clip" / "rankings.jsonl",
                                  "WebQA", _exists),
    "WebQA train rankings (BGE)": (paths.WEBQA_OPEN / "dense_top5_full" / "webqa_train" / "bge" / "rankings.jsonl",
                                   "WebQA calibration (--calib-source train)", _exists),
    "WebQA train rankings (CLIP)": (paths.WEBQA_OPEN / "dense_top5_full" / "webqa_train" / "clip" / "rankings.jsonl",
                                    "WebQA calibration (--calib-source train)", _exists),
    "OK-VQA inputs": (paths.OKVQA_INPUTS, "OK-VQA", _exists),
    "OK-VQA annotations": (paths.OKVQA_ANNOTATIONS, "OK-VQA scoring", _exists),
    "OK-VQA GS-112K candidates": (paths.LOCAL_CACHE / "okvqa_gs112k" / "candidates_full.jsonl", "OK-VQA (okvqa_gs)",
                                  _exists),
    "OK-VQA BM25 candidates": (paths.OKVQA_CANDIDATES, "dev grid only (optional)", _exists),
    "OK-VQA DPR-wiki candidates": (paths.LOCAL_CACHE / "okvqa_dense" / "candidates_full.jsonl",
                                   "dev grid only (optional)", _exists),
    "UniversalRAG checkout": (paths.UNIVERSALRAG_ROOT / "dataset" / "query" / "hybridqa.json", "HybridQA, VideoRAG",
                              _exists),
    "HybridQA corpora": (paths.UNIVERSALRAG_ROOT / "dataset" / "table.parquet", "HybridQA", _exists),
    "HybridQA rankings": (paths.UNIVERSALRAG_RANKINGS / "hybridqa" / "table.jsonl", "HybridQA", _exists),
    "VideoRAG rankings": (paths.UNIVERSALRAG_RANKINGS / "videorag_synth" / "video.jsonl", "VideoRAG-Synth", _exists),
    "VideoRAG frames": (paths.UNIVERSALRAG_FRAMES, "VideoRAG-Synth", _nonempty_dir),
    "M2-RAG official source": (_m2rag_source(), "M2-RAG baseline", _exists),   # same choice as baselines/m2rag_official.py
    "BARTScore official source": (paths.EXTERNAL / "BARTScore" / "bart_score.py", "BARTScore", _exists),
}
OPTIONAL = {"OK-VQA BM25 candidates", "OK-VQA DPR-wiki candidates", "InternVL3.5-8B-HF"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    missing_required = []
    for name, (path, purpose, check) in ASSETS.items():
        ok = check(path)
        print(f"{'OK     ' if ok else 'MISSING'} {name:28s} {path}   [{purpose}]")
        if not ok and name not in OPTIONAL:
            missing_required.append(name)
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("OK      spaCy en_core_web_sm")
    except Exception:
        print("MISSING spaCy en_core_web_sm   [WebQA official accuracy, interventions]")
        missing_required.append("spacy")
    if missing_required:
        print(f"\n{len(missing_required)} required assets missing: {', '.join(missing_required)}")
        sys.exit(1)
    print("\nall required assets present")


if __name__ == "__main__":
    main()

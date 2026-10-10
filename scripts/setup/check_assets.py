"""Report which models, datasets and retrieval artifacts are present, per experiment.

    python -m scripts.setup.check_assets

Prints one line per asset (OK / MISSING, path) and, at the end, which datasets of scripts/paper_runs.py can run.
Exits with status 1 when an asset needed by the main experiments is missing.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths  # noqa: E402

ASSETS = {   # name -> (path, needed by)
    "Qwen3-VL-8B-Instruct": (paths.QWEN_8B / "config.json", "all main runs"),
    "InternVL3.5-8B-HF": (paths.INTERNVL_8B / "config.json", "backbone transfer"),
    "BGE-large-en-v1.5": (paths.BGE / "config.json", "relevance, WebQA/OK-VQA retrieval"),
    "CLIP ViT-L/14": (paths.CLIP / "config.json", "relevance, WebQA image retrieval"),
    "bge-reranker-v2-m3": (paths.RERANKER / "config.json", "cross-encoder baseline"),
    "DeBERTa NLI checker": (paths.NLI_CHECKER / "model.safetensors", "citation metrics"),
    "BARTScore ParaBank": (paths.BARTSCORE / "bart_score.pth", "WebQA QA_approx (--bart)"),
    "WebQA_train_val.json": (paths.WEBQA_JSON, "WebQA"),
    "WebQA open pool": (paths.WEBQA_OPEN / "text_documents.jsonl", "WebQA"),
    "WebQA val rankings": (paths.WEBQA_OPEN / "dense_top5_full" / "webqa_val" / "clip" / "rankings.jsonl", "WebQA"),
    "WebQA train rankings": (paths.WEBQA_OPEN / "dense_top5_full" / "webqa_train" / "clip" / "rankings.jsonl",
                             "WebQA calibration (--calib-source train)"),
    "OK-VQA inputs": (paths.OKVQA_INPUTS, "OK-VQA"),
    "OK-VQA annotations": (paths.OKVQA_ANNOTATIONS, "OK-VQA scoring"),
    "OK-VQA GS-112K candidates": (paths.LOCAL_CACHE / "okvqa_gs112k" / "candidates_full.jsonl", "OK-VQA (okvqa_gs)"),
    "OK-VQA BM25 candidates": (paths.OKVQA_CANDIDATES, "dev grid only (optional)"),
    "OK-VQA DPR-wiki candidates": (paths.LOCAL_CACHE / "okvqa_dense" / "candidates_full.jsonl", "dev grid only (optional)"),
    "UniversalRAG checkout": (paths.UNIVERSALRAG_ROOT / "dataset" / "query" / "hybridqa.json", "HybridQA, VideoRAG"),
    "HybridQA corpora": (paths.UNIVERSALRAG_ROOT / "dataset" / "table.parquet", "HybridQA"),
    "HybridQA rankings": (paths.UNIVERSALRAG_RANKINGS / "hybridqa" / "table.jsonl", "HybridQA"),
    "VideoRAG rankings": (paths.UNIVERSALRAG_RANKINGS / "videorag_synth" / "video.jsonl", "VideoRAG-Synth"),
    "VideoRAG frames": (paths.UNIVERSALRAG_FRAMES, "VideoRAG-Synth"),
    "M2-RAG official source": (paths.EXTERNAL / "M2RAG" / "src", "M2-RAG baseline"),
    "BARTScore official source": (paths.EXTERNAL / "BARTScore" / "bart_score.py", "BARTScore"),
}
OPTIONAL = {"OK-VQA BM25 candidates", "OK-VQA DPR-wiki candidates", "InternVL3.5-8B-HF"}


def main():
    missing_required = []
    for name, (path, purpose) in ASSETS.items():
        ok = Path(path).exists()
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

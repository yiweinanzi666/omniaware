"""Download every model the experiments use into the asset layout of omniaware.paths.

    python -m scripts.setup.download_models                 # all (about 45 GB)
    python -m scripts.setup.download_models --only qwen8b bge clip
    HF_ENDPOINT=https://hf-mirror.com python -m scripts.setup.download_models      # mirror, if needed

Weights come from the Hugging Face Hub (huggingface_hub.snapshot_download); the BARTScore ParaBank
checkpoint comes from the Google Drive link in the BARTScore README (needs ``gdown``). Every target is
skipped when its directory already holds the expected files, so the command can be re-run after a failure.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths  # noqa: E402

BART_CNN_REVISION = "main"
BART_PARABANK_DRIVE = "https://drive.google.com/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m"
NLI_REPO, NLI_REVISION = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"

HF_MODELS = {   # name -> (repo, revision, target, allow_patterns or None, purpose)
    "qwen8b": ("Qwen/Qwen3-VL-8B-Instruct", None, paths.QWEN_8B, None, "primary backbone"),
    "qwen2b": ("Qwen/Qwen3-VL-2B-Instruct", None, paths.QWEN_2B, None, "smoke tests / quick dev"),
    "internvl": ("OpenGVLab/InternVL3_5-8B-HF", None, paths.INTERNVL_8B, None, "backbone transfer"),
    "bge": ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09", paths.BGE,
            ["*.json", "*.txt", "model.safetensors", "README.md"], "text relevance r_j, WebQA/OK-VQA retrieval"),
    "clip": ("openai/clip-vit-large-patch14", None, paths.CLIP,
             ["*.json", "*.txt", "model.safetensors"], "image relevance r_j, WebQA image retrieval"),
    "reranker": ("BAAI/bge-reranker-v2-m3", None, paths.RERANKER, ["*.json", "*.model", "model.safetensors"],
                 "cross-encoder baseline"),
    "nli": (NLI_REPO, NLI_REVISION, paths.NLI_CHECKER, None, "citation support checker (Faith/AP)"),
    "bart_cnn_config": ("facebook/bart-large-cnn", BART_CNN_REVISION, paths.BARTSCORE / "bart-large-cnn-config",
                        ["config.json", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"],
                        "BARTScore architecture and tokenizer (weights come from ParaBank)"),
}


def hf_download(repo, revision, target: Path, allow):
    from huggingface_hub import snapshot_download
    if target.exists() and any(target.glob("*.json")):
        print(f"[skip] {target}", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), allow_patterns=allow)
    print(f"[ok] {repo}@{revision or 'main'} -> {target}", flush=True)


def check_nli():
    manifest = json.loads((ROOT / "src/omniaware/eval/citation/checker_asset_manifest.json").read_text())
    import hashlib
    for rec in manifest["files"]:
        with (paths.NLI_CHECKER / rec["file"]).open("rb") as fh:
            if hashlib.file_digest(fh, "sha256").hexdigest() != rec["sha256"]:
                raise SystemExit(f"NLI checker file differs from the pinned manifest: {rec['file']}")
    print("[ok] NLI checker matches checker_asset_manifest.json", flush=True)


def bartscore():
    """bart_score.pth (ParaBank2) + the official bart_score.py + provenance files that webqa_bart checks."""
    target = paths.BARTSCORE
    weights = target / "bart_score.pth"
    if not weights.exists():
        subprocess.run([sys.executable, "-m", "gdown", BART_PARABANK_DRIVE, "-O", str(weights)], check=True)
    source = ROOT / "external" / "BARTScore" / "bart_score.py"
    if not source.exists():
        subprocess.run([sys.executable, "scripts/fetch_external.py", "--only", "BARTScore"], check=True, cwd=ROOT)
    (target / "official_source").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / "official_source" / "bart_score.py")
    commit = json.loads((ROOT / "third_party/external_repos.json").read_text())
    commit = next(r["ref"] for r in commit["repos"] if r["name"] == "BARTScore")
    files = ["official_source/bart_score.py"] + [f"bart-large-cnn-config/{p.name}"
                                                for p in sorted((target / "bart-large-cnn-config").glob("*"))
                                                if p.is_file()]
    (target / "source_provenance.json").write_text(json.dumps(
        {"bartscore_commit": commit, "bart_config_revision": BART_CNN_REVISION,
         "files": [{"file": f} for f in files]}, indent=1))
    (target / "weights_provenance.json").write_text(json.dumps(
        {"source": BART_PARABANK_DRIVE, "description": "BARTScore trained on ParaBank2 (BARTScore README)"}, indent=1))
    print(f"[ok] BARTScore ParaBank -> {target}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=[*HF_MODELS, "bartscore", "spacy"], default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for name, (repo, rev, target, _, purpose) in HF_MODELS.items():
            print(f"{name:16s} {repo}@{rev or 'main'} -> {target}  ({purpose})")
        return
    wanted = args.only or [*HF_MODELS, "bartscore", "spacy"]
    for name in wanted:
        if name in HF_MODELS:
            repo, rev, target, allow, _ = HF_MODELS[name]
            hf_download(repo, rev, target, allow)
            if name == "nli":
                check_nli()
    if "bartscore" in wanted:
        if "bart_cnn_config" not in wanted:
            hf_download(*HF_MODELS["bart_cnn_config"][:4])
        bartscore()
    if "spacy" in wanted:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)


if __name__ == "__main__":
    main()

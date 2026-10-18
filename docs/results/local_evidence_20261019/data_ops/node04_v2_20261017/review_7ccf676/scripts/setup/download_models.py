"""Download every model the experiments use into the asset layout of omniaware.paths.

    python -m scripts.setup.download_models                 # all (about 45 GB)
    python -m scripts.setup.download_models --only qwen8b bge clip
    python -m scripts.setup.download_models --only nltk     # NLTK 3.9.2 + English Punkt (citation metrics)
    HF_ENDPOINT=https://hf-mirror.com python -m scripts.setup.download_models      # mirror, if needed

Weights come from the Hugging Face Hub (huggingface_hub.snapshot_download); the BARTScore ParaBank
checkpoint comes from the Google Drive link in the BARTScore README (needs ``gdown``).  Every Hub repository
is pinned to a commit (REVISIONS); the commits are the ``main`` heads queried with ``model_info`` on
2026-10-10.  snapshot_download is called every time: it resumes partial files and checks them against the Hub,
so an interrupted run is repaired by rerunning the command.  After each download the weight files are checked
against their own safetensors headers (model_complete).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths  # noqa: E402

BART_PARABANK_DRIVE = "https://drive.google.com/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m"
NLI_CHECKER_MANIFEST = ROOT / "src/omniaware/eval/citation/checker_asset_manifest.json"

# Hub commits (model_info(repo).sha at 2026-10-10).  Fill in a missing entry before downloading: a
# repository without a pinned revision is refused.
REVISIONS = {
    "qwen8b": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",      # Qwen/Qwen3-VL-8B-Instruct
    "qwen2b": "89644892e4d85e24eaac8bacfd4f463576704203",      # Qwen/Qwen3-VL-2B-Instruct
    "internvl": "741a7d03020411e666c6109218ab71e08151ef86",    # OpenGVLab/InternVL3_5-8B-HF
    "bge": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",         # BAAI/bge-large-en-v1.5
    "clip": "32bd64288804d66eefd0ccbe215aa642df71cc41",        # openai/clip-vit-large-patch14
    "reranker": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",    # BAAI/bge-reranker-v2-m3
    "nli": "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",         # MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
    "bart_cnn": "37f520fa929c961707657b28798b30c003dd100b",    # facebook/bart-large-cnn (config and tokenizer only)
}

# NLTK for the citation metrics (docs/citation_support_protocol.md): the version intake_answers.py asserts,
# and the nltk_data revision whose English Punkt tables are frozen by sha256.
NLTK_VERSION = "3.9.2"
NLTK_DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
PUNKT_ZIP_GIT_BLOB = "5e5ff6137d5ee6025e400d1c3a7b21914c48b635"   # packages/tokenizers/punkt_tab.zip at that revision
PUNKT_ENGLISH = ["abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt"]
CITATION_CACHE = paths.CACHE / "e5_citation_support"

def nli_allow_patterns() -> list[str]:
    """Only the files the NLI checker is verified against (no ONNX or .bin copies of the same model)."""
    return [rec["file"] for rec in json.loads(NLI_CHECKER_MANIFEST.read_text(encoding="utf8"))["files"]]


HF_MODELS = {   # name -> (repo, revision, target, allow_patterns or None, purpose)
    "qwen8b": ("Qwen/Qwen3-VL-8B-Instruct", REVISIONS["qwen8b"], paths.QWEN_8B, None, "primary backbone"),
    "qwen2b": ("Qwen/Qwen3-VL-2B-Instruct", REVISIONS["qwen2b"], paths.QWEN_2B, None, "smoke tests / quick dev"),
    "internvl": ("OpenGVLab/InternVL3_5-8B-HF", REVISIONS["internvl"], paths.INTERNVL_8B, None, "backbone transfer"),
    "bge": ("BAAI/bge-large-en-v1.5", REVISIONS["bge"], paths.BGE,
            ["*.json", "*.txt", "model.safetensors", "README.md"], "text relevance r_j, WebQA/OK-VQA retrieval"),
    "clip": ("openai/clip-vit-large-patch14", REVISIONS["clip"], paths.CLIP,
             ["*.json", "*.txt", "model.safetensors"], "image relevance r_j, WebQA image retrieval"),
    "reranker": ("BAAI/bge-reranker-v2-m3", REVISIONS["reranker"], paths.RERANKER,
                 ["*.json", "*.model", "model.safetensors"], "cross-encoder baseline"),
    "nli": ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", REVISIONS["nli"], paths.NLI_CHECKER,
            nli_allow_patterns(), "citation support checker (Faith/AP)"),
    "bart_cnn_config": ("facebook/bart-large-cnn", REVISIONS["bart_cnn"], paths.BARTSCORE / "bart-large-cnn-config",
                        ["config.json", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"],
                        "BARTScore architecture and tokenizer (weights come from ParaBank)"),
}


WEIGHTLESS = {"bart_cnn_config"}


def _shard_complete(path: Path) -> bool:
    """A safetensors file is complete when its size equals the length its own JSON header implies."""
    try:
        with path.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if not 0 < n < 100_000_000:
                return False
            header = json.loads(fh.read(n))
    except (OSError, struct.error, ValueError):
        return False
    ends = [v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
    return path.stat().st_size == 8 + n + (max(ends) if ends else 0)


def model_complete(folder: Path, needs_weights: bool = True) -> bool:
    """config.json plus every weight file: all shards named by model.safetensors.index.json, or model.safetensors."""
    folder = Path(folder)
    if not (folder / "config.json").exists():
        return False
    if not needs_weights:
        return True
    index = folder / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text(encoding="utf8"))["weight_map"].values()))
        return bool(shards) and all(_shard_complete(folder / s) for s in shards)
    single = folder / "model.safetensors"
    return single.exists() and _shard_complete(single)


def hf_download(repo, revision, target: Path, allow) -> None:
    from huggingface_hub import snapshot_download
    if not revision:
        raise SystemExit(f"{repo} has no pinned revision in REVISIONS; fill it in (model_info(repo).sha) first")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), allow_patterns=allow)
    print(f"[ok] {repo}@{revision} -> {target}", flush=True)


def check_nli():
    manifest = json.loads(NLI_CHECKER_MANIFEST.read_text(encoding="utf8"))
    for rec in manifest["files"]:
        with (paths.NLI_CHECKER / rec["file"]).open("rb") as fh:
            if hashlib.file_digest(fh, "sha256").hexdigest() != rec["sha256"]:
                raise SystemExit(f"NLI checker file differs from the pinned manifest: {rec['file']}")
    print("[ok] NLI checker matches checker_asset_manifest.json", flush=True)


# ------------------------------------------------------------------ NLTK --
def _git_blob_sha1(path: Path) -> str:
    """The sha1 git uses for a blob; the GitHub contents API reports it, so the download can be checked."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def extract_punkt_english(zip_path: Path, root: Path) -> Path:
    """Extract the four English Punkt tables to <root>/nltk_data/tokenizers/punkt_tab/english and record their
    sha256 in <root>/sentence_splitter_assets.json.  The first extraction freezes the hashes; later extractions
    must reproduce them.  Paths in the record are relative to OMNIAWARE_ASSETS (intake_answers.py reads them)."""
    target = root / "nltk_data" / "tokenizers" / "punkt_tab" / "english"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for name in PUNKT_ENGLISH:
            member = f"punkt_tab/english/{name}"
            if member not in names:
                raise SystemExit(f"{member} is missing from {zip_path}")
            data = zf.read(member)
            dest = target / name
            dest.write_bytes(data)
            records.append({"path": dest.relative_to(paths.ASSET_ROOT).as_posix(),
                            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    record = {"nltk_version": NLTK_VERSION, "nltk_data_revision": NLTK_DATA_REVISION,
              "punkt_zip_git_blob_sha1": PUNKT_ZIP_GIT_BLOB, "files": records}
    manifest = root / "sentence_splitter_assets.json"
    if manifest.exists():
        frozen = json.loads(manifest.read_text(encoding="utf8"))
        if frozen.get("files") != records:
            raise SystemExit(f"English Punkt tables differ from the sha256 frozen in {manifest}; "
                             "delete that file only if the nltk_data pin was changed on purpose")
    else:
        manifest.write_text(json.dumps(record, indent=1), encoding="utf8")
    print(f"[ok] English Punkt tables (nltk_data {NLTK_DATA_REVISION[:10]}) -> {target}", flush=True)
    return manifest


def nltk_punkt() -> None:
    """NLTK 3.9.2 into the citation cache, and the pinned English Punkt tables (see docs/citation_support_protocol.md)."""
    runtime = CITATION_CACHE / "nltk_runtime"
    if not (runtime / "nltk" / "__init__.py").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(runtime), f"nltk=={NLTK_VERSION}"],
                       check=True)
    zip_path = CITATION_CACHE / "downloads" / f"punkt_tab-{NLTK_DATA_REVISION[:12]}.zip"
    if not zip_path.exists() or _git_blob_sha1(zip_path) != PUNKT_ZIP_GIT_BLOB:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_REVISION}/"
               "packages/tokenizers/punkt_tab.zip")
        tmp = zip_path.with_name(zip_path.name + ".part")
        print(f"[download] {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        if _git_blob_sha1(tmp) != PUNKT_ZIP_GIT_BLOB:
            tmp.unlink()
            raise SystemExit("punkt_tab.zip does not match the pinned nltk_data revision; not used")
        tmp.replace(zip_path)
    extract_punkt_english(zip_path, CITATION_CACHE)


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
        {"bartscore_commit": commit, "bart_config_revision": REVISIONS["bart_cnn"],
         "files": [{"file": f} for f in files]}, indent=1))
    (target / "weights_provenance.json").write_text(json.dumps(
        {"source": BART_PARABANK_DRIVE, "description": "BARTScore trained on ParaBank2 (BARTScore README)"}, indent=1))
    print(f"[ok] BARTScore ParaBank -> {target}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=[*HF_MODELS, "bartscore", "spacy", "nltk"], default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for name, (repo, rev, target, _, purpose) in HF_MODELS.items():
            print(f"{name:16s} {repo}@{rev or 'UNPINNED'} -> {target}  ({purpose})")
        print(f"{'nltk':16s} nltk=={NLTK_VERSION}, nltk_data@{NLTK_DATA_REVISION[:10]} -> {CITATION_CACHE}")
        return
    wanted = args.only or [*HF_MODELS, "bartscore", "spacy", "nltk"]
    for name in wanted:
        if name in HF_MODELS:
            repo, rev, target, allow, _ = HF_MODELS[name]
            hf_download(repo, rev, target, allow)
            if not model_complete(target, needs_weights=name not in WEIGHTLESS):
                raise SystemExit(f"{name}: {target} is incomplete after the download (a shard is missing or "
                                 "truncated); rerun the same command to resume")
            if name == "nli":
                check_nli()
    if "bartscore" in wanted:
        if "bart_cnn_config" not in wanted:
            hf_download(*HF_MODELS["bart_cnn_config"][:4])
        bartscore()
    if "nltk" in wanted:
        nltk_punkt()
    if "spacy" in wanted:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)


if __name__ == "__main__":
    main()

"""OK-VQA val (5,046 questions) inputs from the official release and COCO val2014 images.

    python -m scripts.setup.prepare_okvqa --download            # fetch zips (about 6.5 GB) and prepare
    python -m scripts.setup.prepare_okvqa --coco-dir /data/coco/val2014   # reuse an existing COCO copy

Writes paths.OKVQA_INPUTS (one {"question_id", "question", "image_path"} per line, no answers) and copies the
annotation zip to paths.OKVQA_ANNOTATIONS, where the loader and the official VQA scorer read it. Retrieval
candidates are produced separately (python -m omniaware.okvqa_dense --corpus gs112k ...).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths  # noqa: E402

URLS = {
    "questions": "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip",
    "annotations": "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip",
    "coco_val2014": "http://images.cocodataset.org/zips/val2014.zip",
}
RAW = paths.DATA / "okvqa_local" / "original"


def fetch(url: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"[download] {url}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)
    return target


def read_zip_json(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        return json.loads(zf.read(name))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--coco-dir", type=Path, default=None, help="directory with COCO_val2014_*.jpg")
    args = ap.parse_args(argv)
    q_zip, a_zip = RAW / Path(URLS["questions"]).name, RAW / Path(URLS["annotations"]).name
    if args.download:
        fetch(URLS["questions"], q_zip)
        fetch(URLS["annotations"], a_zip)
    for p in (q_zip, a_zip):
        if not p.exists():
            raise SystemExit(f"missing {p}: download it from okvqa.allenai.org or pass --download")
    coco = args.coco_dir or paths.DATA / "coco" / "val2014"
    if not coco.exists():
        if not args.download:
            raise SystemExit(f"missing COCO val2014 images at {coco}: pass --coco-dir or --download")
        archive = fetch(URLS["coco_val2014"], paths.DATA / "coco" / "val2014.zip")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(paths.DATA / "coco")
    questions = read_zip_json(q_zip)["questions"]
    if paths.OKVQA_ANNOTATIONS.resolve() != a_zip.resolve():
        paths.OKVQA_ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a_zip, paths.OKVQA_ANNOTATIONS)
    paths.OKVQA_INPUTS.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with paths.OKVQA_INPUTS.open("w", encoding="utf8") as out:
        for q in sorted(questions, key=lambda q: q["question_id"]):
            image = coco / f"COCO_val2014_{q['image_id']:012d}.jpg"
            missing += not image.exists()
            out.write(json.dumps({"question_id": q["question_id"], "question": q["question"],
                                  "image_id": q["image_id"], "image_path": str(image)}) + "\n")
    if missing:
        raise SystemExit(f"{missing} question images missing under {coco}")
    print(f"[ok] {len(questions)} questions -> {paths.OKVQA_INPUTS}; annotations -> {paths.OKVQA_ANNOTATIONS}")


if __name__ == "__main__":
    main()

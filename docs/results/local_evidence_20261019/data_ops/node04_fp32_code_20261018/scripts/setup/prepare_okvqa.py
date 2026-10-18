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
import time
import urllib.error
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


def fetch(url: str, target: Path, retries: int = 6, timeout: int = 60) -> Path:
    """Download ``url`` to ``target`` with a timeout, retries with backoff and resume.

    Bytes go to ``<target>.part``; after an interruption the next attempt asks for the rest with a Range request
    (a server that ignores Range makes it start over).  The result is checked against Content-Length before the
    part file is renamed, so a truncated download is never mistaken for a finished one."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    print(f"[download] {url}", flush=True)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if have and resp.status != 206:
                    have = 0                                   # the server ignored Range: start again
                length = resp.headers.get("Content-Length")
                expected = have + int(length) if length else None
                with part.open("ab" if have else "wb") as out:
                    shutil.copyfileobj(resp, out, 1 << 20)
            size = part.stat().st_size
            if expected is not None and size != expected:
                raise OSError(f"incomplete download: {size} of {expected} bytes")
            part.replace(target)
            return target
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part.exists():             # the part file already holds every byte
                part.replace(target)
                return target
            print(f"[retry {attempt}/{retries}] HTTP {exc.code}", flush=True)
        except (OSError, TimeoutError) as exc:
            print(f"[retry {attempt}/{retries}] {exc}", flush=True)
        if attempt < retries:
            time.sleep(min(60, 2 ** attempt))
    raise SystemExit(f"download of {url} failed after {retries} attempts; rerun the command to resume from {part}")


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

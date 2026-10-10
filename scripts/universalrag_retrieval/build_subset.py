"""Build UniversalRAG's unified corpora from the HybridQA and VideoRAG parts only.

UniversalRAG's ``script/0_dataset.sh`` downloads ten datasets (NQ, HotpotQA, InfoSeek, LVBench, ...) and merges
their passages into one paragraph pool. Our evaluation uses HybridQA and VideoRAG-Synth; this script builds the
same files from those two parts, after their own download scripts have run:

    cd external/UniversalRAG/dataset/HybridQA && bash get_hybridqa.sh     # tables + linked passages
    cd external/UniversalRAG/dataset/videorag && bash get_videorag.sh     # HowTo100M videos (yt-dlp) + scripts
    python -m scripts.universalrag_retrieval.build_subset [--extra-paragraphs nq.parquet ...]

Writes, inside the UniversalRAG checkout: dataset/paragraph.parquet (HybridQA passages plus any extra parquet
given), dataset/table.parquet (HybridQA tables), and filters dataset/video.parquet and dataset/clip.parquet to
the VideoRAG videos that were actually downloaded (the originals are kept as *.full.parquet). The candidate
pool is therefore smaller than the full suite's; run 0_dataset.sh instead to reproduce the full pool, and
report which one was used. Videos that YouTube no longer serves are listed in videorag_missing.txt.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths  # noqa: E402

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov")   # yt-dlp picks the container; the index names the .mp4


def find_video(path: Path) -> Path | None:
    """The downloaded file behind an index path: the path itself, or the same name in another container."""
    if path.exists():
        return path
    for ext in VIDEO_EXTENSIONS:
        alt = path.with_suffix(ext)
        if alt.exists():
            return alt
    return None


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=paths.UNIVERSALRAG_ROOT)
    ap.add_argument("--extra-paragraphs", nargs="*", type=Path, default=[],
                    help="further paragraph parquet files (psg_id index, text) to add as distractors")
    args = ap.parse_args(argv)
    ds = args.root / "dataset"

    text = ds / "HybridQA" / "hybridqa_text.parquet"
    table = ds / "HybridQA" / "hybridqa_table.parquet"
    for f in (text, table):
        if not f.exists():
            raise SystemExit(f"missing {f}: run dataset/HybridQA/get_hybridqa.sh first")
    frames = [pd.read_parquet(f) for f in [text, *args.extra_paragraphs]]
    paragraph = pd.concat(frames)
    paragraph = paragraph[~paragraph.index.duplicated()]
    paragraph.to_parquet(ds / "paragraph.parquet")
    shutil.copy2(table, ds / "table.parquet")
    print(f"[subset] paragraph.parquet {len(paragraph)} passages, table.parquet {len(pd.read_parquet(table))} tables")

    for name in ("video", "clip"):
        full = ds / f"{name}.full.parquet"
        if not full.exists():
            shutil.copy2(ds / f"{name}.parquet", full)
        df = pd.read_parquet(full)
        video_of = (lambda i: i.rsplit("_", 1)[0]) if name == "clip" else (lambda i: i)
        keep = [i for i in df.index if "videorag" in i and find_video(args.root / video_of(i)) is not None]
        if name == "video":
            wanted = [i for i in df.index if "videorag" in i]
            missing = sorted(set(wanted) - set(keep))
            (ds / "videorag_missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
            print(f"[subset] video.parquet {len(keep)} VideoRAG videos available, {len(missing)} missing")
        else:
            print(f"[subset] clip.parquet {len(keep)} clips of available VideoRAG videos")
        df.loc[keep].to_parquet(ds / f"{name}.parquet")


if __name__ == "__main__":
    main()

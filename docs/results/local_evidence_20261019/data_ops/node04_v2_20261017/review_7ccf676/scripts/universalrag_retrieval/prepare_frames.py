"""Sample frames for the videos and clips that appear in exported UniversalRAG rankings.

    python -m scripts.universalrag_retrieval.prepare_frames --target videorag_synth --frames 8

For every ranked video (dataset/videorag/videos/<id>.mp4, or the same name in another container) and clip (<video path>_<segment>, with start and
end times from dataset/clip.parquet) writes <UNIVERSALRAG_FRAMES>/<key>/frames.json and PNGs, with ``key``
from ``frame_key``. Frames are the midpoints of ``--frames`` equal temporal bins, as for Video-MME.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths  # noqa: E402
from omniaware.data import frame_key  # noqa: E402
from scripts.universalrag_retrieval.build_subset import find_video  # noqa: E402


def sample_frames(video: Path, out: Path, count: int, start: float | None = None, end: float | None = None):
    import av
    meta_path = out / "frames.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        t0 = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration / 1e6)
        lo, hi = (start or 0.0), (end if end is not None else duration)
        for i in range(count):
            target = lo + (hi - lo) * (2 * i + 1) / (2 * count)
            container.seek(int((t0 + target) / stream.time_base), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                seconds = float(frame.pts * frame.time_base) - t0
                if seconds >= target:
                    path = out / f"{i:02d}.png"
                    frame.to_image().convert("RGB").save(path, compress_level=1)
                    frames.append({"index": i, "target_seconds": target, "actual_seconds": seconds,
                                   "path": path.name})
                    break
    meta = {"video": str(video), "start": lo, "end": hi, "frames": frames}
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()
    root = paths.UNIVERSALRAG_ROOT
    keys = set()
    for modality in ("video", "clip"):
        path = paths.UNIVERSALRAG_RANKINGS / args.target / f"{modality}.jsonl"
        if path.exists():
            for line in path.open():
                keys |= {(modality, c["source_id"]) for c in json.loads(line)["candidates"]}
    times = {}
    if any(m == "clip" for m, _ in keys):
        import pandas as pd
        times = pd.read_parquet(root / "dataset" / "clip.parquet")["frame_time"].to_dict()
    missing = 0
    for modality, sid in sorted(keys):
        video = sid.rsplit("_", 1)[0] if modality == "clip" else sid
        source = find_video(root / video)
        if source is None:
            missing += 1
            continue
        start, end = (float(times[sid][0]), float(times[sid][1])) if modality == "clip" else (None, None)
        sample_frames(source, paths.UNIVERSALRAG_FRAMES / frame_key(sid), args.frames, start, end)
    print(f"[frames] {len(keys) - missing} videos/clips -> {paths.UNIVERSALRAG_FRAMES}")
    if missing:
        print(f"[frames] {missing} ranked videos have no file under {root} in any container; no frames written "
              "for them (data.load_universalrag counts them)", flush=True)


if __name__ == "__main__":
    main()

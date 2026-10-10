"""Merge the per-GPU shard files of one run into <dataset>.jsonl.

    python -m omniaware.merge_shards runs/<tag>/<dataset>

Shards are written by ``omniaware.run --num-shards N --shard-id i`` as
``<dataset>.shardII-of-NN.jsonl``. The merge checks that all N shards exist, that no question appears twice,
and that every question of the evaluation split was answered; the shard files then move to
``<dataset>.shards/``. Re-running the merge after an interrupted shard was resumed is safe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def shard_files(stem: Path) -> list[Path]:
    files = sorted(stem.parent.glob(f"{stem.name}.shard*-of-*.jsonl"))
    if not files:
        return []
    totals = {int(re.search(r"-of-(\d+)\.jsonl$", f.name).group(1)) for f in files}
    if len(totals) != 1:
        raise SystemExit(f"shards of different splits: {sorted(totals)}")
    n = totals.pop()
    expected = [stem.parent / f"{stem.name}.shard{i:02d}-of-{n:02d}.jsonl" for i in range(n)]
    missing = [f.name for f in expected if not f.exists()]
    if missing:
        raise SystemExit(f"missing shards: {missing}")
    return expected


def merge(stem: Path, expected_questions: int | None = None) -> Path:
    stem = Path(stem)
    files = shard_files(stem)
    if not files:
        raise SystemExit(f"no shard files for {stem}")
    rows, seen = [], set()
    for f in files:
        for line in f.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            qid = json.loads(line)["qid"]
            if qid in seen:
                raise SystemExit(f"question {qid} answered twice")
            seen.add(qid)
            rows.append(line)
    if expected_questions is not None and len(rows) != expected_questions:
        raise SystemExit(f"{len(rows)} answers, expected {expected_questions}: resume the unfinished shards first")
    out = stem.with_name(stem.name + ".jsonl")
    if out.exists():
        raise SystemExit(f"{out} exists; remove it to merge again")
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text("".join(r + "\n" for r in rows), encoding="utf8")
    tmp.replace(out)
    archive = stem.with_name(stem.name + ".shards")
    archive.mkdir(exist_ok=True)
    for f in files:
        f.replace(archive / f.name)
    print(f"[merge] {len(rows)} answers from {len(files)} shards -> {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", type=Path, help="runs/<tag>/<dataset> (without .jsonl)")
    ap.add_argument("--expected", type=int, default=None, help="number of evaluation questions, checked if given")
    args = ap.parse_args(argv)
    merge(args.stem, args.expected)


if __name__ == "__main__":
    main()

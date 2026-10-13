"""Merge the per-GPU shard files of one run into <dataset>.jsonl.

    python -m omniaware.merge_shards runs/<tag>/<dataset>                 # checks against <dataset>_config.json
    python -m omniaware.merge_shards runs/<tag>/<dataset> --expected N    # checks the count only, no configuration
    python -m omniaware.merge_shards runs/<tag>/<dataset> --max-empty-rate 0.005   # also fail above that rate

Shards are written by ``omniaware.run --num-shards N --shard-id i`` as
``<dataset>.shardII-of-NN.jsonl``. The merge checks that all N shards exist, that no question appears twice,
and that the answered questions are exactly the evaluation questions of the configuration (their count and the
sha256 of their sorted ids). The shard files then move to ``<dataset>.shards/`` and ``<dataset>.merged.json`` is
written, the completion marker that scripts/launch.py trusts. Re-running the merge after an interrupted shard was
resumed is safe, because nothing is moved until the check passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from . import answers, runio


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


def merge(stem: Path, expected_questions: int | None = None, expected_qids_sha256: str | None = None,
          max_empty_rate: float | None = None) -> Path:
    stem = Path(stem)
    files = shard_files(stem)
    if not files:
        raise SystemExit(f"no shard files for {stem}")
    rows, qids, seen, flags = [], [], set(), []
    for f in files:
        for line in f.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f"{f.name} has a damaged line ({exc}); resume that shard before merging")
            qid = record["qid"]
            if qid in seen:
                raise SystemExit(f"question {qid} answered twice")
            seen.add(qid)
            qids.append(qid)
            flags += answers.question_empty_flags(record)
            rows.append(line)
    if expected_questions is not None and len(rows) != expected_questions:
        raise SystemExit(f"{len(rows)} answers, expected {expected_questions}: resume the unfinished shards first")
    if expected_qids_sha256 is not None and runio.qid_set_sha256(qids) != expected_qids_sha256:
        raise SystemExit(f"the {len(rows)} answered questions are not the configuration's evaluation set: "
                         "resume the unfinished shards, or check the split")
    out = stem.with_name(stem.name + ".jsonl")
    if out.exists():
        raise SystemExit(f"{out} exists; remove it to merge again")
    runio.atomic_write_text(out, "".join(r + "\n" for r in rows))
    archive = stem.with_name(stem.name + ".shards")
    archive.mkdir(exist_ok=True)
    for f in files:
        f.replace(archive / f.name)
    empty_report = answers.summarize_empty(flags)
    runio.write_completion_marker(stem.parent, stem.name, len(rows), out, runio.config_path_for(stem.parent, stem.name),
                                  extra=empty_report)
    if empty_report.get("empty_answer_warning"):
        print(empty_report["empty_answer_warning"], flush=True)
    print(f"[merge] {len(rows)} answers from {len(files)} shards -> {out}; "
          f"empty_answer_rate {empty_report['empty_answer_rate']:.6f}", flush=True)
    if max_empty_rate is not None and empty_report["empty_answer_rate"] > max_empty_rate:
        raise SystemExit(f"empty answer rate {empty_report['empty_answer_rate']:.6f} exceeds --max-empty-rate "
                         f"{max_empty_rate}; the merged output and its completion marker were written")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", type=Path, help="runs/<tag>/<dataset> (without .jsonl)")
    ap.add_argument("--expected", type=int, default=None,
                    help="number of evaluation questions to check, instead of the count in <dataset>_config.json")
    ap.add_argument("--max-empty-rate", type=float, default=None,
                    help="exit non-zero when the empty-answer rate of the merged run exceeds this value")
    args = ap.parse_args(argv)
    if args.expected is not None:
        merge(args.stem, args.expected, max_empty_rate=args.max_empty_rate)
        return
    config_path = runio.config_path_for(args.stem.parent, args.stem.name)
    if not config_path.exists():
        ap.error(f"{config_path.name} not found: pass --expected N to merge without the configuration check")
    config = json.loads(config_path.read_text(encoding="utf8"))
    merge(args.stem, config.get("evaluation_questions"), config.get("evaluation_qids_sha256"),
          max_empty_rate=args.max_empty_rate)


if __name__ == "__main__":
    main()

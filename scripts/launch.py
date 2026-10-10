"""Run one ``omniaware.run`` command on several GPUs.

    python -m scripts.launch --gpus 0,1,2,3 -- --dataset okvqa_gs --split test --calib-source heldout \
        --model 8b --tag v321_main_8b --variants eab:g1.2:ans1:pos:gate
    python -m scripts.launch --gpus auto --procs-per-gpu 2 -- <run arguments>

1. ``--calibrate-only`` on the first GPU fills every cache (relevance, cross-encoder, U calibration, SeaKR
   threshold) and writes the run configuration, so shards never race on cache files;
2. one process per GPU slot answers every N-th evaluation question (``--num-shards N --shard-id i``), with
   ``CUDA_VISIBLE_DEVICES`` pinned to its GPU and its log in ``runs/<tag>/logs/``;
3. the shards are merged into ``runs/<tag>/<dataset>.jsonl`` after checking that the answered questions are exactly
   the configuration's evaluation set, and ``<dataset>.merged.json`` is written.

Interrupted runs resume: each shard skips the questions already in its file, and the merge refuses to run
until the answer count matches. launch trusts only ``<dataset>.merged.json`` (matching configuration and output,
with as many answers as ``evaluation_questions``); a partial single-process ``<dataset>.jsonl`` without that marker
is refused. With one GPU slot the command runs unchanged (no shards), and that process writes the marker itself.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def detect_gpus() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g.strip() for g in visible.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def _arg(run_args: list[str], name: str) -> str:
    return run_args[run_args.index(name) + 1]


def launch(run_args: list[str], gpus: list[str], procs_per_gpu: int = 1, python: str = sys.executable,
           runs_dir: Path | None = None) -> Path:
    from omniaware import paths, runio
    runs_dir = Path(runs_dir or paths.RUNS)
    tag, dataset = _arg(run_args, "--tag"), _arg(run_args, "--dataset")
    run_dir = runs_dir / tag
    out = run_dir / f"{dataset}.jsonl"
    base = [python, "-m", "omniaware.run", *run_args]
    slots = [g for g in gpus for _ in range(procs_per_gpu)] or [None]
    env0 = dict(os.environ, **({"CUDA_VISIBLE_DEVICES": slots[0]} if slots[0] is not None else {}))
    if len(slots) == 1:
        # One process writes the same completion marker itself when it finishes.
        subprocess.run(base, check=True, env=env0)
        return out
    marker = runio.completion_marker(run_dir, dataset).name
    try:
        if runio.verify_completion(run_dir, dataset):
            print(f"[launch] {out} is complete ({marker}); nothing to do", flush=True)
            return out
    except ValueError as exc:
        raise SystemExit(f"[launch] {exc}")
    if out.exists():
        # A single-process run that did not finish leaves a partial <dataset>.jsonl without the marker; shards
        # must not be merged on top of it.
        raise SystemExit(f"[launch] {out} has no {marker}: it is a partial single-process run (or predates the "
                         "marker). Finish it with one GPU slot, or move it aside.")
    subprocess.run(base + ["--calibrate-only"], check=True, env=env0)
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, gpu in enumerate(slots):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        log = (logs / f"{dataset}.shard{i:02d}.log").open("a")
        procs.append((i, subprocess.Popen(base + ["--num-shards", str(len(slots)), "--shard-id", str(i)],
                                          env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"[launch] shard {i}/{len(slots)} on GPU {gpu} -> {log.name}", flush=True)
    failed = []
    while procs:
        for item in list(procs):
            i, p, log = item
            if p.poll() is not None:
                log.close()
                procs.remove(item)
                if p.returncode:
                    failed.append(i)
        time.sleep(5)
    if failed:
        raise SystemExit(f"shards {failed} failed; see {logs}; rerun the same command to resume")
    config = json.loads(runio.config_path_for(run_dir, dataset).read_text(encoding="utf8"))
    from omniaware.merge_shards import merge
    return merge(run_dir / dataset, config.get("evaluation_questions"), config.get("evaluation_qids_sha256"))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m scripts.launch [--gpus 0,1|auto] [--procs-per-gpu K] -- <omniaware.run args>")
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default="auto", help="comma-separated GPU ids, or 'auto' (CUDA_VISIBLE_DEVICES / nvidia-smi)")
    ap.add_argument("--procs-per-gpu", type=int, default=1,
                    help="processes per GPU; one 8B model in bf16 needs about 20 GB")
    args = ap.parse_args(argv[:split])
    gpus = detect_gpus() if args.gpus == "auto" else [g for g in args.gpus.split(",") if g]
    launch(argv[split + 1:], gpus, args.procs_per_gpu)


if __name__ == "__main__":
    main()

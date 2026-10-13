"""Print or run the Omni-Aware v2 GPU stages S1, S2 and S4.

  python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50 --dry-run
  python scripts/v2_runs.py --stage s2 --dataset webqa_full --dry-run
  python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions runs/v2_decisions/webqa_full.json --dry-run

--dry-run prints the command lines and runs nothing. Without it the commands run in order: the omniaware.run
decodes, then scoring and the v2_table steps. The S1 image-evidence check is only printed. Commands use this
interpreter (sys.executable) and relative paths, so run the script from the repository root.
This module does not download models or data. Durations are in docs/V2_RUNS.md.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import runio  # noqa: E402

# dataset -> calibration source. data.LOADERS names; okvqa_gs is the GS-112K retrieval universe of OK-VQA.
DATASETS = {
    "webqa_full": "train",
    "webqa_open_img": "train",
    "okvqa_gs": "heldout",
}
N_CALIB = "500"
N_DEV = "256"
MODEL = "8b"
SPLIT_SEED = "2027"
DECISIONS_DIR = "runs/v2_decisions"


def format_argv(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def run_dir(tag: str) -> str:
    return f"runs/{tag}"


def decisions_path(dataset: str) -> str:
    """Where S2's selection writes the frozen decisions of one dataset. One file per dataset, so that the
    datasets cannot overwrite each other's choices."""
    return f"{DECISIONS_DIR}/{dataset}.json"


def run_command(stage: str, dataset: str, *, n_eval: int | None = None, decisions: str | None = None,
                split: str | None = None, tag: str | None = None) -> list[str]:
    if split is None:
        split = "test" if stage == "s4" else "dev"
    if tag is None:
        tag = f"v2_{stage}_{dataset}"
    cmd = [
        sys.executable, "-m", "omniaware.run",
        "--dataset", dataset,
        "--split", split,
        "--model", MODEL,
        "--tag", tag,
        "--calib-source", DATASETS[dataset],
        "--n-calib", N_CALIB,
        "--n-dev", N_DEV,
        "--split-seed", SPLIT_SEED,
        "--v2", "--v2-reranker",
    ]
    if n_eval is not None:
        cmd += ["--n-eval", str(n_eval)]
    if decisions is not None:
        cmd += ["--decisions", decisions]
    return cmd


def score_command(tag: str, dataset: str) -> list[str]:
    return [sys.executable, "-m", "omniaware.eval.score_official", f"{run_dir(tag)}/{dataset}.jsonl"]


def build_command(tag: str, dataset: str) -> list[str]:
    return [sys.executable, "scripts/analysis/v2_table.py", "build", "--run", f"{run_dir(tag)}/{dataset}.jsonl"]


def gate_command(tag: str, dataset: str) -> list[str]:
    return [sys.executable, "scripts/analysis/v2_table.py", "gate",
            "--table", f"{run_dir(tag)}/{dataset}.v2_actions.jsonl"]


def select_command(tag: str, dataset: str) -> list[str]:
    return [sys.executable, "scripts/analysis/v2_table.py", "select",
            "--table", f"{run_dir(tag)}/{dataset}.v2_actions.jsonl", "--out", decisions_path(dataset)]


def apply_command(tag: str, dataset: str, decisions: str) -> list[str]:
    return [sys.executable, "scripts/analysis/v2_table.py", "apply",
            "--table", f"{run_dir(tag)}/{dataset}.v2_actions.jsonl", "--decisions", decisions,
            "--out", f"{run_dir(tag)}/{dataset}.v2_main.json"]


def image_evidence_command(n: int) -> list[str]:
    return [sys.executable, "scripts/analysis/image_evidence_check.py", "--n", str(n)]


def check_decisions(path_str: str) -> None:
    """Fail closed: S4 needs a readable frozen decisions file. Nothing is substituted for it."""
    try:
        runio.read_frozen_decisions(path_str)
    except runio.FrozenDecisionsError as exc:
        raise SystemExit(f"S4 refuses to run: {exc}") from exc


def plan(stage: str, dataset: str, *, n: int | None, decisions: str | None) -> tuple[list[list[str]], list[list[str]]]:
    """Return (commands run in order, lines that are printed but never executed)."""
    if stage == "s1":
        smoke = 50 if n is None else n
        if smoke < 1:
            raise SystemExit("S1 --n must be a positive number of questions")
        return [run_command(stage, dataset, n_eval=smoke)], [image_evidence_command(smoke)]
    if n is not None:
        raise SystemExit("--n is only for --stage s1")
    if stage == "s2":
        calib_tag, dev_tag = f"v2_s2_{dataset}_calib", f"v2_s2_{dataset}_dev"
        # run.py writes per-question records only for --split, so calibration and dev are decoded separately.
        runs = [run_command(stage, dataset, split="calib", tag=calib_tag),
                run_command(stage, dataset, split="dev", tag=dev_tag)]
        runs += [score_command(tag, dataset) for tag in (calib_tag, dev_tag)]
        runs += [build_command(tag, dataset) for tag in (calib_tag, dev_tag)]
        runs += [gate_command(dev_tag, dataset), select_command(calib_tag, dataset)]
        return runs, []
    if stage == "s4":
        if not decisions:
            raise SystemExit("S4 refuses to run: pass --decisions PATH to the frozen decisions file. "
                             "No default hyperparameters were substituted.")
        check_decisions(decisions)
        tag = f"v2_s4_{dataset}"
        return [run_command(stage, dataset, decisions=decisions, tag=tag), score_command(tag, dataset),
                build_command(tag, dataset), apply_command(tag, dataset, decisions)], []
    raise SystemExit(f"unknown stage {stage}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("s1", "s2", "s4"))
    ap.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    ap.add_argument("--n", type=int, default=None, help="S1 smoke size (default 50)")
    ap.add_argument("--decisions", default=None,
                    help="S4 frozen decisions JSON (required for s4; S2 writes one per dataset to "
                         f"{DECISIONS_DIR}/<dataset>.json)")
    ap.add_argument("--dry-run", action="store_true", help="print the command lines and run nothing")
    args = ap.parse_args(argv)
    if args.stage != "s4" and args.decisions:
        raise SystemExit("--decisions is only for --stage s4")
    commands, printed_only = plan(args.stage, args.dataset, n=args.n, decisions=args.decisions)
    for cmd in commands + printed_only:
        print(format_argv(cmd))
    if args.dry_run:
        return
    for cmd in commands:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

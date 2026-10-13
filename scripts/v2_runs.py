"""Print or run the Omni-Aware v2 GPU stages S1, S2 and S4.

  python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50 --dry-run
  python scripts/v2_runs.py --stage s2 --dataset webqa_full --dry-run
  python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions docs/dev_decisions.json --dry-run

--dry-run prints command lines and writes nothing. Without --dry-run the
omniaware.run lines are executed. The image-evidence check is only printed.
This module does not download models or data. Durations are in docs/V2_RUNS.md.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

# run.py --split is only dev|test. Calib is selected with --n-calib / --calib-source.
# WebQA calib comes from train (>=500). OK-VQA has no train loader, so held-out 500.
DATASETS = {
    "webqa_full": "train",
    "webqa_open_img": "train",
    "okvqa": "heldout",
}
V2_ACTIONS = "direct,static_text,static_image,static_mixed,rerank_static"
N_CALIB = "500"
N_DEV = "256"
MODEL = "8b"
SPLIT_SEED = "2027"


def format_argv(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def read_frozen_decisions(path_str: str) -> None:
    """Fail closed. The file is not expanded into a hyperparameter grid."""
    path = Path(path_str)
    if not path.is_file():
        raise SystemExit(
            f"S4 refuses to run: frozen decisions file not found: {path_str}. "
            "No default hyperparameters were substituted."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"S4 refuses to run: frozen decisions file is not valid JSON: {path_str} ({exc}). "
            "No default hyperparameters were substituted."
        ) from exc
    except OSError as exc:
        raise SystemExit(
            f"S4 refuses to run: could not read frozen decisions file: {path_str} ({exc}). "
            "No default hyperparameters were substituted."
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"S4 refuses to run: frozen decisions file must be a JSON object: {path_str}. "
            "No default hyperparameters were substituted."
        )


def run_command(stage: str, dataset: str, *, n_eval: int | None = None, decisions: str | None = None) -> list[str]:
    split = "test" if stage == "s4" else "dev"
    cmd = [
        "python", "-m", "omniaware.run",
        "--dataset", dataset,
        "--split", split,
        "--model", MODEL,
        "--tag", f"v2_{stage}_{dataset}",
        "--calib-source", DATASETS[dataset],
        "--n-calib", N_CALIB,
        "--n-dev", N_DEV,
        "--split-seed", SPLIT_SEED,
    ]
    if n_eval is not None:
        cmd += ["--n-eval", str(n_eval)]
    # Protocol branch: run.py in this worktree does not define these flags yet.
    cmd += [
        "--v2-actions", V2_ACTIONS,
        "--save-signals",
        "--empty-answer-accounting",
        "--shared-prompt-registry",
    ]
    if decisions is not None:
        cmd += ["--decisions", decisions]
    return cmd


def image_evidence_command(n: int) -> list[str]:
    return ["python", "scripts/analysis/image_evidence_check.py", "--n", str(n)]


def plan(stage: str, dataset: str, *, n: int | None, decisions: str | None) -> tuple[list[list[str]], list[list[str]]]:
    """Return (omniaware.run commands, lines that are printed but never executed)."""
    if stage == "s1":
        smoke = 50 if n is None else n
        if smoke < 1:
            raise SystemExit("S1 --n must be a positive number of questions")
        runs = [run_command(stage, dataset, n_eval=smoke)]
        return runs, [image_evidence_command(smoke)]
    if n is not None:
        raise SystemExit("--n is only for --stage s1")
    if stage == "s2":
        # One dev evaluation. Calib (500) is fixed by --n-calib and --calib-source.
        # --split test is the full val and is not generated here.
        return [run_command(stage, dataset)], []
    if stage == "s4":
        if not decisions:
            raise SystemExit(
                "S4 refuses to run: pass --decisions PATH to the frozen decisions file. "
                "No default hyperparameters were substituted."
            )
        read_frozen_decisions(decisions)
        return [run_command(stage, dataset, decisions=decisions)], []
    raise SystemExit(f"unknown stage {stage}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("s1", "s2", "s4"))
    ap.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    ap.add_argument("--n", type=int, default=None, help="S1 smoke size (default 50)")
    ap.add_argument("--decisions", default=None, help="S4 frozen decisions JSON (required for s4)")
    ap.add_argument("--dry-run", action="store_true", help="print omniaware.run lines and write nothing")
    args = ap.parse_args(argv)
    if args.stage != "s4" and args.decisions:
        raise SystemExit("--decisions is only for --stage s4")
    runs, printed_only = plan(args.stage, args.dataset, n=args.n, decisions=args.decisions)
    for cmd in runs + printed_only:
        print(format_argv(cmd))
    if args.dry_run:
        return
    for cmd in runs:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

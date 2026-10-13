"""Print or run the Omni-Aware v2 GPU stages S1, S2 and S4.

  python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50 --dry-run
  python scripts/v2_runs.py --stage s2 --dataset webqa_full --dry-run
  python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions runs/v2_decisions/webqa_full.json --dry-run

--dry-run prints the command lines and runs nothing. Without it the commands run in order: the omniaware.run
decodes, then scoring and the v2_table steps. A step that exits non-zero stops the stage and its reason is printed;
in S2 a failed gate stops the stage before the selection. The S1 image-evidence check is only printed.
Commands use this interpreter (sys.executable) and the repository's run and cache directories (omniaware.paths),
so the stage does not depend on the working directory. This module does not download models or data.
Durations and the cost of the baseline arms are in docs/V2_RUNS.md.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import paths, runio  # noqa: E402

# dataset -> calibration source. data.LOADERS names; okvqa_gs is the GS-112K retrieval universe of OK-VQA.
# webqa_full (distractor setting) is the main WebQA setting; webqa_open_img is the second, open-retrieval setting.
DATASETS = {
    "webqa_full": "train",
    "webqa_open_img": "train",
    "okvqa_gs": "heldout",
}
WEBQA_DATASETS = ("webqa_full", "webqa_open_img")
N_CALIB = "500"
N_DEV = "256"
MODEL = "8b"
SPLIT_SEED = "2027"
SEAKR_SAMPLES = "20"          # sampled continuations per input for the SeaKR eigen score (upstream value)
S1_MAX_EMPTY_RATE = "0.005"   # smoke gate: more empty answers than this means the prompt or the decode is broken
GATE_EXIT_FAIL = 1            # v2_table.py exit codes: 0 PASS, 1 gate FAIL, 2 bad input
DECISIONS_DIR = paths.RUNS / "v2_decisions"


def format_argv(argv: list[str]) -> str:
    """A command line a user can paste: the Windows command-line rules on Windows, POSIX quoting elsewhere."""
    import shlex
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def run_file(tag: str, name: str) -> str:
    """A file of one run directory (runs/<tag>/<name>), in the directory omniaware.run writes to."""
    return str(paths.RUNS / tag / name)


def decisions_path(dataset: str) -> str:
    """Where S2's selection writes the frozen decisions of one dataset. One file per dataset, so that the
    datasets cannot overwrite each other's choices."""
    return str(DECISIONS_DIR / f"{dataset}.json")


def run_command(stage: str, dataset: str, *, n_eval: int | None = None, decisions: str | None = None,
                split: str | None = None, tag: str | None = None, baselines: bool = False,
                seakr: bool = True) -> list[str]:
    """One omniaware.run command. ``baselines`` adds the external baseline arms (FiTM, M2-RAG, SeaKR) that the
    main table reports next to the v2 actions; ``seakr=False`` leaves out SeaKR (its cost, see docs/V2_RUNS.md)."""
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
    if baselines:
        cmd += ["--fitm", "--m2rag"]
        if seakr:
            cmd += ["--seakr-samples", SEAKR_SAMPLES]
    if n_eval is not None:
        cmd += ["--n-eval", str(n_eval)]
    if decisions is not None:
        cmd += ["--decisions", decisions]
    return cmd


def score_command(tag: str, dataset: str) -> list[str]:
    cmd = [sys.executable, "-m", "omniaware.eval.score_official", run_file(tag, f"{dataset}.jsonl")]
    if dataset in WEBQA_DATASETS:
        cmd.append("--bart")       # normalized BARTScore and QA_approx are written only with --bart
    return cmd


def build_command(tag: str, dataset: str) -> list[str]:
    cmd = [sys.executable, str(ROOT / "scripts" / "analysis" / "v2_table.py"), "build",
           "--run", run_file(tag, f"{dataset}.jsonl")]
    if dataset in WEBQA_DATASETS:
        cmd.append("--bart")
    return cmd


def gate_command(tag: str, dataset: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / "analysis" / "v2_table.py"), "gate",
            "--table", run_file(tag, f"{dataset}.v2_actions.jsonl")]


def select_command(tag: str, dataset: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / "analysis" / "v2_table.py"), "select",
            "--table", run_file(tag, f"{dataset}.v2_actions.jsonl"), "--out", decisions_path(dataset)]


def apply_command(tag: str, dataset: str, decisions: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / "analysis" / "v2_table.py"), "apply",
            "--table", run_file(tag, f"{dataset}.v2_actions.jsonl"), "--decisions", decisions,
            "--out", run_file(tag, f"{dataset}.v2_main.json")]


def image_evidence_command(n: int) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / "analysis" / "image_evidence_check.py"), "--n", str(n)]


def check_decisions(path_str: str) -> None:
    """Fail closed: S4 needs a readable frozen decisions file. Nothing is substituted for it."""
    try:
        runio.read_frozen_decisions(path_str)
    except runio.FrozenDecisionsError as exc:
        raise SystemExit(f"S4 refuses to run: {exc}") from exc


def plan(stage: str, dataset: str, *, n: int | None, decisions: str | None,
         seakr: bool = True) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Return (commands run in order, lines that are printed but never executed, notes printed before the commands)."""
    if stage == "s1":
        smoke = 50 if n is None else n
        if smoke < 1:
            raise SystemExit("S1 --n must be a positive number of questions")
        cmd = run_command(stage, dataset, n_eval=smoke)
        cmd += ["--max-empty-rate", S1_MAX_EMPTY_RATE]
        return [cmd], [image_evidence_command(smoke)], []
    if n is not None:
        raise SystemExit("--n is only for --stage s1")
    notes = []
    if not seakr:
        notes.append("NOTE: --no-seakr: SeaKR is not run. The main table of this dataset has no SeaKR row, and "
                     "the comparison against SeaKR is missing from the paper table.")
    if stage == "s2":
        calib_tag, dev_tag = f"v2_s2_{dataset}_calib", f"v2_s2_{dataset}_dev"
        # run.py writes per-question records only for --split, so calibration and dev are decoded separately.
        # The baseline arms are decoded on the dev run only: the calibration table reads the v2 actions alone.
        runs = [run_command(stage, dataset, split="calib", tag=calib_tag),
                run_command(stage, dataset, split="dev", tag=dev_tag, baselines=True, seakr=seakr)]
        runs += [score_command(tag, dataset) for tag in (calib_tag, dev_tag)]
        runs += [build_command(tag, dataset) for tag in (calib_tag, dev_tag)]
        runs += [gate_command(dev_tag, dataset), select_command(calib_tag, dataset)]
        return runs, [], notes
    if stage == "s4":
        if not decisions:
            raise SystemExit("S4 refuses to run: pass --decisions PATH to the frozen decisions file. "
                             "No default hyperparameters were substituted.")
        check_decisions(decisions)
        tag = f"v2_s4_{dataset}"
        return [run_command(stage, dataset, decisions=decisions, tag=tag, baselines=True, seakr=seakr),
                score_command(tag, dataset), build_command(tag, dataset), apply_command(tag, dataset, decisions)], \
            [], notes
    raise SystemExit(f"unknown stage {stage}")


def execute(commands: list[list[str]]) -> None:
    """Run the commands in order. The first non-zero exit stops the stage with the failing command and its reason.
    A failed gate (exit 1) is reported as a go/no-go decision, and the selection after it is never run."""
    for cmd in commands:
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            continue
        if len(cmd) > 2 and cmd[2] == "gate" and rc == GATE_EXIT_FAIL:
            raise SystemExit("gate FAIL (go/no-go): the development oracle does not beat the strongest single action by "
                             "the required margin, so the calibration selection and S4 are not run. Command: "
                             + format_argv(cmd))
        raise SystemExit(f"step failed with exit code {rc}; the remaining steps were not run. Command: "
                         + format_argv(cmd))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("s1", "s2", "s4"))
    ap.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    ap.add_argument("--n", type=int, default=None, help="S1 smoke size (default 50)")
    ap.add_argument("--decisions", default=None,
                    help="S4 frozen decisions JSON (required for s4; S2 writes one per dataset to "
                         f"{DECISIONS_DIR}/<dataset>.json)")
    ap.add_argument("--no-seakr", action="store_true",
                    help="S2 dev and S4: leave out the SeaKR baseline (its cost is about 20 extra sampled decodes per "
                         "question, see docs/V2_RUNS.md). The main table then has no SeaKR row.")
    ap.add_argument("--dry-run", action="store_true", help="print the command lines and run nothing")
    args = ap.parse_args(argv)
    if args.stage != "s4" and args.decisions:
        raise SystemExit("--decisions is only for --stage s4")
    if args.stage == "s1" and args.no_seakr:
        raise SystemExit("--no-seakr applies to S2 and S4 only; S1 runs no baseline arms")
    commands, printed_only, notes = plan(args.stage, args.dataset, n=args.n, decisions=args.decisions,
                                         seakr=not args.no_seakr)
    for note in notes:
        print(note)
    for cmd in commands + printed_only:
        print(format_argv(cmd))
    if args.dry_run:
        return
    execute(commands)


if __name__ == "__main__":
    main()

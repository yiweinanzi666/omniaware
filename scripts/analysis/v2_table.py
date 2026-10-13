"""Per-question action tables for Omni-Aware v2: build, gate, select, apply.

    python scripts/analysis/v2_table.py build  --run runs/<tag>/<dataset>.jsonl [--bart] [--device cuda]
    python scripts/analysis/v2_table.py gate   --table runs/<tag>/<dataset>.v2_actions.jsonl
    python scripts/analysis/v2_table.py select --table runs/<tag>/<dataset>.v2_actions.jsonl \
        --out runs/v2_decisions/<dataset>.json [--delta 0.1 --epsilon 1.0 --seed 0]
    python scripts/analysis/v2_table.py apply  --table runs/<tag>/<dataset>.v2_actions.jsonl \
        --decisions runs/v2_decisions/<dataset>.json --out runs/<tag>/<dataset>.v2_main.json

build scores every available action of a run written by omniaware.run --v2 (rec["actions"]) with the official
evaluator, and writes <dataset>.v2_actions.jsonl and <dataset>.v2_build_summary.json next to the run. Each table
row is {"qid", "subset", "actions", "scores", "metric"}; scores are on a 0-100 scale.

An unavailable action (available=False) has no entry in its question's "scores", and no score is borrowed
from another action. arbitration needs every scored action to be scored on every calibration row, so select
uses only the actions scored on every calibration question and lists the others in the decisions file. gate
and apply use the same rule for the strongest single action. The oracle of a question is the best of its
available actions.

gate prints PASS or FAIL: the relative gap of the oracle over the strongest single action, against +8%.
select freezes the v2-Rule and v2-Cal decisions on a calibration table (dev_select.select_v2). apply reads a
frozen decisions file and a test table. It never searches, and it chooses from the actions alone.

Exit codes: 0 success (gate PASS), 1 gate FAIL, 2 bad input or inconsistent files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from omniaware import arbitration, runio  # noqa: E402
from omniaware.actions import DIRECT, V2_ACTIONS  # noqa: E402
from omniaware.eval import score_official  # noqa: E402
from scripts.analysis import dev_select  # noqa: E402

GATE_THRESHOLD = 0.08
FAMILY_METRIC = {"webqa": "acc_approx", "okvqa": "vqa_accuracy"}
REFERENCE = "static_mixed"
POLICY_FAMILIES = ("rule", "cal")


class TableError(Exception):
    """A table, run or decisions file is missing or inconsistent. Nothing is written."""


# ---------------------------------------------------------------- reading tables

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise TableError(f"file not found: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TableError(f"{path}: line {number} is not JSON ({exc})") from exc
    if not rows:
        raise TableError(f"{path}: no rows")
    return rows


def _finite_score(value, qid: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TableError(f"question {qid}: score for {name} must be a finite number, not {value!r}")
    return float(value)


def load_table(path: Path) -> tuple[list[dict], str]:
    """Rows of an action table and their metric name. Scores are checked; actions are checked by require_actions."""
    rows = read_jsonl(path)
    seen, metrics = set(), set()
    for row in rows:
        qid = row.get("qid")
        if not isinstance(qid, str) or not qid:
            raise TableError(f"{path}: a row has no string qid")
        if qid in seen:
            raise TableError(f"{path}: duplicate qid {qid!r}")
        seen.add(qid)
        scores = row.get("scores")
        if not isinstance(scores, dict) or not scores:
            raise TableError(f"{path}: question {qid!r} needs a non-empty scores object")
        unknown = set(scores) - set(V2_ACTIONS)
        if unknown:
            raise TableError(f"{path}: question {qid!r} has unknown actions {sorted(unknown)}")
        row["scores"] = {name: _finite_score(value, qid, name) for name, value in scores.items()}
        metrics.add(row.get("metric"))
    if len(metrics) != 1 or None in metrics:
        raise TableError(f"{path}: every row must name the same metric, found {sorted(map(str, metrics))}")
    return rows, metrics.pop()


def require_actions(rows: list[dict], path: Path) -> None:
    for row in rows:
        actions = row.get("actions")
        if not isinstance(actions, dict) or DIRECT not in actions:
            raise TableError(f"{path}: question {row['qid']!r} has no actions object with direct")


def full_actions(rows: list[dict]) -> list[str]:
    """Actions scored on every question, in V2_ACTIONS order. Only these can be a single action or be chosen."""
    return [name for name in V2_ACTIONS if all(name in row["scores"] for row in rows)]


def partial_actions(rows: list[dict]) -> list[str]:
    """Actions scored on some questions but not all: excluded from selection and from the single-action baseline."""
    full = set(full_actions(rows))
    seen = {name for row in rows for name in row["scores"]}
    return [name for name in V2_ACTIONS if name in seen and name not in full]


def _mean(values) -> float | None:
    values = [float(v) for v in values]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------- build

def _action_available(record: dict, name: str) -> bool:
    action = record["actions"].get(name)
    return action is not None and bool(action.get("available", True))


def _score_action(family: str, name: str, answers: dict[str, str], by_qid: dict[str, dict], bart: bool,
                  device: str | None) -> tuple[dict[str, dict], dict]:
    """Official cells of one action on the questions it answered: {qid: {metric: value}} and extra fields."""
    qids = list(answers)
    if family == "okvqa":
        scored, _, _ = score_official.score_okvqa({name: answers}, qids)
        return scored[name], {}
    scored, _, extra = score_official.score_webqa({name: answers}, qids, by_qid, bart=bart, device=device)
    return scored[name], extra or {}


def build(run: Path, bart: bool = False, device: str | None = None) -> dict:
    """Write the action table and its summary next to the run. Returns the summary."""
    records = read_jsonl(run)
    missing = sum("actions" not in r for r in records)
    if missing:
        raise TableError(f"{run}: {missing} records have no rec['actions']; the run needs --v2 (omniaware.run --v2)")
    by_qid = {}
    for record in records:
        if record["qid"] in by_qid:
            raise TableError(f"{run}: duplicate qid {record['qid']!r}")
        by_qid[record["qid"]] = record
        unknown = set(record["actions"]) - set(V2_ACTIONS)
        if unknown or DIRECT not in record["actions"]:
            raise TableError(f"{run}: question {record['qid']!r} needs direct and only known actions, "
                             f"found {sorted(record['actions'])}")
    families = {r.get("dataset") for r in records}
    if len(families) != 1 or next(iter(families)) not in FAMILY_METRIC:
        raise TableError(f"{run}: the run must hold one dataset family, webqa or okvqa, found {sorted(map(str, families))}")
    family = families.pop()
    metric = FAMILY_METRIC[family]

    names = [name for name in V2_ACTIONS if any(name in r["actions"] for r in records)]
    cells, extras = {}, {}
    for name in names:
        answers = {r["qid"]: r["actions"][name]["answer"] for r in records if _action_available(r, name)}
        if answers:
            cells[name], extras[name] = _score_action(family, name, answers, by_qid, bart, device)
    bart_errors = {name: extra["bart_error"] for name, extra in extras.items() if "bart_error" in extra}

    table = []
    for record in records:
        qid = record["qid"]
        scores = {name: float(cells[name][qid][metric]) for name in names if name in cells and qid in cells[name]}
        subset = score_official.webqa_subset_name(record) if family == "webqa" else None
        table.append({"qid": qid, "subset": subset, "actions": record["actions"], "scores": scores,
                      "metric": metric})

    full = full_actions(table)
    summary = {
        "dataset": family, "metric": metric, "bart": bool(bart), "questions": len(table),
        "input": str(run), "input_sha256": runio.sha256_file(run),
        "actions": {name: _action_summary(table, name, cells.get(name), bart, family) for name in names},
        "scored_on_every_question": full,
        "best_single": arbitration.best_single_action({n: [r["scores"][n] for r in table] for n in full}),
        "oracle_mean": _mean(max(r["scores"].values()) for r in table),
        "partially_available": partial_actions(table),
    }
    summary["best_single_mean"] = summary["actions"][summary["best_single"]]["mean"]
    if bart_errors:
        summary["bart_errors"] = bart_errors

    table_path = run.with_name(f"{run.stem}.v2_actions.jsonl")
    runio.atomic_write_text(table_path, "".join(runio.dumps_strict(r, ensure_ascii=False) + "\n" for r in table))
    summary["table"] = {"path": str(table_path), "sha256": runio.sha256_file(table_path)}
    runio.atomic_write_text(run.with_name(f"{run.stem}.v2_build_summary.json"),
                            runio.dumps_strict(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def _action_summary(table: list[dict], name: str, cells: dict | None, bart: bool, family: str) -> dict:
    available = [r for r in table if name in r["scores"]]
    empty = [bool(r["actions"][name].get("answer_empty")) for r in available]
    out = {"questions": len(available), "mean": _mean(r["scores"][name] for r in available),
           "empty_rate": (sum(empty) / len(empty)) if empty else None}
    if bart and family == "webqa" and cells is not None:
        for field in ("normalized_bart", "QA_approx"):
            out[f"{field}_mean"] = _mean(cells[r["qid"]][field] for r in available
                                         if cells[r["qid"]].get(field) is not None)
    return out


# ---------------------------------------------------------------- gate

def relative_gap(oracle_mean: float, best_mean: float) -> float:
    """Relative gap when the baseline is positive; the plain difference otherwise (a zero or negative baseline)."""
    if best_mean > 0:
        return (oracle_mean - best_mean) / best_mean
    return oracle_mean - best_mean


def gate(table: Path) -> tuple[bool, dict]:
    rows, metric = load_table(table)
    full = full_actions(rows)
    if not full:
        raise TableError(f"{table}: no action is scored on every question, so there is no strongest single action")
    means = {name: _mean(r["scores"][name] for r in rows if name in r["scores"]) for name in V2_ACTIONS
             if any(name in r["scores"] for r in rows)}
    best = arbitration.best_single_action({n: [r["scores"][n] for r in rows] for n in full})
    oracle = _mean(max(r["scores"].values()) for r in rows)
    gap = relative_gap(oracle, means[best])
    summary = {"metric": metric, "questions": len(rows), "oracle_mean": oracle, "best_single": best,
               "best_single_mean": means[best], "relative_gap": gap, "threshold": GATE_THRESHOLD,
               "action_means": means, "partially_available": partial_actions(rows)}
    return gap >= GATE_THRESHOLD, summary


def format_gate(passed: bool, summary: dict) -> list[str]:
    lines = [
        "PASS" if passed else "FAIL",
        f"questions={summary['questions']} metric={summary['metric']} oracle_mean={summary['oracle_mean']:.4f} "
        f"best_single={summary['best_single']} best_single_mean={summary['best_single_mean']:.4f} "
        f"relative_gap={summary['relative_gap']:.4f} threshold={summary['threshold']}",
        "action means: " + ", ".join(f"{k}={v:.4f}" for k, v in summary["action_means"].items()),
    ]
    if summary["partially_available"]:
        lines.append("not scored on every question (excluded from the single action): "
                     + ", ".join(summary["partially_available"]))
    return lines


# ---------------------------------------------------------------- select

def select(table: Path, out: Path, delta: float, epsilon: float, seed: int) -> dict:
    rows, metric = load_table(table)
    require_actions(rows, table)
    full = full_actions(rows)
    if DIRECT not in full:
        raise TableError(f"{table}: direct must be scored on every calibration question")
    selector_rows = [{"qid": r["qid"], "actions": r["actions"], "scores": r["scores"]} for r in rows]
    result = dev_select.select_v2(selector_rows, delta=delta, epsilon=epsilon, seed=seed)
    decisions = {
        **result,
        "calibration": {"table": str(table), "table_sha256": runio.sha256_file(table), "questions": len(rows),
                        "metric": metric, "delta": delta, "epsilon": epsilon, "seed": seed,
                        "scored_on_every_question": full, "partially_available": partial_actions(rows)},
    }
    runio.atomic_write_text(out, runio.dumps_strict(decisions, indent=2) + "\n")
    return decisions


# ---------------------------------------------------------------- apply

def decision_fields(decision: dict) -> dict:
    """The fields that define a frozen decision, for the report. Cal models list their kept and dropped features."""
    keys = ("kind", "fallback", "action", "best_single", "confidence", "delta", "epsilon", "n_calib",
            "n_accepted", "calib_utility", "evidence_choice", "tau", "lam", "seed", "n_fit", "n_risk", "n_params")
    out = {key: decision[key] for key in keys if key in decision}
    if decision.get("models"):
        out["models"] = {name: {"features": m["features"], "dropped": m["dropped"], "n_params": m["n_params"]}
                         for name, m in decision["models"].items()}
    return out


def _chosen_scores(decision: dict, rows: list[dict]) -> tuple[list[str], list[float]]:
    """The action each question receives, and its score. The choice reads only the actions; the score is for
    reporting after the choice has been made."""
    chosen = [arbitration.apply_decision(decision, row["actions"])["action"] for row in rows]
    scores = []
    for row, name in zip(rows, chosen):
        if name not in row["scores"]:
            raise TableError(f"question {row['qid']!r}: {name} was chosen but has no score; table and decision disagree")
        scores.append(float(row["scores"][name]))
    return chosen, scores


def _block(rows: list[dict], frozen: dict, best: str, full: list[str]) -> dict:
    """Statistics of the single actions, the oracle and both frozen policies on these questions. The reference
    is the frozen best single action; it is not searched again on this table."""
    best_scores = np.asarray([r["scores"][best] for r in rows], dtype=float)
    oracle = np.asarray([max(r["scores"].values()) for r in rows], dtype=float)
    static = np.asarray([r["scores"][REFERENCE] for r in rows], dtype=float) if REFERENCE in full else None
    out = {
        "questions": len(rows), "best_single_mean": float(best_scores.mean()), "oracle_mean": float(oracle.mean()),
        "singles": {name: {"questions": sum(name in r["scores"] for r in rows),
                           "mean": _mean(r["scores"][name] for r in rows if name in r["scores"])}
                    for name in V2_ACTIONS if any(name in r["scores"] for r in rows)},
        "policies": {},
    }
    gap_oracle = float(oracle.mean() - best_scores.mean())
    for family in POLICY_FAMILIES:
        chosen, policy = _chosen_scores(frozen[family], rows)
        policy = np.asarray(policy, dtype=float)
        gap_policy = float(policy.mean() - best_scores.mean())
        out["policies"][family] = {
            "decision": decision_fields(frozen[family]),
            "mean": float(policy.mean()),
            "vs_best_single": score_official.paired(best_scores, policy),
            "vs_static_mixed": score_official.paired(static, policy) if static is not None else None,
            "capture_ratio": (gap_policy / gap_oracle) if gap_oracle > 0 else None,
            "worse_than_best_single_fraction": float(np.mean(policy < best_scores)),
            "chosen": dict(Counter(chosen)),
        }
    return out


def apply(table: Path, decisions_path: Path, out: Path) -> dict:
    decisions_sha = runio.read_frozen_decisions(decisions_path)
    frozen = dev_select.frozen_v2(decisions_path)
    missing = [family for family in POLICY_FAMILIES if family not in frozen]
    if missing:
        raise TableError(f"{decisions_path}: the frozen decisions lack {missing}")
    rows, metric = load_table(table)
    require_actions(rows, table)
    best = frozen["rule"]["best_single"]
    if frozen["cal"]["best_single"] != best:
        raise TableError(f"{decisions_path}: the rule and cal decisions name different best single actions")
    full = full_actions(rows)
    if best not in full:
        raise TableError(f"{table}: the frozen best single action {best} is not scored on every test question")
    result = {
        "metric": metric, "best_single": best, "frozen_primary": frozen["primary"],
        "table": {"path": str(table), "sha256": runio.sha256_file(table), "questions": len(rows)},
        "decisions": {"path": str(decisions_path), "sha256": decisions_sha},
        "scored_on_every_question": full, "partially_available": partial_actions(rows),
        **_block(rows, frozen, best, full),
    }
    subsets = sorted({r["subset"] for r in rows if r.get("subset")})
    if subsets:
        result["subsets"] = {s: _block([r for r in rows if r.get("subset") == s], frozen, best, full)
                             for s in subsets}
    choices = [{"qid": r["qid"], "subset": r.get("subset")} for r in rows]
    for family in POLICY_FAMILIES:
        chosen, scores = _chosen_scores(frozen[family], rows)
        for choice, name, score in zip(choices, chosen, scores):
            choice[family] = {"action": name, "score": score}
    runio.atomic_write_text(out, runio.dumps_strict(result, ensure_ascii=False, indent=2) + "\n")
    runio.atomic_write_text(out.with_name(f"{out.stem}.choices.jsonl"),
                            "".join(runio.dumps_strict(c, ensure_ascii=False) + "\n" for c in choices))
    return result


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="score each action of a --v2 run and write its action table")
    p.add_argument("--run", type=Path, required=True, help="<run_dir>/<dataset>.jsonl written by omniaware.run --v2")
    p.add_argument("--bart", action="store_true", help="WebQA: also record the normalized_bart and QA_approx means")
    p.add_argument("--device", default=None, help="device of the WebQA BARTScore model")

    p = sub.add_parser("gate", help="PASS when the oracle beats the strongest single action by at least 8%")
    p.add_argument("--table", type=Path, required=True)

    p = sub.add_parser("select", help="freeze v2-Rule and v2-Cal on a calibration table")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="decisions JSON, e.g. runs/v2_decisions/<dataset>.json")
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=1.0, help="tolerance in score points (0-100 scale)")
    p.add_argument("--seed", type=int, default=0, help="split of the calibration rows for v2-Cal")

    p = sub.add_parser("apply", help="apply frozen decisions to a test table; searches nothing")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="main results JSON; choices go next to it")

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            summary = build(args.run, bart=args.bart, device=args.device)
            print(json.dumps({"best_single": summary["best_single"], "oracle_mean": summary["oracle_mean"],
                              "actions": {k: v["mean"] for k, v in summary["actions"].items()},
                              "table": summary["table"]["path"]}, indent=2))
            return 0
        if args.command == "gate":
            passed, summary = gate(args.table)
            print("\n".join(format_gate(passed, summary)))
            return 0 if passed else 1
        if args.command == "select":
            decisions = select(args.table, args.out, args.delta, args.epsilon, args.seed)
            print(json.dumps({k: v["kind"] + (" (fallback)" if v["fallback"] else "")
                              for k, v in decisions["v2"].items() if isinstance(v, dict)}, indent=2))
            return 0
        result = apply(args.table, args.decisions, args.out)
        print(json.dumps({"best_single": result["best_single"], "rule": result["policies"]["rule"]["mean"],
                          "cal": result["policies"]["cal"]["mean"]}, indent=2))
        return 0
    except (TableError, ValueError, runio.FrozenDecisionsError, dev_select.DevDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

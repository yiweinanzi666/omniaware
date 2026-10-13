"""Per-question action tables for Omni-Aware v2: build, gate, select, apply.

    python scripts/analysis/v2_table.py build  --run runs/<tag>/<dataset>.jsonl [--bart] [--device cuda]
    python scripts/analysis/v2_table.py gate   --table runs/<tag>/<dataset>.v2_actions.jsonl
    python scripts/analysis/v2_table.py select --table runs/<tag>/<dataset>.v2_actions.jsonl \
        --out runs/v2_decisions/<dataset>.json [--delta 0.1 --epsilon 1.0 --seed 0]
    python scripts/analysis/v2_table.py apply  --table runs/<tag>/<dataset>.v2_actions.jsonl \
        --decisions runs/v2_decisions/<dataset>.json --out runs/<tag>/<dataset>.v2_main.json
    python scripts/analysis/v2_table.py devcheck --table runs/<dev_tag>/<dataset>.v2_actions.jsonl \
        --decisions runs/v2_decisions/<dataset>.json --out runs/v2_decisions/<dataset>.dev_check.json

build checks the run the way score_official does: the scored questions must be the configured evaluation set
(<dataset>_config.json), and WebQA gold answers must match the dataset. An incomplete or foreign run is refused.
It scores every available action of a run written by omniaware.run --v2 (rec["actions"]) with the official
evaluator, and writes <dataset>.v2_actions.jsonl and <dataset>.v2_build_summary.json next to the run. Each table
row is {"qid", "subset", "actions", "scores", "metric", "baseline_signals", "baseline_scores"}; scores are on a
0-100 scale. baseline_signals holds the U gate (u_gate) and the pre-retrieval normalised entropy (h_sem_pre) of
the run. baseline_scores holds the primary metric of each baseline arm in <dataset>.official_scores.jsonl, which
must exist (run score_official on the run first).

An unavailable action (available=False) has no entry in its question's "scores", and no score is borrowed
from another action. Only an action scored on every question can be the strongest single action a*. An action
scored on some questions only can be chosen on the questions where it is available, but never as a* and never as
the reference action of a question. The oracle of a question is the best of its available actions.

gate prints PASS or FAIL: the relative gap of the oracle over the strongest single action, against +8%. On FAIL
the output says not to continue with select or S4. select freezes the v2-Rule and v2-Cal decisions on a
calibration table (dev_select.select_v2), tunes the two single-threshold adaptive baselines (U gate and NTE) on the
same table without risk control, and records the calibration means of every candidate for the strongest baseline.
apply reads a frozen decisions file and a test table. It never searches: the choices read the actions and the
baseline signals only, and scores are read after the choice, for reporting.
devcheck is the check of docs/PROTOCOL.md section 5 before the test: it applies the frozen decisions to the
development table and prints PASS when the primary selector beats the strongest single action a* (the
calibration's) by at least +5% relative, with a paired bootstrap 95% CI whose lower bound is above 0. It writes the
applied result next to the table (<dataset>.v2_dev_main.json) and the check, with the sha256 of the decisions file,
to --out; scripts/v2_runs.py --stage s4 reads that file.

Exit codes: 0 success (gate or devcheck PASS), 1 gate or devcheck FAIL, 2 bad input or inconsistent files.
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
DEV_CHECK_THRESHOLD = 0.05      # protocol section 5: selector over a* on the development set, relative
FAMILY_METRIC = {"webqa": "acc_approx", "okvqa": "vqa_accuracy"}
REFERENCE = "static_mixed"
POLICY_FAMILIES = ("rule", "cal")
BASELINE_ARMS = ("static_joint", "gated:static", "nte_trigger:static", "nte_shared:static", "seakr",
                 "seakr_trigger:static", "fitm", "m2rag", "universalrag")
ADAPTIVE_SIGNAL = {"u_gate": "u_gate", "nte": "h_sem_pre"}     # baseline name -> baseline_signals field
THRESHOLD_STEPS = 20                                            # threshold grids are 0%, 5%, ..., 100% quantiles


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


def _finite(value, qid, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TableError(f"question {qid}: {name} must be a finite number, not {value!r}")
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
        row["scores"] = {name: _finite(value, qid, f"score for {name}") for name, value in scores.items()}
        metrics.add(row.get("metric"))
    if len(metrics) != 1 or None in metrics:
        raise TableError(f"{path}: every row must name the same metric, found {sorted(map(str, metrics))}")
    return rows, metrics.pop()


def require_actions(rows: list[dict], path: Path) -> None:
    for row in rows:
        actions = row.get("actions")
        if not isinstance(actions, dict) or DIRECT not in actions:
            raise TableError(f"{path}: question {row['qid']!r} has no actions object with direct")
        unknown = set(actions) - set(V2_ACTIONS)
        if unknown:
            raise TableError(f"{path}: question {row['qid']!r} has unknown actions {sorted(unknown)}")


def require_baselines(rows: list[dict], path: Path) -> list[str]:
    """Baseline signals and baseline arm scores of every row. Returns the baseline arms, the same on every row."""
    arms = None
    for row in rows:
        qid = row["qid"]
        signals = row.get("baseline_signals")
        if not isinstance(signals, dict) or any(field not in signals for field in ("u_gate", "h_sem_pre")):
            raise TableError(f"{path}: question {qid!r} has no baseline_signals; rebuild the table with build")
        row["baseline_signals"] = {field: _finite(signals[field], qid, field) for field in ("u_gate", "h_sem_pre")}
        scores = row.get("baseline_scores")
        if not isinstance(scores, dict):
            raise TableError(f"{path}: question {qid!r} has no baseline_scores; rebuild the table with build")
        if arms is None:
            arms = sorted(scores)
        elif sorted(scores) != arms:
            raise TableError(f"{path}: question {qid!r} has baseline arms {sorted(scores)}, other questions have {arms}")
        row["baseline_scores"] = {arm: _finite(value, qid, f"baseline score for {arm}") for arm, value in scores.items()}
    return arms or []


def full_actions(rows: list[dict]) -> list[str]:
    """Actions scored on every question, in V2_ACTIONS order. Only these can be a single action or be chosen."""
    return [name for name in V2_ACTIONS if all(name in row["scores"] for row in rows)]


def partial_actions(rows: list[dict]) -> list[str]:
    """Actions scored on some questions but not all: never a single action, never a reference."""
    full = set(full_actions(rows))
    seen = {name for row in rows for name in row["scores"]}
    return [name for name in V2_ACTIONS if name in seen and name not in full]


def _mean(values) -> float | None:
    values = [float(v) for v in values]
    return float(np.mean(values)) if values else None


def _action_available(actions: dict, name: str) -> bool:
    action = actions.get(name)
    return action is not None and bool(action.get("available", True))


def _score_of(row: dict, name: str) -> float:
    if name not in row["scores"]:
        raise TableError(f"question {row['qid']!r}: {name} has no score; table and decision disagree")
    return float(row["scores"][name])


# ---------------------------------------------------------------- build

def _score_action(family: str, name: str, answers: dict[str, str], by_qid: dict[str, dict], bart: bool,
                  device: str | None) -> tuple[dict[str, dict], dict]:
    """Official cells of one action on the questions it answered: {qid: {metric: value}} and extra fields."""
    qids = list(answers)
    if family == "okvqa":
        scored, _, _ = score_official.score_okvqa({name: answers}, qids)
        return scored[name], {}
    scored, _, extra = score_official.score_webqa({name: answers}, qids, by_qid, bart=bart, device=device)
    return scored[name], extra or {}


def _check_run_questions(run: Path, by_qid: dict[str, dict], family: str) -> dict:
    """The checks score_official makes before it scores a run: the question set against the configuration, and for
    WebQA the gold answers against the dataset. OK-VQA gold answers are not checked by score_official either."""
    qids = list(by_qid)
    check = score_official.check_question_set(run, qids)
    if family == "webqa":
        score_official.check_golds(by_qid, qids, score_official.webqa_references())
    return check


def _official_baseline_scores(run: Path, qids: list[str], metric: str) -> tuple[dict[str, dict[str, float]], dict]:
    """Primary-metric cells of the baseline arms, {arm: {qid: value}}, from the run's official scores. The official
    summary must describe this predictions file and the full run, not a subset."""
    scores_path = run.with_name(f"{run.stem}.official_scores.jsonl")
    summary_path = run.with_name(f"{run.stem}.official_summary.json")
    missing = [str(p) for p in (scores_path, summary_path) if not p.is_file()]
    if missing:
        raise TableError(f"{run}: no official scores ({', '.join(missing)}); run score_official on this run first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("predictions_sha256") != runio.sha256_file(run):
        raise TableError(f"{summary_path}: scored a different predictions file; run score_official again")
    if summary.get("subset"):
        raise TableError(f"{summary_path}: scored on a subset of the run; the baseline arms need the full run")
    rows = read_jsonl(scores_path)
    seen = [row.get("qid") for row in rows]
    if len(set(seen)) != len(seen) or set(seen) != set(qids):
        raise TableError(f"{scores_path}: the official scores cover {len(seen)} questions that differ from the "
                         f"{len(qids)} questions of the run; run score_official again")
    cells: dict[str, dict[str, float]] = {}
    for row in rows:
        for arm in BASELINE_ARMS:
            cell = (row.get("scores") or {}).get(arm)
            if cell is None:
                continue
            if metric not in cell:
                raise TableError(f"{scores_path}: arm {arm!r} of question {row['qid']!r} has no {metric}")
            cells.setdefault(arm, {})[row["qid"]] = _finite(cell[metric], row["qid"], f"{arm} {metric}")
    for arm, values in cells.items():
        if set(values) != set(qids):
            raise TableError(f"{scores_path}: arm {arm!r} is missing for {len(qids) - len(values)} questions")
    return cells, {"path": str(scores_path), "sha256": runio.sha256_file(scores_path),
                   "summary": str(summary_path), "summary_sha256": runio.sha256_file(summary_path)}


def _run_signals(record: dict) -> dict[str, float]:
    qid = record["qid"]
    raw_pre = record.get("raw_pre") or {}
    return {"u_gate": _finite(record.get("u_gate"), qid, "u_gate"),
            "h_sem_pre": _finite(raw_pre.get("h_sem"), qid, "raw_pre.h_sem")}


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
    question_check = _check_run_questions(run, by_qid, family)
    baseline_cells, baseline_source = _official_baseline_scores(run, list(by_qid), metric)
    arms = [arm for arm in BASELINE_ARMS if arm in baseline_cells]

    names = [name for name in V2_ACTIONS if any(name in r["actions"] for r in records)]
    cells, extras = {}, {}
    for name in names:
        answers = {r["qid"]: r["actions"][name]["answer"] for r in records
                   if _action_available(r["actions"], name)}
        if answers:
            cells[name], extras[name] = _score_action(family, name, answers, by_qid, bart, device)
    bart_errors = {name: extra["bart_error"] for name, extra in extras.items() if "bart_error" in extra}

    table = []
    for record in records:
        qid = record["qid"]
        scores = {name: float(cells[name][qid][metric]) for name in names if name in cells and qid in cells[name]}
        subset = score_official.webqa_subset_name(record) if family == "webqa" else None
        table.append({"qid": qid, "subset": subset, "actions": record["actions"], "scores": scores,
                      "metric": metric, "baseline_signals": _run_signals(record),
                      "baseline_scores": {arm: baseline_cells[arm][qid] for arm in arms}})

    full = full_actions(table)
    if not full:
        raise TableError(f"{run}: no action is scored on every question, so there is no strongest single action")
    summary = {
        "dataset": family, "metric": metric, "bart": bool(bart), "questions": len(table),
        "input": str(run), "input_sha256": runio.sha256_file(run), "question_check": question_check,
        "actions": {name: _action_summary(table, name, cells.get(name), bart, family) for name in names},
        "scored_on_every_question": full,
        "best_single": arbitration.best_single_action({n: [r["scores"][n] for r in table] for n in full}),
        "oracle_mean": _mean(max(r["scores"].values()) for r in table),
        "partially_available": partial_actions(table),
        "baseline_arms": arms, "baseline_source": baseline_source,
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
        lines.append("not scored on every question (never a single action or a reference): "
                     + ", ".join(summary["partially_available"]))
    if not passed:
        lines.append("不要继续 select / S4：oracle 相对最强单一动作的差距未达到门槛")
    return lines


# ---------------------------------------------------------------- adaptive baselines

def _threshold_grid(values) -> list[float]:
    """-inf, the 0%, 5%, ..., 100% quantiles of the signal on the calibration rows, and +inf; ascending, no duplicates."""
    arr = np.asarray(values, dtype=float)
    quantiles = [float(q) for q in np.quantile(arr, np.linspace(0.0, 1.0, THRESHOLD_STEPS + 1))]
    return sorted(set([-math.inf, *quantiles, math.inf]))


def _gate_choice(value: float, tau: float, actions: dict, reference: str) -> str:
    """static_mixed when the signal is above tau, direct otherwise. A gated action that is unavailable on the question
    is replaced by the reference action of the question."""
    name = REFERENCE if value > tau else DIRECT
    return name if _action_available(actions, name) else reference


def calibrate_baselines(rows: list[dict], arms: list[str]) -> dict:
    """Tune the U gate and the NTE gate on the calibration rows. Each has one threshold on one signal, chosen from
    the threshold grid by the highest calibration mean. Ties go to the end of the grid that always picks the stronger
    of direct and static_mixed. These thresholds are tuned for the baseline and are not risk controlled."""
    full = full_actions(rows)
    if REFERENCE not in full:
        raise TableError(f"the adaptive baselines need {REFERENCE} scored on every calibration question")
    means = {name: _mean(r["scores"][name] for r in rows) for name in full}
    stronger = REFERENCE if means[REFERENCE] > means[DIRECT] else DIRECT
    out = {
        "risk_controlled": False,
        "note": "single-threshold baselines tuned on the calibration table; no risk control, no test data",
        "stronger_single": stronger,
        "grid": "-inf, the 0%, 5%, ..., 100% quantiles of the signal on the calibration rows, +inf",
    }
    adaptive = {}
    for name, field in ADAPTIVE_SIGNAL.items():
        grid = _threshold_grid([r["baseline_signals"][field] for r in rows])
        order = list(reversed(grid)) if stronger == DIRECT else grid
        best = None
        for tau in order:
            mean = _mean(_score_of(r, _gate_choice(r["baseline_signals"][field], tau, r["actions"], DIRECT))
                         for r in rows)
            if best is None or mean > best[1]:
                best = (tau, mean)
        out[name] = {"signal": field, "tau": arbitration.encode_json_float(best[0]), "above": REFERENCE,
                     "otherwise": DIRECT, "calibration_mean": best[1], "candidates": len(grid)}
        adaptive[name] = best[1]
    out["calibration_means"] = {
        "singles": means,
        "adaptive": adaptive,
        "arms": {arm: _mean(r["baseline_scores"][arm] for r in rows) for arm in arms},
    }
    return out


# ---------------------------------------------------------------- select

def select(table: Path, out: Path, delta: float, epsilon: float, seed: int,
           confidence: str = arbitration.PRIMARY_CONFIDENCE) -> dict:
    rows, metric = load_table(table)
    require_actions(rows, table)
    arms = require_baselines(rows, table)
    full = full_actions(rows)
    if DIRECT not in full:
        raise TableError(f"{table}: direct must be scored on every calibration question")
    selector_rows = [{"qid": r["qid"], "actions": r["actions"], "scores": r["scores"]} for r in rows]
    result = dev_select.select_v2(selector_rows, delta=delta, epsilon=epsilon, seed=seed, confidence=confidence)
    decisions = {
        **result,
        "baselines": calibrate_baselines(rows, arms),
        "calibration": {"table": str(table), "table_sha256": runio.sha256_file(table), "questions": len(rows),
                        "metric": metric, "delta": delta, "epsilon": epsilon, "seed": seed, "confidence": confidence,
                        "scored_on_every_question": full, "partially_available": partial_actions(rows)},
    }
    runio.atomic_write_text(out, runio.dumps_strict(decisions, indent=2) + "\n")
    return decisions


# ---------------------------------------------------------------- apply

def _reference(rule: dict, row: dict) -> str:
    try:
        return arbitration.reference_action(rule, row["actions"])
    except ValueError as exc:
        raise TableError(f"question {row['qid']!r}: {exc}") from exc


def _choose(decision: dict, row: dict) -> str:
    """The action a frozen decision gives one question. A non-finite confidence is reported with its qid."""
    try:
        return arbitration.apply_decision(decision, row["actions"])["action"]
    except ValueError as exc:
        raise TableError(f"question {row['qid']!r}: {exc}") from exc


def _chosen_scores(decision: dict, rows: list[dict]) -> tuple[list[str], list[float]]:
    """The action each question receives, and its score. The choice reads only the actions; the score is for
    reporting after the choice has been made."""
    chosen = [_choose(decision, row) for row in rows]
    return chosen, [_score_of(row, name) for row, name in zip(rows, chosen)]


def _check_compatible(calibration: dict, metric: str, rows: list[dict], table: Path, decisions_path: Path) -> None:
    """The decisions were calibrated on the same metric and on actions the test table also has."""
    if calibration.get("metric") != metric:
        raise TableError(f"{decisions_path}: calibrated on {calibration.get('metric')!r}, the test table is {metric!r}")
    calibrated = set(calibration.get("scored_on_every_question", [])) | set(calibration.get("partially_available", []))
    test_actions = {name for row in rows for name in row["actions"]}
    missing = sorted(calibrated - test_actions)
    if missing:
        raise TableError(f"{table}: the test table has no actions {missing} that the decisions were calibrated on")


def _strongest(baselines: dict, best: str, test_arms: list[str]) -> dict:
    """The strongest baseline, from calibration means only: best_single, the two adaptive gates, and the arms."""
    means = baselines["calibration_means"]
    candidates = [(best, means["singles"][best]), ("u_gate", means["adaptive"]["u_gate"]),
                  ("nte", means["adaptive"]["nte"])]
    for arm, value in means["arms"].items():
        if arm not in test_arms:
            raise TableError(f"the test table has no baseline arm {arm!r} that the decisions were calibrated on")
        candidates.append((arm, value))
    name, value = max(candidates, key=lambda c: c[1])
    return {"name": name, "calibration_mean": value}


def _baseline_choices(rows: list[dict], baselines: dict, references: list[str]) -> dict[str, list[str]]:
    out = {}
    for name, field in ADAPTIVE_SIGNAL.items():
        tau = arbitration.decode_json_float(baselines[name]["tau"])
        out[name] = [_gate_choice(r["baseline_signals"][field], tau, r["actions"], ref)
                     for r, ref in zip(rows, references)]
    return out


def _entry(scores: np.ndarray, rule_scores: np.ndarray, kind: str, **extra) -> dict:
    return {"kind": kind, "mean": float(scores.mean()), **extra,
            "vs_rule": score_official.paired(scores, rule_scores)}


def _block(rows: list[dict], frozen: dict, baselines: dict, strongest: dict, test_arms: list[str]) -> dict:
    """Statistics on these test questions. The reference is a* where it is available and the reference action of the
    frozen rule otherwise. Every baseline is paired with v2-Rule. The strongest baseline is the one chosen on the
    calibration table, not on these rows."""
    rule = frozen["rule"]
    references = [_reference(rule, r) for r in rows]
    reference_scores = np.asarray([_score_of(r, n) for r, n in zip(rows, references)], dtype=float)
    oracle = np.asarray([max(r["scores"].values()) for r in rows], dtype=float)
    gap_oracle = float(oracle.mean() - reference_scores.mean())
    has_static = all(REFERENCE in r["scores"] for r in rows)
    static = np.asarray([r["scores"][REFERENCE] for r in rows], dtype=float) if has_static else None
    policies, by_family = {}, {}
    for family in POLICY_FAMILIES:
        chosen, scores = _chosen_scores(frozen[family], rows)
        policy = np.asarray(scores, dtype=float)
        by_family[family] = policy
        gap_policy = float(policy.mean() - reference_scores.mean())
        policies[family] = {
            "decision": arbitration.decision_fields(frozen[family]),
            "mean": float(policy.mean()),
            "vs_reference": score_official.paired(reference_scores, policy),
            "vs_static_mixed": score_official.paired(static, policy) if static is not None else None,
            "capture_ratio": (gap_policy / gap_oracle) if gap_oracle > 0 else None,
            "worse_than_reference_fraction": float(np.mean(policy < reference_scores)),
            "chosen": dict(Counter(chosen)),
        }
    rule_scores = by_family["rule"]

    entries = {}
    means = baselines["calibration_means"]
    for name in V2_ACTIONS:
        have = [r for r in rows if name in r["scores"]]
        if not have:
            continue
        value = means["singles"].get(name)            # None for an action that is partial on calibration
        if len(have) == len(rows):
            scores = np.asarray([r["scores"][name] for r in rows], dtype=float)
            entries[name] = _entry(scores, rule_scores, "single", calibration_mean=value)
        else:
            entries[name] = {"kind": "single", "questions": len(have),
                             "mean": _mean(r["scores"][name] for r in have), "vs_rule": None,
                             "calibration_mean": value}
    chosen_gates = _baseline_choices(rows, baselines, references)
    for name in ADAPTIVE_SIGNAL:
        scores = np.asarray([_score_of(r, c) for r, c in zip(rows, chosen_gates[name])], dtype=float)
        entries[name] = _entry(scores, rule_scores, "adaptive", tau=baselines[name]["tau"],
                               calibration_mean=means["adaptive"][name], chosen=dict(Counter(chosen_gates[name])))
    for arm in test_arms:
        scores = np.asarray([r["baseline_scores"][arm] for r in rows], dtype=float)
        entries[arm] = _entry(scores, rule_scores, "arm", calibration_mean=means["arms"].get(arm))

    best = strongest["name"]
    return {
        "questions": len(rows), "reference_mean": float(reference_scores.mean()),
        "reference_counts": dict(Counter(references)), "oracle_mean": float(oracle.mean()),
        "policies": policies,
        "baselines": entries,
        "strongest_baseline": {"name": best, "calibration_mean": strongest["calibration_mean"],
                               "mean": entries[best]["mean"], "vs_rule": entries[best]["vs_rule"]},
    }


def apply(table: Path, decisions_path: Path, out: Path) -> dict:
    decisions_sha = runio.read_frozen_decisions(decisions_path)
    doc = json.loads(decisions_path.read_text(encoding="utf-8"))
    frozen = dev_select.frozen_v2(doc)
    missing = [family for family in POLICY_FAMILIES if family not in frozen]
    if missing:
        raise TableError(f"{decisions_path}: the frozen decisions lack {missing}")
    best = frozen["rule"]["best_single"]
    if frozen["cal"]["best_single"] != best:
        raise TableError(f"{decisions_path}: the rule and cal decisions name different best single actions")
    if not isinstance(doc.get("baselines"), dict) or not isinstance(doc.get("calibration"), dict):
        raise TableError(f"{decisions_path}: no calibration record or baselines; write the decisions with select")
    rows, metric = load_table(table)
    require_actions(rows, table)
    test_arms = require_baselines(rows, table)
    _check_compatible(doc["calibration"], metric, rows, table, decisions_path)
    baselines = doc["baselines"]
    strongest = _strongest(baselines, best, test_arms)
    full = full_actions(rows)
    result = {
        "metric": metric, "best_single": best, "frozen_primary": frozen["primary"],
        "table":{"path": str(table), "sha256": runio.sha256_file(table), "questions": len(rows)},
        "decisions": {"path": str(decisions_path), "sha256": decisions_sha},
        "scored_on_every_question": full, "partially_available": partial_actions(rows),
        **_block(rows, frozen, baselines, strongest, test_arms),
    }
    subsets = sorted({r["subset"] for r in rows if r.get("subset")})
    if subsets:
        result["subsets"] = {s: _block([r for r in rows if r.get("subset") == s], frozen, baselines, strongest,
                                       test_arms) for s in subsets}
    choices = [{"qid": r["qid"], "subset": r.get("subset"),
                "reference": _reference(frozen["rule"], r)} for r in rows]
    for family in POLICY_FAMILIES:
        chosen, scores = _chosen_scores(frozen[family], rows)
        for choice, name, score in zip(choices, chosen, scores):
            choice[family] = {"action": name, "score": score}
    references = [c["reference"] for c in choices]
    for name, picked in _baseline_choices(rows, baselines, references).items():
        for choice, action, row in zip(choices, picked, rows):
            choice[name] = {"action": action, "score": _score_of(row, action)}
    runio.atomic_write_text(out, runio.dumps_strict(result, ensure_ascii=False, indent=2) + "\n")
    runio.atomic_write_text(out.with_name(f"{out.stem}.choices.jsonl"),
                            "".join(runio.dumps_strict(c, ensure_ascii=False) + "\n" for c in choices))
    return result


# ---------------------------------------------------------------- devcheck

def dev_check(table: Path, decisions_path: Path, out: Path) -> tuple[bool, dict]:
    """Apply the frozen decisions to the development table and test the primary selector against a*.

    The comparison is the one ``apply`` reports as ``vs_reference``: the frozen primary family against a* (the
    reference action where a* is unavailable), paired over the development questions. PASS needs the relative gain
    (``relative_gap``, the plain difference when a* scores <= 0) to reach DEV_CHECK_THRESHOLD and the lower end of the
    paired bootstrap 95% CI to be above 0. A fallback decision is a* itself and therefore fails. The strongest
    baseline of the calibration is reported beside it and does not decide the check.
    """
    suffix = ".v2_actions.jsonl"
    if not table.name.endswith(suffix):
        raise TableError(f"{table}: expected a development action table named <dataset>{suffix}")
    dev_main = table.with_name(table.name[:-len(suffix)] + ".v2_dev_main.json")
    result = apply(table, decisions_path, dev_main)
    primary = result["frozen_primary"]
    policy = result["policies"][primary]
    paired = policy["vs_reference"]
    gain = relative_gap(paired["ours"], paired["baseline"])
    ci_low, ci_high = paired["paired_ci95_pp"]
    passed = gain >= DEV_CHECK_THRESHOLD and ci_low > 0.0
    summary = {
        "passed": passed, "threshold": DEV_CHECK_THRESHOLD, "primary": primary,
        "fallback": bool(policy["decision"].get("fallback")), "best_single": result["best_single"],
        "metric": result["metric"], "questions": result["questions"],
        "reference_mean": paired["baseline"], "policy_mean": paired["ours"], "delta_pp": paired["delta_pp"],
        "relative_gain": gain, "ci95_pp": [ci_low, ci_high],
        "wins": paired["wins"], "losses": paired["losses"], "ties": paired["ties"],
        "strongest_baseline": result["strongest_baseline"],
        "table": result["table"], "decisions": result["decisions"],
        "dev_main": {"path": str(dev_main), "sha256": runio.sha256_file(dev_main)},
    }
    runio.atomic_write_text(out, runio.dumps_strict(summary, ensure_ascii=False, indent=2) + "\n")
    return passed, summary


def format_dev_check(passed: bool, summary: dict) -> list[str]:
    lines = [
        "PASS" if passed else "FAIL",
        f"questions={summary['questions']} metric={summary['metric']} primary={summary['primary']} "
        f"fallback={summary['fallback']} best_single={summary['best_single']} "
        f"reference_mean={summary['reference_mean']:.4f} policy_mean={summary['policy_mean']:.4f}",
        f"delta_pp={summary['delta_pp']:.4f} ci95_pp=[{summary['ci95_pp'][0]:.4f}, {summary['ci95_pp'][1]:.4f}] "
        f"relative_gain={summary['relative_gain']:.4f} threshold={summary['threshold']}",
    ]
    if not passed:
        lines.append("开发集检查未通过：选择器相对最强单一动作未达到 +5% 或 CI 下界不大于 0。先反馈，不要直接跑 S4")
    return lines


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

    p = sub.add_parser("select", help="freeze v2-Rule, v2-Cal and the adaptive baselines on a calibration table")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="decisions JSON, e.g. runs/v2_decisions/<dataset>.json")
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=1.0, help="tolerance in score points (0-100 scale)")
    p.add_argument("--seed", type=int, default=0, help="split of the calibration rows for v2-Cal")
    p.add_argument("--confidence", default=arbitration.PRIMARY_CONFIDENCE, choices=arbitration.CONFIDENCE_KINDS,
                   help="confidence the selectors read; u_omni (U at the answer token) is pre-registered, the others "
                        "are ablations")

    p = sub.add_parser("apply", help="apply frozen decisions to a test table; searches nothing")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="main results JSON; choices go next to it")

    p = sub.add_parser("devcheck", help="PASS when the frozen selector beats a* on the development table by >= 5%% "
                                        "with a CI lower bound above 0")
    p.add_argument("--table", type=Path, required=True, help="<dev_run>/<dataset>.v2_actions.jsonl")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="check JSON, e.g. runs/v2_decisions/<dataset>.dev_check.json")

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            summary = build(args.run, bart=args.bart, device=args.device)
            print(json.dumps({"best_single": summary["best_single"], "oracle_mean": summary["oracle_mean"],
                              "actions": {k: v["mean"] for k, v in summary["actions"].items()},
                              "baseline_arms": summary["baseline_arms"], "table": summary["table"]["path"]}, indent=2))
            return 0
        if args.command == "gate":
            passed, summary = gate(args.table)
            print("\n".join(format_gate(passed, summary)))
            return 0 if passed else 1
        if args.command == "select":
            decisions = select(args.table, args.out, args.delta, args.epsilon, args.seed, args.confidence)
            print(json.dumps({
                **{k: v["kind"] + (" (fallback)" if v["fallback"] else "")
                   for k, v in decisions["v2"].items() if isinstance(v, dict)},
                "baselines": {name: decisions["baselines"][name]["tau"] for name in ADAPTIVE_SIGNAL},
            }, indent=2))
            return 0
        if args.command == "devcheck":
            passed, summary = dev_check(args.table, args.decisions, args.out)
            print("\n".join(format_dev_check(passed, summary)))
            return 0 if passed else 1
        result = apply(args.table, args.decisions, args.out)
        print(json.dumps({"best_single": result["best_single"], "strongest_baseline": result["strongest_baseline"]["name"],
                          "rule": result["policies"]["rule"]["mean"], "cal": result["policies"]["cal"]["mean"]},
                         indent=2))
        return 0
    except (TableError, ValueError, runio.FrozenDecisionsError, dev_select.DevDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""S2 go/no-go: dev-set oracle upper bound versus the strongest single action.

  python scripts/analysis/v2_gate.py --scores tests/fixtures/v2_gate_example.json

Input is a JSON object {"questions": [{"qid": str, "scores": {action: number}}]}.
Prints PASS or FAIL, then one detail line. Exit 0 on PASS, 1 on FAIL, 2 on bad input.

PASS when the relative gap is at least +8%:
  (mean(oracle) - mean(best)) / mean(best)    if mean(best) > 0
  mean(oracle) - mean(best)                   otherwise
oracle is the per-question max over actions. Ties take the earliest action in
ACTION_ORDER (then any other name, alphabetically).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

THRESHOLD = 0.08
ACTION_ORDER = (
    "direct",
    "static_text",
    "static_image",
    "static_mixed",
    "rerank_static",
)


def bail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def relative_gap(oracle_mean: float, best_mean: float) -> float:
    if best_mean > 0:
        return (oracle_mean - best_mean) / best_mean
    return oracle_mean - best_mean


def _score(value, qid: str, action: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        bail(f"question {qid}: score for {action} must be a finite number")
    return float(value)


def load_questions(path: Path) -> list[dict]:
    if not path.is_file():
        bail(f"scores file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        bail(f"scores file is not valid JSON: {path} ({exc})")
    except OSError as exc:
        bail(f"could not read scores file: {path} ({exc})")
    if not isinstance(payload, dict) or "questions" not in payload:
        bail("scores file must be a JSON object with a questions list")
    questions = payload["questions"]
    if not isinstance(questions, list) or not questions:
        bail("questions must be a non-empty list")
    parsed = []
    keys = None
    for i, row in enumerate(questions):
        if not isinstance(row, dict):
            bail(f"questions[{i}] must be an object")
        qid = row.get("qid")
        scores = row.get("scores")
        if not isinstance(qid, str) or not qid:
            bail(f"questions[{i}] needs a non-empty string qid")
        if not isinstance(scores, dict) or not scores:
            bail(f"question {qid}: scores must be a non-empty object")
        row_keys = tuple(scores)
        if keys is None:
            keys = row_keys
        elif row_keys != keys:
            bail(f"question {qid}: score actions differ from the first question")
        parsed.append({"qid": qid, "scores": {a: _score(scores[a], qid, a) for a in row_keys}})
    return parsed


def summarize(questions: list[dict]) -> dict:
    actions = list(questions[0]["scores"])
    means = {a: sum(q["scores"][a] for q in questions) / len(questions) for a in actions}
    best_mean = max(means.values())
    order = {name: i for i, name in enumerate(ACTION_ORDER)}
    tied = [a for a, m in means.items() if m == best_mean]
    best = min(tied, key=lambda a: (order.get(a, len(ACTION_ORDER)), a))
    oracle_mean = sum(max(q["scores"].values()) for q in questions) / len(questions)
    gap = relative_gap(oracle_mean, means[best])
    return {
        "oracle_mean": oracle_mean,
        "best_single": best,
        "best_single_mean": means[best],
        "relative_gap": gap,
        "passed": gap >= THRESHOLD,
    }


def format_detail(summary: dict) -> str:
    return (
        f"oracle_mean={summary['oracle_mean']:.6f} "
        f"best_single={summary['best_single']} "
        f"best_single_mean={summary['best_single_mean']:.6f} "
        f"relative_gap={summary['relative_gap']:.6f}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="JSON file of per-question action scores")
    args = ap.parse_args(argv)
    summary = summarize(load_questions(Path(args.scores)))
    print("PASS" if summary["passed"] else "FAIL")
    print(format_detail(summary))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()

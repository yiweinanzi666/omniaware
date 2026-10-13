"""Where the uncertainty is measured: the answer-first format check of S1 and the measurement-position analysis.

    python scripts/analysis/answer_slot.py format   --run runs/<tag>/<dataset>.jsonl [--min-answer-first 0.9]
    python scripts/analysis/answer_slot.py position --table runs/<tag>/<dataset>.v2_actions.jsonl [--out PATH]

format reads a --v2 run. For every available action it reports how often the output kept the answer-first format
(answer_first_ok), how often the answer token is the first generated token (answer_token.index == 0), and how often
the first generated token is a function word ("template start", signals.FUNCTION_WORDS). For an answer-first run it
exits 1 when some action keeps the format on fewer than --min-answer-first of its questions: the first-token signals
of the selector are then measured on template words, the failure the answer-first format exists to remove.

position reads a scored action table (v2_table.py build). It compares the confidence of the direct action measured
at the first generated token with the same confidence measured at the answer token, as AUROC for two targets:
direct is correct (score >= 50 on the 0-100 scale), and, on the questions where direct and static_mixed score
differently, static_mixed is the better one (higher uncertainty should mean retrieve). The rows are also split by
whether the first generated token is a template word. Both positions come from the same decode of the same run, so
the comparison isolates the measurement position.

Exit codes: 0 success, 1 format check failed, 2 bad input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import runio  # noqa: E402
from omniaware.actions import DIRECT, V2_ACTIONS  # noqa: E402
from omniaware.signals import FUNCTION_WORDS  # noqa: E402

MIN_ANSWER_FIRST = 0.9
REFERENCE = "static_mixed"
CORRECT = 50.0


class SlotError(Exception):
    """A run or table is missing or lacks the fields this analysis reads."""


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SlotError(f"file not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SlotError(f"{path}: no rows")
    return rows


def first_text(record: dict) -> str | None:
    """Text of the first generated token: the greedy token is the top-1 of first_token.topk."""
    topk = (record.get("first_token") or {}).get("topk") or []
    return topk[0].get("text") if topk else None


def template_start(record: dict) -> bool | None:
    text = first_text(record)
    return None if text is None else text.strip().lower() in FUNCTION_WORDS


def _rate(flags: list) -> float | None:
    known = [bool(f) for f in flags if f is not None]
    return (sum(known) / len(known)) if known else None


# ---------------------------------------------------------------- format

def format_report(records: list[dict]) -> dict:
    formats = {r.get("answer_format", "plain") for r in records}
    if len(formats) != 1:
        raise SlotError(f"the run mixes answer formats {sorted(formats)}")
    answer_format = formats.pop()
    actions = {}
    for name in V2_ACTIONS:
        available = [r["actions"][name] for r in records
                     if name in (r.get("actions") or {}) and r["actions"][name].get("available", True)]
        if not available:
            continue
        slots = [a.get("answer_token") for a in available]
        actions[name] = {
            "questions": len(available),
            "answer_first_ok": _rate([a.get("answer_first_ok") for a in available]),
            "answer_token_is_first": _rate([None if s is None else s["index"] == 0 for s in slots]),
            "template_start": _rate([template_start(a) for a in available]),
        }
    if not actions:
        raise SlotError("no v2 actions in the run; it needs omniaware.run --v2")
    return {"answer_format": answer_format, "questions": len(records), "actions": actions}


def format_passes(report: dict, minimum: float) -> tuple[bool, list[str]]:
    if report["answer_format"] != "answer_first":
        return True, []
    low = [f"{name} answer_first_ok={stats['answer_first_ok']}" for name, stats in report["actions"].items()
           if stats["answer_first_ok"] is None or stats["answer_first_ok"] < minimum]
    return not low, low


# ---------------------------------------------------------------- position

def auroc(score, label) -> float | None:
    """Mann-Whitney AUROC with average ranks for ties; None when a class is empty."""
    score, label = np.asarray(score, dtype=float), np.asarray(label, dtype=bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


SIGNALS = {
    # name: (where, field, sign): confidence = sign * value; higher is more confident
    "first_token.neg_entropy": ("first_token", "entropy_norm", -1.0),
    "first_token.margin": ("first_token", "margin", 1.0),
    "answer_token.neg_entropy": ("answer_token", "entropy_norm", -1.0),
    "answer_token.margin": ("answer_token", "margin", 1.0),
    "answer_token.neg_u_omni": ("answer_token", "u_raw", -1.0),
}


def _confidence(record: dict, signal: str) -> float:
    where, field, sign = SIGNALS[signal]
    value = (record.get(where) or {}).get(field)
    if value is None or not np.isfinite(value):
        raise SlotError(f"{signal} is missing or not finite; the table needs a --v2 run with answer_token")
    return sign * float(value)


def _block(rows: list[dict]) -> dict:
    direct = [r["actions"][DIRECT] for r in rows]
    correct = [r["scores"][DIRECT] >= CORRECT for r in rows]
    differ = [i for i, r in enumerate(rows)
              if REFERENCE in r["scores"] and r["scores"][REFERENCE] != r["scores"][DIRECT]]
    better = [rows[i]["scores"][REFERENCE] > rows[i]["scores"][DIRECT] for i in differ]
    out = {"questions": len(rows), "direct_correct_rate": float(np.mean(correct)) if rows else None,
           "disagree": len(differ), "signals": {}}
    for signal in SIGNALS:
        conf = [_confidence(d, signal) for d in direct]
        out["signals"][signal] = {
            "auroc_direct_correct": auroc(conf, correct),
            # retrieve when direct is unsure: the target is "evidence better", scored by -confidence
            "auroc_evidence_better": auroc([-conf[i] for i in differ], better),
        }
    return out


def position_report(rows: list[dict]) -> dict:
    usable = [r for r in rows if DIRECT in r["scores"] and r["actions"].get(DIRECT, {}).get("available", True)]
    if not usable:
        raise SlotError("no row has a scored direct action")
    starts = {"template_start": [], "answer_start": [], "unknown": []}
    for r in usable:
        flag = template_start(r["actions"][DIRECT])
        starts["unknown" if flag is None else ("template_start" if flag else "answer_start")].append(r)
    return {"metric": usable[0].get("metric"), "all": _block(usable),
            "by_first_token": {k: _block(v) for k, v in starts.items() if v}}


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("format", help="answer-first compliance and template starts of a --v2 run")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--min-answer-first", type=float, default=MIN_ANSWER_FIRST)
    p.add_argument("--out", type=Path, default=None, help="default: <run stem>.answer_format.json beside the run")
    p = sub.add_parser("position", help="AUROC of the direct confidence at the first generated vs the answer token")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="default: <table stem>.answer_slot.json beside the table")
    args = parser.parse_args(argv)
    try:
        if args.command == "format":
            report = format_report(read_jsonl(args.run))
            passed, low = format_passes(report, args.min_answer_first)
            report.update(min_answer_first=args.min_answer_first, passed=passed, below=low)
            out = args.out or args.run.with_name(f"{args.run.stem}.answer_format.json")
            runio.atomic_write_text(out, runio.dumps_strict(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            if not passed:
                print("ANSWER-FIRST FORMAT BELOW THRESHOLD: " + "; ".join(low), file=sys.stderr)
            return 0 if passed else 1
        report = position_report(read_jsonl(args.table))
        out = args.out or args.table.with_name(f"{args.table.stem}.answer_slot.json")
        runio.atomic_write_text(out, runio.dumps_strict(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    except (SlotError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

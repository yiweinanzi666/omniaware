"""Human audit of the secondary metric semantic_acc against the official metric (pre-registered, DEV_DECISIONS v2).

    python scripts/analysis/semantic_audit.py export --table runs/<dev_tag>/<dataset>.v2_actions.jsonl \
        --out-dir runs/<dev_tag>/semantic_audit [--per-stratum 25] [--seed 20261013]
    python scripts/analysis/semantic_audit.py score --key runs/<dev_tag>/semantic_audit/audit_key.json \
        --labels runs/<dev_tag>/semantic_audit/audit_labels.csv

export reads a development action table built with --semantic (v2_table.py build --semantic) and the run it came from.
Every available action of every question is one candidate item, placed in one of four strata by whether the official
metric and semantic_acc count it correct (score >= 50). Up to --per-stratum items are drawn from each stratum with a
fixed seed. audit_items.csv holds, in shuffled order, only what a person needs to judge an answer: an item id, the
question, the reference answer(s) and the predicted answer, plus an empty label column. The action, the scores and
the stratum are in audit_key.json, which the labeller does not open.

score reads the key and the labelled sheet (label 1 = the prediction answers the question as the reference does,
0 = it does not; every item must be labelled). For each metric it reports the agreement with the human labels per
stratum, the population agreement (strata weighted by their sizes in the table), Cohen's kappa on the labelled
sample, and the human-estimated accuracy of the sampled pool. Both metrics are reported in every table whatever the
audit shows; the audit is reported beside them.

Exit codes: 0 success, 2 bad input.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import runio  # noqa: E402

SECONDARY = "semantic_acc"
CORRECT = 50.0
PER_STRATUM = 25
SEED = 20261013
STRATA = ("both_correct", "official_only", "semantic_only", "both_wrong")
COLUMNS = ("item_id", "question", "reference", "prediction", "label")


class AuditError(Exception):
    """A table, run, key or label file is missing or inconsistent."""


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise AuditError(f"file not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stratum(official: float, secondary: float) -> str:
    o, s = official >= CORRECT, secondary >= CORRECT
    return STRATA[0] if o and s else STRATA[1] if o else STRATA[2] if s else STRATA[3]


def run_of(table: Path) -> Path:
    suffix = ".v2_actions.jsonl"
    if not table.name.endswith(suffix):
        raise AuditError(f"{table}: expected an action table named <dataset>{suffix}")
    return table.with_name(table.name[:-len(suffix)] + ".jsonl")


def candidates(table: Path) -> list[dict]:
    rows = read_jsonl(table)
    records = {r["qid"]: r for r in read_jsonl(run_of(table))}
    out = []
    for row in rows:
        if "secondary_scores" not in row:
            raise AuditError(f"{table}: question {row['qid']!r} has no secondary_scores; build with --semantic")
        record = records.get(row["qid"])
        if record is None:
            raise AuditError(f"{table}: question {row['qid']!r} is not in {run_of(table)}")
        for name, official in sorted(row["scores"].items()):
            if name not in row["secondary_scores"]:
                raise AuditError(f"question {row['qid']!r}: {name} has a score but no {SECONDARY}")
            secondary = float(row["secondary_scores"][name])
            out.append({"qid": row["qid"], "action": name, "official": float(official), "secondary": secondary,
                        "stratum": stratum(float(official), secondary), "question": record["question"],
                        "reference": " | ".join(str(g) for g in record["golds"]),
                        "prediction": row["actions"][name]["answer"]})
    return out


def export(table: Path, out_dir: Path, per_stratum: int = PER_STRATUM, seed: int = SEED) -> dict:
    pool = candidates(table)
    rng = np.random.default_rng(seed)
    sizes = {s: sum(c["stratum"] == s for c in pool) for s in STRATA}
    chosen = []
    for s in STRATA:
        members = [c for c in pool if c["stratum"] == s]
        take = min(per_stratum, len(members))
        chosen += [members[i] for i in sorted(rng.choice(len(members), size=take, replace=False))] if take else []
    order = rng.permutation(len(chosen))
    items = [{**chosen[i], "item_id": f"a{k:04d}"} for k, i in enumerate(order)]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "audit_items.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for item in items:
            writer.writerow({"item_id": item["item_id"], "question": item["question"],
                             "reference": item["reference"], "prediction": item["prediction"], "label": ""})
    key = {"table": {"path": str(table), "sha256": runio.sha256_file(table)}, "seed": seed,
           "per_stratum": per_stratum, "correct_at": CORRECT, "stratum_sizes": sizes,
           "items": [{k: item[k] for k in ("item_id", "qid", "action", "official", "secondary", "stratum")}
                     for item in items]}
    runio.atomic_write_text(out_dir / "audit_key.json", runio.dumps_strict(key, indent=2) + "\n")
    return key


def _kappa(metric: list[bool], human: list[bool]) -> float | None:
    m, h = np.asarray(metric, dtype=bool), np.asarray(human, dtype=bool)
    if m.size == 0:
        return None
    observed = float(np.mean(m == h))
    expected = float(m.mean() * h.mean() + (1 - m.mean()) * (1 - h.mean()))
    return None if expected == 1.0 else (observed - expected) / (1 - expected)


def score(key_path: Path, labels_path: Path) -> dict:
    if not key_path.is_file() or not labels_path.is_file():
        raise AuditError(f"need both {key_path} and {labels_path}")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    with labels_path.open(newline="", encoding="utf-8") as handle:
        labels = {row["item_id"]: row["label"].strip() for row in csv.DictReader(handle)}
    items = key["items"]
    missing = [i["item_id"] for i in items if labels.get(i["item_id"]) not in ("0", "1")]
    if missing:
        raise AuditError(f"{len(missing)} items are not labelled 0 or 1, first {missing[:3]}")
    human = {i["item_id"]: labels[i["item_id"]] == "1" for i in items}
    sizes = key["stratum_sizes"]
    total = sum(sizes[s] for s in STRATA if any(i["stratum"] == s for i in items))
    out = {"items": len(items), "stratum_sizes": sizes, "metrics": {}}
    weighted_human = 0.0
    for s in STRATA:
        members = [i for i in items if i["stratum"] == s]
        if members:
            weighted_human += sizes[s] / total * float(np.mean([human[i["item_id"]] for i in members]))
    out["human_accuracy_estimate"] = 100.0 * weighted_human
    for metric, field in (("official", "official"), (SECONDARY, "secondary")):
        per, weighted = {}, 0.0
        for s in STRATA:
            members = [i for i in items if i["stratum"] == s]
            if not members:
                continue
            agree = float(np.mean([(i[field] >= CORRECT) == human[i["item_id"]] for i in members]))
            per[s] = {"items": len(members), "agreement": agree}
            weighted += sizes[s] / total * agree
        out["metrics"][metric] = {
            "agreement_by_stratum": per, "population_agreement": weighted,
            "sample_kappa": _kappa([i[field] >= CORRECT for i in items], [human[i["item_id"]] for i in items])}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="draw the blind, stratified audit sample from a development table")
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <table dir>/semantic_audit")
    p.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    p.add_argument("--seed", type=int, default=SEED)
    p = sub.add_parser("score", help="agreement of the official metric and semantic_acc with the human labels")
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="default: audit_result.json beside the key")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            key = export(args.table, args.out_dir or args.table.parent / "semantic_audit", args.per_stratum, args.seed)
            print(json.dumps({"items": len(key["items"]), "stratum_sizes": key["stratum_sizes"]}, indent=2))
            return 0
        result = score(args.key, args.labels)
        runio.atomic_write_text(args.out or args.key.with_name("audit_result.json"),
                                runio.dumps_strict(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    except (AuditError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

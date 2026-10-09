"""Routing table (paper Table 4): wrong-route rate, coverage, modalities per query, answer score.

    python -m scripts.analysis.routing_metrics runs/route_mh/webqa_open_full.jsonl runs/route_all/... \
        [--arm gated:eab:g1.2:ans2:pos:gate] [--metric acc_approx]

Among triggered queries (retrieve = true) with gold modalities M* and searched modalities M':
WR = share with M' ∩ M* empty, Cov = share with M* ⊆ M'. An empty route (rule 'all', or a cold
start without routing) searches every modality offered by the candidates. The score column is the
official per-question score of ``--arm`` from <run>.official_scores.jsonl when present.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def searched(row):
    route = row.get("route") or {}
    return set(route.get("modalities") or row.get("candidate_modalities") or [])


def routing_metrics(rows, all_modalities):
    triggered = [r for r in rows if r.get("retrieve", True) and r.get("gold_modalities")]
    wr = cov = n_mod = 0
    counts = Counter()
    for r in triggered:
        m_star = set(r["gold_modalities"])
        m_sel = searched(r) or set(all_modalities)
        wr += not (m_sel & m_star)
        cov += m_star <= m_sel
        n_mod += len(m_sel)
        counts["+".join(sorted(m_sel))] += 1
    n = len(triggered)
    return {"questions": len(rows), "triggered": n,
            "wrong_route_rate": 100 * wr / n if n else None,
            "coverage_rate": 100 * cov / n if n else None,
            "modalities_per_query": n_mod / n if n else None,
            "routes": {k: 100 * v / n for k, v in sorted(counts.items())} if n else {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--modalities", nargs="+", default=["image", "text"])
    args = parser.parse_args()
    table = {}
    for path in args.runs:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        result = routing_metrics(rows, args.modalities)
        scores_path = path.with_name(path.stem + ".official_scores.jsonl")
        if args.arm and args.metric and scores_path.exists():
            scores = [json.loads(line)["scores"][args.arm][args.metric] for line in scores_path.read_text().splitlines()]
            result["score"] = sum(scores) / len(scores)
        table[str(path)] = result
    print(json.dumps(table, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""UniversalRAG baseline (Yeo et al., 2026): query-only modality routing with the official router.

The official router (external/UniversalRAG, pinned in third_party/external_repos.json) is trained
and run with the upstream scripts; this module only moves data in and out:

    # 1. export every question of a dataset (no answers) in the upstream query format
    python -m omniaware.baselines.universalrag export --dataset webqa_open_full --output q.json
    # 2. exclude our evaluation questions from the upstream router training data, then train/route upstream
    python -m omniaware.baselines.universalrag filter-train --train route/train/data/train_data.json \\
        --dataset webqa_open_full --output route/train/data/train_data.filtered.json
    #    (upstream) python route/train/train_qwen3_vl.py --train-data ...filtered.json ...
    #    (upstream) python route/train/route_qwen3_vl.py --model-path ... --input q.json --output routes.json
    # 3. run:  python -m omniaware.run ... --route-rule query --query-routes routes.json

Upstream labels map to our corpora: paragraph/document -> text, table -> table, image -> image,
clip/video -> video, no -> no retrieval. The 'universalrag' arm answers directly when the router
says 'no' and otherwise uses static concatenation of the routed top-k.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL_TO_MODALITY = {"paragraph": "text", "document": "text", "table": "table", "image": "image",
                     "clip": "video", "video": "video", "no": None}
INDEX_PREFIX = {"webqa": "webqa_"}


def upstream_index(example: dict) -> str:
    """Upstream query ids prefix the dataset name (e.g. webqa_<guid>)."""
    return INDEX_PREFIX.get(example["dataset"], f"{example['dataset']}_") + str(example["qid"])


def export_queries(examples: list[dict]) -> list[dict]:
    return [{"index": upstream_index(e), "question": e["question"],
             **({"query_image": e["query_images"][0]} if e.get("query_images") else {})} for e in examples]


def load_routes(path: Path) -> dict[str, list[str]]:
    rows = json.loads(Path(path).read_text(encoding="utf8"))
    return {row["index"]: list(row["retrieval"]) for row in rows}


def to_route(labels: list[str], available: set[str]) -> dict:
    """Router labels -> {retrieve, modalities}; modalities absent from the corpus are dropped."""
    unknown = set(labels) - set(LABEL_TO_MODALITY)
    if unknown:
        raise ValueError(f"unknown UniversalRAG labels {sorted(unknown)}")
    mapped = {LABEL_TO_MODALITY[label] for label in labels} - {None}
    return {"mode": "query", "rule": "query", "labels": labels, "retrieve": bool(mapped),
            "modalities": sorted(mapped & available)}


def overlapping(train_rows: list[dict], examples: list[dict]) -> set[str]:
    ours = {upstream_index(e) for e in examples}
    return {row["index"] for row in train_rows if row.get("index") in ours}


def main():
    from .. import data
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("export")
    a.add_argument("--dataset", required=True, choices=sorted(data.LOADERS))
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("filter-train")
    b.add_argument("--train", type=Path, required=True)
    b.add_argument("--dataset", required=True, choices=sorted(data.LOADERS))
    b.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    examples = data.LOADERS[args.dataset]()
    if args.cmd == "export":
        args.output.write_text(json.dumps(export_queries(examples), ensure_ascii=False, indent=1), encoding="utf8")
        print(f"exported {len(examples)} queries -> {args.output}")
    else:
        rows = json.loads(args.train.read_text(encoding="utf8"))
        drop = overlapping(rows, examples)
        kept = [r for r in rows if r.get("index") not in drop]
        args.output.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf8")
        print(f"removed {len(drop)} training queries that are evaluation questions; kept {len(kept)} -> {args.output}")


if __name__ == "__main__":
    main()

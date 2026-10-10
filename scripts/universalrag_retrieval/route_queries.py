"""Route every query with UniversalRAG's official router (the Qwen3-VL multi-label classifier).

Step 2 of the UniversalRAG baseline (src/omniaware/baselines/universalrag.py). The router is trained by
the upstream scripts (external/UniversalRAG/route/train/train_qwen3_vl.py); this script only runs its
inference class on our exported queries:

    # 1. export the queries of a dataset (no answers, no gold)
    python -m omniaware.baselines.universalrag export --dataset webqa_open_full --output q_webqa.json
    # 2. route them with the trained router checkpoint
    python scripts/universalrag_retrieval/route_queries.py --queries q_webqa.json \\
        --model-path <router checkpoint dir> --output routes_webqa.json
    # 3. run the baseline with the routes
    python -m omniaware.run ... --route-rule query --query-routes routes_webqa.json

It runs upstream's Qwen3VLMultiLabelRouter.route_batch unchanged, with upstream's CLI defaults: sigmoid
threshold 0.8, LoRA adapters loaded from the checkpoint, and per-label scores kept. Output is the input
list with "retrieval" (upstream labels, e.g. ["table", "paragraph"] or ["no"]) and "retrieval_scores".
Needs the UniversalRAG environment (torch, transformers, peft, qwen_vl_utils; the router is loaded with
flash_attention_2, as in upstream); --help works without it.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_UNIVERSALRAG = Path(__file__).resolve().parents[2] / "external" / "UniversalRAG"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True,
                    help="exported queries: a list of {index, question, query_image?} (universalrag.py export)")
    ap.add_argument("--model-path", required=True, help="router checkpoint directory (train_qwen3_vl.py output)")
    ap.add_argument("--output", type=Path, required=True, help="routes JSON for --query-routes")
    ap.add_argument("--universalrag", type=Path, default=DEFAULT_UNIVERSALRAG,
                    help="UniversalRAG checkout (default: external/UniversalRAG in this repository)")
    ap.add_argument("--threshold", type=float, default=0.8, help="sigmoid threshold for selecting labels (upstream 0.8)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--classifier-dropout", type=float, default=0.1)
    ap.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True,
                    help="load LoRA adapters from --model-path (upstream CLI default)")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.universalrag.resolve()))   # makes `route` importable, as upstream expects
    from route.train.route_qwen3_vl import Qwen3VLMultiLabelRouter

    rows = json.loads(args.queries.read_text(encoding="utf8"))
    router = Qwen3VLMultiLabelRouter(str(args.model_path), classifier_dropout=args.classifier_dropout,
                                     use_lora=args.use_lora)
    predictions = router.route_batch([row["question"] for row in rows],
                                     query_images=[row.get("query_image") for row in rows],
                                     batch_size=args.batch_size, threshold=args.threshold, return_scores=True)
    routes = [dict(row, retrieval=labels, retrieval_scores=scores)
              for row, (labels, scores) in zip(rows, predictions)]
    args.output.write_text(json.dumps(routes, ensure_ascii=False, indent=1), encoding="utf8")
    print(f"routed {len(routes)} queries -> {args.output}")
    print("label sets:", dict(Counter("+".join(labels) for labels, _ in predictions)))


if __name__ == "__main__":
    main()

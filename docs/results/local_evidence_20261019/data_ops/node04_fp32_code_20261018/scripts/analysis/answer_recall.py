"""Does the packed evidence contain an annotator answer? Split by closed-book (direct) correctness.

    python -m scripts.analysis.answer_recall runs/<tag>/okvqa.jsonl [--metric vqa_accuracy] [--correct-at 50]

"Closed-book correct" uses the same threshold as paper_analyses (--correct-at, default 50, the VQA threshold; a
score of 33.3 is wrong). The earlier > 0 rule counted one-of-ten annotator matches as correct.

Reads the run's chunk ids, the dataset's candidate texts and the run's official scores. A hit is any
reference answer as a whole-word substring of the concatenated packed text (norm() unchanged from the
legacy okvqa_passage_answer_recall.py). Diagnostic of retrieval quality only: answer presence is
necessary, not sufficient, for a correct grounded answer. Short-answer datasets (OK-VQA, MMQA).
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import data  # noqa: E402


def norm(text):
    return ' ' + re.sub(r'[^0-9a-z]+', ' ', text.lower()).strip() + ' '


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metric", default="vqa_accuracy")
    parser.add_argument("--correct-at", type=float, default=50.0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    config = json.loads(args.predictions.with_name(args.predictions.stem + "_config.json").read_text())
    scores = {r["qid"]: r["scores"] for r in map(json.loads, args.predictions.with_suffix("")
              .with_name(args.predictions.stem + ".official_scores.jsonl").read_text().splitlines())}
    examples = {e["qid"]: e for e in data.LOADERS[config["dataset"]]()}
    table = {}
    for r in rows:
        e = examples[r["qid"]]
        text = {c["cid"]: f"{c['title']} {c['text']}" for c in e["candidates"]}
        packed = norm(" ".join(text[cid] for cid in r["chunks"]))
        golds = [norm(a) for a in r["golds"] if norm(a).strip()]
        hit = any(g in packed for g in golds)
        closed = ("closed_correct" if scores[r["qid"]]["direct"][args.metric] >= args.correct_at else "closed_wrong")
        cell = table.setdefault(closed, dict(questions=0, packed_evidence_contains_answer=0))
        cell["questions"] += 1
        cell["packed_evidence_contains_answer"] += hit
    for cell in table.values():
        cell["rate"] = cell["packed_evidence_contains_answer"] / cell["questions"]
    print(json.dumps(dict(questions=len(rows), table=table), indent=1))


if __name__ == "__main__":
    main()

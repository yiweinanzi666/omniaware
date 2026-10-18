"""Export a cite-style run to CiteRec intake files (answers.jsonl, sources.jsonl).

    python -m omniaware.eval.citation.from_run runs/<tag>/<dataset>.jsonl --output runs/<tag>/citerec
    python -m omniaware.eval.citation.intake_answers --input .../answers.jsonl --sources .../sources.jsonl \
        --output .../intake --variants static eab:g1.2:ans2:pos:gate
    python -m omniaware.eval.citation.run_nli --input .../intake/cite_rec_nli_inputs.jsonl --output .../nli.jsonl
    python -m omniaware.eval.citation.score_nli --ledger .../intake/answer_ledger.jsonl \
        --requests .../intake/cite_rec_nli_inputs.jsonl --results .../nli.jsonl --output .../citerec.json

Only arms decoded from the shared evidence input are exported (static and the bias variants);
direct and gated arms do not see the same [E<j>] evidence, and replication renumbers copies.
"""
import argparse
import hashlib
import json
from pathlib import Path


def export(predictions: Path, output: Path, arms=None):
    raw = predictions.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    config_path = predictions.with_name(predictions.stem + "_config.json")
    config = json.loads(config_path.read_text())
    if config.get("answer_style") != "cite":
        raise ValueError("CiteRec needs a run with --answer-style cite")
    shared = ["static"] + [v["name"] for v in config["variants"]]
    arms = list(arms or shared)
    if not set(arms) <= set(shared):
        raise ValueError(f"only arms with the shared evidence input can be exported: {shared}")
    identity = {"predictions_sha256": hashlib.sha256(raw).hexdigest(),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    answers, sources = [], []
    for r in records:
        sources.append({"dataset": r["dataset"], "qid": str(r["qid"]), "status": "CITATION_SOURCE_RECORD_READY",
                        "evidence": r["citation_sources"]})
        for arm in arms:
            answers.append({"dataset": r["dataset"], "qid": str(r["qid"]), "variant": arm,
                            "raw_answer": r["preds"][arm], "raw_token_ids": r["tokens"][arm],
                            "generation_identity": dict(identity, arm=arm)})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("answers.jsonl", answers), ("sources.jsonl", sources)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return {"questions": len(records), "arms": arms, "answers": len(answers)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=None)
    args = parser.parse_args()
    print(json.dumps(export(args.predictions, args.output, args.arms)))

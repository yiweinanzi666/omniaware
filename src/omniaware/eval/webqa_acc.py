"""Score paired training-development predictions with official WebQA acc_approx.

Only the published functions/constants are loaded from pinned source files; the
original evaluation script's top-level CUDA/BART initialization is never run.
The train/val release uses exactly one reference per question.
Scores are in [0, 1]; differences also expose percentage points.
"""

import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import string

import numpy as np

from .. import paths


VARIANTS = ("bias-off", "paper-timing", "first-token")
OFFICIAL_FILES = {
    "eval_1022.py": "6862a280454b576062695724b3417c9e4ce43af5447596ff4efca15d8d1e3f5c",
    "eval_webqa.md": "c6dfcb9a7f8aed27c393e3d1296e038f37f10483de66b771963770482fea1ab3",
}


def _selected_ast(source, function_names=(), constant_names=()):
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in constant_names:
                nodes.append(node)
    return ast.Module(body=nodes, type_ignores=[])


def load_official_metric(source_dir=None, spacy_model="en_core_web_sm"):
    """Return the unmodified published approximate metric and its provenance."""
    source_dir = Path(source_dir or paths.THIRD_PARTY / "webqa")
    sources, source_hashes = {}, {}
    for filename in OFFICIAL_FILES:
        raw = (source_dir / filename).read_bytes()
        source_hashes[filename] = hashlib.sha256(raw).hexdigest()
        if source_hashes[filename] != OFFICIAL_FILES[filename]:   # checked before anything is executed
            raise RuntimeError(f"{source_dir / filename}: sha256 {source_hashes[filename]} does not match the pinned "
                               f"{OFFICIAL_FILES[filename]}; the official source has changed")
        sources[filename] = raw.decode("utf-8")
    import spacy
    from word2number import w2n

    nlp = spacy.load(spacy_model, disable=["ner", "textcat", "parser"])
    namespace = dict(re=re, string=string, Counter=Counter, w2n=w2n, nlp=nlp)
    tree = _selected_ast(
        sources["eval_1022.py"],
        function_names=("detectNum", "toNum", "normalize_text"),
        constant_names=("color_set", "shape_set", "yesno_set"),
    )
    exec(compile(tree, "official_eval/eval_1022.py:selected", "exec"), namespace)
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", sources["eval_webqa.md"], re.DOTALL)
    code = next(block for block in code_blocks if "def webqa_metrics_approx" in block)
    tree = _selected_ast(code, function_names=("_webqa_acc_approx", "webqa_metrics_approx"))
    exec(compile(tree, "official_eval/eval_webqa.md:published", "exec"), namespace)
    provenance = {
        "metric": "WebQA public-data acc_approx component",
        "webqa_commit": "81cc377aaddb2cf9de98cf082a2795ce53ccc1f4",
        "baseline_commit": "279d5ae47faee096f7d4171ba0bacdfaec27d30f",
        "source_sha256": source_hashes,
        "spacy_version": spacy.__version__,
        "spacy_model": nlp.meta.get("name"),
        "spacy_model_version": nlp.meta.get("version"),
        "word2number_version": importlib.metadata.version("word2number"),
        "numpy_version": np.__version__,
        "reference_rule": "exactly one reference, as in all 41732 original train/val examples; multiple references rejected",
        "leaderboard_composite": False,
        "bartscore_computed": False,
    }
    return namespace["webqa_metrics_approx"], provenance


def score_references(prediction, references, category, metric):
    if len(references) != 1:
        raise ValueError("references must contain exactly one string, matching the official train/val release")
    values = [float(metric(prediction, ref, Qcate=category)["acc_approx"]) for ref in references]
    return values[0], values


def score_paired(records, metric, *, bootstrap_resamples=10000, bootstrap_seed=20261002):
    """Require all three variants for every question and resample questions."""
    by_id = {}
    for row in records:
        qid, variant = row.get("qid"), row.get("variant")
        variants = by_id.setdefault(qid, {})
        if variant in variants:
            raise ValueError(f"Duplicate question/variant: {qid}/{variant}")
        variants[variant] = row
    if not by_id:
        raise ValueError("No prediction records")

    scored = []
    matrix = []
    qids = sorted(by_id)
    for qid in qids:
        rows = by_id[qid]
        if set(rows) != set(VARIANTS):
            raise ValueError(f"Unpaired question: {qid}; variants={sorted(rows)}")
        reference = rows[VARIANTS[0]]
        expected = (reference.get("references"), reference.get("category"))
        sample_scores = []
        for variant in VARIANTS:
            row = rows[variant]
            if (row.get("references"), row.get("category")) != expected:
                raise ValueError(f"Reference/category mismatch between variants: {qid}")
            value, values = score_references(row.get("prediction"), *expected, metric)
            scored.append(dict(row, acc_approx=value, reference_acc_approx=values))
            sample_scores.append(value)
        matrix.append(sample_scores)

    scores = np.asarray(matrix, dtype=np.float64)
    differences = scores[:, 2, None] - scores[:, :2]
    rng = np.random.default_rng(bootstrap_seed)
    boot = np.empty((bootstrap_resamples, 2))
    for start in range(0, bootstrap_resamples, 256):
        stop = min(start + 256, bootstrap_resamples)
        indices = rng.integers(0, len(qids), size=(stop - start, len(qids)))
        boot[start:stop] = differences[indices].mean(axis=1)
    comparisons = {}
    for column, baseline in enumerate(VARIANTS[:2]):
        delta = differences[:, column]
        interval = np.quantile(boot[:, column], [.025, .975])
        comparisons[f"first-token_minus_{baseline}"] = {
            "mean_difference": float(delta.mean()),
            "mean_difference_pp": float(100 * delta.mean()),
            "ci95": interval.tolist(),
            "ci95_pp": (100 * interval).tolist(),
            "improved_questions": int((delta > 0).sum()),
            "unchanged_questions": int((delta == 0).sum()),
            "worsened_questions": int((delta < 0).sum()),
            "question_differences": dict(zip(qids, delta.tolist())),
        }
    variant_results = {}
    for column, variant in enumerate(VARIANTS):
        timings = [row["inference_seconds"] for row in scored if row["variant"] == variant and "inference_seconds" in row]
        variant_results[variant] = {
            "question_count": len(qids), "mean_acc_approx": float(scores[:, column].mean()),
            "inference_seconds_count": len(timings),
            "mean_inference_seconds": float(np.mean(timings)) if timings else None,
        }
    summary = {
        "scope": "training-split text candidate-set development experiment",
        "heldout_confirmation": False, "leaderboard_composite": False,
        "question_ids": qids, "variants": variant_results,
        "paired_comparisons": comparisons,
        "bootstrap": {"unit": "question", "resamples": bootstrap_resamples,
                      "seed": bootstrap_seed, "interval": "percentile 95%", "numpy_quantile_method": "linear"},
    }
    return scored, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    metric, provenance = load_official_metric(spacy_model=args.spacy_model)
    scored, summary = score_paired(records, metric)
    summary["metric_provenance"] = provenance
    summary["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scored_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in scored))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"variants": summary["variants"], "comparisons": {
        k: {field: value[field] for field in ("mean_difference_pp", "ci95_pp")}
        for k, value in summary["paired_comparisons"].items()}}, indent=2))


if __name__ == "__main__":
    main()

"""Add official ParaBank BARTScore fluency to public WebQA acc_approx.

This is a public-data approximate composite, not the hidden-test leaderboard
score. It is a supplemental development metric added after inspecting the
lexical metric; it must not silently replace the original development scores.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import string

import numpy as np
import torch
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer

from .. import paths
from .webqa_acc import VARIANTS


DEFAULT_CACHE = paths.BARTSCORE


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_for_bart(text):
    """Exact WebQA eval_1022.py normalize_text_for_bart definition."""
    table = str.maketrans(dict.fromkeys(string.punctuation))
    return " ".join(text.translate(table).split())


def normalized_bart(log_prediction, log_self):
    """WebQA's min(1, exp(score(ref,pred))/exp(score(ref,ref))), stably."""
    if not math.isfinite(log_prediction) or not math.isfinite(log_self):
        raise ValueError("BART log scores must be finite")
    return math.exp(min(0., log_prediction - log_self))


def configure_official_scorer(model, tokenizer, *, device, cache_dir=DEFAULT_CACHE):
    """Use the official score() function unchanged, with an explicit model."""
    cache_dir = Path(cache_dir)
    source = cache_dir / "official_source" / "bart_score.py"
    spec = importlib.util.spec_from_file_location("_official_parabank_bartscore", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.BARTScorer.__new__(module.BARTScorer)
    scorer.device = device
    scorer.max_length = 1024
    scorer.tokenizer = tokenizer
    scorer.model = model.eval().to(device)
    scorer.loss_fct = torch.nn.NLLLoss(reduction="none", ignore_index=model.config.pad_token_id)
    scorer.lsm = torch.nn.LogSoftmax(dim=1)
    return scorer


def load_parabank(*, device="cpu", cache_dir=DEFAULT_CACHE):
    """Strictly load the authors' complete ParaBank weights; no substitute."""
    cache_dir = Path(cache_dir)
    provenance = json.loads((cache_dir / "source_provenance.json").read_text())
    weights_record = json.loads((cache_dir / "weights_provenance.json").read_text())
    weights = cache_dir / "bart_score.pth"
    weights_record.update(bytes=weights.stat().st_size, sha256=sha256(weights))
    for item in provenance["files"]:
        path = cache_dir / item["file"]
        item.update(bytes=path.stat().st_size, sha256=sha256(path))
    config_path = cache_dir / "bart-large-cnn-config"
    config = BartConfig.from_pretrained(config_path, local_files_only=True)
    config._attn_implementation = "eager"
    tokenizer = BartTokenizer.from_pretrained(config_path, local_files_only=True)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = BartForConditionalGeneration(config)
    # The complete ParaBank state supersedes the CNN initialization in the
    # official example. Strict loading avoids downloading redundant CNN weights.
    load_result = model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    scorer = configure_official_scorer(model, tokenizer, device=device, cache_dir=cache_dir)
    metadata = {
        "bartscore_commit": provenance["bartscore_commit"],
        "bart_config_revision": provenance["bart_config_revision"],
        "weights": weights_record,
        "strict_state_dict_missing_keys": load_result.missing_keys,
        "strict_state_dict_unexpected_keys": load_result.unexpected_keys,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": device, "dtype": str(next(model.parameters()).dtype),
        "attention_implementation": "eager", "max_length": 1024,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "source_provenance": provenance,
    }
    return scorer, metadata


def add_fluency(records, scorer, *, batch_size=4, variants=VARIANTS, answer_policy=False):
    """Keep acc_approx unchanged and add reference-to-prediction fluency."""
    by_id = {}
    for row in records:
        qid, variant = row.get("qid"), row.get("variant")
        group = by_id.setdefault(qid, {})
        if variant in group:
            raise ValueError(f"Duplicate qid/variant: {qid}/{variant}")
        group[variant] = row
    if not by_id:
        raise ValueError("No scored predictions")
    qids = sorted(by_id)
    for qid in qids:
        group = by_id[qid]
        if set(group) != set(variants):
            raise ValueError(f"Incomplete pairing: {qid}")
        first = group[variants[0]]
        if not isinstance(first.get("references"), list) or len(first["references"]) != 1:
            raise ValueError("This WebQA release requires exactly one reference")
        for row in group.values():
            if (row["references"], row["category"]) != (first["references"], first["category"]):
                raise ValueError(f"Paired reference/category mismatch: {qid}")
    ordered = [by_id[qid][variant] for qid in qids for variant in variants]
    references = [normalize_for_bart(by_id[qid][variants[0]]["references"][0]) for qid in qids]
    sources = [normalize_for_bart(row["references"][0]) for row in ordered]
    targets = [normalize_for_bart(row["prediction"]) for row in ordered]
    try:
        self_scores = scorer.score(references, references, batch_size=batch_size)
        prediction_scores = scorer.score(sources, targets, batch_size=batch_size)
    except SystemExit as error:
        raise RuntimeError("The official BARTScore implementation exited during inference") from error
    if len(self_scores) != len(qids) or len(prediction_scores) != len(ordered):
        raise ValueError("BARTScore returned an incomplete result")
    self_by_id = dict(zip(qids, self_scores))
    result = []
    for row, log_score in zip(ordered, prediction_scores):
        self_score = self_by_id[row["qid"]]
        fluency = normalized_bart(log_score, self_score)
        result.append(dict(row, bart_reference_to_prediction_logscore=float(log_score),
                           bart_reference_self_logscore=float(self_score),
                           normalized_bart_score=fluency,
                           acc_approx_x_normalized_bart=float(row["acc_approx"] * fluency)))
    return result


def comparison_pairs(variants, baseline):
    variants = tuple(variants)
    if len(variants) < 2 or len(set(variants)) != len(variants):
        raise ValueError("Need at least two distinct variants")
    if baseline is None:
        if variants != VARIANTS:
            raise ValueError("A baseline is required for custom variants")
        return [("first-token", variant) for variant in VARIANTS[:2]]
    if baseline not in variants:
        raise ValueError("baseline must occur in variants")
    return [(variant, baseline) for variant in variants if variant != baseline]


def summarize_composite(records, *, variants=VARIANTS, baseline=None, stage="development",
                        resamples=10000, seed=20261002):
    pairs = comparison_pairs(variants, baseline)
    qids = sorted({row["qid"] for row in records})
    indexed = {(row["qid"], row["variant"]): row for row in records}
    values = np.asarray([[indexed[qid, variant]["acc_approx_x_normalized_bart"]
                          for variant in variants] for qid in qids])
    columns = {variant: index for index, variant in enumerate(variants)}
    differences = np.stack([values[:, columns[candidate]] - values[:, columns[reference]]
                            for candidate, reference in pairs], axis=1)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((resamples, len(pairs)))
    for start in range(0, resamples, 256):
        stop = min(start + 256, resamples)
        indices = rng.integers(0, len(qids), size=(stop - start, len(qids)))
        bootstrap[start:stop] = differences[indices].mean(1)
    comparisons = {}
    for col, (candidate, reference) in enumerate(pairs):
        mean = float(differences[:, col].mean())
        baseline_mean = float(values[:, columns[reference]].mean())
        comparisons[f"{candidate}_minus_{reference}"] = {
            "mean_difference": mean, "mean_difference_score_points": 100 * mean,
            "relative_improvement_percent": 100 * mean / baseline_mean if baseline_mean else None,
            "ci95": np.quantile(bootstrap[:, col], [.025, .975]).tolist(),
            "question_differences": dict(zip(qids, differences[:, col].tolist())),
        }
    return {
        "metric": "public WebQA approximate composite: acc_approx x normalized ParaBank BARTScore",
        "leaderboard_hidden_test_score": False, "heldout_confirmation": False,
        "stage": stage, "baseline": baseline, "heldout_independence_verified": False,
        "supplemental_metric_added_after_development_inspection": True,
        "score_scale": [0, 1], "question_count": len(qids), "question_ids": qids,
        "variants": {variant: {
            "mean_acc_approx": float(np.mean([indexed[qid, variant]["acc_approx"] for qid in qids])),
            "mean_normalized_bart_score": float(np.mean([indexed[qid, variant]["normalized_bart_score"] for qid in qids])),
            "mean_approx_composite": float(values[:, col].mean()),
        } for col, variant in enumerate(variants)},
        "paired_comparisons": comparisons,
        "bootstrap": {"unit": "question", "resamples": resamples, "seed": seed,
                      "interval": "percentile 95%", "numpy_quantile_method": "linear"},
    }




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="scored_predictions.jsonl from score_webqa")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    args = parser.parse_args()
    official_eval = paths.THIRD_PARTY / "webqa" / "eval_1022.py"
    raw = args.input.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    comparison_pairs(args.variants, args.baseline)
    answer_policy = tuple(args.variants) != VARIANTS
    scorer, provenance = load_parabank(device=args.device, cache_dir=args.cache_dir)
    scored = add_fluency(records, scorer, batch_size=args.batch_size,
                        variants=args.variants, answer_policy=answer_policy)
    report = summarize_composite(scored, variants=args.variants, baseline=args.baseline, stage=args.stage)
    report["comparison_mode"] = "answer-policy" if answer_policy else "eab-timing"
    report["bart_provenance"] = provenance
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    report["webqa_evaluator_sha256"] = sha256(official_eval)
    report["normalization"] = "min(1, exp(score(reference,prediction) - score(reference,reference))) after WebQA punctuation removal"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bart_scored_predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in scored))
    (args.output_dir / "bart_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(report["variants"], indent=2))


if __name__ == "__main__":
    main()

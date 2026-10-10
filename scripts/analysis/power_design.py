"""Detectable effect sizes and selection-on-noise for a paired comparison of two arms of a run.

    python -m scripts.analysis.power_design runs/<tag>/<dataset>.official_scores.jsonl \
        --baseline static --candidate eab:g1.2:ans2:pos:gate --metric vqa_accuracy

For a paired comparison the standard error of the mean difference over n questions is sd(delta)/sqrt(n);
the smallest effect detected with 80% power at a two-sided 5% level is about 2.8 * SE (mde_points). The
question-level differences are resampled to each planned sample size. best_of_K_noise is the expected excess of
the best of K resampled mean differences over the observed mean difference: the selection noise that the best of K
candidates picks up on n questions (an upper bound for correlated variants). Use it before tuning on a dev split of
a given size; paper_runs runs it on the dev base grid. --output also writes the JSON to a file.

analyse() and the size grids are unchanged from the legacy round-008 power_design.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SAMPLE_SIZES = (128, 256, 512, 1024, 2048, 4096)
CANDIDATE_COUNTS = (1, 8, 24)


def analyse(name, deltas, base_mean, rng):
    """Power summary of one paired difference; deltas are per-question candidate minus baseline scores."""
    sd, mean = float(deltas.std(ddof=1)), float(deltas.mean())
    rows = []
    for n in SAMPLE_SIZES:
        se = sd / np.sqrt(n)
        row = dict(n=n, standard_error=float(se), mde_points=float(2.8 * se),
                   mde_relative_percent=float(100 * 2.8 * se / base_mean) if base_mean else None)
        for k in CANDIDATE_COUNTS[1:]:
            draws = np.array([[deltas[rng.integers(0, len(deltas), n)].mean() for _ in range(k)] for _ in range(300)])
            best = (draws - mean).max(axis=1)
            row[f'best_of_{k}_noise_points'] = float(best.mean())
            row[f'best_of_{k}_noise_relative_percent'] = float(100 * best.mean() / base_mean) if base_mean else None
        rows.append(row)
    return dict(name=name, questions_in_source=len(deltas), mean_difference=mean, sd_difference=sd, baseline_mean=base_mean,
                share_questions_with_nonzero_difference=float((deltas != 0).mean()), by_sample_size=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=Path, help="<dataset>.official_scores.jsonl from omniaware.eval.score_official")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", type=Path, default=None, help="also write the JSON here")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()]
    base = np.array([r["scores"][args.baseline][args.metric] for r in rows], dtype=float)
    cand = np.array([r["scores"][args.candidate][args.metric] for r in rows], dtype=float)
    result = analyse(f"{args.candidate} - {args.baseline} ({args.metric})", cand - base, float(base.mean()),
                     np.random.default_rng(args.seed))
    text = json.dumps(result, indent=1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return result


if __name__ == "__main__":
    main()

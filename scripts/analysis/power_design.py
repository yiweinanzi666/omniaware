"""Detectable effect sizes for the planned train-256 development comparisons (CPU, saved per-question differences).

For a paired comparison of two arms the standard error of the mean difference over n questions is sd(delta)/sqrt(n).
The smallest effect detected with 80% power at a two-sided 5% level is about 2.8 * SE. The question-level difference
distributions come from saved comparisons of the same kind as the planned ones (an EAB arm against its matched static
arm, and answer-level interface changes). Each distribution is resampled to the planned sample sizes; nothing here is a
result about a candidate and nothing selects a rule.

Two further quantities, both from the same differences:
  * the expected best-of-K estimate when K candidates with true effect zero are compared on n questions (selection on
    noise): K independent size-n resamples of the centered differences, maximum taken. Variants of one mechanism have
    positively correlated differences, so this overstates the noise of a real variant family; it is an upper bound.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COMBO, LEXICAL = 'acc_approx_x_normalized_bart', 'acc_approx'
SAMPLE_SIZES = (128, 256, 512, 1024, 2048, 4096)
CANDIDATE_COUNTS = (1, 8, 24)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def formal_differences(path, left, right):
    deltas = []
    for line in (ROOT / path).open():
        row = json.loads(line)
        deltas.append(row['scores'][right] - row['scores'][left])
    return np.array(deltas), None


def scored_differences(path, arm_field, baseline, candidate, field, keep=None):
    arms = {}
    for line in (ROOT / path).open():
        row = json.loads(line)
        if keep and not keep(row):
            continue
        arms.setdefault(row[arm_field], {}).setdefault(row['qid'], []).append(row[field])
    qids = sorted(arms[baseline])
    base = np.array([np.mean(arms[baseline][q]) for q in qids])
    cand = np.array([np.mean(arms[candidate][q]) for q in qids])
    return 100 * (cand - base), 100 * base.mean()


def analyse(name, deltas, base_mean, rng):
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
    rng = np.random.default_rng(17)
    sources = {}
    out = []
    p = 'runs/okvqa_formal_fusion_codex_20261005/output/scored_questions.jsonl'
    sources[p] = sha(ROOT / p)
    for left, right in (('trigger-static', 'trigger-full'), ('trigger-static+guard', 'trigger-full+guard')):
        deltas, _ = formal_differences(p, left, right)
        base = np.mean([json.loads(l)['scores'][left] for l in (ROOT / p).open()])
        out.append(analyse(f'OK-VQA official accuracy: {right} - {left} (val5046 differences, resampled)', deltas, float(base), rng))
    p = 'runs/eab_positive_confirmation/scored/scored_predictions.jsonl'
    sources[p] = sha(ROOT / p)
    for field, label in ((COMBO, 'Combo'), (LEXICAL, 'Lexical')):
        for baseline, candidate in (('static-five', 'bge-centered-g1.2'), ('static-five', 'target-binding-static')):
            deltas, base = scored_differences(p, 'variant', baseline, candidate, field)
            out.append(analyse(f'WebQA image {label}: {candidate} - {baseline} (val256 differences, resampled)', deltas, base, rng))
    p = 'runs/eab_relative_evidence_codex_20261004/output/scored/scored_predictions.jsonl'
    sources[p] = sha(ROOT / p)
    for field, label in ((COMBO, 'Combo'), (LEXICAL, 'Lexical')):
        deltas, base = scored_differences(p, 'method', 'static', 'relative-evidence-eab', field, keep=lambda r: r['order_kind'] == 'random')
        out.append(analyse(f'WebQA text {label}: relative-evidence-eab - static (T17 128 x 12 orders, resampled)', deltas, base, rng))
    result = dict(rule='MDE = 2.8 * sd(delta) / sqrt(n) (80% power, two-sided 5%); best-of-K noise by resampling', sources=sources, analyses=out)
    (HERE / 'power_design.json').write_text(json.dumps(result, indent=1) + '\n')
    for a in out:
        print(f"{a['name']}\n   mean {a['mean_difference']:+.4f} sd {a['sd_difference']:.4f} nonzero {a['share_questions_with_nonzero_difference']:.1%} baseline {a['baseline_mean']:.3f}")
        for r in a['by_sample_size']:
            if r['n'] in (256, 1024, 4096):
                print(f"   n={r['n']:5d}  MDE {r['mde_points']:.4f} ({r['mde_relative_percent']:.1f}% rel) | best-of-24 noise {r['best_of_24_noise_points']:+.4f} "
                      f"({r['best_of_24_noise_relative_percent']:+.1f}%)")


if __name__ == '__main__':
    main()

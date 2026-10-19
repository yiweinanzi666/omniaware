"""Evidence-order experiments: fixed permutation schedule, chunk replication, order statistics.

Copied verbatim from the legacy T17 order experiment (see MIGRATION_SOURCES.json, objects),
except that replication_counts takes gamma explicitly instead of a module constant.
"""
import itertools
import math
import random

import numpy as np

PERMUTATION_SEEDS = (17, 23, 29)


def permutation_schedule():
    anchors = [tuple(range(5)), tuple(reversed(range(5)))]
    available = [p for p in itertools.permutations(range(5)) if p not in anchors]
    schedule = [dict(order_id=name, kind='anchor', seed=None, permutation=list(p))
        for name, p in zip(('sorted', 'reverse'), anchors, strict=True)]
    for seed in PERMUTATION_SEEDS:
        chosen = random.Random(seed).sample(available, 4)
        for index, p in enumerate(chosen):
            schedule.append(dict(order_id=f's{seed}-p{index+1}', kind='random', seed=seed, permutation=list(p)))
        available = [p for p in available if p not in chosen]
    return schedule


def replication_counts(relevances, uncertainty, gate, gamma=1.2):
    """Adjacent whole-chunk copies max(1, round(exp(gamma * U * r))) when the gate is open, else 1.

    For every gamma*U*r >= 0 the log of the copy count differs from the bias gamma*U*r by at most log(1.5),
    whereas ceiling gave two copies to any positive bias. The copies are not truncated: the replication prompt
    grows with them, and its length is recorded per question."""
    return [max(1, round(math.exp(gamma * uncertainty * value))) if gate else 1 for value in relevances]


def order_std_statistics(baseline, candidate, *, resamples=10000, seed=20261002):
    """Bootstrap questions, then recompute permutation dataset means and std."""
    baseline, candidate = np.asarray(baseline, dtype=float), np.asarray(candidate, dtype=float)
    if baseline.ndim != 2 or baseline.shape != candidate.shape or not all(baseline.shape):
        raise ValueError("Need equal, nonempty question-by-permutation matrices")
    if resamples < 1:
        raise ValueError("Need positive bootstrap resamples")
    base_std = float(baseline.mean(axis=0).std(ddof=0))
    candidate_std = float(candidate.mean(axis=0).std(ddof=0))
    boot = np.empty(resamples)
    rng = np.random.default_rng(seed)
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        questions = rng.integers(0, len(baseline), size=(stop-start, len(baseline)))
        boot[start:stop] = (candidate[questions].mean(axis=1).std(axis=1, ddof=0)
                           - baseline[questions].mean(axis=1).std(axis=1, ddof=0))
    interval = np.quantile(boot, [.025, .975])
    return dict(statistic="std(ddof=0) of permutation-specific full-dataset means",
                baseline_std=base_std, candidate_std=candidate_std,
                difference=candidate_std-base_std, difference_score_points=100*(candidate_std-base_std),
                ci95=interval.tolist(), ci95_score_points=(100*interval).tolist())

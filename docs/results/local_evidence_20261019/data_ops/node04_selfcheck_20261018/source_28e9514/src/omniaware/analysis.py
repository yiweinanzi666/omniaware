"""Per-question analysis quantities shared by the runner and the analysis scripts."""
from __future__ import annotations

import math

import numpy as np

ARD_TEMPERATURE = 0.1   # paper Eq. (ard): T_r


def _softmax(values, temperature):
    v = np.asarray(values, dtype=float) / temperature
    v = np.exp(v - v.max())
    return v / v.sum()


def jsd(p, q, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in nats between two distributions."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(a * (np.log(a + eps) - np.log(b + eps))))  # noqa: E731
    return (kl(p, m) + kl(q, m)) / 2


def ard(chunk_mass: dict[str, float], relevance: dict[str, float], temperature: float = ARD_TEMPERATURE) -> float:
    """Attention-relevance divergence: JSD(normalised chunk attention mass || softmax(r / T_r))."""
    cids = [c for c in relevance if c in chunk_mass]
    mass = [max(chunk_mass[c], 0.0) for c in cids]
    if not cids or sum(mass) <= 0:
        return math.nan
    return jsd(mass, _softmax([relevance[c] for c in cids], temperature))


def aurc(uncertainty, error) -> float:
    """Area under the risk-coverage curve: answer the most confident (lowest-uncertainty) first."""
    order = np.argsort(np.asarray(uncertainty, dtype=float), kind="stable")
    err = np.asarray(error, dtype=float)[order]
    risks = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risks.mean())


def spearman(x, y) -> float:
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort(kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for value in np.unique(v):          # average ranks for ties
            idx = v == value
            r[idx] = r[idx].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])

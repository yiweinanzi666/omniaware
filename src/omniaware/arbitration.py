"""Action selector for Omni-Aware v2.

The input is per-question action records (answers and first-token signals) plus official scores on a
calibration set. The output is a JSON-serialisable decision that maps one question's actions to one action.

Two families are fitted on calibration data:

- v2-Rule (no training): keep direct when its confidence clears a threshold, otherwise use an evidence action.
- v2-Cal: one ridge regression per non-reference action predicts its score gain over the strongest single
  action; the largest predicted gain is taken only when it exceeds a threshold.

Both families pass through the same risk control: a one-sided paired t test on the mean score gap to the
strongest single action a*, walked along a fixed sequence that starts at the policy equal to a*.
Only numpy and the standard library are used; this module reads no files.
"""
from __future__ import annotations

import math
import string
import unicodedata
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .actions import DIRECT, EVIDENCE_ACTIONS, V2_ACTIONS

CONFIDENCE_KINDS = ("margin", "neg_entropy")
QUANTILE_STEPS = 20                 # threshold grids are 0%, 5%, ..., 100% quantiles
RIDGE_ALPHA = 1.0                   # on standardised features; the intercept is not penalised
CAL_FEATURES = ("conf_direct", "conf_star", "conf_gap", "token_f1_to_star", "kl_to_direct", "retriever_top1")


# ---------------------------------------------------------------- JSON and text helpers

def encode_json_float(value: float) -> float | str:
    """Write +inf and -inf as the strings "inf" and "-inf" so decisions are strict JSON."""
    value = float(value)
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def decode_json_float(value: float | str) -> float:
    """Read back a value written by ``encode_json_float``."""
    if isinstance(value, str):
        if value not in ("inf", "-inf"):
            raise ValueError(f"not an encoded float: {value!r}")
        return float(value)
    return float(value)


def normalize_answer(text: str) -> list[str]:
    """Lower-case, drop punctuation, and split on whitespace."""
    kept = "".join(
        ch for ch in text.lower()
        if ch not in string.punctuation and not unicodedata.category(ch).startswith("P")
    )
    return kept.split()


def token_f1(a: str, b: str) -> float:
    """Multiset token F1 over ``normalize_answer`` tokens. Two empty answers agree (1.0); one empty gives 0.0."""
    left, right = normalize_answer(a), normalize_answer(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2.0 * precision * recall / (precision + recall)


def _confidence(record: dict, kind: str) -> float:
    """Higher is more confident. ``margin`` is the first-token top-1 minus top-2 margin; ``neg_entropy`` is -entropy.

    A non-finite value is refused: a NaN or infinite confidence would decide the policy silently.
    """
    token = record["first_token"]
    if kind == "margin":
        raw, sign = token["margin"], 1.0
    elif kind == "neg_entropy":
        raw, sign = token["entropy"], -1.0
    else:
        raise ValueError(f"confidence must be one of {CONFIDENCE_KINDS}, not {kind!r}")
    value = math.nan if raw is None else sign * float(raw)      # a non-finite value is stored as null
    if not math.isfinite(value):
        raise ValueError(f"the {kind} confidence of an action is not finite; the decision cannot be applied")
    return value


def _available(actions: dict, name: str) -> bool:
    return name in actions and bool(actions[name].get("available", True))


def _nan_if_none(value) -> float:
    return math.nan if value is None else float(value)


# ---------------------------------------------------------------- strongest actions

def _mean_scores(scores_by_action: dict[str, Sequence[float]]) -> dict[str, float]:
    means, lengths = {}, set()
    for name, values in scores_by_action.items():
        if name not in V2_ACTIONS:
            raise ValueError(f"unknown action {name!r}")
        arr = np.asarray(values, dtype=float)
        lengths.add(arr.size)
        means[name] = float(arr.mean()) if arr.size else math.nan
    if not means or len(lengths) != 1 or 0 in lengths:
        raise ValueError("scores_by_action needs one non-empty score sequence per action, all of one length")
    return means


def _ranked(means: dict[str, float]) -> list[str]:
    """Highest mean first. Ties follow V2_ACTIONS order."""
    return sorted(means, key=lambda name: (-means[name], V2_ACTIONS.index(name)))


def best_single_action(scores_by_action: dict[str, Sequence[float]]) -> str:
    """The action with the highest mean score (a*). Ties follow V2_ACTIONS order."""
    return _ranked(_mean_scores(scores_by_action))[0]


def best_evidence_action(scores_by_action: dict[str, Sequence[float]]) -> str:
    """The evidence action with the highest mean score (e*). Ties follow V2_ACTIONS order."""
    means = {name: m for name, m in _mean_scores(scores_by_action).items() if name in EVIDENCE_ACTIONS}
    if not means:
        raise ValueError("best_evidence_action needs at least one evidence action")
    return _ranked(means)[0]


# ---------------------------------------------------------------- applying a decision

def apply_decision(decision: dict, actions: dict) -> dict:
    """Return ``{"action": name, "answer": text}`` for one question.

    Reads only the decision and the question's ``actions``. Scores and ``gold_in_topk`` are never read,
    and an unavailable action is never returned.
    """
    kind = decision["kind"]
    if kind == "rule":
        name = _rule_action(decision, actions)
    elif kind == "cal":
        name = _cal_action(decision, actions)
    elif kind == "single":
        name = reference_action(decision, actions)
    else:
        raise ValueError(f"unknown decision kind {kind!r}")
    return {"action": name, "answer": actions[name]["answer"]}


def policy_scores(decision: dict, rows: list[dict]) -> list[float]:
    """Official score of the action each calibration or evaluation row receives under ``decision``."""
    out = []
    for row in rows:
        name = apply_decision(decision, row["actions"])["action"]
        out.append(float(row["scores"][name]))
    return out


def reference_action(decision: dict, actions: dict) -> str:
    """The reference action of one question: a* when it is available, otherwise the first available action of the
    frozen ``action_ranking`` (highest calibration mean first). This is the action the fallback decision gives.

    Raises ValueError when no action of the question is available.
    """
    for name in decision["action_ranking"]:
        if _available(actions, name):
            return name
    raise ValueError("no available action on this question")


def _rule_action(decision: dict, actions: dict) -> str:
    kind = decision["confidence"]
    if _available(actions, DIRECT) and _confidence(actions[DIRECT], kind) >= decode_json_float(decision["tau"]):
        return DIRECT
    evidence = [name for name in decision["evidence_ranking"] if _available(actions, name)]
    if not evidence:
        return reference_action(decision, actions)
    if decision["evidence_choice"] == "best":
        return evidence[0]
    chosen = evidence[0]
    for name in evidence[1:]:
        if _confidence(actions[name], kind) > _confidence(actions[chosen], kind):
            chosen = name
    return chosen


def _cal_action(decision: dict, actions: dict) -> str:
    """The predicted-gain policy. Its features are relative to a* and to direct, so when either is unavailable on
    the question no gain can be predicted and the reference action is used."""
    star = decision["best_single"]
    if not (_available(actions, DIRECT) and _available(actions, star)):
        return reference_action(decision, actions)
    best = _best_predicted_gain(decision["models"], actions, star, decision["confidence"])
    if best is not None and best[1] > decode_json_float(decision["lam"]):
        return best[0]
    return reference_action(decision, actions)


def decision_fields(decision: dict) -> dict:
    """The fields that define a frozen decision, for reports. Cal models list their kept and dropped features."""
    keys = ("kind", "fallback", "action", "best_single", "confidence", "delta", "epsilon", "n_calib",
            "n_accepted", "calib_utility", "evidence_choice", "tau", "lam", "seed", "n_fit", "n_risk", "n_params")
    out = {key: decision[key] for key in keys if key in decision}
    if decision.get("models"):
        out["models"] = {name: {"features": m["features"], "dropped": m["dropped"], "n_params": m["n_params"]}
                         for name, m in decision["models"].items()}
    return out


# ---------------------------------------------------------------- v2-Rule

def _rule_candidates(common: dict, choice: str, thresholds: list[float]) -> list[dict]:
    return [
        {**common, "kind": "rule", "fallback": False, "evidence_choice": choice, "tau": encode_json_float(tau)}
        for tau in thresholds
    ]


def select_rule(calib_rows: list[dict], delta: float = 0.1, epsilon: float = 1.0,
                confidence: str = "margin") -> dict:
    """Choose a v2-Rule policy by risk control on the calibration rows.

    The candidates are ``evidence_choice`` in {"best", "confident"} crossed with thresholds tau on the direct
    confidence. Each family is a fixed sequence at level delta / 2, so the union of the two has level delta.
    Thresholds are quantiles of direct confidence on the calibration rows (labels are not used). Both families
    start from the same end and move away from it: tau = -inf (always direct) when a* is direct, and tau = +inf
    (always evidence) when a* is an evidence action. For "best" the starting policy equals a* exactly, because
    e* = a* when a* is evidence. For "confident" the starting policy picks the most confident evidence action,
    which is not a* in general; if that start does not pass the test, the whole family is rejected.
    The accepted candidates from both families are pooled, and the one with the largest mean gap wins;
    ties go to the earlier candidate, with "best" before "confident". If nothing is accepted or the best
    mean gap is <= 0, the decision is the fallback "always a*".

    Guarantee, stated as it holds: if questions are i.i.d. and the one-sided paired t test is asymptotically
    valid, then with probability at least 1 - delta the chosen policy satisfies
    E[score_policy] >= E[score_a*] - epsilon. a* is picked on the same calibration rows, which adds mild
    optimism. The t statistic uses a normal approximation for every n; for n < 30 this is less conservative
    than the t distribution would be.
    """
    _check_settings(delta, epsilon, confidence)
    _, _, ranking = _calibration_setup(calib_rows)
    star = ranking[0]
    common = _common_fields(ranking, delta, epsilon, confidence)
    base = _base_scores(calib_rows, star)
    direct_conf = [_confidence(row["actions"][DIRECT], confidence) for row in calib_rows]
    thresholds = _unique_in_order([-math.inf, *_quantile_grid(direct_conf), math.inf])
    if star != DIRECT:
        thresholds.reverse()
    level = delta / 2.0
    pool = []
    for choice in ("best", "confident"):
        pool.extend(_accept_prefix(_rule_candidates(common, choice, thresholds), calib_rows, base, epsilon, level))
    return _finish(pool, common, len(calib_rows))


# ---------------------------------------------------------------- v2-Cal

def _cal_features(name: str, actions: dict, star: str, kind: str) -> dict[str, float]:
    """Features for one action on one question. Missing values are NaN and are never filled."""
    rec, star_rec = actions[name], actions[star]
    retriever = rec.get("retriever") or {}
    return {
        "conf_direct": _confidence(actions[DIRECT], kind),
        "conf_star": _confidence(star_rec, kind),
        "conf_gap": _confidence(rec, kind) - _confidence(star_rec, kind),
        "token_f1_to_star": token_f1(rec["answer"], star_rec["answer"]),
        "kl_to_direct": _nan_if_none(rec.get("kl_to_direct")),
        "retriever_top1": _nan_if_none(retriever.get("top1_score")),
    }


def _fit_model(name: str, feats: list[dict[str, float]], targets: np.ndarray) -> dict:
    """Ridge regression on standardised features. Returns weights with the intercept first, in raw feature units.

    A feature missing on every row, or constant on every row, is dropped and recorded. A feature missing on
    some rows but not others is an error.
    """
    x = np.array([[f[c] for c in CAL_FEATURES] for f in feats], dtype=float)
    kept, dropped = [], []
    for j, feature in enumerate(CAL_FEATURES):
        column = x[:, j]
        missing = np.isnan(column)
        if missing.all():
            dropped.append(feature)
        elif missing.any():
            raise ValueError(f"{name}: feature {feature!r} is missing on some calibration questions but not others; "
                             "refusing to impute it")
        elif np.ptp(column) == 0.0:
            dropped.append(feature)
        else:
            kept.append(j)
    if not kept:
        return {"features": [], "weights": [float(targets.mean())], "dropped": dropped, "n_params": 1}
    xk = x[:, kept]
    mu, sd = xk.mean(axis=0), xk.std(axis=0)
    z = (xk - mu) / sd
    y_centred = targets - targets.mean()
    coef_std = np.linalg.solve(z.T @ z + RIDGE_ALPHA * np.eye(len(kept)), z.T @ y_centred)
    coef = coef_std / sd
    intercept = float(targets.mean() - mu @ coef)
    return {
        "features": [CAL_FEATURES[j] for j in kept],
        "weights": [intercept] + [float(c) for c in coef],
        "dropped": dropped,
        "n_params": 1 + len(kept),
    }


def _fit_cal_models(fit_rows: list[dict], star: str, names: list[str], kind: str) -> dict[str, dict]:
    """One model per non-reference action, fitted on the fit half. Rows where that action is unavailable are skipped."""
    models = {}
    for name in names:
        if name == star:
            continue
        feats, targets = [], []
        for row in fit_rows:
            if not _available(row["actions"], name):
                continue
            feats.append(_cal_features(name, row["actions"], star, kind))
            targets.append(float(row["scores"][name]) - float(row["scores"][star]))
        if feats:
            models[name] = _fit_model(name, feats, np.asarray(targets, dtype=float))
    return models


def _predict_gain(model: dict, feats: dict[str, float]) -> float:
    values = []
    for feature in model["features"]:
        value = feats[feature]
        if math.isnan(value):
            raise ValueError(f"feature {feature!r} is missing on this question; the fitted model cannot skip it")
        values.append(value)
    weights = model["weights"]
    return float(weights[0] + np.dot(np.asarray(weights[1:], dtype=float), np.asarray(values, dtype=float)))


def _best_predicted_gain(models: dict[str, dict], actions: dict, star: str, kind: str) -> tuple[str, float] | None:
    """The available action with the largest predicted gain over a*. Ties follow the insertion order of ``models``."""
    best = None
    for name, model in models.items():
        if not _available(actions, name):
            continue
        gain = _predict_gain(model, _cal_features(name, actions, star, kind))
        if best is None or gain > best[1]:
            best = (name, gain)
    return best


def _split_halves(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded split of calibration rows into a fit half and a disjoint risk-control half."""
    if len(rows) < 4:
        raise ValueError("v2-Cal needs at least 4 calibration questions to split into two halves")
    order = np.random.default_rng(seed).permutation(len(rows))
    half = len(rows) // 2
    return [rows[i] for i in order[:half]], [rows[i] for i in order[half:]]


def select_cal(calib_rows: list[dict], delta: float = 0.1, epsilon: float = 1.0, seed: int = 0,
               confidence: str = "margin") -> dict:
    """Choose a v2-Cal policy: fit per-action gain models on one half, risk-control the threshold on the other.

    The policy takes the action with the largest predicted gain over a* when that gain exceeds lam, and a*
    otherwise. The fixed sequence over lam runs from +inf (always a*) downwards, at level delta. Acceptance,
    selection, and fallback follow ``select_rule``. The guarantee and its caveats are the same, and the
    risk-control half is disjoint from the fit half.
    """
    _check_settings(delta, epsilon, confidence)
    names, _, ranking = _calibration_setup(calib_rows)
    star = ranking[0]
    fit_rows, risk_rows = _split_halves(calib_rows, seed)
    common = {**_common_fields(ranking, delta, epsilon, confidence),
              "seed": int(seed), "n_fit": len(fit_rows), "n_risk": len(risk_rows)}
    models = _fit_cal_models(fit_rows, star, names, confidence)
    gains = []
    for row in fit_rows:
        best = _best_predicted_gain(models, row["actions"], star, confidence) if models else None
        if best is not None:
            gains.append(best[1])
    if not gains:
        return _fallback(common, len(calib_rows), 0)
    lams = _unique_in_order([math.inf, *reversed(_quantile_grid(gains)), -math.inf])
    total_params = sum(model["n_params"] for model in models.values())
    candidates = [
        {**common, "kind": "cal", "fallback": False, "lam": encode_json_float(lam),
         "models": models, "n_params": total_params}
        for lam in lams
    ]
    pool = _accept_prefix(candidates, risk_rows, _base_scores(risk_rows, star), epsilon, delta)
    return _finish(pool, common, len(calib_rows))


# ---------------------------------------------------------------- risk control shared by both families

def _check_settings(delta: float, epsilon: float, kind: str) -> None:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1")
    if not (math.isfinite(epsilon) and epsilon >= 0.0):
        raise ValueError("epsilon must be a finite non-negative number")
    if kind not in CONFIDENCE_KINDS:
        raise ValueError(f"confidence must be one of {CONFIDENCE_KINDS}, not {kind!r}")


def _calibration_setup(rows: list[dict]) -> tuple[list[str], dict[str, np.ndarray], list[str]]:
    """Validate calibration rows. Returns the action names, their score arrays, and the mean ranking.

    An action may be unavailable on some questions (an empty modality pool); it then has no score there and
    its mean is taken over the questions where it is available. Only an action available on every question
    can be a*, because the paired gap needs its score on every row. The ranking lists those actions first.
    """
    if not rows:
        raise ValueError("calibration rows are empty")
    names = [name for name in V2_ACTIONS if any(name in row["scores"] for row in rows)]
    by_action, always = {}, []
    for name in names:
        values = []
        for row in rows:
            if name not in row["actions"]:
                raise ValueError(f"question {row.get('qid')!r}: {name!r} is missing from actions")
            if not _available(row["actions"], name):
                continue
            score = row["scores"].get(name)
            if score is None or not math.isfinite(float(score)):
                raise ValueError(f"question {row.get('qid')!r}: score for available action {name!r} is missing "
                                 "or not finite")
            values.append(float(score))
        if not values:
            continue
        by_action[name] = np.asarray(values, dtype=float)
        if len(values) == len(rows):
            always.append(name)
    if DIRECT not in always:
        raise ValueError("direct must be available and scored on every calibration question")
    means = {name: float(arr.mean()) for name, arr in by_action.items()}
    ranking = _ranked({name: means[name] for name in always}) + _ranked(
        {name: means[name] for name in means if name not in always})
    return list(by_action), by_action, ranking


def _common_fields(ranking: list[str], delta: float, epsilon: float, kind: str) -> dict:
    return {
        "best_single": ranking[0],
        "action_ranking": list(ranking),
        "evidence_ranking": [name for name in ranking if name in EVIDENCE_ACTIONS],
        "confidence": kind,
        "delta": float(delta),
        "epsilon": float(epsilon),
    }


def _fallback(common: dict, n_calib: int, n_accepted: int) -> dict:
    return {**common, "kind": "single", "fallback": True, "action": common["best_single"],
            "calib_utility": None, "n_calib": n_calib, "n_accepted": n_accepted}


def _finish(pool: list[tuple[float, dict]], common: dict, n_calib: int) -> dict:
    """Pick the largest mean gap in the pool. The pool is in fixed-sequence order, so ties keep the earlier one."""
    if not pool:
        return _fallback(common, n_calib, 0)
    best_u, best = pool[0]
    for utility, decision in pool[1:]:
        if utility > best_u:
            best_u, best = utility, decision
    if best_u <= 0.0:
        return _fallback(common, n_calib, len(pool))
    return {**best, "calib_utility": best_u, "n_calib": n_calib, "n_accepted": len(pool)}


def _accept_prefix(candidates: list[dict], rows: list[dict], base: np.ndarray, epsilon: float,
                   level: float) -> list[tuple[float, dict]]:
    """Walk the fixed sequence and stop at the first candidate whose test fails. Returns (mean gap, candidate)."""
    accepted = []
    for candidate in candidates:
        gaps = np.asarray(policy_scores(candidate, rows), dtype=float) - base
        if _one_sided_p(gaps, epsilon) > level:
            break
        accepted.append((float(gaps.mean()), candidate))
    return accepted


def _one_sided_p(gap: np.ndarray, epsilon: float) -> float:
    """p-value for H0: E[gap] < -epsilon, from the one-sided paired t statistic.

    The t tail is replaced by the standard normal tail for every n (math.erfc, no scipy). For n < 30 this is
    less conservative than the t distribution. A constant gap has zero variance: p = 0 if its mean is at
    least -epsilon, otherwise p = 1. A policy identical to a* therefore passes at epsilon = 0.
    """
    gap = np.asarray(gap, dtype=float)
    n = gap.size
    if n < 2:
        raise ValueError("the paired test needs at least two questions")
    mean = float(gap.mean())
    if np.ptp(gap) == 0.0:
        return 0.0 if mean >= -epsilon else 1.0
    t = (mean + epsilon) / (float(gap.std(ddof=1)) / math.sqrt(n))
    return 0.5 * math.erfc(t / math.sqrt(2.0))


def _base_scores(rows: list[dict], star: str) -> np.ndarray:
    return np.asarray([float(row["scores"][star]) for row in rows], dtype=float)


def _quantile_grid(values) -> list[float]:
    """Quantiles at 0%, 5%, ..., 100% in ascending order, duplicates removed."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("threshold grid needs a non-empty, finite set of values")
    return _unique_in_order([float(q) for q in np.quantile(arr, np.linspace(0.0, 1.0, QUANTILE_STEPS + 1))])


def _unique_in_order(values: list[float]) -> list[float]:
    """Drop adjacent duplicates. The grids above are monotone, so this removes every duplicate."""
    out: list[float] = []
    for value in values:
        if not out or value != out[-1]:
            out.append(value)
    return out

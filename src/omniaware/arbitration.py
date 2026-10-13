"""Offline Omni-Aware v2 selector.

Reads per-action records (answers and signals). Does not decode, prompt, or build those records.
``retriever.gold_in_topk`` is never read. Thresholds are chosen by utility under a fixed-sequence
binomial test, not by maximum accuracy.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

V2_ACTIONS = ("direct", "static_text", "static_image", "static_mixed", "rerank_static")

# weights[0] is the intercept. action_features() returns the other six, in this order.
FEATURE_NAMES = (
    "intercept",
    "margin",
    "entropy_norm",
    "seq_logprob_per_token",
    "kl_to_direct",
    "token_f1_to_direct",
    "retriever_top1_score",
)
N_PARAMS = 7

_TAU1_GRID = (-1e9, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9)
_THETA_GRID = (1.0, 0.5, 0.0)


def token_f1(a: str, b: str) -> float:
    """Whitespace-token multiset F1 in [0, 1]. Empty/empty is 1. Empty vs non-empty is 0."""
    left, right = a.split(), b.split()
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


def _direct_is_confident(direct: dict, tau1: float, confidence: str) -> bool:
    if confidence == "margin":
        return float(direct["first_token"]["margin"]) >= float(tau1)
    if confidence == "entropy":
        return float(direct["first_token"]["entropy_norm"]) <= float(tau1)
    raise ValueError(f"confidence must be 'margin' or 'entropy', not {confidence!r}")


def _agrees(name: str, actions: dict, theta: float) -> bool:
    """True when this answer has token F1 >= theta with a different action, including direct."""
    answer = actions[name]["answer"]
    for other in V2_ACTIONS:
        if other == name or other not in actions:
            continue
        if token_f1(answer, actions[other]["answer"]) >= float(theta):
            return True
    return False


def _evidence_rank(name: str, actions: dict, confidence: str) -> tuple:
    """Lower is better. Margin: higher margin first. Entropy: lower entropy_norm first. Ties: V2_ACTIONS order."""
    order = V2_ACTIONS.index(name)
    if confidence == "margin":
        return (-float(actions[name]["first_token"]["margin"]), order)
    return (float(actions[name]["first_token"]["entropy_norm"]), order)


def select_v2_rule(actions: dict, tau1: float, theta: float, confidence: str = "margin") -> dict:
    """v2-Rule. Confident direct is kept. Otherwise the agreeing evidence action with the best confidence.

    ``confidence='margin'``: direct is confident when ``first_token.margin >= tau1``. Evidence actions are
    ranked by higher margin. ``confidence='entropy'``: direct is confident when ``entropy_norm <= tau1``
    (lower entropy is more confident). Evidence actions are ranked by lower ``entropy_norm``.

    An evidence action is kept only when its answer has token F1 >= theta with at least one different
    action (any action, including direct). If none qualify, the choice is direct. ``used_evidence`` is
    true only when the returned action is not direct. ``gold_in_topk`` is not read.
    """
    direct = actions["direct"]
    if _direct_is_confident(direct, tau1, confidence):
        return {"action": "direct", "answer": direct["answer"], "used_evidence": False}
    keepers = [name for name in V2_ACTIONS if name != "direct" and name in actions and _agrees(name, actions, theta)]
    if not keepers:
        return {"action": "direct", "answer": direct["answer"], "used_evidence": False}
    best = min(keepers, key=lambda name: _evidence_rank(name, actions, confidence))
    return {"action": best, "answer": actions[best]["answer"], "used_evidence": True}


def action_features(name: str, rec: dict) -> np.ndarray:
    """Length-6 feature vector. The caller adds the intercept.

    For a non-direct action, ``rec['direct_answer']`` must hold the direct answer (the caller copies it
    in; this function does not look at other actions). Feature 5 is 1 for direct.
    """
    first = rec["first_token"]
    gen_len = rec.get("gen_len", 1)
    kl = rec.get("kl_to_direct")
    retriever = rec.get("retriever") or {}
    top1 = retriever.get("top1_score")
    if name == "direct":
        agree = 1.0
    else:
        if "direct_answer" not in rec:
            raise ValueError("action_features needs rec['direct_answer'] for a non-direct action")
        agree = token_f1(rec["answer"], rec["direct_answer"])
    return np.asarray([
        float(first["margin"]),
        float(first["entropy_norm"]),
        float(rec["seq_logprob"]) / max(float(gen_len), 1.0),
        0.0 if kl is None else float(kl),
        float(agree),
        0.0 if top1 is None else float(top1),
    ], dtype=float)


def _with_direct_answer(name: str, actions: dict) -> dict:
    rec = actions[name]
    if name == "direct":
        return rec
    return {**rec, "direct_answer": actions["direct"]["answer"]}


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic(features: np.ndarray, labels: np.ndarray, l2: float = 1e-2, max_iter: int = 40) -> np.ndarray:
    """IRLS / Newton logistic regression. Returns 7 weights in the original feature space, intercept first.

    Features are standardised only inside the solver. The returned weights already undo that, so prediction
    is ``sigmoid(w[0] + w[1:] · features)``. A fixed L2 penalty keeps separable and tiny samples finite; it
    is not an extra learned parameter.
    """
    raw = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != N_PARAMS - 1:
        raise ValueError(f"expected (n, {N_PARAMS - 1}) features, got {raw.shape}")
    if raw.shape[0] != y.shape[0] or raw.shape[0] == 0:
        raise ValueError("features and labels must be non-empty and the same length")
    center = raw.mean(axis=0)
    scale = raw.std(axis=0)
    varying = scale >= 1e-8
    scale = np.where(varying, scale, 1.0)
    standard = (raw - center) / scale
    standard[:, ~varying] = 0.0
    design = np.column_stack([np.ones(len(standard)), standard])
    weights = np.zeros(N_PARAMS, dtype=float)
    penalty = np.eye(N_PARAMS, dtype=float) * l2
    for _ in range(max_iter):
        prob = np.clip(_sigmoid(design @ weights), 1e-6, 1.0 - 1e-6)
        grad = design.T @ (prob - y) + l2 * weights
        hess = design.T @ (design * (prob * (1.0 - prob))[:, None]) + penalty
        step = np.linalg.solve(hess, grad)
        updated = weights - step
        if not np.isfinite(updated).all():
            break
        weights = updated
        if float(np.linalg.norm(step)) <= 1e-8:
            break
    slope = np.zeros(N_PARAMS - 1, dtype=float)
    slope[varying] = weights[1:][varying] / scale[varying]
    intercept = float(weights[0] - np.sum(weights[1:][varying] * center[varying] / scale[varying]))
    mapped = np.concatenate([[intercept], slope])
    if not np.isfinite(mapped).all():
        return np.zeros(N_PARAMS, dtype=float)
    return mapped


def fit_v2_cal(rows: list[dict]) -> dict:
    """One shared logistic regression P(correct | features) on calibration rows.

    Each row is ``{"actions": {name: record}, "correct": {name: 0 or 1}}``. Returns seven parameters
    (intercept plus six shared weights). ``weights[0]`` is the intercept.
    """
    xs, ys = [], []
    for row in rows:
        actions, correct = row["actions"], row["correct"]
        for name in V2_ACTIONS:
            if name not in actions or name not in correct:
                continue
            xs.append(action_features(name, _with_direct_answer(name, actions)))
            ys.append(float(correct[name]))
    if not xs:
        raise ValueError("fit_v2_cal received no labelled actions")
    weights = _fit_logistic(np.vstack(xs), np.asarray(ys, dtype=float))
    if len(weights) != N_PARAMS or N_PARAMS >= 10:
        raise RuntimeError(f"v2-Cal parameter count must be 7 and < 10, got {len(weights)}")
    return {
        "weights": [float(w) for w in weights],
        "n_params": N_PARAMS,
        "feature_names": list(FEATURE_NAMES),
        "intercept_index": 0,
    }


def _proba(weights: list[float], feats: np.ndarray) -> float:
    logit = float(weights[0] + np.dot(np.asarray(weights[1:], dtype=float), feats))
    logit = min(30.0, max(-30.0, logit))
    return float(1.0 / (1.0 + math.exp(-logit)))


def select_v2_cal(model: dict, actions: dict) -> dict:
    """Pick the action with the highest predicted P(correct). Ties keep the earlier V2_ACTIONS entry."""
    weights = model["weights"]
    if len(weights) != N_PARAMS or model.get("n_params", N_PARAMS) >= 10:
        raise ValueError(f"v2-Cal model must have {N_PARAMS} parameters")
    probs = {}
    best_name, best_p = None, None
    for name in V2_ACTIONS:
        if name not in actions:
            continue
        prob = _proba(weights, action_features(name, _with_direct_answer(name, actions)))
        probs[name] = prob
        if best_name is None or prob > best_p:
            best_name, best_p = name, prob
    if best_name is None:
        raise ValueError("select_v2_cal received no actions")
    return {"action": best_name, "answer": actions[best_name]["answer"], "probs": probs}


def utility(policy_scores, best_single_scores) -> float:
    """Mean score gap versus the strongest single action. This is what thresholds optimise."""
    policy = np.asarray(policy_scores, dtype=float)
    baseline = np.asarray(best_single_scores, dtype=float)
    if policy.shape != baseline.shape:
        raise ValueError(f"utility arrays differ in length: {policy.shape} vs {baseline.shape}")
    if policy.size == 0:
        raise ValueError("utility is undefined on an empty sample")
    return float((policy - baseline).mean())


def best_single_action(scores_by_action: dict[str, np.ndarray]) -> str:
    """Highest mean score. Ties are broken by ``V2_ACTIONS`` order, then by name."""
    if not scores_by_action:
        raise ValueError("best_single_action received no actions")
    lengths = {len(np.asarray(v)) for v in scores_by_action.values()}
    if len(lengths) != 1:
        raise ValueError("best_single_action arrays differ in length")

    def sort_key(name: str):
        mean = float(np.mean(np.asarray(scores_by_action[name], dtype=float)))
        order = V2_ACTIONS.index(name) if name in V2_ACTIONS else len(V2_ACTIONS)
        return (-mean, order, name)

    return min(scores_by_action, key=sort_key)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p). Pure Python; no scipy."""
    if n < 0:
        raise ValueError("n must be non-negative")
    k, n, p = int(k), int(n), float(p)
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logs = []
    for i in range(k + 1):
        log_coef = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        logs.append(log_coef + i * math.log(p) + (n - i) * math.log(1.0 - p))
    pivot = max(logs)
    total = math.exp(pivot) * sum(math.exp(term - pivot) for term in logs)
    return float(min(1.0, max(0.0, total)))


def _fallback(best_action: str | None) -> dict:
    if not best_action:
        raise ValueError("ltt_select fallback needs the best single action's name")
    return {"name": "best_single", "action": best_action, "fallback": True}


def ltt_select(candidates, scores_by_candidate, best_single_scores, delta: float = 0.1, best_action: str | None = None) -> dict:
    """Fixed-sequence Learn-then-Test. Candidates are already ordered conservative -> aggressive.

    定义（与 docs/DEV_DECISIONS.md「v2 预注册」一致，如实陈述，不加强结论）：

    - 损失 L_i = 1，当策略在第 i 题的得分 < 最强单一动作的得分；否则为 0。
    - 原假设 H0：真实退化率 E[L] > alpha，其中 alpha = delta。
    - p 值 = P(X <= 观测到的退化次数)，X ~ Binomial(n, alpha)。用纯 Python 计算，不用 scipy。
      当 p 值 <= delta 时拒绝 H0，该候选被接受为安全。
    - 固定序列：候选已经按保守到激进排好，本函数不重新排序。按该顺序走，只要检验拒绝 H0 就接受；
      遇到第一个不能拒绝 H0 的候选就停止，其后的候选都不再接受。被接受的前缀在固定序列规则下、
      水平 delta 上是同时有效的。
    - 只在被接受的前缀里，选择效用最高的候选。效用是 mean(得分_策略 - 得分_最强单一动作)，
      不是准确率。效用并列时取更早（更保守）的候选。
    - 若前缀为空，或前缀里的最高效用 < 0，返回
      {"name": "best_single", "action": <best_single_action 的名字>, "fallback": True}。

    假设：最强单一动作是在同一份校准样本上选出的，因此这个有限样本声明略偏反保守
    （anti-conservative）。题目按 i.i.d. 处理。这里实现的总体声明是：在校准抽样上，以至少
    1 - delta 的概率，每一个被接受的候选都满足真实的
    P(score_policy < score_best_single) <= delta。这弱于「永远不差于最强单一动作」。
    不声称退化率为 0。
    """
    baseline = np.asarray(best_single_scores, dtype=float)
    accepted = []
    for candidate in candidates:
        scores = np.asarray(scores_by_candidate[candidate["name"]], dtype=float)
        if scores.shape != baseline.shape:
            raise ValueError(f"{candidate['name']}: score length {scores.shape} != {baseline.shape}")
        degradations = int(np.sum(scores < baseline))
        p_value = _binom_cdf(degradations, scores.size, delta)  # alpha = delta
        if p_value <= float(delta):
            accepted.append((candidate, scores))
        else:
            break
    if not accepted:
        return _fallback(best_action)
    chosen, chosen_u = None, None
    for candidate, scores in accepted:
        gap = utility(scores, baseline)
        if chosen is None or gap > chosen_u:
            chosen, chosen_u = candidate, gap
    if chosen_u < 0:
        return _fallback(best_action)
    out = dict(chosen)
    out["fallback"] = False
    out["utility"] = float(chosen_u)
    return out


def _format_threshold(value: float) -> str:
    if value == -1e9:
        return "-1e9"
    if value == 1e9:
        return "1e9"
    return format(value, "g")


def preregistered_grid() -> list[dict]:
    """Margin rule, conservative to aggressive.

    tau1 ascends (-1e9 always keeps direct on ordinary margins; 1e9 always considers evidence).
    Within each tau1, theta descends (1 is the stricter agreement).
    """
    grid = []
    for tau1 in _TAU1_GRID:
        for theta in _THETA_GRID:
            grid.append({
                "name": f"margin_tau{_format_threshold(tau1)}_theta{_format_threshold(theta)}",
                "tau1": float(tau1),
                "theta": float(theta),
                "confidence": "margin",
            })
    return grid


def apply_candidate(candidate, actions) -> dict:
    """Dispatch one grid candidate to ``select_v2_rule``."""
    return select_v2_rule(
        actions,
        tau1=float(candidate["tau1"]),
        theta=float(candidate["theta"]),
        confidence=candidate.get("confidence", "margin"),
    )


def summarize_policy(policy_scores, baseline_scores, cluster_ids=None) -> dict:
    """Paired comparison of the policy against a baseline. Reuses ``score_official.paired``."""
    from omniaware.eval.score_official import paired
    return paired(baseline_scores, policy_scores, cluster_ids=cluster_ids)


def assert_disjoint_qids(calib, eval_ids) -> None:
    """Raise if the calibration and evaluation qids overlap or repeat."""
    calib, eval_ids = list(calib), list(eval_ids)
    if len(set(calib)) != len(calib) or len(set(eval_ids)) != len(eval_ids):
        raise ValueError("qid list contains duplicates")
    overlap = set(calib) & set(eval_ids)
    if overlap:
        shown = ", ".join(str(q) for q in sorted(overlap)[:5])
        raise ValueError(f"calib and eval qids overlap: {shown}")


def split_half(qids, seed) -> tuple[list, list]:
    """Shuffle qids with ``seed`` and split into disjoint calibration and evaluation halves.

    The calibration half is the first floor(n/2) of the permutation. The checker runs before return.
    """
    ids = [str(q) for q in qids]
    if len(ids) != len(set(ids)):
        raise ValueError("qid list contains duplicates")
    if len(ids) < 2:
        raise ValueError("need at least two qids to split")
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    mid = len(shuffled) // 2
    calib, eval_ids = shuffled[:mid], shuffled[mid:]
    assert_disjoint_qids(calib, eval_ids)
    return calib, eval_ids

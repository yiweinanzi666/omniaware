"""Synthetic tests for the v2 selector. No run files, and gold_in_topk is never read by the selector."""
import copy
import json
import math

import numpy as np
import pytest

from omniaware.actions import DIRECT, EVIDENCE_ACTIONS, V2_ACTIONS
from omniaware.arbitration import (
    _available,
    _fit_model,
    _one_sided_p,
    _split_halves,
    apply_decision,
    best_evidence_action,
    best_single_action,
    decode_json_float,
    encode_json_float,
    normalize_answer,
    policy_scores,
    select_cal,
    select_rule,
    token_f1,
)

EVIDENCE = list(EVIDENCE_ACTIONS)


def _record(answer, margin=0.5, available=True, kl=0.1, top1=0.3):
    return {
        "answer": answer,
        "available": available,
        "seq_logprob": -1.0,
        "gen_len": 3,
        "first_token": {"margin": margin, "entropy": 0.5, "entropy_norm": 0.5},
        "kl_to_direct": kl,
        "retriever": {"top1_score": top1, "top1_top2_margin": 0.05, "gold_in_topk": False},
    }


def _row(qid, scores, margins=None, answers=None, unavailable=(), kl=0.1):
    margins, answers = margins or {}, answers or {}
    actions = {
        name: _record(answers.get(name, f"{qid}-{name}"), margin=margins.get(name, 0.5),
                      available=name not in unavailable, kl=kl)
        for name in scores
    }
    return {"qid": qid, "actions": actions, "scores": dict(scores)}


def _oracle_rows(n, prefix):
    """Direct is right and confident on half the questions. A static evidence action is right and the only
    confident alternative on the other half. The oracle scores 100 on every question."""
    rows = []
    for i in range(n):
        direct_right = i % 2 == 0
        scores = {name: 0.0 for name in V2_ACTIONS}
        margins = {name: 0.3 for name in V2_ACTIONS}
        if direct_right:
            scores[DIRECT] = 100.0
            margins[DIRECT] = 2.0
        else:
            scores["static_text"] = 100.0
            margins[DIRECT] = 0.1
            margins["static_text"] = 3.0
        rows.append(_row(f"{prefix}{i}", scores, margins=margins))
    return rows


def _random_rows(n, seed):
    """Signals are correlated with correctness, so the cal family has something to fit."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        margin = float(rng.uniform(0.0, 3.0))
        gold = f"g{i % 4}"
        scores, margins, answers = {}, {}, {}
        for name in V2_ACTIONS:
            if name == DIRECT:
                p_correct = 1.0 / (1.0 + math.exp(-(margin - 1.5)))
                margins[name] = margin
            else:
                p_correct = 0.7 if name == "static_image" else 0.4
                margins[name] = float(rng.uniform(0.0, 3.0))
            correct = bool(rng.random() < p_correct)
            scores[name] = 100.0 if correct else 0.0
            answers[name] = gold if correct else f"w{int(rng.integers(0, 9))}"
        rows.append(_row(f"r{seed}-{i}", scores, margins=margins, answers=answers,
                         kl=float(rng.uniform(0.0, 1.0))))
    return rows


def _mixed_star_rows(n, prefix):
    """static_mixed has the highest mean (60 against 40 for direct). Direct is right and confident on 40% of
    questions, and mixed is right on the other 60%, where direct is wrong and unsure."""
    rows = []
    for i in range(n):
        direct_right = i % 5 < 2
        scores = {name: 0.0 for name in V2_ACTIONS}
        margins = {name: 0.3 for name in V2_ACTIONS}
        if direct_right:
            scores[DIRECT] = 100.0
            margins[DIRECT] = 2.0
        else:
            scores["static_mixed"] = 100.0
            margins[DIRECT] = 0.1
        rows.append(_row(f"{prefix}{i}", scores, margins=margins))
    return rows


def _worse_on_thirty_better_on_half(n):
    """Against direct (a*), the policy 'use static_text when direct is unsure' gains 10 on 50% of questions and
    loses 5 on 30%. The mean gap is +3.5."""
    rows = []
    for i in range(n):
        slot = i % 10
        scores = {name: 0.0 for name in V2_ACTIONS}
        margins = {DIRECT: 0.1}
        if slot < 5:
            scores[DIRECT], scores["static_text"] = 0.0, 10.0
        elif slot < 8:
            scores[DIRECT], scores["static_text"] = 10.0, 5.0
        else:
            scores[DIRECT], scores["static_text"] = 50.0, 0.0
            margins[DIRECT] = 2.0
        unavailable = [name for name in EVIDENCE if name != "static_text"]
        rows.append(_row(f"w{i}", scores, margins=margins, unavailable=unavailable))
    return rows


def _constant_gap_rows(n):
    """direct scores 50 and static_text scores 48 on every question, with identical margins."""
    rows = []
    for i in range(n):
        scores = {name: 0.0 for name in V2_ACTIONS}
        scores[DIRECT], scores["static_text"] = 50.0, 48.0
        rows.append(_row(f"k{i}", scores, margins={name: 0.5 for name in V2_ACTIONS}))
    return rows


def _gold_flipped(actions):
    flipped = copy.deepcopy(actions)
    for rec in flipped.values():
        rec["retriever"]["gold_in_topk"] = not rec["retriever"]["gold_in_topk"]
        rec["scores"] = {"direct": -1.0}
    return flipped


def test_token_f1_ignores_case_and_punctuation():
    assert normalize_answer("Wood.") == ["wood"]
    assert token_f1("Wood.", "wood") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("", "a") == 0.0
    assert token_f1("a a", "a") == pytest.approx(2.0 / 3.0)
    assert token_f1("a b", "a c") == pytest.approx(0.5)


def test_perfect_confidence_reaches_the_oracle_without_fallback():
    calib = _oracle_rows(40, "c")
    held = _oracle_rows(40, "h")
    decision = select_rule(calib, delta=0.1, epsilon=1.0)
    assert decision["kind"] == "rule" and decision["fallback"] is False
    assert decision["best_single"] == DIRECT
    policy = np.asarray(policy_scores(decision, held))
    oracle = np.asarray([max(row["scores"].values()) for row in held])
    assert np.array_equal(policy, oracle)
    assert decision["calib_utility"] == pytest.approx(50.0)


def test_pure_noise_signals_fall_back_to_the_strongest_single_action():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        scores = {DIRECT: 60.0 + rng.normal(0, 1)}
        scores.update({name: 40.0 + rng.normal(0, 1) for name in EVIDENCE})
        margins = {name: float(rng.uniform(0, 3)) for name in V2_ACTIONS}
        answers = {name: f"w{int(rng.integers(0, 5))}" for name in V2_ACTIONS}
        rows.append(_row(f"n{i}", scores, margins=margins, answers=answers, kl=float(rng.uniform())))
    for decision in (select_rule(rows), select_cal(rows, seed=0)):
        assert decision["fallback"] is True
        assert decision["kind"] == "single"
        assert decision["action"] == DIRECT
        assert apply_decision(decision, rows[0]["actions"])["action"] == DIRECT


def test_fixed_sequence_starts_at_a_star_when_a_star_is_evidence():
    calib = _mixed_star_rows(50, "c")
    held = _mixed_star_rows(50, "h")
    decision = select_rule(calib, delta=0.1, epsilon=1.0)
    assert decision["best_single"] == "static_mixed"
    assert decision["fallback"] is False
    assert decision["calib_utility"] > 0.0
    policy = np.asarray(policy_scores(decision, held))
    held_mixed = np.asarray([row["scores"]["static_mixed"] for row in held])
    assert policy.mean() > held_mixed.mean()
    oracle = np.asarray([max(row["scores"].values()) for row in held])
    assert np.array_equal(policy, oracle)


def test_policy_worse_on_thirty_percent_and_better_on_half_is_accepted():
    calib = _worse_on_thirty_better_on_half(100)
    decision = select_rule(calib, delta=0.1, epsilon=1.0)
    assert decision["best_single"] == DIRECT
    assert decision["fallback"] is False
    direct = np.asarray([row["scores"][DIRECT] for row in calib])
    gaps = np.asarray(policy_scores(decision, calib)) - direct
    assert np.mean(gaps > 0) == pytest.approx(0.5)
    assert np.mean(gaps < 0) == pytest.approx(0.3)
    assert decision["calib_utility"] == pytest.approx(3.5)


def test_epsilon_and_delta_are_independent_knobs():
    calib = _constant_gap_rows(40)
    # the fixed sequence reaches a policy with constant gap -2: accepted only when epsilon > 2
    loose = select_rule(calib, delta=0.1, epsilon=3.0)["n_accepted"]
    tight = select_rule(calib, delta=0.1, epsilon=1.0)["n_accepted"]
    assert loose > tight
    assert select_rule(calib, delta=0.01, epsilon=3.0)["n_accepted"] == loose
    assert select_rule(calib, delta=0.5, epsilon=3.0)["n_accepted"] == loose
    assert select_rule(calib, delta=0.01, epsilon=1.0)["n_accepted"] == tight

    grows = _worse_on_thirty_better_on_half(100)
    assert select_rule(grows, epsilon=0.5)["n_accepted"] <= select_rule(grows, epsilon=5.0)["n_accepted"]


def test_one_sided_p_matches_the_normal_tail_and_handles_constant_gaps():
    gap = np.array([1.0, 2.0, 3.0, 4.0])
    t = 2.5 / (float(gap.std(ddof=1)) / math.sqrt(4))
    assert _one_sided_p(gap, 0.0) == pytest.approx(0.5 * math.erfc(t / math.sqrt(2.0)))
    assert _one_sided_p(np.full(5, -2.0), 3.0) == 0.0
    assert _one_sided_p(np.full(5, -2.0), 1.0) == 1.0


def test_cal_drops_missing_features_and_refuses_partial_missingness():
    rows = _random_rows(60, seed=1)
    for row in rows:
        row["actions"]["static_text"]["kl_to_direct"] = None
    fit, _ = _split_halves(rows, seed=0)
    feats = [
        {"conf_direct": 1.0, "conf_star": 1.0, "conf_gap": 0.1 * i, "token_f1_to_star": 0.5 * (i % 2),
         "kl_to_direct": math.nan, "retriever_top1": 0.2 + 0.01 * i}
        for i in range(len(fit))
    ]
    model = _fit_model("static_text", feats, np.linspace(0.0, 1.0, len(fit)))
    assert "kl_to_direct" in model["dropped"]
    assert "kl_to_direct" not in model["features"]
    assert len(model["weights"]) == model["n_params"] == 1 + len(model["features"])
    assert all(math.isfinite(w) for w in model["weights"])

    partial = _random_rows(60, seed=1)
    partial[5]["actions"]["static_mixed"]["kl_to_direct"] = None
    with pytest.raises(ValueError, match="missing"):
        select_cal(partial, seed=0)


def test_unavailable_actions_are_never_selected():
    rows = _random_rows(60, seed=2)
    for row in rows[::3]:
        row["actions"]["static_image"]["available"] = False
    for decision in (select_cal(rows, seed=0), select_rule(rows)):
        for row in rows:
            chosen = apply_decision(decision, row["actions"])["action"]
            assert _available(row["actions"], chosen)


def test_cal_fit_and_risk_halves_are_disjoint_and_cover_the_calibration_set():
    rows = _random_rows(41, seed=3)
    fit, risk = _split_halves(rows, seed=5)
    fit_ids = {row["qid"] for row in fit}
    risk_ids = {row["qid"] for row in risk}
    assert fit_ids.isdisjoint(risk_ids)
    assert fit_ids | risk_ids == {row["qid"] for row in rows}
    assert [row["qid"] for row in _split_halves(rows, seed=5)[0]] == [row["qid"] for row in fit]
    decision = select_cal(rows, seed=5)
    assert decision["n_fit"] == len(fit) and decision["n_risk"] == len(risk)
    assert decision["n_fit"] + decision["n_risk"] == len(rows)


def test_apply_decision_ignores_scores_and_gold_in_topk():
    calib = _random_rows(40, seed=7)
    for decision in (select_rule(calib), select_cal(calib, seed=7)):
        for row in calib:
            plain = apply_decision(decision, row["actions"])
            assert plain == apply_decision(decision, _gold_flipped(row["actions"]))


def test_infinite_thresholds_survive_strict_json():
    assert encode_json_float(-math.inf) == "-inf"
    assert encode_json_float(math.inf) == "inf"
    assert encode_json_float(0.25) == 0.25
    assert decode_json_float("-inf") == -math.inf
    assert decode_json_float("inf") == math.inf
    assert decode_json_float(0.25) == 0.25
    with pytest.raises(ValueError):
        decode_json_float("nan")

    always_direct = {"kind": "rule", "confidence": "margin", "tau": encode_json_float(-math.inf),
                     "evidence_choice": "best", "evidence_ranking": list(EVIDENCE), "action_ranking": [DIRECT]}
    text = json.dumps(always_direct, allow_nan=False)
    loaded = json.loads(text)
    actions = _oracle_rows(1, "x")[0]["actions"]
    assert apply_decision(loaded, actions)["action"] == DIRECT

    calib = _oracle_rows(40, "c")
    held = _oracle_rows(40, "h")
    decision = select_rule(calib)
    loaded = json.loads(json.dumps(decision, allow_nan=False))
    assert policy_scores(loaded, held) == policy_scores(decision, held)


def test_strongest_action_ties_follow_v2_order():
    scores = {"static_text": [1.0, 3.0], DIRECT: [2.0, 2.0], "static_image": [0.0, 4.0]}
    assert best_single_action(scores) == DIRECT
    assert best_evidence_action(scores) == "static_text"


def test_neg_entropy_confidence_is_accepted_by_both_families():
    calib = _oracle_rows(40, "c")
    rule = select_rule(calib, confidence="neg_entropy")
    cal = select_cal(calib, seed=0, confidence="neg_entropy")
    assert rule["confidence"] == cal["confidence"] == "neg_entropy"
    with pytest.raises(ValueError, match="confidence"):
        select_rule(calib, confidence="top2")


def test_a_policy_identical_to_a_star_passes_at_epsilon_zero():
    """With epsilon = 0 the first candidate in the sequence (equal to a*) has a constant zero gap and must pass,
    otherwise the whole sequence is rejected before any informative candidate is tested."""
    assert _one_sided_p(np.zeros(50), 0.0) == 0.0
    rows = _oracle_rows(400, "eps0")
    decision = select_rule(rows, delta=0.1, epsilon=0.0)
    assert not decision["fallback"]


def test_an_action_available_on_some_questions_can_still_be_selected():
    """static_image has no candidates on half the questions. It cannot be a*, but where it is available and
    direct is unsure it is the right answer, and the rule must be able to pick it."""
    def rows(n, prefix):
        out = []
        for i in range(n):
            has_image = i % 2 == 0
            direct_right = i % 4 in (1, 2)
            scores = {DIRECT: 100.0 if direct_right else 0.0, "static_mixed": 20.0}
            margins = {DIRECT: 5.0 if direct_right else 0.1, "static_mixed": 0.2}
            unavailable = ()
            if has_image:
                scores["static_image"] = 0.0 if direct_right else 100.0
                margins["static_image"] = 3.0
            else:
                unavailable = ("static_image",)
            row = _row(f"{prefix}{i}", {**scores, **({} if has_image else {"static_image": 0.0})},
                       margins=margins, unavailable=unavailable)
            if not has_image:
                del row["scores"]["static_image"]
            out.append(row)
        return out

    decision = select_rule(rows(400, "c"), delta=0.1, epsilon=1.0)
    assert decision["best_single"] == DIRECT and not decision["fallback"]
    assert "static_image" in decision["evidence_ranking"]
    evaluation = rows(200, "e")
    chosen = [apply_decision(decision, row["actions"])["action"] for row in evaluation]
    assert "static_image" in chosen
    for row, name in zip(evaluation, chosen):
        assert row["actions"][name]["available"]
    assert np.mean(policy_scores(decision, evaluation)) > np.mean([row["scores"][DIRECT] for row in evaluation])

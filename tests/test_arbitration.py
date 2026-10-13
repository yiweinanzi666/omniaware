"""Synthetic tests for the v2 selector. No run files and no gold-in-topk leakage."""
import copy
import json
import math

import numpy as np
import pytest

from omniaware.arbitration import (
    V2_ACTIONS,
    action_features,
    apply_candidate,
    assert_disjoint_qids,
    best_single_action,
    fit_v2_cal,
    ltt_select,
    preregistered_grid,
    select_v2_cal,
    select_v2_rule,
    split_half,
    summarize_policy,
    token_f1,
    utility,
)
from omniaware.eval.score_official import paired
from scripts.analysis.dev_select import apply_frozen_v2, frozen_v2, main as dev_main, select_v2


def _action(answer, margin=1.0, entropy_norm=0.4, seq_logprob=-0.5, gen_len=4,
            kl=0.0, top1=0.2, gold=False):
    return {
        "answer": answer,
        "answer_empty": answer.strip() == "",
        "seq_logprob": seq_logprob,
        "gen_len": gen_len,
        "first_token": {
            "topk": [{"id": 10, "logprob": -0.2}, {"id": 11, "logprob": -1.2}],
            "entropy": 1.2,
            "entropy_norm": entropy_norm,
            "margin": margin,
            "vocab_size": 151936,
        },
        "kl_to_direct": kl,
        "retriever": {"top1_score": top1, "top1_top2_margin": 0.05, "gold_in_topk": gold},
    }


def _question(qid, specs, scores):
    actions = {name: _action(answer, margin=margin) for name, (answer, margin) in specs.items()}
    return {"qid": qid, "actions": actions, "scores": scores}


def _perfect_records():
    """Direct is right and high-margin on the easy half, wrong and low-margin on the hard half.

    On the hard half two evidence actions agree on the right answer and outrank the others on margin.
    On the easy half those evidence actions agree on a wrong answer, so a rule that always opens the
    evidence branch is worse than direct. Ordinary margins sit in (0, 1e9).
    """
    easy = {
        "direct": ("easy right", 5.0),
        "static_text": ("easy wrong", 4.0),
        "static_image": ("easy wrong", 3.5),
        "static_mixed": ("mixed easy", 0.4),
        "rerank_static": ("rerank easy", 0.3),
    }
    hard = {
        "direct": ("wrong direct", 0.1),
        "static_text": ("right answer", 3.0),
        "static_image": ("right answer", 2.0),
        "static_mixed": ("mixed only", 0.4),
        "rerank_static": ("rerank only", 0.3),
    }
    easy_scores = {"direct": 1.0, "static_text": 0.0, "static_image": 0.0, "static_mixed": 0.0, "rerank_static": 0.0}
    hard_scores = {"direct": 0.0, "static_text": 1.0, "static_image": 1.0, "static_mixed": 0.0, "rerank_static": 0.0}
    records = []
    for i in range(12):
        records.append(_question(f"e{i}", easy, easy_scores))
    for i in range(12):
        records.append(_question(f"h{i}", hard, hard_scores))
    return records


def _policy_scores(decision, records):
    scores = []
    for rec in records:
        if decision.get("fallback"):
            name = decision["action"]
        else:
            name = apply_candidate(decision, rec["actions"])["action"]
        scores.append(rec["scores"][name])
    return np.asarray(scores, dtype=float)


def test_token_f1_is_a_whitespace_multiset():
    assert token_f1("", "") == 1.0
    assert token_f1("", "a") == 0.0
    assert token_f1("a b", "") == 0.0
    assert token_f1("a b", "b a") == 1.0
    assert token_f1("a a", "a") == pytest.approx(2.0 / 3.0)
    assert token_f1("a b", "a c") == pytest.approx(0.5)


def test_rule_selects_the_agreed_answer_when_direct_is_unsure():
    actions = {
        "direct": _action("wrong direct", margin=0.1),
        "static_text": _action("right answer", margin=2.5),
        "static_image": _action("right answer", margin=1.5),
        "static_mixed": _action("something else", margin=4.0),  # higher margin, no agreement
        "rerank_static": _action("wrong direct", margin=0.2),
    }
    out = select_v2_rule(actions, tau1=0.5, theta=1.0)
    assert out == {"action": "static_text", "answer": "right answer", "used_evidence": True}
    # margin >= 0, so tau1=0 keeps direct; tau1=-1e9 always keeps direct on this margin
    assert select_v2_rule(actions, tau1=0.0, theta=0.0)["action"] == "direct"
    assert select_v2_rule(actions, tau1=-1e9, theta=0.0)["used_evidence"] is False
    opened = select_v2_rule(actions, tau1=1e9, theta=1.0)
    assert opened["action"] == "static_text" and opened["used_evidence"] is True
    # no action agrees with any other action: fall back to direct without using evidence
    alone = {name: _action(name, margin=0.01) for name in V2_ACTIONS}
    fallback = select_v2_rule(alone, tau1=1.0, theta=1.0)
    assert fallback == {"action": "direct", "answer": "direct", "used_evidence": False}
    # equal margins: the earlier action in V2_ACTIONS wins
    tied = {
        "direct": _action("no", margin=0.0),
        "static_text": _action("yes", margin=2.0),
        "static_image": _action("yes", margin=2.0),
        "static_mixed": _action("other", margin=0.1),
        "rerank_static": _action("third", margin=0.1),
    }
    assert select_v2_rule(tied, tau1=0.5, theta=1.0)["action"] == "static_text"


def test_entropy_confidence_ranks_low_entropy_among_agreeing_actions():
    actions = {
        "direct": _action("unsure", margin=3.0, entropy_norm=0.9),
        "static_text": _action("same", margin=4.0, entropy_norm=0.45),
        "static_image": _action("same", margin=0.2, entropy_norm=0.15),
        "static_mixed": _action("other", margin=5.0, entropy_norm=0.05),
        "rerank_static": _action("third", margin=0.0, entropy_norm=0.8),
    }
    out = select_v2_rule(actions, tau1=0.5, theta=1.0, confidence="entropy")
    assert out["action"] == "static_image"  # lowest entropy among actions that agree; mixed does not
    sure = dict(actions)
    sure["direct"] = _action("unsure", entropy_norm=0.1)
    kept = select_v2_rule(sure, tau1=0.5, theta=0.0, confidence="entropy")
    assert kept == {"action": "direct", "answer": "unsure", "used_evidence": False}


def test_perfect_signals_ltt_reaches_the_oracle():
    grid = preregistered_grid()
    assert len(grid) == 24
    assert grid[0] == {"name": "margin_tau-1e9_theta1", "tau1": -1e9, "theta": 1.0, "confidence": "margin"}
    assert grid[-1]["tau1"] == 1e9 and grid[-1]["theta"] == 0.0
    assert [item["tau1"] for item in grid] == sorted(item["tau1"] for item in grid)
    for tau in {item["tau1"] for item in grid}:
        thetas = [item["theta"] for item in grid if item["tau1"] == tau]
        assert thetas == [1.0, 0.5, 0.0]

    records = _perfect_records()
    scores_by_action = {
        name: np.array([rec["scores"][name] for rec in records], dtype=float) for name in V2_ACTIONS
    }
    assert best_single_action(scores_by_action) == "direct"
    decision = select_v2(records, delta=0.1)["v2"]
    assert decision["fallback"] is False
    assert decision["name"] == "margin_tau0.25_theta1"
    oracle = np.array([max(rec["scores"].values()) for rec in records], dtype=float)
    policy = _policy_scores(decision, records)
    assert np.array_equal(policy, oracle)
    assert decision["utility"] == pytest.approx(utility(policy, scores_by_action["direct"]))
    assert decision["utility"] == pytest.approx(0.5)
    summary = summarize_policy(policy, scores_by_action["direct"])
    assert summary == paired(scores_by_action["direct"], policy)
    clusters = [rec["qid"][:1] for rec in records]
    assert summarize_policy(policy, scores_by_action["direct"], cluster_ids=clusters) == paired(
        scores_by_action["direct"], policy, cluster_ids=clusters)


def test_noise_falls_back_to_the_strongest_single_on_the_held_out_half():
    """Scores do not depend on the confidence features. Thresholds are fit on the calib half only.

    The calib half's strongest action is static_text. The eval half's strongest action is direct.
    A frozen fallback must still emit static_text on the eval half.
    """
    rng = np.random.default_rng(0)
    records = []
    for i in range(40):
        actions = {}
        for name in V2_ACTIONS:
            actions[name] = _action(
                answer=f"noise-{name}-{int(rng.integers(0, 8))}",
                margin=float(rng.uniform(0.0, 3.0)),
                entropy_norm=float(rng.uniform(0.0, 1.0)),
                seq_logprob=float(rng.normal()),
                gen_len=int(rng.integers(1, 6)),
                kl=float(rng.uniform(0.0, 1.0)),
                top1=float(rng.uniform(0.0, 1.0)),
                gold=bool(rng.integers(0, 2)),
            )
        scores = {name: float(rng.normal()) for name in V2_ACTIONS}
        records.append({"qid": f"q{i}", "actions": actions, "scores": scores})
    calib_ids, eval_ids = split_half([rec["qid"] for rec in records], seed=0)
    assert set(calib_ids).isdisjoint(eval_ids)
    by_id = {rec["qid"]: rec for rec in records}
    for rec in (by_id[q] for q in calib_ids):
        rec["scores"]["static_text"] += 8.0
    for rec in (by_id[q] for q in eval_ids):
        rec["scores"]["direct"] += 8.0
    calib = [by_id[q] for q in calib_ids]
    held_out = [by_id[q] for q in eval_ids]
    assert_disjoint_qids([rec["qid"] for rec in calib], [rec["qid"] for rec in held_out])
    decisions = select_v2(calib, delta=0.1)
    assert decisions["v2"]["fallback"] is True
    assert decisions["v2"]["name"] == "best_single"
    assert decisions["v2"]["action"] == "static_text"
    frozen = frozen_v2(decisions)
    assert frozen == decisions["v2"]
    for rec in held_out:
        chosen = apply_frozen_v2(frozen, rec["actions"])
        assert chosen["action"] == "static_text"
        assert chosen["answer"] == rec["actions"]["static_text"]["answer"]
        assert chosen["used_evidence"] is True
    # the held-out half would prefer direct; the frozen choice does not refit there
    eval_scores = {name: np.array([rec["scores"][name] for rec in held_out]) for name in V2_ACTIONS}
    assert best_single_action(eval_scores) == "direct"


def test_split_half_refuses_overlap():
    calib, eval_ids = split_half([f"q{i}" for i in range(11)], seed=1)
    assert set(calib).isdisjoint(eval_ids)
    assert len(calib) + len(eval_ids) == 11
    assert len(calib) == 5
    again, _ = split_half([f"q{i}" for i in range(11)], seed=1)
    assert calib == again
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_qids(["a", "b"], ["b", "c"])
    with pytest.raises(ValueError, match="duplicates"):
        split_half(["a", "a", "b", "c"], seed=0)
    with pytest.raises(ValueError, match="duplicates"):
        assert_disjoint_qids(["a", "a"], ["b"])


def test_ltt_binomial_boundary_and_negative_utility_fallback():
    def trial(n):
        scores = np.zeros(n)
        return ltt_select(
            [{"name": "flat", "tau1": 0.0, "theta": 1.0, "confidence": "margin"}],
            {"flat": scores}, scores, delta=0.1, best_action="direct")
    assert trial(22)["fallback"] is False
    assert trial(22)["utility"] == pytest.approx(0.0)
    assert trial(21)["fallback"] is True and trial(21)["action"] == "direct"
    # one large loss is still inside the binomial acceptance region at n=40, but utility is negative
    n = 40
    worse = np.zeros(n)
    worse[0] = -1.0
    chosen = ltt_select(
        [{"name": "slightly_worse"}], {"slightly_worse": worse}, np.zeros(n),
        delta=0.1, best_action="static_text")
    assert chosen == {"name": "best_single", "action": "static_text", "fallback": True}


def test_thresholds_follow_utility_not_accuracy():
    """Utility inside the accepted prefix, not hit rate and not the best mean on the whole grid.

    ``deep`` gains 2 on four questions and ties the rest (strict-improvement rate 4/24, utility 8/24).
    ``broad`` gains 0.1 on every question (improvement rate 1, utility 0.1). Both have zero
    degradations, so both are accepted, and utility selects ``deep``. ``blocker`` then fails the
    binomial test, so the walk stops. ``tempting`` has the highest mean score on the grid but sits
    after that failure and must not be selected.
    """
    n = 24
    best = np.zeros(n)
    deep = np.array([2.0] * 4 + [0.0] * (n - 4))
    broad = np.full(n, 0.1)
    blocker = np.zeros(n)
    blocker[0] = -0.01
    tempting = np.ones(n)
    candidates = [
        {"name": "deep", "tau1": 0.0, "theta": 1.0, "confidence": "margin"},
        {"name": "broad", "tau1": 0.5, "theta": 0.5, "confidence": "margin"},
        {"name": "blocker", "tau1": 1.0, "theta": 0.5, "confidence": "margin"},
        {"name": "tempting", "tau1": 4.0, "theta": 0.0, "confidence": "margin"},
    ]
    scores = {"deep": deep, "broad": broad, "blocker": blocker, "tempting": tempting}
    chosen = ltt_select(candidates, scores, best, delta=0.1, best_action="direct")
    improvement_rate = lambda scores: float(np.mean(np.asarray(scores) > best))
    assert utility(deep, best) > utility(broad, best)
    assert improvement_rate(broad) > improvement_rate(deep)
    assert float(tempting.mean()) > float(deep.mean())
    assert utility(tempting, best) > utility(deep, best)
    assert chosen["fallback"] is False
    assert chosen["name"] == "deep"
    assert chosen["utility"] == pytest.approx(utility(deep, best))
    # same utility: the earlier, more conservative candidate wins
    tied = ltt_select(
        [{"name": "first"}, {"name": "second"}],
        {"first": np.ones(n), "second": np.ones(n)},
        np.zeros(n), delta=0.1, best_action="direct")
    assert tied["name"] == "first" and tied["fallback"] is False


def test_calibrator_has_seven_parameters_and_ranks_separable_labels():
    tiny = []
    for margin, label, answer in ((0.2, 0, "low"), (1.5, 1, "high")):
        actions = {name: _action(answer, margin=margin, kl=None if name == "direct" else 0.2) for name in V2_ACTIONS}
        tiny.append({"actions": actions, "correct": {name: label for name in V2_ACTIONS}})
    tiny_model = fit_v2_cal(tiny)
    assert tiny_model["n_params"] == 7
    assert tiny_model["n_params"] < 10
    assert len(tiny_model["weights"]) == 7
    assert tiny_model["intercept_index"] == 0
    assert tiny_model["feature_names"][0] == "intercept"
    assert len(tiny_model["feature_names"]) == 7
    assert all(math.isfinite(w) for w in tiny_model["weights"])

    rows = []
    for _ in range(12):
        actions, correct = {}, {}
        for name in V2_ACTIONS:
            high = name in ("direct", "static_image")
            actions[name] = _action("same", margin=3.0 if high else 0.0, entropy_norm=0.3,
                                     seq_logprob=-0.4, gen_len=5, kl=0.1, top1=0.4)
            correct[name] = 1 if high else 0
        rows.append({"actions": actions, "correct": correct})
    model = fit_v2_cal(rows)
    assert model["n_params"] == 7 and all(math.isfinite(w) for w in model["weights"])
    query = {name: _action("same", margin=0.0, entropy_norm=0.3, seq_logprob=-0.4, gen_len=5, kl=0.1, top1=0.4)
             for name in V2_ACTIONS}
    # static_text was the low-margin, label-0 action in training; only its features change here
    query["static_text"] = _action("same", margin=3.0, entropy_norm=0.3, seq_logprob=-0.4, gen_len=5, kl=0.1, top1=0.4)
    picked = select_v2_cal(model, query)
    assert picked["action"] == "static_text"
    assert picked["answer"] == "same"
    assert picked["probs"]["static_text"] > picked["probs"]["direct"]
    assert picked["probs"]["static_text"] > picked["probs"]["static_mixed"]
    direct_feats = action_features("direct", query["direct"])
    assert direct_feats.shape == (6,)
    assert direct_feats[0] == pytest.approx(0.0)
    assert direct_feats[4] == pytest.approx(1.0)
    text_feats = action_features("static_text", {**query["static_text"], "direct_answer": "same"})
    assert text_feats[0] == pytest.approx(3.0)
    assert text_feats[4] == pytest.approx(1.0)
    missing = _action("cat", margin=0.3, seq_logprob=-2.0, gen_len=0, kl=None)
    del missing["retriever"]
    bare = action_features("direct", missing)
    assert bare[2] == pytest.approx(-2.0)  # gen_len 0 uses a denominator of 1
    assert bare[3] == pytest.approx(0.0)
    assert bare[5] == pytest.approx(0.0)


def test_gold_in_topk_does_not_change_the_decision():
    actions = {
        "direct": _action("wrong direct", margin=0.1, gold=False),
        "static_text": _action("right answer", margin=2.5, gold=False),
        "static_image": _action("right answer", margin=1.5, gold=True),
        "static_mixed": _action("something else", margin=0.2, gold=False),
        "rerank_static": _action("nope", margin=0.2, gold=True),
    }
    flipped = copy.deepcopy(actions)
    for rec in flipped.values():
        rec["retriever"]["gold_in_topk"] = not rec["retriever"]["gold_in_topk"]
        rec["retriever"]["top1_top2_margin"] = 9.0
    assert select_v2_rule(actions, 0.5, 1.0) == select_v2_rule(flipped, 0.5, 1.0)
    base = action_features("static_text", {**actions["static_text"], "direct_answer": actions["direct"]["answer"]})
    other = action_features("static_text", {**flipped["static_text"], "direct_answer": flipped["direct"]["answer"]})
    assert np.array_equal(base, other)

    rows = []
    for high_names, gold in ((("direct", "static_text"), False), (("static_image",), True)):
        built, correct = {}, {}
        for name in V2_ACTIONS:
            high = name in high_names
            built[name] = _action("same", margin=3.0 if high else 0.0, gold=gold)
            correct[name] = 1 if high else 0
        rows.append({"actions": built, "correct": correct})
    # repeat so the fit is not a single row
    rows = rows * 8
    model = fit_v2_cal(rows)
    first = select_v2_cal(model, actions)
    second = select_v2_cal(model, flipped)
    assert first["action"] == second["action"]
    assert first["probs"].keys() == second["probs"].keys()
    for name in first["probs"]:
        assert first["probs"][name] == pytest.approx(second["probs"][name])

    records = _perfect_records()
    changed = copy.deepcopy(records)
    for rec in changed:
        for action in rec["actions"].values():
            action["retriever"]["gold_in_topk"] = not action["retriever"]["gold_in_topk"]
    assert select_v2(records, delta=0.1)["v2"] == select_v2(changed, delta=0.1)["v2"]


def test_select_v2_test_stage_reads_the_frozen_file(tmp_path):
    path = tmp_path / "dev_decisions.json"
    frozen = {"phase": 2, "v2": {"name": "best_single", "action": "direct", "fallback": True}}
    path.write_text(json.dumps(frozen), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    loaded = frozen_v2(path)
    assert loaded == frozen["v2"]
    out = dev_main(["--stage", "test", "--output", str(path)])
    assert out == {"v2": frozen["v2"]}
    assert path.read_text(encoding="utf-8") == before
    with pytest.raises(SystemExit, match="does not search"):
        frozen_v2({"phase": 2})
    # a synthetic calibration file goes through the mechanical step and lands under "v2"
    records = _perfect_records()
    record_path = tmp_path / "calib.jsonl"
    record_path.write_text("".join(json.dumps(rec) + "\n" for rec in records), encoding="utf-8")
    written = tmp_path / "out.json"
    result = dev_main(["--v2-records", str(record_path), "--output", str(written), "--delta", "0.1"])
    assert result["v2"]["name"] == "margin_tau0.25_theta1"
    assert json.loads(written.read_text(encoding="utf-8"))["v2"]["name"] == "margin_tau0.25_theta1"

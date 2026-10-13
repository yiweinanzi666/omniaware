"""The first-answer-token signals: answer-first WebQA format, answer-slot location, U_omni at the answer token, the
route family of v2-Rule, and the format / measurement-position analysis."""
import json
import math

import pytest
import torch

from omniaware import answers, arbitration, prompts, run, signals
from omniaware.actions import DIRECT, EVIDENCE_ACTIONS, V2_ACTIONS
from omniaware.qwen_eab import predictive_entropy, step_signals
from scripts.analysis import answer_slot as slot_cli
from tests import test_v2_prompts

fake_run = test_v2_prompts.fake_run          # the fake model and assets of the v2 run tests (a fixture)

LOGITS = torch.tensor([3.0, 2.0] + [0.0] * 10)


# ---------------------------------------------------------------- format and parsing

def test_webqa_default_instruction_is_answer_first_and_the_sentence_style_is_the_ablation():
    assert prompts.answer_format("webqa_full", "default") == "answer_first"
    assert prompts.answer_format("webqa_open_img", "default") == "answer_first"
    assert prompts.answer_format("webqa_full", "sentence") == "plain"
    assert prompts.answer_format("okvqa_gs", "default") == "plain"
    style = prompts.instruction_for_style("webqa", "sentence", cite_instruction="c", short_instruction="s")
    assert style["instruction"] == prompts.SENTENCE_INSTRUCTION and style["answer_format"] == "plain"
    default = prompts.instruction_for_style("webqa", "default", cite_instruction="c", short_instruction="s")
    assert default["answer_format"] == "answer_first" and "first line" in default["instruction"]


def test_answer_first_output_scores_the_sentence_and_keeps_the_short_answer():
    parsed = answers.parse_answer("Red\nThe car in the picture is red.\nSources: [E1]", answer_first=True)
    assert parsed["short"] == "Red" and parsed["answer"] == "The car in the picture is red."
    assert parsed["answer_first_ok"] is True and parsed["answer_empty"] is False
    labelled = answers.parse_answer("Answer: No\nSentence: No, the tower is not taller.", answer_first=True)
    assert labelled["short"] == "No" and labelled["answer"] == "No, the tower is not taller."


def test_a_missing_sentence_line_falls_back_to_the_short_answer_and_is_flagged():
    parsed = answers.parse_answer("Red", answer_first=True)
    assert parsed["answer"] == "Red" and parsed["short"] == "Red" and parsed["answer_first_ok"] is False


def test_a_reversed_order_is_not_answer_first():
    parsed = answers.parse_answer("The car in the picture is red.\nRed", answer_first=True)
    assert parsed["answer_first_ok"] is False


def test_a_refusal_on_either_line_is_a_refusal():
    for text in ("I don't know.\nThe car is red.", "Red\nI cannot determine the answer from the provided image."):
        parsed = answers.parse_answer(text, answer_first=True)
        assert parsed["answer_empty"] and parsed["refusal"] and parsed["answer"] == ""


def test_a_plain_output_keeps_every_line_and_reports_no_format():
    parsed = answers.parse_answer("Red\nThe car is red.")
    assert parsed["answer"] == "Red\nThe car is red." and parsed["short"] == parsed["answer"]
    assert parsed["answer_first_ok"] is None


def test_the_record_format_decides_how_the_empty_kinds_parse_the_preds():
    record = {"answer_format": "answer_first", "preds": {"direct": "Unknown\nI cannot answer this question."}}
    assert answers.question_empty_kinds(record) == [("direct", "refusal")]


# ---------------------------------------------------------------- answer slot

@pytest.mark.parametrize("tokens,question,answer_first,expected", [
    (["Red", "\n", "The", " car", " is", " red"], "What color is the car?", True, 0),
    (["The", " Eiff", "el", " Tower", "\n", "It"], "Which landmark is this?", True, 1),
    (["Answer", ":", " No", "\n", "No"], "Is it taller?", True, 2),
    (["Red"], "Is the car red or blue?", True, 0),                  # a choice answer repeats a question word
    (["The", " car", " is", " red", "."], "What color is the car?", False, 3),
    (["No", ",", " the", " car"], "Is the car red?", False, 0),
    (["The", " the"], "q", False, 0),                                # nothing answer-bearing: the first position
    ([], "q", True, 0),
])
def test_answer_slot_finds_the_first_answer_bearing_token(tokens, question, answer_first, expected):
    assert signals.answer_slot(tokens, question, answer_first) == expected


def test_step_signals_match_the_first_token_entropy_and_skip_padded_rows():
    logits = torch.tensor([3.0, 2.0, 0.5, -1.0, 9.0])                 # the last row is padding (vocab 4)
    entropy, margin = step_signals(logits, 4)
    assert entropy == pytest.approx(predictive_entropy(logits, 4))
    assert margin == pytest.approx(1.0)


def test_with_answer_token_reads_the_slot_and_forms_u_omni():
    base = signals.action_record("Red\nThe car is red.", LOGITS, [1, 2, 3], [-.1, -.2, -.3], 12, None,
                                 {"top1_score": None}, [], answer_first=True)
    rec = signals.with_answer_token(base, [0.4, 0.1, 0.2], [0.5, 3.0, 2.0], ["The", " Eiff", "el"], "Which?", True,
                                    h_grd=0.6, beta=0.3)
    slot = rec["answer_token"]
    assert slot["index"] == 1 and slot["text"] == " Eiff" and slot["steps_recorded"] is True
    assert slot["u_raw"] == pytest.approx(0.6 + 0.3 * 0.1) and slot["margin"] == pytest.approx(3.0)
    bare = signals.with_answer_token(base, None, None, None, "Which?", True, h_grd=0.0, beta=0.3)
    assert bare["answer_token"]["index"] == 0 and bare["answer_token"]["steps_recorded"] is False
    assert bare["answer_token"]["entropy_norm"] == pytest.approx(base["first_token"]["entropy_norm"])


# ---------------------------------------------------------------- selector

def _act(u, *, available=True, route=None, answer="a"):
    rec = {"answer": answer, "short": answer, "available": available, "kl_to_direct": 0.1,
           "first_token": {"margin": 1.0, "entropy": 0.5, "entropy_norm": 0.5},
           "answer_token": {"margin": 1.0, "entropy_norm": 0.5, "u_raw": u}, "retriever": {"top1_score": 0.3}}
    if route is not None:
        rec["route"] = route
    return rec


def _route(*modalities):
    return {"mode": "attention", "modalities": list(modalities), "statistics": {}}


def test_u_omni_is_the_pre_registered_confidence_and_lower_u_is_more_confident():
    assert arbitration.PRIMARY_CONFIDENCE == "u_omni"
    decision = {"kind": "rule", "confidence": "u_omni", "tau": arbitration.encode_json_float(-0.5),
                "evidence_choice": "best", "evidence_ranking": list(EVIDENCE_ACTIONS),
                "action_ranking": list(V2_ACTIONS)}
    sure = {DIRECT: _act(0.2), "static_mixed": _act(0.9)}
    unsure = {DIRECT: _act(0.8), "static_mixed": _act(0.9)}
    assert arbitration.apply_decision(decision, sure)["action"] == DIRECT          # -0.2 >= -0.5
    assert arbitration.apply_decision(decision, unsure)["action"] == "static_mixed"
    del sure[DIRECT]["answer_token"]
    with pytest.raises(ValueError, match="answer_token"):
        arbitration.apply_decision(decision, sure)


def test_route_choice_follows_the_static_mixed_attention_route():
    decision = {"kind": "rule", "confidence": "u_omni", "tau": arbitration.encode_json_float(math.inf),
                "evidence_choice": "route", "evidence_ranking": ["static_mixed", "static_text", "static_image"],
                "action_ranking": list(V2_ACTIONS)}

    def question(route, image_available=True):
        return {DIRECT: _act(0.5), "static_text": _act(0.5), "static_image": _act(0.5, available=image_available),
                "static_mixed": _act(0.5, route=route)}

    assert arbitration.apply_decision(decision, question(_route("image")))["action"] == "static_image"
    assert arbitration.apply_decision(decision, question(_route("text")))["action"] == "static_text"
    assert arbitration.apply_decision(decision, question(_route("image", "text")))["action"] == "static_mixed"
    skipped = {"mode": "skipped", "modalities": [], "statistics": {}}
    assert arbitration.apply_decision(decision, question(skipped))["action"] == "static_mixed"
    assert arbitration.apply_decision(decision, question(_route("image"), False))["action"] == "static_mixed"


def _calib(n, routable):
    """Direct is right and sure on even questions. On odd ones the evidence decides: image is right where the route
    says image (i % 4 == 1), text where it says text (i % 4 == 3), and static_mixed scores 60 on both. static_mixed
    has the best evidence mean (e*), so "best" reaches 60 where "route" reaches 100."""
    rows = []
    for i in range(n):
        direct_right, image_side = i % 2 == 0, i % 4 == 1
        skipped = {"mode": "skipped", "modalities": [], "statistics": {}}
        route = (_route("image" if image_side else "text") if routable else skipped)
        actions = {DIRECT: _act(0.1 if direct_right else 0.9), "static_text": _act(0.5),
                   "static_image": _act(0.5), "static_mixed": _act(0.5, route=route)}
        odd = not direct_right
        scores = {DIRECT: 100.0 if direct_right else 0.0,
                  "static_text": 100.0 if odd and not image_side else 0.0,
                  "static_image": 100.0 if odd and image_side else 0.0,
                  "static_mixed": 60.0 if odd else 0.0}
        rows.append({"qid": f"q{i}", "actions": actions, "scores": scores})
    return rows


def test_the_route_family_is_tested_only_when_some_question_has_an_attention_route():
    routed = arbitration.select_rule(_calib(200, True))
    assert routed["rule_families"] == ["best", "confident", "route"]
    assert {t["family"] for t in routed["trace"]} >= {"rule:route"}
    assert all(t["level"] == pytest.approx(0.1 / 3) for t in routed["trace"])
    # direct when it is sure, otherwise the routed image evidence, which is right exactly there
    assert not routed["fallback"] and routed["evidence_choice"] == "route" and routed["calib_utility"] > 0
    plain = arbitration.select_rule(_calib(200, False))
    assert plain["rule_families"] == ["best", "confident"]
    assert all(t["level"] == pytest.approx(0.05) for t in plain["trace"])


def test_token_f1_feature_compares_the_short_answers():
    actions = {DIRECT: _act(0.5, answer="x"), "static_mixed": _act(0.5, answer="x")}
    actions[DIRECT].update(answer="The car is red.", short="red")
    actions["static_mixed"].update(answer="The car in the image is red indeed.", short="red")
    feats = arbitration._cal_features("static_mixed", actions, DIRECT, "u_omni")
    assert feats["token_f1_to_star"] == pytest.approx(1.0)


# ---------------------------------------------------------------- run records

@pytest.mark.usefixtures("tiny_reserves")
def test_a_v2_run_records_the_answer_token_and_the_evidence_route(fake_run, tmp_path):
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout", "--n-dev", "4",
              "--image-mode", "clip", "--tag", "slot", "--v2", "--v2-reranker"])
    rows = [json.loads(line) for line in (fake_run / "runs" / "slot" / "okvqa.jsonl").read_text().splitlines()]
    assert rows and all(r["answer_format"] == "plain" for r in rows)
    for row in rows:
        for name, action in row["actions"].items():
            if not action["available"]:
                continue
            slot = action["answer_token"]
            assert math.isfinite(slot["u_raw"]) and slot["beta"] == pytest.approx(0.3)
            assert slot["u_raw"] == pytest.approx(slot["h_grd"] + slot["beta"] * slot["entropy_norm"])
            assert ("route" in action) == (name != DIRECT)


# ---------------------------------------------------------------- analysis

def _run_row(i, ok, first="Red"):
    def action(name):
        return {"available": True, "answer_first_ok": ok if name != "static_image" else True,
                "first_token": {"topk": [{"id": 1, "text": first}]}, "answer_token": {"index": 0}}
    return {"qid": f"q{i}", "answer_format": "answer_first",
            "actions": {name: action(name) for name in (DIRECT, "static_mixed", "static_image")}}


def test_format_check_fails_an_answer_first_run_whose_actions_drop_the_format(tmp_path, capsys):
    good = tmp_path / "good.jsonl"
    good.write_text("".join(json.dumps(_run_row(i, True)) + "\n" for i in range(10)))
    assert slot_cli.main(["format", "--run", str(good)]) == 0
    report = json.loads((tmp_path / "good.answer_format.json").read_text())
    assert report["passed"] and report["actions"][DIRECT]["template_start"] == 0.0
    bad = tmp_path / "bad.jsonl"
    bad.write_text("".join(json.dumps(_run_row(i, i < 5, first="The")) + "\n" for i in range(10)))
    assert slot_cli.main(["format", "--run", str(bad)]) == 1
    report = json.loads((tmp_path / "bad.answer_format.json").read_text())
    assert not report["passed"] and report["actions"][DIRECT]["template_start"] == 1.0
    assert "ANSWER-FIRST FORMAT BELOW THRESHOLD" in capsys.readouterr().err


def test_position_analysis_separates_the_first_token_from_the_answer_token(tmp_path):
    rows = []
    for i in range(40):
        right = i % 2 == 0
        direct = {"available": True, "first_token": {"entropy_norm": 0.1, "margin": 5.0,     # template: no signal
                                                     "topk": [{"id": 1, "text": "The"}]},
                  "answer_token": {"entropy_norm": 0.05 if right else 0.6, "margin": 4.0 if right else 0.2,
                                   "u_raw": 0.1 if right else 0.9}}
        rows.append({"qid": f"q{i}", "metric": "acc_approx", "actions": {DIRECT: direct},
                     "scores": {DIRECT: 100.0 if right else 0.0, "static_mixed": 0.0 if right else 100.0}})
    table = tmp_path / "t.v2_actions.jsonl"
    table.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert slot_cli.main(["position", "--table", str(table)]) == 0
    report = json.loads((tmp_path / "t.v2_actions.answer_slot.json").read_text())
    signals_all = report["all"]["signals"]
    assert signals_all["first_token.neg_entropy"]["auroc_direct_correct"] == pytest.approx(0.5)
    assert signals_all["answer_token.neg_u_omni"]["auroc_direct_correct"] == pytest.approx(1.0)
    assert signals_all["answer_token.neg_u_omni"]["auroc_evidence_better"] == pytest.approx(1.0)
    assert set(report["by_first_token"]) == {"template_start"}

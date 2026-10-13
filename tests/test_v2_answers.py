"""A1: citation-only, blank, and answer-plus-citation parses."""
import json

import pytest

from omniaware.answers import empty_kind, parse_answer, question_empty_kinds, summarize_empty


def test_citation_only_sources_line_is_empty():
    for text in ("Sources: none", "  sources: none  ", "Sources: [E1], [E2]", "Sources:"):
        parsed = parse_answer(text)
        assert parsed["answer_empty"] is True and parsed["citation_only"] is True
        assert parsed["answer"] == ""


def test_citation_only_identifier_line_is_empty():
    parsed = parse_answer("[E1] [E2]\n[E3]")
    assert parsed == {"answer": "", "answer_empty": True, "citation_only": True, "refusal": False}


def test_empty_string_and_whitespace_are_empty_but_not_citation_only():
    for text in ("", "   ", "\n\t"):
        parsed = parse_answer(text)
        assert parsed["answer_empty"] is True and parsed["citation_only"] is False
        assert parsed["answer"] == ""


def test_answer_plus_citation_keeps_the_answer_text():
    parsed = parse_answer("The Eiffel Tower is in Paris.\nSources: [E1]")
    assert parsed["answer_empty"] is False and parsed["citation_only"] is False
    assert parsed["answer"] == "The Eiffel Tower is in Paris."


def test_refusal_is_empty_and_not_a_citation():
    parsed = parse_answer("I don't know.")
    assert parsed["answer_empty"] is True and parsed["citation_only"] is False and parsed["refusal"] is True
    assert parsed["answer"] == ""
    # A real short answer is not a refusal.
    assert parse_answer("none")["answer_empty"] is False
    assert parse_answer("no")["answer_empty"] is False


def test_empty_answer_rate_warning_names_the_rate():
    quiet = summarize_empty([("static", None)] * 3)
    assert quiet["empty_answer_rate"] == 0.0 and "empty_answer_warning" not in quiet
    loud = summarize_empty([("static", "blank")] + [("static", None)] * 99)
    assert loud["empty_answer_rate"] == pytest.approx(0.01)
    assert loud["empty_answer_rate"] > 0.005
    assert "EMPTY ANSWER RATE" in loud["empty_answer_warning"]
    assert f"{loud['empty_answer_rate']:.6f}" in loud["empty_answer_warning"]


def test_summary_round_trips_as_json():
    text = json.dumps(summarize_empty([("static", "blank"), ("direct", None)]))
    assert "empty_answer_rate" in text


def test_question_empty_kinds_cover_preds_and_available_actions_only():
    record = {
        "preds": {"direct": "Sources: none", "static": "Paris", "gated:static": ""},
        "actions": {
            "direct": {"available": True, "answer_empty": False},
            "static_image": {"available": False, "answer_empty": True},
            "static_text": {"available": True, "answer_empty": True},
        },
    }
    assert question_empty_kinds(record) == [("direct", "citation_only"), ("static", None), ("gated:static", "blank"),
                                            ("action:direct", None), ("action:static_text", "blank")]


def test_preds_keep_the_raw_text_when_parsed_separately():
    raw = "Paris\n[E1] [E2]"
    assert question_empty_kinds({"preds": {"static": raw}}) == [("static", None)]
    assert parse_answer(raw)["answer"] == "Paris"


REFUSALS = [
    "I'm sorry, I cannot answer this question.",
    "I cannot determine the answer from the provided image.",
    "Sorry, I can't answer that based on the given information.",
    "I don't know the answer to this question.",
    "There is not enough information to answer the question.",
    "The provided evidence does not contain the answer.",
    "I am unable to answer this question.",
    "Insufficient information",
]
REAL_ANSWERS = [
    "No",
    "None",
    "Yes",
    "I cannot see any person.",
    "I cannot determine the exact number, it is about five.",
    "I can't tell which one, the image shows a red car.",
    "The evidence shows Paris.",
    "No information about the dog is visible.",
]


@pytest.mark.parametrize("text", REFUSALS)
def test_whole_sentence_refusals_are_empty(text):
    parsed = parse_answer(text)
    assert parsed["answer_empty"] is True and parsed["answer"] == ""


@pytest.mark.parametrize("text", REAL_ANSWERS)
def test_real_short_answers_are_not_refusals(text):
    parsed = parse_answer(text)
    assert parsed["answer_empty"] is False and parsed["answer"] == text


def test_refusal_followed_by_a_citation_line_is_still_a_refusal():
    assert parse_answer("I'm sorry, I cannot answer this question.\nSources: [E1]")["answer_empty"] is True


def test_refusal_clause_followed_by_an_answer_is_an_answer():
    text = "I cannot determine the exact number from the image, it is about five."
    assert parse_answer(text)["answer_empty"] is False


def test_refusals_are_reported_apart_and_do_not_raise_the_empty_rate():
    # S1 checks --max-empty-rate on blank and citation-only strings only; a refusal is model behaviour, kept visible.
    record = {"preds": {"direct": "I cannot determine the answer from the provided information.", "static": "Paris"},
              "actions": {"direct": {"available": True, "answer_empty": True, "citation_only": False, "refusal": True},
                          "static_image": {"available": True, "answer_empty": True, "citation_only": False,
                                           "refusal": False}}}
    summary = summarize_empty(question_empty_kinds(record))
    assert summary["n_answers"] == 4
    assert summary["n_refusal"] == 2 and summary["refusal_rate"] == pytest.approx(0.5)
    assert summary["n_empty"] == 1 and summary["n_blank"] == 1 and summary["n_citation_only"] == 0
    assert summary["empty_answer_rate"] == pytest.approx(0.25)
    assert summary["empty_by_arm"]["direct"] == {"n": 1, "blank": 0, "citation_only": 0, "refusal": 1}
    assert summary["empty_by_arm"]["action:static_image"]["blank"] == 1


def test_only_refusals_leave_the_empty_rate_at_zero():
    summary = summarize_empty([("direct", "refusal")] * 10)
    assert summary["empty_answer_rate"] == 0.0 and "empty_answer_warning" not in summary
    assert summary["refusal_rate"] == 1.0


def test_an_action_record_without_the_refusal_field_counts_as_blank():
    assert empty_kind({"answer_empty": True, "citation_only": False}) == "blank"
    assert empty_kind({"answer_empty": False}) is None

import torch

from omniaware import interventions as iv
from omniaware.qwen_eab import Encoded


def _chunk(cid, text, gold=None, modality="text"):
    return {"cid": cid, "modality": modality, "title": cid, "text": text, "gold": gold}


def test_target_is_highest_ranked_text_chunk_with_answer():
    chunks = [_chunk("a", "nothing here"), _chunk("b", "It is in Paris, France."), _chunk("c", "paris again"),
              _chunk("i", "Paris", modality="image")]
    rel = {"a": 0.9, "b": 0.5, "c": 0.7, "i": 1.0}
    assert iv.find_target(chunks, rel, ["Paris"])["cid"] == "c"
    assert iv.answer_span("Comparison", ["paris"]) is None       # word-bounded


def test_counterfactual_and_random_edit_controls():
    c = _chunk("b", "Paris is big. Paris is old and has many museums.")
    cf = iv.counterfactual(c, ["Paris"], "Rome")
    assert "Paris" not in cf["text"] and cf["text"].count("Rome") == 2
    re_ = iv.random_edit(c, ["Paris"], "Rome", "seed")
    assert re_["text"].count("Paris") == 2 and "Rome" in re_["text"]


def test_entity_types_and_substitutes():
    assert iv.entity_type("1999") == "year" and iv.entity_type("12") == "number" and iv.entity_type("Yes") == "yesno"
    pool = iv.substitute_pool([{"qid": "q1", "golds": ["red"]}, {"qid": "q2", "golds": ["blue"]},
                               {"qid": "q3", "golds": ["1990"]}])
    assert iv.pick_substitute("q1", ["red"], "red", pool) == "blue"
    assert iv.pick_substitute("q3", ["1990"], "1990", pool) is None


def test_distractors_keep_target_and_positions():
    e = {"candidates": [_chunk(c, f"text {c}", gold=(c == "g")) for c in "abcdefgh"]}
    e["candidates"][5]["text"] = "contains Paris"        # 'f' has the answer, must not be a distractor
    rel = {c: 1 - i / 10 for i, c in enumerate("abcdefgh")}
    chunks = [c for c in e["candidates"] if c["cid"] in "abcd"]
    new, extra = iv.distractors(e, chunks, rel, "b", ["Paris"])
    assert extra == ["e", "h"]                          # 'f' has the answer, 'g' is a gold fact
    assert [c["cid"] for c in new] == ["a", "b", "h", "e"]   # d, c (lowest non-target) swapped in place


def test_inject_moves_to_front_under_rank_order():
    e = {"question": "What is the capital?"}
    chunks = [_chunk("a", "x", gold=True), _chunk("b", "y"), _chunk("c", "z")]
    new, rel = iv.inject(e, chunks, {"a": 0.9, "b": 0.5, "c": 0.2}, ["Paris"], "Rome", "inject_answer", "rank")
    assert new[0]["cid"] == iv.INJECT_ID and rel[iv.INJECT_ID] == 1.0 and [c["cid"] for c in new][1:] == ["a", "b"]
    assert iv.matches("I think it is Rome.", "Rome") and not iv.matches("Romeo", "Rome")


def test_mask_keeps_positions_and_noise_is_clipped_and_seeded():
    enc = Encoded(torch.arange(10)[None], torch.zeros(1, 10, dtype=torch.long), None, None, {"c": (3, 6)}, 8, 10)
    m = iv.mask_tokens(enc, "c", 0)
    assert m.input_ids[0, 3:6].tolist() == [0, 0, 0] and m.input_ids[0, 6] == 6 and enc.input_ids[0, 3] == 3
    n1 = iv.relevance_noise({"a": 0.99, "b": 0.01}, 0.5, "s")
    assert n1 == iv.relevance_noise({"b": 0.01, "a": 0.99}, 0.5, "s") and all(0 <= v <= 1 for v in n1.values())

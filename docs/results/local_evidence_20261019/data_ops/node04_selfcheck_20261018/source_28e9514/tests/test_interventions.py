import torch

from omniaware import interventions as iv
from omniaware import pipeline
from omniaware.qwen_eab import Encoded


def _chunk(cid, text, gold=None, modality="text"):
    return {"cid": cid, "modality": modality, "title": cid, "text": text, "gold": gold}


def test_target_is_highest_ranked_text_chunk_with_answer():
    chunks = [_chunk("a", "nothing here"), _chunk("b", "It is in Paris, France."), _chunk("c", "paris again"),
              _chunk("i", "Paris", modality="image")]
    rel = {"a": 0.9, "b": 0.5, "c": 0.7, "i": 1.0}
    assert iv.find_target(chunks, rel, ["Paris"])["cid"] == "c"
    assert iv.answer_span("Comparison", ["paris"]) is None       # word-bounded


def test_answers_are_searched_where_the_prompt_reads_them():
    # the 160-word cut of pipeline._trim: a gold after the cut is not in the prompt, so it is not a target
    late = " ".join(["filler"] * 170) + " Paris"
    chunks = [_chunk("late", late), _chunk("early", "Paris is in France.")]
    rel = {"late": 1.0, "early": 0.1}
    assert iv.answer_span(iv.prompt_text(chunks[0]), ["Paris"]) is None
    assert iv.find_target(chunks, rel, ["Paris"])["cid"] == "early"


def test_prompt_text_is_what_build_segments_writes():
    e = {"query_images": [], "question": "Where?", "dataset": "okvqa"}
    chunks = [_chunk("t", " ".join(f"w{i}" for i in range(200))),
              {"cid": "tb", "modality": "table", "title": "", "text": "| a | b |", "gold": None},
              {"cid": "im", "modality": "image", "title": "cap", "text": " ".join(["cap"] + ["x"] * 70),
               "image": "x.png", "gold": None},
              {"cid": "v", "modality": "video", "title": "", "text": " ".join(["spoken"] * 170), "frames": [],
               "gold": None}]
    segments, _ = pipeline.build_segments(e, chunks)
    prompt = "".join(s.text or "" for s in segments)
    for c in chunks:
        if c["modality"] == "image":   # title and caption are written as two pieces of the prompt
            assert c["title"] in prompt and pipeline._trim(c["text"], 60) in prompt
        else:
            assert iv.prompt_text(c) in prompt


def test_counterfactual_and_random_edit_controls():
    c = _chunk("b", "Paris is big. Paris is old and has many museums.")
    cf = iv.counterfactual(c, ["Paris"], "Rome")
    assert "Paris" not in cf["text"] and cf["text"].count("Rome") == 2
    re_ = iv.random_edit(c, ["Paris"], "Rome", "seed")
    assert re_["text"].count("Paris") == 2 and "Rome" in re_["text"]


def test_counterfactual_replaces_every_gold_alias_case_insensitively():
    c = _chunk("b", "New York and NYC; new york city; nyc again.")
    cf = iv.counterfactual(c, ["New York", "NYC"], "Rome")
    assert "new york" not in cf["text"].lower() and "nyc" not in cf["text"].lower()
    assert cf["text"] == "Rome and Rome; Rome city; Rome again."


def test_random_edit_replaces_whole_words_of_answer_length_and_spares_golds():
    text = "I love New York city and Boston"
    c = _chunk("b", text)
    words = text.split()
    for seed in range(30):
        out = iv.random_edit(c, ["New York"], "Rome", f"s{seed}")
        assert out is not None and "New York" in out["text"]           # the gold occurrence is untouched
        new = out["text"].split()
        # exactly one run of two whole words (the answer's length) became one entity word
        assert any(new == words[:i] + ["Rome"] + words[i + 2:] for i in range(len(words) - 1))


def test_random_edit_fails_when_no_window_of_answer_length_avoids_the_gold():
    assert iv.random_edit(_chunk("b", "New York"), ["New York"], "Rome", "s") is None


def test_answer_span_order_does_not_depend_on_set_order():
    # equal length: the lexicographically first gold is chosen whatever the order of the input list
    assert iv.answer_span("ab cd", ["cd", "ab"]) == iv.answer_span("ab cd", ["ab", "cd"]) == (0, 2, "ab")
    # longer gold first
    assert iv.answer_span("New York", ["New", "New York"])[2] == "New York"


def test_yes_no_entities_are_judged_on_the_whole_answer():
    assert not iv.matches("there is no way", "no")
    assert iv.matches("No, it is not.", "no") and iv.matches("no", "no") and iv.matches("Yes", "yes")
    assert not iv.matches("the answer is yes", "yes")
    assert iv.matches("I think it is Rome.", "Rome") and not iv.matches("Romeo", "Rome")


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


def test_mask_span_keeps_token_ids_and_positions_and_noise_is_clipped_and_seeded():
    enc = Encoded(torch.arange(10)[None], torch.zeros(1, 10, dtype=torch.long), None, None, {"c": (3, 6)}, 8, 10)
    m = iv.mask_span(enc, "c")
    assert m.blocked == {"c"} and enc.blocked == frozenset()
    assert torch.equal(m.input_ids, enc.input_ids) and m.spans == enc.spans
    n1 = iv.relevance_noise({"a": 0.99, "b": 0.01}, 0.5, "s")
    assert n1 == iv.relevance_noise({"b": 0.01, "a": 0.99}, 0.5, "s") and all(0 <= v <= 1 for v in n1.values())

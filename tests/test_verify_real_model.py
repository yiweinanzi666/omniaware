"""scripts/verify_real_model.py on the tiny random models (the parts that need no processor or checkpoint)."""
import pytest

from omniaware.pipeline import Variant
from scripts import verify_real_model as vrm

from eab_fixtures import encode, make_internvl, make_qwen


def _spec(q, enc, rows, gamma=3.0):
    return Variant(rows, gamma, rows=rows, form="pos", gate=False).spec(enc, {"c1": 1.0, "c2": 0.2}, 1.0, 0.0,
                                                                         q.n_layers)


def test_token_comparison_compares_image_pads_as_runs():
    ours = [1, 2, 9, 9, 9, 3]                                 # 9 = image pad
    assert vrm.compare_token_sequences(ours, [1, 2, 9, 9, 9, 3], 9)["ok"]
    more_pads = vrm.compare_token_sequences(ours, [1, 2, 9, 9, 9, 9, 3], 9)
    assert more_pads["ok"] and not more_pads["image_runs_equal"]     # pad count is reported, not a mismatch
    wrong = vrm.compare_token_sequences(ours, [1, 2, 9, 4, 3], 9)
    assert not wrong["ok"] and wrong["first_mismatch"] == 3
    assert wrong["ours_token"] == 3 and wrong["theirs_token"] == 4
    truncated = vrm.compare_token_sequences(ours, [1, 2, 9, 9, 9], 9)     # collapsed: [1, 2, -1] vs [1, 2, -1, 3]
    assert not truncated["ok"] and truncated["first_mismatch"] == 3 and truncated["theirs_token"] is None


def test_segmented_prefill_matches_one_forward_on_tiny_model():
    q = make_qwen()
    result = vrm.segmented_check(q, encode(q))
    assert result["ok"] and result["max_abs_diff"] < 1e-4 and result["argmax_equal"]


def test_reference_check_covers_every_arm_on_tiny_model():
    q = make_qwen()
    enc = encode(q)
    arms = {"ans1": _spec(q, enc, "ans1"), "ans2": _spec(q, enc, "ans2")}
    result = vrm.reference_check(q, enc, arms, steps=4)
    assert result["ok"] and set(result["arms"]) == {"static", "ans1", "ans2"}
    for arm in result["arms"].values():
        assert arm["t_end_attention_max_abs_diff"] < 1e-4 and arm["first_logits_max_abs_diff"] < 1e-4


def test_reference_check_covers_native_internvl_path():
    q = make_internvl()
    enc = encode(q)
    result = vrm.reference_check(q, enc, {"ans1": _spec(q, enc, "ans1")}, steps=3)
    assert result["ok"], result


def test_statistics_report_modalities_and_group0_change():
    q = make_qwen()
    examples = [{"qid": "a", "dataset": "okvqa", "question": "Which one?", "query_images": [], "chunks": [
        {"cid": "c1", "modality": "text", "title": "T", "text": "alpha beta gamma delta epsilon"},
        {"cid": "c2", "modality": "table", "title": "", "text": "x|y\n1|2", "r": 0.5}]}]
    rows = vrm.statistics_rows(q, examples, gamma=3.0)
    assert set(rows[0]["modality"]) == {"text", "table"}
    assert all(0.0 <= m["H"] <= 1.0 and 0.0 < m["M"] <= 1.0 for m in rows[0]["modality"].values())
    # A positive bias on the chunks moves attention mass off the non-chunk positions (group 0), strictly.
    assert rows[0]["group0_mass_eab"] < rows[0]["group0_mass_static"]
    summary = vrm.summarize_statistics(rows)
    assert summary["per_modality_H"]["text"]["n"] == 1 and summary["group0_mass_change"]["n"] == 1
    assert summary["group0_mass_change"]["mean"] < 0


def test_default_examples_without_an_image_have_text_evidence_only():
    (example,) = vrm.default_examples(None)
    assert [c["modality"] for c in example["chunks"]] == ["text"] and example["query_images"] == []


@pytest.mark.parametrize("ids,expected", [([5, 7, 7, 7], ([5, -1], [3])), ([], ([], []))])
def test_collapse_image_runs(ids, expected):
    assert vrm.collapse_image_runs(ids, 7) == expected

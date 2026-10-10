import pytest

from omniaware import data
from omniaware.data import candidate_shortfall, check_disjoint, image_subset, is_image_question, split_examples


def _examples(prefix, n, **extra):
    return [{"qid": f"{prefix}{i:04d}", **extra} for i in range(n)]


def _ids(part):
    return [e["qid"] for e in part]


def test_heldout_calibration_is_disjoint_and_never_evaluated():
    sp = split_examples(_examples("v", 1000), n_calib=10, n_dev=20, reserve=100)
    ids = {k: {e["qid"] for e in sp[k]} for k in ("calib", "dev", "test")}
    assert [len(ids[k]) for k in ("calib", "dev", "test")] == [10, 20, 880]
    assert not ids["calib"] & (ids["dev"] | ids["test"])
    assert not ids["dev"] & ids["test"]
    assert {e["qid"] for e in sp["eval"]} == ids["dev"] | ids["test"]


def test_test_set_and_dev_set_do_not_depend_on_the_calibration_size():
    # S-3: the calibration-size ablation may change calibration only.
    universe = _examples("v", 5046)                                    # OK-VQA val
    splits = [split_examples(universe, n_calib=n, n_dev=data.N_DEV["okvqa"], reserve=data.CALIB_RESERVE["okvqa"])
              for n in (100, 250, 500)]
    assert _ids(splits[0]["test"]) == _ids(splits[1]["test"]) == _ids(splits[2]["test"])
    assert _ids(splits[0]["dev"]) == _ids(splits[1]["dev"]) == _ids(splits[2]["dev"])
    assert len(splits[0]["test"]) == 5046 - 500 - 256
    assert set(_ids(splits[0]["calib"])) < set(_ids(splits[2]["calib"]))   # a prefix of the same reserve


def test_calibration_beyond_the_reserve_is_refused():
    with pytest.raises(ValueError, match="reserve"):
        split_examples(_examples("v", 1000), n_calib=501, n_dev=10, reserve=500)
    with pytest.raises(ValueError, match="reserve"):
        split_examples(_examples("v", 1000), n_calib=0, n_dev=10, reserve=500)


def test_videorag_protocol_leaves_a_nonempty_evaluation_set():
    # F-1: 374 VideoRAG-Synth questions used to leave dev and test empty.
    sp = split_examples(_examples("v", 374), n_calib=data.CALIB_RESERVE["videorag_synth"],
                        n_dev=data.N_DEV["videorag_synth"], reserve=data.CALIB_RESERVE["videorag_synth"])
    assert [len(sp[k]) for k in ("calib", "dev", "test")] == [100, 50, 224]


def test_empty_evaluation_set_is_an_error():
    with pytest.raises(ValueError, match="test.*empty"):
        split_examples(_examples("v", 374), n_calib=100, n_dev=274, reserve=100)


def test_dev_larger_than_the_universe_is_an_error():
    with pytest.raises(ValueError, match="dev split needs"):
        split_examples(_examples("v", 100), n_calib=10, n_dev=60, reserve=50)


def test_disjointness_check_rejects_an_overlap():
    with pytest.raises(ValueError, match="dev and test share"):
        check_disjoint({"calib": [{"qid": "a"}], "dev": [{"qid": "b"}], "test": [{"qid": "b"}]})


def test_training_pool_calibration_and_dev_come_from_train_and_test_is_all_of_val():
    # D2 (WebQA): calibration and dev from the training split; every validation question is test.
    val = _examples("v", 300)
    sp = split_examples(val, n_calib=50, n_dev=40, calib_pool=_examples("t", 200))
    assert all(e["qid"].startswith("t") for e in sp["calib"] + sp["dev"])
    assert len(sp["calib"]) == 50 and len(sp["dev"]) == 40
    assert set(_ids(sp["test"])) == {e["qid"] for e in val}
    assert not {e["qid"] for e in sp["calib"]} & {e["qid"] for e in sp["dev"]}


def test_training_pool_test_set_is_independent_of_the_calibration_size():
    val, train = _examples("v", 300), _examples("t", 1000)
    tests = [_ids(split_examples(val, n_calib=n, n_dev=256, calib_pool=train)["test"]) for n in (100, 250, 500)]
    assert tests[0] == tests[1] == tests[2]


def test_split_is_deterministic_and_order_independent():
    a = split_examples(_examples("v", 40), n_calib=5, n_dev=5, reserve=5)
    b = split_examples(list(reversed(_examples("v", 40))), n_calib=5, n_dev=5, reserve=5)
    assert _ids(a["test"]) == _ids(b["test"])


def test_image_subsets_are_cut_from_the_full_split():
    # S-1: webqa-img must use the same partition as webqa-full, restricted to image questions.
    val = [{"qid": f"v{i}", "gold_modalities": ["image", "text"] if i % 3 == 0 else (["image"] if i % 2 else ["text"])}
           for i in range(300)]
    train = [{"qid": f"t{i}", "gold_modalities": ["image"] if i % 2 else ["text"]} for i in range(300)]
    full = split_examples(val, n_calib=40, n_dev=30, calib_pool=train)
    img = image_subset(full)
    for part in ("calib", "dev", "test"):
        assert _ids(img[part]) == [e["qid"] for e in full[part] if is_image_question(e)]
    assert not {e["qid"] for e in full["dev"]} & {e["qid"] for e in img["test"]}
    assert all(is_image_question(e) for e in img["test"]) and img["test"]


def test_candidate_shortfall_counts_short_questions_and_modality_gaps():
    text = {"modality": "text"}
    examples = [
        {"qid": "a", "fixed_evidence": False, "candidates": [dict(text)] * 3},                         # short, no image
        {"qid": "b", "fixed_evidence": False, "candidates": [dict(text)] * 5 + [{"modality": "image"}]},
        {"qid": "c", "fixed_evidence": True, "candidates": [dict(text)] * 2},                            # not short
    ]
    rep = candidate_shortfall(examples, k=5)
    assert rep["fewer_than_k"]["count"] == 1 and rep["fewer_than_k"]["qids"] == ["a"]
    assert rep["without_modality"] == {"image": {"count": 2, "qids": ["a", "c"]}}


def test_protocol_sizes_resolve_per_family():
    assert data.protocol_family("okvqa_gs") == data.protocol_family("okvqa_dense") == "okvqa"
    assert data.protocol_family("webqa_open_img") == "webqa"
    assert data.protocol_family("videorag_synth") == "videorag_synth"
    assert data.CALIB_RESERVE["videorag_synth"] == 100 and data.N_DEV["videorag_synth"] == 50

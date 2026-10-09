from omniaware.data import split_examples


def _examples(prefix, n):
    return [{"qid": f"{prefix}{i:04d}"} for i in range(n)]


def test_heldout_calibration_is_disjoint_and_never_evaluated():
    sp = split_examples(_examples("v", 100), n_calib=10, n_dev=20)
    ids = {k: {e["qid"] for e in sp[k]} for k in ("calib", "dev", "test")}
    assert [len(ids[k]) for k in ("calib", "dev", "test")] == [10, 20, 70]
    assert not ids["calib"] & (ids["dev"] | ids["test"])
    assert not ids["dev"] & ids["test"]
    assert {e["qid"] for e in sp["eval"]} == ids["dev"] | ids["test"]


def test_training_pool_calibration_keeps_every_eval_question():
    sp = split_examples(_examples("v", 50), n_calib=8, n_dev=10, calib_pool=_examples("t", 30))
    assert all(e["qid"].startswith("t") for e in sp["calib"]) and len(sp["calib"]) == 8
    assert len(sp["dev"]) + len(sp["test"]) == 50


def test_split_is_deterministic_and_order_independent():
    a = split_examples(_examples("v", 40), n_calib=5, n_dev=5)
    b = split_examples(list(reversed(_examples("v", 40))), n_calib=5, n_dev=5)
    assert [e["qid"] for e in a["test"]] == [e["qid"] for e in b["test"]]

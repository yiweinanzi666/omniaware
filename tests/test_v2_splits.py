"""A6: WebQA calib/dev/test are disjoint, test is all val, an empty eval split raises."""
import pytest

from omniaware import data
from omniaware.data import check_disjoint, split_examples
from omniaware.runio import qid_set_sha256


def _examples(prefix, n):
    return [{"qid": f"{prefix}{i:04d}"} for i in range(n)]


def test_webqa_protocol_defaults_and_full_val_is_the_test_set():
    assert data.N_CALIB_TRAIN["webqa"] >= 500
    assert data.N_DEV["webqa"] == 256
    assert data.WEBQA_VAL_SIZE == 4966
    val = _examples("v", data.WEBQA_VAL_SIZE)
    train = _examples("t", 2000)
    sp = split_examples(val, n_calib=data.N_CALIB_TRAIN["webqa"], n_dev=data.N_DEV["webqa"],
                        calib_pool=train, reserve=data.TRAIN_RESERVE["webqa"])
    assert len(sp["calib"]) == 500 and len(sp["dev"]) == 256 and len(sp["test"]) == 4966
    assert {e["qid"] for e in sp["test"]} == {e["qid"] for e in val}
    assert all(e["qid"].startswith("t") for e in sp["calib"] + sp["dev"])
    check_disjoint(sp)
    hashes = {part: qid_set_sha256(e["qid"] for e in sp[part]) for part in ("calib", "dev", "test")}
    assert len(set(hashes.values())) == 3
    assert hashes["test"] == qid_set_sha256(e["qid"] for e in val)


def test_smaller_fixture_is_not_required_to_be_4966():
    val, train = _examples("v", 40), _examples("t", 80)
    sp = split_examples(val, n_calib=10, n_dev=8, calib_pool=train)
    assert len(sp["test"]) == 40
    check_disjoint(sp)


def test_empty_test_split_raises_including_videorag_sized_pools():
    # F-1: a 374-question pool cannot give calib 500 and a non-empty test. The held-out protocol
    # uses a 100-question reserve so the test split is non-empty, and an empty test raises.
    with pytest.raises(ValueError, match="empty"):
        split_examples(_examples("v", 374), n_calib=100, n_dev=274, reserve=100)
    sp = split_examples(_examples("v", 374), n_calib=data.CALIB_RESERVE["videorag_synth"],
                        n_dev=data.N_DEV["videorag_synth"], reserve=data.CALIB_RESERVE["videorag_synth"])
    assert sp["test"] and sp["dev"] and sp["calib"]
    check_disjoint(sp)


def test_pairwise_overlap_raises():
    with pytest.raises(ValueError, match="share"):
        check_disjoint({"calib": [{"qid": "a"}], "dev": [{"qid": "b"}], "test": [{"qid": "a"}]})

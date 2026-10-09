import pytest

from omniaware.baselines import universalrag


def test_label_mapping_drops_unavailable_and_no():
    r = universalrag.to_route(["paragraph", "image", "video"], {"text", "image"})
    assert r["retrieve"] and r["modalities"] == ["image", "text"]
    r = universalrag.to_route(["no"], {"text", "image"})
    assert not r["retrieve"] and r["modalities"] == []
    with pytest.raises(ValueError):
        universalrag.to_route(["audio"], {"text"})


def test_training_overlap_uses_upstream_index():
    ex = [{"dataset": "webqa", "qid": "abc", "question": "q?", "query_images": []}]
    assert universalrag.export_queries(ex) == [{"index": "webqa_abc", "question": "q?"}]
    train = [{"index": "webqa_abc"}, {"index": "webqa_zzz"}, {"index": "nq_1"}]
    assert universalrag.overlapping(train, ex) == {"webqa_abc"}

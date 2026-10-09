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


def test_router_labels_parse_like_upstream_and_suite_ids_are_kept():
    from omniaware.baselines import universalrag as ur
    assert ur.parse_labels("table+paragraph") == ["table", "paragraph"]
    assert ur.parse_labels(["no", "image"]) == ["image"] and ur.parse_labels("error") == ["no"]
    assert ur.parse_labels(None) == ["no"]
    assert ur.upstream_index({"dataset": "hybridqa", "qid": "hybridqa_abc"}) == "hybridqa_abc"
    assert ur.upstream_index({"dataset": "webqa", "qid": "d5b"}) == "webqa_d5b"

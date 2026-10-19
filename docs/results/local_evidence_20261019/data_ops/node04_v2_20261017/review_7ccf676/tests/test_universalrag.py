import subprocess
import sys
from pathlib import Path

import pytest

from omniaware.baselines import universalrag


def test_label_mapping_drops_unavailable_and_no():
    r = universalrag.to_route(["paragraph", "image", "video"], {"text", "image"})
    assert r["retrieve"] and r["modalities"] == ["image", "text"] and r["fallback"] is False
    r = universalrag.to_route(["no"], {"text", "image"})
    assert not r["retrieve"] and r["modalities"] == [] and r["fallback"] is False
    with pytest.raises(ValueError):
        universalrag.to_route(["audio"], {"text"})


def test_clip_and_video_are_separate_modalities():
    assert universalrag.to_route(["clip"], {"clip", "video"})["modalities"] == ["clip"]
    assert universalrag.to_route(["video"], {"clip", "text"})["modalities"] == []
    assert universalrag.to_route(["clip", "video"], {"video"})["modalities"] == ["video"]


def test_labels_absent_from_the_corpus_do_not_retrieve_and_are_flagged():
    r = universalrag.to_route(["table", "video"], {"text", "image"})
    assert r == {"mode": "query", "rule": "query", "labels": ["table", "video"], "retrieve": False,
                 "modalities": [], "fallback": True}
    r = universalrag.to_route(["paragraph", "video"], {"text"})
    assert r["retrieve"] and r["modalities"] == ["text"] and r["fallback"] is False


def test_training_overlap_uses_upstream_index():
    ex = [{"dataset": "webqa", "qid": "abc", "question": "q?", "query_images": []}]
    assert universalrag.export_queries(ex) == [{"index": "webqa_abc", "question": "q?"}]
    train = [{"index": "webqa_abc"}, {"index": "webqa_zzz"}, {"index": "nq_1"}]
    assert universalrag.overlapping(train, ex) == {"webqa_abc"}


def test_router_labels_parse_like_upstream_and_suite_ids_are_kept():
    from omniaware.baselines import universalrag as ur
    assert ur.parse_labels("table+paragraph") == ["table", "paragraph"]
    assert ur.parse_labels(["no", "image"]) == ["image"] and ur.parse_labels(["clip", "video"]) == ["clip", "video"]
    assert ur.parse_labels(None) == ["no"] and ur.parse_labels([]) == ["no"]
    assert ur.upstream_index({"dataset": "hybridqa", "qid": "hybridqa_abc"}) == "hybridqa_abc"
    assert ur.upstream_index({"dataset": "webqa", "qid": "d5b"}) == "webqa_d5b"


def test_unknown_router_labels_are_errors_not_dropped():
    with pytest.raises(ValueError):
        universalrag.parse_labels(["table", "audio"])
    with pytest.raises(ValueError):
        universalrag.parse_labels("error")
    with pytest.raises(ValueError):
        universalrag.to_route(["error"], {"text"})


def test_universalrag_route_ranks_by_suite_score_not_relevance():
    from omniaware.run import select_chunks
    candidates = [{"cid": "t1", "modality": "text", "suite_score": .2},
                  {"cid": "t2", "modality": "text", "suite_score": .9},
                  {"cid": "i1", "modality": "image", "suite_score": .5}]
    e = {"qid": "q", "candidates": candidates, "fixed_evidence": False}
    rel = {"t1": .9, "t2": .1, "i1": .4}       # Omni-Aware's order would be t1, i1, t2
    route = universalrag.to_route(["paragraph", "image"], {"text", "image"})
    assert [c["cid"] for c in select_chunks(e, rel, 2, "rank", 0, 0, route)] == ["t2", "i1"]
    assert [c["cid"] for c in select_chunks(e, rel, 2, "rank", 0, 0, None)] == ["t1", "i1"]
    with pytest.raises(ValueError):
        select_chunks({**e, "candidates": [{"cid": "t1", "modality": "text"}]}, {"t1": .5}, 1, "rank", 0, 0, route)


def test_route_queries_script_has_a_help_without_the_router_environment():
    script = Path(__file__).resolve().parents[1] / "scripts" / "universalrag_retrieval" / "route_queries.py"
    done = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "--model-path" in done.stdout

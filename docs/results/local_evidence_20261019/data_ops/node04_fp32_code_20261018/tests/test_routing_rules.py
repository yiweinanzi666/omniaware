import pytest
import torch

from omniaware import routing
from scripts.analysis.routing_metrics import routing_metrics


def _attention():
    # text: substantial and diffuse; image: larger mass but concentrated on one token
    attn = torch.tensor([0.0, 0.0] + [0.1] * 4 + [0.55, 0.05])
    spans = {"t1": (2, 6), "i1": (6, 8)}
    return attn, spans, {"t1": "text", "i1": "image"}


@pytest.mark.parametrize("rule,expected", [("mh", "text"), ("mass", "image"), ("entropy", "text")])
def test_attention_rules_rank_by_their_factor(rule, expected):
    attn, spans, mods = _attention()
    route = routing.route_modalities(attn, spans, mods, margin=0.0, rule=rule)
    assert route["modalities"] == [expected] and route["rule"] == rule


def test_retriever_rule_uses_best_calibrated_score_and_margin():
    cands = [{"cid": "t1", "modality": "text"}, {"cid": "t2", "modality": "text"}, {"cid": "i1", "modality": "image"}]
    assert routing.retriever_route(cands, {"t1": 0.2, "t2": 0.9, "i1": 0.7}, margin=0.08)["modalities"] == ["text"]
    assert routing.retriever_route(cands, {"t1": 0.2, "t2": 0.9, "i1": 0.85}, margin=0.08)["modalities"] == ["text", "image"]


def test_wrong_route_and_coverage():
    rows = [
        {"retrieve": True, "gold_modalities": ["image"], "route": {"modalities": ["image"]}},
        {"retrieve": True, "gold_modalities": ["image"], "route": {"modalities": ["text"]}},
        {"retrieve": True, "gold_modalities": ["text"], "route": {"modalities": ["text", "image"]}},
        {"retrieve": True, "gold_modalities": ["text"], "route": {"modalities": []}},      # all modalities
        {"retrieve": False, "gold_modalities": ["text"], "route": {"modalities": ["image"]}},  # not triggered
    ]
    m = routing_metrics(rows, ["image", "text"])
    assert m["triggered"] == 4
    assert m["wrong_route_rate"] == 25.0 and m["coverage_rate"] == 75.0
    assert m["modalities_per_query"] == 1.5


def test_single_supplied_modality_skips_routing():
    """Paper, Modality Routing: when the supplied context covers fewer than two candidate modalities, routing is
    skipped and retrieval searches the full pool (OK-VQA: only the query image, against a text corpus)."""
    from types import SimpleNamespace
    attn = torch.ones(6)
    spans = {"__query__": (0, 4)}
    route = routing.route_modalities(attn, spans, {"__query__": "image"})
    assert route["mode"] == "skipped" and route["modalities"] == [] and route["rule"] == "mh"
    enc = SimpleNamespace(spans=spans)
    assert routing.initial_route(attn, enc, {"query_images": ["q.png"]})["modalities"] == []
    assert routing.evidence_route(attn, enc, [{"cid": "__query__", "modality": "image"}])["mode"] == "skipped"
    # No supplied context at all is still a cold start, which the probe round handles.
    assert routing.initial_route(None, SimpleNamespace(spans={}), {"query_images": []})["mode"] == "cold_start"


def test_okvqa_style_query_image_routes_without_error():
    """With --modality-routing, an OK-VQA question (query image, text-only candidates) used to raise ValueError
    in select_chunks, because the initial route pointed at a modality without candidates."""
    from types import SimpleNamespace
    from omniaware import run
    cands = [{"cid": f"k{j}", "modality": "text", "title": "", "text": f"passage {j}"} for j in range(1, 8)]
    e = {"qid": "q0", "fixed_evidence": False, "candidates": cands, "query_images": ["q.png"]}
    attn = torch.tensor([0.0] * 4 + [0.5] * 2)
    spans = {"__query__": (0, 4), "k1": (4, 6)}
    route = routing.initial_route(attn, SimpleNamespace(spans=spans), e)
    assert route["modalities"] == []
    rel = {c["cid"]: 1.0 - int(c["cid"][1:]) / 10 for c in cands}
    chosen = run.select_chunks(e, rel, 5, "rank", 0, route=route)
    assert [c["cid"] for c in chosen] == ["k1", "k2", "k3", "k4", "k5"]
    # Two supplied modalities still route, and the route filters the candidates as before.
    two = routing.route_modalities(torch.tensor([0.0, 0.0, 0.1, 0.1, 0.9, 0.9]),
                                   {"__query__": (0, 2), "k1": (2, 4), "i1": (4, 6)},
                                   {"__query__": "image", "k1": "text", "i1": "image"}, margin=0.0)
    assert two["mode"] == "attention" and two["modalities"]

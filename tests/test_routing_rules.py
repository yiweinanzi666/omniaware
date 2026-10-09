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

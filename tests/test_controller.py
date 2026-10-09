import math

import pytest
import torch

from omniaware import routing
from omniaware.pipeline import Norm, Variant, parse_variant
from omniaware.qwen_eab import grounding_entropy, predictive_entropy


def test_norm_maps_median_to_half_and_clips():
    n = Norm.fit([0.0, 1.0, 2.0, 3.0, 4.0])          # median 2, IQR 2
    assert n(2.0) == pytest.approx(0.5)
    assert n(2.0 + 1.5) == pytest.approx(1.0)
    assert n(-100) == 0.0 and n(100) == 1.0


def test_entropies_are_normalised():
    assert predictive_entropy(torch.zeros(1000)) == pytest.approx(1.0)
    peaked = torch.full((1000,), -1e4)
    peaked[3] = 0
    assert predictive_entropy(peaked) == pytest.approx(0.0, abs=1e-6)
    attn = torch.tensor([0.5, 0.25, 0.25, 0.0])
    assert grounding_entropy(attn, [1, 2]) == pytest.approx(1.0)
    assert grounding_entropy(attn, [0]) == 0.0           # N <= 1


def test_paper_bias_form_and_gate():
    v = parse_variant("eab:g1.2:ans2:pos:gate")
    enc = type("E", (), {"length": 50, "split": 40})()
    spec = v.spec(enc, {"a": 1.0, "b": 0.5}, u=0.8, tau_bias=0.4, n_layers=36)
    assert spec.values == pytest.approx({"a": 1.2 * 0.8, "b": 1.2 * 0.8 * 0.5})
    assert spec.q_start == 50 and spec.layers is None
    assert v.spec(enc, {"a": 1.0}, u=0.4, tau_bias=0.4, n_layers=36) is None   # gate: U <= tau_bias
    assert parse_variant("eab:g1.2:ans1").spec(enc, {"a": 1.0}, 0.9, 0.4, 36).q_start == 49
    pasta = parse_variant("pasta:g1.2:ans2").spec(enc, {"a": 0.1, "b": 0.9}, 0.1, 0.4, 36)
    assert set(pasta.values.values()) == {1.2}                                  # uniform, ungated


def test_removed_forms_are_rejected():
    with pytest.raises(ValueError):
        Variant("x", 1.0, form="supk").spec(type("E", (), {"length": 1, "split": 0})(), {"a": 1}, 1, 0, 1)


def test_routing_prefers_substantial_and_diffuse_modality():
    attn = torch.tensor([0.0] * 2 + [0.1, 0.1, 0.1, 0.1] + [0.55, 0.05])
    spans = {"t1": (2, 6), "i1": (6, 8)}
    route = routing.route_modalities(attn, spans, {"t1": "text", "i1": "image"})
    s = route["statistics"]
    assert s["text"]["entropy"] == pytest.approx(1.0) and s["image"]["entropy"] < 0.5
    assert route["modalities"][0] == "text"
    assert math.isclose(s["text"]["mass"] + s["image"]["mass"], 1.0)

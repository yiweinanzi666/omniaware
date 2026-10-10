"""SeaKR port: upstream eigen score, EOS-state sampling on a tiny Qwen3-VL, selection and threshold."""
import ast
from pathlib import Path

import pytest
import torch

from omniaware.baselines import seakr
from eab_fixtures import encode

UPSTREAM = Path(__file__).resolve().parents[1] / "external/SeaKR/vllm_uncertainty/vllm/engine/llm_engine.py"


@pytest.mark.skipif(not UPSTREAM.exists(), reason="external/SeaKR not fetched")
def test_eigen_score_body_is_upstream():
    tree = ast.parse(UPSTREAM.read_text(encoding="utf8"))
    upstream = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_compute_eigen_score")
    ours = next(n for n in ast.parse(Path(seakr.__file__).read_text(encoding="utf8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "eigen_score")
    body = lambda f: [ast.dump(s) for s in f.body if not isinstance(s, ast.Expr)]  # noqa: E731
    assert [b.replace("Attribute(value=Name(id='self', ctx=Load()), attr='eigen_alpha', ctx=Load())",
                      "Name(id='alpha', ctx=Load())") for b in body(upstream)] == body(ours)


def test_eigen_score_grows_with_spread():
    torch.manual_seed(0)
    tight = torch.randn(1, 64).repeat(10, 1) + 1e-3 * torch.randn(10, 64)
    spread = torch.randn(10, 64)
    assert seakr.eigen_score(tight) < seakr.eigen_score(spread)


def test_threshold_matching_and_document_choice():
    scores = [-9.0, -8.0, -7.0, -6.0, -5.0]
    thr = seakr.matched_threshold(scores, 0.4)
    assert sum(s > thr for s in scores) == 2
    assert seakr.choose_document({"a": -7.0, "b": -9.0, "c": -10.0}, {"a": "x", "b": "y", "c": " "}) == "b"
    assert seakr.scaled_layer(32) == 15 and seakr.scaled_layer(36) == 17


def test_missing_eigen_scores_are_none_not_infinity(monkeypatch):
    monkeypatch.setattr(seakr, "sampled_eos_states", lambda *a, **k: [torch.ones(8)])
    assert seakr.score_input(None, None, 4, 10) is None          # one EOS sample: no eigen score
    monkeypatch.setattr(seakr, "sampled_eos_states", lambda *a, **k: [])
    assert seakr.score_input(None, None, 4, 10) is None
    assert seakr.is_missing(None) and seakr.is_missing(float("inf")) and not seakr.is_missing(-6.0)


def test_missing_scores_never_retrieve_and_stay_in_the_rate():
    scores = [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, None, None]
    thr = seakr.matched_threshold(scores, 0.3)
    retrieved = sum(s is not None and s > thr for s in scores)
    assert abs(retrieved / len(scores) - 0.3) <= 1 / len(scores)
    assert seakr.matched_threshold([None, None], 0.5) == float("inf")           # nothing can retrieve
    assert seakr.matched_threshold([-1.0, None], 1.0) == float("-inf")         # every finite score retrieves


def test_documents_without_a_score_come_after_scored_ones():
    assert seakr.choose_document({"a": None, "b": -3.0, "c": -9.0}, {"a": "x", "b": "y", "c": "z"}) == "c"
    assert seakr.choose_document({"a": None, "b": None}, {"a": "x", "b": "y"}) == "a"
    assert seakr.choose_document({"a": -9.0, "b": None}, {"a": " ", "b": "y"}) == "b"    # empty answer last


def test_eos_states_from_sampling_on_tiny_qwen(qwen):
    enc = encode(qwen)
    states = seakr.sampled_eos_states(qwen, enc, n=6, max_new_tokens=200, layer=1, seed=3)
    assert states and all(s.shape == (64,) for s in states)
    again = seakr.sampled_eos_states(qwen, enc, n=6, max_new_tokens=200, layer=1, seed=3)
    assert len(again) == len(states) and all(torch.equal(a, b) for a, b in zip(states, again))

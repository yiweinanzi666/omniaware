"""B2: first-token signals and the tiny-model decode path."""
import math

import pytest
import torch

from eab_fixtures import encode, make_qwen
from omniaware import signals
from omniaware.routing import evidence_route


def test_first_token_signals_are_nonnegative_and_width_k():
    logits = torch.tensor([0.2, 3.0, 1.0, 0.0, -1.0, 2.0, 0.5, 0.1, -0.4, 1.5, 0.3, 0.7])
    out = signals.first_token_signals(logits, k=10)
    assert out["entropy"] >= 0
    assert out["entropy_norm"] == pytest.approx(out["entropy"] / math.log(out["vocab_size"]))
    assert out["margin"] >= 0
    assert out["margin"] == pytest.approx(3.0 - 2.0)
    assert len(out["topk"]) == 10
    assert out["topk"][0]["id"] == 1
    assert {key for row in out["topk"] for key in row} == {"id", "logprob"}


def test_kl_to_direct_is_zero_for_the_same_distribution_and_none_is_not_stored_here():
    logits = torch.arange(12, dtype=torch.float)
    assert signals.kl_to(logits, logits) == pytest.approx(0.0, abs=1e-5)
    other = logits + torch.tensor([0, 4] + [0] * 10, dtype=torch.float)
    assert signals.kl_to(other, logits) > 0


def test_gold_in_topk_is_stored_and_not_read_by_routing():
    import inspect
    from omniaware import routing
    assert "gold_in_topk" not in inspect.getsource(routing)
    assert "gold_in_topk" not in inspect.getsource(evidence_route)
    candidates = [
        {"cid": "t1", "modality": "text", "gold": True},
        {"cid": "t2", "modality": "text", "gold": False},
        {"cid": "i1", "modality": "image", "gold": False},
    ]
    scores = {"t1": 0.2, "t2": 0.9, "i1": 0.5}
    selected = [candidates[1], candidates[2]]
    rec = signals.retriever_signals(candidates, scores, selected)
    assert rec["top1_score"] == pytest.approx(0.9)
    assert rec["top1_top2_margin"] == pytest.approx(0.4)
    assert rec["gold_in_topk"] is False
    assert rec["per_modality"]["text"]["gold_in_topk"] is False
    direct = signals.retriever_signals(candidates, scores, None)
    assert direct["gold_in_topk"] is None
    assert direct["per_modality"]["text"]["top1_score"] == pytest.approx(0.9)


def test_unavailable_action_has_the_same_fields_and_is_marked():
    rec = signals.unavailable_record(signals.retriever_signals([], {}, []))
    assert rec["available"] is False and rec["answer"] == "" and rec["answer_empty"] is True
    assert rec["seq_logprob"] is None and rec["first_token"] is None and rec["gen_len"] == 0
    logits = torch.arange(12, dtype=torch.float)
    live = signals.action_record("Paris", logits, [3], [-0.5], 12, None, rec["retriever"])
    assert live["available"] is True
    assert set(rec) == set(live)


def test_sequence_logprob_is_none_only_without_token_logprobs():
    logits = torch.arange(12, dtype=torch.float)
    assert signals.sequence_logprob(logits, [1, 2], None, 12) is None
    assert signals.sequence_logprob(logits, [1, 2], [-0.25, -0.75], 12) == pytest.approx(-1.0)
    assert signals.sequence_logprob(logits, [], [], 12) == 0.0


def test_tiny_qwen_generation_populates_action_fields():
    qwen = make_qwen()
    enc = encode(qwen)
    res, _, _ = qwen.run(enc, None, 4, decode_static=True)
    result = res["static"]
    assert result.token_logprobs is not None and len(result.token_logprobs) == len(result.tokens)
    direct = signals.action_record(result.text, result.first_logits, result.tokens, result.token_logprobs,
                                   qwen.vocab_size, None,
                                   signals.retriever_signals([], {}, None))
    evidence = signals.action_record(result.text, result.first_logits, result.tokens, result.token_logprobs,
                                     qwen.vocab_size, result.first_logits,
                                     {"top1_score": 0.5, "top1_top2_margin": 0.1, "gold_in_topk": False})
    for rec in (direct, evidence):
        assert rec["first_token"]["entropy"] >= 0
        assert rec["first_token"]["entropy_norm"] >= 0
        assert rec["first_token"]["margin"] >= 0
        assert len(rec["first_token"]["topk"]) >= 10
        assert rec["gen_len"] == len(result.tokens)
        assert rec["seq_logprob"] == pytest.approx(sum(result.token_logprobs))
        assert set(rec) == {"available", "answer", "answer_empty", "citation_only", "seq_logprob", "gen_len",
                            "first_token", "kl_to_direct", "retriever"}
        assert rec["available"] is True
    assert direct["kl_to_direct"] is None
    assert evidence["kl_to_direct"] == pytest.approx(0.0, abs=1e-5)
    assert isinstance(direct["answer_empty"], bool) and isinstance(direct["citation_only"], bool)

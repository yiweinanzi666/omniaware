"""First-token and retriever signals for one decoded action. Pure tensor math; no model weights.

``gold_in_topk`` is written for offline analysis. Nothing in this module, and no routing decision,
reads it back to choose an action.
"""
from __future__ import annotations

import math

import torch

from .answers import parse_answer


def _as_logits(logits, vocab_size: int | None) -> torch.Tensor:
    flat = torch.as_tensor(logits, dtype=torch.float32).flatten()
    if vocab_size is not None:
        if not 1 < vocab_size <= flat.numel():
            raise ValueError(f"vocab_size {vocab_size} must lie in (1, {flat.numel()}]")
        flat = flat[:vocab_size]
    if flat.numel() < 2:
        raise ValueError("first-token signals need at least two logits")
    return flat


def first_token_signals(logits, k: int = 10, vocab_size: int | None = None) -> dict:
    """Entropy (raw and / log|V|), margin (top1 logit - top2 logit), and top-k ids with logprobs."""
    if k < 1:
        raise ValueError("k must be >= 1")
    flat = _as_logits(logits, vocab_size)
    if flat.numel() < k:
        raise ValueError(f"need at least {k} logits for top-k, got {flat.numel()}")
    logp = torch.log_softmax(flat, dim=-1)
    prob = logp.exp()
    entropy = float((-(prob * logp).sum()).clamp_min(0))
    vocab = int(flat.numel())
    entropy_norm = entropy / math.log(vocab)
    top_vals, top_idx = torch.topk(flat, k)
    # margin from the two largest logits, not from a re-sort of the top-k logprobs
    margin = float(top_vals[0] - top_vals[1])
    if margin < 0:
        margin = 0.0
    return {
        "topk": [{"id": int(i), "logprob": float(logp[int(i)])} for i in top_idx],
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "margin": margin,
        "vocab_size": vocab,
    }


def kl_to(evidence_logits, direct_logits, vocab_size: int | None = None) -> float:
    """KL(p_evidence || p_direct) on the first-token distribution. Non-negative up to rounding."""
    p = torch.log_softmax(_as_logits(evidence_logits, vocab_size), dim=-1)
    q = torch.log_softmax(_as_logits(direct_logits, vocab_size), dim=-1)
    return float((p.exp() * (p - q)).sum().clamp_min(0))


def sequence_logprob(logits, tokens: list[int] | None, token_logprobs: list[float] | None,
                     vocab_size: int | None) -> float | None:
    """Sum of greedy-token logprobs. A one-token decode can use the first-step distribution."""
    tokens = list(tokens or [])
    if token_logprobs is not None:
        return float(sum(token_logprobs))
    if not tokens:
        return 0.0
    if len(tokens) != 1:
        return None
    logp = torch.log_softmax(_as_logits(logits, vocab_size), dim=-1)
    return float(logp[int(tokens[0])])


def _rank_stats(pool: list[dict], scores: dict[str, float], selected_ids: set[str] | None) -> dict:
    ranked = sorted(pool, key=lambda c: (-float(scores.get(c["cid"], float("-inf"))), str(c["cid"])))
    if not ranked:
        return {"top1_score": None, "top1_top2_margin": None, "gold_in_topk": None}
    s1 = float(scores[ranked[0]["cid"]])
    s2 = float(scores[ranked[1]["cid"]]) if len(ranked) > 1 else None
    margin = None if s2 is None else s1 - s2
    if selected_ids is None:
        gold = None
    else:
        known = [c for c in pool if c.get("gold") is not None]
        if not known:
            gold = None
        else:
            gold = any(bool(c.get("gold")) and c["cid"] in selected_ids for c in pool)
    return {"top1_score": s1, "top1_top2_margin": margin, "gold_in_topk": gold}


def retriever_signals(candidates: list[dict], scores: dict[str, float], selected: list[dict] | None,
                      modality: str | None = None) -> dict:
    """Retriever top-1, top1-top2 gap, and gold-in-topk for the chunks this action sends.

    ``selected is None`` (the direct action) stores the pool's top-1 scores and leaves
    ``gold_in_topk`` unset: no top-k was sent to the model. ``per_modality`` repeats the same
    three keys per modality. ``gold_in_topk`` is not an input to any decision.
    """
    pool = [c for c in candidates if modality is None or c["modality"] == modality]
    selected_ids = None if selected is None else {c["cid"] for c in selected}
    if modality is not None:
        if selected_ids is not None:
            selected_ids = {c["cid"] for c in selected if c["modality"] == modality}
        return _rank_stats(pool, scores, selected_ids)
    top = _rank_stats(pool, scores, selected_ids)
    per = {}
    for m in sorted({c["modality"] for c in candidates}):
        per[m] = retriever_signals(candidates, scores, selected, modality=m)
    top["per_modality"] = per
    return top


def action_record(text: str, logits, tokens: list[int] | None, token_logprobs: list[float] | None,
                  vocab_size: int | None, direct_logits, retriever: dict, k: int = 10) -> dict:
    """One available v2 action. ``direct_logits is None`` marks the direct action (``kl_to_direct`` is None)."""
    parsed = parse_answer(text)
    first = first_token_signals(logits, k=k, vocab_size=vocab_size)
    return {
        "available": True,
        "answer": parsed["answer"],
        "answer_empty": parsed["answer_empty"],
        "citation_only": parsed["citation_only"],
        "seq_logprob": sequence_logprob(logits, tokens, token_logprobs, vocab_size),
        "gen_len": len(tokens or []),
        "first_token": first,
        "kl_to_direct": None if direct_logits is None else kl_to(logits, direct_logits, vocab_size),
        "retriever": retriever,
    }


def unavailable_record(retriever: dict) -> dict:
    """A v2 action with no candidates (for example a modality pool that is empty). Nothing is decoded, so no
    selector may choose it: ``available`` is False and the answer fields are empty."""
    return {
        "available": False,
        "answer": "",
        "answer_empty": True,
        "citation_only": False,
        "seq_logprob": None,
        "gen_len": 0,
        "first_token": None,
        "kl_to_direct": None,
        "retriever": retriever,
    }

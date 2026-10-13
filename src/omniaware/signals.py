"""First-token, answer-token and retriever signals for one decoded action. Pure tensor math; no model weights.

``first_token`` describes the first generated token. ``answer_token`` describes the first token of the answer: the
position where the paper's uncertainty is defined (H_sem over the first answer token, H_grd on the row that predicts
it). In the answer-first format (prompts.ANSWER_FIRST) the two coincide unless the short answer opens with a function
word ("The Eiffel Tower"); in a plain output the answer token is located after template words and restated question
words (``answer_slot``). ``answer_token.u_raw`` = H_grd + beta * H_sem at that position is the confidence the
pre-registered selector reads (arbitration, confidence "u_omni").

``gold_in_topk`` is written for offline analysis. Nothing in this module, and no routing decision,
reads it back to choose an action.
"""
from __future__ import annotations

import math
import re

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
    """Sum of greedy-token logprobs. A one-token decode can use the first-step distribution.

    An empty decode has no greedy token, so it has no sequence logprob (None), not a score of 0.
    """
    tokens = list(tokens or [])
    if not tokens:
        return None
    if token_logprobs is not None:
        return float(sum(token_logprobs))
    if len(tokens) != 1:
        return None
    logp = torch.log_softmax(_as_logits(logits, vocab_size), dim=-1)
    return float(logp[int(tokens[0])])


def _top_stats(ids: list[str], scores: dict[str, float]) -> dict:
    """Top-1 score and top1-top2 gap over the given candidate ids (None where undefined)."""
    if not ids:
        return {"top1_score": None, "top1_top2_margin": None}
    ranked = sorted(ids, key=lambda cid: (-float(scores[cid]), str(cid)))
    s1 = float(scores[ranked[0]])
    margin = s1 - float(scores[ranked[1]]) if len(ranked) > 1 else None
    return {"top1_score": s1, "top1_top2_margin": margin}


def _selected_stats(pool: list[dict], scores: dict[str, float], selected: list[dict] | None,
                    modality: str | None = None) -> dict:
    """Top-1 of the evidence the action sends (within ``modality`` when given), and gold_in_topk over it.

    gold_in_topk is None when no evidence was sent (direct, or an unavailable action) or when no candidate of the
    pool has a gold label.
    """
    if not selected:
        return {"top1_score": None, "top1_top2_margin": None, "gold_in_topk": None}
    chosen = [c for c in selected if modality is None or c["modality"] == modality]
    top = _top_stats([c["cid"] for c in chosen], scores)
    known = any(c.get("gold") is not None for c in pool)
    if known:
        chosen_ids = {c["cid"] for c in selected}
        gold = any(bool(c.get("gold")) for c in pool if c["cid"] in chosen_ids and
                   (modality is None or c["modality"] == modality))
    else:
        gold = None
    return {**top, "gold_in_topk": gold}


def retriever_signals(candidates: list[dict], scores: dict[str, float], selected: list[dict] | None) -> dict:
    """Retriever statistics of one action.

    The top-level ``top1_score`` / ``top1_top2_margin`` / ``gold_in_topk`` describe the chunks this action sends
    (``selected``), scored by ``scores``. The direct action sends none (``selected is None``), so its values are
    None. ``pool`` holds the same statistics over the whole candidate pool and per modality (``per_modality``, each
    entry with the selected evidence of that modality). No selector reads ``gold_in_topk``.
    """
    sent = _selected_stats(candidates, scores, selected)
    per = {}
    for m in sorted({c["modality"] for c in candidates}):
        pool_m = [c for c in candidates if c["modality"] == m]
        per[m] = {**_top_stats([c["cid"] for c in pool_m], scores),
                  "gold_in_topk": _selected_stats(candidates, scores, selected, modality=m)["gold_in_topk"]}
    pool = {**_top_stats([c["cid"] for c in candidates], scores), "per_modality": per}
    return {**sent, "pool": pool}


def action_record(text: str, logits, tokens: list[int] | None, token_logprobs: list[float] | None,
                  vocab_size: int | None, direct_logits, retriever: dict, chunks: list[str], k: int = 10,
                  answer_first: bool = False) -> dict:
    """One available v2 action. ``direct_logits is None`` marks the direct action (``kl_to_direct`` is None).
    ``chunks`` are the ids of the evidence chunks in the order the model read them ([] for direct).
    ``answer`` is the scored text and ``short`` the short answer (parse_answer)."""
    parsed = parse_answer(text, answer_first)
    first = first_token_signals(logits, k=k, vocab_size=vocab_size)
    return {
        "available": True,
        "answer": parsed["answer"],
        "short": parsed["short"],
        "answer_first_ok": parsed["answer_first_ok"],
        "answer_empty": parsed["answer_empty"],
        "citation_only": parsed["citation_only"],
        "refusal": parsed["refusal"],
        "seq_logprob": sequence_logprob(logits, tokens, token_logprobs, vocab_size),
        "gen_len": len(tokens or []),
        "first_token": first,
        "kl_to_direct": None if direct_logits is None else kl_to(logits, direct_logits, vocab_size),
        "retriever": retriever,
        "chunks": list(chunks),
    }


def unavailable_record(retriever: dict) -> dict:
    """A v2 action with no candidates (for example a modality pool that is empty). Nothing is decoded, so no
    selector may choose it: ``available`` is False, the answer fields are empty and no chunk is listed."""
    return {
        "available": False,
        "answer": "",
        "short": "",
        "answer_first_ok": None,
        "answer_empty": True,
        "citation_only": False,
        "refusal": False,
        "seq_logprob": None,
        "gen_len": 0,
        "first_token": None,
        "kl_to_direct": None,
        "retriever": retriever,
        "chunks": [],
    }


# Words that never carry the answer: articles, pronouns, auxiliaries, prepositions, conjunctions and sentence openers.
FUNCTION_WORDS = frozenset("""
a an the this that these those there here it its it's they them their he she his her we our you your i my me
is are was were be been being am do does did has have had will would can could should may might must shall
of in on at to for from by with about as into onto over under between among through during before after above below
and or but nor so yet if then than because while although though whether either neither both each every all any some
not according based answer short sentence image picture photo shown depicted
""".split())
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_LABEL = re.compile(r"(?i)^\s*(?:short\s+)?answer\s*:\s*")


def answer_slot(token_texts: list[str], question: str, answer_first: bool) -> int:
    """Index of the first generated token of the first answer-bearing word; 0 when there is none.

    The decoded text is rebuilt from the token strings. In the answer-first format only the first line is
    searched (an "Answer:" label is skipped) and the first word outside FUNCTION_WORDS starts the answer; question
    words are allowed, because a choice answer repeats one. In a plain output the restated question words are
    skipped as well, since a sentence answer opens by restating the question.
    """
    if not token_texts:
        return 0
    text, owner = "", []
    for index, piece in enumerate(token_texts):
        text += piece
        owner += [index] * len(piece)
    start, end = 0, len(text)
    if answer_first:
        lead = len(text) - len(text.lstrip())
        newline = text.find("\n", lead)
        end = newline if newline >= 0 else len(text)
        label = _LABEL.match(text[:end])
        start = label.end() if label else 0
    skip = set(FUNCTION_WORDS)
    if not answer_first:
        skip |= {w.lower() for w in _WORD.findall(question)}
    for match in _WORD.finditer(text, start, end):
        if match.group().lower() not in skip:
            return owner[match.start()]
    return 0


def with_answer_token(record: dict, step_entropy: list[float] | None, step_margin: list[float] | None,
                      token_texts: list[str] | None, question: str, answer_first: bool, h_grd: float,
                      beta: float, route: dict | None = None) -> dict:
    """Add ``answer_token`` (and the evidence ``route`` when given) to an available action record.

    ``answer_token`` holds the slot index and its token, the normalised entropy and the margin of the distribution
    that produced it, H_grd of the t_end row over this action's evidence (0 without evidence), beta, and
    ``u_raw`` = h_grd + beta * entropy_norm. Without per-step signals (or an empty decode) the slot is the first
    generated position and its values come from ``first_token``.
    """
    first = record["first_token"]
    have_steps = bool(step_entropy) and bool(step_margin) and token_texts is not None
    index = answer_slot(token_texts, question, answer_first) if have_steps else 0
    if have_steps:
        entropy, margin, text = float(step_entropy[index]), float(step_margin[index]), token_texts[index]
    else:
        entropy, margin, text = float(first["entropy_norm"]), float(first["margin"]), None
    out = {**record, "answer_token": {"index": index, "text": text, "steps_recorded": have_steps,
                                      "entropy_norm": entropy, "margin": margin, "h_grd": float(h_grd),
                                      "beta": float(beta), "u_raw": float(h_grd) + float(beta) * entropy}}
    if route is not None:
        out["route"] = route
    return out


def input_trace(enc, image_token_id: int | None) -> dict:
    """What one action's prompt contained, read from its encoding: length, a hash of the token ids, and the
    number of visual tokens of each image in prompt order.

    The image counts come from the runs of the image placeholder token in ``input_ids``, so they describe the
    input the model received, not what the packing code intended. ``image_token_id`` is None for a model whose
    placeholder is unknown; the image fields are then null.
    """
    import hashlib
    ids = getattr(enc, "input_ids", None)
    if ids is None:
        return {"prompt_tokens": int(enc.length), "prompt_sha256": None, "n_images": None, "image_tokens": None}
    flat = torch.as_tensor(ids).flatten().tolist()
    trace = {"prompt_tokens": int(enc.length),
             "prompt_sha256": hashlib.sha256(",".join(map(str, flat)).encode()).hexdigest()}
    if image_token_id is None:
        return {**trace, "n_images": None, "image_tokens": None}
    runs, current = [], 0
    for token in flat:
        if token == image_token_id:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return {**trace, "n_images": len(runs), "image_tokens": runs}


def decode_trace(result, max_new_tokens: int, token_text=None) -> dict:
    """The greedy decode of one action: token ids, their logprobs, wall time, and whether it stopped at the
    length limit. ``token_text`` maps a token id to its string for the first-token candidates."""
    tokens = [int(t) for t in (result.tokens or [])]
    logprobs = None if result.token_logprobs is None else [round(float(v), 5) for v in result.token_logprobs]
    return {"tokens": tokens, "token_logprobs": logprobs, "hit_token_cap": len(tokens) >= int(max_new_tokens),
            "seconds": None if getattr(result, "seconds", None) is None else round(float(result.seconds), 4),
            "token_text": token_text}


def with_trace(record: dict, input_info: dict, decode_info: dict) -> dict:
    """Attach the input and decode trace to an action record. The selector reads none of these fields."""
    token_text = decode_info.pop("token_text")
    if token_text is not None and record.get("first_token"):
        for candidate in record["first_token"]["topk"]:
            candidate["text"] = token_text(candidate["id"])
    return {**record, "input": input_info, "decode": decode_info}

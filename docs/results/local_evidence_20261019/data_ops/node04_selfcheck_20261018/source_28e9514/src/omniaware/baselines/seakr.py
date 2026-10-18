"""SeaKR (Yao et al., ACL 2025) single-hop protocol ported to Qwen3-VL.

Upstream (external/SeaKR, THU-KEG/SeaKR, GPL-3.0) runs LLaMA-2 in a modified vLLM:
- each prompt is decoded once greedily and sampled n=20 times (temperature 1.0, top_p 0.9);
- the hidden state after decoder layer ``selected_intermediate_layer`` (15 of LLaMA-2-7B's 32)
  at the position that emits EOS is kept for every non-empty sample that ends with EOS;
- eigen_score = (1/k) logdet(Z J Z^T + alpha I) over those k states (alpha 1e-3);
- eval_singlehop.ipynb: if the question-only eigen score is below the threshold (-6.0 for LLaMA-2)
  answer directly; otherwise answer once per retrieved document and keep the answer whose
  eigen score is lowest.

Here the layer is scaled to the backbone's depth, documents are our top-k chunks, and the threshold
is either fixed or matched on the unlabeled calibration questions to a target retrieval rate.

Deviations from the upstream protocol (the paper's shared protocol requires them, or they are not
yet verified):
- k = 5 chunks per question (the paper's common k); upstream's notebook scores 10 rows per question.
- EOS samples: we keep a sample if it emitted at least one non-EOS token; upstream keeps one whose
  decoded text is non-empty after strip(). Upstream stacks nothing for samples without EOS and fails;
  we drop them.
- Sampling is seeded with seed=42 for every question, so runs reproduce; upstream draws from an
  unseeded vLLM stream.
- An input with fewer than two EOS samples has no eigen score. It is reported as missing (None),
  never as infinity: a missing score does not trigger retrieval, and the count is recorded.
- Threshold: matched on the calibration questions to the retrieval rate of U (upstream fixes -6.0
  for LLaMA-2), and the layer is scaled to the backbone depth (scaled_layer).
- The eigen-score body is the upstream formula (tests/test_seakr.py compares the AST).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..qwen_eab import _crop

LLAMA2_LAYER, LLAMA2_DEPTH = 15, 32


def eigen_score(z: torch.Tensor, alpha: float = 1e-3) -> float:
    """Same body as upstream vllm_uncertainty/vllm/engine/llm_engine.py:_compute_eigen_score (AST-checked)."""
    z = z.to(torch.float32)
    k, d = z.shape
    j_d = torch.eye(d) - (1/d) * torch.ones(d, d)
    j_d = j_d.to(z.device)
    sigma = torch.einsum('ij,jk,kl->il', z, j_d, z.t())
    return ((1/k) * torch.logdet(sigma + alpha * torch.eye(k, device=sigma.device))).item()


def scaled_layer(n_layers: int) -> int:
    """Upstream layer 15 of 32, at the same relative depth."""
    return round(LLAMA2_LAYER * n_layers / LLAMA2_DEPTH)


def _top_p_sample(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator) -> int:
    probs = torch.softmax(logits.float().cpu() / temperature, dim=-1)
    sorted_probs, order = torch.sort(probs, descending=True)
    keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
    sorted_probs = sorted_probs * keep
    choice = torch.multinomial(sorted_probs / sorted_probs.sum(), 1, generator=generator)
    return int(order[choice])


@torch.no_grad()
def sampled_eos_states(qwen, enc, n: int = 20, max_new_tokens: int = 100, layer: int | None = None,
                       temperature: float = 1.0, top_p: float = 0.9, seed: int = 42) -> list[torch.Tensor]:
    """Hidden states (after decoder ``layer``) at the step that emits EOS, one per non-empty sample."""
    layer = scaled_layer(qwen.n_layers) if layer is None else layer
    generator = torch.Generator().manual_seed(seed)
    cache = qwen._prefix(enc)
    suffix = enc.input_ids[:, enc.split:]
    states = []
    for _ in range(n):
        _crop(cache, enc.split)
        ids, emitted = suffix, 0
        for _ in range(max_new_tokens):
            out = qwen.model(input_ids=ids.to(qwen.device), past_key_values=cache, use_cache=True,
                             output_hidden_states=True, logits_to_keep=1)
            token = _top_p_sample(out.logits[0, -1], temperature, top_p, generator)
            if token in qwen.eos:
                if emitted:   # upstream drops empty samples and samples without EOS
                    states.append(out.hidden_states[layer + 1][0, -1].float().cpu())
                break
            emitted += 1
            ids = torch.tensor([[token]])
    _crop(cache, enc.split)
    return states


def is_missing(score: float | None) -> bool:
    """A question without a usable eigen score (fewer than two EOS samples, or a non-finite value)."""
    return score is None or not math.isfinite(score)


def score_input(qwen, enc, n: int, max_new_tokens: int, layer: int | None = None, seed: int = 42) -> float | None:
    """Eigen score of one input, or None (missing) when fewer than two EOS samples were drawn."""
    states = sampled_eos_states(qwen, enc, n, max_new_tokens, layer, seed=seed)
    return eigen_score(torch.stack(states)) if len(states) > 1 else None


def matched_threshold(calibration_scores: list[float | None], retrieval_rate: float) -> float:
    """Threshold at which the calibration questions retrieve at ``retrieval_rate``.

    Missing scores never retrieve, so they stay in the denominator: the threshold is chosen so that
    the finite scores above it make up retrieval_rate * n of the n calibration questions. When that
    many cannot be reached (too many missing), every finite score retrieves (threshold -inf). With no
    finite score at all the threshold is +inf and nothing retrieves.
    """
    n = len(calibration_scores)
    scores = np.asarray([s for s in calibration_scores if not is_missing(s)], dtype=float)
    if n == 0 or scores.size == 0:
        return float("inf")
    fraction = retrieval_rate * n / scores.size   # share of the finite scores that must retrieve
    if fraction >= 1:
        return float("-inf")
    return float(np.quantile(scores, 1 - fraction))


def choose_document(doc_scores: dict[str, float | None], doc_answers: dict[str, str]) -> str | None:
    """eval_singlehop.ipynb: keep the answer with the lowest eigen score.

    Empty answers come last. Among non-empty answers, documents with a score come before documents
    without one, which are ordered by candidate id.
    """
    def key(c):
        score = doc_scores[c]
        return (not doc_answers[c].strip(), is_missing(score), 0.0 if is_missing(score) else score, c)
    ranked = sorted(doc_scores, key=key)
    return ranked[0] if ranked else None

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
"""
from __future__ import annotations

import numpy as np
import torch

from ..qwen_eab import _crop

LLAMA2_LAYER, LLAMA2_DEPTH = 15, 32


def eigen_score(z: torch.Tensor, alpha: float = 1e-3) -> float:
    """Verbatim from upstream vllm_uncertainty/vllm/engine/llm_engine.py:_compute_eigen_score."""
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


def score_input(qwen, enc, n: int, max_new_tokens: int, layer: int | None = None, seed: int = 42) -> float:
    states = sampled_eos_states(qwen, enc, n, max_new_tokens, layer, seed=seed)
    return eigen_score(torch.stack(states)) if len(states) > 1 else float("inf")


def matched_threshold(calibration_scores: list[float], retrieval_rate: float) -> float:
    """Threshold whose share of calibration scores above it equals the target retrieval rate."""
    scores = np.asarray([s for s in calibration_scores if np.isfinite(s)], dtype=float)
    return float(np.quantile(scores, 1 - retrieval_rate)) if len(scores) else float("inf")


def choose_document(doc_scores: dict[str, float], doc_answers: dict[str, str]) -> str | None:
    """eval_singlehop.ipynb: keep the answer with the lowest eigen score (empty answers last)."""
    ranked = sorted(doc_scores, key=lambda c: (doc_scores[c] if doc_answers[c].strip() else float("inf"), c))
    return ranked[0] if ranked else None

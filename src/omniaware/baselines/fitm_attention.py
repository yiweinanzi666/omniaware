"""Hsieh et al. document-mean attention calibration, equations (4) and (5).

Leading tensor axes remain independent (layers, batches, query heads, queries).
Reference construction and when to freeze alpha belong to the Qwen adapter.
"""

import math

import torch


TEMPERATURE = 5e-5


def _precision(value):
    return value if value.dtype == torch.float64 else value.float()


def validate_documents(documents, key_count):
    flat = [index for document in documents for index in document]
    if (not documents or any(not document for document in documents)
            or len(set(flat)) != len(flat) or any(index < 0 or index >= key_count for index in flat)):
        raise ValueError("Need nonempty, disjoint document token sets within the actual input")


def document_means(log_attention, documents):
    """Mean probability per document; input is log-softmax attention [...,K]."""
    probability = _precision(log_attention).exp()
    return torch.stack([probability[..., indices].mean(-1) for indices in documents], dim=-1)


def calibrated_alpha(original_means, reference_means, *, temperature=TEMPERATURE):
    """Stable softmax of mean-attention differences, independently on every row."""
    if not math.isfinite(temperature) or temperature <= 0 or original_means.shape != reference_means.shape:
        raise ValueError("Need matching original/reference shapes and positive finite temperature")
    difference = _precision(original_means) - _precision(reference_means)
    if not torch.isfinite(difference).all():
        raise ValueError("Document mean differences must be finite")
    # Subtract before division too, so very small temperatures cannot overflow positive logits.
    logits = (difference - difference.max(-1, keepdim=True).values) / temperature
    return logits.softmax(-1)


def redistribute_document_attention(log_attention, alpha, documents):
    """Eq. (5) with C=S/sum(N_k*alpha_k); non-document probabilities are copied.

    Group-conditional softmax avoids dividing by tiny or underflowed m_k.
    It is algebraically the same rescaling of within-document probabilities.
    """
    log_attention = _precision(log_attention)
    alpha = _precision(alpha)
    if alpha.shape != log_attention.shape[:-1] + (len(documents),):
        raise ValueError("Alpha must match all leading attention axes and the document count")
    if not torch.isfinite(alpha).all() or (alpha < 0).any() or (alpha.sum(-1) <= 0).any():
        raise ValueError("Alpha must be finite, nonnegative and have positive row mass")
    original = log_attention.exp()
    masses = torch.stack([original[..., indices].sum(-1) for indices in documents], -1)
    counts = alpha.new_tensor([len(indices) for indices in documents])
    total_mass = masses.sum(-1, keepdim=True)
    length_weight = counts * alpha
    normalization = total_mass / length_weight.sum(-1, keepdim=True)
    target_mass = normalization * length_weight
    result = original.clone()
    for index, indices in enumerate(documents):
        conditional = log_attention[..., indices].softmax(-1)
        if not torch.isfinite(conditional).all():
            raise ValueError("Every controlled document needs visible finite-score tokens")
        result[..., indices] = target_mass[..., index:index + 1] * conditional
    after = torch.stack([result[..., indices].sum(-1) for indices in documents], -1)
    return result, dict(document_mass_before=masses, document_mass_after=after,
                        document_mean_before=masses / counts,
                        document_mean_after=after / counts, C=normalization,
                        document_token_counts=counts)

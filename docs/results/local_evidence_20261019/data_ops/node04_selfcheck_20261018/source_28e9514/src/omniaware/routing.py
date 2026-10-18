"""Paper modality routing from pre-retrieval evidence attention (M_m H_m), and its ablations.

Attention rules (Table 4): 'mh' ranks modalities by M_m * H_m (paper), 'mass' by M_m alone,
'entropy' by H_m alone.  Non-attention rules: 'retriever' ranks modalities by their best
per-modality-calibrated retriever score, 'all' searches every modality, 'oracle' uses the gold
modalities.  In every ranked rule the runner-up is added when it is within ``margin`` (delta).
"""

import math

ATTENTION_RULES = ("mh", "mass", "entropy")
KEY = {"mh": "score", "mass": "mass", "entropy": "entropy"}


def modality_groups(spans, modalities) -> dict[str, list[int]]:
    """Token positions of the supplied chunks (those present in ``spans``), grouped by modality."""
    groups = {}
    for chunk, modality in modalities.items():
        if chunk in spans:
            groups.setdefault(modality, []).extend(range(*spans[chunk]))
    return groups


def modality_statistics(attention, groups, eps=1e-9) -> tuple[dict, float]:
    """Per modality m: M_m (share of the total attention mass), H_m (entropy of the conditional distribution
    within m, normalised by log|I_m|, 0 when |I_m| <= 1) and M_m * H_m. Also returns the total mass over the
    groups."""
    weights = {m: [float(attention[i]) for i in ids] for m, ids in groups.items()}
    total = sum(sum(values) for values in weights.values())
    stats = {}
    if total <= eps:
        return stats, total
    for modality, values in weights.items():
        mass = sum(values) / total
        conditional = [value / total / (mass + eps) for value in values]
        entropy = (-sum(p * math.log(p + eps) for p in conditional)
                   / math.log(max(len(values), 2))) if len(values) > 1 else 0.0
        stats[modality] = {"mass": mass, "entropy": entropy,
                           "score": mass * entropy, "tokens": len(values)}
    return stats, total


def route_modalities(attention, spans, modalities, margin=0.08, eps=1e-9, rule="mh"):
    """Return the leading route and its near-tie. No supplied modality is a cold start (the caller runs a
    first retrieval round); a single supplied modality skips routing (paper, Modality Routing: retrieval then
    searches the full pool, e.g. an OK-VQA query image against a text corpus)."""
    groups = modality_groups(spans, modalities)
    if not groups:
        return {"mode": "cold_start", "modalities": [], "statistics": {}}
    if len(groups) < 2:
        return {"mode": "skipped", "rule": rule, "modalities": [], "statistics": {}}
    if attention is None:
        raise ValueError("Modality routing requires captured initial-evidence attention")
    stats, total = modality_statistics(attention, groups, eps)
    if total <= eps:
        return {"mode": "no_evidence_mass", "modalities": [], "statistics": {}}
    if rule not in ATTENTION_RULES:
        raise ValueError(rule)
    ranked = sorted((m for m in stats if stats[m]["mass"] > eps), key=lambda m: (-stats[m][KEY[rule]], m))
    selected = ranked[:1]
    if len(ranked) > 1 and stats[ranked[0]][KEY[rule]] - stats[ranked[1]][KEY[rule]] <= margin:
        selected.append(ranked[1])
    return {"mode": "attention", "rule": rule, "modalities": selected, "statistics": stats}


def initial_route(attention, enc, example, rule="mh"):
    modalities = {}
    if example["query_images"]:
        modalities["__query__"] = "image"
    return route_modalities(attention, enc.spans, modalities, rule=rule)


def evidence_route(attention, enc, chunks, margin=0.08, rule="mh"):
    """Cold start (paper B.5): a first joint retrieval round supplies the evidence spans;
    route by M_m H_m (or an ablation rule) over the unbiased final-layer attention on those spans."""
    return route_modalities(attention, enc.spans, {c["cid"]: c["modality"] for c in chunks}, margin, rule=rule)


def retriever_route(candidates, relevance, margin=0.08):
    """Route by each modality's best calibrated retriever score (per-modality ECDF, so comparable)."""
    best = {}
    for c in candidates:
        best[c["modality"]] = max(best.get(c["modality"], 0.0), relevance[c["cid"]])
    if not best:
        return {"mode": "cold_start", "rule": "retriever", "modalities": [], "statistics": {}}
    ranked = sorted(best, key=lambda m: (-best[m], m))
    selected = ranked[:1]
    if len(ranked) > 1 and best[ranked[0]] - best[ranked[1]] <= margin:
        selected.append(ranked[1])
    return {"mode": "retriever", "rule": "retriever", "modalities": selected,
            "statistics": {m: {"best_relevance": v} for m, v in best.items()}}

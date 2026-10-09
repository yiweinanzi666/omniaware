"""Paper modality routing from pre-retrieval evidence attention (M_m H_m)."""

import math


def route_modalities(attention, spans, modalities, margin=0.08, eps=1e-9):
    """Return the leading route and its near-tie, or an explicit cold start."""
    groups = {}
    for chunk, modality in modalities.items():
        if chunk in spans:
            groups.setdefault(modality, []).extend(range(*spans[chunk]))
    if not groups:
        return {"mode": "cold_start", "modalities": [], "statistics": {}}
    if attention is None:
        raise ValueError("Modality routing requires captured initial-evidence attention")
    weights = {m: [float(attention[i]) for i in ids] for m, ids in groups.items()}
    total = sum(sum(values) for values in weights.values())
    if total <= eps:
        return {"mode": "no_evidence_mass", "modalities": [], "statistics": {}}
    stats = {}
    for modality, values in weights.items():
        mass = sum(values) / total
        conditional = [value / total / (mass + eps) for value in values]
        entropy = (-sum(p * math.log(p + eps) for p in conditional)
                   / math.log(max(len(values), 2))) if len(values) > 1 else 0.0
        stats[modality] = {"mass": mass, "entropy": entropy,
                           "score": mass * entropy, "tokens": len(values)}
    ranked = sorted((m for m in stats if stats[m]["mass"] > eps),
                    key=lambda m: (-stats[m]["score"], m))
    selected = ranked[:1]
    if len(ranked) > 1 and stats[ranked[0]]["score"] - stats[ranked[1]]["score"] <= margin:
        selected.append(ranked[1])
    return {"mode": "attention", "modalities": selected, "statistics": stats}


def initial_route(attention, enc, example):
    modalities = {c["cid"]: c["modality"] for c in example.get("query_chunks", [])}
    if example["query_images"]:
        modalities["__query__"] = "image"
    return route_modalities(attention, enc.spans, modalities)


def evidence_route(attention, enc, chunks, margin=0.08):
    """Cold start (paper B.5): a first joint retrieval round supplies the evidence spans;
    route by M_m H_m over the unbiased final-layer attention on those spans."""
    return route_modalities(attention, enc.spans, {c["cid"]: c["modality"] for c in chunks}, margin)

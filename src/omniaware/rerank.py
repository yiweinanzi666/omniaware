"""Cross-encoder evidence selection with per-modality ECDF calibration (bge-reranker-v2-m3).

Functions are copied verbatim from the legacy pipeline (see MIGRATION_SOURCES.json, objects):
rank_sources selects the cross-encoder top-k and presents it in BGE order (appendix:
"BGE top-5 vs five facts that bge-reranker-v2-m3 selects"); select_calibrated selects by the
per-modality ECDF of the cross-encoder logit (appendix: "percentile calibration").
"""
from bisect import bisect_right
import math

import torch

TOP_K, MAX_LENGTH = 5, 512


def encode_bge(texts, tokenizer, model, batch_size):
    import torch
    vectors, lengths = [], []
    for offset in range(0, len(texts), batch_size):
        batch = tokenizer(texts[offset:offset+batch_size], padding=True, truncation=True,
                          max_length=MAX_LENGTH, return_tensors='pt')
        lengths.extend(batch['attention_mask'].sum(1).tolist())
        with torch.inference_mode():
            cls = model(**batch).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(cls.float(), p=2, dim=1))
    return torch.cat(vectors), lengths


def score_ce(question, candidates, tokenizer, model, batch_size):
    import torch
    values, lengths = [], []
    for offset in range(0, len(candidates), batch_size):
        pairs = [[question, c['text']] for c in candidates[offset:offset+batch_size]]
        batch = tokenizer(pairs, padding=True, truncation=True, max_length=MAX_LENGTH,
                          return_tensors='pt')
        lengths.extend(batch['attention_mask'].sum(1).tolist())
        with torch.inference_mode():
            logits = model(**batch, return_dict=True).logits.view(-1).float().tolist()
        values.extend(logits)
    return values, lengths


def rank_sources(candidates, bge_scores, ce_scores):
    scored = []
    for candidate, bge_score, ce_score in zip(candidates, bge_scores, ce_scores, strict=True):
        if not math.isfinite(bge_score) or not math.isfinite(ce_score):
            raise ValueError('retrieval score must be finite')
        scored.append(dict(candidate, source_id=f"{candidate['modality']}:{candidate['fact_id']}",
                           bge_cosine=float(bge_score), ce_logit=float(ce_score)))
    tie = lambda c: (str(c['fact_id']), c['modality'])
    by_bge = sorted(scored, key=lambda c: (-c['bge_cosine'], *tie(c)))
    by_ce = sorted(scored, key=lambda c: (-c['ce_logit'], *tie(c)))
    for rank, candidate in enumerate(by_bge, 1):
        candidate['bge_rank'] = rank
    for rank, candidate in enumerate(by_ce, 1):
        candidate['ce_rank'] = rank
    selected = {c['source_id'] for c in by_ce[:TOP_K]}
    return scored, [c for c in by_bge if c['source_id'] in selected]


QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '


def select_calibrated(candidates, calibration):
    scored = []
    for candidate in candidates:
        score = candidate['ce_logit']
        pool = calibration['modalities'][candidate['modality']]
        values = pool['sorted_logits']
        if not math.isfinite(score):
            raise ValueError('CE logit must be finite')
        percentile = bisect_right(values, score) / pool['count']
        scored.append(dict(candidate, ce_modality_percentile=percentile))
    tie = lambda c: (str(c['fact_id']), c['modality'])
    by_calibration = sorted(scored, key=lambda c: (-c['ce_modality_percentile'], -c['ce_logit'], *tie(c)))
    selected = {c['source_id'] for c in by_calibration[:5]}
    ordered = sorted((c for c in scored if c['source_id'] in selected),
                     key=lambda c: (-c['bge_cosine'], *tie(c)))
    return scored, ordered


def fit_ecdfs(records):
    pools = {'text': [], 'image': []}
    for row in records:
        for candidate in row['candidate_scores']:
            value = float(candidate['ce_logit'])
            if not math.isfinite(value):
                raise ValueError('CE calibration logits must be finite')
            pools[candidate['modality']].append(value)
    if any(not values for values in pools.values()):
        raise ValueError('CE calibration requires both modalities')
    return {modality: dict(count=len(values), sorted_logits=sorted(values))
            for modality, values in pools.items()}


def calibrated_relevance(logit, modality, calibration):
    if not math.isfinite(logit):
        raise ValueError('CE query logit must be finite')
    pool = calibration['modalities'][modality]
    return bisect_right(pool['sorted_logits'], logit) / pool['count']


def normalize_features(features):
    norms = features.float().norm(dim=-1, keepdim=True)
    if not torch.isfinite(features).all() or not (norms > 0).all():
        raise ValueError('CLIP produced non-finite or zero features')
    return features.float() / norms

"""Cross-encoder evidence selection with per-modality ECDF calibration (bge-reranker-v2-m3).

Functions are copied verbatim from the legacy pipeline (see MIGRATION_SOURCES.json, objects):
rank_sources selects the cross-encoder top-k and presents it in BGE order (appendix:
"BGE top-5 vs five facts that bge-reranker-v2-m3 selects"); select_calibrated selects by the
per-modality ECDF of the cross-encoder logit (appendix: "percentile calibration").

The cross-encoder is text-only (bge-reranker-v2-m3 takes a (question, text) pair). It never sees an
image or a video frame: an image candidate is scored by its caption, and a video by its transcript
(empty when there is none). The "CE + static" baseline therefore selects images by caption.
The order shown to the model is the Omni-Aware calibrated relevance (ce_select passes ``rel`` as
the BGE score), not the raw BGE cosine.
"""
from bisect import bisect_right
import gc
import json
import math
import os
from pathlib import Path

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


MODEL_ID = 'BAAI/bge-large-en-v1.5'
MODEL_REVISION = 'd4aa6901d3a41ba39fb536a557fa166f842b0e09'
MODEL_FILES = ['README.md', 'config.json', 'model.safetensors',
               'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
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


def normalize_features(features):
    norms = features.float().norm(dim=-1, keepdim=True)
    if not torch.isfinite(features).all() or not (norms > 0).all():
        raise ValueError('CLIP produced non-finite or zero features')
    return features.float() / norms


def ce_logits(examples, cache_path, model_path, device=None, batch_size=16):
    """qid -> cid -> bge-reranker logit of (question, candidate text); cached, resumable."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_path = Path(cache_path)
    scores = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [e for e in examples if e["qid"] not in scores]
    if not todo:
        return scores
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    class _OnDevice:
        def __call__(self, *a, **kw):
            return tokenizer(*a, **kw).to(device)

    for e in todo:
        texts = [{"text": f"{c['title']}: {c['text']}" if c["title"] and c["title"] != c["text"] else c["text"]}
                 for c in e["candidates"]]
        values, _ = score_ce(e["question"], texts, _OnDevice(), model, batch_size)
        scores[e["qid"]] = {c["cid"]: v for c, v in zip(e["candidates"], values, strict=True)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(scores))
    os.replace(tmp, cache_path)
    del model
    gc.collect()
    return scores


def ce_select(candidates, rel, logits, calibration=None):
    """CE top-k presented in retriever-relevance order (rank_sources), or per-modality CE-percentile
    top-k (select_calibrated) when a calibration {'modalities': {m: {count, sorted_logits}}} is given."""
    rows = [dict(c, fact_id=c["cid"]) for c in candidates]
    scored, selected = rank_sources(rows, [rel[c["cid"]] for c in candidates], [logits[c["cid"]] for c in candidates])
    if calibration is not None:
        _, selected = select_calibrated(scored, calibration)
    by_cid = {c["cid"]: c for c in candidates}
    return [by_cid[s["fact_id"]] for s in selected]

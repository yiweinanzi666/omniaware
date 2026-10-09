"""CPU caption/text retrieval: CE selects five sources, BGE orders the five."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from .prepare_webqa import ROOT, MODEL_REVISION, QUERY_INSTRUCTION, sha256_file, write_jsonl


INPUTS = ROOT / 'data/multimodal_development/inputs.jsonl'
OUTPUT = ROOT / 'runs/multimodal_retrieval_development'
BGE = ROOT / 'cache/bge-large-en-v1.5'
CE = ROOT / 'cache/bge-reranker-v2-m3'
CE_REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
PROTOCOL = ROOT / 'MULTIMODAL_32_PROTOCOL.md'
TOP_K, MAX_LENGTH = 5, 512
CANDIDATE_FIELDS = {'caption', 'fact', 'fact_id', 'image_id', 'imgUrl', 'modality',
                    'snippet_id', 'text', 'title', 'url'}


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_cached_records(path, rows):
    records = read_jsonl(path)
    cached = {record['qid']: record for record in records}
    if len(cached) != len(records) or set(cached) - {row['qid'] for row in rows}:
        raise ValueError('cached records contain duplicate or unexpected qids')
    return cached


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def load_inputs(path):
    rows = read_jsonl(path)
    if not rows or len({row['qid'] for row in rows}) != len(rows):
        raise ValueError('input requires nonempty, unique qids')
    for row in rows:
        if set(row) != {'qid', 'question', 'source_split', 'subset', 'candidates'}:
            raise ValueError('retrieval accepts only the frozen label-free input schema')
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError(f"empty question: {row['qid']}")
        candidates = row['candidates']
        keys = [(c['modality'], str(c['fact_id'])) for c in candidates]
        if len(candidates) < TOP_K or len(keys) != len(set(keys)):
            raise ValueError(f"need at least five distinct sources: {row['qid']}")
        for c in candidates:
            if set(c) - CANDIDATE_FIELDS or c['modality'] not in ('text', 'image'):
                raise ValueError(f"invalid candidate schema: {row['qid']}")
            if not isinstance(c['text'], str) or not c['text'].strip():
                raise ValueError(f"empty retrieval text: {row['qid']} {c['fact_id']}")
            if c['text'] != c['caption' if c['modality'] == 'image' else 'fact']:
                raise ValueError('candidate.text must retain its original caption/fact')
    return rows


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


def validate_bge_record(row, record):
    if record['qid'] != row['qid'] or record['source_row_sha256'] != row_hash(row):
        raise ValueError('saved BGE record does not match input source')
    count = len(row['candidates'])
    if len(record['bge_cosines']) != count or len(record['candidate_input_tokens']) != count:
        raise ValueError('saved BGE record has mismatched candidate counts')
    if any(not math.isfinite(score) or abs(score) > 1.000001 for score in record['bge_cosines']):
        raise ValueError('saved BGE cosine must be finite and within [-1, 1]')


def validate_retrieval_record(row, record, bge_record):
    if record['source_row_sha256'] != row_hash(row) or any(
            record[key] != row[key] for key in ('qid', 'question', 'source_split', 'subset')):
        raise ValueError('saved retrieval record does not match input source')
    candidates = record['candidates']
    if len(candidates) != len(row['candidates']):
        raise ValueError('saved retrieval record has mismatched candidate counts')
    expected, selected = rank_sources(row['candidates'], bge_record['bge_cosines'],
                                      [c['ce_logit'] for c in candidates])
    for actual, wanted, tokens in zip(candidates, expected, bge_record['candidate_input_tokens'], strict=True):
        if any(actual.get(key) != value for key, value in wanted.items()) or actual['bge_input_tokens'] != tokens:
            raise ValueError('saved retrieval candidate differs from source or model scores')
    selected_ids = [c['source_id'] for c in selected]
    actual_by_id = {c['source_id']: c for c in candidates}
    if (len(selected_ids) != TOP_K or record['selected_source_ids'] != selected_ids
            or record['ordered_sources'] != [actual_by_id[source_id] for source_id in selected_ids]
            or record['selected_image_count'] != sum(c['modality'] == 'image' for c in selected)):
        raise ValueError('saved retrieval selection differs from CE top-five/BGE ordering')


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


def prepare_config(rows, args):
    path = args.output / 'config.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['identity']['inputs_sha256'] != sha256_file(args.inputs):
            raise ValueError('saved retrieval input differs; use the original inputs or a separate output directory')
        return saved
    import torch
    import transformers
    identity = dict(schema_version=1, inputs_path=str(args.inputs.resolve()),
        inputs_sha256=sha256_file(args.inputs), source_code_sha256=sha256_file(__file__),
        protocol_sha256=sha256_file(PROTOCOL), samples=len(rows),
        qids=[row['qid'] for row in rows], candidate_occurrences=sum(len(row['candidates']) for row in rows),
        device='cpu', dtype='float32', attention='sdpa', threads=args.threads,
        batch_size=args.batch_size, max_length=MAX_LENGTH, top_k=TOP_K,
        retrieval_text='candidate.text exactly: original caption for image, original fact for text; title excluded',
        selection='descending CE raw logit, then string fact_id, then modality; select five',
        presentation='descending BGE cosine, then string fact_id, then modality',
        bge=dict(model_id='BAAI/bge-large-en-v1.5', revision=MODEL_REVISION,
            query_instruction=QUERY_INSTRUCTION, pooling='CLS L2-normalized cosine',
            assets_sha256={p.name:sha256_file(p) for p in sorted(BGE.iterdir()) if p.is_file()}),
        ce=dict(model_id='BAAI/bge-reranker-v2-m3', revision=CE_REVISION,
            input='question/candidate.text pair; no instruction',
            assets_sha256={p.name:sha256_file(p) for p in sorted(CE.iterdir()) if p.is_file()}),
        torch=torch.__version__, transformers=transformers.__version__,
        relevance='Not assigned here; root will fit separate text/image-caption BGE calibration ECDFs',
        cost_boundary='This CPU run only; model load and each question scoring reported separately; no T17 timing')
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), identity=identity)
    save_json(path, config)
    (args.output / 'source').mkdir(exist_ok=True)
    for source in (Path(__file__), PROTOCOL, Path(__file__).with_name('test_multimodal_retrieval.py')):
        (args.output / 'source' / source.name).write_bytes(source.read_bytes())
    return config


def run(args):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    rows = load_inputs(args.inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    config = prepare_config(rows, args)
    batch_size = config['identity']['batch_size']
    torch.set_num_threads(config['identity']['threads'])
    torch.manual_seed(17)
    config_hash = sha256_file(args.output / 'config.json')
    started = time.perf_counter()
    bge_path, result_path = args.output/'bge_scores.jsonl', args.output/'retrieval.jsonl'
    bge = load_cached_records(bge_path, rows)
    for row in rows:
        if row['qid'] in bge:
            validate_bge_record(row, bge[row['qid']])
    timing_path = args.output / 'load_timings.json'
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    if len(bge) != len(rows):
        tick = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(BGE, local_files_only=True)
        model = AutoModel.from_pretrained(BGE, local_files_only=True, dtype=torch.float32,
                                          attn_implementation='sdpa').to('cpu').eval()
        timings['bge_model_load_seconds'] = time.perf_counter()-tick
        save_json(timing_path, timings)
        for index, row in enumerate(rows, 1):
            if row['qid'] in bge:
                continue
            tick = time.perf_counter()
            texts = [QUERY_INSTRUCTION+row['question']] + [c['text'] for c in row['candidates']]
            vectors, lengths = encode_bge(texts, tokenizer, model, batch_size)
            scores = (vectors[1:] @ vectors[0]).tolist()
            bge[row['qid']] = dict(qid=row['qid'], source_row_sha256=row_hash(row), config_sha256=config_hash,
                bge_cosines=scores, query_input_tokens=lengths[0], candidate_input_tokens=lengths[1:],
                bge_seconds=time.perf_counter()-tick)
            write_jsonl(bge_path, list(bge.values()))
            print(json.dumps(dict(stage='bge', questions=index, total=len(rows), seconds=bge[row['qid']]['bge_seconds'])),flush=True)
        del vectors, model, tokenizer
        gc.collect()
    results = load_cached_records(result_path, rows)
    for row in rows:
        validate_bge_record(row, bge[row['qid']])
        if row['qid'] in results:
            validate_retrieval_record(row, results[row['qid']], bge[row['qid']])
    if len(results) != len(rows):
        tick = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(CE, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(CE, local_files_only=True,
            dtype=torch.float32, attn_implementation='sdpa').to('cpu').eval()
        timings['ce_model_load_seconds'] = time.perf_counter()-tick
        save_json(timing_path, timings)
        for index, row in enumerate(rows, 1):
            if row['qid'] in results:
                continue
            tick = time.perf_counter()
            b = bge[row['qid']]
            scores, lengths = score_ce(row['question'], row['candidates'], tokenizer, model, batch_size)
            candidates, selected = rank_sources(row['candidates'], b['bge_cosines'], scores)
            for candidate, bge_length, ce_length in zip(candidates, b['candidate_input_tokens'], lengths, strict=True):
                candidate.update(bge_input_tokens=bge_length, ce_input_tokens=ce_length)
            results[row['qid']] = dict(qid=row['qid'], question=row['question'], source_split=row['source_split'],
                subset=row['subset'], config_sha256=config_hash, source_row_sha256=row_hash(row),
                candidates=candidates, ordered_sources=selected, selected_source_ids=[c['source_id'] for c in selected],
                selected_image_count=sum(c['modality']=='image' for c in selected),
                bge_seconds=b['bge_seconds'], ce_and_order_seconds=time.perf_counter()-tick)
            write_jsonl(result_path, list(results.values()))
            print(json.dumps(dict(stage='ce', questions=index, total=len(rows),
                selected_images=results[row['qid']]['selected_image_count'],
                seconds=results[row['qid']]['ce_and_order_seconds'])),flush=True)
        del model, tokenizer
        gc.collect()
    ordered = [results[row['qid']] for row in rows]
    for row, record in zip(rows, ordered, strict=True):
        validate_retrieval_record(row, record, bge[row['qid']])
    bge_seconds = sum(r['bge_seconds'] for r in ordered)
    ce_seconds = sum(r['ce_and_order_seconds'] for r in ordered)
    completion = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        questions=len(rows), scored_candidate_occurrences=sum(len(r['candidates']) for r in ordered),
        selected_sources=sum(len(r['ordered_sources']) for r in ordered),
        selected_modalities=dict(Counter(c['modality'] for r in ordered for c in r['ordered_sources'])),
        selected_images_per_question=dict(Counter(r['selected_image_count'] for r in ordered)),
        config_sha256=config_hash, retrieval_sha256=sha256_file(result_path), bge_sha256=sha256_file(bge_path),
        timings=dict(**timings, bge_scoring_seconds=bge_seconds, ce_and_order_seconds=ce_seconds,
                     cpu_scoring_seconds=bge_seconds+ce_seconds, invocation_elapsed_seconds=time.perf_counter()-started,
                     candidate_occurrences_per_scoring_second=config['identity']['candidate_occurrences']/(bge_seconds+ce_seconds)))
    save_json(args.output/'completion.json', completion)
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=INPUTS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    print(json.dumps(run(parser.parse_args()), indent=2),flush=True)

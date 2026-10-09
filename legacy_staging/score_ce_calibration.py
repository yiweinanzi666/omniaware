"""Fit CE logit ECDFs for the frozen 500 label-free calibration queries."""

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import time

from .multimodal_retrieval import (CE, CE_REVISION, MAX_LENGTH, load_cached_records,
                                   load_inputs, row_hash, save_json, score_ce)
from .prepare_webqa import ROOT, sha256_file, write_jsonl


INPUTS = ROOT / 'data/multimodal_calibration/inputs.jsonl'
OUTPUT = ROOT / 'runs/ce_calibration'
PROTOCOL = ROOT / 'CE_CALIBRATION_PROTOCOL.md'
MODALITY_COUNTS = {'text': 7698, 'image': 8666}


def validate_record(row, record):
    if record['qid'] != row['qid'] or record['source_row_sha256'] != row_hash(row):
        raise ValueError('saved CE calibration record differs from actual input')
    if len(record['candidate_scores']) != len(row['candidates']):
        raise ValueError('saved CE calibration candidate count differs')
    for source, score in zip(row['candidates'], record['candidate_scores'], strict=True):
        if (score['modality'] != source['modality'] or score['fact_id'] != source['fact_id']
                or score['source_id'] != f"{source['modality']}:{source['fact_id']}"):
            raise ValueError('saved CE calibration candidate identity differs')
        if not math.isfinite(score['ce_logit']):
            raise ValueError('CE calibration logits must be finite')


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


def snapshot_execution(args, rows):
    import torch
    import transformers
    execution_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = args.output / 'executions' / execution_id
    (directory / 'source').mkdir(parents=True)
    files = [Path(__file__), Path(__file__).with_name('multimodal_retrieval.py'),
             Path(__file__).with_name('prepare_webqa.py'), PROTOCOL]
    for source in files:
        (directory / 'source' / source.name).write_bytes(source.read_bytes())
    config = dict(execution_id=execution_id, pid=os.getpid(),
        inputs_path=str(args.inputs.resolve()), inputs_sha256=sha256_file(args.inputs),
        code_sha256={source.name: sha256_file(source) for source in files},
        model_id='BAAI/bge-reranker-v2-m3', model_revision=CE_REVISION,
        model_assets_sha256={p.name: sha256_file(p) for p in sorted(CE.iterdir()) if p.is_file()},
        questions=len(rows), candidate_occurrences=sum(len(r['candidates']) for r in rows),
        device='cpu', dtype='float32', attention='sdpa', threads=4, batch_size=2,
        seed=17, max_length=MAX_LENGTH, torch=torch.__version__, transformers=transformers.__version__,
        input='question/candidate.text pair; original fact or image caption; no instruction or title',
        mode='real_numerical_smoke' if args.smoke else '500_query_calibration',
        cost_boundary='this CE CPU execution only; no BGE, pixels, Qwen, answers, or roles')
    save_json(directory / 'execution.json', config)
    return execution_id, directory


def run(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    started = time.perf_counter()
    rows = load_inputs(args.inputs)
    counts = Counter(c['modality'] for row in rows for c in row['candidates'])
    if len(rows) != 500 or any(row['source_split'] != 'train' for row in rows) or counts != MODALITY_COUNTS:
        raise ValueError('protocol requires the frozen 500 train queries and 16364 candidates')
    execution_id, execution = snapshot_execution(args, rows)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    path = args.output / 'ce_scores.jsonl'
    saved = {} if args.smoke else load_cached_records(path, rows)
    for row in rows:
        if row['qid'] in saved:
            validate_record(row, saved[row['qid']])
    scored_questions = 0
    if args.smoke or len(saved) != len(rows):
        tick = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(CE, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(CE, local_files_only=True,
            dtype=torch.float32, attn_implementation='sdpa').to('cpu').eval()
        save_json(execution / 'model_load.json', dict(model_load_seconds=time.perf_counter()-tick))
        if args.smoke:
            row = rows[0]
            candidates = [next(c for c in row['candidates'] if c['modality'] == modality)
                          for modality in ('text', 'image')]
            tick = time.perf_counter()
            batch2, lengths2 = score_ce(row['question'], candidates, tokenizer, model, 2)
            batch1, lengths1 = score_ce(row['question'], candidates, tokenizer, model, 1)
            differences = [abs(a-b) for a,b in zip(batch2, batch1, strict=True)]
            if (not all(math.isfinite(v) for v in batch2 + batch1)
                    or max(differences) > 1e-4 or lengths2 != lengths1):
                raise ValueError('real CE batch1/batch2 numerical verification failed')
            completion = dict(status='passed', execution_id=execution_id, qid=row['qid'],
                source_row_sha256=row_hash(row), source_ids=[f"{c['modality']}:{c['fact_id']}" for c in candidates],
                logits_batch2=batch2, logits_batch1=batch1, max_absolute_difference=max(differences),
                input_tokens=lengths2, scoring_seconds=time.perf_counter()-tick,
                invocation_elapsed_seconds=time.perf_counter()-started)
            del model, tokenizer
            gc.collect()
            save_json(execution / 'completion.json', completion)
            save_json(args.output / 'completion.json', completion)
            return completion
        for index, row in enumerate(rows, 1):
            if row['qid'] in saved:
                continue
            tick = time.perf_counter()
            logits, lengths = score_ce(row['question'], row['candidates'], tokenizer, model, 2)
            record = dict(qid=row['qid'], source_row_sha256=row_hash(row), execution_id=execution_id,
                candidate_scores=[dict(source_id=f"{c['modality']}:{c['fact_id']}", modality=c['modality'],
                    fact_id=c['fact_id'], ce_logit=value, input_tokens=length)
                    for c,value,length in zip(row['candidates'], logits, lengths, strict=True)],
                ce_seconds=time.perf_counter()-tick)
            validate_record(row, record)
            saved[row['qid']] = record
            write_jsonl(path, list(saved.values()))
            scored_questions += 1
            if index == 1 or index % 10 == 0 or index == len(rows):
                print(json.dumps(dict(stage='ce_calibration', questions=index, total=len(rows),
                    elapsed_seconds=time.perf_counter()-started)), flush=True)
        del model, tokenizer
        gc.collect()
    records = [saved[row['qid']] for row in rows]
    for row, record in zip(rows, records, strict=True):
        validate_record(row, record)
    ecdf = dict(model_id='BAAI/bge-reranker-v2-m3', model_revision=CE_REVISION,
        score_records_sha256=sha256_file(path), question_count=len(rows), max_length=MAX_LENGTH,
        transform='r=bisect_right(sorted_logits[modality],ce_logit)/count[modality]',
        modalities=fit_ecdfs(records))
    save_json(args.output / 'ce_ecdf.json', ecdf)
    seconds = sum(record['ce_seconds'] for record in records)
    loads = [json.loads(p.read_text())['model_load_seconds'] for p in (args.output / 'executions').glob('*/model_load.json')]
    completion = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        execution_id=execution_id, questions=len(records), candidate_occurrences=sum(counts.values()),
        modalities={m:p['count'] for m,p in ecdf['modalities'].items()},
        ce_scores_sha256=sha256_file(path), ecdf_sha256=sha256_file(args.output / 'ce_ecdf.json'),
        cpu_scoring_seconds=seconds, accumulated_model_load_seconds=sum(loads),
        candidate_occurrences_per_second=sum(counts.values())/seconds,
        invocation_scored_questions=scored_questions, invocation_elapsed_seconds=time.perf_counter()-started)
    save_json(execution / 'completion.json', completion)
    save_json(args.output / 'completion.json', completion)
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=INPUTS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--smoke', action='store_true')
    print(json.dumps(run(parser.parse_args()), indent=2), flush=True)

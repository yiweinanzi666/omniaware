"""Compare one frozen modality-ECDF selection with saved raw-CE top-five."""

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .multimodal_retrieval import read_jsonl, row_hash, save_json
from .prepare_webqa import ROOT, sha256_file, write_jsonl


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


def measure(selected, gold_images, gold_all):
    hits = len(set(selected) & gold_images)
    return dict(image_recall=hits / len(gold_images), image_any_hit=int(hits > 0),
                image_all_gold=int(hits == len(gold_images)),
                source_precision=len(set(selected) & gold_all) / len(selected),
                selected_images=sum(s.startswith('image:') for s in selected))


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    groups = ('multimodal_development', 'multimodal_development_expansion')
    input_paths = [ROOT / 'data' / group / 'inputs.jsonl' for group in groups]
    retrieval_paths = [ROOT / 'runs' / group / 'retrieval.jsonl' for group in
                       ('multimodal_retrieval_development', 'multimodal_retrieval_expansion')]
    role_paths = [ROOT / 'data' / group / 'candidate_roles.jsonl' for group in groups]
    inputs = {row['qid']: row for path in input_paths for row in read_jsonl(path)}
    original = {row['qid']: row for path in retrieval_paths for row in read_jsonl(path)}
    pilot_ids = {row['qid'] for row in read_jsonl(input_paths[0])}
    calibration = json.loads(args.ecdf.read_text())
    for modality, pool in calibration['modalities'].items():
        values = pool['sorted_logits']
        if not values or pool['count'] != len(values) or values != sorted(values) or not all(map(math.isfinite, values)):
            raise ValueError(f'Invalid calibration distribution: {modality}')
    if len(inputs) != 128 or set(inputs) != set(original):
        raise ValueError('Require all 128 input and retrieval questions')
    protocol = ROOT / 'CE_MODALITY_SELECTION_PROTOCOL.md'
    save_json(args.output / 'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        stage='train development', questions=128, candidates=['raw_ce', 'ce_modality_ecdf'],
        primary_metric='mean per-question positive-image recall at mixed total top-five',
        protocol_sha256=sha256_file(protocol), code_sha256=sha256_file(__file__),
        inputs={str(path): sha256_file(path) for path in [*input_paths, *retrieval_paths, args.ecdf]},
        bootstrap=dict(unit='paired question', samples=10000, seed=20261003),
        model_calls=0))
    (args.output / 'source').mkdir(exist_ok=True)
    for path in (Path(__file__), protocol):
        (args.output / 'source' / path.name).write_bytes(path.read_bytes())
    transformed = {}
    ecdf_hash = sha256_file(args.ecdf)
    plan_hash = sha256_file(args.output / 'plan.json')
    for qid, row in original.items():
        if row['source_row_sha256'] != row_hash(inputs[qid]):
            raise ValueError(f'Retrieval differs from input: {qid}')
        expected_ids = {f"{c['modality']}:{c['fact_id']}" for c in inputs[qid]['candidates']}
        if expected_ids != {c['source_id'] for c in row['candidates']}:
            raise ValueError(f'Retrieval candidates differ from input: {qid}')
        scored, ordered = select_calibrated(row['candidates'], calibration)
        transformed[qid] = dict(row, candidates=scored, ordered_sources=ordered,
            selected_source_ids=[c['source_id'] for c in ordered],
            selected_image_count=sum(c['modality'] == 'image' for c in ordered),
            selection_method='CE modality ECDF top-five; original BGE presentation order',
            selection_calibration_sha256=ecdf_hash, selection_plan_sha256=plan_hash,
            derivation_source_row_sha256=row_hash(row))
    write_jsonl(args.output / 'retrieval.jsonl', list(transformed.values()))
    roles = {row['qid']: row for path in role_paths for row in read_jsonl(path)}
    if set(roles) != set(inputs):
        raise ValueError('Gold roles must cover exactly the input questions')
    results = []
    for qid in sorted(inputs):
        labels = roles[qid]['candidate_roles']
        candidate_ids = {c['source_id'] for c in original[qid]['candidates']}
        if candidate_ids != {f"{c['modality']}:{c['fact_id']}" for c in labels}:
            raise ValueError(f'Role candidate set differs: {qid}')
        gold = {f"image:{c['fact_id']}" for c in labels if c['modality'] == 'image' and c['role'] == 'pos'}
        gold_all = {f"{c['modality']}:{c['fact_id']}" for c in labels if c['role'] == 'pos'}
        if not gold:
            raise ValueError(f'Image-task query has no positive image: {qid}')
        for method, records in [('raw_ce', original), ('ce_modality_ecdf', transformed)]:
            selected = records[qid]['selected_source_ids']
            results.append(dict(qid=qid, cohort='pilot32' if qid in pilot_ids else 'expansion96',
                method=method, selected_source_ids=selected, gold_image_ids=sorted(gold),
                **measure(selected, gold, gold_all)))
    write_jsonl(args.output / 'per_question.jsonl', results)
    summaries = {}
    for cohort in ('all128', 'pilot32', 'expansion96'):
        selected_rows = [row for row in results if cohort == 'all128' or row['cohort'] == cohort]
        by_method = {method: [row for row in selected_rows if row['method'] == method]
                     for method in ('raw_ce', 'ce_modality_ecdf')}
        summaries[cohort] = {method: {key: float(np.mean([row[key] for row in rows]))
            for key in ('image_recall', 'image_any_hit', 'image_all_gold', 'source_precision', 'selected_images')}
            for method, rows in by_method.items()}
        delta = np.array([row['image_recall'] for row in by_method['ce_modality_ecdf']]) - np.array(
            [row['image_recall'] for row in by_method['raw_ce']])
        rng = np.random.default_rng(20261003)
        boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
        summaries[cohort]['paired_recall'] = dict(questions=len(delta), difference_pp=float(100 * delta.mean()),
            ci95_pp=(100 * np.quantile(boot, [.025, .975])).tolist())
    summary = dict(completed_utc=datetime.now(timezone.utc).isoformat(), cohorts=summaries,
        changed_selections=sum(set(original[q]['selected_source_ids']) != set(transformed[q]['selected_source_ids'])
                               for q in inputs),
        candidate_roles_sha256={str(p): sha256_file(p) for p in role_paths},
        plan_sha256=sha256_file(args.output / 'plan.json'),
        retrieval_sha256=sha256_file(args.output / 'retrieval.jsonl'),
        answer_references_or_generated_answers_read=False, model_calls=0)
    save_json(args.output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ecdf', type=Path, default=ROOT / 'runs/ce_calibration/ce_ecdf.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/ce_modality_selection_development')
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))

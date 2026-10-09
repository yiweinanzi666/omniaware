"""Freeze full official E5 source records without labels, model calls, or old-answer edits."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prepare_eab_recovery_confirmation import project

SOURCE_FILES = {}
CITATION_INSTRUCTION = (
    'Answer the question using the supplied evidence and images. '
    'Write each answer as a complete factual statement on a separate line. '
    'Cite the supplied evidence that supports each statement using its exact [E<number>] identifier. '
    'Use only identifiers present in the supplied evidence. Do not include explanations or a reference list.'
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def register(path, expected=None):
    path = Path(path)
    value = digest(path)
    if expected:
        assert value == expected, path
    SOURCE_FILES[str(path.resolve())] = value
    return dict(path=str(path.resolve()), sha256=value)


def rows(path):
    register(path)
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(name, records):
    path = OUT / name
    with path.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    return dict(path=name, rows=len(records), bytes=path.stat().st_size, sha256=digest(path))


def source_record(number, source_id, modality, text, **metadata):
    return dict(citation_id=f'E{number}', source_id=source_id, modality=modality,
                rendered_text=text, rendered_text_sha256=hashlib.sha256(text.encode()).hexdigest(), **metadata)


def base(dataset, row, evidence, origin, **metadata):
    assert len({e['source_id'] for e in evidence}) == len(evidence)
    return dict(dataset=dataset, qid=str(row['qid']), question=row['question'],
                source_split=row['source_split'], evidence=evidence, origin=origin,
                status='CITATION_SOURCE_RECORD_READY', intended_variants=['static_concat', 'eab'],
                citation_instruction=CITATION_INSTRUCTION, raw_answer=None,
                generation_calls=0, nli_calls=0, **metadata)


def okvqa():
    assets = {str(r['qid']):r for r in rows(ROOT / 'data/okvqa_val5046_assets/validation_candidates.jsonl')}
    prepared = rows(ROOT / 'data/okvqa_val5046_prepared/validation_examples.jsonl')
    assert len(assets) == len(prepared) == 5046
    result = []
    for row in prepared:
        asset = assets[str(row['qid'])]
        trace_path = ROOT / 'data/okvqa_val5046_prepared' / row['retrieved']['trace']
        origin = register(trace_path, row['retrieved']['trace_sha256'])
        trace = json.loads(trace_path.read_text())
        assert len(trace['knowledge']) == 5
        evidence = []
        for n, source in enumerate(trace['knowledge'], 1):
            assert source['text'] in trace['prompt'] and source['retained_tokens'] <= 256
            evidence.append(source_record(n, source['source_id'], 'text', source['text'],
                document_id=source['doc_id'], original_rank=source['rank'], r_j=source['r_j'],
                retained_tokens=source['retained_tokens'], original_tokens=source['original_tokens']))
        assert Path(asset['image_path']).is_file()
        result.append(base('okvqa', row, evidence, origin,
            question_image=dict(source_id='query_image', path=asset['image_path'],
                sha256=asset['image_sha256'], role='question_condition_not_text_citation'),
            original_input_refs={k:row[k] for k in ('initial', 'retrieved', 'image_inputs')},
            uncertainty_policy='reuse original frozen query-image observation and 500-question calibration'))
    return result


def mmqa():
    # Execute only the exact existing pure renderer; never import its model dependencies.
    source_path = ROOT / 'mmqa_packing.py'
    register(source_path)
    tree = ast.parse(source_path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'source_text')
    namespace = {'json':json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
    renderer = namespace['source_text']
    inputs = rows(ROOT / 'data/mmqa_dev1280/inputs.jsonl')
    examples = {r['qid']:r for r in rows(ROOT / 'data/mmqa_dev1280/examples.jsonl')}
    relevance = {r['qid']:r for r in rows(ROOT / 'runs/mmqa_dev1280/relevance/scores.jsonl')}
    assert len(inputs) == len(examples) == len(relevance) == 1280
    result = []
    for row in inputs:
        example = examples[row['qid']]
        spec = example['prompts']['original']
        origin = register(spec['trace_file'], spec['trace_file_sha256'])
        prompt = json.loads(Path(spec['trace_file']).read_text())['prompt']
        scores = {s['source_id']:s for s in relevance[row['qid']]['candidate_scores']}
        assert [s['source_id'] for s in row['candidates']] == example['source_ids']
        evidence = []
        for n, source in enumerate(row['candidates'], 1):
            text, score = renderer(source), scores[source['source_id']]
            assert text in prompt
            metadata = dict(document_id=source['document_id'], r_j=score['r_j'], original_rank=n)
            if source['modality'] == 'image':
                assert Path(source['image_path']).is_file()
                metadata.update(image_path=source['image_path'], image_sha256=score['image_sha256'],
                    nli_visible_content='original image title only; pixels are not passed to text NLI')
            evidence.append(source_record(n, source['source_id'], source['modality'], text, **metadata))
        result.append(base('mmqa_table', row, evidence, origin,
            original_input_refs=spec, uncertainty_policy='reuse original frozen full-QA observation and calibration'))
    return result


def webqa():
    inputs = rows(ROOT / 'runs/webqa_full4966_cpu_retrieval/inputs.jsonl')
    categories = {r['qid']:r for r in rows(ROOT / 'runs/webqa_full4966_readonly_audit/validation_ids.jsonl')}
    assert len(inputs) == len(categories) == 4966
    assert sum(r['category'] != 'text' for r in categories.values()) == 2511
    pooled = write_rows('webqa4966_unlabeled_pooled_inputs.jsonl', inputs)
    # Fixed snapshot of only fully written final JSON paths, never partial retrieval outputs.
    paths = {p.stem:p for p in (ROOT / 'runs/webqa_full4966_formal/prepared').glob('*.json')}
    result = []
    for row in inputs:
        category = categories[row['qid']]['category']
        evidence, origin = [], dict(path=pooled['path'], sha256=pooled['sha256'], qid=row['qid'])
        metadata = dict(category=category, official_image_question=category != 'text',
                        historical_1247_membership=None, candidate_count=len(row['candidates']))
        if row['qid'] in paths:
            path = paths[row['qid']]
            origin = register(path)
            prepared = json.loads(path.read_text())
            assert prepared['question'] == row['question']
            assert len(prepared['evidence']) == 5
            for n, source in enumerate(prepared['evidence'], 1):
                extra = dict(document_id=source['fact_id'], r_j=source['r_j'], original_rank=n)
                if source['modality'] == 'image':
                    extra.update(image_path=source['image_path'],
                        image_sha256=prepared['image_payload_sha256'][source['image_path']],
                        nli_visible_content='original retained caption only; pixels are not passed to text NLI')
                evidence.append(source_record(n, source['source_id'], source['modality'], source['text'], **extra))
            assert [e['source_id'] for e in evidence] == prepared['selected_source_ids']
            metadata.update(image_provenance=prepared['image_provenance'], retrieval_sha256=prepared['retrieval_sha256'])
        record = base('webqa', row, evidence, origin, **metadata)
        if not evidence:
            record['status'] = 'PENDING_EXISTING_CPU_TOP5_PREPARATION'
        record['uncertainty_policy'] = 'reuse original query-only observation and frozen 500-question calibration'
        result.append(record)
    return result, pooled


def citation_audit():
    paths = sorted((ROOT / 'runs').glob('*/predictions.jsonl'))
    paths += [ROOT / 'runs/webqa_ctx_controls/qa_scored/predictions.jsonl']
    import re
    records = []
    for path in paths:
        counts = Counter(rows=0, rows_with_E_citation=0, rows_with_numeric_citation=0, incomplete_snapshot_lines=0)
        # Hash and parse one immutable in-memory snapshot of potentially still-active output.
        payload = path.read_bytes()
        for line in payload.decode().splitlines():
            try:
                row = project(line, {'qid','variant','prediction','raw_answer','answer_token_ids','raw_token_ids'})
            except (ValueError, IndexError):
                counts['incomplete_snapshot_lines'] += 1
                continue
            counts['rows'] += 1
            answer = row.get('raw_answer', row.get('prediction', ''))
            if isinstance(answer, str) and re.search(r'\[E\d+\]', answer):
                counts['rows_with_E_citation'] += 1
            if isinstance(answer, str) and re.search(r'\[\d+\]', answer):
                counts['rows_with_numeric_citation'] += 1
        records.append(dict(path=str(path.relative_to(ROOT)), snapshot_sha256=hashlib.sha256(payload).hexdigest(),
                            snapshot_bytes=len(payload), **counts))
    write('actual_prediction_citation_audit.json', dict(scope='All direct runs/*/predictions.jsonl plus current WebQA ctx QA output; archive and scored duplicates excluded',
        decoded_fields=['qid','variant','prediction','raw_answer','answer_token_ids','raw_token_ids'],
        gold_fields_decoded=False, files=records, totals=dict(sum((Counter({k:v for k,v in r.items() if isinstance(v,int) and k!='snapshot_bytes'}) for r in records), Counter()))))


def main():
    outputs = []
    for name, builder in [('okvqa5046',okvqa), ('mmqa1280',mmqa)]:
        data = builder()
        item = write_rows(name + '_citation_inputs.jsonl', data)
        item['source_occurrences'] = sum(len(r['evidence']) for r in data)
        item['status_counts'] = dict(Counter(r['status'] for r in data))
        outputs.append(item)
        print(json.dumps(item), flush=True)
    data, pooled = webqa()
    item = write_rows('webqa4966_citation_inputs.jsonl', data)
    item['status_counts'] = dict(Counter(r['status'] for r in data))
    item['source_occurrences'] = sum(len(r['evidence']) for r in data)
    outputs.extend([item, pooled])
    citation_audit()
    write('source_files_sha256.json', SOURCE_FILES)
    report = dict(prepared_utc=datetime.now(timezone.utc).isoformat(), status='SOURCES_PREPARED_WEBQA_TOP5_CONTINUATION_PENDING',
        full_target_questions=11292, webqa_image_questions=2511, outputs=outputs,
        citation_instruction=CITATION_INSTRUCTION, source_files_verified=len(SOURCE_FILES),
        raw_reference_fields_read=False, model_loads=0, generation_calls=0, nli_calls=0,
        new_citation_answers=0, image_payload_verification='existence plus inherited frozen hash references; no new pixel inference',
        historical_table5_replication_claimed=False)
    write('preparation.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

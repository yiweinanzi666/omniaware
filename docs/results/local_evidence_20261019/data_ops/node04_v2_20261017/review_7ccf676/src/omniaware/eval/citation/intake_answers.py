"""Prepare complete-denominator CiteRec requests from newly generated E5 answers."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from .attribution import sentence_spans, splitter_name
from .requests import requests_for_answer

MARKER = re.compile(r'\[(E\d+)\]')


def prepare_answer(answer, source):
    for key in ('dataset', 'qid', 'variant', 'raw_answer', 'raw_token_ids', 'generation_identity'):
        if key not in answer:
            raise ValueError(f'Missing raw generation field: {key}')
    assert source['status'] == 'CITATION_SOURCE_RECORD_READY'
    assert (answer['dataset'], str(answer['qid'])) == (source['dataset'], source['qid'])
    assert isinstance(answer['raw_answer'], str) and isinstance(answer['raw_token_ids'], list)
    assert answer['generation_identity'], 'Record generation config and source-input identity'
    assert all(isinstance(token, int) for token in answer['raw_token_ids'])
    statements = []
    for begin, end in sentence_spans(answer['raw_answer']):
        raw = answer['raw_answer'][begin:end]
        text = MARKER.sub('', raw).strip()
        if text:
            statements.append(dict(text=text, citation_ids=list(dict.fromkeys(MARKER.findall(raw))),
                                   raw_span=[begin,end], raw_sentence=raw))
    cited = list(dict.fromkeys(MARKER.findall(answer['raw_answer'])))
    status = ('EMPTY_ANSWER_ZERO' if not statements else
              'UNCITED_ANSWER_ZERO' if not cited else 'READY_FOR_NLI')
    row = dict(answer, statements=statements, evidence=source['evidence'])
    requests = requests_for_answer(row) if status == 'READY_FOR_NLI' else []
    # Invalid IDs remain visible. They supply no premise and never earn support.
    for request in requests:
        request['dataset'] = answer['dataset']
        statement = statements[request['statement_index']]
        request['raw_sentence_span'] = statement['raw_span']
        request['raw_sentence'] = statement['raw_sentence']
        request['sentence_local_citation_ids'] = statement['citation_ids']
        request['sentence_splitter'] = splitter_name()   # the splitter shared with Faith/AP (attribution.sentence_spans)
    denominator = max(1, len(statements))
    ledger = dict(dataset=answer['dataset'], qid=str(answer['qid']), variant=answer['variant'],
        status=status, statement_denominator=denominator, emitted_citation_ids=cited,
        empty_answer=not statements, nli_requests=len(requests),
        structural_zero_statements=denominator if not requests else
            sum(r['status'] == 'ALL_CITATIONS_INVALID' for r in requests),
        source_record_sha256=hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        raw_answer_record_sha256=hashlib.sha256(json.dumps(answer, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    return ledger, requests


def prepare(input_path, sources_paths, output, variants=None):
    """Every (dataset, qid) source must have one answer per variant (default: all variants seen)."""
    sources = {}
    for path in sources_paths:
        for line in path.open():
            row = json.loads(line)
            key = (row['dataset'], row['qid'])
            assert key not in sources
            sources[key] = row
    ledgers, requests, seen = [], [], set()
    for line in input_path.open():
        answer = json.loads(line)
        key = (answer['dataset'], str(answer['qid']), answer['variant'])
        assert key not in seen, f'Duplicate answer: {key}'
        seen.add(key)
        ledger, values = prepare_answer(answer, sources[key[:2]])
        ledgers.append(ledger)
        requests.extend(values)
    variants = sorted(variants or {key[2] for key in seen})
    expected = {(dataset, qid, variant) for dataset, qid in sources for variant in variants}
    assert seen == expected, f'Incomplete full denominator: missing={len(expected-seen)}, unexpected={len(seen-expected)}'
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('answer_ledger.jsonl',ledgers), ('cite_rec_nli_inputs.jsonl',requests)]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    report = dict(status='INTAKE_ONLY_NO_NLI_RESULTS', answers=len(ledgers), expected_answers=len(expected),
                  statements=sum(r['statement_denominator'] for r in ledgers), requests=len(requests),
                  generation_calls=0, nli_calls=0, sentence_splitter=splitter_name(),
                  ledger_sha256=hashlib.sha256((output / 'answer_ledger.jsonl').read_bytes()).hexdigest(),
                  requests_sha256=hashlib.sha256((output / 'cite_rec_nli_inputs.jsonl').read_bytes()).hexdigest(),
                  answer_input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  source_files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources_paths})
    (output / 'intake.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', default=None, help='arms that every question must have')
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.sources, args.output, args.variants)))

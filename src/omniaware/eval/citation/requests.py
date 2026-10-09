"""Build NLI inputs from actual [E<number>] citations; never invent citations."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def requests_for_answer(row):
    required = ('qid', 'variant', 'raw_answer', 'raw_token_ids', 'generation_identity', 'statements', 'evidence')
    for field in required:
        row[field]
    cited = list(dict.fromkeys(re.findall(r'\[(E\d+)\]', row['raw_answer'])))
    if not cited:
        raise ValueError(f"No actual citation markers: {row['qid']}/{row['variant']}")
    evidence = {e['citation_id']: e for e in row['evidence']}
    assert len(evidence) == len(row['evidence']) and row['statements']
    valid = [c for c in cited if c in evidence]
    invalid = [c for c in cited if c not in evidence]
    premise = '\n\n'.join(evidence[c]['rendered_text'] for c in valid)
    result = []
    for index, statement in enumerate(row['statements']):
        assert statement['text'].strip()
        assert set(statement['citation_ids']) <= set(cited)
        result.append(dict(qid=row['qid'], variant=row['variant'], statement_index=index,
            answer_record_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            hypothesis=statement['text'], premise=premise,
            all_actual_citation_ids=cited, valid_citation_ids=valid, invalid_citation_ids=invalid,
            cited_source_ids=[evidence[c]['source_id'] for c in valid],
            requested_metric='CiteRec', status='NLI_INPUT_ONLY' if valid else 'ALL_CITATIONS_INVALID',
            generation_identity=row['generation_identity']))
    return result


def prepare(source, destination):
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    requests = [request for row in rows for request in requests_for_answer(row)]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'cite_rec_nli_inputs.jsonl').write_text(''.join(
        json.dumps(r, ensure_ascii=False) + '\n' for r in requests))
    report = dict(status='INPUTS_ONLY_NO_NLI_RESULTS', answers=len(rows), statements=len(requests),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        requests_sha256=hashlib.sha256((destination / 'cite_rec_nli_inputs.jsonl').read_bytes()).hexdigest(),
        generation_calls=0, nli_calls=0,
        remaining='Fix exact Table5 checker/alias/sentence and support aggregation protocol, then execute and record real NLI outputs')
    (destination / 'intake.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), ensure_ascii=False))

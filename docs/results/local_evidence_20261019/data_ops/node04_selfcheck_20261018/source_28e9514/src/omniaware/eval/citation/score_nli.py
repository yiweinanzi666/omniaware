"""Aggregate only complete E5 results, retaining uncited and empty answers as zero."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from .run_nli import expected_checker


def key(row):
    return row['dataset'], row['qid'], row['variant'], row['statement_index']


def score(ledger_path, request_path, result_path):
    intake = json.loads((ledger_path.parent / 'intake.json').read_text())
    assert hashlib.sha256(ledger_path.read_bytes()).hexdigest() == intake['ledger_sha256']
    assert hashlib.sha256(request_path.read_bytes()).hexdigest() == intake['requests_sha256']
    assert intake['answers'] == intake['expected_answers'] > 0
    expected = {}
    for line in request_path.open():
        request = json.loads(line)
        assert key(request) not in expected
        expected[key(request)] = dict(sha256=hashlib.sha256(line.encode()).hexdigest(), status=request['status'])
    values = {}
    checker = expected_checker()
    for line in result_path.open():
        row = json.loads(line)
        identity = key(row)
        assert identity in expected and identity not in values
        request = expected[identity]
        assert row['input_sha256'] == request['sha256']
        assert row['checker'] == checker, 'Checker identity or full-input execution policy changed'
        if request['status'] == 'ALL_CITATIONS_INVALID':
            assert row['status'] == 'STRUCTURAL_ZERO' and row['entailment'] == 0
            assert row['logits'] is None and row['input_tokens'] == 0
        else:
            assert request['status'] == 'NLI_INPUT_ONLY' and row['status'] == 'NLI_COMPLETED'
            logits = row['logits']
            assert len(logits) == 3 and all(math.isfinite(x) for x in logits)
            label = max(range(3), key=lambda index: logits[index])
            assert row['predicted_label_id'] == label and row['entailment'] == int(label == 0)
            assert row['input_tokens'] > 0
        values[identity] = row['entailment']
    assert values.keys() == expected.keys(), 'Incomplete NLI results are not a zero or a partial metric'
    groups, seen = defaultdict(list), set()
    for line in ledger_path.open():
        row = json.loads(line)
        identity = row['dataset'], row['qid'], row['variant']
        assert identity not in seen
        seen.add(identity)
        support = sum(values[identity + (i,)] for i in range(row['statement_denominator'])) if row['nli_requests'] else 0
        groups[(row['dataset'], row['variant'])].append(support / row['statement_denominator'])
    assert len(seen) == intake['expected_answers']
    assert {k[:3] for k in values} <= seen
    return dict(status='COMPLETE_CITEREC', definition='Mean per-answer fraction of entailed statement units; uncited and empty answers remain zero.',
        groups=[dict(dataset=k[0], variant=k[1], answers=len(v), CiteRec=100*sum(v)/len(v)) for k,v in sorted(groups.items())])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(score(args.ledger, args.requests, args.results), indent=2) + '\n')

"""CPU fixtures test provenance and denominator behavior, never experimental scores."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from intake_answers import prepare, prepare_answer
from score_nli import score
from run_nli import expected_checker


class IntakeTest(unittest.TestCase):
    def setUp(self):
        self.source = dict(dataset='fixture', qid='q', status='CITATION_SOURCE_RECORD_READY', evidence=[
            dict(citation_id='E1', source_id='s1', rendered_text='First source.'),
            dict(citation_id='E2', source_id='s2', rendered_text='Second source.')])
        self.answer = dict(dataset='fixture', qid='q', variant='eab',
            raw_answer='First statement [E2].\nSecond statement [E1][E2].',
            raw_token_ids=[1,2], generation_identity={'kind':'CPU_FIXTURE_NOT_EXPERIMENT'})

    def test_all_answer_citations_and_deduplication(self):
        ledger, requests = prepare_answer(self.answer, self.source)
        self.assertEqual(ledger['statement_denominator'], 2)
        self.assertEqual(len(requests), 2)
        for row in requests:
            self.assertEqual(row['premise'], 'Second source.\n\nFirst source.')
            self.assertEqual(row['all_actual_citation_ids'], ['E2','E1'])

    def test_sentence_segmentation_matches_alce_api_not_lines(self):
        self.answer['raw_answer'] = 'Dr. Smith described the first source [E1]. Another statement cites the second [E2].'
        _, requests = prepare_answer(self.answer, self.source)
        import nltk
        self.assertEqual([r['raw_sentence'] for r in requests], nltk.sent_tokenize(self.answer['raw_answer']))
        self.assertEqual(len(requests), 2)
        self.assertEqual([r['sentence_local_citation_ids'] for r in requests], [['E1'],['E2']])
        for row in requests:
            begin,end = row['raw_sentence_span']
            self.assertEqual(row['raw_sentence'], self.answer['raw_answer'][begin:end])

    def test_uncited_answer_remains_in_denominator(self):
        self.answer['raw_answer'] = 'One statement.\nTwo statements.'
        ledger, requests = prepare_answer(self.answer, self.source)
        self.assertEqual((ledger['statement_denominator'], ledger['structural_zero_statements']), (2,2))
        self.assertEqual(requests, [])

    def test_empty_answer_remains_in_answer_denominator(self):
        self.answer['raw_answer'] = ' \n'
        ledger, requests = prepare_answer(self.answer, self.source)
        self.assertEqual(ledger['statement_denominator'], 1)
        self.assertEqual(ledger['status'], 'EMPTY_ANSWER_ZERO')
        self.assertEqual(requests, [])

    def test_invalid_citations_are_not_manufactured(self):
        self.answer['raw_answer'] = 'Claim [E99].'
        ledger, requests = prepare_answer(self.answer, self.source)
        self.assertEqual(requests[0]['invalid_citation_ids'], ['E99'])
        self.assertEqual(requests[0]['premise'], '')
        self.assertEqual(ledger['structural_zero_statements'], 1)

    def test_mixed_invalid_citation_contributes_no_text(self):
        self.answer['raw_answer'] = 'Claim [E99][E1].'
        _, requests = prepare_answer(self.answer, self.source)
        self.assertEqual(requests[0]['premise'], 'First source.')
        self.assertEqual(requests[0]['invalid_citation_ids'], ['E99'])

    def test_raw_records_preserved_and_missing_identity_rejected(self):
        before = copy.deepcopy((self.answer, self.source))
        prepare_answer(self.answer, self.source)
        self.assertEqual((self.answer, self.source), before)
        del self.answer['raw_token_ids']
        with self.assertRaises(ValueError):
            prepare_answer(self.answer, self.source)

    def test_complete_denominator_and_missing_nli_rejection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)
            source = path / 'sources.jsonl'
            source.write_text(json.dumps(self.source) + '\n')
            answers = path / 'answers.jsonl'
            uncited = dict(self.answer, variant='static_concat', raw_answer='Uncited statement.')
            answers.write_text(json.dumps(self.answer) + '\n' + json.dumps(uncited) + '\n')
            report = prepare(answers, [source], path)
            self.assertEqual(report['answers'], 2)
            requests = path / 'cite_rec_nli_inputs.jsonl'
            results = path / 'results.jsonl'
            fixture_results = []
            for line in requests.open():
                request = json.loads(line)
                fixture_results.append(dict(dataset=request['dataset'], qid=request['qid'], variant=request['variant'],
                    statement_index=request['statement_index'], input_sha256=hashlib.sha256(line.encode()).hexdigest(),
                    status='NLI_COMPLETED', entailment=1, fixture=True, checker=expected_checker(),
                    logits=[10.,0.,0.], predicted_label_id=0, input_tokens=10))
            results.write_text(''.join(json.dumps(r) + '\n' for r in fixture_results))
            scored = score(path / 'answer_ledger.jsonl', requests, results)
            self.assertEqual([g['CiteRec'] for g in scored['groups']], [100.0, 0.0])
            results.write_text(json.dumps(fixture_results[0]) + '\n')
            with self.assertRaises(AssertionError):
                score(path / 'answer_ledger.jsonl', requests, results)

    def test_missing_answer_is_not_silently_excluded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)
            source, answers = path / 'sources.jsonl', path / 'answers.jsonl'
            source.write_text(json.dumps(self.source) + '\n')
            answers.write_text(json.dumps(self.answer) + '\n')
            with self.assertRaises(AssertionError):
                prepare(answers, [source], path)

    def rejected_result(self, answer_text, mutate):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)
            source, answers = path / 'sources.jsonl', path / 'answers.jsonl'
            source.write_text(json.dumps(self.source) + '\n')
            answers.write_text(''.join(json.dumps(dict(self.answer, variant=arm, raw_answer=answer_text)) + '\n'
                                       for arm in ('static_concat','eab')))
            prepare(answers, [source], path)
            requests = path / 'cite_rec_nli_inputs.jsonl'
            results = []
            for line in requests.open():
                request = json.loads(line)
                row = {name:request[name] for name in ('dataset','qid','variant','statement_index')}
                row.update(input_sha256=hashlib.sha256(line.encode()).hexdigest(), checker=expected_checker(),
                    status='NLI_COMPLETED', logits=[10.,0.,0.], predicted_label_id=0, entailment=1, input_tokens=10)
                mutate(row)
                results.append(row)
            result_path = path / 'results.jsonl'
            result_path.write_text(''.join(json.dumps(r) + '\n' for r in results))
            with self.assertRaises(AssertionError):
                score(path / 'answer_ledger.jsonl', requests, result_path)

    def test_all_invalid_request_cannot_receive_positive_score(self):
        self.rejected_result('Claim [E99].', lambda row: None)

    def test_wrong_checker_truncation_or_inconsistent_argmax_rejected(self):
        for mutate in (
            lambda row: row['checker'].update(model='WRONG_CHECKPOINT'),
            lambda row: row['checker'].update(truncation=True),
            lambda row: row.update(logits=[0.,10.,0.], predicted_label_id=1, entailment=1)):
            self.rejected_result('Claim [E1].', mutate)


if __name__ == '__main__':
    unittest.main()

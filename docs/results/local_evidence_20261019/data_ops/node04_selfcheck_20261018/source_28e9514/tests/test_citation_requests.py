"""Citation request construction only; synthetic rows, no checker results."""

import unittest
from omniaware.eval.citation.requests import requests_for_answer


class CitationIntakeTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(qid='fixture', variant='synthetic', raw_answer='Alpha [E2]. Beta [E1] [E9] [E2].',
            raw_token_ids=[1,2], generation_identity={'fixture': True},
            statements=[dict(text='Alpha.', citation_ids=['E2']), dict(text='Beta.', citation_ids=['E1','E9','E2'])],
            evidence=[dict(citation_id='E1', source_id='text:a', rendered_text='First evidence'),
                      dict(citation_id='E2', source_id='text:b', rendered_text='Second evidence')])

    def test_each_statement_is_judged_on_the_evidence_it_cites(self):
        rows = requests_for_answer(self.row)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['premise'], 'Second evidence')                 # not the answer-wide union
        self.assertEqual(rows[1]['premise'], 'First evidence\n\nSecond evidence')
        self.assertEqual(rows[0]['valid_citation_ids'], ['E2'])
        self.assertEqual(rows[1]['invalid_citation_ids'], ['E9'])
        self.assertEqual(rows[1]['sentence_citation_ids'], ['E1', 'E9', 'E2'])
        self.assertTrue(all(r['all_actual_citation_ids'] == ['E2', 'E1', 'E9'] for r in rows))

    def test_uncited_statement_is_a_structural_zero_not_a_neighbours_premise(self):
        self.row['statements'][0]['citation_ids'] = []
        rows = requests_for_answer(self.row)
        self.assertEqual((rows[0]['status'], rows[0]['premise']), ('ALL_CITATIONS_INVALID', ''))
        self.assertEqual(rows[1]['status'], 'NLI_INPUT_ONLY')

    def test_answer_without_any_marker_is_rejected(self):
        self.row['raw_answer'] = 'Alpha without citations.'
        with self.assertRaisesRegex(ValueError, 'No actual citation'):
            requests_for_answer(self.row)


if __name__ == '__main__':
    unittest.main()

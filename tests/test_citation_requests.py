"""Citation request construction only; synthetic rows, no checker results."""

import unittest
from .prepare_citation_support_intake import requests_for_answer


class CitationIntakeTests(unittest.TestCase):
    def test_entire_answer_citations_are_concatenated_for_every_statement(self):
        row = dict(qid='fixture', variant='synthetic', raw_answer='Alpha [E2]. Beta [E1] [E9] [E2].',
            raw_token_ids=[1,2], generation_identity={'fixture': True},
            statements=[dict(text='Alpha.', citation_ids=['E2']), dict(text='Beta.', citation_ids=['E1','E9','E2'])],
            evidence=[dict(citation_id='E1', source_id='text:a', rendered_text='First evidence'),
                      dict(citation_id='E2', source_id='text:b', rendered_text='Second evidence')])
        rows = requests_for_answer(row)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['premise'] == 'Second evidence\n\nFirst evidence' for r in rows))
        self.assertTrue(all(r['invalid_citation_ids'] == ['E9'] for r in rows))
        row['raw_answer'] = 'Alpha without citations.'
        with self.assertRaisesRegex(ValueError, 'No actual citation'):
            requests_for_answer(row)


if __name__ == '__main__':
    unittest.main()

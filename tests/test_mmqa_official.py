"""Synthetic cases covering the official list-answer scoring contract."""

import unittest

import pytest

from omniaware import paths

if not (paths.MMQA_OFFICIAL_EVAL / 'evaluate.py').exists():
    pytest.skip('upstream MMQA evaluator not under the asset root', allow_module_level=True)

from omniaware.eval.mmqa_official import list_em, list_f1, score_predictions


class OfficialMMQAScoringTests(unittest.TestCase):
    def test_complete_list_is_order_independent(self):
        self.assertEqual(list_em(['blue', 'red'], ['red', 'blue']), 1.0)
        self.assertEqual(list_f1(['blue', 'red'], ['red', 'blue']), 1.0)

    def test_answer_items_are_one_reference(self):
        scores, instances = score_predictions(
            {'q': ['red']}, [{'qid': 'q', 'gold_answers': [['red', 'blue']]}])
        self.assertEqual(instances['q'], {'list_em': 0.0, 'list_f1': 0.5})
        self.assertEqual(scores, {'list_em': 0.0, 'list_f1': 50.0})

    def test_number_normalization_and_consistency(self):
        self.assertEqual(list_em('two cats', '2 cats'), 1.0)
        self.assertEqual(list_f1('3 cats', '2 cats'), 0.0)

    def test_missing_prediction_remains_in_denominator(self):
        scores, instances = score_predictions(
            {'q1': ['red']}, [{'qid': 'q1', 'gold_answers': [['red']]},
                             {'qid': 'q2', 'gold_answers': [['blue']]}])
        self.assertEqual(scores, {'list_em': 50.0, 'list_f1': 50.0})
        self.assertEqual(set(instances), {'q1', 'q2'})
        self.assertEqual(instances['q2'], {'list_em': 0.0, 'list_f1': 0.0})


if __name__ == '__main__':
    unittest.main()

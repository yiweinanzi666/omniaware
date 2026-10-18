"""Metric contract tests; no model inference or benchmark claim."""

import copy
import unittest

import pytest

spacy = pytest.importorskip('spacy')
pytest.importorskip('word2number')

# spaCy installs without its English model (README: `python -m spacy download en_core_web_sm`); load_official_metric
# needs it, so the module skips until the model is downloaded instead of erroring with E050. is_package checks the
# installed package without loading the pipeline at collection time.
if not spacy.util.is_package('en_core_web_sm'):
    pytest.skip('spaCy model en_core_web_sm not installed (python -m spacy download en_core_web_sm)',
                allow_module_level=True)

from omniaware.eval.webqa_acc import load_official_metric, score_paired, score_references


class OfficialMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metric, cls.provenance = load_official_metric()
        cls.metric = staticmethod(cls.metric)

    def test_published_empty_and_normalization_semantics(self):
        for prediction, reference, category, expected in (
            ("", "", "text", 0.),
            ("", "cat", "text", 0.),
            ("cats", "A cat runs", "text", .5),
            ("A", "a", "text", 1.),
            ("two", "2", "number", 1.),
            ("2.0", "2", "number", 0.),
            ("yes, it is", "yes", "YesNo", 1.),
            ("red and blue", "blue", "color", 2 / 3),
        ):
            with self.subTest(prediction=prediction, reference=reference, category=category):
                self.assertAlmostEqual(self.metric(prediction, reference, Qcate=category)["acc_approx"], expected)

    def test_reference_rule_is_explicit_and_shared(self):
        value, per_reference = score_references("dog", ["dog"], "text", self.metric)
        self.assertEqual(per_reference, [1.])
        self.assertEqual(value, 1.)
        self.assertIn("multiple references rejected", self.provenance["reference_rule"])
        for references in ([], ["cat", "dog"], "dog"):
            with self.subTest(references=references), self.assertRaisesRegex(ValueError, "exactly one"):
                score_references("dog", references, "text", self.metric)

    def rows(self):
        predictions = {
            "q1": {"bias-off": "cat", "paper-timing": "cat", "first-token": "cat"},
            "q2": {"bias-off": "dog", "paper-timing": "dog", "first-token": "cat"},
        }
        return [dict(qid=qid, variant=variant, prediction=prediction,
                     references=["cat"], category="text", inference_seconds=.5)
                for qid, variants in predictions.items() for variant, prediction in variants.items()]

    def test_question_paired_bootstrap_and_means(self):
        scored, report = score_paired(self.rows(), self.metric)
        self.assertEqual(len(scored), 6)
        self.assertEqual(report["variants"]["first-token"]["mean_acc_approx"], 1.)
        self.assertEqual(report["variants"]["paper-timing"]["mean_acc_approx"], .5)
        for result in report["paired_comparisons"].values():
            self.assertEqual(result["mean_difference_pp"], 50.)
            self.assertEqual(result["ci95"], [0., 1.])
            self.assertEqual(result["question_differences"], {"q1": 0., "q2": 1.})
        self.assertEqual(report["bootstrap"]["resamples"], 10000)
        self.assertEqual(report["bootstrap"]["unit"], "question")
        self.assertFalse(report["heldout_confirmation"])
        self.assertFalse(report["leaderboard_composite"])
        _, reversed_report = score_paired(list(reversed(self.rows())), self.metric)
        self.assertEqual(report, reversed_report)

    def test_duplicate_and_unpaired_questions_rejected(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            score_paired(rows + [copy.deepcopy(rows[0])], self.metric)
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            score_paired(rows[:-1], self.metric)
        rows[-1]["qid"] = "another-question"
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            score_paired(rows, self.metric)

    def test_reference_changes_between_variants_rejected(self):
        rows = self.rows()
        rows[-1]["references"] = ["dog"]
        with self.assertRaisesRegex(ValueError, "Reference/category mismatch"):
            score_paired(rows, self.metric)



if __name__ == "__main__":
    unittest.main()

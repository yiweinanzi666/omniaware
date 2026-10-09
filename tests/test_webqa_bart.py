"""CPU checks of BARTScore math and paired public-data composite reporting."""

import copy
import math
import unittest

import torch
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer

from experiments.eab_first_token.score_bart import (
    DEFAULT_CACHE, add_fluency, configure_official_scorer, normalize_for_bart,
    normalized_bart, summarize_composite,
)


class RecordingScorer:
    def __init__(self):
        self.calls = []

    def score(self, sources, targets, batch_size=4):
        self.calls.append((sources, targets))
        if sources == targets:
            return [-2.] * len(sources)
        return [-3., -2., -1.] * (len(sources) // 3)


class CompositeTests(unittest.TestCase):
    def rows(self):
        return [dict(qid=qid, variant=variant, prediction="Prediction!",
                     references=["Reference, text."], category="text", acc_approx=value)
                for qid in ("q1", "q2")
                for variant, value in (("bias-off", .2), ("paper-timing", .4), ("first-token", .6))]

    def test_webqa_punctuation_and_case_are_preserved_exactly(self):
        self.assertEqual(normalize_for_bart("What's 2.5?\n (Blue)"), "Whats 25 Blue")
        self.assertEqual(normalize_for_bart(""), "")

    def test_probability_ratio_direction_and_clipping(self):
        self.assertAlmostEqual(normalized_bart(-3., -2.), math.exp(-3.) / math.exp(-2.))
        self.assertEqual(normalized_bart(-1., -2.), 1.)
        self.assertEqual(normalized_bart(-1000., -1000.), 1.)
        with self.assertRaises(ValueError):
            normalized_bart(float("nan"), -2.)

    def test_reference_to_prediction_direction_and_original_metric_unchanged(self):
        rows = self.rows()
        before = copy.deepcopy(rows)
        scorer = RecordingScorer()
        actual = add_fluency(rows, scorer)
        self.assertEqual(rows, before)
        self.assertEqual(scorer.calls[0], (["Reference text"] * 2, ["Reference text"] * 2))
        self.assertEqual(scorer.calls[1], (["Reference text"] * 6, ["Prediction"] * 6))
        self.assertAlmostEqual(actual[0]["acc_approx_x_normalized_bart"], .2 / math.e)
        self.assertEqual(actual[1]["acc_approx_x_normalized_bart"], .4)
        self.assertEqual(actual[2]["acc_approx_x_normalized_bart"], .6)
        self.assertEqual(actual[0]["acc_approx"], .2)

    def test_pairing_is_required_before_bart_execution(self):
        scorer = RecordingScorer()
        for rows in (self.rows()[:-1], self.rows() + [self.rows()[0]]):
            with self.assertRaises(ValueError):
                add_fluency(rows, scorer)
        self.assertFalse(scorer.calls)

    def test_composite_is_development_not_hidden_test_score(self):
        scored = add_fluency(self.rows(), RecordingScorer())
        report = summarize_composite(scored)
        self.assertEqual(report["question_count"], 2)
        self.assertFalse(report["heldout_confirmation"])
        self.assertFalse(report["leaderboard_hidden_test_score"])
        self.assertTrue(report["supplemental_metric_added_after_development_inspection"])
        result = report["paired_comparisons"]["first-token_minus_paper-timing"]
        self.assertAlmostEqual(result["relative_improvement_percent"], 50.)
        for limit in result["ci95"]:
            self.assertAlmostEqual(limit, .2)
        self.assertEqual(report["bootstrap"]["unit"], "question")

    def test_two_and_three_policy_variants_keep_real_names(self):
        class FlatScorer:
            def score(self, sources, targets, batch_size=4):
                return [-2.] * len(sources)
        for variants in (("short-answer", "complete-answer"),
                         ("short-answer", "complete-answer", "knowledge-fallback")):
            rows = [dict(qid="q1", variant=variant, prompt_policy=variant,
                         prediction="Answer", references=["Reference"], category="text",
                         question="Question", evidence_ids=["e1"], evidence_sha256="a" * 64,
                         config_sha256="b" * 64, prompt_sha256=str(i) * 64,
                         system_prompt_sha256=str(i) * 64, gate_enabled=False, acc_approx=(i + 1) / 4)
                    for i, variant in enumerate(variants)]
            with self.subTest(variants=variants):
                scored = add_fluency(rows, FlatScorer(), variants=variants, answer_policy=True)
                report = summarize_composite(scored, variants=variants, baseline="short-answer", stage="confirmation")
                self.assertEqual(set(report["variants"]), set(variants))
                self.assertEqual(report["stage"], "confirmation")
                self.assertFalse(report["heldout_independence_verified"])
                for variant in variants[1:]:
                    self.assertIn(f"{variant}_minus_short-answer", report["paired_comparisons"])




class OfficialScoreTests(unittest.TestCase):
    @torch.inference_mode()
    def test_official_function_matches_independent_token_log_probabilities(self):
        torch.manual_seed(43)
        tokenizer = BartTokenizer.from_pretrained(DEFAULT_CACHE / "bart-large-cnn-config", local_files_only=True)
        config = BartConfig(vocab_size=len(tokenizer), d_model=16, encoder_layers=1,
                            decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
                            encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=64,
                            bos_token_id=0, pad_token_id=1, eos_token_id=2, decoder_start_token_id=2)
        config._attn_implementation = "eager"
        model = BartForConditionalGeneration(config).eval()
        scorer = configure_official_scorer(model, tokenizer, device="cpu")
        sources = ["A long source sentence.", "Short.", ""]
        targets = ["Target.", "A longer target sentence.", ""]
        actual = scorer.score(sources, targets, batch_size=3)
        src = tokenizer(sources, padding=True, return_tensors="pt")
        tgt = tokenizer(targets, padding=True, return_tensors="pt")
        logits = model(input_ids=src.input_ids, attention_mask=src.attention_mask,
                       labels=tgt.input_ids).logits
        token_logs = logits.log_softmax(-1).gather(-1, tgt.input_ids.unsqueeze(-1)).squeeze(-1)
        target_mask = tgt.input_ids != config.pad_token_id
        expected = (token_logs * target_mask).sum(-1) / tgt.attention_mask.sum(-1)
        torch.testing.assert_close(torch.tensor(actual), expected, rtol=1e-6, atol=1e-6)
        singles = [scorer.score([source], [target], batch_size=1)[0]
                   for source, target in zip(sources, targets)]
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(singles), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

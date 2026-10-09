"""Actual random tiny Qwen3VL CPU probes; no pretrained model or GPU runs."""

import pytest

pytest.importorskip('spacy')
pytest.importorskip('scipy')

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import spacy
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GenerationConfig, PreTrainedTokenizerFast

from omniaware.baselines.dragin_qwen_observation import (
    capture_draft_sdpa, observe_dragin_draft, reconstruct_draft_attention,
)
from omniaware.baselines.dragin_rind import generation_entropy_nats, received_attention
from tiny_qwen import tiny_model


def text_inputs():
    return dict(input_ids=torch.tensor([[2, 3, 4, 5, 6, 7]]),
                attention_mask=torch.ones(1, 6, dtype=torch.long))


class DraftAttentionTests(unittest.TestCase):
    @torch.inference_mode()
    def test_actual_all_query_capture_preserves_logits_and_matches_eager(self):
        model, inputs = tiny_model("sdpa"), text_inputs()
        native = model(**inputs, use_cache=False).logits
        original_forward = model.model.language_model.layers[-1].self_attn.forward
        original_sdpa = F.scaled_dot_product_attention
        with capture_draft_sdpa(model) as capture:
            observed = model(**inputs, use_cache=False).logits
        torch.testing.assert_close(observed, native, rtol=0, atol=0)
        self.assertEqual(capture["calls"], 1)
        self.assertEqual(capture["query_shape"], [1, 4, 6, 8])
        self.assertEqual(capture["key_shape"], [1, 2, 6, 8])
        self.assertTrue(capture["enable_gqa"])
        reconstructed = reconstruct_draft_attention(capture)
        model.model.language_model.set_attn_implementation("eager")
        eager = model(**inputs, use_cache=False, output_attentions=True).attentions[-1]
        torch.testing.assert_close(reconstructed, eager.double(), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(reconstructed.sum(-1), torch.ones(1, 4, 6).double())
        torch.testing.assert_close(reconstructed.triu(1), torch.zeros_like(reconstructed))
        torch.testing.assert_close(torch.tensor(received_attention(reconstructed, 6)),
                                   eager.double().max(dim=-2).values.mean(dim=1)[0], rtol=1e-6, atol=1e-7)
        self.assertEqual(model.model.language_model.layers[-1].self_attn.forward, original_forward)
        self.assertIs(F.scaled_dot_product_attention, original_sdpa)
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with capture_draft_sdpa(model):
                raise RuntimeError("probe failure")
        self.assertEqual(model.model.language_model.layers[-1].self_attn.forward, original_forward)
        self.assertIs(F.scaled_dot_product_attention, original_sdpa)

    def test_full_causal_bool_additive_masks_and_gqa_match_native_sdpa(self):
        generator = torch.Generator().manual_seed(24)
        query = torch.randn(1, 4, 5, 8, generator=generator, dtype=torch.float64)
        for gqa in (False, True):
            heads = 2 if gqa else 4
            key = torch.randn(1, heads, 5, 8, generator=generator, dtype=torch.float64)
            value = torch.randn(1, heads, 5, 8, generator=generator, dtype=torch.float64)
            allowed = torch.ones(5, 5, dtype=torch.bool).tril()
            allowed[2] = False  # Verify SDPA's fully masked row behavior too.
            additive = torch.zeros(1, 1, 5, 5).double().masked_fill(~allowed, -torch.inf)
            additive[..., 4, 1] = -.7
            for mask, causal in ((None, True), (allowed, False), (additive, False)):
                with self.subTest(gqa=gqa, causal=causal, mask_dtype=None if mask is None else mask.dtype):
                    capture = dict(query=query, key=key, mask=mask, scale=.23,
                                   enable_gqa=gqa, is_causal=causal, dropout_p=0., calls=1)
                    weights = reconstruct_draft_attention(capture)
                    expanded_value = value.repeat_interleave(2, dim=1) if gqa else value
                    expected = F.scaled_dot_product_attention(query, key, value,
                        attn_mask=mask, is_causal=causal, scale=.23, enable_gqa=gqa)
                    torch.testing.assert_close(weights @ expanded_value, expected, rtol=1e-10, atol=1e-10)


class DraftPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocabulary = ["<pad>", "<unk>", "cat", "dog", "bird", "tree"]
        vocabulary += [f"word{i}" for i in range(6, 63)] + ["<eos>"]
        backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
        cls.nlp = spacy.load("en_core_web_sm")

    @torch.inference_mode()
    def test_pipeline_preserves_native_greedy_scores_inputs_and_draft_only_forward(self):
        model, inputs = tiny_model("sdpa"), text_inputs()
        config = GenerationConfig(max_new_tokens=4, do_sample=False, eos_token_id=63,
            pad_token_id=0, suppress_tokens=[0, 1] + list(range(6, 64)),
            output_scores=True, return_dict_in_generate=True)
        original_config = deepcopy(config.to_dict())
        native = model.generate(**inputs, generation_config=config)
        calls = []
        def remember(module, args, kwargs):
            calls.append({key: value.clone() if isinstance(value, torch.Tensor) else value
                          for key, value in kwargs.items() if key != "past_key_values"})
        handle = model.register_forward_pre_hook(remember, with_kwargs=True)
        try:
            record = observe_dragin_draft(model, inputs, self.tokenizer, self.nlp, generation_config=config)
        finally:
            handle.remove()
        draft = native.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        self.assertEqual(record["draft_token_ids"], draft)
        self.assertEqual(record["input_token_ids"], inputs["input_ids"][0].tolist())
        self.assertEqual(record["generation_inputs"]["attention_mask"], inputs["attention_mask"].tolist())
        torch.testing.assert_close(torch.tensor(record["token_entropy_nats"]).double(),
                                   torch.tensor(generation_entropy_nats(native.scores)), rtol=1e-6, atol=1e-7)
        self.assertEqual(len(calls), 5)  # Four greedy steps plus one fresh probe.
        self.assertEqual(calls[-1]["input_ids"].tolist(), [draft])
        self.assertEqual(calls[-1]["attention_mask"].tolist(), [[1] * len(draft)])
        self.assertFalse(calls[-1]["use_cache"])
        self.assertEqual(record["sdpa"]["query_shape"][-2], 4)
        self.assertEqual(record["sdpa"]["key_shape"][-2], 4)
        self.assertEqual(record["processed_score_count"], 4)
        self.assertEqual(record["processed_score_shape"], [1, 64])
        self.assertEqual(record["stop_reason"], "length_limit")
        self.assertTrue(record["length_limit_hit"])
        self.assertFalse(record["eos_emitted"])
        self.assertTrue(record["words"])
        self.assertEqual(config.to_dict(), original_config)
        self.assertGreater(record["draft_generation_seconds"], 0)
        self.assertGreater(record["draft_probe_seconds"], 0)
        self.assertGreater(record["rind_scoring_seconds"], 0)
        self.assertGreaterEqual(record["total_seconds"], sum(record[key] for key in
            ("draft_generation_seconds", "draft_probe_seconds", "rind_scoring_seconds")))
        self.assertEqual((record["generation_calls"], record["extra_forward_calls"]), (1, 1))
        json.dumps(record, allow_nan=False)

    @torch.inference_mode()
    def test_immediate_eos_keeps_raw_token_and_special_only_zero_score(self):
        model = tiny_model("sdpa")
        config = GenerationConfig(max_new_tokens=1, eos_token_id=63, pad_token_id=0,
                                  suppress_tokens=list(range(63)))
        record = observe_dragin_draft(model, text_inputs(), self.tokenizer, self.nlp, generation_config=config)
        self.assertEqual(record["draft_token_ids"], [63])
        self.assertEqual(record["raw_draft_text"], "<eos>")
        self.assertEqual(record["draft_text"], "")
        self.assertEqual(record["special_token_positions"], [0])
        self.assertEqual(record["words"], [])
        self.assertEqual(record["masked_max_score"], 0.)
        self.assertFalse(record["retrieve"])
        self.assertTrue(record["eos_emitted"] and record["ended_with_eos"])
        self.assertTrue(record["reached_max_new_tokens"])
        self.assertFalse(record["length_limit_hit"])
        self.assertEqual(record["stop_reason"], "eos")
        self.assertEqual(record["sdpa"]["query_shape"][-2], 1)
        json.dumps(record, allow_nan=False)

    @torch.inference_mode()
    def test_failed_scoring_retains_completed_draft_and_probe_costs(self):
        model, progress = tiny_model("sdpa"), {}
        config = GenerationConfig(max_new_tokens=1, eos_token_id=63, pad_token_id=0,
                                  suppress_tokens=list(range(63)))
        with patch("omniaware.baselines.dragin_qwen_observation.score_draft",
                   side_effect=RuntimeError("scoring failed")):
            with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                observe_dragin_draft(model, text_inputs(), self.tokenizer, self.nlp,
                                     generation_config=config, call_record=progress)
        self.assertEqual(progress["active_stage"], "rind_scoring")
        self.assertEqual(progress["draft_token_ids"], [63])
        self.assertGreater(progress["draft_generation_seconds"], 0)
        self.assertGreater(progress["draft_probe_seconds"], 0)
        self.assertEqual((progress["generation_calls"], progress["extra_forward_calls"]), (1, 1))


if __name__ == "__main__":
    unittest.main()

"""Real random 18-layer tiny Qwen CPU tests; no pretrained weights or GPU."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3VLForConditionalGeneration

from omniaware.baselines.fitm_attention import redistribute_document_attention
from omniaware.baselines.fitm_qwen_adapter import (
    DUMMY_TEXT, METHOD_NAME, corrected_sdpa_output, document_indices,
    make_null_reference, prepare_fitm_calibration, qwen_fitm,
)
from tiny_qwen import image_prompt, tiny_model


def model18():
    config = deepcopy(tiny_model("sdpa").config)
    config.text_config.num_hidden_layers = 18
    torch.manual_seed(17)
    model = Qwen3VLForConditionalGeneration(config).eval()
    model.model.language_model.set_attn_implementation("sdpa")
    return model


def tokenizer64():
    vocabulary = ["<pad>", "<unk>", "This", "is", "a", "reference", "document."]
    vocabulary += [f"word{i}" for i in range(7, 63)] + ["<eos>"]
    backend = Tokenizer(models.WordLevel({token: index for index, token in enumerate(vocabulary)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")


def text_case():
    return dict(input_ids=torch.tensor([[2, 3, 4, 5, 6, 7, 8]]), attention_mask=torch.ones(1, 7, dtype=torch.long)), [
        dict(source_id="text:a", text_token_indices=[1, 2], vision_token_indices=[]),
        dict(source_id="text:b", text_token_indices=[3, 4, 5], vision_token_indices=[])]


class FitmQwenTests(unittest.TestCase):
    def test_correction_matches_explicit_attention_with_masks_and_gqa(self):
        generator = torch.Generator().manual_seed(31)
        query = torch.randn(1, 4, 6, 8, generator=generator)
        documents = [[0, 1], [2, 3, 4]]
        alpha = torch.tensor([.8, .2]).expand(1, 4, 1, 2)
        for gqa in (False, True):
            heads = 2 if gqa else 4
            key = torch.randn(1, heads, 6, 8, generator=generator)
            value = torch.randn(1, heads, 6, 8, generator=generator)
            boolean = torch.ones(1, 1, 6, 6, dtype=torch.bool).tril()
            boolean[..., 5, 4] = False
            additive = torch.zeros(1, 1, 6, 6).masked_fill(~boolean, -torch.inf)
            additive[..., 5, 1] = -.7
            for mask, causal in ((None, True), (boolean, False), (additive, False)):
                with self.subTest(gqa=gqa, causal=causal, mask_dtype=None if mask is None else mask.dtype):
                    native = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                                           is_causal=causal, enable_gqa=gqa)
                    actual, stats = corrected_sdpa_output(native, query, key, value, mask, alpha, documents,
                                                         is_causal=causal, enable_gqa=gqa)
                    expanded_key = key.repeat_interleave(2, dim=1) if gqa else key
                    expanded_value = value.repeat_interleave(2, dim=1) if gqa else value
                    scores = query[..., -1:, :] @ expanded_key.transpose(-1, -2) / 8 ** .5
                    if mask is not None:
                        scores = scores.masked_fill(~mask[..., -1:, :], -torch.inf) if mask.dtype == torch.bool else scores + mask[..., -1:, :]
                    adjusted, _ = redistribute_document_attention(scores.log_softmax(-1), alpha, documents)
                    torch.testing.assert_close(actual[..., -1:, :], adjusted @ expanded_value, rtol=2e-5, atol=2e-7)
                    torch.testing.assert_close(actual[..., :-1, :], native[..., :-1, :], rtol=0, atol=0)
                    torch.testing.assert_close(stats["document_mass_before"].sum(-1), stats["document_mass_after"].sum(-1))
                    if mask is not None:
                        self.assertTrue(torch.equal(adjusted[..., 4], torch.zeros_like(adjusted[..., 4])))

    @torch.inference_mode()
    def test_actual_last_16_layers_prefill_decode_and_native_restoration(self):
        model, tokenizer = model18(), tokenizer64()
        inputs, sources = text_case()
        documents = document_indices(sources)
        native = model(**inputs, use_cache=False).logits
        forwards = [layer.self_attn.forward for layer in model.model.language_model.layers]
        sdpa = F.scaled_dot_product_attention
        with qwen_fitm(model, documents) as observed:
            observed_logits = model(**inputs, use_cache=False).logits
        torch.testing.assert_close(observed_logits, native, rtol=0, atol=0)
        self.assertEqual(observed["layer_indices"], list(range(2, 18)))
        self.assertTrue(all(value == 1 for value in observed["calls"].values()))
        alpha, record = prepare_fitm_calibration(model, inputs, sources, tokenizer)
        self.assertEqual(record["method"], METHOD_NAME)
        self.assertEqual(record["extra_forward_calls_started"], 3)
        self.assertEqual([attempt["stage"] for attempt in record["forward_attempts"]],
                         ["reference_0", "reference_1", "unbiased_actual_prompt"])
        self.assertTrue(all(attempt["status"] == "completed" for attempt in record["forward_attempts"]))
        self.assertFalse(record["dummy_constant_relevance_proven"])
        self.assertEqual(record["dummy_text"], DUMMY_TEXT)
        self.assertGreater(record["reference_forward_seconds"], 0)
        self.assertGreater(record["actual_observation_seconds"], 0)
        self.assertAlmostEqual(record["extra_forward_seconds"],
                               record["reference_forward_seconds"] + record["actual_observation_seconds"])
        self.assertGreaterEqual(record["total_calibration_seconds"],
                                record["extra_forward_seconds"] + record["reference_preparation_seconds"])
        for layer in range(2, 18):
            torch.testing.assert_close(alpha[layer].sum(-1), torch.ones(1, 4, 1))
        with qwen_fitm(model, documents, alpha=alpha):
            actual = model(**inputs, use_cache=False).logits
        torch.testing.assert_close(actual[:, :-1], native[:, :-1], rtol=0, atol=0)
        self.assertGreater((actual[:, -1] - native[:, -1]).abs().max().item(), 1e-6)
        with qwen_fitm(model, documents, alpha=alpha) as generated:
            model.generate(**inputs, max_new_tokens=3, do_sample=False, eos_token_id=None,
                           pad_token_id=0, suppress_tokens=[59, 60, 61, 62])
        self.assertTrue(all(value == 3 for value in generated["calls"].values()))
        for layer in range(2, 18):
            self.assertEqual(generated["first"][layer]["query_tokens"], 7)
            self.assertEqual(generated["last"][layer]["query_tokens"], 1)
            self.assertEqual(generated["last"][layer]["key_tokens"], 9)
            stats = generated["last"][layer]
            torch.testing.assert_close(stats["document_mass_before"].sum(-1), stats["document_mass_after"].sum(-1))
        with patch("omniaware.baselines.fitm_qwen_adapter.corrected_sdpa_output", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with qwen_fitm(model, documents, alpha=alpha):
                    model(**inputs, use_cache=False)
        self.assertIs(F.scaled_dot_product_attention, sdpa)
        self.assertEqual([layer.self_attn.forward for layer in model.model.language_model.layers], forwards)
        json.dumps(record, allow_nan=False)

    @torch.inference_mode()
    def test_failed_forward_preserves_reference_and_partial_attempt_cost(self):
        model, progress = model18(), {}
        inputs, sources = text_case()
        original_sdpa = F.scaled_dot_product_attention
        with patch.object(model, "forward", side_effect=RuntimeError("native forward failed")):
            with self.assertRaisesRegex(RuntimeError, "native forward failed"):
                prepare_fitm_calibration(model, inputs, sources, tokenizer64(), call_record=progress)
        self.assertEqual(progress["status"], "failed")
        self.assertEqual(progress["extra_forward_calls_started"], 1)
        self.assertEqual(progress["references"][0]["dummy_text"], DUMMY_TEXT)
        self.assertTrue(progress["references"][0]["replacement_token_ids"])
        self.assertEqual(progress["forward_attempts"][0]["status"], "failed")
        self.assertGreater(progress["forward_attempts"][0]["elapsed_seconds_until_failure"], 0)
        self.assertIs(F.scaled_dot_product_attention, original_sdpa)
        json.dumps(progress, allow_nan=False)

    def test_null_reference_replaces_content_and_keeps_vision_markers(self):
        tokenizer = tokenizer64()
        inputs = dict(input_ids=torch.tensor([[11, 12, 13, 14]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
        sources = [dict(source_id="image:a", text_token_indices=[2], vision_token_indices=[],
                        structural_token_indices=[1, 3]),
                   dict(source_id="text:b", text_token_indices=[0], vision_token_indices=[])]
        reference, metadata = make_null_reference(inputs, sources, 0, tokenizer)
        self.assertEqual(reference["input_ids"][0, [0, 1, 3]].tolist(), [11, 12, 14])   # markers and outside text
        self.assertNotEqual(reference["input_ids"][0, 2].item(), 13)                    # content replaced
        self.assertEqual(metadata["text_token_indices"], [2])
        self.assertEqual(document_indices(sources)[0], [1, 2, 3])                       # markers stay in the document
        self.assertEqual(inputs["input_ids"].tolist(), [[11, 12, 13, 14]])

    @torch.inference_mode()
    def test_visual_null_preserves_positions_and_all_reference_costs(self):
        model, tokenizer = model18(), tokenizer64()
        inputs = image_prompt()
        originals = {key: value.clone() for key, value in inputs.items()}
        sources = [dict(source_id="image:a", text_token_indices=[4], vision_token_indices=[2]),
                   dict(source_id="text:b", text_token_indices=[0], vision_token_indices=[])]
        reference, metadata = make_null_reference(inputs, sources, 0, tokenizer)
        self.assertEqual(metadata["visual_null"], "processor_pixel_values_zero")
        self.assertEqual(metadata["pixel_patch_range"], [0, 4])
        self.assertEqual(metadata["replacement_token_ids"], [2])
        self.assertEqual(reference["input_ids"].tolist(), [[2, 60, 61, 62, 2, 7]])
        self.assertEqual(reference["pixel_values"].count_nonzero().item(), 0)
        for key in ("attention_mask", "image_grid_thw", "mm_token_type_ids"):
            torch.testing.assert_close(reference[key], inputs[key], rtol=0, atol=0)
        for key in originals:
            torch.testing.assert_close(inputs[key], originals[key], rtol=0, atol=0)
        positions = model.model.get_rope_index(inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"], mm_token_type_ids=inputs["mm_token_type_ids"])[0]
        reference_positions = model.model.get_rope_index(reference["input_ids"], image_grid_thw=reference["image_grid_thw"],
            attention_mask=reference["attention_mask"], mm_token_type_ids=reference["mm_token_type_ids"])[0]
        torch.testing.assert_close(reference_positions, positions, rtol=0, atol=0)
        alpha, record = prepare_fitm_calibration(model, inputs, sources, tokenizer)
        self.assertEqual(record["extra_forward_calls_started"], 3)
        self.assertEqual(record["references"][0]["visual_null"], "processor_pixel_values_zero")
        self.assertIsNone(record["references"][1]["visual_null"])
        with qwen_fitm(model, document_indices(sources), alpha=alpha):
            output = model(**inputs, use_cache=False)
        self.assertTrue(torch.isfinite(output.logits).all())
        json.dumps(record, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

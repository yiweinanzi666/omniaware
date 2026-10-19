"""Local processor and tiny CPU checks; no pretrained model weights are loaded."""

import pytest

pytest.importorskip('torchvision')

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from omniaware.baselines.m2rag_qwen import LocalQwenVLM
from tiny_qwen import tiny_model
from omniaware import paths

# The processor files (tokenizer and image processor) come from the local Qwen3-VL-8B directory. A fresh server runs
# the self-check before download_models, so the module is skipped until those files exist.
PROCESSOR_FILES = ("preprocessor_config.json", "tokenizer_config.json")
if not all((paths.QWEN_8B / name).is_file() for name in PROCESSOR_FILES):
    pytest.skip(f'Qwen3-VL-8B processor files not under {paths.QWEN_8B} (scripts/setup/download_models.py)',
                allow_module_level=True)

class LocalQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(
            paths.QWEN_8B, local_files_only=True,
        )

    def backend(self, output_tokens, **kwargs):
        model = SimpleNamespace(device=torch.device("cpu"),
                                generation_config=SimpleNamespace(eos_token_id=self.processor.tokenizer.eos_token_id))
        model.generate = Mock(side_effect=lambda **inputs: torch.cat(
            [inputs["input_ids"], torch.tensor([output_tokens])], dim=1,
        ))
        return LocalQwenVLM(model, self.processor, max_new_tokens=3, **kwargs)

    def test_one_user_message_caption_and_fixed_generate_budget(self):
        tokenizer = self.processor.tokenizer
        answer = tokenizer.encode("verified", add_special_tokens=False)
        backend = self.backend(answer + [tokenizer.eos_token_id])
        messages = [{"type": "text", "text": "Question: "},
                    {"type": "image_caption", "image_caption": {"caption": "caption text"}}]
        original = copy.deepcopy(messages)
        result = backend.chat(messages)
        call = backend.calls[0]
        expected = [{"role": "user", "content": [
            {"type": "text", "text": "Question: "}, {"type": "text", "text": "caption text"}]}]
        self.assertEqual(call["messages"], expected)
        self.assertEqual(messages, original)
        messages[0]["text"] = "later official mutation"
        self.assertEqual(call["messages"], expected)
        actual_inputs = backend.model.generate.call_args.kwargs
        expected_inputs = self.processor.apply_chat_template(
            expected, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            processor_kwargs={"add_special_tokens": False},
        )
        torch.testing.assert_close(actual_inputs["input_ids"], expected_inputs["input_ids"])
        self.assertEqual(call["input_token_ids"], expected_inputs["input_ids"][0].tolist())
        self.assertEqual(call["status"], "completed")
        decode = actual_inputs["generation_config"]
        self.assertEqual(decode.max_new_tokens, 3)
        self.assertFalse(decode.do_sample)
        self.assertEqual(decode.num_beams, 1)
        self.assertEqual(result["response"], "verified")
        self.assertEqual(result["usage"]["prompt_tokens"], expected_inputs["input_ids"].shape[1])
        self.assertEqual(result["usage"]["completion_tokens"], len(answer) + 1)
        self.assertEqual(result["usage"]["total_tokens"], sum(result["usage"][key] for key in ("prompt_tokens", "completion_tokens")))
        self.assertTrue(call["ended_with_eos"])
        self.assertFalse(call["hit_token_cap"])
        self.assertGreaterEqual(call["elapsed_seconds"], call["generation_seconds"])

    def test_token_cap_and_eos_at_cap_are_distinct(self):
        for tokens, ended, capped in (([10, 11, 12], False, True),
                                     ([10, 11, self.processor.tokenizer.eos_token_id], True, False)):
            backend = self.backend(tokens)
            backend.chat([{"type": "text", "text": "A"}])
            self.assertEqual(backend.calls[0]["ended_with_eos"], ended)
            self.assertEqual(backend.calls[0]["hit_token_cap"], capped)

    def test_real_local_image_mapping_and_pixel_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (1600, 800), "navy").save(image_path)
            original_url = "https://official.invalid/images/sample.png"
            backend = self.backend([10, 11, 12], image_paths={original_url: image_path},
                                   image_size={"shortest_edge": 65536, "longest_edge": 262144})
            messages = [{"type": "image_url", "image_url": {"url": original_url}},
                        {"type": "text", "text": "Describe this image."}]
            backend.chat(messages)
            inputs = backend.model.generate.call_args.kwargs
            self.assertIn("pixel_values", inputs)
            self.assertEqual(inputs["pixel_values"].shape[1], 3 * 2 * 16 * 16)
            call = backend.calls[0]
            self.assertEqual(len(call["image_grid_thw"]), 1)
            self.assertGreater(call["vision_tokens"], 0)
            self.assertLessEqual(call["vision_tokens"], 256)
            self.assertEqual(call["messages"][0]["content"][0]["path"], str(image_path.resolve()))
            self.assertEqual(messages[0]["image_url"]["url"], original_url)
            # Official file:// inputs use the same processor path.
            backend.chat([{"type": "image_url", "image_url": {"url": image_path.as_uri()}}])
            self.assertEqual(backend.calls[1]["image_grid_thw"], call["image_grid_thw"])

    def test_image_errors_are_visible_and_do_not_download(self):
        backend = self.backend([10])
        with self.assertRaisesRegex(ValueError, "explicit local"):
            backend.chat([{"type": "image_url", "image_url": {"url": "https://invalid.example/no-image.png"}}])
        with self.assertRaises(FileNotFoundError):
            backend.chat([{"type": "image_url", "image_url": {"url": "/missing/m2rag.png"}}])
        backend.model.generate.assert_not_called()

    def test_official_token_count_and_trimming_semantics(self):
        backend = self.backend([10])
        text = "alpha beta gamma delta"
        tokens = self.processor.tokenizer(text).input_ids
        self.assertEqual(backend.num_tokens_from_text(text), len(tokens))
        for side, expected in (("left", tokens[-2:]), ("right", tokens[:2])):
            self.assertEqual(backend.trim_text(text, 2, side), self.processor.tokenizer.decode(expected, skip_special_tokens=True))
        self.assertEqual(backend.trim_text(text, 0, "left"), "")
        self.assertEqual(backend.trim_text(text, 0, "right"), "")

    def test_generation_failure_preserves_started_input_and_propagates(self):
        backend = self.backend([10])
        backend.model.generate.side_effect = RuntimeError("generation failure")
        with self.assertRaisesRegex(RuntimeError, "generation failure"):
            backend.chat([{"type": "text", "text": "Retain this input."}])
        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertEqual(call["status"], "failed")
        self.assertEqual(call["error"], "generation failure")
        self.assertGreaterEqual(call["generation_attempt_seconds"], 0)
        self.assertEqual(call["input_token_ids"], backend.model.generate.call_args.kwargs["input_ids"][0].tolist())
        self.assertEqual(call["generation"]["max_new_tokens"], 3)
        self.assertNotIn("response", call)

    def test_real_tiny_cpu_qwen_text_and_visual_generation(self):
        processor = Qwen3VLProcessor(
            tokenizer=self.processor.tokenizer, chat_template=self.processor.chat_template,
            video_processor=self.processor.video_processor,
            image_processor=Qwen2VLImageProcessor(
                patch_size=2, temporal_patch_size=2, merge_size=2,
                size={"shortest_edge": 16, "longest_edge": 64},
            ),
        )
        model = tiny_model("sdpa")
        model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        model.config.image_token_id = processor.image_token_id
        model.config.video_token_id = processor.video_token_id
        model.config.vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        model.config.vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        backend = LocalQwenVLM(model, processor, max_new_tokens=2)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "tiny.png"
            Image.new("RGB", (8, 8), "red").save(image_path)
            for content in ([{"type": "text", "text": "A"}],
                            [{"type": "image_url", "image_url": {"url": str(image_path)}},
                             {"type": "text", "text": "A"}]):
                with self.subTest(visual=len(content) == 2), patch.object(model, "generate", wraps=model.generate) as generate:
                    result = backend.chat(content)
                    self.assertEqual(generate.call_args.kwargs["generation_config"].max_new_tokens, 2)
                    self.assertEqual(result["usage"]["completion_tokens"], 2)
                    self.assertEqual("pixel_values" in generate.call_args.kwargs, len(content) == 2)
        self.assertEqual(backend.calls[0]["vision_tokens"], 0)
        self.assertEqual(backend.calls[1]["vision_tokens"], 4)


if __name__ == "__main__":
    unittest.main()

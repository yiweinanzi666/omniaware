"""CPU/mock checks against the actual pinned M2RAG sources and retained T15 input."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest

from omniaware.baselines.m2rag_official import (
    OFFICIAL_COMMIT, OFFICIAL_ROOT, build_official_summarizer, query_from_t15,
)
from omniaware.baselines.m2rag_webqa import official_query


from omniaware.paths import ASSET_ROOT as ROOT
import pytest

if not (OFFICIAL_ROOT / 'src').exists():
    pytest.skip('pinned official M2RAG source not under the asset root', allow_module_level=True)


class RecordingBackend:
    name = "vlm-recording-mock"

    def __init__(self, responses=("mock document",)):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, messages):
        self.calls.append(deepcopy(messages))
        return {"response": next(self.responses),
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}

    def num_tokens_from_text(self, text):
        return len(text.split())

    def trim_text(self, text, max_tokens, side="right"):
        words = text.split()
        return " ".join(words[:max_tokens] if side == "right" else words[-max_tokens:])


def image_query():
    return {
        "query_id": "image-fixture", "content": "Describe the image.",
        "webpages": [{
            "webpage_id": "source-image", "title": "Image source", "url": "",
            "cleaned_webpage_splits": [{"split_id": 0, "score": 0,
                "text": "Before image. <IMAGE_PLACEHOLDER>[7] After image."}],
            "images": [{"image_id": 7, "split_id": 0, "valid": True, "final_score": 0,
                "image_url": "https://example.test/original.png",
                "cached_image_url": "file:///mock/local.png", "detailed_image_caption": "A square."}],
        }],
        "aux_images": [],
    }


class OfficialM2RAGTests(unittest.TestCase):
    def test_vendor_commit_clean_and_loaded_source_is_official(self):
        self.assertEqual(subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OFFICIAL_ROOT, text=True).strip(), OFFICIAL_COMMIT)
        before = sys.modules.get("src")
        summarizer = build_official_summarizer(RecordingBackend())
        self.assertIs(sys.modules.get("src"), before)
        self.assertEqual(Path(inspect.getfile(type(summarizer).summarize_core_single_stage)).resolve(),
                         (OFFICIAL_ROOT / "src/summarizer/multi_modal/end_to_end.py").resolve())
        self.assertEqual(subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=OFFICIAL_ROOT, text=True), "")

    def test_question_image_is_in_the_summarizer_input(self):
        # OK-VQA's question image is an official auxiliary image: it must reach the backend's messages.
        query = official_query("okvqa-question", "What is the man holding?", [],
                               query_images=["file:///mock/question.jpg"])
        backend = RecordingBackend()
        build_official_summarizer(backend).summarize(query_list=[query], dry_run=False)
        urls = [m["image_url"]["url"] for m in backend.calls[0] if m["type"] == "image_url"]
        self.assertEqual(urls, ["file:///mock/question.jpg"])

    @unittest.skipUnless((ROOT / 'runs/reranker_order_development/predictions.jsonl').exists(),
                         'legacy T15 run outputs not under the asset root')
    def test_all_128_saved_inputs_preserve_order_text_and_native_prompt(self):
        rows = [json.loads(line) for line in
                (ROOT / "runs/reranker_order_development/predictions.jsonl").read_text().splitlines()]
        rows = [row for row in rows if row["variant"] == "cross-selected-bge-order"]
        self.assertEqual(len(rows), 128)
        backend = RecordingBackend()
        summarizer = build_official_summarizer(backend)
        native_class = type(summarizer).__mro__[1]
        prompt_template = sys.modules[native_class.__module__].VLM_MULTI_MODAL_SINGLE_STAGE_SUMMARY_PROMPT_TEMPLATE
        for row in rows:
            original = deepcopy(row)
            query = query_from_t15(row)
            result = summarizer.summarize(query_list=[query], dry_run=True)
            self.assertEqual(row, original)
            self.assertEqual(len(result["webpages"]), 5)
            self.assertEqual([page["webpage_id"] for page in result["webpages"]], row["evidence_ids"])
            self.assertEqual([page["content"] for page in result["webpages"]],
                             [block["text"] for block in row["evidence_blocks"]])
            messages = result["model_input"]
            self.assertEqual(messages[0], {"type": "text", "text": prompt_template.format(text_prompt=row["question"])})
            for i, block in enumerate(row["evidence_blocks"]):
                self.assertEqual(messages[i + 1]["text"], f"## Webpage {i}\n```markdown\n{block['text']}\n```\n")
                self.assertLessEqual(block["retained_tokens"], 256)
                self.assertEqual(block["block"], f"[Evidence {i + 1}]\n{block['text']}\n[/Evidence {i + 1}]")
            self.assertEqual(messages[-1], {"type": "text", "text": "# Generated Document"})
            self.assertEqual(result["num_input_images"]["total_images"], 0)
        self.assertEqual(backend.calls, [])

    def test_single_stage_calls_injected_backend_without_decoding_kwargs(self):
        backend = RecordingBackend(("Actual mock document",))
        summarizer = build_official_summarizer(backend)
        self.assertIs(summarizer.model, backend)
        query = query_from_t15({"qid": "q", "question": "Question?", "evidence_blocks": [
            {"fact_id": "f", "text": "Retained text."}]})
        result = summarizer.summarize(query_list=[query], dry_run=False)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result["model_response"], "Actual mock document")
        self.assertEqual(result["usage"], {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})
        with self.assertRaises(TypeError):
            summarizer.summarize(query_list=[query], max_new_tokens=1)
        self.assertEqual(len(backend.calls), 1)

    def test_image_input_and_native_output_processing_keep_url_contract(self):
        backend = RecordingBackend(("A square. ![Caption](images/0.png)",))
        summarizer = build_official_summarizer(backend)
        result = summarizer.summarize(query_list=[image_query()], dry_run=False)
        sent_images = [part for part in backend.calls[0] if part["type"] == "image_url"]
        self.assertEqual(sent_images, [{"type": "image_url", "image_url": {"url": "file:///mock/local.png"}}])
        returned_images = [part for part in result["model_input"] if part["type"] == "image_url"]
        self.assertEqual(returned_images[0]["image_url"]["url"], "https://example.test/original.png")
        self.assertEqual(result["num_input_images"]["total_images"], 1)
        self.assertEqual(result["num_output_images"], 1)
        self.assertEqual(result["output_images"][0]["detailed_image_caption"], "A square.")
        self.assertIn('<img', result["processed_response"])
        self.assertIn('<IMAGE_PLACEHOLDER>', result["placeholder_response"])

    def test_multistage_executes_all_three_official_stages_with_image(self):
        backend = RecordingBackend(("First line.\nSecond line.", "Square caption\n0", "Refined text."))
        summarizer = build_official_summarizer(backend, multi_stage=True)
        result = summarizer.summarize(query_list=[image_query()], dry_run=False)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(result["stage_1_model_response"], "First line.\nSecond line.")
        self.assertEqual(result["stage_2_model_responses"][0]["result"]["response"], "Square caption\n0")
        self.assertEqual(result["stage_3_model_responses"][0]["result"]["response"], "Refined text.")
        self.assertEqual(result["usage"], {"prompt_tokens": 30, "completion_tokens": 9, "total_tokens": 39})
        self.assertEqual(result["num_output_images"], 1)
        self.assertIsNone(result["model_response"])
        self.assertIn("Refined text.", result["processed_response"])

    def test_multistage_without_images_degenerates_to_one_call_even_for_dry_run(self):
        backend = RecordingBackend()
        summarizer = build_official_summarizer(backend, multi_stage=True)
        query = query_from_t15({"qid": "q", "question": "Question?", "evidence_blocks": [
            {"fact_id": "f", "text": "Retained text."}]})
        result = summarizer.summarize(query_list=[query], dry_run=True)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result["stage_2_model_inputs"], [])
        self.assertEqual(result["stage_3_model_inputs"], [])
        self.assertEqual(result["num_input_images"]["total_images"], 0)

    def test_negative_one_upstream_regression_and_local_no_image_path(self):
        backend = RecordingBackend(("First line.\nSecond line.", "Not relevant\n-1"))
        summarizer = build_official_summarizer(backend, multi_stage=True)
        upstream = type(summarizer).__mro__[1]
        self.assertEqual(upstream.parse_image_selection_result(summarizer, "Not relevant\n-1", 1),
                         {"summary_line_id": 1, "image_caption": "Not relevant"})
        for text, expected in (("Not relevant\n-1", -1), ("Caption\n+1", 1),
                               ("Caption\n0", 0), ("Caption\n2", -1), ("No index", -1)):
            self.assertEqual(summarizer.parse_image_selection_result(text, 1)["summary_line_id"], expected)
        result = summarizer.summarize(query_list=[image_query()], dry_run=False)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(result["num_output_images"], 0)
        self.assertEqual(result["stage_3_model_inputs"], [])
        self.assertEqual(result["stage_3_model_responses"], [])
        self.assertEqual(result["processed_response"], "First line.\nSecond line.")


if __name__ == "__main__":
    unittest.main()

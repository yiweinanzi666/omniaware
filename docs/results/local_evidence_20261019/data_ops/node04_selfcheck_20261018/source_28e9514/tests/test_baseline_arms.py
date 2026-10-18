"""Baseline glue: the HF generate path matches our static decoding; FITM document spans and fallback;
M2-RAG arm inputs (shared pixel budget, question images)."""
import types

import torch

from eab_fixtures import encode
from omniaware.baselines import arms, m2rag_official, m2rag_qwen


@torch.no_grad()
def test_hf_greedy_matches_runner_static(qwen):
    enc = encode(qwen)
    res, _, _ = qwen.run(enc, None, 8, decode_static=True)
    text = arms.greedy(qwen, arms.hf_inputs(qwen, enc), 8)
    assert text == res["static"].text


def test_fitm_sources_cover_chunk_spans(qwen):
    enc = encode(qwen)
    sources = arms.fitm_sources(qwen, enc, ["c1", "c2"])
    for s, cid in zip(sources, ["c1", "c2"]):
        a, b = enc.spans[cid]
        assert s["text_token_indices"] == list(range(a, b)) and s["vision_token_indices"] == []
        assert s["structural_token_indices"] == []


def test_fitm_sources_keep_image_markers_out_of_the_content(qwen):
    # <|vision_start|> (152), two image pads (150), <|vision_end|> (153), then a text chunk.
    enc = types.SimpleNamespace(input_ids=torch.tensor([[1, 152, 150, 150, 153, 5]]),
                                spans={"img": (1, 5), "txt": (5, 6)})
    img, txt = arms.fitm_sources(qwen, enc, ["img", "txt"])
    assert img["text_token_indices"] == [] and img["vision_token_indices"] == [2, 3]
    assert img["structural_token_indices"] == [1, 4]
    assert txt["text_token_indices"] == [5] and txt["structural_token_indices"] == []


def test_fitm_without_candidates_returns_the_static_answer(qwen):
    text, record = arms.fitm_answer(qwen, None, [], 8, static_text="static answer")
    assert text == "static answer" and record == {"fallback": "no_candidates", "extra_forward_calls": 0}


def test_pixel_budget_matches_the_other_arms():
    budget = arms.pixel_budget(types.SimpleNamespace(size={"shortest_edge": 64 * 32 * 32, "longest_edge": 256 * 32 * 32}))
    assert budget == {"shortest_edge": 65536, "longest_edge": 262144}


def test_m2rag_arm_gets_question_images_and_the_shared_pixel_budget(qwen, monkeypatch):
    budgets, summarized = [], []

    class RecordingBackend:
        def __init__(self, model, processor, *, max_new_tokens, image_paths=None, image_size=None):
            budgets.append(image_size)
            self.calls = []

        def chat(self, messages):
            return {"response": "generated answer", "usage": {}}

    class RecordingSummarizer:
        def summarize(self, query_list, dry_run):
            summarized.extend(query_list)
            return {"placeholder_response": "generated document", "output_images": []}

    monkeypatch.setattr(m2rag_qwen, "LocalQwenVLM", RecordingBackend)
    monkeypatch.setattr(m2rag_official, "build_official_summarizer", lambda backend, **kw: RecordingSummarizer())
    qwen_like = types.SimpleNamespace(tok=qwen.tok, model=None, processor=None,
                                      size={"shortest_edge": 64 * 32 * 32, "longest_edge": 256 * 32 * 32})
    e = {"qid": "q1", "question": "What is held?", "query_images": ["/synthetic/question.jpg"]}
    chunks = [{"cid": "t1", "modality": "text", "title": "", "text": "alpha beta", "image": None}]
    answer, info = arms.m2rag_answer(qwen_like, e, chunks, {"t1": 0.5}, 8)
    assert answer == "generated answer" and info["query_images"] == 1
    assert budgets == [{"shortest_edge": 65536, "longest_edge": 262144}] * 2   # summarizer and extractor
    [query] = summarized
    assert [a["cached_image_url"] for a in query["aux_images"]] == ["/synthetic/question.jpg"]

"""A2 / B-3: the answer instruction is checked on the prompts a run really builds.

Every prompt that ``run.main`` encodes must contain the run's instruction: the direct and routed inputs, the joint
input, the v2 actions, the EAB arms, calibration, FiTM, SeaKR's sampling and per-document answers. M2-RAG's
extractor is checked on the messages it sends. The check compares against the instruction text itself, not
against a hash of it.
"""
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from omniaware import data, paths, pipeline, prompts, rerank, run
from omniaware.actions import V2_ACTIONS
from omniaware.baselines import arms as baseline_arms, m2rag_official, m2rag_qwen, m2rag_webqa, seakr
from omniaware.qwen_eab import Result

pytestmark = pytest.mark.usefixtures("tiny_reserves")
ANSWER_STYLES = ("default", "short", "cite")


def _expected_instruction(answer_style: str) -> str:
    return prompts.instruction_for_style("okvqa", answer_style, cite_instruction=pipeline.CITATION_INSTRUCTION,
                                         short_instruction=pipeline.SHORT_INSTRUCTION)["instruction"]


def _examples(n):
    out = []
    for i in range(n):
        cands = [{"cid": f"k{j}", "modality": "text", "title": f"T{j}", "text": f"passage {j} of q{i}",
                  "image": None, "gold": None} for j in range(1, 6)]
        cands += [{"cid": f"i{j}", "modality": "image", "title": f"cap {j}", "text": f"cap {j}",
                   "image": f"img{j}.jpg", "gold": None} for j in range(1, 3)]
        out.append({"dataset": "okvqa", "qid": f"q{i:03d}", "question": f"question {i}?", "qcate": "okvqa",
                    "query_images": [], "candidates": cands, "fixed_evidence": False, "golds": ["x"]})
    return out


@dataclass
class FakeEncoded:
    length: int
    split: int
    text: str
    spans: dict = field(default_factory=dict)
    input_ids: torch.Tensor | None = None
    blocked: frozenset = frozenset()


class FakeQwen:
    """QwenEAB stand-in. ``encode`` keeps the prompt text the model would read; ``prompts`` collects it."""

    prompts: list[str] = []

    def __init__(self, model_path, **kw):
        self.device, self.n_layers = "cpu", 4
        self.last_prefill_seconds = 0.0
        self.vocab_size = 32
        self.eos = frozenset({2})
        self.tok = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 2})()

    def encode(self, segments, split_marker):
        spans, pos = {}, 0
        for s in segments:
            n = len(s.text or "")
            if s.chunk:
                a, _ = spans.get(s.chunk, (pos, pos))
                spans[s.chunk] = (a, pos + n)
            pos += n
        text = "".join(s.text or "" for s in segments)
        FakeQwen.prompts.append(text)
        return FakeEncoded(length=pos, split=sum(len(s.text or "") for s in segments[:split_marker]), text=text,
                           spans=spans, input_ids=torch.ones(1, pos, dtype=torch.long))

    def run(self, enc, make_variants, max_new_tokens, decode_static=True):
        attn = torch.rand(enc.length) + 1e-3
        logits = torch.randn(32) * (enc.length % 7 + 1)
        results = {}
        decode = decode_static(attn, logits) if callable(decode_static) else decode_static
        if decode:
            results["static"] = Result(f"static@{enc.length}", logits, [1, 3], 0.01, attn, [-0.5, -1.0])
        variants = make_variants(attn, logits) if make_variants else {}
        for name in variants:
            results[name] = Result(f"{name}@{enc.length}", logits, [1], 0.01, attn, [-0.5])
        return results, attn, logits


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    FakeQwen.prompts = []
    examples = _examples(14)
    monkeypatch.setitem(data.LOADERS, "okvqa", lambda **kw: examples)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "LOCAL_CACHE", tmp_path / "cache")
    monkeypatch.setattr(run, "QwenEAB", FakeQwen)
    monkeypatch.setattr(run, "load_relevance", lambda exs, path, **kw: {
        e["qid"]: {c["cid"]: ({"image_clip": 1.0 / int(c["cid"][1:])} if c["modality"] == "image"
                              else {"text": 1.0 / int(c["cid"][1:])}) for c in e["candidates"]} for e in exs})
    monkeypatch.setattr(rerank, "ce_logits", lambda exs, path, model, **kw: {
        e["qid"]: {c["cid"]: float(c["cid"][1:]) for c in e["candidates"]} for e in exs})
    return tmp_path


@pytest.mark.parametrize("answer_style", ANSWER_STYLES)
def test_every_prompt_of_a_run_carries_the_run_instruction(fake_run, monkeypatch, answer_style):
    seen = {"fitm": [], "seakr": [], "m2rag": []}

    def fake_fitm(qwen, enc, chunk_ids, max_new_tokens, static_text=None):
        seen["fitm"].append(enc.text)
        return "fitm answer", {"extra_forward_calls": 0}

    def fake_score(qwen, enc, n, max_new_tokens, layer=None, seed=42):
        seen["seakr"].append(enc.text)
        return 1.0

    def fake_m2rag(qwen, e, chunks, rel, max_new_tokens, document_tokens=384, instruction=None):
        seen["m2rag"].append(instruction)
        return "m2rag answer", {"calls": 0}

    monkeypatch.setattr(baseline_arms, "fitm_answer", fake_fitm)
    monkeypatch.setattr(baseline_arms, "m2rag_answer", fake_m2rag)
    monkeypatch.setattr(seakr, "score_input", fake_score)
    tag = f"prompts_{answer_style}"
    run.main(["--dataset", "okvqa", "--split", "test", "--n-calib", "6", "--calib-source", "heldout",
              "--n-dev", "4", "--tag", tag, "--image-mode", "clip", "--answer-style", answer_style,
              "--v2", "--v2-reranker", "--fitm", "--m2rag", "--seakr-samples", "4", "--seakr-threshold", "0"])
    expected = _expected_instruction(answer_style)
    out = paths.RUNS / tag
    config = json.loads((out / "okvqa_config.json").read_text(encoding="utf8"))
    assert config["instruction"] == expected
    assert config["instruction_sha256"] == prompts.instruction_sha256(expected)
    assert FakeQwen.prompts and seen["fitm"] and seen["seakr"] and seen["m2rag"]
    missing = [text for text in FakeQwen.prompts if expected not in text]
    assert missing == [], f"{len(missing)} prompts without the run instruction, first: {missing[:1]}"
    assert all(instruction == expected for instruction in seen["m2rag"])
    assert all(expected in text for text in seen["fitm"] + seen["seakr"])
    rows = [json.loads(line) for line in (out / "okvqa.jsonl").read_text(encoding="utf8").splitlines()]
    assert rows and all(set(r["actions"]) == set(V2_ACTIONS) for r in rows)
    assert all(r["actions"][name]["available"] for r in rows for name in ("direct", "static_text", "static_image",
                                                                          "static_mixed", "rerank_static"))
    assert all(set(r["parsed"]) == set(r["preds"]) for r in rows)
    assert all(r["preds"]["direct"] == r["parsed"]["direct"]["answer"] for r in rows)   # clean text: same string
    # the completion marker counts the decoded arms and the available actions (B-5)
    marker = json.loads((out / "okvqa.merged.json").read_text(encoding="utf8"))
    counted = sum(len(r["preds"]) + sum(1 for a in r["actions"].values() if a["available"]) for r in rows)
    assert marker["n_answers"] == counted and marker["empty_answer_rate"] == 0.0


def test_m2rag_extractor_messages_carry_the_instruction(monkeypatch):
    sent = []

    class RecordingBackend:
        def __init__(self, model, processor, *, max_new_tokens, image_size=None, **kw):
            self.calls = []

        def chat(self, messages):
            sent.append(messages)
            return {"response": "generated answer", "usage": {}}

    class RecordingSummarizer:
        def summarize(self, query_list, dry_run):
            return {"placeholder_response": "generated document", "output_images": []}

    monkeypatch.setattr(m2rag_qwen, "LocalQwenVLM", RecordingBackend)
    monkeypatch.setattr(m2rag_official, "build_official_summarizer", lambda backend, **kw: RecordingSummarizer())
    monkeypatch.setattr(m2rag_webqa, "prepare_sources", lambda tok, sources, image_paths, rel: [])
    qwen_like = SimpleNamespace(tok=None, model=None, processor=None,
                                size={"shortest_edge": 64 * 32 * 32, "longest_edge": 256 * 32 * 32})
    e = {"qid": "q1", "dataset": "okvqa", "question": "What is held?", "query_images": []}
    expected = prompts.instruction_text("okvqa")
    answer, _ = baseline_arms.m2rag_answer(qwen_like, e, [], {}, 8, instruction=expected)
    assert answer == "generated answer"
    text = "".join(part.get("text", "") for part in sent[0] if part.get("type") == "text")
    assert expected in text and "What is held?" in text


def test_the_m2rag_extractor_falls_back_to_the_family_instruction(monkeypatch):
    sent = []

    class RecordingBackend:
        def __init__(self, model, processor, *, max_new_tokens, image_size=None, **kw):
            self.calls = []

        def chat(self, messages):
            sent.append(messages)
            return {"response": "x", "usage": {}}

    monkeypatch.setattr(m2rag_qwen, "LocalQwenVLM", RecordingBackend)
    monkeypatch.setattr(m2rag_official, "build_official_summarizer", lambda backend, **kw: SimpleNamespace(
        summarize=lambda query_list, dry_run: {"placeholder_response": "d", "output_images": []}))
    monkeypatch.setattr(m2rag_webqa, "prepare_sources", lambda tok, sources, image_paths, rel: [])
    qwen_like = SimpleNamespace(tok=None, model=None, processor=None,
                                size={"shortest_edge": 64 * 32 * 32, "longest_edge": 256 * 32 * 32})
    e = {"qid": "q2", "dataset": "webqa", "question": "Where?", "query_images": []}
    baseline_arms.m2rag_answer(qwen_like, e, [], {}, 8)
    text = "".join(part.get("text", "") for part in sent[0] if part.get("type") == "text")
    assert prompts.instruction_text("webqa") in text


def test_no_arm_uses_a_different_instruction():
    # The M2-RAG summarizer writes the evidence document and does not answer the question, so it is not an arm
    # of the answer comparison; every answering arm uses the run's instruction (see the test above).
    assert prompts.EXCEPTIONS == {}


def test_webqa_and_okvqa_wording_is_the_frozen_default():
    webqa = prompts.instruction_text("webqa")
    okvqa = prompts.instruction_text("okvqa")
    assert "complete sentence" in webqa and "key object" in webqa
    assert "shortest word or phrase" in okvqa
    assert "Return exactly two lines." in webqa
    assert all("complete names, numbers and units" in text for text in (webqa, okvqa))
    assert "complete sentence" in prompts.instruction_text("videorag")
    assert prompts.instruction_text("hybridqa")


def test_build_segments_uses_the_registry_for_every_default_arm():
    question = {"dataset": "webqa", "question": "Where is the tower?", "query_images": []}
    text = pipeline.build_segments(question, None)[0][-1].text
    assert prompts.instruction_text("webqa") in text
    cite = pipeline.build_segments(question, None, "cite")[0][-1].text
    assert pipeline.CITATION_INSTRUCTION in cite
    assert prompts.instruction_text("webqa") not in cite


def test_overrides_are_not_chosen_by_this_test_as_the_default():
    custom = prompts.instruction_text("okvqa", overrides={"okvqa": "CUSTOM"})
    assert custom == "CUSTOM"
    assert prompts.instruction_text("okvqa") != "CUSTOM"

"""Run the migrated baselines (Found-in-the-Middle, M2-RAG) on our encoded inputs.

Each function answers one question with the same backbone, prompt format, candidate chunks and
generation cap as the Omni-Aware arms of the run, so the arms differ only in the method.
"""
from __future__ import annotations

import torch
from transformers import GenerationConfig

from ..qwen_eab import STATE


def hf_inputs(qwen, enc) -> dict:
    """Encoded prompt -> keyword inputs for the Hugging Face forward/generate of the backbone."""
    kw = dict(input_ids=enc.input_ids.to(qwen.device), attention_mask=torch.ones_like(enc.input_ids).to(qwen.device))
    if getattr(qwen, "kind", "qwen3_vl") == "internvl":
        if enc.pixel_values is not None:
            kw["pixel_values"] = enc.pixel_values.to(qwen.device, qwen.model.dtype)
        return kw
    kw["mm_token_type_ids"] = enc.mm_token_type_ids.to(qwen.device)
    if enc.pixel_values is not None:
        kw.update(pixel_values=enc.pixel_values.to(qwen.device, qwen.model.dtype),
                  image_grid_thw=enc.image_grid_thw.to(qwen.device))
    return kw


def generation_config(qwen, max_new_tokens: int) -> GenerationConfig:
    return GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                            eos_token_id=sorted(qwen.eos), pad_token_id=qwen.tok.eos_token_id)


@torch.no_grad()
def greedy(qwen, inputs: dict, max_new_tokens: int) -> str:
    STATE.reset()
    if hasattr(qwen.model, "model") and hasattr(qwen.model.model, "rope_deltas"):
        qwen.model.model.rope_deltas = None
    out = qwen.model.generate(**inputs, generation_config=generation_config(qwen, max_new_tokens))
    return qwen.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ------------------------------------------------------- Found-in-the-Middle --
def fitm_sources(qwen, enc, chunk_ids: list[str]) -> list[dict]:
    """Document token indices of each retrieved chunk: content text, image patches, and the vision
    start/end markers of an image (structural: part of the document, but not replaced by a null)."""
    ids = enc.input_ids[0].tolist()
    vision = set(i for i, t in enumerate(ids) if t == qwen.image_token_id)
    markers = {qwen.vision_start_id, qwen.vision_end_id}
    sources = []
    for cid in chunk_ids:
        a, b = enc.spans[cid]
        sources.append({"source_id": cid,
                        "text_token_indices": [i for i in range(a, b) if i not in vision and ids[i] not in markers],
                        "vision_token_indices": [i for i in range(a, b) if i in vision],
                        "structural_token_indices": [i for i in range(a, b) if ids[i] in markers]})
    return sources


@torch.no_grad()
def fitm_answer(qwen, enc, chunk_ids: list[str], max_new_tokens: int, static_text: str | None = None) -> tuple[str, dict]:
    """FiTM answer for the input's chunks. With no candidates there is nothing to calibrate, so the
    static answer on the same input (``static_text``) is returned and the fallback is recorded."""
    if not chunk_ids:
        return static_text, {"fallback": "no_candidates", "extra_forward_calls": 0}
    from .fitm_qwen_adapter import document_indices, prepare_fitm_calibration, qwen_fitm
    if getattr(qwen, "kind", "qwen3_vl") != "qwen3_vl":
        raise NotImplementedError("the FITM null-reference adapter handles Qwen3-VL image grids only")
    inputs = hf_inputs(qwen, enc)
    sources = fitm_sources(qwen, enc, chunk_ids)
    STATE.reset()
    qwen.model.model.rope_deltas = None
    alpha, record = prepare_fitm_calibration(qwen.model, inputs, sources, qwen.tok)
    with qwen_fitm(qwen.model, document_indices(sources), alpha=alpha):
        text = greedy(qwen, inputs, max_new_tokens)
    return text, {"extra_forward_calls": record["extra_forward_calls_started"]}


# ------------------------------------------------------------------- M2-RAG --
def pixel_budget(qwen) -> dict:
    """The processor size QwenEAB.encode gives a full-budget image (its cap is longest_edge), so that
    M2-RAG's images get the same pixel budget as every other arm."""
    cap = qwen.size["longest_edge"]
    return {"shortest_edge": min(qwen.size["shortest_edge"], cap), "longest_edge": cap}


PROTOCOL_DIFFERENCES = {
    "m2rag.summarizer": "The official M2RAG single-stage summarizer writes an intermediate document (at most 384 "
                        "tokens) from the chunks before the answer extractor reads it. Every other arm reads the "
                        "chunks directly. This is the M2RAG protocol and is kept.",
    "m2rag.extractor_prompt": "The extractor's own constraints (EXTRACTOR_INSTRUCTION) follow the run's shared "
                              "answer instruction. The official constraints are kept.",
    "m2rag.chat_template": "The official flat-content user message through LocalQwenVLM, not the ChatML prompt of "
                           "the other arms.",
    "m2rag.video": "M2-RAG accepts no video evidence, so it is not run on VideoRAG-Synth.",
    "fitm": "None: Found-in-the-Middle reads the joint input with the run's prompt and pixel budget.",
}


def m2rag_answer(qwen, e: dict, chunks: list[dict], rel: dict[str, float], max_new_tokens: int,
                 document_tokens: int = 384, instruction: str | None = None) -> tuple[str, dict]:
    """Official M2RAG single-stage summarizer over the same chunks, then the document-only answer extractor.

    The question's own images (OK-VQA query images) reach both the summarizer and the extractor, as
    the question image does for every other arm. Images use the same pixel budget as the other arms, and each
    source text is the word-budgeted text every other arm shows (pipeline.chunk_text). The chat template is the
    official one. The remaining differences are listed in PROTOCOL_DIFFERENCES and written to the run configuration.
    ``instruction`` is the run's answer instruction; run.py passes the one every arm decodes with. Without it, the
    dataset family's instruction is used.
    """
    from .m2rag_official import build_official_summarizer
    from .m2rag_qwen import LocalQwenVLM
    from .m2rag_webqa import extractor_messages, official_query, prepare_sources
    from .. import pipeline, prompts
    query_images = list(e.get("query_images") or [])
    if instruction is None and e.get("dataset"):
        instruction = prompts.instruction_text(e["dataset"])
    sources, image_paths = [], {}
    for j, c in enumerate(chunks):
        fact_id = str(900000 + j) if c["modality"] == "image" else c["cid"]
        text = pipeline.chunk_text(c)
        if c["modality"] == "image" and c.get("title") and c["title"] != text:
            text = f"{c['title']}: {text}" if text else c["title"]
        sources.append({"source_id": c["cid"], "fact_id": fact_id, "modality": c["modality"], "text": text})
        if c["modality"] == "image":
            image_paths[fact_id] = c["image"]
    prepared = prepare_sources(qwen.tok, sources, image_paths, {c["cid"]: rel[c["cid"]] for c in chunks})
    STATE.reset()
    size = pixel_budget(qwen)
    backend = LocalQwenVLM(qwen.model, qwen.processor, max_new_tokens=document_tokens, image_size=size)
    summarizer = build_official_summarizer(backend, multi_stage=False)
    query = official_query(e["qid"], e["question"], prepared, query_images)
    document = summarizer.summarize(query_list=[query], dry_run=False)
    extractor = LocalQwenVLM(qwen.model, qwen.processor, max_new_tokens=max_new_tokens, image_size=size)
    answer = extractor.chat(extractor_messages(e["question"], document, query_images, instruction))["response"].strip()
    return answer, {"calls": len(backend.calls) + len(extractor.calls), "query_images": len(query_images)}

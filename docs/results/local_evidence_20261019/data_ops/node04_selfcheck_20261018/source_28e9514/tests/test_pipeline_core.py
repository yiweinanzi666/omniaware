"""Prompt labels, uncertainty inputs and the chunk-identifier contract of pipeline.py."""
import pytest
import torch

from omniaware import pipeline
from omniaware.qwen_eab import Encoded


def _chunks():
    return [
        {"cid": "t1", "modality": "text", "title": "Title", "text": "a passage about Paris", "image": None},
        {"cid": "i1", "modality": "image", "title": "", "text": "", "image": "img.png"},
        {"cid": "v1", "modality": "video", "title": "", "text": "a transcript", "frames": ["f0.png", "f1.png"]},
        {"cid": "b1", "modality": "table", "title": "", "text": "a|b\n1|2"},
    ]


def _question(query_images=()):
    return {"dataset": "okvqa", "question": "Where is it?", "query_images": list(query_images)}


@pytest.mark.parametrize("style", ["default", "short", "cite"])
def test_every_evidence_chunk_is_labelled_E_j_in_every_answer_style(style):
    segments, _ = pipeline.build_segments(_question(), _chunks(), style)
    text = "".join(s.text for s in segments if s.text)
    for j, label in enumerate(["[E1] (text)", "[E2] (image)", "[E3] (video)", "[E4] (table)"], 1):
        assert label in text, (style, label)
    assert "Document" not in text


def test_citation_sources_use_the_same_identifiers():
    segments, _ = pipeline.build_segments(_question(), _chunks(), "cite")
    sources = pipeline.citation_sources(segments, _chunks())
    assert [s["citation_id"] for s in sources] == ["E1", "E2", "E3", "E4"]
    assert sources[0]["rendered_text"] == "Title: a passage about Paris"
    assert sources[3]["source_id"] == "b1"


def test_build_segments_takes_no_pixel_override():
    with pytest.raises(TypeError):
        pipeline.build_segments(_question(), _chunks(), "default", {"t1": 64})


def _enc(spans):
    return Encoded(input_ids=torch.zeros(1, 8, dtype=torch.long), mm_token_type_ids=torch.zeros(1, 8, dtype=torch.long),
                   pixel_values=None, image_grid_thw=None, spans=spans, split=6, length=8)


def test_raw_uncertainty_rejects_evidence_without_a_span():
    attn = torch.full((8,), 0.125)
    logits = torch.zeros(32)
    enc = _enc({"c1": (2, 5)})
    raw = pipeline.raw_uncertainty(attn, enc, ["c1"], logits, 0.3, 32)
    assert raw["h_grd"] == pytest.approx(1.0) and raw["h_sem"] == pytest.approx(1.0)
    with pytest.raises(KeyError, match="c9"):
        pipeline.raw_uncertainty(attn, enc, ["c1", "c9"], logits, 0.3, 32)


def test_context_ids_only_name_a_query_image_that_exists():
    assert pipeline.context_ids({"query_images": []}) == []
    assert pipeline.context_ids({"query_images": ["q.png"]}) == ["__query__"]
    # A text-only question has no __query__ span; its pre-retrieval H_grd is the empty-evidence value 0.
    enc = _enc({})
    raw = pipeline.raw_uncertainty(torch.full((8,), 0.125), enc, pipeline.context_ids({"query_images": []}),
                                   torch.zeros(32), 0.3, 32)
    assert raw["h_grd"] == 0.0


def test_norm_maps_the_calibration_median_to_one_half():
    norm = pipeline.Norm.fit([0.1, 0.2, 0.3, 0.4, 0.5])
    assert norm(0.3) == pytest.approx(0.5)
    assert 0.0 <= norm(0.0) <= norm(0.6) <= 1.0


def test_robust_scale_fallback_is_the_one_norm_and_zsum_use():
    # A constant calibration sample (IQR ~ 0) falls back to scale 1 for Norm and zsum alike.
    assert pipeline.robust_scale(0.0) == 1.0
    assert pipeline.robust_scale(pipeline.IQR_FLOOR / 10) == 1.0
    assert pipeline.robust_scale(0.5) == 0.5

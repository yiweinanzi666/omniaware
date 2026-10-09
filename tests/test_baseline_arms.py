"""Baseline glue: the HF generate path matches our static decoding; FITM document spans."""
import torch

from eab_fixtures import encode
from omniaware.baselines import arms


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

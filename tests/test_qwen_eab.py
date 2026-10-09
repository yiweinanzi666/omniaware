"""EAB attention hook inside a tiny randomly initialised Qwen3-VL (CPU, no weights needed)."""
import math
import types

import pytest
import torch

from omniaware import qwen_eab
from omniaware.pipeline import Variant
from omniaware.qwen_eab import STATE, QwenEAB, Segment

transformers = pytest.importorskip("transformers")
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration  # noqa: E402


class CharTokenizer:
    """Deterministic text-only stand-in for the Qwen tokenizer."""

    eos_token_id = 157

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 140 + 1 for ch in text]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(map(str, tokens))

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 155, "<|endoftext|>": 156}[token]


@pytest.fixture(scope="module")
def qwen():
    config = Qwen3VLConfig(
        text_config=dict(vocab_size=160, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[0]),
        image_token_id=150, video_token_id=151, vision_start_token_id=152, vision_end_token_id=153)
    config.text_config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(config).eval()
    # Same attribute setup as QwenEAB.__init__, without loading a checkpoint.
    q = QwenEAB.__new__(QwenEAB)
    q.device, q.model, q.tok, q.processor = "cpu", model, CharTokenizer(), None
    q.layers = []
    for m in model.modules():
        if m.__class__.__name__ == "Qwen3VLTextAttention":
            m._erc_layer = m.layer_idx
            q.layers.append(m.layer_idx)
    q.n_layers = len(q.layers)
    q.eos = {155, 156, 157}
    q.vocab_size = 160
    return q


def encode(q):
    segments = [Segment(text="<user>"), Segment(text="Document 1: alpha beta gamma", chunk="c1"),
                Segment(text="Document 2: delta epsilon", chunk="c2"), Segment(text="Question: which? Answer:")]
    return q.encode(segments, split_marker=3)


def spec(enc, rows, gamma, u=1.0):
    return Variant("v", gamma, rows=rows, form="pos", gate=False).spec(enc, {"c1": 1.0, "c2": 0.2}, u, 0.4, 2)


def test_hook_matches_explicit_softmax_with_bias():
    torch.manual_seed(1)
    b, h, kvh, q_len, kv_len, d = 1, 4, 2, 3, 7, 8
    query, key, value = torch.randn(b, h, q_len, d), torch.randn(b, kvh, kv_len, d), torch.randn(b, kvh, kv_len, d)
    module = types.SimpleNamespace(_erc_layer=0, scaling=1 / math.sqrt(d), num_key_value_groups=h // kvh,
                                   is_causal=True, training=False)
    bias = torch.tensor([0.0, 2.0, 2.0, -1.0, 0.0, 0.0, 0.0])
    STATE.reset()
    STATE.bias, STATE.q_start = bias, kv_len - 2   # only the last two query rows are biased
    try:
        out, _ = qwen_eab._eab_sdpa(module, query, key, value, None, scaling=module.scaling)
    finally:
        STATE.reset()
    k = key.repeat_interleave(h // kvh, dim=1)
    v = value.repeat_interleave(h // kvh, dim=1)
    logits = query @ k.transpose(-1, -2) * module.scaling
    rows = torch.arange(kv_len - q_len, kv_len)[:, None]
    logits = logits.masked_fill(torch.arange(kv_len)[None, :] > rows, float("-inf"))
    logits[..., 1:, :] += bias        # rows kv_len-2 and kv_len-1 are query rows 1 and 2
    expected = (torch.softmax(logits, -1) @ v).transpose(1, 2)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_shared_prefix_equals_full_forward(qwen):
    enc = encode(qwen)
    _, _, base_logits = qwen.run(enc, None, 1, decode_static=False)
    qwen.model.model.rope_deltas = None
    full = qwen.model(input_ids=enc.input_ids, mm_token_type_ids=enc.mm_token_type_ids).logits[0, -1].float()
    torch.testing.assert_close(base_logits, full, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_zero_bias_reproduces_static(qwen):
    enc = encode(qwen)
    res, _, _ = qwen.run(enc, lambda a, lg: {"ans1": spec(enc, "ans1", 0.0), "ans2": spec(enc, "ans2", 0.0)},
                         6, decode_static=True)
    assert res["ans1"].tokens == res["static"].tokens == res["ans2"].tokens
    torch.testing.assert_close(res["ans1"].first_logits, res["static"].first_logits, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_decode_only_eab_cannot_change_first_token_but_eab_p_can(qwen):
    enc = encode(qwen)
    res, _, _ = qwen.run(enc, lambda a, lg: {"ans2": spec(enc, "ans2", 8.0), "ans1": spec(enc, "ans1", 8.0)},
                         4, decode_static=True)
    # Decode-only EAB (paper default): first-token logits are the unbiased prefill's.
    assert torch.equal(res["ans2"].first_logits, res["static"].first_logits)
    assert res["ans2"].tokens[:1] == res["static"].tokens[:1]
    # EAB-P biases the last prompt row, so the first-token distribution moves.
    assert not torch.allclose(res["ans1"].first_logits, res["static"].first_logits, atol=1e-3)


@torch.no_grad()
def test_decode_only_eab_changes_later_rows(qwen):
    enc = encode(qwen)
    cache = qwen._prefix(enc)
    qwen._activate(enc, None)
    base = qwen._step(enc.input_ids[:, enc.split:], cache)
    first = torch.tensor([[int(base.argmax())]])
    unbiased_second = qwen._step(first, cache)
    qwen_eab._crop(cache, enc.length)
    qwen._activate(enc, spec(enc, "ans2", 8.0))
    biased_second = qwen._step(first, cache)
    qwen._activate(enc, None)
    assert not torch.allclose(unbiased_second, biased_second, atol=1e-3)

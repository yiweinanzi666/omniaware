"""EAB attention hook inside a tiny randomly initialised Qwen3-VL (CPU, no weights needed)."""
import dataclasses
import math
import types

import pytest
import torch

from omniaware import interventions, qwen_eab
from omniaware.pipeline import Variant
from omniaware.qwen_eab import STATE

from eab_fixtures import encode


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


def test_blocked_keys_get_exactly_zero_weight_and_the_rest_is_renormalised():
    torch.manual_seed(2)
    b, h, kvh, q_len, kv_len, d = 1, 4, 2, 3, 7, 8
    query, key, value = torch.randn(b, h, q_len, d), torch.randn(b, kvh, kv_len, d), torch.randn(b, kvh, kv_len, d)
    module = types.SimpleNamespace(_erc_layer=0, scaling=1 / math.sqrt(d), num_key_value_groups=h // kvh,
                                   is_causal=True, training=False)
    blocked = torch.tensor([False, False, True, True, False, False, False])
    STATE.reset()
    STATE.blocked = blocked
    try:
        out, _ = qwen_eab._eab_sdpa(module, query, key, value, None, scaling=module.scaling)
        # The output does not depend on the values of blocked keys: their weight is exactly zero on every row.
        other = value.clone()
        other[:, :, blocked] = torch.randn(b, kvh, int(blocked.sum()), d) * 100
        out_other, _ = qwen_eab._eab_sdpa(module, query, key.clone(), other, None, scaling=module.scaling)
    finally:
        STATE.reset()
    assert torch.isfinite(out).all() and torch.equal(out, out_other)
    # Removing the blocked keys leaves the same distribution on the rest (proportional weights).
    keep = (~blocked).nonzero().flatten()
    k = key.repeat_interleave(h // kvh, dim=1)[:, :, keep]
    v = value.repeat_interleave(h // kvh, dim=1)[:, :, keep]
    logits = query @ k.transpose(-1, -2) * module.scaling
    rows = torch.arange(kv_len - q_len, kv_len)[:, None]
    logits = logits.masked_fill(keep[None, :] > rows, float("-inf"))
    expected = (torch.softmax(logits, -1) @ v).transpose(1, 2)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_blocked_span_is_invisible_to_every_row(qwen):
    enc = encode(qwen)
    a, b = enc.spans["c1"]
    other = enc.input_ids.clone()     # different tokens with the same length: positions stay put
    other[0, a:b] = torch.tensor([(t * 7 + 3) % 140 + 1 for t in other[0, a:b].tolist()])
    alt = dataclasses.replace(enc, input_ids=other)
    res, _, _ = qwen.run(interventions.mask_span(enc, "c1"), None, 4, decode_static=True)
    res_alt, _, _ = qwen.run(interventions.mask_span(alt, "c1"), None, 4, decode_static=True)
    torch.testing.assert_close(res["static"].first_logits, res_alt["static"].first_logits, atol=1e-5, rtol=1e-5)
    assert res["static"].tokens == res_alt["static"].tokens
    # the same change without the block does move the answer distribution: the span is visible there
    plain, _, _ = qwen.run(enc, None, 1, decode_static=True)
    plain_alt, _, _ = qwen.run(alt, None, 1, decode_static=True)
    assert not torch.allclose(plain["static"].first_logits, plain_alt["static"].first_logits, atol=1e-3)


@torch.no_grad()
def test_blocked_span_gets_no_eab_bias(qwen):
    enc = encode(qwen)
    biased = qwen._bias_tensor(enc, spec(enc, "ans1", 1.0))
    masked = qwen._bias_tensor(interventions.mask_span(enc, "c1"), spec(enc, "ans1", 1.0))
    a, b = enc.spans["c1"]
    assert biased[a:b].abs().sum() > 0 and torch.equal(masked[a:b], torch.zeros(b - a))
    c2a, c2b = enc.spans["c2"]
    assert torch.equal(masked[c2a:c2b], biased[c2a:c2b])      # other chunks keep their bias


@torch.no_grad()
def test_blocked_span_at_token_zero_is_rejected(qwen):
    enc = encode(qwen)
    shifted = dataclasses.replace(enc, spans={**enc.spans, "c1": (0, 5)}, blocked=frozenset({"c1"}))
    with pytest.raises(ValueError):
        qwen._prefix(shifted)


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


@torch.no_grad()
def test_arms_report_time_and_first_answer_row_attention(qwen):
    enc = encode(qwen)
    res, _, _ = qwen.run(enc, lambda a, lg: {"ans1": spec(enc, "ans1", 1.2), "ans2": spec(enc, "ans2", 1.2)},
                         4, decode_static=True)
    assert qwen.last_prefill_seconds > 0
    for name, r in res.items():
        assert r.seconds > 0
        if len(r.tokens) >= 2:          # the first answer row exists only once a token was fed back
            assert r.attention.shape[-1] == enc.length + 1
            torch.testing.assert_close(r.attention.sum(), torch.tensor(1.0), atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_capture_layer_switch(qwen):
    enc = encode(qwen)
    _, last, _ = qwen.run(enc, None, 1, decode_static=False)
    qwen.capture_at = 0
    try:
        _, first, _ = qwen.run(enc, None, 1, decode_static=False)
    finally:
        qwen.capture_at = -1
    assert last.shape == first.shape and not torch.allclose(last, first)

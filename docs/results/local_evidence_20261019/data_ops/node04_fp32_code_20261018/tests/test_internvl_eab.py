"""EAB on a tiny native InternVL (Qwen3 LM): same hook, 1-D RoPE prefill, same first-token property."""
import torch

from eab_fixtures import encode, make_internvl
from omniaware.pipeline import Variant


def spec(enc, rows, gamma):
    return Variant("v", gamma, rows=rows, form="pos", gate=False).spec(enc, {"c1": 1.0, "c2": 0.2}, 1.0, 0.4, 2)


@torch.no_grad()
def test_internvl_prefix_hook_and_first_token():
    q = make_internvl()
    assert q.layers == [0, 1]
    enc = encode(q)
    _, _, base = q.run(enc, None, 1, decode_static=False)
    full = q.model(input_ids=enc.input_ids).logits[0, -1].float()
    torch.testing.assert_close(base, full, atol=1e-4, rtol=1e-4)
    res, attn, _ = q.run(enc, lambda a, lg: {"ans2": spec(enc, "ans2", 8.0), "ans1": spec(enc, "ans1", 8.0)},
                         4, decode_static=True)
    assert attn is not None and attn.shape[0] == enc.length
    assert torch.equal(res["ans2"].first_logits, res["static"].first_logits)
    assert not torch.allclose(res["ans1"].first_logits, res["static"].first_logits, atol=1e-3)

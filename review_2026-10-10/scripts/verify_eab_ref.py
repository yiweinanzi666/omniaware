"""Independent check of QwenEAB.run (cache + early/late re-runs + decode) against a no-cache reference.

Reference: full forward of prompt+generated tokens with an explicit causal mask and an explicit additive
bias on rows >= q_start (computed here, not through qwen_eab's hook). Runs on the tiny random Qwen3-VL.
"""
import sys, math
REPO = "C:/Users/22688/Desktop/www_code/omniaware"
sys.path[:0] = [REPO + "/src", REPO + "/tests"]

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward as ORIG_SDPA

import omniaware.qwen_eab as qe
from omniaware.qwen_eab import QwenEAB, Encoded, STATE
from omniaware.pipeline import Variant
from eab_fixtures import make_qwen, encode as text_encode

torch.set_num_threads(4)
REF = {"bias": None, "q_start": 0, "layers": None}


def ref_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    layer = getattr(module, "_erc_layer", None)
    if layer is None:
        return ORIG_SDPA(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                         is_causal=is_causal, **kw)
    B, H, Q, D = query.shape
    KV = key.shape[2]
    rep = H // key.shape[1]
    k = key.repeat_interleave(rep, dim=1).float()
    v = value.repeat_interleave(rep, dim=1).float()
    s = scaling if scaling is not None else module.scaling
    logits = (query.float() @ k.transpose(-1, -2)) * s
    qpos = torch.arange(KV - Q, KV)
    kpos = torch.arange(KV)
    causal = kpos[None, :] > qpos[:, None]
    logits = logits.masked_fill(causal[None, None], float("-inf"))
    if REF["bias"] is not None and (REF["layers"] is None or layer in REF["layers"]):
        add = torch.zeros(KV)
        n = min(KV, REF["bias"].numel())
        add[:n] = REF["bias"][:n]
        rows = (qpos >= REF["q_start"])
        logits = logits + torch.where(rows[:, None], add[None, :], torch.zeros(1))[None, None]
    w = torch.softmax(logits, dim=-1)
    out = (w @ v).transpose(1, 2).to(query.dtype)
    return out, None


def use(fn):
    AttentionInterface.register("sdpa", fn)


def bias_vec(q, enc, spec):
    col = torch.zeros(enc.length)
    for cid, val in spec.values.items():
        if cid in enc.spans and val != 0.0:
            a, b = enc.spans[cid]
            col[a:b] = float(val)
    return col


@torch.no_grad()
def reference_decode(q, enc, spec, steps):
    """Greedy decode with a full no-cache forward per step; returns (first_logits, tokens)."""
    ids = enc.input_ids.clone()
    toks, first = [], None
    REF["bias"] = None if spec is None else bias_vec(q, enc, spec)
    REF["q_start"] = 0 if spec is None else spec.q_start
    REF["layers"] = None if spec is None or spec.layers is None else spec.layers
    use(ref_sdpa)
    try:
        for step in range(steps):
            q.model.model.rope_deltas = None
            out = q.model(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids))
            logits = out.logits[0, -1].float()
            if first is None:
                first = logits
            nxt = int(logits.argmax())
            if nxt in q.eos:
                break
            toks.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    finally:
        use(qe._eab_sdpa)
        REF["bias"] = None
    return first, toks


def compare(name, q, enc, spec_fn, steps=5):
    res, _, _ = q.run(enc, lambda a, lg: spec_fn(enc), steps, decode_static=True)
    for arm in res:
        if arm == "static":
            spec = None
        else:
            spec = spec_fn(enc)[arm]
        ref_first, ref_toks = reference_decode(q, enc, spec, steps)
        ok_tok = res[arm].tokens == ref_toks
        diff = (res[arm].first_logits - ref_first).abs().max().item()
        print(f"{name:28s} arm={arm:8s} tokens_match={ok_tok} first_logit_maxdiff={diff:.2e} "
              f"hook_tokens={res[arm].tokens} ref_tokens={ref_toks}")
    return res


def main():
    q = make_qwen()
    enc = text_encode(q)
    # spec with rows = ans1 (t_end) and ans2 (decode-only), gamma large so the effect is visible
    def specs(enc):
        return {"ans1": Variant("ans1", 3.0, rows="ans1", form="pos", gate=False).spec(enc, {"c1": 1.0, "c2": 0.2}, 1.0, 0.4, 2),
                "ans2": Variant("ans2", 3.0, rows="ans2", form="pos", gate=False).spec(enc, {"c1": 1.0, "c2": 0.2}, 1.0, 0.4, 2)}
    print("text-only prompt: length", enc.length, "split", enc.split, "spans", enc.spans)
    use(qe._eab_sdpa)
    compare("text prompt, hook vs ref", q, enc, specs, steps=5)

    # Layer-restricted variant (mid layers) to test layer masking.
    def specs_mid(enc):
        return {"ans1_mid": Variant("x", 3.0, rows="ans1", form="pos", gate=False, layers="mid").spec(enc, {"c1": 1.0, "c2": 0.2}, 1.0, 0.4, 2)}
    compare("text prompt, layers=mid", q, enc, specs_mid, steps=4)

    # Multi-image prompt with a manual image block (bypasses the processor): checks image token positions and
    # the cache/rope path with an image in the prefix.
    VS, VE, IMG = 152, 153, 150
    seg_a = [5, 6, 7]
    seg_b = [9, 10, 11, 12]
    img_tokens = [IMG] * 4        # grid [1,4,4] -> (4*4)/(2*2)=4 tokens
    tail = [20, 21, 22, 23, 24]   # question tail (split point below)
    ids = seg_a + [VS] + img_tokens + [VE] + seg_b + tail
    types = [0] * len(seg_a) + [0] + [1] * 4 + [0] + [0] * len(seg_b) + [0] * len(tail)
    spans = {"__query__": (len(seg_a), len(seg_a) + 6), "c1": (len(seg_a) + 6, len(seg_a) + 6 + len(seg_b))}
    split = len(seg_a) + 6 + len(seg_b)
    g = torch.Generator().manual_seed(3)
    pix = torch.randn(16, 3 * 2 * 16 * 16, generator=g)
    grid = torch.tensor([[1, 4, 4]])
    enc_img = Encoded(input_ids=torch.tensor([ids]), mm_token_type_ids=torch.tensor([types]), pixel_values=pix,
                      image_grid_thw=grid, spans=spans, split=split, length=len(ids))

    def specs_img(enc):
        vals = {"__query__": 0.0, "c1": 1.0}
        return {"ans1": Variant("ans1", 3.0, rows="ans1", form="pos", gate=False).spec(enc, {"c1": 1.0}, 1.0, 0.4, 2),
                "ans2": Variant("ans2", 3.0, rows="ans2", form="pos", gate=False).spec(enc, {"c1": 1.0}, 1.0, 0.4, 2),
                "q": Variant("q", 3.0, rows="q", form="pos", gate=False).spec(enc, {"c1": 1.0}, 1.0, 0.4, 2)}

    # The hook path for images uses the vision tower, so compare with the same pixel_values in both paths.
    q.model.model.rope_deltas = None
    use(qe._eab_sdpa)
    cache = q._prefix(enc_img)
    print("image prompt: split", split, "length", len(ids), "rope_deltas", q.model.model.rope_deltas)
    compare_img = {}
    for name, spec in specs_img(enc_img).items():
        q.run  # keep reference
    res, _, _ = q.run(enc_img, lambda a, lg: specs_img(enc_img), 4, decode_static=True)
    for arm in res:
        spec = None if arm == "static" else specs_img(enc_img)[arm]
        # reference needs vision features: compute full forward with pixel_values, bias rows via REF
        REF["bias"] = None if spec is None else bias_vec(q, enc_img, spec)
        REF["q_start"] = 0 if spec is None else spec.q_start
        REF["layers"] = None
        use(ref_sdpa)
        toks, first = [], None
        ids_t = enc_img.input_ids.clone()
        types_t = enc_img.mm_token_type_ids.clone()
        try:
            with torch.no_grad():
                for step in range(4):
                    q.model.model.rope_deltas = None
                    out = q.model(input_ids=ids_t, mm_token_type_ids=types_t,
                                  pixel_values=pix, image_grid_thw=grid)
                    lg = out.logits[0, -1].float()
                    if first is None:
                        first = lg
                    nxt = int(lg.argmax())
                    if nxt in q.eos:
                        break
                    toks.append(nxt)
                    ids_t = torch.cat([ids_t, torch.tensor([[nxt]])], dim=1)
                    types_t = torch.cat([types_t, torch.zeros(1, 1, dtype=torch.long)], dim=1)
        finally:
            use(qe._eab_sdpa)
            REF["bias"] = None
        diff = (res[arm].first_logits - first).abs().max().item()
        print(f"image prompt arm={arm:8s} tokens_match={res[arm].tokens == toks} first_logit_maxdiff={diff:.2e} "
              f"hook={res[arm].tokens} ref={toks}")


if __name__ == "__main__":
    main()

"""Verify the Omni-Aware prompt and EAB path on a real checkpoint (run once on the GPU server, per backbone).

    python -m scripts.verify_real_model --model /path/to/qwen3-vl-8b-instruct --image sample.jpg \
        [--examples questions.json] [--dtype float32] [--max-pixels 262144] [--out result.json]

Checks (each reports ``ok``; the exit status is 1 when a check fails; the JSON report is printed):
  template    the hand-written ChatML prompt tokenises token by token like processor.apply_chat_template,
              for a text question and an image question (runs of image pads are compared as runs).
  segmented   the prefix + suffix prefill (the shared-cache path) gives the same first-token logits as one
              forward over the whole prompt.
  reference   every arm of QwenEAB.run (static, ans1, ans2) matches a no-cache reference: greedy tokens,
              first-token logits and the t_end-row attention of the last layer. The reference writes its own
              masked softmax, so it does not use qwen_eab's hook.
  statistics  not a pass/fail check: per modality the attention mass M_m and entropy H_m (paper Eq. routing),
              H_grd and H_pred, and the change of the attention mass on non-chunk positions (group 0: the
              instruction, the query image and the sink) under the ans1 EAB arm. These are the quantities the
              paper's claims about routing and about the bias's effect on group 0 depend on.

With a tiny random model the same functions run on CPU (tests/test_verify_real_model.py).
Bfloat16 checkpoints carry rounding noise, so the tolerances below are loose; tighten them with fp32 runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward as ORIG_SDPA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import pipeline, qwen_eab, routing  # noqa: E402
from omniaware.pipeline import Variant  # noqa: E402
from omniaware.qwen_eab import QwenEAB  # noqa: E402

TOLERANCE = {"logits": 0.05, "attention": 0.02}


# ------------------------------------------------------------------ template --
def collapse_image_runs(ids, image_token_id: int):
    """Replace every run of image pad tokens by one marker (-1). Returns (collapsed ids, run lengths)."""
    out, runs = [], []
    for t in ids:
        if t == image_token_id:
            if out and out[-1] == -1:
                runs[-1] += 1
            else:
                out.append(-1)
                runs.append(1)
        else:
            out.append(t)
    return out, runs


def compare_token_sequences(ours, theirs, image_token_id: int) -> dict:
    """Token-by-token comparison of two prompts. Image pads are compared as runs: their number depends on the
    image grid (pixel budget), which is reported separately as ``image_runs_equal``."""
    ours, theirs = list(ours), list(theirs)
    a, ra = collapse_image_runs(ours, image_token_id)
    b, rb = collapse_image_runs(theirs, image_token_id)
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    return {"ok": first is None, "len_ours": len(ours), "len_theirs": len(theirs),
            "first_mismatch": first,
            "ours_token": a[first] if first is not None and first < len(a) else None,
            "theirs_token": b[first] if first is not None and first < len(b) else None,
            "image_runs_equal": ra == rb, "image_runs_ours": ra, "image_runs_theirs": rb}


def template_check(q, dataset: str, question: str, image_path: str | None = None) -> dict:
    """Our segments versus processor.apply_chat_template for the same question (and query image)."""
    e = {"dataset": dataset, "question": question, "query_images": [image_path] if image_path else []}
    segments, split = pipeline.build_segments(e, None)
    enc = q.encode(segments, split)
    text = f"Question: {question}\n{pipeline.INSTRUCTION[dataset]}"
    if image_path:
        content = [{"type": "image", "image": Image.open(image_path).convert("RGB")},
                   {"type": "text", "text": "\n" + text}]
    else:
        content = [{"type": "text", "text": text}]
    out = q.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                          tokenize=True, return_dict=True)
    theirs = torch.as_tensor(out["input_ids"]).reshape(-1).tolist()
    return compare_token_sequences(enc.input_ids[0].tolist(), theirs, q.image_token_id)


# ----------------------------------------------------------------- forwards --
@torch.no_grad()
def _last_logits(q, enc, ids, types) -> torch.Tensor:
    """One forward over ``ids`` (prompt or prompt + generated tokens) without the KV cache."""
    kw = dict(input_ids=ids.to(q.device))
    if getattr(q, "kind", "qwen3_vl") == "internvl":
        if enc.pixel_values is not None:
            kw["pixel_values"] = enc.pixel_values.to(q.device, q.model.dtype)
    else:
        q.model.model.rope_deltas = None
        kw["mm_token_type_ids"] = types.to(q.device)
        if enc.pixel_values is not None:
            kw.update(pixel_values=enc.pixel_values.to(q.device, q.model.dtype),
                      image_grid_thw=enc.image_grid_thw.to(q.device))
    return q.model(**kw).logits[0, -1].float().cpu()


def segmented_check(q, enc, tol: float = TOLERANCE["logits"]) -> dict:
    """The cache path (prefix prefill, then the suffix) against one forward over the whole prompt."""
    _, _, seg = q.run(enc, None, 1, decode_static=False)
    full = _last_logits(q, enc, enc.input_ids, enc.mm_token_type_ids)
    diff = float((seg - full).abs().max())
    argmax_equal = int(seg.argmax()) == int(full.argmax())
    return {"ok": diff <= tol and argmax_equal, "max_abs_diff": diff, "argmax_equal": argmax_equal,
            "prompt_tokens": enc.length, "split": enc.split}


def _bias_vector(enc, arm) -> torch.Tensor:
    bias = torch.zeros(enc.length)
    if arm is not None:
        for cid, value in arm.values.items():
            if cid in enc.spans and value != 0.0:
                a, b = enc.spans[cid]
                bias[a:b] = float(value)
    return bias


@torch.no_grad()
def reference_decode(q, enc, arm, steps: int):
    """No-cache greedy decoding: each step is a full forward whose attention is written out here (explicit
    causal mask, explicit bias on the rows at or after q_start in the biased layers). Returns the first-token
    logits, the head-averaged last-layer attention of the t_end row (the last prompt row), and the tokens."""
    bias = _bias_vector(enc, arm)
    captured = {}

    def ref_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        layer = getattr(module, "_erc_layer", None)
        if layer is None:
            return ORIG_SDPA(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                             is_causal=is_causal, **kw)
        _, heads, q_len, _ = query.shape
        kv_len = key.shape[2]
        rep = heads // key.shape[1]
        k = key.repeat_interleave(rep, dim=1).double()
        v = value.repeat_interleave(rep, dim=1).double()
        s = scaling if scaling is not None else module.scaling
        logits = (query.double() @ k.transpose(-1, -2)) * s
        q_pos = torch.arange(kv_len - q_len, kv_len, device=query.device)
        k_pos = torch.arange(kv_len, device=query.device)
        logits = logits.masked_fill((k_pos[None, :] > q_pos[:, None])[None, None], float("-inf"))
        if arm is not None and (arm.layers is None or layer in arm.layers):
            # Keys after the prompt (generated tokens) carry no bias, as in the hook.
            col = torch.zeros(kv_len, dtype=torch.float64, device=query.device)
            col[:min(kv_len, bias.numel())] = bias[:kv_len].to(device=query.device, dtype=torch.float64)
            rows = (q_pos >= arm.q_start)[:, None]
            logits = logits + torch.where(rows, col[None, :], torch.zeros_like(col[:1]))
        w = torch.softmax(logits, dim=-1)
        if layer == q.n_layers - 1 and "row" not in captured:      # first forward = the prompt
            captured["row"] = w[0].mean(dim=0)[-1].float().cpu()
        return (w @ v).transpose(1, 2).to(query.dtype), None

    ids, types = enc.input_ids.clone(), enc.mm_token_type_ids.clone()
    tokens, first = [], None
    AttentionInterface.register("sdpa", ref_sdpa)
    try:
        for _ in range(steps):
            logits = _last_logits(q, enc, ids, types)
            if first is None:
                first = logits
            nxt = int(torch.argmax(logits))
            if nxt in q.eos:
                break
            tokens.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
            types = torch.cat([types, torch.zeros(1, 1, dtype=types.dtype)], dim=1)
    finally:
        AttentionInterface.register("sdpa", qwen_eab._eab_sdpa)     # restore the hook
    return first, captured.get("row"), tokens


def reference_check(q, enc, arms: dict, steps: int, tol=TOLERANCE) -> dict:
    """Every arm of QwenEAB.run against reference_decode. ``arms`` maps names to BiasSpec (None = unbiased);
    the static arm is always included. Token agreement is reported; the pass criterion is numerical."""
    variants = {name: spec for name, spec in arms.items() if spec is not None}
    res, _, _ = q.run(enc, lambda a, lg: variants, steps, decode_static=True)
    report = {}
    for name, r in res.items():
        first_ref, row_ref, tokens_ref = reference_decode(q, enc, arms.get(name), steps)
        logit_diff = float((r.first_logits - first_ref).abs().max())
        att_diff = (float((r.attention - row_ref).abs().max())
                    if r.attention is not None and row_ref is not None else None)
        report[name] = {"ok": logit_diff <= tol["logits"] and att_diff is not None and att_diff <= tol["attention"],
                        "tokens_match": r.tokens == tokens_ref, "first_logits_max_abs_diff": logit_diff,
                        "t_end_attention_max_abs_diff": att_diff, "tokens": r.tokens, "reference_tokens": tokens_ref}
    return {"ok": all(v["ok"] for v in report.values()), "prompt_tokens": enc.length, "arms": report}


# -------------------------------------------------------------- statistics --
def _summary(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    return {"n": int(v.size), "mean": float(v.mean()), "p10": float(p10), "p50": float(p50), "p90": float(p90)}


@torch.no_grad()
def statistics_rows(q, examples: list[dict], gamma: float) -> list[dict]:
    """Per question: M_m and H_m per modality on the unbiased t_end attention (the route's statistics), H_grd and
    H_pred, and the attention mass on group 0 (non-chunk positions) and on the first token, before and after
    the ans1 EAB arm with gamma."""
    rows = []
    for e in examples:
        chunks = e["chunks"]
        segments, split = pipeline.build_segments(e, chunks, e.get("answer_style", "default"))
        enc = q.encode(segments, split)
        _, grounding, logits = q.run(enc, None, 1, decode_static=False)
        modality = {c["cid"]: c["modality"] for c in chunks}
        stats, _ = routing.modality_statistics(grounding, routing.modality_groups(enc.spans, modality))
        rel = {c["cid"]: float(c.get("r", 1.0)) for c in chunks}
        arm = Variant("ans1", gamma, rows="ans1", form="pos", gate=False).spec(enc, rel, 1.0, 0.0, q.n_layers)
        res, _, _ = q.run(enc, lambda a, lg: {"eab": arm}, 1, decode_static=False)
        biased = res["eab"].attention
        chunk_positions = sorted({i for cid in modality if cid in enc.spans for i in range(*enc.spans[cid])})
        chunk_set = set(chunk_positions)
        group0 = [i for i in range(enc.length) if i not in chunk_set]
        rows.append({
            "qid": e.get("qid"), "prompt_tokens": enc.length,
            "modality": {m: {"M": s["mass"], "H": s["entropy"], "tokens": s["tokens"]} for m, s in stats.items()},
            "h_grd": qwen_eab.grounding_entropy(grounding, chunk_positions),
            "h_pred": qwen_eab.predictive_entropy(logits, q.vocab_size),
            "group0_mass_static": float(grounding[group0].sum()) if group0 else None,
            "group0_mass_eab": float(biased[group0].sum()) if group0 else None,
            "sink_mass_static": float(grounding[0]), "sink_mass_eab": float(biased[0])})
    return rows


def summarize_statistics(rows: list[dict]) -> dict:
    modalities = sorted({m for r in rows for m in r["modality"]})
    change = [r["group0_mass_eab"] - r["group0_mass_static"] for r in rows if r["group0_mass_static"] is not None]
    return {
        "questions": len(rows),
        "per_modality_H": {m: _summary([r["modality"][m]["H"] for r in rows if m in r["modality"]])
                           for m in modalities},
        "per_modality_M": {m: _summary([r["modality"][m]["M"] for r in rows if m in r["modality"]])
                           for m in modalities},
        "h_grd": _summary([r["h_grd"] for r in rows]),
        "h_pred": _summary([r["h_pred"] for r in rows]),
        "group0_mass_static": _summary([r["group0_mass_static"] for r in rows]),
        "group0_mass_eab": _summary([r["group0_mass_eab"] for r in rows]),
        "group0_mass_change": _summary(change),
        "sink_mass_change": _summary([r["sink_mass_eab"] - r["sink_mass_static"] for r in rows]),
    }


# -------------------------------------------------------------------- CLI --
def _chunk_spec(e_chunks: list[dict]) -> dict:
    return {c["cid"]: 1.0 - 0.2 * j for j, c in enumerate(e_chunks)}


def default_examples(image: str | None) -> list[dict]:
    chunks = [{"cid": "t1", "modality": "text", "title": "Background", "text": "A short placeholder passage."}]
    if image:
        chunks.append({"cid": "i1", "modality": "image", "title": "", "text": "", "image": image})
    return [{"qid": "default", "dataset": "okvqa", "question": "What is shown?",
             "query_images": [image] if image else [], "chunks": chunks}]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="local checkpoint directory (Qwen3-VL or native InternVL3.5)")
    ap.add_argument("--image", default=None, help="a local image for the image-question checks")
    ap.add_argument("--dataset", default="okvqa", choices=sorted(pipeline.INSTRUCTION))
    ap.add_argument("--question", default="What is shown in the picture?")
    ap.add_argument("--examples", default=None,
                    help="JSON list of {qid, dataset, question, query_images, chunks:[{cid, modality, title, text, "
                         "image, frames, r}]} for the statistics (default: one text chunk and the --image)")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-pixels", type=int, default=256 * 32 * 32)
    ap.add_argument("--max-patches", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=1.2)
    ap.add_argument("--steps", type=int, default=8, help="greedy steps in the reference check")
    ap.add_argument("--checks", nargs="+", default=["template", "segmented", "reference", "statistics"],
                    choices=["template", "segmented", "reference", "statistics"])
    ap.add_argument("--out", default=None, help="also write the JSON report here")
    args = ap.parse_args(argv)

    q = QwenEAB(args.model, device=args.device, dtype=getattr(torch, args.dtype), max_pixels=args.max_pixels,
                max_patches=args.max_patches)
    report = {"model": str(args.model), "kind": q.kind, "layers": q.n_layers, "dtype": args.dtype,
              "max_pixels": args.max_pixels, "checks": {}}
    text_chunks = [{"cid": "t1", "modality": "text", "title": "Background", "text": "A short placeholder passage."},
                   {"cid": "t2", "modality": "text", "title": "Other", "text": "A second placeholder passage."}]
    text_e = {"dataset": args.dataset, "question": args.question, "query_images": []}
    prompts = {}
    segs, split = pipeline.build_segments(text_e, text_chunks)
    prompts["text"] = q.encode(segs, split)
    if args.image:
        image_e = {"dataset": args.dataset, "question": args.question, "query_images": [args.image],
                   "chunks": text_chunks[:1]}
        segs, split = pipeline.build_segments(image_e, text_chunks[:1])
        prompts["image"] = q.encode(segs, split)

    checks = report["checks"]
    if "template" in args.checks:
        checks["template"] = {"text": template_check(q, args.dataset, args.question)}
        if args.image:
            checks["template"]["image"] = template_check(q, args.dataset, args.question, args.image)
        checks["template"]["ok"] = all(v["ok"] for k, v in checks["template"].items() if k != "ok")
    if "segmented" in args.checks:
        checks["segmented"] = {name: segmented_check(q, enc) for name, enc in prompts.items()}
        checks["segmented"]["ok"] = all(v["ok"] for v in checks["segmented"].values())
    if "reference" in args.checks:
        ref = {}
        for name, enc in prompts.items():
            rel = _chunk_spec(text_chunks[:1] if name == "image" else text_chunks)
            arms = {"ans1": Variant("ans1", args.gamma, rows="ans1", form="pos", gate=False).spec(
                enc, rel, 1.0, 0.0, q.n_layers),
                "ans2": Variant("ans2", args.gamma, rows="ans2", form="pos", gate=False).spec(
                    enc, rel, 1.0, 0.0, q.n_layers)}
            ref[name] = reference_check(q, enc, arms, args.steps)
        ref["ok"] = all(v["ok"] for v in ref.values() if isinstance(v, dict))
        checks["reference"] = ref
    if "statistics" in args.checks:
        if args.examples:
            examples = json.loads(Path(args.examples).read_text(encoding="utf-8"))
        else:
            examples = default_examples(args.image)
        rows = statistics_rows(q, examples, args.gamma)
        report["statistics"] = summarize_statistics(rows)
    report["ok"] = all(v.get("ok", True) for v in checks.values())
    text = json.dumps(report, indent=1, ensure_ascii=False, default=float)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

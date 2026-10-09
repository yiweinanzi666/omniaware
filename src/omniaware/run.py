"""Decode direct / static / EAB / gated arms on one dataset split and write per-question JSONL.

Example (score afterwards with omniaware.eval.score_official):
  python -m omniaware.run --dataset okvqa --split dev --calib-source heldout --model 8b --tag okvqa_dev
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from . import data, order as order_mod, paths, pipeline, relevance, rerank, routing
from .qwen_eab import QwenEAB


def load_relevance(examples: list[dict], cache_path: Path) -> dict:
    raw = relevance.load_raw(cache_path) if cache_path.exists() else {}
    todo = [e for e in examples if e["qid"] not in raw]
    if todo:
        t0 = time.time()
        bge = relevance.BGE()
        need_clip = any(c["modality"] == "image" for e in todo for c in e["candidates"])
        clip = relevance.CLIP() if need_clip else None
        raw.update(relevance.raw_scores(todo, bge, clip))
        relevance.save_raw(raw, cache_path)
        del bge, clip
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"[relevance] scored {len(todo)} examples in {time.time() - t0:.1f}s", flush=True)
    return raw


# Loader name -> dataset family used for prompts and generation caps.
DATASET_FAMILY = {"webqa_img": "webqa", "webqa_full": "webqa", "webqa_open_img": "webqa", "webqa_open_full": "webqa",
                  "mmqa_table": "mmqa_table",
                  "videomme": "videomme", "okvqa": "okvqa", "okvqa_dense": "okvqa"}


def select_chunks(e: dict, rel: dict[str, float], k: int, order: str, seed: int, min_per_modality: int = 0,
                  route: dict | None = None, ce: dict | None = None) -> list[dict]:
    """Top-k by calibrated relevance, or by the cross-encoder when ``ce`` = {"logits", "calibration"}."""
    candidates = e["candidates"]
    if route and route["modalities"]:
        missing = set(route["modalities"]) - {c["modality"] for c in candidates}
        if missing:
            raise ValueError(f"{e['qid']}: no retrieval candidates for requested modalities {sorted(missing)}")
        candidates = [c for c in candidates if c["modality"] in route["modalities"]]
    if e["fixed_evidence"]:
        chosen = list(candidates)
    elif ce is not None:
        chosen = rerank.ce_select(candidates, rel, ce["logits"], ce["calibration"])
    else:
        ranked = sorted(candidates, key=lambda c: -rel[c["cid"]])
        picked = []
        if min_per_modality:
            for m in sorted({c["modality"] for c in ranked}):
                picked += [c for c in ranked if c["modality"] == m][:min_per_modality]
            picked = sorted(picked, key=lambda c: -rel[c["cid"]])[:k]
        chosen = picked + [c for c in ranked if c not in picked][:k - len(picked)]
        chosen = sorted(chosen, key=lambda c: -rel[c["cid"]])
    if order == "shuffle":
        chosen = list(chosen)
        random.Random(f"{seed}:{e['qid']}").shuffle(chosen)
    elif order == "reverse":
        chosen = list(reversed(chosen))
    elif order.startswith("perm:"):
        # One fixed permutation of the relevance ranks, identical for every question (order.py).
        permutation = PERMUTATIONS[order[len("perm:"):]]
        if len(chosen) != len(permutation):
            raise ValueError(f"{e['qid']}: permutation needs exactly {len(permutation)} chunks, got {len(chosen)}")
        chosen = [chosen[i] for i in permutation]
    return chosen


PERMUTATIONS = {o["order_id"]: o["permutation"] for o in order_mod.permutation_schedule()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(data.LOADERS))
    ap.add_argument("--model", default="2b", choices=["2b", "8b"])
    ap.add_argument("--split", required=True, choices=["dev", "test"],
                    help="dev: tune here; test: run once with frozen settings (see docs/PROTOCOL.md)")
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--calib-source", choices=["train", "heldout"], default="train",
                    help="train: unlabeled training-split queries (paper); heldout: carve them out of the eval data")
    ap.add_argument("--n-dev", type=int, default=256, help="size of the frozen dev split")
    ap.add_argument("--split-seed", type=int, default=2027)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=None, help="evaluate only a slice of the split (default: all)")
    ap.add_argument("--mmqa-scope", choices=["tableq", "legacy299"], default="tableq")
    ap.add_argument("--video-frames-dir", default=None, help="Video-MME frame directory (default: 8-frame set)")
    ap.add_argument("--subtitle-words", type=int, default=220, help="Video-MME subtitle word cap; 0 keeps all")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-per-modality", type=int, default=0,
                    help="keep the top-m chunks of every modality before global top-k filling")
    ap.add_argument("--mmqa-provided-context", action="store_true",
                    help="MMQA diagnostic: keep all provided table rows as initial context in both branches")
    ap.add_argument("--table-format", choices=["rows", "cells"], default="rows",
                    help="MMQA prompt display; cells labels every value with its column, keeping retrieval text unchanged")
    ap.add_argument("--modality-routing", action="store_true",
                    help="route retrieval by the paper's initial-evidence M_m H_m; cold start keeps joint retrieval")
    ap.add_argument("--first-round-routing", action="store_true",
                    help="cold start: route by M_m H_m over a first joint top-k round, then retrieve top-k "
                         "inside the routed modalities")
    ap.add_argument("--order", default="rank",
                    help="rank | reverse | shuffle (per-question, --seed) | perm:<id> with a fixed schedule id: "
                         + ", ".join(PERMUTATIONS))
    ap.add_argument("--replication-gamma", type=float, default=None,
                    help="also decode the chunk-replication arm: ceil(exp(gamma*U*r)) adjacent copies when "
                         "U > tau_bias (paper: replication baseline, Prop. 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selector", default="relevance", choices=["relevance", "ce", "ce_calibrated"],
                    help="ce: bge-reranker top-5 shown in retriever order; ce_calibrated: top-5 by per-modality "
                         "CE percentile. With a CE selector, EAB's r_j is the per-modality CE percentile.")
    ap.add_argument("--image-mode", default="mix", choices=["mix", "clip", "caption"])
    ap.add_argument("--beta", type=float, default=0.30)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--tau-bias", type=float, default=0.40)
    ap.add_argument("--eab-uncertainty", choices=["pre", "post"], default="pre",
                    help="pre (paper): reuse the retrieval-gate U for EAB; post: recompute U on the packed input")
    ap.add_argument("--answer-style", default="default",
                    choices=["default", "short", "table", "json_list", "strict_letter", "cite"],
                    help="short: one word/phrase; table: MMQA table reasoning; json_list: MMQA JSON array "
                         "(scored as a list); strict_letter: Video-MME single option letter; cite: statements with "
                         "[E<j>] citations for CiteRec (raise --max-new-tokens). Applies to every arm.")
    ap.add_argument("--variants", default="pasta:g1.2:ans2,eab:g1.2:ans2:pos:gate")
    ap.add_argument("--max-pixels", type=int, default=256 * 32 * 32)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the per-dataset generation cap (e.g. 10 to reproduce earlier short-answer runs)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)
    if args.order not in ("rank", "shuffle", "reverse") and args.order.removeprefix("perm:") not in PERMUTATIONS:
        ap.error(f"unknown --order {args.order}")
    if args.mmqa_provided_context and args.dataset != "mmqa_table":
        ap.error("--mmqa-provided-context requires --dataset mmqa_table")
    if args.table_format == "cells" and args.dataset != "mmqa_table":
        ap.error("--table-format cells requires --dataset mmqa_table")
    if args.selector != "relevance" and (args.k != rerank.TOP_K or args.min_per_modality):
        ap.error(f"CE selectors take exactly k={rerank.TOP_K} and no --min-per-modality")
    if args.first_round_routing and args.mmqa_provided_context:
        ap.error("--first-round-routing is the cold-start pathway; provided tables already give initial evidence")

    out_dir = paths.RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    max_new = args.max_new_tokens or pipeline.MAX_NEW_TOKENS[DATASET_FAMILY[args.dataset]]
    config_path = out_dir / f"{args.dataset}_config.json"
    previous = None
    if (out_dir / f"{args.dataset}.jsonl").exists():
        if not config_path.exists():
            ap.error("saved answers have no configuration; use a new --tag")
        previous = json.loads(config_path.read_text())
        saved = dict(previous)
        saved["variants"] = ",".join(v["name"] for v in saved["variants"])
        requested = dict(vars(args), max_new_tokens=max_new)
        changed = [k for k, v in requested.items() if saved.get(k) != v]
        if changed:
            ap.error(f"saved answers use different {', '.join(changed)}; use a new --tag")
    cache_dir = paths.LOCAL_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    loader_kwargs = {}
    if args.dataset == "mmqa_table":
        loader_kwargs["scope"] = args.mmqa_scope
    if args.dataset == "videomme":
        loader_kwargs.update(frames_dir=args.video_frames_dir, max_subtitle_words=args.subtitle_words or None)
    examples = data.LOADERS[args.dataset](**loader_kwargs)
    # Input variants get their own relevance / calibration caches.
    data_tag = "".join(f"_{k}-{Path(str(v)).name if k == 'frames_dir' else v}"
                       for k, v in sorted(loader_kwargs.items()) if v is not None)
    calib_pool = None
    if args.calib_source == "train":
        if args.dataset not in data.TRAIN_LOADERS:
            ap.error(f"{args.dataset} has no training-split loader; use --calib-source heldout")
        calib_pool = data.TRAIN_LOADERS[args.dataset]()
    # Dense candidate files may cover only the intended local slice.  Keep the
    # original OK-VQA universe for split identities instead of reshuffling it.
    split_source = data.LOADERS["okvqa"]() if args.dataset == "okvqa_dense" else examples
    sp = data.split_examples(split_source, args.n_calib, args.n_dev, args.split_seed, calib_pool)
    calib = sp["calib"]
    end = None if args.n_eval is None else args.offset + args.n_eval
    evals = sp[args.split][args.offset:end]
    if args.dataset == "okvqa_dense":
        by_qid = {e["qid"]: e for e in examples}
        missing = [e["qid"] for e in calib + evals if e["qid"] not in by_qid]
        if missing:
            ap.error(f"dense candidates are missing {len(missing)} requested calibration/evaluation questions; "
                     "generate candidates for the original OK-VQA split before running this slice")
        calib = [by_qid[e["qid"]] for e in calib]
        evals = [by_qid[e["qid"]] for e in evals]
    for e in calib + evals:
        if args.table_format == "cells":
            for c in e["candidates"]:
                if c["modality"] == "table":
                    c["prompt_text"] = c["cell_text"]
        if args.mmqa_provided_context:
            e["query_chunks"] = [c for c in e["candidates"] if c["modality"] == "table"]
    assert not {e["qid"] for e in calib} & {e["qid"] for e in evals}, "calibration overlaps evaluation"
    print(f"[data] {args.dataset}: {len(examples)} total, calib {len(calib)} ({args.calib_source}), "
          f"{args.split} {len(evals)}", flush=True)

    raw = load_relevance(calib + evals, cache_dir / f"relevance_{args.dataset}{data_tag}.json")
    cal = relevance.Calibrator.fit(raw, [e["qid"] for e in calib])

    def rel_of(e):
        return {c["cid"]: cal.relevance(raw[e["qid"]][c["cid"]], args.image_mode) for c in e["candidates"]}

    ce_of = ce_rel_of = None
    if args.selector != "relevance":
        ce_table = rerank.ce_logits(calib + evals, cache_dir / f"ce_{args.dataset}{data_tag}.json", paths.RERANKER)
        ce_cal = relevance.Calibrator.fit(
            {q: {c: {f"ce_{m}": v} for c, (m, v) in rows.items()} for q, rows in
             ((e["qid"], {c["cid"]: (c["modality"], ce_table[e["qid"]][c["cid"]]) for c in e["candidates"]})
              for e in calib)}, [e["qid"] for e in calib])
        calibration = {"modalities": {k[3:]: {"count": len(v), "sorted_logits": v}
                                      for k, v in ce_cal.sorted.items()}}

        def ce_of(e):
            return {"logits": ce_table[e["qid"]],
                    "calibration": calibration if args.selector == "ce_calibrated" else None}

        def ce_rel_of(e):
            return {c["cid"]: ce_cal.ecdf(f"ce_{c['modality']}", ce_table[e["qid"]][c["cid"]]) for c in e["candidates"]}

    model_path = paths.QWEN_8B if args.model == "8b" else paths.QWEN_2B
    t0 = time.time()
    qwen = QwenEAB(model_path, max_pixels=args.max_pixels)
    print(f"[model] {args.model} loaded on {qwen.device} in {time.time() - t0:.1f}s, {qwen.n_layers} layers", flush=True)

    # ---- U calibration (frozen on the calibration split, cached per dataset/model/k/order)
    style = data_tag + ("" if args.answer_style == "default" else f"_{args.answer_style}")
    style += f"_mpm{args.min_per_modality}" if args.min_per_modality else ""
    style += "_provided_table" if args.mmqa_provided_context else ""
    style += "_route" if args.modality_routing else ""
    style += "_r1route" if args.first_round_routing else ""
    style += "_cells" if args.table_format == "cells" else ""
    style += f"_px{args.max_pixels}" if args.max_pixels != 256 * 32 * 32 else ""
    style += f"_sel-{args.selector}" if args.selector != "relevance" else ""

    def evidence_segments(e, chunks, rel):
        return pipeline.build_segments(e, chunks, args.answer_style)

    routed = args.modality_routing or args.first_round_routing

    def resolve_route(e, rel, initial, order="rank", seed=0):
        """Initial-evidence route; cold-start queries take the first-round pathway when enabled."""
        if not args.first_round_routing or (initial and initial["modalities"]):
            return initial
        first = select_chunks(e, rel, args.k, order, seed, args.min_per_modality, None)
        enc_1 = qwen.encode(*evidence_segments(e, first, rel))
        _, attn_1, _ = qwen.run(enc_1, None, 1, decode_static=False)
        route = routing.evidence_route(attn_1, enc_1, first)
        route["first_round"] = [c["cid"] for c in first]
        return route
    ucal_path = cache_dir / (f"ucal_{args.dataset}_{args.model}_k{args.k}_{args.image_mode}_n{args.n_calib}"
                             f"_{args.calib_source}{args.split_seed}{style}.json")
    ucal = json.loads(ucal_path.read_text()) if ucal_path.exists() else {}
    t0 = time.time()
    for e in calib:
        if e["qid"] in ucal:
            continue
        rel = rel_of(e)
        enc_d = qwen.encode(*pipeline.build_segments(e, None, args.answer_style))
        _, attn_d, logits_d = qwen.run(enc_d, None, 1, decode_static=False)
        route = routing.initial_route(attn_d, enc_d, e) if args.modality_routing else None
        route = resolve_route(e, rel, route)
        chunks = select_chunks(e, rel, args.k, "rank", 0, args.min_per_modality, route, ce_of and ce_of(e))
        query_ids = ["__query__"] + [c["cid"] for c in e.get("query_chunks", [])]
        pre = pipeline.raw_uncertainty(attn_d, enc_d, query_ids, logits_d, args.beta)
        enc = qwen.encode(*evidence_segments(e, chunks, rel))
        _, attn, logits = qwen.run(enc, None, 1, decode_static=False)
        post = pipeline.raw_uncertainty(attn, enc, [c["cid"] for c in chunks], logits, args.beta)
        ucal[e["qid"]] = {"pre": pre, "post": post}
        if routed:
            ucal[e["qid"]]["route"] = route
        ucal_path.write_text(json.dumps(ucal))
    if time.time() - t0 > 1:
        print(f"[ucal] {len(calib)} calibration examples in {time.time() - t0:.1f}s", flush=True)
    # Raw entropies are reusable across beta values; the cached weighted sum is not.
    for cached in ucal.values():
        for phase in ("pre", "post"):
            entropy = cached[phase]
            entropy["s"] = entropy["h_grd"] + args.beta * entropy["h_sem"]
    norm_pre = pipeline.Norm.fit([ucal[e["qid"]]["pre"]["s"] for e in calib])
    norm_post = pipeline.Norm.fit([ucal[e["qid"]]["post"]["s"] for e in calib])

    variants = [pipeline.parse_variant(v) for v in args.variants.split(",") if v]

    out_path = out_dir / f"{args.dataset}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["qid"] for l in out_path.open()}
    config_path.write_text(json.dumps({
        **vars(args), "norm_pre": vars(norm_pre), "norm_post": vars(norm_post),
        "variants": [vars(v) for v in variants], "model_path": str(model_path),
        "max_new_tokens": max_new,
    }, indent=1))

    t_start = time.time()
    with out_path.open("a") as fh:
        for i, e in enumerate(evals):
            if e["qid"] in done:
                continue
            t0 = time.time()
            rel = rel_of(e)
            rec = {"qid": e["qid"], "dataset": e["dataset"], "qcate": e["qcate"], "question": e["question"],
                   "golds": e["golds"],
                   "query_chunks": [c["cid"] for c in e.get("query_chunks", [])],
                   "options": e.get("options"), "gold_letter": e.get("gold_letter"),
                   "preds": {}}
            # Pre-retrieval uncertainty (paper Eq. 4): prefill without retrieved evidence.
            # Every compared arm is decoded here from the same prompt; no cached baselines.
            enc_d = qwen.encode(*pipeline.build_segments(e, None, args.answer_style))
            pre_info = {}

            def gate_off(attn_d, logits_d):
                query_ids = ["__query__"] + [c["cid"] for c in e.get("query_chunks", [])]
                pre = pipeline.raw_uncertainty(attn_d, enc_d, query_ids, logits_d, args.beta)
                pre_info.update(raw_pre=pre, u_pre=norm_pre(pre["s"]))
                if args.modality_routing:
                    pre_info["route"] = routing.initial_route(attn_d, enc_d, e)
                return True

            res_d, _, _ = qwen.run(enc_d, None, max_new, decode_static=gate_off)
            rec["u_pre"] = pre_info["u_pre"]
            rec["raw_pre"] = pre_info["raw_pre"]
            rec["retrieve"] = pre_info["u_pre"] > args.tau
            rec["preds"]["direct"] = res_d["static"].text
            rec["tokens"] = {"direct": res_d["static"].tokens}

            route = resolve_route(e, rel, pre_info.get("route"), args.order, args.seed)
            chunks = select_chunks(e, rel, args.k, args.order, args.seed, args.min_per_modality, route,
                                   ce_of and ce_of(e))
            eab_rel = ce_rel_of(e) if ce_rel_of else rel
            rec["chunks"] = [c["cid"] for c in chunks]
            rec["rel"] = {c["cid"]: round(eab_rel[c["cid"]], 4) for c in chunks}
            rec["gold_in_topk"] = any(c.get("gold") for c in chunks) if any(
                c.get("gold") is not None for c in e["candidates"]) else None
            if routed:
                rec["route"] = route
            segments, split = evidence_segments(e, chunks, rel)
            enc = qwen.encode(segments, split)
            if args.answer_style == "cite":
                rec["citation_sources"] = pipeline.citation_sources(segments, chunks)
            rel_sel = {c["cid"]: eab_rel[c["cid"]] for c in chunks}
            info = {}

            def make_variants(attn, logits):
                post = pipeline.raw_uncertainty(attn, enc, list(rel_sel), logits, args.beta)
                u = norm_post(post["s"])
                u_eab = rec["u_pre"] if args.eab_uncertainty == "pre" else u
                info.update(raw_post=post, u_post=u, u_eab=u_eab, mass=pipeline.chunk_masses(attn, enc))
                return {v.name: v.spec(enc, rel_sel, u_eab, args.tau_bias, qwen.n_layers) for v in variants}

            res, _, _ = qwen.run(enc, make_variants, max_new, decode_static=True)
            for name, r in res.items():
                rec["preds"][name] = r.text
                rec["tokens"][name] = r.tokens
            arms = ["static"] + [v.name for v in variants]
            if args.replication_gamma is not None:
                counts = order_mod.replication_counts([rel_sel[c["cid"]] for c in chunks], info["u_eab"],
                                                      info["u_eab"] > args.tau_bias, args.replication_gamma)
                rec["replication_counts"] = counts
                if all(n == 1 for n in counts):
                    rec["preds"]["replication"] = rec["preds"]["static"]
                else:
                    copies = [c for c, n in zip(chunks, counts) for _ in range(n)]
                    enc_r = qwen.encode(*evidence_segments(e, copies, rel))
                    res_r, _, _ = qwen.run(enc_r, None, max_new, decode_static=True)
                    rec["preds"]["replication"] = res_r["static"].text
                    rec["tokens"]["replication"] = res_r["static"].tokens
                    rec["replication_prompt_tokens"] = enc_r.length
                arms.append("replication")
            # Gated arms (paper Eq. 4): the direct answer when U_pre <= tau, else the retrieval arm.
            # "gated:static" is the paper's "U_omni, static" row; "gated:<variant>" the full system.
            for name in arms:
                rec["preds"]["gated:" + name] = rec["preds"][name] if rec["retrieve"] else rec["preds"]["direct"]
            rec["u_post"] = info["u_post"]
            rec["u_eab"] = info["u_eab"]
            rec["raw_post"] = info["raw_post"]
            rec["mass"] = {k: round(v, 5) for k, v in info["mass"].items()}
            rec["prompt_tokens"] = enc.length
            rec["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(evals)}] {e['qid'][:12]} {rec['seconds']}s P={enc.length} "
                  f"U_pre={rec['u_pre']:.2f} retrieve={rec['retrieve']}", flush=True)
    print(f"[done] {args.dataset} {len(evals)} in {time.time() - t_start:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

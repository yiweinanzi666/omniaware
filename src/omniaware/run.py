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

import numpy as np
import torch

from . import analysis, data, interventions, order as order_mod, paths, pipeline, relevance, rerank, routing
from .baselines import arms as baseline_arms, seakr, universalrag
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
                  "okvqa": "okvqa", "okvqa_dense": "okvqa", "okvqa_gs": "okvqa",
                  **{name: family for name, (_, family, _) in data.UNIVERSALRAG_TARGETS.items()}}


def select_chunks(e: dict, rel: dict[str, float], k: int, order: str, seed: int, min_per_modality: int = 0,
                  route: dict | None = None, ce: dict | None = None) -> list[dict]:
    """Top-k by calibrated relevance, or by the cross-encoder when ``ce`` = {"logits", "calibration"}.

    A UniversalRAG query route ranks by the suite retriever's own score (``suite_score``) instead of
    our relevance, so that the baseline's candidate order does not depend on Omni-Aware's scores.
    """
    candidates = e["candidates"]
    if route and route["modalities"]:
        missing = set(route["modalities"]) - {c["modality"] for c in candidates}
        if missing:
            raise ValueError(f"{e['qid']}: no retrieval candidates for requested modalities {sorted(missing)}")
        candidates = [c for c in candidates if c["modality"] in route["modalities"]]
    score = rel
    if route and route.get("rule") == "query":
        unscored = [c["cid"] for c in e["candidates"] if "suite_score" not in c]
        if unscored:
            raise ValueError(f"{e['qid']}: UniversalRAG route needs suite_score on every candidate, missing {unscored[:3]}")
        score = {c["cid"]: c["suite_score"] for c in e["candidates"]}
    if e["fixed_evidence"]:
        chosen = list(candidates)
    elif ce is not None:
        chosen = rerank.ce_select(candidates, rel, ce["logits"], ce["calibration"])
    else:
        ranked = sorted(candidates, key=lambda c: -score[c["cid"]])
        picked = []
        if min_per_modality:
            for m in sorted({c["modality"] for c in ranked}):
                picked += [c for c in ranked if c["modality"] == m][:min_per_modality]
            picked = sorted(picked, key=lambda c: -score[c["cid"]])[:k]
        chosen = picked + [c for c in ranked if c not in picked][:k - len(picked)]
        chosen = sorted(chosen, key=lambda c: -score[c["cid"]])
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
# How a run is executed, not what it computes: excluded from the saved configuration and its checks.
EXECUTION_ARGS = {"num_shards", "shard_id", "calibrate_only"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(data.LOADERS))
    ap.add_argument("--model", default="2b", choices=["2b", "8b", "internvl3.5-8b"])
    ap.add_argument("--max-patches", type=int, default=1, help="InternVL: 448px tiles per image (256 tokens each)")
    ap.add_argument("--split", required=True, choices=["dev", "test"],
                    help="dev: tune here; test: run once with frozen settings (see docs/PROTOCOL.md)")
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--calib-source", choices=["train", "heldout"], default="train",
                    help="train: unlabeled training-split queries (paper); heldout: carve them out of the eval data")
    ap.add_argument("--n-dev", type=int, default=256, help="size of the frozen dev split")
    ap.add_argument("--split-seed", type=int, default=2027)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=None, help="evaluate only a slice of the split (default: all)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-per-modality", type=int, default=0,
                    help="keep the top-m chunks of every modality before global top-k filling")
    ap.add_argument("--modality-routing", action="store_true",
                    help="route retrieval by the paper's initial-evidence M_m H_m; cold start keeps joint retrieval")
    ap.add_argument("--route-rule", default="mh",
                    choices=["mh", "mass", "entropy", "retriever", "all", "oracle", "query"],
                    help="routing rule (Table 4): mh = M_m*H_m (paper); mass / entropy = one factor; retriever = best "
                         "calibrated retriever score per modality; all = every modality; oracle = gold modalities; query = UniversalRAG router (--query-routes). "
                         "Attention rules need --modality-routing or --first-round-routing.")
    ap.add_argument("--query-routes", default=None,
                    help="UniversalRAG router output (route_qwen3_vl.py --output) for --route-rule query")
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
    ap.add_argument("--universalrag-top", type=int, default=5,
                    help="UniversalRAG suite: candidates per modality list from the exported rankings")
    ap.add_argument("--universalrag-modalities", nargs="+", default=None,
                    choices=["paragraph", "document", "table", "image", "video", "clip"],
                    help="UniversalRAG suite: modality lists in the candidate pool (default per benchmark)")
    ap.add_argument("--intervention", nargs="+", choices=interventions.KINDS, default=None,
                    help="evidence interventions on eligible questions (retrieval on, EAB active, gold answer in a "
                         "retrieved text chunk); each re-decodes static and every variant")
    ap.add_argument("--relevance-noise", type=float, default=0.0,
                    help="Gaussian sigma added to calibrated relevance (ranking and EAB), clipped to [0, 1]")
    ap.add_argument("--noise-draw", type=int, default=0, help="noise draw index (paper averages three draws)")
    ap.add_argument("--relevance-norm", default="ecdf", choices=["ecdf", "raw", "minmax"],
                    help="r_j and ranking: per-modality ECDF (paper), raw retriever scores, or per-question min-max "
                         "(ablations)")
    ap.add_argument("--norm-from", default=None,
                    help="reuse norm_pre/norm_post from another run's *_config.json (calibration transfer)")
    ap.add_argument("--no-calibration", action="store_true",
                    help="heuristic default statistics: Norm(median 0, IQR 1) and per-question min-max relevance")
    ap.add_argument("--image-mode", default="mix", choices=["mix", "clip", "caption"])
    ap.add_argument("--beta", type=float, default=0.30)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--tau-bias", type=float, default=0.40)
    ap.add_argument("--eab-uncertainty", choices=["pre", "post"], default="pre",
                    help="pre (paper): reuse the retrieval-gate U for EAB; post: recompute U on the packed input")
    ap.add_argument("--answer-style", default="default", choices=["default", "short", "cite"],
                    help="short: one word/phrase; cite: statements with [E<j>] citations for the citation metrics "
                         "(raise --max-new-tokens). Applies to every arm.")
    ap.add_argument("--variants", default="eab:g1.2:ans1:pos:gate",
                    help="comma-separated EAB variants (pipeline.parse_variant); paper default eab:g1.2:ans1:pos:gate")
    ap.add_argument("--probe-uncertainty", action="store_true",
                    help="cold start (no supplied context): a probe round packs the joint top-k by calibrated "
                         "relevance; U and, with --first-round-routing, the M_m*H_m route come from its prefill")
    ap.add_argument("--u-combine", choices=["sum", "zsum"], default="sum",
                    help="sum: H_grd + beta*H_pred (paper); zsum: each term standardised on calibration first")
    ap.add_argument("--attn-layer", choices=["last", "mid"], default="last",
                    help="decoder layer whose head-averaged attention gives H_grd, M_m*H_m and ARD")
    ap.add_argument("--max-pixels", type=int, default=256 * 32 * 32)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the per-dataset generation cap (e.g. 10 to reproduce earlier short-answer runs)")
    ap.add_argument("--seakr-samples", type=int, default=0,
                    help="SeaKR baseline: samples per input for the eigen score (upstream 20; 0 = off)")
    ap.add_argument("--seakr-max-new-tokens", type=int, default=100, help="SeaKR sampling length (upstream 100)")
    ap.add_argument("--seakr-threshold", type=float, default=None,
                    help="fixed SeaKR eigen threshold; default: matched on the calibration questions to U's "
                         "retrieval rate there")
    ap.add_argument("--fitm", action="store_true", help="Found-in-the-Middle baseline arm on the static input")
    ap.add_argument("--m2rag", action="store_true",
                    help="M2-RAG baseline arm (official single-stage summarizer) on the same chunks")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="multi-GPU: split the evaluation questions round-robin over this many processes "
                         "(scripts/launch.py starts them, one per GPU, after a --calibrate-only pass)")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--calibrate-only", action="store_true",
                    help="fill every cache (relevance, CE, U calibration, SeaKR) and write the config, then stop")
    ap.add_argument("--tag", required=True)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.order not in ("rank", "shuffle", "reverse") and args.order.removeprefix("perm:") not in PERMUTATIONS:
        ap.error(f"unknown --order {args.order}")
    if args.selector != "relevance" and (args.k != rerank.TOP_K or args.min_per_modality):
        ap.error(f"CE selectors take exactly k={rerank.TOP_K} and no --min-per-modality")
    if args.no_calibration:
        args.relevance_norm = "minmax"
    if (args.route_rule == "query") != (args.query_routes is not None):
        ap.error("--route-rule query and --query-routes go together")
    if args.m2rag and DATASET_FAMILY[args.dataset] == "videorag":
        ap.error("M2-RAG accepts no video evidence (an em dash in the paper's main table)")
    if (args.fitm or args.m2rag) and args.model == "internvl3.5-8b":
        ap.error("the Found-in-the-Middle and M2-RAG adapters support the Qwen3-VL backbones only")
    if args.route_rule in ("mass", "entropy") and not (args.modality_routing or args.first_round_routing):
        ap.error("--route-rule mass/entropy ablates the attention route; add --first-round-routing or --modality-routing")

    if not 0 <= args.shard_id < args.num_shards:
        ap.error("--shard-id must lie in [0, --num-shards)")
    out_dir = paths.RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    max_new = args.max_new_tokens or pipeline.MAX_NEW_TOKENS[DATASET_FAMILY[args.dataset]]
    config_path = out_dir / f"{args.dataset}_config.json"
    out_path = out_dir / (f"{args.dataset}.jsonl" if args.num_shards == 1
                          else f"{args.dataset}.shard{args.shard_id:02d}-of-{args.num_shards:02d}.jsonl")
    previous = None
    if out_path.exists() or (out_dir / f"{args.dataset}.jsonl").exists():
        if not config_path.exists():
            ap.error("saved answers have no configuration; use a new --tag")
        previous = json.loads(config_path.read_text())
        saved = dict(previous)
        saved["variants"] = ",".join(v["name"] for v in saved["variants"])
        requested = {k: v for k, v in dict(vars(args), max_new_tokens=max_new).items() if k not in EXECUTION_ARGS}
        changed = [k for k, v in requested.items() if saved.get(k) != v]
        if changed:
            ap.error(f"saved answers use different {', '.join(changed)}; use a new --tag")
    cache_dir = paths.LOCAL_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    loader_kwargs = {}
    if args.dataset in data.UNIVERSALRAG_TARGETS:
        loader_kwargs.update(top=args.universalrag_top,
                             modalities=tuple(args.universalrag_modalities) if args.universalrag_modalities else None)
    examples = data.LOADERS[args.dataset](**loader_kwargs)
    # Input variants get their own relevance / calibration caches.
    data_tag = "".join(f"_{k}-{v}" for k, v in sorted(loader_kwargs.items()) if v is not None)
    calib_pool = None
    if args.calib_source == "train":
        if args.dataset not in data.TRAIN_LOADERS:
            ap.error(f"{args.dataset} has no training-split loader; use --calib-source heldout")
        calib_pool = data.TRAIN_LOADERS[args.dataset]()
    # Dense candidate files may cover only the intended local slice.  Keep the
    # original OK-VQA universe for split identities instead of reshuffling it.
    retrieved_okvqa = args.dataset in ("okvqa_dense", "okvqa_gs")
    split_source = data.load_okvqa(candidates_path=False) if retrieved_okvqa else examples
    sp = data.split_examples(split_source, args.n_calib, args.n_dev, args.split_seed, calib_pool)
    calib = sp["calib"]
    end = None if args.n_eval is None else args.offset + args.n_eval
    evals = sp[args.split][args.offset:end]
    if retrieved_okvqa:
        by_qid = {e["qid"]: e for e in examples}
        missing = [e["qid"] for e in calib + evals if e["qid"] not in by_qid]
        if missing:
            ap.error(f"dense candidates are missing {len(missing)} requested calibration/evaluation questions; "
                     "generate candidates for the original OK-VQA split before running this slice")
        calib = [by_qid[e["qid"]] for e in calib]
        evals = [by_qid[e["qid"]] for e in evals]
    assert not {e["qid"] for e in calib} & {e["qid"] for e in evals}, "calibration overlaps evaluation"
    print(f"[data] {args.dataset}: {len(examples)} total, calib {len(calib)} ({args.calib_source}), "
          f"{args.split} {len(evals)}", flush=True)

    raw = load_relevance(calib + evals, cache_dir / f"relevance_{args.dataset}{data_tag}.json")
    cal = relevance.Calibrator.fit(raw, [e["qid"] for e in calib])

    def rel_of(e):
        if args.relevance_norm == "ecdf":
            return {c["cid"]: cal.relevance(raw[e["qid"]][c["cid"]], args.image_mode) for c in e["candidates"]}
        scores = {c["cid"]: relevance.Calibrator.raw(raw[e["qid"]][c["cid"]], args.image_mode) for c in e["candidates"]}
        if args.relevance_norm == "raw":
            return scores
        lo, hi = min(scores.values(), default=0.0), max(scores.values(), default=1.0)
        return {c: (s - lo) / (hi - lo) if hi > lo else 1.0 for c, s in scores.items()}

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

    model_path = {"8b": paths.QWEN_8B, "2b": paths.QWEN_2B, "internvl3.5-8b": paths.INTERNVL_8B}[args.model]
    t0 = time.time()
    qwen = QwenEAB(model_path, max_pixels=args.max_pixels, max_patches=args.max_patches)
    qwen.capture_at = qwen.n_layers // 2 if args.attn_layer == "mid" else -1
    print(f"[model] {args.model} loaded on {qwen.device} in {time.time() - t0:.1f}s, {qwen.n_layers} layers", flush=True)

    # ---- U calibration (frozen on the calibration split, cached per dataset/model/k/order)
    style = data_tag + ("" if args.answer_style == "default" else f"_{args.answer_style}")
    style += f"_mpm{args.min_per_modality}" if args.min_per_modality else ""
    style += "_route" if args.modality_routing else ""
    style += "_r1route" if args.first_round_routing else ""
    style += "_probe" if args.probe_uncertainty else ""
    style += "_attn-mid" if args.attn_layer == "mid" else ""
    style += f"_rule-{args.route_rule}" if args.route_rule != "mh" else ""
    style += f"_px{args.max_pixels}" if args.max_pixels != 256 * 32 * 32 else ""
    style += f"_patches{args.max_patches}" if args.max_patches != 1 else ""
    style += f"_sel-{args.selector}" if args.selector != "relevance" else ""
    style += f"_rel-{args.relevance_norm}" if args.relevance_norm != "ecdf" else ""

    def evidence_segments(e, chunks, rel):
        return pipeline.build_segments(e, chunks, args.answer_style)

    def variant_specs(e, enc, rel_sel, u_eab):
        context_ids = ["__query__"] if e["query_images"] else []
        specs = {}
        for v in variants:
            rel_v = rel_sel
            if v.relevance == "random":
                rng = random.Random(f"random-relevance:{args.seed}:{e['qid']}")
                rel_v = {c: rng.random() for c in rel_sel}
            if v.context:
                rel_v = {**rel_v, **{c: 1.0 for c in context_ids}}
            specs[v.name] = v.spec(enc, rel_v, u_eab, args.tau_bias, qwen.n_layers)
        return specs

    def intervene(e, rec, chunks, rel_all, enc, info):
        """Paper Table `causal`: re-decode static and every variant on perturbed evidence, keeping the
        retrieval decision and the uncertainty EAB uses fixed."""
        golds = e["golds"]
        rel_sel = {c["cid"]: rel_all[c["cid"]] for c in chunks}
        target = interventions.find_target(chunks, rel_sel, golds)
        eligible = bool(rec["retrieve"] and any(info["eab_active"].values()) and target is not None)
        out = {"target": target and target["cid"], "eligible": eligible}
        if not eligible:
            return out
        gold = interventions.answer_span(target["text"], golds)[2]
        entity = interventions.pick_substitute(e["qid"], golds, gold, substitute_pool, iv_nlp)
        out.update(gold_in_target=gold, entity=entity)
        for kind in args.intervention:
            chunks_x, rel_x, enc_x, extra = chunks, rel_sel, None, {}
            if kind == "mask":
                enc_x = interventions.mask_tokens(enc, target["cid"], pad_id)
            elif kind == "rerun":
                enc_x = enc
            elif kind == "distractors":
                swapped = interventions.distractors(e, chunks, rel_all, target["cid"], golds)
                if swapped is None:
                    out[kind] = None
                    continue
                chunks_x, extra["distractors"] = swapped
                rel_x = {c["cid"]: rel_all[c["cid"]] for c in chunks_x}
            elif entity is None:
                out[kind] = None
                continue
            elif kind == "cf":
                chunks_x = [interventions.counterfactual(c, golds, entity) if c is target else c for c in chunks]
            elif kind == "random_edit":
                edited = interventions.random_edit(target, golds, entity, f"random-edit:{e['qid']}")
                if edited is None:
                    out[kind] = None
                    continue
                chunks_x = [edited if c is target else c for c in chunks]
            else:
                injected = interventions.inject(e, chunks, rel_all, golds, entity, kind, args.order)
                if injected is None:
                    out[kind] = None
                    continue
                chunks_x, rel_x = injected
            segments_x = None
            if enc_x is None:
                segments_x, split_x = evidence_segments(e, chunks_x, rel_x)
                enc_x = qwen.encode(segments_x, split_x)
            res_x, _, _ = qwen.run(enc_x, lambda a, lg: variant_specs(e, enc_x, rel_x, info["u_eab"]), max_new,
                                   decode_static=True)
            entry = {"preds": {n: r.text for n, r in res_x.items()}, "chunks": [c["cid"] for c in chunks_x], **extra}
            if args.answer_style == "cite" and segments_x is not None:
                entry["citation_sources"] = pipeline.citation_sources(segments_x, chunks_x)
            out[kind] = entry
        return out

    routed = (args.modality_routing or args.first_round_routing
              or args.route_rule in ("retriever", "all", "oracle", "query"))
    query_routes = universalrag.load_routes(args.query_routes) if args.query_routes else None

    def cold(e):
        return not e["query_images"]

    def probe(e, rel):
        """Cold-start probe round (paper Sec. 4.2-4.3): the joint top-k by calibrated relevance in rank
        order, one zero-bias prefill. H_grd over these spans makes U informative without supplied context,
        and the same attention gives the M_m*H_m route."""
        first = select_chunks(e, rel, args.k, "rank", 0, args.min_per_modality, None)
        enc_1 = qwen.encode(*evidence_segments(e, first, rel))
        _, attn_1, logits_1 = qwen.run(enc_1, None, 1, decode_static=False)
        return {"chunks": first, "enc": enc_1, "attn": attn_1, "seconds": qwen.last_prefill_seconds,
                "raw": pipeline.raw_uncertainty(attn_1, enc_1, [c["cid"] for c in first], logits_1, args.beta)}

    def resolve_route(e, rel, initial, calibrating=False, probe_out=None):
        """Initial-evidence route; cold-start queries take the probe-round pathway when enabled.
        Non-attention rules (retriever, all, oracle) replace the attention route entirely."""
        if args.route_rule == "oracle":
            return {"mode": "oracle", "rule": "oracle", "modalities": sorted(e["gold_modalities"])}
        if args.route_rule == "all":
            return {"mode": "all", "rule": "all", "modalities": []}
        if args.route_rule == "retriever":
            return routing.retriever_route(e["candidates"], rel)
        if args.route_rule == "query":
            index = universalrag.upstream_index(e)
            if index not in query_routes:
                if calibrating:   # calibration questions (e.g. training split) need no router output
                    return None
                raise KeyError(f"no UniversalRAG route for {index} in {args.query_routes}")
            return universalrag.to_route(query_routes[index], {c["modality"] for c in e["candidates"]})
        if not args.first_round_routing or (initial and initial["modalities"]):
            return initial
        p = probe_out or probe(e, rel)
        route = routing.evidence_route(p["attn"], p["enc"], p["chunks"], rule=args.route_rule)
        route["first_round"] = [c["cid"] for c in p["chunks"]]
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
        route = routing.initial_route(attn_d, enc_d, e, args.route_rule) if args.modality_routing else None
        p = probe(e, rel) if args.probe_uncertainty and cold(e) else None
        route = resolve_route(e, rel, route, calibrating=True, probe_out=p)
        chunks = select_chunks(e, rel, args.k, "rank", 0, args.min_per_modality, route, ce_of and ce_of(e))
        pre = pipeline.raw_uncertainty(attn_d, enc_d, ["__query__"], logits_d, args.beta)
        enc = qwen.encode(*evidence_segments(e, chunks, rel))
        _, attn, logits = qwen.run(enc, None, 1, decode_static=False)
        post = pipeline.raw_uncertainty(attn, enc, [c["cid"] for c in chunks], logits, args.beta)
        ucal[e["qid"]] = {"pre": pre, "post": post}
        if p:
            ucal[e["qid"]]["probe"] = p["raw"]
        if routed:
            ucal[e["qid"]]["route"] = route
        ucal_path.write_text(json.dumps(ucal))
    if time.time() - t0 > 1:
        print(f"[ucal] {len(calib)} calibration examples in {time.time() - t0:.1f}s", flush=True)
    # Raw entropies are cached; the combination (beta, sum/zsum) and Norm are fitted here.
    calibrators = {}
    for phase in ("pre", "post", "probe"):
        raws = [ucal[e["qid"]][phase] for e in calib if phase in ucal[e["qid"]]]
        if raws:
            comb = pipeline.Combiner.fit(raws, args.beta, args.u_combine)
            calibrators[phase] = (comb, pipeline.Norm.fit([comb(r) for r in raws]))
    if args.norm_from:
        source = json.loads(Path(args.norm_from).read_text())
        for phase in ("pre", "post", "probe"):
            if f"norm_{phase}" in source and source[f"norm_{phase}"]:
                comb = (pipeline.Combiner.from_json(source[f"comb_{phase}"]) if source.get(f"comb_{phase}")
                        else pipeline.Combiner(args.beta))
                calibrators[phase] = (comb, pipeline.Norm(**source[f"norm_{phase}"]))
    if args.no_calibration:
        calibrators = {phase: (pipeline.Combiner(args.beta), pipeline.Norm(0.0, 1.0))
                       for phase in ("pre", "post", "probe")}
    norm_pre, norm_post = calibrators["pre"][1], calibrators["post"][1]

    def u_of(phase, raw):
        comb, norm = calibrators[phase]
        return norm(comb(raw))

    def u_gate_cal(cached):
        return u_of("probe", cached["probe"]) if "probe" in cached else u_of("pre", cached["pre"])
    # Trigger ablations (Table 3): next-token entropy alone. 'nte' matches U's retrieval rate on the
    # calibration questions with its own threshold (a per-format threshold); 'nte_shared' normalises
    # beta*H_pred like U and uses the shared tau.
    u_rate = sum(u_gate_cal(ucal[e["qid"]]) > args.tau for e in calib) / max(len(calib), 1)
    h_cal = [ucal[e["qid"]]["pre"]["h_sem"] for e in calib]
    nte_threshold = float(np.quantile(h_cal, 1 - u_rate)) if h_cal else float("inf")
    nte_norm = pipeline.Norm.fit([args.beta * h for h in h_cal])

    variants = [pipeline.parse_variant(v) for v in args.variants.split(",") if v]

    seakr_threshold, seakr_missing_calibration = None, None
    if args.seakr_samples:
        scache_path = cache_dir / (f"seakr_{args.dataset}_{args.model}_n{args.seakr_samples}_m{args.seakr_max_new_tokens}"
                                   f"_c{args.n_calib}_{args.calib_source}{args.split_seed}{style}.json")
        scache = json.loads(scache_path.read_text()) if scache_path.exists() else {}
        for e in calib:
            if e["qid"] not in scache:
                enc_d = qwen.encode(*pipeline.build_segments(e, None, args.answer_style))
                scache[e["qid"]] = seakr.score_input(qwen, enc_d, args.seakr_samples, args.seakr_max_new_tokens)
                scache_path.write_text(json.dumps(scache))
        calibration_scores = [scache[e["qid"]] for e in calib]
        seakr_missing_calibration = sum(seakr.is_missing(s) for s in calibration_scores)
        seakr_threshold = (args.seakr_threshold if args.seakr_threshold is not None
                           else seakr.matched_threshold(calibration_scores, u_rate))
        print(f"[seakr] threshold {seakr_threshold:.4f} (U retrieval rate on calibration {u_rate:.3f}; "
              f"{seakr_missing_calibration} of {len(calib)} calibration questions have no eigen score)", flush=True)

    substitute_pool, iv_nlp, pad_id = None, None, None
    if args.intervention:
        try:
            import spacy
            iv_nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            print("[interventions] spaCy en_core_web_sm unavailable: entity types from surface rules only", flush=True)
        substitute_pool = interventions.substitute_pool(evals, iv_nlp)
        if "mask" in args.intervention:
            pad_id = qwen.tok.pad_token_id if qwen.tok.pad_token_id is not None else qwen.tok.eos_token_id

    done = set()
    if out_path.exists():
        done = {json.loads(l)["qid"] for l in out_path.open()}
    config = json.dumps({
        **{k: v for k, v in vars(args).items() if k not in EXECUTION_ARGS},
        "norm_pre": vars(norm_pre), "norm_post": vars(norm_post),
        "norm_probe": vars(calibrators["probe"][1]) if "probe" in calibrators else None,
        **{f"comb_{ph}": calibrators[ph][0].to_json() for ph in calibrators},
        "variants": [vars(v) for v in variants], "model_path": str(model_path),
        "max_new_tokens": max_new, "seakr_threshold_used": seakr_threshold,
        "seakr_missing_calibration": seakr_missing_calibration,
        "intervention_entity_types": ("spacy+rules" if iv_nlp else "rules") if args.intervention else None,
        "nte_threshold": nte_threshold, "nte_norm": vars(nte_norm), "u_retrieval_rate_calibration": u_rate,
        "evaluation_questions": len(evals),
    }, indent=1)
    # Shards write the same configuration; replace atomically so readers never see a partial file.
    tmp = config_path.with_name(f".{config_path.name}.{args.shard_id}.tmp")
    tmp.write_text(config)
    tmp.replace(config_path)
    if args.calibrate_only:
        print(f"[calibrate-only] caches and {config_path} ready", flush=True)
        return

    t_start = time.time()
    tallies = {"universalrag_fallback": 0, "seakr_missing": 0}   # questions in this process only
    with out_path.open("a") as fh:
        for i, e in enumerate(evals):
            if i % args.num_shards != args.shard_id or e["qid"] in done:
                continue
            t0 = time.time()
            rel = rel_of(e)
            if args.relevance_noise:
                rel = interventions.relevance_noise(rel, args.relevance_noise,
                                                    f"noise:{args.relevance_noise}:{args.noise_draw}:{e['qid']}")
            rec = {"qid": e["qid"], "dataset": e["dataset"], "qcate": e["qcate"], "question": e["question"],
                   "golds": e["golds"],
                   "preds": {}}
            # Pre-retrieval uncertainty (paper Eq. 4): prefill without retrieved evidence.
            # Every compared arm is decoded here from the same prompt; no cached baselines.
            enc_d = qwen.encode(*pipeline.build_segments(e, None, args.answer_style))
            pre_info = {}

            def gate_off(attn_d, logits_d):
                pre = pipeline.raw_uncertainty(attn_d, enc_d, ["__query__"], logits_d, args.beta)
                pre_info.update(raw_pre=pre, u_pre=u_of("pre", pre))
                if args.modality_routing:
                    pre_info["route"] = routing.initial_route(attn_d, enc_d, e, args.route_rule)
                return True

            res_d, _, _ = qwen.run(enc_d, None, max_new, decode_static=gate_off)
            direct_prefill = qwen.last_prefill_seconds
            rec["u_pre"] = pre_info["u_pre"]
            rec["raw_pre"] = pre_info["raw_pre"]
            probe_out = probe(e, rel) if args.probe_uncertainty and cold(e) else None
            if probe_out:
                rec["raw_probe"] = probe_out["raw"]
                rec["u_probe"] = u_of("probe", probe_out["raw"])
                rec["probe_chunks"] = [c["cid"] for c in probe_out["chunks"]]
            # The gate reads U on the evidence spans in the input: supplied context, or the probe round.
            rec["u_gate"] = rec.get("u_probe", rec["u_pre"])
            rec["retrieve"] = rec["u_gate"] > args.tau
            rec["preds"]["direct"] = res_d["static"].text
            rec["tokens"] = {"direct": res_d["static"].tokens}

            route = resolve_route(e, rel, pre_info.get("route"), probe_out=probe_out)
            chunks = select_chunks(e, rel, args.k, args.order, args.seed, args.min_per_modality, route,
                                   ce_of and ce_of(e))
            # Joint input (D4): the same top-k over all candidates, with no probe and no route. The
            # external baselines (CE, FiTM, M2-RAG, SeaKR) read this input; the Omni-Aware arms keep `chunks`.
            joint = select_chunks(e, rel, args.k, args.order, args.seed, args.min_per_modality, None,
                                  ce_of and ce_of(e))
            rec["chunks_joint"] = [c["cid"] for c in joint]
            eab_rel = ce_rel_of(e) if ce_rel_of else rel
            if ce_rel_of and args.relevance_noise:
                eab_rel = interventions.relevance_noise(eab_rel, args.relevance_noise,
                                                        f"noise-ce:{args.relevance_noise}:{args.noise_draw}:{e['qid']}")
            rec["chunks"] = [c["cid"] for c in chunks]
            rec["rel"] = {c["cid"]: round(eab_rel[c["cid"]], 4) for c in chunks}
            rec["chunk_gold"] = [c.get("gold") for c in chunks]
            rec["chunk_modalities"] = [c["modality"] for c in chunks]
            # whether the chunk text states a gold answer (error analysis: biased chunk without the answer)
            rec["chunk_has_answer"] = [bool(interventions.answer_span(f"{c.get('title') or ''} {c.get('text') or ''}",
                                                                      e["golds"])) for c in chunks]
            rec["gold_in_topk"] = any(c.get("gold") for c in chunks) if any(
                c.get("gold") is not None for c in e["candidates"]) else None
            if routed:
                rec["route"] = route
            if "gold_modalities" in e:
                rec["gold_modalities"] = e["gold_modalities"]
            segments, split = evidence_segments(e, chunks, rel)
            enc = qwen.encode(segments, split)
            if args.answer_style == "cite":
                rec["citation_sources"] = pipeline.citation_sources(segments, chunks)
            rel_sel = {c["cid"]: eab_rel[c["cid"]] for c in chunks}
            info = {}

            def make_variants(attn, logits):
                post = pipeline.raw_uncertainty(attn, enc, list(rel_sel), logits, args.beta)
                u = u_of("post", post)
                u_eab = rec["u_gate"] if args.eab_uncertainty == "pre" else u
                info.update(raw_post=post, u_post=u, u_eab=u_eab, mass=pipeline.chunk_masses(attn, enc))
                specs = variant_specs(e, enc, rel_sel, u_eab)
                info["eab_active"] = {name: s is not None for name, s in specs.items()}
                return specs

            res, _, _ = qwen.run(enc, make_variants, max_new, decode_static=True)
            rec["prefill_seconds"] = {"direct": direct_prefill, "evidence": qwen.last_prefill_seconds,
                                      **({"probe": probe_out["seconds"]} if probe_out else {})}
            rec["arm_seconds"] = {"direct": res_d["static"].seconds}
            rec["ard"] = {}
            for name, r in res.items():
                rec["preds"][name] = r.text
                rec["tokens"][name] = r.tokens
                rec["arm_seconds"][name] = r.seconds
                if r.attention is not None:
                    rec["ard"][name] = analysis.ard(pipeline.chunk_mass(r.attention, {c["cid"]: enc.spans[c["cid"]]
                                                                                    for c in chunks}), rel_sel)
            # Static joint (paper: "Static concatenation"): the same decoding on the joint input, with no probe
            # and no route. Reused from the routed decoding when both inputs are the same chunks.
            if rec["chunks_joint"] == rec["chunks"]:
                enc_j, res_j = enc, res
            else:
                enc_j = qwen.encode(*evidence_segments(e, joint, rel))
                res_j, _, _ = qwen.run(enc_j, None, max_new, decode_static=True)
            rec["preds"]["static_joint"] = res_j["static"].text
            rec["tokens"]["static_joint"] = res_j["static"].tokens
            rec["arm_seconds"]["static_joint"] = res_j["static"].seconds
            arms = ["static"] + [v.name for v in variants]
            if args.route_rule == "query":
                # UniversalRAG baseline: its router decides both whether and where to retrieve. A route whose
                # labels are all absent from the corpus does not retrieve (fallback, counted).
                rec["preds"]["universalrag"] = rec["preds"]["static"] if route["retrieve"] else rec["preds"]["direct"]
                rec["universalrag_fallback"] = int(route["fallback"])
                tallies["universalrag_fallback"] += rec["universalrag_fallback"]
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
            # Baseline arms read the joint input (D4). Wall time: everything each method runs itself.
            joint_ids = rec["chunks_joint"]
            rec["baseline_seconds"] = {}
            if args.fitm:
                t_b = time.perf_counter()
                rec["preds"]["fitm"], rec["fitm"] = baseline_arms.fitm_answer(
                    qwen, enc_j, joint_ids, max_new, static_text=rec["preds"]["static_joint"])
                rec["baseline_seconds"]["fitm"] = time.perf_counter() - t_b
            if args.m2rag:
                t_b = time.perf_counter()
                rec["preds"]["m2rag"], rec["m2rag"] = baseline_arms.m2rag_answer(
                    qwen, e, joint, rel, max_new)
                rec["baseline_seconds"]["m2rag"] = time.perf_counter() - t_b
            if args.seakr_samples:
                t_b = time.perf_counter()
                direct_eigen = seakr.score_input(qwen, enc_d, args.seakr_samples, args.seakr_max_new_tokens)
                # A missing eigen score (None) does not retrieve; it is recorded, not treated as infinity.
                missing = seakr.is_missing(direct_eigen)
                info_s = {"direct_eigen": direct_eigen, "missing": missing,
                          "retrieve": not missing and direct_eigen > seakr_threshold}
                tallies["seakr_missing"] += missing
                rec["preds"]["seakr_trigger:static"] = (rec["preds"]["static"] if info_s["retrieve"]
                                                        else rec["preds"]["direct"])
                if info_s["retrieve"]:
                    doc_scores, doc_answers = {}, {}
                    for c in joint:
                        enc_c = qwen.encode(*evidence_segments(e, [c], rel))
                        doc_answers[c["cid"]] = qwen.run(enc_c, None, max_new, decode_static=True)[0]["static"].text
                        doc_scores[c["cid"]] = seakr.score_input(qwen, enc_c, args.seakr_samples,
                                                                 args.seakr_max_new_tokens)
                    best = seakr.choose_document(doc_scores, doc_answers)
                    info_s.update(doc_eigen=doc_scores, chosen=best,
                                  doc_missing=sum(seakr.is_missing(s) for s in doc_scores.values()))
                    rec["preds"]["seakr"] = doc_answers[best] if best else rec["preds"]["direct"]
                else:
                    rec["preds"]["seakr"] = rec["preds"]["direct"]
                rec["seakr"] = info_s
                rec["baseline_seconds"]["seakr"] = time.perf_counter() - t_b
            # Gated arms (paper Eq. 4): the direct answer when U_pre <= tau, else the retrieval arm.
            # "gated:static" is the paper's "U_omni, static" row; "gated:<variant>" the full system.
            # Only the Omni-Aware arms are gated; the external baselines are not.
            for name in arms:
                rec["preds"]["gated:" + name] = rec["preds"][name] if rec["retrieve"] else rec["preds"]["direct"]
            h_sem = rec["raw_pre"]["h_sem"]
            rec["preds"]["nte_trigger:static"] = rec["preds"]["static"] if h_sem > nte_threshold else rec["preds"]["direct"]
            rec["preds"]["nte_shared:static"] = (rec["preds"]["static"] if nte_norm(args.beta * h_sem) > args.tau
                                                 else rec["preds"]["direct"])
            rec["eab_active"] = info["eab_active"]
            rec["u_post"] = info["u_post"]
            rec["u_eab"] = info["u_eab"]
            rec["raw_post"] = info["raw_post"]
            rec["mass"] = {k: round(v, 5) for k, v in info["mass"].items()}
            rec["prompt_tokens"] = enc.length
            rec["span_tokens"] = {c["cid"]: [c["modality"], enc.spans[c["cid"]][1] - enc.spans[c["cid"]][0]]
                                  for c in chunks}
            if args.intervention:
                rec["interventions"] = intervene(e, rec, chunks, eab_rel, enc, info)
            rec["seconds"] = round(time.time() - t0, 2)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(evals)}] {e['qid'][:12]} {rec['seconds']}s P={enc.length} "
                  f"U_pre={rec['u_pre']:.2f} retrieve={rec['retrieve']}", flush=True)
    notes = []
    if args.route_rule == "query":
        notes.append(f"universalrag_fallback {tallies['universalrag_fallback']}")
    if args.seakr_samples:
        notes.append(f"seakr_missing {tallies['seakr_missing']}")
    print(f"[done] {args.dataset} {len(evals)} in {time.time() - t_start:.1f}s -> {out_path}"
          + "".join(f"; {note}" for note in notes), flush=True)


if __name__ == "__main__":
    main()

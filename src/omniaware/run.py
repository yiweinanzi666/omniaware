"""Decode direct / static / EAB / gated arms on one dataset split and write per-question JSONL.

Example (score afterwards with omniaware.eval.score_official):
  python -m omniaware.run --dataset okvqa --split dev --calib-source heldout --model 8b --tag okvqa_dev
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import (analysis, answers, arbitration, data, interventions, order as order_mod, paths, pipeline, prompts,
               relevance, rerank, routing, runio, signals)
from .actions import DIRECT, V2_ACTIONS
from .baselines import arms as baseline_arms, seakr, universalrag
from .qwen_eab import QwenEAB


def relevance_fingerprint(e: dict) -> str:
    """Cache key of one question's raw retriever scores (question, candidates, encoder paths)."""
    return runio.fingerprint(e, {"encoders": runio.encoder_identity()})


def load_relevance(examples: list[dict], cache_path: Path, write: bool = True) -> dict:
    """qid -> cid -> raw channel scores. A cached question is reused only if its fingerprint matches; stale or
    missing questions are recomputed and saved atomically. With write=False (sharded workers) they are an error."""
    cached = relevance.load_raw(cache_path) if cache_path.exists() else {}
    raw, todo = {}, []
    for e in examples:
        fp = relevance_fingerprint(e)
        entry = cached.get(e["qid"])
        if isinstance(entry, dict) and entry.get("fingerprint") == fp:
            raw[e["qid"]] = entry["scores"]
        else:
            todo.append((e, fp))
    if todo and not write:
        raise SystemExit(f"{len(todo)} relevance scores are missing or stale in {cache_path}; "
                         "run --calibrate-only first")
    if todo:
        t0 = time.time()
        todo_examples = [e for e, _ in todo]
        bge = relevance.BGE()
        # CLIP scores images and the mean frame of every video or clip (paper: the video channels are the
        # transcript BGE score and the CLIP score of the mean frame embedding).
        need_clip = any(c["modality"] in ("image", "video", "clip") for e in todo_examples for c in e["candidates"])
        clip = relevance.CLIP() if need_clip else None
        fresh = relevance.raw_scores(todo_examples, bge, clip)
        raw.update(fresh)
        relevance.save_raw({e["qid"]: {"fingerprint": fp, "scores": fresh[e["qid"]]} for e, fp in todo}, cache_path)
        del bge, clip
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"[relevance] scored {len(todo)} examples in {time.time() - t0:.1f}s", flush=True)
    return raw


# Loader name -> dataset family used for prompts and generation caps.
DATASET_FAMILY = {"webqa_img": "webqa", "webqa_full": "webqa", "webqa_open_img": "webqa", "webqa_open_full": "webqa",
                  "okvqa": "okvqa", "okvqa_dense": "okvqa", "okvqa_gs": "okvqa", "okvqa_gs_full": "okvqa",
                  **{name: family for name, (_, family, _) in data.UNIVERSALRAG_TARGETS.items()}}


def replication_copies(chunks: list[dict], counts: list[int]) -> list[dict]:
    """Adjacent copies of each chunk (replication arm). Each copy gets its own id ``<cid>#<k>``, so every copy is
    its own [E<j>] source in the cite prompt (pipeline.citation_sources) and its own span."""
    return [{**c, "cid": f"{c['cid']}#{k}"} for c, n in zip(chunks, counts) for k in range(n)]


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
        unscored = [c["cid"] for c in candidates if c["cid"] not in ce["logits"]]
        if unscored:
            raise ValueError(f"{e['qid']}: no cross-encoder score for candidates {unscored[:3]}")
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


def rerank_chunks(e: dict, rel: dict[str, float], logits: dict[str, float]) -> list[dict]:
    """The cross-encoder top-k exactly as the CE baseline (--selector ce) selects and shows it: the top
    rerank.TOP_K candidates by cross-encoder logit over the whole pool, presented in relevance order."""
    return select_chunks(e, rel, rerank.TOP_K, "rank", 0, ce={"logits": logits, "calibration": None})


PERMUTATIONS = {o["order_id"]: o["permutation"] for o in order_mod.permutation_schedule()}
# How a run is executed, not what it computes: excluded from the saved configuration and its checks.
EXECUTION_ARGS = {"num_shards", "shard_id", "calibrate_only", "trace_prompts", "max_empty_rate"}
# Mechanism-section arm only. The default --variants list does not include it.
ANALYSIS_EAB = "eab:g1.2:ans1:pos:gate"


def apply_analysis_eab(args) -> None:
    """Append the ans1 mechanism arm when --analysis-eab is set. Default runs stay without an eab: variant."""
    if not getattr(args, "analysis_eab", False):
        return
    parts = [v for v in (args.variants or "").split(",") if v]
    if ANALYSIS_EAB not in parts:
        parts.append(ANALYSIS_EAB)
    args.variants = ",".join(parts)


def modality_pool(example: dict, modality: str) -> dict:
    """Same question, restricted to one modality, so top-k is not the routed mix."""
    return {**example, "candidates": [c for c in example["candidates"] if c["modality"] == modality],
            "fixed_evidence": False}


def v2_action_chunks(e: dict, rel: dict[str, float], joint: list[dict], ce_logits: dict[str, float], k: int,
                     seed: int) -> dict[str, list[dict] | None]:
    """Candidate list of each v2 action. None is direct (the question alone, no evidence).

    static_text and static_image are the retriever top-k inside one modality pool. static_mixed is the joint top-k
    by calibrated relevance (the list of the static joint arm). rerank_static is the cross-encoder top-k of the CE
    baseline (rerank_chunks). An empty list means the action has no candidates.
    """
    return {DIRECT: None,
            "static_text": select_chunks(modality_pool(e, "text"), rel, k, "rank", seed),
            "static_image": select_chunks(modality_pool(e, "image"), rel, k, "rank", seed),
            "static_mixed": joint,
            "rerank_static": rerank_chunks(e, rel, ce_logits)}


def token_text_of(qwen):
    """Token id -> string for the first-token candidates, or None when the model exposes no tokenizer decode."""
    decode = getattr(getattr(qwen, "tok", None), "decode", None)
    return None if decode is None else (lambda token_id: decode([int(token_id)]))


def decode_v2_actions(qwen, e: dict, rel: dict[str, float], chunks_by_action: dict, direct: tuple, decoded: dict,
                      max_new: int, answer_style: str, beta: float) -> dict:
    """One record per v2 action (signals.action_record), in V2_ACTIONS order.

    ``direct`` is the (result, encoding) of the direct decode; its first-token logits give KL_to_direct.
    ``decoded`` maps a tuple of chunk ids to the (result, encoding) already decoded from the same prompt; new
    decodes are added to it. An evidence action with no candidates is recorded as unavailable: it is neither
    decoded nor replaced by another action's result. Each available record carries the trace of its own input
    (prompt length and hash, visual tokens per image) and decode (token ids, logprobs, time, length-limit flag).

    Each available record also carries ``answer_token`` (signals.with_answer_token): the answer slot of the decode
    and U_omni there, H_grd of the t_end row over the action's evidence (the supplied context for direct) plus
    beta * H_sem of the answer token. Evidence actions carry ``route``, the paper's M_m H_m routing over their
    evidence spans on the same row (routing.route_modalities, margin 0.08); the v2-Rule "route" family reads it
    from static_mixed.
    """
    direct_logits = direct[0].first_logits
    image_token_id = getattr(qwen, "image_token_id", None)
    answer_first = prompts.answer_format(e["dataset"], answer_style) == "answer_first"
    records = {}
    for name in V2_ACTIONS:
        chosen = chunks_by_action[name]
        if name != DIRECT and not chosen:
            records[name] = signals.unavailable_record(signals.retriever_signals(e["candidates"], rel, chosen))
            continue
        if name == DIRECT:
            result, enc = direct
        else:
            key = tuple(c["cid"] for c in chosen)
            if key not in decoded:
                enc = qwen.encode(*pipeline.build_segments(e, chosen, answer_style))
                decoded[key] = (qwen.run(enc, None, max_new, decode_static=True)[0]["static"], enc)
            result, enc = decoded[key]
        if result.token_logprobs is None:
            raise RuntimeError(f"{e['qid']}: the {name} decode recorded no token logprobs")
        record = action_signal_record(qwen, e, rel, name, chosen, result, enc, direct_logits, answer_first, beta)
        records[name] = signals.with_trace(record, signals.input_trace(enc, image_token_id),
                                           signals.decode_trace(result, max_new, token_text_of(qwen)))
    return records


def action_signal_record(qwen, e: dict, rel: dict[str, float], name: str, chosen: list[dict] | None, result, enc,
                         direct_logits, answer_first: bool, beta: float) -> dict:
    """The signal record of one available action from one decode of its prompt (``result``, ``enc``).

    The full decode (decode_v2_actions) and the deployment path (deploy_v2) both build their records here, so the
    selector reads the same quantities in both: the first token, the answer token with U_omni = H_grd + beta * H_sem
    (H_grd of the t_end row over the action's evidence, the supplied context for direct), and for an evidence
    action the M_m H_m route of its evidence (routing.route_modalities, margin 0.08).
    """
    chosen = chosen or []
    record = signals.action_record(
        result.text, result.first_logits, result.tokens, result.token_logprobs, qwen.vocab_size,
        None if name == DIRECT else direct_logits, signals.retriever_signals(e["candidates"], rel, chosen),
        [] if name == DIRECT else [c["cid"] for c in chosen], answer_first=answer_first)
    evidence_ids = pipeline.context_ids(e) if name == DIRECT else [c["cid"] for c in chosen]
    h_grd = pipeline.raw_uncertainty(result.attention, enc, evidence_ids, result.first_logits, beta,
                                     qwen.vocab_size)["h_grd"]
    route = None if name == DIRECT else routing.route_modalities(
        result.attention, enc.spans, {c["cid"]: c["modality"] for c in chosen})
    decode_text = token_text_of(qwen)
    texts = None if decode_text is None else [decode_text(t) for t in result.tokens]
    return signals.with_answer_token(record, getattr(result, "step_entropy", None),
                                     getattr(result, "step_margin", None), texts, e["question"], answer_first, h_grd,
                                     beta, route)


def candidate_scores(e: dict, rel: dict[str, float], ce_logits: dict[str, float] | None) -> list[dict]:
    """Every candidate of the question with its scores, best calibrated relevance first.

    Written for analysis after the run (where the gold evidence ranked, how close it was to the cut). ``gold`` is
    copied from the data and is not read by any selection in this module.
    """
    rows = [{"cid": c["cid"], "modality": c["modality"], "relevance": round(float(rel[c["cid"]]), 5),
             "ce_logit": None if ce_logits is None else round(float(ce_logits[c["cid"]]), 5),
             "gold": c.get("gold")} for c in e["candidates"]]
    return sorted(rows, key=lambda row: (-row["relevance"], str(row["cid"])))


def prompt_sample(e: dict, chunks_by_action: dict, answer_style: str) -> dict:
    """The full prompt of every available v2 action of one question, for reading by a person.

    Text segments are joined as the model receives them; an image is shown as ``<image:path>`` at its position.
    """
    actions = {}
    for name in V2_ACTIONS:
        chosen = chunks_by_action[name]
        if name != DIRECT and not chosen:
            continue
        segments, _ = pipeline.build_segments(e, chosen, answer_style)
        actions[name] = {"prompt": "".join(s.text if s.image is None else f"<image:{s.image}>" for s in segments),
                         "images": [s.image for s in segments if s.image is not None]}
    return {"qid": e["qid"], "question": e["question"], "actions": actions}


# Greedy tokens a deployment decodes on an action whose signal the rule reads, to reach its answer token. In the
# answer-first format the answer token is the first or second generated token; signals.answer_slot looks no further.
DEPLOY_SIGNAL_TOKENS = 8


def deploy_v2(qwen, e: dict, rel: dict[str, float], chunks_by_action: dict, decision: dict, max_new: int,
              answer_style: str, beta: float) -> dict:
    """Apply a frozen v2-Rule decision the way a deployment would: read only the signals the decision needs, then
    decode the one chosen action.

    Reading the signal of an action is one prefill of its prompt followed by at most DEPLOY_SIGNAL_TOKENS greedy
    tokens, enough to reach the answer token where U_omni is defined; the record is built by
    action_signal_record, as in the full decode, so the rule reads the same quantities (answer_token, and the
    route of static_mixed for the "route" family). Nothing here reads a score or a gold field. The returned record
    counts the passes, their decoded tokens and the wall time of this path alone; the answer is parsed in the
    run's answer format.
    """
    t0 = time.perf_counter()
    answer_first = prompts.answer_format(e["dataset"], answer_style) == "answer_first"
    actions = {name: {"available": name == DIRECT or bool(chunks_by_action[name]), "answer": None}
               for name in V2_ACTIONS}
    encodings, passes, direct_logits = {}, [], None

    def encoding(name):
        if name not in encodings:
            encodings[name] = qwen.encode(*pipeline.build_segments(e, chunks_by_action[name], answer_style))
        return encodings[name]

    while True:
        needed = arbitration.signals_needed(decision, actions)
        if not needed:
            break
        for name in needed:
            enc = encoding(name)
            result = qwen.run(enc, None, DEPLOY_SIGNAL_TOKENS, decode_static=True)[0]["static"]
            if name == DIRECT:
                direct_logits = result.first_logits
            actions[name] = action_signal_record(qwen, e, rel, name, chunks_by_action[name], result, enc,
                                                 direct_logits, answer_first, beta)
            passes.append({"action": name, "decoded_tokens": len(result.tokens or [])})
    chosen = arbitration.apply_decision(decision, actions)["action"]
    result = qwen.run(encoding(chosen), None, max_new, decode_static=True)[0]["static"]
    return {"action": chosen, "answer": answers.parse_answer(result.text, answer_first)["answer"], "raw": result.text,
            "prefilled": [p["action"] for p in passes], "prefill_calls": len(passes),
            "signal_tokens": sum(p["decoded_tokens"] for p in passes), "decode_calls": 1,
            "seconds": round(time.perf_counter() - t0, 4)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(data.LOADERS))
    ap.add_argument("--model", default="2b", choices=["2b", "8b", "internvl3.5-8b"])
    ap.add_argument("--max-patches", type=int, default=1, help="InternVL: 448px tiles per image (256 tokens each)")
    ap.add_argument("--split", required=True, choices=["calib", "dev", "test"],
                    help="calib: decode the calibration questions; dev: tune here; test: run once with frozen settings")
    ap.add_argument("--n-calib", type=int, default=None,
                    help="calibration questions (default: the dataset's protocol size, data.CALIB_RESERVE / N_CALIB_TRAIN)")
    ap.add_argument("--calib-source", choices=["train", "heldout"], default="train",
                    help="train: calibration questions from the training split, labelled (the v2 table scores them); "
                         "heldout: carved out of the evaluation data")
    ap.add_argument("--n-dev", type=int, default=None,
                    help="size of the frozen dev split (default: data.N_DEV); the test split starts after it, so "
                         "changing it changes the test set")
    ap.add_argument("--split-seed", type=int, default=2027)
    ap.add_argument("--allow-missing-assets", action="store_true",
                    help="drop questions or candidates whose assets are missing instead of failing; the dropped qids "
                         "and counts are written to the configuration")
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
                    help="also decode the chunk-replication arm: max(1, round(exp(gamma*U*r))) adjacent copies when "
                         "U > tau_bias, no truncation (paper: replication baseline, Prop. 2)")
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
                    help="heuristic default statistics: Norm(median 0, IQR 1); relevance normalisation is unchanged "
                         "(use --relevance-norm minmax for that)")
    ap.add_argument("--image-mode", default="mix", choices=["mix", "clip", "caption"])
    ap.add_argument("--beta", type=float, default=0.30)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--tau-bias", type=float, default=0.40)
    ap.add_argument("--eab-uncertainty", choices=["pre", "post"], default="pre",
                    help="pre (paper): reuse the retrieval-gate U for EAB; post: recompute U on the packed input")
    ap.add_argument("--answer-style", default="default", choices=["default", "short", "cite", "sentence"],
                    help="short: one word/phrase; cite: statements with [E<j>] citations for the citation metrics "
                         "(raise --max-new-tokens); sentence: the sentence-only WebQA instruction, for the "
                         "measurement-position ablation (the default WebQA instruction is answer-first). Applies to "
                         "every arm.")
    ap.add_argument("--variants", default="",
                    help="comma-separated EAB variants (pipeline.parse_variant). Empty by default: the main "
                         "method does not decode an eab: arm")
    ap.add_argument("--analysis-eab", action="store_true",
                    help="mechanism section only: also decode ans1 EAB "
                         f"({ANALYSIS_EAB}). Not part of the main method")
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
    ap.add_argument("--v2", action="store_true",
                    help="v2 action records: decode direct, static_text, static_image, static_mixed and rerank_static "
                         "for every question (rec['actions']). Needs --v2-reranker")
    ap.add_argument("--v2-reranker", action="store_true",
                    help="load cross-encoder logits for the v2 rerank_static action only; the selector and every other "
                         "arm keep relevance. Requires --v2")
    ap.add_argument("--decisions", default=None,
                    help="frozen decisions JSON. Checked and hashed into the config; this process does not search a grid")
    ap.add_argument("--trace-prompts", type=int, default=20,
                    help="with --v2: write the full prompt of every action for the first N questions of this "
                         "process to <output>.prompts.ndjson (0 disables)")
    ap.add_argument("--max-empty-rate",type=float, default=None,
                    help="single-process runs: after the outputs and completion marker are written, exit non-zero when "
                         "the rate of blank or citation-only answers exceeds this value (default: only warn above "
                         "0.5%%). Refusals are reported apart (refusal_rate, per arm) and do not count toward it")
    return ap


def check_v2_args(ap, args) -> None:
    """v2 options. static_mixed is the relevance top-k, so the selector stays relevance under --v2; the cross-encoder
    logits of rerank_static come from --v2-reranker, which changes no other arm."""
    if args.v2_reranker and not args.v2:
        ap.error("--v2-reranker only applies to the v2 actions: add --v2")
    if args.v2 and args.selector != "relevance":
        ap.error("--v2 keeps static_mixed on the relevance top-k: drop --selector and pass --v2-reranker instead")
    if args.v2 and not args.v2_reranker:
        ap.error("--v2 needs cross-encoder logits for rerank_static: add --v2-reranker")
    if args.v2 and args.k != rerank.TOP_K:
        ap.error(f"--v2 rerank_static is the cross-encoder baseline, which takes exactly k={rerank.TOP_K}")
    if args.v2 and args.order != "rank":
        ap.error("--v2 shows static_text, static_image and rerank_static in relevance order; drop --order")


def check_decisions_file(ap, args) -> None:
    """Hash the frozen decisions file into the configuration. It is never expanded into a grid."""
    args.decisions_sha256 = None
    if not args.decisions:
        return
    try:
        args.decisions_sha256 = runio.read_frozen_decisions(args.decisions)
    except runio.FrozenDecisionsError as exc:
        ap.error(str(exc))


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    apply_analysis_eab(args)
    check_decisions_file(ap, args)
    check_v2_args(ap, args)
    if args.order not in ("rank", "shuffle", "reverse") and args.order.removeprefix("perm:") not in PERMUTATIONS:
        ap.error(f"unknown --order {args.order}")
    if args.selector != "relevance" and (args.k != rerank.TOP_K or args.min_per_modality):
        ap.error(f"CE selectors take exactly k={rerank.TOP_K} and no --min-per-modality")
    # --no-calibration only swaps the U statistics for defaults (below); the relevance normalisation is a
    # separate switch (--relevance-norm) so that the ablation changes one thing.
    if (args.route_rule == "query") != (args.query_routes is not None):
        ap.error("--route-rule query and --query-routes go together")
    if args.m2rag and DATASET_FAMILY[args.dataset] == "videorag":
        ap.error("M2-RAG accepts no video evidence (an em dash in the paper's main table)")
    if (args.fitm or args.m2rag) and args.model == "internvl3.5-8b":
        ap.error("the Found-in-the-Middle and M2-RAG adapters support the Qwen3-VL backbones only")
    if args.route_rule in ("mass", "entropy") and not (args.modality_routing or args.first_round_routing):
        ap.error("--route-rule mass/entropy ablates the attention route; add --first-round-routing or --modality-routing")
    norm_source = None
    if args.norm_from:
        norm_source = json.loads(Path(args.norm_from).read_text())
        # The transferred Norm must exist for every phase this run uses: pre and post always, and the probe phase
        # for cold-start targets (--probe-uncertainty: the gate reads U on the probe round). A source without that
        # phase (OK-VQA has no probe round) cannot calibrate such a target, so the transfer fails here.
        needed = ["pre", "post"] + (["probe"] if args.probe_uncertainty else [])
        missing = [ph for ph in needed if not norm_source.get(f"norm_{ph}")]
        if missing:
            ap.error(f"--norm-from {args.norm_from} has no norm_{'/'.join(missing)}; transfer calibration from a run "
                     "of the same kind (a cold-start target needs a source that has a probe round)")

    if not 0 <= args.shard_id < args.num_shards:
        ap.error("--shard-id must lie in [0, --num-shards)")
    if args.max_empty_rate is not None and not 0 <= args.max_empty_rate <= 1:
        ap.error("--max-empty-rate must lie in [0, 1]")
    out_dir = paths.RUNS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # One writer per output file: a second process on the same tag (or shard) is refused while the first runs.
    lock_name = (f"{args.dataset}.lock" if args.num_shards == 1
                 else f"{args.dataset}.shard{args.shard_id:02d}-of-{args.num_shards:02d}.lock")
    log_name = lock_name.removesuffix(".lock") + ".log"
    with runio.run_lock(out_dir / lock_name), runio.tee_output(out_dir / log_name, " ".join(sys.argv if argv is None
                                                                                       else ["omniaware.run", *argv])):
        evaluate(ap, args, out_dir, norm_source)


def evaluate(ap, args, out_dir: Path, norm_source: dict | None) -> None:
    """Decode the questions of one run into its output directory (the caller holds the run lock)."""
    max_new = args.max_new_tokens or pipeline.MAX_NEW_TOKENS[DATASET_FAMILY[args.dataset]]
    config_path = out_dir / f"{args.dataset}_config.json"
    out_path = out_dir / (f"{args.dataset}.jsonl" if args.num_shards == 1
                          else f"{args.dataset}.shard{args.shard_id:02d}-of-{args.num_shards:02d}.jsonl")
    # Only the process that writes the caches (single process, or --calibrate-only) may create or replace them;
    # shards read them and must find the configuration left by --calibrate-only.
    writes_caches = args.num_shards == 1 or args.calibrate_only
    has_outputs = runio.has_outputs(out_dir, args.dataset)
    if (has_outputs or args.num_shards > 1) and not config_path.exists():
        ap.error("saved answers have no configuration; use a new --tag" if has_outputs else
                 f"shards need {config_path.name}: run the same command with --calibrate-only first")
    if config_path.exists() and (has_outputs or args.num_shards > 1):
        saved = json.loads(config_path.read_text(encoding="utf8"))
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
    # Input variants get their own relevance / calibration caches.
    data_tag = "".join(f"_{k}-{v}" for k, v in sorted(loader_kwargs.items()) if v is not None)
    if args.allow_missing_assets:
        loader_kwargs["allow_missing"] = True
    # Image subsets are cut from their full split (data.IMAGE_SUBSET_OF), so the full loader defines the universe.
    split_universe = data.IMAGE_SUBSET_OF.get(args.dataset, args.dataset)
    examples = data.LOADERS[split_universe](**loader_kwargs)
    if args.route_rule == "query" and any("suite_score" not in c for e in examples for c in e["candidates"]):
        # Checked before the model loads. The released WebQA candidates come from this repository's BGE/CLIP
        # retrieval and carry no suite score, so the UniversalRAG baseline is not defined on them.
        ap.error(f"--route-rule query needs a UniversalRAG suite_score on every candidate; {args.dataset} has "
                 "candidates without one")
    dropped = {"universe": getattr(examples, "meta", {}).get("dropped", {})}
    # Protocol sizes (docs/PROTOCOL.md section 1, data.CALIB_RESERVE / N_CALIB_TRAIN / N_DEV).
    family = data.protocol_family(args.dataset)
    if args.n_dev is None:
        args.n_dev = data.N_DEV[family]
    elif args.n_dev != data.N_DEV[family]:
        print(f"[data] --n-dev {args.n_dev} differs from the protocol size {data.N_DEV[family]}: the test set changes",
              flush=True)
    calib_pool, reserve = None, None
    if args.calib_source == "train":
        if split_universe not in data.TRAIN_LOADERS:
            ap.error(f"{args.dataset} has no training-split loader; use --calib-source heldout")
        calib_pool = data.TRAIN_LOADERS[split_universe](**({"allow_missing": True} if args.allow_missing_assets else {}))
        dropped["calibration_pool"] = getattr(calib_pool, "meta", {}).get("dropped", {})
        reserve = data.TRAIN_RESERVE[family]
        if args.n_calib is None:
            args.n_calib = data.N_CALIB_TRAIN[family]
    else:
        reserve = data.CALIB_RESERVE.get(family)
        if reserve is None:
            ap.error(f"{args.dataset} has no held-out calibration reserve; use --calib-source train")
        if args.n_calib is None:
            args.n_calib = reserve
    # Dense candidate files may cover only the intended local slice.  Keep the
    # original OK-VQA universe for split identities instead of reshuffling it.
    retrieved_okvqa = args.dataset in ("okvqa_dense", "okvqa_gs", "okvqa_gs_full")
    split_source = data.load_okvqa(candidates_path=False) if retrieved_okvqa else examples
    sp = data.split_examples(split_source, args.n_calib, args.n_dev, args.split_seed, calib_pool, reserve=reserve)
    if (family == "webqa" and args.calib_source == "train" and args.dataset not in data.IMAGE_SUBSET_OF
            and len(split_source) == data.WEBQA_VAL_SIZE and len(sp["test"]) != data.WEBQA_VAL_SIZE):
        raise ValueError(f"WebQA test split must be all {data.WEBQA_VAL_SIZE} validation questions, "
                         f"got {len(sp['test'])}")
    if args.dataset in data.IMAGE_SUBSET_OF:
        sp = data.image_subset(sp)
    data.check_disjoint(sp)                       # runtime check: calib, dev, test pairwise disjoint
    if not sp["test"]:
        raise ValueError(f"{args.dataset}: the test split is empty")
    calib = sp["calib"]
    end = None if args.n_eval is None else args.offset + args.n_eval
    evals = sp[args.split][args.offset:end]
    if not evals:
        raise ValueError(f"{args.dataset}: the {args.split} split is empty (offset {args.offset}, n-eval {args.n_eval})")
    if retrieved_okvqa:
        by_qid = {e["qid"]: e for e in examples}
        missing = [e["qid"] for e in calib + evals if e["qid"] not in by_qid]
        if missing:
            ap.error(f"dense candidates are missing {len(missing)} requested calibration/evaluation questions; "
                     "generate candidates for the original OK-VQA split before running this slice")
        calib = [by_qid[e["qid"]] for e in calib]
        evals = [by_qid[e["qid"]] for e in evals]
    print(f"[data] {args.dataset}: {len(examples)} total, calib {len(calib)} ({args.calib_source}), "
          f"dev {len(sp['dev'])}, {args.split} {len(evals)}", flush=True)
    shortfall = data.candidate_shortfall(calib + evals, args.k)
    if shortfall["fewer_than_k"]["count"] or shortfall["without_modality"]:
        print(f"[data] candidates: {shortfall['fewer_than_k']['count']} questions with fewer than k={args.k}; "
              + ", ".join(f"{m}: {v['count']} without" for m, v in shortfall["without_modality"].items()), flush=True)

    raw = load_relevance(calib + evals, cache_dir / f"relevance_{args.dataset}{data_tag}.json", write=writes_caches)
    cal = relevance.Calibrator.fit(raw, [e["qid"] for e in calib], args.image_mode)

    def rel_of(e):
        if args.relevance_norm == "ecdf":
            return {c["cid"]: cal.relevance(raw[e["qid"]][c["cid"]], args.image_mode) for c in e["candidates"]}
        scores = {c["cid"]: relevance.Calibrator.raw(raw[e["qid"]][c["cid"]], args.image_mode) for c in e["candidates"]}
        if args.relevance_norm == "raw":
            return scores
        lo, hi = min(scores.values(), default=0.0), max(scores.values(), default=1.0)
        return {c: (s - lo) / (hi - lo) if hi > lo else 1.0 for c, s in scores.items()}

    # Cross-encoder logits are needed on the evaluated questions; the calibration questions only when the CE
    # selector's per-modality percentile is fitted on them.
    ce_questions = calib + evals if args.selector != "relevance" else evals
    ce_table = None
    if args.selector != "relevance" or args.v2_reranker:
        ce_table = rerank.ce_logits(ce_questions, cache_dir / f"ce_{args.dataset}{data_tag}.json", paths.RERANKER,
                                    write=writes_caches)
    ce_of = ce_rel_of = None
    if args.selector != "relevance":
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
    # One line per launch (a resumed or sharded run appends): code version, libraries, hardware, model loading.
    environment = runio.environment_snapshot(sys.argv, qwen)
    runio.append_jsonl(out_path.with_suffix(".env.ndjson"), environment)
    print(f"[env] commit {environment['git_commit']} dirty={environment['git_dirty']} torch {environment['torch']} "
          f"transformers {environment['transformers']} attn {environment['model']['attn_implementation']} "
          f"dtype {environment['model']['dtype']} gpus {environment['gpus']}", flush=True)
    prompts_path = out_path.with_suffix(".prompts.ndjson")
    # With --v2 and a frozen decisions file, the frozen v2-Rule is also run as a deployment would run it.
    deploy_decision = None
    if args.v2 and args.decisions:
        deploy_decision = json.loads(Path(args.decisions).read_text(encoding="utf8")).get("v2", {}).get("rule")
        if deploy_decision is None:
            ap.error(f"--decisions {args.decisions} holds no v2 rule decision (v2.rule)")

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
        """Paper Table `causal`: re-decode static, every variant and (with --replication-gamma) the replication
        arm on perturbed evidence, keeping the retrieval decision and the uncertainty EAB uses fixed.

        The target is the highest-ranked text chunk whose prompt text (after the word cut) holds a gold answer.
        Eligibility is per arm (eligible_by_arm). Each requested kind has one entry per eligible question: the
        arm predictions, or {"failed_reason": ...} when the perturbation cannot be built (fixed denominator)."""
        golds = e["golds"]
        rel_sel = {c["cid"]: rel_all[c["cid"]] for c in chunks}
        target = interventions.find_target(chunks, rel_sel, golds)
        retrieved = bool(rec["retrieve"]) and target is not None
        active = {"static": True, **{v.name: info["eab_active"][v.name] for v in variants}}
        if args.replication_gamma is not None:
            active["replication"] = info["u_eab"] > args.tau_bias
        out = {"target": target and target["cid"], "eligible": retrieved,
               "eligible_by_arm": {arm: retrieved and bool(on) for arm, on in active.items()}}
        if not retrieved:
            return out
        target_p = interventions.as_prompt_chunk(target)     # the text the model reads, for edits
        gold = interventions.answer_span(target_p["text"], golds)[2]
        entity = interventions.pick_substitute(e["qid"], golds, gold, substitute_pool, iv_nlp)
        out.update(gold_in_target=gold, entity=entity, attempted=list(args.intervention))
        segments, split = evidence_segments(e, chunks, rel_sel)

        def decode(kind, chunks_x, rel_x, enc_x, segments_x, split_x):
            """Decode static, the variants and replication (copy counts recomputed on the perturbed chunks)."""
            res_x, _, _ = qwen.run(enc_x, lambda a, lg: variant_specs(e, enc_x, rel_x, info["u_eab"]), max_new,
                                   decode_static=True)
            entry = {"preds": {n: r.text for n, r in res_x.items()}, "chunks": [c["cid"] for c in chunks_x]}
            if args.answer_style == "cite":
                entry["citation_sources"] = pipeline.citation_sources(segments_x, chunks_x)
            if args.replication_gamma is not None:
                counts = order_mod.replication_counts([rel_x[c["cid"]] for c in chunks_x], info["u_eab"],
                                                      info["u_eab"] > args.tau_bias, args.replication_gamma)
                if all(n == 1 for n in counts):
                    entry["preds"]["replication"] = entry["preds"]["static"]
                    if args.answer_style == "cite":
                        entry["replication_citation_sources"] = entry["citation_sources"]
                else:
                    copies = replication_copies(chunks_x, counts)
                    seg_r, split_r = evidence_segments(e, copies, rel_x)
                    enc_r = qwen.encode(seg_r, split_r)
                    if kind == "mask":     # every copy of the target is hidden
                        for c in copies:
                            if c["cid"].split("#")[0] == target["cid"]:
                                enc_r = interventions.mask_span(enc_r, c["cid"])
                    res_r, _, _ = qwen.run(enc_r, None, max_new, decode_static=True)
                    entry["preds"]["replication"] = res_r["static"].text
                    if args.answer_style == "cite":
                        entry["replication_citation_sources"] = pipeline.citation_sources(seg_r, copies)
            return entry

        for kind in args.intervention:
            extra = {}
            if kind == "mask":
                out[kind] = decode(kind, chunks, rel_sel, interventions.mask_span(enc, target["cid"]), segments, split)
                continue
            if kind == "rerun":
                out[kind] = decode(kind, chunks, rel_sel, enc, segments, split)
                continue
            if kind == "distractors":
                swapped = interventions.distractors(e, chunks, rel_all, target["cid"], golds)
                if swapped is None:
                    out[kind] = {"failed_reason": "fewer than two candidates beyond the top k without the answer"}
                    continue
                chunks_x, extra["distractors"] = swapped
                rel_x = {c["cid"]: rel_all[c["cid"]] for c in chunks_x}
            elif entity is None:   # cf and random_edit share the questions: both need the same-type entity
                out[kind] = {"failed_reason": "no same-type entity among calibration/training golds"}
                continue
            elif kind == "cf":
                chunks_x = [interventions.counterfactual(target_p, golds, entity) if c is target else c
                            for c in chunks]
                rel_x = rel_sel
            elif kind == "random_edit":
                edited = interventions.random_edit(target_p, golds, entity, f"random-edit:{e['qid']}")
                if edited is None:
                    out[kind] = {"failed_reason": "no window of whole words as long as the answer avoids every gold"}
                    continue
                chunks_x = [edited if c is target else c for c in chunks]
                rel_x = rel_sel
            else:
                injected = interventions.inject(e, chunks, rel_all, golds, entity, kind, args.order)
                if injected is None:
                    out[kind] = {"failed_reason": "no non-gold chunk to replace"}
                    continue
                chunks_x, rel_x = injected
            segments_x, split_x = evidence_segments(e, chunks_x, rel_x)
            out[kind] = {**decode(kind, chunks_x, rel_x, qwen.encode(segments_x, split_x), segments_x, split_x),
                         **extra}
        return out

    routed = (args.modality_routing or args.first_round_routing
              or args.route_rule in ("retriever", "all", "oracle", "query"))
    query_routes = universalrag.load_routes(args.query_routes) if args.query_routes else None

    def cold(e):
        return not e["query_images"]

    # Wall-clock of the probe rounds of the question being evaluated (reset per question).
    timing = {"probe": 0.0}

    def probe(e, rel):
        """Cold-start probe round (paper Sec. 4.2-4.3): the joint top-k by calibrated relevance in rank
        order, one zero-bias prefill. H_grd over these spans makes U informative without supplied context,
        and the same attention gives the M_m*H_m route."""
        t_probe = time.perf_counter()
        first = select_chunks(e, rel, args.k, "rank", 0, args.min_per_modality, None)
        enc_1 = qwen.encode(*evidence_segments(e, first, rel))
        _, attn_1, logits_1 = qwen.run(enc_1, None, 1, decode_static=False)
        out = {"chunks": first, "enc": enc_1, "attn": attn_1, "seconds": qwen.last_prefill_seconds,
               "raw": pipeline.raw_uncertainty(attn_1, enc_1, [c["cid"] for c in first], logits_1, args.beta,
                                               qwen.vocab_size)}
        timing["probe"] += time.perf_counter() - t_probe
        return out

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
    # The answer instruction of this run's arms (prompts.py); M2-RAG's extractor reads the same string.
    # Its sha256 keys the U and SeaKR calibration caches: their raw values come from prompts that contain it.
    run_instruction = prompts.instruction_for_style(family, args.answer_style,
                                                    cite_instruction=pipeline.CITATION_INSTRUCTION,
                                                    short_instruction=pipeline.SHORT_INSTRUCTION)
    ucal_path = cache_dir / (f"ucal_{args.dataset}_{args.model}_k{args.k}_{args.image_mode}_n{args.n_calib}"
                             f"_{args.calib_source}{args.split_seed}{style}.json")
    # Each calibration entry is keyed by its fingerprint: question content, every setting its raw values depend
    # on, and (for route-rule query) the hash of the UniversalRAG route file.
    query_routes_sha = runio.sha256_file(args.query_routes) if args.query_routes else None
    ucal_settings = {"encoders": runio.encoder_identity(), "model": args.model, "model_path": str(model_path),
                     "answer_style": args.answer_style,
                     "instruction_sha256": run_instruction["instruction_sha256"], "k": args.k, "image_mode": args.image_mode, "beta": args.beta,
                     "min_per_modality": args.min_per_modality, "modality_routing": args.modality_routing,
                     "route_rule": args.route_rule, "first_round_routing": args.first_round_routing,
                     "probe_uncertainty": args.probe_uncertainty, "selector": args.selector,
                     "relevance_norm": args.relevance_norm, "attn_layer": args.attn_layer,
                     "max_pixels": args.max_pixels, "max_patches": args.max_patches,
                     "query_routes_sha256": query_routes_sha}
    ucal_fp = {e["qid"]: runio.fingerprint(e, ucal_settings) for e in calib}
    saved_ucal = json.loads(ucal_path.read_text(encoding="utf8")) if ucal_path.exists() else {}
    ucal = {e["qid"]: saved_ucal[e["qid"]] for e in calib
            if e["qid"] in saved_ucal and saved_ucal[e["qid"]].get("fingerprint") == ucal_fp[e["qid"]]}
    stale = [e for e in calib if e["qid"] not in ucal]
    if stale and not writes_caches:
        raise SystemExit(f"{len(stale)} calibration entries of {ucal_path.name} are missing or stale; "
                         "run --calibrate-only first")
    t0 = time.time()
    for e in stale:
        rel = rel_of(e)
        enc_d = qwen.encode(*pipeline.build_segments(e, None, args.answer_style))
        _, attn_d, logits_d = qwen.run(enc_d, None, 1, decode_static=False)
        route = routing.initial_route(attn_d, enc_d, e, args.route_rule) if args.modality_routing else None
        p = probe(e, rel) if args.probe_uncertainty and cold(e) else None
        route = resolve_route(e, rel, route, calibrating=True, probe_out=p)
        chunks = select_chunks(e, rel, args.k, "rank", 0, args.min_per_modality, route, ce_of and ce_of(e))
        pre = pipeline.raw_uncertainty(attn_d, enc_d, pipeline.context_ids(e), logits_d, args.beta, qwen.vocab_size)
        enc = qwen.encode(*evidence_segments(e, chunks, rel))
        _, attn, logits = qwen.run(enc, None, 1, decode_static=False)
        post = pipeline.raw_uncertainty(attn, enc, [c["cid"] for c in chunks], logits, args.beta, qwen.vocab_size)
        ucal[e["qid"]] = {"fingerprint": ucal_fp[e["qid"]], "pre": pre, "post": post}
        if p:
            ucal[e["qid"]]["probe"] = p["raw"]
        if routed:
            ucal[e["qid"]]["route"] = route
        runio.atomic_write_text(ucal_path, json.dumps(ucal))
    if time.time() - t0 > 1:
        print(f"[ucal] {len(calib)} calibration examples in {time.time() - t0:.1f}s", flush=True)
    # Raw entropies are cached; the combination (beta, sum/zsum) and Norm are fitted here.
    calibrators = {}
    for phase in ("pre", "post", "probe"):
        raws = [ucal[e["qid"]][phase] for e in calib if phase in ucal[e["qid"]]]
        if raws:
            comb = pipeline.Combiner.fit(raws, args.beta, args.u_combine)
            calibrators[phase] = (comb, pipeline.Norm.fit([comb(r) for r in raws]))
    if norm_source:
        for phase in ("pre", "post", "probe"):
            if norm_source.get(f"norm_{phase}"):
                comb = (pipeline.Combiner.from_json(norm_source[f"comb_{phase}"]) if norm_source.get(f"comb_{phase}")
                        else pipeline.Combiner(args.beta))
                calibrators[phase] = (comb, pipeline.Norm(**norm_source[f"norm_{phase}"]))
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

    seakr_settings = {"model": args.model, "model_path": str(model_path), "answer_style": args.answer_style,
                      "instruction_sha256": run_instruction["instruction_sha256"], "samples": args.seakr_samples, "max_new_tokens": args.seakr_max_new_tokens}

    def seakr_fingerprint(e):
        return runio.fingerprint(e, seakr_settings)

    seakr_threshold, seakr_missing_calibration = None, None
    if args.seakr_samples:
        scache_path = cache_dir / (f"seakr_{args.dataset}_{args.model}_n{args.seakr_samples}_m{args.seakr_max_new_tokens}"
                                   f"_c{args.n_calib}_{args.calib_source}{args.split_seed}{style}.json")
        saved_scache = json.loads(scache_path.read_text(encoding="utf8")) if scache_path.exists() else {}
        scache = {e["qid"]: saved_scache[e["qid"]] for e in calib
                  if e["qid"] in saved_scache and saved_scache[e["qid"]].get("fingerprint") == seakr_fingerprint(e)}
        missing_scores = [e for e in calib if e["qid"] not in scache]
        if missing_scores and not writes_caches:
            raise SystemExit(f"{len(missing_scores)} SeaKR calibration scores in {scache_path.name} are missing or "
                             "stale; run --calibrate-only first")
        for e in missing_scores:
            enc_d = qwen.encode(*pipeline.build_segments(e, None, args.answer_style))
            scache[e["qid"]] = {"fingerprint": seakr_fingerprint(e),
                                "score": seakr.score_input(qwen, enc_d, args.seakr_samples, args.seakr_max_new_tokens)}
            runio.atomic_write_text(scache_path, json.dumps(scache))
        calibration_scores = [scache[e["qid"]]["score"] for e in calib]
        seakr_missing_calibration = sum(seakr.is_missing(s) for s in calibration_scores)
        seakr_threshold = (args.seakr_threshold if args.seakr_threshold is not None
                           else seakr.matched_threshold(calibration_scores, u_rate))
        print(f"[seakr] threshold {seakr_threshold:.4f} (U retrieval rate on calibration {u_rate:.3f}; "
              f"{seakr_missing_calibration} of {len(calib)} calibration questions have no eigen score)", flush=True)

    substitute_pool, iv_nlp = None, None
    if args.intervention:
        try:
            import spacy
            iv_nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            print("[interventions] spaCy en_core_web_sm unavailable: entity types from surface rules only", flush=True)
        # Counterfactual entities come from calibration questions (training split or held-out calibration) only.
        # Their golds build the pool; they are never used for calibration, and evaluation golds never enter it.
        substitute_pool = interventions.substitute_pool(calib, iv_nlp)

    # Fingerprints of every cache this configuration was calibrated from (a resumed or sharded run must match them).
    cache_fingerprints = {
        "relevance": runio.digest([relevance_fingerprint(e) for e in calib + evals]),
        "ce": (runio.digest([rerank.ce_fingerprint(e, paths.RERANKER) for e in ce_questions])
               if ce_table is not None else None),
        "ucal": runio.digest([ucal_fp[e["qid"]] for e in calib]),
        "seakr": runio.digest([seakr_fingerprint(e) for e in calib]) if args.seakr_samples else None,
    }
    # Fingerprint of the evaluated questions and their candidate lists: analysis scripts pair or pool runs only
    # when these agree (paper_analyses checks it), so a stale candidate set cannot slip into a table.
    candidate_sha = runio.digest([[e["qid"], [c["cid"] for c in e["candidates"]]] for e in evals])
    config_text = runio.dumps_strict({
        **{k: v for k, v in vars(args).items() if k not in EXECUTION_ARGS},
        "norm_pre": vars(norm_pre), "norm_post": vars(norm_post),
        "norm_probe": vars(calibrators["probe"][1]) if "probe" in calibrators else None,
        **{f"comb_{ph}": calibrators[ph][0].to_json() for ph in calibrators},
        "variants": [vars(v) for v in variants], "model_path": str(model_path),
        "max_new_tokens": max_new, "seakr_threshold_used": seakr_threshold,
        "seakr_missing_calibration": seakr_missing_calibration,
        "intervention_entity_types": ("spacy+rules" if iv_nlp else "rules") if args.intervention else None,
        "nte_threshold": nte_threshold, "nte_norm": vars(nte_norm), "u_retrieval_rate_calibration": u_rate,
        "evaluation_questions": len(evals), "candidate_sha256": candidate_sha,
        "split_sizes": {"universe": len(examples), "calib": len(calib), "dev": len(sp["dev"]), "test": len(sp["test"])},
        "dropped_for_missing_assets": dropped,
        "candidate_shortfall": shortfall,
        "evaluation_qids_sha256": runio.qid_set_sha256([e["qid"] for e in evals]),
        "split_qids_sha256": {part: runio.qid_set_sha256(e["qid"] for e in sp[part])
                              for part in ("calib", "dev", "test")},
        **run_instruction,
        "cache_fingerprints": cache_fingerprints,
        "protocol_differences": baseline_arms.PROTOCOL_DIFFERENCES,
    }, indent=1)
    if config_path.exists() and (has_outputs or args.num_shards > 1):
        saved = json.loads(config_path.read_text(encoding="utf8"))
        current = json.loads(config_text)
        differ = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        if differ:
            raise SystemExit(f"saved answers were produced with a different {', '.join(differ)}; "
                             f"use a new --tag (configuration: {config_path})")
    if writes_caches:
        runio.atomic_write_text(config_path, config_text)
    if args.calibrate_only:
        print(f"[calibrate-only] caches and {config_path} ready", flush=True)
        return

    t_start = time.time()
    # A half-written last line of an interrupted run is cut off before the questions it held are answered again.
    done = {r["qid"] for r in runio.load_partial_jsonl(out_path)}
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
                   "golds": e["golds"], "answer_format": run_instruction["answer_format"],
                   "preds": {}}
            # Pre-retrieval uncertainty (paper Eq. 4): prefill without retrieved evidence.
            # Every compared arm is decoded here from the same prompt; no cached baselines.
            timing["probe"] = 0.0
            t_direct = time.perf_counter()
            enc_d = qwen.encode(*pipeline.build_segments(e, None, args.answer_style))
            pre_info = {}

            def gate_off(attn_d, logits_d):
                pre = pipeline.raw_uncertainty(attn_d, enc_d, pipeline.context_ids(e), logits_d, args.beta,
                                               qwen.vocab_size)
                pre_info.update(raw_pre=pre, u_pre=u_of("pre", pre))
                if args.modality_routing:
                    pre_info["route"] = routing.initial_route(attn_d, enc_d, e, args.route_rule)
                return True

            res_d, _, _ = qwen.run(enc_d, None, max_new, decode_static=gate_off)
            direct_prefill = qwen.last_prefill_seconds
            direct_s = time.perf_counter() - t_direct          # encode + gate + prefill + direct decode
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

            t_route, probe_before = time.perf_counter(), timing["probe"]
            route = resolve_route(e, rel, pre_info.get("route"), probe_out=probe_out)
            chunks = select_chunks(e, rel, args.k, args.order, args.seed, args.min_per_modality, route,
                                   ce_of and ce_of(e))
            route_s = time.perf_counter() - t_route - (timing["probe"] - probe_before)   # route + index lookup
            # Joint input: the same top-k over all candidates, with no probe and no route. The
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
            if "cluster_id" in e:     # VideoRAG: the source video (clustered bootstrap, universalrag_metrics)
                rec["cluster_id"] = e["cluster_id"]
            # whether the prompt text of the chunk states a gold answer (error analysis: biased chunk without it)
            rec["chunk_has_answer"] = [bool(interventions.answer_span(interventions.prompt_text(c), e["golds"]))
                                       for c in chunks]
            rec["gold_in_topk"] = any(c.get("gold") for c in chunks) if any(
                c.get("gold") is not None for c in e["candidates"]) else None
            if routed:
                rec["route"] = route
            if "gold_modalities" in e:
                rec["gold_modalities"] = e["gold_modalities"]
            t_enc = time.perf_counter()
            segments, split = evidence_segments(e, chunks, rel)
            enc = qwen.encode(segments, split)
            encode_s = time.perf_counter() - t_enc
            if args.answer_style == "cite":
                rec["citation_sources"] = pipeline.citation_sources(segments, chunks)
            rel_sel = {c["cid"]: eab_rel[c["cid"]] for c in chunks}
            info = {}

            def make_variants(attn, logits):
                post = pipeline.raw_uncertainty(attn, enc, list(rel_sel), logits, args.beta, qwen.vocab_size)
                u = u_of("post", post)
                u_eab = rec["u_gate"] if args.eab_uncertainty == "pre" else u
                info.update(raw_post=post, u_post=u, u_eab=u_eab, mass=pipeline.chunk_masses(attn, enc))
                specs = variant_specs(e, enc, rel_sel, u_eab)
                info["eab_active"] = {name: s is not None for name, s in specs.items()}
                return specs

            res, _, _ = qwen.run(enc, make_variants, max_new, decode_static=True)
            rec["prefill_seconds"] = {"direct": direct_prefill, "evidence": qwen.last_prefill_seconds,
                                      **({"probe": probe_out["seconds"]} if probe_out else {})}
            # Measured stages on the question's paths (paper latency): analysis adds the stages of each arm's
            # actual path. direct = encode + gate + direct prefill + direct decode; probe = every probe round of the
            # question (cold start or first-round routing); route = route + index lookup; encode = evidence packing.
            rec["seconds_stage"] = {"direct": direct_s, "probe": timing["probe"], "route": route_s, "encode": encode_s}
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
                # Copies per chunk, aligned with rec["chunks"]; the prompt keeps every copy (no truncation).
                counts = order_mod.replication_counts([rel_sel[c["cid"]] for c in chunks], info["u_eab"],
                                                      info["u_eab"] > args.tau_bias, args.replication_gamma)
                rec["replication_counts"] = counts
                if all(n == 1 for n in counts):
                    rec["preds"]["replication"] = rec["preds"]["static"]
                    rec["replication_prompt_tokens"] = enc.length      # no copies: the static prompt itself
                    if args.answer_style == "cite":
                        rec["replication_citation_sources"] = rec["citation_sources"]
                else:
                    copies = replication_copies(chunks, counts)
                    seg_r, split_r = evidence_segments(e, copies, rel)
                    enc_r = qwen.encode(seg_r, split_r)
                    res_r, _, _ = qwen.run(enc_r, None, max_new, decode_static=True)
                    rec["preds"]["replication"] = res_r["static"].text
                    rec["tokens"]["replication"] = res_r["static"].tokens
                    rec["replication_prompt_tokens"] = enc_r.length
                    if args.answer_style == "cite":   # copies are numbered [E1..] in their own prompt
                        rec["replication_citation_sources"] = pipeline.citation_sources(seg_r, copies)
                arms.append("replication")
            # Baseline arms read the joint input. Wall time: everything each method runs itself.
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
                    qwen, e, joint, rel, max_new, instruction=run_instruction["instruction"])
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
            if args.v2:
                # static_mixed reuses the joint decode above; rerank_static ranks by cross-encoder logit (--v2-reranker)
                chunks_by_action = v2_action_chunks(e, rel, joint, ce_table[e["qid"]], args.k, args.seed)
                rec["actions"] = decode_v2_actions(qwen, e, rel, chunks_by_action, (res_d["static"], enc_d),
                                                   {tuple(rec["chunks_joint"]): (res_j["static"], enc_j)}, max_new,
                                                   args.answer_style, args.beta)
                rec["candidate_scores"] = candidate_scores(e, rel, ce_table[e["qid"]])
                if i < args.trace_prompts:
                    runio.append_jsonl(prompts_path, prompt_sample(e, chunks_by_action, args.answer_style))
                if deploy_decision is not None:
                    # The deployment path of the frozen rule: prefill what it reads, decode what it chooses. The
                    # full decode of every action above stays the reference; the two must agree.
                    deployed = deploy_v2(qwen, e, rel, chunks_by_action, deploy_decision, max_new, args.answer_style,
                                         args.beta)
                    full = arbitration.apply_decision(deploy_decision, rec["actions"])
                    deployed["matches_full_decode"] = (deployed["action"] == full["action"]
                                                       and deployed["answer"] == full["answer"])
                    rec["v2_deploy"] = deployed
            # preds keep the model's raw output; the parse (answer, answer_empty, citation_only) is stored beside it
            answer_first = rec["answer_format"] == "answer_first"
            rec["parsed"] = {name: answers.parse_answer(text, answer_first) for name, text in rec["preds"].items()}
            rec["seconds"] = round(time.time() - t0, 2)
            fh.write(runio.dumps_strict(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(evals)}] {e['qid'][:12]} {rec['seconds']}s P={enc.length} "
                  f"U_pre={rec['u_pre']:.2f} retrieve={rec['retrieve']}", flush=True)
    notes = []
    if args.num_shards == 1:
        # A single process owns every question: mark the file complete only if all of them are in it.
        written = runio.load_partial_jsonl(out_path)
        answered = [r["qid"] for r in written]
        if sorted(answered) != sorted(e["qid"] for e in evals):
            raise SystemExit(f"{out_path} holds {len(answered)} answers for {len(evals)} evaluation questions")
        empty_report = answers.summarize_empty([k for row in written for k in answers.question_empty_kinds(row)])
        if empty_report.get("empty_answer_warning"):
            print(empty_report["empty_answer_warning"], flush=True)
        marker = runio.write_completion_marker(out_dir, args.dataset, len(evals), out_path, config_path,
                                               extra=empty_report)
        notes.append(f"empty_answer_rate {empty_report['empty_answer_rate']:.6f}")
        notes.append(f"refusal_rate {empty_report['refusal_rate']:.6f}")
        if args.max_empty_rate is not None and empty_report["empty_answer_rate"] > args.max_empty_rate:
            raise SystemExit(f"empty answer rate {empty_report['empty_answer_rate']:.6f} exceeds --max-empty-rate "
                             f"{args.max_empty_rate} ({empty_report['n_empty']} of {empty_report['n_answers']} "
                             f"decoded strings are blank or citation-only; {empty_report['n_refusal']} refusals are "
                             f"reported apart); outputs and {marker.name} were written")
    if args.route_rule == "query":
        notes.append(f"universalrag_fallback {tallies['universalrag_fallback']}")
    if args.seakr_samples:
        notes.append(f"seakr_missing {tallies['seakr_missing']}")
    if torch.cuda.is_available():
        notes.append(f"peak_gpu_memory_gib {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}")
    print(f"[done] {args.dataset} {len(evals)} in {time.time() - t_start:.1f}s -> {out_path}"
          + "".join(f"; {note}" for note in notes), flush=True)


if __name__ == "__main__":
    main()

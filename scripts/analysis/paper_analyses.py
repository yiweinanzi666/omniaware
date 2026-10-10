"""Aggregators for the paper's analysis tables, from run files and their official scores.

Every input is a run file ``runs/<tag>/<dataset>.jsonl`` written by ``omniaware.run`` and scored by
``omniaware.eval.score_official`` (``<dataset>.official_scores.jsonl`` next to it). ``--metric`` picks the
per-question official score (0-100); a question is *correct* when its score >= ``--correct-at`` (default 50).

    order          order sensitivity: std over the 12 seeded permutations, sorted - reverse gap, flip rate
                   (one run per --order perm:<id>, all 14 schedule entries)
    uncertainty    AURC and Spearman rho of U_pre, next-token entropy, H_grd and SeaKR scores vs answer error
    ard            mean attention-relevance divergence per arm
    interventions  Mask / CF / random-edit / rerun change rates, Misled (injection) on eligible questions
    gold25         score on questions whose top chunk is not gold while a gold fact is at ranks 2-5
    noise          score per arm averaged over the noise draws of one sigma (one run per draw)
    latency        mean end-to-end ms: always-retrieve arms vs the U-gated system
    stats          dataset statistics: evidence tokens per modality, relevance spread r1 - r5, retrieval and
                   EAB activation rates, single-token answers and first-token changes
    selective      AURC / rho with every system scored on its own answers (U -> gated EAB, NTE, SeaKR)
    agreement      NTE vs U trigger agreement and the scores on skipped / retrieved questions
    transitions    wrong->correct and correct->wrong shares between two arms (--arms before after)
    paired         paired bootstrap across two runs: <baseline run> <our run> --arms <baseline arm> <our arm>
    ranges         raw retriever score ranges per channel (pass cache/relevance_<dataset>*.json)
    errors         rule-based pre-labelling of an arm's errors into the paper's four categories, plus a
                   seeded sample for manual verification

    python -m scripts.analysis.paper_analyses order runs/perm_*/webqa_img.jsonl --metric acc_approx \
        --arms static gated:eab:g1.2:ans2:pos:gate
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import analysis, interventions, order as order_mod  # noqa: E402


# ------------------------------------------------------------------ loading --
def load_run(path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    config_path = Path(path).with_name(Path(path).stem + "_config.json")
    return rows, (json.loads(config_path.read_text()) if config_path.exists() else {})


def load_scores(path: Path, metric: str) -> dict[str, dict[str, float]]:
    """qid -> arm -> official score of ``metric``."""
    scores_path = Path(path).with_name(Path(path).stem + ".official_scores.jsonl")
    if not scores_path.exists():
        raise SystemExit(f"score the run first: python -m omniaware.eval.score_official {path}")
    out = {}
    for line in scores_path.read_text().splitlines():
        d = json.loads(line)
        out[d["qid"]] = {arm: s[metric] for arm, s in d["scores"].items()}
    return out


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None


# -------------------------------------------------------------------- order --
def order_table(runs: list[Path], metric: str, arms: list[str], correct_at: float, baseline: str | None) -> dict:
    by_order = {}
    for path in runs:
        _, config = load_run(path)
        order = config.get("order", "")
        if not order.startswith("perm:"):
            raise SystemExit(f"{path}: not a --order perm:<id> run")
        by_order[order[len("perm:"):]] = load_scores(path, metric)
    schedule = order_mod.permutation_schedule()
    missing = [o["order_id"] for o in schedule if o["order_id"] not in by_order]
    if missing:
        raise SystemExit(f"missing order runs: {missing}")
    qids = sorted(set.intersection(*(set(s) for s in by_order.values())))
    random_ids = [o["order_id"] for o in schedule if o["kind"] == "random"]
    out = {"questions": len(qids), "orders": random_ids, "arms": {}}
    matrices = {}
    for arm in arms:
        m = np.array([[by_order[o][q][arm] / 100 for o in random_ids] for q in qids])   # question x order
        matrices[arm] = m
        means = 100 * m.mean(axis=0)
        per_seed = [float(np.std(means[[i for i, o in enumerate(random_ids) if o.startswith(f"s{s}-")]]))
                    for s in order_mod.PERMUTATION_SEEDS]
        correct = 100 * m >= correct_at
        sorted_mean = mean([by_order["sorted"][q][arm] for q in qids])
        reverse_mean = mean([by_order["reverse"][q][arm] for q in qids])
        out["arms"][arm] = {"mean": float(means.mean()), "std": float(np.std(means)),
                            "std_per_seed_mean": float(np.mean(per_seed)), "std_per_seed": per_seed,
                            "sorted": sorted_mean, "reverse": reverse_mean, "sorted_minus_reverse": sorted_mean - reverse_mean,
                            "flip_rate": float(100 * np.mean(correct.any(axis=1) & ~correct.all(axis=1)))}
    if baseline and baseline in matrices:
        out["std_vs_baseline"] = {arm: order_mod.order_std_statistics(matrices[baseline], matrices[arm])
                                  for arm in arms if arm != baseline}
    return out


# -------------------------------------------------------------- uncertainty --
UNCERTAINTIES = {
    "U_omni": lambda r: r.get("u_gate", r.get("u_pre")),
    "U_pre": lambda r: r.get("u_pre"),
    "U_probe": lambda r: r.get("u_probe"),
    "NTE": lambda r: (r.get("raw_pre") or {}).get("h_sem"),
    "H_grd": lambda r: (r.get("raw_pre") or {}).get("h_grd"),
    "SeaKR": lambda r: (r.get("seakr") or {}).get("direct_eigen"),
}


def uncertainty_table(path: Path, metric: str, arm: str) -> dict:
    rows, _ = load_run(path)
    scores = load_scores(path, metric)
    out = {"arm": arm}
    for name, get in UNCERTAINTIES.items():
        pairs = [(get(r), 1 - scores[r["qid"]][arm] / 100) for r in rows if get(r) is not None]
        if len(pairs) < 2:
            continue
        u, err = zip(*pairs)
        out[name] = {"n": len(pairs), "AURC": 100 * analysis.aurc(u, err), "spearman": analysis.spearman(u, err)}
    return out


# ---------------------------------------------------------------------- ARD --
def ard_table(path: Path) -> dict:
    rows, _ = load_run(path)
    values = defaultdict(list)
    for r in rows:
        for arm, v in (r.get("ard") or {}).items():
            values[arm].append(v)
    return {arm: {"n": len(v), "ARD": mean(v)} for arm, v in values.items()}


# ------------------------------------------------------------ interventions --
def intervention_table(path: Path) -> dict:
    rows, _ = load_run(path)
    eligible = [r for r in rows if (r.get("interventions") or {}).get("eligible")]
    out = {"questions": len(rows), "eligible": len(eligible), "kinds": {}}
    kinds = sorted({k for r in eligible for k in r["interventions"] if k in interventions.KINDS})
    for kind in kinds:
        per_arm = defaultdict(lambda: Counter())
        for r in eligible:
            entry = r["interventions"].get(kind)
            if not entry:
                continue
            entity = r["interventions"]["entity"]
            for arm, pred in entry["preds"].items():
                c = per_arm[arm]
                c["n"] += 1
                c["changed"] += interventions.normalize(pred) != interventions.normalize(r["preds"][arm])
                if entity and kind in ("cf", "random_edit", "inject_answer", "inject_instruction"):
                    c["follows"] += interventions.matches(pred, entity)
        rates = {}
        for arm, c in per_arm.items():
            rates[arm] = {"n": c["n"], "changed": 100 * c["changed"] / c["n"]}
            if kind in ("cf", "random_edit"):
                rates[arm]["follows_entity"] = 100 * c["follows"] / c["n"]
            if kind.startswith("inject"):
                rates[arm]["misled"] = 100 * c["follows"] / c["n"]
        out["kinds"][kind] = rates
    return out


def gold25_table(path: Path, metric: str) -> dict:
    """Gold@2-5 under rank order: chunks are listed by relevance, so position = rank."""
    rows, config = load_run(path)
    if config.get("order", "rank") != "rank":
        raise SystemExit("Gold@2-5 needs a rank-order run")
    scores = load_scores(path, metric)
    subset = [r for r in rows if r.get("chunk_gold") and r["chunk_gold"][0] is False and any(r["chunk_gold"][1:5])]
    arms = sorted(scores[rows[0]["qid"]]) if rows else []
    return {"questions": len(rows), "gold_2_5": len(subset),
            "scores": {arm: mean([scores[r["qid"]][arm] for r in subset]) for arm in arms}}


def noise_table(runs: list[Path], metric: str) -> dict:
    by_sigma = defaultdict(list)
    for path in runs:
        _, config = load_run(path)
        by_sigma[config.get("relevance_noise", 0.0)].append(path)
    out = {}
    for sigma, paths in sorted(by_sigma.items()):
        draws = []
        for path in paths:
            scores = load_scores(path, metric)
            arms = next(iter(scores.values())).keys()
            draws.append({arm: mean([s[arm] for s in scores.values()]) for arm in arms})
        out[str(sigma)] = {"draws": len(draws),
                           "scores": {arm: mean([d[arm] for d in draws]) for arm in draws[0]},
                           "std_over_draws": {arm: statistics.pstdev([d[arm] for d in draws]) for arm in draws[0]}}
    return out


# ------------------------------------------------------------------ latency --
def latency_table(path: Path) -> dict:
    """Generation-side end-to-end ms (retrieval lookup is cached and excluded):
    always-retrieve arm = evidence prefill + arm decode;
    gated arm = direct prefill (computes U) [+ probe prefill at cold start] + (retrieve ? evidence prefill
    + arm decode : direct decode);
    universalrag = (router says retrieve ? evidence prefill + static decode : direct prefill + direct decode),
    router time excluded; FITM, M2-RAG, SeaKR = wall time of everything the method runs itself."""
    rows, _ = load_run(path)
    rows = [r for r in rows if "arm_seconds" in r]
    arms = sorted({a for r in rows for a in r["arm_seconds"] if a != "direct"})
    out = {"questions": len(rows), "retrieval_rate": 100 * mean([float(r["retrieve"]) for r in rows]) if rows else None,
           "prefill_direct_ms": 1000 * mean([r["prefill_seconds"]["direct"] for r in rows]),
           "prefill_probe_ms": (1000 * mean([r["prefill_seconds"]["probe"] for r in rows if "probe" in r["prefill_seconds"]])
                                if any("probe" in r["prefill_seconds"] for r in rows) else None),
           "direct_only": 1000 * mean([r["prefill_seconds"]["direct"] + r["arm_seconds"]["direct"] for r in rows])}
    routed = [r for r in rows if (r.get("route") or {}).get("rule") == "query"]
    if routed:
        out["universalrag"] = 1000 * mean([
            r["prefill_seconds"]["evidence"] + r["arm_seconds"]["static"] if r["route"]["retrieve"]
            else r["prefill_seconds"]["direct"] + r["arm_seconds"]["direct"] for r in routed])
    for name in sorted({b for r in rows for b in (r.get("baseline_seconds") or {})}):
        out[name] = 1000 * mean([r["baseline_seconds"][name] for r in rows if name in (r.get("baseline_seconds") or {})])
    for arm in arms:
        out[f"always:{arm}"] = 1000 * mean([r["prefill_seconds"]["evidence"] + r["arm_seconds"][arm] for r in rows])
        out[f"gated:{arm}"] = 1000 * mean([
            r["prefill_seconds"]["direct"] + r["prefill_seconds"].get("probe", 0.0)
            + (r["prefill_seconds"]["evidence"] + r["arm_seconds"][arm]
                                              if r["retrieve"] else r["arm_seconds"]["direct"]) for r in rows])
    return out


# -------------------------------------------------------------------- stats --
def stats_table(path: Path, arms: list[str]) -> dict:
    rows, config = load_run(path)
    tokens = defaultdict(list)
    for r in rows:
        per_q = Counter()
        for _, (modality, n) in (r.get("span_tokens") or {}).items():
            per_q[modality] += n
        for modality, n in per_q.items():
            tokens[modality].append(n)
    spread = [sorted(r["rel"].values(), reverse=True) for r in rows if len(r.get("rel") or {}) >= 5]
    out = {"questions": len(rows), "prompt_tokens": mean([r.get("prompt_tokens") for r in rows]),
           "evidence_tokens_per_question": {m: {"mean": mean(v), "median": statistics.median(v)}
                                            for m, v in tokens.items()},
           "relevance_r1": mean([s[0] for s in spread]), "relevance_r5": mean([s[4] for s in spread]),
           "relevance_r1_minus_r5": mean([s[0] - s[4] for s in spread]),
           "retrieval_rate": 100 * mean([float(r["retrieve"]) for r in rows]) if rows else None,
           "gold_in_topk": mean([float(r["gold_in_topk"]) for r in rows if r.get("gold_in_topk") is not None]),
           "eab_active": {}, "single_token_answers": {}, "first_token_changed_vs_static": {}}
    variant_names = [v["name"] for v in config.get("variants", [])]
    for v in variant_names:
        act = [r["eab_active"][v] for r in rows if v in (r.get("eab_active") or {})]
        out["eab_active"][v] = 100 * mean([float(a) for a in act]) if act else None
    for arm in arms:
        toks = [r["tokens"][arm] for r in rows if arm in (r.get("tokens") or {})]
        out["single_token_answers"][arm] = 100 * mean([float(len(t) == 1) for t in toks]) if toks else None
        if arm != "static":
            pairs = [(r["tokens"][arm], r["tokens"]["static"]) for r in rows
                     if arm in r.get("tokens", {}) and "static" in r.get("tokens", {})]
            out["first_token_changed_vs_static"][arm] = (
                100 * mean([float(a[:1] != b[:1]) for a, b in pairs]) if pairs else None)
    return out


# ------------------------------------------------------------------- errors --
CATEGORIES = ("missing_topk_evidence", "answer_normalization", "unsupported_citation",
              "ambiguous_visual_grounding", "other")


def categorize(r: dict, arm: str, score: float, faith: float | None, correct_at: float) -> str | None:
    if score >= correct_at:
        return None
    pred = r["preds"].get(arm) or ""
    if any(interventions.matches(pred, g) for g in r["golds"] if g):
        return "answer_normalization"           # gold string present, official metric still fails
    if r.get("gold_in_topk") is False:
        return "missing_topk_evidence"
    if faith is not None and faith == 0:
        return "unsupported_citation"
    if r.get("query_images") or "image" in {m for m, _ in (r.get("span_tokens") or {}).values()}:
        return "ambiguous_visual_grounding"
    return "other"


def errors_table(path: Path, metric: str, arm: str, correct_at: float, attribution: Path | None,
                 sample: int, seed: int) -> dict:
    rows, _ = load_run(path)
    scores = load_scores(path, metric)
    faith = {}
    if attribution:
        details = json.loads(attribution.read_text())["details"].get(arm, [])
        faith = {d["qid"]: d["faith"] for d in details}
    labelled = []
    for r in rows:
        cat = categorize(r, arm, scores[r["qid"]][arm], faith.get(r["qid"]), correct_at)
        if cat:
            labelled.append({"qid": r["qid"], "category": cat, "question": r["question"], "golds": r["golds"],
                             "pred": r["preds"][arm], "chunks": r.get("chunks")})
    counts = Counter(x["category"] for x in labelled)
    n = len(labelled)
    picked = random.Random(seed).sample(labelled, min(sample, n))
    return {"arm": arm, "errors": n, "shares": {c: 100 * counts[c] / n if n else None for c in CATEGORIES},
            "rule_note": "rule-based pre-labels; verify the sample manually before reporting",
            "sample_for_manual_check": picked}


# ---------------------------------------------------------- selective risk --
def selective_table(paths_: list[Path], metric: str, eab_arm: str) -> dict:
    """Paper appendix 'Selective risk': every system is scored on its own answers.
    U_omni -> gated:<EAB>, NTE -> nte_trigger:static, SeaKR -> seakr (from the run(s) that hold them)."""
    systems = {"U_omni": (lambda r: r.get("u_gate", r.get("u_pre")), f"gated:{eab_arm}"),
               "NTE": (lambda r: (r.get("raw_pre") or {}).get("h_sem"), "nte_trigger:static"),
               "SeaKR": (lambda r: (r.get("seakr") or {}).get("direct_eigen"), "seakr")}
    out = {}
    for path in paths_:
        rows, _ = load_run(path)
        scores = load_scores(path, metric)
        for name, (get, arm) in systems.items():
            if name in out:
                continue
            pairs = [(get(r), 1 - scores[r["qid"]][arm] / 100) for r in rows
                     if get(r) is not None and arm in scores[r["qid"]] and math.isfinite(get(r))]
            if len(pairs) >= 2:
                u, err = zip(*pairs)
                out[name] = {"run": str(path), "arm": arm, "n": len(pairs), "AURC": 100 * analysis.aurc(u, err),
                             "spearman": analysis.spearman(u, err)}
    return out


# --------------------------------------------------------- trigger agreement --
def agreement_table(path: Path, metric: str, eab_arm: str) -> dict:
    """Appendix 'Trigger agreement': NTE (its matched threshold) against the U gate on the same questions."""
    rows, config = load_run(path)
    scores = load_scores(path, metric)
    thr = config["nte_threshold"]
    nte = [r["raw_pre"]["h_sem"] > thr for r in rows]
    u = [bool(r["retrieve"]) for r in rows]
    n = len(rows)

    def score(arm, mask):
        return mean([scores[r["qid"]][arm] for r, m in zip(rows, mask) if m])
    return {"questions": n,
            "both_retrieve": 100 * sum(a and b for a, b in zip(nte, u)) / n,
            "both_skip": 100 * sum(not a and not b for a, b in zip(nte, u)) / n,
            "only_nte": 100 * sum(a and not b for a, b in zip(nte, u)) / n,
            "only_u": 100 * sum(b and not a for a, b in zip(nte, u)) / n,
            "no_retrieval_score": {"NTE": score("direct", [not a for a in nte]), "U": score("direct", [not b for b in u])},
            "retrieved_score": {"NTE (static)": score("static", nte), "U (EAB)": score(eab_arm, u)}}


# --------------------------------------------------------------- transitions --
def transitions_table(path: Path, metric: str, before: str, after: str, correct_at: float) -> dict:
    """Error analysis: share of questions turned wrong->correct and correct->wrong from ``before`` to ``after``."""
    rows, _ = load_run(path)
    scores = load_scores(path, metric)
    b = [scores[r["qid"]][before] >= correct_at for r in rows]
    a = [scores[r["qid"]][after] >= correct_at for r in rows]
    n = len(rows)
    losses = Counter()
    for r, x, y in zip(rows, b, a):
        if x and not y and r.get("rel"):
            chunks = r.get("chunks") or list(r["rel"])
            top = max(range(len(chunks)), key=lambda j: r["rel"][chunks[j]])
            modality = (r.get("chunk_modalities") or ["text"] * len(chunks))[top]
            if modality == "image":
                losses["top_chunk_image"] += 1
            elif modality == "table":
                losses["top_chunk_table"] += 1
            elif modality == "video":
                losses["top_chunk_video"] += 1
            elif r.get("chunk_has_answer") and not r["chunk_has_answer"][top]:
                losses["top_text_chunk_without_answer"] += 1
            else:
                losses["other"] += 1
    n_loss = sum(losses.values())
    return {"questions": n, "before": before, "after": after,
            "wrong_to_correct": 100 * sum((not x) and y for x, y in zip(b, a)) / n,
            "correct_to_wrong": 100 * sum(x and not y for x, y in zip(b, a)) / n,
            "losses_by_top_biased_chunk": {k: 100 * v / n_loss for k, v in losses.items()} if n_loss else {}}


# ------------------------------------------------------------ significance --
def paired_test(a_path: Path, a_arm: str, b_path: Path, b_arm: str, metric: str) -> dict:
    """Paired bootstrap of b_arm (run b) against a_arm (run a) on the questions both runs answered; across runs,
    e.g. the controller against SeaKR or UniversalRAG. VideoRAG questions are clustered by source video."""
    sys.path.insert(0, str(ROOT / "src"))
    from omniaware.eval.score_official import paired
    sa, sb = load_scores(a_path, metric), load_scores(b_path, metric)
    qids = sorted(set(sa) & set(sb))
    clusters = [q.rsplit("_", 1)[0] for q in qids] if all(q.startswith("videorag") for q in qids) else None
    result = paired([sa[q][a_arm] for q in qids], [sb[q][b_arm] for q in qids], clusters)
    return {"questions": len(qids), "baseline": f"{a_path}:{a_arm}", "ours": f"{b_path}:{b_arm}", **result}


# ---------------------------------------------------------- raw score ranges --
def score_ranges(cache: Path) -> dict:
    """Raw retriever scores per channel from a relevance cache (cache/relevance_<dataset>*.json)."""
    raw = json.loads(Path(cache).read_text())
    per = defaultdict(list)
    for entry in raw.values():
        # Entries are {"fingerprint", "scores"} since the relevance cache became fingerprinted; older files are
        # {cid: channels} directly.
        chunks = entry["scores"] if "scores" in entry else entry
        for channels in chunks.values():
            for k, v in channels.items():
                per[k].append(v)
    return {k: {"n": len(v), "min": float(np.min(v)), "p5": float(np.percentile(v, 5)),
                "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}
            for k, v in per.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", choices=["order", "uncertainty", "selective", "agreement", "transitions", "paired", "ard",
                                         "interventions", "gold25", "noise", "latency", "stats", "ranges", "errors"])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--metric", default=None, help="official per-question metric (e.g. acc_approx, accuracy)")
    ap.add_argument("--arms", nargs="+", default=["static"])
    ap.add_argument("--baseline", default="static", help="order: arm whose std the others are compared to")
    ap.add_argument("--correct-at", type=float, default=50.0)
    ap.add_argument("--attribution", type=Path, default=None, help="errors: <run>.attribution.json")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args(argv)
    needs_metric = {"order", "uncertainty", "selective", "agreement", "transitions", "paired", "gold25", "noise",
                    "errors"}
    if args.analysis in needs_metric and not args.metric:
        ap.error(f"{args.analysis} needs --metric")
    one = args.runs[0]
    if args.analysis == "order":
        result = order_table(args.runs, args.metric, args.arms, args.correct_at, args.baseline)
    elif args.analysis == "uncertainty":
        result = {str(p): uncertainty_table(p, args.metric, args.arms[0]) for p in args.runs}
    elif args.analysis == "selective":
        result = selective_table(args.runs, args.metric, args.arms[0])
    elif args.analysis == "agreement":
        result = {str(p): agreement_table(p, args.metric, args.arms[0]) for p in args.runs}
    elif args.analysis == "transitions":
        if len(args.arms) != 2:
            ap.error("transitions needs --arms <before> <after>")
        result = {str(p): transitions_table(p, args.metric, args.arms[0], args.arms[1], args.correct_at)
                  for p in args.runs}
    elif args.analysis == "paired":
        if len(args.runs) != 2 or len(args.arms) != 2:
            ap.error("paired needs two runs (baseline, ours) and --arms <baseline arm> <our arm>")
        result = paired_test(args.runs[0], args.arms[0], args.runs[1], args.arms[1], args.metric)
    elif args.analysis == "ranges":
        result = {str(p): score_ranges(p) for p in args.runs}
    elif args.analysis == "ard":
        result = {str(p): ard_table(p) for p in args.runs}
    elif args.analysis == "interventions":
        result = {str(p): intervention_table(p) for p in args.runs}
    elif args.analysis == "gold25":
        result = {str(p): gold25_table(p, args.metric) for p in args.runs}
    elif args.analysis == "noise":
        result = noise_table(args.runs, args.metric)
    elif args.analysis == "latency":
        result = {str(p): latency_table(p) for p in args.runs}
    elif args.analysis == "stats":
        result = {str(p): stats_table(p, args.arms) for p in args.runs}
    else:
        result = errors_table(one, args.metric, args.arms[0], args.correct_at, args.attribution,
                              args.sample, args.seed)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()

"""Aggregators for the paper's analysis tables, from run files and their official scores.

Every input is a run file ``runs/<tag>/<dataset>.jsonl`` written by ``omniaware.run`` and scored by
``omniaware.eval.score_official`` (``<dataset>.official_scores.jsonl`` next to it). ``--metric`` picks the
per-question official score (0-100); a question is *correct* when its score >= ``--correct-at`` (default 50, the
VQA threshold: a VQA score of 33.3 counts as wrong; answer_recall.py uses the same default).

Runs are only combined when they answer the same questions with the same configuration: qid sets must be equal
(no silent intersection), and the fields listed in STRICT_FIELDS / PAIR_FIELDS must agree (candidate fingerprint,
model, split, calibration). Restricting a run to a subset (e.g. the image questions) is explicit: --restrict-modality.

    order          order sensitivity: std over the 12 seeded permutations, sorted - reverse gap, flip rate and the
                   std across seeds (one run per --order perm:<id>, all 14 schedule entries)
    uncertainty    AURC and Spearman rho of U_pre, next-token entropy, H_grd and SeaKR scores vs answer error
    ard            mean attention-relevance divergence per arm (finite values only; n_valid and n_missing)
    interventions  Mask / CF / random-edit / rerun change rates, Misled (injection) with a fixed denominator
    gold25         score on questions whose top chunk is not gold while a gold fact is at ranks 2-5
    noise          score per arm averaged over the three noise draws of one sigma (qids aligned)
    latency        measured end-to-end ms per arm (see latency_table)
    stats          dataset statistics: evidence tokens per modality, relevance spread r1 - r5, retrieval and
                   EAB activation rates, single-token answers and first-token changes
    selective      AURC / rho with every system scored on its own answers (U and NTE from the main run, SeaKR from
                   the SeaKR run given by --seakr-run)
    agreement      NTE vs U trigger agreement and the scores on skipped / retrieved questions
    transitions    wrong->correct and correct->wrong shares between two arms (--arms before after)
    paired         paired bootstrap across two runs: <baseline run> <our run> --arms <baseline arm> <our arm>
    column         one table column: the controller against every --compare <run>:<arm> baseline, with Holm-adjusted
                   two-sided p-values (--restrict-modality for subset baselines). The strongest baseline is named only
                   when the caller passes it with --strongest <run>:<arm>; it is never chosen on the test column
    tau            gated score and retrieval rate for several tau values, offline from one run (the gate only decides
                   retrieve = u_gate > tau and every arm is decoded for every question); checked against the run
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
from omniaware import analysis, answers, interventions, order as order_mod  # noqa: E402

# Fields that must agree for runs to be aggregated (order, noise) or paired (across runs).
STRICT_FIELDS = ("dataset", "split", "model", "model_path", "calib_source", "n_calib", "split_seed", "n_dev", "k",
                 "answer_style", "variants", "replication_gamma", "relevance_norm", "norm_pre", "norm_post", "comb_pre",
                 "norm_probe", "evaluation_questions", "candidate_sha256", "max_new_tokens", "fitm", "m2rag",
                 "seakr_samples", "u_combine", "attn_layer", "route_rule", "selector", "beta", "tau", "tau_bias",
                 "no_calibration", "seed")
PAIR_FIELDS = ("split", "model", "calib_source", "n_calib", "split_seed", "k", "answer_style", "dataset",
               "candidate_sha256")


def answer_of(row: dict, arm: str) -> str:
    """Parsed answer of one arm of a run record. rec["preds"] keeps the raw model output, so the answer is
    rec["parsed"][arm]["answer"]; a record without that field (written before the parse was stored) is parsed here."""
    parsed = (row.get("parsed") or {}).get(arm)
    return parsed["answer"] if parsed is not None else answers.parse_answer(row["preds"][arm])["answer"]


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


def qid_set(rows: list[dict]) -> frozenset:
    return frozenset(r["qid"] for r in rows)


def same_config(configs: list[dict], fields=STRICT_FIELDS) -> None:
    """Every configuration must agree on ``fields``; the first disagreement is reported with its values."""
    if not all(configs):
        raise SystemExit("runs need their *_config.json (rerun with the current code)")
    for field in fields:
        values = [json.dumps(c.get(field), sort_keys=True) for c in configs]
        if len(set(values)) > 1:
            raise SystemExit(f"configurations disagree on {field}: {sorted(set(values))[:3]}")


def same_questions(named: list[tuple[str, frozenset]]) -> int:
    """Every run must answer exactly the same questions; returns the common n."""
    ref_name, ref = named[0]
    for name, qs in named[1:]:
        if qs != ref:
            raise SystemExit(f"{name} answers {len(qs)} questions, {ref_name} answers {len(ref)}; "
                             f"the symmetric difference has {len(qs ^ ref)} questions (no intersection is taken)")
    return len(ref)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# -------------------------------------------------------------------- order --
def order_table(runs: list[Path], metric: str, arms: list[str], correct_at: float, baseline: str | None) -> dict:
    by_order, configs, rows_by_order = {}, {}, {}
    for path in runs:
        rows, config = load_run(path)
        order = config.get("order", "")
        if not order.startswith("perm:"):
            raise SystemExit(f"{path}: not a --order perm:<id> run")
        key = order[len("perm:"):]
        by_order[key] = load_scores(path, metric)
        configs[key] = config
        rows_by_order[key] = qid_set(rows)
    schedule = order_mod.permutation_schedule()
    missing = [o["order_id"] for o in schedule if o["order_id"] not in by_order]
    if missing:
        raise SystemExit(f"missing order runs: {missing}")
    same_config([configs[o["order_id"]] for o in schedule])     # 'order' itself is not in STRICT_FIELDS
    n = same_questions([(o["order_id"], rows_by_order[o["order_id"]]) for o in schedule])
    qids = sorted(rows_by_order[schedule[0]["order_id"]])
    random_ids = [o["order_id"] for o in schedule if o["kind"] == "random"]
    out = {"questions": n, "orders": random_ids, "n_per_order": {o["order_id"]: n for o in schedule},
           "arms": {}}
    matrices = {}
    for arm in arms:
        m = np.array([[by_order[o][q][arm] / 100 for o in random_ids] for q in qids])   # question x order
        matrices[arm] = m
        means = 100 * m.mean(axis=0)
        per_seed = {s: [i for i, o in enumerate(random_ids) if o.startswith(f"s{s}-")]
                    for s in order_mod.PERMUTATION_SEEDS}
        seed_means = {s: float(means[idx].mean()) for s, idx in per_seed.items()}
        correct = 100 * m >= correct_at
        sorted_mean = mean([by_order["sorted"][q][arm] for q in qids])
        reverse_mean = mean([by_order["reverse"][q][arm] for q in qids])
        out["arms"][arm] = {"mean": float(means.mean()), "std": float(np.std(means)),
                            "std_per_seed_mean": float(np.mean([np.std(means[idx]) for idx in per_seed.values()])),
                            "std_per_seed": [float(np.std(means[idx])) for idx in per_seed.values()],
                            "std_across_seeds": float(np.std(list(seed_means.values()))),   # the paper's second '±'
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
    """Mean attention-relevance divergence per arm over finite values only. n_missing counts the questions
    where the arm has no value or a non-finite one."""
    rows, _ = load_run(path)
    arms = sorted({a for r in rows for a in (r.get("ard") or {})})
    out = {}
    for arm in arms:
        vals = [(r.get("ard") or {}).get(arm) for r in rows]
        valid = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        out[arm] = {"n_valid": len(valid), "n_missing": len(rows) - len(valid),
                    "ARD": mean(valid) if valid else None}
    return out


# ------------------------------------------------------------ interventions --
def _failed(entry) -> tuple[bool, str | None]:
    """An intervention entry counts as failed when it is None, has no predictions, or carries a 'failed' key."""
    if entry is None or not isinstance(entry, dict):
        return True, "no_entry"
    if "preds" not in entry or any("failed" in k for k in entry):
        return True, entry.get("failed_reason") or "unspecified"
    return False, None


def intervention_table(path: Path) -> dict:
    """Fixed denominator: every kind is over the same eligible questions. Failures are counted with their
    reason; 'changed' is given as a share of successful interventions and 'changed_of_eligible' counts a failure as
    unchanged. Eligibility is per arm when the run records eab_active."""
    rows, _ = load_run(path)
    eligible = [r for r in rows if (r.get("interventions") or {}).get("eligible")]
    out = {"questions": len(rows), "eligible": len(eligible), "kinds": {}}
    kinds = sorted({k for r in eligible for k in r["interventions"] if k in interventions.KINDS})
    for kind in kinds:
        per_arm = defaultdict(Counter)
        failures = Counter()
        success = 0
        for r in eligible:
            entry = r["interventions"].get(kind)
            failed, reason = _failed(entry)
            if failed:
                failures[reason] += 1
                continue
            success += 1
            entity = r["interventions"].get("entity")
            for arm, raw in entry["preds"].items():
                if (r.get("eab_active") or {}).get(arm) is False:
                    continue                        # this EAB arm was inactive on the question: not eligible for it
                answer = answers.parse_answer(raw)["answer"]
                c = per_arm[arm]
                c["n"] += 1
                c["changed"] += interventions.normalize(answer) != interventions.normalize(answer_of(r, arm))
                if entity and kind in ("cf", "random_edit", "inject_answer", "inject_instruction"):
                    c["follows"] += interventions.matches(answer, entity)
        attempted = len(eligible)
        rates = {}
        for arm, c in per_arm.items():
            rates[arm] = {"n": c["n"], "changed": 100 * c["changed"] / c["n"] if c["n"] else None,
                          "changed_of_eligible": 100 * c["changed"] / attempted if attempted else None}
            if kind in ("cf", "random_edit"):
                rates[arm]["follows_entity"] = 100 * c["follows"] / c["n"] if c["n"] else None
            if kind.startswith("inject"):
                rates[arm]["misled"] = 100 * c["follows"] / c["n"] if c["n"] else None
        out["kinds"][kind] = {"attempted": attempted, "success": success, "failed": sum(failures.values()),
                              "failure_reasons": dict(failures),
                              "failure_rate": 100 * sum(failures.values()) / attempted if attempted else None,
                              "arms": rates}
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
    """Noise draws: every sigma has exactly the draws {0, 1, 2}; configurations and questions must agree;
    scores are aligned by qid before the draws are averaged."""
    by_sigma = defaultdict(list)
    for path in runs:
        rows, config = load_run(path)
        by_sigma[config.get("relevance_noise", 0.0)].append((path, rows, config))
    out = {}
    for sigma, items in sorted(by_sigma.items()):
        draws = sorted(c.get("noise_draw") for _, _, c in items)
        if draws != [0, 1, 2]:
            raise SystemExit(f"sigma {sigma}: noise draws must be exactly 0, 1, 2 (found {draws})")
        same_config([c for _, _, c in items])
        n = same_questions([(str(p), qid_set(rows)) for p, rows, _ in items])
        per_draw = []
        for path, rows, _ in items:
            scores = load_scores(path, metric)
            arms = sorted(next(iter(scores.values())))
            per_draw.append({arm: mean([scores[r["qid"]][arm] for r in rows]) for arm in arms})
        arms = sorted(per_draw[0])
        out[str(sigma)] = {"draws": len(per_draw), "questions": n,
                           "scores": {arm: mean([d[arm] for d in per_draw]) for arm in arms},
                           "std_over_draws": {arm: statistics.pstdev([d[arm] for d in per_draw]) for arm in arms}}
    return out


# ------------------------------------------------------------------ latency --
def latency_table(path: Path) -> dict:
    """Measured end-to-end wall-clock per arm, in ms (paper latency). Nothing is assembled across runs: every
    stage is timed inside run.py with perf_counter on the question it belongs to, and an arm's time is the sum of
    the measured stages on that arm's own path:
      direct           encode + gate + direct prefill + direct decode                  (seconds_stage.direct)
      probe            every probe round of the question (cold start or first-round routing; seconds_stage.probe)
      retrieval(a)     route + index lookup + evidence encode + evidence prefill + decode of arm a
      gated:a          if retrieve: direct prefill (the gate needs it, not the direct decode) + probe + retrieval(a)
                       else: direct path + probe
      always:a         probe + retrieval(a)   (the routed static input, probe included when the route used it)
      universalrag     retrieve: route (index lookup) + encode + evidence prefill + static decode; else direct path.
                       The UniversalRAG router runs offline, so its own inference time is not in this number.
      static_joint     (when present) encode + evidence prefill + decode: no probe, no route
      baseline arms    their own measured wall-clock (baseline_seconds: FiTM, M2-RAG, SeaKR)
    Rows without seconds_stage (runs made before the timing fields) are an error."""
    rows, _ = load_run(path)
    if any("seconds_stage" not in r for r in rows):
        raise SystemExit(f"{path}: no measured seconds_stage (run made before the timing fields; rerun)")
    rows = [r for r in rows if "arm_seconds" in r]
    ms = lambda vals: 1000 * mean(vals)
    stage = lambda r: r["seconds_stage"]

    def retrieval(r, arm):
        s = stage(r)
        return s["route"] + s["encode"] + r["prefill_seconds"]["evidence"] + r["arm_seconds"][arm]

    def direct_path(r):
        return stage(r)["direct"]

    def probe(r):
        return stage(r)["probe"]

    arms = sorted({a for r in rows for a in r["arm_seconds"] if a != "direct"})
    out = {"questions": len(rows),
           "retrieval_rate": 100 * mean([float(r["retrieve"]) for r in rows]) if rows else None,
           "direct": ms([direct_path(r) for r in rows]),
           "probe": ms([probe(r) for r in rows]),
           "prefill_direct": ms([r["prefill_seconds"]["direct"] for r in rows])}
    routed = [r for r in rows if (r.get("route") or {}).get("rule") == "query"]
    if routed:
        out["universalrag"] = ms([
            retrieval(r, "static") if r["route"]["retrieve"] else direct_path(r) for r in routed])
    for name in sorted({b for r in rows for b in (r.get("baseline_seconds") or {})}):
        out[name] = ms([r["baseline_seconds"][name] for r in rows if name in (r.get("baseline_seconds") or {})])
    if "static_joint" in arms:
        out["static_joint"] = ms([stage(r)["encode"] + r["prefill_seconds"]["evidence"] + r["arm_seconds"]["static_joint"]
                                  for r in rows])
    for arm in arms:
        if arm == "static_joint":
            continue
        out[f"always:{arm}"] = ms([probe(r) + retrieval(r, arm) for r in rows])
        out[f"gated:{arm}"] = ms([
            (direct_path(r) - r["arm_seconds"]["direct"] + probe(r) + retrieval(r, arm)) if r["retrieve"]
            else direct_path(r) + probe(r) for r in rows])
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
    pred = answer_of(r, arm)
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
                             "pred": answer_of(r, arm), "chunks": r.get("chunks")})
    counts = Counter(x["category"] for x in labelled)
    n = len(labelled)
    picked = random.Random(seed).sample(labelled, min(sample, n))
    return {"arm": arm, "errors": n, "shares": {c: 100 * counts[c] / n if n else None for c in CATEGORIES},
            "rule_note": "rule-based pre-labels; verify the sample manually before reporting",
            "sample_for_manual_check": picked}


# ---------------------------------------------------------- selective risk --
def _risk(rows: list[dict], scores: dict, get, arm: str, path: Path) -> dict | None:
    pairs = [(get(r), 1 - scores[r["qid"]][arm] / 100) for r in rows
             if get(r) is not None and arm in scores[r["qid"]] and math.isfinite(get(r))]
    if len(pairs) < 2:
        return None
    u, err = zip(*pairs)
    return {"run": str(path), "arm": arm, "n": len(pairs), "AURC": 100 * analysis.aurc(u, err),
            "spearman": analysis.spearman(u, err)}


def selective_table(path: Path, seakr_path: Path, metric: str, eab_arm: str) -> dict:
    """Paper appendix 'Selective risk': every system is scored on its own answers. U_omni -> gated:<EAB> and NTE ->
    nte_trigger:static come from the main run; SeaKR -> seakr comes from the SeaKR run (each system from its own
    run, never from whichever file happens to come first)."""
    rows, _ = load_run(path)
    scores = load_scores(path, metric)
    out = {"U_omni": _risk(rows, scores, lambda r: r.get("u_gate", r.get("u_pre")), f"gated:{eab_arm}", path),
           "NTE": _risk(rows, scores, lambda r: (r.get("raw_pre") or {}).get("h_sem"), "nte_trigger:static", path)}
    s_rows, _ = load_run(seakr_path)
    out["SeaKR"] = _risk(s_rows, load_scores(seakr_path, metric),
                         lambda r: (r.get("seakr") or {}).get("direct_eigen"), "seakr", seakr_path)
    return {k: v for k, v in out.items() if v is not None}


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


# ----------------------------------------------------------------- transitions --
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


# ------------------------------------------------------------- significance --
def _clusters(qids: list[str]) -> list[str] | None:
    return [q.rsplit("_", 1)[0] for q in qids] if qids and all(q.startswith("videorag") for q in qids) else None


def _paired(sa: dict, sb: dict, qids: list[str], a_arm: str, b_arm: str, metric_name: str) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from omniaware.eval.score_official import paired
    return paired([sa[q][a_arm] for q in qids], [sb[q][b_arm] for q in qids], _clusters(qids))


def check_pairable(a_cfg: dict, b_cfg: dict, restricted: bool) -> None:
    """Cross-run pairs must share split, model, calibration, candidate fingerprint and dataset family. A restricted
    baseline (a subset of a wider run, e.g. the image questions) may differ in dataset name and candidate list."""
    if not a_cfg or not b_cfg:
        raise SystemExit("paired runs need their *_config.json (rerun with the current code)")
    fields = [f for f in PAIR_FIELDS if not (restricted and f in ("dataset", "candidate_sha256"))]
    for field in fields:
        if a_cfg.get(field) != b_cfg.get(field):
            raise SystemExit(f"runs cannot be paired: {field} differs ({a_cfg.get(field)!r} vs {b_cfg.get(field)!r})")
    if str(a_cfg.get("dataset", "")).split("_")[0] != str(b_cfg.get("dataset", "")).split("_")[0]:
        raise SystemExit("runs cannot be paired: different dataset families")


def paired_test(a_path: Path, a_arm: str, b_path: Path, b_arm: str, metric: str, restrict: str | None = None,
                subset: bool = False) -> dict:
    """Paired bootstrap of b_arm (run b) against a_arm (run a) on the same questions, e.g. the controller against
    SeaKR or UniversalRAG. Question sets must be equal; ``restrict`` keeps the baseline's questions with that gold
    modality (the explicit subset pairing); ``subset`` allows a smaller run to be paired with a wider one, using the
    smaller run's questions, and must be asked for explicitly. VideoRAG questions are clustered by source video."""
    a_rows, a_cfg = load_run(a_path)
    b_rows, b_cfg = load_run(b_path)
    check_pairable(a_cfg, b_cfg, restricted=restrict is not None)
    sa, sb = load_scores(a_path, metric), load_scores(b_path, metric)
    a_q = qid_set([r for r in a_rows if restrict is None or restrict in (r.get("gold_modalities") or [])])
    b_q = qid_set([r for r in b_rows if restrict is None or restrict in (r.get("gold_modalities") or [])])
    if a_q != b_q:
        if not subset or not (a_q <= b_q or b_q <= a_q):
            raise SystemExit(f"question sets differ ({len(a_q)} vs {len(b_q)}); pair subsets explicitly "
                             f"(--restrict-modality or --pair-subset), never by intersection")
    qids = sorted(a_q & b_q)
    result = _paired(sa, sb, qids, a_arm, b_arm, metric)
    return {"questions": len(qids), "baseline": f"{a_path}:{a_arm}", "ours": f"{b_path}:{b_arm}", **result}


def column_table(ours: Path, our_arm: str, metric: str, compare: list[str], restrict: str | None = None,
                 strongest: str | None = None) -> dict:
    """One table column: the controller against every baseline (``<run>:<arm>``), Holm-adjusted over the column.
    No baseline is named the strongest unless the caller passes ``strongest`` (a dev decision or a fixed choice):
    choosing it by the column's own scores would select on the test set. Holm adjusts over all baselines."""
    o_rows, o_cfg = load_run(ours)
    o_scores = load_scores(ours, metric)
    o_q = qid_set(o_rows)
    comps = []
    for spec in compare:
        path, arm = spec.rsplit(":", 1)
        b_rows, b_cfg = load_run(Path(path))
        # a baseline from the same dataset is paired as it is; one from another dataset (the wider WebQA run for
        # WebQA-img) must be restricted explicitly to the questions of this column
        same = b_cfg.get("dataset") == o_cfg.get("dataset")
        if not same and restrict is None:
            raise SystemExit(f"{path} is dataset {b_cfg.get('dataset')!r}, not {o_cfg.get('dataset')!r}: "
                             "restrict it explicitly (--restrict-modality)")
        check_pairable(o_cfg, b_cfg, restricted=not same)
        keep = None if same else restrict
        b_q = qid_set([r for r in b_rows if keep is None or keep in (r.get("gold_modalities") or [])])
        if b_q != o_q:
            raise SystemExit(f"{path}: {len(b_q)} questions vs {len(o_q)} in {ours}; restrict the baseline explicitly "
                             "(--restrict-modality) or fix the run")
        qids = sorted(o_q)
        b_scores = load_scores(Path(path), metric)
        res = _paired(b_scores, o_scores, qids, arm, our_arm, metric)
        comps.append({"baseline": f"{path}:{arm}", "n": len(qids), "baseline_mean": res["baseline"],
                      "ours_mean": res["ours"], "delta_pp": res["delta_pp"], "ci95_pp": res["paired_ci95_pp"],
                      "p_two_sided": res["p_two_sided"]})
    if not comps:
        raise SystemExit("column needs at least one --compare baseline")
    adjusted = holm([c["p_two_sided"] for c in comps])
    for c, p in zip(comps, adjusted):
        c["p_holm"] = p
    out = {"ours": f"{ours}:{our_arm}", "n": len(o_q), "ours_mean": comps[0]["ours_mean"],
           "comparisons": comps, "m": len(comps), "strongest": None, "strongest_delta_pp": None,
           "strongest_p_holm": None}
    if strongest is not None:
        named = [c for c in comps if c["baseline"] == strongest]
        if not named:
            raise SystemExit(f"--strongest {strongest} is not one of the --compare baselines of this column")
        out.update(strongest=strongest, strongest_delta_pp=named[0]["delta_pp"], strongest_p_holm=named[0]["p_holm"])
    return out


# ---------------------------------------------------------- tau sensitivity --
TAU_GRID = (0.25, 0.35, 0.45)


def tau_table(path: Path, metric: str, arm: str, taus: list[float]) -> dict:
    """Gated score for several tau values, from one run. Exact: the gate only decides retrieve = (u_gate > tau),
    the gated arm is preds[arm] when retrieving and preds[direct] otherwise, and every arm is decoded for every
    question, so no model call is needed. Checked against the run's own gated score at the run's tau (must match)."""
    rows, config = load_run(path)
    scores = load_scores(path, metric)
    u = [r["u_gate"] for r in rows]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in u):
        raise SystemExit(f"{path}: u_gate missing or non-finite")
    direct = [scores[r["qid"]]["direct"] for r in rows]
    own = [scores[r["qid"]][arm] for r in rows]
    gated_arm = [scores[r["qid"]][f"gated:{arm}"] for r in rows]
    out = {"questions": len(rows), "arm": arm, "taus": {}}
    run_tau = config.get("tau")
    for tau in sorted(set(taus) | ({run_tau} if run_tau is not None else set())):
        retrieve = [x > tau for x in u]
        vals = [o if rt else d for rt, o, d in zip(retrieve, own, direct)]
        out["taus"][f"{tau:g}"] = {"retrieval_rate": 100 * mean([float(x) for x in retrieve]), "gated_score": mean(vals)}
    if run_tau is not None:
        ref = mean(gated_arm)
        mine = out["taus"][f"{run_tau:g}"]["gated_score"]
        if abs(ref - mine) > 1e-9:
            raise SystemExit(f"tau check failed: offline {mine} vs the run's gated score {ref}")
        out["check_at_run_tau"] = {"tau": run_tau, "run_gated_score": ref, "offline_gated_score": mine}
    return out


# ------------------------------------------------------------------- ranges --
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


# ---------------------------------------------------------------------- CLI --
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", choices=["order", "uncertainty", "selective", "agreement", "transitions", "paired",
                                         "column", "tau", "ard", "interventions", "gold25", "noise", "latency",
                                         "stats", "ranges", "errors"])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--metric", default=None, help="official per-question metric (e.g. acc_approx, accuracy)")
    ap.add_argument("--arms", nargs="+", default=["static"])
    ap.add_argument("--baseline", default="static", help="order: arm whose std the others are compared to")
    ap.add_argument("--compare", action="append", default=None,
                    help="column: a baseline as <run>:<arm> (repeat for each baseline of the column)")
    ap.add_argument("--strongest", default=None,
                    help="column: the strongest baseline as <run>:<arm>, fixed by the caller (e.g. from the dev decisions); "
                         "omitted: the column lists its baselines without naming one")
    ap.add_argument("--seakr-run", type=Path, default=None, help="selective: the SeaKR run (SeaKR system)")
    ap.add_argument("--restrict-modality", default=None, help="column/paired: baseline keeps questions with this gold modality")
    ap.add_argument("--pair-subset", action="store_true", help="paired: allow pairing a subset run explicitly")
    ap.add_argument("--taus", nargs="+", type=float, default=list(TAU_GRID), help="tau: values of the gate")
    ap.add_argument("--correct-at", type=float, default=50.0)
    ap.add_argument("--attribution", type=Path, default=None, help="errors: <run>.attribution.json")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args(argv)
    needs_metric = {"order", "uncertainty", "selective", "agreement", "transitions", "paired", "column", "tau", "gold25",
                    "noise", "errors"}
    if args.analysis in needs_metric and not args.metric:
        ap.error(f"{args.analysis} needs --metric")
    one = args.runs[0]
    if args.analysis == "order":
        result = order_table(args.runs, args.metric, args.arms, args.correct_at, args.baseline)
    elif args.analysis == "uncertainty":
        result = {str(p): uncertainty_table(p, args.metric, args.arms[0]) for p in args.runs}
    elif args.analysis == "selective":
        if args.seakr_run is None:
            ap.error("selective needs --seakr-run (SeaKR is scored on its own run)")
        result = selective_table(one, args.seakr_run, args.metric, args.arms[0])
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
        result = paired_test(args.runs[0], args.arms[0], args.runs[1], args.arms[1], args.metric,
                             restrict=args.restrict_modality, subset=args.pair_subset)
    elif args.analysis == "column":
        if not args.compare:
            ap.error("column needs --compare <run>:<arm> (one or more)")
        result = column_table(one, args.arms[0], args.metric, args.compare, restrict=args.restrict_modality,
                              strongest=args.strongest)
    elif args.analysis == "tau":
        result = {str(p): tau_table(p, args.metric, args.arms[0], args.taus) for p in args.runs}
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

"""Consistency checks of the analysis layer: runs are aggregated or paired only when they agree (M-46..M-55, D18)."""
import json

import numpy as np
import pytest

from omniaware import order as order_mod
from scripts.analysis import paper_analyses as pa
from scripts.analysis import routing_metrics as rm


CFG = {"split": "test", "model": "8b", "calib_source": "heldout", "n_calib": 500, "split_seed": 2027, "k": 5,
       "answer_style": "default", "dataset": "webqa_open_img", "candidate_sha256": "c0", "relevance_norm": "ecdf",
       "variants": [{"name": "eab"}]}


def _row(q, **kw):
    base = {"qid": q, "question": "q?", "golds": ["Paris"], "preds": {"static": "Rome", "eab": "Paris"},
            "retrieve": True, "u_pre": 0.5, "raw_pre": {"h_sem": 1.0, "h_grd": 0.0}, "u_gate": 0.5,
            "rel": {"a": 0.9, "b": 0.7}, "chunks": ["a", "b"], "chunk_gold": [False, True],
            "tokens": {"static": [5], "eab": [6]}, "eab_active": {"eab": True},
            "span_tokens": {"a": ["text", 10]}, "prompt_tokens": 100,
            "prefill_seconds": {"direct": 0.1, "evidence": 0.3}, "arm_seconds": {"direct": 0.05, "static": 0.2, "eab": 0.2},
            "seconds_stage": {"direct": 0.15, "probe": 0.0, "route": 0.01, "encode": 0.02},
            "ard": {"static": 0.3, "eab": 0.1}, "gold_in_topk": True}
    base.update(kw)
    return base


def _write(dirpath, rows, scores, config):
    dirpath.mkdir(parents=True, exist_ok=True)
    run = dirpath / "webqa_img.jsonl"
    run.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (dirpath / "webqa_img_config.json").write_text(json.dumps(config))
    (dirpath / "webqa_img.official_scores.jsonl").write_text("".join(
        json.dumps({"qid": q, "scores": {a: {"acc": v} for a, v in s.items()}}) + "\n" for q, s in scores.items()))
    return run


def _order_runs(root, qids=("q1", "q2"), drop=None, bad_hash=None, seed_shift=False):
    runs = []
    for o in order_mod.permutation_schedule():
        oid = o["order_id"]
        cfg = {**CFG, "order": f"perm:{oid}", **({"candidate_sha256": "other"} if oid == bad_hash else {})}
        rows = [_row(q) for q in qids if not (oid == drop and q == qids[-1])]
        # seed_shift moves every order of seed s by 2 * index(s): the seed means then differ
        seeded = oid[:1] == "s" and oid[1:3].isdigit()
        shift = 2.0 * order_mod.PERMUTATION_SEEDS.index(int(oid[1:3])) if (seed_shift and seeded) else 0.0
        scores = {q: {"static": 50.0 + (i % 3) + shift, "eab": 100.0} for i, q in enumerate(qids)}
        runs.append(_write(root / oid, rows, scores, cfg))
    return runs


def test_order_table_refuses_mismatched_questions_and_configs(tmp_path):
    t = pa.order_table(_order_runs(tmp_path / "ok"), "acc", ["static", "eab"], 50, "static")
    assert t["questions"] == 2 and set(t["n_per_order"].values()) == {2}
    assert len(t["arms"]["static"]["std_per_seed"]) == len(order_mod.PERMUTATION_SEEDS)
    with pytest.raises(SystemExit, match="answers 1 questions"):
        pa.order_table(_order_runs(tmp_path / "drop", drop="s17-p1"), "acc", ["static", "eab"], 50, "static")
    with pytest.raises(SystemExit, match="candidate_sha256"):
        pa.order_table(_order_runs(tmp_path / "hash", bad_hash="s23-p2"), "acc", ["static", "eab"], 50, "static")


def test_order_std_across_seeds_is_the_std_of_seed_means(tmp_path):
    t = pa.order_table(_order_runs(tmp_path / "seeds", seed_shift=True), "acc", ["static"], 50, "static")
    # seed means 50.5, 52.5, 54.5 (the sorted and reverse orders are not in any seed)
    assert t["arms"]["static"]["std_across_seeds"] == pytest.approx(float(np.std([50.5, 52.5, 54.5])))


def test_noise_draws_are_exactly_three_and_aligned_by_qid(tmp_path):
    def runs_for(draws, sigma=0.1):
        out = []
        for d in draws:
            cfg = {**CFG, "relevance_noise": sigma, "noise_draw": d}
            scores = {"q1": {"static": 50.0 + d, "eab": 100.0}, "q2": {"static": 50.0 + d, "eab": 100.0}}
            out.append(_write(tmp_path / f"s{sigma}_d{d}", [_row("q1"), _row("q2")], scores, cfg))
        return out
    t = pa.noise_table(runs_for([0, 1, 2]), "acc")
    assert t["0.1"]["draws"] == 3 and t["0.1"]["questions"] == 2
    assert t["0.1"]["scores"]["static"] == pytest.approx(51.0)
    with pytest.raises(SystemExit, match="exactly 0, 1, 2"):
        pa.noise_table(runs_for([0, 1]), "acc")


def test_intervention_failures_are_counted_with_a_fixed_denominator(tmp_path):
    rows = [_row("q1", interventions={"eligible": True, "entity": "Rome",
                                      "cf": {"preds": {"static": "Paris", "eab": "Paris"}}}),
            _row("q2", interventions={"eligible": True, "entity": "Rome", "cf": None}),
            _row("q3", interventions={"eligible": True, "entity": "Rome",
                                      "cf": {"failed_reason": "no substitute", "failed": True}}),
            _row("q4", eab_active={"eab": False},
                 interventions={"eligible": True, "entity": "Rome", "cf": {"preds": {"static": "Paris", "eab": "Paris"}}})]
    run = _write(tmp_path / "iv", rows, {r["qid"]: {"static": 0.0, "eab": 0.0} for r in rows}, {})
    t = pa.intervention_table(run)
    cf = t["kinds"]["cf"]
    assert t["eligible"] == 4 and cf["attempted"] == 4 and cf["success"] == 2 and cf["failed"] == 2
    assert cf["failure_reasons"] == {"no_entry": 1, "no substitute": 1} and cf["failure_rate"] == 50.0
    assert cf["arms"]["static"]["n"] == 2 and cf["arms"]["static"]["changed_of_eligible"] == 50.0   # 2 of 4 changed
    assert cf["arms"]["eab"]["n"] == 1          # q4's EAB arm was inactive, so it is not eligible for it (M-49)


def test_ard_uses_finite_values_and_reports_missing(tmp_path):
    rows = [_row("q1", ard={"eab": 0.2}), _row("q2", ard={"eab": float("nan")}), _row("q3", ard={})]
    run = _write(tmp_path / "ard", rows, {r["qid"]: {"static": 0.0} for r in rows}, {})
    t = pa.ard_table(run)
    assert t["eab"]["n_valid"] == 1 and t["eab"]["n_missing"] == 2 and t["eab"]["ARD"] == pytest.approx(0.2)


def test_latency_refuses_runs_without_measured_stages(tmp_path):
    row = _row("q1")
    row.pop("seconds_stage")
    run = _write(tmp_path / "lat", [row], {"q1": {"static": 0.0}}, {})
    with pytest.raises(SystemExit, match="seconds_stage"):
        pa.latency_table(run)


def test_holm_adjusts_in_step_down_order():
    assert pa.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert pa.holm([0.2]) == [0.2]


def test_column_lists_every_baseline_and_names_no_strongest_unless_told(tmp_path):
    qids = [f"q{i}" for i in range(6)]
    rows = [_row(q, gold_modalities=["image"]) for q in qids]
    ours = _write(tmp_path / "main", rows, {q: {"gated:eab": 100.0, "static": 50.0} for q in qids}, CFG)
    seakr = _write(tmp_path / "seakr", rows, {q: {"seakr": float(v)} for q, v in zip(qids, [40, 30, 50, 20, 60, 40])}, CFG)
    ce = _write(tmp_path / "ce", rows, {q: {"static": float(v)} for q, v in zip(qids, [100, 100, 80, 100, 100, 100])}, CFG)
    t = pa.column_table(ours, "gated:eab", "acc", [f"{ce}:static", f"{seakr}:seakr"])
    # ce has the higher mean, but the column does not choose on its own scores: no strongest is named
    assert t["m"] == 2 and t["strongest"] is None and t["strongest_p_holm"] is None
    by = {c["baseline"]: c for c in t["comparisons"]}
    assert by[f"{seakr}:seakr"]["p_two_sided"] < 0.05 and by[f"{ce}:static"]["p_two_sided"] > 0.05
    assert all(c["p_holm"] >= c["p_two_sided"] - 1e-12 for c in t["comparisons"])


def test_column_names_the_strongest_baseline_only_when_the_caller_fixes_it(tmp_path):
    qids = [f"q{i}" for i in range(6)]
    rows = [_row(q, gold_modalities=["image"]) for q in qids]
    ours = _write(tmp_path / "main", rows, {q: {"gated:eab": 100.0, "static": 50.0} for q in qids}, CFG)
    seakr = _write(tmp_path / "seakr", rows, {q: {"seakr": float(v)} for q, v in zip(qids, [40, 30, 50, 20, 60, 40])}, CFG)
    ce = _write(tmp_path / "ce", rows, {q: {"static": float(v)} for q, v in zip(qids, [100, 100, 80, 100, 100, 100])}, CFG)
    fixed = f"{seakr}:seakr"
    t = pa.column_table(ours, "gated:eab", "acc", [f"{ce}:static", f"{seakr}:seakr"], strongest=fixed)
    assert t["strongest"] == fixed
    by = {c["baseline"]: c for c in t["comparisons"]}
    assert t["strongest_p_holm"] == by[fixed]["p_holm"] and t["strongest_delta_pp"] == by[fixed]["delta_pp"]
    with pytest.raises(SystemExit, match="not one of the --compare"):
        pa.column_table(ours, "gated:eab", "acc", [f"{ce}:static"], strongest=fixed)


def test_column_refuses_a_baseline_from_another_dataset(tmp_path):
    qids = ["q1", "q2"]
    rows = [_row(q) for q in qids]
    ours = _write(tmp_path / "main", rows, {q: {"gated:eab": 100.0} for q in qids}, CFG)
    other = _write(tmp_path / "other", rows, {q: {"seakr": 1.0} for q in qids}, {**CFG, "dataset": "hybridqa"})
    with pytest.raises(SystemExit, match="dataset"):
        pa.column_table(ours, "gated:eab", "acc", [f"{other}:seakr"])


def test_subset_baseline_must_be_restricted_explicitly(tmp_path):
    rows = [_row(f"q{i}", gold_modalities=["image"] if i % 2 == 0 else ["text"]) for i in range(6)]
    img_rows = [r for r in rows if r["gold_modalities"] == ["image"]]
    ours = _write(tmp_path / "img", img_rows, {r["qid"]: {"gated:eab": 80.0} for r in img_rows}, CFG)
    # the wider WebQA run is another dataset: without an explicit restriction nothing is pooled
    full = _write(tmp_path / "full", rows, {r["qid"]: {"universalrag": 50.0} for r in rows},
                  {**CFG, "dataset": "webqa_open_full", "candidate_sha256": "full"})
    with pytest.raises(SystemExit, match="restrict it explicitly"):
        pa.column_table(ours, "gated:eab", "acc", [f"{full}:universalrag"])
    # a same-dataset baseline with a different question set is refused too (no intersection is taken)
    same = _write(tmp_path / "same", rows, {r["qid"]: {"seakr": 1.0} for r in rows}, CFG)
    with pytest.raises(SystemExit, match="answers 6 questions|questions"):
        pa.column_table(ours, "gated:eab", "acc", [f"{same}:seakr"])
    # the explicit restriction keeps the image questions (the paper's WebQA-img column of UniversalRAG)
    t = pa.column_table(ours, "gated:eab", "acc", [f"{full}:universalrag"], restrict="image")
    assert t["n"] == 3 and t["comparisons"][0]["delta_pp"] == pytest.approx(30.0)


def test_cross_run_pairs_must_share_the_candidate_fingerprint(tmp_path):
    rows = [_row("q1"), _row("q2")]
    a = _write(tmp_path / "a", rows, {"q1": {"seakr": 1.0}, "q2": {"seakr": 2.0}}, CFG)
    b = _write(tmp_path / "b", rows, {"q1": {"eab": 1.0}, "q2": {"eab": 2.0}}, {**CFG, "candidate_sha256": "other"})
    with pytest.raises(SystemExit, match="candidate_sha256"):
        pa.paired_test(a, "seakr", b, "eab", "acc")


def test_tau_sweep_is_exact_offline_and_checked_against_the_run(tmp_path):
    rows = [_row(f"q{i}", u_gate=u) for i, u in enumerate([0.1, 0.3, 0.5, 0.7])]
    gated = {"q0": 0.0, "q1": 0.0, "q2": 100.0, "q3": 100.0}          # the run's tau = 0.35 retrieves when u > 0.35
    scores = {r["qid"]: {"direct": 0.0, "eab": 100.0, "gated:eab": gated[r["qid"]]} for r in rows}
    run = _write(tmp_path / "tau", rows, scores, {"tau": 0.35})
    t = pa.tau_table(run, "acc", "eab", [0.25, 0.35, 0.45])
    assert t["taus"]["0.35"]["retrieval_rate"] == 50.0 and t["taus"]["0.35"]["gated_score"] == 50.0
    # u = 0.1, 0.3, 0.5, 0.7: retrieval above 0.25 keeps 3 of 4, above 0.45 keeps 2 of 4
    assert t["taus"]["0.25"]["retrieval_rate"] == 75.0 and t["taus"]["0.45"]["retrieval_rate"] == 50.0
    assert t["check_at_run_tau"]["offline_gated_score"] == t["check_at_run_tau"]["run_gated_score"]
    tampered = {**scores, "q3": {"direct": 0.0, "eab": 100.0, "gated:eab": 0.0}}
    bad = _write(tmp_path / "tau_bad", rows, tampered, {"tau": 0.35})
    with pytest.raises(SystemExit, match="tau check failed"):
        pa.tau_table(bad, "acc", "eab", [0.35])


def test_selective_takes_seakr_from_the_seakr_run(tmp_path):
    main_rows = [_row("q1", u_gate=0.9, raw_pre={"h_sem": 0.9, "h_grd": 0.0}),
                 _row("q2", u_gate=0.1, raw_pre={"h_sem": 0.1, "h_grd": 0.0})]
    main = _write(tmp_path / "main", main_rows, {"q1": {"gated:eab": 0.0, "nte_trigger:static": 0.0},
                                                 "q2": {"gated:eab": 100.0, "nte_trigger:static": 100.0}}, {})
    seakr_rows = [_row("q1", seakr={"direct_eigen": 2.0}), _row("q2", seakr={"direct_eigen": 1.0})]
    seakr = _write(tmp_path / "seakr", seakr_rows, {"q1": {"seakr": 0.0}, "q2": {"seakr": 100.0}}, {})
    sel = pa.selective_table(main, seakr, "acc", "eab")
    assert sel["SeaKR"]["run"] == str(seakr) and sel["SeaKR"]["n"] == 2
    assert sel["U_omni"]["run"] == str(main) and sel["NTE"]["run"] == str(main)


def test_paired_across_runs_requires_equal_question_sets(tmp_path):
    rows = [_row(f"q{i}") for i in range(4)]
    cfg = {k: CFG[k] for k in pa.PAIR_FIELDS if k in CFG}
    a = _write(tmp_path / "a", rows, {r["qid"]: {"seakr": 0.0} for r in rows}, {**cfg, "dataset": "webqa_open_img"})
    b = _write(tmp_path / "b", rows[:3], {r["qid"]: {"eab": 1.0} for r in rows[:3]}, {**cfg, "dataset": "webqa_open_img"})
    with pytest.raises(SystemExit, match="pair subsets explicitly"):
        pa.paired_test(a, "seakr", b, "eab", "acc")
    assert pa.paired_test(a, "seakr", b, "eab", "acc", subset=True)["questions"] == 3


def test_routing_score_is_aligned_by_qid(tmp_path):
    rows = [{"qid": "q1"}, {"qid": "q2"}]
    scores = tmp_path / "r.official_scores.jsonl"
    scores.write_text(json.dumps({"qid": "q2", "scores": {"gated:eab": {"acc": 0.0}}}) + "\n"
                      + json.dumps({"qid": "q1", "scores": {"gated:eab": {"acc": 100.0}}}) + "\n")
    assert rm.aligned_score(rows, scores, "gated:eab", "acc") == pytest.approx(50.0)
    with pytest.raises(SystemExit, match="no score"):
        rm.aligned_score(rows + [{"qid": "q3"}], scores, "gated:eab", "acc")

import json

import pytest

from omniaware import order as order_mod
from scripts.analysis import paper_analyses as pa


def _write(dirpath, rows, scores, config):
    dirpath.mkdir(parents=True, exist_ok=True)
    run = dirpath / "webqa_img.jsonl"
    run.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (dirpath / "webqa_img_config.json").write_text(json.dumps(config))
    (dirpath / "webqa_img.official_scores.jsonl").write_text("".join(
        json.dumps({"qid": q, "scores": {a: {"acc": v} for a, v in s.items()}}) + "\n" for q, s in scores.items()))
    return run


def _row(q, **kw):
    base = {"qid": q, "question": "q?", "golds": ["Paris"], "preds": {"static": "Rome", "eab": "Paris"},
            "retrieve": True, "u_pre": 0.5, "raw_pre": {"h_sem": 1.0, "h_grd": 0.0},
            "rel": {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3, "e": 0.1}, "chunk_gold": [False, True, False, False, False],
            "tokens": {"static": [5], "eab": [6, 7]}, "eab_active": {"eab": True},
            "span_tokens": {"a": ["text", 10], "b": ["image", 64]}, "prompt_tokens": 100,
            "prefill_seconds": {"direct": 0.1, "evidence": 0.3}, "arm_seconds": {"direct": 0.05, "static": 0.2, "eab": 0.2},
            "seconds_stage": {"direct": 0.15, "probe": 0.0, "route": 0.01, "encode": 0.02},
            "ard": {"static": 0.3, "eab": 0.1}, "gold_in_topk": True, "baseline_seconds": {"seakr": 1.5}}
    base.update(kw)
    return base


def test_order_table_std_gap_and_flips(tmp_path):
    runs = []
    for i, o in enumerate(order_mod.permutation_schedule()):
        scores = {"q1": {"static": 100.0 * (i % 2), "eab": 100.0}, "q2": {"static": 0.0, "eab": 100.0}}
        runs.append(_write(tmp_path / o["order_id"], [_row("q1"), _row("q2")], scores, {"order": f"perm:{o['order_id']}"}))
    t = pa.order_table(runs, "acc", ["static", "eab"], 50, "static")
    assert t["arms"]["eab"]["std"] == 0 and t["arms"]["eab"]["flip_rate"] == 0
    assert t["arms"]["static"]["std"] > 0 and t["arms"]["static"]["flip_rate"] == 50
    assert t["arms"]["static"]["sorted_minus_reverse"] == -50      # sorted is i=0, reverse i=1


def test_latency_gold25_stats_uncertainty_interventions(tmp_path):
    rows = [_row("q1"), _row("q2", retrieve=False, u_pre=0.1, chunk_gold=[True, False, False, False, False]),
            _row("q3", interventions={"eligible": True, "entity": "Rome",
                                      "cf": {"preds": {"static": "Rome", "eab": "Paris"}},
                                      "inject_answer": {"preds": {"static": "Rome", "eab": "Rome"}}})]
    scores = {"q1": {"static": 0.0, "eab": 100.0}, "q2": {"static": 100.0, "eab": 100.0},
              "q3": {"static": 0.0, "eab": 100.0}}
    run = _write(tmp_path / "r", rows, scores, {"order": "rank", "variants": [{"name": "eab"}]})
    lat = pa.latency_table(run)
    # always:static = probe + route + encode + evidence prefill + static decode (0.53 s on every row)
    assert abs(lat["always:static"] - 530) < 1e-6 and abs(lat["seakr"] - 1500) < 1e-6
    # retrieving rows: direct prefill only (0.15 - 0.05) + the retrieval path (0.53); q2 keeps the direct path (0.15)
    assert abs(lat["gated:static"] - (630 + 150 + 630) / 3) < 1e-6
    assert abs(lat["direct"] - 150) < 1e-6
    g = pa.gold25_table(run, "acc")
    assert g["gold_2_5"] == 2 and g["scores"]["eab"] == 100
    s = pa.stats_table(run, ["static", "eab"])
    assert s["single_token_answers"]["static"] == 100 and s["first_token_changed_vs_static"]["eab"] == 100
    assert abs(s["relevance_r1_minus_r5"] - 0.8) < 1e-9 and s["eab_active"]["eab"] == 100
    u = pa.uncertainty_table(run, "acc", "static")
    assert u["U_omni"]["n"] == 3 and u["U_omni"]["spearman"] > 0
    iv = pa.intervention_table(run)
    assert iv["eligible"] == 1
    assert iv["kinds"]["cf"]["arms"]["static"]["follows_entity"] == 100
    assert iv["kinds"]["cf"]["arms"]["eab"]["follows_entity"] == 0
    assert iv["kinds"]["inject_answer"]["arms"]["eab"]["misled"] == 100
    assert iv["kinds"]["cf"]["attempted"] == 1 and iv["kinds"]["cf"]["failed"] == 0
    assert abs(pa.ard_table(run)["eab"]["ARD"] - 0.1) < 1e-9
    e = pa.errors_table(run, "acc", "static", 50, None, 5, 0)
    assert e["errors"] == 2 and e["shares"]["ambiguous_visual_grounding"] == 100


def test_selective_agreement_transitions_ranges(tmp_path):
    rows = [_row("q1", raw_pre={"h_sem": 0.9, "h_grd": 0.0}, u_gate=0.9, retrieve=True, seakr={"direct_eigen": 3.0}),
            _row("q2", raw_pre={"h_sem": 0.1, "h_grd": 0.0}, u_gate=0.1, retrieve=False, seakr={"direct_eigen": 1.0}),
            _row("q3", raw_pre={"h_sem": 0.8, "h_grd": 0.0}, u_gate=0.2, retrieve=False, seakr={"direct_eigen": 2.0})]
    arms = lambda d, s, e: {"direct": d, "static": s, "eab": e, "gated:eab": e, "nte_trigger:static": s, "seakr": s}
    scores = {"q1": arms(0, 0, 100), "q2": arms(100, 100, 100), "q3": arms(0, 100, 0)}
    run = _write(tmp_path / "r", rows, scores, {"order": "rank", "variants": [{"name": "eab"}], "nte_threshold": 0.5})
    sel = pa.selective_table(run, run, "acc", "eab")
    assert set(sel) == {"U_omni", "NTE", "SeaKR"} and sel["U_omni"]["arm"] == "gated:eab"
    ag = pa.agreement_table(run, "acc", "eab")
    assert abs(ag["both_retrieve"] - 100 / 3) < 1e-9 and abs(ag["only_nte"] - 100 / 3) < 1e-9
    assert ag["no_retrieval_score"]["U"] == 50 and ag["retrieved_score"]["U (EAB)"] == 100
    tr = pa.transitions_table(run, "acc", "static", "eab", 50)
    assert abs(tr["wrong_to_correct"] - 100 / 3) < 1e-9 and abs(tr["correct_to_wrong"] - 100 / 3) < 1e-9
    cache = tmp_path / "relevance.json"
    cache.write_text(json.dumps({"q1": {"a": {"text": 0.2}, "b": {"image_clip": 0.3}}, "q2": {"a": {"text": 0.6}}}))
    rg = pa.score_ranges(cache)
    assert rg["text"]["min"] == 0.2 and rg["text"]["max"] == 0.6 and rg["image_clip"]["n"] == 1


def test_paired_across_runs_and_loss_breakdown(tmp_path):
    rows = [_row(f"q{i}", chunks=["a", "b"], rel={"a": 0.9, "b": 0.1},
                 chunk_modalities=["text", "image"], chunk_has_answer=[False, True]) for i in range(4)]
    cfg = {"split": "test", "model": "8b", "calib_source": "train", "n_calib": 500, "split_seed": 2027, "k": 5,
           "answer_style": "default", "dataset": "videorag_synth", "candidate_sha256": "abc"}
    base = _write(tmp_path / "base", rows, {f"q{i}": {"seakr": 0.0, "gated:static": 100.0, "gated:eab": 0.0}
                                             for i in range(4)}, cfg)
    ours = _write(tmp_path / "ours", rows[:3], {f"q{i}": {"gated:eab": 100.0, "gated:static": 100.0} for i in range(3)}, cfg)
    with pytest.raises(SystemExit, match="pair subsets explicitly"):    # 3 vs 4 questions: no silent intersection
        pa.paired_test(base, "seakr", ours, "gated:eab", "acc")
    t = pa.paired_test(base, "seakr", ours, "gated:eab", "acc", subset=True)
    assert t["questions"] == 3 and t["delta_pp"] == 100.0
    tr = pa.transitions_table(base, "acc", "gated:static", "gated:eab", 50)
    assert tr["correct_to_wrong"] == 100 and tr["losses_by_top_biased_chunk"] == {"top_text_chunk_without_answer": 100.0}


def test_answer_of_reads_the_parsed_answer_and_parses_old_records():
    # rec["preds"] keeps the raw output with its citation line; the answer is the parsed one.
    raw = "Paris\n\n[E1]"
    new = {"preds": {"static": raw}, "parsed": {"static": {"answer": "Paris", "answer_empty": False}}}
    assert pa.answer_of(new, "static") == "Paris"
    # A record written before the parse was stored is parsed here, with the same rule.
    old = {"preds": {"static": raw}}
    assert pa.answer_of(old, "static") == "Paris"
    assert pa.answer_of({"preds": {"static": "Sources: none"}}, "static") == ""
    # The error categories match the gold on the parsed answer, not on the citation text.
    row = {"qid": "q", "golds": ["Paris"], "preds": {"static": raw}}
    assert pa.categorize(row, "static", 0.0, None, 50) == "answer_normalization"

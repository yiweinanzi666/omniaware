import json

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
    assert abs(lat["always:static"] - 500) < 1e-6 and abs(lat["seakr"] - 1500) < 1e-6
    assert abs(lat["gated:static"] - (600 + 150 + 600) / 3) < 1e-6
    g = pa.gold25_table(run, "acc")
    assert g["gold_2_5"] == 2 and g["scores"]["eab"] == 100
    s = pa.stats_table(run, ["static", "eab"])
    assert s["single_token_answers"]["static"] == 100 and s["first_token_changed_vs_static"]["eab"] == 100
    assert abs(s["relevance_r1_minus_r5"] - 0.8) < 1e-9 and s["eab_active"]["eab"] == 100
    u = pa.uncertainty_table(run, "acc", "static")
    assert u["U_omni"]["n"] == 3 and u["U_omni"]["spearman"] > 0
    iv = pa.intervention_table(run)
    assert iv["eligible"] == 1
    assert iv["kinds"]["cf"]["static"]["follows_entity"] == 100 and iv["kinds"]["cf"]["eab"]["follows_entity"] == 0
    assert iv["kinds"]["inject_answer"]["eab"]["misled"] == 100
    assert abs(pa.ard_table(run)["eab"]["ARD"] - 0.1) < 1e-9
    e = pa.errors_table(run, "acc", "static", 50, None, 5, 0)
    assert e["errors"] == 2 and e["shares"]["ambiguous_visual_grounding"] == 100

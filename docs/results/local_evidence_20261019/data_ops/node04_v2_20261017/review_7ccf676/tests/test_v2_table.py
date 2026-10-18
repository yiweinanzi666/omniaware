"""v2_table build -> gate -> select -> apply on tiny synthetic runs, plus the properties the table must keep."""
import json
import zipfile

import pytest

from omniaware import paths, runio
from omniaware.eval import score_official, webqa_acc
from scripts.analysis import v2_table as vt


def _gold(i):
    """A letters-only answer, unique per index (the VQA evaluator sees plain words)."""
    return f"item{chr(97 + i % 26)}{chr(97 + i // 26)}"


def _action(answer, margin, *, available=True, kl=None):
    if not available:
        return {"available": False, "answer": "", "answer_empty": True, "citation_only": False,
                "seq_logprob": None, "gen_len": 0, "first_token": None, "kl_to_direct": None,
                "retriever": {"top1_score": None}}
    return {"available": True, "answer": answer, "answer_empty": False, "citation_only": False,
            "seq_logprob": None, "gen_len": 1, "first_token": {"margin": margin, "entropy": 1.0, "entropy_norm": 0.5},
            "answer_token": {"margin": margin, "entropy_norm": 0.5, "u_raw": -margin},
            "kl_to_direct": kl, "retriever": {"top1_score": 0.5}}


def _okvqa_records(qids, *, mixed_unavailable=(), direct_unavailable=()):
    """Direct is right on even questions and static_mixed on odd ones (so the oracle is 100 everywhere and the best
    single action is a 50 tie, which goes to direct). static_text is missing on every fourth question, so it is a
    partially available action. static_image is always available and is wrong except every fifth question.
    u_gate is low where direct is right, so the U gate can separate the two; h_sem_pre is constant."""
    records = []
    for i, qid in enumerate(qids):
        gold = _gold(i)
        direct_right = i % 2 == 0
        mixed = _action(gold, 0.5) if not direct_right else _action("nope", 0.5)
        if qid in mixed_unavailable:
            mixed = _action("", 0.0, available=False)
        direct = _action(gold if direct_right else "wrong", 0.9 if direct_right else 0.1)
        if qid in direct_unavailable:
            direct = _action("", 0.0, available=False)
        records.append({
            "qid": qid, "dataset": "okvqa", "qcate": "okvqa", "golds": [gold],
            "u_gate": 0.1 if direct_right else 0.9, "raw_pre": {"h_sem": 0.0},
            "actions": {
                "direct": direct,
                "static_text": _action(gold, 0.4) if i % 4 else _action("", 0.0, available=False),
                "static_image": _action(gold if i % 5 == 0 else "nope", 0.3),
                "static_mixed": mixed,
            },
        })
    return records


def _okvqa_arms(n):
    """Official scores of two baseline arms: gated:static is right on every third question (mean 35) and
    universalrag is 60 everywhere, so universalrag is the strongest baseline among the arms."""
    return {"gated:static": [100.0 if i % 3 == 0 else 0.0 for i in range(n)], "universalrag": [60.0] * n}


def _write_okvqa_annotations(tmp_path, monkeypatch, records):
    annotations = {"annotations": [
        {"question_id": int(r["qid"]), "question_type": "other", "answer_type": "other",
         "answers": [{"answer": r["golds"][0], "answer_id": k + 1} for k in range(10)]}
        for r in records]}
    zpath = tmp_path / "ann.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("mscoco_val2014_annotations.json", json.dumps(annotations))
    monkeypatch.setattr(paths, "OKVQA_ANNOTATIONS", zpath)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _write_run(path, records, *, metric, arms):
    """A run as omniaware.run --v2 and score_official leave it: the predictions, the configured question set, the
    official scores of the baseline arms (arms maps an arm to one value per record), and the official summary."""
    _jsonl(path, records)
    qids = [r["qid"] for r in records]
    path.with_name(f"{path.stem}_config.json").write_text(json.dumps(
        {"evaluation_questions": len(qids), "evaluation_qids_sha256": runio.qid_set_sha256(qids)}), encoding="utf-8")
    official = [{"qid": q, "scores": {"direct": {metric: 0.0}, **{arm: {metric: values[i]}
                                                                  for arm, values in arms.items()}}}
                for i, q in enumerate(qids)]
    _jsonl(path.with_name(f"{path.stem}.official_scores.jsonl"), official)
    path.with_name(f"{path.stem}.official_summary.json").write_text(json.dumps(
        {"predictions_sha256": runio.sha256_file(path), "subset": False, "questions": len(qids)}), encoding="utf-8")
    return path


def _table_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


CALIB_QIDS = [str(q) for q in range(1, 41)]
TEST_QIDS = [str(q) for q in range(101, 141)]


@pytest.fixture
def okvqa_flow(tmp_path, monkeypatch):
    """Calibration and test runs with their annotations, their configurations and their official scores."""
    calib = _okvqa_records(CALIB_QIDS)
    test = _okvqa_records(TEST_QIDS, mixed_unavailable={"102"})
    _write_okvqa_annotations(tmp_path, monkeypatch, calib + test)
    run_dir = tmp_path / "runs" / "okvqa"
    run_dir.mkdir(parents=True)
    calib_run = _write_run(run_dir / "okvqa_gs_calib.jsonl", calib, metric="vqa_accuracy", arms=_okvqa_arms(len(calib)))
    test_run = _write_run(run_dir / "okvqa_gs_test.jsonl", test, metric="vqa_accuracy", arms=_okvqa_arms(len(test)))
    return {"dir": tmp_path, "calib_run": calib_run, "test_run": test_run}


def _select(okvqa_flow, tmp_path):
    """Build and select on the calibration run. Returns the decisions path."""
    calib = okvqa_flow["calib_run"]
    vt.build(calib)
    decisions = tmp_path / "decisions.json"
    vt.select(calib.with_name("okvqa_gs_calib.v2_actions.jsonl"), decisions, 0.1, 1.0, 0)
    return decisions


# ---------------------------------------------------------------- build

def test_build_scores_available_actions_only_and_writes_the_table(okvqa_flow):
    run = okvqa_flow["calib_run"]
    summary = vt.build(run)
    rows = _table_rows(run.with_name("okvqa_gs_calib.v2_actions.jsonl"))
    assert len(rows) == len(CALIB_QIDS)
    by_qid = {r["qid"]: r for r in rows}
    # static_text is missing on question 1 (index 0, a multiple of 4): no score there, nothing borrowed from direct.
    assert "static_text" not in by_qid["1"]["scores"]
    assert by_qid["1"]["actions"]["static_text"]["available"] is False
    assert by_qid["1"]["scores"]["direct"] == pytest.approx(100.0)       # index 0: direct is right
    assert by_qid["1"]["scores"]["static_mixed"] == 0.0
    assert by_qid["2"]["scores"]["direct"] == 0.0 and by_qid["2"]["scores"]["static_mixed"] == pytest.approx(100.0)
    assert all(r["metric"] == "vqa_accuracy" and r["subset"] is None for r in rows)
    assert summary["metric"] == "vqa_accuracy" and summary["questions"] == len(CALIB_QIDS)
    assert summary["input_sha256"] == runio.sha256_file(run)
    assert summary["partially_available"] == ["static_text"]
    assert summary["scored_on_every_question"] == ["direct", "static_image", "static_mixed"]
    assert summary["best_single"] == "direct"             # 50 tie with static_mixed; V2_ACTIONS order breaks it
    assert summary["oracle_mean"] == pytest.approx(100.0)
    assert summary["actions"]["static_text"]["questions"] == 30
    assert summary["actions"]["direct"]["empty_rate"] == 0.0
    assert summary["question_check"]["mode"] == "exact"


def test_build_carries_the_baseline_signals_and_scores_into_every_row(okvqa_flow):
    run = okvqa_flow["calib_run"]
    summary = vt.build(run)
    by_qid = {r["qid"]: r for r in _table_rows(run.with_name("okvqa_gs_calib.v2_actions.jsonl"))}
    assert by_qid["1"]["baseline_signals"] == {"u_gate": 0.1, "h_sem_pre": 0.0}
    assert by_qid["2"]["baseline_signals"] == {"u_gate": 0.9, "h_sem_pre": 0.0}
    # index 0 is a multiple of 3: gated:static is right there; universalrag is 60 everywhere
    assert by_qid["1"]["baseline_scores"] == {"gated:static": 100.0, "universalrag": 60.0}
    assert by_qid["2"]["baseline_scores"] == {"gated:static": 0.0, "universalrag": 60.0}
    assert summary["baseline_arms"] == ["gated:static", "universalrag"]


def test_build_refuses_a_run_without_actions(tmp_path):
    rows = [{"qid": "1", "dataset": "okvqa", "golds": ["x"], "preds": {"direct": "x"}}]
    run = _jsonl(tmp_path / "okvqa_gs.jsonl", rows)
    with pytest.raises(vt.TableError, match="--v2"):
        vt.build(run)


def test_build_refuses_an_incomplete_run_and_exits_two(okvqa_flow, capsys):
    run = okvqa_flow["calib_run"]
    records = _table_rows(run)[:-1]      # the run is missing its last question
    _jsonl(run, records)
    with pytest.raises(ValueError, match="configuration expects 40"):
        vt.build(run)
    assert vt.main(["build", "--run", str(run)]) == 2
    assert "configuration expects 40" in capsys.readouterr().err
    assert not run.with_name("okvqa_gs_calib.v2_actions.jsonl").exists()


def test_build_refuses_a_run_without_its_configuration(okvqa_flow):
    run = okvqa_flow["calib_run"]
    run.with_name("okvqa_gs_calib_config.json").unlink()
    with pytest.raises(ValueError, match="no okvqa_gs_calib_config.json"):
        vt.build(run)


def test_build_needs_official_scores_of_the_same_predictions(okvqa_flow):
    run = okvqa_flow["calib_run"]
    run.with_name("okvqa_gs_calib.official_scores.jsonl").unlink()
    with pytest.raises(vt.TableError, match="score_official"):
        vt.build(run)

    run = _write_run(okvqa_flow["dir"] / "runs" / "okvqa" / "okvqa_gs_calib.jsonl",
                     _okvqa_records(CALIB_QIDS), metric="vqa_accuracy", arms=_okvqa_arms(len(CALIB_QIDS)))
    records = _table_rows(run)
    records[0]["actions"]["direct"]["answer"] = "changed after scoring"
    _jsonl(run, records)
    with pytest.raises(vt.TableError, match="different predictions"):
        vt.build(run)


def test_build_refuses_missing_u_gate_or_h_sem(okvqa_flow, tmp_path):
    records = _okvqa_records(CALIB_QIDS)
    del records[3]["u_gate"]
    run = _write_run(tmp_path / "okvqa_gs_nosignal.jsonl", records, metric="vqa_accuracy",
                     arms=_okvqa_arms(len(CALIB_QIDS)))
    with pytest.raises(vt.TableError, match="u_gate"):
        vt.build(run)
    records = _okvqa_records(CALIB_QIDS)
    del records[3]["raw_pre"]
    run = _write_run(tmp_path / "okvqa_gs_noh.jsonl", records, metric="vqa_accuracy",
                     arms=_okvqa_arms(len(CALIB_QIDS)))
    with pytest.raises(vt.TableError, match="h_sem"):
        vt.build(run)


def test_build_webqa_subsets_come_from_gold_modalities(tmp_path, monkeypatch):
    class Exact:
        def __call__(self, prediction, reference, Qcate):
            return {"acc_approx": float(prediction.strip().lower() == reference.strip().lower())}
    monkeypatch.setattr(webqa_acc, "load_official_metric", lambda: (Exact(), {"stand_in": True}))
    monkeypatch.setattr(score_official, "webqa_references", lambda: {"g1": ["Paris"], "g2": ["Rome"]})
    rows = [
        {"qid": "g1", "dataset": "webqa", "qcate": "text", "golds": ["Paris"], "gold_modalities": ["image"],
         "u_gate": 0.2, "raw_pre": {"h_sem": 0.1},
         "actions": {"direct": _action("Paris", 0.9), "static_mixed": _action("Rome", 0.2)}},
        {"qid": "g2", "dataset": "webqa", "qcate": "text", "golds": ["Rome"], "gold_modalities": ["text"],
         "u_gate": 0.7, "raw_pre": {"h_sem": 0.3},
         "actions": {"direct": _action("Milan", 0.1), "static_mixed": _action("Rome", 0.2)}},
    ]
    arms = {"gated:static": [100.0, 0.0]}
    run = _write_run(tmp_path / "webqa_full.jsonl", rows, metric="acc_approx", arms=arms)
    summary = vt.build(run)
    table = {r["qid"]: r for r in _table_rows(run.with_name("webqa_full.v2_actions.jsonl"))}
    assert table["g1"]["subset"] == "image_question" and table["g2"]["subset"] == "text_question"
    assert table["g1"]["scores"] == {"direct": 100.0, "static_mixed": 0.0}
    assert table["g1"]["metric"] == "acc_approx"
    assert table["g2"]["baseline_scores"] == {"gated:static": 0.0}
    assert summary["best_single"] == "direct"             # both actions score 50: the tie goes to direct

    rows[0].pop("gold_modalities")
    run = _write_run(tmp_path / "webqa_nomod.jsonl", rows, metric="acc_approx", arms=arms)
    with pytest.raises(ValueError, match="gold_modalities"):
        vt.build(run)


def test_build_webqa_refuses_gold_answers_that_differ_from_the_dataset(tmp_path, monkeypatch):
    class Exact:
        def __call__(self, prediction, reference, Qcate):
            return {"acc_approx": float(prediction.strip().lower() == reference.strip().lower())}
    monkeypatch.setattr(webqa_acc, "load_official_metric", lambda: (Exact(), {"stand_in": True}))
    monkeypatch.setattr(score_official, "webqa_references", lambda: {"g1": ["Berlin"]})
    rows = [{"qid": "g1", "dataset": "webqa", "qcate": "text", "golds": ["Paris"], "gold_modalities": ["text"],
             "u_gate": 0.2, "raw_pre": {"h_sem": 0.1},
             "actions": {"direct": _action("Paris", 0.9), "static_mixed": _action("Rome", 0.2)}}]
    run = _write_run(tmp_path / "webqa_gold.jsonl", rows, metric="acc_approx", arms={})
    with pytest.raises(ValueError, match="differ"):
        vt.build(run)
    assert vt.main(["build", "--run", str(run)]) == 2


# ---------------------------------------------------------------- gate

def _gate_rows(path, rows):
    return _jsonl(path, [{"qid": f"q{i}", "metric": "vqa_accuracy", "actions": {}, "scores": s}
                         for i, s in enumerate(rows)])


def test_gate_passes_when_the_oracle_clears_eight_percent(tmp_path, capsys):
    table = _gate_rows(tmp_path / "t.jsonl", [{"direct": 100.0, "static_mixed": 0.0},
                                              {"direct": 0.0, "static_mixed": 100.0}])
    assert vt.main(["gate", "--table", str(table)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "PASS" and "best_single=direct" in out[1] and "relative_gap=1.0000" in out[1]
    assert not any("select" in line for line in out)


def test_gate_fails_when_the_oracle_equals_the_best_single(tmp_path, capsys):
    table = _gate_rows(tmp_path / "t.jsonl", [{"direct": 60.0, "static_mixed": 50.0},
                                              {"direct": 60.0, "static_mixed": 50.0}])
    assert vt.main(["gate", "--table", str(table)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "FAIL"
    assert "不要继续 select / S4" in out[-1]


def test_gate_zero_baseline_uses_the_difference_not_a_ratio(tmp_path):
    table = _gate_rows(tmp_path / "t.jsonl", [{"rerank_static": 0.0, "static_text": 0.5},
                                              {"rerank_static": 0.0, "static_text": -0.5}])
    passed, summary = vt.gate(table)
    assert summary["best_single"] == "static_text"        # the tie at 0.0 goes to the earlier action
    assert summary["relative_gap"] == pytest.approx(0.25) and passed
    zeros = _gate_rows(tmp_path / "z.jsonl", [{"direct": 0.0}, {"direct": 0.0}])
    passed, summary = vt.gate(zeros)
    assert not passed and summary["relative_gap"] == 0.0


def test_gate_oracle_uses_partial_actions_but_the_single_action_does_not(tmp_path):
    table = _gate_rows(tmp_path / "t.jsonl", [{"direct": 50.0, "static_mixed": 50.0, "static_image": 100.0},
                                              {"direct": 50.0, "static_mixed": 50.0}])
    passed, summary = vt.gate(table)
    assert summary["best_single"] == "direct" and summary["partially_available"] == ["static_image"]
    assert summary["oracle_mean"] == pytest.approx(75.0) and passed


def test_gate_bad_input_exits_two(tmp_path, capsys):
    assert vt.main(["gate", "--table", str(tmp_path / "missing.jsonl")]) == 2
    assert "file not found" in capsys.readouterr().err
    bad = _gate_rows(tmp_path / "bad.jsonl", [{"direct": float("nan")}])
    assert vt.main(["gate", "--table", str(bad)]) == 2


# ---------------------------------------------------------------- select / apply

def test_select_records_the_calibration_table_and_keeps_partial_actions(okvqa_flow):
    run = okvqa_flow["calib_run"]
    vt.build(run)
    table = run.with_name("okvqa_gs_calib.v2_actions.jsonl")
    out = okvqa_flow["dir"] / "decisions" / "okvqa_gs.json"
    decisions = vt.select(table, out, delta=0.1, epsilon=1.0, seed=0)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["calibration"]["table_sha256"] == runio.sha256_file(table)
    assert saved["calibration"]["questions"] == len(CALIB_QIDS)
    assert saved["calibration"]["partially_available"] == ["static_text"]
    assert "static_text" in saved["v2"]["rule"]["evidence_ranking"]
    assert saved["calibration"]["delta"] == 0.1 and saved["calibration"]["epsilon"] == 1.0
    assert saved["v2"]["primary"] == "rule" and decisions["v2"]["rule"]["best_single"] == "direct"
    # Direct is right exactly where its margin is high, so v2-Rule accepts a switch to static_mixed.
    assert saved["v2"]["rule"]["fallback"] is False and saved["v2"]["rule"]["kind"] == "rule"


def test_select_freezes_the_adaptive_baselines_without_risk_control(okvqa_flow, tmp_path):
    saved = json.loads(_select(okvqa_flow, tmp_path).read_text(encoding="utf-8"))
    base = saved["baselines"]
    assert base["risk_controlled"] is False
    # Direct and static_mixed tie at 50 on calibration, so the stronger end is direct (always direct = tau +inf).
    assert base["stronger_single"] == "direct"
    # U is 0.1 where direct is right and 0.9 elsewhere. Any tau in [0.1, 0.9) scores 100; the quantile grid also
    # holds 0.5, and the tie goes to the end nearest always-direct, so tau = 0.5.
    assert base["u_gate"]["tau"] == pytest.approx(0.5) and base["u_gate"]["calibration_mean"] == pytest.approx(100.0)
    assert base["u_gate"]["above"] == "static_mixed" and base["u_gate"]["otherwise"] == "direct"
    # h_sem_pre is constant: every threshold ties, and the tie goes to the always-direct end, tau = +inf.
    assert base["nte"]["tau"] == "inf" and base["nte"]["calibration_mean"] == pytest.approx(50.0)
    means = base["calibration_means"]
    assert means["singles"]["direct"] == pytest.approx(50.0) and means["singles"]["static_mixed"] == pytest.approx(50.0)
    assert means["adaptive"] == {"u_gate": pytest.approx(100.0), "nte": pytest.approx(50.0)}
    assert means["arms"] == {"gated:static": pytest.approx(35.0), "universalrag": pytest.approx(60.0)}


def test_apply_reads_no_scores_to_choose_and_never_picks_an_unavailable_action(okvqa_flow, tmp_path):
    decisions = _select(okvqa_flow, tmp_path)

    test_run = okvqa_flow["test_run"]
    vt.build(test_run)
    table = test_run.with_name("okvqa_gs_test.v2_actions.jsonl")
    main_out = tmp_path / "main.json"
    result = vt.apply(table, decisions, main_out)
    choices = [json.loads(x) for x in main_out.with_name("main.choices.jsonl").read_text().splitlines()]
    chosen = {c["qid"]: (c["rule"]["action"], c["cal"]["action"], c["u_gate"]["action"], c["nte"]["action"])
              for c in choices}

    # Tamper with every test score and every baseline score: no choice may move, because the choices are read from
    # the actions and the baseline signals alone.
    tampered = tmp_path / "tampered.jsonl"
    rows = _table_rows(table)
    for row in rows:
        row["scores"] = {name: 100.0 - value for name, value in row["scores"].items()}
        row["baseline_scores"] = {arm: 100.0 - value for arm, value in row["baseline_scores"].items()}
    _jsonl(tampered, rows)
    tampered_out = tmp_path / "tampered_main.json"
    vt.apply(tampered, decisions, tampered_out)
    tampered_choices = [json.loads(x) for x in
                        tampered_out.with_name("tampered_main.choices.jsonl").read_text().splitlines()]
    assert {c["qid"]: (c["rule"]["action"], c["cal"]["action"], c["u_gate"]["action"], c["nte"]["action"])
            for c in tampered_choices} == chosen

    # Question 102 (index 1, direct wrong) has static_mixed unavailable: v2-Rule falls to the available
    # evidence action, static_image, and never to the unavailable one.
    by_table = {r["qid"]: r for r in rows}
    for qid, picks in chosen.items():
        for name in picks[:2]:
            assert by_table[qid]["actions"][name]["available"] is True
    assert chosen["102"][0] == "static_image"
    assert result["best_single"] == "direct"
    assert result["partially_available"] == ["static_text", "static_mixed"]


def test_apply_takes_the_best_single_from_the_decisions_not_the_test_set(okvqa_flow, tmp_path):
    decisions = _select(okvqa_flow, tmp_path)
    test_run = okvqa_flow["test_run"]
    vt.build(test_run)
    table = test_run.with_name("okvqa_gs_test.v2_actions.jsonl")
    rows = _table_rows(table)
    for row in rows:   # on the test set static_mixed looks far better than direct
        row["scores"]["static_mixed"], row["scores"]["direct"] = 100.0, 0.0
    flipped = _jsonl(tmp_path / "flipped.jsonl", rows)
    result = vt.apply(flipped, decisions, tmp_path / "flipped_main.json")
    assert result["best_single"] == "direct"
    assert result["baselines"]["static_mixed"]["mean"] == pytest.approx(100.0)
    assert result["baselines"]["direct"]["mean"] == pytest.approx(0.0)


def test_strongest_baseline_is_frozen_on_calibration_and_the_gates_stay_frozen(okvqa_flow, tmp_path):
    decisions = _select(okvqa_flow, tmp_path)
    before = decisions.read_bytes()
    test_run = okvqa_flow["test_run"]
    vt.build(test_run)
    table = test_run.with_name("okvqa_gs_test.v2_actions.jsonl")
    rows = _table_rows(table)
    # On the test set universalrag is far stronger than anything else, and the arm now beats the U gate.
    for row in rows:
        row["baseline_scores"] = {"gated:static": 0.0, "universalrag": 100.0}
    flipped = _jsonl(tmp_path / "flipped_arms.jsonl", rows)
    result = vt.apply(flipped, decisions, tmp_path / "flipped_arms_main.json")
    assert decisions.read_bytes() == before                 # apply writes nothing into the decisions
    assert result["strongest_baseline"]["name"] == "u_gate"  # calibration winner: 100 against 60 for universalrag
    assert result["strongest_baseline"]["calibration_mean"] == pytest.approx(100.0)
    assert result["baselines"]["universalrag"]["mean"] == pytest.approx(100.0)
    assert result["baselines"]["u_gate"]["tau"] == pytest.approx(0.5)


def test_apply_reports_every_baseline_and_pairs_v2_rule_with_it(okvqa_flow, tmp_path):
    decisions = _select(okvqa_flow, tmp_path)
    test_run = okvqa_flow["test_run"]
    vt.build(test_run)
    result = vt.apply(test_run.with_name("okvqa_gs_test.v2_actions.jsonl"), decisions, tmp_path / "main.json")
    blocks = result["baselines"]
    assert {"direct", "static_mixed", "static_text", "static_image", "u_gate", "nte", "gated:static",
            "universalrag"} <= set(blocks)
    assert blocks["u_gate"]["kind"] == "adaptive" and blocks["gated:static"]["kind"] == "arm"
    assert blocks["u_gate"]["mean"] == pytest.approx(97.5)   # question 102 gets its reference (direct, wrong)
    assert blocks["u_gate"]["vs_rule"]["delta_pp"] == pytest.approx(0.0)
    assert blocks["universalrag"]["vs_rule"]["delta_pp"] == pytest.approx(
        result["policies"]["rule"]["mean"] - 60.0)
    assert set(blocks["universalrag"]["vs_rule"]) >= {"paired_ci95_pp", "wins", "losses", "ties"}
    assert result["strongest_baseline"]["name"] == "u_gate"
    assert result["strongest_baseline"]["vs_rule"] == blocks["u_gate"]["vs_rule"]


def test_a_star_unavailable_on_a_test_question_still_applies(okvqa_flow, tmp_path, monkeypatch):
    decisions = _select(okvqa_flow, tmp_path)
    test = _okvqa_records(TEST_QIDS, mixed_unavailable={"102"}, direct_unavailable={"105"})
    run = _write_run(okvqa_flow["dir"] / "runs" / "okvqa" / "okvqa_gs_test_no_direct.jsonl", test,
                     metric="vqa_accuracy", arms=_okvqa_arms(len(test)))
    vt.build(run)
    out = tmp_path / "no_direct_main.json"
    result = vt.apply(run.with_name("okvqa_gs_test_no_direct.v2_actions.jsonl"), decisions, out)
    choices = {c["qid"]: c for c in
               (json.loads(x) for x in out.with_name("no_direct_main.choices.jsonl").read_text().splitlines())}
    assert choices["105"]["reference"] == "static_mixed"     # a* is direct, unavailable here: first available in ranking
    for family in ("rule", "cal", "u_gate", "nte"):
        assert choices["105"][family]["action"] != "direct"
    assert result["reference_counts"] == {"direct": 39, "static_mixed": 1}     # 102: direct; 105: static_mixed


def test_apply_names_the_question_when_a_confidence_is_not_finite(okvqa_flow, tmp_path):
    decisions = _select(okvqa_flow, tmp_path)
    test_run = okvqa_flow["test_run"]
    vt.build(test_run)
    table = test_run.with_name("okvqa_gs_test.v2_actions.jsonl")
    rows = _table_rows(table)
    rows[4]["actions"]["direct"]["answer_token"]["u_raw"] = None          # a NaN value is stored as null
    broken = _jsonl(tmp_path / "broken.jsonl", rows)
    with pytest.raises(vt.TableError, match=f"question {rows[4]['qid']!r}"):
        vt.apply(broken, decisions, tmp_path / "broken_main.json")


def test_apply_refuses_decisions_calibrated_on_another_metric_or_action_set(okvqa_flow, tmp_path):
    decisions = _select(okvqa_flow, tmp_path)
    test_run = okvqa_flow["test_run"]
    vt.build(test_run)
    table = test_run.with_name("okvqa_gs_test.v2_actions.jsonl")

    rows = _table_rows(table)
    for row in rows:
        row["metric"] = "acc_approx"
    other_metric = _jsonl(tmp_path / "other_metric.jsonl", rows)
    with pytest.raises(vt.TableError, match="calibrated on"):
        vt.apply(other_metric, decisions, tmp_path / "x.json")

    rows = _table_rows(table)
    for row in rows:
        row["actions"].pop("static_text", None)
    no_static_text = _jsonl(tmp_path / "no_static_text.jsonl", rows)
    with pytest.raises(vt.TableError, match="static_text"):
        vt.apply(no_static_text, decisions, tmp_path / "y.json")


def test_apply_refuses_decisions_without_calibration_or_baselines(okvqa_flow, tmp_path):
    decisions = _select(okvqa_flow, tmp_path)
    doc = json.loads(decisions.read_text(encoding="utf-8"))
    test_run = okvqa_flow["test_run"]
    vt.build(test_run)
    table = test_run.with_name("okvqa_gs_test.v2_actions.jsonl")
    for key in ("baselines", "calibration"):
        without = tmp_path / f"without_{key}.json"
        without.write_text(json.dumps({k: v for k, v in doc.items() if k != key}), encoding="utf-8")
        assert vt.main(["apply", "--table", str(table), "--decisions", str(without),
                        "--out", str(tmp_path / "z.json")]) == 2


def test_full_flow_build_gate_select_apply_through_the_command_line(okvqa_flow, tmp_path, capsys):
    calib, test = okvqa_flow["calib_run"], okvqa_flow["test_run"]
    assert vt.main(["build", "--run", str(calib)]) == 0
    assert vt.main(["build", "--run", str(test)]) == 0
    calib_table = calib.with_name("okvqa_gs_calib.v2_actions.jsonl")
    test_table = test.with_name("okvqa_gs_test.v2_actions.jsonl")
    assert vt.main(["gate", "--table", str(calib_table)]) == 0
    decisions = tmp_path / "decisions" / "okvqa_gs.json"
    assert vt.main(["select", "--table", str(calib_table), "--out", str(decisions)]) == 0
    main_out = tmp_path / "okvqa_main.json"
    assert vt.main(["apply", "--table", str(test_table), "--decisions", str(decisions), "--out", str(main_out)]) == 0
    result = json.loads(main_out.read_text(encoding="utf-8"))
    # The rule switches exactly the questions where direct is wrong. Question 102 has static_mixed unavailable, so
    # it gets static_image (wrong there): 39 questions at 100 and one at 0. The oracle also counts static_text,
    # which is right on question 102, so it is 100 and the rule captures 47.5 of the 50 points available.
    rule = result["policies"]["rule"]
    assert rule["mean"] == pytest.approx(97.5)
    assert rule["capture_ratio"] == pytest.approx(0.95)
    assert rule["vs_reference"]["delta_pp"] == pytest.approx(47.5)
    assert result["oracle_mean"] == pytest.approx(100.0)
    assert result["table"]["sha256"] == runio.sha256_file(test_table)
    assert result["decisions"]["sha256"] == runio.sha256_file(decisions)
    assert result["strongest_baseline"]["name"] == "u_gate"


def test_apply_refuses_a_decisions_file_without_a_v2_entry(okvqa_flow, tmp_path, capsys):
    test = okvqa_flow["test_run"]
    vt.build(test)
    table = test.with_name("okvqa_gs_test.v2_actions.jsonl")
    decisions = _jsonl(tmp_path / "bad.json", [{"note": "no v2 entry"}])
    assert vt.main(["apply", "--table", str(table), "--decisions", str(decisions),
                    "--out", str(tmp_path / "out.json")]) == 2
    assert "frozen decisions" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


# ---------------------------------------------------------------- devcheck

def test_devcheck_passes_when_the_frozen_rule_beats_a_star_on_the_development_table(okvqa_flow, tmp_path, capsys):
    decisions = _select(okvqa_flow, tmp_path)
    dev_run = okvqa_flow["test_run"]           # stands in for the S2 development run
    vt.build(dev_run)
    table = dev_run.with_name("okvqa_gs_test.v2_actions.jsonl")
    out = tmp_path / "decisions.dev_check.json"
    assert vt.main(["devcheck", "--table", str(table), "--decisions", str(decisions), "--out", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "PASS" and "primary=rule" in lines[1]
    check = json.loads(out.read_text(encoding="utf-8"))
    # Same numbers as apply: the rule scores 97.5 against a* = direct at 50.
    assert check["passed"] is True and check["primary"] == "rule" and check["best_single"] == "direct"
    assert check["reference_mean"] == pytest.approx(50.0) and check["policy_mean"] == pytest.approx(97.5)
    assert check["relative_gain"] == pytest.approx(0.95) and check["ci95_pp"][0] > 0
    assert check["decisions"]["sha256"] == runio.sha256_file(decisions)
    assert check["table"]["sha256"] == runio.sha256_file(table)
    dev_main = table.with_name("okvqa_gs_test.v2_dev_main.json")
    assert check["dev_main"] == {"path": str(dev_main), "sha256": runio.sha256_file(dev_main)}


def test_devcheck_fails_when_the_selector_gains_nothing_on_the_development_table(okvqa_flow, tmp_path, capsys):
    decisions = _select(okvqa_flow, tmp_path)
    dev_run = okvqa_flow["test_run"]
    vt.build(dev_run)
    rows = _table_rows(dev_run.with_name("okvqa_gs_test.v2_actions.jsonl"))
    for row in rows:                           # every action scores the same: no selector can beat a*
        row["scores"] = {name: 40.0 for name in row["scores"]}
    flat = _jsonl(tmp_path / "flat.v2_actions.jsonl", rows)
    out = tmp_path / "decisions.dev_check.json"
    assert vt.main(["devcheck", "--table", str(flat), "--decisions", str(decisions), "--out", str(out)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "FAIL" and "不要直接跑 S4" in lines[-1]
    check = json.loads(out.read_text(encoding="utf-8"))
    assert check["passed"] is False and check["delta_pp"] == 0.0 and check["relative_gain"] == 0.0


def test_devcheck_needs_the_gain_and_a_positive_ci_lower_bound(monkeypatch, tmp_path):
    def fake_apply(table, decisions, out):
        out.write_text("{}", encoding="utf-8")
        return {"frozen_primary": "rule", "best_single": "direct", "metric": "vqa_accuracy", "questions": 10,
                "strongest_baseline": {}, "table": {}, "decisions": {},
                "policies": {"rule": {"decision": {"fallback": False}, "vs_reference": paired}}}

    monkeypatch.setattr(vt, "apply", fake_apply)
    table = tmp_path / "d.v2_actions.jsonl"
    base = {"baseline": 50.0, "wins": 1, "losses": 0, "ties": 9}
    paired = {**base, "ours": 53.0, "delta_pp": 3.0, "paired_ci95_pp": [0.5, 5.0]}      # +6%, CI above 0
    assert vt.dev_check(table, tmp_path / "x.json", tmp_path / "c.json")[0] is True
    paired = {**base, "ours": 53.0, "delta_pp": 3.0, "paired_ci95_pp": [-0.5, 6.0]}     # +6%, CI crosses 0
    assert vt.dev_check(table, tmp_path / "x.json", tmp_path / "c.json")[0] is False
    paired = {**base, "ours": 52.0, "delta_pp": 2.0, "paired_ci95_pp": [0.5, 3.5]}      # +4%, CI above 0
    assert vt.dev_check(table, tmp_path / "x.json", tmp_path / "c.json")[0] is False


def test_devcheck_refuses_a_table_that_is_not_an_action_table(okvqa_flow, tmp_path, capsys):
    decisions = _select(okvqa_flow, tmp_path)
    assert vt.main(["devcheck", "--table", str(tmp_path / "dev.jsonl"), "--decisions", str(decisions),
                    "--out", str(tmp_path / "c.json")]) == 2
    assert "v2_actions.jsonl" in capsys.readouterr().err


def test_deployment_summary_reports_cost_and_agreement():
    deployed = [{"action": "direct", "prefill_calls": 1, "decode_calls": 1, "seconds": 0.2, "matches_full_decode": True},
                {"action": "static_mixed", "prefill_calls": 4, "decode_calls": 1, "seconds": 0.6,
                 "matches_full_decode": True}]
    summary = vt.deployment_summary(deployed, 2)
    assert summary["prefill_calls_mean"] == 2.5 and summary["decode_calls_mean"] == 1.0
    assert summary["chosen"] == {"direct": 1, "static_mixed": 1} and summary["matches_full_decode"] == 2

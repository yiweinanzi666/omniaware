"""v2_table build -> gate -> select -> apply on tiny synthetic runs, plus the properties the table must keep."""
import json
import zipfile

import pytest

from omniaware import paths, runio
from omniaware.eval import webqa_acc
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
            "kl_to_direct": kl, "retriever": {"top1_score": 0.5}}


def _okvqa_records(qids, *, mixed_unavailable=()):
    """Direct is right on even questions and static_mixed on odd ones (so the oracle is 100 everywhere and the best
    single action is a 50 tie, which goes to direct). static_text is missing on every fourth question, so it is a
    partially available action. static_image is always available and is wrong except every fifth question."""
    records = []
    for i, qid in enumerate(qids):
        gold = _gold(i)
        direct_right = i % 2 == 0
        mixed = _action(gold, 0.5) if not direct_right else _action("nope", 0.5)
        if qid in mixed_unavailable:
            mixed = _action("", 0.0, available=False)
        records.append({
            "qid": qid, "dataset": "okvqa", "qcate": "okvqa", "golds": [gold],
            "actions": {
                "direct": _action(gold if direct_right else "wrong", 0.9 if direct_right else 0.1),
                "static_text": _action(gold, 0.4) if i % 4 else _action("", 0.0, available=False),
                "static_image": _action(gold if i % 5 == 0 else "nope", 0.3),
                "static_mixed": mixed,
            },
        })
    return records


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


def _table_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


CALIB_QIDS = [str(q) for q in range(1, 41)]
TEST_QIDS = [str(q) for q in range(101, 141)]


@pytest.fixture
def okvqa_flow(tmp_path, monkeypatch):
    """Calibration and test runs with their annotations, as omniaware.run --v2 would write them."""
    calib = _okvqa_records(CALIB_QIDS)
    test = _okvqa_records(TEST_QIDS, mixed_unavailable={"102"})
    _write_okvqa_annotations(tmp_path, monkeypatch, calib + test)
    run_dir = tmp_path / "runs" / "okvqa"
    run_dir.mkdir(parents=True)
    calib_run = _jsonl(run_dir / "okvqa_gs_calib.jsonl", calib)
    test_run = _jsonl(run_dir / "okvqa_gs_test.jsonl", test)
    return {"dir": tmp_path, "calib_run": calib_run, "test_run": test_run}


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


def test_build_refuses_a_run_without_actions(tmp_path):
    rows = [{"qid": "1", "dataset": "okvqa", "golds": ["x"], "preds": {"direct": "x"}}]
    run = _jsonl(tmp_path / "okvqa_gs.jsonl", rows)
    with pytest.raises(vt.TableError, match="--v2"):
        vt.build(run)


def test_build_webqa_subsets_come_from_gold_modalities(tmp_path, monkeypatch):
    class Exact:
        def __call__(self, prediction, reference, Qcate):
            return {"acc_approx": float(prediction.strip().lower() == reference.strip().lower())}
    monkeypatch.setattr(webqa_acc, "load_official_metric", lambda: (Exact(), {"stand_in": True}))
    rows = [
        {"qid": "g1", "dataset": "webqa", "qcate": "text", "golds": ["Paris"], "gold_modalities": ["image"],
         "actions": {"direct": _action("Paris", 0.9), "static_mixed": _action("Rome", 0.2)}},
        {"qid": "g2", "dataset": "webqa", "qcate": "text", "golds": ["Rome"], "gold_modalities": ["text"],
         "actions": {"direct": _action("Milan", 0.1), "static_mixed": _action("Rome", 0.2)}},
    ]
    run = _jsonl(tmp_path / "webqa_full.jsonl", rows)
    summary = vt.build(run)
    table = {r["qid"]: r for r in _table_rows(run.with_name("webqa_full.v2_actions.jsonl"))}
    assert table["g1"]["subset"] == "image_question" and table["g2"]["subset"] == "text_question"
    assert table["g1"]["scores"] == {"direct": 100.0, "static_mixed": 0.0}
    assert table["g1"]["metric"] == "acc_approx"
    assert summary["best_single"] == "direct"             # both actions score 50: the tie goes to direct

    rows[0].pop("gold_modalities")
    run = _jsonl(tmp_path / "webqa_nomod.jsonl", rows)
    with pytest.raises(ValueError, match="gold_modalities"):
        vt.build(run)


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


def test_gate_fails_when_the_oracle_equals_the_best_single(tmp_path, capsys):
    table = _gate_rows(tmp_path / "t.jsonl", [{"direct": 60.0, "static_mixed": 50.0},
                                              {"direct": 60.0, "static_mixed": 50.0}])
    assert vt.main(["gate", "--table", str(table)]) == 1
    assert capsys.readouterr().out.splitlines()[0] == "FAIL"


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


def test_apply_reads_no_scores_to_choose_and_never_picks_an_unavailable_action(okvqa_flow, tmp_path):
    calib = okvqa_flow["calib_run"]
    vt.build(calib)
    decisions = tmp_path / "decisions.json"
    vt.select(calib.with_name("okvqa_gs_calib.v2_actions.jsonl"), decisions, 0.1, 1.0, 0)

    test_run = okvqa_flow["test_run"]
    vt.build(test_run)
    table = test_run.with_name("okvqa_gs_test.v2_actions.jsonl")
    main_out = tmp_path / "main.json"
    result = vt.apply(table, decisions, main_out)
    choices = [json.loads(x) for x in main_out.with_name("main.choices.jsonl").read_text().splitlines()]
    chosen = {c["qid"]: (c["rule"]["action"], c["cal"]["action"]) for c in choices}

    # Tamper with every test score: the choices must not move, because they are read from the actions alone.
    tampered = tmp_path / "tampered.jsonl"
    rows = _table_rows(table)
    for row in rows:
        row["scores"] = {name: 100.0 - value for name, value in row["scores"].items()}
    _jsonl(tampered, rows)
    tampered_out = tmp_path / "tampered_main.json"
    vt.apply(tampered, decisions, tampered_out)
    tampered_choices = [json.loads(x) for x in
                        tampered_out.with_name("tampered_main.choices.jsonl").read_text().splitlines()]
    assert {c["qid"]: (c["rule"]["action"], c["cal"]["action"]) for c in tampered_choices} == chosen

    # Question 102 (index 1, direct wrong) has static_mixed unavailable: v2-Rule falls to the available
    # evidence action, static_image, and never to the unavailable one.
    by_table = {r["qid"]: r for r in rows}
    for qid, (rule_action, cal_action) in chosen.items():
        for name in (rule_action, cal_action):
            assert by_table[qid]["actions"][name]["available"] is True
    assert chosen["102"][0] == "static_image"
    assert result["best_single"] == "direct"
    assert result["partially_available"] == ["static_text", "static_mixed"]


def test_apply_takes_the_best_single_from_the_decisions_not_the_test_set(okvqa_flow, tmp_path):
    calib = okvqa_flow["calib_run"]
    vt.build(calib)
    decisions = tmp_path / "decisions.json"
    vt.select(calib.with_name("okvqa_gs_calib.v2_actions.jsonl"), decisions, 0.1, 1.0, 0)
    test_run = okvqa_flow["test_run"]
    vt.build(test_run)
    table = test_run.with_name("okvqa_gs_test.v2_actions.jsonl")
    rows = _table_rows(table)
    for row in rows:   # on the test set static_mixed looks far better than direct
        row["scores"]["static_mixed"], row["scores"]["direct"] = 100.0, 0.0
    flipped = _jsonl(tmp_path / "flipped.jsonl", rows)
    result = vt.apply(flipped, decisions, tmp_path / "flipped_main.json")
    assert result["best_single"] == "direct"
    assert result["singles"]["static_mixed"]["mean"] == pytest.approx(100.0)


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
    assert rule["vs_best_single"]["delta_pp"] == pytest.approx(47.5)
    assert result["oracle_mean"] == pytest.approx(100.0)
    assert result["table"]["sha256"] == runio.sha256_file(test_table)
    assert result["decisions"]["sha256"] == runio.sha256_file(decisions)


def test_apply_refuses_a_decisions_file_without_a_v2_entry(okvqa_flow, tmp_path, capsys):
    test = okvqa_flow["test_run"]
    vt.build(test)
    table = test.with_name("okvqa_gs_test.v2_actions.jsonl")
    decisions = _jsonl(tmp_path / "bad.json", [{"note": "no v2 entry"}])
    assert vt.main(["apply", "--table", str(table), "--decisions", str(decisions),
                    "--out", str(tmp_path / "out.json")]) == 2
    assert "frozen decisions" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()

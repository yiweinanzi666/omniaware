"""The pre-registered secondary metric semantic_acc and its human audit. No model is loaded: a stub checker
returns entailment probabilities from a table."""
import csv
import json

import pytest

from omniaware import runio
from omniaware.eval import score_official, semantic
from scripts.analysis import semantic_audit as audit
from scripts.analysis import v2_table as vt
from tests import test_v2_table as tvt


class StubChecker:
    """Entailment is 0.9 when the prediction statement contains the reference word, else 0.1."""

    def __init__(self):
        self.pairs = []

    def entailment(self, pairs):
        self.pairs += pairs
        return [0.9 if h.split("Answer: ")[1].lower() in p.split("Answer: ")[1].lower() else 0.1 for p, h in pairs]


def test_okvqa_references_need_three_annotators_or_fall_back_to_the_most_frequent():
    strong = {"answers": [{"answer": a} for a in ["dog"] * 3 + ["puppy"] * 2 + ["cat"] * 5]}
    assert semantic.okvqa_references(strong) == ["cat", "dog"]
    weak = {"answers": [{"answer": a} for a in ["red", "red", "crimson", "maroon", "blue", "teal", "x", "y", "z", "w"]]}
    assert semantic.okvqa_references(weak) == ["red"]


def test_semantic_acc_credits_entailment_scores_empty_as_zero_and_scores_each_pair_once():
    checker = StubChecker()
    preds = {"a": {"q1": "It is a dog.", "q2": "", "q3": "blue"}, "b": {"q1": "It is a dog.", "q2": "cat", "q3": ""}}
    out = semantic.score_semantic("okvqa", preds, ["q1", "q2", "q3"], {q: f"what {q}?" for q in ("q1", "q2", "q3")},
                                  {"q1": ["dog"], "q2": ["cat"], "q3": ["red"]}, checker)
    assert out == {"a": {"q1": 100.0, "q2": 0.0, "q3": 0.0}, "b": {"q1": 100.0, "q2": 100.0, "q3": 0.0}}
    assert len(checker.pairs) == 3                          # q1 shared by both arms; empty answers never sent
    with pytest.raises(ValueError):
        semantic.score_semantic("hybridqa", preds, ["q1"], {}, {}, checker)


def test_semantic_identity_names_the_pinned_checker_and_the_rule():
    identity = semantic.checker_identity()
    assert identity["threshold"] == 0.5 and identity["min_annotators"] == 3 and identity["truncation"] is False
    assert identity["model"].startswith("MoritzLaurer/DeBERTa-v3-large")


@pytest.fixture
def semantic_flow(okvqa_flow, monkeypatch):
    """The OK-VQA calibration and test runs of test_v2_table, with semantic_acc in their official scores."""
    monkeypatch.setattr(semantic, "load_checker", lambda device=None: StubChecker())
    for run in (okvqa_flow["calib_run"], okvqa_flow["test_run"]):
        records = [json.loads(line) for line in run.read_text().splitlines()]
        run.write_text("".join(json.dumps({**r, "question": f"question {r['qid']}?"}) + "\n" for r in records))
        summary_path = run.with_name(f"{run.stem}.official_summary.json")
        summary = json.loads(summary_path.read_text())
        summary_path.write_text(json.dumps({**summary, "predictions_sha256": runio.sha256_file(run)}))
        path = run.with_name(f"{run.stem}.official_scores.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            for cell in row["scores"].values():
                cell["semantic_acc"] = 100.0 - cell["vqa_accuracy"]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return okvqa_flow


okvqa_flow = tvt.okvqa_flow


def test_the_table_carries_semantic_scores_and_the_main_output_reports_them_without_changing_a_choice(
        semantic_flow, tmp_path):
    calib, test = semantic_flow["calib_run"], semantic_flow["test_run"]
    vt.build(calib, semantic=True)
    summary = vt.build(test, semantic=True)
    assert summary["secondary_metric"]["name"] == "semantic_acc"
    rows = tvt._table_rows(test.with_name("okvqa_gs_test.v2_actions.jsonl"))
    assert all(set(r["secondary_scores"]) == set(r["scores"]) for r in rows)
    assert all(set(r["baseline_secondary"]) == set(r["baseline_scores"]) for r in rows)
    decisions = tmp_path / "d.json"
    vt.select(calib.with_name("okvqa_gs_calib.v2_actions.jsonl"), decisions, 0.1, 1.0, 0)
    table = test.with_name("okvqa_gs_test.v2_actions.jsonl")
    with_secondary = vt.apply(table, decisions, tmp_path / "with.json")
    plain_rows = [{k: v for k, v in r.items() if k not in ("secondary_scores", "baseline_secondary")} for r in rows]
    plain = tmp_path / "plain.v2_actions.jsonl"
    plain.write_text("".join(json.dumps(r) + "\n" for r in plain_rows))
    without = vt.apply(plain, decisions, tmp_path / "without.json")
    assert with_secondary["policies"]["rule"]["chosen"] == without["policies"]["rule"]["chosen"]
    assert "secondary" in with_secondary and "secondary" not in without
    secondary = with_secondary["secondary"]
    assert secondary["metric"] == "semantic_acc" and {"rule", "cal"} <= set(secondary["policies"])
    assert {"u_gate", "nte", "gated:static", "universalrag"} <= set(secondary["baselines"])


def test_build_with_semantic_needs_semantic_official_scores(okvqa_flow, monkeypatch):
    monkeypatch.setattr(semantic, "load_checker", lambda device=None: StubChecker())
    with pytest.raises(vt.TableError, match="semantic_acc"):
        vt.build(okvqa_flow["test_run"], semantic=True)


def test_audit_export_is_blind_and_stratified_and_score_reports_both_metrics(semantic_flow, tmp_path):
    test = semantic_flow["test_run"]
    vt.build(test, semantic=True)
    table = test.with_name("okvqa_gs_test.v2_actions.jsonl")
    out = tmp_path / "audit"
    assert audit.main(["export", "--table", str(table), "--out-dir", str(out), "--per-stratum", "5"]) == 0
    with (out / "audit_items.csv").open(newline="") as handle:
        items = list(csv.DictReader(handle))
    assert set(items[0]) == set(audit.COLUMNS)              # no action, score or stratum column
    key = json.loads((out / "audit_key.json").read_text())
    assert len(items) == len(key["items"]) and all(i["label"] == "" for i in items)
    for s in audit.STRATA:
        assert sum(i["stratum"] == s for i in key["items"]) == min(5, key["stratum_sizes"][s])
    labels = out / "audit_labels.csv"                        # a labeller who always agrees with the official metric
    with labels.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "label"])
        writer.writeheader()
        for i in key["items"]:
            writer.writerow({"item_id": i["item_id"], "label": "1" if i["official"] >= 50 else "0"})
    assert audit.main(["score", "--key", str(out / "audit_key.json"), "--labels", str(labels)]) == 0
    result = json.loads((out / "audit_result.json").read_text())
    assert result["metrics"]["official"]["population_agreement"] == pytest.approx(1.0)
    assert 0.0 <= result["metrics"]["semantic_acc"]["population_agreement"] <= 1.0
    assert result["metrics"]["official"]["sample_kappa"] == pytest.approx(1.0)


def test_audit_score_refuses_unlabelled_items(semantic_flow, tmp_path, capsys):
    test = semantic_flow["test_run"]
    vt.build(test, semantic=True)
    out = tmp_path / "audit"
    audit.export(test.with_name("okvqa_gs_test.v2_actions.jsonl"), out, 3)
    assert audit.main(["score", "--key", str(out / "audit_key.json"), "--labels", str(out / "audit_items.csv")]) == 2
    assert "not labelled" in capsys.readouterr().err


def test_score_official_adds_semantic_after_the_official_metric(monkeypatch):
    monkeypatch.setattr(semantic, "load_checker", lambda device=None: StubChecker())
    by_qid = {"q1": {"question": "colour?", "golds": ["The car is red."]}}
    scores = {"static": {"q1": {"acc_approx": 100.0}}}
    metrics = ["acc_approx"]
    identity = score_official.add_semantic("webqa", scores, metrics, {"static": {"q1": "The car is red."}}, ["q1"],
                                           by_qid)
    assert metrics == ["acc_approx", "semantic_acc"] and scores["static"]["q1"]["semantic_acc"] == 100.0
    assert identity["name"] == "semantic_acc"

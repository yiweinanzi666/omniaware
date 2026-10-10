"""Run the official OK-VQA and Video-MME evaluators on tiny synthetic runs."""
import json
import zipfile

import pytest

from omniaware import paths, runio
from omniaware.eval import score_official


def _write_run(path, dataset, preds, golds=("unused",)):
    rows = [{"qid": q, "dataset": dataset, "qcate": "x", "golds": list(golds), "preds": p} for q, p in preds.items()]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _write_config(path, qids):
    """The configuration omniaware.run writes next to a run: the evaluation set it expects."""
    config = path.with_name(f"{path.stem}_config.json")
    config.write_text(json.dumps({"evaluation_questions": len(qids),
                                  "evaluation_qids_sha256": runio.qid_set_sha256(qids)}))
    return config


def test_okvqa_uses_vqa_accuracy_not_any_match(tmp_path, monkeypatch):
    def ann(qid, answers):
        return {"question_id": int(qid), "question_type": "other", "answer_type": "other",
                "answers": [{"answer": a, "answer_id": i + 1} for i, a in enumerate(answers)]}
    annotations = {"annotations": [ann("1", ["car"] + ["truck"] * 9), ann("2", ["dog"] * 10)]}
    zpath = tmp_path / "ann.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("mscoco_val2014_annotations.json", json.dumps(annotations))
    monkeypatch.setattr(paths, "OKVQA_ANNOTATIONS", zpath)
    run = _write_run(tmp_path / "okvqa.jsonl", "okvqa",
                     {"1": {"static": "car", "eab": "truck"}, "2": {"static": "hotdog", "eab": "dog"}})
    _write_config(run, ["1", "2"])
    summary, details = score_official.score_run(run, references=("static",))
    by_q = {d["qid"]: d["scores"] for d in details}
    # One of ten annotators said "car": VQA accuracy is 1/3, not full credit.
    assert by_q["1"]["static"]["vqa_accuracy"] == pytest.approx(30.0, abs=0.1)
    assert by_q["1"]["eab"]["vqa_accuracy"] == pytest.approx(100.0)
    # No substring credit: "hotdog" does not match "dog".
    assert by_q["2"]["static"]["vqa_accuracy"] == 0
    cmp = summary["comparisons"]["static"]["eab"]["vqa_accuracy"]
    assert cmp["wins"] == 2 and cmp["losses"] == 0
    assert summary["question_check"]["mode"] == "exact" and summary["subset"] is False


def _universalrag_root(tmp_path, monkeypatch, queries):
    """A UniversalRAG suite tree with only the query files (golds are read from them)."""
    root = tmp_path / "ur"
    for name, rows in queries.items():
        (root / "dataset" / "query").mkdir(parents=True, exist_ok=True)
        (root / "dataset" / "query" / f"{name}.json").write_text(json.dumps(rows))
    monkeypatch.setattr(paths, "UNIVERSALRAG_ROOT", root)


def test_universalrag_suite_scores_and_clusters_videorag_by_video(tmp_path, monkeypatch):
    _universalrag_root(tmp_path, monkeypatch, {
        "videorag_synth": [{"index": "videorag_synth_v1_1", "answer": "Detach the caliper."},
                           {"index": "videorag_synth_v1_2", "answer": "Tighten the bolts."},
                           {"index": "videorag_synth_v2_1", "answer": "Add salt."}],
        "hybridqa": [{"index": "hybridqa_x", "answer": "1960", "aliases": []}]})
    rows = [{"qid": "videorag_synth_v1_1", "dataset": "videorag", "golds": ["Detach the caliper."],
             "preds": {"static": "Remove the wheel.", "eab": "Detach the caliper."}},
            {"qid": "videorag_synth_v1_2", "dataset": "videorag", "golds": ["Tighten the bolts."],
             "preds": {"static": "Tighten the bolts.", "eab": "Tighten the bolts."}},
            {"qid": "videorag_synth_v2_1", "dataset": "videorag", "golds": ["Add salt."],
             "preds": {"static": "Add pepper.", "eab": "Add salt."}}]
    path = tmp_path / "videorag_synth.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _write_config(path, [r["qid"] for r in rows])
    summary, details = score_official.score_run(path, references=("static",))
    assert summary["metrics"] == ["rouge_l"] and summary["means"]["eab"]["rouge_l"] == pytest.approx(100.0)
    assert summary["comparisons"]["static"]["eab"]["rouge_l"]["delta_pp"] > 0
    hq = tmp_path / "hybridqa.jsonl"
    hq.write_text(json.dumps({"qid": "hybridqa_x", "dataset": "hybridqa", "golds": ["1960"],
                              "preds": {"static": "In 1960.", "eab": "1960"}}) + "\n")
    _write_config(hq, ["hybridqa_x"])
    summary, _ = score_official.score_run(hq, references=("static",))
    assert summary["means"]["eab"]["em"] == 100 and summary["means"]["static"]["em"] == 0
    assert summary["means"]["static"]["f1"] == pytest.approx(200 / 3)


def test_runs_with_inconsistent_arms_are_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    rows = [{"qid": "1", "dataset": "okvqa", "preds": {"static": "a", "eab": "b"}},
            {"qid": "2", "dataset": "okvqa", "preds": {"static": "a"}}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(AssertionError, match="arms differ"):
        score_official.score_run(path)


# ------------------------------------------------ question-set integrity (M-57) --

def test_question_set_must_match_the_configuration(tmp_path):
    path = _write_run(tmp_path / "okvqa.jsonl", "okvqa", {"1": {"static": "a"}, "2": {"static": "b"}})
    with pytest.raises(ValueError, match="no okvqa_config.json"):
        score_official.check_question_set(path, ["1", "2"])
    assert score_official.check_question_set(path, ["1", "2"], subset=True)["mode"] == "subset"
    _write_config(path, ["1", "2", "3"])          # the run was meant to answer three questions
    with pytest.raises(ValueError, match="expects 3"):
        score_official.check_question_set(path, ["1", "2"])
    assert score_official.check_question_set(path, ["1", "2"], subset=True)["mode"] == "subset"
    _write_config(path, ["1", "9"])               # same count, different questions
    with pytest.raises(ValueError, match="question set differs"):
        score_official.check_question_set(path, ["1", "2"])
    _write_config(path, ["1", "2"])
    assert score_official.check_question_set(path, ["2", "1"]) == {
        "mode": "exact", "questions": 2, "question_set_sha256": runio.qid_set_sha256(["1", "2"])}


def _webqa_fixture(tmp_path, monkeypatch, answers):
    """A tiny WebQA annotation file; answers maps guid -> list of gold answers."""
    raw = {guid: {"split": "val", "Q": f"question {guid}?", "Qcate": "text", "A": golds,
                  "txt_posFacts": [{"title": "t", "fact": f"fact {guid}"}], "txt_negFacts": [],
                  "img_posFacts": [], "img_negFacts": []} for guid, golds in answers.items()}
    path = tmp_path / "WebQA_train_val.json"
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(paths, "WEBQA_JSON", path)


class _ExactMetric:
    """Stand-in for the official approximation (its spaCy model is not needed to test the integrity checks)."""
    def __call__(self, prediction, reference, Qcate):
        return {"acc_approx": float(prediction.strip().lower() == reference.strip().lower())}


@pytest.fixture
def exact_webqa_metric(monkeypatch):
    from omniaware.eval import webqa_acc
    monkeypatch.setattr(webqa_acc, "load_official_metric", lambda: (_ExactMetric(), {"stand_in": True}))


def _webqa_run(tmp_path, golds_in_run, preds):
    rows = [{"qid": q, "dataset": "webqa", "qcate": "text", "golds": golds_in_run[q], "preds": p}
            for q, p in preds.items()]
    path = tmp_path / "webqa_open_full.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _write_config(path, list(preds))
    return path


def test_webqa_scores_against_the_single_reference(tmp_path, monkeypatch, exact_webqa_metric):
    _webqa_fixture(tmp_path, monkeypatch, {"g1": ["Paris"], "g2": ["Rome"]})
    path = _webqa_run(tmp_path, {"g1": ["Paris"], "g2": ["Rome"]},
                      {"g1": {"static": "Paris", "eab": "paris"}, "g2": {"static": "Milan", "eab": "Rome"}})
    summary, _ = score_official.score_run(path, references=("static",))
    assert summary["means"]["static"]["acc_approx"] == pytest.approx(50.0)
    assert summary["means"]["eab"]["acc_approx"] == pytest.approx(100.0)


def test_webqa_multiple_references_are_refused_not_truncated(tmp_path, monkeypatch, exact_webqa_metric):
    _webqa_fixture(tmp_path, monkeypatch, {"g1": ["Paris", "Lutetia"]})
    path = _webqa_run(tmp_path, {"g1": ["Paris", "Lutetia"]}, {"g1": {"static": "Paris"}})
    with pytest.raises(ValueError, match="exactly one per question"):
        score_official.score_run(path)


def test_webqa_golds_must_match_the_annotation_file(tmp_path, monkeypatch, exact_webqa_metric):
    _webqa_fixture(tmp_path, monkeypatch, {"g1": ["Paris"]})
    path = _webqa_run(tmp_path, {"g1": ["London"]}, {"g1": {"static": "Paris"}})
    with pytest.raises(ValueError, match="differ from the dataset"):
        score_official.score_run(path)


def test_webqa_question_missing_from_annotation_is_refused(tmp_path, monkeypatch, exact_webqa_metric):
    _webqa_fixture(tmp_path, monkeypatch, {"g1": ["Paris"]})
    path = _webqa_run(tmp_path, {"g2": ["Rome"]}, {"g2": {"static": "Rome"}})
    with pytest.raises(ValueError, match="cannot be verified"):
        score_official.score_run(path)


def test_bartscore_runs_on_the_requested_device_and_failure_keeps_acc(tmp_path, monkeypatch, exact_webqa_metric):
    from omniaware.eval import webqa_bart
    _webqa_fixture(tmp_path, monkeypatch, {"g1": ["Paris"]})
    path = _webqa_run(tmp_path, {"g1": ["Paris"]}, {"g1": {"static": "Paris"}})
    seen = {}

    def failing_parabank(*, device="cpu", cache_dir=None):
        seen["device"] = device
        raise RuntimeError("weights missing")
    monkeypatch.setattr(webqa_bart, "load_parabank", failing_parabank)
    summary, details = score_official.score_run(path, references=("static",), bart=True, device="cpu")
    assert seen["device"] == "cpu"
    assert summary["metrics"] == ["acc_approx"] and "weights missing" in summary["bart_error"]
    assert summary["means"]["static"]["acc_approx"] == pytest.approx(100.0)
    assert details[0]["scores"]["static"]["acc_approx"] == pytest.approx(100.0)


def test_default_device_is_cuda_only_when_available(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert score_official.default_device(None) == "cpu"
    assert score_official.default_device("cuda:1") == "cuda:1"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert score_official.default_device(None) == "cuda"


def test_summary_and_details_are_strict_json(tmp_path, monkeypatch):
    monkeypatch.setattr(score_official, "score_run", lambda *a, **k: (
        {"means": {"static": {"acc": float("nan")}}}, [{"qid": "q", "scores": {"x": float("inf")}}]))
    monkeypatch.setattr("sys.argv", ["score_official", str(tmp_path / "run.jsonl")])
    score_official.main()
    summary = json.loads((tmp_path / "run.official_summary.json").read_text())
    assert summary["means"]["static"]["acc"] is None
    details = json.loads((tmp_path / "run.official_scores.jsonl").read_text())
    assert details["scores"]["x"] is None

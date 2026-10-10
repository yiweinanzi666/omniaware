"""Run the official OK-VQA evaluator and the UniversalRAG-suite scorers on tiny synthetic runs."""
import json
import zipfile

import pytest

from omniaware import paths
from omniaware.eval import score_official


def _write_run(path, dataset, preds):
    rows = [{"qid": q, "dataset": dataset, "qcate": "x", "golds": ["unused"], "preds": p} for q, p in preds.items()]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


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
    summary, details = score_official.score_run(run, references=("static",))
    by_q = {d["qid"]: d["scores"] for d in details}
    # One of ten annotators said "car": VQA accuracy is 1/3, not full credit.
    assert by_q["1"]["static"]["vqa_accuracy"] == pytest.approx(30.0, abs=0.1)
    assert by_q["1"]["eab"]["vqa_accuracy"] == pytest.approx(100.0)
    # No substring credit: "hotdog" does not match "dog".
    assert by_q["2"]["static"]["vqa_accuracy"] == 0
    cmp = summary["comparisons"]["static"]["eab"]["vqa_accuracy"]
    assert cmp["wins"] == 2 and cmp["losses"] == 0


def test_universalrag_suite_scores_and_clusters_videorag_by_video(tmp_path):
    rows = [{"qid": "videorag_synth_v1_1", "dataset": "videorag", "golds": ["Detach the caliper."],
             "preds": {"static": "Remove the wheel.", "eab": "Detach the caliper."}},
            {"qid": "videorag_synth_v1_2", "dataset": "videorag", "golds": ["Tighten the bolts."],
             "preds": {"static": "Tighten the bolts.", "eab": "Tighten the bolts."}},
            {"qid": "videorag_synth_v2_1", "dataset": "videorag", "golds": ["Add salt."],
             "preds": {"static": "Add pepper.", "eab": "Add salt."}}]
    path = tmp_path / "videorag_synth.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary, details = score_official.score_run(path, references=("static",))
    assert summary["metrics"] == ["rouge_l"] and summary["means"]["eab"]["rouge_l"] == pytest.approx(100.0)
    assert summary["comparisons"]["static"]["eab"]["rouge_l"]["delta_pp"] > 0
    hq = tmp_path / "hybridqa.jsonl"
    hq.write_text(json.dumps({"qid": "hybridqa_x", "dataset": "hybridqa", "golds": ["1960"],
                              "preds": {"static": "In 1960.", "eab": "1960"}}) + "\n")
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

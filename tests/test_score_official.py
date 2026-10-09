"""Run the official OK-VQA and Video-MME evaluators on tiny synthetic runs."""
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


def test_videomme_parses_letters_and_clusters_by_video(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"question_id": ["001-1", "001-2", "002-1"], "videoID": ["v1", "v1", "v2"],
                      "answer": ["A", "B", "C"]})
    pq.write_table(table, tmp_path / "official_annotations.parquet")
    monkeypatch.setattr(paths, "VIDEOMME_DIR", tmp_path)
    run = _write_run(tmp_path / "videomme.jsonl", "videomme", {
        "001-1": {"static": "A", "eab": "The answer is A."},
        "001-2": {"static": "none of these", "eab": "B"},
        "002-1": {"static": "D", "eab": "C. something"}})
    summary, _ = score_official.score_run(run, references=("static",))
    assert summary["means"]["static"]["choice_accuracy"] == pytest.approx(100 / 3)
    assert summary["means"]["eab"]["choice_accuracy"] == pytest.approx(100.0)
    assert summary["answered"] == {"static": 2, "eab": 3}


def test_runs_with_inconsistent_arms_are_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    rows = [{"qid": "1", "dataset": "okvqa", "preds": {"static": "a", "eab": "b"}},
            {"qid": "2", "dataset": "okvqa", "preds": {"static": "a"}}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(AssertionError, match="arms differ"):
        score_official.score_run(path)

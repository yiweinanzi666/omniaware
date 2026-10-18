"""A3 / B-7: official scores only; WebQA subsets from gold modalities; one name per metric; shard merge statistics."""
import json
from pathlib import Path

import pytest

from omniaware import merge_shards, paths, runio
from omniaware.eval import score_official


PAPER_SCRIPTS = [
    Path("scripts/paper_runs.py"),
    Path("scripts/analysis/paper_analyses.py"),
    Path("src/omniaware/eval/score_official.py"),
]


def test_okvqa_table_path_is_score_okvqa_and_the_diagnostic_is_gone():
    assert score_official.SCORERS["okvqa"] is score_official.score_okvqa
    assert not hasattr(score_official, "diagnostic_substring_accuracy")
    for path in PAPER_SCRIPTS:
        assert "diagnostic_substring_accuracy" not in path.read_text(encoding="utf8")


class _ExactMetric:
    def __call__(self, prediction, reference, Qcate):
        return {"acc_approx": float(prediction.strip().lower() == reference.strip().lower())}


@pytest.fixture
def exact_webqa_metric(monkeypatch):
    from omniaware.eval import webqa_acc
    monkeypatch.setattr(webqa_acc, "load_official_metric", lambda: (_ExactMetric(), {"stand_in": True}))


def _webqa_run(tmp_path, monkeypatch, preds, golds, modalities):
    raw = {guid: {"split": "val", "Q": "q?", "Qcate": "text", "A": [golds[guid]],
                  "txt_posFacts": [{"title": "t", "fact": f"fact {guid}"}], "txt_negFacts": [],
                  "img_posFacts": [], "img_negFacts": []}
           for guid in golds}
    ann = tmp_path / "WebQA_train_val.json"
    ann.write_text(json.dumps(raw))
    monkeypatch.setattr(paths, "WEBQA_JSON", ann)
    rows = []
    for qid, pred in preds.items():
        rows.append({"qid": qid, "dataset": "webqa", "qcate": "text", "golds": [golds[qid]],
                     "gold_modalities": modalities[qid], "preds": {"static": pred}})
    path = tmp_path / "webqa_open_full.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    path.with_name("webqa_open_full_config.json").write_text(json.dumps({
        "evaluation_questions": len(preds),
        "evaluation_qids_sha256": runio.qid_set_sha256(preds)}))
    return path


def test_webqa_subset_needs_gold_modalities_and_ignores_qcate():
    with pytest.raises(ValueError, match="gold_modalities"):
        score_official.webqa_subset_name({"qid": "w1", "qcate": "image"})
    assert score_official.webqa_subset_name({"gold_modalities": ["image"], "qcate": "text"}) == "image_question"
    assert score_official.webqa_subset_name({"gold_modalities": ["text"], "qcate": "img"}) == "text_question"


def test_citation_dump_is_not_scored_as_the_answer(tmp_path, monkeypatch, exact_webqa_metric):
    path = _webqa_run(
        tmp_path, monkeypatch,
        {"g1": "Sources: none", "g2": "Paris is the capital.\nSources: [E1]"},
        {"g1": "Paris", "g2": "Paris is the capital."},
        {"g1": ["text"], "g2": ["image"]},
    )
    summary, _ = score_official.score_run(path, references=())
    assert summary["primary_metric"] == "acc_approx"
    assert summary["means"]["static"]["acc_approx"] == pytest.approx(50.0)
    assert set(summary["means"]["static"]) == {"acc_approx"}
    assert "normalized_bart" not in summary and "QA_approx" not in summary
    assert summary["subsets"]["text_question"]["n"] == 1
    assert summary["subsets"]["image_question"]["n"] == 1
    assert summary["empty_answer_rate"] == pytest.approx(0.5)
    assert "EMPTY ANSWER RATE" in summary["empty_answer_warning"]
    assert f"{summary['empty_answer_rate']:.6f}" in summary["empty_answer_warning"]


def test_webqa_summary_names_each_metric_once_when_fluency_is_computed(tmp_path, monkeypatch, exact_webqa_metric):
    from omniaware.eval import webqa_bart
    path = _webqa_run(tmp_path, monkeypatch, {"g1": "Paris"}, {"g1": "Paris"}, {"g1": ["text"]})

    def fake_load(*, device="cpu", cache_dir=None):
        return object(), {}

    def fake_add(rows, scorer, variants=None, **kwargs):
        out = []
        for row in rows:
            out.append(dict(row, normalized_bart_score=0.5,
                            acc_approx_x_normalized_bart=row["acc_approx"] * 0.5,
                            bart_reference_to_prediction_logscore=-1.0,
                            bart_reference_self_logscore=-1.0))
        return out

    monkeypatch.setattr(webqa_bart, "load_parabank", fake_load)
    monkeypatch.setattr(webqa_bart, "add_fluency", fake_add)
    summary, details = score_official.score_run(path, references=(), bart=True, device="cpu")
    assert summary["metrics"] == ["acc_approx", "normalized_bart", "QA_approx"]
    assert summary["primary_metric"] == "acc_approx"
    means = summary["means"]["static"]
    assert means["acc_approx"] == pytest.approx(100.0)
    assert means["normalized_bart"] == pytest.approx(50.0)
    assert means["QA_approx"] == pytest.approx(50.0)
    assert "normalized_bart" not in summary and "QA_approx" not in summary     # one place: means
    cell = details[0]["scores"]["static"]
    assert set(cell) == {"acc_approx", "normalized_bart", "QA_approx"}
    assert cell["QA_approx"] == pytest.approx(50.0)


def test_shard_merge_writes_the_empty_answer_rate_over_preds_and_available_actions(tmp_path):
    stem = tmp_path / "okvqa"
    first = {"qid": "q1", "preds": {"direct": "Paris", "static": "Sources: none"},
             "actions": {"direct": {"available": True, "answer_empty": False},
                         "static_image": {"available": False, "answer_empty": True}}}
    second = {"qid": "q2", "preds": {"direct": "", "static": "Paris"}, "actions": {}}
    (tmp_path / "okvqa.shard00-of-01.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (first, second)), encoding="utf8")
    merge_shards.merge(stem, expected_questions=2)
    marker = json.loads(runio.completion_marker(tmp_path, "okvqa").read_text(encoding="utf8"))
    # q1: direct, static, direct-action (static_image is unavailable and not counted); q2: direct, static
    assert marker["n_answers"] == 5 and marker["n_empty"] == 2
    assert marker["empty_answer_rate"] == pytest.approx(0.4)
    assert "EMPTY ANSWER RATE" in marker["empty_answer_warning"]
    assert marker["n_refusal"] == 0 and marker["empty_by_arm"]["static"]["citation_only"] == 1

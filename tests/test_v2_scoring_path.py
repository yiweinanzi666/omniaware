"""A3: official scores only, and WebQA summaries emit acc_approx, BART, and QA_approx."""
import json
from pathlib import Path

import pytest

from omniaware import paths, runio
from omniaware.eval import score_official


PAPER_SCRIPTS = [
    Path("scripts/paper_runs.py"),
    Path("scripts/analysis/paper_analyses.py"),
    Path("src/omniaware/eval/score_official.py"),
]


def test_okvqa_table_path_is_score_okvqa_not_the_diagnostic():
    assert score_official.SCORERS["okvqa"] is score_official.score_okvqa
    for path in PAPER_SCRIPTS:
        text = path.read_text(encoding="utf8")
        assert "diagnostic_substring_accuracy" not in text or path.name == "score_official.py"
    # The diagnostic lives in score_official and is not called from the scoring function.
    body = Path("src/omniaware/eval/score_official.py").read_text(encoding="utf8")
    score_run = body.split("def score_run", 1)[1].split("\ndef main", 1)[0]
    assert "diagnostic_substring_accuracy" not in score_run
    assert "score_okvqa" in body


def test_diagnostic_substring_is_not_the_official_match():
    assert score_official.diagnostic_substring_accuracy("hotdog", "dog") == 1.0
    assert score_official.diagnostic_substring_accuracy("car", "truck") == 0.0


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
    assert summary["normalized_bart"] is None and summary["QA_approx"] is None
    assert "normalized_bart" in summary and "QA_approx" in summary
    assert summary["subsets"]["text_question"]["n"] == 1
    assert summary["subsets"]["image_question"]["n"] == 1
    assert summary["empty_answer_rate"] == pytest.approx(0.5)
    assert "EMPTY ANSWER RATE" in summary["empty_answer_warning"]
    assert f"{summary['empty_answer_rate']:.6f}" in summary["empty_answer_warning"]


def test_webqa_summary_emits_bart_and_qa_approx_when_fluency_is_computed(tmp_path, monkeypatch, exact_webqa_metric):
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
    summary, _ = score_official.score_run(path, references=(), bart=True, device="cpu")
    assert summary["metrics"] == ["acc_approx", "normalized_bart", "QA_approx"]
    assert summary["primary_metric"] == "acc_approx"
    assert summary["means"]["static"]["acc_approx"] == pytest.approx(100.0)
    assert summary["means"]["static"]["normalized_bart"] == pytest.approx(50.0)
    assert summary["means"]["static"]["QA_approx"] == pytest.approx(50.0)
    assert summary["normalized_bart"]["static"] == pytest.approx(50.0)
    assert summary["QA_approx"]["static"] == pytest.approx(50.0)

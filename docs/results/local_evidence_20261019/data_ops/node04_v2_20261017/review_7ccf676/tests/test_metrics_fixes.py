"""Scoring fixes: UniversalRAG metrics (normalisation, golds, BERTScore references, clusters), WebQA source pins."""
import shutil
import sys
import types

import pytest

from omniaware import paths
from omniaware.data import videorag_video_id
from omniaware.eval import universalrag_metrics as um


class _Scores:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


def test_normalisation_folds_full_width_and_cjk_punctuation():
    assert um.exact_match("Paris。", "Paris") == 1.0
    assert um.exact_match("Ｐａｒｉｓ", "Paris") == 1.0          # NFKC: full-width letters
    assert um.f1("Paris、 France", "Paris France") == 1.0       # punctuation is deleted, as in SQuAD ("a,b" -> "ab")
    assert um.exact_match("Paris’s", "Paris's") == 1.0           # curly apostrophe
    assert um.exact_match("Paris", "Rome") == 0.0


def test_gold_that_normalises_to_nothing_or_is_missing_is_an_error():
    with pytest.raises(ValueError):                                  # an empty answer would match "。"
        um.score("hybridqa", {"a": {"q": ""}}, ["q"], {"q": {"golds": ["。"]}})
    with pytest.raises(ValueError):
        um.score("hybridqa", {"a": {"q": "x"}}, ["q"], {"q": {"golds": []}})
    with pytest.raises(ValueError):
        um.score("videorag", {"a": {"q": "x"}}, ["q"], {"q": {"golds": ["..."]}})


def test_bertscore_takes_the_best_gold_with_a_fixed_model(monkeypatch):
    calls = {}

    def fake_score(cands, refs, model_type=None, num_layers=None, verbose=False):
        calls.update(model_type=model_type, num_layers=num_layers)
        return None, None, _Scores([1.0 if c == r else 0.1 for c, r in zip(cands, refs)])

    monkeypatch.setitem(sys.modules, "bert_score", types.SimpleNamespace(score=fake_score))
    records = {"videorag_synth_v1_1": {"golds": ["Paris", "Lyon"]}, "videorag_synth_v2_1": {"golds": ["Rome"]}}
    q1, q2 = "videorag_synth_v1_1", "videorag_synth_v2_1"
    scores, metrics, extra = um.score("videorag", {"a": {q1: "Lyon", q2: "Milan"}}, [q1, q2], records,
                                      bertscore=True)
    assert metrics == ["rouge_l", "bertscore_f1"]
    assert scores["a"][q1]["bertscore_f1"] == pytest.approx(100.0)   # the second gold matches exactly
    assert scores["a"][q2]["bertscore_f1"] == pytest.approx(10.0)
    assert calls == {"model_type": um.BERTSCORE_MODEL, "num_layers": um.BERTSCORE_LAYERS}
    assert extra["bertscore"]["model_type"] == um.BERTSCORE_MODEL


def test_videorag_clusters_use_the_record_field_or_a_strictly_parsed_query_id():
    records = {"videorag_synth_vidA_1": {"golds": ["x"], "cluster_id": "vidA"},
               "videorag_synth_vid_B_2": {"golds": ["x"]}, "videorag_synth_vidC_3": {"golds": ["x"]}}
    preds = {"a": {q: "x" for q in records}}
    _, _, extra = um.score("videorag", preds, list(records), records)
    assert extra["clusters"] == ["vidA", "vid_B", "vidC"]
    with pytest.raises(ValueError):
        um.score("videorag", {"a": {"q": "x"}}, ["q"], {"q": {"golds": ["x"]}})


def test_video_id_comes_from_the_query_index_and_other_formats_raise():
    assert videorag_video_id("videorag_synth_abc_def_3") == "abc_def"
    assert videorag_video_id("videorag_synth_v1_12") == "v1"
    with pytest.raises(ValueError):
        videorag_video_id("video_12")


def test_webqa_official_source_with_a_changed_hash_is_rejected(tmp_path):
    source = paths.THIRD_PARTY / "webqa"
    for name in ("eval_1022.py", "eval_webqa.md"):
        shutil.copy(source / name, tmp_path / name)
    (tmp_path / "eval_1022.py").write_text((tmp_path / "eval_1022.py").read_text() + "\n# edited\n")
    with pytest.raises(RuntimeError, match="sha256"):
        from omniaware.eval.webqa_acc import load_official_metric
        load_official_metric(source_dir=tmp_path)

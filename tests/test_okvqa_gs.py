import json

import numpy as np
import pytest

from omniaware import data, okvqa_dense, relevance


def test_gs_csv_becomes_corpus_jsonl(tmp_path, monkeypatch):
    csv_path = tmp_path / "okvqa_train_corpus.csv"
    csv_path.write_text('id,text\n0,"Bananas are   yellow."\n1,""\n2,"Snow is cold, white."\n', encoding="utf8")
    monkeypatch.setitem(okvqa_dense.CORPORA, "gs112k", (tmp_path / "gs" / "corpus.jsonl", tmp_path / "gs"))
    okvqa_dense.use_corpus("gs112k")
    try:
        out = okvqa_dense.prepare_gs_csv(csv_path)
        rows = [json.loads(line) for line in out.read_text(encoding="utf8").splitlines()]
        assert rows == [{"id": 0, "contents": "Bananas are yellow."}, {"id": 1, "contents": "Snow is cold, white."}]
        assert okvqa_dense.OUT.name == "gs"
        assert len(okvqa_dense._offsets(None)) == 2              # the passage count comes from the corpus itself
    finally:
        okvqa_dense.use_corpus("wiki3m")


def test_csv_without_text_column_is_rejected_and_writes_nothing(tmp_path, monkeypatch):
    csv_path = tmp_path / "shifted.csv"
    csv_path.write_text("id,title,body\n0,Bananas,Bananas are yellow.\n", encoding="utf8")
    corpus = tmp_path / "gs" / "corpus.jsonl"
    monkeypatch.setitem(okvqa_dense.CORPORA, "gs112k", (corpus, tmp_path / "gs"))
    okvqa_dense.use_corpus("gs112k")
    try:
        with pytest.raises(ValueError, match="no 'text' column"):
            okvqa_dense.prepare_gs_csv(csv_path)                 # the last column is never taken as the sentence
        assert not corpus.exists()
    finally:
        okvqa_dense.use_corpus("wiki3m")


def test_corpora_are_named_by_their_source():
    """gs112k is built from the training questions only; the test-inclusive 168k corpus is a separate entry."""
    assert set(okvqa_dense.CORPORA) == {"wiki3m", "gs112k", "gs168k_full"}
    assert okvqa_dense.CORPORA["gs112k"][0].parent.name == "okvqa_gs112k"
    assert okvqa_dense.CORPORA["gs168k_full"][0].parent.name == "okvqa_gs168k_full"
    assert okvqa_dense.GS_CORPORA == ("gs112k", "gs168k_full")


class RecordingBGE:
    """Stands in for relevance.BGE; records every call so the prefixes can be checked."""
    calls = []

    def __init__(self, device=None):
        self.device = "cpu"

    def encode(self, texts, query=False, batch=32):
        RecordingBGE.calls.append((list(texts), query))
        vecs = np.ones((len(texts), okvqa_dense.DIM), dtype=np.float32)
        return vecs / np.sqrt(okvqa_dense.DIM)


def test_passages_have_no_prefix_and_questions_use_the_bge_instruction(tmp_path, monkeypatch):
    csv_path = tmp_path / "okvqa_train_corpus.csv"
    csv_path.write_text("id,text\n0,Bananas are yellow.\n1,Snow is cold.\n", encoding="utf8")
    monkeypatch.setitem(okvqa_dense.CORPORA, "gs112k", (tmp_path / "gs" / "corpus.jsonl", tmp_path / "gs"))
    monkeypatch.setattr(relevance, "BGE", RecordingBGE)
    okvqa_dense.use_corpus("gs112k")
    RecordingBGE.calls = []
    try:
        okvqa_dense.prepare_gs_csv(csv_path)
        okvqa_dense.encode(limit=None, batch=8)
        assert RecordingBGE.calls == [(["Bananas are yellow.", "Snow is cold."], False)]   # no 'passage:' prefix

        examples = [{"qid": "q1", "question": "What color are bananas?", "candidates": []}]
        monkeypatch.setattr(data, "load_okvqa", lambda candidates_path=None: examples)
        RecordingBGE.calls = []
        out = okvqa_dense.search(None, k=2, limit_queries=None, chunk=1, tag="test")
        assert RecordingBGE.calls[0] == (["What color are bananas?"], True)  # the same call as relevance.raw_scores
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 1 and len(rows[0]["candidates"]) == 2            # k=2 over the 2 corpus lines
    finally:
        okvqa_dense.use_corpus("wiki3m")

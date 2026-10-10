import json

from omniaware import okvqa_dense


def test_gs112k_csv_becomes_corpus_jsonl(tmp_path, monkeypatch):
    csv_path = tmp_path / "okvqa_full_corpus.csv"
    csv_path.write_text('id,text\n0,"Bananas are   yellow."\n1,""\n2,"Snow is cold, white."\n', encoding="utf8")
    monkeypatch.setitem(okvqa_dense.CORPORA, "gs112k", (tmp_path / "gs" / "corpus.jsonl", tmp_path / "gs"))
    okvqa_dense.use_corpus("gs112k")
    try:
        out = okvqa_dense.prepare_gs112k(csv_path)
        rows = [json.loads(line) for line in out.read_text(encoding="utf8").splitlines()]
        assert rows == [{"id": 0, "contents": "Bananas are yellow."}, {"id": 1, "contents": "Snow is cold, white."}]
        assert okvqa_dense.OUT.name == "gs"
    finally:
        okvqa_dense.use_corpus("wiki3m")

"""WebQA open-retrieval loader on a synthetic pool laid out like the FAISS pipeline's artifacts."""
import json

import pytest

from omniaware import data, paths


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_open_retrieval_candidates_and_gold_aliases(tmp_path, monkeypatch):
    root = tmp_path / "open"
    _jsonl(root / "text_documents.jsonl", [
        {"row_index": 0, "document_id": "text-sha256:a", "text": "Paris is in France.", "source_snippet_ids": ["s1", "s9"]},
        {"row_index": 1, "document_id": "text-sha256:b", "text": "Unrelated.", "source_snippet_ids": ["s2"]}])
    _jsonl(root / "image_corpus.jsonl", [
        {"fact_id": "image:300", "image_id": "300", "captions": ["  Eiffel   tower ", "Eiffel tower"]},
        {"fact_id": "image:301", "image_id": "301", "captions": ["A cat"]}])
    wave = tmp_path / "legacy" / "experiments" / "eab_first_token" / "runs" / "x" / "clip_wave0"
    image = tmp_path / "img300.jpg"
    image.write_bytes(b"jpg")
    _jsonl(wave / "assets.jsonl", [{"image_id": "300", "path": str(image)},
                                   {"image_id": "301", "path": "/elsewhere/experiments/eab_first_token/img301.jpg"}])
    (root / "faiss_clip_full").mkdir(parents=True)
    (root / "faiss_clip_full" / "plan.json").write_text(json.dumps({"inputs": [{"directory": str(wave)}]}))
    rankings = tmp_path / "rankings"
    _jsonl(rankings / "bge" / "rankings.jsonl", [{"qid": "g1", "candidates": [
        {"rank": 1, "source_id": "text-sha256:b"}, {"rank": 2, "source_id": "text-sha256:a"}]}])
    _jsonl(rankings / "clip" / "rankings.jsonl", [{"qid": "g1", "candidates": [
        {"rank": 1, "source_id": "image:301"}, {"rank": 2, "source_id": "image:300"}]}])
    webqa = {"g1": {"split": "val", "Q": "Where is it?", "Qcate": "text", "A": ["\"In Paris.\""],
                    "txt_posFacts": [{"snippet_id": "s9"}], "txt_negFacts": [],
                    "img_posFacts": [{"image_id": 300}], "img_negFacts": []},
             "g2": {"split": "val", "Q": "Unranked?", "Qcate": "text", "A": ["x"],
                    "txt_posFacts": [], "txt_negFacts": [], "img_posFacts": [{"image_id": 1}], "img_negFacts": []}}
    (tmp_path / "webqa.json").write_text(json.dumps(webqa))
    monkeypatch.setattr(paths, "WEBQA_OPEN", root)
    monkeypatch.setattr(paths, "WEBQA_JSON", tmp_path / "webqa.json")
    monkeypatch.setattr(paths, "ASSET_ROOT", tmp_path / "assets")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "img301.jpg").write_bytes(b"jpg")

    # g2 (image question) has no rankings: an error by default, dropped and recorded with allow_missing.
    with pytest.raises(data.MissingDataError, match="no ranking"):
        data.load_webqa_open("img", rankings=rankings)
    loaded = data.load_webqa_open("img", rankings=rankings, allow_missing=True)
    assert loaded.meta["dropped"] == {"no ranking": {"items": 1, "qids": ["g2"]}}
    (e,) = loaded
    by_cid = {c["cid"]: c for c in e["candidates"]}
    assert [c["cid"] for c in e["candidates"]] == ["t1", "t2", "i1", "i2"]
    assert by_cid["t2"]["gold"] and not by_cid["t1"]["gold"]          # alias s9 of the deduplicated text
    assert by_cid["i2"]["gold"] and by_cid["i2"]["image"] == str(image)
    assert by_cid["i2"]["text"] == "Eiffel tower"
    # A path recorded on another host is remapped under the local asset root.
    assert by_cid["i1"]["image"] == str(tmp_path / "assets" / "img301.jpg")
    assert e["golds"] == ["In Paris."] and e["missing_images"] == 0
    assert e["gold_modalities"] == ["image", "text"]                  # M-37: both kinds of positive fact

    # A ranked image whose file is absent is an error by default, and dropped and counted with allow_missing.
    (tmp_path / "assets" / "img301.jpg").unlink()
    with pytest.raises(data.MissingDataError, match="ranked image missing"):
        data.load_webqa_open("img", rankings=rankings, allow_missing=False)
    loaded = data.load_webqa_open("img", rankings=rankings, allow_missing=True)
    (e,) = loaded
    assert [c["cid"] for c in e["candidates"]] == ["t1", "t2", "i2"] and e["missing_images"] == 1
    assert loaded.meta["dropped"]["ranked image missing"] == {"items": 1, "qids": ["g1"]}

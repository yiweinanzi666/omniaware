"""Loader policy (fail-fast, allow_missing, drop records) and per-question loader fixes on synthetic assets."""
import json
import zipfile

import pytest

from omniaware import data, paths


def _webqa(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    (images / "1.jpg").write_bytes(b"jpg")          # image 9 and 404 are missing
    webqa = {
        "g1": {"split": "val", "Q": "Q1?", "Qcate": "text", "A": ["a"],
               "txt_posFacts": [{"title": "T", "fact": "F one"}], "txt_negFacts": [],
               "img_posFacts": [{"image_id": 1, "title": "", "caption": "cap"}], "img_negFacts": []},
        "g2": {"split": "val", "Q": "Q2?", "Qcate": "text", "A": ["b"],
               "txt_posFacts": [{"title": "", "fact": "F two"}], "txt_negFacts": [],
               "img_posFacts": [], "img_negFacts": [{"image_id": 9, "title": "", "caption": ""}]},
        "g3": {"split": "val", "Q": "Q3?", "Qcate": "text", "A": ["c"],
               "txt_posFacts": [], "txt_negFacts": [{"title": "", "fact": "N"}],
               "img_posFacts": [{"image_id": 404, "title": "", "caption": ""}], "img_negFacts": []},
    }
    (tmp_path / "webqa.json").write_text(json.dumps(webqa))
    monkeypatch.setattr(paths, "WEBQA_JSON", tmp_path / "webqa.json")
    monkeypatch.setattr(paths, "WEBQA_IMAGES", images)


def test_webqa_missing_facts_fail_fast_by_default(tmp_path, monkeypatch):
    _webqa(tmp_path, monkeypatch)
    with pytest.raises(data.MissingDataError) as err:
        data.load_webqa("all", "val")
    assert "fact image missing" in str(err.value) and "g2" in str(err.value) and "g3" in str(err.value)


def test_webqa_allow_missing_drops_facts_and_questions_and_records_them(tmp_path, monkeypatch):
    _webqa(tmp_path, monkeypatch)
    e = data.load_webqa("all", "val", allow_missing=True)
    assert [x["qid"] for x in e] == ["g1", "g2"]                     # g3 has no gold left
    assert e.meta["dropped"] == {"fact image missing": {"items": 2, "qids": ["g2", "g3"]},
                                 "no gold fact left": {"items": 1, "qids": ["g3"]}}
    g2 = {c["cid"] for c in e[1]["candidates"]}
    assert g2 == {"tp0"} and e[1]["missing_images"] == 1


def test_webqa_gold_modalities_record_every_positive_kind(tmp_path, monkeypatch):
    # M-37: a question with both a text and an image positive fact records both modalities.
    _webqa(tmp_path, monkeypatch)
    e = data.load_webqa("all", "val", allow_missing=True)
    by = {x["qid"]: x for x in e}
    assert by["g1"]["gold_modalities"] == ["image", "text"]
    assert by["g2"]["gold_modalities"] == ["text"]


def test_webqa_image_subset_keeps_only_image_questions(tmp_path, monkeypatch):
    _webqa(tmp_path, monkeypatch)
    e = data.load_webqa("img", "val", allow_missing=True)
    assert [x["qid"] for x in e] == ["g1"]
    assert e.meta["dropped"]["no gold fact left"]["qids"] == ["g3"]


def _okvqa(tmp_path, monkeypatch):
    zip_path = tmp_path / "val.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("mscoco.json", json.dumps({"annotations": [
            {"question_id": 1, "answers": [{"answer": "yellow"}]},
            {"question_id": 2, "answers": [{"answer": "cold"}]},
            {"question_id": 3, "answers": [{"answer": "x"}]}]}))
    img = tmp_path / "q1.jpg"
    img.write_bytes(b"jpg")
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps(r) + "\n" for r in [
        {"question_id": 1, "question": "What colour?", "image_path": str(img)},
        {"question_id": 2, "question": "Is it cold?", "image_path": str(img)},
        {"question_id": 3, "question": "Missing image?", "image_path": str(tmp_path / "gone.jpg")}]))
    cands = tmp_path / "cands.jsonl"
    cands.write_text("".join(json.dumps(r) + "\n" for r in [
        {"question_id": 1, "candidates": [{"rank": 1, "text": "Bananas are yellow."},
                                          {"rank": 2, "text": "Snow\nSnow is cold."}]},
        {"question_id": 2, "candidates": []},
        {"question_id": 3, "candidates": [{"rank": 1, "text": "x"}]}]))
    monkeypatch.setattr(paths, "OKVQA_ANNOTATIONS", zip_path)
    monkeypatch.setattr(paths, "OKVQA_INPUTS", inputs)
    monkeypatch.setattr(paths, "OKVQA_CANDIDATES", cands)


def test_okvqa_single_line_passage_is_text_without_title(tmp_path, monkeypatch):
    # M-18: a passage without a line break used to become title = text, encoded as "X: X".
    _okvqa(tmp_path, monkeypatch)
    e = data.load_okvqa(allow_missing=True)
    assert [x["qid"] for x in e] == ["1"] and e[0]["golds"] == ["yellow"]
    k1, k2 = e[0]["candidates"]
    assert (k1["title"], k1["text"]) == ("", "Bananas are yellow.")
    assert (k2["title"], k2["text"]) == ("Snow", "Snow is cold.")


def test_okvqa_missing_candidates_or_images_fail_fast_and_are_recorded(tmp_path, monkeypatch):
    _okvqa(tmp_path, monkeypatch)
    with pytest.raises(data.MissingDataError, match="no candidates"):
        data.load_okvqa()
    e = data.load_okvqa(allow_missing=True)
    assert e.meta["dropped"] == {"no candidates": {"items": 1, "qids": ["2"]},
                                 "query image missing": {"items": 1, "qids": ["3"]}}
    # The split universe (no candidates loaded) does not need the images.
    assert len(data.load_okvqa(candidates_path=False)) == 3

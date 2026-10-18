"""Download bookkeeping: completeness of model folders, pinned revisions, the NLTK Punkt extraction, --help."""
import json
import struct
import zipfile

import pytest

from omniaware import paths
from scripts.setup import check_assets, download_models as dm


def _write_safetensors(path, nbytes):
    """A valid safetensors file with one U8 tensor of ``nbytes`` bytes (header + data, nothing else)."""
    header = {"w": {"dtype": "U8", "shape": [nbytes], "data_offsets": [0, nbytes]}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * nbytes)
    return path


def test_model_complete_needs_every_shard_at_its_full_size(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    assert not dm.model_complete(folder)                        # no weights yet
    shard1 = _write_safetensors(folder / "model-00001-of-00002.safetensors", 100)
    shard2 = _write_safetensors(folder / "model-00002-of-00002.safetensors", 50)
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": shard1.name, "b": shard2.name}}))
    assert dm.model_complete(folder)
    with shard2.open("r+b") as fh:                              # an interrupted download: last bytes missing
        fh.truncate(shard2.stat().st_size - 1)
    assert not dm.model_complete(folder)
    shard2.unlink()                                             # a shard that was never fetched
    assert not dm.model_complete(folder)
    single = tmp_path / "single"
    single.mkdir()
    (single / "config.json").write_text("{}")
    _write_safetensors(single / "model.safetensors", 10)
    assert dm.model_complete(single)
    (tmp_path / "config_only").mkdir()
    (tmp_path / "config_only" / "config.json").write_text("{}")
    assert dm.model_complete(tmp_path / "config_only", needs_weights=False)


def test_every_hub_repository_is_pinned_to_a_commit():
    assert set(dm.REVISIONS) == {"qwen8b", "qwen2b", "internvl", "bge", "clip", "reranker", "nli", "bart_cnn"}
    for name, (repo, rev, _, _, _) in dm.HF_MODELS.items():
        assert rev and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), (name, rev)


def test_nli_downloads_only_the_manifest_files():
    allow = dm.HF_MODELS["nli"][3]
    manifest = json.loads(dm.NLI_CHECKER_MANIFEST.read_text(encoding="utf8"))
    assert allow == [rec["file"] for rec in manifest["files"]]
    assert not any(p.endswith(".onnx") or p.endswith(".bin") for p in allow)


def test_git_blob_sha1_matches_git(tmp_path):
    blob = tmp_path / "hello.txt"
    blob.write_bytes(b"hello\n")
    assert dm._git_blob_sha1(blob) == "ce013625030ba8dba906f756967f9e9ca394464a"   # git hash-object of "hello\n"


def _punkt_zip(path, payload):
    with zipfile.ZipFile(path, "w") as zf:
        for name in dm.PUNKT_ENGLISH:
            zf.writestr(f"punkt_tab/english/{name}", payload + name)
        zf.writestr("punkt_tab/german/abbrev_types.txt", "not english")
    return path


def test_punkt_english_is_extracted_and_its_hashes_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ASSET_ROOT", tmp_path)
    root = tmp_path / "cache" / "e5_citation_support"
    manifest = dm.extract_punkt_english(_punkt_zip(tmp_path / "a.zip", "x"), root)
    record = json.loads(manifest.read_text())
    assert [f["path"].rsplit("/", 1)[1] for f in record["files"]] == dm.PUNKT_ENGLISH
    assert all(f["path"].startswith("cache/e5_citation_support/nltk_data/tokenizers/punkt_tab/english/")
               for f in record["files"])
    assert not (root / "nltk_data" / "tokenizers" / "punkt_tab" / "german").exists()
    dm.extract_punkt_english(_punkt_zip(tmp_path / "b.zip", "x"), root)      # same content: accepted again
    with pytest.raises(SystemExit, match="frozen"):                           # changed content: refused
        dm.extract_punkt_english(_punkt_zip(tmp_path / "c.zip", "y"), root)


def test_punkt_zip_without_the_english_tables_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ASSET_ROOT", tmp_path)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("punkt_tab/german/abbrev_types.txt", "x")
    with pytest.raises(SystemExit, match="missing"):
        dm.extract_punkt_english(bad, tmp_path / "cache" / "e5_citation_support")


def test_check_assets_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as info:
        check_assets.main(["--help"])
    assert info.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_webqa_distractor_images_need_every_train_and_val_image_fact(tmp_path, monkeypatch):
    webqa = {"q1": {"split": "val", "img_posFacts": [{"image_id": 1}], "img_negFacts": [{"image_id": 2}]},
             "q2": {"split": "train", "img_posFacts": [], "img_negFacts": [{"image_id": 3}]},
             "q3": {"split": "test", "img_posFacts": [{"image_id": 9}], "img_negFacts": []}}
    monkeypatch.setattr(paths, "WEBQA_JSON", tmp_path / "WebQA_train_val.json")
    images = tmp_path / "images"
    assert not check_assets._webqa_distractor_images(images)          # no annotation file, no directory
    paths.WEBQA_JSON.write_text(json.dumps(webqa))
    images.mkdir()
    for iid in (1, 2):
        (images / f"{iid}.jpg").write_bytes(b"")
    assert not check_assets._webqa_distractor_images(images)          # the train image 3 is missing
    (images / "3.jpg").write_bytes(b"")
    assert check_assets._webqa_distractor_images(images)              # image 9 belongs to a test question

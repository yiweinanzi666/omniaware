"""prepare_webqa_open builds exactly what data.load_webqa_open reads (stand-in encoders, CPU)."""
import base64
import io
import json
import random

import numpy as np
import pytest
from PIL import Image

from omniaware import data, paths, relevance
from scripts.setup import prepare_webqa_open as pw


class FakeBGE:
    def encode(self, texts, query=False, batch=32):
        out = np.array([[1.0 if "Paris" in t else 0.0, 1.0 if "cat" in t else 0.0, 0.1] for t in texts])
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class FakeCLIP:
    def encode_text(self, texts):
        return FakeBGE().encode(texts)

    def encode_images(self, image_paths, batch=16):
        return FakeBGE().encode(["Paris" if "300" in p else "cat" for p in image_paths])


WEBQA = {"g1": {"split": "val", "Q": "\"Where is the Eiffel tower, Paris?\"", "Qcate": "text", "A": ["\"Paris.\""],
                "txt_posFacts": [{"snippet_id": "s1", "title": "Paris", "fact": "The tower is in Paris."}],
                "txt_negFacts": [{"snippet_id": "s2", "title": "Cats", "fact": "A cat sleeps."}],
                "img_posFacts": [{"image_id": 300, "caption": "Eiffel tower Paris"}],
                "img_negFacts": [{"image_id": 301, "caption": "A cat"}]},
         "g2": {"split": "train", "Q": "What sleeps? cat", "Qcate": "text", "A": ["A cat."],
                "txt_posFacts": [{"snippet_id": "s2", "title": "Cats", "fact": "A cat sleeps."}],
                "txt_negFacts": [], "img_posFacts": [], "img_negFacts": []}}


def _jpeg(seed, size=32):
    rnd = random.Random(seed)
    img = Image.new("RGB", (size, size))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(size * size)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _tsv_line(iid, raw: bytes) -> str:
    return f"{iid}\t{base64.b64encode(raw).decode()}"


@pytest.fixture
def webqa_env(tmp_path, monkeypatch):
    """A WebQA source directory and an empty open-retrieval output directory, wired through monkeypatching."""
    src = tmp_path / "webqa"
    src.mkdir()
    (src / "WebQA_train_val.json").write_text(json.dumps(WEBQA))
    out = tmp_path / "open"
    monkeypatch.setattr(pw, "OUT", out)
    monkeypatch.setattr(paths, "WEBQA_OPEN", out)
    monkeypatch.setattr(paths, "WEBQA_JSON", src / "WebQA_train_val.json")
    monkeypatch.setattr(paths, "ASSET_ROOT", tmp_path)
    monkeypatch.setattr(relevance, "BGE", FakeBGE)
    monkeypatch.setattr(relevance, "CLIP", FakeCLIP)
    return src, out


def test_build_pool_images_encode_search_then_load(webqa_env):
    src, out = webqa_env
    lines = [_tsv_line(300, _jpeg(300)), _tsv_line(301, _jpeg(301))]
    (src / "imgs.tsv").write_text("\n".join(lines) + "\n")

    pw.build_pool(src)
    docs = [json.loads(l) for l in (out / "text_documents.jsonl").read_text().splitlines()]
    assert len(docs) == 2 and sorted(d["source_snippet_ids"] for d in docs) == [["s1"], ["s2"]]
    stats = json.loads((out / "pool_stats.json").read_text())
    assert stats["text_documents"] == 2 and stats["image_corpus"] == 2
    assert stats["sources"]["WebQA_train_val.json"] == pw._sha256(src / "WebQA_train_val.json")

    pw.extract_images(src, workers=1)
    assert (out / "images" / "300.jpg").exists()
    assert not (out / "images" / "missing_images.jsonl").read_text().strip()
    assert (out / "images" / "bad_images.jsonl").read_text() == ""
    for kind in ("text", "image"):
        for shard in range(2):
            pw.encode_shard(kind, shard, 2, batch=4)
        pw.write_manifest(kind, 2)
    pw.search("val", top=2)
    pw.search("train", top=2)

    (e,) = data.load_webqa_open("all", split="val")
    by = {c["cid"]: c for c in e["candidates"]}
    assert by["t1"]["gold"] and by["t1"]["text"].startswith("Paris:")      # Paris passage ranked first
    assert by["i1"]["gold"] and by["i1"]["image"].endswith("300.jpg")
    assert e["golds"] == ["Paris."] and e["missing_images"] == 0
    (t,) = data.load_webqa_open("all", split="train")
    assert t["qid"] == "g2"


def test_truncated_jpeg_is_recorded_and_rerun_completes(webqa_env, capsys):
    """A truncated JPEG (a known defect of the official imgs.tsv) must not abort the extraction, must be listed
    as bad, must not be indexed, and rerunning the step must finish the same way."""
    src, out = webqa_env
    pw.build_pool(src)                                    # pool holds image 300 and 301
    good, broken = _jpeg(300), _jpeg(301)
    truncated = broken[: len(broken) // 2]
    (src / "imgs.tsv").write_text("\n".join([_tsv_line(300, good),
                                             f"301\t{base64.b64encode(truncated).decode()}"]) + "\n")

    pw.extract_images(src, workers=1)                     # must not raise
    bad = [json.loads(l) for l in (out / "images" / "bad_images.jsonl").read_text().splitlines()]
    assert [b["image_id"] for b in bad] == ["301"] and "truncated" in bad[0]["error"].lower()
    assert (out / "images" / "300.jpg").exists() and not (out / "images" / "301.jpg").exists()
    assert json.loads((out / "images" / "assets.jsonl").read_text().splitlines()[0])["image_id"] == "300"
    assert not list((out / "images").glob("*.tmp"))       # no temporary files left behind
    output = capsys.readouterr().out
    assert "undecodable" in output and "the loader counts them" not in output

    pw.extract_images(src, workers=1)                     # rerun completes with the same outcome
    rerun_bad = [json.loads(l) for l in (out / "images" / "bad_images.jsonl").read_text().splitlines()]
    assert rerun_bad == bad


def test_missing_pool_images_are_listed_not_silently_dropped(webqa_env, capsys):
    src, out = webqa_env
    pw.build_pool(src)
    (src / "imgs.tsv").write_text(_tsv_line(300, _jpeg(300)) + "\n")   # 301 absent from imgs.tsv
    pw.extract_images(src, workers=1)
    missing = [json.loads(l)["image_id"] for l in (out / "images" / "missing_images.jsonl").read_text().splitlines()]
    assert missing == ["301"]
    assert "missing from imgs.tsv: 1" in capsys.readouterr().out


def test_interrupted_shard_is_redone_and_done_marker_gates_skipping(webqa_env):
    src, out = webqa_env
    lines = [_tsv_line(300, _jpeg(300)), _tsv_line(301, _jpeg(301))]
    (src / "imgs.tsv").write_text("\n".join(lines) + "\n")
    pw.build_pool(src)
    pw.extract_images(src, workers=1)
    emb = out / "emb"
    emb.mkdir(parents=True, exist_ok=True)
    (emb / "text.shard00-of-01.npy").write_bytes(b"\x93NUMPY truncated")      # left by an interrupted write
    pw.encode_shard("text", 0, 1, batch=4)                                   # no .done -> redone
    assert (emb / "text.shard00-of-01.done").exists()
    assert np.load(emb / "text.shard00-of-01.npy").shape[0] == 2
    assert not list(emb.glob("*.tmp"))
    before = (emb / "text.shard00-of-01.npy").stat().st_mtime_ns
    pw.encode_shard("text", 0, 1, batch=4)                                   # .done present -> skipped
    assert (emb / "text.shard00-of-01.npy").stat().st_mtime_ns == before


def test_load_rejects_missing_shard_and_row_mismatch(webqa_env):
    src, out = webqa_env
    (src / "imgs.tsv").write_text(_tsv_line(300, _jpeg(300)) + "\n" + _tsv_line(301, _jpeg(301)) + "\n")
    pw.build_pool(src)
    pw.extract_images(src, workers=1)
    for kind in ("text", "image"):
        for shard in range(2):
            pw.encode_shard(kind, shard, 2, batch=4)
        pw.write_manifest(kind, 2)
    (out / "emb" / "image.shard01-of-02.npy").unlink()        # a shard disappears (with its marker)
    (out / "emb" / "image.shard01-of-02.done").unlink()
    with pytest.raises(SystemExit, match="do not cover"):
        pw._load("image")
    pw.encode_shard("image", 1, 2, batch=4)                    # re-encode the missing shard
    pw.write_manifest("image", 2)
    ids_path = out / "emb" / "image.shard00-of-02.ids.json"    # ids no longer match the vectors
    ids_path.write_text(json.dumps(["image:999"] * 5))
    with pytest.raises(SystemExit):
        pw._load("image")


def test_search_refuses_a_different_webqa_json_and_oversized_top(webqa_env):
    src, out = webqa_env
    (src / "imgs.tsv").write_text(_tsv_line(300, _jpeg(300)) + "\n" + _tsv_line(301, _jpeg(301)) + "\n")
    pw.build_pool(src)
    pw.extract_images(src, workers=1)
    for kind in ("text", "image"):
        pw.encode_shard(kind, 0, 1, batch=4)
        pw.write_manifest(kind, 1)
    other = src / "other"
    other.mkdir()
    (other / "WebQA_train_val.json").write_text(json.dumps({**WEBQA, "g3": WEBQA["g1"]}))
    with pytest.raises(SystemExit, match="differs from the WebQA JSON"):
        pw.search("val", top=2, webqa_json=other / "WebQA_train_val.json")
    with pytest.raises(SystemExit, match="exceeds the text index"):
        pw.search("val", top=50)
    pw.search("val", top=2, chunk=1)                           # the default JSON, chunked scoring


def test_multi_gpu_failure_kills_the_other_shards(monkeypatch):
    """One failing shard must stop the rest (no orphan encoders) and fail the command; success waits for all."""
    class FakeProc:
        def __init__(self, code=None):
            self.code, self.returncode, self.killed = code, None, False
            if code is not None:
                self.returncode = code

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self):
            return self.returncode

    slow = FakeProc()                                            # shard 1 would run until it is killed
    queue = [FakeProc(code=1), slow]                             # shard 0 fails at once
    monkeypatch.setattr(pw.subprocess, "Popen", lambda *a, **k: queue.pop(0))
    monkeypatch.setattr(pw.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match=r"shard\(s\) \[0\] failed"):
        pw._run_shards("text", ["0", "1"], batch=4)
    assert slow.killed                                           # the other shard was stopped

    queue = [FakeProc(code=0), FakeProc(code=0)]                 # both finish: no error, all waited for
    monkeypatch.setattr(pw.subprocess, "Popen", lambda *a, **k: queue.pop(0))
    pw._run_shards("text", ["0", "1"], batch=4)

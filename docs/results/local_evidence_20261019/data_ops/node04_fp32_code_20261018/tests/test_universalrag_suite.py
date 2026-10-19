import json

import numpy as np
import pandas as pd
import pytest

from omniaware import data, pipeline, relevance
from omniaware.eval import universalrag_metrics as um


def _tree(tmp_path):
    root, rk, fr = tmp_path / "ur", tmp_path / "rankings", tmp_path / "frames"
    (root / "dataset" / "query").mkdir(parents=True)
    (root / "dataset" / "query" / "hybridqa.json").write_text(json.dumps([
        {"index": "h1", "question": "Founded when ?", "answer": "1960", "aliases": [], "source": "HybridQA",
         "gt_tables": "T_0", "gt_retrieval": "table+paragraph"}]))
    (root / "dataset" / "query" / "videorag_synth.json").write_text(json.dumps([
        {"index": "videorag_synth_abc_1", "question": "Next step?", "answer": "Detach the caliper.", "source": "videorag_synth",
         "gt_videos": ["dataset/videorag/videos/abc.mp4"], "gt_retrieval": "video"}]))
    pd.DataFrame({"text": ["Table: T\nclub | founded\nA | 1960", "Table: other"]},
                 index=pd.Index(["hybridqa-T_0", "hybridqa-U_0"], name="psg_id")).to_parquet(root / "dataset" / "table.parquet")
    pd.DataFrame({"text": ["A was founded in 1960.", "Unrelated."]},
                 index=pd.Index(["hybridqa-T_0_0", "hybridqa-U_0_3"], name="psg_id")).to_parquet(root / "dataset" / "paragraph.parquet")
    scripts = root / "dataset" / "videorag" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "abc.txt").write_text("First remove the wheel, then detach the caliper.")
    for target, lists in {"hybridqa": {"table": ["hybridqa-U_0", "hybridqa-T_0"],
                                       "paragraph": ["hybridqa-T_0_0", "hybridqa-U_0_3"]},
                          "videorag_synth": {"video": ["dataset/videorag/videos/abc.mp4",
                                                       "dataset/videorag/videos/zzz.mp4"]}}.items():
        (rk / target).mkdir(parents=True)
        for m, ids in lists.items():
            (rk / target / f"{m}.jsonl").write_text(json.dumps(
                {"qid": "h1" if target == "hybridqa" else "videorag_synth_abc_1",
                 "candidates": [{"source_id": i, "score": 1.0 - 0.1 * rank} for rank, i in enumerate(ids)]}) + "\n")
    key = fr / data.frame_key("dataset/videorag/videos/abc.mp4")
    key.mkdir(parents=True)
    (key / "frames.json").write_text(json.dumps({"frames": [{"path": "00.png"}, {"path": "01.png"}]}))
    return root, rk, fr


def test_hybridqa_loader_marks_gt_table_and_passages(tmp_path):
    root, rk, fr = _tree(tmp_path)
    [e] = data.load_universalrag("hybridqa", root=root, rankings=rk, frames_dir=fr)
    assert e["dataset"] == "hybridqa" and e["gold_modalities"] == ["table", "text"] and e["golds"] == ["1960"]
    by = {c["cid"]: c for c in e["candidates"]}
    assert by["t1"]["gold"] is False and by["t2"]["gold"] is True and by["p1"]["gold"] is True
    assert by["t2"]["modality"] == "table" and by["p1"]["modality"] == "text"
    assert by["t1"]["suite_score"] == pytest.approx(1.0) and by["t2"]["suite_score"] == pytest.approx(0.9)   # rankings' scores


def test_videorag_loader_builds_video_chunks_with_frames_and_transcript(tmp_path):
    root, rk, fr = _tree(tmp_path)
    with pytest.raises(data.MissingDataError, match="frames missing"):   # zzz.mp4 has no frames
        data.load_universalrag("videorag_synth", ("video",), root=root, rankings=rk, frames_dir=fr)
    loaded = data.load_universalrag("videorag_synth", ("video",), root=root, rankings=rk, frames_dir=fr,
                                    allow_missing=True)
    assert loaded.meta["dropped"] == {"frames missing": {"items": 1, "qids": ["videorag_synth_abc_1"]}}
    [e] = loaded
    assert e["cluster_id"] == "abc"            # the source video, from the query index
    [v] = e["candidates"]
    assert v["modality"] == "video" and v["gold"] is True and len(v["frames"]) == 2 and "caliper" in v["text"]
    segs, split = pipeline.build_segments(e, e["candidates"], "cite")
    video = [s for s in segs if s.chunk == "v1"]
    assert sum(s.image is not None for s in video) == 2 and video[0].text.startswith("[E1] (video)")
    assert "Transcript:" in video[-1].text and "[E<number>]" in segs[-1].text
    assert "complete sentence" in pipeline.build_segments(e, e["candidates"])[0][-1].text
    assert pipeline.citation_sources(segs, e["candidates"])[0]["source_id"] == "v1"


def test_clip_is_a_separate_modality_that_keeps_frames_and_video_scores(tmp_path):
    # M-44 / D6: clips are their own modality, but are still video chunks (frames, transcript, video channels).
    root, rk, fr = _tree(tmp_path)
    clip_sid = "dataset/videorag/videos/abc.mp4_0"
    (rk / "videorag_synth" / "clip.jsonl").write_text(json.dumps(
        {"qid": "videorag_synth_abc_1", "candidates": [{"source_id": clip_sid, "score": 0.9}]}) + "\n")
    key = fr / data.frame_key(clip_sid)
    key.mkdir(parents=True)
    (key / "frames.json").write_text(json.dumps({"frames": [{"path": "00.png"}]}))
    [e] = data.load_universalrag("videorag_synth", ("video", "clip"), root=root, rankings=rk, frames_dir=fr,
                                 allow_missing=True)
    by = {c["cid"]: c for c in e["candidates"]}
    assert by["v1"]["modality"] == "video" and by["c1"]["modality"] == "clip"
    assert by["c1"]["gold"] is True and len(by["c1"]["frames"]) == 1 and "caliper" in by["c1"]["text"]
    segs, _ = pipeline.build_segments(e, e["candidates"], "cite")
    clip = [s for s in segs if s.chunk == "c1"]
    assert sum(s.image is not None for s in clip) == 1 and "(video)" in clip[0].text

    class FakeBGE:
        def encode(self, texts, query=False):
            return np.ones((len(texts), 3))

    class FakeCLIP:
        def encode_text(self, texts):
            return np.ones((len(texts), 3))

        def encode_images(self, images):
            return np.ones((len(images), 3))

    scores = relevance.raw_scores([e], FakeBGE(), FakeCLIP())[e["qid"]]
    assert set(scores["c1"]) == {"clip_text", "clip_frames"}          # clips are calibrated apart from videos
    assert set(scores["v1"]) == {"video_text", "video_clip"}


def test_video_relevance_channels():
    cal = relevance.Calibrator({"video_text": [0.1, 0.2, 0.3, 0.4], "video_clip": [0.0, 1.0]})
    # Two-channel percentiles need calibrated candidates scored on both channels (see test_relevance_calibration).
    with pytest.raises(ValueError):
        cal.relevance({"video_text": 0.35, "video_clip": 0.5})
    assert relevance.Calibrator.raw({"video_text": 0.2, "video_clip": 0.4}) == pytest.approx(0.3)


def test_universalrag_metrics():
    assert um.exact_match("The 1960.", "1960") == 1 and um.f1("founded 1960", "1960") == pytest.approx(2 / 3)
    assert um.rouge_l("detach the caliper", "Detach the brake caliper.") == pytest.approx(2 * 1 * 0.75 / 1.75)
    recs = {"q": {"golds": ["Jeep Wagoneer", "Wagoneer"]}}
    with pytest.raises(ValueError):
        um.score("infoseek", {"a": {"q": "wagoneer"}}, ["q"], recs)
    s, m, _ = um.score("hybridqa", {"a": {"q": "a jeep wagoneer"}}, ["q"], recs)
    assert s["a"]["q"]["em"] == 100

"""Official Video-MME protocol: frame-aligned subtitles (video-slicer rule) and the official prompt."""
from pathlib import Path

import pytest

from omniaware.data import _parse_srt_intervals, frame_aligned_subtitles
from omniaware.pipeline import build_segments

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "external" / "video-slicer" / "demo"


def test_strict_interval_first_match_and_empty_skipped(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nfirst\nline two\n\n"
                   "2\n00:00:02,000 --> 00:00:04,000\nsecond\n\n3\n00:00:10,000 --> 00:00:12,000\n<i>third</i>\n")
    intervals = _parse_srt_intervals(srt)
    assert intervals == [(0.0, 2.0, "first line two"), (2.0, 4.0, "second"), (10.0, 12.0, "third")]
    # t=2.0 lies on a boundary: start < t < end is false for both, so no subtitle (official rule).
    assert frame_aligned_subtitles(intervals, [1.0, 2.0, 3.0, 6.0, 11.0]) == "first line two\nsecond\nthird"


@pytest.mark.skipif(not (DEMO / "demo.srt").exists(), reason="external/video-slicer not fetched")
def test_matches_official_demo_output():
    expected = (DEMO / "subtitles.txt").read_text(encoding="utf-8")
    # Demo frames are named 00:01, 00:05, 00:09, 00:13 (whole seconds of the actual frame times).
    assert frame_aligned_subtitles(_parse_srt_intervals(DEMO / "demo.srt"), [1.9, 5.9, 9.9, 13.9]) == expected


def test_official_prompt_layout():
    e = {"dataset": "videomme", "question": "What color?", "options": ["A. Red.", "B. Blue.", "C. Green.", "D. Gray."],
         "query_images": []}
    chunks = [{"cid": "f0", "modality": "image", "image": "f0.jpg", "title": "", "text": ""},
              {"cid": "sub", "modality": "text", "title": "Subtitles", "text": "hello there"}]
    segs, split = build_segments(e, chunks, "videomme_official")
    text = "".join(s.text or "" for s in segs)
    assert "This video's subtitles are listed below:\nhello there\n" in text
    assert text.endswith("Respond with only the letter (A, B, C, or D) of the correct option.\nWhat color?\n"
                         "A. Red.\nB. Blue.\nC. Green.\nD. Gray.\nThe best answer is:<|im_end|>\n<|im_start|>assistant\n")
    assert [s.chunk for s in segs[:split] if s.image] == ["f0"]
    no_sub, _ = build_segments(e, None, "videomme_official")
    assert "subtitles" not in "".join(s.text or "" for s in no_sub)

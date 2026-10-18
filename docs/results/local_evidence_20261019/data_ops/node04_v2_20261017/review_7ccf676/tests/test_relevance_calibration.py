"""Relevance r_j: per-channel ECDFs, and the percentile of a combined score over several channels."""
import numpy as np
import pytest

from omniaware import relevance


def _raw(n, channels, rho, seed):
    """Calibration candidates whose two channel scores are standard normal with correlation rho."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    x = np.column_stack([z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho ** 2) * z[:, 1]])
    return {f"c{i}": {channels[0]: float(x[i, 0]), channels[1]: float(x[i, 1])} for i in range(n)}


@pytest.mark.parametrize("channels,image_mode", [(("video_text", "video_clip"), "mix"),
                                                 (("image_clip", "image_cap"), "mix")])
@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_combined_relevance_is_a_percentile_on_the_calibration_set(channels, image_mode, rho):
    raw = {"q": _raw(4000, channels, rho, seed=1)}
    cal = relevance.Calibrator.fit(raw, ["q"], image_mode)
    r = np.array([cal.relevance(ch, image_mode) for ch in raw["q"].values()])
    assert r.min() > 0.0 and r.max() == pytest.approx(1.0)
    # A percentile of the calibration set is uniform on (0, 1]: mean 1/2, sd 1/sqrt(12), 5% and 95% quantiles.
    assert r.mean() == pytest.approx(0.5, abs=0.01)
    assert r.std() == pytest.approx(np.sqrt(1 / 12), abs=0.01)
    assert np.quantile(r, 0.05) == pytest.approx(0.05, abs=0.02)
    assert np.quantile(r, 0.95) == pytest.approx(0.95, abs=0.02)


def test_mean_of_percentiles_alone_would_not_be_uniform():
    """The regression this file guards: the mean of two channel percentiles is compressed toward 1/2."""
    raw = {"q": _raw(4000, ("video_text", "video_clip"), 0.0, seed=2)}
    cal = relevance.Calibrator.fit(raw, ["q"])
    means = np.array([np.mean([cal.ecdf(k, ch[k]) for k in ("video_text", "video_clip")])
                      for ch in raw["q"].values()])
    assert means.std() < 0.25        # about 0.20 for independent channels: the reason for the second ECDF


def test_single_channel_candidates_use_their_own_percentile():
    raw = {"q": _raw(500, ("image_clip", "image_cap"), 0.0, seed=3)}
    cal = relevance.Calibrator.fit(raw, ["q"], "clip")
    channels = {"image_clip": 0.7, "image_cap": -2.0}
    assert cal.relevance(channels, "clip") == cal.ecdf("image_clip", 0.7)


def test_partial_channels_use_the_present_one():
    """A candidate that lacks one of the calibrated channels (e.g. a video without transcript) is ranked on the
    channel it has, which is a percentile of the same calibration distribution."""
    raw = {"q": _raw(300, ("video_text", "video_clip"), 0.3, seed=4)}
    cal = relevance.Calibrator.fit(raw, ["q"])
    assert cal.relevance({"video_clip": 0.2}) == cal.ecdf("video_clip", 0.2)


def test_missing_calibration_data_is_an_error_not_a_constant():
    cal = relevance.Calibrator({"text": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError):
        cal.ecdf("image_clip", 0.2)
    with pytest.raises(ValueError):
        cal.relevance({"video_text": 0.2})             # no calibration pool for this channel
    with pytest.raises(ValueError):
        cal.relevance({})                               # nothing scored for this candidate


def test_two_channels_need_calibration_candidates_scored_on_both():
    raw = {"q": {"c0": {"video_text": 0.1}, "c1": {"video_text": 0.4}}}
    cal = relevance.Calibrator.fit(raw, ["q"])
    with pytest.raises(ValueError, match="scored on the channels"):
        cal.relevance({"video_text": 0.2, "video_clip": 0.5})


def test_image_mode_selects_channels_as_before():
    raw = {"q": _raw(400, ("image_clip", "image_cap"), 0.2, seed=5)}
    cal = relevance.Calibrator.fit(raw, ["q"], "clip")
    both = {"image_clip": 0.1, "image_cap": 0.9}
    assert cal.relevance(both, "clip") == cal.ecdf("image_clip", 0.1)
    assert relevance.used_channels(both, "caption") == ["image_cap"]
    assert relevance.used_channels(both, "mix") == ["image_clip", "image_cap"]


class _FakeBGE:
    def encode(self, texts, query=False, batch=32):
        out = np.zeros((len(texts), 4))
        for i, text in enumerate(texts):
            out[i, len(text) % 4] = 1.0
        return out


def test_captioned_image_with_cid_starting_f_keeps_its_caption_score():
    """The old filter `not cid.startswith("f")` was dead code for loader ids; a caption must never be dropped by
    the identifier's spelling."""
    example = {"qid": "q", "question": "what is it?", "candidates": [
        {"cid": "f1", "modality": "image", "title": "A title", "text": "caption", "image": None},
        {"cid": "i2", "modality": "image", "title": "", "text": "", "image": None},
    ]}
    scores = relevance.raw_scores([example], _FakeBGE(), None)["q"]
    assert "image_cap" in scores["f1"] and "image_cap" not in scores["i2"]

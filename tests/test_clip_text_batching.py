"""CLIP query encoding must stay bounded for the full WebQA training split."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omniaware.relevance import CLIP


class Encoding(dict):
    def to(self, device):
        return self


@pytest.mark.parametrize("wrapped", [False, True])
def test_clip_batches_preserve_normalized_embeddings_and_order(wrapped):
    clip = CLIP.__new__(CLIP)
    clip.device = "cpu"
    calls = []

    def tokenize(texts, **kwargs):
        calls.append(len(texts))
        # Refuse a request larger than the device can hold.
        assert len(texts) <= 3
        return Encoding(input_ids=torch.tensor([[int(t), 1.0] for t in texts]))

    def features(input_ids):
        return SimpleNamespace(pooler_output=input_ids) if wrapped else input_ids

    clip.proc = SimpleNamespace(tokenizer=tokenize)
    clip.model = SimpleNamespace(get_text_features=features, config=SimpleNamespace(projection_dim=2))
    actual = clip.encode_text([str(i) for i in range(11)], batch=3)
    expected = np.array([[i, 1.0] for i in range(11)], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert calls == [3, 3, 3, 2]
    assert clip.encode_text([]).shape == (0, 2)
    with pytest.raises(ValueError, match="positive"):
        clip.encode_text(["1"], batch=0)

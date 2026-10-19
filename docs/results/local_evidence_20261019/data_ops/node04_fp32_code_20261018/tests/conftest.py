import pytest

from eab_fixtures import make_qwen
from omniaware import data


@pytest.fixture(scope="session")
def qwen():
    return make_qwen()


@pytest.fixture
def tiny_reserves(monkeypatch):
    """The synthetic universes of the run tests hold a few dozen questions: shrink the protocol reserves to the
    --n-calib those tests use, so that calibration, dev and test are all non-empty."""
    for family, reserve in {"okvqa": 6, "webqa": 4, "videorag_synth": 4}.items():
        monkeypatch.setitem(data.CALIB_RESERVE, family, reserve)
    monkeypatch.setitem(data.TRAIN_RESERVE, "webqa", 4)

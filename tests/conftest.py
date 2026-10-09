import pytest

from eab_fixtures import make_qwen


@pytest.fixture(scope="session")
def qwen():
    return make_qwen()

"""Precision propagation, calibration isolation and safe resume across actual run entry points."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from omniaware import run
from omniaware.baselines import seakr
from test_cache_resume import BASE, _setup
from test_run_smoke import FakeQwen

pytestmark = pytest.mark.usefixtures("tiny_reserves")


def test_dtype_reaches_model_and_isolates_calibration_and_resume(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    loaded, scored = [], []

    class Model(FakeQwen):
        def __init__(self, model_path, **kw):
            super().__init__(model_path, **kw)
            loaded.append(kw["dtype"])
            self.model = SimpleNamespace(dtype=kw["dtype"])

    def score(qwen, enc, *args, **kw):
        scored.append(qwen.model.dtype)
        return float(enc.length)

    monkeypatch.setattr(run, "QwenEAB", Model)
    monkeypatch.setattr(seakr, "score_input", score)
    args = BASE + ["--seakr-samples", "4", "--calibrate-only"]
    cache = tmp_path / "cache"
    run.main(args + ["--tag", "bf16", "--dtype", "bfloat16"])
    old = {p: p.read_bytes() for p in cache.glob("*.json")}
    scored.clear()
    run.main(args + ["--tag", "fp32", "--dtype", "float32"])
    assert loaded == [torch.bfloat16, torch.float32]
    assert scored and set(scored) == {torch.float32}  # no old SeaKR calibration hit
    for p, content in old.items():
        assert p.read_bytes() == content
    for kind in ("ucal", "seakr"):
        bf, = cache.glob(f"{kind}_*bfloat16.json")
        fp, = cache.glob(f"{kind}_*float32.json")
        a, b = json.loads(bf.read_text()), json.loads(fp.read_text())
        assert a.keys() == b.keys()
        assert all(a[q]["fingerprint"] != b[q]["fingerprint"] for q in a)
    cfg = json.loads((tmp_path / "runs/fp32/okvqa_config.json").read_text())
    assert cfg["dtype"] == "float32"
    env = json.loads((tmp_path / "runs/fp32/okvqa.env.ndjson").read_text().splitlines()[0])
    assert env["model"]["dtype"] == "torch.float32"

    run.main(BASE + ["--tag", "answers", "--n-eval", "2", "--dtype", "bfloat16"])
    answers = tmp_path / "runs/answers/okvqa.jsonl"
    original = answers.read_bytes()
    with pytest.raises(SystemExit) as error:
        run.main(BASE + ["--tag", "answers", "--n-eval", "2", "--dtype", "float32"])
    assert error.value.code == 2
    assert "saved answers use different dtype" in capsys.readouterr().err
    assert answers.read_bytes() == original


def test_wrong_loaded_dtype_fails_before_calibration(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)

    class WrongModel(FakeQwen):
        def __init__(self, model_path, **kw):
            super().__init__(model_path, **kw)
            self.model = SimpleNamespace(dtype=torch.bfloat16)

    monkeypatch.setattr(run, "QwenEAB", WrongModel)
    with pytest.raises(RuntimeError, match="requested model dtype"):
        run.main(BASE + ["--tag", "wrong", "--dtype", "float32"])
    assert not list((tmp_path / "cache").glob("ucal_*.json"))


@pytest.mark.parametrize("stage,extra", [("s1", []), ("s2", ["--position-ablation"])])
def test_stage_cli_propagates_dtype_and_isolated_paths(tmp_path, stage, extra):
    root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "fp32_20261018"
    env = dict(os.environ, OMNIAWARE_RUNS=str(isolated / "runs"),
               OMNIAWARE_CACHE=str(isolated / "cache"))
    result = subprocess.run([sys.executable, str(root / "scripts/v2_runs.py"),
                             "--stage", stage, "--dataset", "webqa_full", "--dtype", "float32",
                             "--dry-run", *extra], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    decodes = [line for line in result.stdout.splitlines() if "-m omniaware.run " in line]
    assert len(decodes) == (1 if stage == "s1" else 3)
    assert all("--dtype float32" in line for line in decodes)
    if stage == "s2":
        assert "--analysis-eab" not in decodes[0]
        assert decodes[1].endswith("--analysis-eab")
        assert str(isolated / "runs/v2_decisions") in result.stdout
    assert str(isolated / "runs") in result.stdout
    assert str(root / "runs") not in result.stdout
    probe = subprocess.run([sys.executable, "-c",
                            "from omniaware import paths; print(paths.LOCAL_CACHE)"],
                           cwd=root, env=dict(env, PYTHONPATH=str(root / "src")),
                           capture_output=True, text=True)
    assert probe.returncode == 0 and probe.stdout.strip() == str(isolated / "cache")
    assert not isolated.exists()  # dry-run did not create or overwrite assets/results

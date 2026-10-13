"""Dry-run checks for S1, S2 and S4. These tests do not launch a model."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.v2_runs as v2

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "scripts" / "v2_runs.py"
ACTIONS = "direct,static_text,static_image,static_mixed,rerank_static"


def _cli(*args):
    return subprocess.run(
        [sys.executable, str(RUNS), *args],
        cwd=ROOT, capture_output=True, text=True,
    )


@pytest.mark.parametrize("dataset,source", [("webqa_full", "train"), ("okvqa", "heldout")])
def test_s1_dry_run_names_dataset_dev_split_and_actions(dataset, source):
    tag = ROOT / "runs" / f"v2_s1_{dataset}"
    assert not tag.exists()
    result = _cli("--stage", "s1", "--dataset", dataset, "--n", "50", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert lines[1] == "python scripts/analysis/image_evidence_check.py --n 50"
    cmd = lines[0]
    assert f"--dataset {dataset}" in cmd
    assert "--split dev" in cmd
    assert "--split test" not in cmd
    assert "--n-eval 50" in cmd
    assert f"--calib-source {source}" in cmd
    assert "--n-calib 500" in cmd
    assert "--n-dev 256" in cmd
    assert ACTIONS in cmd
    assert "--empty-answer-accounting" in cmd
    assert "--shared-prompt-registry" in cmd
    assert "--save-signals" in cmd
    assert not tag.exists()


@pytest.mark.parametrize(
    "dataset,source",
    [("webqa_full", "train"), ("webqa_open_img", "train"), ("okvqa", "heldout")],
)
def test_s2_dry_run_generates_calib_and_dev_not_test(dataset, source):
    result = _cli("--stage", "s2", "--dataset", dataset, "--dry-run")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    calib, dev = lines
    assert f"--dataset {dataset}" in calib and f"--dataset {dataset}" in dev
    assert "--split calib" in calib
    assert "--split dev" in dev
    assert "--split test" not in result.stdout
    assert "--n-eval" not in result.stdout
    assert f"--calib-source {source}" in calib and f"--calib-source {source}" in dev
    assert "--n-calib 500" in calib and "--n-calib 500" in dev
    assert "--n-dev 256" in calib and "--n-dev 256" in dev
    assert ACTIONS in calib and ACTIONS in dev
    assert "--save-signals" in calib and "--save-signals" in dev
    assert f"--tag v2_s2_{dataset}_calib" in calib
    assert f"--tag v2_s2_{dataset}_dev" in dev
    assert "image_evidence_check.py" not in result.stdout


def test_s4_dry_run_is_one_test_command(tmp_path):
    decisions = tmp_path / "dev_decisions.json"
    decisions.write_text(json.dumps({"note": "frozen"}), encoding="utf-8")
    result = _cli(
        "--stage", "s4", "--dataset", "webqa_full", "--decisions", str(decisions), "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    cmd = lines[0]
    assert cmd.count("python -m omniaware.run") == 1
    assert "--dataset webqa_full" in cmd
    assert "--split test" in cmd
    assert "--split dev" not in cmd
    assert "--n-eval" not in cmd
    assert ACTIONS in cmd
    assert str(decisions) in cmd
    assert "--tau" not in cmd
    assert "--beta" not in cmd


def test_s4_okvqa_uses_heldout_calib(tmp_path):
    decisions = tmp_path / "dev_decisions.json"
    decisions.write_text("{}", encoding="utf-8")
    result = _cli("--stage", "s4", "--dataset", "okvqa", "--decisions", str(decisions), "--dry-run")
    assert result.returncode == 0, result.stderr
    cmd = result.stdout.strip()
    assert "--dataset okvqa" in cmd
    assert "--split test" in cmd
    assert "--calib-source heldout" in cmd
    assert ACTIONS in cmd
    assert str(decisions) in cmd


def test_s4_missing_decisions_file_is_rejected(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    result = _cli("--stage", "s4", "--dataset", "webqa_full", "--decisions", str(missing), "--dry-run")
    assert result.returncode != 0
    assert "omniaware.run" not in result.stdout
    assert "refuses" in result.stderr
    assert "No default hyperparameters were substituted." in result.stderr
    assert not missing.exists()


def test_dry_run_does_not_launch_a_model(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError(f"subprocess was called: {args}")

    monkeypatch.setattr(v2.subprocess, "run", boom)
    v2.main(["--stage", "s2", "--dataset", "webqa_open_img", "--dry-run"])
    out = capsys.readouterr().out
    assert "webqa_open_img" in out
    assert ACTIONS in out

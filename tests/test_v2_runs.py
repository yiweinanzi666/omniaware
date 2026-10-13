"""Dry-run checks for S1, S2 and S4. These tests do not launch a model or a scoring step."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.v2_runs as v2

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "scripts" / "v2_runs.py"
PY = shlex.quote(sys.executable)


def _cli(*args):
    return subprocess.run(
        [sys.executable, str(RUNS), *args],
        cwd=ROOT, capture_output=True, text=True,
    )


@pytest.mark.parametrize("dataset,source", [("webqa_full", "train"), ("okvqa_gs", "heldout")])
def test_s1_dry_run_names_dataset_dev_split_and_v2_flags(dataset, source):
    tag = ROOT / "runs" / f"v2_s1_{dataset}"
    assert not tag.exists()
    result = _cli("--stage", "s1", "--dataset", dataset, "--n", "50", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 2
    assert lines[1] == f"{PY} scripts/analysis/image_evidence_check.py --n 50"
    cmd = lines[0]
    assert cmd.startswith(PY + " -m omniaware.run ")
    assert f"--dataset {dataset}" in cmd
    assert "--split dev" in cmd and "--split test" not in cmd
    assert "--n-eval 50" in cmd
    assert f"--calib-source {source}" in cmd
    assert "--n-calib 500" in cmd and "--n-dev 256" in cmd
    assert "--v2 --v2-reranker" in cmd
    assert "--selector" not in cmd
    assert not any(flag in cmd for flag in ("--v2-actions", "--save-signals", "--empty-answer-accounting",
                                            "--shared-prompt-registry"))
    assert not tag.exists()


@pytest.mark.parametrize("dataset,source", [("webqa_full", "train"), ("webqa_open_img", "train"),
                                            ("okvqa_gs", "heldout")])
def test_s2_dry_run_decodes_calib_and_dev_then_scores_gates_and_selects(dataset, source):
    result = _cli("--stage", "s2", "--dataset", dataset, "--dry-run")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    calib_tag, dev_tag = f"v2_s2_{dataset}_calib", f"v2_s2_{dataset}_dev"
    decodes, rest = lines[:2], lines[2:]
    calib, dev = decodes
    assert calib.startswith(PY + " -m omniaware.run ") and dev.startswith(PY + " -m omniaware.run ")
    assert f"--tag {calib_tag}" in calib and "--split calib" in calib
    assert f"--tag {dev_tag}" in dev and "--split dev" in dev
    for cmd in decodes:
        assert f"--dataset {dataset}" in cmd and f"--calib-source {source}" in cmd
        assert "--n-calib 500" in cmd and "--n-dev 256" in cmd and "--v2 --v2-reranker" in cmd
        assert "--n-eval" not in cmd and "--split test" not in cmd
    assert rest == [
        f"{PY} -m omniaware.eval.score_official runs/{calib_tag}/{dataset}.jsonl",
        f"{PY} -m omniaware.eval.score_official runs/{dev_tag}/{dataset}.jsonl",
        f"{PY} scripts/analysis/v2_table.py build --run runs/{calib_tag}/{dataset}.jsonl",
        f"{PY} scripts/analysis/v2_table.py build --run runs/{dev_tag}/{dataset}.jsonl",
        f"{PY} scripts/analysis/v2_table.py gate --table runs/{dev_tag}/{dataset}.v2_actions.jsonl",
        f"{PY} scripts/analysis/v2_table.py select --table runs/{calib_tag}/{dataset}.v2_actions.jsonl "
        f"--out runs/v2_decisions/{dataset}.json",
    ]
    assert "image_evidence_check.py" not in result.stdout


def test_s4_dry_run_decodes_test_then_scores_builds_and_applies(tmp_path):
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"note": "frozen"}), encoding="utf-8")
    result = _cli("--stage", "s4", "--dataset", "webqa_full", "--decisions", str(decisions), "--dry-run")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    run_line, score, build, apply_line = lines
    assert run_line.startswith(PY + " -m omniaware.run ") and "--dataset webqa_full" in run_line
    assert "--split test" in run_line and "--split dev" not in run_line and "--n-eval" not in run_line
    quoted = shlex.quote(str(decisions))
    assert "--v2 --v2-reranker" in run_line and f"--decisions {quoted}" in run_line
    assert "--tau" not in run_line and "--beta" not in run_line
    tag = "v2_s4_webqa_full"
    assert score == f"{PY} -m omniaware.eval.score_official runs/{tag}/webqa_full.jsonl"
    assert build == f"{PY} scripts/analysis/v2_table.py build --run runs/{tag}/webqa_full.jsonl"
    assert apply_line == (f"{PY} scripts/analysis/v2_table.py apply --table runs/{tag}/webqa_full.v2_actions.jsonl "
                          f"--decisions {quoted} --out runs/{tag}/webqa_full.v2_main.json")


def test_s4_okvqa_gs_uses_heldout_calib(tmp_path):
    decisions = tmp_path / "dev_decisions.json"
    decisions.write_text("{}", encoding="utf-8")
    result = _cli("--stage", "s4", "--dataset", "okvqa_gs", "--decisions", str(decisions), "--dry-run")
    assert result.returncode == 0, result.stderr
    cmd = result.stdout.splitlines()[0]
    assert "--dataset okvqa_gs" in cmd and "--split test" in cmd and "--calib-source heldout" in cmd
    assert shlex.quote(str(decisions)) in cmd


def test_plain_okvqa_is_not_a_dataset_of_the_v2_stages():
    result = _cli("--stage", "s2", "--dataset", "okvqa", "--dry-run")
    assert result.returncode != 0 and "invalid choice" in result.stderr


def test_s4_missing_decisions_file_is_rejected(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    result = _cli("--stage", "s4", "--dataset", "webqa_full", "--decisions", str(missing), "--dry-run")
    assert result.returncode != 0
    assert "omniaware.run" not in result.stdout
    assert "refuses" in result.stderr
    assert "No default hyperparameters were substituted." in result.stderr
    assert not missing.exists()


def test_s4_decisions_must_be_a_json_object(tmp_path):
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    result = _cli("--stage", "s4", "--dataset", "webqa_full", "--decisions", str(bad), "--dry-run")
    assert result.returncode != 0 and "must be a JSON object" in result.stderr


def test_dry_run_does_not_launch_a_model(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError(f"subprocess was called: {args}")

    monkeypatch.setattr(v2.subprocess, "run", boom)
    v2.main(["--stage", "s2", "--dataset", "webqa_open_img", "--dry-run"])
    out = capsys.readouterr().out
    assert "webqa_open_img" in out and "--v2 --v2-reranker" in out

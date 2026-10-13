"""Dry-run checks for S1, S2 and S4, and the stage's stop rules. These tests do not launch a model or a scoring step."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.v2_runs as v2
from omniaware import paths, runio

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "scripts" / "v2_runs.py"
IMAGE_CHECK = ROOT / "scripts" / "analysis" / "image_evidence_check.py"
V2_TABLE = ROOT / "scripts" / "analysis" / "v2_table.py"


def _cli(*args):
    return subprocess.run([sys.executable, str(RUNS), *args], cwd=ROOT, capture_output=True, text=True)


def _fmt(*argv):
    return v2.format_argv(list(argv))


def _decisions_with_check(path, passed=True, content=None):
    """A decisions file and the development check that S2 writes beside it (sha256 of that decisions file)."""
    path.write_text(json.dumps(content if content is not None else {"note": "frozen"}), encoding="utf-8")
    check = {"passed": passed, "relative_gain": 0.07 if passed else 0.01,
             "ci95_pp": [0.4, 3.0] if passed else [-1.0, 2.0],
             "threshold": 0.05, "fallback": False, "decisions": {"path": str(path), "sha256": runio.sha256_file(path)}}
    Path(v2.dev_check_path(str(path))).write_text(json.dumps(check), encoding="utf-8")
    return path


@pytest.mark.parametrize("dataset,source", [("webqa_full", "train"), ("okvqa_gs", "heldout")])
def test_s1_dry_run_names_dataset_dev_split_flags_and_the_empty_answer_gate(dataset, source):
    result = _cli("--stage", "s1", "--dataset", dataset, "--n", "50", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 3
    assert lines[1] == _fmt(sys.executable, str(ROOT / "scripts" / "analysis" / "answer_slot.py"), "format", "--run",
                            str(paths.RUNS / f"v2_s1_{dataset}" / f"{dataset}.jsonl"), "--min-answer-first", "0.9")
    assert lines[2] == _fmt(sys.executable, str(IMAGE_CHECK), "--n", "50")
    cmd = lines[0]
    assert cmd.startswith(_fmt(sys.executable, "-m", "omniaware.run") + " ")
    assert f"--dataset {dataset}" in cmd
    assert "--split dev" in cmd and "--split test" not in cmd
    assert "--n-eval 50" in cmd
    assert f"--calib-source {source}" in cmd
    assert "--n-calib 500" in cmd and "--n-dev 256" in cmd
    assert "--v2 --v2-reranker" in cmd
    assert "--max-empty-rate 0.005" in cmd
    assert "--fitm" not in cmd and "--seakr-samples" not in cmd and "--selector" not in cmd


def test_s2_dry_run_decodes_calib_and_dev_then_scores_builds_gates_and_selects_for_webqa():
    result = _cli("--stage", "s2", "--dataset", "webqa_full", "--dry-run")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    calib_tag, dev_tag = "v2_s2_webqa_full_calib", "v2_s2_webqa_full_dev"
    calib, dev, *rest = lines
    assert calib.startswith(_fmt(sys.executable, "-m", "omniaware.run") + " ") and f"--tag {calib_tag}" in calib
    assert "--split calib" in calib
    assert f"--tag {dev_tag}" in dev and "--split dev" in dev
    # The baseline arms run on the dev decode only; the calibration table reads the v2 actions alone.
    assert "--fitm --m2rag --seakr-samples 20" in dev
    assert "--fitm" not in calib and "--seakr-samples" not in calib
    for cmd in (calib, dev):
        assert "--calib-source train" in cmd and "--n-calib 500" in cmd and "--n-dev 256" in cmd
        assert "--v2 --v2-reranker" in cmd and "--n-eval" not in cmd and "--split test" not in cmd
    run_file = lambda tag: str(paths.RUNS / tag / "webqa_full.jsonl")   # noqa: E731
    assert rest == [
        _fmt(sys.executable, "-m", "omniaware.eval.score_official", run_file(calib_tag), "--bart"),
        _fmt(sys.executable, "-m", "omniaware.eval.score_official", run_file(dev_tag), "--bart"),
        _fmt(sys.executable, str(V2_TABLE), "build", "--run", run_file(calib_tag), "--bart"),
        _fmt(sys.executable, str(V2_TABLE), "build", "--run", run_file(dev_tag), "--bart"),
        _fmt(sys.executable, str(ROOT / "scripts" / "analysis" / "answer_slot.py"), "position", "--table",
             str(paths.RUNS / dev_tag / "webqa_full.v2_actions.jsonl")),
        _fmt(sys.executable, str(V2_TABLE), "gate", "--table",
             str(paths.RUNS / dev_tag / "webqa_full.v2_actions.jsonl")),
        _fmt(sys.executable, str(V2_TABLE), "select", "--table",
             str(paths.RUNS / calib_tag / "webqa_full.v2_actions.jsonl"), "--out",
             str(paths.RUNS / "v2_decisions" / "webqa_full.json")),
        _fmt(sys.executable, str(V2_TABLE), "devcheck", "--table",
             str(paths.RUNS / dev_tag / "webqa_full.v2_actions.jsonl"), "--decisions",
             str(paths.RUNS / "v2_decisions" / "webqa_full.json"), "--out",
             str(paths.RUNS / "v2_decisions" / "webqa_full.dev_check.json")),
    ]
    assert "image_evidence_check.py" not in result.stdout


def test_position_ablation_decodes_the_dev_set_again_in_the_sentence_format():
    result = _cli("--stage", "s2", "--dataset", "webqa_full", "--position-ablation", "--dry-run")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    sentence = [line for line in lines if "--tag v2_s2_webqa_full_dev_sentence" in line]
    assert len(sentence) == 1 and sentence[0].endswith("--answer-style sentence") and "--fitm" not in sentence[0]
    table = str(paths.RUNS / "v2_s2_webqa_full_dev_sentence" / "webqa_full.v2_actions.jsonl")
    assert any(line.endswith(_fmt("position", "--table", table)) for line in lines)
    assert lines[-3].split()[2] == "gate"                # the ablation runs before the gate, select and devcheck
    refused = _cli("--stage", "s2", "--dataset", "okvqa_gs", "--position-ablation", "--dry-run")
    assert refused.returncode != 0 and "WebQA" in refused.stderr


def test_s2_okvqa_has_no_bart_flags_and_uses_heldout_calibration():
    result = _cli("--stage", "s2", "--dataset", "okvqa_gs", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "--bart" not in result.stdout
    assert all("--calib-source heldout" in line for line in result.stdout.splitlines()[:2])


@pytest.mark.parametrize("dataset", ["webqa_full", "webqa_open_img"])
def test_webqa_is_the_bart_dataset_of_both_settings(dataset):
    assert dataset in v2.WEBQA_DATASETS
    assert "--bart" in v2.score_command("t", dataset) and "--bart" in v2.build_command("t", dataset)


def test_no_seakr_leaves_the_baseline_arm_out_and_says_so():
    result = _cli("--stage", "s2", "--dataset", "webqa_full", "--no-seakr", "--dry-run")
    assert result.returncode == 0, result.stderr
    first = result.stdout.splitlines()[0]
    assert first.startswith("NOTE: --no-seakr") and "no SeaKR row" in first
    dev = [line for line in result.stdout.splitlines() if "--tag v2_s2_webqa_full_dev" in line][0]
    assert "--fitm --m2rag" in dev and "--seakr-samples" not in dev


def test_no_seakr_is_rejected_for_s1():
    result = _cli("--stage", "s1", "--dataset", "webqa_full", "--no-seakr", "--dry-run")
    assert result.returncode != 0 and "S1 runs no baseline arms" in result.stderr


def test_s4_dry_run_decodes_test_with_baselines_then_scores_builds_and_applies(tmp_path):
    decisions = _decisions_with_check(tmp_path / "decisions.json")
    result = _cli("--stage", "s4", "--dataset", "webqa_full", "--decisions", str(decisions), "--dry-run")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    run_line, score, build, apply_line = lines
    assert run_line.startswith(_fmt(sys.executable, "-m", "omniaware.run") + " ")
    assert "--dataset webqa_full" in run_line and "--split test" in run_line and "--n-eval" not in run_line
    assert "--v2 --v2-reranker" in run_line and "--fitm --m2rag --seakr-samples 20" in run_line
    assert f"--decisions {v2.format_argv([str(decisions)])}" in run_line
    tag = "v2_s4_webqa_full"
    run_file = str(paths.RUNS / tag / "webqa_full.jsonl")
    assert score == _fmt(sys.executable, "-m", "omniaware.eval.score_official", run_file, "--bart")
    assert build == _fmt(sys.executable, str(V2_TABLE), "build", "--run", run_file, "--bart")
    assert apply_line == _fmt(sys.executable, str(V2_TABLE), "apply", "--table",
                              str(paths.RUNS / tag / "webqa_full.v2_actions.jsonl"), "--decisions", str(decisions),
                              "--out", str(paths.RUNS / tag / "webqa_full.v2_main.json"))


def test_s4_okvqa_gs_uses_heldout_calib_and_no_bart(tmp_path):
    decisions = _decisions_with_check(tmp_path / "dev_decisions.json", content={})
    result = _cli("--stage", "s4", "--dataset", "okvqa_gs", "--decisions", str(decisions), "--dry-run")
    assert result.returncode == 0, result.stderr
    run_line, score, build, _apply = result.stdout.splitlines()
    assert "--dataset okvqa_gs" in run_line and "--split test" in run_line and "--calib-source heldout" in run_line
    assert "--bart" not in score and "--bart" not in build


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


def test_s4_refuses_without_a_development_check(tmp_path):
    decisions = tmp_path / "webqa_full.json"
    decisions.write_text(json.dumps({"note": "frozen"}), encoding="utf-8")
    result = _cli("--stage", "s4", "--dataset", "webqa_full", "--decisions", str(decisions), "--dry-run")
    assert result.returncode != 0 and "omniaware.run" not in result.stdout
    assert "no development check" in result.stderr and "webqa_full.dev_check.json" in result.stderr


def test_s4_refuses_a_development_check_of_other_decisions(tmp_path):
    decisions = _decisions_with_check(tmp_path / "webqa_full.json")
    decisions.write_text(json.dumps({"note": "changed after the check"}), encoding="utf-8")
    result = _cli("--stage", "s4", "--dataset", "webqa_full", "--decisions", str(decisions), "--dry-run")
    assert result.returncode != 0 and "different decisions file" in result.stderr


def test_s4_refuses_a_failed_development_check_unless_allowed(tmp_path):
    decisions = _decisions_with_check(tmp_path / "webqa_full.json", passed=False)
    refused = _cli("--stage", "s4", "--dataset", "webqa_full", "--decisions", str(decisions), "--dry-run")
    assert refused.returncode != 0 and "development check failed" in refused.stderr
    assert "--allow-failed-dev-check" in refused.stderr and "omniaware.run" not in refused.stdout
    allowed = _cli("--stage", "s4", "--dataset", "webqa_full", "--decisions", str(decisions),
                   "--allow-failed-dev-check", "--dry-run")
    assert allowed.returncode == 0, allowed.stderr
    lines = allowed.stdout.splitlines()
    assert lines[0].startswith("NOTE: --allow-failed-dev-check") and "relative_gain=0.01" in lines[0]
    assert len(lines) == 5


def test_allow_failed_dev_check_is_only_for_s4():
    result = _cli("--stage", "s2", "--dataset", "webqa_full", "--allow-failed-dev-check", "--dry-run")
    assert result.returncode != 0 and "only for --stage s4" in result.stderr


def test_dry_run_does_not_launch_a_model(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError(f"subprocess was called: {args}")

    monkeypatch.setattr(v2.subprocess, "run", boom)
    v2.main(["--stage", "s2", "--dataset", "webqa_open_img", "--dry-run"])
    out = capsys.readouterr().out
    assert "webqa_open_img" in out and "--v2 --v2-reranker" in out


class _Exit:
    """Fake subprocess.run: records the commands and returns the scripted exit codes in order."""

    def __init__(self, codes):
        self.codes, self.calls = list(codes), []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, self.codes.pop(0))


def test_gate_failure_stops_the_stage_before_the_selection(monkeypatch):
    fake = _Exit([1])                    # the gate reports FAIL (exit 1); the selection is the next command
    monkeypatch.setattr(v2.subprocess, "run", fake)
    commands = [[sys.executable, str(V2_TABLE), "gate", "--table", "dev.jsonl"],
                [sys.executable, str(V2_TABLE), "select", "--table", "calib.jsonl", "--out", "d.json"]]
    with pytest.raises(SystemExit) as info:
        v2.execute(commands)
    assert "gate FAIL" in str(info.value) and "not run" in str(info.value)
    assert [call[2] for call in fake.calls] == ["gate"]


def test_dev_check_failure_ends_the_stage_and_names_the_s4_rule(monkeypatch):
    fake = _Exit([0, 1])
    monkeypatch.setattr(v2.subprocess, "run", fake)
    commands = [[sys.executable, str(V2_TABLE), "select", "--table", "calib.jsonl", "--out", "d.json"],
                [sys.executable, str(V2_TABLE), "devcheck", "--table", "dev.jsonl", "--decisions", "d.json",
                 "--out", "d.dev_check.json"]]
    with pytest.raises(SystemExit) as info:
        v2.execute(commands)
    assert "development check FAIL" in str(info.value) and "--allow-failed-dev-check" in str(info.value)


def test_format_failure_stops_s1_and_names_the_report(monkeypatch):
    fake = _Exit([0, 1])
    monkeypatch.setattr(v2.subprocess, "run", fake)
    commands = [[sys.executable, "-m", "omniaware.run", "--tag", "v2_s1_webqa_full"],
                [sys.executable, "answer_slot.py", "format", "--run", "webqa_full.jsonl"]]
    with pytest.raises(SystemExit) as info:
        v2.execute(commands)
    assert "answer-first format FAIL" in str(info.value) and "answer_format.json" in str(info.value)


def test_a_failing_step_stops_the_stage_with_its_command(monkeypatch):
    fake = _Exit([2, 0])
    monkeypatch.setattr(v2.subprocess, "run", fake)
    commands = [[sys.executable, "-m", "omniaware.run", "--tag", "t"], [sys.executable, "-m", "omniaware.eval"]]
    with pytest.raises(SystemExit) as info:
        v2.execute(commands)
    assert "exit code 2" in str(info.value) and "omniaware.run" in str(info.value)
    assert len(fake.calls) == 1


def test_the_printed_command_quotes_for_the_platform():
    if os.name == "nt":
        assert v2.format_argv(["x", "a b"]) == 'x "a b"'
    else:
        assert v2.format_argv(["x", "a b"]) == "x 'a b'"

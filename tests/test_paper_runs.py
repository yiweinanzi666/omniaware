"""The command list behind the paper tables must stay valid for the current runner, and must refuse to run unfrozen."""
import pytest

from omniaware import pipeline, run
import scripts.paper_runs as pr


def _frozen(**kw):
    base = {**pr.PLACEHOLDER, "model": "8b", "phase": 2, "eab_variant": "eab:g0.9:ans1:pos:gate", "gamma": 0.9,
            "eab_rows": "ans1", "eab_form": "pos", "beta": 0.5, "tau_bias": 0.6, "tau": 0.45}
    base.update(kw)
    return base


def _cmd(stage, needle, tag=None):
    for c in stage:
        if needle in c and (tag is None or tag in c):
            return c
    raise AssertionError(f"no command with {needle} {tag}")


def test_every_paper_run_command_parses_and_its_variants_exist():
    parser = run.build_parser()
    n = 0
    for model in ("8b", "internvl3.5-8b"):
        for st in pr.stages(model, "test").values():
            for cmd in st["commands"]:
                if cmd[:3] != ["python", "-m", "omniaware.run"]:
                    continue
                args = parser.parse_args(cmd[3:])
                for v in args.variants.split(","):
                    if v:
                        pipeline.parse_variant(v)
                assert not (args.m2rag and args.dataset == "videorag_synth")
                assert not (args.fitm and args.model == "internvl3.5-8b")
                n += 1
    assert n > 150


def test_ablation_rows_survive_any_frozen_form(monkeypatch):
    monkeypatch.setattr(pr, "decisions", lambda model: _frozen(eab_variant="eab:g1.2:q:ctr:gate", gamma=1.2,
                                                               eab_rows="q", eab_form="ctr", beta=0.3, tau_bias=0.4,
                                                               tau=0.35))
    cmds = [c for c in pr.stages("8b", "test")["ablation"]["commands"] if "--replication-gamma" in c]
    variants = cmds[0][cmds[0].index("--variants") + 1].split(",")
    assert len(variants) == 7 and "eab:g1.2:ans2:ctr:gate" in variants and "eab:g1.2:q:fixed:gate" in variants


def test_frozen_choices_reach_every_test_command(monkeypatch):
    monkeypatch.setattr(pr, "decisions", lambda model: _frozen())
    st = pr.stages("8b", "test")
    main = st["main"]["commands"]
    runs_ = [c for c in main if c[:3] == ["python", "-m", "omniaware.run"]]
    assert runs_ and all("eab:g0.9:ans1:pos:gate" in c for c in runs_ if "--variants" in c)
    for c in runs_:
        i = c.index("--beta")
        assert c[i + 1] == "0.5" and c[c.index("--tau-bias") + 1] == "0.6" and c[c.index("--tau") + 1] == "0.45"
    assert "NOT FROZEN" not in st["main"]["fills"]


def test_sensitivity_grid_is_labelled_and_takes_explicit_hyperparameters(monkeypatch):
    monkeypatch.setattr(pr, "decisions", lambda model: _frozen())
    abl = pr.stages("8b", "test")["ablation"]
    assert "SENSITIVITY ONLY" in abl["fills"]
    cell = _cmd(abl["commands"], "--tau-bias", "v321_beta0.15_tb0.2_8b")
    assert cell[cell.index("--beta") + 1] == "0.15" and cell[cell.index("--tau-bias") + 1] == "0.2"


def test_unfrozen_test_stage_refuses_to_run(monkeypatch, capsys):
    monkeypatch.setattr(pr, "DECISIONS", pr.ROOT / "runs" / "does-not-exist.json")
    with pytest.raises(SystemExit, match="needs the frozen test choices"):
        pr.main(["--stage", "main", "--run"])
    pr.main(["--list"])
    assert "未冻结" in capsys.readouterr().out


def test_missing_okvqa_candidates_are_an_error_not_a_skip(monkeypatch, tmp_path):
    missing = {k: tmp_path / f"{k}.jsonl" for k in ("okvqa", "okvqa_dense", "okvqa_gs")}
    monkeypatch.setattr(pr, "okvqa_candidate_files", lambda: missing)
    with pytest.raises(SystemExit, match="missing OK-VQA candidate corpus"):
        pr.main(["--stage", "dev", "--run"])
    assert len(pr.stages("8b")["dev"]["missing"]) == 3


def test_dev_stage_runs_every_candidate_and_selects_in_two_phases(monkeypatch):
    monkeypatch.setattr(pr, "okvqa_candidate_files", lambda: {})
    dev = pr.stages("8b")["dev"]["commands"]
    base = [c for c in dev if c[:3] == ["python", "-m", "omniaware.run"] and "v321dev_base_8b" in c]
    assert base and all(c[c.index("--variants") + 1].count("eab:g") == 30 for c in base)
    assert any(c[-3:] == ["--phase", "1", "--model"] or "--phase" in c for c in dev)
    assert pr.stages("8b")["dev_beta"]["requires"] == "phase1"


def test_beta_tau_grid_uses_the_phase1_candidate(monkeypatch):
    monkeypatch.setattr(pr, "load_decisions", lambda model: {"phase": 1, "model": "8b", "eab_variant":
                                                             "eab:g0.6:q:pos:gate", "okvqa_loader": "okvqa_gs",
                                                             "u_combine": "zsum", "attn_layer": "mid"})
    beta = pr.stages("8b")["dev_beta"]["commands"]
    runs_ = [c for c in beta if c[:3] == ["python", "-m", "omniaware.run"]]
    assert len(runs_) == 15 * 4 and all(c[c.index("--variants") + 1] == "eab:g0.6:q:pos:gate" for c in runs_)
    assert all(c[c.index("--u-combine") + 1] == "zsum" and c[c.index("--attn-layer") + 1] == "mid" for c in runs_)


def test_noprobe_keeps_first_round_routing():
    cmd = _cmd(pr.stages("8b")["ablation"]["commands"], "--tag", "v321_ablation_noprobe_8b")
    assert "--first-round-routing" in cmd and "--probe-uncertainty" not in cmd


def test_nocal_changes_only_the_statistics():
    cmd = _cmd(pr.stages("8b")["ablation"]["commands"], "--tag", "v321_ablation_nocal_8b")
    assert "--no-calibration" in cmd and "--relevance-norm" not in cmd


def test_seakr_runs_on_the_full_split_and_order_on_the_subsample():
    main = pr.stages("8b")["main"]["commands"]
    seakr = [c for c in main if "--seakr-samples" in c]
    assert seakr and all("--n-eval" not in c for c in seakr)
    order = pr.stages("8b")["order"]["commands"]
    assert all(c[c.index("--n-eval") + 1] == "1000" for c in order if "--order" in c)


def test_universalrag_uses_the_router_not_an_echo():
    ur = pr.stages("8b")["universalrag"]["commands"]
    assert not any(c[0] == "echo" for c in ur)
    assert any(c[:3] == ["python", "-m", "scripts.universalrag_retrieval.route_queries"] for c in ur)


def test_decode_only_control_when_the_frozen_form_is_ans2(monkeypatch):
    monkeypatch.setattr(pr, "decisions", lambda model: _frozen(eab_variant="eab:g1.2:ans2:pos:gate", gamma=1.2,
                                                               eab_rows="ans2", eab_form="pos"))
    order = pr.stages("8b")["order"]["commands"]
    perm = [c for c in order if "--order" in c][0]
    arms = perm[perm.index("--variants") + 1].split(",")
    assert arms == ["eab:g1.2:ans2:pos:gate", "eab:g1.2:ans1:pos:gate"]     # control is ans1, not a duplicate
    assert "ans1" in pr.stages("8b")["ablation"]["fills"] and "decode-only" in pr.stages("8b")["ablation"]["fills"]


def test_control_arm_names():
    assert pr.control_arm("ans1", "pos", 1.2) == ("eab:g1.2:ans2:pos:gate", "")
    assert pr.control_arm("q", "ctr", 0.9) == ("eab:g0.9:ans2:ctr:gate", "")
    control, note = pr.control_arm("ans2", "pos", 1.2)
    assert control == "eab:g1.2:ans1:pos:gate" and "ans1" in note


def test_globs_are_expanded_in_python_and_an_empty_match_fails(tmp_path):
    (tmp_path / "v_1").mkdir()
    (tmp_path / "v_1" / "x.jsonl").write_text("")
    (tmp_path / "v_2").mkdir()
    (tmp_path / "v_2" / "x.jsonl").write_text("")
    hits = pr.expand_globs(["python", "-m", "analysis", str(tmp_path / "v_*" / "x.jsonl"), "--arms", "static"])
    assert hits[3:5] == [str(tmp_path / "v_1" / "x.jsonl"), str(tmp_path / "v_2" / "x.jsonl")] and hits[-1] == "static"
    with pytest.raises(SystemExit, match="no file matches"):
        pr.expand_globs(["python", str(tmp_path / "nothing_*.jsonl")])


def test_analysis_commands_never_use_a_shell():
    src = (pr.ROOT / "scripts" / "paper_runs.py").read_text(encoding="utf8")
    assert "shell=True" not in src


def test_print_shows_patterns_without_expanding_or_refusing(monkeypatch, capsys):
    monkeypatch.setattr(pr, "okvqa_candidate_files", lambda: {"okvqa": pr.ROOT / "nope.jsonl"})
    pr.main(["--stage", "analysis", "--print", "--model", "8b"])     # nothing has run yet: no glob matches exist
    out = capsys.readouterr().out
    assert "--compare" in out and "--taus" in out
    pr.main(["--stage", "order", "--print", "--model", "8b"])
    assert "runs/v321_perm_*_8b" in capsys.readouterr().out          # the pattern is printed as written

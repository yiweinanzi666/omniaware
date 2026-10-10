"""The command list behind the paper tables must stay valid for the current runner."""
from omniaware import pipeline, run
from scripts.paper_runs import stages


def test_every_paper_run_command_parses_and_its_variants_exist():
    parser = run.build_parser()
    n = 0
    for model in ("8b", "internvl3.5-8b"):
        for st in stages(model, "test").values():
            for cmd in st["commands"]:
                if cmd[:3] != ["python", "-m", "omniaware.run"]:
                    continue
                args = parser.parse_args(cmd[3:])
                for v in args.variants.split(","):
                    pipeline.parse_variant(v)
                assert not (args.m2rag and args.dataset == "videorag_synth")
                assert not (args.fitm and args.model == "internvl3.5-8b")
                n += 1
    assert n > 150


def test_ablation_rows_survive_any_frozen_form(monkeypatch):
    import scripts.paper_runs as pr
    monkeypatch.setattr(pr, "decisions", lambda model: {**pr.DEFAULTS, "eab_variant": "eab:g1.2:q:ctr:gate"})
    cmds = [c for c in pr.stages("8b", "test")["ablation"]["commands"] if "--replication-gamma" in c]
    variants = cmds[0][cmds[0].index("--variants") + 1].split(",")
    assert len(variants) == 7 and "eab:g1.2:ans2:ctr:gate" in variants and "eab:g1.2:q:fixed:gate" in variants

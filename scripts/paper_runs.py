"""Every run, scoring and analysis command behind paper V3.21, in one place.

    python -m scripts.paper_runs --list                        # stages, frozen status and what each one fills
    python -m scripts.paper_runs --stage dev --run             # 1. pre-registered dev grid (--split dev)
    python -m scripts.analysis.dev_select --phase 1            #    -> (form, gamma), U setting, OK-VQA corpus
    python -m scripts.paper_runs --stage dev_beta --run        # 2. beta x tau_b grid on the phase-1 candidate
    python -m scripts.analysis.dev_select --phase 2            #    -> beta, tau_b, tau; runs/dev_decisions.json
    python -m scripts.paper_runs --stage main --run            # 3. test stages read the frozen choices
    python -m scripts.paper_runs --stage main --run --gpus auto # every run sharded over all visible GPUs

The dev stage compares 30 EAB candidates (6 forms x 5 gammas), three uncertainty settings and three OK-VQA
retrieval corpora; docs/DEV_DECISIONS.md states the decision rules before any result is seen. Test stages refuse to
run until runs/dev_decisions.json holds a complete (phase 2) decision for the model: --list and --print show the
commands with the placeholder values marked as not frozen.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.analysis.dev_select import (BETAS, DEV_DATASETS, EAB_CANDIDATES, OKVQA_LOADERS, TAU_BIASES,  # noqa: E402
                                        TAUS, eab_name)
from omniaware import paths  # noqa: E402

CALIB = {"webqa_open_full": "train", "webqa_open_img": "train", "okvqa": "heldout", "okvqa_dense": "heldout",
         "okvqa_gs": "heldout", "hybridqa": "heldout", "videorag_synth": "heldout"}
METRIC = {"webqa_open_full": "acc_approx", "webqa_open_img": "acc_approx", "okvqa": "vqa_accuracy",
          "okvqa_dense": "vqa_accuracy", "okvqa_gs": "vqa_accuracy", "hybridqa": "em", "videorag_synth": "rouge_l"}
COLD = {"webqa_open_full", "webqa_open_img", "hybridqa", "videorag_synth"}   # no supplied context
SUBSAMPLE = "1000"     # the 14-ordering runs (and their OK-VQA / WebQA-img / HybridQA / VideoRAG siblings): first 1,000
# SeaKR runs on the full test split (D5); only the order experiment is subsampled and the paper says so.
PAPER_EAB = eab_name(1.2, "ans1", "pos")
PAPER_FLAGS = ["--beta", "0.3", "--tau-bias", "0.4", "--tau", "0.35"]
PLACEHOLDER = {"eab_variant": PAPER_EAB, "gamma": 1.2, "eab_rows": "ans1", "eab_form": "pos", "beta": 0.3,
               "tau_bias": 0.4, "tau": 0.35, "u_combine": "sum", "attn_layer": "last", "okvqa_loader": "okvqa_gs"}
DECISIONS = ROOT / "runs" / "dev_decisions.json"


def load_decisions(model: str) -> dict | None:
    """The decisions file for this model, at any phase; None when missing or for another model."""
    if not DECISIONS.exists():
        return None
    d = json.loads(DECISIONS.read_text())
    return d if d.get("model") == model else None


def decisions(model: str) -> dict | None:
    """Frozen test-stage choices: only a complete phase-2 file counts (S-7)."""
    d = load_decisions(model)
    return d if d and d.get("phase") == 2 else None


def okvqa_candidate_files() -> dict[str, Path]:
    dense = os.environ.get("ERC_OKVQA_DENSE_CANDIDATES", paths.LOCAL_CACHE / "okvqa_dense" / "candidates_full.jsonl")
    return {"okvqa_dense": Path(dense), "okvqa_gs": paths.LOCAL_CACHE / "okvqa_gs112k" / "candidates_full.jsonl"}


def control_arm(rows: str, form: str, gamma: float) -> tuple[str, str]:
    """The decode-only control of the ablation: the frozen EAB with its first-answer row removed (ans2).
    If the frozen rows are already ans2 that control would be the same string as the main form; the control is then
    ans1 (EAB applied to the first answer row), and the fills text says so (M-38)."""
    control_rows = "ans1" if rows == "ans2" else "ans2"
    note = ("" if rows != "ans2" else "frozen form is ans2: the decode-only control is ans1 instead")
    return eab_name(gamma, control_rows, form), note


def stages(model: str, split: str = "test") -> dict[str, dict]:
    dec = decisions(model)
    frozen = dec is not None
    dec = dec or PLACEHOLDER
    p1 = load_decisions(model) or {}
    eab = dec["eab_variant"]
    gamma, rows, form = dec["gamma"], dec["eab_rows"], dec["eab_form"]
    decode_only, decode_note = control_arm(rows, form, gamma)
    okvqa = dec["okvqa_loader"]
    datasets = ["webqa_open_full", "webqa_open_img", okvqa, "hybridqa", "videorag_synth"]
    g = f"g{gamma:g}"
    ablation_variants = ",".join(dict.fromkeys([
        eab, decode_only,
        f"eab:{g}:{rows}:{form}:gate:all:retriever:ctx",      # EAB + context bias
        f"eab:{g}:{rows}:{form}:nogate",                       # always retrieve, relevance bias
        f"eab:{g}:{rows}:pos_nou:gate",                        # relevance bias without U scaling
        f"eab:{g}:{rows}:fixed:gate",                          # uniform (PASTA-style) bias, gated
        f"eab:{g}:{rows}:{form}:gate:all:random",              # random relevance, gated
    ]))
    frozen_flags = ["--beta", f"{dec['beta']:g}", "--tau-bias", f"{dec['tau_bias']:g}", "--tau", f"{dec['tau']:g}"]
    frozen_note = "" if frozen else " [NOT FROZEN: runs/dev_decisions.json missing, another model, or phase 1 only]"

    def method(d, u_combine=None, attn_layer=None, probe=True):
        flags = ["--u-combine", u_combine or dec["u_combine"], "--attn-layer", attn_layer or dec["attn_layer"]]
        if d in COLD and probe:
            flags += ["--probe-uncertainty", "--first-round-routing"]
        elif d in COLD:
            flags += ["--first-round-routing"]      # D17: the no-probe ablation keeps first-round routing
        return flags

    def run(d, tag, *extra, split_=None, hyper=None, **kw):
        return ["python", "-m", "omniaware.run", "--dataset", d, "--split", split_ or split, "--calib-source",
                CALIB[d], "--model", model, "--tag", tag, *method(d, **kw),
                *(frozen_flags if hyper is None else hyper), *extra]

    def score(d, tag):
        return ["python", "-m", "omniaware.eval.score_official", f"runs/{tag}/{d}.jsonl", "--bart"]

    def analyse(kind, path, *extra):
        return ["python", "-m", "scripts.analysis.paper_analyses", kind, path, *extra]

    s = {}
    candidates = okvqa_candidate_files()
    dev_missing = [str(p) for p in candidates.values() if not p.exists()]

    # ---- dev: phase 1 (base grid, U settings), phase 2 (beta x tau_b) -------------------------------------------
    dev = []
    for d in ["webqa_open_full", *OKVQA_LOADERS, "hybridqa", "videorag_synth"]:
        tag = f"v321dev_base_{model}"
        dev += [run(d, tag, "--variants", ",".join(EAB_CANDIDATES), split_="dev", hyper=PAPER_FLAGS,
                    u_combine="sum", attn_layer="last"), score(d, tag)]
    for setting, kw in (("zsum", {"u_combine": "zsum", "attn_layer": "last"}),
                        ("mid", {"u_combine": "sum", "attn_layer": "mid"})):
        for d in ["webqa_open_full", "okvqa_gs", "hybridqa", "videorag_synth"]:
            tag = f"v321dev_{setting}_{model}"
            dev += [run(d, tag, "--variants", "", split_="dev", hyper=PAPER_FLAGS, **kw), score(d, tag)]
    for d in DEV_DATASETS:      # pre-registration check of the design (docs/PROTOCOL.md): MDE and best-of-K noise
        dev.append(["python", "-m", "scripts.analysis.power_design", f"runs/v321dev_base_{model}/{d}.official_scores.jsonl",
                    "--baseline", "static", "--candidate", PAPER_EAB, "--metric", METRIC[d],
                    "--output", f"runs/dev_power/{model}_{d}.json"])
    dev.append(["python", "-m", "scripts.analysis.dev_select", "--phase", "1", "--model", model])
    s["dev"] = {"fills": "nothing in the paper: phase-1 choices (EAB form x gamma, U setting, OK-VQA corpus) -> "
                         "runs/dev_decisions.json",
                "commands": dev, "requires": None, "missing": dev_missing}

    dev_beta = []
    p1_ok = p1.get("okvqa_loader", okvqa)
    if p1:
        cells_datasets = ["webqa_open_full", p1_ok, "hybridqa", "videorag_synth"]
        for beta in BETAS:
            for tb in TAU_BIASES:
                tag = f"v321dev_beta{beta:g}_tb{tb:g}_{model}"
                hyper = ["--beta", f"{beta:g}", "--tau-bias", f"{tb:g}", "--tau", "0.35"]   # tau is offline (dev_select)
                for d in cells_datasets:
                    dev_beta += [run(d, tag, "--variants", p1["eab_variant"], split_="dev", hyper=hyper,
                                     u_combine=p1["u_combine"], attn_layer=p1["attn_layer"]), score(d, tag)]
        dev_beta.append(["python", "-m", "scripts.analysis.dev_select", "--phase", "2", "--model", model])
    s["dev_beta"] = {"fills": "nothing in the paper: phase-2 choices (beta, tau_b; tau offline) -> runs/dev_decisions.json"
                              + ("" if p1 else " [needs the phase-1 file first]"),
                     "commands": dev_beta, "requires": "phase1" if not p1 else None, "missing": []}

    # ---- test stages (read the frozen choices) -----------------------------------------------------------------
    main = []
    if model == "internvl3.5-8b":     # backbone transfer: static vs the controller, with its own calibration
        for d in datasets:
            main += [run(d, f"v321_main_{model}", "--variants", eab, "--replication-gamma", "1.2"),
                     score(d, f"v321_main_{model}")]
        transfer_order = []
        from omniaware.order import permutation_schedule
        for o in permutation_schedule():
            tag = f"v321_perm_{o['order_id']}_{model}"
            transfer_order += [run(okvqa, tag, "--variants", eab, "--replication-gamma", "1.2", "--order",
                                   f"perm:{o['order_id']}", "--n-eval", SUBSAMPLE), score(okvqa, tag)]
        transfer_order.append(analyse("order", f"runs/v321_perm_*_{model}/{okvqa}.jsonl", "--metric", METRIC[okvqa],
                                      "--arms", "static", "replication", eab))
        return {"main": {"fills": "appendix backbone transfer (InternVL3.5-8B)" + frozen_note, "commands": main,
                         "requires": "frozen", "missing": []},
                "order": {"fills": "appendix: OK-VQA order deviations of InternVL3.5-8B" + frozen_note,
                          "commands": transfer_order, "requires": "frozen", "missing": []}}
    for d in datasets:
        extra = ["--variants", eab, "--replication-gamma", "1.2"]
        if model != "internvl3.5-8b":
            extra.append("--fitm")
        if d != "videorag_synth":
            extra.append("--m2rag")
        main += [run(d, f"v321_main_{model}", *extra), score(d, f"v321_main_{model}"),
                 run(d, f"v321_ce_{model}", "--selector", "ce", "--variants", eab), score(d, f"v321_ce_{model}"),
                 # SeaKR on the full test split (D5): no --n-eval
                 run(d, f"v321_seakr_{model}", "--variants", eab, "--seakr-samples", "20"),
                 score(d, f"v321_seakr_{model}")]
    s["main"] = {"fills": "Table main (UniversalRAG in its own stage; SeaKR on the full test split); "
                          "backbone transfer with --model internvl3.5-8b" + frozen_note,
                 "commands": main, "requires": "frozen", "missing": []}

    ur = []
    for d in ("webqa_open_full", "hybridqa", "videorag_synth"):
        tag = f"v321_universalrag_{model}"
        ur += [["python", "-m", "omniaware.baselines.universalrag", "export", "--dataset", d,
                "--output", f"runs/{tag}/{d}.queries.json"],
               # G3's router (scripts/universalrag_retrieval/route_queries.py): official UniversalRAG router, fixed checkpoint
               ["python", "-m", "scripts.universalrag_retrieval.route_queries", "--queries",
                f"runs/{tag}/{d}.queries.json", "--model-path", str(paths.UNIVERSALRAG_ROUTER),
                "--output", f"runs/{tag}/{d}.routes.json"],
               run(d, tag, "--variants", eab, "--route-rule", "query", "--query-routes", f"runs/{tag}/{d}.routes.json"),
               score(d, tag)]
    s["universalrag"] = {"fills": "Table main, UniversalRAG row (arm 'universalrag'); WebQA-img column = the image "
                                  "questions of webqa_open_full (analysis 'column'); OK-VQA column: none (the router "
                                  "reads text queries only)" + frozen_note,
                         "commands": ur, "requires": "frozen", "missing": []}

    abl = []
    for d in (okvqa, "webqa_open_img"):
        tag = f"v321_ablation_{model}"
        abl += [run(d, tag, "--variants", ablation_variants, "--replication-gamma", "1.2"), score(d, tag)]
        for flags, name in ((["--relevance-norm", "raw"], "raw"), (["--relevance-norm", "minmax"], "minmax"),
                            (["--no-calibration"], "nocal")):      # D16: nocal changes only the U statistics
            abl += [run(d, f"v321_ablation_{name}_{model}", "--variants", eab, *flags),
                    score(d, f"v321_ablation_{name}_{model}")]
        # Calibration size: the default is the whole reserve (OK-VQA held-out 500, WebQA training pool 500 of a
        # 1,000 reserve), so OK-VQA adds 100 and WebQA-img adds 100 and 1,000; dev and test do not move (D1, D2).
        for n in ((100,) if d == okvqa else (100, 1000)):
            abl += [run(d, f"v321_calib{n}_{model}", "--variants", eab, "--n-calib", str(n)),
                    score(d, f"v321_calib{n}_{model}")]
        # SENSITIVITY ONLY (not selected, not in dev_select): the beta x tau_b grid is reported on test
        for beta in BETAS:
            for tb in TAU_BIASES:
                t = f"v321_beta{beta:g}_tb{tb:g}_{model}"
                hyper = ["--beta", f"{beta:g}", "--tau-bias", f"{tb:g}", "--tau", f"{dec['tau']:g}"]
                abl += [run(d, t, "--variants", eab, hyper=hyper), score(d, t)]
    for d in ("webqa_open_img", "hybridqa"):
        t = f"v321_ablation_noprobe_{model}"          # cold start without the probe round: U = Norm(beta H_pred)
        abl += [run(d, t, "--variants", eab, probe=False), score(d, t)]
    # Calibration transfer only between cold-start datasets: their gate reads the probe-round statistic, which
    # OK-VQA (supplied context, no probe round) does not have, so --norm-from refuses an OK-VQA source.
    for d in ("videorag_synth", "webqa_open_img"):
        t = f"v321_transfer_hybridqa_{model}"
        abl += [run(d, t, "--variants", eab, "--norm-from", f"runs/v321_main_{model}/hybridqa_config.json"),
                score(d, t)]
    s["ablation"] = {"fills": "Table uomni_ablation (incl. decode-only EAB" + (f" [{decode_note}]" if decode_note else "")
                              + ", no probe round); appendix calibration size and transfer, score normalization; "
                              "beta x tau_b grid = SENSITIVITY ONLY, reported and never selected" + frozen_note,
                     "commands": abl, "requires": "frozen", "missing": []}

    route = []
    for rule in ["all", "retriever", "query", "mass", "entropy", "mh", "oracle"]:
        tag = f"v321_route_{rule}_{model}"
        extra = ["--variants", eab, "--route-rule", rule]
        if rule == "query":
            extra += ["--query-routes", f"runs/v321_universalrag_{model}/webqa_open_full.routes.json"]
        route += [run("webqa_open_full", tag, *extra), score("webqa_open_full", tag),
                  ["python", "-m", "scripts.analysis.routing_metrics", f"runs/{tag}/webqa_open_full.jsonl",
                   "--arm", f"gated:{eab}", "--metric", "acc_approx"]]
    s["routing"] = {"fills": "Table routing (WebQA, open retrieval)" + frozen_note, "commands": route,
                    "requires": "frozen", "missing": []}

    from omniaware.order import permutation_schedule
    order = []
    for d in (okvqa, "webqa_open_img", "hybridqa", "videorag_synth"):
        for o in permutation_schedule():
            tag = f"v321_perm_{o['order_id']}_{model}"
            extra = ["--variants", f"{eab},{decode_only}", "--replication-gamma", "1.2", "--order",
                     f"perm:{o['order_id']}", "--n-eval", SUBSAMPLE]
            if model != "internvl3.5-8b":
                extra.append("--fitm")
            order += [run(d, tag, *extra), score(d, tag)]
        arms = ["static", "replication", eab, decode_only] + (["fitm"] if model != "internvl3.5-8b" else [])
        order += [analyse("order", f"runs/v321_perm_*_{model}/{d}.jsonl", "--metric", METRIC[d], "--arms", *arms),
                  analyse("ard", f"runs/v321_perm_sorted_{model}/{d}.jsonl")]
    s["order"] = {"fills": "Table order and appendix order_full (Std., gap, flip, ARD); first 1,000 test questions per "
                           "dataset (SUBSAMPLE, disclosed)" + frozen_note, "commands": order,
                  "requires": "frozen", "missing": []}

    cite = []
    for d in (okvqa, "webqa_open_img", "hybridqa"):
        tag = f"v321_cite_{model}"
        cite += [run(d, tag, "--variants", ablation_variants, "--replication-gamma", "1.2", "--answer-style", "cite",
                     "--max-new-tokens", "128"),
                 ["python", "-m", "omniaware.eval.citation.attribution", f"runs/{tag}/{d}.jsonl", "--device", "cuda"]]
    s["citation"] = {"fills": "AP columns of Table uomni_ablation; appendix citation support" + frozen_note,
                     "commands": cite, "requires": "frozen", "missing": []}

    iv = []
    tag = f"v321_interventions_{model}"
    iv += [run("webqa_open_img", tag, "--variants", eab, "--replication-gamma", "1.2", "--answer-style", "cite",
               "--max-new-tokens", "128", "--intervention", "mask", "cf", "random_edit", "rerun", "distractors",
               "inject_answer", "inject_instruction"),
           score("webqa_open_img", tag),
           analyse("interventions", f"runs/{tag}/webqa_open_img.jsonl"),
           ["python", "-m", "omniaware.eval.citation.attribution", f"runs/{tag}/webqa_open_img.jsonl",
            "--intervention", "distractors", "--device", "cuda"],
           analyse("gold25", f"runs/v321_main_{model}/webqa_open_img.jsonl", "--metric", "acc_approx")]
    for sigma in ("0.1", "0.2"):
        for draw in range(3):
            t = f"v321_noise{sigma}_d{draw}_{model}"
            iv += [run("webqa_open_img", t, "--variants", eab, "--replication-gamma", "1.2", "--relevance-noise",
                       sigma, "--noise-draw", str(draw)), score("webqa_open_img", t)]
    iv.append(analyse("noise", f"runs/v321_noise*_{model}/webqa_open_img.jsonl", "--metric", "acc_approx",
                      "--arms", "static", "replication", eab))
    s["interventions"] = {"fills": "Table causal (interventions, Gold@2-5, noise, Misled)" + frozen_note, "commands": iv,
                          "requires": "frozen", "missing": []}

    an = []
    for d in datasets:
        path = f"runs/v321_main_{model}/{d}.jsonl"
        seakr = f"runs/v321_seakr_{model}/{d}.jsonl"
        ce = f"runs/v321_ce_{model}/{d}.jsonl"
        ur_path = f"runs/v321_universalrag_{model}/webqa_open_full.jsonl" if d.startswith("webqa_open") else (
            f"runs/v321_universalrag_{model}/{d}.jsonl")
        an += [analyse("uncertainty", path, "--metric", METRIC[d], "--arms", "direct"),
               analyse("uncertainty", seakr, "--metric", METRIC[d], "--arms", "direct"),
               analyse("latency", path), analyse("latency", seakr),
               analyse("latency", ce),                                    # CE and Omni+CE (gated:<EAB> of the CE run)
               analyse("stats", path, "--arms", "static", eab),
               analyse("errors", path, "--metric", METRIC[d], "--arms", f"gated:{eab}"),
               # F11: U and NTE from the main run, SeaKR from the SeaKR run (each system from its own run)
               analyse("selective", path, "--seakr-run", seakr, "--metric", METRIC[d], "--arms", eab),
               analyse("agreement", path, "--metric", METRIC[d], "--arms", f"gated:{eab}"),
               analyse("transitions", path, "--metric", METRIC[d], "--arms", "gated:static", f"gated:{eab}"),
               analyse("stats", ce, "--arms", "static"),       # reranker recall
               analyse("ranges", f"cache/relevance_{d}*.json"),
               analyse("tau", path, "--metric", METRIC[d], "--arms", eab, "--taus", *map(str, TAUS))]
        # significance of the controller against every baseline of this column; strongest + Holm (M-10, D18)
        baselines = [f"{path}:static", f"{seakr}:seakr", f"{ce}:static"]
        if d != "videorag_synth":
            baselines.append(f"{path}:m2rag")
        if d in ("webqa_open_full", "webqa_open_img", "hybridqa", "videorag_synth"):
            baselines.append(f"{ur_path}:universalrag")
            if d != "webqa_open_img":                               # same run as webqa_open_full: reported once
                an.append(analyse("latency", ur_path))              # UniversalRAG row (router time reported apart)
        restrict = ["--restrict-modality", "image"] if d == "webqa_open_img" else []   # image questions of the full run
        an.append(analyse("column", path, "--metric", METRIC[d], "--arms", f"gated:{eab}",
                          *[x for b in baselines for x in ("--compare", b)], *restrict))
    for name in ("raw", "minmax"):
        an.append(analyse("ard", f"runs/v321_ablation_{name}_{model}/webqa_open_img.jsonl"))
    s["analysis"] = {"fills": "selective risk, latency, dataset statistics, first-token shares, error analysis, column "
                              "significance (strongest baseline, Holm); tau = SENSITIVITY ONLY, offline from the main run, "
                              "never selected" + frozen_note,
                     "commands": an, "requires": "frozen", "missing": []}
    return s


def expand_globs(cmd: list[str]) -> list[str]:
    """Shell-free globbing: a pattern argument becomes its sorted matches; no match is an error (M-45)."""
    out = []
    for tok in cmd:
        if any(c in tok for c in "*?"):
            hits = sorted(glob.glob(tok))
            if not hits:
                raise SystemExit(f"no file matches {tok}")
            out.extend(hits)
        else:
            out.append(tok)
    return out


def check_stage(name: str, st: dict, model: str) -> None:
    if st["missing"]:
        raise SystemExit(f"stage {name}: missing OK-VQA candidate corpus, refusing to skip silently (S-12): "
                         + ", ".join(st["missing"]))
    if st["requires"] == "frozen" and decisions(model) is None:
        raise SystemExit(f"stage {name} needs the frozen test choices: {DECISIONS} is missing, for another model, or not "
                         f"complete (phase 2). Run the dev stage and scripts.analysis.dev_select --phase 1 and --phase 2.")
    if st["requires"] == "phase1" and load_decisions(model) is None:
        raise SystemExit(f"stage {name} needs the phase-1 choices in {DECISIONS}; run dev_select --phase 1 first")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--model", default="8b", choices=["8b", "internvl3.5-8b"])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--gpus", default=None,
                    help="run every omniaware.run command through scripts.launch on these GPUs ('auto' or '0,1,...')")
    ap.add_argument("--procs-per-gpu", type=int, default=1)
    args = ap.parse_args(argv)
    all_stages = stages(args.model, args.split)
    if args.list or not args.stage:
        dec = decisions(args.model)
        print(f"frozen choices: {dec if dec else '未冻结 (runs/dev_decisions.json missing, another model, or phase 1 only)'}")
        for name, st in all_stages.items():
            status = " [missing corpora]" if st["missing"] else ""
            print(f"{name:14s} {len(st['commands']):4d} commands  -> {st['fills']}{status}")
        return
    if args.stage not in all_stages:
        raise SystemExit(f"unknown stage {args.stage}; choose from {', '.join(all_stages)}")
    st = all_stages[args.stage]
    if args.run:                    # --print only shows the commands (with 未冻结 / missing-corpus status)
        check_stage(args.stage, st, args.model)
    for cmd in st["commands"]:
        if args.gpus and cmd[:3] == ["python", "-m", "omniaware.run"]:
            cmd = ["python", "-m", "scripts.launch", "--gpus", args.gpus, "--procs-per-gpu", str(args.procs_per_gpu),
                   "--", *cmd[3:]]
        if args.print or not args.run:          # patterns are shown as written; they are expanded only to run
            print(" ".join(shlex.quote(c) for c in cmd))
        if args.run:
            subprocess.run(expand_globs(cmd), check=True)


if __name__ == "__main__":
    main()

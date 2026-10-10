"""Every run, scoring and analysis command behind paper V3.21, in one place.

    python -m scripts.paper_runs --list                        # stages and what each one fills
    python -m scripts.paper_runs --stage dev --run             # 1. pre-registered dev grid (--split dev)
    python -m scripts.analysis.dev_select                      # 2. freeze choices -> runs/dev_decisions.json
    python -m scripts.paper_runs --stage main --run            # 3. test stages read the frozen choices
    python -m scripts.paper_runs --stage main --run --gpus auto # every run sharded over all visible GPUs

The dev stage compares the six EAB forms, three uncertainty settings and three OK-VQA retrieval corpora;
docs/DEV_DECISIONS.md states the decision rules before any result is seen. Test stages run once with the
frozen settings. docs/PAPER_TABLE_MAP.md maps each table cell to a run tag and an arm.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIB = {"webqa_open_full": "train", "webqa_open_img": "train", "okvqa": "heldout", "okvqa_dense": "heldout",
         "okvqa_gs": "heldout", "hybridqa": "heldout", "videorag_synth": "heldout"}
METRIC = {"webqa_open_full": "acc_approx", "webqa_open_img": "acc_approx", "okvqa": "vqa_accuracy",
          "okvqa_dense": "vqa_accuracy", "okvqa_gs": "vqa_accuracy", "hybridqa": "em", "videorag_synth": "rouge_l"}
COLD = {"webqa_open_full", "webqa_open_img", "hybridqa", "videorag_synth"}   # no supplied context
EAB_FORMS = [f"eab:g1.2:{rows}:{form}:gate" for rows in ("ans1", "ans2", "q") for form in ("pos", "ctr")]
SUBSAMPLE = "1000"     # SeaKR and the 14-ordering runs: first 1,000 test questions of each dataset
DEFAULTS = {"eab_variant": "eab:g1.2:ans1:pos:gate", "u_combine": "sum", "attn_layer": "last",
            "okvqa_loader": "okvqa_gs"}


def decisions(model: str) -> dict:
    path = ROOT / "runs" / "dev_decisions.json"
    if path.exists():
        d = json.loads(path.read_text())
        if d.get("model") == model:
            return {**DEFAULTS, **{k: d[k] for k in DEFAULTS}}
    return dict(DEFAULTS)


def stages(model: str, split: str) -> dict[str, dict]:
    dec = decisions(model)
    eab = dec["eab_variant"]
    decode_only = eab.replace(":ans1:", ":ans2:").replace(":q:", ":ans2:")
    okvqa = dec["okvqa_loader"]
    datasets = ["webqa_open_full", "webqa_open_img", okvqa, "hybridqa", "videorag_synth"]
    _, gamma, rows, form = eab.split(":")[:4]
    ablation_variants = ",".join(dict.fromkeys([
        eab, decode_only,
        f"eab:{gamma}:{rows}:{form}:gate:all:retriever:ctx",     # EAB + context bias
        f"eab:{gamma}:{rows}:{form}:nogate",                      # always retrieve, relevance bias
        f"eab:{gamma}:{rows}:pos_nou:gate",                       # relevance bias without U scaling
        f"eab:{gamma}:{rows}:fixed:gate",                         # uniform (PASTA-style) bias, gated
        f"eab:{gamma}:{rows}:{form}:gate:all:random",             # random relevance, gated
    ]))

    def method(d, u_combine=None, attn_layer=None, probe=True):
        flags = ["--u-combine", u_combine or dec["u_combine"], "--attn-layer", attn_layer or dec["attn_layer"]]
        if d in COLD and probe:
            flags += ["--probe-uncertainty", "--first-round-routing"]
        return flags

    def run(d, tag, *extra, split_=None, **kw):
        return ["python", "-m", "omniaware.run", "--dataset", d, "--split", split_ or split, "--calib-source",
                CALIB[d], "--model", model, "--tag", tag, *method(d, **kw), *extra]

    def score(d, tag):
        return ["python", "-m", "omniaware.eval.score_official", f"runs/{tag}/{d}.jsonl", "--bart"]

    def analyse(kind, path, *extra):
        return ["python", "-m", "scripts.analysis.paper_analyses", kind, path, *extra]

    s = {}
    dev = []
    sys.path.insert(0, str(ROOT / "src"))
    from omniaware import paths
    optional = {"okvqa": paths.OKVQA_CANDIDATES, "okvqa_dense": paths.LOCAL_CACHE / "okvqa_dense" / "candidates_full.jsonl"}
    skipped = [d for d, f in optional.items() if not Path(f).exists()]
    for d in [d for d in ["webqa_open_full", "okvqa", "okvqa_dense", "okvqa_gs", "hybridqa", "videorag_synth"]
              if d not in skipped]:
        tag = f"v321dev_base_{model}"
        dev += [run(d, tag, "--variants", ",".join(EAB_FORMS), split_="dev", u_combine="sum", attn_layer="last"),
                score(d, tag)]
    for setting, kw in (("zsum", {"u_combine": "zsum", "attn_layer": "last"}),
                        ("mid", {"u_combine": "sum", "attn_layer": "mid"})):
        for d in ["webqa_open_full", "okvqa_gs", "hybridqa", "videorag_synth"]:
            tag = f"v321dev_{setting}_{model}"
            dev += [run(d, tag, "--variants", "", split_="dev", **kw), score(d, tag)]
    dev.append(["python", "-m", "scripts.analysis.dev_select", "--model", model])
    s["dev"] = {"fills": "nothing in the paper: pre-registered choices -> runs/dev_decisions.json"
                         + (f" (skipped, candidates not built: {', '.join(skipped)})" if skipped else ""),
                "commands": dev}

    main = []
    if model == "internvl3.5-8b":     # backbone transfer: static vs the controller, with its own calibration
        for d in datasets:
            main += [run(d, f"v321_main_{model}", "--variants", eab, "--replication-gamma", "1.2"),
                     score(d, f"v321_main_{model}")]
        transfer_order = []
        sys.path.insert(0, str(ROOT / "src"))
        from omniaware.order import permutation_schedule
        for o in permutation_schedule():
            tag = f"v321_perm_{o['order_id']}_{model}"
            transfer_order += [run(okvqa, tag, "--variants", eab, "--replication-gamma", "1.2", "--order",
                                   f"perm:{o['order_id']}", "--n-eval", SUBSAMPLE), score(okvqa, tag)]
        transfer_order.append(analyse("order", f"runs/v321_perm_*_{model}/{okvqa}.jsonl", "--metric", METRIC[okvqa],
                                      "--arms", "static", "replication", eab))
        return {"main": {"fills": "appendix backbone transfer (InternVL3.5-8B)", "commands": main},
                "order": {"fills": "appendix: OK-VQA order deviations of InternVL3.5-8B", "commands": transfer_order}}
    for d in datasets:
        extra = ["--variants", eab, "--replication-gamma", "1.2"]
        if model != "internvl3.5-8b":
            extra.append("--fitm")
        if d != "videorag_synth":
            extra.append("--m2rag")
        main += [run(d, f"v321_main_{model}", *extra), score(d, f"v321_main_{model}"),
                 run(d, f"v321_ce_{model}", "--selector", "ce", "--variants", eab), score(d, f"v321_ce_{model}"),
                 run(d, f"v321_seakr_{model}", "--variants", eab, "--seakr-samples", "20", "--n-eval", SUBSAMPLE),
                 score(d, f"v321_seakr_{model}")]
    s["main"] = {"fills": "Table main (UniversalRAG in its own stage; SeaKR on 1,000 questions per dataset); "
                          "backbone transfer with --model internvl3.5-8b", "commands": main}

    ur = []
    for d in ("webqa_open_full", "hybridqa", "videorag_synth"):
        tag = f"v321_universalrag_{model}"
        ur += [["python", "-m", "omniaware.baselines.universalrag", "export", "--dataset", d,
                "--output", f"runs/{tag}/{d}.queries.json"],
               ["echo", f"route runs/{tag}/{d}.queries.json with the official UniversalRAG router "
                        f"(see baselines/universalrag.py) into runs/{tag}/{d}.routes.json"],
               run(d, tag, "--variants", eab, "--route-rule", "query", "--query-routes", f"runs/{tag}/{d}.routes.json"),
               score(d, tag)]
    s["universalrag"] = {"fills": "Table main, UniversalRAG row (arm 'universalrag')", "commands": ur}

    abl = []
    for d in (okvqa, "webqa_open_img"):
        tag = f"v321_ablation_{model}"
        abl += [run(d, tag, "--variants", ablation_variants, "--replication-gamma", "1.2"), score(d, tag)]
        for flag, name in ((["--relevance-norm", "raw"], "raw"), (["--relevance-norm", "minmax"], "minmax"),
                           (["--no-calibration"], "nocal")):
            abl += [run(d, f"v321_ablation_{name}_{model}", "--variants", eab, *flag),
                    score(d, f"v321_ablation_{name}_{model}")]
        for n in (100, 1000):
            abl += [run(d, f"v321_calib{n}_{model}", "--variants", eab, "--n-calib", str(n)),
                    score(d, f"v321_calib{n}_{model}")]
        for beta in ("0", "0.15", "0.3", "0.5", "1.0"):
            for tb in ("0.2", "0.4", "0.6"):
                t = f"v321_beta{beta}_tb{tb}_{model}"
                abl += [run(d, t, "--variants", eab, "--beta", beta, "--tau-bias", tb), score(d, t)]
    for d in ("webqa_open_img", "hybridqa"):
        t = f"v321_ablation_noprobe_{model}"          # cold start without the probe round: U = Norm(beta H_pred)
        abl += [run(d, t, "--variants", eab, probe=False), score(d, t)]
    for d in ("hybridqa", "videorag_synth", "webqa_open_img"):
        t = f"v321_transfer_okvqa_{model}"
        abl += [run(d, t, "--variants", eab, "--norm-from", f"runs/v321_main_{model}/{okvqa}_config.json"),
                score(d, t)]
    s["ablation"] = {"fills": "Table uomni_ablation (incl. decode-only EAB, no probe round); appendix beta/tau_bias, "
                              "calibration size and transfer, score normalization", "commands": abl}

    route = []
    for rule in ["all", "retriever", "query", "mass", "entropy", "mh", "oracle"]:
        tag = f"v321_route_{rule}_{model}"
        extra = ["--variants", eab, "--route-rule", rule]
        if rule == "query":
            extra += ["--query-routes", f"runs/v321_universalrag_{model}/webqa_open_full.routes.json"]
        route += [run("webqa_open_full", tag, *extra), score("webqa_open_full", tag),
                  ["python", "-m", "scripts.analysis.routing_metrics", f"runs/{tag}/webqa_open_full.jsonl",
                   "--arm", f"gated:{eab}", "--metric", "acc_approx"]]
    s["routing"] = {"fills": "Table routing (WebQA, open retrieval)", "commands": route}

    sys.path.insert(0, str(ROOT / "src"))
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
    s["order"] = {"fills": "Table order and appendix order_full (Std., gap, flip, ARD); 1,000 questions per dataset",
                  "commands": order}

    cite = []
    for d in (okvqa, "webqa_open_img", "hybridqa"):
        tag = f"v321_cite_{model}"
        cite += [run(d, tag, "--variants", ablation_variants, "--replication-gamma", "1.2", "--answer-style", "cite",
                     "--max-new-tokens", "128"),
                 ["python", "-m", "omniaware.eval.citation.attribution", f"runs/{tag}/{d}.jsonl", "--device", "cuda"]]
    s["citation"] = {"fills": "AP columns of Table uomni_ablation; appendix citation support", "commands": cite}

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
    s["interventions"] = {"fills": "Table causal (interventions, Gold@2-5, noise, Misled)", "commands": iv}

    an = []
    for d in datasets:
        path = f"runs/v321_main_{model}/{d}.jsonl"
        an += [analyse("uncertainty", path, "--metric", METRIC[d], "--arms", "direct"),
               analyse("uncertainty", f"runs/v321_seakr_{model}/{d}.jsonl", "--metric", METRIC[d], "--arms", "direct"),
               analyse("latency", path), analyse("latency", f"runs/v321_seakr_{model}/{d}.jsonl"),
               analyse("stats", path, "--arms", "static", eab),
               analyse("errors", path, "--metric", METRIC[d], "--arms", f"gated:{eab}"),
               analyse("selective", f"runs/v321_seakr_{model}/{d}.jsonl", path, "--metric", METRIC[d], "--arms", eab),
               analyse("agreement", path, "--metric", METRIC[d], "--arms", f"gated:{eab}"),
               analyse("transitions", path, "--metric", METRIC[d], "--arms", "gated:static", f"gated:{eab}"),
               analyse("stats", f"runs/v321_ce_{model}/{d}.jsonl", "--arms", "static"),       # reranker recall
               analyse("ranges", f"cache/relevance_{d}*.json"),
               # significance of the controller against baselines held in other runs (table main daggers)
               analyse("paired", path, path, "--metric", METRIC[d], "--arms", "m2rag" if d != "videorag_synth" else "static",
                       f"gated:{eab}"),
               analyse("paired", f"runs/v321_seakr_{model}/{d}.jsonl", path, "--metric", METRIC[d],
                       "--arms", "seakr", f"gated:{eab}"),
               analyse("paired", f"runs/v321_ce_{model}/{d}.jsonl", path, "--metric", METRIC[d],
                       "--arms", "static", f"gated:{eab}")]
        if d in ("webqa_open_full", "hybridqa", "videorag_synth"):
            an.append(analyse("paired", f"runs/v321_universalrag_{model}/{d}.jsonl", path, "--metric", METRIC[d],
                              "--arms", "universalrag", f"gated:{eab}"))
    for name in ("raw", "minmax"):
        an.append(analyse("ard", f"runs/v321_ablation_{name}_{model}/webqa_open_img.jsonl"))
    s["analysis"] = {"fills": "selective risk, latency, dataset statistics, first-token shares, error analysis",
                     "commands": an}
    return s


def main():
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
    args = ap.parse_args()
    all_stages = stages(args.model, args.split)
    if args.list or not args.stage:
        print(f"frozen choices: {decisions(args.model)}")
        for name, st in all_stages.items():
            print(f"{name:14s} {len(st['commands']):4d} commands  -> {st['fills']}")
        return
    for cmd in all_stages[args.stage]["commands"]:
        if args.gpus and cmd[:3] == ["python", "-m", "omniaware.run"]:
            cmd = ["python", "-m", "scripts.launch", "--gpus", args.gpus, "--procs-per-gpu", str(args.procs_per_gpu),
                   "--", *cmd[3:]]
        line = " ".join(shlex.quote(c) for c in cmd)
        if args.print or not args.run:
            print(line)
        if args.run:
            subprocess.run(line if "*" in line else cmd, shell="*" in line, check=True)


if __name__ == "__main__":
    main()

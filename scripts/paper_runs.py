"""Every run, scoring and analysis command behind the tables of paper V3.21, in one place.

    python -m scripts.paper_runs --list                       # stages and what each one fills
    python -m scripts.paper_runs --stage main --print         # commands of one stage
    python -m scripts.paper_runs --stage main --run [--model 8b] [--split test]

Each command writes runs/<tag>/<dataset>.jsonl; docs/PAPER_TABLE_MAP.md says which arm of which run fills
each table cell. Settings are frozen here; tuning happens on --split dev with other tags (docs/PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

DATASETS = {   # loader -> (calibration source, official metric used by the analyses)
    "webqa_open_full": ("train", "acc_approx"),
    "webqa_open_img": ("train", "acc_approx"),
    "okvqa": ("heldout", "vqa_accuracy"),
    "hybridqa": ("heldout", "em"),
    "videorag_synth": ("heldout", "rouge_l"),
}
EAB = "eab:g1.2:ans2:pos:gate"
EAB_P = "eab:g1.2:ans1:pos:gate"
ABLATION_VARIANTS = ",".join([
    EAB, EAB_P,
    "eab:g1.2:ans2:pos:gate:all:retriever:ctx",       # EAB + context bias
    "eab:g1.2:ans2:pos:nogate",                       # always retrieve, relevance bias (ungated arm)
    "eab:g1.2:ans2:pos_nou:gate",                     # relevance bias without U scaling
    "eab:g1.2:ans2:fixed:gate",                       # uniform (PASTA-style) bias, gated
    "eab:g1.2:ans2:pos:gate:all:random",              # random relevance, gated
])
ROUTING_RULES = ["all", "retriever", "query", "mass", "entropy", "mh", "oracle"]


def run_cmd(dataset, tag, *extra, model="8b", split="test"):
    calib, _ = DATASETS[dataset]
    return ["python", "-m", "omniaware.run", "--dataset", dataset, "--split", split, "--calib-source", calib,
            "--model", model, "--tag", tag, *extra]


def score_cmd(dataset, tag, bart=True):
    return ["python", "-m", "omniaware.eval.score_official", f"runs/{tag}/{dataset}.jsonl", *(["--bart"] if bart else [])]


def stages(model: str, split: str) -> dict[str, dict]:
    s = {}
    main = []
    for d in DATASETS:
        extra = ["--variants", EAB, "--seakr-samples", "20", "--replication-gamma", "1.2"]
        if model != "internvl3.5-8b":
            extra.append("--fitm")
        if d != "videorag_synth":
            extra.append("--m2rag")
        tag = f"v321_main_{model}"
        main += [run_cmd(d, tag, *extra, model=model, split=split), score_cmd(d, tag)]
        tag_ce = f"v321_ce_{model}"
        main += [run_cmd(d, tag_ce, "--selector", "ce", "--variants", EAB, model=model, split=split),
                 score_cmd(d, tag_ce)]
    s["main"] = {"fills": "Table main (all rows but UniversalRAG); backbone transfer with --model internvl3.5-8b",
                 "commands": main}
    ur = []
    for d in ("webqa_open_full", "hybridqa", "videorag_synth"):
        tag = f"v321_universalrag_{model}"
        ur += [["python", "-m", "omniaware.baselines.universalrag", "export", "--dataset", d,
                "--output", f"runs/{tag}/{d}.queries.json"],
               ["echo", f"route runs/{tag}/{d}.queries.json with the official UniversalRAG router "
                        f"(see baselines/universalrag.py) into runs/{tag}/{d}.routes.json"],
               run_cmd(d, tag, "--variants", EAB, "--route-rule", "query", "--query-routes",
                       f"runs/{tag}/{d}.routes.json", model=model, split=split),
               score_cmd(d, tag)]
    s["universalrag"] = {"fills": "Table main, UniversalRAG row (arm 'universalrag')", "commands": ur}
    abl = []
    for d in ("okvqa", "webqa_open_img"):
        tag = f"v321_ablation_{model}"
        abl += [run_cmd(d, tag, "--variants", ABLATION_VARIANTS, "--replication-gamma", "1.2", model=model,
                        split=split), score_cmd(d, tag)]
        for flag, name in ((["--relevance-norm", "raw"], "raw"), (["--relevance-norm", "minmax"], "minmax"),
                           (["--no-calibration"], "nocal")):
            abl += [run_cmd(d, f"v321_ablation_{name}_{model}", "--variants", EAB, *flag, model=model, split=split),
                    score_cmd(d, f"v321_ablation_{name}_{model}")]
        for n in (100, 1000):
            abl += [run_cmd(d, f"v321_calib{n}_{model}", "--variants", EAB, "--n-calib", str(n), model=model,
                            split=split), score_cmd(d, f"v321_calib{n}_{model}")]
        for beta in ("0", "0.15", "0.5", "1.0"):
            for tb in ("0.2", "0.4", "0.6"):
                t = f"v321_beta{beta}_tb{tb}_{model}"
                abl += [run_cmd(d, t, "--variants", EAB, "--beta", beta, "--tau-bias", tb, model=model,
                                split=split), score_cmd(d, t)]
    for d in ("hybridqa", "videorag_synth", "webqa_open_img"):
        t = f"v321_transfer_okvqa_{model}"
        abl += [run_cmd(d, t, "--variants", EAB, "--norm-from", f"runs/v321_main_{model}/okvqa_config.json",
                        model=model, split=split), score_cmd(d, t)]
    s["ablation"] = {"fills": "Table uomni_ablation; appendix beta/tau_bias, calibration size and transfer, "
                              "score normalization", "commands": abl}
    route = []
    for rule in ROUTING_RULES:
        tag = f"v321_route_{rule}_{model}"
        extra = ["--variants", EAB, "--route-rule", rule]
        if rule in ("mh", "mass", "entropy"):
            extra.append("--first-round-routing")
        if rule == "query":
            extra += ["--query-routes", f"runs/v321_universalrag_{model}/webqa_open_full.routes.json"]
        route += [run_cmd("webqa_open_full", tag, *extra, model=model, split=split),
                  score_cmd("webqa_open_full", tag),
                  ["python", "-m", "scripts.analysis.routing_metrics", f"runs/{tag}/webqa_open_full.jsonl",
                   "--arm", f"gated:{EAB}", "--metric", "acc_approx"]]
    s["routing"] = {"fills": "Table routing (WebQA, open retrieval)", "commands": route}
    order = []
    from omniaware.order import permutation_schedule   # noqa: E402 (needs src on the path)
    for d in ("okvqa", "webqa_open_img", "hybridqa", "videorag_synth"):
        for o in permutation_schedule():
            tag = f"v321_perm_{o['order_id']}_{model}"
            extra = ["--variants", EAB, "--replication-gamma", "1.2", "--order", f"perm:{o['order_id']}"]
            if model != "internvl3.5-8b":
                extra.append("--fitm")
            order += [run_cmd(d, tag, *extra, model=model, split=split), score_cmd(d, tag)]
        metric = DATASETS[d][1]
        order_arms = ["static", "replication", EAB] + (["fitm"] if model != "internvl3.5-8b" else [])
        order.append(["python", "-m", "scripts.analysis.paper_analyses", "order", f"runs/v321_perm_*_{model}/{d}.jsonl",
                      "--metric", metric, "--arms", *order_arms])
        order.append(["python", "-m", "scripts.analysis.paper_analyses", "ard", f"runs/v321_perm_sorted_{model}/{d}.jsonl"])
    s["order"] = {"fills": "Table order and appendix order_full (Std., gap, flip, ARD)", "commands": order}
    cite = []
    for d in ("okvqa", "webqa_open_img", "hybridqa"):
        tag = f"v321_cite_{model}"
        cite += [run_cmd(d, tag, "--variants", ABLATION_VARIANTS, "--replication-gamma", "1.2", "--answer-style", "cite",
                         "--max-new-tokens", "128", model=model, split=split),
                 ["python", "-m", "omniaware.eval.citation.attribution", f"runs/{tag}/{d}.jsonl", "--device", "cuda"]]
    s["citation"] = {"fills": "AP columns of Table uomni_ablation; appendix citation support", "commands": cite}
    iv = []
    tag = f"v321_interventions_{model}"
    iv += [run_cmd("webqa_open_img", tag, "--variants", EAB, "--answer-style", "cite", "--max-new-tokens", "128",
                   "--intervention", "mask", "cf", "random_edit", "rerun", "distractors", "inject_answer",
                   "inject_instruction", model=model, split=split),
           score_cmd("webqa_open_img", tag),
           ["python", "-m", "scripts.analysis.paper_analyses", "interventions", f"runs/{tag}/webqa_open_img.jsonl"],
           ["python", "-m", "omniaware.eval.citation.attribution", f"runs/{tag}/webqa_open_img.jsonl",
            "--intervention", "distractors", "--device", "cuda"],
           ["python", "-m", "scripts.analysis.paper_analyses", "gold25", f"runs/v321_main_{model}/webqa_open_img.jsonl",
            "--metric", "acc_approx"]]
    for sigma in ("0.1", "0.2"):
        for draw in range(3):
            t = f"v321_noise{sigma}_d{draw}_{model}"
            iv += [run_cmd("webqa_open_img", t, "--variants", EAB, "--replication-gamma", "1.2", "--relevance-noise",
                           sigma, "--noise-draw", str(draw), model=model, split=split), score_cmd("webqa_open_img", t)]
    iv.append(["python", "-m", "scripts.analysis.paper_analyses", "noise", f"runs/v321_noise*_{model}/webqa_open_img.jsonl",
               "--metric", "acc_approx", "--arms", "static", "replication", EAB])
    s["interventions"] = {"fills": "Table causal (interventions, Gold@2-5, noise, Misled)", "commands": iv}
    an = []
    for d, (_, metric) in DATASETS.items():
        path = f"runs/v321_main_{model}/{d}.jsonl"
        an += [["python", "-m", "scripts.analysis.paper_analyses", "uncertainty", path, "--metric", metric,
                "--arms", "static"],
               ["python", "-m", "scripts.analysis.paper_analyses", "latency", path],
               ["python", "-m", "scripts.analysis.paper_analyses", "stats", path, "--arms", "static", EAB],
               ["python", "-m", "scripts.analysis.paper_analyses", "errors", path, "--metric", metric,
                "--arms", f"gated:{EAB}"]]
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
    args = ap.parse_args()
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    all_stages = stages(args.model, args.split)
    if args.list or not args.stage:
        for name, st in all_stages.items():
            print(f"{name:14s} {len(st['commands']):4d} commands  -> {st['fills']}")
        return
    commands = all_stages[args.stage]["commands"]
    for cmd in commands:
        line = " ".join(shlex.quote(c) for c in cmd)
        if args.print or not args.run:
            print(line)
        if args.run:
            if "*" in line:
                subprocess.run(line, shell=True, check=True)
            else:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

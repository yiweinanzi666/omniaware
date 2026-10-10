"""Pre-registered dev decisions (docs/DEV_DECISIONS.md), applied mechanically to the dev runs.

    python -m scripts.analysis.dev_select --runs runs --model 8b [--output runs/dev_decisions.json]

Reads runs/v321dev_<setting>_<model>/<dataset>.jsonl and their official scores and writes the frozen choices
that scripts/paper_runs.py uses for the test runs:

  eab_variant   ONE global EAB form: the largest mean same-input gain over static, averaged over datasets
                (each dataset weighs equally); 'eab_effective' iff the bootstrap 95% lower bound > 0
  u_setting     the uncertainty setting (sum/zsum x last/mid layer) with the highest mean Spearman rho between
                the gate score and the error of the direct answer; compared with next-token entropy
  okvqa_loader  the OK-VQA retrieval corpus with the largest static - direct gap; 'okvqa_retrieval_helps'
                iff that gap is >= 0
  framing       1: EAB effective and U better than NTE; 2: only one of them; 3: neither (see the doc)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import analysis  # noqa: E402
from scripts.analysis.paper_analyses import load_run, load_scores, mean  # noqa: E402

EAB_FORMS = [f"eab:g1.2:{rows}:{form}:gate" for rows in ("ans1", "ans2", "q") for form in ("pos", "ctr")]
DEV_DATASETS = ["webqa_open_full", "okvqa_gs", "hybridqa", "videorag_synth"]
OKVQA_LOADERS = ["okvqa", "okvqa_dense", "okvqa_gs"]
U_SETTINGS = ["base", "zsum", "mid"]          # sum/last (paper), zsum/last, sum/mid
METRIC = {"webqa_open_full": "acc_approx", "okvqa": "vqa_accuracy", "okvqa_dense": "vqa_accuracy",
          "okvqa_gs": "vqa_accuracy", "hybridqa": "em", "videorag_synth": "rouge_l"}


def _path(runs: Path, setting: str, model: str, dataset: str) -> Path:
    return runs / f"v321dev_{setting}_{model}" / f"{dataset}.jsonl"


def _available(paths):
    return [p for p in paths if p.exists() and p.with_name(p.stem + ".official_scores.jsonl").exists()]


def pooled_gain(per_dataset: dict[str, np.ndarray], resamples: int = 5000, seed: int = 2027) -> dict:
    """Mean over datasets of the per-dataset mean delta, with a stratified bootstrap interval."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([d.mean() for d in per_dataset.values()]))
    boot = np.mean([d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
                    for d in per_dataset.values()], axis=0)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"gain": point, "ci95": [float(lo), float(hi)],
            "per_dataset": {k: float(v.mean()) for k, v in per_dataset.items()}}


def select_eab(runs, model):
    results = {}
    for form in EAB_FORMS:
        per = {}
        for d in DEV_DATASETS:
            for path in _available([_path(runs, "base", model, d)]):
                scores = load_scores(path, METRIC[d])
                if all(form in s for s in scores.values()):
                    per[d] = np.array([s[form] - s["static"] for s in scores.values()])
        if per:
            results[form] = pooled_gain(per)
    if not results:
        return None, results
    best = max(results, key=lambda f: results[f]["gain"])
    return best, results


def gate_rho(path: Path, metric: str) -> dict:
    rows, _ = load_run(path)
    scores = load_scores(path, metric)
    err = [1 - scores[r["qid"]]["direct"] / 100 for r in rows]
    u = [r.get("u_gate", r["u_pre"]) for r in rows]
    nte = [r["raw_pre"]["h_sem"] for r in rows]
    return {"u": analysis.spearman(u, err), "nte": analysis.spearman(nte, err),
            "u_aurc": 100 * analysis.aurc(u, err), "nte_aurc": 100 * analysis.aurc(nte, err)}


def select_u(runs, model):
    results = {}
    for setting in U_SETTINGS:
        per = {d: gate_rho(p, METRIC[d]) for d in DEV_DATASETS for p in _available([_path(runs, setting, model, d)])}
        if per:
            results[setting] = {"rho_u": mean([v["u"] for v in per.values()]),
                                "rho_nte": mean([v["nte"] for v in per.values()]), "per_dataset": per}
    if not results:
        return None, results
    best = max(results, key=lambda s: results[s]["rho_u"] if results[s]["rho_u"] is not None else -1)
    return best, results


def select_okvqa(runs, model):
    results = {}
    for loader in OKVQA_LOADERS:
        for path in _available([_path(runs, "base", model, loader)]):
            scores = load_scores(path, METRIC[loader])
            results[loader] = {"static": mean([s["static"] for s in scores.values()]),
                               "direct": mean([s["direct"] for s in scores.values()])}
            results[loader]["gap"] = results[loader]["static"] - results[loader]["direct"]
    if not results:
        return None, results
    return max(results, key=lambda k: results[k]["gap"]), results


def decide(runs: Path, model: str) -> dict:
    eab, eab_all = select_eab(runs, model)
    u, u_all = select_u(runs, model)
    okvqa, ok_all = select_okvqa(runs, model)
    eab_effective = bool(eab and eab_all[eab]["ci95"][0] > 0)
    u_better = bool(u and u_all[u]["rho_u"] is not None and u_all[u]["rho_nte"] is not None
                    and u_all[u]["rho_u"] > u_all[u]["rho_nte"])
    framing = 1 if eab_effective and u_better else (2 if eab_effective or u_better else 3)
    return {
        "model": model,
        "eab_variant": eab or "eab:g1.2:ans1:pos:gate", "eab_effective": eab_effective,
        "u_combine": "zsum" if u == "zsum" else "sum", "attn_layer": "mid" if u == "mid" else "last",
        "u_better_than_nte": u_better,
        "okvqa_loader": okvqa or "okvqa_gs",
        "okvqa_retrieval_helps": bool(okvqa and ok_all[okvqa]["gap"] >= 0),
        "framing": framing,
        "evidence": {"eab": eab_all, "u": u_all, "okvqa": ok_all},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--model", default="8b")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    result = decide(args.runs, args.model)
    out = args.output or args.runs / "dev_decisions.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, indent=1))
    return result


if __name__ == "__main__":
    main()

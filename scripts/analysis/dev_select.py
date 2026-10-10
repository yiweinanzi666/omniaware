"""Pre-registered dev decisions (docs/DEV_DECISIONS.md), applied mechanically to the dev runs.

    python -m scripts.analysis.dev_select --phase 1 [--runs runs] [--model 8b]   # base grid -> runs/dev_decisions.json
    python -m scripts.paper_runs --stage dev_beta --run                          # beta x tau_b grid (phase-1 choice)
    python -m scripts.analysis.dev_select --phase 2 [--runs runs] [--model 8b]   # beta, tau_b, tau -> complete file

Phase 1 reads runs/v321dev_base_<model>/<dataset>.jsonl (all EAB candidates) and the sum/zsum/mid U-setting runs,
and chooses, in this order: the OK-VQA corpus (largest static - direct gap), the uncertainty setting (highest mean
Spearman rho), and the EAB candidate (form x gamma) with the largest pooled same-input gain over static, judged by a
max-statistic bootstrap over all candidates. Phase 2 reads runs/v321dev_beta<b>_tb<t>_<model>/ for the phase-1
candidate and chooses beta and tau_b by the pooled gain; tau is then chosen offline from the same runs: the gate
only decides retrieve = (u_gate > tau), and every arm is decoded for every question, so the gated score for any
tau is exact without rerunning the model (see paper_analyses tau).

Every file read is checked (present, scored, finite) and its sha256 is written to the decisions file.
The test-stage commands in scripts/paper_runs.py refuse to run unless the phase-2 file exists for the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import analysis  # noqa: E402
from scripts.analysis.paper_analyses import load_run, load_scores, mean  # noqa: E402

ALPHA = 0.05
RESAMPLES = 5000
SEED = 2027
DELTA = 0.08                                    # fixed prior setting, not tuned (paper Sec. 4.1)
EAB_GAMMAS = (0.6, 0.9, 1.2, 1.5, 2.0)
EAB_ROWS = ("ans1", "ans2", "q")
EAB_FORMS = ("pos", "ctr")
BETAS = (0.0, 0.15, 0.3, 0.5, 1.0)
TAU_BIASES = (0.2, 0.4, 0.6)
TAUS = (0.25, 0.35, 0.45)
PAPER = {"gamma": 1.2, "beta": 0.30, "tau_bias": 0.40, "tau": 0.35, "rows": "ans1", "form": "pos"}
DEV_DATASETS = ["webqa_open_full", "okvqa_gs", "hybridqa", "videorag_synth"]   # U-setting runs exist for these
# D3: gs168k_full is not a candidate; the BM25 candidates (loader "okvqa") cannot be rebuilt from this repository
# (README 4.2), so the dev choice is between the two corpora whose build steps are complete.
OKVQA_LOADERS = ["okvqa_dense", "okvqa_gs"]
U_SETTINGS = ["base", "zsum", "mid"]          # sum/last (paper), zsum/last, sum/mid
METRIC = {"webqa_open_full": "acc_approx", "okvqa": "vqa_accuracy", "okvqa_dense": "vqa_accuracy",
          "okvqa_gs": "vqa_accuracy", "hybridqa": "em", "videorag_synth": "rouge_l"}


def eab_name(gamma: float, rows: str, form: str) -> str:
    """Canonical variant string; the same spelling is used by paper_runs for every run and by this module."""
    return f"eab:g{gamma:g}:{rows}:{form}:gate"


EAB_CANDIDATES = [eab_name(g, r, f) for g in EAB_GAMMAS for r in EAB_ROWS for f in EAB_FORMS]


class DevDataError(SystemExit):
    """A dev run is missing, unscored, or has non-finite scores: the decisions are not written."""


def _dir(runs: Path, setting: str, model: str) -> Path:
    return runs / f"v321dev_{setting}_{model}"


def dev_file(runs: Path, setting: str, model: str, dataset: str) -> Path:
    return _dir(runs, setting, model) / f"{dataset}.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_run(runs: Path, setting: str, model: str, dataset: str) -> Path:
    path = dev_file(runs, setting, model, dataset)
    scores = path.with_name(path.stem + ".official_scores.jsonl")
    missing = [str(p) for p in (path, scores) if not p.exists()]
    if missing:
        raise DevDataError(f"dev run missing: {', '.join(missing)} (run `paper_runs --stage dev --run` first)")
    return path


def score_matrix(runs: Path, setting: str, model: str, dataset: str, names: list[str]) -> tuple[list[str], np.ndarray]:
    """Questions (in run order) x arms; refuses missing arms and non-finite scores."""
    path = require_run(runs, setting, model, dataset)
    rows, _ = load_run(path)
    scores = load_scores(path, METRIC[dataset])
    qids = [r["qid"] for r in rows]
    absent = {n for q in qids for n in names if n not in scores.get(q, {})}
    if absent:
        raise DevDataError(f"{path}: arms missing from the scores: {sorted(absent)[:5]}")
    table = np.array([[scores[q][n] for n in names] for q in qids], dtype=float)
    if not np.isfinite(table).all():
        raise DevDataError(f"{path}: non-finite official scores (NaN would silently decide the selection)")
    return qids, table


def max_stat_gain(diffs: dict[str, np.ndarray], resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Pooled same-input gains of K candidates with a max-statistic bootstrap over the K candidates.

    diffs[dataset] is (questions x K): candidate minus static on the same question. The pooled gain is the mean
    over datasets of the per-dataset means (each dataset weighs equally). Questions are resampled jointly for all
    candidates (multinomial counts, paired across candidates). For each resample b, t*_bk = (g*_bk - g_k)/se_k and
    M*_b = max_k t*_bk. The simultaneous one-sided lower bound of candidate k is g_k - q_{1-alpha}(M*) * se_k; the
    best candidate is 'effective' only if its bound is above zero. This corrects for choosing the best of K.
    """
    rng = np.random.default_rng(seed)
    k = next(iter(diffs.values())).shape[1]
    point = np.mean([d.mean(axis=0) for d in diffs.values()], axis=0)
    boot = np.zeros((resamples, k))
    for d in diffs.values():
        n = len(d)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=resamples)
        boot += (counts @ d) / n / len(diffs)
    se = boot.std(axis=0, ddof=1)
    se = np.where(se > 0, se, 1e-12)
    max_t = ((boot - point) / se).max(axis=1)
    q = float(np.quantile(max_t, 1 - ALPHA))
    best = int(np.argmax(point))
    p_best = float(np.mean(max_t >= point[best] / se[best]))
    return {"gain": point, "se": se, "lower_bound": point - q * se, "best": best, "p_max_stat_best": p_best,
            "critical_value": q}


def select_eab(runs: Path, model: str, datasets: list[str]) -> dict:
    """Phase 1: one global EAB candidate (gamma, rows, form) on the chosen OK-VQA corpus and the other datasets."""
    diffs = {}
    for d in datasets:
        _, table = score_matrix(runs, "base", model, d, EAB_CANDIDATES + ["static"])
        diffs[d] = table[:, :-1] - table[:, [-1]]          # candidate minus static, same question
    res = max_stat_gain(diffs)
    names = EAB_CANDIDATES
    best = names[res["best"]]
    return {"candidate": best, "gamma": float(best.split(":")[1][1:]), "rows": best.split(":")[2],
            "form": best.split(":")[3], "effective": bool(res["lower_bound"][res["best"]] > 0),
            "gain": float(res["gain"][res["best"]]), "lower_bound": float(res["lower_bound"][res["best"]]),
            "p_max_stat": res["p_max_stat_best"], "critical_value": res["critical_value"],
            "candidates": {n: {"gain": float(g), "lower_bound": float(lb)}
                           for n, g, lb in zip(names, res["gain"], res["lower_bound"])},
            "datasets": datasets}


def gate_rho(path: Path, metric: str) -> dict:
    rows, _ = load_run(path)
    scores = load_scores(path, metric)
    err = [1 - scores[r["qid"]]["direct"] / 100 for r in rows]
    if not all(math.isfinite(x) for x in err):
        raise DevDataError(f"{path}: non-finite direct scores")
    u = [r.get("u_gate", r["u_pre"]) for r in rows]
    nte = [r["raw_pre"]["h_sem"] for r in rows]
    return {"u": analysis.spearman(u, err), "nte": analysis.spearman(nte, err),
            "u_aurc": 100 * analysis.aurc(u, err), "nte_aurc": 100 * analysis.aurc(nte, err)}


def select_u(runs: Path, model: str) -> dict:
    results = {}
    for setting in U_SETTINGS:      # 'base' is the paper's setting (sum / last layer)
        per = {d: gate_rho(require_run(runs, setting, model, d), METRIC[d]) for d in DEV_DATASETS}
        results[setting] = {"rho_u": mean([v["u"] for v in per.values()]),
                            "rho_nte": mean([v["nte"] for v in per.values()]), "per_dataset": per}
    best = max(results, key=lambda s: results[s]["rho_u"] if results[s]["rho_u"] is not None else -1)
    return {"setting": best, "u_combine": "zsum" if best == "zsum" else "sum",
            "attn_layer": "mid" if best == "mid" else "last",
            "rho_u": results[best]["rho_u"], "rho_nte": results[best]["rho_nte"], "all": results}


def select_okvqa(runs: Path, model: str) -> dict:
    results = {}
    for loader in OKVQA_LOADERS:
        qids, table = score_matrix(runs, "base", model, loader, ["static", "direct"])
        results[loader] = {"static": float(table[:, 0].mean()), "direct": float(table[:, 1].mean())}
        results[loader]["gap"] = results[loader]["static"] - results[loader]["direct"]
    best = max(results, key=lambda k: results[k]["gap"])
    return {"loader": best, "retrieval_helps": bool(results[best]["gap"] >= 0), "all": results}


def select_beta_tau_bias(runs: Path, model: str, candidate: str, okvqa: str) -> dict:
    """Phase 2: beta and tau_b cells (each its own run) scored by the pooled gain of the phase-1 candidate."""
    datasets = ["webqa_open_full", okvqa, "hybridqa", "videorag_synth"]
    grid = {}
    for beta in BETAS:
        for tb in TAU_BIASES:
            setting = f"beta{beta:g}_tb{tb:g}"
            per = []
            for d in datasets:
                _, table = score_matrix(runs, setting, model, d, [candidate, "static"])
                per.append((table[:, 0] - table[:, 1]).mean())
            grid[(beta, tb)] = float(np.mean(per))          # each dataset weighs equally
    best = max(((b, t) for b in BETAS for t in TAU_BIASES), key=lambda c: grid[c])
    return {"beta": best[0], "tau_bias": best[1], "datasets": datasets,
            "gain_grid": {f"beta{b:g}_tb{t:g}": v for (b, t), v in grid.items()}}


def select_tau(runs: Path, model: str, candidate: str, okvqa: str, beta: float, tau_bias: float) -> dict:
    """Phase 2, offline: tau changes only the gate (retrieve = u_gate > tau); every arm is decoded for every
    question, so the gated score for each tau is read from the selected cell's own run."""
    setting = f"beta{beta:g}_tb{tau_bias:g}"
    datasets = ["webqa_open_full", okvqa, "hybridqa", "videorag_synth"]
    score = {}
    for tau in TAUS:
        per = []
        for d in datasets:
            path = require_run(runs, setting, model, d)
            rows, _ = load_run(path)
            _, table = score_matrix(runs, setting, model, d, [candidate, "direct"])
            u = np.array([r.get("u_gate", r["u_pre"]) for r in rows], dtype=float)
            if not np.isfinite(u).all():
                raise DevDataError(f"{path}: non-finite u_gate")
            per.append(np.where(u > tau, table[:, 0], table[:, 1]).mean())
        score[tau] = float(np.mean(per))
    best = max(TAUS, key=lambda t: (score[t], -abs(t - PAPER["tau"])))
    return {"tau": best, "gated_score": {f"{t:g}": v for t, v in score.items()}, "datasets": datasets}


def file_hashes(runs: Path, model: str, settings: list[str], datasets: list[str]) -> dict:
    out = {}
    for setting in settings:
        for d in datasets:
            path = dev_file(runs, setting, model, d)
            if path.exists():
                out[str(path)] = sha256(path)
                scores = path.with_name(path.stem + ".official_scores.jsonl")
                if scores.exists():
                    out[str(scores)] = sha256(scores)
    return out


def phase1(runs: Path, model: str) -> dict:
    okvqa = select_okvqa(runs, model)
    u = select_u(runs, model)
    eab_datasets = ["webqa_open_full", okvqa["loader"], "hybridqa", "videorag_synth"]
    eab = select_eab(runs, model, eab_datasets)
    u_better = bool(u["rho_u"] is not None and u["rho_nte"] is not None and u["rho_u"] > u["rho_nte"])
    framing = 1 if eab["effective"] and u_better else (2 if eab["effective"] or u_better else 3)
    hashes = file_hashes(runs, model, U_SETTINGS, sorted(set(OKVQA_LOADERS + DEV_DATASETS)))
    return {"model": model, "phase": 1, "delta": DELTA, "okvqa_loader": okvqa["loader"],
            "okvqa_retrieval_helps": okvqa["retrieval_helps"],
            "u_combine": u["u_combine"], "attn_layer": u["attn_layer"], "u_setting": u["setting"],
            "u_better_than_nte": u_better,
            "eab_candidate": eab["candidate"], "gamma": eab["gamma"], "eab_rows": eab["rows"], "eab_form": eab["form"],
            "eab_variant": eab["candidate"], "eab_effective": eab["effective"], "framing": framing,
            "dev_files": hashes,
            "evidence": {"okvqa": okvqa, "u": {k: v for k, v in u.items() if k != "all"}, "u_all": u["all"],
                         "eab": eab}}


def phase2(runs: Path, model: str, decided: dict) -> dict:
    if decided.get("phase") != 1:
        raise DevDataError("phase 2 needs the phase-1 decisions in runs/dev_decisions.json")
    cand, okvqa = decided["eab_variant"], decided["okvqa_loader"]
    grid = select_beta_tau_bias(runs, model, cand, okvqa)
    tau = select_tau(runs, model, cand, okvqa, grid["beta"], grid["tau_bias"])
    settings = [f"beta{b:g}_tb{t:g}" for b in BETAS for t in TAU_BIASES]
    hashes = {**decided["dev_files"], **file_hashes(runs, model, settings, grid["datasets"])}
    out = dict(decided)
    out.update(phase=2, beta=grid["beta"], tau_bias=grid["tau_bias"], tau=tau["tau"],
               dev_files=hashes, evidence={**decided["evidence"], "beta_tau_bias": grid, "tau": tau})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--model", default="8b")
    ap.add_argument("--phase", type=int, choices=[1, 2], required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or args.runs / "dev_decisions.json"
    if args.phase == 1:
        result = phase1(args.runs, args.model)
    else:
        prev = json.loads(out.read_text()) if out.exists() else {}
        if prev.get("model") != args.model:
            raise DevDataError(f"{out} is missing or for another model; run --phase 1 first")
        result = phase2(args.runs, args.model, prev)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ("evidence", "dev_files")}, indent=1))
    return result


if __name__ == "__main__":
    main()

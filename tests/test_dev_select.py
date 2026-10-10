import json

import numpy as np
import pytest

from scripts.analysis import dev_select as ds


def _write(runs, setting, dataset, rows, scores, model="8b"):
    d = runs / f"v321dev_{setting}_{model}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{dataset}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / f"{dataset}.official_scores.jsonl").write_text("".join(
        json.dumps({"qid": q, "scores": {a: {ds.METRIC[dataset]: v} for a, v in s.items()}}) + "\n"
        for q, s in scores.items()))


def _questions(n, rng, boost=None):
    rows, scores = [], {}
    for i in range(n):
        hard = i % 2 == 0
        u = 0.8 if hard else 0.2
        rows.append({"qid": f"q{i}", "u_pre": u, "u_gate": u, "raw_pre": {"h_sem": float(rng.random())}})
        s = {"direct": 0.0 if hard else 100.0, "static": 50.0}
        for name in ds.EAB_CANDIDATES:
            s[name] = 50.0 + (boost[1] if boost and name == boost[0] else 0.0) + rng.normal(0, 1)
        scores[f"q{i}"] = s
    return rows, scores


def _base_tree(tmp_path, rng, boost=("eab:g0.9:q:ctr:gate", 10.0)):
    """Dev grid for phase 1: base runs for every OK-VQA loader and dev dataset, zsum / mid U settings."""
    for dataset in list(ds.DEV_DATASETS) + ["okvqa", "okvqa_dense"]:
        rows, scores = _questions(60, rng, boost)
        if dataset == "okvqa_dense":
            scores = {q: dict(s, static=80.0) for q, s in scores.items()}
        if dataset == "okvqa":
            scores = {q: dict(s, static=20.0) for q, s in scores.items()}   # largest static - direct gap for okvqa_gs
        _write(tmp_path, "base", dataset, rows, scores)
        if dataset in ds.DEV_DATASETS:
            for setting in ("zsum", "mid"):
                _write(tmp_path, setting, dataset, [dict(r, u_gate=float(rng.random())) for r in rows], scores)


def test_phase1_follows_the_preregistered_rules(tmp_path):
    rng = np.random.default_rng(0)
    _base_tree(tmp_path, rng)
    d = ds.phase1(tmp_path, "8b")
    assert d["phase"] == 1 and d["eab_variant"] == "eab:g0.9:q:ctr:gate" and d["gamma"] == 0.9
    assert d["eab_rows"] == "q" and d["eab_form"] == "ctr" and d["eab_effective"]
    assert d["u_combine"] == "sum" and d["attn_layer"] == "last" and d["u_better_than_nte"]
    assert d["okvqa_loader"] == "okvqa_dense" and d["okvqa_retrieval_helps"]     # static - direct gap of 30 points
    assert d["framing"] == 1
    # every file used is recorded with its sha256
    assert any(p.endswith("webqa_open_full.jsonl") for p in d["dev_files"])
    assert all(len(h) == 64 for h in d["dev_files"].values())


def test_missing_dev_run_and_nan_scores_refuse_to_decide(tmp_path):
    rng = np.random.default_rng(1)
    _base_tree(tmp_path, rng)
    (tmp_path / "v321dev_base_8b" / "hybridqa.jsonl").unlink()
    with pytest.raises(SystemExit, match="dev run missing"):
        ds.phase1(tmp_path, "8b")
    # restore a clean hybridqa run, then poison one score
    rows, scores = _questions(60, rng, ("eab:g0.9:q:ctr:gate", 10.0))
    scores["q3"]["eab:g1.2:ans1:pos:gate"] = float("nan")
    _write(tmp_path, "base", "hybridqa", rows, scores)
    with pytest.raises(SystemExit, match="non-finite"):
        ds.phase1(tmp_path, "8b")


def test_phase2_chooses_beta_and_tau_offline_and_tau_ties_go_to_the_paper_value(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    _base_tree(tmp_path, rng)
    decided = ds.phase1(tmp_path, "8b")
    monkeypatch.setattr(ds, "BETAS", (0.3, 1.0))
    monkeypatch.setattr(ds, "TAU_BIASES", (0.4,))
    okvqa = decided["okvqa_loader"]
    cand = decided["eab_variant"]
    for beta in (0.3, 1.0):
        for dataset in ["webqa_open_full", okvqa, "hybridqa", "videorag_synth"]:
            rows, scores = [], {}
            for i in range(40):
                u = float(rng.random())
                rows.append({"qid": f"q{i}", "u_pre": u, "u_gate": u, "raw_pre": {"h_sem": 0.1}})
                # direct is always 0. beta 0.3: the candidate never helps; beta 1.0: it is right exactly when U > 0.3
                right = (beta == 1.0 and u > 0.3)
                scores[f"q{i}"] = {"direct": 0.0, "static": 50.0, cand: 100.0 if right else 0.0}
            _write(tmp_path, f"beta{beta:g}_tb0.4", dataset, rows, scores)
    out = ds.phase2(tmp_path, "8b", decided)
    assert out["phase"] == 2 and out["beta"] == 1.0 and out["tau_bias"] == 0.4
    # gated score is largest when tau <= 0.3 (retrieval happens exactly when the candidate is right): 0.25 wins
    assert out["tau"] == 0.25
    assert out["evidence"]["tau"]["gated_score"]["0.25"] >= out["evidence"]["tau"]["gated_score"]["0.45"]
    assert any("beta1_tb0.4" in p for p in out["dev_files"])


def test_tau_tie_breaks_to_the_paper_value(tmp_path):
    rng = np.random.default_rng(3)
    rows, scores = [], {}
    for i in range(30):
        u = float(rng.random())
        rows.append({"qid": f"q{i}", "u_pre": u, "u_gate": u, "raw_pre": {"h_sem": 0.1}})
        scores[f"q{i}"] = {"direct": 50.0, "static": 50.0, "cand": 50.0}     # every gated score is 50
    for dataset in ["webqa_open_full", "okvqa_gs", "hybridqa", "videorag_synth"]:
        _write(tmp_path, "beta0.3_tb0.4", dataset, rows, scores)
    sel = ds.select_tau(tmp_path, "8b", "cand", "okvqa_gs", 0.3, 0.4)
    assert sel["tau"] == 0.35


def test_max_statistic_correction_has_the_nominal_size_under_the_null():
    """Zero true gain for all K candidates (correlated with each other), 4 datasets of 256 questions: the 'effective'
    rule must reject at about the nominal 5%, while the naive per-candidate interval for the best candidate does not."""
    rng = np.random.default_rng(0)
    sims, n, n_ds, k = 150, 256, 4, 6
    corrected = naive = 0
    for s in range(sims):
        diffs = {}
        for d in range(n_ds):
            common = rng.normal(0, 25.0, n)
            own = rng.normal(0, 25.0, (n, k))
            diffs[f"d{d}"] = 0.7 ** 0.5 * common[:, None] + (1 - 0.7) ** 0.5 * own
        res = ds.max_stat_gain(diffs, resamples=400, seed=s)
        best = res["best"]
        corrected += res["lower_bound"][best] > 0
        # naive: the best candidate's own percentile bootstrap, no correction for the choice of the best
        boot = np.zeros(400)
        r2 = np.random.default_rng(s + 1000)
        for v in diffs.values():
            idx = r2.integers(0, n, size=(400, n))
            boot += v[idx, best].mean(axis=1) / n_ds
        naive += np.quantile(boot, 0.025) > 0
    assert corrected / sims <= 0.10
    assert naive / sims > corrected / sims


def test_decisions_need_the_exact_candidate_grid():
    assert len(ds.EAB_CANDIDATES) == 30 and ds.eab_name(1.2, "ans1", "pos") == "eab:g1.2:ans1:pos:gate"
    assert ds.eab_name(2.0, "q", "ctr") == "eab:g2:q:ctr:gate"

import json

import numpy as np

from scripts.analysis import dev_select as ds


def _write(runs, setting, dataset, rows, scores):
    d = runs / f"v321dev_{setting}_8b"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{dataset}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / f"{dataset}.official_scores.jsonl").write_text("".join(
        json.dumps({"qid": q, "scores": {a: {ds.METRIC[dataset]: v} for a, v in s.items()}}) + "\n"
        for q, s in scores.items()))


def test_decisions_follow_the_preregistered_rules(tmp_path):
    rng = np.random.default_rng(0)
    for dataset in ds.DEV_DATASETS + ["okvqa", "okvqa_dense"]:
        rows, scores = [], {}
        for i in range(60):
            hard = i % 2 == 0
            u = 0.8 if hard else 0.2
            rows.append({"qid": f"q{i}", "u_pre": u, "u_gate": u, "raw_pre": {"h_sem": rng.random()}})
            s = {"direct": 0.0 if hard else 100.0, "static": 50.0}
            for form in ds.EAB_FORMS:
                s[form] = 50.0 + (10.0 if form == "eab:g1.2:q:ctr:gate" else 0.0) + rng.normal(0, 1)
            if dataset == "okvqa_dense":
                s["static"] = 80.0
            scores[f"q{i}"] = s
        _write(tmp_path, "base", dataset, rows, scores)
        if dataset in ds.DEV_DATASETS:
            for setting in ("zsum", "mid"):
                noisy = [dict(r, u_gate=rng.random()) for r in rows]
                _write(tmp_path, setting, dataset, noisy, scores)
    d = ds.decide(tmp_path, "8b")
    assert d["eab_variant"] == "eab:g1.2:q:ctr:gate" and d["eab_effective"]
    assert d["u_combine"] == "sum" and d["attn_layer"] == "last" and d["u_better_than_nte"]
    assert d["okvqa_loader"] == "okvqa_dense" and d["okvqa_retrieval_helps"]
    assert d["framing"] == 1

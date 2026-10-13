"""Exploratory offline check of the v2 selector on old pipeline dumps.

The selector itself lives in ``omniaware.arbitration`` (another worktree). This
script does not reimplement it. ``--help``, empty-answer tagging, and the local
half split do not import that module. The import happens inside ``run``, and
only the thresholded policies use it.

    python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2

Every number written under ``--out`` is exploratory, from the old pipeline, and
includes known bugs. It is not for the paper main table.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

LABEL = "探索性、旧管线、含已知 bug 的数据"
N1_CITED_RANGE = (433, 551)  # direction doc §1.2, not a count from this script
WEBQA_DIRECT = "direct"
WEBQA_STATIC = "always-static"
ACTION_DIRECT = "direct"
ACTION_STATIC = "static_mixed"
OKVQA_STATIC_KEY = "static"  # preds/scores key; static_joint is checked, not substituted

DEFAULT_WEBQA = Path(
    r"C:\rc\experiments\v318_missing_experiments_5h_root_20261008"
    r"\pro6000_resume_20261009\official_scoring_pipeline\server_results_37028"
    r"\results\full4966_scoring\seed17\OFFICIAL_SCORES.jsonl"
)
DEFAULT_OKVQA = Path(
    r"C:\rc\experiments\omniaware_github_20261009\a100_20261012"
    r"\results\reward_gate_v1\fast_math_v1\runs\tf32_ce_calibrated_k5_dev256"
)

_SOURCES_LINE = re.compile(r"(?i)^\s*sources\s*:")
_SOURCES_ANY = re.compile(r"(?i)sources\s*:")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory offline v2 check on old WebQA and OK-VQA dumps. "
            f"Every number it writes is {LABEL}. Not for the paper main table."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--out", type=Path, default=Path("runs/offline_v2"))
    parser.add_argument("--webqa", type=Path, default=DEFAULT_WEBQA)
    parser.add_argument("--okvqa", type=Path, default=DEFAULT_OKVQA)
    parser.add_argument("--delta", type=float, default=0.1)
    return parser


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def _text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _parsed_answer(row: dict):
    parsed = row.get("parsed")
    if isinstance(parsed, dict) and "answer" in parsed:
        return _text(parsed.get("answer"))
    return None


def _sources_only(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and all(_SOURCES_LINE.match(ln) for ln in lines)


def empty_answer_reasons(row: dict) -> list[str]:
    """A1 empty answer: blank prediction or parsed answer, a Sources-only
    response, or an empty answer body sitting next to a Sources line.

    A normal answer with a following Sources line is not empty.
    """
    reasons = []
    pred_present = "prediction" in row
    pred = _text(row.get("prediction")) if pred_present else ""
    parsed = _parsed_answer(row)
    raw_present = "raw_response" in row
    raw = _text(row.get("raw_response")) if raw_present else ""
    answer_present = "answer" in row and not pred_present and parsed is None
    answer = _text(row.get("answer")) if answer_present else ""
    if pred_present and pred.strip() == "":
        reasons.append("blank_prediction")
    if parsed is not None and parsed.strip() == "":
        reasons.append("blank_parsed_answer")
    if answer_present and answer.strip() == "":
        reasons.append("blank_answer")
    if ((pred_present and _sources_only(pred))
            or (raw_present and _sources_only(raw))
            or (parsed is not None and _sources_only(parsed))
            or (answer_present and _sources_only(answer))):
        reasons.append("sources_only")
    if raw_present and _SOURCES_ANY.search(raw):
        body = "\n".join(
            ln for ln in raw.splitlines()
            if ln.strip() and not _SOURCES_LINE.match(ln.strip())
        ).strip()
        if body == "":
            reasons.append("empty_answer_with_sources")
    out = []
    for reason in reasons:
        if reason not in out:
            out.append(reason)
    return out


def is_empty_answer(row: dict) -> bool:
    return bool(empty_answer_reasons(row))


def answer_text_for_selection(row: dict) -> str:
    """Answer string handed to the selector. Sources-only text is kept as stored."""
    parsed = _parsed_answer(row)
    if parsed is not None:
        return parsed
    if "prediction" in row:
        return _text(row.get("prediction"))
    return _text(row.get("answer"))


def local_split_half(qids, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic half split used by the unit test and by the blocked report.

    This is not ``arbitration.split_half``. Thresholds are fit with that
    function when the module is present. Ids are sorted before shuffling so
    the input order cannot change the split.
    """
    ids = sorted({str(q) for q in qids})
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    mid = len(ids) // 2
    calib, ev = ids[:mid], ids[mid:]
    if set(calib) & set(ev):
        raise RuntimeError("local_split_half produced an overlap")
    return calib, ev


def proxy_margin_from_u(u: float) -> float:
    """Proxy margin from u_initial. Not a first-token logit margin.

    Lower u_initial is treated as more confident. If u is in [0, 1],
    margin = 1 - u. Otherwise margin = -u.
    """
    u = float(u)
    if 0.0 <= u <= 1.0:
        return 1.0 - u
    return -u


def entropy_norm_proxy(h_sem: float) -> float:
    """Clip h_sem into [0, 1]. In this OK-VQA dump h_sem is already a small
    normalized entropy, not a logit margin."""
    return float(min(1.0, max(0.0, float(h_sem))))


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _new_variant_stats() -> dict:
    return {"n": 0, "empty": 0, "reasons": Counter(), "sources_none_exact": 0}


def _variant_public(stats: dict) -> dict:
    n = stats["n"]
    empty = stats["empty"]
    return {
        "n": n,
        "empty": empty,
        "rate": (empty / n) if n else None,
        "reasons": dict(stats["reasons"]),
        "sources_none_exact": stats["sources_none_exact"],
        "label": LABEL,
    }


def _blank_signal(margin, entropy_norm, retriever_top1, kl_to_direct):
    return {
        "margin": margin,
        "entropy_norm": entropy_norm,
        "seq_logprob": None,
        "gen_len": None,
        "kl_to_direct": kl_to_direct,
        "retriever_top1": retriever_top1,
    }


def load_webqa(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"WebQA jsonl not found: {path}")
    stats = {}
    keys = set()
    per = {}
    duplicate = Counter()
    n_rows = 0
    for row in iter_jsonl(path):
        n_rows += 1
        keys.update(row.keys())
        variant = row.get("variant")
        if not isinstance(variant, str):
            continue
        bucket = stats.setdefault(variant, _new_variant_stats())
        bucket["n"] += 1
        reasons = empty_answer_reasons(row)
        if reasons:
            bucket["empty"] += 1
            bucket["reasons"].update(reasons)
        raw = _text(row.get("raw_response")) if "raw_response" in row else ""
        if raw.strip().lower() == "sources: none":
            bucket["sources_none_exact"] += 1
        if variant not in (WEBQA_DIRECT, WEBQA_STATIC):
            continue
        qid = str(row["qid"])
        slot = per.setdefault(qid, {})
        if variant in slot:
            duplicate[variant] += 1
            continue
        slot[variant] = {
            "answer": answer_text_for_selection(row),
            "score": _finite(row.get("acc_approx")),
            "bart": _finite(row.get("normalized_bart_score")),
            "u": _finite(row.get("u_initial")),
            "subset": row.get("subset"),
            "empty": bool(reasons),
        }
    questions = []
    skipped = Counter()
    u_values = []
    margins = []
    u_mismatch = 0
    subset_mismatch = 0
    for qid, arms in per.items():
        if WEBQA_DIRECT not in arms or WEBQA_STATIC not in arms:
            skipped["missing_arm"] += 1
            continue
        direct, static = arms[WEBQA_DIRECT], arms[WEBQA_STATIC]
        if direct["score"] is None or static["score"] is None:
            skipped["missing_acc_approx"] += 1
            continue
        if direct["u"] is None:
            skipped["missing_u_initial"] += 1
            continue
        if static["u"] is not None and abs(direct["u"] - static["u"]) > 1e-9:
            u_mismatch += 1
        if static["subset"] != direct["subset"]:
            subset_mismatch += 1
        margin = proxy_margin_from_u(direct["u"])
        u_values.append(direct["u"])
        margins.append(margin)
        questions.append({
            "qid": qid,
            "subset": direct["subset"],
            "answers": {ACTION_DIRECT: direct["answer"], ACTION_STATIC: static["answer"]},
            "scores": {ACTION_DIRECT: direct["score"], ACTION_STATIC: static["score"]},
            "bart": {ACTION_DIRECT: direct["bart"], ACTION_STATIC: static["bart"]},
            "empty": {ACTION_DIRECT: direct["empty"], ACTION_STATIC: static["empty"]},
            "correct": {
                ACTION_DIRECT: int(direct["score"] >= 0.5),
                ACTION_STATIC: int(static["score"] >= 0.5),
            },
            "signals": {
                ACTION_DIRECT: _blank_signal(margin, None, None, 0.0),
                ACTION_STATIC: _blank_signal(margin, None, None, None),
            },
        })
    return {
        "path": str(path),
        "n_rows": n_rows,
        "keys": sorted(keys),
        "qa_approx_present": "QA_approx" in keys,
        "acc_approx_x_normalized_bart_present": "acc_approx_x_normalized_bart" in keys,
        "empty": {
            name: _variant_public(stats[name]) for name in (WEBQA_DIRECT, WEBQA_STATIC) if name in stats
        },
        "other_variants": {
            name: _variant_public(bucket) for name, bucket in sorted(stats.items())
            if name not in (WEBQA_DIRECT, WEBQA_STATIC)
        },
        "duplicate_rows_ignored": dict(duplicate),
        "skipped": dict(skipped),
        "u_mismatch_questions": u_mismatch,
        "subset_mismatch_questions": subset_mismatch,
        "u_initial": _range(u_values),
        "proxy_margin": _range(margins),
        "questions": questions,
    }


def load_okvqa(directory: Path) -> dict:
    if not directory.is_dir():
        raise FileNotFoundError(f"OK-VQA directory not found: {directory}")
    pred_path = directory / "okvqa_gs.jsonl"
    score_path = directory / "okvqa_gs.official_scores.jsonl"
    if not pred_path.is_file() or not score_path.is_file():
        raise FileNotFoundError(f"OK-VQA preds or official scores missing under {directory}")
    official = {}
    score_arms = Counter()
    for row in iter_jsonl(score_path):
        qid = str(row["qid"])
        scores = row.get("scores") or {}
        official[qid] = scores
        score_arms.update(scores.keys())
    stats = {"direct": _new_variant_stats(), OKVQA_STATIC_KEY: _new_variant_stats()}
    questions = []
    skipped = Counter()
    h_pre, h_post, h_grd = [], [], []
    margins_pre, margins_post, top1s = [], [], []
    joint_equal = 0
    joint_n = 0
    joint_score_equal = 0
    joint_score_n = 0
    pred_rows = 0
    for row in iter_jsonl(pred_path):
        pred_rows += 1
        qid = str(row["qid"])
        preds = row.get("preds") or {}
        if "static_joint" in preds and OKVQA_STATIC_KEY in preds:
            joint_n += 1
            if preds["static_joint"] == preds[OKVQA_STATIC_KEY]:
                joint_equal += 1
        arm_scores = official.get(qid, {})
        if (OKVQA_STATIC_KEY in arm_scores and "static_joint" in arm_scores
                and "vqa_accuracy" in arm_scores[OKVQA_STATIC_KEY]
                and "vqa_accuracy" in arm_scores["static_joint"]):
            joint_score_n += 1
            if arm_scores[OKVQA_STATIC_KEY]["vqa_accuracy"] == arm_scores["static_joint"]["vqa_accuracy"]:
                joint_score_equal += 1
        for arm in stats:
            if arm not in preds:
                continue
            text = _text(preds[arm])
            tagged = {"prediction": text, "raw_response": text}
            reasons = empty_answer_reasons(tagged)
            stats[arm]["n"] += 1
            if reasons:
                stats[arm]["empty"] += 1
                stats[arm]["reasons"].update(reasons)
            if text.strip().lower() == "sources: none":
                stats[arm]["sources_none_exact"] += 1
        if ACTION_DIRECT not in preds or OKVQA_STATIC_KEY not in preds:
            skipped["missing_pred"] += 1
            continue
        if qid not in official:
            skipped["missing_official_row"] += 1
            continue
        direct_score = _score_field(arm_scores, ACTION_DIRECT)
        static_score = _score_field(arm_scores, OKVQA_STATIC_KEY)
        if direct_score is None or static_score is None:
            skipped["missing_vqa_accuracy"] += 1
            continue
        try:
            hsem_pre = float(row["raw_pre"]["h_sem"])
            hsem_post = float(row["raw_post"]["h_sem"])
        except (KeyError, TypeError, ValueError):
            skipped["missing_h_sem"] += 1
            continue
        rel = row.get("rel") or {}
        top1 = max(float(v) for v in rel.values()) if rel else None
        h_pre.append(hsem_pre)
        h_post.append(hsem_post)
        margins_pre.append(-hsem_pre)
        margins_post.append(-hsem_post)
        if top1 is not None:
            top1s.append(top1)
        try:
            h_grd.append(float(row["raw_pre"]["h_grd"]))
        except (KeyError, TypeError, ValueError):
            pass
        direct_text = _text(preds[ACTION_DIRECT])
        static_text = _text(preds[OKVQA_STATIC_KEY])
        questions.append({
            "qid": qid,
            "subset": None,
            "answers": {ACTION_DIRECT: direct_text, ACTION_STATIC: static_text},
            "scores": {ACTION_DIRECT: direct_score, ACTION_STATIC: static_score},
            "bart": {ACTION_DIRECT: None, ACTION_STATIC: None},
            "empty": {
                ACTION_DIRECT: is_empty_answer({"prediction": direct_text, "raw_response": direct_text}),
                ACTION_STATIC: is_empty_answer({"prediction": static_text, "raw_response": static_text}),
            },
            "correct": {
                ACTION_DIRECT: int(direct_score >= 50.0),
                ACTION_STATIC: int(static_score >= 50.0),
            },
            "signals": {
                ACTION_DIRECT: _blank_signal(-hsem_pre, entropy_norm_proxy(hsem_pre), None, 0.0),
                ACTION_STATIC: _blank_signal(-hsem_post, entropy_norm_proxy(hsem_post), top1, None),
            },
        })
    return {
        "path": str(directory),
        "predictions": str(pred_path),
        "official_scores": str(score_path),
        "n_pred_rows": pred_rows,
        "n_score_rows": len(official),
        "score_arms": dict(score_arms),
        "static_key": OKVQA_STATIC_KEY,
        "static_joint_answer_equal": joint_equal,
        "static_joint_answer_n": joint_n,
        "static_joint_score_equal": joint_score_equal,
        "static_joint_score_n": joint_score_n,
        "score_field": "vqa_accuracy",
        "empty": {name: _variant_public(bucket) for name, bucket in stats.items()},
        "skipped": dict(skipped),
        "h_sem_pre": _range(h_pre),
        "h_sem_post": _range(h_post),
        "h_grd_pre": _range(h_grd),
        "proxy_margin_direct": _range(margins_pre),
        "proxy_margin_static": _range(margins_post),
        "retriever_top1": _range(top1s),
        "questions": questions,
    }


def _score_field(arm_scores: dict, arm: str):
    block = arm_scores.get(arm)
    if not isinstance(block, dict) or "vqa_accuracy" not in block:
        return None
    return _finite(block["vqa_accuracy"])


def _range(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        mid = ordered[n // 2]
    else:
        mid = (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {
        "n": n,
        "min": ordered[0],
        "median": mid,
        "max": ordered[-1],
        "label": LABEL,
    }


def _split_sizes(qids, seed: int) -> dict:
    calib, ev = local_split_half(qids, seed)
    return {
        "n": len(set(map(str, qids))),
        "n_calib": len(calib),
        "n_eval": len(ev),
        "disjoint": set(calib).isdisjoint(ev),
        "covers": set(calib) | set(ev) == set(map(str, qids)),
        "label": LABEL,
    }


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _mean(values) -> dict:
    present = [float(v) for v in values if v is not None]
    missing = len(values) - len(present)
    if not present:
        return {"n": 0, "n_missing": missing, "mean": None, "label": LABEL}
    return {
        "n": len(present),
        "n_missing": missing,
        "mean": sum(present) / len(present),
        "label": LABEL,
    }


def _oracle_gap(policy, baseline, oracle):
    num = 0.0
    den = 0.0
    for p, b, o in zip(policy, baseline, oracle):
        num += float(p) - float(b)
        den += float(o) - float(b)
    return {
        "numerator_sum_policy_minus_best": num,
        "denominator_sum_oracle_minus_best": den,
        "fraction": (num / den) if den > 0 else None,
        "definition": (
            "sum(policy - best_single) / sum(oracle - best_single) on the eval half; "
            "null when the denominator is not > 0. Oracle is the per-question max of "
            "the available actions' official scores."
        ),
        "label": LABEL,
    }


def _import_arbitration():
    """Import the selector. Missing module returns None; a broken module raises."""
    root = Path(__file__).resolve().parents[2]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    path = root / "src" / "omniaware" / "arbitration.py"
    try:
        import omniaware.arbitration as arb
        from omniaware.arbitration import (  # noqa: WPS433
            V2_ACTIONS, best_single_action, fit_v2_cal, ltt_select,
            preregistered_grid, select_v2_cal, select_v2_rule, split_half,
            summarize_policy, token_f1, utility,
        )
    except ModuleNotFoundError:
        return None
    except ImportError:
        if not path.is_file():
            return None
        raise
    return SimpleNamespace(
        V2_ACTIONS=V2_ACTIONS,
        best_single_action=best_single_action,
        fit_v2_cal=fit_v2_cal,
        ltt_select=ltt_select,
        preregistered_grid=preregistered_grid,
        select_v2_cal=select_v2_cal,
        select_v2_rule=select_v2_rule,
        split_half=split_half,
        summarize_policy=summarize_policy,
        token_f1=token_f1,
        utility=utility,
        ActionRecord=getattr(arb, "ActionRecord", None),
        FirstToken=getattr(arb, "FirstToken", None),
    )


def make_action(mod, name: str, answer: str, signal: dict):
    """Dict record. The selector indexes ``answer`` and ``first_token``; it does not read attributes.

    These dumps have no sequence log-prob. Missing ``seq_logprob`` / ``gen_len`` / ``entropy_norm``
    become 0 / 1 / 0 so v2-Cal can run. That fill carries no information from the file.
    """
    del mod
    entropy = signal["entropy_norm"]
    seq = signal["seq_logprob"]
    gen = signal["gen_len"]
    return {
        "answer": answer,
        "seq_logprob": 0.0 if seq is None else float(seq),
        "gen_len": 1 if not gen else int(gen),
        "kl_to_direct": signal["kl_to_direct"],
        "first_token": {
            "margin": float(signal["margin"]),
            "entropy": 0.0 if entropy is None else float(entropy),
            "entropy_norm": 0.0 if entropy is None else float(entropy),
        },
        "retriever": {"top1_score": signal["retriever_top1"]},
    }


def question_actions(mod, question: dict) -> dict:
    return {
        name: make_action(mod, name, question["answers"][name], question["signals"][name])
        for name in (ACTION_DIRECT, ACTION_STATIC)
    }


def _candidate(item) -> dict:
    if isinstance(item, dict):
        return item
    out = {}
    for key in ("name", "tau1", "theta", "confidence"):
        if hasattr(item, key):
            out[key] = getattr(item, key)
    return out


def _call_rule(mod, actions, cand: dict) -> dict:
    fn = mod.select_v2_rule
    tau1, theta = cand["tau1"], cand["theta"]
    conf = cand.get("confidence", "margin")
    sig = inspect.signature(fn)
    if "confidence" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        out = fn(actions, tau1, theta, confidence=conf)
    else:
        out = fn(actions, tau1, theta)
    if not isinstance(out, dict) or "action" not in out:
        raise TypeError(f"select_v2_rule returned {out!r}")
    return out


def _require_split(qids: list[str], calib, ev) -> tuple[list[str], list[str]]:
    calib = [str(q) for q in calib]
    ev = [str(q) for q in ev]
    cs, es, allq = set(calib), set(ev), set(qids)
    if (len(calib) != len(cs) or len(ev) != len(es) or cs & es or cs | es != allq
            or len(calib) + len(ev) != len(qids)):
        raise RuntimeError(
            "split_half is not a disjoint cover of the question ids: "
            f"n={len(qids)} calib={len(calib)} eval={len(ev)} overlap={len(cs & es)}"
        )
    return calib, ev


def _scores_for(questions, action: str) -> list[float]:
    return [q["scores"][action] for q in questions]


def _selected_scores(mod, questions, cand: dict) -> tuple[list[float], list[str], Counter]:
    scores, actions = [], []
    counts = Counter()
    evidence = 0
    for question in questions:
        chosen = _call_rule(mod, question_actions(mod, question), cand)
        action = chosen["action"]
        if action not in question["scores"]:
            raise KeyError(f"selector chose {action!r}, which has no official score")
        scores.append(question["scores"][action])
        actions.append(action)
        counts[action] += 1
        if chosen.get("used_evidence"):
            evidence += 1
    counts["__used_evidence__"] = evidence
    return scores, actions, counts


def _model_size(model) -> dict:
    if isinstance(model, dict):
        weights = model.get("weights")
        n_params = model.get("n_params")
    else:
        weights = getattr(model, "weights", None)
        n_params = getattr(model, "n_params", None)
    listed = None if weights is None else [float(x) for x in list(weights)]
    return {
        "n_params": None if n_params is None else int(n_params),
        "n_weights": None if listed is None else len(listed),
        "weights": listed,
        "label": LABEL,
    }


def _subset_table(questions, columns: dict[str, list]) -> dict:
    groups = {}
    for index, question in enumerate(questions):
        key = question.get("subset") or "all"
        bucket = groups.setdefault(key, {name: [] for name in columns})
        for name, values in columns.items():
            bucket[name].append(values[index])
    return {
        key: {name: _mean(values) for name, values in cols.items()}
        for key, cols in groups.items()
    }


def evaluate_dataset(mod, bundle: dict, seed: int, delta: float, objective: str) -> dict:
    questions = bundle["questions"]
    by_id = {q["qid"]: q for q in questions}
    qids = sorted(by_id)
    calib_ids, eval_ids = _require_split(qids, *mod.split_half(qids, seed))
    calib = [by_id[q] for q in calib_ids]
    ev = [by_id[q] for q in eval_ids]
    scores_by_action = {name: _scores_for(calib, name) for name in (ACTION_DIRECT, ACTION_STATIC)}
    best_name = mod.best_single_action(scores_by_action)
    if not isinstance(best_name, str) or best_name not in scores_by_action:
        raise TypeError(f"best_single_action returned {best_name!r}")
    grid = list(mod.preregistered_grid())
    cands = [_candidate(item) for item in grid]
    if any("name" not in c or "tau1" not in c or "theta" not in c for c in cands):
        raise TypeError("preregistered_grid items need name, tau1, and theta")
    if len({c["name"] for c in cands}) != len(cands):
        raise ValueError("preregistered_grid names are not unique")
    # Grid order is kept. ltt_select requires conservative-to-aggressive already.
    scores_by_candidate = {}
    for cand in cands:
        scores, _, _ = _selected_scores(mod, calib, cand)
        scores_by_candidate[cand["name"]] = scores
    decision = mod.ltt_select(
        grid, scores_by_candidate, scores_by_action[best_name], delta=delta, best_action=best_name,
    )
    if not isinstance(decision, dict) or "name" not in decision or "fallback" not in decision:
        raise TypeError(f"ltt_select returned {decision!r}")
    decision = _plain(decision)
    if decision["fallback"]:
        policy = _scores_for(ev, best_name)
        chosen_actions = [best_name] * len(ev)
        counts = Counter(chosen_actions)
        chosen_cand = None
    else:
        cand = next((c for c in cands if c["name"] == decision["name"]), None)
        if cand is None:
            raise KeyError(f"ltt_select named {decision['name']!r}, which is not in the grid")
        policy, chosen_actions, counts = _selected_scores(mod, ev, cand)
        chosen_cand = {k: cand[k] for k in ("name", "tau1", "theta") if k in cand}
        if "confidence" in cand:
            chosen_cand["confidence"] = cand["confidence"]
    baseline = _scores_for(ev, best_name)
    oracle = [max(q["scores"].values()) for q in ev]
    bart_policy = [ev[i]["bart"].get(chosen_actions[i]) for i in range(len(ev))]
    rule = {
        "decision": decision,
        "candidate": chosen_cand,
        "delta": delta,
        "eval": _eval_block(mod, ev, policy, baseline, oracle, bart_policy, counts),
        "label": LABEL,
    }
    cal = _fit_cal(mod, calib, ev, baseline)
    return {
        "label": LABEL,
        "objective": objective,
        "seed": seed,
        "n_questions": len(questions),
        "n_calib": len(calib),
        "n_eval": len(ev),
        "best_single_action_on_calib": best_name,
        "declared_actions": [str(name) for name in mod.V2_ACTIONS],
        "actions_in_dump": [ACTION_DIRECT, ACTION_STATIC],
        "grid": [_plain(c) for c in cands],
        "v2_rule": rule,
        "v2_cal": cal,
        "eval_by_subset": _subset_table(ev, {
            "direct": _scores_for(ev, ACTION_DIRECT),
            "static_mixed": _scores_for(ev, ACTION_STATIC),
            "v2_rule": policy,
            "oracle": oracle,
            "best_single": baseline,
        }),
    }


def _eval_block(mod, ev, policy, baseline, oracle, bart_policy, counts) -> dict:
    return {
        "policy_mean": _mean(policy),
        "best_single_mean": _mean(baseline),
        "direct_mean": _mean(_scores_for(ev, ACTION_DIRECT)),
        "static_mixed_mean": _mean(_scores_for(ev, ACTION_STATIC)),
        "oracle_mean": _mean(oracle),
        "utility": _plain(mod.utility(policy, baseline)),
        "paired": _plain(mod.summarize_policy(policy, baseline)),
        "oracle_gap": _oracle_gap(policy, baseline, oracle),
        "action_counts": {k: int(v) for k, v in counts.items()},
        "bart_policy_mean": _mean(bart_policy),
        "bart_direct_mean": _mean([q["bart"][ACTION_DIRECT] for q in ev]),
        "bart_static_mixed_mean": _mean([q["bart"][ACTION_STATIC] for q in ev]),
        "label": LABEL,
    }


def _fit_cal(mod, calib, ev, baseline) -> dict:
    # Labels from the calibration half only. Eval labels are not passed in.
    try:
        rows = [{"actions": question_actions(mod, q), "correct": q["correct"]} for q in calib]
        model = mod.fit_v2_cal(rows)
        size = _model_size(model)
        scores = []
        counts = Counter()
        bart_policy = []
        for question in ev:
            chosen = mod.select_v2_cal(model, question_actions(mod, question))
            if not isinstance(chosen, dict) or "action" not in chosen:
                raise TypeError(f"select_v2_cal returned {chosen!r}")
            action = chosen["action"]
            if action not in question["scores"]:
                raise KeyError(f"v2-Cal chose {action!r}, which has no official score")
            scores.append(question["scores"][action])
            bart_policy.append(question["bart"].get(action))
            counts[action] += 1
        oracle = [max(q["scores"].values()) for q in ev]
        return {
            "status": "ok",
            "fit_on": "calib_half_only",
            "model": size,
            "eval": {
                "policy_mean": _mean(scores),
                "best_single_mean": _mean(baseline),
                "utility": _plain(mod.utility(scores, baseline)),
                "paired": _plain(mod.summarize_policy(scores, baseline)),
                "oracle_gap": _oracle_gap(scores, baseline, oracle),
                "action_counts": {k: int(v) for k, v in counts.items()},
                "bart_policy_mean": _mean(bart_policy),
                "label": LABEL,
            },
            "label": LABEL,
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "label": LABEL}


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _mean_cell(block: dict | None) -> str:
    if not block:
        return "null"
    return f"{_fmt(block.get('mean'))} (n={block.get('n')})"


def render_report(state: dict) -> str:
    lines = [
        "# Offline v2 exploratory check",
        "",
        f"Every number in this file is {LABEL}. Not for the paper main table.",
        "",
        "## Reproduce",
        "",
        "```",
        "python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2",
        "```",
        "",
        f"Seed used: {state['seed']}. Exit code this run will return: {state['exit_code']}.",
        "",
        "## Empty answers (A1)",
        "",
        f"Numbers in this section are {LABEL}.",
        "",
        "A row is empty if the prediction or the parsed answer is blank, or `raw_response` / "
        "the prediction is only a `Sources:` line, or the answer body is empty while a "
        "`Sources:` line is present. A non-empty answer followed by a Sources line is not empty.",
        "",
        "| slice | empty / n | rate | exact raw `Sources: none` |",
        "|---|---:|---:|---:|",
    ]
    for dataset, variants in (("WebQA", state["empty"]["webqa"]), ("OK-VQA", state["empty"]["okvqa"])):
        for name, bucket in variants.items():
            rate = bucket["rate"]
            lines.append(
                f"| {dataset} {name} | {bucket['empty']}/{bucket['n']} | {_fmt(rate)} | "
                f"{bucket['sources_none_exact']} |"
            )
    lines += [
        "",
        "### Comparison with the cited ~10% N1 figure",
        "",
        "The root-cause report (`Omni-Aware_根因溯源报告_20261013.md`) does not contain the "
        "N1 / `Sources: none` / 10% count. That figure is in the direction document, section 1.2: "
        f"each WebQA arm has {N1_CITED_RANGE[0]}–{N1_CITED_RANGE[1]} questions (about 10%) whose "
        "raw output is only a `Sources: none` line. Those two numbers are quoted from that "
        "document. They are not measurements from this script.",
        "",
        state["n1_explanation"],
        "",
        "`QA_approx` is absent from the WebQA file. The file does contain "
        "`acc_approx_x_normalized_bart`. This script does not treat that field as `QA_approx` "
        "and does not use it as the selection objective.",
        "",
        "## Splits",
        "",
        f"Numbers in this section are {LABEL}.",
        "",
        state["split_prose"],
        "",
        "| dataset | n | calib | eval | disjoint |",
        "|---|---:|---:|---:|---|",
    ]
    for name, bucket in state["splits"]["reported"].items():
        lines.append(
            f"| {name} | {bucket['n']} | {bucket['n_calib']} | {bucket['n_eval']} | {bucket['disjoint']} |"
        )
    lines += [
        "",
        "## Signals",
        "",
        "WebQA direct margin is a proxy margin from u_initial: `1 - u_initial` when u is in "
        "[0, 1], otherwise `-u_initial`. Lower `u_initial` is treated as more confident. "
        "This is not a first-token logit margin. `static_mixed` uses that same proxy margin "
        "because the dump has no separate evidence confidence. `static_text`, `static_image`, "
        "and `rerank_static` are omitted. WebQA `entropy_norm`, sequence log-prob, and retriever "
        "top-1 are absent from the file. The selector requires numbers, so those missing features "
        "are passed as 0 (generation length 1). They add no information. Direct `kl_to_direct` is 0 "
        "because that action is the reference; the evidence arm's KL is missing and is also passed as "
        "absent (`None`, which the selector treats as 0).",
        "",
        f"WebQA `u_initial` on paired questions: {state['webqa_u']}. "
        f"Proxy margin from u_initial: {state['webqa_margin']}. "
        f"Questions where the two arms' `u_initial` differ by more than 1e-9: "
        f"{state['webqa_u_mismatch']}.",
        "",
        "OK-VQA uses `preds[\"direct\"]` and `preds[\"static\"]`. "
        f"`static` answers equal `static_joint` on {state['okvqa_joint_answers']}. "
        f"Official scores use the field `vqa_accuracy` "
        f"({state['okvqa_joint_scores']}). "
        "Direct proxy margin is `-h_sem` from `raw_pre`. Evidence proxy margin is `-h_sem` "
        "from `raw_post`. `entropy_norm` is `h_sem` clipped to [0, 1]. Retriever top-1 for the "
        "evidence arm is `max(rel)` when `rel` is present. `h_grd` is in the file and is not a "
        "calibration feature. `gold_in_topk` is not passed to the selector. Sequence log-prob "
        "is absent, so the `seq_logprob / gen_len` feature is 0.",
        "",
        f"OK-VQA h_sem pre {state['okvqa_h_pre']}; h_sem post {state['okvqa_h_post']}; "
        f"direct proxy margin {state['okvqa_margin_pre']}; evidence proxy margin "
        f"{state['okvqa_margin_post']}.",
        "",
        "WebQA selection objective is `acc_approx`. OK-VQA selection objective is the official "
        "`vqa_accuracy` already in `okvqa_gs.official_scores.jsonl`. No new metric is computed. "
        "`normalized_bart_score` is a secondary column on WebQA when a policy is scored. "
        "v2-Cal 0/1 labels, fit on the calibration half only, are `acc_approx >= 0.5` on WebQA "
        "(the repo correct-at 50 rule applied to a 0–1 score) and `vqa_accuracy >= 50` on OK-VQA. "
        "Reported means use the official scores, not these labels.",
        "",
        "## Selector",
        "",
        f"Numbers in this section are {LABEL}.",
        "",
    ]
    if state["exit_code"] == 2:
        lines += [
            "arbitration.py is not in this worktree yet.",
            "",
            "v2-Rule, v2-Cal, paired CIs, and the oracle-gap fraction were not computed. "
            "This file does not contain those numbers.",
            "",
        ]
    elif state.get("selector_error"):
        lines += [
            "The arbitration import succeeded, and the selector call then failed. "
            "No accuracy below was filled in after the failure.",
            "",
            "```",
            state["selector_error"],
            "```",
            "",
        ]
    for dataset, result in state.get("results", {}).items():
        lines += _result_section(dataset, result)
    lines.append("")
    return "\n".join(lines)


def _result_section(dataset: str, result: dict) -> list[str]:
    if result.get("status") == "error":
        return [
            f"### {dataset}",
            "",
            f"Numbers in this section are {LABEL}.",
            "",
            "Selector error, no policy scores:",
            "",
            "```",
            result.get("error", ""),
            "```",
            "",
        ]
    rule = result["v2_rule"]["eval"]
    lines = [
        f"### {dataset}",
        "",
        f"Numbers in this section are {LABEL}.",
        "",
        f"Objective: `{result['objective']}`. Questions: {result['n_questions']}. "
        f"Calib {result['n_calib']} / eval {result['n_eval']}. "
        f"Best single action chosen on the calib half: `{result['best_single_action_on_calib']}`. "
        f"v2-Rule decision: `{json.dumps(result['v2_rule']['decision'], ensure_ascii=False)}`. "
        f"Candidate: `{json.dumps(result['v2_rule']['candidate'], ensure_ascii=False)}`.",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| eval direct mean | {_mean_cell(rule['direct_mean'])} |",
        f"| eval static_mixed mean | {_mean_cell(rule['static_mixed_mean'])} |",
        f"| eval best single mean | {_mean_cell(rule['best_single_mean'])} |",
        f"| eval v2-Rule mean | {_mean_cell(rule['policy_mean'])} |",
        f"| eval oracle mean | {_mean_cell(rule['oracle_mean'])} |",
        f"| utility(policy, best) | {_fmt(rule['utility'])} |",
        f"| oracle-gap fraction | {_fmt(rule['oracle_gap']['fraction'])} |",
        f"| oracle-gap numerator | {_fmt(rule['oracle_gap']['numerator_sum_policy_minus_best'])} |",
        f"| oracle-gap denominator | {_fmt(rule['oracle_gap']['denominator_sum_oracle_minus_best'])} |",
        f"| secondary normalized BART, direct | {_mean_cell(rule['bart_direct_mean'])} |",
        f"| secondary normalized BART, static_mixed | {_mean_cell(rule['bart_static_mixed_mean'])} |",
        f"| secondary normalized BART, v2-Rule | {_mean_cell(rule['bart_policy_mean'])} |",
        "",
        f"v2-Rule action counts on eval: `{json.dumps(rule['action_counts'], ensure_ascii=False)}`.",
        "",
        "Paired comparison from `summarize_policy` (policy vs best single on the eval half):",
        "",
        "```json",
        json.dumps(rule["paired"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    cal = result["v2_cal"]
    lines.append(f"v2-Cal status: `{cal.get('status')}`.")
    lines.append("")
    if cal.get("status") == "ok":
        ceval = cal["eval"]
        lines += [
            f"v2-Cal parameters: n_params={cal['model']['n_params']}, "
            f"n_weights={cal['model']['n_weights']}. Fit on the calib half only.",
            "",
            "| quantity | value |",
            "|---|---:|",
            f"| eval v2-Cal mean | {_mean_cell(ceval['policy_mean'])} |",
            f"| utility(policy, best) | {_fmt(ceval['utility'])} |",
            f"| oracle-gap fraction | {_fmt(ceval['oracle_gap']['fraction'])} |",
            f"| secondary normalized BART, v2-Cal | {_mean_cell(ceval['bart_policy_mean'])} |",
            "",
            "```json",
            json.dumps(ceval["paired"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    elif cal.get("error"):
        lines += ["```", cal["error"], "```", ""]
    if result.get("eval_by_subset"):
        lines += [f"Eval means by subset (`{result['objective']}`):", "", "```json",
                  json.dumps(result["eval_by_subset"], ensure_ascii=False, indent=2), "```", ""]
    return lines


def _n1_explanation(webqa: dict, okvqa: dict) -> str:
    direct = webqa["empty"][WEBQA_DIRECT]
    static = webqa["empty"][WEBQA_STATIC]
    ok_direct = okvqa["empty"]["direct"]
    ok_static = okvqa["empty"][OKVQA_STATIC_KEY]
    others = webqa.get("other_variants") or {}
    other_bits = ", ".join(
        f"`{name}` exact `Sources: none` {bucket['sources_none_exact']}/{bucket['n']} "
        f"(A1 empty {bucket['empty']}/{bucket['n']})"
        for name, bucket in others.items()
    )
    other_sentence = (
        f" The same file's other variants are not used by the selector: {other_bits}."
        if other_bits else ""
    )
    return (
        f"This script measures WebQA `{WEBQA_DIRECT}` {direct['empty']}/{direct['n']} "
        f"(rate {_fmt(direct['rate'])}) and `{WEBQA_STATIC}` {static['empty']}/{static['n']} "
        f"(rate {_fmt(static['rate'])}). The denominator is every row of that variant in "
        f"`{webqa['path']}`, not the paired-question subset and not a Sources-filtered subset. "
        f"Exact `raw_response` equal to `Sources: none` is {direct['sources_none_exact']} for "
        f"direct and {static['sources_none_exact']} for always-static. Direct "
        f"{direct['sources_none_exact']} is below the cited minimum {N1_CITED_RANGE[0]}. "
        f"always-static {static['sources_none_exact']} falls inside {N1_CITED_RANGE[0]}–"
        f"{N1_CITED_RANGE[1]}.{other_sentence} The cited range is an across-arm summary; this "
        "file's direct arm is lower, so the rate is not forced to 10%. OK-VQA empty answers are "
        f"direct {ok_direct['empty']}/{ok_direct['n']} and static {ok_static['empty']}/{ok_static['n']}. "
        "That dump stores short strings in `preds` and has no citation `raw_response`, so it is "
        "a different field, a different variant key (`static`), and a different denominator "
        f"({ok_direct['n']} development questions, not the WebQA full set)."
    )


def _split_prose(splits: dict) -> str:
    if splits["thresholds"] == "arbitration.split_half":
        return (
            "Thresholds are chosen only on the calibration half from `arbitration.split_half` "
            f"(seed {splits['seed']}). The table is that split. `local_split_half` is the unit-test "
            "split and is not the split used for thresholds."
        )
    return (
        "Thresholds were not chosen. `arbitration.split_half` is not available in this worktree. "
        "The table is `local_split_half`: sort the question ids, shuffle with `random.Random(seed)`, "
        "and give the first half to calib and the rest to eval. That split was not used to fit a selector."
    )


def run(args: argparse.Namespace) -> int:
    webqa = load_webqa(args.webqa)
    okvqa = load_okvqa(args.okvqa)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    empty = {
        "label": LABEL,
        "definition": (
            "Empty if prediction or parsed answer is blank, or raw_response/prediction is only "
            "a Sources line, or the answer body is empty while a Sources line exists."
        ),
        "webqa": empty_public(webqa),
        "okvqa": {
            "path": okvqa["path"],
            "static_key": okvqa["static_key"],
            "variants": okvqa["empty"],
            "label": LABEL,
        },
    }
    local_splits = {
        "webqa": _split_sizes([q["qid"] for q in webqa["questions"]], args.seed),
        "okvqa": _split_sizes([q["qid"] for q in okvqa["questions"]], args.seed),
    }
    splits = {
        "label": LABEL,
        "seed": args.seed,
        "thresholds": "unavailable",
        "local_split_half": local_splits,
        "arbitration_split_half": None,
        "reported": local_splits,
    }
    inputs = {
        "label": LABEL,
        "webqa_path": webqa["path"],
        "okvqa_path": okvqa["path"],
        "webqa_n_rows": webqa["n_rows"],
        "webqa_paired_questions": len(webqa["questions"]),
        "webqa_skipped": webqa["skipped"],
        "webqa_duplicate_rows_ignored": webqa["duplicate_rows_ignored"],
        "webqa_u_mismatch_questions": webqa["u_mismatch_questions"],
        "qa_approx": "absent" if not webqa["qa_approx_present"] else "present",
        "acc_approx_x_normalized_bart_present": webqa["acc_approx_x_normalized_bart_present"],
        "proxy_margin": "proxy margin from u_initial",
        "webqa_u_initial": webqa["u_initial"],
        "webqa_proxy_margin": webqa["proxy_margin"],
        "okvqa_static_key": OKVQA_STATIC_KEY,
        "okvqa_static_joint_answer_equal": okvqa["static_joint_answer_equal"],
        "okvqa_static_joint_answer_n": okvqa["static_joint_answer_n"],
        "okvqa_static_joint_score_equal": okvqa["static_joint_score_equal"],
        "okvqa_static_joint_score_n": okvqa["static_joint_score_n"],
        "okvqa_score_field": okvqa["score_field"],
        "okvqa_skipped": okvqa["skipped"],
        "okvqa_paired_questions": len(okvqa["questions"]),
        "okvqa_h_sem_pre": okvqa["h_sem_pre"],
        "okvqa_h_sem_post": okvqa["h_sem_post"],
        "okvqa_h_grd_pre_not_a_feature": okvqa["h_grd_pre"],
        "okvqa_proxy_margin_direct": okvqa["proxy_margin_direct"],
        "okvqa_proxy_margin_static": okvqa["proxy_margin_static"],
        "calibration_labels": {
            "webqa": "acc_approx >= 0.5, calib half only, not the reported metric",
            "okvqa": "vqa_accuracy >= 50, calib half only, not the reported metric",
        },
    }
    _write(out / "empty_answers.json", empty)
    _write(out / "inputs.json", inputs)
    state = {
        "seed": args.seed,
        "exit_code": 2,
        "empty": {"webqa": empty["webqa"]["variants"], "okvqa": empty["okvqa"]["variants"]},
        "splits": splits,
        "n1_explanation": _n1_explanation(webqa, okvqa),
        "split_prose": "",
        "webqa_u": _fmt_range(webqa["u_initial"]),
        "webqa_margin": _fmt_range(webqa["proxy_margin"]),
        "webqa_u_mismatch": webqa["u_mismatch_questions"],
        "okvqa_joint_answers": (
            f"{okvqa['static_joint_answer_equal']}/{okvqa['static_joint_answer_n']}"
        ),
        "okvqa_joint_scores": (
            f"static vqa_accuracy equals static_joint on "
            f"{okvqa['static_joint_score_equal']}/{okvqa['static_joint_score_n']}"
        ),
        "okvqa_h_pre": _fmt_range(okvqa["h_sem_pre"]),
        "okvqa_h_post": _fmt_range(okvqa["h_sem_post"]),
        "okvqa_margin_pre": _fmt_range(okvqa["proxy_margin_direct"]),
        "okvqa_margin_post": _fmt_range(okvqa["proxy_margin_static"]),
        "results": {},
        "selector_error": None,
    }
    try:
        mod = _import_arbitration()
    except Exception:
        mod = None
        state["exit_code"] = 1
        state["selector_error"] = traceback.format_exc()
    if mod is None and state["exit_code"] != 1:
        print("arbitration.py is not in this worktree yet")
        state["split_prose"] = _split_prose(splits)
        (out / "REPORT.md").write_text(render_report(state), encoding="utf-8")
        _write(out / "splits.json", splits)
        return 2
    if mod is None:
        state["split_prose"] = _split_prose(splits)
        (out / "REPORT.md").write_text(render_report(state), encoding="utf-8")
        _write(out / "splits.json", splits)
        print(state["selector_error"], file=sys.stderr)
        return 1
    try:
        results = {
            "webqa": evaluate_dataset(mod, webqa, args.seed, args.delta, "acc_approx"),
            "okvqa": evaluate_dataset(mod, okvqa, args.seed, args.delta, "vqa_accuracy"),
        }
    except Exception:
        state["exit_code"] = 1
        state["selector_error"] = traceback.format_exc()
        state["split_prose"] = _split_prose(splits)
        (out / "REPORT.md").write_text(render_report(state), encoding="utf-8")
        _write(out / "splits.json", splits)
        print(state["selector_error"], file=sys.stderr)
        return 1
    # Sizes are the halves evaluate_dataset already checked with split_half.
    arb_splits = {}
    for name, result in results.items():
        arb_splits[name] = {
            "n": result["n_questions"],
            "n_calib": result["n_calib"],
            "n_eval": result["n_eval"],
            "disjoint": True,
            "covers": result["n_calib"] + result["n_eval"] == result["n_questions"],
            "label": LABEL,
        }
    splits["thresholds"] = "arbitration.split_half"
    splits["arbitration_split_half"] = arb_splits
    splits["reported"] = arb_splits
    state["exit_code"] = 0
    state["splits"] = splits
    state["split_prose"] = _split_prose(splits)
    state["results"] = results
    _write(out / "splits.json", {k: v for k, v in splits.items()})
    _write(out / "results.json", {"label": LABEL, "datasets": results})
    (out / "REPORT.md").write_text(render_report(state), encoding="utf-8")
    return 0


def empty_public(webqa: dict) -> dict:
    return {
        "path": webqa["path"],
        "n_rows": webqa["n_rows"],
        "variants": webqa["empty"],
        "other_variants_not_used_by_selector": webqa["other_variants"],
        "label": LABEL,
    }


def _fmt_range(block: dict | None) -> str:
    if not block:
        return "absent"
    return f"min {_fmt(block['min'])}, median {_fmt(block['median'])}, max {_fmt(block['max'])}, n={block['n']}"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

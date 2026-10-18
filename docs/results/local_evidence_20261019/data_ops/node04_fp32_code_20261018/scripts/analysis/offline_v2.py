"""Exploratory offline check of the v2 selectors on two legacy dumps. Not the paper's main table.

    python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2

The dumps come from the legacy pipeline, which has known bugs, so every number is exploratory. WebQA: the official
scores of the direct and always-static variants (always-static is the static_mixed action). OK-VQA: the dev run
tf32_ce_calibrated_k5_dev256 (direct and static, the latter as static_mixed).

Each question becomes a row of the per-question action table (docs/V2_REWORK_SPEC.md, section 2). The sorted
question ids are permuted with numpy.random.default_rng(seed); the first half calibrates, the second half is
evaluated. The calibration half fits and risk-controls v2-Rule and v2-Cal, and also sets a naive single-threshold
baseline that has no risk control. A second analysis keeps only the questions where both direct and static_mixed
answers are non-empty, and repeats the split and every step on those questions. Writes results.json and REPORT.md
under --out.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from omniaware import answers, arbitration, runio  # noqa: E402
from omniaware.actions import DIRECT, V2_ACTIONS  # noqa: E402
from omniaware.eval import score_official  # noqa: E402

STATIC = "static_mixed"
ACTIONS = (DIRECT, STATIC)
WEBQA_VARIANTS = {"direct": DIRECT, "always-static": STATIC}
OKVQA_ARMS = {"direct": DIRECT, "static": STATIC}
WEBQA_SCORES = Path(r"C:\rc\experiments\v318_missing_experiments_5h_root_20261008\pro6000_resume_20261009"
                    r"\official_scoring_pipeline\server_results_37028\results\full4966_scoring\seed17"
                    r"\OFFICIAL_SCORES.jsonl")
OKVQA_DIR = Path(r"C:\rc\experiments\omniaware_github_20261009\a100_20261012\results\reward_gate_v1"
                 r"\fast_math_v1\runs\tf32_ce_calibrated_k5_dev256")
DELTA = 0.1
LEGACY_CONFIDENCE = "margin"        # the legacy dumps hold only a margin proxy, no answer_token
EPSILONS = (0.0, 1.0)
AUDIT_EPSILON = 1.0                 # the v2-Rule setting whose direct choices the WebQA note audits
GRID_STEPS = 20
SUBSET_WEBQA = {"img": "image_question", "text": "text_question"}
CAL_NOTE = "（对内部拆分种子敏感，见下方种子范围）"
SHORT_WORDS = 3                     # "<= 3 words" against ">= 4 words"
MIN_NONEMPTY = 8                    # v2-Cal needs at least 4 calibration questions, so 8 questions for the split


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _action(raw_text: str, margin: float, top1: float | None) -> dict:
    """One available action. Only the first-token margin and the retriever top-1 are known in these dumps; every
    other signal is missing and recorded as None (never filled in)."""
    parsed = answers.parse_answer(raw_text)
    return {"available": True, "answer": parsed["answer"], "answer_empty": parsed["answer_empty"],
            "citation_only": parsed["citation_only"], "seq_logprob": None, "gen_len": None,
            "first_token": {"margin": margin, "entropy": None, "entropy_norm": None},
            "kl_to_direct": None, "retriever": {"top1_score": top1}}


def load_webqa(path: Path) -> list[dict]:
    """One row per WebQA question: direct and always-static (static_mixed). The single confidence proxy is
    margin = 1 - u_initial, shared by both actions, because the dump has no first-token logits."""
    arms: dict[str, dict[str, dict]] = {}
    for row in read_jsonl(path):
        action = WEBQA_VARIANTS.get(row["variant"])
        if action is None:
            continue
        slot = arms.setdefault(row["qid"], {})
        if action in slot:
            raise ValueError(f"{path}: two {row['variant']} rows for question {row['qid']}")
        slot[action] = row
    rows = []
    for qid in sorted(arms):
        if set(arms[qid]) != set(ACTIONS):
            raise ValueError(f"question {qid}: needs both direct and always-static rows")
        direct, static = arms[qid][DIRECT], arms[qid][STATIC]
        margin = 1.0 - float(direct["u_initial"])
        rows.append({
            "qid": qid, "subset": SUBSET_WEBQA[direct["subset"]],
            "actions": {DIRECT: _action(direct["raw_response"], margin, None),
                        STATIC: _action(static["raw_response"], margin, None)},
            "scores": {DIRECT: 100.0 * float(direct["acc_approx"]), STATIC: 100.0 * float(static["acc_approx"])},
            "metric": "acc_approx",
        })
    return rows


def load_okvqa(directory: Path) -> list[dict]:
    """One row per OK-VQA question. margin is -h_sem of the question's own entropy: pre-retrieval for direct and
    post-retrieval for static_mixed. The retriever top-1 of static_mixed is the largest relevance among the evidence
    it received (``rel`` holds the relevance of each selected evidence); direct received no evidence."""
    records = {r["qid"]: r for r in read_jsonl(directory / "okvqa_gs.jsonl")}
    official = {r["qid"]: r["scores"] for r in read_jsonl(directory / "okvqa_gs.official_scores.jsonl")}
    if set(records) != set(official):
        raise ValueError(f"{directory}: predictions and official scores cover different questions")
    rows = []
    for qid in sorted(records):
        rec = records[qid]
        top1 = max(rec["rel"].values()) if rec.get("rel") else None
        rows.append({
            "qid": qid, "subset": None,
            "actions": {DIRECT: _action(rec["preds"]["direct"], -float(rec["raw_pre"]["h_sem"]), None),
                        STATIC: _action(rec["preds"]["static"], -float(rec["raw_post"]["h_sem"]), top1)},
            "scores": {DIRECT: float(official[qid]["direct"]["vqa_accuracy"]),
                       STATIC: float(official[qid]["static"]["vqa_accuracy"])},
            "metric": "vqa_accuracy",
        })
    return rows


def split_halves(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Calibration and evaluation halves. The sorted question ids are permuted with the seed; the first half
    calibrates. Rows keep the sorted order inside each half."""
    ordered = sorted(r["qid"] for r in rows)
    permuted = [str(q) for q in np.random.default_rng(seed).permutation(ordered)]
    half = len(ordered) // 2
    calib_ids, eval_ids = set(permuted[:half]), set(permuted[half:])
    if calib_ids & eval_ids or len(calib_ids) + len(eval_ids) != len(ordered):
        raise RuntimeError("the calibration and evaluation halves overlap or lose questions")
    by_qid = {r["qid"]: r for r in rows}
    return ([by_qid[q] for q in ordered if q in calib_ids], [by_qid[q] for q in ordered if q in eval_ids])


def both_nonempty(row: dict) -> bool:
    return not any(row["actions"][action]["answer_empty"] for action in ACTIONS)


def length_class(answer: str) -> str:
    """'short' for at most SHORT_WORDS normalised words, 'long' otherwise (the empty answer is short)."""
    return "short" if len(arbitration.normalize_answer(answer)) <= SHORT_WORDS else "long"


def naive_threshold(calib: list[dict]) -> dict:
    """Baseline without risk control: direct when its margin >= tau, else static_mixed. tau runs over the 21
    quantiles of the calibration margins and the best calibration mean wins (the earliest quantile on ties)."""
    margins = np.array([r["actions"][DIRECT]["first_token"]["margin"] for r in calib], dtype=float)
    scores = {a: np.array([r["scores"][a] for r in calib], dtype=float) for a in ACTIONS}
    best = None
    for tau in np.unique(np.quantile(margins, np.linspace(0.0, 1.0, GRID_STEPS + 1))):
        mean = float(np.where(margins >= tau, scores[DIRECT], scores[STATIC]).mean())
        if best is None or mean > best["calib_mean"]:
            best = {"kind": "naive_threshold", "tau": float(tau), "calib_mean": mean}
    return best


def naive_choices(tau: float, rows: list[dict]) -> list[str]:
    return [DIRECT if r["actions"][DIRECT]["first_token"]["margin"] >= tau else STATIC for r in rows]


def oracle_choices(rows: list[dict]) -> list[str]:
    """The best available action of each question; ties go to the earlier action in V2_ACTIONS order."""
    return [max(r["scores"], key=lambda a: (r["scores"][a], -V2_ACTIONS.index(a))) for r in rows]


def _policy_block(name: str, rows: list[dict], choices: list[str], best_scores: np.ndarray,
                  oracle: np.ndarray, decision: dict | None) -> dict:
    scores = np.asarray([r["scores"][c] for r, c in zip(rows, choices)], dtype=float)
    gap_oracle = float(oracle.mean() - best_scores.mean())
    return {
        "name": name, "mean": float(scores.mean()),
        "vs_best_single": score_official.paired(best_scores, scores),
        "capture_ratio": float((scores.mean() - best_scores.mean()) / gap_oracle) if gap_oracle > 0 else None,
        "chosen": dict(Counter(choices)),
        "decision": arbitration.decision_fields(decision) if decision and "kind" in decision else decision,
    }


def cal_seed_spread(calib: list[dict], evaluation: list[dict], seeds: range = range(20)) -> dict:
    """v2-Cal at epsilon=1 over several seeds of its internal fit/risk split. The spread shows how much the
    result depends on that split; the main result keeps the run seed and is not picked from this range."""
    means = []
    for seed in seeds:
        cal = arbitration.select_cal(calib, delta=DELTA, epsilon=1.0, seed=seed, confidence=LEGACY_CONFIDENCE)
        choices = [arbitration.apply_decision(cal, r["actions"])["action"] for r in evaluation]
        means.append(float(np.mean([r["scores"][c] for r, c in zip(evaluation, choices)])))
    return {"seeds": len(means), "min": min(means), "median": float(np.median(means)), "max": max(means)}


def direct_audit(evaluation: list[dict], choices: list[str]) -> dict:
    """What the direct choices of v2-Rule rest on: static_mixed being empty, and direct and static_mixed answers in
    different length classes. Counted on the questions where v2-Rule chose direct and on the whole evaluation half."""
    picked = [r for r, c in zip(evaluation, choices) if c == DIRECT]

    def differs(r: dict) -> bool:
        return length_class(r["actions"][DIRECT]["answer"]) != length_class(r["actions"][STATIC]["answer"])

    return {"rule_direct": len(picked),
            "rule_direct_static_empty": sum(bool(r["actions"][STATIC]["answer_empty"]) for r in picked),
            "rule_direct_length_differs": sum(differs(r) for r in picked),
            "evaluation": len(evaluation),
            "evaluation_length_differs": sum(differs(r) for r in evaluation)}


def evaluate_dataset(rows: list[dict], calib: list[dict], evaluation: list[dict], seed: int) -> dict:
    """Fit on the calibration half, report on the evaluation half. The strongest single action is the calibration
    winner; every block is paired against it on the evaluation half."""
    best = arbitration.best_single_action({a: [r["scores"][a] for r in calib] for a in ACTIONS})
    best_scores = np.asarray([r["scores"][best] for r in evaluation], dtype=float)
    oracle = np.asarray([max(r["scores"].values()) for r in evaluation], dtype=float)
    n = len(evaluation)
    naive = naive_threshold(calib)
    blocks = [
        _policy_block(DIRECT, evaluation, [DIRECT] * n, best_scores, oracle, None),
        _policy_block(STATIC, evaluation, [STATIC] * n, best_scores, oracle, None),
        _policy_block(f"最强单一动作 {best}", evaluation, [best] * n, best_scores, oracle, None),
        _policy_block("朴素基线", evaluation, naive_choices(naive["tau"], evaluation), best_scores, oracle, naive),
        _policy_block("逐题上限", evaluation, oracle_choices(evaluation), best_scores, oracle, None),
    ]
    audit = None
    for eps in EPSILONS:
        rule = arbitration.select_rule(calib, delta=DELTA, epsilon=eps, confidence=LEGACY_CONFIDENCE)
        choices = [arbitration.apply_decision(rule, r["actions"])["action"] for r in evaluation]
        blocks.append(_policy_block(f"v2-Rule ε={eps:g}", evaluation, choices, best_scores, oracle, rule))
        if eps == AUDIT_EPSILON:
            audit = direct_audit(evaluation, choices)
    for eps in EPSILONS:
        cal = arbitration.select_cal(calib, delta=DELTA, epsilon=eps, seed=seed, confidence=LEGACY_CONFIDENCE)
        choices = [arbitration.apply_decision(cal, r["actions"])["action"] for r in evaluation]
        blocks.append(_policy_block(f"v2-Cal ε={eps:g}{CAL_NOTE}", evaluation, choices, best_scores, oracle, cal))
    empty = {}
    for action in ACTIONS:
        flags = [bool(r["actions"][action]["answer_empty"]) for r in rows]
        empty[action] = {"empty": sum(flags), "questions": len(flags), "rate": sum(flags) / len(flags)}
    return {"questions": len(rows), "calib": len(calib), "eval": n, "best_single": best,
            "calib_qids_sha256": runio.digest(sorted(r["qid"] for r in calib)),
            "eval_qids_sha256": runio.digest(sorted(r["qid"] for r in evaluation)),
            "naive": naive, "empty_rates": empty, "blocks": blocks, "direct_audit": audit,
            "cal_seed_spread": cal_seed_spread(calib, evaluation)}


def run(webqa_path: Path, okvqa_dir: Path, seed: int) -> dict:
    inputs = {"webqa": webqa_path, "okvqa_preds": okvqa_dir / "okvqa_gs.jsonl",
              "okvqa_official": okvqa_dir / "okvqa_gs.official_scores.jsonl"}
    loaded = {"webqa": load_webqa(webqa_path), "okvqa": load_okvqa(okvqa_dir)}
    results = {"exploratory": True, "paper_table": False, "seed": seed, "delta": DELTA, "epsilons": list(EPSILONS),
               "inputs": {k: {"path": str(v), "sha256": runio.sha256_file(v)} for k, v in inputs.items()},
               "datasets": {}, "both_nonempty": {}}
    for name, rows in loaded.items():
        calib, evaluation = split_halves(rows, seed)
        results["datasets"][name] = evaluate_dataset(rows, calib, evaluation, seed)
        kept = [r for r in rows if both_nonempty(r)]
        if len(kept) < MIN_NONEMPTY:
            raise ValueError(f"{name}: {len(kept)} questions have both answers non-empty; "
                             f"the split needs at least {MIN_NONEMPTY}")
        kept_calib, kept_eval = split_halves(kept, seed)
        results["both_nonempty"][name] = evaluate_dataset(kept, kept_calib, kept_eval, seed)
        results["both_nonempty"][name]["dropped"] = len(rows) - len(kept)
    return results


# ---------------------------------------------------------------- report

def _fmt_decision(block: dict) -> str:
    d = block["decision"]
    if not d:
        return ""
    if d.get("kind") == "naive_threshold":
        return f"tau={d['tau']:.2f}"
    if d.get("kind") == "single" or d.get("fallback"):
        return f"回退为 {d.get('action', d.get('best_single'))}"
    parts = [f"kind={d['kind']}"]
    if d["kind"] == "rule":
        parts.append(f"evidence={d['evidence_choice']} tau={arbitration.decode_json_float(d['tau']):.2f}")
    if d["kind"] == "cal":
        parts.append(f"lam={arbitration.decode_json_float(d['lam']):.2f}")
    parts.append(f"n_accepted={d.get('n_accepted')}")
    return " ".join(parts)


def _block_row(block: dict) -> str:
    v = block["vs_best_single"]
    lo, hi = v["paired_ci95_pp"]
    rel = "—" if v["relative_percent"] is None else f"{v['relative_percent']:+.2f}%"
    cap = "—" if block["capture_ratio"] is None else f"{block['capture_ratio']:.2f}"
    chosen = "，".join(f"{k} {n}" for k, n in sorted(block["chosen"].items()))
    return (f"| {block['name']} | {block['mean']:.2f} | {v['delta_pp']:+.2f} [{lo:.2f}, {hi:.2f}] | {rel} | "
            f"{cap} | {chosen} |")


def _decision_lines(block: dict) -> list[str]:
    lines = [f"- {block['name']}：{_fmt_decision(block)}"]
    models = (block["decision"] or {}).get("models")
    for action, model in (models or {}).items():
        lines.append(f"  - {action}：保留特征 {model['features']}，丢弃 {model['dropped']}，参数 {model['n_params']} 个")
    return lines


def _section(title: str, data: dict) -> list[str]:
    lines = [f"## {title}（校准 {data['calib']} 题，评测 {data['eval']} 题）", "",
             f"最强单一动作（校准半选出）：{data['best_single']}。", "",
             "| 策略 | 评测半均分 | 配对差 [95% CI] | 相对提升 | 上限差距比例 | 被选动作计数 |",
             "|---|---:|---:|---:|---:|---|"]
    lines += [_block_row(b) for b in data["blocks"]]
    lines += ["", "决定的关键字段："]
    for block in data["blocks"]:
        if block["decision"]:
            lines += _decision_lines(block)
    spread = data["cal_seed_spread"]
    lines += ["", f"v2-Cal ε=1 对内部拆分种子的敏感性（种子 0–{spread['seeds'] - 1}）：评测半均分 "
              f"最小 {spread['min']:.2f}，中位 {spread['median']:.2f}，最大 {spread['max']:.2f}。"]
    lines += ["", "空答案率（全部题目）：" + "；".join(
        f"{a} {e['empty']}/{e['questions']} = {e['rate']:.2f}" for a, e in data["empty_rates"].items()), ""]
    return lines


def _audit_lines(audit: dict) -> list[str]:
    return [
        f"v2-Rule ε={AUDIT_EPSILON:g} 在评测半选中 direct 的题：{audit['rule_direct']} 题，其中 static_mixed 为空答案"
        f"（含仅 Sources 行）的题 {audit['rule_direct_static_empty']} 题，direct 与 static_mixed 答案长度类别不同的题 "
        f"{audit['rule_direct_length_differs']} 题。",
        f"评测半全部 {audit['evaluation']} 题中，direct 与 static_mixed 答案长度类别不同（≤{SHORT_WORDS} 词对 "
        f"≥{SHORT_WORDS + 1} 词，按规范化词数计）的题 {audit['evaluation_length_differs']} 题。",
    ]


def render(results: dict) -> str:
    seed = results["seed"]
    lines = [
        "# v2 离线探索（旧管线）", "",
        "**探索性数字，不进论文主表。** 数据来自含已知 bug 的旧管线：WebQA 官方分数的 direct 与 always-static 变体，"
        "OK-VQA 开发运行 tf32_ce_calibrated_k5_dev256。只用于检查选择器代码和信号方向。", "",
        f"复现：`python scripts/analysis/offline_v2.py --seed {seed} --out runs/offline_v2`。"
        f"对排序后的 qid 做 numpy.random.default_rng({seed}).permutation，前一半校准，后一半评测；δ={results['delta']}，"
        "ε 取 0 与 1（0–100 量纲）。配对差是 评测半 相对最强单一动作 的差值，单位为分，CI 为 95% 百分位区间。", "",
        "v2-Cal 的结果依赖内部拆分种子（种子 20 个取值的范围见各节末尾）。", "",
    ]
    for name, data in results["datasets"].items():
        lines += _section(f"{name}", data)
        if data["direct_audit"] is not None and name == "webqa":
            lines += ["WebQA 增益的来源检查（旧管线伪影）：", ""] + [f"- {x}" for x in _audit_lines(data["direct_audit"])]
            lines += [""]
    lines += ["# 只在两路答案都非空的题上重算", "",
              "保留 direct 与 static_mixed 答案都非空的题；校准与评测都只用这些题，划分、δ、ε、种子与上表相同。", ""]
    for name, data in results["both_nonempty"].items():
        lines += _section(f"{name}，两路非空，去掉 {data['dropped']} 题", data)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--out", type=Path, default=Path("runs/offline_v2"))
    parser.add_argument("--webqa", type=Path, default=WEBQA_SCORES, help="WebQA OFFICIAL_SCORES.jsonl")
    parser.add_argument("--okvqa-dir", type=Path, default=OKVQA_DIR, help="OK-VQA run directory")
    args = parser.parse_args(argv)
    try:
        results = run(args.webqa, args.okvqa_dir, args.seed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    runio.atomic_write_text(args.out / "results.json", runio.dumps_strict(results, ensure_ascii=False, indent=2) + "\n")
    runio.atomic_write_text(args.out / "REPORT.md", render(results))
    print((args.out / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

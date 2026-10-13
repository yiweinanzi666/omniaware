# Offline v2 exploratory check

Every number in this file is 探索性、旧管线、含已知 bug 的数据. Not for the paper main table.

## Reproduce

```
python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2
```

Seed used: 20261013. Exit code this run will return: 0.

## Empty answers (A1)

Numbers in this section are 探索性、旧管线、含已知 bug 的数据.

A row is empty if the prediction or the parsed answer is blank, or `raw_response` / the prediction is only a `Sources:` line, or the answer body is empty while a `Sources:` line is present. A non-empty answer followed by a Sources line is not empty.

| slice | empty / n | rate | exact raw `Sources: none` |
|---|---:|---:|---:|
| WebQA direct | 410/4966 | 0.08256141764 | 410 |
| WebQA always-static | 447/4966 | 0.09001208216 | 447 |
| OK-VQA direct | 0/256 | 0 | 0 |
| OK-VQA static | 0/256 | 0 | 0 |

### Comparison with the cited ~10% N1 figure

The root-cause report (`Omni-Aware_根因溯源报告_20261013.md`) does not contain the N1 / `Sources: none` / 10% count. That figure is in the direction document, section 1.2: each WebQA arm has 433–551 questions (about 10%) whose raw output is only a `Sources: none` line. Those two numbers are quoted from that document. They are not measurements from this script.

This script measures WebQA `direct` 410/4966 (rate 0.08256141764) and `always-static` 447/4966 (rate 0.09001208216). The denominator is every row of that variant in `C:\rc\experiments\v318_missing_experiments_5h_root_20261008\pro6000_resume_20261009\official_scoring_pipeline\server_results_37028\results\full4966_scoring\seed17\OFFICIAL_SCORES.jsonl`, not the paired-question subset and not a Sources-filtered subset. Exact `raw_response` equal to `Sources: none` is 410 for direct and 447 for always-static. Direct 410 is below the cited minimum 433. always-static 447 falls inside 433–551. The same file's other variants are not used by the selector: `trigger-eab` exact `Sources: none` 512/4966 (A1 empty 513/4966), `trigger-replication` exact `Sources: none` 522/4966 (A1 empty 523/4966), `trigger-static` exact `Sources: none` 512/4966 (A1 empty 513/4966). The cited range is an across-arm summary; this file's direct arm is lower, so the rate is not forced to 10%. OK-VQA empty answers are direct 0/256 and static 0/256. That dump stores short strings in `preds` and has no citation `raw_response`, so it is a different field, a different variant key (`static`), and a different denominator (256 development questions, not the WebQA full set).

`QA_approx` is absent from the WebQA file. The file does contain `acc_approx_x_normalized_bart`. This script does not treat that field as `QA_approx` and does not use it as the selection objective.

## Splits

Numbers in this section are 探索性、旧管线、含已知 bug 的数据.

Thresholds are chosen only on the calibration half from `arbitration.split_half` (seed 20261013). The table is that split. `local_split_half` is the unit-test split and is not the split used for thresholds.

| dataset | n | calib | eval | disjoint |
|---|---:|---:|---:|---|
| webqa | 4966 | 2483 | 2483 | True |
| okvqa | 256 | 128 | 128 | True |

## Signals

WebQA direct margin is a proxy margin from u_initial: `1 - u_initial` when u is in [0, 1], otherwise `-u_initial`. Lower `u_initial` is treated as more confident. This is not a first-token logit margin. `static_mixed` uses that same proxy margin because the dump has no separate evidence confidence. `static_text`, `static_image`, and `rerank_static` are omitted. WebQA `entropy_norm`, sequence log-prob, and retriever top-1 are absent from the file. The selector requires numbers, so those missing features are passed as 0 (generation length 1). They add no information. Direct `kl_to_direct` is 0 because that action is the reference; the evidence arm's KL is missing and is also passed as absent (`None`, which the selector treats as 0).

WebQA `u_initial` on paired questions: min 0.2519305238, median 0.5175348493, max 1, n=4966. Proxy margin from u_initial: min 0, median 0.4824651507, max 0.7480694762, n=4966. Questions where the two arms' `u_initial` differ by more than 1e-9: 0.

OK-VQA uses `preds["direct"]` and `preds["static"]`. `static` answers equal `static_joint` on 256/256. Official scores use the field `vqa_accuracy` (static vqa_accuracy equals static_joint on 256/256). Direct proxy margin is `-h_sem` from `raw_pre`. Evidence proxy margin is `-h_sem` from `raw_post`. `entropy_norm` is `h_sem` clipped to [0, 1]. Retriever top-1 for the evidence arm is `max(rel)` when `rel` is present. `h_grd` is in the file and is not a calibration feature. `gold_in_topk` is not passed to the selector. Sequence log-prob is absent, so the `seq_logprob / gen_len` feature is 0.

OK-VQA h_sem pre min 7.96999719e-08, median 0.01922335299, max 0.1850796157, n=256; h_sem post min 4.693214373e-08, median 0.01158766154, max 0.2043742546, n=256; direct proxy margin min -0.1850796157, median -0.01922335299, max -7.96999719e-08, n=256; evidence proxy margin min -0.2043742546, median -0.01158766154, max -4.693214373e-08, n=256.

WebQA selection objective is `acc_approx`. OK-VQA selection objective is the official `vqa_accuracy` already in `okvqa_gs.official_scores.jsonl`. No new metric is computed. `normalized_bart_score` is a secondary column on WebQA when a policy is scored. v2-Cal 0/1 labels, fit on the calibration half only, are `acc_approx >= 0.5` on WebQA (the repo correct-at 50 rule applied to a 0–1 score) and `vqa_accuracy >= 50` on OK-VQA. Reported means use the official scores, not these labels.

## Selector

Numbers in this section are 探索性、旧管线、含已知 bug 的数据.

### webqa

Numbers in this section are 探索性、旧管线、含已知 bug 的数据.

Objective: `acc_approx`. Questions: 4966. Calib 2483 / eval 2483. Best single action chosen on the calib half: `static_mixed`. v2-Rule decision: `{"name": "best_single", "action": "static_mixed", "fallback": true}`. Candidate: `null`.

| quantity | value |
|---|---:|
| eval direct mean | 0.4127519215 (n=2483) |
| eval static_mixed mean | 0.4132004936 (n=2483) |
| eval best single mean | 0.4132004936 (n=2483) |
| eval v2-Rule mean | 0.4132004936 (n=2483) |
| eval oracle mean | 0.545261195 (n=2483) |
| utility(policy, best) | 0 |
| oracle-gap fraction | 0 |
| oracle-gap numerator | 0 |
| oracle-gap denominator | 327.9067214 |
| secondary normalized BART, direct | 0.1742469702 (n=2483) |
| secondary normalized BART, static_mixed | 0.1742841034 (n=2483) |
| secondary normalized BART, v2-Rule | 0.1742841034 (n=2483) |

v2-Rule action counts on eval: `{"static_mixed": 2483}`.

Paired comparison from `summarize_policy` (policy vs best single on the eval half):

```json
{
  "baseline": 0.4132004936465373,
  "ours": 0.4132004936465373,
  "delta_pp": 0.0,
  "relative_percent": 0.0,
  "paired_ci95_pp": [
    0.0,
    0.0
  ],
  "p_two_sided": 1.0,
  "wins": 0,
  "losses": 0,
  "ties": 2483
}
```

v2-Cal status: `ok`.

v2-Cal parameters: n_params=7, n_weights=7. Fit on the calib half only.

| quantity | value |
|---|---:|
| eval v2-Cal mean | 0.4127519215 (n=2483) |
| utility(policy, best) | -0.0004485721272 |
| oracle-gap fraction | -0.003396711685 |
| secondary normalized BART, v2-Cal | 0.1742469702 (n=2483) |

```json
{
  "baseline": 0.4132004936465373,
  "ours": 0.41275192151931644,
  "delta_pp": -0.0004485721272208135,
  "relative_percent": -0.10856040448115582,
  "paired_ci95_pp": [
    -0.0184402702861263,
    0.017099027685546546
  ],
  "p_two_sided": 0.9582083583283343,
  "wins": 591,
  "losses": 550,
  "ties": 1342
}
```

Eval means by subset (`acc_approx`):

```json
{
  "text": {
    "direct": {
      "n": 1195,
      "n_missing": 0,
      "mean": 0.4220906020988302,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "static_mixed": {
      "n": 1195,
      "n_missing": 0,
      "mean": 0.5278283224828569,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "v2_rule": {
      "n": 1195,
      "n_missing": 0,
      "mean": 0.5278283224828569,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "oracle": {
      "n": 1195,
      "n_missing": 0,
      "mean": 0.6150559869335577,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "best_single": {
      "n": 1195,
      "n_missing": 0,
      "mean": 0.5278283224828569,
      "label": "探索性、旧管线、含已知 bug 的数据"
    }
  },
  "img": {
    "direct": {
      "n": 1288,
      "n_missing": 0,
      "mean": 0.4040875400810255,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "static_mixed": {
      "n": 1288,
      "n_missing": 0,
      "mean": 0.3068493636314734,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "v2_rule": {
      "n": 1288,
      "n_missing": 0,
      "mean": 0.3068493636314734,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "oracle": {
      "n": 1288,
      "n_missing": 0,
      "mean": 0.4805059337708267,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "best_single": {
      "n": 1288,
      "n_missing": 0,
      "mean": 0.3068493636314734,
      "label": "探索性、旧管线、含已知 bug 的数据"
    }
  }
}
```

### okvqa

Numbers in this section are 探索性、旧管线、含已知 bug 的数据.

Objective: `vqa_accuracy`. Questions: 256. Calib 128 / eval 128. Best single action chosen on the calib half: `direct`. v2-Rule decision: `{"name": "margin_tau-1e9_theta1", "tau1": -1000000000.0, "theta": 1.0, "confidence": "margin", "fallback": false, "utility": 0.0}`. Candidate: `{"name": "margin_tau-1e9_theta1", "tau1": -1000000000.0, "theta": 1.0, "confidence": "margin"}`.

| quantity | value |
|---|---:|
| eval direct mean | 58.59375 (n=128) |
| eval static_mixed mean | 32.03125 (n=128) |
| eval best single mean | 58.59375 (n=128) |
| eval v2-Rule mean | 58.59375 (n=128) |
| eval oracle mean | 60.625 (n=128) |
| utility(policy, best) | 0 |
| oracle-gap fraction | 0 |
| oracle-gap numerator | 0 |
| oracle-gap denominator | 260 |
| secondary normalized BART, direct | null (n=0) |
| secondary normalized BART, static_mixed | null (n=0) |
| secondary normalized BART, v2-Rule | null (n=0) |

v2-Rule action counts on eval: `{"direct": 128, "__used_evidence__": 0}`.

Paired comparison from `summarize_policy` (policy vs best single on the eval half):

```json
{
  "baseline": 58.59375,
  "ours": 58.59375,
  "delta_pp": 0.0,
  "relative_percent": 0.0,
  "paired_ci95_pp": [
    0.0,
    0.0
  ],
  "p_two_sided": 1.0,
  "wins": 0,
  "losses": 0,
  "ties": 128
}
```

v2-Cal status: `ok`.

v2-Cal parameters: n_params=7, n_weights=7. Fit on the calib half only.

| quantity | value |
|---|---:|
| eval v2-Cal mean | 58.125 (n=128) |
| utility(policy, best) | -0.46875 |
| oracle-gap fraction | -0.2307692308 |
| secondary normalized BART, v2-Cal | null (n=0) |

```json
{
  "baseline": 58.59375,
  "ours": 58.125,
  "delta_pp": -0.46875,
  "relative_percent": -0.8,
  "paired_ci95_pp": [
    -1.40625,
    0.0
  ],
  "p_two_sided": 0.6186762647470506,
  "wins": 0,
  "losses": 1,
  "ties": 127
}
```

Eval means by subset (`vqa_accuracy`):

```json
{
  "all": {
    "direct": {
      "n": 128,
      "n_missing": 0,
      "mean": 58.59375,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "static_mixed": {
      "n": 128,
      "n_missing": 0,
      "mean": 32.03125,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "v2_rule": {
      "n": 128,
      "n_missing": 0,
      "mean": 58.59375,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "oracle": {
      "n": 128,
      "n_missing": 0,
      "mean": 60.625,
      "label": "探索性、旧管线、含已知 bug 的数据"
    },
    "best_single": {
      "n": 128,
      "n_missing": 0,
      "mean": 58.59375,
      "label": "探索性、旧管线、含已知 bug 的数据"
    }
  }
}
```


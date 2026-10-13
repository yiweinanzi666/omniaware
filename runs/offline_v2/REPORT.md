# Offline v2 exploratory check

Every number in this file is 探索性、旧管线、含已知 bug 的数据. Not for the paper main table.

## Reproduce

```
python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2
```

Seed used: 20261013. Exit code this run will return: 2.

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

Thresholds were not chosen. `arbitration.split_half` is not available in this worktree. The table is `local_split_half`: sort the question ids, shuffle with `random.Random(seed)`, and give the first half to calib and the rest to eval. That split was not used to fit a selector.

| dataset | n | calib | eval | disjoint |
|---|---:|---:|---:|---|
| webqa | 4966 | 2483 | 2483 | True |
| okvqa | 256 | 128 | 128 | True |

## Signals

WebQA direct margin is a proxy margin from u_initial: `1 - u_initial` when u is in [0, 1], otherwise `-u_initial`. Lower `u_initial` is treated as more confident. This is not a first-token logit margin. `static_mixed` uses that same proxy margin because the dump has no separate evidence confidence. `static_text`, `static_image`, and `rerank_static` are omitted. WebQA `entropy_norm`, sequence log-prob, and retriever top-1 are missing and are passed as missing, not filled in. Direct `kl_to_direct` is 0 because that action is the reference; the evidence arm's KL is missing.

WebQA `u_initial` on paired questions: min 0.2519305238, median 0.5175348493, max 1, n=4966. Proxy margin from u_initial: min 0, median 0.4824651507, max 0.7480694762, n=4966. Questions where the two arms' `u_initial` differ by more than 1e-9: 0.

OK-VQA uses `preds["direct"]` and `preds["static"]`. `static` answers equal `static_joint` on 256/256. Official scores use the field `vqa_accuracy` (static vqa_accuracy equals static_joint on 256/256). Direct proxy margin is `-h_sem` from `raw_pre`. Evidence proxy margin is `-h_sem` from `raw_post`. `entropy_norm` is `h_sem` clipped to [0, 1]. Retriever top-1 for the evidence arm is `max(rel)` when `rel` is present. `h_grd` is in the file and is not a calibration feature. `gold_in_topk` is not passed to the selector. Sequence log-prob is absent, so the `seq_logprob / gen_len` feature is left missing.

OK-VQA h_sem pre min 7.96999719e-08, median 0.01922335299, max 0.1850796157, n=256; h_sem post min 4.693214373e-08, median 0.01158766154, max 0.2043742546, n=256; direct proxy margin min -0.1850796157, median -0.01922335299, max -7.96999719e-08, n=256; evidence proxy margin min -0.2043742546, median -0.01158766154, max -4.693214373e-08, n=256.

WebQA selection objective is `acc_approx`. OK-VQA selection objective is the official `vqa_accuracy` already in `okvqa_gs.official_scores.jsonl`. No new metric is computed. `normalized_bart_score` is a secondary column on WebQA when a policy is scored. v2-Cal 0/1 labels, fit on the calibration half only, are `acc_approx >= 0.5` on WebQA (the repo correct-at 50 rule applied to a 0–1 score) and `vqa_accuracy >= 50` on OK-VQA. Reported means use the official scores, not these labels.

## Selector

Numbers in this section are 探索性、旧管线、含已知 bug 的数据.

arbitration.py is not in this worktree yet.

v2-Rule, v2-Cal, paired CIs, and the oracle-gap fraction were not computed. This file does not contain those numbers.


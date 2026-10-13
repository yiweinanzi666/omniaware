# v2 离线探索（旧管线）

**探索性数字，不进论文主表。** 数据来自含已知 bug 的旧管线：WebQA 官方分数的 direct 与 always-static 变体，OK-VQA 开发运行 tf32_ce_calibrated_k5_dev256。只用于检查选择器代码和信号方向。

复现：`python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2`。对排序后的 qid 做 numpy.random.default_rng(20261013).permutation，前一半校准，后一半评测；δ=0.1，ε 取 0 与 1（0–100 量纲）。配对差是 评测半 相对最强单一动作 的差值，单位为分，CI 为 95% 百分位区间。

## webqa（校准 2483 题，评测 2483 题）

最强单一动作（校准半选出）：static_mixed。

| 策略 | 评测半均分 | 配对差 [95% CI] | 相对提升 | 上限差距比例 | 被选动作计数 |
|---|---:|---:|---:|---:|---|
| direct | 41.28 | -0.04 [-1.90, 1.78] | -0.11% | -0.00 | direct 2483 |
| static_mixed | 41.32 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | static_mixed 2483 |
| 最强单一动作 static_mixed | 41.32 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | static_mixed 2483 |
| 朴素基线 | 43.31 | +1.99 [0.96, 3.01] | +4.81% | 0.15 | direct 1013，static_mixed 1470 |
| 逐题上限 | 54.53 | +13.21 [12.10, 14.36] | +31.96% | 1.00 | direct 1933，static_mixed 550 |
| v2-Rule ε=0 | 43.31 | +1.99 [0.96, 3.01] | +4.81% | 0.15 | direct 1013，static_mixed 1470 |
| v2-Rule ε=1 | 43.31 | +1.99 [0.96, 3.01] | +4.81% | 0.15 | direct 1013，static_mixed 1470 |
| v2-Cal ε=0 | 41.32 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | static_mixed 2483 |
| v2-Cal ε=1 | 43.75 | +2.43 [1.28, 3.59] | +5.88% | 0.18 | direct 1565，static_mixed 918 |

决定的关键字段：
- 朴素基线：tau=0.62
- v2-Rule ε=0：kind=rule evidence=best tau=0.62 n_accepted=22
- v2-Rule ε=1：kind=rule evidence=best tau=0.62 n_accepted=28
- v2-Cal ε=0：回退为 static_mixed
- v2-Cal ε=1：kind=cal lam=-3.23 n_accepted=18
  - direct：保留特征 ['conf_direct', 'conf_star', 'token_f1_to_star']，丢弃 ['conf_gap', 'kl_to_direct', 'retriever_top1']，参数 4 个

v2-Cal ε=1 对内部拆分种子的敏感性（种子 0–19）：评测半均分 最小 42.68，中位 43.30，最大 43.61。

空答案率（全部题目）：direct 410/4966 = 0.08；static_mixed 447/4966 = 0.09

## okvqa（校准 128 题，评测 128 题）

最强单一动作（校准半选出）：direct。

| 策略 | 评测半均分 | 配对差 [95% CI] | 相对提升 | 上限差距比例 | 被选动作计数 |
|---|---:|---:|---:|---:|---|
| direct | 58.59 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | direct 128 |
| static_mixed | 32.03 | -26.56 [-34.69, -18.91] | -45.33% | -13.08 | static_mixed 128 |
| 最强单一动作 direct | 58.59 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | direct 128 |
| 朴素基线 | 58.59 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | direct 127，static_mixed 1 |
| 逐题上限 | 60.62 | +2.03 [0.31, 4.38] | +3.47% | 1.00 | direct 124，static_mixed 4 |
| v2-Rule ε=0 | 58.59 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | direct 128 |
| v2-Rule ε=1 | 58.59 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | direct 128 |
| v2-Cal ε=0 | 58.59 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | direct 128 |
| v2-Cal ε=1 | 58.59 | +0.00 [0.00, 0.00] | +0.00% | 0.00 | direct 128 |

决定的关键字段：
- 朴素基线：tau=-0.18
- v2-Rule ε=0：回退为 direct
- v2-Rule ε=1：回退为 direct
- v2-Cal ε=0：回退为 direct
- v2-Cal ε=1：回退为 direct

v2-Cal ε=1 对内部拆分种子的敏感性（种子 0–19）：评测半均分 最小 58.59，中位 58.59，最大 58.59。

空答案率（全部题目）：direct 0/256 = 0.00；static_mixed 0/256 = 0.00


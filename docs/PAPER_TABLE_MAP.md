# v2 论文表格与命令的对应

本表对应 `paper/main.tex`、`paper/sec_experiments.tex`、`paper/sec_appendix.tex` 的当前稿。v2 实验**还没有跑**。表中的数字在论文里都是 `\ph`。脚本由另一分支实现；下面的命令是接口，即使脚本尚未合入本分支，也按这些命令填写。

当前稿**没有图**。`paper/fig/fig2.pdf` 和 `paper/fig/fig3.pdf` 画的是 v1 控制器（U 门控、路由、EAB），v2 稿不收入，因此没有对应的论文图。

`offline_v2` **不是**下表任何一行的来源：

```bash
python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2
```

该命令的结果是探索性的，来自旧管线，含已知 bug。

## 命令接口

```bash
python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50
python scripts/v2_runs.py --stage s1 --dataset okvqa --n 50
python scripts/v2_runs.py --stage s2 --dataset webqa_full
python scripts/v2_runs.py --stage s2 --dataset webqa_open_img
python scripts/v2_runs.py --stage s2 --dataset okvqa
python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions docs/dev_decisions.json
python scripts/v2_runs.py --stage s4 --dataset okvqa --decisions docs/dev_decisions.json
python scripts/analysis/v2_gate.py --scores <dev-scores.json>
python scripts/analysis/image_evidence_check.py --n 50
```

`<dev-scores.json>` 是 S2 写出的开发集官方分数文件，不是论文主表。

GPU 时间为估计值，不是实测：S1 约 0.5 天、1 张 GPU；S2 约 4–8 小时、1 张 A100；S4 约 1–2 天、1–2 张 A100。

## 表

| 论文位置 | 标签 | 产出命令 | 说明 |
|---|---|---|---|
| 度量研究，WebQA（acc_approx、归一化 BART、QA_approx；全体 / 图像 / 文本） | `tab:webqa` | `python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions docs/dev_decisions.json` | 主设置是 distractor。图像列和文本列是同一次 `webqa_full` 输出的子集，不是另一次打乱。强基线的身份来自校准集，写在 `docs/dev_decisions.json`，不在测试集上重选 |
| 度量研究，OK-VQA（官方 VQA accuracy） | `tab:okvqa` | `python scripts/v2_runs.py --stage s4 --dataset okvqa --decisions docs/dev_decisions.json` | 只填官方 VQA accuracy |
| 主表（相对强基线与相对 `static_mixed`，含退化比例） | `tab:main` | 上两条 S4 命令 | 动作：`direct`、`static_text`、`static_image`、`static_mixed`、`rerank_static`。对照：NTE-trigger、SeaKR、v1 的 U 门控、UniversalRAG。方法：v2-Rule、v2-Cal。EAB 不是这张表的臂。相对提升的分母是校准集上最好的单一动作。区间是按题配对 bootstrap 的 95% CI，多重比较用 Holm。贪心解码不报 seed 标准差 |
| 消融（逐个去掉信号、v2-Rule 与 v2-Cal、\(\delta\) 敏感性） | `tab:ablation` | 同上两条 S4；阈值与系数来自 `docs/dev_decisions.json` | 没有单独的消融命令。特征在 S4 保存的信号上重算选择，不在测试集上重选网格。\(\delta\) 敏感性只描述开发集或校准集，报告点是冻结决定 |
| 开放检索（第二设置） | `tab:open` | `python scripts/v2_runs.py --stage s2 --dataset webqa_open_img` | 当前接口**没有**该设置的 S4。此表不是冻结测试。单元格仍是 `\ph` |
| 成本（调用次数、输入 token、延迟） | `tab:cost` | 两条 S4 命令的运行记录 | 论文与 always-`static_mixed` 和 SeaKR 比较。记录里没有计时字段就不填，不把估计的 GPU 时长写进表。省钱版（只在 direct 信号低于阈值时解码证据动作）同样来自这些运行的配置，未跑则为 `\ph` |
| 开发集 go/no-go（**不是论文表**） | — | `python scripts/analysis/v2_gate.py --scores <dev-scores.json>` | 判据见 `docs/PROTOCOL.md`。PASS/FAIL 不是主表数字 |
| 冒烟（**不是论文表**） | — | 两条 `--stage s1 --n 50` 命令 | 空答案率、格式一致性、图像诊断的管线检查 |
| 图像证据诊断（度量研究中的一段，没有数字表） | 正文指向 `image_evidence_check` | `python scripts/analysis/image_evidence_check.py --n 50` | 50 是命令里的题数，不是准确率。结论未产出，论文写 `\ph` |

## 分析小节

| 论文位置 | 来源 | 说明 |
|---|---|---|
| Attention steering moves attention, not answers | **无 v2 命令**。数字只引用 2026-10-13 根因报告里已经印出的数，并在正文标明出处 | 不是 S1/S2/S4 的结果，也不是 `offline_v2`。EAB 不作为主方法臂 |
| 失败案例与格式敏感性 | 两条 S4 命令 | 未跑之前全部是 `\ph`。提示词在开发集上冻结，不在测试集上挑选 |

## 附录

| 段落 | 来源 |
|---|---|
| 切分、指标、强基线、预注册 | `docs/PROTOCOL.md` 与 `docs/dev_decisions.json`。测试命令即上表 S4。不从测试集网格填数 |
| 退役的 EAB 局部恒等式 | 数学推导，不是实验表，没有对应的 `v2_runs.py` 阶段 |
| WebQA 评分定义 | 官方 acc_approx、归一化 BART、QA_approx 的定义。数值仍来自 S4，未跑则为 `\ph` |
| 补充占位（开放检索、成本、退化比例） | 与 `tab:open`、`tab:cost`、`tab:main` 相同的命令 |

泛化实验（MMQA、视频）在章节结构里保留为未跑的 `\ph`。当前命令接口没有对应阶段，本表不编造命令。

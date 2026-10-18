# v2 论文表格与命令的对应

本表对应 `paper/main.tex`、`paper/sec_experiments.tex`、`paper/sec_appendix.tex` 的当前稿。v2 实验**还没有跑**。表中的数字在论文里都是 `\ph`。

`offline_v2` **不是**下表任何一行的来源。它是探索性的，来自旧管线，含已知 bug：

```bash
python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2
```

当前稿**没有图**。`paper/fig/fig2.pdf` 和 `paper/fig/fig3.pdf` 画的是 v1 控制器（U 门控、路由、EAB），v2 稿不收入，因此没有对应的论文图。

## 命令接口

阶段命令由 `scripts/v2_runs.py` 生成（`--dry-run` 只打印，不执行）。选择、评分与拼表步骤由 `scripts/analysis/v2_table.py` 的子命令完成。`v2_table.py` 由另一子任务实现，合入本分支之前这些子命令不存在；命令形式以 `v2_runs.py --dry-run` 的输出为准。

```bash
# S1：每个数据集 50 题冒烟（图像诊断命令只打印）
python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50
python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --n 50
# S2：校准集与开发集的解码 -> 官方评分 -> build -> gate（开发集）-> select（校准集）
python scripts/v2_runs.py --stage s2 --dataset webqa_full
python scripts/v2_runs.py --stage s2 --dataset webqa_open_img
python scripts/v2_runs.py --stage s2 --dataset okvqa_gs
# S4：测试集的解码 -> 官方评分 -> build -> apply（冻结决定）
python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions runs/v2_decisions/webqa_full.json
python scripts/v2_runs.py --stage s4 --dataset okvqa_gs --decisions runs/v2_decisions/okvqa_gs.json
# 图像证据诊断（只打印，不执行 GPU 描述）
python scripts/analysis/image_evidence_check.py --n 50
```

`v2_table.py` 的四个子命令：

- `build --run runs/<tag>/<dataset>.jsonl`：读运行输出，对每个动作的答案做官方评分，写出逐题动作表 `<dataset>.v2_actions.jsonl`。
- `gate --table <dev>.v2_actions.jsonl`：开发集 go/no-go，+8% 判据，输出 PASS 或 FAIL。
- `select --table <calib>.v2_actions.jsonl --out runs/v2_decisions/<dataset>.json`：在校准集上生成冻结决定，主决定为 rule，附带 cal。
- `apply --table <test>.v2_actions.jsonl --decisions runs/v2_decisions/<dataset>.json --out <dataset>.v2_main.json`：把冻结决定应用到测试集，产出主表输入。不搜索网格。

`<dev-scores.json>` 这一写法已废止。开发集判据直接读 S2 的开发动作表。

GPU 时间为估计值，不是实测：S1 约 0.5 天、1 张 GPU；S2 约 4–8 小时、1 张 A100；S4 约 1–2 天、1–2 张 A100。

## 表

| 论文位置 | 标签 | 产出命令 | 说明 |
|---|---|---|---|
| 度量研究，WebQA（acc_approx、归一化 BART、QA_approx；全体 / 图像 / 文本） | `tab:webqa` | `v2_runs.py --stage s4 --dataset webqa_full --decisions runs/v2_decisions/webqa_full.json`，随后 `v2_table.py build`、`apply` | 主设置是 distractor。图像列和文本列是同一次 `webqa_full` 输出的子集，不是另一次打乱。强基线的身份来自校准集，写在冻结决定文件中，不在测试集上重选 |
| 度量研究，OK-VQA（官方 VQA accuracy） | `tab:okvqa` | `v2_runs.py --stage s4 --dataset okvqa_gs --decisions runs/v2_decisions/okvqa_gs.json`，随后 `v2_table.py build`、`apply` | 只填官方 VQA accuracy。检索语料为 GS-112K（`okvqa_gs`） |
| 主表（相对强基线与相对 `static_mixed`，含配对 CI） | `tab:main` | 上两条 S4 命令 | 动作：`direct`、`static_text`、`static_image`、`static_mixed`、`rerank_static`。对照：NTE-trigger、SeaKR、v1 的 U 门控、UniversalRAG。方法：v2-Rule（主决定）、v2-Cal。EAB 不是这张表的臂。相对提升的分母是校准集上最好的单一动作。区间是按题配对 bootstrap 的 95% CI，多重比较用 Holm。贪心解码不报 seed 标准差。风险控制列是平均效用（配对差均值）及其 CI，不是退化比例 |
| 消融（逐个去掉 v2-Cal 的一个特征、v2-Rule 与 v2-Cal、δ 与 ε 的开发集描述） | `tab:ablation` | 同上两条 S4 与 S2 的 `select` | 每个消融变体是对校准动作表重跑 `v2_table.py select`（去掉一个特征或换一个决定族），再 `apply`。该变体开关尚未在 `v2_table.py` 中定义，实现之前相应单元格保持 `\ph`。δ 与 ε 的行只描述开发集或校准集，报告点是冻结决定 |
| 开放检索（第二设置） | `tab:open` | `python scripts/v2_runs.py --stage s2 --dataset webqa_open_img` | 当前接口**没有**该设置的 S4。此表不是冻结测试。单元格仍是 `\ph` |
| 成本（调用次数、输入 token、延迟） | `tab:cost` | 两条 S4 命令的运行记录 | 论文与 always-`static_mixed` 和 SeaKR 比较。记录里没有计时字段就不填，不把估计的 GPU 时长写进表。省钱版（只在 direct 信号低于阈值时解码证据动作）同样来自这些运行的配置，未跑则为 `\ph` |
| 开发集 go/no-go（**不是论文表**） | — | S2 自动调用的 `v2_table.py gate --table runs/v2_s2_<dataset>_dev/<dataset>.v2_actions.jsonl` | 判据见 `docs/PROTOCOL.md` 与 `docs/DEV_DECISIONS.md`。PASS/FAIL 不是主表数字 |
| 冒烟（**不是论文表**） | — | 两条 `--stage s1 --n 50` 命令 | 空答案率、格式一致性、图像诊断的管线检查 |
| 图像证据诊断（度量研究中的一段，没有数字表） | 正文指向 `image_evidence_check` | `python scripts/analysis/image_evidence_check.py --n 50` | 50 是命令里的题数，不是准确率。结论未产出，论文写 `\ph` |

## 分析小节

| 论文位置 | 来源 | 说明 |
|---|---|---|
| Attention steering moves attention, not answers | S2 开发集与 S4 运行中的 EAB 分析臂（`--analysis-eab`，DEV_DECISIONS 2026-10-15 修订）。答案层面：`<数据集>.official_summary.json` 里 `eab:g1.2:ans1:pos:gate` 与 `static` 的官方分数，以及逐题 `preds` 中两臂答案不同的题数。注意力层面：逐题记录 `ard` 里这两个臂的 ARD。2026-10-13 根因报告里的数来自旧管线，引用时在正文标明出处 | EAB 臂只由 `score_official` 评分，不进入 `v2_table.py` 的任何一步，也不在主表里。取值为论文值（γ = 1.2、ans1、pos、门控），没有在开发集上选择；输入是 `static` 的联合 top-k，不另开探测轮。测试集上的数要等 S4 跑完，此前写 `\ph`。EAB 不作为主方法臂，也不是 `offline_v2` 的结果 |
| 失败案例与格式敏感性 | 两条 S4 命令 | 未跑之前全部是 `\ph`。提示词在开发集上冻结，不在测试集上挑选 |

## 附录

| 段落 | 来源 |
|---|---|
| 切分、指标、强基线、预注册 | `docs/PROTOCOL.md` 与 `docs/DEV_DECISIONS.md`（v2 预注册）。冻结决定为 `runs/v2_decisions/<dataset>.json`。测试命令即上表 S4。不从测试集网格填数 |
| 退役的 EAB 局部恒等式 | 数学推导，不是实验表，没有对应的 `v2_runs.py` 阶段 |
| WebQA 评分定义 | 官方 acc_approx、归一化 BART、QA_approx 的定义。数值仍来自 S4，未跑则为 `\ph` |
| 补充占位（开放检索、成本、平均效用的配对 CI） | 与 `tab:open`、`tab:cost`、`tab:main` 相同的命令 |

泛化实验（MMQA、视频）在章节结构里保留为未跑的 `\ph`。当前命令接口没有对应阶段，本表不编造命令。

# Dev 阶段预先登记的决策规则（2026-10-11，在看到任何 dev 结果之前写定）

目的：避免在测试集上挑选设置，也避免事后按结果选择论文主张。规则由 `scripts/analysis/dev_select.py` 机械执行，结果写入 `runs/dev_decisions.json`，`scripts/paper_runs.py` 的所有 test 阶段读取它。

## 运行

```bash
python -m scripts.paper_runs --stage dev --run      # 每个数据集 --split dev（默认 256 题）
python -m scripts.analysis.dev_select               # -> runs/dev_decisions.json
```

dev 网格：
- **EAB 形式（6 种）**：起始行 {ans1（从预测第一个答案 token 的行开始，论文默认）、ans2（decode-only）、q（从问题开始）} × 偏置形式 {pos：γUr，ctr：γU(2r−1)}。
- **不确定性设置（3 种）**：sum + 末层（论文）、zsum（两项各自标准化）+ 末层、sum + 中间层。
- **OK-VQA 检索语料（3 种）**：BM25 维基（`okvqa`）、BGE 维基 3M（`okvqa_dense`）、BGE GS-112K（`okvqa_gs`）。
- 冷启动数据集（WebQA、HybridQA、VideoRAG-Synth）一律使用探测轮（`--probe-uncertainty --first-round-routing`）。

## 规则

1. **EAB 形式只选一种，全局通用。** 指标是同一输入下 EAB 相对 static 的逐题增益：先在每个数据集内取平均，再对数据集取平均（每个数据集权重相同）。选平均增益最大的形式。"EAB 有效"当且仅当分层 bootstrap 的 95% 区间下界大于 0。不允许按数据集分别选择。
2. **不确定性设置。** 指标是门控分数与直答错误的 Spearman ρ，跨数据集取平均；选 ρ 最大的设置。"U 优于 NTE"当且仅当所选设置的平均 ρ 大于下一 token 熵的平均 ρ。
3. **OK-VQA 语料。** 选 static − direct 差值最大的语料。"检索有用"当且仅当这个差值 ≥ 0；否则 OK-VQA 在论文中作为"检索可能有害、门控用来避免伤害"的场景报告，不作为增益来源。
4. **其余超参数不再调整。** (τ, τ_b, β, γ, δ) = (0.35, 0.40, 0.30, 1.2, 0.08)、k = 5 保持论文值；β、τ_b 的敏感性只在附录报告。
5. **最佳候选的乐观偏差。** dev 上的选择存在 best-of-K 偏差（EAB 有 6 个候选），所以论文中的所有数字只来自 test，dev 数字不进正文。

## 论文框架（由规则 1、2 机械决定）

| framing | 条件 | 论文主张 |
|---|---|---|
| 1 | EAB 有效，且 U 优于 NTE | 完整主张：一次 prefill 统一决定是否检索、往哪检索，以及如何用证据 |
| 2 | 两者只有一项成立 | 主张收缩到成立的部分；另一部分改为分析，如实报告负结果 |
| 3 | 两者都不成立 | 不投 WWW 2027 正会（改投短文或下一个会议） |

结果出来后，规则只能用于决定，不能修改。若需修改规则，必须在本文件中注明修改时间和原因，并在论文附录中披露。

# 实验协议

本协议针对旧实验中发现的问题制定（审查报告 2026-10-10，未纳入仓库；问题编号 F/S/M 见修复记录）。论文里的任何结果都必须按本协议产生。协议已按 2026-10-10 的统一决策 D1–D18 更新：论文和本文件描述的都是这些决策实施后的代码行为。

## 1. 数据划分

| 数据集（`--dataset`） | 评测数据（test） | calib | dev | 说明 |
|---|---|---|---|---|
| `okvqa` / `okvqa_dense` / `okvqa_gs` | OK-VQA val 5,046 − 500 − 256 = **4,290** | 500（保留区，`heldout`） | 256 | 三种语料候选之一；默认 `okvqa_gs`（GS 训练语料 112,724 句）；`okvqa_gs` 的全量 168,306 句版只在附录作对照 |
| `webqa_img` / `webqa_full`（旧名，仅用于历史运行） | WebQA val 图像题 2,511 / 全部 4,966 | 训练集前 `n_calib` 题 | 训练集接下来的 256 题 | **distractor 设置**：每题自带的正负事实池，不是开放检索 |
| `webqa_open_img` / `webqa_open_full` | 同上（**全部 4,966 道 val 为 test**；`webqa_open_img` 为同一划分下的图像题子集） | 训练集前 `n_calib` 题（需要先为 train 生成检索结果） | 训练集接下来的 256 题 | **开放检索**：在全局文本池和图片池中取 BGE 文本 top-5 与 CLIP 图像 top-5；论文的"Web 检索"设定以此为准 |
| `hybridqa` | UniversalRAG 套件 HybridQA 2,000 − 500 − 256 = **1,244** | 500（保留区，`heldout`） | 256 | 候选为 UniversalRAG 检索器在 HybridQA 段落池与表格池中各取的 top-5（所有臂共享）；语料为套件子集（仅 HybridQA 段落与表格） |
| `videorag_synth` | UniversalRAG 套件 VideoRAG-Synth 374 − 100 − 50 = **224** | 100（保留区，`heldout`） | 50 | 候选为整段视频与视频片段各 top-5（两者为独立模态）；每个候选 = 均匀采样帧 + 字幕；语料为可下载视频的子集 |

已删除的设置：MMQA-table（每题自带 gold 表格，检索不起作用）、Video-MME（证据固定，不涉及检索）、InfoSeek 与 VideoRAG-WikiHow（2026-10-11 决定）。

- `--split-seed`（默认 2027）把数据打乱后，依次取出 calib、dev、test：`calib = order[:CALIB_RESERVE][:n_calib]`，`dev = order[CALIB_RESERVE:CALIB_RESERVE+N_DEV]`，`test = order[CALIB_RESERVE+N_DEV:]`（D1）。因此 **test 与 `--n-calib` 无关**。
- WebQA 的 calib 与 dev 都取自 train（D2）：train 打乱后 `calib = 前 n_calib`，`dev = 接下来的 256`；val 全部作为 test，`webqa_open_img` 的 test 是同一划分下的图像题子集，全量版和图像版不再各自独立打乱 val。
- `--n-calib` 默认取该数据集的保留区大小；超过保留区就报错（因此 OK-VQA 不能跑 1,000 条校准，见 §2）。评测集为空也报错。运行时断言 calib、dev、test 两两不交。
- `--split dev|test`：dev 阶段只跑 `--split dev`；test 阶段只跑 `--split test`，并读取 `runs/dev_decisions.json`（见 `docs/DEV_DECISIONS.md`）。
- 调参（τ、τ_b、β、γ、k、提示词、帧数等）**只看 dev**。test 在设置冻结后只跑一次。
- 在 test 上看过结果再改设置，那么这一轮 test 只能算 dev，必须重新划分或换数据集确认。

## 2. 校准

- 校准用 `--n-calib` 个**无标签**问题，估计 U 的 median/IQR、探测轮统计量 S，以及各模态检索分数的 ECDF。默认值为保留区大小（OK-VQA、HybridQA 500；VideoRAG-Synth 100；WebQA 为训练集上的 `n_calib`）。
- 校准过程不读取答案，也不按准确率选阈值（旧的 `--select-tau` 已删除）。
- 校准统计量（median、IQR、ECDF、`norm_probe`）写入运行配置；续跑时校准统计量也必须一致。
- 论文中的校准规模消融只用 100 和 500（保留区上限）。
- 校准迁移（`--norm-from`）只在冷启动数据集之间进行（WebQA-img、HybridQA、VideoRAG-Synth），它们共享探测轮统计量。来源没有探测统计量（例如 OK-VQA 无探测轮）时必须报错，不得静默沿用目标集自己的统计量。
- 校准缓存按数据集、模型、k、输入选项和校准来源分别存放。

## 3. 对照臂（同一次运行、同一提示词，题目相同）

| 臂 | 含义 |
|---|---|
| `direct` | 不加检索证据（只保留问题自带的图片或表格上下文） |
| `static_joint` | **不探测、不路由**的联合 top-k 静态拼接（D4）。论文主表的 "Static concatenation" 行对应此臂 |
| `static` | 路由后的静态拼接（Omni-Aware 的输入，无 EAB、无门控）。OK-VQA 不路由，此时与 `static_joint` 相同 |
| `<eab 变体>` | 在 `static` 的同一输入上加 EAB（默认 `eab:g1.2:ans1:pos:gate`，即论文默认 EAB 形式；DEV 冻结后以冻结形式为准） |
| `eab:...:ans2:...`（decode-only） | 从 t_ans 行开始加偏置；第一个答案 token 不受影响（见 `tests/test_qwen_eab.py`） |
| `pasta:*` | 不区分相关度的统一偏置（PASTA 风格对照） |
| `gated:static` | U ≤ τ 时用 direct，否则用 `static`（论文的 "U_omni, static"） |
| `gated:<eab 变体>` | 完整系统 |
| `replication`、`gated:replication` | 复制对照：每块复制 `max(1, round(exp(γ·U·r)))` 份，**不截断**，并记录 prompt token 数（D11）；门控与 EAB 相同 |

论文主表中的对照方法（都在同一次运行里、用相同候选池、k、解码上限和提示词格式生成）：
- **Cross-encoder + static**：`--selector ce`，取 `static_joint` 臂（CE 选出的联合 top-k）。
- **Found-in-the-Middle**（`--fitm`）、**M2-RAG**（`--m2rag`）：输入均为联合 top-k（D4）。M2-RAG 含问题图像；其官方单阶段摘要之后接一个答案抽取器（项目新增的一步，见论文附录）。M2-RAG 不支持视频。
- **SeaKR**（`--seakr-samples 20`）：在全量 test 上运行（D5）；触发阈值在校准集上与 U 的检索率对齐，检索后选特征值得分最低的文档作答。
- **UniversalRAG**（`--route-rule query --query-routes`）：路由器用其官方脚本训练，训练数据先剔除我们的评测题。clip 与 video 是两个独立模态（D6）；路由标签全都不在语料中时按"不检索"处理（取 direct 的答案）并记入计数；遇到未知标签报错。候选顺序用套件自带检索器的分数，而不是 Omni 的 rel。OK-VQA 一列在论文中标 "—"（router 只看文本查询，语料只有一种模态）。WebQA-img 一列从 `webqa_open_full` 的 UniversalRAG 运行结果按图像题过滤得到。
- **全部外部基线**（CE、FiTM、M2-RAG、SeaKR）都使用联合 top-k 输入（`static_joint` 的输入）。**Omni-Aware 本身及其消融**（routed static、EAB 变体、replication、noprobe）使用路由后的输入。
- 消融表中的对照：NTE 触发（`nte_trigger:static`、`nte_shared:static`；NTE 阈值为与 U 检索率匹配的分位数）、PASTA 式统一偏置（`pasta:*`）、复制对照、各路由规则。
- 已删除、不得出现在结果表里的方法：DRAGIN、M2-RAG 多阶段、SAM-RAG、ViDoRAG、Visual-RAG。

- 要判断 EAB 本身的效果，比较 `<eab>` 与 `static`，或 `gated:<eab>` 与 `gated:static`；要判断检索门控的效果，比较 `gated:static` 与 `static` 和 `direct`；要判断路由的效果，比较 `static` 与 `static_joint`。
- 消融 noprobe = 去掉 `--probe-uncertainty`，但保留 `--first-round-routing`（D17）。
- `--no-calibration` 只把 Norm 改成默认统计量，不改变 relevance 归一化；要用 minmax 必须另传 `--relevance-norm minmax`（D16）。
- 提示词、帧数、答案格式（`--answer-style`）对所有臂同时生效，所以这些改动带来的提升**不能**算作方法贡献。如果要报告这类工程改进，单独列一行。

## 4. 评分与统计

```bash
python -m omniaware.eval.score_official runs/<tag>/<dataset>.jsonl
```

- 只用官方指标（见 CONVENTIONS §5）。WebQA validation 没有隐藏测试集的关键词，只能报告 acc_approx（以及 × BARTScore 的近似综合分），不能称为排行榜分数。
- 对每个臂与参考臂做**双侧配对 bootstrap**（5,000 次；VideoRAG-Synth 按源视频聚类重采样），输出 p 值（D18）。同时报告 **Holm 校正**后的 p 值。
- "最强基线"由代码按列自动确定：同一列中所有基线臂里平均分最高者。论文不得手工指定。
- greedy 解码下结果是确定的。"多个 seed"只有在 seed 确实控制了某个随机过程（例如顺序打乱或采样）时才有意义，要写明 seed 控制的是什么。
- 延迟（D15）：`run.py` 记录每个臂的端到端 wall-clock（包括 probe、路由、检索查表、编码、prefill、解码）；分析脚本读取这些实测值，不再拼段。`static_joint` 不包含 probe。

## 5. 顺序敏感性

用 `--order perm:<id>` 对 `order.permutation_schedule()` 中的 **14 个排列**各跑一次：2 个锚点（sorted、reverse）加 12 个随机排列（种子 17、23、29，每个种子 4 个）。每次换一个 `--tag`。证据集合在各次运行间不变，排列对所有题目相同。

- 顺序实验只使用**前 1,000 道 test 题**（VideoRAG-Synth 为其全部 224 道）。论文必须披露这一子集（D5）。
- 报告以下几项：
  - 各排列下数据集均值的标准差，以及按题整体重采样的区间（`order.order_std_statistics`，10,000 次重采样）；
  - sorted 与 reverse 之差；
  - 逐题翻转率。

`--order shuffle` 是按题随机打乱，只能用于开发阶段的诊断，不能用于顺序表。

## 6. 调参前的统计功效

在 dev 上比较多个候选之前，先用 `scripts/analysis/power_design.py` 估计当前 dev 大小下的最小可检测效应，以及 best-of-K 的噪声。如果候选之间的差距小于这个噪声，挑出"最好"的那个就没有意义。旧仓库在 48 到 576 题上反复挑选候选，正是这个问题。dev 选择的 6 选 1 校正见 `docs/DEV_DECISIONS.md` 规则 1。

## 7. 引用支持度

`--answer-style cite` 加 `eval/citation/`，完整规则见 `docs/citation_support_protocol.md`。检查的是文字蕴含（DeBERTa-v3 NLI）；图像证据只用其标题或说明文字作为前提，不能称为图像蕴含验证。

- 归因（Faith、AP）支持 `gated:*` 与 `replication` 臂。gated 臂中未检索的题不计入 AP 分母，同时报告覆盖率（D9）。
- **alias 命中与 NLI 判定分别报告**（D9）。论文的 Faith/AP 两种判定都给出。alias 命中与 NLI 判定不做短路组合。
- 前提的长度处理见 `docs/citation_support_protocol.md`（由 G6 维护）。

## 8. 干预与检索错误

- 干预只在满足以下条件的题上做：触发检索、EAB 激活（U > τ_b）、gold 答案字符串出现在某个检索到的文本块中。**目标块是"实际 prompt 中、含答案的、相关度最高的文本块"**，不是偏置最大的块（D10）。检索决定和 EAB 用的 U 固定为未扰动时的值。
- **分母固定**：每个干预、每个臂的分母都是这同一批合格题。无法构造或解码的题计为失败，失败数和失败原因都要记录并与比率一起报告。
- **Mask**：通过 attention 层面屏蔽目标 span（所有行对这些 key 的 logit 设为 −∞），并且不对该 span 施加 EAB；位置不变（D10）。不再用 pad token 替换。
- **反事实（CF）**：替换目标块中**全部** gold 别名（不再只替换第一个）为同类型实体（年份、数字、颜色、是/否、spaCy 实体类型，或按词数），由题号哈希确定，可复现。"跟随"指答案匹配该实体。
- **随机编辑（random_edit）**：把非答案的完整词片段替换为等长的完整词（不切断词），与 CF 用同一批题（D10）。对照还包括不改动的重跑（rerun）。
- **replication 臂**也要在干预输入上重新解码（D10）。
- 干扰项：top-k 之外得分最高、不含答案且不是 gold 的两个候选，替换两个排名最低的非目标块，位置不变。
- 注入：一段像相关网页的文字（错误答案或指令）替换排名最低的非 gold 块，相关度设为 1；rank 顺序下放在第一位。
- 相关度噪声：在校准后的相关度上加高斯噪声并截断到 [0, 1]，影响排序和偏置；每个 σ 跑三次（`--noise-draw 0/1/2`）取平均。
- Gold@2–5 只能用 rank 顺序的运行，按 `chunk_gold` 选题。

## 9. 报告规则

- 论文中的每个数字都要附上 run tag、commit 和评分文件。不能重算的数字一律不写。
- 负结果和不显著的结果同样保留在 `runs/` 中，不按结果好坏挑选。
- 论文中所有比例型结论都要给出分母（题数）和失败数。

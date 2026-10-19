# Omni-Aware v2 交付说明

分支 `v2/arbitration`，起点是 `main` 的 `5c66f73`。没有推送到远端。`main` 与 `origin/main` 仍是 `5c66f73`。`C:\rc\` 只读，没有修改。

2026-10-13 的返工改动已由协调者统一提交到本分支（见 `git log main..v2/arbitration`）。

## 1. 总表

状态只取三类（定义见 `docs/REVIEW_V2_STATUS.md` 的开头）：

- **本次修复**：在 v2/arbitration 分支上完成，相对 `main` 有对应改动。
- **main 中已修复，本次未改动**：修复已在 `main` 中，本分支没有改动该修复所在的代码。
- **未修（原因）**：仍未修，括号内写明原因。

测试列是锁定该项的测试；没有单独测试时写“无”。

| 条目 | 状态 | 改动文件 | 测试 | 备注 |
|---|---|---|---|---|
| R1 EAB 推不动答案 | 本次修复 | `paper/main.tex`，`paper/sec_experiments.tex`，`src/omniaware/run.py` | `tests/test_v2_review.py` | EAB 退出主方法，论文改为机制分析。没有新的 GPU 数字。 |
| R1b 实际偏置小于上界 | 本次修复 | 同上 | 无 | 作为负结果写进 “Attention steering moves attention, not answers”。数字只引用根因报告。 |
| R2 EAB 实现不是失效 bug | main 中已修复，本次未改动 | `src/omniaware/qwen_eab.py` | `tests/test_qwen_eab.py`，`tests/test_v2_review.py` | 非 sdpa 显式报错在 `main` 中，分支未改动。 |
| R3 格式拖垮 static | 本次修复 | `src/omniaware/prompts.py`，`src/omniaware/pipeline.py` | `tests/test_v2_prompts.py` | 各臂共用冻结指令。措辞未在开发集上重选，本机没有 GPU。 |
| R4 QA 分被 BART 主导 | 本次修复 | `src/omniaware/eval/score_official.py` | `tests/test_v2_scoring_path.py` | 主指标是 acc_approx。BART 与 QA_approx 同时输出；没有权重时这两项为 null，不编造。 |
| R5 组件没被真正测到 | 本次修复 | `src/omniaware/signals.py`，`src/omniaware/arbitration.py`，`src/omniaware/run.py` | `tests/test_v2_signals.py`，`tests/test_arbitration.py` | 信号与选择器已实现，合成数据上有测试。没有新的全量运行。 |
| R6 正结果来自小样本或已消费的开发题 | 本次修复 | `src/omniaware/data.py`，`docs/DEV_DECISIONS.md`，`scripts/analysis/dev_select.py` | `tests/test_v2_splits.py`，`tests/test_arbitration.py` | 切分与预注册已写入。+8% 判据没有在新开发集上计算。 |
| R7 评分口径不统一 | 本次修复 | `src/omniaware/eval/score_official.py` | `tests/test_v2_scoring_path.py` | 主表路径走官方评分。 |
| R8 论文声明与实现不符 | 本次修复 | `paper/main.tex`，`paper/sec_experiments.tex`，`paper/sec_appendix.tex`，`README.md` | 无 | 旧声明改为撤回或 `\ph{}`。编译结果见第 7 节。 |
| P0-1 / A1 空答案与纯引用 | 本次修复 | `src/omniaware/answers.py`，`src/omniaware/run.py`，`src/omniaware/merge_shards.py` | `tests/test_v2_answers.py` | 主实验不用 cite。`preds` 保留原始输出，解析结果在 `parsed`。单卡与多卡都统计 `empty_answer_rate`，超过 0.5% 打印 `EMPTY ANSWER RATE`。 |
| P0-2 / A2 各臂同一指令 | 本次修复 | `src/omniaware/prompts.py`，`src/omniaware/baselines/arms.py`，`src/omniaware/baselines/m2rag_webqa.py` | `tests/test_v2_prompts.py` | 测试在真实构造路径上抓取各臂的 prompt 文本（B-3）。`EXCEPTIONS` 为空。指令 sha256 写入运行配置。 |
| P0-3 / A3 官方评分 | 本次修复 | `src/omniaware/eval/score_official.py` | `tests/test_v2_scoring_path.py` | OK-VQA 只走 `score_okvqa`。WebQA 输出 acc_approx、归一化 BART、QA_approx，以及由 `gold_modalities` 定义的图像/文本子集。动作答案的官方评分在 `v2_table.py build` 中，该步骤尚未合入（见第 2 节 A-4）。 |
| P0-4 / A4 EAB 与 ans1 | 本次修复 | `src/omniaware/pipeline.py`，`src/omniaware/run.py`，`scripts/paper_runs.py` | `tests/test_v2_review.py` | 默认 `--variants` 不含 EAB。`--analysis-eab` 才加入 ans1。ans2 只作为 decode-only。 |
| P0-5 / A5 图像证据诊断 | 未修（原因：默认 50 题的真实 WebQA 诊断需要 GPU，本机没有跑；`--fixture` 路径可在无数据时运行） | `scripts/analysis/image_evidence_check.py` | `tests/test_v2_review.py` | GPU 描述命令只打印。 |
| P0-6 / A6 切分 | 本次修复 | `src/omniaware/data.py`，`src/omniaware/run.py` | `tests/test_v2_splits.py`，`tests/test_splits.py` | WebQA 校准与开发来自 train，测试为全部 val。分支新增完整 val 的 4966 校验。三者不交。空测试集报错。三份 qid 的 sha256 写入配置。 |
| P0-7 / A7 审查协议 | 本次修复 | `docs/REVIEW_V2_STATUS.md`，`scripts/paper_runs.py`，`src/omniaware/baselines/arms.py` | `tests/test_v2_review.py`，`tests/test_paper_runs.py` | 状态表已统一为三类，见该文件。 |
| F-1 VideoRAG 评测集为空 | main 中已修复，本次未改动 | `src/omniaware/data.py`，`src/omniaware/run.py` | `tests/test_v2_splits.py` | 空 test 报错。默认 reserve 100、dev 50。 |
| F-2 M2-RAG 看不到问题图 | main 中已修复，本次未改动 | `src/omniaware/baselines/arms.py`，`src/omniaware/baselines/m2rag_webqa.py` | `tests/test_v2_review.py` | 问题图进入 summarizer 与 extractor。 |
| F-3 论文占位与旧数字 | 本次修复 | `paper/*.tex` | 无 | 未跑的数字是 `\ph{}`。 |
| S-1 评测规模 | 本次修复 | `src/omniaware/data.py`，`paper/sec_experiments.tex` | `tests/test_v2_splits.py` | 核心切分在 `main`；分支补充了完整 val 的 4966 校验。论文改为实际切分。 |
| S-2 测试集上选参与无来源数字 | 本次修复 | `paper/sec_appendix.tex`，`paper/sec_experiments.tex`，`scripts/paper_runs.py` | `tests/test_dev_select.py` | 论文中的无来源数字已撤回。没有新的开发集网格。 |
| S-3 n_calib 改变 test | main 中已修复，本次未改动 | `src/omniaware/data.py` | `tests/test_splits.py` | test 不随 n_calib 移动。 |
| S-4 冷启动基线共用探测和路由 | 本次修复 | `scripts/paper_runs.py`，`src/omniaware/run.py` | `tests/test_v2_review.py` | 基线读 joint top-k（`main` 中已是如此）。主表运行的探测轮与路由按原协议保留，返工时撤回了曾设的 `probe=False, route=False`，删除了写死的 `baseline_retrieval` 字段。 |
| S-5 SeaKR 只跑 1000 题 | main 中已修复，本次未改动 | `scripts/paper_runs.py` | `tests/test_v2_review.py`，`tests/test_paper_runs.py` | 主表 SeaKR 不传更小的 `--n-eval`。order 实验的 1000 题子集在 stage 说明里标明。 |
| S-6 UniversalRAG 静默退回 | main 中已修复，本次未改动 | `src/omniaware/baselines/universalrag.py`，`scripts/paper_runs.py` | 既有路由测试 | 空路由不再变成联合检索。OK-VQA 列没有可执行的路由器命令，stage 说明里写了。 |
| S-7 缺决定文件时用默认值 | main 中已修复，本次未改动 | `scripts/paper_runs.py`，`scripts/analysis/dev_select.py` | `tests/test_paper_runs.py`，`tests/test_dev_select.py` | 缺文件或非法 JSON 直接失败。分支只增加了 v2 入口，未改动该检查。 |
| S-8 缓存指纹 | main 中已修复，本次未改动 | `src/omniaware/run.py`，`src/omniaware/runio.py`，`src/omniaware/rerank.py` | `tests/test_cache_resume.py` | 当前代码按内容指纹命中缓存。 |
| S-9 半成品当已合并 | main 中已修复，本次未改动 | `scripts/launch.py`，`src/omniaware/runio.py` | 既有 launch 测试 | 当前代码拒绝没有完成标记的半成品。 |
| S-10 坏图中断抽取 | main 中已修复，本次未改动 | `scripts/setup/prepare_webqa_open.py` | 既有准备脚本测试 | 当前代码按批记录坏图。 |
| S-11 模型完成判定 | main 中已修复，本次未改动 | `scripts/setup/download_models.py` | 无新测试 | 当前代码检查权重分片。 |
| S-12 OK-VQA 三种语料的构建说明 | 未修（原因：BM25 语料的输入由旧主机生成，本仓库无法复现；构建说明未补） | `scripts/paper_runs.py`，`README.md` | `tests/test_paper_runs.py` | 缺候选文件时 dev 阶段失败（`main` 中已有）。v2 使用 `okvqa_gs`。 |
| S-13 Punkt 下载 | main 中已修复，本次未改动 | `scripts/setup/download_models.py` | `tests/test_citation_intake.py` 在本机仍会跳过 | 下载步骤在。本机没装 Punkt，所以测试跳过。 |
| S-14 缩小语料仍称 unified corpora | 本次修复 | `paper/sec_experiments.tex` | 无 | 论文不再使用该措辞。代码仍用缩小语料。 |
| S-15 M2-RAG 像素预算 | 本次修复 | `src/omniaware/baselines/arms.py`，`src/omniaware/baselines/m2rag_webqa.py` | `tests/test_v2_review.py` | 抽取器带上该次运行的共享指令。问题图与 `max_pixels` 的传递在 `main` 中。 |
| S-16 FiTM 温度 | 未修（原因：温度 5e-5 没有原文量纲依据，没有改） | `src/omniaware/baselines/fitm_attention.py` | 无 | 温度仍是 5e-5。 |
| S-17 真实权重的 chat template | 未修（原因：禁止下载模型） | `src/omniaware/qwen_eab.py` | `tests/test_v2_signals.py` 只用小模型 | 真实权重上的核对需要下载模型。 |
| S-18 EAB 选择偏差 | main 中已修复，本次未改动 | `scripts/analysis/dev_select.py` | `tests/test_dev_select.py` | 已是 max-statistic。EAB 不再是主方法。 |
| S-19 因果编辑 | main 中已修复，本次未改动 | `src/omniaware/interventions.py` | 既有 intervention 测试 | 当前代码已修。 |
| S-20 引用指标不是门控答案 | main 中已修复，本次未改动 | `src/omniaware/eval/citation/attribution.py`，`src/omniaware/run.py` | 既有 citation 测试 | cite 仍是单独的 `--answer-style cite`。 |
| S-21 OK-VQA 全量语料 | main 中已修复，本次未改动 | `src/omniaware/okvqa_dense.py`，`scripts/analysis/dev_select.py` | `tests/test_okvqa_gs.py` | 默认 gs112k。含测试题的 full 语料不参与 dev 选择。 |
| B1 五个动作同批解码 | 本次修复 | `src/omniaware/run.py`，`src/omniaware/signals.py` | `tests/test_v2_signals.py`，`tests/test_v2_review.py` | direct、static_text、static_image、static_mixed、rerank_static。`rerank_static` 没有 cross-encoder 时报错，词面重叠已删除，`rerank_static_source` 配置项已删除。 |
| B2 信号字段 | 本次修复 | `src/omniaware/signals.py`，`src/omniaware/run.py` | `tests/test_v2_signals.py` | 熵非负，margin ≥ 0。`gold_in_topk` 只存储。`--v2` 下某个动作缺逐 token logprob 时运行报错。 |
| B3 选择器与风险控制 | 本次修复 | `src/omniaware/arbitration.py` | `tests/test_arbitration.py` | 2026-10-13 返工重写：v2-Rule、v2-Cal（每个证据动作一个岭回归，预测效用差）、对平均效用的固定序列单侧配对检验。详见第 2 节。 |
| B4 预注册 | 本次修复 | `docs/DEV_DECISIONS.md` | `tests/test_arbitration.py`，`tests/test_dev_select.py` | 2026-10-13 改写 v2 预注册一节，取代旧的退化率版本。见本分支 2026-10-13 的返工提交。 |
| C 离线探索 | 本次修复 | `scripts/analysis/offline_v2.py`（重写，约 300 行），`runs/offline_v2/` | `tests/test_offline_v2.py` | 结果见 `runs/offline_v2/REPORT.md`。探索性数字，来自含已知 bug 的旧管线。 |
| D S1/S2/S4 与门槛 | 本次修复（脚本与流程）；GPU 运行未做 | `scripts/v2_runs.py`，`scripts/analysis/v2_table.py`（build / gate / select / apply），`docs/V2_RUNS.md`，`README.md` | `tests/test_v2_runs.py`，`tests/test_v2_table.py` | 判据并入 `v2_table.py gate`，独立的 `v2_gate.py` 已删除。本机没有 GPU，S1/S2/S4 没有运行。 |
| E 文档与论文 | 本次修复 | `README.md`，`docs/PROTOCOL.md`，`docs/PAPER_TABLE_MAP.md`，`docs/REVIEW_V2_STATUS.md`，`DELIVERY_v2.md`，`paper/*.tex` | 无 | 未跑实验的数字是 `\ph{}`。 |

## 2. 2026-10-13 返工

对应 `CURSOR_REVIEW_v2_验收意见.md`。A、B、C 三节逐条说明处理结果。

| 验收条目 | 处理结果 | 状态 |
|---|---|---|
| A-1 风险控制的损失定义 | 改为对平均效用的单侧配对检验（`arbitration._one_sided_p`）：H0 为 E[d] < −ε，t 统计量，正态上尾 p 值。δ 与 ε 分开，预注册 δ = 0.10、ε = 1.0。`α` 的概念已删除。 | 本次修复（选择器重写） |
| A-2 固定序列起点 | 起点改为等价于 a\* 的策略：a\* 为 direct 时 tau 从 −∞ 递增；a\* 为证据动作时 tau 从 +∞ 递减。规则的默认动作改为 a\*，不再是 direct。 | 本次修复（选择器重写） |
| A-3 v2-Cal 没有动作身份 | 每个证据动作一个岭回归，预测相对 a\* 的效用差。特征为 `[1, conf(direct), conf(a*), conf(b)−conf(a*), token F1, KL, retriever top-1]`。缺失的列丢弃并记录，部分缺失报错，不填 0。校准集对半，v2-Cal 经过同一套风险控制，`select_v2` 调用它。 | 本次修复（选择器重写） |
| A-4 流水线没有接通 | `scripts/analysis/v2_table.py`（build / gate / select / apply）给每个动作官方评分并拼成逐题表；`v2_runs.py` 的 S2 与 S4 生成 score、build、gate、select、apply 命令。CPU 上用小模型和合成数据端到端走通过。 | 本次修复 |
| A-5 `rerank_static` 被静默换成词面重叠 | 没有 cross-encoder 时报错，词面重叠代码已删除，`rerank_static_source` 配置项已删除。v2 使用 `--v2 --v2-reranker`，没有采用验收意见中建议的 `--selector ce`：`check_v2_args` 规定 `--v2` 时选择器必须保持 relevance，因此 static_mixed 保持联合相关度 top-k，只有 rerank_static 使用 cross-encoder 分数。 | 本次修复（命令形式与验收意见不同，原因见左） |
| B-1 无效的契约参数 | `--save-signals`、`--empty-answer-accounting`、`--shared-prompt-registry`、`--v2-actions` 已从 `run.py` 删除（HEAD 中存在）。 | 本次修复 |
| B-2 v2 动作解码对所有运行强制开启 | 由 `--v2` 显式开启。动作分组与解码抽成 `v2_action_chunks` 与 `decode_v2_actions`。调用点仍在 `main()` 的主循环中，没有完全抽成独立函数。 | 本次修复（部分：调用点仍在 `main()` 内） |
| B-3 A2 的测试是同义反复 | `tests/test_v2_prompts.py` 通过 `run.main` 跑 tiny 模型，抓取各臂实际编码的 prompt，断言包含该次运行的指令。覆盖直答、路由、联合、v2 各动作、校准、FiTM、SeaKR，以及 M2-RAG 抽取器消息。 | 本次修复 |
| B-4 `paper_runs.py` 主表探测与路由被关 | 已撤回：主方法运行保留探测轮与路由。`baseline_retrieval` 写死字符串已删除。`order` 实验的 `route=False` 是原有设计（固定 chunk 集合），与主表无关。 | 本次修复 |
| B-5 空答案处理 | `preds` 保留原始输出，解析结果在 `rec["parsed"]`。多卡合并路径统计 `empty_answer_rate`，统计包括 available 的 v2 动作。模态候选池为空时该动作标记为 `available=false`，不复用 direct 的结果。 | 本次修复 |
| B-6 `v2_runs.py` 的数据集与陈旧注释 | 数据集改为 `okvqa_gs`。过期注释已删除。S2 生成评分、build、gate、select 命令，S4 生成评分、build、apply 命令。 | 本次修复 |
| B-7 WebQA 子集划分的回退 | 缺少 `gold_modalities` 时报错（`score_official.webqa_subset_name`），不再按 qcate 猜测。 | 本次修复 |
| C 节 同一常量定义多次 | `V2_ACTIONS` 只在 `omniaware/actions.py` 定义。`v2_gate.py` 已删除。 | 本次修复 |
| C 节 重复实现与离线脚本 | `token_f1` 做大小写与标点归一化。`ltt_select`、`split_half`、`_TAU1_GRID` 等已从选择器删除。`score_official` 的同值别名字段已去掉。`offline_v2.py` 重写为约 300 行，直接导入选择器，空答案判断复用 `answers.parse_answer`。 | 本次修复 |
| C 节 命名 | `keepers`、`N_PARAMS`、`diagnostic_substring_accuracy` 已删除。选择器不再有 `split_half` 与 `assert_disjoint_qids`。 | 本次修复 |

### 离线结果（待重新生成）

旧的离线探索结果（任务 C 的评测半效用为 0）来自被替换的选择器，已被返工取代，不再作为结论引用。

返工后的离线结果，包括验收门槛要求的并排基线（评测半效用为正，且不低于按 `u_initial` 单阈值的朴素基线：WebQA 评测半 acc 43.31，始终 static 为 41.32），应见 `runs/offline_v2/REPORT.md`（由 `offline_v2.py` 重新生成）。本文件不填写任何数字。`offline_v2.py` 重写完成前，该文件仍是旧输出，不可用。

## 3. 验收门槛的状态

- 离线（WebQA，旧管线，探索性）：评测半上 v2-Rule 43.31，朴素单阈值基线 43.31，始终 static 41.32，通过。只保留两路答案都非空的题时，v2-Rule 为 48.34 对 47.02（+2.8%），即原先 +4.8% 中约一半来自识别旧管线的空答案。OK-VQA 上所有策略回退为直答，增益为 0，“收益为正”这一条在 OK-VQA 上没有满足。详见 `runs/offline_v2/REPORT.md`。
- 阶段接线：build → gate → select → apply 已在 CPU 上用小模型和合成数据走通；真实模型与真实数据上的运行没有做。
- 选择器：合成数据测试覆盖了 A-1、A-2、A-3 与 B-3 的要点（见 `tests/test_arbitration.py`），但这不等同于在真实数据上的结果。

## 4. 没做或做不到

- 没有 GPU，没有下载模型或数据。S1、S2、S4 没有实际运行。开发集 +8% 门槛因此没有 PASS/FAIL。
- 50 道 WebQA 图像题的真实诊断没有跑。脚本的 GPU 描述探针只打印命令。
- WebQA 提示词没有在开发集上重选。注册表里是冻结的候选句。
- 归一化 BART 与 QA_approx 在没有 ParaBank / 相关权重时保持 null。
- FiTM 温度仍是 5e-5。真实权重上的 chat template 没有核对。
- 选择器支持只在部分题上可用的动作（例如没有图像候选的题上的 `static_image`）：这类动作不能成为最强单一动作，但在可用的题上可以被选中。
- `dev_select.py` 的 `select_v2`（`--v2-records`）与 `v2_table.py select` 调用同样的选择器函数，是两个入口。协调者需确定正式流程只保留一个。
- 离线数字不能进入论文主表。

## 5. 提交

返工改动已提交，见 `git log main..v2/arbitration` 中 2026-10-13 的返工提交。

返工前的提交（`git log --oneline main..HEAD`，截至 `3f7516a`）：

```
3f7516a Record the v2 delivery status, offline results, and pytest output.
565cdab Accept the v2 run flags, including a calibration split decode.
9e8add3 Merge the v2 protocol fixes into v2/arbitration.
3e4135c Drop EAB from the main method, keep cold-start baselines on joint top-k, and write v2 action records (A4, A7).
541a872 Add a CPU WebQA image-evidence diagnostic and print the GPU probe without running it (A5).
e6c87b6 Record first-token entropy, margin, sequence logprob, and KL for each decoded action (B2).
ee0a7a8 Lock WebQA calibration, dev, and the full validation test split as pairwise disjoint (A6).
f3392df Score OK-VQA only with score_okvqa and emit WebQA acc_approx, normalized BART, and QA_approx (A3).
c1f698e Give every arm one frozen per-dataset instruction and record its sha256 (A2).
0714580 Merge the v2 arbitration selector into v2/arbitration.
5035ef5 Parse blank, citation-only, and refusal answers before they are scored (A1).
bb08c50 Merge the v2 protocol and paper draft into v2/arbitration.
9fec5cc Match the v2-Cal feature list to the seven-parameter selector.
982f97a Record the fixed L2 penalty used by the v2 calibrator (B4).
496f061 Merge the exploratory offline v2 check into v2/arbitration.
af80369 Score the old dumps with the v2 selector on the held-out half.
473f10e Preregister the v2 selector grid, utility, and go/no-go rule (B4).
9829fef Add the v2 arbitration selector and its dev selection step (B3).
3c96e4e Rewrite the paper draft around risk-controlled arbitration and retire EAB as analysis.
1513923 Map each v2 paper table to the command that produces it.
ff10ada Rewrite the experiment protocol around the v2 splits, metrics, and strong baseline.
777ae47 Document the v2 direction and withdraw historical-cache gains.
171ffc0 Record measured empty-answer rates and half-split sizes from the old dumps.
646f8cb Merge v2 GPU stage scripts into v2/arbitration.
1afcfa4 Add the exploratory offline v2 checker and its unit tests.
a773a06 Decode the calibration split in S2 so the selector has B2 signals.
27f07db Add S1, S2, and S4 dry-run GPU commands for task D.
a16ee0a Add the S2 dev-set oracle gap gate for task D.
```

注：旧的提交说明写的是 `565cdab` 为最新，实际 HEAD 为 `3f7516a`。上面的列表以当前 HEAD 为准。

## 6. pytest

`python -m pytest -p no:warnings` 的汇总行（第二轮修复后）：`449 passed, 5 skipped, 8 subtests passed`。5 个跳过是原有的环境相关测试（缺少 torchvision、spacy、NLTK 数据或资产目录）。

## 7. 论文编译

在 `paper/` 中执行 `latexmk -pdf -interaction=nonstopmode main.tex`，退出码 0，输出 `main.pdf`，8 页，559597 字节。日志中没有 `!` 错误，没有未定义的引用或引文，只有一条 1.6pt 的 `Overfull \vbox` 版式警告。

该 PDF 被 `.gitignore` 忽略，没有提交。


## 8. 2026-10-13 第二轮审查与修复

第二轮由 5 个只读审查（协议、端到端、文档与论文、代码质量、清单核对）加一组蒙特卡洛模拟组成，报告在 `C:\rc\_reports\audit2\`。确认并修复的问题：

| 问题 | 处理 |
|---|---|
| 检索器信号取的是整个候选池的第一名，不是该动作实际使用的证据 | `signals.retriever_signals` 的顶层统计改为按选中的证据计算，候选池统计移到 `retriever["pool"]`。 |
| 动作记录没有保存使用了哪些证据 | 每个动作记录新增 `chunks`。 |
| 直答不可用时仍可能被规则选中；应用阶段不校验非有限置信度 | 已修，`arbitration.reference_action` 给出逐题参照动作。 |
| 最强单一动作在某道测试题上不可用时 `apply` 报错 | 改用逐题参照动作。 |
| 拼表不核对题集是否完整 | `v2_table build` 复用 `score_official` 的题集与 gold 校验。 |
| v2 主表没有自适应检索对照 | 表里并入基线臂分数；校准集上为 U 门控与 NTE 各冻结一个单阈值；`apply` 输出 v2 相对每个基线的配对比较，最强基线在校准集上确定。S2 开发运行与 S4 运行带 `--fitm --m2rag --seakr-samples 20`。 |
| GPU 脚本的评分命令不带 `--bart` | WebQA 的评分与 build 命令加上 `--bart`。 |
| `rerank_static` 与 CE 基线的证据顺序不同 | 两者走同一个函数。 |
| v1 分析脚本读取的 EAB 列在主表运行里不存在 | v1 管线（`paper_runs.py`）的运行恢复 EAB 臂；它只用于机制分析与基线，不是 v2 主表。 |
| UniversalRAG 的 query 路由在 WebQA 上必然失败 | 运行在加载模型之前检查候选是否带 suite 分数；v1 的相关阶段不再包含 WebQA。 |
| 旧主表脚本在测试集上选最强基线 | `column_table` 只在调用方显式给出时标注最强基线。 |
| M2-RAG 的证据文本预算与其他臂不同 | 改为使用同一词数预算；其余协议差异列在 `baselines.arms.PROTOCOL_DIFFERENCES` 并写入运行配置。 |
| 图像证据诊断不检查真实输入 | 重写为用与运行相同的函数构造输入段后检查。真实数据上的诊断仍未运行。 |
| 空输出被记为最高置信度；整句拒答有漏网；同一运行标签并发写入无锁 | 已修；新增 `--max-empty-rate`，S1 冒烟使用 0.005。 |
| 论文与文档的不实表述 | “preregistered”、“提示词在开发集上选定”、省钱版、“+33%”的来源、两处方法描述已改正。 |

蒙特卡洛（36 种情形：二值与连续分数、独立与高度相关、校准集 100 / 500 / 2000 题、最强动作为直答、证据或并列，各 500 次）：选择器选出真实期望得分低于最强单一动作减 ε 的策略的频率为 0，保证的上限是 0.10。结果在 `C:\rc\_reports\audit2\work_stats\guarantee_log.txt`。统计设计未改。

仍未解决：
- 所有 GPU 运行（S1 / S2 / S4）与真实数据上的图像证据诊断。
- OK-VQA 的校准集与开发集取自评测数据的保留部分（仓库里没有 OK-VQA 的 train 加载器），三者不交，但不是取自 train。
- 校准运行不带 SeaKR / FiTM / M2-RAG 臂，所以“校准集上的最强基线”只在单一动作、U 门控与 NTE 之间确定；`apply` 仍会逐个报告 v2 相对 SeaKR / FiTM / M2-RAG 的比较。
- 相关工作缺 CRAG、CAD、DoLa 与风险控制方面的引用；`\tbd{AI-use disclosure}` 未填。
- 此前列出的三项：BM25 版 OK-VQA 语料、FiTM 温度、真实权重下的对话模板。

# REVIEW 2026-10-10 致命 / 严重条目状态（v2/arbitration 分支）

对照 `REVIEW_2026-10-10.md` 的 F-* 与 S-*。审查稿里的行号已过期，下面以当前代码为准。`DELIVERY_v2.md` 第 1 节的 F-*、S-* 行与本表一致。

每个条目只取三类状态之一：

- **本次修复**：在 v2/arbitration 分支上完成，即相对 `main` 有对应改动（包括分支上的提交和 2026-10-13 的返工）。
- **main 中已修复，本次未改动**：修复已在 `main` 中，`git log main` 可见，本分支没有改动该修复所在的代码。
- **未修（原因）**：仍未修，括号内写明原因。

判断方法：对每个实现文件运行 `git diff main --stat -- <file>` 与 `git log main --oneline -- <file>`，并对照修复所在的函数。若文件在分支上有改动，但改动的不是该条修复，按修复所在的函数判断。

| 编号 | 状态 | 说明 | 位置 |
|---|---|---|---|
| F-1 | main 中已修复，本次未改动 | VideoRAG-Synth 默认 reserve=100、dev=50，374 题时 test 非空。评测切分为空时报错（`run.py`）。分支在 `data.py` 只新增 `WEBQA_VAL_SIZE` 常量，与本项无关。 | `src/omniaware/data.py`（`CALIB_RESERVE`、`split_examples`）；`src/omniaware/run.py`；`tests/test_splits.py`、`tests/test_v2_splits.py` |
| F-2 | main 中已修复，本次未改动 | OK-VQA 的问题图会作为 `aux_images` 进入 M2-RAG summarizer，并进入 extractor 的消息。分支在同一文件中只加了共享指令参数（见 S-15），不改问题图路径。 | `src/omniaware/baselines/arms.py`（`m2rag_answer`）；`src/omniaware/baselines/m2rag_webqa.py`（`official_query`、`extractor_messages`）；`tests/test_baseline_arms.py`、`tests/test_v2_review.py` |
| F-3 | 本次修复 | 论文中未跑出的数字全部是 `\ph{}`，V3.20 的数字不再保留。主审已把它从致命问题里拿掉，作为状态说明保留。 | `paper/*.tex` |
| S-1 | 本次修复 | WebQA 校准 ≥500、开发 256 都来自 train，且开发在校准 reserve 之外；测试是全部 val。核心切分在 `main`；分支补充了完整 val 的 4966 校验（`WEBQA_VAL_SIZE`）。图像列从同一次全量切分里取图像题。 | `src/omniaware/data.py`（`N_CALIB_TRAIN`、`TRAIN_RESERVE`、`N_DEV`、`WEBQA_VAL_SIZE`、`image_subset`）；`src/omniaware/run.py`（`split_qids_sha256`）；`tests/test_splits.py`、`tests/test_v2_splits.py` |
| S-2 | 本次修复 | 论文中无来源的数字（“0.6 points”“70%”“β=0.30 在测试集上最优”）已撤回，论文稿中不再出现。代码侧 `paper_runs` 的 β×τ_b 测试网格标为 SENSITIVITY ONLY（该注释在 `main`）。选择仍在开发集上，用 max-statistic。 | `paper/sec_appendix.tex`、`paper/sec_experiments.tex`；`scripts/paper_runs.py`；`scripts/analysis/dev_select.py` |
| S-3 | main 中已修复，本次未改动 | held-out 与 train-pool 都先留出固定 reserve，test 不随 `n_calib` 移动。 | `src/omniaware/data.py`（`split_examples`）；`tests/test_splits.py` |
| S-4 | 本次修复 | 基线（Static、FiTM、M2-RAG、CE、SeaKR）读 joint top-k（D4，`main` 中已是如此），不读探测轮与路由。主表运行的探测轮与路由按原协议保留：返工时撤回了分支上曾设的 `probe=False, route=False`，并删除了写死的 `baseline_retrieval` 字段。消融里的 noprobe 仍是单独的消融。 | `scripts/paper_runs.py`（main 阶段）；`src/omniaware/run.py`（`chunks_joint`）；`tests/test_v2_review.py` |
| S-5 | main 中已修复，本次未改动 | 主表 SeaKR 命令不传 `--n-eval`，与同阶段其它臂的题集相同。order 实验的 1000 题子集在 stage 说明里标明。 | `scripts/paper_runs.py`；`tests/test_paper_runs.py`、`tests/test_v2_review.py` |
| S-6 | main 中已修复，本次未改动 | 路由标签都不在语料里时不再静默变成联合检索（`retrieve` 为假并记 fallback）。未知标签报错。 | `src/omniaware/baselines/universalrag.py`（`to_route`、`LABEL_TO_MODALITY`、`parse_labels`）；`scripts/paper_runs.py` |
| S-7 | main 中已修复，本次未改动 | test 阶段没有 phase-2 决定文件（缺文件、换模型、只有 phase 1）直接失败。dev 分数缺失或非有限时 `DevDataError`，不写决定。分支在 `dev_select.py` 增加了 v2 入口，未改动这一检查。 | `scripts/paper_runs.py`（`check_stage`）；`scripts/analysis/dev_select.py`（`DevDataError`）；`tests/test_paper_runs.py`、`tests/test_dev_select.py` |
| S-8 | main 中已修复，本次未改动 | relevance / CE / ucal / SeaKR 缓存按内容指纹命中，候选或设置变了就重算；分片 worker 遇到过期缓存报错。 | `src/omniaware/run.py`（`load_relevance`、`ucal_fp`、`seakr_fingerprint`）；`src/omniaware/runio.py`（`fingerprint`）；`src/omniaware/rerank.py`（`ce_fingerprint`） |
| S-9 | main 中已修复，本次未改动 | 多卡 launch 不把没有完成标记的 `<dataset>.jsonl` 当成已合并。 | `scripts/launch.py`（`verify_completion`）；`src/omniaware/runio.py`（`write_completion_marker`） |
| S-10 | main 中已修复，本次未改动 | 图片抽取按批提交，坏图记入 bad 列表并继续，临时文件再 `replace`。 | `scripts/setup/prepare_webqa_open.py`（`extract_images`、`_decode`、`_batched`） |
| S-11 | main 中已修复，本次未改动 | 模型完成判定检查权重分片，而不只是目录里有没有任意 json。 | `scripts/setup/download_models.py`（`model_complete`）；`scripts/setup/check_assets.py` |
| S-12 | 未修（原因：BM25 语料的输入由旧主机生成，仓库中没有，无法复现；README 中的构建说明也没有补齐） | dev 阶段缺候选文件时直接失败，不跳过（`check_stage`，在 `main` 中）。OK-VQA 候选中的 BM25 维基（`okvqa`）已在 DEV_DECISIONS 中移除，v2 使用 `okvqa_gs`。 | `scripts/paper_runs.py`（`check_stage`）；`scripts/analysis/dev_select.py`（`OKVQA_LOADERS`）；`README.md` 第 4 节 |
| S-13 | main 中已修复，本次未改动 | Punkt 资产有下载步骤：固定 nltk 版本和 punkt 修订。本机未装 Punkt，相关测试仍会跳过，那是环境问题，不是缺少步骤。 | `scripts/setup/download_models.py`（`NLTK_VERSION`、`extract_punkt_english`）；`scripts/setup/check_assets.py` |
| S-14 | 本次修复 | 论文稿不再使用“unified corpora”的措辞。HybridQA 与 VideoRAG-Synth 用的仍是缩小语料，论文未跑的单元格都是 `\ph{}`，不写成结果。 | `paper/sec_experiments.tex`；`scripts/universalrag_retrieval/build_subset.py` |
| S-15 | 本次修复 | M2-RAG 的抽取器消息带上该次运行的共享答案指令（`arms.m2rag_answer` 的 `instruction` 参数，由 `run.py` 传入）。问题图与 `max_pixels` 的传递在 `main` 中。 | `src/omniaware/baselines/arms.py`（`pixel_budget`、`m2rag_answer`）；`src/omniaware/baselines/m2rag_webqa.py`（`extractor_messages`） |
| S-16 | 未修（原因：FiTM 温度 5e-5 的量纲需要对照 Hsieh et al. 2024 的原文确认，没有依据不改，否则会连带改动已有的数值测试） | FiTM 温度仍是 5e-5。 | `src/omniaware/baselines/fitm_attention.py` |
| S-17 | 未修（原因：真实 Qwen3-VL / InternVL 权重的 chat template 与分段 prefill 比对需要下载模型，本任务禁止下载） | 现有测试只用随机小模型。 | `src/omniaware/qwen_eab.py`；`tests/eab_fixtures.py` |
| S-18 | main 中已修复，本次未改动 | EAB 形式选择用全部候选上的 max-statistic bootstrap，不再用未校正的区间。 | `scripts/analysis/dev_select.py`（`max_stat_gain`） |
| S-19 | main 中已修复，本次未改动 | Mask 用 attention mask 挡住目标 span，该 span 不再加 EAB。反事实替换全部 gold 别名。随机编辑换成等长的完整词窗口，并避开 gold。 | `src/omniaware/interventions.py`；`src/omniaware/qwen_eab.py`（`blocked`） |
| S-20 | main 中已修复，本次未改动 | 引用打分默认包含 `gated:<arm>` 和 replication。冷启动 `--norm-from` 在来源没有 probe 的 Norm 时失败。 | `src/omniaware/eval/citation/attribution.py`（默认 `decoded`）；`src/omniaware/run.py` |
| S-21 | main 中已修复，本次未改动 | 默认 GS 语料是 train-only 的 gs112k（112,724 句）。含测试题检索结果的 gs168k_full 只做附录，不参与 dev 选择。 | `src/omniaware/okvqa_dense.py`（`GS_CORPORA`）；`scripts/analysis/dev_select.py`（`OKVQA_LOADERS` 不含 full） |

## 空答案、指令与 rerank 的当前行为

主实验不使用引用指令：`--answer-style cite` 只在 citation 阶段。空答案由 `src/omniaware/answers.py` 标记：`preds` 保留模型的原始输出，解析结果写在 `rec["parsed"]`。单卡与多卡合并路径都统计 `empty_answer_rate`，统计包括 `available` 的 v2 动作。超过 0.005 时汇总里有一行 `EMPTY ANSWER RATE`。各臂指令在 `src/omniaware/prompts.py`，`EXCEPTIONS` 默认为空。

`rerank_static` 需要 cross-encoder 分数（`--v2-reranker`）。没有 cross-encoder 时运行报错，不再用词面重叠替代，配置项 `rerank_static_source` 已删除。`--selector ce` 只用于 v1 的 CE 对照臂；v2 的 static_mixed 保持 relevance 的联合 top-k，与 `--v2` 互斥。

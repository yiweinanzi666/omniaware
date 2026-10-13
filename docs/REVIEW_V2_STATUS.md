# REVIEW 2026-10-10 致命 / 严重条目状态（v2 协议代理）

对照 `REVIEW_2026-10-10.md` 的 F-* 与 S-*。审查稿里的行号已过期，下面以当前代码为准。状态只取：已修 / 不修(原因) / 不适用。

| 编号 | 状态 | 说明 | 位置 |
|---|---|---|---|
| F-1 | 已修 | 评测/测试切分为空会报错。VideoRAG-Synth 默认 reserve=100、dev=50，374 题时 test 非空，不再把全部题划进校准集。 | `src/omniaware/data.py`（`split_examples`、`CALIB_RESERVE`、`N_DEV`）；`src/omniaware/run.py`（空 test / 空 eval 切片）；`tests/test_splits.py`、`tests/test_v2_splits.py` |
| F-2 | 已修 | OK-VQA 的问题图会作为 `aux_images` 进入 M2-RAG summarizer，并进入 extractor 的消息。本代理未改选择逻辑，只加了回归测试。 | `src/omniaware/baselines/arms.py`（`m2rag_answer`）；`src/omniaware/baselines/m2rag_webqa.py`（`official_query`、`extractor_messages`）；`tests/test_baseline_arms.py`、`tests/test_v2_review.py` |
| F-3 | 不适用 | 主审已把它从致命问题里拿掉，改为“论文仍是占位符”的状态说明。占位符在 `paper/**`，本代理不改论文。 | `REVIEW_2026-10-10.md` 状态说明 |
| S-1 | 已修 | WebQA 校准 ≥500、开发 256 都来自 train，且开发在校准 reserve 之外；测试是全部 val（完整文件为 4966）。图像列从同一次全量切分里取图像题，不再各自 shuffle。论文里的旧句由论文代理改。 | `src/omniaware/data.py`（`N_CALIB_TRAIN`、`TRAIN_RESERVE`、`N_DEV`、`WEBQA_VAL_SIZE`、`image_subset`）；`src/omniaware/run.py`（`split_qids_sha256`）；`tests/test_splits.py`、`tests/test_v2_splits.py` |
| S-2 | 不修(原因) | 写死的 “0.6 points”“70%”“β=0.30 在测试集上最优”在 `paper/**`，本代理禁止改论文。`scripts/analysis/dev_select.py` 也禁止修改。代码侧：γ 的选择在 dev，并用 max-statistic；`paper_runs` 的 β×τ_b 测试网格已标成 SENSITIVITY ONLY，不进入选择。 | `scripts/analysis/dev_select.py`；`scripts/paper_runs.py`（ablation 的 sensitivity 注释） |
| S-3 | 已修 | held-out 与 train-pool 都先留出固定 reserve，test 不随 `n_calib` 移动。n_calib=100 不再把主 dev 题卷进 test。 | `src/omniaware/data.py`（`split_examples`）；`tests/test_splits.py` |
| S-4 | 已修 | 主表上的 Static / CE / FiTM / M2-RAG / SeaKR 不再带 `--probe-uncertainty` 和 `--first-round-routing`。它们读 joint top-k（`static_joint` / `chunks_joint`），不读探测轮路由。配置里写 `baseline_retrieval=joint_topk_no_probe_no_route`。消融里的 noprobe 仍保留 first-round routing，那是单独的消融，不是主表基线。 | `scripts/paper_runs.py`（main 阶段 `probe=False, route=False`）；`src/omniaware/run.py`（`chunks_joint`、基线读 `joint`）；`tests/test_v2_review.py` |
| S-5 | 已修 | 主表 SeaKR 命令不传 `--n-eval`，与同阶段其它臂的题集相同。order 实验仍用前 1000 题，stage 说明里写了 SUBSAMPLE，不把它当成全量测试。 | `scripts/paper_runs.py`（main 与 order）；`tests/test_paper_runs.py`、`tests/test_v2_review.py` |
| S-6 | 已修 | 路由标签都不在语料里时不再静默变成联合检索：`retrieve` 为假并记 fallback。`clip` 与 `video` 保持不同标签。未知标签会报错而不是变成 `no`。主表 WebQA-img 列是 full 运行的图像子集（analysis `column`）；OK-VQA 列在 stage 说明里写明没有命令（路由器只读文本）。论文单元格里的破折号由论文代理改。 | `src/omniaware/baselines/universalrag.py`（`to_route`、`LABEL_TO_MODALITY`、`parse_labels`）；`scripts/paper_runs.py`（universalrag stage 的 fills） |
| S-7 | 已修 | test 阶段没有 phase-2 的 `dev_decisions.json`（或缺文件、换模型、只有 phase 1）会直接失败，不再静默用硬编码默认值。dev 分数缺失或非有限时 `dev_select` 抛错，不写决定。本代理未改 `dev_select.py`（禁止修改）；当前代码已经是失败而不是静默写出。 | `scripts/paper_runs.py`（`decisions`、`check_stage`）；`scripts/analysis/dev_select.py`（`DevDataError`）；`tests/test_paper_runs.py`、`tests/test_dev_select.py` |
| S-8 | 已修 | relevance / CE / ucal / SeaKR 缓存按内容指纹命中，候选或设置变了就重算；分片 worker 遇到过期缓存会报错。 | `src/omniaware/run.py`（`load_relevance`、`ucal_fp`、`seakr_fingerprint`）；`src/omniaware/runio.py`（`fingerprint`）；`src/omniaware/rerank.py`（`ce_fingerprint`） |
| S-9 | 已修 | 多卡 launch 不再把没有完成标记的 `<dataset>.jsonl` 当成已合并。没有 `.merged.json` 的半成品会失败并要求移走或用单卡跑完。 | `scripts/launch.py`（`verify_completion`）；`src/omniaware/runio.py`（`write_completion_marker`） |
| S-10 | 已修 | 图片抽取按批提交，坏图记入 bad 列表并继续，临时文件再 `replace`。审查里“一张坏 JPEG 让整个 map 失败”的行为已经不在。 | `scripts/setup/prepare_webqa_open.py`（`extract_images`、`_decode`、`_batched`） |
| S-11 | 已修 | 模型完成判定检查权重分片，而不只是目录里有没有任意 json。WebQA 的 BGE 与 CLIP rankings 都会检查。 | `scripts/setup/download_models.py`（`model_complete`）；`scripts/setup/check_assets.py` |
| S-12 | 不修(原因) | 三种 OK-VQA 语料的构建步骤写在 README，本代理禁止改 README。`dev_select` 的候选列表也禁止修改。`paper_runs` 在 dev 阶段缺候选文件时会失败而不是跳过（S-12 的静默跳过部分已在 `check_stage`）。BM25 语料 `okvqa` 仍没有纳入本仓库能完成的构建。 | `scripts/paper_runs.py`（`check_stage`）；`scripts/analysis/dev_select.py`（`OKVQA_LOADERS`）；README 禁止修改 |
| S-13 | 已修 | Punkt 资产有下载步骤：固定 nltk 3.9.2 和 punkt 修订，写到 citation cache。测试在没有该资产时仍可能跳过，那是环境里没跑过下载，不是缺少生成步骤。 | `scripts/setup/download_models.py`（`NLTK_VERSION`、`extract_punkt_english`）；`scripts/setup/check_assets.py` |
| S-14 | 不修(原因) | HybridQA / VideoRAG 用的是缩小语料，论文仍写 “unified corpora”。改正文在 `paper/**`，本代理禁止修改。 | `scripts/universalrag_retrieval/build_subset.py`；`paper/**` |
| S-15 | 已修 | 问题图和与其它臂相同的 `max_pixels` 会传给 summarizer 和 extractor。抽取器同时带上该次运行的共享答案指令（`prompts.instruction_text`）。两段式流程本身是 M2-RAG 的官方步骤；论文里的披露由论文代理写。 | `src/omniaware/baselines/arms.py`（`pixel_budget`、`m2rag_answer`）；`src/omniaware/baselines/m2rag_webqa.py` |
| S-16 | 不修(原因) | FiTM 温度仍是 5e-5。审查要求先对照 Hsieh et al. 2024 的量纲，不能在没有原文依据时改温度，否则会把已有数值测试和基线一起改掉。 | `src/omniaware/baselines/fitm_attention.py` |
| S-17 | 不修(原因) | 真实 Qwen3-VL / InternVL 权重上的 chat template 与分段 prefill 比对需要下载模型。本任务禁止下载模型；现有测试只用随机小模型。 | `src/omniaware/qwen_eab.py`；`tests/eab_fixtures.py` |
| S-18 | 已修 | EAB 形式选择用全部候选上的 max-statistic bootstrap，不再用同一份 dev 上未校正的区间。本代理未改该文件（禁止修改）；当前实现已经是校正后的选择。 | `scripts/analysis/dev_select.py`（`max_stat_gain`） |
| S-19 | 已修 | Mask 用 attention mask 挡住目标 span，并且该 span 不再加 EAB。反事实替换全部 gold 别名。随机编辑换成等长的完整词窗口，并避开 gold。等长 gold 的顺序按长度和字典序，不依赖 `PYTHONHASHSEED`。 | `src/omniaware/interventions.py`；`src/omniaware/qwen_eab.py`（`blocked`） |
| S-20 | 已修 | 引用打分默认包含 `gated:<arm>` 和 replication。冷启动 `--norm-from` 在来源没有 probe 的 Norm 时失败，不会静默留下目标集自己的探测轮。Acc 与 AP 仍来自两次解码（cite 是单独的 `--answer-style cite`），表注由论文代理写。 | `src/omniaware/eval/citation/attribution.py`（默认 `decoded`）；`src/omniaware/run.py`（`--norm-from` 缺 probe 即报错） |
| S-21 | 已修 | 默认 GS 语料是 train-only 的 gs112k（112,724 句）。含测试题检索结果的 gs168k_full 标明只做附录、不参与 dev 选择。 | `src/omniaware/okvqa_dense.py`（模块说明、`GS_CORPORA`）；`scripts/analysis/dev_select.py`（`OKVQA_LOADERS` 不含 full） |

主实验不使用引用指令：`--answer-style cite` 只在 citation 阶段。空答案由 `src/omniaware/answers.py` 标记，运行汇总写 `empty_answer_rate`；超过 0.005 时汇总里有一行 `EMPTY ANSWER RATE` 和该比率。各臂指令在 `src/omniaware/prompts.py`，`EXCEPTIONS` 默认为空。`rerank_static` 在本次运行没有加载 cross-encoder 时用词面重叠重排同一候选池（配置项 `rerank_static_source=lexical_overlap`），不下载 reranker；`--selector ce` 时用 cross-encoder 分数。

# 新旧代码对照

旧仓库 `EMNLP-reproduction`（commit `c8d08ebb`）有 33,975 个文件，其中 4,764 个 `.py`。按内容去重后约 2,100 份不同的代码，包含 `erc` 包的至少 19 份副本、十几个评分器版本，以及大量 receipt/status 文件。

迁移分两轮，都遵循"先原样复制、再单独提交改动"：
- **第一轮**（提交 `fa7ad4f`）：34 个文件，主方法、官方评测、数据准备。
- **第二轮**（提交 `2b0b7e5` 起）：59 个文件（含随后补入的 NLI 检查器清单），对照方法、引用支持度、WebQA 开放检索、分析工具；另有 24 个对象（函数或常量）从整个脚本中按 AST 原样提取。

每个文件或对象的来源路径和 SHA256 都记录在 `MIGRATION_SOURCES.json`（`files`、`files_round2`、`extracted_objects`）。后续每一处改动都是单独提交，可以用 `git log -p` 逐项审查。旧仓库保持不变，作为只读档案。

## 第一轮：主方法与官方评测

| 新位置 | 旧位置 | 后续改动 |
|---|---|---|
| `src/omniaware/qwen_eab.py` | `experiments/emnlp_repro_claude/erc/qwen_eab.py` | 无（钩子已在 transformers 5.5.3 下用测试验证） |
| `src/omniaware/pipeline.py` | `erc/pipeline.py` | 删除宽松评分、focus 图像预算、融合相关度和 sup/ctrk 等形式；增加 json_list / strict_letter / cite 答案格式 |
| `src/omniaware/run.py` | `erc/run.py` | 所有对照臂同输入生成；删除 `--select-tau`；dev/test 划分、训练集校准；EAB 默认用 U_pre；删除 query-coverage、order-vote；增加复制臂、固定排列、CE 选证、引用输出 |
| `src/omniaware/data.py` | `erc/data.py` | 加载器名称明确化；WebQA 缺图计数；MMQA 默认 369 题（MMQA 已于 2026-10-11 删除，此处仅为历史记录）；Video 帧目录与字幕长度可选；新的划分函数（D1：保留区 + dev + test）；WebQA 开放检索加载器 |
| `src/omniaware/relevance.py` | `erc/relevance.py` | 删除 query-coverage |
| `src/omniaware/routing.py` | `erc/routing.py` | 无 |
| `src/omniaware/okvqa_dense.py` | `erc/okvqa_dense.py` | 缓存目录 |
| `src/omniaware/paths.py` | `erc/paths.py` | 仓库内路径；`OMNIAWARE_ASSETS`（兼容 `ERC_ASSET_ROOT`）；`WEBQA_OPEN` |
| `src/omniaware/answer_format.py`、`table_render.py`（**已删除**，2026-10-11，见下节） | `pro6000_three_official_20261008/answer_format.py`、`table_rows_lossless.py` | 无 |
| `src/omniaware/eval/score_official.py` | `pro6000_three_official_20261008/score_official.py` | bootstrap 与视频辅助函数不变；`main()` 改为对任意运行的所有臂评分 |
| `src/omniaware/eval/mmqa_official.py`（已删除）、`webqa_acc.py`、`webqa_bart.py` | `eab_first_token/mmqa_official_eval.py`、`score_webqa.py`、`score_bart.py` | import 与官方源码路径 |
| `third_party/*`（`videomme/` 已删除） | VQA、Video-MME、WebQA 官方评测（见 `third_party/README.md`） | 无 |
| `scripts/data_prep/*`（Video-MME 帧准备脚本已删除） | `eab_first_token/prepare_*`、`fetch_*`、`prepare_dense_video.py` | 资产根目录改为环境变量；32 帧脚本不再改写运行输入 |

## 第二轮：对照方法、引用支持度、开放检索、分析

| 新位置 | 旧位置 | 后续改动 |
|---|---|---|
| `src/omniaware/baselines/dragin_rind.py`、`dragin_qwen_observation.py`（**已删除**，DRAGIN 于 2026-10-11 删除，见下节） | `eab_first_token/` 同名文件 | 无（RIND 打分、按校准集检索率定阈值、单次决策的草稿观测） |
| `src/omniaware/baselines/fitm_attention.py`、`fitm_qwen_adapter.py` | `eab_first_token/` 同名文件 | 无（Hsieh et al. 公式 4、5，以及 Qwen 适配） |
| `src/omniaware/baselines/m2rag_official.py`、`m2rag_qwen.py`、`m2rag_webqa.py` | `eab_first_token/` 同名文件 | 官方源码从资产目录读取；删除只服务"M2-RAG + EAB"组合的 `official_stage_one_inputs` |
| `src/omniaware/rerank.py` | `multimodal_retrieval.py`、`analyze_ce_modality_selection.py`、`score_visual_candidates.py`、`prepare_webqa.py` 中的函数与常量 | 按函数原样提取；新增 CE 打分缓存与 `ce_select`；两模态专用的 `fit_ecdfs` 改用 `relevance.Calibrator` |
| `src/omniaware/order.py` | `run_order_mechanism.py`、`score_order_mechanism.py` 中的函数 | 按函数原样提取；`replication_counts` 的 γ 改为参数。2026-10-10 起复制份数为 `max(1, round(exp(γUr)))`，不截断（D11） |
| `src/omniaware/eval/citation/*` | `prepare_citation_support_intake.py`、`runs/e5_citation_support_preparation/{intake_answers,run_nli,score_nli}.py`、`checker_asset_manifest.json` | 包内 import；资产目录；支持任意对照臂；记录而不是强制旧主机的 torch/transformers 版本；新增 `from_run.py` |
| `scripts/webqa_open_retrieval/*` | `runs/phase1_dense_retrieval_codex_20261006/` 中的语料池、编码、索引、查询编码、检索及校验脚本 | 输出目录指向主机上已有的旧目录，以复用已建好的索引；查询编码支持任意问题集；删除编排脚本和只服务 train256 的融合与连接步骤（已删除的文件见 `MIGRATION_SOURCES.json` 中标注为 `removed` 的条目） |
| `scripts/data_prep/build_full_dpr_index.py` | `runs/claude_direct_repair_20261005/round_005/candidates/` | 资产根目录 |
| `scripts/analysis/power_design.py`、`answer_recall.py` | `round_008/analysis/power_design.py`、`round_005/analysis/okvqa_passage_answer_recall.py` | 核心函数不变；输入改为任意运行的输出 |
| `tests/test_{fitm_*,m2rag_*,citation_*,rerank,order,webqa_open}.py`、`tests/tiny_qwen.py`（`test_dragin_*` 已删除） | 对应旧测试 | import；缺少资产时跳过；M2-RAG WebQA 测试改用合成输入 |
| `docs/citation_support_protocol.md` | `runs/e5_citation_support_preparation/E5_PROTOCOL_AND_HANDOFF.md` | 无 |

## 未迁移及原因

| 旧内容 | 原因 |
|---|---|
| `erc/scoring.py`、`erc/aliases.py`、`YESNO_REFERENCE_POLARITIES.json` 及所有评分器快照 | 宽松子串匹配会误判（例如 `'no'` 匹配 `'snow'`，`'a'` 匹配 `'banana'`），再加上按题补的同义词规则。论文只能用官方评测 |
| `erc/report.py`、`tables.py`、`summarize.py`、`derive_tau.py`、`prep_relevance.py` | 依赖宽松评分或缓存 baseline；已由 `score_official.py` 取代 |
| `cached_baselines/` 以及所有"复用 baseline"的流程 | 旧 baseline 的提示词、token 上限和输入都与 Ours 不同 |
| 旧 first-token 管线：`eab.py`、`qwen_adapter.py`、`qwen_paper_eab.py`、`qwen_first_answer_eab.py`、`qwen_review_eab.py`、`multimodal_packing.py`、`controlled_generation.py`、`native_vision_cache.py`、`observe_*` | 已被 `qwen_eab.py` 取代；注意力观测由 `qwen_eab` 的捕获功能覆盖 |
| `eab_first_token` 的 run/prepare/score/audit 脚本（约 400 个），以及 T0x、E1–E5 任务队列和 `continue_*`、`run_*_after_*` 调度器 | 每个脚本绑定某一轮固定的输入和输出目录；其中可复用的算法已在第二轮按函数提取 |
| 开发阶段的候选方法：anchored/LPER EAB、centered/rank-centered EAB、draft fusion、case anchor、evidence facts、sentence selection、answer policy、reasoned verification、token budget、visual budget、target binding、query-only router（qrouter） | 实测为负或没有被采用；结论记录在审查报告与旧仓库中 |
| UniversalRAG 路由器的训练脚本，以及 InfoSeek/MRAG 图像恢复链 | 路由器按其官方脚本训练（训练数据先剔除评测题），结果以 `--query-routes` 文件输入，不在本仓库重新实现；训练代码和数据恢复链很长，需要时再从旧仓库迁移。注意：UniversalRAG **本身**是论文主表的对照方法，其候选导出与评测接口已迁移（`src/omniaware/baselines/universalrag.py`、`scripts/universalrag_retrieval/*`），见 [PAPER_TABLE_MAP.md](PAPER_TABLE_MAP.md)（修复记录 2026-10-10，原文"论文的结果表里没有这个方法"有误） |
| `runs/claude_direct_repair_*`、`goal_diagnosis_codex_*`、`local_m4_*`、`pro6000_preparation_*` 等目录 | 绝大多数是一次性的审计、复现、按轮次的测试和运行编排；可复用的只有已迁移的三个小工具 |
| WebQA 开放检索中的图像修复脚本（`prepare_official_jpeg_assets` 等） | 只用于一次性修复少数图片；索引已经建好 |
| Pro6000 的 MMQA 投票、分歧复核、表头复核脚本 | 多次生成后投票属于额外推理开销的工程技巧，不是本方法；如需保留，应作为单独的对照方法重新实现 |
| `reference/self-aware-mrag-main` | 原 EMNLP 代码，未实现 EAB / U_omni（见审查报告 §1.2）；其中的 SAM-RAG、ViDoRAG 等基线依赖 FlashRAG 框架和宽松评分 |
| `deliveries/`、`Each Stage Review Work/`、根目录约 30 个 `EMNLP_INCREMENT_*.json` | 交付快照与进度记录，不是代码 |

需要追溯时，到旧仓库用 `git show c8d08ebb:<路径>` 读取。

## 2026-10-11 范围调整后删除的迁移代码

以下迁移过来的代码已随评测与对照方法的删减一并删除（历史保留在 git 中）：MMQA 加载器、官方评测包装、
JSON 列表答案解析（`answer_format.py`）与表格渲染（`table_render.py`）；Video-MME 加载器、官方答案解析
（`third_party/videomme/`）、帧准备脚本；DRAGIN 的 RIND 打分与 Qwen 观测（`baselines/dragin_*`）。

## 2026-10-10 核对：MIGRATION_SOURCES.json 中不存在的目标

`MIGRATION_SOURCES.json` 的 `files` 与 `files_round2` 中有 36 个 `target` 在仓库中不存在。逐个核对后，每个条目都增加了 `status` 字段：

- `removed`：对应的功能已按上节删除（MMQA、Video-MME、DRAGIN、旧 WebQA 修复与 train256 编排脚本，以及 E5 交接准备脚本）；或者该文件从未作为独立模块迁入（仅作为 staging 源，其中有用的函数已按 AST 提取到对应模块，见 `extracted_objects`）。
- `renamed` / `merged` / `extracted`：目标已改为仓库中的现有路径（例如 `okvqa_passage_answer_recall.py` 改为 `scripts/analysis/answer_recall.py`；两个 staging 测试并入 `tests/test_rerank.py`、`tests/test_order.py`；staging 中的函数提取到 `src/omniaware/rerank.py`）。

`extracted_objects` 中的条目不对应文件，未修改。

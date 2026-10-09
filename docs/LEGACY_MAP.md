# 新旧代码对照

旧仓库 `EMNLP-reproduction`（commit `c8d08ebb`）有 33,975 个文件，其中 4,764 个 `.py`，包含 `erc` 包的至少 19 份副本、十几个评分器版本，以及大量 receipt/status 文件。本仓库只迁移其中有用的 34 个文件：

- 第一次提交 `fa7ad4f` 是**逐字节复制**，每个文件的来源路径与 SHA256 记录在 `MIGRATION_SOURCES.json`；
- 之后每一处改动都是单独提交，可以用 `git log -p` 逐项审查。

旧仓库保持不变，作为只读档案。

## 已迁移

| 新位置 | 旧位置 | 后续改动 |
|---|---|---|
| `src/omniaware/qwen_eab.py` | `experiments/emnlp_repro_claude/erc/qwen_eab.py` | 无（钩子已在 transformers 5.5.3 下用测试验证） |
| `src/omniaware/pipeline.py` | `erc/pipeline.py` | 删除宽松评分、focus 图像预算、融合相关度和 sup/ctrk 等形式；增加 json_list / strict_letter 答案格式 |
| `src/omniaware/run.py` | `erc/run.py` | 所有对照臂同输入生成；删除 `--select-tau`；增加 dev/test 划分、训练集校准；EAB 默认用 U_pre；删除 query-coverage、order-vote 等选项 |
| `src/omniaware/data.py` | `erc/data.py` | 加载器名称明确化；WebQA 缺图计数；MMQA 默认 369 题；Video 帧目录与字幕长度可选；新的划分函数 |
| `src/omniaware/relevance.py` | `erc/relevance.py` | 删除 query-coverage |
| `src/omniaware/routing.py` | `erc/routing.py` | 无 |
| `src/omniaware/okvqa_dense.py` | `erc/okvqa_dense.py` | 缓存目录 |
| `src/omniaware/paths.py` | `erc/paths.py` | 仓库内路径；`OMNIAWARE_ASSETS`（兼容 `ERC_ASSET_ROOT`） |
| `src/omniaware/answer_format.py` | `pro6000_three_official_20261008/answer_format.py` | 无 |
| `src/omniaware/table_render.py` | `pro6000_three_official_20261008/table_rows_lossless.py` | 无 |
| `src/omniaware/eval/score_official.py` | `pro6000_three_official_20261008/score_official.py` | bootstrap 与视频辅助函数不变；`main()` 改为对任意运行的所有臂评分 |
| `src/omniaware/eval/mmqa_official.py` | `experiments/eab_first_token/mmqa_official_eval.py` | 官方源码路径 |
| `src/omniaware/eval/webqa_acc.py` | `eab_first_token/score_webqa.py` | 官方源码路径 |
| `src/omniaware/eval/webqa_bart.py` | `eab_first_token/score_bart.py` | import 与路径 |
| `third_party/vqa_eval/vqaEval_py3.py` | `okvqa_full_official_independent_root_20261008T1922/evidence/` | 无 |
| `third_party/videomme/official_video_eval.py` | `pro6000_three_official_20261008/` | 无 |
| `third_party/webqa/*` | `webqa_official_vs_lenient_20261008/official_source/` | 无 |
| `scripts/data_prep/*` | `eab_first_token/prepare_*`、`fetch_*`，以及 `prepare_dense_video.py` | 资产根目录改为环境变量；32 帧脚本不再改写运行输入 |
| `tests/test_{mmqa_official,webqa_acc,webqa_bart,answer_format,table_render}.py` | 对应旧测试 | import；缺少资产时跳过 |

## 未迁移及原因

| 旧内容 | 原因 |
|---|---|
| `erc/scoring.py`、`erc/aliases.py`、`resources/YESNO_REFERENCE_POLARITIES.json` 及所有评分器快照 | 宽松子串匹配会误判（例如 `'no'` 匹配 `'snow'`，`'a'` 匹配 `'banana'`），再加上按题补的同义词规则。论文只能用官方评测 |
| `erc/report.py`、`tables.py`、`summarize.py`、`derive_tau.py`、`prep_relevance.py` | 依赖宽松评分或缓存 baseline；功能已由 `score_official.py` 取代 |
| `experiments/emnlp_repro_codex_review/cached_baselines/` 以及所有"复用 baseline"的流程 | 旧 baseline 的提示词、token 上限和输入都与 Ours 不同，比较不公平 |
| `experiments/eab_first_token/` 下其余约 400 个脚本（旧 first-token 管线、T0x/E1–E5 任务队列、continue/run_after 调度器、各类 audit/verify 脚本） | 旧管线已被 `erc` 取代；调度和审计脚本只服务于某一轮的运行 |
| `emnlp_repro_claude/candidates/*`、`emnlp_repro_codex_review/*` 的几百个候选与复核目录 | 开发阶段的一次性候选；结论已记录在审查报告中；其中的源码都是 `erc` 的副本 |
| Pro6000 的 MMQA 投票、分歧复核、表头复核脚本 | 多次生成后投票属于额外推理开销的工程技巧，不是本方法；如需保留，应作为单独的对照方法重新实现 |
| `reference/self-aware-mrag-main` | 原 EMNLP 代码，未实现 EAB / U_omni（见审查报告 §1.2），不是本仓库方法的来源 |
| `deliveries/`、`Each Stage Review Work/`、根目录约 30 个 `EMNLP_INCREMENT_*.json` | 交付快照与进度记录，不是代码 |

需要追溯时，到旧仓库用 `git show c8d08ebb:<路径>` 读取。

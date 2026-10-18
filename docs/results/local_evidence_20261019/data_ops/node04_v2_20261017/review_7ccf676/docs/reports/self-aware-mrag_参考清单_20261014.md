# self-aware-mrag 参考清单（2026-10-14）

`https://github.com/yiweinanzi/self-aware-mrag` 是 2025 年 10–12 月的 Self-Aware-MRAG 代码（FlashRAG 框架，Qwen3-VL-8B）。作者确认它不是 EMNLP 投稿版的最终代码，是目前能拿到的最新版本。本仓库只把它当参考：代码不迁移，结果不复用，论文数字也不从这里取证，与[根因溯源报告](Omni-Aware_根因溯源报告_20261013.md) §2 的结论一致。它与本仓库的代码、协议或文档不一致时，一律以本仓库为准。

下文的路径和行号都指 commit `9fe9af84f849b8a448064e0e531b37f581fde662`。简称如下：
- `pipeline`：`FlashRAG/flashrag/pipeline/self_aware_pipeline_qwen3vl.py`
- `estimator`：`FlashRAG/flashrag/modules/uncertainty_estimator.py`
- `improved`：`FlashRAG/flashrag/modules/uncertainty_estimator_improved.py`

## 1. 内容范围

它实现"跨模态不确定性"、位置感知融合和归因，覆盖 OK-VQA、A-OKVQA、MRAG-Bench、MultiModalQA 四个数据集。不确定性有两个实现，由配置项 `use_improved_estimator` 选择：
- `estimator`（`CrossModalUncertaintyEstimator`）：问题文本隐藏状态的 Gram 特征值熵、CLIP 图像特征的统计量、CLIP 文本特征与图像特征之间的 JS 散度。
- `improved`（`ImprovedUncertaintyEstimator`）：文本部分按问题长度和关键词表打分，视觉部分用 CLIP 图像特征的统计量，对齐部分用 CLIP 图文相似度。2025 年 11–12 月的 OK-VQA 运行用的是这一个。

EMNLP 方法的 H_grd、首 token 熵 H_sem、U_omni、M_m·H_m 路由和 EAB/DCA 都是之后才出现的；WebQA 在 `FlashRAG/flashrag/config/dataset_registry.py` 中未启用。EMNLP 主表的数字（67.88、54.65、59.30、72.15、66.82 等）在这个仓库的任何文件里都找不到。

## 2. 可参考的工程实现

需要时按 `CONVENTIONS.md` 重写并补测试，不复制原文件。原文件中的路径都指向旧服务器，需要替换。

| 内容 | 位置 | 用途 |
|---|---|---|
| MMQA 表格序列化：表头一行，每行单元格用 ` \| ` 连接，保留行边界 | `pipeline:395-415` | 论文保留 MMQA-full 路由表时使用。MMQA 已于 2026-10-11 从本仓库删除，恢复前先在 `docs/DEV_DECISIONS.md` 记录决定 |
| MMQA 文件读取：问题（`MMQA_{split}.jsonl.gz`）；表格与图像元数据（`MMQA_tables.jsonl.gz`、`MMQA_images.jsonl.gz`） | 问题在 `FlashRAG/flashrag/dataset/multimodalqa_dataset.py`；表格和图像在 `pipeline:384-446` | 同上 |
| A-OKVQA 数据加载（其中的 `evaluate_answer` 是子串判分，不用） | `FlashRAG/flashrag/dataset/aokvqa_dataset.py` | 重跑图像类结果表时使用 |
| MRAG-Bench 四选一提示与字母回映射 | `pipeline:330-343`、`pipeline:730-738` | 同上 |
| MRAG-Bench 图像语料的 CLIP 图像特征 FAISS 索引 | `FlashRAG/experiments/build_mrag_image_indexes.py` | 同上 |
| 以查询图像检索语料描述文本的 CLIP 检索器 | `FlashRAG/flashrag/retriever/clip_image_retriever.py` | 同上 |
| DPR Wikipedia（`psgs_w100.tsv`）前 300 万段的语料与 BGE 索引构建 | `FlashRAG/tools/build_corpus_wiki_only_3m.py`、`FlashRAG/tools/rebuild_index_wiki_3m.py` | 对照语料规模时使用；本仓库的 OK-VQA 稠密检索用 DPR Wikipedia 全量（`scripts/data_prep/build_full_dpr_index.py`） |
| ViDoRAG 官方代码副本 | `ViDoRAG-main/` | 需要 ViDoRAG 时从上游 `Alibaba-NLP/ViDoRAG` 按固定 commit 获取，并登记到 `third_party/external_repos.json` |

## 3. 有启发的观察

这些数字用的是宽松判分或小样本，只作动机和教训，不进论文。

1. **加入不确定性模块没有带来收益。** 实验设置：
   - OK-VQA val 全量 5045 题，2025-11-25/26；
   - Qwen3-VL-8B，BGE 检索 wiki_3m，top-5，`improved` 估计器，宽松判分。

   | | 准确率 | 检索率 |
   |---|---|---|
   | Baseline_MuRAG | 63.35% | 90.35% |
   | 五个不确定性变体 | 62.62–62.87% | 92.8% |

   基线的两个数只见于 `📊导师汇报-自感知多模态RAG-2025-11-27.md`，仓库里没有对应的结果文件。五个变体的数据在 `FlashRAG/experiments/results_unified_ablation/unified_ablation_results_20251126_233745.json`。五个变体彼此相差不到 0.3 个点，都比基线低 0.5–0.7 个点，检索也没有变少。这与 v2 的出发点一致：先决定这道题要不要证据、要哪类证据。

   这份结果不是 `9fe9af84` 的 pipeline 产生的。结果里有 7% 的题没检索，每题保留的文档数在 0 到 5 之间；而 `9fe9af84` 的 pipeline 在同样的配置（`force_retrieval=True`）下每题都检索，也不过滤文档（§4 第 1 条）。
2. **不检索的结果不比检索差。** 一次 20 题的阈值扫描（文件未记录数据集），见 `_ARCHIVE_2025-11-13/04_experiment_logs/threshold_sweep_results.json`：
   - 阈值从 0.30 调到 0.60，检索率从 95% 降到 0%，EM 始终是 0.65；
   - 只有阈值 0.25（检索率同样是 95%）那一档是 0.60，相差一道题。
3. **Wikipedia 检索会取回同主题的干扰段落。** 见 `case_study_images/CASE_STUDY.md` 第 1 例：
   - 图片是一辆停放的摩托车，问题是"能用来做什么运动"，标准答案 race；
   - 检索 top-3 是 Squash、Roller hockey、Lacrosse 词条；
   - 两种方法都答错。
   
   这个例子可以说明检索会引入干扰。放进论文前要走论文修改的审批流程。
4. **100 道题排不出稳定的名次。** 同一组 100 道 MRAG-Bench 题：
   - 2025-10-29 的运行：本方法 EM 61%，基线 60–63%（`_ARCHIVE_2025-11-13/backups/_ARCHIVED/2025-10-30-old-experiments/results_baseline_comparison_100_real/`）；
   - 10-30 的运行（改用 wiki_3m 语料）：本方法 59%，基线 51–53%（`FlashRAG/experiments/archived/20251125/results_baseline_comparison_100_wiki3m/`）。
   
   两次之间本方法有 34 题、MuRAG 有 31 题的答案变了。比较方法之前，要先用 `scripts/analysis/power_design.py` 估算所需样本量。
5. **判分规则决定结论。** 本方法和基线用的是不同的宽松规则（§4 第 4 条）。根因溯源报告 §2 记录过，宽松判分能把同一批 100 题的预测从 59 抬到 72。这与 `CONVENTIONS.md` §5 只用官方评测的规定一致。

## 4. 不要照搬

| # | 问题 | 位置 | 后果 |
|---|---|---|---|
| 1 | 检索不受不确定性控制：<br>• 检索调用（第 536、540、551 行）不在第 506 行的判断分支里，每题都检索；<br>• 判断分支只给 `should_retrieve` 赋 True，`force_retrieval` 默认也为 True；<br>• 把 `force_retrieval` 设为 False 时，U≤τ 的题里 `should_retrieve` 没有赋值，读取它时（第 750 或 756 行）抛 `UnboundLocalError`，`run()` 捕获后把这道题静默丢掉；<br>• `estimator` 自带的 `should_retrieve` 也固定返回 True | `pipeline:504-551`、`pipeline:750-756`、`pipeline:1209-1210`；`estimator:460` | 不确定性只影响模态选择和查询改写；关掉强制检索后，低不确定性的题会从结果中消失 |
| 2 | 位置感知融合不改变顺序：调制系数是标量，归一化后被约掉；检索分数降序、位置权重也降序，二者相乘后顺序不变 | `pipeline:833-886` | 实际效果只是保留前 3 篇 |
| 3 | 归因读取 `doc['text']`，而文档字段是 `contents` | `pipeline:690` | 归因在空字符串上计算 |
| 4 | 判分规则不统一：本方法用 pipeline 自带的"精确、双向子串或关键词"匹配，空答案也判对；MuRAG、ViDoRAG、SAM-RAG（`samrag_adapted.py`）、VisRAG 的实现用另一套规则（只看前 3 个标准答案做词干和同义词匹配，再加"至少一位标注者"的 VQA 匹配）；mR²AG、RagVL 由各实验脚本判分 | `pipeline:1166-1193`；`FlashRAG/experiments/baselines/evaluation_helper.py`、`FlashRAG/experiments/baselines/answer_matcher.py:73` | 本方法与基线不可比 |
| 5 | MMQA 只给本方法题目自带的上下文（同一题的表格与候选图像），基线自己检索；"图像"只是 ID 加标题文字 | `pipeline:372-462`；`FlashRAG/experiments/run_all_baselines_MultimodalVQA_multimodal.py:1713` | 输入不对等，图像内容没有进入模型 |
| 6 | 一道 MMQA 测试题（Ben Piazza 饰演 Mr. Simms）进了代码：<br>• MMQA 表格题提示写有"'Mr. Simms'和'Simms'是不同的"（生效）<br>• 表格标题缺省为 'Filmography'<br>• 表格格式化函数专门标记 Mr. Simms 所在行（只在文档含"【表格信息】"时触发）<br>• 答案支持度验证用这道题做 few-shot（该验证已注释停用） | `pipeline:1002`；`pipeline:399`；`pipeline:919-943`；`pipeline:191-198` | 测试信息进入提示 |
| 7 | 失败时用随机数或零向量顶替 | `estimator:693`、`estimator:787`、`estimator:895`；`improved:221`；`FlashRAG/experiments/build_multimodalqa_indexes_complete.py:145` | 失败被静默掩盖 |
| 8 | SeaKR 仿写（pipeline 未调用）：只跑前 3 层，不传注意力掩码和位置信息；层调用失败时退回 embedding 均值，而 embedding 查表是确定性的，k 个样本完全相同 | `estimator:767-895` | 算出来的不是 SeaKR；本仓库用 `src/omniaware/baselines/seakr.py` |
| 9 | 2025-11-24 的"消融"是模拟运行：检索文档是 "Document about …" 占位句，准确率 3%；`SIMPLE_ABLATION_REPORT` 的数字直接写在脚本里 | `FlashRAG/experiments/archived/20251125/run_real_ablation_simple.py:149`；`FlashRAG/experiments/archived/20251125/run_ablation_simple.py:131-199` | 这些结果不是模型输出 |
| 10 | 生成长度写死为 `max_new_tokens=5`，不读配置 | `pipeline:1129` | 多词答案被截断 |
| 11 | MMQA、MRAG-Bench、A-OKVQA 的方法对比大多只有 10 题；README 的 MRAG-Bench 表是 100 题 | `results_multimodalqa_*`、`FlashRAG/experiments/results_*`、`README.md` | 支撑不了方法排序 |

## 5. 与本仓库现有实现的对应

| 需要的功能 | 用本仓库的 |
|---|---|
| SeaKR | `src/omniaware/baselines/seakr.py`（按上游公式实现，有 AST 对照测试） |
| cross-encoder 选证 | `src/omniaware/rerank.py` |
| 模态路由 | `src/omniaware/routing.py` |
| 答案评分 | `python -m omniaware.eval.score_official` |

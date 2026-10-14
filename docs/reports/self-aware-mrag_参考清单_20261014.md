# self-aware-mrag 参考清单（2026-10-14）

`https://github.com/yiweinanzi/self-aware-mrag` 是 2025 年 10–12 月的 Self-Aware-MRAG 代码（FlashRAG 框架，Qwen3-VL-8B）。作者确认它早于 EMNLP 投稿版的最终代码，是目前能拿到的最新版本。本仓库只把它当参考：代码不迁移，结果不复用，论文数字也不从这里取证，与[根因溯源报告](Omni-Aware_根因溯源报告_20261013.md) §2 的结论一致。

下文的路径和行号都指 commit `9fe9af84f849b8a448064e0e531b37f581fde662`。`pipeline` 是 `FlashRAG/flashrag/pipeline/self_aware_pipeline_qwen3vl.py` 的简称，`estimator` 是 `FlashRAG/flashrag/modules/uncertainty_estimator.py` 的简称。

## 1. 内容范围

它实现三路"跨模态不确定性"、位置感知融合和归因，覆盖 OK-VQA、A-OKVQA、MRAG-Bench、MultiModalQA 四个数据集。三路不确定性分别是：问题文本隐藏状态的 Gram 特征值熵、CLIP 图像特征的统计量、CLIP 文本特征与图像特征之间的 JS 散度。

EMNLP 方法的 H_grd、首 token 熵 H_sem、U_omni、M_m·H_m 路由和 EAB/DCA 都是之后才出现的；WebQA 在 `FlashRAG/flashrag/config/dataset_registry.py` 中未启用。EMNLP 主表的数字（67.88、54.65、59.30、72.15、66.82 等）不在它的任何结果文件里。

## 2. 可参考的工程实现

需要时按 `CONVENTIONS.md` 重写并补测试，不复制原文件。

| 内容 | 位置 | 用途 |
|---|---|---|
| MMQA 表格序列化：表头一行，每行单元格用 ` \| ` 连接，保留行边界 | `pipeline:397-417` | 论文保留 MMQA-full 路由表时使用。MMQA 已于 2026-10-11 从本仓库删除，恢复前先在 `docs/DEV_DECISIONS.md` 记录决定 |
| MMQA 问题与语料读取（`MMQA_{split}.jsonl.gz`、`MMQA_tables/texts/images.jsonl.gz`） | `FlashRAG/flashrag/dataset/multimodalqa_dataset.py` | 同上 |
| A-OKVQA 加载器 | `FlashRAG/flashrag/dataset/aokvqa_dataset.py` | 重跑图像类结果表时使用 |
| MRAG-Bench 四选一提示与字母回映射 | `pipeline:330-343`、`pipeline:730-738` | 同上 |
| MRAG-Bench 图像语料的 CLIP 编码与检索 | `FlashRAG/experiments/build_mrag_image_indexes.py`、`FlashRAG/flashrag/retriever/clip_image_retriever.py` | 同上 |
| Wikipedia 前 300 万段的语料与 BGE 索引构建 | `FlashRAG/tools/build_corpus_wiki_only_3m.py`、`FlashRAG/tools/rebuild_index_wiki_3m.py` | 对照语料规模时使用；本仓库的 OK-VQA 稠密检索用 DPR Wikipedia 全量（`scripts/data_prep/build_full_dpr_index.py`） |
| ViDoRAG 官方代码副本 | `ViDoRAG-main/` | 需要 ViDoRAG 时从上游 `Alibaba-NLP/ViDoRAG` 按固定 commit 获取，并登记到 `third_party/external_repos.json` |

## 3. 有启发的观察

这些数字用的是宽松判分或小样本，只作动机和教训，不进论文。

1. **不确定性模块在一直检索的设定下没有带来收益。** 实验设置：
   - OK-VQA val 全量 5045 题，2025-11-25/26；
   - Qwen3-VL-8B，BGE 检索 wiki_3m，top-5，宽松判分。

   一直检索的 MuRAG 基线是 63.35%，这个数只见于 `📊导师汇报-自感知多模态RAG-2025-11-27.md`，仓库里没有对应的结果文件。五个不确定性变体在 62.62–62.87% 之间，检索率都是 92.8%（`FlashRAG/experiments/results_unified_ablation/unified_ablation_results_20251126_233745.json`）。各变体其实都在检索（见 §4 第 1 条），彼此的差异小于 0.3 个点。v2 由此把问题前移：先决定这道题要不要证据、要哪类证据。
2. **不检索的结果不比检索差。** 一次 20 题的阈值扫描（文件未记录数据集）：检索率为 0–5% 时 EM 是 0.65，检索率为 70–95% 时 EM 是 0.60–0.65（`_ARCHIVE_2025-11-13/04_experiment_logs/threshold_sweep_results.json`）。
3. **Wikipedia 检索会取回同主题的干扰段落。** 见 `case_study_images/CASE_STUDY.md` 第 1 例：
   - 摩托车竞速图片，问题是"能用来做什么运动"，标准答案 race；
   - 检索 top-3 是 Squash、Roller hockey、Lacrosse 词条；
   - 两种方法都答错。
   
   这个例子可以说明检索会引入干扰。放进论文前要走论文修改的审批流程。
4. **100 道题排不出稳定的名次。** 同一组 100 道 MRAG-Bench 题：
   - 2025-10-29 的运行：本方法 EM 61%，基线 60–63%（`_ARCHIVE_2025-11-13/backups/_ARCHIVED/2025-10-30-old-experiments/results_baseline_comparison_100_real/`）；
   - 10-30 换成 wiki_3m 语料后：本方法 59%，基线 51–53%（`FlashRAG/experiments/archived/20251125/results_baseline_comparison_100_wiki3m/`）。
   
   两次之间本方法有 34 题、MuRAG 有 31 题的答案变了。这正是本仓库先用 `scripts/analysis/power_design.py` 算样本量的原因。
5. **判分规则决定结论。** 本方法和基线各用一套宽松规则（§4 第 4 条）。根因溯源报告 §2 记录过，宽松判分能把同一批 100 题的预测从 59 抬到 72。`CONVENTIONS.md` §5 只允许官方评测，依据就在这里。

## 4. 不要照搬

| # | 问题 | 位置 | 后果 |
|---|---|---|---|
| 1 | 检索被强制打开：`force_retrieval` 默认为 True；估计器的 `should_retrieve` 固定返回 True | `pipeline:504`；`estimator:460` | "自适应检索"没有门控，各变体检索率都在 93% 左右 |
| 2 | 位置感知融合不改变顺序：调制系数是标量，归一化后被约掉；检索分数降序、位置权重也降序，二者相乘后顺序不变 | `pipeline:834-886` | 实际效果只是保留前 3 篇 |
| 3 | 归因读取 `doc['text']`，而文档字段是 `contents` | `pipeline:690` | 归因在空字符串上计算 |
| 4 | 两套判分规则：本方法用"精确、双向子串或关键词"匹配，空答案也判对；基线只看前 3 个标准答案，做词干和同义词匹配，再加 VQA 单人匹配 | `pipeline:1166-1193`；`FlashRAG/experiments/baselines/answer_matcher.py:73`、`FlashRAG/experiments/baselines/evaluation_helper.py` | 本方法与基线不可比 |
| 5 | MMQA 只给本方法提供题目自带的上下文（同一题的表格与候选图像），基线自己检索；"图像"只是 ID 加标题文字 | `pipeline:372-460`；`FlashRAG/experiments/run_all_baselines_MultimodalVQA_multimodal.py:1713` | 输入不对等，图像内容没有进入模型 |
| 6 | 一道 MMQA 测试题（"Mr. Simms"）被写进表格提示和 few-shot 示例 | `pipeline:191-198`、`pipeline:919-943`、`pipeline:1002` | 测试信息泄漏 |
| 7 | 提取失败时用随机矩阵或零向量顶替 | `estimator:693`、`estimator:787`、`estimator:895`；`FlashRAG/experiments/build_multimodalqa_indexes_complete.py:145` | 失败被静默掩盖 |
| 8 | SeaKR 仿写：只跑前 3 层，不传注意力掩码和位置信息；层调用失败时退回 embedding 均值，而 embedding 查表是确定性的，k 个样本完全相同 | `estimator:767-895` | 算出来的不是 SeaKR；本仓库用 `src/omniaware/baselines/seakr.py` |
| 9 | 2025-11-24 的"消融"是模拟运行：检索文档是 "Document about …" 占位句，准确率 3%；`SIMPLE_ABLATION_REPORT` 的数字直接写在脚本里 | `FlashRAG/experiments/archived/20251125/run_real_ablation_simple.py:149`；`FlashRAG/experiments/archived/20251125/run_ablation_simple.py:131-199` | 这些结果不是模型输出 |
| 10 | 生成长度只有 `max_new_tokens=5` | `pipeline:1129` | 多词答案被截断 |
| 11 | MMQA、MRAG-Bench、A-OKVQA 的方法对比大多只有 10 题；README 的 MRAG-Bench 表是 100 题 | `results_multimodalqa_*`、`FlashRAG/experiments/results_*`、`README.md` | 支撑不了方法排序 |

## 5. 与本仓库现有实现的对应

| 需要的功能 | 用本仓库的 |
|---|---|
| SeaKR | `src/omniaware/baselines/seakr.py`（按上游公式，AST 对照测试） |
| cross-encoder 选证 | `src/omniaware/rerank.py` |
| 模态路由 | `src/omniaware/routing.py` |
| 答案评分 | `python -m omniaware.eval.score_official` |

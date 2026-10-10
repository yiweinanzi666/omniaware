# 论文全链路代码覆盖清单

对照对象：论文 V3.21（`OmniAware_V3.21_Overleaf`，按 2026-10-10 / 10-11 的范围决定修改）。状态说明：
- ✅ 已实现并有测试（CPU 上用小模型 / 合成数据验证；真实数字要在 GPU 主机上跑）
- 🟡 代码已有，缺数据或资产（需要下载，见 §7）
- 🗑️ 已删除（代码与论文中都已删除）

## 0. 范围决定

| 决定 | 原因 |
|---|---|
| 删除 Visual-RAG、ViDoRAG、Llama-3.2-Vision、SAM-RAG | Visual-RAG 是数据集；ViDoRAG 面向文档页面且依赖 API；Llama-3.2-Vision 用交叉注意力；SAM-RAG 无可运行代码 |
| 只保留 DeBERTa-v3 NLI；删除其他检查器、长答案分析、150 题人工审核 | |
| InternVL2-40B → InternVL3.5-8B | |
| 自适应检索对照只用 SeaKR；删除 DRAGIN | 两者定位重复，DRAGIN 只能改成单次决定使用 |
| M2-RAG 只用官方单阶段 | |
| NTE 触发、PASTA 式统一偏置改为消融 | 都是本方法去掉某一部分的版本 |
| 评测：WebQA（开放检索）、OK-VQA、HybridQA、VideoRAG-Synth | 删除 MMQA-table、MMQA-full、Video-MME、InfoSeek、VideoRAG-WikiHow |
| 路由实验在 WebQA 开放检索上做 | WebQA gold 模态已知 |

## 1. 数据集与评测

| 设置 | 状态 | 加载器 / 评测 |
|---|---|---|
| WebQA 开放检索（图像题 / 文本题分组报告） | 🟡 索引已建，val 检索结果待在主机上生成 | `webqa_open_img/full`；官方 acc_approx × ParaBank BARTScore |
| OK-VQA val 5,046 | ✅ | `okvqa` / `okvqa_dense`；官方 VQAEval |
| HybridQA | 🟡 | `hybridqa`；EM / F1（UniversalRAG Table 1） |
| VideoRAG-Synth | 🟡 | `videorag_synth`；ROUGE-L、BERTScore；bootstrap 按源视频聚类 |

UniversalRAG 套件的候选来自它自己的检索器（`scripts/universalrag_retrieval/export_rankings.py`），所有对照臂共享同一候选池；Omni-Aware 在这些候选上重新计算 BGE / CLIP 相关度。视频与片段打包为一个 `video` 证据块（均匀采样帧 + 字幕），EAB 偏置覆盖整个块。

## 2. 本方法与消融

| 项目 | 状态 |
|---|---|
| U_omni、触发阈值、β=0、NTE 触发（匹配检索率 / 共享 τ） | ✅ |
| SeaKR 触发 | ✅ |
| 路由 M_m·H_m 及消融（mass、entropy、retriever、all、oracle） | ✅ |
| EAB、EAB-P、复制对照、不乘 U、PASTA 式统一偏置、始终检索 | ✅ |
| 上下文偏置、随机相关度、原始分数、按题 min-max、跨数据集校准、不校准 | ✅ |
| InternVL3.5-8B 骨干 | ✅（EAB 钩子在小模型上验证） |

## 3. 对照方法（主表，同输入的对照臂）

| 方法 | 官方代码 | 运行选项 |
|---|---|---|
| Direct、Static（始终检索） | — | 默认臂 |
| Cross-encoder 重排 + Static | bge-reranker-v2-m3 | `--selector ce` |
| SeaKR | THU-KEG/SeaKR | `--seakr-samples 20` |
| Found-in-the-Middle | 无官方代码（按论文公式） | `--fitm`（仅 Qwen3-VL） |
| M2-RAG | maziao/M2RAG | `--m2rag`（不支持视频，VideoRAG 列为 "—"） |
| UniversalRAG | wgcyeo/UniversalRAG | `--route-rule query --query-routes` |

## 4. 指标与分析（`scripts/analysis/paper_analyses.py`）

| 项目 | 状态 |
|---|---|
| 官方答案指标、配对 bootstrap | ✅ |
| 顺序：12 个带种子排列的标准差、正序−逆序、翻转率 | ✅ `order` |
| ARD | ✅ `ard` |
| Faith / AP / Inv（别名匹配 + DeBERTa NLI） | ✅ `omniaware.eval.citation.attribution` |
| WR、Cov | ✅ `routing_metrics.py` |
| AURC、Spearman ρ（U、NTE、H_grd、SeaKR） | ✅ `uncertainty` |
| 端到端延迟 | ✅ `latency` |
| 干预、Gold@2–5、相关度噪声、注入 | ✅ `--intervention`、`--relevance-noise`，汇总 `interventions` / `gold25` / `noise` |
| 数据统计、错误分类 | ✅ `stats`、`errors` |

## 5. 骨干模型

| 模型 | 状态 |
|---|---|
| Qwen3-VL-8B | ✅ |
| InternVL3.5-8B | 🟡 代码 ✅，权重未下载 |

## 6. 不实现的内容

- "3 个 seed 的均值 ± 标准差"：greedy 解码下 seed 不影响结果，改用 bootstrap 置信区间（论文中已改）。
- 相关工作中只引用、没有作为实验对照的方法（含 MAD-RAG、"Beyond the Query"）。

## 7. 需要下载的资产（在 GPU 主机上、经确认后进行）

| 资产 | 用途 | 规模 |
|---|---|---|
| InternVL3.5-8B（HF 格式） | 骨干迁移 | 约 16 GB |
| WikiTables-WithLinks | HybridQA 语料 | 数百 MB |
| HowTo100M 视频（UniversalRAG videos.txt）+ 字幕 | VideoRAG-Synth | 数百 GB；YouTube 下载需注意服务条款 |
| UniversalRAG 特征（VLM2Vec、Qwen3-Embedding） | 导出共享候选池 | 由其 `script/1_preprocess.sh` 生成 |
| spaCy en_core_web_sm、bert_score | 反事实实体类型；VideoRAG BERTScore | 小 |

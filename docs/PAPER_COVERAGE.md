# 论文全链路代码覆盖清单

对照对象：论文 V3.21（`OmniAware_V3.21_Overleaf`，按 2026-10-10 / 10-11 的范围决定修改；2026-10-10 按统一决策 D1–D18 更新，见 [DEV_DECISIONS.md](DEV_DECISIONS.md) 与 [PAPER_TABLE_MAP.md](PAPER_TABLE_MAP.md)）。状态说明：
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
| M2-RAG 只用官方单阶段摘要，外加一个项目新增的答案抽取器（论文附录披露，S-15） | 官方摘要器不产出短答案；抽取器不是官方代码的一部分 |
| NTE 触发、PASTA 式统一偏置改为消融 | 都是本方法去掉某一部分的版本 |
| 评测：WebQA（开放检索）、OK-VQA、HybridQA、VideoRAG-Synth | 删除 MMQA-table、MMQA-full、Video-MME、InfoSeek、VideoRAG-WikiHow |
| 路由实验在 WebQA 开放检索上做 | WebQA gold 模态已知 |

## 1. 数据集与评测

| 设置 | 状态 | 加载器 / 评测 |
|---|---|---|
| WebQA 开放检索（全部 4,966 道 val 为 test；图像题子集 2,511 道） | 🟡 索引已建，val 检索结果待在主机上生成 | `webqa_open_img/full`；官方 acc_approx × ParaBank BARTScore；calib 与 dev 取自训练集（D2） |
| OK-VQA：calib 500 / dev 256 / test 4,290 | ✅ | `okvqa`、`okvqa_dense`、`okvqa_gs`（默认 GS 训练语料 112,724 句，D3）；官方 VQAEval |
| HybridQA：calib 500 / dev 256 / test 1,244 | 🟡 | `hybridqa`；语料为套件子集（仅 HybridQA 段落与表格，S-14）；EM / F1（UniversalRAG Table 1） |
| VideoRAG-Synth：calib 100 / dev 50 / test 224 | 🟡 | `videorag_synth`；语料为可下载视频的子集；ROUGE-L、BERTScore；bootstrap 按源视频聚类 |

UniversalRAG 套件的候选来自它自己的检索器（`scripts/universalrag_retrieval/export_rankings.py`），所有对照臂共享同一候选池。clip 与 video 是两个独立模态（D6）。Omni-Aware 在这些候选上重新计算 BGE / CLIP 相关度，EAB 偏置覆盖整个证据块（均匀采样帧 + 字幕）。UniversalRAG 的候选顺序用套件自带检索器的分数（D6）。

## 2. 本方法与消融

| 项目 | 状态 |
|---|---|
| U_omni、触发阈值、β=0、NTE 触发（匹配检索率 / 共享 τ） | ✅ |
| SeaKR 触发 | ✅ |
| 路由 M_m·H_m 及消融（mass、entropy、retriever、all、oracle） | ✅ |
| EAB、EAB-P、复制对照（`max(1, round(exp(γUr)))`，不截断，D11）、不乘 U、PASTA 式统一偏置、始终检索 | ✅ |
| noprobe 消融（去掉探测轮，保留 first-round routing，D17） | ✅ |
| 上下文偏置、随机相关度、原始分数、按题 min-max、跨数据集校准（仅冷启动数据集之间）、不校准（D16：只改 Norm 统计量） | ✅ |
| 校准规模：100 与 500（保留区上限）；1,000 超过 OK-VQA 保留区，不运行（D1） | ✅ |
| InternVL3.5-8B 骨干 | ✅（EAB 钩子在小模型上验证） |

## 3. 对照方法（主表，同输入的对照臂）

| 方法 | 官方代码 | 运行选项 |
|---|---|---|
| Direct；Static concatenation（`static_joint`，联合 top-k，不探测、不路由，D4） | — | 默认臂 |
| Cross-encoder 重排 + Static（联合输入） | bge-reranker-v2-m3 | `--selector ce`，取 `static_joint` 臂 |
| SeaKR（联合输入，全量 test，D5） | THU-KEG/SeaKR | `--seakr-samples 20` |
| Found-in-the-Middle（联合输入） | 无官方代码（按论文公式；温度以原文为准，G3 核实） | `--fitm`（仅 Qwen3-VL） |
| M2-RAG（联合输入 + 问题图像；像素上限与其它臂一致） | maziao/M2RAG | `--m2rag`（不支持视频，VideoRAG 列为 "—"） |
| UniversalRAG（OK-VQA 列为 "—"，D6） | wgcyeo/UniversalRAG | `--route-rule query --query-routes` |

所有外部基线都使用联合 top-k 输入；Omni-Aware 及其消融使用路由后的输入（D4）。

## 4. 指标与分析（`scripts/analysis/paper_analyses.py`）

| 项目 | 状态 |
|---|---|
| 官方答案指标、配对 bootstrap | ✅ |
| 顺序：14 个排列（2 个锚点 + 12 个带种子的随机排列）的标准差、正序−逆序、翻转率；前 1,000 道 test 题（D5） | ✅ `order` |
| ARD（t_end 行、含偏置、末层头平均，D14） | ✅ `ard` |
| Faith / AP / Inv（alias 与 DeBERTa NLI 分别报告；门控臂 AP 只在检索题上计算，D9） | ✅ `omniaware.eval.citation.attribution` |
| 配对 bootstrap：双侧 p 值，最强基线由代码确定，Holm 校正（D18） | ✅ `score_official`、`paper_analyses paired` |
| 干预：attention 层面 mask，CF 替换全部别名，固定分母与失败计数（D10） | ✅ `interventions` |
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

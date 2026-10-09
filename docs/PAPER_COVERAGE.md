# 论文全链路代码覆盖清单

对照对象：论文 V3.20 的正文与附录（按 2026-10-10 的范围决定修改后）。状态说明：
- ✅ 已实现并有测试（CPU 上用小模型 / 合成数据验证；真实数字要在 GPU 主机上跑）
- 🟡 代码已有，缺数据或资产（需要下载，见 §7）
- ⚠️ 待你决定
- 🗑️ 已决定删除（论文中也要相应删除）

## 0. 范围决定（2026-10-10）

| 决定 | 原因 |
|---|---|
| 删除 Visual-RAG、ViDoRAG、Llama-3.2-Vision | Visual-RAG 是数据集；ViDoRAG 面向文档页面且依赖 API；Llama-3.2-Vision 用交叉注意力，span 接口不成立 |
| 只保留 DeBERTa-v3 NLI；删除其他检查器、长答案分析、150 题人工审核 | |
| MMQA-full → WebQA 开放检索上的路由实验 | WebQA gold 模态已知，全语料索引已有 |
| InternVL2-40B → InternVL3.5-8B | |
| SAM-RAG → UniversalRAG | UniversalRAG 有官方代码，且是路由方法 |
| 自适应检索对照采用 SeaKR（另有 DRAGIN） | |
| Video-MME → VideoRAG（UniversalRAG 套件）；新增 HybridQA、InfoSeek | 更贴合 Web 主线，且 UniversalRAG 正好用 Qwen3-VL-8B / InternVL3.5-8B 报告过这些集合 |

## 1. 数据集与官方评测

| 设置 | 状态 | 加载器 / 评测 |
|---|---|---|
| WebQA 开放检索（全池） | 🟡 | `webqa_open_img/full`；官方 acc_approx × ParaBank BARTScore |
| WebQA 图像题 / 文本题 | ✅ | 取代论文中无法定义的 1,247 子集 |
| OK-VQA val 5,046 | ✅ | 官方 VQAEval |
| HybridQA（Web 表格 + 链接段落） | 🟡 | `hybridqa`；EM / F1（UniversalRAG Table 1） |
| InfoSeek（实体图像问答） | 🟡 | `infoseek`；Acc（别名精确匹配） |
| VideoRAG-WikiHow / Synth（HowTo100M 视频） | 🟡 | `videorag_wikihow/_synth`；ROUGE-L、BERTScore |
| Video-MME-Short | ✅ | 保留代码；正文建议换成 VideoRAG |
| MMQA-table | ⚠️ | 每题自带 gold 表格，检索不起作用；建议由 HybridQA 取代 |
| MMQA-full | 🗑️ | |

UniversalRAG 套件的候选来自它自己的检索器（`scripts/universalrag_retrieval/export_rankings.py`），所有对照臂共享同一候选池；Omni-Aware 在这些候选上重新计算 BGE / CLIP 相关度。视频与片段打包为一个 `video` 证据块（均匀采样帧 + 字幕），EAB 偏置覆盖整个块。

## 2. 本方法与消融

| 项目 | 状态 |
|---|---|
| U_omni、触发阈值、β=0、NTE 触发（匹配检索率 / 共享 τ） | ✅ |
| DRAGIN 触发、SeaKR 触发 | ✅ |
| 路由 M_m·H_m 及消融（mass、entropy、retriever、all、oracle、query） | ✅ |
| EAB、EAB-P、复制对照、不乘 U、统一偏置加门控、始终检索 | ✅ |
| 上下文偏置、随机相关度、原始分数、按题 min-max、跨数据集校准、不校准 | ✅ |
| InternVL3.5-8B 骨干 | ✅（EAB 钩子在小模型上验证） |

## 3. 对照方法（同输入的对照臂）

| 方法 | 官方代码 | 状态 |
|---|---|---|
| Direct、Static、PASTA 式统一偏置、cross-encoder + static、复制对照 | — | ✅ |
| M2-RAG（单阶段 / 多阶段） | maziao/M2RAG | ✅ `--m2rag` |
| DRAGIN | oneal2000/DRAGIN | ✅ `--dragin`（需要 spaCy） |
| Found-in-the-Middle | 无官方代码 | ✅ `--fitm`（按论文公式） |
| SeaKR | THU-KEG/SeaKR | ✅ `--seakr-samples` |
| UniversalRAG | wgcyeo/UniversalRAG | ✅ `--route-rule query --query-routes`（路由器用其官方脚本训练） |

## 4. 指标与分析（`scripts/analysis/paper_analyses.py`）

| 项目 | 状态 |
|---|---|
| 官方答案指标、配对 bootstrap | ✅ |
| 顺序：12 个带种子排列的标准差、正序−逆序、翻转率、与 static 的标准差差值置信区间 | ✅ `order` |
| ARD（末层、首个答案行的块注意力 vs softmax(r/0.1)） | ✅ `ard`（运行时记录） |
| Faith / AP / Inv（别名匹配 + DeBERTa NLI） | ✅ `omniaware.eval.citation.attribution` |
| WR、Cov | ✅ `routing_metrics.py` |
| AURC、Spearman ρ（U、NTE、H_grd、SeaKR、DRAGIN） | ✅ `uncertainty` |
| 端到端延迟（prefill + 每臂解码，CUDA 同步计时） | ✅ `latency` |
| 干预：遮盖、反事实、随机编辑、重跑、干扰项、注入（答案 / 指令） | ✅ `--intervention`，汇总 `interventions` |
| Gold@2–5、相关度噪声 σ | ✅ `gold25`、`--relevance-noise` + `noise` |
| 数据统计（各模态 token、r1−r5、检索率、EAB 激活率、单 token 答案、首 token 变化） | ✅ `stats` |
| 错误分类（规则预标注 + 抽样供人工核对） | ✅ `errors` |
| 附加 NLI 检查器、人工审核、长答案行为 | 🗑️ |

## 5. 骨干模型

| 模型 | 状态 |
|---|---|
| Qwen3-VL-8B | ✅ |
| InternVL3.5-8B | 🟡 代码 ✅，权重未下载 |
| InternVL2-40B、Llama-3.2-Vision-11B | 🗑️ |

## 6. 不实现的内容

- "3 个 seed 的均值 ± 标准差"：greedy 解码下 seed 不影响结果，改用 bootstrap 置信区间（待确认）。
- 相关工作中只引用、没有作为实验对照的方法。

## 7. 需要下载的资产（都要在 GPU 主机上、经你确认后进行）

| 资产 | 用途 | 规模 |
|---|---|---|
| InternVL3.5-8B（HF 格式） | 骨干迁移 | 约 16 GB |
| WikiTables-WithLinks（HybridQA） | HybridQA 语料 | 数百 MB |
| Wiki6M + OVEN 图像分片（InfoSeek） | InfoSeek 语料与查询图像 | 数十 GB |
| HowTo100M 视频 8,620 个（YouTube）+ 字幕 | VideoRAG | 数百 GB；YouTube 下载需注意服务条款 |
| UniversalRAG 特征（VLM2Vec、Qwen3-Embedding） | 导出共享候选池 | 由其 `script/1_preprocess.sh` 生成 |
| spaCy en_core_web_sm、bert_score | DRAGIN、反事实实体类型；VideoRAG BERTScore | 小 |

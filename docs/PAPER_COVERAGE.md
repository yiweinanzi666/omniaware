# 论文全链路代码覆盖清单

对照对象：论文 V3.20 的正文与附录。状态说明：
- ✅ 已实现并有测试
- 🟡 部分实现，或只有模块、尚未接入运行
- ❌ 缺失
- 🗑️ 已决定删除（论文中也要相应删除）
- 🔍 调研中

## 0. 范围决定（2026-10-10）

| 决定 | 原因 |
|---|---|
| 删除 Visual-RAG | 它是基准数据集，不是方法 |
| 删除 ViDoRAG | 面向文档页面图像，依赖 API 大模型，不适合本文的任务 |
| 删除 Llama-3.2-Vision | 图像走交叉注意力，方法的 span 接口不成立 |
| 删除附加 NLI 检查器（FLAN-T5-xxl、"Llama-3-8B-NLI"）、长答案行为分析、150 题人工审核 | 只保留 DeBERTa-v3 NLI |
| MMQA-full → WebQA 开放检索上的路由实验 | WebQA 的 gold 模态已知（图像题 / 文本题），全语料索引已有，也更贴合 Web 主线 |
| InternVL2-40B → InternVL3.5-8B | 骨干迁移的结论不依赖模型大小，8B 成本低得多 |
| SAM-RAG → UniversalRAG | SAM-RAG 没有可运行的官方代码；UniversalRAG 有官方代码，也正好是路由方法 |
| 自适应检索对照方法采用 SeaKR | |
| Video-MME、更贴合 Web 的评测 | 🔍 调研中，见 §1 |

## 1. 数据集与官方评测

| 设置 | 状态 | 说明 |
|---|---|---|
| WebQA full val 4,966（开放检索） | 🟡 | 加载器 `webqa_open_full`、FAISS 索引已有；val 查询编码和检索结果要在主机上生成 |
| WebQA 图像题 / 文本题分组 | ✅ | 取代论文中无法定义的 1,247 子集 |
| OK-VQA val 5,046 | ✅ | 前提是先把检索修好 |
| MMQA-table | ⚠️ | 建议删除（每题自带 gold 表格，检索不起作用），待确认 |
| MMQA-full | 🗑️ | 由 WebQA 开放检索上的路由实验取代 |
| Video-MME-Short 900 | 🔍 | 已按官方协议实现；与 Web 主线关系弱，替代方案调研中 |
| 更贴合 Web 主线的评测 | 🔍 | 调研中 |

## 2. 本方法与消融

| 项目 | 状态 |
|---|---|
| U_omni、触发阈值、β=0（只用 H_grd） | ✅ |
| NTE 触发、每种格式单独阈值 | ❌ |
| DRAGIN 触发、SeaKR 触发 | 🟡 / ❌ |
| 路由 M_m·H_m 及消融（只用 M_m、只用 H_m、检索器分数、全部模态、oracle），在 WebQA 开放检索上 | ❌ → 本轮实现 |
| EAB、EAB-P、复制对照、不乘 U、统一偏置加门控、始终检索 | ✅ |
| EAB + 给上下文加偏置、随机相关度、原始分数、按题 min-max、跨数据集校准、不校准 | ❌ |

## 3. 对照方法

| 方法 | 官方代码 | 状态 |
|---|---|---|
| Direct、Static、PASTA 式统一偏置、cross-encoder + static、按相关度排序、复制对照 | — | ✅ |
| M2-RAG | maziao/M2RAG | 🟡 适配器已有，未接入运行 |
| DRAGIN | oneal2000/DRAGIN | 🟡 核心已有，未接入运行 |
| Found-in-the-Middle | 无官方代码 | 🟡 按论文公式实现，未接入运行 |
| SeaKR | THU-KEG/SeaKR | ❌ → 本轮移植 |
| UniversalRAG | wgcyeo/UniversalRAG | ❌ → 本轮接入 |
| 更新的方法（2025–2026） | | 🔍 调研中 |
| SAM-RAG、ViDoRAG、Visual-RAG | | 🗑️ |

## 4. 指标与分析

| 项目 | 状态 |
|---|---|
| 官方答案指标、配对 bootstrap | ✅ |
| 顺序：各排列的标准差 | ✅ |
| 顺序：正序与逆序之差、翻转率、跨运行汇总；ARD | ❌ |
| CiteRec / AP / Faith（只用 DeBERTa-v3 NLI） | 🟡 |
| WR、Cov | ❌ → 本轮实现 |
| AURC、Spearman ρ；端到端延迟 | ❌ |
| 干预（遮盖、反事实、干扰项）、检索噪声、注入误导段落 | ❌ |
| 统计功效与 best-of-K | ✅ |
| 附加 NLI 检查器、人工审核、长答案行为 | 🗑️ |

## 5. 骨干模型

| 模型 | 状态 |
|---|---|
| Qwen3-VL-8B | ✅ |
| InternVL3.5-8B | ❌ → 本轮实现 EAB 钩子 |
| InternVL2-40B、Llama-3.2-Vision-11B | 🗑️ |

## 6. 不实现的内容

- "3 个 seed 的均值 ± 标准差"：greedy 解码下 seed 不影响结果，改用 bootstrap 置信区间（待确认）。
- 相关工作中只引用、没有作为实验对照的方法。

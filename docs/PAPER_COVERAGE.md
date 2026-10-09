# 论文全链路代码覆盖清单

对照对象：论文 V3.20 的正文与附录（`main.tex`、`sec_experiments.tex`、`sec_appendix.tex`）。
状态说明：✅ 已实现并有测试；🟡 部分实现，或只有模块、未接入运行；❌ 缺失；⛔ 按论文的写法无法实现，需要改论文。

## 1. 数据集与官方评测

| 论文中的设置 | 状态 | 说明 |
|---|---|---|
| WebQA full val 4,966（开放检索，约 39 万图片、54 万文本） | 🟡 | 加载器 `webqa_open_full`、FAISS 索引已有；val 查询编码和检索结果还要在主机上生成 |
| WebQA img-backed 1,247 | ⛔ | 论文没有给出筛选规则（`\tbd`）。val 实际是 2,511 道图像题加 2,455 道文本题，1,247 对不上任何自然子集 |
| OK-VQA val 5,046 | ✅ | BM25 / BGE 维基百科段落；官方 VQAEval |
| MMQA-table | ✅ | 369 题（旧口径 299）；官方 list F1/EM |
| MMQA-full 2,441（在文本、表格、图片之间路由） | ❌ | 缺少全量 dev 加载器，以及在 MMQA 文本、表格、图片三个语料上的检索 |
| Video-MME-Short 900 | ✅ | 8 帧或 32 帧，字幕长度可选；官方选项解析 |
| WebQA 评分 acc_approx × BARTScore | ✅ | 官方源码固定版本 |

## 2. 本方法与消融（表 3、表 4、附录）

| 项目 | 状态 | 说明 |
|---|---|---|
| U_omni（H_grd + β·H_pred，median/IQR 归一化）、触发阈值 τ | ✅ | |
| 只用 H_grd（β=0） | ✅ | `--beta 0` |
| NTE 触发（只用下一词熵） | ❌ | 需要一个只用 H_pred 的触发选项 |
| DRAGIN 触发 | 🟡 | `baselines/dragin_*` 已有，未接入 |
| 每种格式单独阈值 | ❌ | |
| 模态路由 M_m·H_m（含 δ 并列、冷启动首轮联合检索） | ✅ | 并列时两个模态如何合并占用 k 个位置，论文是 `\tbd` |
| 路由消融：只用 M_m、只用 H_m、检索器分数、全部模态、oracle | ❌ | 需要 MMQA-full |
| EAB decode-only（ans2）、EAB-P（ans1） | ✅ | |
| EAB + 给上下文也加偏置（r=1） | ❌ | |
| 复制对照 | ✅ | `--replication-gamma` |
| 偏置不乘 U（γ·r） | ✅ | `form=pos_nou` |
| 统一偏置加门控 | ✅ | `eab:g1.2:ans2:fixed:gate` |
| 随机相关度加门控 | ❌ | |
| 原始检索分数（不做按模态百分位）、按题 min-max | ❌ | |
| 始终检索加相关度偏置 | ✅ | `eab:...:nogate` |
| γ 网格、全部层与后几层 | 🟡 | 已有 `mid` / `late`（后半层）；论文写的是"后四层" |
| 校准集大小（100/500/1,000）、不校准（启发式默认值） | 🟡 | 大小用 `--n-calib`；"不校准"缺失 |
| 跨数据集复用校准统计 | ❌ | |
| β 与 τ_bias 敏感性网格 | ✅ | 改参数分别运行即可 |

## 3. 对照方法（表 2）

| 方法 | 官方代码 | 状态 | 说明 |
|---|---|---|---|
| Direct、Static、PASTA-style 统一偏置 | — | ✅ | |
| Cross-encoder + static | FlagOpen/FlagEmbedding | ✅ | `--selector ce` |
| M2-RAG | maziao/M2RAG（MIT） | 🟡 | 适配器已有，未接入运行 |
| DRAGIN | oneal2000/DRAGIN（Apache-2.0） | 🟡 | RIND 核心已有，未接入 |
| SeaKR | THU-KEG/SeaKR（GPL-3.0） | ❌ | 官方实现基于 LLaMA-2 与修改过的 vLLM，读取内部状态；需要移植到 Qwen3-VL |
| SAM-RAG | SAM-RAG/SAM_RAG（Apache-2.0） | ⛔ | 官方仓库 README 写明"代码仍在准备中，不可使用"，没有可运行的官方实现 |
| ViDoRAG | Alibaba-NLP/ViDoRAG（无许可证声明） | ❌ | 面向文档页面图像（ColQwen2 + 多智能体），示例调用 qwen-vl-max API；用在自然图像和 WebQA 上需要大幅改造 |
| Visual-RAG | visual-rag/visual-rag（无许可证声明） | ⛔ | **这是一个基准数据集，不是方法**，仓库里只有标注文件、没有方法代码。论文把它当作方法列在表 2，需要改论文 |
| Found-in-the-Middle | 无官方代码 | 🟡 | 按论文公式实现，已有测试，未接入运行 |
| 按相关度排序的拼接 | — | ✅ | `--order perm:sorted` |

## 4. 指标与分析

| 项目 | 状态 | 说明 |
|---|---|---|
| 官方答案指标、配对 bootstrap（按视频聚类） | ✅ | |
| 顺序：各排列的标准差 | ✅ | `order.order_std_statistics` |
| 顺序：正序与逆序之差、翻转率、跨运行汇总 | ❌ | 需要一个读取 14 次运行的汇总脚本 |
| ARD（注意力与相关度的 JSD，T_r=0.1） | ❌ | 注意力质量已经记录，缺计算 |
| AP、Faith、无效引用率 | 🟡 | CiteRec 流水线已有；论文中 AP/Faith 的定义与 CiteRec 不同，需要按论文定义另算 |
| NLI 检查器 FLAN-T5-xxl、"Llama-3-8B-NLI" | ❌ | 后者不是一个明确存在的模型名，需要论文澄清 |
| 150 题人工审核 | ⛔ | 需要真人标注 |
| AURC、Spearman ρ | ❌ | |
| WR、Cov（路由） | ❌ | 需要 MMQA-full 的 gold 模态 |
| 端到端延迟（分阶段计时） | ❌ | 目前只记录每题总秒数 |
| 干预：遮盖目标块、反事实替换、干扰项 | ❌ | |
| 检索错误：Gold@2–5、相关度加噪声 σ、注入误导段落 | ❌ | WebQA 的 gold 标记已有 |
| 长答案行为（256 token、无支持陈述、重复三元组） | ❌ | |
| 错误分类与案例 | ❌ | |
| 统计功效与 best-of-K | ✅ | `scripts/analysis/power_design.py` |

## 5. 其他骨干模型

| 模型 | 状态 | 说明 |
|---|---|---|
| Qwen3-VL-8B | ✅ | |
| InternVL2-40B | ❌ | EAB 钩子目前只支持 Qwen3-VL；需要为 InternVL 写注意力钩子和 span 映射 |
| Llama-3.2-Vision-11B | ⛔ | 图像通过交叉注意力层进入模型，不在解码器序列里，论文的 span 接口对图像证据不成立（论文局限性部分自己写着 `\tbd`）；而且权重需要先接受 Meta 许可 |

## 6. 论文里有、但不应该实现的

- "3 个 seed 的均值 ± 标准差"：greedy 解码下 seed 不影响结果。如果确实要报告，必须先定义 seed 控制的是什么（例如证据顺序）。
- 相关工作里只是引用、没有作为实验对照的方法（Self-RAG、FLARE、Adaptive-RAG、MMA-RAG、UniversalRAG、OpenDecoder、PAI、MS-PoE、VisRAG 等），不需要实现。

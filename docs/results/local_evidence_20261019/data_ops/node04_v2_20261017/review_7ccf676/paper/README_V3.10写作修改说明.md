# Omni-Aware V3.10：写作修改说明

本轮只改写作，不改数据。对照依据是 `科研全流程与论文写作指南.md` 第二、三篇。
对比稿：`../diff_V3.9_V3.10/diff.pdf`。V3.9 的占位和待核对事项（`README_V3.9修改说明.md`）全部保留。

## 编译状态

正文止于第 8 页，全文 14 页（与 V3.9 相同），无编译错误、无未定义引用。

| 指标（正文） | V3.9 | V3.10 |
|---|---|---|
| 平均句长（词） | 22.9 | 20.9 |
| 超过 35 词的句子 | 29 | 18 |
| 分号 | 45 | 22 |

## 一、术语统一（指南 11.4、15.2）

| 概念 | 统一为 | 去掉的写法 |
|---|---|---|
| 基线拼接 | static concatenation（表中同名；紧凑表的表注说明 Static 的含义） | static packing、Static concat |
| 打包单元 | chunk；source 只用于"出处"（source identifier、cite sources） | 指代 chunk 的 source |
| token span | evidence span（在问题定义中定义一次） | source-aligned token span、evidence-token span、retrieved span、chunk span |
| U_omni | uncertainty score | prefill score |
| 检索决策 | trigger（规则）/ retrieval decision（结果） | retrieval policy |
| 数据集条目 | question（数据集规模、准确率）；query（方法里的输入） | 4,966 validation queries 等 |
| 对齐诊断指标 | **ARD**（attention–relevance divergence） | PosBias（名不副实）。图 4 的坐标轴仍写 PosBias，图注已说明 |
| 基线组名 | Shared-pipeline variants | Adapter-controlled（方法里没有 adapter） |

## 二、符号去重（指南 12.5）

| 原符号 | 新符号 | 原因 |
|---|---|---|
| 生成器 M_θ | G_θ | 与 M_m、M_causal、M_bias 冲突 |
| span 质量 m_j；附录中的可见副本数 m_j(t_q) | μ_j；c_j(a) | 与模态下标 m 冲突 |
| 头数 H | n_h | 与 H_m 冲突 |
| 模态 token 集合 𝕋_m | 𝓘_m | 与 span T_j 冲突 |
| t_q（"final query position"） | t_end | 与 t_end 指同一位置 |
| key 维度 d | d_k | 与 chunk d_j 冲突 |
| ARD 温度 τ_r | T_r | 与阈值 τ 冲突 |
| 求和下标 t | i′ | 与位置 t 冲突 |
| 附录 B 的平均相关度 μ(λ) | r̄(λ) | 与新的 μ_j 冲突 |
| "query row / query position"（注意力） | attention row | 与用户 query 冲突 |

M_m（模态质量）与 M_causal、M_bias（mask）仍共用字母 M，但下标不同；图 2、图 3 里印的是 M_bias，所以没有改名。

## 三、逐节修改

**摘要（10.2）**：47 词的挑战句拆成两句，每句一个挑战；"latent space … occupies there" 改写通顺；效果句补上"controlled comparisons show both contribute"；口径与引言、结论统一为绝对值（72.15 versus 68.90）。

**引言（10.3）**
- Challenge 1 把 span 长度（占位）挪到第一页；增加一条来自主表的现象证据：4 个按格式设计的基线中，只有 M2-RAG 能跑全 5 个评测集。
- 方法段从六个新概念减到只引入 evidence span 和 EAB。
- 贡献改为可验证的表述：贡献 1 给出 3.63 和 8.72 两个度量；贡献 3 指向 Proposition 1、2；贡献 4 给出"59%–70% 的标准差降幅"。

**相关工作（10.4）**
- 节标题改为 Related Work。
- 第一组改名为 "Web and multimodal RAG"，开头补上 Web 问答引擎（WebGPT、WebGLM、WebQA）。**新增的两条 bib 条目（`nakano2021webgpt`、`liu2023webglm`）请到 DBLP 核对。**
- 第二组结尾改为对整组的收束：这些方法读的是生成文本或 query 的信号，不读模型对不同模态证据的注意力。

**问题定义（10.5）**：在这里正式定义 evidence span，并说明 X_sys 和 X_query 不属于 evidence span，与方法节对齐（删掉了方法节里未定义的 "answer examples"）。

**方法（10.6、10.7）**
- 路由补上设计理由：为什么同时要求 mass 大、熵高。
- 两个阈值补上设计理由，并引用超参表中的已有数据（τ_bias=0.20 时为 67.45，τ_bias=0.40 时为 67.88）。
- 局部解释改写为正文中的 **Proposition 1（Bounded relevance tilt）**，证明放在附录 B；附录 B 改名为 "Proofs for Evidence Attention Bias"，replication 等价为 Proposition 2。
- 4.5 节改名为 Computational Cost；mask 形状、KV cache、ε 移到附录 A 的 Implementation。

**实验（6.1、10.8、12.3）**
- RQ 与挑战对应：RQ2 对应 Challenge 1，RQ3 对应 Challenge 2。
- 设置部分的顺序改为：数据集 → 指标 → 基线 → 提示 → 实现。
- 结果段补上原因：
  - 5.2 节：在 4 个设置上，相对提升随 static concatenation 的 sorted–reverse gap 单调增大（8.72 → 9.35 → 10.45 → 11.20，对应 2.69% → 4.72% → 6.63% → 11.05%）；WebQA-img 的增益按阈值和 EAB 拆分（占位）。
  - 5.3 节：NTE 为什么低于 always-retrieve（跳过检索的题准确率 78.40 对 82.15）；raw 分数为什么是最大降幅。
  - 引用支持为什么提升；校准迁移为什么下降（WebQA 是冷启动，没有 grounding 项）；错误分析指出 EAB 受限于检索分数的质量。
- 5.4.4 节原来一段 5 个话题，拆为选择性风险、校准与迁移、开销三段。
- 有小节标题的地方去掉了重复的斜体段首。
- 表格：主表给最强基线加下划线，Latency 加 ↓；Table 5 的 Std. 和 Flip 加 ↓；各表注说明 Static、Replication 的全称。正文引用附录表时都注明了 Appendix。

**结论（10.11）**：结构为"框架与挑战 → 模块 → 实验（含数字）→ 未来方向"。为了守住 8 页，未来方向压缩成一句。

## 四、新增或变化的 `\ai{}` 句子（共 9 处，填数字后核对方向）

`sec_experiments.tex` 中的位置：
- :77 reranker 与本文方法增益可叠加
- :112 按格式设阈值需要每个模态单独调参
- :114 EAB-P 在 Video-MME 上增益最大
- :138 单用 M 或 H 会提高 WR（新增了原因解释）
- :163 FITM 降低了标准差但仍高于 EAB
- :163 InternVL2-40B 上排序相同
- :187 三项干预实验
- :202 **新增**：reranker 的增益来自补回检索遗漏
- :202 **新增**：EAB 的损失主要来自相关度错误

## 五、为守住篇幅删掉或压缩的内容

- 正文删除了与 semantic entropy 的区分句。`kuhn2023semantic` 因此不再被引用，参考文献里会自动消失；如需保留，可以在附录 A 补回一句。
- 引言第一段删掉了 "The last decision also determines whether the cited chunks support the answer."
- 相关工作第一组对多模态 RAG 的逐篇介绍，压缩为按思路分三类各一句。

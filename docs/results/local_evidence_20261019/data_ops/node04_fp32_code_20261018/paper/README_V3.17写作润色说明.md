# Omni-Aware V3.17：用 Gemini 润色全文写作

对比稿：`../diff_V3.16_V3.17/diff.pdf`。全文 12 页，正文止于第 8 页，无编译错误、无未定义引用。数据、图、`\ph`、`\tbd`、`\ai` 均未改动。

## 做法

按写作指南第 22.2 节的建议，**让 AI 提出具体修改，而不是整段重写**，避免抹平语气、悄悄改变原意。

1. **第一轮，逐句修改**：把全文按节分成 24 组，每次连同全文正文一起发给 gemini-3.8-flash-high，只要逐句的"原句 → 改句 → 理由"。提示词里写明的约束：
   - 引用、交叉引用、占位符、数字、限定语一律不动；
   - 术语锁定（evidence span、static concatenation、chunk replication、trigger 与 retrieval decision、question 与 query 等）；
   - 不得用写作指南第 18、19 章列出的 AI 高频词和句式；
   - 每节总长度不得增加。
2. **自动检查**：每条建议都核对原句能否唯一匹配，引用、占位符、数学式、数字有没有被改，有没有引入禁用词。
3. **人工审读**：93 条建议逐条审读，结果是采纳 68 条、改写后采纳 22 条、拒绝 3 条。逐条对照见 `Gemini逐句修改对照.md`。
4. **第二轮，审稿式检查**：让 Gemini 以 WWW 审稿人身份检查论证结构，只找问题、不改写，共提出 25 条意见（处理结果见下文）。

## 第一轮的主要改动

- **段落逻辑**：
  - 5.4.1 节首句改为结论句"When a fixed set of retrieved chunks is permuted, EAB reduces order sensitivity"；
  - 4.4 节命题后一段把"bounded relevance prior"提前作为论点；
  - 问题定义中"context 可以为空"一句挪到定义 instance 之后。
- **指代**：二十余处 it、they、the two 等代词改为具体名词（accuracy、these signals、the target chunk、EAB 等）。
- **长句拆分**：
  - Related Work 第一组的三分句长句；
  - 4.4 节 Bias gate 和 μ_j、λ 的定义；
  - 5.4.3 节命题界与 Misled 列的关系；
  - 5.4.4 节延迟一段；
  - 附录 C 中的 checker 对比和附录 E 的局限性。
- **术语统一**：
  - Repeating chunks、Replication 统一为 chunk replication；
  - retrieved queries 统一为 triggered queries；
  - 附录中作为数据集条目的 query、example 改为 question；
  - 4.2 节 row 改为 attention row。
- **语法**：
  - 补上缺失的谓语（the coverage rate … **is** the share；videos **are** given；random relevance **gives**）；
  - 主谓一致（relevance … differs）；
  - 引导短语后加逗号；
  - RQ 问句首字母大写；
  - 结论改为现在时（We present）。
- **压缩**：
  - 引言第一段的赛道声明句与上一句重复了"必须引用来源"，合并后保留赛道声明；
  - 摘要的 benchmark 句去掉了重复的 "benchmarks"；
  - 删去 "the two WebQA sets overlap"，因为前文已说明 WebQA-img 是子集。

## 拒绝或改写的典型情况

Gemini 的建议约四分之一有问题，主要有以下几类：

| 问题 | 例子 | 处理 |
|---|---|---|
| 改变语义 | 把 "The initial $\Uomni$" 改成 "The uncertainty score"，丢了 initial（论文强调的是不重算初始值）；把 "OpenDecoder fine-tunes it"（指 generator）改成 "the decoder" | 拒绝或改回 |
| 语法不通 | "modality-level attention **routes which modality** to search"；"When retrieval **triggers**" | 改写 |
| 同义反复 | "this **correlation** is only an **association**" | 改为 "an association observed over only four settings" |
| 降低精确度 | "more citations support the answer"，原文是 "more citations point to chunks that support the answer" | 保留原文的精确说法 |
| 删掉定义 | 删除 "Here $\beta$ weights predictive entropy"（这是 β 的定义） | 拒绝 |
| 破坏 LaTeX | 有三条建议丢了收尾的 `$` | 手工修正后采纳 |

## 第二轮：Gemini 审稿意见的处理

| # | 意见 | 处理 |
|---|---|---|
| 1 | **默认的 decode-only EAB 不改变第一个答案 token，而 OK-VQA、Video-MME 等任务的答案多由第一个 token 决定，主表的增益从何而来？** | **这是需要你用数据回答的问题，见下文"需要你处理的"第 1 条** |
| 2 | 正文中出现 `\ai{}` 标记 | 不成立，这是我们自己的审阅宏，投稿前设置 `\reviewmarksfalse` 即可去掉 |
| 3 | 路由说"不需要额外的前向计算"，但输入中没有的模态没有注意力 | **已改**：4.3 节改为 "For modalities in the input, routing reads the attention of the same prefill, with no further forward pass."（没有的模态在后文用检索器分数） |
| 5 | "one threshold serves all benchmarks" 有夸大之嫌 | 前一句已写明 S 按 benchmark 和 backbone 分别估计，曾补上 "after this calibration"，但因超页撤回 |
| 6 | 78.5% 的查询要做两次 prefill，延迟为何反而更低？ | 需要事实，见下文第 2 条 |
| 7、9 | Llama-3.2-Vision 的适配方式；冷启动联合检索的时机 | 已有 `\tbd` 占位，等你补充 |
| 10 | 附录 C 中 score normalization 一段混入了 trigger 重叠分析 | **已改**：拆出 "Trigger agreement" 小段 |
| 4、8、12、13、15–25 | 贡献 3 夸大命题、摘要只报 OK-VQA 的标准差、β 未定义、基线列举过于机械、Video-MME 与 Web 的关系等 | 逐条核对原文后不改：要么原文已有相应表述（命题的局部性在 4.4 和 5.4.1 节都已说明；H_pred 有 Eq. 2 定义；基线段首句已说明分组目的），要么建议本身会造成过度声明（如把四个设置上的相关性写成因果），要么与"不利结果只压缩不删"的底线冲突 |

## 为守住 8 页做的压缩

- 4.3 节："(Section 5.3 tests each factor)" 改为 "(Section 5.3)"；
- 5.4.2 节：删 "themselves"；
- 5.4.3 节："which the Misled column measures" 改为 "(Misled column)"；
- 4.3 节："The initial $\Uomni$" 保持原文。

## 需要你处理的

1. **decode-only EAB 与首 token 的矛盾**（审稿人很可能会问）。5.1 节说明了首 token 决定的答案比例（占位），附录 C 也写了 decode-only EAB 在 32 题上首 token 从未改变；而主结果使用 decode-only EAB，在 OK-VQA 上从 66.15 提升到 67.88。请确认：
   - 主表中各数据集实际用的是 decode-only 还是 EAB-P；
   - OK-VQA、Video-MME 的答案是否多为单 token；
   - 如果增益来自多 token 答案，相应的比例是多少。

   填入 5.1 节的 `\ph` 后，正文可能需要补一句解释。
2. **延迟**：triggered queries 有两次 prefill（初始输入一次，打包后一次），但总延迟 382 ms 低于 always-retrieve 的 430 ms。请确认第一次 prefill 是否只覆盖初始输入（冷启动时只有问题，开销很小），我再据此补半句解释。

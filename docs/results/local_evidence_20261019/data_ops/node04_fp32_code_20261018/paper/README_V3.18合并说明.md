# Omni-Aware V3.18：合并朋友版本的新增内容

- **基线**：V3.17，即已加入 Gemini 润色的版本。
- **合入来源**：朋友的版本 `OmniAware_V3.17.pdf`。它基于 V3.16 修改，没有包含 Gemini 的改动，共 13 页，超出 12 页上限。
- **对比稿**：`../diff_V3.17_V3.18/diff.pdf`。
- **结果**：全文 12 页，正文止于第 8 页，无编译错误、无未定义引用。

## 一、朋友版本改了什么

我用词级 6-gram 比对了 V3.16 和朋友版本的 PDF 文本，确认实质改动只有下面几处；其余差异都是排版造成的。

| # | 位置 | 改动 | 处理 |
|---|---|---|---|
| 1 | 5.1 节 Evaluation metrics | 原文 "WebQA and MMQA use their official answer metrics" 改为说明 WebQA 评分把答案准确率和基于 BARTScore 的流畅度分开，并指向新附录 D | **合入**，压缩为一句："WebQA scores separate answer accuracy from BARTScore-based fluency (Appendix D), MMQA uses its official answer metric, …" |
| 2 | 5.3 节 | BM25 那组的 "a lexical scorer" 改为 "the official OK-VQA scorer" | **合入** |
| 3 | 附录 C 的 BM25 段与 Table 8 | 原来的占位全部换成真实结果：5,046 题、6 个臂、官方评测器，并写明运行设置 | **合入，数字一个不少**，表格改成行内列出（为了篇幅） |
| 4 | 新增附录 D：Public WebQA Scoring Diagnostics | WebQA 公开评测指标的定义（Eq. 12），以及在开发集上做的答案形式、证据选择诊断（Table 9） | **合入**：公式和定义完整保留；Table 9 改为正文叙述，保留关键数字 |
| 5 | 附录导言 | 列出的顺序写成了 "(E), (D)" | 为守住篇幅，导言整句删除 |

朋友版本中引用 DPR 语料却没有给出参考文献。我补上了 `karpukhin2020dpr`（EMNLP 2020，第 6769–6781 页），已经过 Crossref 核实。

## 二、为了守住 12 页，呈现方式上的改动

所有数字都保留，只改变呈现形式：

| 原内容 | 现在 |
|---|---|
| BM25 的 Table 8（6 行） | 一句话依次列出 6 个臂的分数：47.94、27.70、32.05、31.35、31.56、31.70 |
| 朋友版本的 Table 9（共 15 行开发诊断） | 正文保留其中的关键对比：cached static-five（45.93 acc，21.29 QA）、sentence form + recovery（68.27 acc，6.71 QA，流畅度从 24.10 降到 9.45）、text-priority（25.29 QA，+4.00 [0.51, 7.57]）、modality-routed（23.13）、576 张图像题上的 sentence form（1.56 → 15.12），以及 alias scorer。**删去的中间变体**：original candidate、text recovery、fact-answer、两个 query-form policy、sentence form only，以及 576 题中的 raw 和 384-token 两行 |
| 证据选择 Table 10 | 改为正文叙述，所有数值和置信区间都写进句子 |
| 案例 Table 11（两行都是占位） | 改成行内的 `\tbd`，等你提供例子 |
| 附录导言句 | 删除 |
| 附录中的几处措辞 | 缩短了 AURC、BART 设置、局限性中视频一句的写法 |

如果你更希望保留表格形式，就需要在别处删掉约一栏的内容。

## 三、需要你确认的问题（审稿人很可能会问）

1. **BM25 实验中，直接回答（47.94）比所有检索臂（27.70–32.05）高出约 16 分。** 正文写了 "direct answering is above every retrieval arm"，这是实话，但审稿人会追问：为什么加检索反而大幅变差？如果原因是"从 DPR Wikipedia 按问题做 BM25 检索，噪声太大"，建议补一句解释。
2. **BM25 运行的阈值写的是 (τ, β, γ) = (0.40, 0.30, 1.20)**，而主实验用的是 τ=0.35、τ_bias=0.40。请确认这里的 0.40 是 τ 还是 τ_bias，以及另一个阈值取多少。
3. **"cached" 和 "raw" static-five 的区别没有定义。** 我写成了 "cached static five-chunk packing"，请补一句说明。
4. **Table 2 中 WebQA 报告的到底是哪个分数？** 新的 5.1 节只说评分会区分准确率和流畅度。如果主表用的是官方的 QA = Acc × Fluency，或者只用了准确率，最好写明。
5. **附录 D 是"自适应开发诊断"**：变体是看过这些题目之后才设计的，BART 评分也是后加的。朋友如实披露了这一点，我原样保留。这对诚实性有利，但会让审稿人知道这批结果存在开发集过拟合风险，所以只能作为诊断，不能当作证据。

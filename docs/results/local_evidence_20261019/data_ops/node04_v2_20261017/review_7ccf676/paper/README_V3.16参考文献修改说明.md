# Omni-Aware V3.16：参考文献修正

对比稿：`../diff_V3.15_V3.16/diff.pdf`。全文 12 页，正文止于第 8 页，无编译错误、无未定义引用。

依据：GPT 的参考文献核查（贴过来的内容只到 [22]，后半部分被截断）。**每一条修改我都用 Crossref 的正式出版元数据或原文独立核实过**，没有直接照搬 GPT 的结论。

## 一、书目修改

| 条目 | 修改 | 核实依据 |
|---|---|---|
| WebQA | 题名改为 "WebQA: Multihop and Multimodal **QA**" | Crossref + arXiv |
| WebQA 作者顺序 | **不改**。IEEE/Crossref 的顺序（Chang, Cao, Narang, …）与 arXiv 不同，我们现在的顺序与 arXiv 一致。这一点 GPT 没有提到 | arXiv 2109.00590 |
| REVEAL | 作者顺序改为 Cordelia Schmid 在 David A. Ross 之前；补 DOI 10.1109/CVPR52729.2023.02238 | Crossref |
| SAM-RAG | 题名按正式出版版本，改为 "SAM-RAG: **An** Self-adaptive Framework …"（原文就是 An，语法虽然不通，但引用要照录） | Crossref（IJCNN 2025） |
| Visual-RAG | CoRR 改为 **SIGIR 2026**，第 3473–3480 页，DOI 10.1145/3805712.3808615 | Crossref |
| mR²AG | CoRR 改为 **IEEE Transactions on Multimedia 2026**（early access，第 1–11 页，DOI 10.1109/TMM.2026.3723597），补全 13 位作者 | Crossref |
| MMA-RAG | **保持 arXiv 版本**。Crossref 查不到 IJCNN 2026 或其他正式出版记录。如果作者主页确实写了 IJCNN 2026，等正式记录出来后再更新 | Crossref 检索 |
| MuRAG | William Cohen 改为 William W. Cohen | GPT 依据原文 PDF |
| Found in the Middle (Hsieh) | Long Le 改为 Long T. Le | 同上 |
| Adaptive-RAG | Jong Park 改为 Jong C. Park | 同上 |
| FLARE | Frank Xu 改为 Frank F. Xu | 同上 |
| PAI | 补页码 125–140。年份保留 2024：网络版 2024 年上线，纸本 2025 年出版 | Crossref |
| WebGLM | 补 DOI 10.1145/3580305.3599931 | Crossref |
| 新增 M3-Embedding（Findings of ACL 2024） | bge-reranker-v2-m3 的模型卡说明它基于 BGE-M3，并要求引用这篇，所以这里不应引用 C-Pack | 模型卡 + Crossref |

## 二、正文中的相应修改

| 位置 | 修改 | 依据 |
|---|---|---|
| 5.1 WebQA 检索池 | "约 390k 个文本和图像事实"改为"约 390k 张图像和 540k 条文本片段" | WebQA 论文 6.2 节："390k images and 540k text sources" |
| 5.1 基线 | bge-reranker-v2-m3 的引用由 C-Pack 改为 M3-Embedding | 模型卡 |
| 引言、相关工作：SAM-RAG | "用按格式设计的规则检查视觉证据的相关性"改为"过滤检索到的图像和段落，并核验其相关性、有用性和支持度" | 论文摘要 |
| 引言、相关工作：MMA-RAG | 改为"对输入和检索到的视觉内容的隐藏状态做分类，决定是否使用这些内容"（它是检索之后的决策，不是触发器） | 论文摘要 |
| 相关工作第二组的收束句 | "这些方法只读生成文本或 query"改为"读生成文本、query 或池化的隐藏状态"，因为 MMA-RAG 读的是隐藏状态 | 同上 |

为了守住 8 页，结论改短了：删掉了"标准差降低 59%–70%"（摘要和 5.4.1 节仍有）和未来工作的具体内容（改为指向附录 E）。5.2 节删掉了一句复述 Table 2 数字的话。附录 A 的协议说明压缩了措辞，内容不变。

## 三、还需要你处理的

1. **GPT 核查结果的后半部分被截断了**：[22]–[37] 的逐条结论，以及它说的"10 项正文审查意见"，我都没看到。把完整内容（或它生成的 Excel 表）发给我，我再逐条核实。
2. M2-RAG（arXiv 2411.16365）引用的是 2025 年的 v4 版本，GPT 建议写明版本号。ACM 的 CoRR 格式不太方便标注版本；如果你希望写上，可以改成 `abs/2411.16365v4`。
3. MMA-RAG 的 IJCNN 2026：如果你能找到正式的 IEEE 记录，告诉我 DOI 和页码。

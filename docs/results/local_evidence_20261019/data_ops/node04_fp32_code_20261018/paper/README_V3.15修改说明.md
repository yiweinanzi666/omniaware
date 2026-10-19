# Omni-Aware V3.15：补齐 GPT 评审中的剩余文字问题

对比稿：`../diff_V3.14_V3.15/diff.pdf`。
全文 12 页，正文止于第 8 页，无编译错误。剩余标记：`\tbd` 44 处、`\ph` 223 处、`\ai` 9 处。

| GPT 编号 | 问题 | 修改 |
|---|---|---|
| S5 | Table 3 的 "Relevance bias, ungated" 同时改变了检索率和 gate，不是单纯的 gate 消融 | 改名为 "Always retrieve, relevance bias"；新增一行 "Relevance bias without U scaling"（检索率固定在 78.5%，偏置为 γ·r_j，不乘 U），作为与 OpenDecoder 式冻结先验的对照（`\ph`）；5.3 节补一句 |
| S5 | "PASTA-style" 并不是完整的 PASTA | 基线描述写明：所有 head 用同一个常数，没有 PASTA 的 head profiling |
| S5 | M2-RAG 用的哪个版本、如何改造 | 加 `\tbd` |
| S6 | 输入 token 上限、各模态配额、基线调参预算、dev 集来源和大小 | 加 `\tbd` |
| 4.1 | 正文先引用了附录表（Table 7），早于 Table 5、6 出现 | 正文 4 处和 Table 5 表注都改为引用 Appendix C |
| 4.1 | O(N) 只是信号后处理的开销 | 补一句：触发检索的查询还需要第二次 prefill，EAB 需要 eager 或 SDPA attention，所以报告的是端到端延迟 |
| S8 | 图 2(A) 是拼出来的"传统方案" | 图注写明这是示意性的组合，每个被引用的系统只用到其中部分组件 |
| F4 | 冷启动时联合检索发生在 trigger 之前还是之后，结果是否复用 | 附录 A 加 `\tbd` |
| S7② | 无效引用（Inv.）的定义和分母 | 附录 A 加 `\tbd` |
| 4.1 | NLI 模型的 checkpoint 和阈值 | 附录 A 加 `\tbd` |
| 4.1 | 错误分析的样本数和抽样方式 | 附录 D 加 `\tbd` |
| S2 | ECDF 对齐的是分位数，不等于证据正确性 | 写入局限节 |

图的修改意见见 `图修改意见汇总.md`。

## 补充（复核后发现漏改的一项）

| GPT 编号 | 问题 | 修改 |
|---|---|---|
| S7② | AP 和 Faith 的分母写得有歧义（原文 "both are averaged over examples"） | 附录 A 改为：Faith 对所有答案取平均，AP 只对至少有一个引用的答案取平均；补上引用覆盖率（`\ph`），并写明覆盖率 × AP 不会超过 Faith |

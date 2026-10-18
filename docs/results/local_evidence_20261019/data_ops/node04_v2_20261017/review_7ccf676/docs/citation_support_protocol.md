> 2026-10-11：MMQA 已从评测中删除，下文中 MMQA 的行仅作历史记录。论文的 Faith / AP / Inv 由 `omniaware.eval.citation.attribution` 计算（别名匹配后接本协议的 DeBERTa 检查器）。

# E5 新运行的引用支持实验：准备与后续接口

本包准备 T08 的 CiteRec 新实验，使用当前正式全集和既有来源。它不恢复旧 Table 5 的原始输出，不声称本次检查器就是旧表的精确检查器，也不回填历史数值。当前生成调用、NLI 调用及真实 CiteRec 结果均为 0。

## 已保存的真实输入

| 数据 | 固定问题数 | 本次可用来源记录 | 来源范围 |
|---|---:|---:|---|
| OK-VQA val | 5046 | 5046 | 原顺序 5 条 BM25 知识，共 25230 条；逐条复用原 trace 中保留的文本，未扩充到截断前全文 |
| MMQA TableQ dev | 1280 | 1280 | 全部原有 table/text/image 上下文，共 26461 条；未缩成 top5 |
| WebQA val | 4966 | 257 | 当前已完成正式 top5 的 257 题，共 1285 条；其余 4709 题保留全池化候选并标 pending |

WebQA 全集含 2511 个官方图像类问题和 2455 个 text 问题。`webqa4966_unlabeled_pooled_inputs.jsonl` 保存全部 4966 题的真实候选，未保存答案或正负角色。旧表 1247 题的具体 qid 集合未恢复，本次不任意选取 1247 题。既有 CPU 检索/准备完成后，可在新版本目录重跑准备脚本；冻结后的本包不要就地换成不同 snapshot。

每条来源记录包含未来引用编号 `[E1]` 等、原 source_id、原文本、文本 SHA256、原来源顺序和 relevance；图像还保留原文件路径和既有 SHA256。编号只加入未来生成输入，不插入历史答案。原始输入和 trace 哈希记录在 `source_files_sha256.json`，本次实际校验 6591 个来源文件。图像记录继承原准备中的哈希，另检查文件存在；本次没有重新逐张计算全部图像哈希。

OK-VQA 的 question image 仍作为问题条件，E1–E5 只对应知识文本。MMQA 图像的文本支持表示只有原 image title，WebQA 图像只有原 retained caption。像素仍提供给答案生成。本实验的检查范围明确是 **text-verbalized citation entailment**，不将文字 NLI 表述为图像蕴含验证。

## 生成与 U 的边界

`static_concat` 和 `eab` 使用完全相同的新引用输出指令、来源及来源顺序。新指令要求每一条完整事实陈述独占一行，并使用实际 `[E#]` 引用。该最终输出格式是本次声明的新干预；生成时保存原 raw_answer、raw_token_ids、generation_identity，不使用会移除引用的答案抽取器。

U 的观测不随 citation prompt 重算：OK-VQA 复用原 query-image 初始观察和冻结 500 题校准；WebQA 复用原 query-only 初始观察和冻结 500 题校准；MMQA 则按现有正式定义，复用 **original SYSTEM unbiased full-QA entropy** 及 MMQA500 median/IQR。MMQA 的实际定义并非 query-only。各数据集的正式计划路径与哈希保存在 `formal_protocol_refs.json`。

本包不实现或启动答案生成。下一阶段生成 runner 应固定两臂的具体 EAB adapter、原 gate/路由语义和新输出 token cap：OK-VQA 既有正式实现为 first-answer adapter，MMQA/WebQA 为 paper token2；这些差异不可在新输入打包中被隐式改变。

## 固定的新检查器

- 模型：`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- revision：`b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`。
- 标签：0 entailment，1 neutral，2 contradiction；判定为 `argmax(logits) == 0`，无额外阈值、alias shortcut 或 QA gold 调参。
- 8 个推断资产均已下载，按上游 Git blob SHA1 或 LFS SHA256 校验。权重 870182606 bytes，SHA256 `c03cd208bf920b4fbbb182a0535859a566e8acc3477d2b536bf87b769978524b`。
- 资产位于 `cache/e5_citation_support/deberta_nli/b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7/`，配置、分词器、模型卡和权重的具体哈希见该目录 `asset_manifest.json`。

当前模型加载次数为 0。未来的最小 `run_nli.py` 为 CPU float32、batch1、单线程，记录 pair token 长度、logits、标签、耗时、版本和输入 SHA256；premise 为该陈述自己引用的有效块（见下），不截断；超过 512 token 时与 Faith/AP 一样按窗口分段，取蕴含概率最大的窗口，记录该窗口的 logits 与窗口数。命令失败会保留未完成状态，不能将失败当 0 分或发布部分指标。Faith/AP 的检查器调用（`attribution.DebertaChecker`）使用同一分窗规则（`attribution.premise_windows`），见"Faith / AP"一节。

## CiteRec 的明确计算规则

从 raw_answer 提取真正发出的 `[E<number>]`，答案级按首次出现顺序去重（记为 `all_actual_citation_ids`，仅用于记录）。分句后，逐句去除引用标记的非空文本作为 hypothesis，不是按换行符计数。每个句子的 premise 只使用**该句自己引用的**有效编号的原 rendered_text，按句内出现顺序用两个换行连接；不使用答案级的引用并集。保存原句字符区间、原句及句内引用编号。新 prompt 的一行完整陈述要求只用于输出组织，不替代分句。

分句与 Faith/AP 共用同一个函数 `attribution.sentence_spans`：NLTK 可用且英文 Punkt 数据存在时用 `PunktTokenizer('english')`，否则用同一个正则 `(?<=[.!?])\s+|\n+`。所用分句器的名称写入每个输出（`sentence_splitter`），两者不得混用。NLTK 版本与 nltk_data 的固定（revision `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`、SHA256）原先由 intake 单独检查，现由 `download`/setup 步骤负责；`PunktTokenizer.span_tokenize` 与 `nltk.sent_tokenize` 的一致性仍需在含缩写的 fixture 上核对。该分句器与 ALCE 使用同一 API/实现家族；不声称恢复 ALCE 历史运行时未固定的 NLTK 包版本。

无引用答案的全部陈述支持为 0；空答案也保留在答案分母，按一个零支持单元记录。非法编号不添加任何文本，并保留在记录中。全非法引用的陈述为 0；有效、非法混合时，仅有效文本进入 premise，非法项不获得支持。

每个答案先计算支持陈述数 / 陈述数，再在同一 dataset/variant 的所有答案上求均值，乘 100 得到 CiteRec。缺失答案和缺失 NLI 输出均导致完整性失败，不会从分母删除。`intake_answers.py` 检查传入完整来源清单的两臂覆盖；`score_nli.py` 核对 intake 的文件哈希和全请求覆盖。评分时同时核对固定 checker/runtime/full-input policy；全非法引用必须是 STRUCTURAL_ZERO、0 分且无 logits，正常请求的标签与 entailment 必须与三类 logits 的 argmax 一致。

## Faith / AP（`src/omniaware/eval/citation/attribution.py`）

- 对答案 y 的每个有效引用 e（方括号形式 `[E<j>]`，`1 <= j <= 源数`）：S(y, e) = S_alias ∪ S_nli。两者分别报告（`Faith_alias`、`AP_alias`、`Faith_nli`、`AP_nli`），头条数字 Faith、AP 是并集。
- S_alias：某个 gold 别名（规范化后、按词边界）同时出现在引用 e 的陈述中，并出现在 e 的文本中。
- S_nli：引用 e 的某个陈述被 e 的文本蕴含（DeBERTa-v3-large MNLI，判定 `argmax == entailment`）。没有陈述引用 e 时用整个答案作为 hypothesis。NLI 对每个被引用的块都计算，**不因 alias 命中而短路**。
- 前提不截断。超过 512 token 的前提按 DeBERTa 窗口分段：预算为 `512 − hypothesis token 数 − 3`，窗口间重叠半个窗口，覆盖每个前提 token；每个窗口独立计算蕴含概率，取蕴含概率最大的窗口，其 argmax 为蕴含则判定为蕴含。短前提只有一个窗口，判定与直接 argmax 相同。
- 方括号外的裸 `E<数字>`（如 "E5 模型"、"Figure E2"）不计为无效引用；无效引用只统计方括号形式（`[9]`、`[E9]`、`[e3]`、`[E 2]`）。
- 分句与 CiteRec 相同（见上文，`sentence_spans`）。
- 门控臂（`gated:<arm>`）只在检索被触发的题上适用；未检索的题不计入 AP 与 Faith 的分母，并输出 `gate_coverage`（适用题占全部题的比例）。
- `replication` 臂只有在 cite 运行中解码了时才可评分，且使用它自己的复制编号 `[E1..]`（每份副本是独立来源，`cid#k`）。

## 与官方 ALCE 的差别

官方 ALCE repository 固定在 `246c476a4edfc564266b7346b6e29ef4861ae937`，只用来核对定义；没有下载或替换为 ALCE 数据集。官方 `eval.py` 对每个句子提取局部 `[1]` 引用并拼接该句引用，以 NLTK 分句；默认使用 `google/t5_xxl_true_nli_mixture`，输出字面 `1` 表示蕴含。其 precision 还含单项和 leave-one-out 检查。

T08 本次要求答案全部引用并集，因此本包是 ALCE-style CiteRec，不能声称与官方逐句局部引用实现完全一致。官方源码和已安装 DeBERTa 源码快照、哈希及差异记录都随包保存。

## 长输入与实际验证范围

当前 38 个直接预测文件及指定 WebQA ctx 输出的快照合计 22906 条，没有发现实际 `[E#]` 或数字引用。该 census 不是对所有历史归档的穷尽声明。不存在可送入 NLI 的真实 citation premise。

分词器只统计了“每题全部可用来源都被引用”的潜在来源串，未包含尚未生成的 hypothesis 和 pair special tokens：OK-VQA max 1377 / p95 695；MMQA max 4866 / p95 2375；WebQA 当前 257 题 max 345 / p95 251。这不是实际引用长度或实际 NLI 调用成本。

当前 DeBERTa config 的 `position_biased_input=false` 使绝对 position embedding 不参与输入；relative attention 按真实长度构造距离并 bucket/clamp。源码未发现 512-token 硬上限，但 attention 内存随长度平方增长。此结论只来自配置与源码，尚无真实权重 forward、内存基准或长文本质量测试。下一阶段根据真实生成的引用串再实测运行资源。

## 后续可运行接口

1. 继续完成 WebQA 正式 top5 来源，冻结同一全集的完整 E5 citation input 版本；固定并运行两臂引用生成。
2. `python3 -B intake_answers.py --input NEW_ACTUAL_ANSWERS.jsonl --sources COMPLETE_DATASET_CITATION_INPUTS.jsonl --output NEW_INTAKE_DIR`。
3. 在模型资源允许时：`python3 -B run_nli.py --input NEW_INTAKE_DIR/cite_rec_nli_inputs.jsonl --output NEW_INTAKE_DIR/nli_results.jsonl`。
4. `python3 -B score_nli.py --ledger NEW_INTAKE_DIR/answer_ledger.jsonl --requests NEW_INTAKE_DIR/cite_rec_nli_inputs.jsonl --results NEW_INTAKE_DIR/nli_results.jsonl --output NEW_INTAKE_DIR/cite_rec.json`。

以上命令中的占位路径不是已完成输出。CPU fixture tests 只验证句子分割与 ALCE API 一致、引用并集、无引用/空答案分母、非法引用、原始记录保留、缺失答案和缺失 NLI 结果拒绝；fixture 不进入真实实验结果。

## 论文的 Faith / AP / Inv 口径（D9，2026-10-10）

论文主表和附录中的 Faith、AP、Inv 由 `omniaware.eval.citation.attribution` 按下面的口径计算（与上面的 E5/CiteRec 实验是两套独立的计算）：

- **归因支持 `gated:*` 与 `replication` 臂。** gated 臂中未检索的题不计入 AP 分母，同时报告覆盖率（retrieved share）。
- **alias 命中与 NLI 判定分别报告。** alias 判定为"gold 别名同时出现在引用它的答案陈述与被引块中"；NLI 判定为 DeBERTa 检查器的蕴含 logit 最大。两者不做短路组合（此前 alias 命中会跳过 NLI，已取消）。
- **Faith** 对所有答案取平均，**AP** 只对至少有一条引用的答案取平均；覆盖率 × AP 不超过 Faith。
- **前提长度。** 不截断；超长前提的分窗规则见上文"前提不截断"一条，CiteRec 与 Faith/AP 共用。

论文的 Faith/AP 数字必须能由 `runs/<tag>/<dataset>.jsonl` 用上述命令重算。

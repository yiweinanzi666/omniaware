# D 部分：评分与评测指标 review

审查对象：`C:\Users\22688\Desktop\www_code\omniaware`（HEAD `51df499`）。本报告只读仓库，没有修改仓库任何文件。
一次性探针脚本位于本目录：`probe_okvqa_universalrag.py`、`probe_webqa.py`、`probe_citation.py`、`probe_unicode.py`、`probe_vqa_quirks.py`。

## 0. 环境与运行范围（先说清楚能验什么、不能验什么）

- 已运行：`python -B -m pytest -p no:cacheprovider -W ignore` 于 `test_score_official.py test_attribution.py test_universalrag.py test_universalrag_suite.py test_webqa_bart.py test_citation_requests.py test_okvqa_gs.py`，结果 `21 passed, 1 skipped, 2 subtests passed`（skip 为 `test_webqa_bart.py::OfficialScoreTests`，缺官方 `bart_score.py`）。
- 因环境缺依赖/资产而未能执行的测试：`tests/test_webqa_acc.py` 整个模块 skip（缺 `spacy`；缺 `word2number`，缺 `en_core_web_sm` 模型）；`tests/test_citation_intake.py` 整个模块 skip（缺 NLTK Punkt 资产）。因此 WebQA 指标、CiteRec 流水线在本环境没有任何被执行的测试。
- 未安装且未下载（按约束）：`spacy`、`word2number`、`bert_score`、`rouge_score`、`en_core_web_sm`；`assets/`、`cache/`、`data/` 目录不存在，BART 检查点与 DeBERTa 检查器不可用。
- 探针中 WebQA 指标用的是替身：`nlp` 为恒等词形还原（lemma = 原词），`w2n` 仅覆盖 0-20 与整十数。因此依赖词形还原的匹配（如 `cats` vs `cat`）没有被复现，结论仅针对与词形无关的路径。
- 探针中 DeBERTa 检查器用恒假函数替代，只测试 alias 短路与解析逻辑，未测试 NLI 本身。

---

## 问题清单

### E-01 [严重] 引用支持 Faith / AP 的别名短路不检查蕴含，且与标准答案耦合

- 位置：`src/omniaware/eval/citation/attribution.py:55-62`（`alias_support`）、`:74-75`（`support[j] = alias_support(...) or any(entails(...))`）；论文 `paper/sec_appendix.tex:15`（定义）、`paper/sec_experiments.tex:146`（"Higher answer scores come with better citation support"）。
- 证据（`probe_citation.py`，`entails` 恒为 False）：
  - `"Paris is in Germany [E1]."`，chunk `"Paris is a city in France."`，gold `["Paris"]` -> `{'E1': True} faith 1.0`。chunk 与答案矛盾，仍计为支持。
  - `"No, the claim is false [E1]."`，chunk `"There is no evidence for this claim."`，gold `["no"]` -> `{'E1': True} faith 1.0`。短 gold 通过词边界命中无关句子。
  - OK-VQA 的 gold 是 10 个标注者的全部字符串（`src/omniaware/data.py:173`；运行记录 `run.py:535` 写入 `golds`，`attribution.py:103` 直接传入），任一命中即短路：`"The bus is red [E1]."` 命中 `red` -> `True`。
- 影响：Faith/AP 对"答对与否"敏感：答案含 gold、chunk 也含同一字符串即计为支持，无论 chunk 是否蕴含答案。论文"答案分数越高，引用支持越好"的结论，部分是构造性的（准确率上升会机械地抬高 Faith）。EAB 相对 static 的 Faith/AP 变化因此不能作为"引用质量提升"的独立证据。论文附录写的是"alias 后接 NLI"，但没有说明 alias 会绕过蕴含判断。
- 修复建议：
  1. 主指标只用 NLI（DeBERTa 蕴含），alias 仅作为诊断列报告；或报告"由 alias 决定的支持比例"（按臂）。
  2. alias 对短 gold（如 `no`、`yes`、单字符数字）设最小长度或禁用。
  3. 论文中明确写出 alias 会绕过蕴含，并给出 NLI-only 版本的 Faith，检验"高准确率 -> 高 Faith"的结论是否依旧。
  4. 与 `docs/citation_support_protocol.md:33,47` 的新协议对齐：新协议明确"无 alias shortcut"，而论文 Faith 仍使用 alias，两套定义不一致。
- 推测：未在真实运行上验证其幅度（需要真实 `[E#]` 运行，仓库中目前没有，见 `docs/citation_support_protocol.md:57`）。机制本身已用探针证实。

### E-02 [中等，取决于各臂答案长度分布] WebQA acc_approx 对冗长/对冲答案几乎不惩罚，唯一惩罚来自 BART 融合（未验证）

- 位置：`third_party/webqa/eval_webqa.md`（代码块约第 10-56 行）中 `accuracy = recall`（第 54 行）：非 color/shape/number/YesNo 类别使用 recall；`src/omniaware/eval/webqa_acc.py:43-80`（`load_official_metric`，只执行已发布的函数与常量）；`src/omniaware/eval/score_official.py:87`（调用时传入完整答案 `r['golds'][:1]`）。
- 证据（`probe_webqa.py`，用替身词形还原，结论与词形无关）：
  - `"It is not Paris"`，gold `Paris`，text -> `1.0`。
  - `"Paris, Berlin, Tokyo, Rome, Lima and Paris"` -> `1.0`。
  - 60 个词的长答案且包含 `Paris` -> `1.0`。
  - `"yes and no"`，gold `no`，YesNo -> `0.6667`；`"No, not yes"` -> `0.6667`（对冲式答案得部分分）。
  - `"blue, red, green, black, white, orange, pink"`，gold `blue`，color -> `0.25`。
- 影响：WebQA 的 QA_approx 对答案长度敏感。若 EAB 臂的答案比 static 更长/更啰嗦，recall 会抬高其 acc，只能靠 BART 融合（`webqa_bart.py:126-144`）来抵消，而融合的惩罚强度在本环境无法验证。论文主表 WebQA 列的排序依赖这一点。
- 注意：官方 `eval_1022.py` 的 text 类别同样使用 recall，所以"冗长即满分"是继承自官方设计；差异在于本实现用的是完整参考答案而非关键词，论文已说明是近似（`paper/sec_appendix.tex:124`）。
- 修复建议：报告每臂的答案 token 长度分布、BART 分量均值，并做按长度分桶的分析；在论文中加一句"acc_approx 不惩罚额外内容，排序需结合融合分量与长度解读"。
- 推测：各臂长度差异的大小未知（需要真实运行输出）。

### E-03 [严重（披露/选择偏差）] BART 复合指标在检视词汇指标之后加入，论文未披露

- 位置：`src/omniaware/eval/webqa_bart.py:3-5`（docstring："a supplemental development metric added after inspecting the lexical metric; it must not silently replace..."）、`:191`（`supplemental_metric_added_after_development_inspection: True`）、`:190`（`heldout_independence_verified: False`）；论文 `paper/main.tex:130`（"All settings are fixed on development queries by rules stated before the evaluation runs"）、`paper/sec_experiments.tex:18`（"no number in this paper comes from development questions"）、`paper/sec_appendix.tex:119-124`（QA_approx 的定义，没有说明加入时间）。
- 证据：上述代码自述与论文自述直接对照。论文只说"用于 WebQA 的 QA_approx"，未提及该指标是在看过词汇指标之后才添加的。
- 影响：若"检视"用的是评测集分数（而非 256 题开发集），WebQA 主指标就存在选择偏差，与任务中"在测试集反复调参"的历史问题同类。即使只用开发集，论文的"评测前固定"声明也不完整。
- 修复建议：在论文和 README 中写明：加入时间、依据、检视的是哪个划分（开发/校准/评测）；若检视过评测划分，需用未触碰的划分重算，或把 QA_approx 降为补充指标并与词汇指标并列报告。
- 推测：检视的划分不明（代码与论文均未说明）。

### E-04 [中等，条件性，但影响 OK-VQA 所有臂] 官方 vqaEval 只在"标注不一致"时规范化预测

- 位置：`third_party/vqa_eval/vqaEval_py3.py:98-104`（仅当 `len(set(gtAnswers)) > 1` 时才对 gt 与预测做 `processPunctuation`/`processDigitArticle`）；调用方 `src/omniaware/eval/score_official.py:64-77` 未做任何预处理。
- 证据（`probe_okvqa_universalrag.py`，官方模块未修改，10 个标注）：
  - 全部标注为 `Dog`，预测 `dog` -> `0.0`。
  - 全部标注为 `dog`，预测 `dog.` -> `0.0`。
  - 全部标注为 `2`，预测 `two` -> `0.0`。
  - 9 个 `dog` + 1 个 `dogs`，预测 `dog.` -> `100.0`。
- 评价：这是官方实现的行为，代码忠实复现，不是集成错误。但它使得得分对输出格式（大小写、句末标点、数字词）敏感。若各臂的输出格式不同（例如 EAB 更常输出句号），会造成臂间系统性偏差；论文称 "official VQA accuracy"，应注明这一点。
- 修复建议：对所有臂统一做预规范（小写、去掉句末标点、数字词转数字）后再交给官方函数，并同时报告"原始"与"规范化"两个数，检查臂间差异是否稳定。
- 推测：OK-VQA 标注是否多为小写、各臂输出格式差异多大，均需真实数据确认。

### E-05 [轻微] 官方 vqaEval 的其他怪癖（复现而未修，应在文档中注明）

- `third_party/vqa_eval/vqaEval_py3.py:137-139`：`self.periodStrip.sub("", outText, re.UNICODE)` 把 `re.UNICODE`（值 32）当作 `count` 参数，只替换前 32 个句点。`probe_vqa_quirks.py`：40 个句点仅去掉 32 个。
- `vqaEval_py3.py:62`：正则 `(?!<=\d)` 应为 `(?<!\d)`。代码推断，未单独运行。
- `vqaEval_py3.py:106`：`item != gtAnsDatum` 是字典相等比较，两个内容完全相同（没有 `answer_id`）的标注会被同时排除。`probe_vqa_quirks.py`：相同字典时得分 53.3，不同 id 时 60.0。真实 OK-VQA 标注带 `answer_id`，因此实际影响很小。
- 影响：极少数答案（句点超过 32 个）或异常标注格式，影响可忽略；但"unmodified official"的表述应注明这些怪癖被原样保留。

### E-06 [中等（实际现在无错，但校验缺失）] WebQA 官方源文件的 SHA-256 只计算、不校验

- 位置：`src/omniaware/eval/webqa_acc.py:25-28`（`OFFICIAL_FILES` 声明了期望哈希）、`:50-52`（读取并计算哈希）、`:70`（写入 provenance）。全文件中没有任何地方把计算值与 `OFFICIAL_FILES` 比较（已用 grep 确认）。
- 证据：当前磁盘文件哈希与记录一致（`Get-FileHash`：`eval_1022.py` = `6862A280…`，`eval_webqa.md` = `C6DFCB9A…`）。但代码不会在文件被改动时报错，provenance 中的 `source_sha256` 会照常写入，看起来像是"已校验"。
- 修复建议：在 `load_official_metric` 中对 `source_hashes[filename] != OFFICIAL_FILES[filename]` 直接 `raise`。同理，`webqa_bart.py` 的 `official_source/bart_score.py` 目前没有任何哈希记录，应补上。
- 推测：无。

### E-07 [中等（需数据核实）] score_official 对 WebQA 只取 `golds[:1]`，绕过了"多参考拒绝"

- 位置：`src/omniaware/eval/score_official.py:87-88`（`r['golds'][:1]`），与 `webqa_acc.py:85`（`if len(references) != 1: raise`）、`webqa_acc.py:76`（provenance 中写 "multiple references rejected"）。
- 证据：由于截断发生在调用方，`score_references` 收到的永远是长度为 1 的列表，拒绝分支从不触发。若某条 WebQA open-val 记录的 `v["A"]` 含多个答案（`src/omniaware/data.py:154`、`:71` 直接取 `v["A"]`），额外答案会被静默丢弃。仓库中没有 WebQA 数据，无法统计。
- 修复建议：当 `len(golds) != 1` 时直接报错或计数并写入 summary，不要静默截断。
- 推测：数据是否真的含多答案需要数据确认。

### E-08 [中等] 引用支持 NLI 的 DeBERTa 检查器对前提截断到 512 token，与协议"不截断"冲突

- 位置：`src/omniaware/eval/citation/attribution.py:131`（`truncation="only_first", max_length=512`）；对照 `src/omniaware/eval/citation/run_nli.py:22`（docstring/identity `truncation=False`）、`:54`（`truncation=False`）；协议 `docs/citation_support_protocol.md:37`（"完全保留真实引用并集，不截断、不拆窗"）、`:33`（判定规则）。
- 影响：论文 Faith/AP 使用的是 `attribution.py` 的 `DebertaChecker`（协议首行明确"论文的 Faith / AP / Inv 由 `omniaware.eval.citation.attribution` 计算"），因此论文 Faith 中超过 512 token 的 chunk 会被截断，产生假阴性。`run_nli.py` 与 `attribution.py` 对"同一个检查器"使用了两套输入策略，且协议只约束了前者。
- 修复建议：统一为无截断（DeBERTa 配置显示 `position_biased_input=false`，协议也称源码未见 512 硬上限，但需要实测内存），或至少报告被截断的 premise 比例。论文中写明截断策略。注意：协议第 61 行称 DeBERTa 配置中 `position_biased_input=false` 且源码未见 512 硬上限，这是协议作者的说法，本报告未验证。
- 推测：影响幅度取决于 chunk 长度分布（协议第 59 行给出 OK-VQA 全部来源拼接后最大 1377 token，单 chunk 长度未知）。需要真实 chunk 长度统计。

### E-09 [中等（未来风险；CiteRec 目前无结果）] CiteRec 的 premise 为整答案的全部引用并集，每个句子共用

- 位置：`src/omniaware/eval/citation/requests.py:21`（`premise = '\n\n'.join(evidence[c]... for c in valid)`）、`:28`（每句 hypothesis 共用 premise）；协议 `docs/citation_support_protocol.md:41`（"每个句子都使用该答案的同一个全部引用 premise"）。
- 证据（`probe_citation.py` 第 6 项）：答案 `"Paris is in France [E1]. Berlin is in Germany [E2]."`，第一句只引用 E1，但发给 NLI 的 premise 为 `Paris is in France.\n\nBerlin is in Germany.`。若第一句改为引用 E2，其支持判定仍会用到 E1 的文本，因此"错误引用"也能获得支持。
- 影响：与 `attribution.py:72-75` 的"逐引用 chunk"定义不一致，同一代码库中有两种支持定义。CiteRec 当前为 0 结果（协议 `:5`），尚未进入论文，因此属于上线前风险。
- 修复建议：CiteRec 与 Faith 统一为"句子自己的引用"作为 premise；若坚持并集定义，在论文中明确区分两者，且不要把 CiteRec 与 Faith 相提并论。

### E-10 [轻微] 无效引用标记的正则会把普通文本计为无效引用，抬高 Inv

- 位置：`src/omniaware/eval/citation/attribution.py:26-27`（`LOOSE`）、`:40`（`invalid += len(LOOSE.findall(stripped))`）。
- 证据（`probe_citation.py` 第 4 项）：`"The E5 model improved accuracy [E1]."` -> `valid=[1] invalid=1`；`"Figure E2 shows the result [E1]."` -> `invalid=1`；`"Using e5 embeddings [E1]."` -> `invalid=1`。
- 影响：回答中提到模型名、图号等 `E<数字>` 形式时，Inv 被抬高，且分母是全部标记，会影响 Inv 的解读。
- 修复建议：只把括号形式（`[e3]`、`(E2)`、`[ 3 ]`）计为无效标记，裸 `E5` 不计。

### E-11 [轻微] 句子切分不一致：Faith 用正则，CiteRec 用 Punkt

- 位置：`src/omniaware/eval/citation/attribution.py:28`（`SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")`）、`:44-52`（`statements`）；对照 `src/omniaware/eval/citation/intake_answers.py:42`（`PunktTokenizer.span_tokenize`）。
- 证据（`probe_citation.py` 第 5 项）：`statements("Dr. Smith said X [E1]. Y [E2].")` -> `[('Dr.', set()), ('Smith said X .', {1}), ('Y .', {2})]`。假设句 `Smith said X .` 残缺，并且 `Dr.` 作为独立句子进入分母。
- 修复建议：Faith 也使用 Punkt（与 CiteRec 一致）。

### E-12 [中等] 跨运行的 paired 检验：基线来自另一个运行目录，只取 qid 交集，不核对配置

- 位置：`scripts/paper_runs.py:229-236`（`analyse("paired", "runs/v321_seakr_.../d.jsonl", path, ...)`、`runs/v321_ce_...`、`runs/v321_universalrag_...`）；`scripts/analysis/paper_analyses.py:384-395`（`paired_test`，`qids = sorted(set(sa) & set(sb))`，不核对两个运行的配置、证据或校准）。
- 影响：论文 `†` 所标的比较，其基线（SeaKR、Cross-encoder、UniversalRAG）来自不同的运行目录。如果这些运行的检索、校准、token 上限或提示不同，差异就不完全来自方法。
- 修复建议：在 `paired_test` 中核对两个运行的 `_config.json` 哈希（以及 evidence 来源哈希），不一致就拒绝；在论文中注明基线运行是否与主运行同配置。
- 推测：各目录配置是否一致需要检查实际运行的 config。

### E-13 [中等] 论文的 `†`（paired-bootstrap p<0.05）没有对应的 p 值实现

- 位置：`src/omniaware/eval/score_official.py:60`（只输出 `paired_ci95_pp`，百分位 95% CI）；`src/omniaware/eval/webqa_acc.py:132-133`（同样只输出 CI）；论文 `paper/sec_experiments.tex:21`（caption："$^\dagger$: paired-bootstrap $p<0.05$"）、`paper/sec_experiments.tex:18`（"5,000 resamples"，与代码的 20×250=5000 一致）。
- 证据：代码库中没有 `p_value` 计算，也没有把 CI 转换为 `†` 的规则（`scripts/paper_runs.py:229` 的注释提到 "table main daggers"，但未见判定代码）。
- 影响：`†` 的依据不可追溯。若用的是"CI 不含 0"，则它不等价于 p<0.05 的双侧检验。
- 修复建议：输出双侧 bootstrap p 值（例如 `2*min(P(d<=0), P(d>=0))`）并写入 summary，同时在论文中说明映射规则；或把 caption 改为"95% CI 不含 0"。

### E-14 [中等] BERTScore 只用第一个参考答案，ROUGE-L 用全部参考答案取最大

- 位置：`src/omniaware/eval/universalrag_metrics.py:77`（ROUGE-L：`max(rouge_l(...) for g in golds[q])`）与 `:82`（BERTScore：`refs = [golds[q][0] for q in qids]`）。BERTScore 的模型未指定（`:79-83` 调用 `bert_score(... lang="en")`，即该包的默认英文模型，代码推断，未运行）。
- 影响：同一个 VideoRAG-Synth 指标内部，两个分数使用了不同的参考集合。论文附录中的 BERTScore 数字无法解释为"多参考"指标。
- 修复建议：两者统一使用全部参考（BERTScore 对每个参考取最大），并在代码中写明模型名与版本。

### E-15 [轻微] 空预测与空 gold 的边界

- 位置：`src/omniaware/eval/universalrag_metrics.py:17-22`（`normalize_answer`）、`:25-26`（EM）、`:33-34`（F1）、`:65`（golds 过滤 `if g`）、`:71-72`（`max()`）。
- 证据（`probe_okvqa_universalrag.py`）：`em('', 'the') = 1.0`，`f1('', 'a') = 1.0`（gold 规范化后为空，空预测被判对）；`golds=[]` 时 `max()` 抛 `ValueError`。
- 修复建议：空预测强制得 0；gold 规范化后为空时直接报错，而不是判对。

### E-16 [轻微（取决于模型是否输出非 ASCII 标点）] 标点规范化只处理 ASCII

- 位置：`src/omniaware/eval/universalrag_metrics.py:20`（`string.punctuation`，仅 ASCII）；`third_party/vqa_eval/vqaEval_py3.py:64-66`（`punct` 列表，仅 ASCII）。
- 证据（`probe_unicode.py`，控制台编码使部分字符显示为替代字符，但行为已确认）：`"Paris。"` 对 `"Paris"` 的 EM/F1 为 0；`"Paris’s"` 对 `"Paris's"` 为 0；`"café"` 对 `"cafe"` 为 0。ROUGE-L 因分词而仍得分。官方 VQA 对 `Paris。` 也得 0。
- 修复建议：统一使用 Unicode 标点类别（`unicodedata` 的 `P*`）并做 NFKC 归一化，对所有臂一致。
- 推测：模型在英文提示下输出中文标点的频率未知。

### E-17 [轻微] score_run 把 None 预测静默当作空串，无计数

- 位置：`src/omniaware/eval/score_official.py:118`（`by_qid[q]['preds'][arm] or ''`）。
- 影响：生成失败或缺失的预测会被计为错误，且没有记录；臂间空预测率的差异在输出中不可见。
- 修复建议：在 summary 中为每个臂记录空/None 预测的计数。

### E-18 [轻微] 测试覆盖与过期 docstring

- 本环境中以下测试被 skip，相关指标/流水线没有被执行：`tests/test_webqa_acc.py`（缺 spacy 与 word2number 与模型）、`tests/test_citation_intake.py`（缺 NLTK Punkt 资产）、`tests/test_webqa_bart.py::OfficialScoreTests`（缺官方 `bart_score.py`）。
- `tests/test_webqa_acc.py` 的 `test_published_empty_and_normalization_semantics` 期望 `("cats", "A cat runs", "text") -> 0.5`，依赖真实词形还原器，本环境无法复现。
- `tests/test_score_official.py:1` 的 docstring 仍写着 "OK-VQA and Video-MME"，而 Video-MME 已在提交 `ea47284` 中删除。

### E-19 [中等（推测，需要 UniversalRAG 的确切提示）] HybridQA 的 EM 对完整句子输出结构性为 0

- 位置：论文 `paper/sec_experiments.tex:14`（"HybridQA and VideoRAG-Synth keep the answer instructions of the UniversalRAG suite (the exact answer, and a complete sentence)"）；`src/omniaware/eval/universalrag_metrics.py:25-26`（EM）。
- 证据（`probe_okvqa_universalrag.py`）：`em("In 1960", "1960") = 0.0`，`f1 = 0.6667`。
- 影响：若提示要求"完整句子"，EM 会系统性地把正确但带句子的答案判错，而 F1 会部分得分。EM 与 F1 的结论可能因此相互矛盾。
- 修复建议：核对 UniversalRAG 的确切 HybridQA 提示（仓库未包含），并报告输出是否为"精确答案"的比例；若提示是完整句，应主报 F1 或改用答案抽取。
- 推测：提示内容未在仓库中核实。

---

## 已核实无问题的部分（避免误报）

- OK-VQA：`score_okvqa` 使用未修改的官方 `vqaEval_py3.py`（`score_official.py:68-77`），gold 直接读取原始标注文件（`:64-67`，不走 data.py 的清洗）。`min(1, n/3)` 与 10 选 9 平均正确（`vqaEval_py3.py:106-108`）。没有子串匹配（`not a cat` 对 `cat` 为 0，`hotdog` 对 `dog` 为 0）。
- WebQA 归一化：`webqa_bart.py:35-38` 与 `eval_1022.py:122-124`（`normalize_text_for_bart`）逐字一致；`webqa_acc.py` 通过 AST 只加载了已发布的函数与常量，并且 hash 与 provenance 相符（见 E-06 的校验缺失问题，但文件本身未被改动）。
- BART 融合的方向与归一化：`webqa_bart.py:126-131` 计算 `score(ref, pred)` 与 `score(ref, ref)`，与 `eval_1022.py:126-130, 183-184` 的 `compute_bartscore_ParaBank(C*len(A), A)` 一致；`normalized_bart` 等价于 `min(1, exp(s(r,y)-s(r,r)))`，与论文式 (eq:webqa) 一致（`paper/sec_appendix.tex:121`）。
- HybridQA EM/F1：SQuAD v1.1 归一化、max over (answer + aliases) 正确（`universalrag_metrics.py:17-43, 71-72`）。
- ROUGE-L：分词（`_rouge_tokens`）与 LCS 的 precision/recall/F 正确；未对照 `rouge_score` 库（未安装），属于凭记忆判断。
- 所有臂经过同一个 `score_run`（`score_official.py:119-125`），后处理只有 `strip()`（`baselines/arms.py:39`、`qwen_eab.py:324,339,352`），没有臂专属的截断或前缀剥离（`run.py:561-669`）。`universalrag`、`gated:*`、`nte_*` 臂复用同一个 `preds` 字符串，是定义上的复用而非数据差异。
- 配对 bootstrap：5000 次重采样（`score_official.py:51`，20×250），VideoRAG 按视频聚类（`score_official.py:44-57`；聚类键由 `universalrag_metrics.py:88` 生成），与论文 `sec_experiments.tex:18` 的描述一致（但 p 值问题见 E-13）。
- 三个 WebQA 指标路径（`acc_approx`、`acc × BART`、`webqa_bart.summarize_composite` 的 CI）在代码层面都是按问题配对的。

---

## 没来得及深入的方向

1. WebQA 的 BART 融合：官方 `bart_score.py` 不在仓库，无法核对每 token 平均、截断、batch 填充与 `load` 的严格性。`tests/test_webqa_bart.py::OfficialScoreTests` 正是为此而写，但在本环境被 skip。
2. 词形还原与数字词：未安装 spacy/word2number，未验证 `cats` vs `cat` 等词形匹配，也未验证 `w2n` 对复合数字（"two hundred"）的处理。
3. 与 `rouge_score`、`bert_score` 的数值对照：未安装，未能直接比较。
4. NLI 检查器的真实行为：模型与资产缺失，未测试 DeBERTa 对真实 chunk 的蕴含判断、长 premise 的截断比例、显存/时间。
5. 真实运行的答案统计：各臂答案长度、标点/大小写分布、OK-VQA 标注的大小写情况、WebQA 多答案比例，都需要真实数据，无法在本环境完成。
6. 基线运行的配置一致性：E-12 中各 `runs/v321_*` 目录的 config 没有逐一比较。
7. M2-RAG 基线：其最终答案经过一个独立的提取器（`baselines/arms.py:71-100`，文档 token 上限 384，与其他臂的上限不同），其输出形式与其他臂是否可比未分析。
8. 论文中大量数字仍是占位符（`\ph{}`、`\tbd{}`），因此无法核对论文数字本身，只能核对指标定义。

---

## 文件路径（绝对路径）

- 报告：`C:\Users\22688\AppData\Local\Temp\claude\C--Users-22688-Desktop-www-code\8e3454d9-d7cd-4088-813b-e62b2bcd35b9\scratchpad\review\D_eval.md`
- 探针：同目录下 `probe_okvqa_universalrag.py`、`probe_webqa.py`、`probe_citation.py`、`probe_unicode.py`、`probe_vqa_quirks.py`
- 主要代码：`C:\Users\22688\Desktop\www_code\omniaware\src\omniaware\eval\score_official.py`、`...\eval\webqa_acc.py`、`...\eval\webqa_bart.py`、`...\eval\universalrag_metrics.py`、`...\eval\citation\attribution.py`、`...\eval\citation\requests.py`、`...\eval\citation\run_nli.py`
- 官方参考：`C:\Users\22688\Desktop\www_code\omniaware\third_party\vqa_eval\vqaEval_py3.py`、`...\third_party\webqa\eval_1022.py`、`...\third_party\webqa\eval_webqa.md`
- 协议与论文：`C:\Users\22688\Desktop\www_code\omniaware\docs\citation_support_protocol.md`、`...\paper\sec_experiments.tex`、`...\paper\sec_appendix.tex`、`...\paper\main.tex`

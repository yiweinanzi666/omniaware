# 对照方法（baselines）正确性与公平性审查

仓库：C:\Users\22688\Desktop\www_code\omniaware（只读审查，未修改仓库文件；一次性脚本与输出只在 scratchpad\review\ 下）

## 0. 结论概览

- 最严重的问题是 M2-RAG 在 OK-VQA 上丢失了问题图片（B1）。这会让 M2-RAG 的 OK-VQA 数字失去意义，而 M2-RAG 又是 WebQA 上的最强基线。
- 论文中"所有方法共享候选、骨干、提示词、截断规则"的表述与代码不符。冷启动数据集上，Static、CE、FITM、M2-RAG、SeaKR 都继承了 Omni 的 M_m·H_m 路由（B2）；M2-RAG 额外加了一次项目自定义的答案抽取（B5）；SeaKR 只在 1000 题子集上跑且正文未披露（B4）。
- FITM 的温度 5e-5 让校准系数近似 one-hot，整行注意力被集中到单个文档（B3，已数值复现）。FITM 的 Qwen 适配也没有在论文中披露（B8）。
- 基线没有任何 dev 调参，论文中的 "[[tuning budget]]" 占位符未填，公平性声明无法成立（B7）。
- 外部依赖的 commit 与上游 HEAD 一致（ea6b99、d75fe7、d29ab3 已核对）。但 M2-RAG 的 commit 只在测试中校验，运行时不校验（B13）。
- 测试：指定的 9 个测试文件中，8 个被收集，共 29 个用例，28 通过，1 跳过（test_m2rag_official.py:71，legacy T15 输出缺失）。test_m2rag_qwen.py 整文件因缺少 torchvision 被跳过（2 个用例未运行），所以 M2-RAG 的 Qwen 后端在本环境从未执行（B14）。运行时可能在仓库内生成 __pycache__，未生成 .pytest_cache（使用了 -p no:cacheprovider）。

## 1. 问题清单

| ID | 严重程度 | 标题 | 位置 | 推测 |
|---|---|---|---|---|
| B1 | 致命 | M2-RAG 在 OK-VQA 上丢失问题图片 | arms.py:77-91；m2rag_webqa.py:42 | 否 |
| B2 | 严重 | 冷启动数据集上 Static/CE/FITM/M2-RAG/SeaKR 继承 Omni 的路由，候选集不同 | paper_runs.py:61-62；run.py:565-566, :600, :634-654；run.py:52-56 | 幅度推测 |
| B3 | 严重 | FITM 温度 5e-5 使 α 近似 one-hot，单个文档吞掉整行注意力 | fitm_attention.py:12, :32-41, :58-68；tests/test_fitm_attention.py:29, :32-33 | 与原文量纲推测 |
| B4 | 严重 | SeaKR 只在前 1000 题上跑，其余方法跑全量，正文未披露 | paper_runs.py:29, :120；paper/sec_experiments.tex:40 | 否 |
| B5 | 严重 | M2-RAG 多了项目自定义抽取器，且截断、预算、提示词与其他臂不同，正文未披露 | arms.py:71-91；m2rag_webqa.py:3-25, :45-59；pipeline.py:20, :51, :61-62 | 否 |
| B6 | 严重 | UniversalRAG 路由到语料中不存在的模态时静默退化为全模态联合检索 | universalrag.py:62-69；run.py:52, :613-615 | 频率推测 |
| B7 | 严重 | 基线无 dev 调参；论文 "[[tuning budget]]" 未填；公平性声明不成立 | paper/sec_experiments.tex:12；paper_runs.py:83-90；dev_select.py | 否 |
| B8 | 中等 | FITM 的 Qwen 适配（null reference、视觉置零、逐层逐头 α、冻结 α）未在论文中披露 | fitm_qwen_adapter.py:1-5, :19-20, :118-151, :162-233；docs/PAPER_COVERAGE.md:50 | 否 |
| B9 | 中等 | SeaKR 移植偏离上游：文档数、采样过滤、阈值、种子；docstring 称 "verbatim" 不准确 | seakr.py:1-14, :54, :63-67；external/SeaKR eval_singlehop.ipynb cell 8 | 上游文档数推测 |
| B10 | 中等 | 论文对 SeaKR 成本的描述偏低（漏掉逐文档评分的约 100 次采样） | paper/sec_experiments.tex:175；run.py:642-654 | 否 |
| B11 | 中等 | CE 臂的 "BGE 顺序" 实际是 Omni 的 ECDF 校准相关度；tie-break 用 cid | rerank.py:1-6, :54-56, :82-83, :128-132；run.py:262-266 | 否 |
| B12 | 中等 | bge-reranker-v2-m3 无 revision/哈希校验；ce_logits 缓存只按 qid 复用 | rerank.py:97-125, :107-108；third_party/README.md:15 | 否 |
| B13 | 中等 | M2-RAG commit 只在测试中校验；运行时优先使用 cache 副本 | m2rag_official.py:17, :19-20；tests/test_m2rag_official.py:61 | 否 |
| B14 | 中等 | 本环境 M2-RAG Qwen 后端整文件未执行（缺 torchvision） | tests/test_m2rag_qwen.py:5 | 否 |
| B15 | 中等 | FITM 测试是自洽测试，并锁定饱和行为，无法发现 B3 | tests/test_fitm_qwen_adapter.py:45-74；tests/test_fitm_attention.py:29, :32-33 | 否 |
| B16 | 中等 | UniversalRAG 训练集过滤只按 index；路由器是手工步骤，未固定 | universalrag.py:9-12, :72-74, :94-98；paper_runs.py:129-131 | 泄漏程度推测 |
| B17 | 中等（潜在，论文主表未用） | 旧 WebQA 闭集加载器把 gold 编码进 cid 且 gold 排在前 | data.py:47-50, :54-61；rerank.py:54-56；run.py:52-69 | 是 |
| B18 | 轻微 | InternVL 下 SeaKR 路径未验证（FITM/M2-RAG 有守卫，符合论文） | run.py:199-200；arms.py:58-59；seakr.py:50-62 | 是 |
| B19 | 轻微 | 空候选时 FITM 抛异常，会中断整个 run | fitm_attention.py:19-23；arms.py:56-67 | 频率推测 |
| B20 | 轻微 | 杂项：M2-RAG 解析器覆盖在单阶段下是惰性代码；SeaKR 每题种子相同；[:5] 硬编码；raw_files 分支不可达；CE 的 MODEL_* 常量未使用 | m2rag_official.py:73-80；seakr.py:54；rerank.py:84, :65-69；fetch_external.py:43-47 | 否 |

## 2. 详细问题

### B1 致命：M2-RAG 在 OK-VQA 上丢失问题图片

- 位置：src/omniaware/baselines/arms.py:77-91（m2rag_answer 只从 chunks 构造 sources，从不读取 e["query_images"]）；src/omniaware/baselines/m2rag_webqa.py:42（official_query 中硬编码 aux_images=[]）。
- 证据：
  - data.py:192 中 OK-VQA 的 query_images 非空。
  - 在 baselines 目录 grep "query_images"，只有 universalrag.py:43 引用。
  - 一次性脚本 checks.py 的输出：`3 official_query aux_images: []`；`3 message part types sent to backend: ['text', 'text', 'text']`，即官方 summarizer 发给后端的消息中没有任何图片。
  - paper_runs.py:116-117 对所有非视频数据集加上 --m2rag，因此 OK-VQA 也受影响（run.py:197-198 只拦截视频）；调用点见 run.py:637-641。
- 影响：OK-VQA 的问题图片是唯一的上下文（paper/sec_experiments.tex:8 写明 "Only OK-VQA supplies context, the question image"）。M2-RAG 在 OK-VQA 上实际上变成了纯文本 RAG，其分数被人为压低，Omni 相对它的增益被夸大。M2-RAG 也是 WebQA 上的最强基线（paper/main.tex:125 "versus M2-RAG on WebQA"；sec_experiments.tex:52），因此对显著性检验的 "strongest baseline" 判断也会受影响。
- 修复：把 e["query_images"] 作为 aux_images 传入 official_query，并在 extractor 消息中保留它；为 OK-VQA 增加测试，断言 summarizer 与 extractor 的消息中包含问题图片；重跑 OK-VQA 上的 M2-RAG。
- 推测：否。机制已通过脚本复现。官方 aux_images 的语义是否正是用于这种情形，需要对照官方 README，但这不影响结论。

### B2 严重：冷启动数据集上 Static/CE/FITM/M2-RAG/SeaKR 继承 Omni 的路由

- 位置：
  - scripts/paper_runs.py:61-62：COLD 数据集（webqa_open_full、webqa_open_img、hybridqa、videorag_synth）加上 `--probe-uncertainty --first-round-routing`。
  - paper_runs.py:65-67（run() 调用 method(d)）以及 :118-121（main、ce、seakr 运行都经过这里）。
  - run.py:377-378（routed 标志）、:410-415（resolve_route 用探测轮推导 M_m·H_m 路由）、:565-566（select_chunks 使用该路由）。
  - run.py:600（static 臂与 Omni 共用同一个 enc）、:634、:639-640、:644-654（fitm、m2rag、seakr 使用同一组 chunks）。
  - run.py:52-56（select_chunks 对 route.modalities 做过滤）；routing.py:53-56（evidence_route）。
  - 论文佐证：docs/DEV_DECISIONS.md:16 "冷启动数据集一律使用探测轮"；docs/PAPER_TABLE_MAP.md 主表各行（Static、CE、FITM、M2-RAG 都来自 v321_main 或 v321_ce，即同一运行）。
- 证据：脚本 checks.py 中 `2b route modalities=['image'] top-2 -> ['i1']`，而 `2a route=None top-2 -> ['t1','t2']`。当路由选中图像模态时，static 只剩图像候选。
- 影响：
  - 论文 sec_experiments.tex:12 写的是 "always-retrieve static concatenation"，并且 "every method reads the same candidates"（sec_experiments.tex:12 及 docs 中的类似表述）。实际上 Static 行是"Omni 路由后的 top-k"，而不是"共享候选池上的 joint top-k"。CE、FITM、M2-RAG、SeaKR 同理。
  - 因为 Omni 和 Static 共享同一路由，Omni 相对 Static 的增益不再只代表"触发 + EAB"，还混入了 Static 对路由错误的脆弱性。
  - 影响的具体大小需要重跑，我没有 runs/ 目录。推测：在 WebQA 的文本/图像池差异较大时，Static 会被压低。
- 修复：为 Static、CE、FITM、M2-RAG、SeaKR 单独跑一组不带 --first-round-routing 的 joint top-k（至少对 Static 和 CE 这么做），或者在论文中明确写出 "baselines inherit Omni's route" 并说明这对解读的影响。两种方案都需要在 sec_experiments.tex:12 中修正文字。
- 推测：幅度是推测；机制已核实。

### B3 严重：FITM 温度 5e-5 使校准系数近似 one-hot

- 位置：src/omniaware/baselines/fitm_attention.py:12（TEMPERATURE = 5e-5）；:32-41（calibrated_alpha，logits = (difference - max)/T）；:58-68（目标质量 target_mass = normalization × counts × α，非目标文档的 α 近似为 0，因此它们的质量也趋近 0）。
- 证据：脚本 scratchpad\review\fitm_saturation.py 的输出。构造一行 K=600 的注意力，5 个 chunk 各 100 个 token，chunk 1 的注意力是均匀的 2 倍。
  - 原始文档均值：[0.001429, 0.002857, 0.001429, 0.001429, 0.001429]；参考均值全部为 0.001667。
  - α（T=5e-5）：[3.9e-13, 1.0, 3.9e-13, 3.9e-13, 3.9e-13]。
  - 校准后的文档质量：[3.3e-13, 0.857, 3.3e-13, 3.3e-13, 3.3e-13]。
  - 当文档均值差只有 1e-4 时，α = [0.088, 0.649, 0.088, 0.088, 0.088]，已经明显偏向单个文档。
  - tests/test_fitm_attention.py:32-33 中的期望值本身就是 one-hot（`[[1,0,0],[0,1,0]]`），说明测试把饱和行为当成了正确结果。
- 影响：按这个实现，FITM 在大多数行上会把全部文档质量压到"均值差最大的那一个文档"上，其余文档的质量趋近 0。这不是位置偏差校准，而是删除了其余证据。如果论文 Table order 中 FITM 的偏差下降（sec_experiments.tex:144 "lowers the deviation to ..."）来自这种退化，那么该结论不能代表 Found-in-the-Middle。
- 修复：
  1. 查阅 Hsieh et al. 2024 原文，核对 α 公式（式 4）与温度值的量纲。原文中的文档均值是否为概率量级（约 1/K）？若是，T=5e-5 与原文不一致，需要改正。
  2. 在 scratchpad 里补一个独立的小例子，用手算值做 golden 测试，替换 test_fitm_attention.py:29 对常数的 pin。
  3. 在论文附录中披露温度选择及其理由。
- 推测：是。原文 T 的取值和量纲无法在本环境核对（无法访问原文）。饱和本身已确凿，但它是否也是原文的行为需要核实。

### B4 严重：SeaKR 只在前 1000 题上跑，正文未披露

- 位置：scripts/paper_runs.py:29（SUBSAMPLE = "1000"）、:120（`--n-eval SUBSAMPLE` 用于 v321_seakr）；docs/PAPER_TABLE_MAP.md:16（SeaKR 行写明"每个数据集前 1,000 题"）；scripts/analysis/paper_analyses.py:390（paired_test 取 qid 交集，SeaKR 与其他臂的显著性在 1000 题上计算）。
- 论文正文 sec_experiments.tex:12、:40、:56、main.tex:130 均未提及子集。sec_experiments.tex:40 的表行与其他行无区别。
- 影响：主表 SeaKR 行与其他行的数字来自不同的题目集合，不可直接比较。显著性检验只在 1000 题上做。main.tex:130 "outperforms ... SeaKR" 的声称受影响。
- 修复：SeaKR 跑全量（成本需要估计：每题 20 次采样，触发时再加 5 × 20 次）；或者在表注中写明子集，并对 Omni 与其他臂同样在子集上报告。
- 推测：否。

### B5 严重：M2-RAG 的 QA 流程不是论文所述的"官方单阶段摘要"

- 位置：
  - arms.py:71-91：summarizer 生成 384 个 token 的文档（默认 document_tokens=384，arms.py:72、:86）；随后由项目的 extractor 输出答案（arms.py:89-90，预算为 max_new）。
  - m2rag_webqa.py:3-10：EXTRACTOR_INSTRUCTION 为项目新增，要求 "Return only the answer. If the response does not answer the question, return 'Insufficient information'"。
  - m2rag_webqa.py:13-25：prepare_sources 对每个文本源截断到 256 个子词（:19-20），对所有模态统一应用。
  - pipeline.py:20 MAX_DOC_WORDS = 160，pipeline.py:23-25（按词截断）；pipeline.py:51（图像 caption 截断到 60 词）；pipeline.py:61-62（表格完全不截断）。
  - external/M2RAG 中 grep "Insufficient" 与答案抽取相关代码，没有任何结果。官方代码只有摘要，没有答案抽取。
  - paper/sec_experiments.tex:12 声称 "All methods share ... the prompt format, decoding limits, ... truncation rules"；sec_experiments.tex:14 声称 "All configurations receive identical instructions"。
- 影响：
  - M2-RAG 的 QA 分数来自"官方摘要 + 项目抽取器"，而论文只说了 "official single-stage summarizer"。抽取器的指令与 pipeline.py:11-16 的 INSTRUCTION（例如 WebQA "one short sentence"）不同，因此答案格式也不同。
  - 证据截断不一致：M2-RAG 把表格截到 256 个 token，而 Static 不截断；M2-RAG 的 caption 不截断，而 Static 截到 60 词。HybridQA 最受影响。
  - M2-RAG 比其他方法多了一次生成调用，生成的文档长度（384）与其他臂的解码上限也不同。
- 修复：在论文中披露抽取器和 384 预算；统一截断函数（所有臂使用同一 token 级截断）；为 M2-RAG 使用与其他臂相同的答案格式指令，或者在论文中说明差异并给出敏感性分析。
- 推测：否。

### B6 严重：UniversalRAG 路由到语料中不存在的模态时静默退化为全模态联合检索

- 位置：src/omniaware/baselines/universalrag.py:62-69（to_route：`retrieve = bool(mapped)`，而 `modalities = sorted(mapped & available)`，即当路由标签都不在语料中时，retrieve=True 且 modalities=[]）；src/omniaware/run.py:52（select_chunks 仅在 route["modalities"] 非空时过滤）；run.py:613-615（UniversalRAG 臂取 static 的预测）。
- 证据：checks.py 的输出：
  - `to_route(['table'], {text,image})` → `{'retrieve': True, 'modalities': []}`。
  - `to_route(['video'], {text,image})` → 同上。
  - `2c UniversalRAG route for ['video'] (modalities empty) top-2 -> ['t1','t2']`，即检索了全部模态。
- 现有测试 tests/test_universalrag.py:6-10 只覆盖了"部分标签可用"的情形，没有覆盖"全部标签都不可用"。
- 影响：UniversalRAG 的预测既不是"不检索"，也不是 UniversalRAG 的路由结果，而是 Omni 默认的联合检索。论文 sec_experiments.tex:12 声称 UniversalRAG "decides from the query whether and where to retrieve"，与此不符。
- 修复：对 retrieve=True 且 modalities 为空的情况显式处理（视为"路由到空集"、按 direct 处理，或者计入 wrong-route 并报告比例）；加上对应测试。同时统计 routes.json 中此类查询的数量。
- 推测：此类查询的频率是推测。需要检查 routes 文件。

### B7 严重：基线没有 dev 调参，论文占位符未填，公平性声明不成立

- 位置：
  - paper/sec_experiments.tex:12：`baselines receive \tbd{[[tuning budget]]}`，占位符仍在。
  - scripts/paper_runs.py:83-90：dev 阶段只跑 Omni 的 EAB 形式（EAB_FORMS）、zsum 与 mid；没有任何基线。
  - scripts/analysis/dev_select.py（文件头部）：只选择 Omni 的 EAB 形式、不确定性设置和 OK-VQA 语料。
  - 基线超参全部固定：FITM 温度（fitm_attention.py:12）、M2-RAG 文档预算（arms.py:72）、SeaKR 的 n=20 与 T=1.0（paper_runs.py:120；seakr.py:51）、SeaKR 阈值由 calibration 的检出率匹配（run.py:487-488）、UniversalRAG 的 top（run.py:128）。
- 影响：Omni 的形式、不确定性设置和语料都经过 dev 选择，而基线使用默认值，没有 dev 选择的机会。论文中"公平"的声明无法成立，也不能说"基线获得了相同的调参预算"。
- 修复：对每个基线在同一 dev 上给出同样数量的候选（或说明为何不需要），并把选择规则写入 DEV_DECISIONS.md；或者在论文中明确写 "baselines use upstream or fixed defaults and receive no dev tuning"，并删除占位符。
- 推测：否。

### B8 中等：FITM 的 Qwen 适配未在论文中披露

- 位置：
  - fitm_qwen_adapter.py:1-5（docstring 自己承认 "null content, independent layer/head weights, frozen alpha and query timing are explicit project adaptation choices"）。
  - fitm_qwen_adapter.py:19-20（DUMMY_TEXT）、:118-151（make_null_reference，文本替换为 "This is a reference document." 的循环 token）、:148（视觉 null 将 pixel_values 置零）、:162-233（每个 target 一次 null forward，共 K+1 次前向）。
  - docs/PAPER_COVERAGE.md:50："Found-in-the-Middle | 无官方代码（按论文公式）"。
  - paper/sec_appendix.tex 中没有任何 FITM 适配的说明（grep 无结果）；sec_experiments.tex:12 只写 "Found-in-the-Middle attention calibration on the static input"。
- 影响：读者会以为这是 FITM 原方法。视觉 null 把图像 patch 置零，会产生异常的图像 token 表征，而不是"空白"参考。额外前向的成本也没有在主表中体现。
- 修复：在附录中列出每一条与原方法的偏离（null reference 的构造、视觉置零、每层每头独立 α、只校准最后一行、α 在首个答案 token 前冻结、K+1 次前向）。
- 推测：否。

### B9 中等：SeaKR 移植偏离上游

- 位置：
  - seakr.py:1-14（docstring 称 "verbatim" 与上游一致，但下列偏离未说明）。
  - seakr.py:54（每次调用固定 seed=42，所有题目的采样随机流相同）。
  - seakr.py:63-67（EOS 样本过滤用 `if emitted`，即"发出过至少一个非 EOS token"，而上游是 `cpl.text.strip()`，见 external/SeaKR/vllm_uncertainty/vllm/engine/llm_engine.py:725）。
  - run.py:650-654：对 k=5 个 chunk 逐个评分；上游 external/SeaKR/eval_singlehop.ipynb cell 8 中 `rag_batch = rag_results.iloc[10*i : 10*i+10]`，每题取 10 行。
- 证据：eigen score 公式与上游一致（tests/test_seakr.py:14-22 的 AST 比较通过）；层索引 `hidden_states[layer+1]` 与上游 llama.py:299 的 0 基输出索引一致；阈值方向（检索当 score > threshold）与上游 notebook 一致。这些地方没有问题。
- 影响：
  - 文档数：论文要求统一 k=5，但上游 SeaKR 每题使用 10 个文档。这是一个需要披露的偏离。
  - 采样过滤：上游对无 EOS 的非空样本，`torch.stack` 中会出现 None，实际上会报错；移植版直接丢弃。这不是致命问题，但 docstring 的 "upstream drops ..." 不准确。
  - 阈值：上游固定为 -6.0（LLaMA-2 的值），移植版改为 calibration 检出率匹配，这已在 sec_experiments.tex:12 中说明，可以接受。
- 修复：在论文附录中披露文档数（k=5 而非上游 10）；修正 seakr.py 的 docstring；每题使用不同的种子（例如 hash(qid)）。
- 推测：上游 notebook 的 10 行是否对应 10 个文档，需要核对 main_simpleqa.py 中的生成脚本，此处为推测。

### B10 中等：SeaKR 的成本描述偏低

- 位置：paper/sec_experiments.tex:175（"against ... the 20 sampled generations behind each SeaKR decision"）；run.py:642-654。
- 证据：代码在触发时，对直接问题做 20 次采样（run.py:644），然后对每个 chunk 做一次贪心作答（run.py:652）并做 20 次采样评分（run.py:653-654）。即触发时的额外成本约为 20 + 5 × (1 + 20) = 125 次解码，而不是论文所述的 20 次。
- 影响：延迟表中 SeaKR 的成本被低估（它的墙钟时间本身是实测的，因此表中的数字可能是对的，但文字描述不对）。
- 修复：更正文字，或者在延迟表注中写明每次触发的解码次数。
- 推测：否。

### B11 中等：CE 臂的"BGE 顺序"实际是 Omni 的校准相关度顺序；tie-break 依赖 cid

- 位置：rerank.py:1-6（docstring 写 "BGE order"）；rerank.py:128-132（ce_select 把 `rel`（Omni 的 ECDF 校准相关度）当作 bge_scores 传入 rank_sources）；run.py:262-266（relevance_norm 默认 ecdf，见 run.py:139）；rerank.py:54-56、:82-83（tie-break 为 `(str(fact_id), modality)`，而 ce_select 中 fact_id=cid）。
- 影响：CE 臂的呈现顺序与论文定义（BGE 余弦顺序）不同，而呈现顺序对位置敏感的答案有影响。论文 sec_experiments.tex 中 CE 行的数字因此不能与 "BGE order" 的定义一一对应。tie 时按 cid 排序在 open 加载器中 cid 不含标签信息，影响有限；在 B17 所述的闭集加载器中则会泄漏（见 B17）。
- 修复：CE 臂的呈现顺序使用原始 BGE 余弦（不用 ECDF），或者在论文中改称 "retriever order (calibrated)"；tie-break 改为与 gold 无关的中性 key。
- 推测：否。

### B12 中等：bge-reranker-v2-m3 无版本校验；缓存只按 qid 复用

- 位置：rerank.py:97-125；:102-103（`todo = [e for e in examples if e["qid"] not in scores]`，只按 qid 判断缓存是否命中）；:107-108（从 model_path 加载，无 revision 或哈希）；:65-69（MODEL_ID、MODEL_REVISION、MODEL_FILES 只在 scripts/webqa_open_retrieval/encode_global_text.py:50-58 用于检索器，CE 完全不用）；third_party/README.md:15（写 cache/bge-reranker-v2-m3，无哈希清单）。
- 影响：候选文本、标题格式或 reranker 权重变化后，旧缓存会被静默复用，与题目中"复用旧缓存"的担忧一致。无法从仓库证明论文 CE 数字来自声明的模型版本。
- 修复：缓存键加入模型 revision、文件哈希和候选文本哈希；仿照 encode_global_text.py 的做法记录 model_files_sha256。
- 推测：否。

### B13 中等：M2-RAG 的 commit 只在测试中校验；运行时优先使用 cache 副本

- 位置：src/omniaware/baselines/m2rag_official.py:17（OFFICIAL_COMMIT 常量）、:19-20（cache/official_m2rag 存在时优先，否则回退到 external/M2RAG）；tests/test_m2rag_official.py:61（唯一使用 OFFICIAL_COMMIT 的地方，且只在测试中）。
- 证据：当前 external/M2RAG 的 HEAD 为 ea6b9977f5a153215b38cd9a6aa3e4702ffbda22，与 external_repos.json:15 一致（已核对）。但运行时不做任何比较，cache 副本没有任何哈希检查。
- 影响：若 cache/official_m2rag 存在且不是该 commit，M2-RAG 会悄悄使用其他代码。third_party/README.md:13 把两种来源并列写出。
- 修复：在 _load_official 中比对 `git rev-parse HEAD`（或文件哈希），不符则报错。
- 推测：否。

### B14 中等：本环境中 M2-RAG 的 Qwen 后端从未执行

- 位置：tests/test_m2rag_qwen.py:5（`pytest.importorskip('torchvision')`）。pytest 输出：`SKIPPED [1] tests\test_m2rag_qwen.py:5: could not import 'torchvision'`。另有 tests/test_m2rag_official.py:71 的 legacy T15 测试被跳过。
- 影响：LocalQwenVLM（生成所有 M2-RAG 数字的路径）在本环境未执行，"测试通过"不能覆盖它。
- 修复：安装 CPU 版 torchvision 后重跑；为 m2rag_answer 增加一个 mock 端到端测试（包括 B1 的问题图片断言）。
- 推测：否。

### B15 中等：FITM 的测试是自洽测试，且锁定了饱和行为

- 位置：tests/test_fitm_qwen_adapter.py:45-74（参照实现调用的也是 redistribute_document_attention，与被测函数同一份代码）；tests/test_fitm_attention.py:29（`assertEqual(TEMPERATURE, .00005)`，只是 pin 常数）、:32-33（期望 one-hot）。
- 影响：这些测试能验证数值实现与自身的一致性，但无法发现公式或温度与原文不一致（B3）。
- 修复：用独立推导的小例子或原文数值做 golden 测试；把 pin 常数的断言改为对原文值的引用。
- 推测：否。

### B16 中等：UniversalRAG 训练集过滤只按 index；路由器为手工步骤

- 位置：universalrag.py:72-74（overlapping 只比较 index）、:94-98（filter-train 只用 index 删除）；universalrag.py:9-12（路由器训练与推理为手工命令）；scripts/paper_runs.py:129-131（`echo route ... with the official UniversalRAG router` 只是打印提示）。
- 影响：若上游训练集中有与评测题文本相同但 index 不同的样本，泄漏仍然存在（推测）。路由器的 checkpoint、训练参数、推理参数未固定，无法复现。论文 sec_experiments.tex:12 的 "after removing our evaluation questions" 只被部分满足。
- 修复：同时按规范化问题文本去重；记录路由器 commit、checkpoint 哈希与运行命令；把路由生成脚本化并写入 manifest。
- 推测：泄漏程度是推测。

### B17 中等（潜在，论文主表未用）：旧 WebQA 闭集加载器把 gold 编码进 cid

- 位置：src/omniaware/data.py:47-50（文本 gold 候选先于负例排列，cid 为 `t{'p' if gold else 'n'}{i}`，即 `tp0`/`tn0`）；:54-61（图像同理，`ip`/`in`）；rerank.py:54-56、:82-83（tie-break 用 fact_id=cid）；run.py:52-69（`sorted(key=-rel)` 是稳定排序，tie 时保留列表顺序，gold 在前）。
- 对照：open 加载器 data.py:128-160 的 cid 为 `t{rank}`/`i{rank}`，不含 gold 信息，论文主表使用的是 webqa_open_*（paper_runs.py:23-26、:48）。
- 影响：在 webqa_full 或 webqa_img 上运行时，relevance 或 CE logit 并列会优先或后置 gold 候选，形成泄漏通道。论文主表不受影响，但 run.py 仍允许这些数据集。
- 修复：从 cid 中删除 gold 信息；候选列表在进入排序前做中性打乱；所有排序使用与 gold 无关的 key。
- 推测：是（对论文数字的影响推测为零，因为主表不用该加载器，需要确认 paper_runs 不调用它）。

### B18 轻微：InternVL 下 SeaKR 路径未验证

- 位置：run.py:199-200（FITM 和 M2-RAG 对 internvl3.5-8b 有守卫）；arms.py:58-59（FITM 只支持 qwen3_vl）；seakr.py:50-62（通过 qwen._prefix、_crop，需要 qwen_eab.py:246-260 的 internvl 分支）。
- 评估：论文的 backbone transfer 表只包含 static 与 Omni（paper_runs.py:97-111），因此 FITM、M2-RAG 不跑 InternVL 与论文一致，没有问题。SeaKR 在 InternVL 下代码路径存在，但没有 InternVL 权重，无法验证；论文也未报告 InternVL 下的 SeaKR，所以不影响论文结论。
- 推测：是。

### B19 轻微：空候选时 FITM 抛异常

- 位置：fitm_attention.py:19-23（validate_documents 对空列表抛 ValueError）；arms.py:56-67（chunk_ids 为空时 fitm_sources 返回 []，随后 prepare_fitm_calibration 抛异常）。
- 影响：run.py 对该调用没有 try/except，整个 run 会中断。这是"响亮失败"而不是静默错误，但长跑中会丢失进度。是否会出现空候选需要检查数据。
- 修复：在 run 层面捕获并记录，或在 fitm_answer 中对空 chunk 直接返回 direct。
- 推测：频率是推测。

### B20 轻微：杂项

- m2rag_official.py:73-80：对 parse_image_selection_result 的"修复"只在 multi-stage 路径生效（external/M2RAG/src/summarizer/multi_modal/end_to_end.py:558），单阶段配置下为惰性代码。docstring 中 "repair" 的说法会让读者误以为它影响结果，应删除或注明。
- seakr.py:54：每次调用固定 seed=42，所有题目的采样随机流相同（见 B9）。
- rerank.py:84：`by_calibration[:5]` 硬编码，与 TOP_K 不一致（当前 TOP_K 也是 5，无实际影响）。
- fetch_external.py:43-47：raw_files 分支不可达（external_repos.json 中没有任何 raw_files 键）；fetch_raw（:24-38）因此也不会执行。
- rerank.py:65-69：MODEL_ID、MODEL_REVISION、MODEL_FILES 对 CE 无用（见 B12）。

## 3. 已核实没有问题的部分

- eigen score 与上游 `_compute_eigen_score` 的 AST 一致（tests/test_seakr.py:14-22 通过，未跳过）。
- SeaKR 的层索引 `hidden_states[layer+1]` 与上游 0 基的 `i == selected_intermediate_layer` 输出一致；阈值方向一致；选文档规则（min eigen，空答案最后）与上游 notebook 一致。
- 外部仓库 commit 全部钉死，且与本地 HEAD 一致：M2RAG ea6b9977、SeaKR d75fe707、UniversalRAG d29ab3ac（已核对 external/*/.git）。fetch_external.py:50-52 对已存在的不匹配 checkout 会报错退出，这是正确的。
- 校准集与评测集的不相交性有 assert（run.py:254），划分逻辑 data.py split_examples 正确。
- Omni 的探测轮不增加候选（与 Static 相同的 top-k 集合），额外成本只是一次探测 prefill，已在延迟列说明。
- gold 信息未用于路由：oracle 路由仅出现在路由消融中（paper_runs.py:162），并已标注。CE 与 SeaKR 不使用 gold。
- FITM 的 null reference 只替换 chunk 的 token 与像素，不使用 gold。
- FITM 与 M2-RAG 的守卫（run.py:197-200）与论文一致：M2-RAG 不跑视频，二者都不跑 InternVL。
- 对 CE 的 k 限制（run.py:191-192）保证 CE 与其他臂使用相同的 k=5。
- test_hf_greedy_matches_runner_static（tests/test_baseline_arms.py:8-13）确认 HF generate 与自定义 greedy 的结果一致。

## 4. 没来得及深入的方向

- B2、B6 的幅度：需要真实 runs/ 输出（当前仓库中没有 runs/ 目录），或者用 Qwen3-VL 重跑 Static 的 joint top-k 与 UniversalRAG 的路由统计。
- B3：需要 Hsieh et al. 2024 原文的式 (4)(5) 与温度取值，以及其 Ā 的量纲，本环境无法访问。
- B5：需要官方 M2-RAG 的 evaluate 脚本，确认官方 WebQA 评测是否使用摘要本身做 QA，而不是额外抽取。
- B16：需要上游 UniversalRAG 训练数据，检查与 WebQA、HybridQA、VideoRAG 评测题的文本重复。
- B19：需要检查 WebQA 开放检索与 OK-VQA 的候选是否存在完全为空的题目。
- B18：需要 InternVL3.5-8B 权重才能验证 SeaKR 的 InternVL 路径。
- 没有下载任何模型或数据集，也没有运行真实 Qwen 推理；所有数值结论均来自 CPU 小脚本或代码阅读。

## 5. 复现文件

- scratchpad\review\fitm_saturation.py：B3 的数值复现。
- scratchpad\review\checks.py：B1、B2、B6 的机制复现（UniversalRAG 路由回退、select_chunks 过滤、M2-RAG summarizer 消息检查）。
- pytest 命令：`python -m pytest -q tests/test_baseline_arms.py tests/test_fitm_attention.py tests/test_fitm_qwen_adapter.py tests/test_m2rag_official.py tests/test_m2rag_qwen.py tests/test_m2rag_webqa.py tests/test_seakr.py tests/test_universalrag.py tests/test_universalrag_suite.py -p no:cacheprovider`，结果：29 个用例中 28 通过，1 跳过（test_m2rag_official.py:71）；test_m2rag_qwen.py 整文件因缺少 torchvision 被跳过。

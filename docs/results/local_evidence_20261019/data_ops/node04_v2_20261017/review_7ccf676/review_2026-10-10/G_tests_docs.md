# G：测试套件质量 + 文档/迁移声明核实 + 真实模型差异

仓库：`C:\Users\22688\Desktop\www_code\omniaware`（分支 main，HEAD 51df499，工作树干净）。
严重程度：致命 / 严重 / 中等 / 轻微。标注"推测"的条目说明了需要什么证据才能确认。

---

## 0. 范围、环境与改动声明

- 仓库文件未做任何修改。唯一的副作用：我第一次运行 `python -m pytest -q` 时重写了 `.pytest_cache/v/cache/nodeids`（git 忽略），并在 src/tests/scripts 下生成了 `__pycache__`（git 忽略）。`.pytest_cache/v/cache/lastfailed` 的 mtime 仍是 11:34，未被我的运行修改。
- 一次性验证脚本只写在 `scratchpad\review\`：`check_fitm_span.py`、`check_fitm_rope.py`、`empty_runs\`（dev_select 空目录实验）。
- 环境核实：Python 3.11；torch 2.7.1+cpu；transformers 5.5.3（已用 `python -c` 确认）；nltk 3.8.1；sentencepiece/pandas/pyarrow/tokenizers 存在。**未安装**：spacy、en_core_web_sm、word2number、torchvision、bert_score、faiss。`assets/`、`cache/` 目录不存在。
- 未下载任何模型或数据，未 push，未改历史。

---

## 1. pytest 结果摘要

| 项目 | 结果 |
|---|---|
| 命令 | `python -m pytest -q`（首次）；复核：`python -m pytest -rsf -p no:cacheprovider` |
| 汇总 | **107 passed, 5 skipped, 8 subtests passed in 3.24s**，退出码 0 |
| pyflakes | `python -m pyflakes src tests` 无输出，退出码 0 |

**5 个 skip 项（按模块，本环境）**

| 位置 | 原因 | 跳过的测试函数数 |
|---|---|---|
| `tests/test_citation_intake.py:12-13` | 模块级 skip：NLTK Punkt 资产不在 `paths.CACHE` 下 | 11 |
| `tests/test_webqa_acc.py:8` | 模块级 `importorskip('spacy')`（spacy 未安装） | 5 |
| `tests/test_m2rag_qwen.py:5` | 模块级 `importorskip('torchvision')`（未安装） | 7 |
| `tests/test_webqa_bart.py:102` | `skipUnless`，BART 资产缺失（类 `OfficialScoreTests`） | 1 |
| `tests/test_m2rag_official.py:71` | `skipUnless`，legacy T15 运行输出缺失 | 1 |

**结论：107 passed 的绿灯不代表这 25 个测试通过。** 其中包括 WebQA 官方指标（`test_webqa_acc`）和引用分句/分母/NLI 拒绝逻辑（`test_citation_intake`），它们正是论文数字的评分保护测试。

**`lastfailed` 缓存**（`.pytest_cache/v/cache/lastfailed`，mtime 11:34，早于本次运行）记录 17 个此前失败的测试：`test_webqa_acc.py` 5 个、`test_citation_intake.py` 11 个（`IntakeTest` 及其 10 个方法），以及 `tests/test_score_official.py::test_videomme_parses_letters_and_clusters_by_video`。现在前 16 个所在模块被跳过（本环境缺依赖/资产），第 17 个测试已不存在（Video-MME 已删除）。缓存只记录 ID，**无法从缓存确定它们当时失败的根因**（推测：当时环境有 spacy/NLTK 资产，或测试在旧代码上失败；需要在完整资产环境重跑才能确认）。

---

## 2. 测试质量

### 2.1 通过但没测到关键性质的测试

| 位置 | 问题 | 证据 |
|---|---|---|
| `tests/test_run_smoke.py:31-63`（`FakeQwen`） | 替换了整个 `QwenEAB`，输出文本只由 `enc.length` 决定（L58-61）。因此所有"运行级"测试都不验证 EAB 偏置是否作用于正确 span、gate 是否改变了输出、干预/复制是否真的改变了证据。 | `text=f"{name}@{enc.length}"` |
| `tests/test_run_smoke.py:95` | ARD 断言只是 `0 <= v <= 0.7`（JSD 上界 ln2 的粗略界）。 | |
| `tests/test_run_smoke.py:126` | 复制臂"不同于 static"只靠长度不同来成立。 | `!=` 基于长度字符串 |
| `tests/test_run_smoke.py:275` | SeaKR 断言 `preds["seakr"] != preds["direct"]` 同样只靠长度。 | |
| `tests/test_run_smoke.py:385` | `preds["gated:eab..."] in (expected, preds["eab..."])`：析取断言，几乎任何输出都通过。 | |
| `tests/test_qwen_eab.py:87-96` | `test_arms_report_time_and_first_answer_row_attention` 只检查 `attention.shape` 与 `sum==1`，未检查它是哪一行（见 F14）。 | |
| `tests/test_qwen_eab.py:18-38` | 钩子数学测试只覆盖 `attention_mask=None`（L28 传 `None`），未覆盖生产代码的布尔/加性 4D mask 分支（`qwen_eab.py:70-74`）。 | |
| `tests/test_internvl_eab.py:13-24` | "InternVL"测试全程文本输入（`eab_fixtures.encode` 只生成文本段，L52-55），从未走 `pixel_values`、`image_seq_length`、`start/end_image_token_id` 路径。 | `encode(q)` |
| `tests/test_fitm_qwen_adapter.py:147-151` | 视觉 null 测试手工构造 `sources`（`text_token_indices=[4]`，`vision_token_indices=[2]`），没有用真实的"图像 span = vision_start + image tokens + vision_end"布局，因此没有覆盖 F15。 | |
| `tests/test_dev_select.py:108-131` | 只覆盖"base 胜出、framing 1"的正路径。没有覆盖：数据集缺失、framing 2/3、zsum/mid 被选中、无 dev 结果时的行为（F2）。 | |
| `tests/test_paper_runs.py:6-20` | 只检查命令能被 `run.build_parser()` 解析、变体可解析；没有检查 `decisions()` 缺失时是否报错（F1）。 | |
| `tests/test_score_official.py:1` | 文件头写"official OK-VQA and Video-MME evaluators"，Video-MME 已删除（轻微）。 | |
| `tests/test_webqa_acc.py`（整文件） | 本环境整文件跳过（见 §1）。WebQA 指标语义、"exactly one reference"、配对 bootstrap 均未在本环境运行。 | |

### 2.2 关键但缺失的测试（建议新增）

1. **WebQA 指标在完整环境下运行**（spacy + en_core_web_sm + word2number），并把 `score_webqa` 路径（而非仅 `score_references`）接入"多参考应拒绝"的断言（F8）。
2. **图像编码路径**：`QwenEAB.encode` 中 `Segment(image=...)` 分支（`qwen_eab.py:184-200`），包括 `max_pixels` 分支、`image_grid_thw` 与 token 计数 `grid.prod()//merge²`（L211）；需要一个带 `processor` 的小 Qwen3-VL（`tests/tiny_qwen.py` 已能构造 processor 所需的小配置，测试 `test_m2rag_qwen.py` 已经在这么做，但它被 torchvision 跳过）。
3. **InternVL 像素路径**：`_internvl_images`（L175-179）、`vision_start_id`/`vision_end_id`（L162-163）、`image_seq_length`（L164），以及 `pixel_values` 与占位 token 数量一致的断言。
4. **`QwenEAB.__init__` 本身**：`kind` 检测（L152）、`dtype`/`attn_implementation` 传参（L156）、`eos` 集合（L171-172）、`max_pixels` 设置（L170）。目前 `eab_fixtures.make_qwen` 用 `QwenEAB.__new__`（`eab_fixtures.py:38`）绕过了它。
5. **生产 `tag_text_layers` 在 Qwen 上**（`qwen_eab.py:130-136`）。fixture 按类名 `Qwen3VLTextAttention` 手动标层（`eab_fixtures.py:41-44`），与生产代码走的路径不同。
6. **4D 布尔/加性 mask 的钩子分支**（`qwen_eab.py:70-74`）。
7. **FITM 的真实 span 布局**（见 F15）。
8. **dev_select**：缺失数据集、framing 2/3、zsum/mid 胜出、无 run 时应失败（见 F2）。
9. **paper_runs.decisions()**：`dev_decisions.json` 缺失或 model 不匹配时应失败而不是回退（见 F1）。
10. **UniversalRAG 路由**：仅含语料中不存在模态的标签时（F3），以及未知标签（F13）。
11. **缓存失效**：候选内容变化后，`relevance_*`、`ucal_*`、`seakr_*`、`ce_*` 是否重算（F10）。
12. **`scripts/launch.py:launch()`**：Popen 分片、失败分片、续跑（目前只测了 `detect_gpus`，`test_run_smoke.py:414-415`）。
13. **运行级 EAB 语义**：用真实 `QwenEAB`（tiny）跑 `run.main`，断言 `eab_active` 与 gate 一致，且 EAB 输出确实与 static 不同（gate 打开时）。
14. **chat template 等价性**：手写 ChatML（`pipeline.py:41,71`）与 `processor.apply_chat_template` 的 token 序列逐 token 比较（目前只有被跳过的 `test_m2rag_qwen.py` 在做）。
15. **分段分词边界**：`QwenEAB.encode` 逐段 `tok.encode`（`qwen_eab.py:216`）与整串分词的一致性（用真实 tokenizer）。
16. **ARD 注意力行语义**（见 F14）。
17. **VideoRAG 聚类键**：用真实 upstream query index 格式（见 F12）。

---

## 3. 文档声明核实表

约定：属实 = 代码与测试都支持；部分属实 = 核心成立但有例外或缺口；不属实 = 与代码矛盾；无法核实 = 需要数据/资产/模型。

| # | 文档 / 位置 | 声明 | 结论 | 证据 |
|---|---|---|---|---|
| 1 | CONVENTIONS §2 | `python -m pyflakes src tests` 无警告 | **属实** | 运行，无输出，rc=0 |
| 2 | CONVENTIONS §6, PROTOCOL §2 | 校准过程不读取答案，不按准确率选阈值；`--select-tau` 已删除 | **属实（代码审阅）** | `run.py:420-439` 只用 qid、prompt、logits、attention；`run.py:470-473` 的 `u_rate`、`nte_threshold` 只用校准集的无标签量；grep 无 `select_tau` |
| 3 | PROTOCOL §1, CONVENTIONS §6 | calib / dev / test 互不相交，且校准不参与评测 | **部分属实** | `run.py:254` 运行时断言只检查 calib∩eval；dev/test 不相交由 `data.split_examples`（`data.py:357-372`）按构造保证，`tests/test_splits.py` 通过，但**运行时没有 dev/test 断言** |
| 4 | DEV_DECISIONS §1, §4（规则 4） | 固定 (τ,τ_b,β,γ,δ)=(0.35,0.40,0.30,1.2,0.08)，k=5 | **属实** | `run.py:147-149`、`run.py:155`、`run.py:103`、`routing.py:15` 的 `margin=0.08` |
| 5 | DEV_DECISIONS 规则 1 | EAB 形式按数据集内平均再跨数据集平均，选最大；CI 下界>0 为有效 | **部分属实** | `dev_select.py:46-54,57-71,114` 与规则一致；但 EAB 只在 **base U 设置**下比较（`dev_select.py:62`），且池化数据集列表包含 `okvqa_gs`（`dev_select.py:31`），在规则 3 选语料之前就已固定。见 F7 |
| 6 | DEV_DECISIONS 规则 2 | U 与 NTE 的 Spearman ρ 比较，跨数据集平均 | **属实** | `dev_select.py:74-94,115-116` |
| 7 | DEV_DECISIONS 规则 3 | 选 static−direct 最大的 OK-VQA 语料；gap≥0 为"检索有用" | **属实** | `dev_select.py:97-107,124`；但缺失语料被静默忽略（F7） |
| 8 | DEV_DECISIONS 规则 5 / 框架表 | framing 1/2/3 的定义 | **属实** | `dev_select.py:117` |
| 9 | DEV_DECISIONS 头部 | "2026-10-11，在看到任何 dev 结果之前写定" | **无法核实，且日期矛盾** | 文件与 `scripts/analysis/dev_select.py` 都只在提交 `cd48ab8`（2026-10-10 10:59:26 +0800）出现，同一提交还把默认 EAB 从 ans2 改为 ans1，并新增 probe 轮、zsum、mid、GS-112K。文档写的日期（10-11）晚于提交时间和今天（10-10）。见 F6 |
| 10 | PROTOCOL §3 | 默认 EAB 变体为 `eab:g1.2:ans2:pos:gate`，"即论文 decode-only" | **不属实（文档陈旧）** | 代码默认是 ans1：`run.py:155`；`pipeline.py:136` 写明 ans1 为"paper default since V3.21"；DEV_DECISIONS 也写 ans1。见 F23 |
| 11 | PROTOCOL §3 | ans2 在数学上无法改变第一个答案 token（见 tests/test_qwen_eab.py） | **属实** | `test_qwen_eab.py:60-68` 通过 |
| 12 | PROTOCOL §3 | 所有对照臂用**相同候选池、k、解码上限和提示词格式** | **部分属实** | direct/static/EAB/gated/FITM/SeaKR 单文档共用同一 encode（`run.py:539,585`、`arms.py:60-67`）。M2-RAG 使用其官方 summarizer 提示词，文档 cap 固定为 384（`arms.py:72,86-91`），不是同一提示词格式。见 F24 |
| 13 | PROTOCOL §4 | 配对 bootstrap 5,000 次 | **部分属实** | `score_official.py:51-57` 为 5000；但 `webqa_acc.py:90` 默认 10000。见 F28 |
| 14 | PROTOCOL §4 | VideoRAG-Synth 按源视频聚类重采样 | **部分属实** | `universalrag_metrics.py:88` 用 `q.rsplit("_",1)[0]` 从 qid 推导簇键，不是源视频字段。见 F12 |
| 15 | PROTOCOL §5 | 14 个排列（sorted、reverse + 12 个固定随机排列） | **属实** | `order.py:12-25`；`tests/test_order.py` 通过 |
| 16 | PROTOCOL §8 | 反事实实体由题号哈希确定 | **属实** | `interventions.py:96`（sha256 of `cf:{qid}`） |
| 17 | PROTOCOL §8 | 干预只在触发检索、EAB 激活、gold 出现在排名最高的文本块上 | **属实（代码审阅）** | `run.py:327-328`，`interventions.find_target` |
| 18 | PROTOCOL §1, README §4.1 | WebQA 开放检索为精确内积 top-5，等同 FAISS IndexFlatIP | **属实** | `scripts/webqa_open_retrieval/search_dense_queries.py:109`（检查 IndexFlatIP）；`scripts/setup/prepare_webqa_open.py:18,188`（精确 topk） |
| 19 | CONVENTIONS §5, PROTOCOL §4 | OK-VQA 使用官方 VQAEval，不做子串匹配 | **属实** | `score_official.py:64-77`；`tests/test_score_official.py:17-36` 通过 |
| 20 | CONVENTIONS §5 | WebQA 使用官方 acc_approx（AST 从固定源码提取） | **部分属实** | 提取真实（`webqa_acc.py:31-80`）；但本环境未运行（spacy 缺失）。另见 F8、F9 |
| 21 | CONVENTIONS §5 | 禁止宽松匹配、同义词表、按题修补评分规则 | **部分属实** | OK-VQA、WebQA 指标无子串匹配。但 `eval/citation/attribution.py:55-62` 对 Faith/AP 使用别名匹配（词界正则 + 标签），而 `citation_support_protocol.md:1` 说这是论文的做法。CONVENTIONS 没有豁免。见 F25 |
| 22 | CONVENTIONS §5 | 入口只有一个：`python -m omniaware.eval.score_official` | **部分属实** | 另有入口：`webqa_acc.py:162`（main）、`attribution.py`、`webqa_bart.py`、`universalrag_metrics.py`。见 F25 |
| 23 | CONVENTIONS §6 | 同一次运行、同一输入、同一提示词生成所有对照臂；不与旧运行缓存比较 | **部分属实** | 预测不复用（`run.py:538-539` 每题重算，属实）。但校准/相关度/CE/SeaKR 校准缓存按 qid 复用（`run.py:25-26,420-421,482-483`；`rerank.py:102-104`）。见 F10 |
| 24 | LEGACY_MAP §未迁移；PROTOCOL 删除列表 | 旧 `cached_baselines` 与复用 baseline 的流程未迁移 | **属实** | 仓库中不存在相关代码；`run.py` 每题重算所有臂 |
| 25 | LEGACY_MAP §第一轮 | `qwen_eab.py` "钩子已在 transformers 5.5.3 下用测试验证" | **部分属实** | 验证只在随机小模型、文本输入上完成（`eab_fixtures.py`），图像与 InternVL 像素路径未验证（F5） |
| 26 | LEGACY_MAP §第一轮 vs §2026-10-11 | 第一轮迁移了 `answer_format.py`、`table_render.py`、`mmqa_official.py`；2026-10-11 删除 | **属实** | 三个文件不存在（已核实） |
| 27 | LEGACY_MAP §第一轮 | `data.py`：`MMQA 默认 369 题` | **陈旧** | MMQA 已删除（`git log` ea47284）；`data.py` 中已无 MMQA 加载器 |
| 28 | MIGRATION_SOURCES.json | 第一轮 34 个文件、第二轮 59 个文件，每个都记录了来源与 SHA256 | **部分属实** | 计数属实（34、59）。但 93 个目标中 **36 个不存在**：10 个第一轮（已在 LEGACY_MAP 删除清单中）、26 个第二轮（DRAGIN 4 个、`legacy_staging/*` 9 个、`webqa_open_retrieval` 9 个、`docs/*DENSE*` 3 个，以及被改名为 `answer_recall.py` 的 `okvqa_passage_answer_recall.py` 1 个）。其中 `legacy_staging/*` 与 `scripts/analysis/okvqa_passage_answer_recall.py` 的改名在任何文档中都未说明。源文件 SHA256 无法核实（旧仓库不在本环境）。见 F26 |
| 29 | LEGACY_MAP §未迁移（UniversalRAG 行） | "UniversalRAG 路由 … 论文的结果表里没有这个方法"，不迁移 | **不属实（与其他文档矛盾）** | 代码已迁移（`baselines/universalrag.py`）；`PAPER_COVERAGE.md:52` 把它列在主表；`PROTOCOL.md:42` 与 README §4.4 与之相符。见 F24 |
| 30 | PAPER_COVERAGE §2–§4 | ✅ "已实现并有测试" | **部分属实** | 许多 ✅ 依赖的测试在本环境被跳过（WebQA 指标、citation 分句/NLI 拒绝、M2-RAG Qwen 适配器）。见 F4 |
| 31 | PAPER_COVERAGE §2 | InternVL3.5-8B 骨干 ✅（"EAB 钩子在小模型上验证"） | **部分属实** | 只在文本小模型上验证（F5） |
| 32 | citation_support_protocol §"CiteRec 的明确计算规则" | NLTK 分句 `PunktTokenizer.span_tokenize` 与 `nltk.sent_tokenize` 在 fixture 上逐项一致 | **无法核实（本环境）** | 对应测试 `test_citation_intake.py:37-43` 被跳过。本环境 nltk 为 3.8.1，文档写的是 3.9.2 |
| 33 | citation_support_protocol §"CiteRec" | 缺失答案、缺失 NLI、全非法引用、wrong checker、argmax 不一致都被拒绝 | **无法核实（本环境）** | `test_citation_intake.py:74-148` 被跳过；代码 `score_nli.py` 未在本环境运行 |
| 34 | README §8 | 需要额外资产的测试（BARTScore、spaCy、NLI 检查器）缺资产时自动跳过 | **部分属实** | 跳过机制存在，但 `test_citation_intake.py:12-13` 跳过的是 NLTK Punkt 资产，不是 NLI 检查器。见 F30 |
| 35 | README §1 | `python -m pytest` 在 CPU 上约 10 秒 | **属实** | 实测 3.24 秒 |
| 36 | README §1 | `transformers>=4.57`（Qwen3-VL、InternVL3.5-HF 需要） | **无法核实（4.57 未测）** | `pyproject.toml:9`；全部验证只在 5.5.3。见 F20 |
| 37 | README §2 | `OMNIAWARE_RUNS` 是可选变量，默认 ./runs | **属实，但与 dev 决策流程不一致** | `paths.py:55` 读取环境变量；`dev_select.py:132` 默认 `Path("runs")`（相对当前目录）；`paper_runs.py:35` 读取 `ROOT/runs`。见 F2 |
| 38 | README §5.2 | dev 网格为"6 种 EAB 形式 × 3 种 U 设置 × 3 种 OK-VQA 语料" | **部分属实** | `paper_runs.py:84-90` 确实生成这些运行，但 `dev_select` 的 EAB 选择只用 base（`dev_select.py:62`）；OK-VQA 缺失语料时静默减少（`paper_runs.py:79-82`）。见 F7 |
| 39 | README §5.1 | `--n-eval` 中 SeaKR 与顺序实验各取 1,000 题 | **属实** | `paper_runs.py:29` SUBSAMPLE="1000" |
| 40 | README §8 / CONVENTIONS §3 | 多卡分片与合并有端到端测试 | **部分属实** | `tests/test_run_smoke.py:388-415` 测试了分片与 merge；`scripts/launch.py:launch()` 的 Popen、失败与续跑路径未测。见 F27 |
| 41 | CONVENTIONS §4 / merge_shards 文档 | 合并时核对每道题恰好回答一次 | **部分属实** | 检查重复 qid（`merge_shards.py:44-46`）与总数（L48-49），但只核对题数，不核对题目身份集合（轻微） |
| 42 | `third_party/webqa/provenance.json` / `scripts/…` 固定 commit 与 SHA256 | 固定的源文件与 pin 一致 | **内容属实，但代码不校验** | `sha256sum` 与 `webqa_acc.py:25-28` 的 pin 一致（已核实）；`webqa_acc.py:49-53` 只计算并记录，从不比较。见 F9 |
| 43 | `external/ALCE` 固定 commit `246c476a…` | 与 citation 协议一致 | **属实** | `git -C external/ALCE rev-parse HEAD` = 246c476a4edfc564266b7346b6e29ef4861ae937 |
| 44 | SeaKR `eigen_score` "Verbatim from upstream" | 与上游 `_compute_eigen_score` 逐字相同（alpha 替换） | **属实** | `tests/test_seakr.py:15-22` 通过；上游 `external/SeaKR/vllm_uncertainty/vllm/engine/llm_engine.py:738-744`；上游默认 alpha 1e-3（`main_multihop.py:146`） |
| 45 | SeaKR 层号 "upstream layer 15 of 32" | 缩放到目标深度 | **属实** | `seakr.py:22,35-37`；`test_seakr.py:37` |
| 46 | README §2 / paths | 资产目录与环境变量 | **属实（代码）** | `paths.py:15-17,47-56` |
| 47 | `external_repos.json`（InternVL 行） | "InternVL2-40B backbone transfer" | **陈旧** | 论文与代码已改为 InternVL3.5-8B（`PAPER_COVERAGE.md:70`，`paths.py:24`）。见 F38 |
| 48 | `citation_support_protocol.md` 第 1 行 | 论文的 Faith/AP/Inv 由 `omniaware.eval.citation.attribution` 计算，别名匹配后接 DeBERTa | **属实（代码）**，但与 CONVENTIONS §5 冲突 | `attribution.py:55-62,65-78`。见 F25 |
| 49 | CONVENTIONS §5 | 评分不在推理时进行，不读取模型输入之外的信息影响生成 | **属实（代码审阅）** | `run.py` 中评分无参与；`relevance`/`run` 只用候选内容；`interventions.pick_substitute` 使用其他评测题的 gold 作为替代池（`run.py:498`），属于评测数据参与构造干预，但不影响生成的选择（见 F33） |

---

## 4. 真实模型差异

### 4.1 已在本地 transformers 5.5.3 源码中核实的 API

路径：`C:\Users\22688\AppData\Roaming\Python\Python311\site-packages\transformers\models\`

- Qwen3-VL：`Qwen3VLForConditionalGeneration`（`qwen3_vl/modeling_qwen3_vl.py:1392`）✓；`Qwen3VLTextAttention`（L436）✓，与 `eab_fixtures.py:42` 的类名匹配；`Qwen3VLModel.rope_deltas`（L964）✓，与 `qwen_eab.py:253,259-260` 使用的属性一致；`logits_to_keep`（L1457）✓。
- Qwen3-VL 的图像数量计数（`_get_image_nums_and_video_nums`，L1616-1665）依赖 vision_start 位置；但 `get_rope_index` 的图像网格消费顺序由 **`mm_token_type_ids` 分组**决定（L1700-1760 与 L1031-1100 附近的 `itertools.groupby`）。我在 tiny 模型上验证：把 vision_start/end 换成文本 token 后，rope 位置不变（`check_fitm_rope.py` 输出 identical: True）。因此 F15 的 rope 层面问题在 5.5.3 中**不成立**，只剩 token 身份层面的问题。
- InternVL：`InternVLForConditionalGeneration`（`internvl/modeling_internvl.py:739`）✓；`InternVLModel.language_model = AutoModel.from_config(text_config)`（L535），所以 `model.model.language_model.layers` 路径（`qwen_eab.py:133`）✓；`logits_to_keep`（L785）✓。
- InternVL 处理器：`start_image_token_id`、`end_image_token_id`、`image_seq_length` 属性存在（`internvl/processing_internvl.py:59-63`）✓；占位构造 `start + image_token*seq*num_patches + end` 与代码的 `encode()` 一致（L105）。
- 图像处理器（GotOcr2 系列）返回 `num_patches`（`got_ocr2/image_processing_got_ocr2.py:268`，`image_processing_pil_got_ocr2.py:271`）✓，与 `qwen_eab.py:178` 的读取一致；支持 `crop_to_patches`、`max_patches`（L34-47）✓。
- Qwen 图像处理器接受 `size={"shortest_edge","longest_edge"}`（`qwen2_vl/image_processing_qwen2_vl.py:95,110-119`）✓；`merge_size` 可从 processor 读取（`qwen3_vl/processing_qwen3_vl.py:128`）✓。
- `DynamicCache.crop` 支持负数（`cache_utils.py:139-145`）✓，`_crop` 依赖它（`qwen_eab.py:87-91`）✓。

**注意**：具体检查点的 `preprocessor_config.json`、`chat_template`、`generation_config.json` 不在本地，以下相关结论都需要检查点才能确认。

### 4.2 最可能在真实模型上出错的位置（按可能性排序）

1. **图像输入路径未被任何测试执行**（`qwen_eab.py:184-200`、`_internvl_images` L175-179）。`QwenEAB.encode` 的图像分支只在带 processor 的真实环境中运行，所有测试都用 `processor=None`、文本段。第一次真实跑的几乎肯定是这里（严重，F5）。
2. **对话模板是手写的 ChatML**（`pipeline.py:41,71` 写死 `<|im_start|>user\n`、`<|im_end|>\n<|im_start|>assistant\n`），不是 `processor.apply_chat_template`。Qwen3-VL-Instruct 的模板通常不加系统提示；InternVL3.5 的官方模板可能加系统提示（推测，需要检查点的 `chat_template`）。没有任何活跃测试比较两者的 token 序列（`test_m2rag_qwen.py` 做了比较但被跳过）。
3. **分段分词边界**（`qwen_eab.py:216` 逐段 `tok.encode(s.text, add_special_tokens=False)`）：整串分词与逐段分词在边界处可能不同（推测，需要真实 tokenizer 对比）。EAB 的 span 是按逐段得到的，所以内部一致，但与模型见过的整串分布可能有偏差。
4. **InternVL 默认只用 1 个 448 tile**（`run.py:93` `--max-patches` 默认 1；`qwen_eab.py:147-148`）。官方推理使用动态分块（多 tile 加缩略图），骨干迁移的结果与官方输入不可直接比较（推测，需要确认论文是否说明）。
5. **`transformers>=4.57` 的兼容性**：所有验证只在 5.5.3。4.57 的 `get_rope_index`、`AttentionInterface`、`repeat_kv` 的位置与签名未核实（推测，需要在 4.57 环境跑 `tests/test_qwen_eab.py`）。
6. **EAB 钩子 mask 分支**：生产路径会在 transformers 传入布尔或加性 4D mask 时进入 `qwen_eab.py:70-74`，而测试只覆盖 `None`。batch=1 无 padding 时 transformers 5.x 很可能传 `None`，所以这里的风险取决于具体版本（推测）。
7. **Qwen3-VL DeepStack**（视觉特征注入前几层）：本次没有核实 5.5.3 中的注入位置是否与 `_prefix`/`_step` 的 KV 缓存分段兼容（推测；decode 步不再传 pixel_values，理论上正确，需要真实模型验证）。
8. **`rope_deltas` 为共享可变状态**（`qwen_eab.py:253-261`，`arms.py:36-37,63`）：每次 `_prefix` 都重置，FITM 与 greedy 也重置。逻辑上正确，但需要在多题批量运行中验证不串题（推测）。
9. **bf16 下的 mask 下界**（`qwen_eab.py:64`，`neg = finfo(dtype).min`，之后 `+= col`）：数值上不溢出（已用 float32 推理），在 bf16 下应当仍安全（推测，需要真实 bf16 验证）。

---

## 5. 问题清单

### F1 — 冻结决策文件缺失时测试阶段静默回退到默认设置
- **严重程度：严重**
- **位置**：`scripts/paper_runs.py:34-40`（`decisions()`），`:30-31`（`DEFAULTS`）
- **证据**：`decisions()` 在 `runs/dev_decisions.json` 不存在或 `model` 不匹配时直接返回 `DEFAULTS`，没有任何报错或警告。我运行 `python -m scripts.paper_runs --list`（本环境 `runs/dev_decisions.json` 不存在）输出：`frozen choices: {'eab_variant': 'eab:g1.2:ans1:pos:gate', 'u_combine': 'sum', 'attn_layer': 'last', 'okvqa_loader': 'okvqa_gs'}`。
- **影响**：test 阶段（`--stage main --run` 等）可以在没有 dev 冻结的情况下跑，并且输出看起来正常。这直接违反 PROTOCOL §1 "test 在设置冻结后只跑一次" 和 CONVENTIONS "不能静默跳过"。
- **修复建议**：没有 `dev_decisions.json`（或 model 不匹配）时 `--run` 应当报错退出；把决策文件的哈希写进每个 test 运行的 config；给 `paper_runs` 加测试。
- **推测**：否。

### F2 — `dev_select` 在没有 dev 结果时静默输出 framing=3 与默认值；路径与其他组件不一致
- **严重程度：严重**
- **位置**：`scripts/analysis/dev_select.py:110-127`（`decide`），`:132`（`--runs` 默认 `Path("runs")`，相对当前目录），`:137`（输出路径）；`scripts/paper_runs.py:35`（读取 `ROOT/"runs"`）；`src/omniaware/paths.py:55`（`RUNS` 由 `OMNIAWARE_RUNS` 决定）；`src/omniaware/run.py:206`（运行输出写到 `paths.RUNS`）。
- **证据**：`python -m scripts.analysis.dev_select --runs <空目录> --model 8b --output <空目录>/dev_decisions.json` 退出码 0，输出 `"eab_variant": "eab:g1.2:ans1:pos:gate", "framing": 3`，并写入文件（实际运行过）。原因：`select_eab/select_u/select_okvqa` 对不存在的文件静默返回 `None`，`decide` 用默认值填充（`:120,123`），`framing` 最终为 3（`:117`）。
- **影响**：两个问题叠加。（a）`dev_select` 默认只看 `./runs`；如果按 README §2 设置了 `OMNIAWARE_RUNS`，而没有给 `--runs`，它读不到任何 dev 结果，但仍然成功写出决策文件，框架为 3（"不投 WWW 正会"），之后 `paper_runs` 读取默认值（F1）。（b）`paper_runs` 读的是 `ROOT/runs`，与 `run.py` 写入的目录不一定相同。
- **修复建议**：没有可用的 dev 运行时报错退出；统一用 `paths.RUNS` 查找和写入决策文件；把参与决策的运行文件列表与哈希写进 `dev_decisions.json`。
- **推测**：否（已复现）。

### F3 — UniversalRAG 路由在语料缺模态时静默退化为联合检索
- **严重程度：严重（推测频率；若标签频繁出现则为严重）**
- **位置**：`src/omniaware/baselines/universalrag.py:62-69`（`to_route`）；`src/omniaware/run.py:52-56`（`select_chunks` 把空 `modalities` 当作"不过滤"）；`src/omniaware/run.py:613-615`（`universalrag` 臂取 `route["retrieve"]`）。
- **证据**：`to_route(['video'], {'text','image'})` 返回 `{'retrieve': True, 'modalities': []}`（已复现）。`select_chunks`（`run.py:52`）`if route and route["modalities"]:` 为假，于是候选为全部模态（`run.py:56` 之前没有过滤），即 joint top-k。`universalrag` 臂则因为 `retrieve=True` 使用 static（联合检索）的输出。
- **影响**：路由器输出 video/clip 而 WebQA 语料中没有视频时，baseline 的"路由"静默变成联合检索，结果与文档 "modalities absent from the corpus are dropped"（`universalrag.py:63` 注释）不符。这会影响主表中 UniversalRAG 一行的公平性。
- **修复建议**：语料中没有任何被路由的模态时，应该 `retrieve=False`（或计数并报告），不能回退到联合检索；增加测试 `to_route(['video'], {'text','image'})`。
- **推测**：频率未知。需要在真实路由输出中统计"被路由模态全部不可用"的题数。

### F4 — 25 个测试在本环境未执行，"107 passed" 掩盖了评分保护测试
- **严重程度：严重**
- **位置**：见 §1 表格；`.pytest_cache/v/cache/lastfailed`（11:34）列出的 17 条记录。
- **证据**：`python -m pytest -rsf` 输出 5 个 SKIPPED；模块级跳过的文件共 23 个测试函数（`test_citation_intake.py` 11、`test_webqa_acc.py` 5、`test_m2rag_qwen.py` 7），另 2 个单用例跳过。`import spacy`、`import word2number`、`import torchvision` 失败；`paths.CACHE` 下没有 NLTK Punkt 资产。
- **影响**：论文的 WebQA 指标语义（`test_webqa_acc.py`）与引用分句/分母/拒绝规则（`test_citation_intake.py`）没有被本环境验证。绿灯会让人误以为它们通过。PAPER_COVERAGE 的 ✅ 也依赖这些测试。
- **修复建议**：CI 中把这些依赖设为必需（或把"跳过"计入失败）；在 README 写明跳过时的覆盖范围；在装有 spacy/NLTK 的环境重跑并记录结果。
- **推测**：否（已核实）。

### F5 — 真实模型的图像路径与 InternVL 像素路径零测试覆盖
- **严重程度：严重（对真实运行而言）**
- **位置**：`src/omniaware/qwen_eab.py:184-200`（图像分支）、`:175-179`（`_internvl_images`）、`:147-173`（`__init__`）、`:130-136`（`tag_text_layers` 对 Qwen 路径的使用）；`tests/eab_fixtures.py:26-79`（`make_qwen` 绕过 `__init__`，`make_internvl` 也不带图像）；`tests/test_internvl_eab.py:16`（`encode(q)` 为文本段）。
- **证据**：`grep` 所有测试，没有任何测试向 `QwenEAB.encode` 传入 `Segment(image=...)`，也没有调用 `QwenEAB.__init__`。`eab_fixtures.py:39,72` 都设置 `processor=None`。
- **影响**：第一次真实运行几乎肯定在这里暴露问题（processor 调用签名、`size` 字段、`image_grid_thw` 形状、token 计数、InternVL 的 `num_patches` 与占位数量）。这些问题在真实模型下才会出现，不是小模型能发现的。
- **修复建议**：用 `processor`（可以用 `tests/tiny_qwen.py` 的小配置和 processor 构造）覆盖 `encode` 的图像分支、InternVL 的 `_internvl_images`，并断言 `pixel_values` 的行数等于 `image_grid_thw.prod()/merge²` 之和（Qwen）或 `num_patches*256`（InternVL）。
- **推测**：否（覆盖缺口已核实）。真实失败形式为推测。

### F6 — 预注册时间不可核实，且文档日期晚于提交与当前日期
- **严重程度：中等**
- **位置**：`docs/DEV_DECISIONS.md:1`（"2026-10-11，在看到任何 dev 结果之前写定"）；`git log` 中 `cd48ab8`（2026-10-10 10:59:26 +0800）。
- **证据**：规则文件与 `dev_select.py` 都只在提交 `cd48ab8` 中首次出现。同一提交还：把默认 EAB 从 `ans2` 改为 `ans1`（`git show cd48ab8 -- src/omniaware/run.py` 的 L34-35）、新增 probe 轮、zsum、mid、GS-112K 候选，并标注 "pre-registered dev selection"。文档日期（10-11）晚于提交时间（10-10 10:59）与当前时间（10-10 12:04）。
- **影响**：无法从 git 证明规则是在看到 dev 结果之前写的。新增的候选（zsum、mid、GS-112K、probe）与规则在同一提交中出现，选择的候选集合也可能是事后决定的。这与 DEV_DECISIONS 的目的（避免事后挑选）直接相关。
- **修复建议**：把规则文件作为单独提交，并在规则中记录 dev 运行的 commit 哈希；修正日期；若已看过 dev 结果，诚实说明。
- **推测**：是。是否在看到 dev 结果后才定稿需要 dev 运行的时间戳证据（`runs/` 不在仓库中）。

### F7 — EAB 选择只在 base U 下评估；池化数据集硬编码；缺失数据被静默丢弃
- **严重程度：中等**
- **位置**：`scripts/analysis/dev_select.py:31`（`DEV_DATASETS` 含 `okvqa_gs`）；`:62`（EAB 只用 `_path(runs,"base",...)`）；`:42-43`（`_available` 静默过滤）；`scripts/paper_runs.py:79-82`（缺失的 OK-VQA 语料被跳过）。
- **证据**：DEV_DECISIONS 规则 1 没有规定池化数据集，代码固定为 4 个数据集，其中 `okvqa_gs` 在规则 3 选语料之前就已加入。`select_eab` 只看 base，而 `u_combine`/`attn_layer` 是后面另外选的（`decide` L112），所以被选中的 U 设置（zsum 或 mid）从未与 EAB 形式一起比较。
- **影响**：DEV_DECISIONS 的"6 × 3 网格"描述与执行不一致；如果规则 3 选中的不是 okvqa_gs，EAB 的池化仍然用 okvqa_gs；数据集缺失时池化静默变少。
- **修复建议**：在规则中明确池化数据集；EAB 选择应在最终选中的 U 设置下重新比较（或明确只用 base 并写进规则）；缺失时报错。
- **推测**：否。

### F8 — WebQA 评分只取第一个参考答案，绕过了"必须恰好一个参考"规则
- **严重程度：中等**
- **位置**：`src/omniaware/eval/score_official.py:87-88`（`r['golds'][:1]`）；对照 `src/omniaware/eval/webqa_acc.py:84-85`（`score_references` 对 `len(references) != 1` 报错）与 `webqa_acc.py:76`（provenance 声明 "exactly one reference"）。
- **证据**：`score_webqa` 每次都传 `[:1]`，所以 `score_references` 的检查永远不会触发。`tests/test_webqa_acc.py:39-41` 只在直接调用 `score_references` 时测试该规则。
- **影响**：如果某个 WebQA 样本有多个 `A` 答案，它会被静默截断为第一个，评分不再是官方规则。`data.py:70` 把全部 `A` 放入 golds，所以截断真实发生在评分端。
- **修复建议**：去掉 `[:1]`，直接把整个 golds 传给 `score_references`，让它在多参考时报错；或者在加载时计数并报告多参考样本。
- **推测**：是。数据中多参考样本的数量未核实（训练/验证集的数据不在仓库）；provenance 声称 41,732 个原始样本都只有一个参考。

### F9 — 第三方 pinned 文件的哈希只记录不校验
- **严重程度：中等**
- **位置**：`src/omniaware/eval/webqa_acc.py:25-28`（`OFFICIAL_FILES` 期望哈希）、`:49-53`（读取并计算哈希），`:70`（写入 provenance）。
- **证据**：读取 `OFFICIAL_FILES` 中的每个文件后，从未用期望哈希做比较或 assert。我用 `sha256sum` 验证：`third_party/webqa/eval_1022.py` = `6862a280…`，`eval_webqa.md` = `c6dfcb9a…`，与 pin 一致。
- **影响**：当前内容正确，但如果有人替换了源文件，评分会静默使用新代码，只有事后看 provenance 才能发现。CONVENTIONS §5 把官方评测作为硬要求。
- **修复建议**：不匹配时 `raise`。
- **推测**：否。

### F10 — 校准与相关度缓存只按 qid 复用，不校验候选内容
- **严重程度：中等**
- **位置**：`src/omniaware/run.py:25-26`（`load_relevance` 直接复用 `raw[qid]`）、`:420-421`（`ucal` 跳过已有 qid）、`:482-483`（SeaKR 校准缓存）；`src/omniaware/rerank.py:102-104`（CE 缓存按 qid）。
- **证据**：这些缓存的键只有 qid（和数据集 + 配置后缀），没有候选的哈希或检索版本。`relevance_{dataset}{data_tag}.json` 也不区分 `--universalrag-top` 之外的候选来源变化。
- **影响**：如果候选集被重建（例如 WebQA 开放检索重跑、OK-VQA 语料更换、UniversalRAG 导出的排名变化），旧的相关度、U 校准和 CE 分数会被静默复用，并且与新候选不一致。这与 CONVENTIONS §6 与 LEGACY_MAP 的"不复用缓存"声明有冲突（预测不复用，但校准缓存复用）。
- **修复建议**：缓存键加入候选内容的哈希（或数据文件的 sha256）；不匹配时重算并计数。
- **推测**：是。需要真实资产变化的场景来确认影响大小。

### F11 — `Calibrator.ecdf` 通道缺失时静默返回 0.5
- **严重程度：中等**
- **位置**：`src/omniaware/relevance.py:144-148`
- **证据**：`if not arr: return 0.5`。当校准集中没有某个通道（如 `image_cap`、`video_clip`）时，所有该通道的候选都得到 0.5，且不报错、不计数。
- **影响**：路由与排序会被静默扭曲（所有该模态的候选得分相同），违反 CONVENTIONS §2 "失败要明确报错"与 §5 的"数据缺失必须计数并报告"。
- **修复建议**：通道缺失时报错，或至少计数并写入配置。
- **推测**：否（代码路径已核实；触发频率取决于数据）。

### F12 — VideoRAG 的簇键从 qid 字符串推导，依赖未核实的 index 格式
- **严重程度：中等（推测）**
- **位置**：`src/omniaware/eval/universalrag_metrics.py:88`（`extra = {"clusters": [q.rsplit("_", 1)[0] for q in qids]}`）；`src/omniaware/data.py:319`（qid 直接取 `q["index"]`，不取源视频）。
- **证据**：数据加载器的 qid 是 upstream 的 `index`（`data.py:287-288`），而簇键是 qid 去掉最后一个 `_` 后的部分。源视频在 `gt_videos` 中（`data.py:292`）可以直接用，但没有被使用。`tests/test_score_official.py:39-50` 与 `tests/test_run_smoke.py:343` 都假设 qid 形如 `videorag_synth_<video>_<n>`，这个假设没有来自真实数据。
- **影响**：如果真实 index 不是这种形式（例如 `videorag_synth_<n>`），"按源视频聚类"会退化为按 qid 的前缀分组，置信区间错误（且 PROTOCOL §4 声称按源视频聚类）。
- **修复建议**：簇键应取源视频（`gt_videos` 或 `source_id`），并把它写入每条记录，而不是从 qid 字符串推导。
- **推测**：是。需要 UniversalRAG 的 `videorag_synth.json` 的真实 index 格式来确认。

### F13 — `parse_labels` 把未知标签静默映射为 "no"
- **严重程度：中等**
- **位置**：`src/omniaware/baselines/universalrag.py:53-59`
- **证据**：`parse_labels(['audio'])` 返回 `['no']`（已复现）。注释说明 "'error' dropped; nothing valid -> ['no']"，但未知标签与 'error' 不一定是同一种情况。`to_route` 只在过滤之后的标签上做检查（`:64-66`），所以未知标签不会报错，而是变成"不检索"。
- **影响**：路由器的未知输出被当作"不检索"，与 CONVENTIONS §2 "失败要明确报错"冲突，并影响 UniversalRAG 基线。
- **修复建议**：未知标签计数并报错（或至少单独计数），不要并入 "no"。
- **推测**：是。上游 `eval.py` 的 normalize 规则未在本地核实，需要它来确定"未知"的处理是否与上游一致。

### F14 — 注意力捕获在偏置之前，且"第一个答案行"与 EAB 的首个被偏置行不是同一行
- **严重程度：中等（推测）**
- **位置**：`src/omniaware/qwen_eab.py:54-58`（捕获在偏置之前计算）、`:60-80`（偏置）；`:279-281`（docstring："the first row EAB biases"）、`:291-293`（捕获的是第一个生成 token 的行）；`src/omniaware/pipeline.py:166`（ans1 的 q_start = enc.length − 1）；`tests/test_qwen_eab.py:87-96`。
- **证据**：捕获（L54-58）在 `attention_mask` 修改之前计算 `softmax(q·kᵀ)`，因此记录的是未加偏置的 logits。`_decode` 中 `capture=len(tokens)==1`（L291）捕获的是第一个**生成** token 的行，即位置 `enc.length` 的行，它的输出预测第二个答案 token。而对 `ans1`（论文默认），第一个被偏置的行是 `enc.length−1`（预测第一个答案 token 的最后一个提示行），不是被捕获的那一行。只有 `ans2` 的首个被偏置行才与捕获行一致。
- **影响**：`Result.attention` 进入 `rec["ard"]`（`run.py:609-611`）。对 ans1 变体，ARD 计算的是另一行、且未加偏置的注意力，与 docstring 与测试名称描述的"first answer row"不符。这影响 ARD 分析（论文附录）的解释。
- **修复建议**：明确 ARD 应该使用哪一行与哪种 logits（偏置后还是偏置前）；更正 docstring；测试要断言被捕获行的索引与 q_start 的关系。
- **推测**：是。需要论文 ARD 定义（"attention of which row, biased or not"）来确定哪一种是 bug。

### F15 — FITM 空值参考覆盖了视觉 span 的标记 token
- **严重程度：中等**
- **位置**：`src/omniaware/baselines/arms.py:46-51`（`fitm_sources` 只把 `image_token_id` 归为视觉，`<|vision_start|>`/`<|vision_end|>` 被当作文本）；`src/omniaware/baselines/fitm_qwen_adapter.py:123-128`（对 `text_token_indices` 写入虚拟 token）。
- **证据**：我用 `tests/tiny_qwen.image_prompt()`（ids `[2,60,61,62,6,7]`）和真实布局的 span `(1,4)` 复现：`fitm_sources` 返回 `text_token_indices=[1,3]`，`vision_token_indices=[2]`；`make_null_reference` 后输入变为 `[2,2,61,3,6,7]`（已复现）。即 vision_start(60)、vision_end(62) 被替换成文本 token。`make_null_reference` 之后的前向仍然成功（tiny 模型）。
- **影响**：对带图像证据的块，FITM 的 null 参考与真实输入在标记 token 上不同，偏离"只替换 source 内容"的设计；被替换的标记 token 进入了 document-mean 的计算，影响 α。由于 5.5.3 的 rope 由 `mm_token_type_ids` 决定，位置不变（见 §4.1），所以影响限于 token 身份与注意力。仅影响 FITM 臂（只支持 Qwen3-VL）。
- **修复建议**：`fitm_sources` 应把 vision_start/vision_end 归入视觉或保留（不替换）；增加测试：真实 span 布局下 `make_null_reference` 不改变标记 token。
- **推测**：影响大小需要真实模型上 α 的对比（推测）。

### F16 — 运行级测试的 FakeQwen 使预测只由长度决定，多数断言形同虚设
- **严重程度：中等**
- **位置**：`tests/test_run_smoke.py:31-63`（FakeQwen）；断言 `:95,126,275,385`。
- **证据**：`FakeQwen.run` 返回 `Result(f"{name}@{enc.length}")`，`make_variants` 的 spec 从未被检查。`test_cold_start_probe_round_gates_and_routes` 的最后一个断言接受任一输出（`:385`）；`test_seakr_arms` 的 `!=` 只靠长度（`:275`）。
- **影响**：运行级的 EAB 语义（spec→bias→span、gate、干预、复制）没有任何端到端的验证。`run.py` 的大部分逻辑只有"格式与计数"被测。
- **修复建议**：在运行级测试中用真实 `QwenEAB`（tiny 模型）或让 FakeQwen 基于 spec 计算输出（例如按偏置改变输出），并断言 gate 关闭时与 static 完全相同、打开时不同。
- **推测**：否。

### F17 — `dev_select` 测试只覆盖正路径
- **严重程度：中等**
- **位置**：`tests/test_dev_select.py:108-131`
- **证据**：数据使得 base U 必然胜出（`:112-118,124-126`），EAB 形式的 CI 必然 > 0，NTE 的 ρ 接近 0。没有任何测试覆盖：缺失数据集、缺失 dev 运行（F2）、framing 2、framing 3、zsum/mid 胜出、并列。
- **影响**：F2 与 F7 的问题未被测试捕获。
- **修复建议**：增加上述分支的测试，尤其是"无 dev 结果应报错"。
- **推测**：否。

### F18 — `make_qwen` 绕过 `QwenEAB.__init__`，并以不同方式标层
- **严重程度：中等**
- **位置**：`tests/eab_fixtures.py:38`（`QwenEAB.__new__`）、`:41-44`（按类名 `Qwen3VLTextAttention` 标层）；生产 `src/omniaware/qwen_eab.py:130-136`（`tag_text_layers` 按结构标层）、`:147-173`（`__init__`）。
- **证据**：fixture 手动设置 `q.device, q.model, q.tok, q.processor`，没有 `q.kind`（依赖 `getattr` 默认值，`qwen_eab.py:188,249`）。`make_internvl`（`:73-75`）则用了 `tag_text_layers`。所以 Qwen 的生产标层路径与 `__init__` 都未被测试。
- **影响**：`__init__` 中的 `kind` 检测、`eos` 集合、`vocab_size`、`max_pixels`/`size`、`merge_size` 都是未测试代码。
- **修复建议**：让 fixture 通过 `__init__` 的一个可注入路径（例如接受已加载的 model 与 processor），然后调用生产的 `tag_text_layers`。
- **推测**：否。

### F19 — 钩子测试只覆盖 `attention_mask=None`
- **严重程度：中等（推测）**
- **位置**：`tests/test_qwen_eab.py:22-28`（传 `None`）；生产 `qwen_eab.py:65-74`（`None`、布尔、加性三个分支）。
- **证据**：`test_hook_matches_explicit_softmax_with_bias` 只用 `None`。布尔与加性分支只在生产代码中。
- **影响**：如果 transformers 在某些情况下（如带 padding、或 `create_causal_mask` 返回布尔 mask）传入布尔/加性 mask，这两个分支没有测试保护。
- **修复建议**：参数化测试：None、布尔（含 padding 列）、加性（含 −inf）。
- **推测**：是。需要确认 5.5.3 在 batch=1 无 padding 时的 mask 形态。

### F20 — transformers 版本声明与实际验证不一致
- **严重程度：中等**
- **位置**：`pyproject.toml:9`（`transformers>=4.57`）；README §1（`transformers>=4.57`）。
- **证据**：所有测试与源码检查只在 transformers 5.5.3（已核实版本）。代码使用 `from_pretrained(dtype=...)`、`AttentionInterface.register`、`repeat_kv`、`DynamicCache.crop` 等。
- **影响**：4.57 下的 `get_rope_index`（可能依赖 vision_start 而不是 `mm_token_type_ids`）、签名差异未验证（推测）。用户按 README 安装 4.57 时可能得到不同行为。
- **修复建议**：要么把下界提高到实际测试过的版本，要么在 4.57 环境中跑测试并记录。
- **推测**：是。

### F21 — InternVL 默认单 tile，与官方动态分块不同
- **严重程度：中等（推测）**
- **位置**：`src/omniaware/run.py:93`（`--max-patches` 默认 1）；`src/omniaware/qwen_eab.py:147-148,165`；`paper_runs.py` 未传 `--max-patches`。
- **证据**：`_internvl_images` 以 `crop_to_patches = max_patches > 1` 处理，默认只得到一个 448 的 tile（256 token）。InternVL3.5 的官方推理使用动态分块与缩略图（需要检查点的 `preprocessor_config.json` 确认）。
- **影响**：骨干迁移（论文附录）的 InternVL 结果反映的是"单 tile 输入"的 InternVL，而不是官方推理设置；论文需要说明，否则不可比。
- **修复建议**：在论文与 README 中说明；或在主实验中采用官方设置。
- **推测**：是。需要论文中的设置说明与 InternVL 的官方推理配置。

### F22 — 对话模板是手写的 ChatML，没有与 processor 的模板做比对
- **严重程度：中等（推测）**
- **位置**：`src/omniaware/pipeline.py:41,71`（`<|im_start|>user\n`、`<|im_end|>\n<|im_start|>assistant\n`）；`src/omniaware/qwen_eab.py:216`。
- **证据**：没有任何活跃测试比较 `enc.input_ids` 与 `processor.apply_chat_template` 的输出。`tests/test_m2rag_qwen.py:55-57` 做了这种比较，但整文件被 torchvision 跳过（F4）。
- **影响**：模板不一致（如 InternVL 的系统提示、Qwen3-VL 的 think 标记）会让模型在分布外输入上运行，所有臂同样受影响，但绝对数字会偏离论文设置。
- **修复建议**：用检查点的 `chat_template` 生成期望 token，断言一致（或明确记录差异）。
- **推测**：是。需要检查点的 `chat_template`（不在本地）。

### F23 — PROTOCOL 中的默认 EAB 变体写错（ans2 vs 代码的 ans1）
- **严重程度：中等**
- **位置**：`docs/PROTOCOL.md:33`；代码 `src/omniaware/run.py:155`、`src/omniaware/pipeline.py:136`、`scripts/analysis/dev_select.py:120`。
- **证据**：PROTOCOL 说默认是 `eab:g1.2:ans2:pos:gate`"即论文 decode-only"；代码与 DEV_DECISIONS 都是 ans1，并且 `git show cd48ab8` 把默认改成了 ans1。
- **影响**：读文档的人会误解默认设置；与 F6 的"规则是否事后修改"问题直接相关。
- **修复建议**：更正 PROTOCOL；在 PROTOCOL 中列出 ans1 与 ans2 的角色。
- **推测**：否。

### F24 — 文档对 UniversalRAG 与 M2-RAG 的描述矛盾
- **严重程度：中等**
- **位置**：`docs/LEGACY_MAP.md:55`（"论文的结果表里没有这个方法"，未迁移）；与 `docs/PAPER_COVERAGE.md:52`、`docs/PROTOCOL.md:42`、`README.md` §4.4、`scripts/paper_runs.py:125-134` 矛盾。另：`docs/PROTOCOL.md:39-41` 与 `src/omniaware/baselines/arms.py:71-91`（M2-RAG 使用官方 summarizer 与 384 文档 cap）。
- **证据**：LEGACY_MAP 说 UniversalRAG 不在论文表中，而 PAPER_COVERAGE 把它列在主表（§3）并给出运行选项，`paper_runs` 有 `universalrag` 阶段。M2-RAG 的提示词与 PROTOCOL "提示词格式相同"的说法不符。
- **影响**：读者无法判断 UniversalRAG 是否属于论文的主表对照。
- **修复建议**：统一为一种说法；M2-RAG 的提示词差异写入 PROTOCOL。
- **推测**：否。

### F25 — "入口只有一个"与别名匹配的归因评分、以及 CONVENTIONS 的"禁止宽松匹配"冲突
- **严重程度：中等**
- **位置**：`docs/CONVENTIONS.md:67-68`（"入口只有一个"、"禁止宽松匹配、同义词表"）；`src/omniaware/eval/citation/attribution.py:55-62`（别名匹配）、`:65-78`（S(y,e)）；`src/omniaware/eval/webqa_acc.py:162`（另一个 main）；`docs/citation_support_protocol.md:1`；`docs/LEGACY_MAP.md:49`（以"宽松子串匹配"为由不迁移评分器）。
- **证据**：归因的别名匹配使用词界正则与标签，比子串匹配严格，但它是一种与标签相关的匹配规则；文档对它的态度前后不一。
- **影响**：引用指标（Faith/AP）的定义依赖一个 CONVENTIONS 没有豁免的匹配规则；读者可能认为"违反规范"。
- **修复建议**：在 CONVENTIONS 中明确引用指标的匹配规则与豁免；列出所有评分入口。
- **推测**：否。

### F26 — MIGRATION_SOURCES.json 的目标路径与删除/改名记录不一致
- **严重程度：中等（provenance）**
- **位置**：`MIGRATION_SOURCES.json`（`files` 34 条、`files_round2` 59 条）；`docs/LEGACY_MAP.md` §2026-10-11。
- **证据**：93 个目标中 36 个不存在（已核实）。其中 10 个（第一轮 answer_format、table_render、mmqa、videomme、相关测试）与 LEGACY_MAP 删除清单一致；但 `legacy_staging/*`（9 个）、`scripts/webqa_open_retrieval/` 的 9 个脚本（如 `calibrate_dense_relevance.py`、`continue_dense_cpu.py`）、`docs/*DENSE*`（3 个）、`scripts/analysis/okvqa_passage_answer_recall.py`（实际为 `answer_recall.py`）在文档中没有说明。
- **影响**：溯源声明（"每个文件或对象的来源路径和 SHA256 都记录"）对这些条目不可用；读者无法区分"删除"与"改名"。
- **修复建议**：在 manifest 中加入 `status` 字段（kept / renamed / deleted）并更新 LEGACY_MAP。
- **推测**：否（已核实不存在）；源文件 SHA256 无法在本环境核实。

### F27 — `scripts/launch.py` 的多进程编排未测试；merge 只核对题数
- **严重程度：中等（launch）；轻微（merge）**
- **位置**：`scripts/launch.py:45-84`（`launch`）；`src/omniaware/merge_shards.py:48-49`。
- **证据**：`tests/test_run_smoke.py:388-415` 只测试 `run.main` 分片与 `merge`，以及 `launch.detect_gpus`。没有测试 `launch()` 的 Popen、失败分片的 `SystemExit`、续跑时的 `out.exists()` 早退。merge 只检查 `len(rows) == expected`（`merge_shards.py:48`），不检查 qid 集合与 split 一致。
- **影响**：README §5 的"中断后用原命令重跑即可续跑"与"合并前核对每道题恰好回答一次"只部分有测试保护。
- **修复建议**：测试 `launch()` 的失败与续跑路径（可用假的 `python` 可执行文件）；merge 应核对 qid 集合与 split 的一致性。
- **推测**：否。

### F28 — bootstrap 次数在两个评分器之间不一致
- **严重程度：轻微**
- **位置**：`src/omniaware/eval/score_official.py:51-57`（5000）；`src/omniaware/eval/webqa_acc.py:90`（10000）；`docs/PROTOCOL.md:58`（5,000）。
- **影响**：同一文档对两种评分给出了不同的统计设定，读者可能混淆；`webqa_acc` 的 10000 不在文档中。
- **修复建议**：统一或在文档中注明各自用途。
- **推测**：否。

### F29 — 测试文件与缓存中残留的 Video-MME 痕迹
- **严重程度：轻微**
- **位置**：`tests/test_score_official.py:1`（docstring 提到 Video-MME）；`.pytest_cache/v/cache/lastfailed` 中的 `test_videomme_parses_letters_and_clusters_by_video`（该测试已不存在）。
- **修复建议**：更新 docstring；清理缓存（缓存是 git 忽略的，不影响结果）。
- **推测**：否。

### F30 — README 与引用协议对跳过原因的描述不准确；NLTK 版本不一致
- **严重程度：轻微**
- **位置**：`README.md:294`（"BARTScore、spaCy、NLI 检查器"）；`tests/test_citation_intake.py:12-13`（跳过的是 NLTK Punkt 资产）；`docs/citation_support_protocol.md:43`（NLTK 3.9.2）。
- **证据**：本环境 nltk 为 3.8.1（已核实）。
- **修复建议**：更正 README 的跳过列表；在环境说明中写明 NLTK 版本要求。
- **推测**：否。

### F31 — `relevance.py` 中无来源的死条件
- **严重程度：轻微**
- **位置**：`src/omniaware/relevance.py:115`（`not c["cid"].startswith("f")`）
- **证据**：代码库中所有候选 cid 前缀为 t、i、k、p、d、v、c、`__query__`、`__inject__`，没有 "f" 开头的。该条件没有注释或来源。
- **修复建议**：删除，或注释说明它排除的是哪类候选。
- **推测**：否。

### F32 — "Qwen3-VL 的默认 256-token 预算"的说法未核实
- **严重程度：轻微（推测）**
- **位置**：`src/omniaware/qwen_eab.py:144`（docstring）；`src/omniaware/run.py:164`（`--max-pixels` 默认 `256*32*32`）。
- **证据**：本地 transformers 的 Qwen2-VL 图像处理器默认 `longest_edge = 28*28*1280`（`qwen2_vl/image_processing_qwen2_vl.py:95`）。256 token 是本仓库的默认，不是 Qwen3-VL 的默认（Qwen3-VL 检查点的 `preprocessor_config.json` 不在本地）。
- **修复建议**：把"256 token"写成本仓库的设定，不要写成 Qwen3-VL 的默认。
- **推测**：是（需要检查点配置）。

### F33 — 干预的替代实体池使用其他评测题的 gold 答案
- **严重程度：轻微**
- **位置**：`src/omniaware/run.py:498`（`substitute_pool(evals, iv_nlp)`）；`src/omniaware/interventions.py:89-97`。
- **证据**：替代实体取自 `evals`（dev+test）的 golds。这只用于构造反事实，不参与评分或选择，但它说明干预使用了评测标签。
- **修复建议**：在 PROTOCOL §8 中说明；或改用训练/校准题的答案作为池。
- **推测**：否。

### F34 — 随 `--m2rag` 运行，`gated:m2rag` 不会生成
- **严重程度：轻微（推测其是否为设计意图）**
- **位置**：`src/omniaware/run.py:636-640`（m2rag 不加入 `arms`）；`:664-665`（`gated:` 循环只遍历 `arms`）。
- **证据**：`fitm` 会被加入 `arms`（L636），`m2rag` 不会。因此 `preds` 中有 `m2rag`，但没有 `gated:m2rag`。
- **影响**：如果论文表中有门控版本的 M2-RAG，就无法从输出中得到。
- **修复建议**：确认设计意图；若需要则加入 `arms`。
- **推测**：是（意图未知）。

### F35 — 配对检验的不相交只在构造时保证，运行时只检查 calib∩eval
- **严重程度：轻微**
- **位置**：`src/omniaware/run.py:254`；`src/omniaware/data.py:357-372`。
- **证据**：`assert not {calib} & {evals}` 只覆盖 calib 与当前 split；dev 与 test 的不相交只通过 `split_examples` 的构造保证（`tests/test_splits.py` 验证）。
- **修复建议**：在运行时同时断言 dev、test、calib 两两不相交（运行 `--split test` 时也应能检查 dev 的存在与否）。
- **推测**：否。

### F36 — SeaKR 推理与校准对无穷分数的处理不一致
- **严重程度：轻微**
- **位置**：`src/omniaware/run.py:645`（`direct_eigen > seakr_threshold`，inf 为 True）；`src/omniaware/baselines/seakr.py:77,82`（inf 在校准 `matched_threshold` 中被排除）。
- **证据**：样本不足（<=1 个 EOS 状态）时 `score_input` 返回 inf。推理时 inf 被当作"检索"，而校准时 inf 不参与阈值估计。
- **影响**：样本不足的题在推理端被系统性地判为检索，但它们没有计入校准的检索率。
- **修复建议**：统计样本不足的题数并报告；明确 inf 的处理规则。
- **推测**：否（影响大小取决于样本不足的频率）。

### F37 — 测试在 tests/ 目录下创建临时目录
- **严重程度：轻微**
- **位置**：`tests/test_citation_intake.py:83,109,118`（`tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)`）
- **影响**：如果测试崩溃，可能在仓库内留下残留目录（会被 `.gitignore` 或 git 状态暴露）。
- **修复建议**：使用 `tmp_path` 或系统临时目录。
- **推测**：否。

### F38 — `external_repos.json` 中 InternVL 的角色描述陈旧
- **严重程度：轻微**
- **位置**：`third_party/external_repos.json`（InternVL 条目，"InternVL2-40B backbone transfer"）。
- **证据**：论文与代码已改为 InternVL3.5-8B（`docs/PAPER_COVERAGE.md:70`，`src/omniaware/paths.py:24`）。
- **修复建议**：更新 role 字段。
- **推测**：否。

---

## 6. 没来得及深入的方向

1. **`scripts/analysis/paper_analyses.py`（467 行）**：只读了前 80 行与 `analysis.py`；`order_table`、`latency_table`、`intervention_table`、`paired_test` 等的统计定义没有逐行核对，只依赖 `test_paper_analyses.py` 的合成数据。
2. **`baselines/m2rag_official.py`、`m2rag_qwen.py`、`m2rag_webqa.py`**：只读了调用端（`arms.py:71-91`）；官方 summarizer 的提示词与 token 截断没有核对。`test_m2rag_qwen.py` 被跳过。
3. **`baselines/fitm_attention.py` 的数学**（`redistribute_document_attention` 等）：只看了测试的断言，没有核对 Hsieh et al. 公式 4、5，以及 `TEMPERATURE=0.00005` 的来源。
4. **`eval/universalrag_metrics.py` 的 EM/F1/ROUGE-L 实现**：只看了 `score` 与测试。BERTScore 的调用未核对。
5. **`eval/webqa_bart.py`**（BARTScore 的 ParaBank 加权）：没有读源码，`test_webqa_bart.py` 大部分在本环境运行，但依赖 BART 资产的部分被跳过。
6. **`rerank.py` 的 `ce_select` 与 `ce_calibrated`**：只读了 `ce_logits` 与测试的 `rank_sources`。
7. **`scripts/setup/*`、`scripts/data_prep/*`、`scripts/webqa_open_retrieval/*`**：没有审阅（它们在本环境无资产，无法运行）。`check_assets.py` 的声明未核对。
8. **`okvqa_dense.py` 与 GS-112K 的来源**：没有核实 GS-112K 语料是否与 OK-VQA 标注有重叠（潜在的标签泄漏）。
9. **`scripts/analysis/power_design.py`**：PROTOCOL §6 声称在比较候选前应运行它，但代码中没有任何强制步骤；未检查其计算。
10. **真实模型的端到端运行**：未在任何真实检查点上运行。Qwen3-VL DeepStack 注入、InternVL 的 `chat_template`、`generation_config.json` 与 `preprocessor_config.json`、bf16 的数值差异、显存与 batch 行为均未验证。
11. **分词边界**（§4.2 第 3 点）：未用真实 tokenizer 对比逐段与整串分词。
12. **`run.py` 中 `--norm-from` 的跨数据集校准迁移**（`run.py:449-455`，`paper_runs.py:155-157`）：是否引入了 test 标签的使用，未核对（推测无，因为只读取 config 中的 norm）。
13. **`interventions.py` 中 `inject`、`distractors`、`mask_tokens`** 的完整实现：只看了测试与部分调用。
14. **`merge_shards` 与 `run.py` 的 `done` 续跑逻辑在并发写入时的安全性**：未审阅并发写入。

---

## 摘要（每条一行：严重程度 + file:line）

- F1 严重 — `scripts/paper_runs.py:34-40`：dev 决策缺失时 test 阶段静默回退到默认设置。
- F2 严重 — `scripts/analysis/dev_select.py:110-127,132,137`：无 dev 结果时静默输出 framing=3；路径与 `paths.RUNS` 不一致。
- F3 严重（推测频率） — `src/omniaware/baselines/universalrag.py:62-69` + `src/omniaware/run.py:52-56,613-615`：路由模态全缺失时退化为联合检索。
- F4 严重 — `tests/test_webqa_acc.py:8`、`tests/test_citation_intake.py:12-13`、`tests/test_m2rag_qwen.py:5`：25 个测试本环境未执行，"107 passed" 掩盖。
- F5 严重 — `src/omniaware/qwen_eab.py:184-200,175-179,147-173`：真实图像与 InternVL 像素路径、`__init__` 零测试。
- F6 中等（推测） — `docs/DEV_DECISIONS.md:1`：预注册时间不可核实，日期晚于提交与今天。
- F7 中等 — `scripts/analysis/dev_select.py:31,42-43,62`：EAB 只在 base U 下比较，池化数据集硬编码，缺失数据静默丢弃。
- F8 中等（推测） — `src/omniaware/eval/score_official.py:87-88`：WebQA 评分取 `golds[:1]`，绕过单参考规则（`webqa_acc.py:84-85`）。
- F9 中等 — `src/omniaware/eval/webqa_acc.py:25-28,49-53`：第三方 pin 哈希只记录不校验（当前内容与 pin 一致）。
- F10 中等（推测） — `src/omniaware/run.py:25-26,420-421,482-483`、`src/omniaware/rerank.py:102-104`：缓存只按 qid 复用，不校验候选内容。
- F11 中等 — `src/omniaware/relevance.py:144-148`：通道缺失时 ECDF 静默返回 0.5。
- F12 中等（推测） — `src/omniaware/eval/universalrag_metrics.py:88`：VideoRAG 簇键从 qid 字符串推导。
- F13 中等（推测） — `src/omniaware/baselines/universalrag.py:53-59`：未知路由标签静默变为 "no"。
- F14 中等（推测） — `src/omniaware/qwen_eab.py:54-58,279-293`：ARD 注意力为偏置前、且行与 ans1 的首个被偏置行不同；docstring 与测试只查形状。
- F15 中等 — `src/omniaware/baselines/arms.py:46-51` + `fitm_qwen_adapter.py:123-128`：FITM 空值参考覆盖了 vision_start/end（已复现）。
- F16 中等 — `tests/test_run_smoke.py:31-63,95,126,275,385`：FakeQwen 使运行级断言只靠长度/形状。
- F17 中等 — `tests/test_dev_select.py:108-131`：只覆盖正路径，缺失数据与无结果分支未测。
- F18 中等 — `tests/eab_fixtures.py:38,41-44`：fixture 绕过 `__init__` 且与生产标层方式不同。
- F19 中等（推测） — `tests/test_qwen_eab.py:22-28`：钩子测试只覆盖 mask=None。
- F20 中等 — `pyproject.toml:9`：声明 transformers>=4.57，实际只在 5.5.3 验证。
- F21 中等（推测） — `src/omniaware/run.py:93`，`qwen_eab.py:147-148`：InternVL 默认单 tile，与官方动态分块不同。
- F22 中等（推测） — `src/omniaware/pipeline.py:41,71`：手写 ChatML，无 apply_chat_template 比对。
- F23 中等 — `docs/PROTOCOL.md:33`：默认 EAB 写成 ans2，代码为 ans1。
- F24 中等 — `docs/LEGACY_MAP.md:55` 对比 `PAPER_COVERAGE.md:52`、`PROTOCOL.md:42`：UniversalRAG 是否在论文主表互相矛盾。
- F25 中等 — `docs/CONVENTIONS.md:67-68`、`src/omniaware/eval/citation/attribution.py:55-62`：入口与别名匹配规则与 CONVENTIONS 冲突。
- F26 中等 — `MIGRATION_SOURCES.json`（36/93 目标缺失）：`legacy_staging/*` 等删除/改名未在文档中说明。
- F27 中等（launch）/轻微（merge） — `scripts/launch.py:45-84`，`src/omniaware/merge_shards.py:48-49`：launch 编排未测；merge 只核对题数。
- F28 轻微 — `score_official.py:51-57` vs `webqa_acc.py:90`：bootstrap 次数 5000 与 10000 不一致。
- F29 轻微 — `tests/test_score_official.py:1`：docstring 提到已删除的 Video-MME。
- F30 轻微 — `README.md:294`，`tests/test_citation_intake.py:12-13`，`docs/citation_support_protocol.md:43`：跳过原因与 NLTK 版本描述不准确。
- F31 轻微 — `src/omniaware/relevance.py:115`：无来源的死条件 `startswith("f")`。
- F32 轻微（推测） — `src/omniaware/qwen_eab.py:144`：把 256 token 说成 Qwen3-VL 默认。
- F33 轻微 — `src/omniaware/run.py:498`：干预替代池使用评测题的 gold。
- F34 轻微（推测） — `src/omniaware/run.py:636-640,664-665`：`gated:m2rag` 不生成。
- F35 轻微 — `src/omniaware/run.py:254`：运行时只断言 calib∩eval，dev/test 不相交仅靠构造保证。
- F36 轻微 — `src/omniaware/run.py:645`，`baselines/seakr.py:77,82`：SeaKR 推理与校准对 inf 的处理不一致。
- F37 轻微 — `tests/test_citation_intake.py:83,109,118`：临时目录建在 tests/ 下。
- F38 轻微 — `third_party/external_repos.json`（InternVL 行）：角色描述陈旧（InternVL2-40B）。

### 已核实属实的声明（摘要）
pyflakes 无警告；校准不读取答案；`--select-tau` 已删除；calib/eval 运行时断言；ans2 不改首个 token（测试通过）；14 个排列；反事实实体的哈希选择；精确内积检索；SeaKR eigen 与上游逐字一致；OK-VQA 使用官方 VQAEval；第三方 pin 的哈希与文件内容一致（但代码不校验）；ALCE 固定 commit；未迁移 baseline 预测缓存（预测每题重算）。

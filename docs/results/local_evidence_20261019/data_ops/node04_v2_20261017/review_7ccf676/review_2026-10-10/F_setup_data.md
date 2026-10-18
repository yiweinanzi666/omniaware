# F. 可复现性与数据准备审查（从零服务器按 README 执行）

仓库：`C:\Users\22688\Desktop\www_code\omniaware`（下文路径均相对该根目录）。审查时间 2026-10-10。
本轮未下载任何模型或大数据集；外部核对只用 HEAD 请求、HF API 元数据、GitHub API 和 raw 小文件。
验证脚本与中间输出在 `scratchpad\review\`（`hf_check.py`、`nli_manifest_check.py`、`verify_pil_executor.py`、`verify_pool_and_skip.py`、`verify_images_e2e.py`）。

严重程度定义：致命 = 按 README 会卡死或产生无法察觉的错误结果；严重 = 结果或数据集口径不对、需人工大量干预、或与论文主张不符；中等 = 可复现性或可维护性有明显缺陷，有绕过办法；轻微 = 文档、体验、次要一致性。

---

## 0. 已核实通过的项（避免重复劳动）

- 测试：`python -m pytest -q tests/test_setup_webqa_open.py tests/test_webqa_open.py tests/test_okvqa_gs.py` 三个用例全部通过。全量 `python -m pytest -q` 全部通过，2 至 5 个跳过（缺 spaCy、torchvision 及资产门控，见 F24/F18），耗时约 38 s（README 写"约 10 秒"，见 F25）。
- `python -m pyflakes src tests scripts/setup scripts/data_prep scripts/webqa_open_retrieval scripts/universalrag_retrieval scripts/fetch_external.py scripts/launch.py scripts/paper_runs.py` 无警告。
- 下载来源可达（HTTP 200）：okvqa.allenai.org 两个 zip，images.cocodataset.org val2014.zip，WebQA GitHub 仓库，luomancs GS 仓库，BARTScore 仓库，BARTScore 的 Google Drive ID（与上游 README 第 29 行一致）。DPR `psgs_w100.tsv.gz` 可达，Content-Length 4,694,541,059，与 `scripts/data_prep/prepare_okvqa_knowledge.py:19` 的 SIZE 一致。
- HF 仓库均存在：Qwen/Qwen3-VL-8B-Instruct、Qwen/Qwen3-VL-2B-Instruct、OpenGVLab/InternVL3_5-8B-HF、BAAI/bge-large-en-v1.5（`d4aa6901…` 存在）、openai/clip-vit-large-patch14、BAAI/bge-reranker-v2-m3、MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli（`b3546ea6…` 存在）、facebook/bart-large-cnn。
- NLI 清单 `src/omniaware/eval/citation/checker_asset_manifest.json` 中 `model.safetensors`、`spm.model` 的 sha256 与 HF LFS 哈希一致。
- `third_party/external_repos.json` 中 11 个上游 commit 在 GitHub API 上全部存在。
- README 中出现的 argparse 参数（run、launch、paper_runs、score_official、citation attribution、universalrag export/filter-train、prepare_webqa_open、prepare_okvqa、okvqa_dense、export_rankings、prepare_frames、routing_metrics、download_models、fetch_external）与代码一致，`--model 8b`、`--route-rule query --query-routes` 等取值也对得上。
- 路径与键的一致性：WebQA 产物（`text_documents.jsonl`、`image_corpus.jsonl`、`faiss_clip_full/plan.json`、`dense_top5_full/webqa_<split>/{bge,clip}/rankings.jsonl`）与 `src/omniaware/data.py:90-164` 的读取一致；`candidates_full.jsonl` 在 `okvqa_dense.py`、`data.py:201/208` 与 `check_assets.py:32-34` 一致；UniversalRAG 的 rankings 目录名（`hybridqa`、`videorag_synth`）在 `export_rankings.py:48-49`、`data.py:272` 与 `check_assets.py:37-38` 一致；`frame_key` 在 `prepare_frames.py:67` 与 `data.py:308` 一致。
- 查询编码不使用答案：`prepare_webqa_open.py:175`（只取 Q）、`okvqa_dense.py:121`（只取 question）。
- `filter-train` 先读后写，同一文件路径安全（`src/omniaware/baselines/universalrag.py:94-98`）；索引格式（`webqa_<32hex>`、`hybridqa_<id>`）与上游 `dataset/query/*.json` 一致（已读取核对）。
- 校准与评测划分不相交（`src/omniaware/data.py:357-372`）。
- `gold` 只用于记录，不进入提示词（`src/omniaware/run.py:573-579`）。
- 文本与标题重复已有保护（`src/omniaware/pipeline.py:64`，`title != text` 判断），GS 无标题段落不会被重复写入提示词。
- `scripts/launch.py:80-84`：任一分片失败则中止合并，合并前核对题数。
- BGE 修订号与查询指令在 `scripts/setup/download_models.py:32`、`src/omniaware/rerank.py:66-69`、`src/omniaware/relevance.py:21` 一致。

---

## 1. 问题清单（按严重程度排序）

### F01 严重：OK-VQA 的 GS 语料包含评测题的检索结果，且名称 "GS-112K" 与所用文件（168,306 条）不符
- 位置：`README.md:135`（"下载完整测试语料 CSV（168,306 条）"）、`README.md:140`（`--n-passages 168306`）、`README.md:128`（"val 5,046 题"）、`src/omniaware/data.py:208`、`src/omniaware/okvqa_dense.py:28`、`paper/sec_experiments.tex:8`（"GS-112K Google Search corpus … (168,306 sentences)"）。
- 证据：上游仓库 README（`https://raw.githubusercontent.com/luomancs/retriever_reader_for_okvqa/master/README.md`，Corpus 一节）写明：`okvqa_train_corpus` 为 "collected based on the training data"，size 112,724；`okvqa_full_corpus` 为 "collected based on the training data and testing data"，168,306。本仓库的评测集 OK-VQA val（5,046 题）即 OK-VQA 的测试集。因此 README 选用的 `okvqa_full_corpus` 含有评测题收集来的检索结果。名称里的 112K 恰好是只用训练题的版本的规模，但本仓库没有使用它。
- 影响：知识库由评测题的 Google 检索结果构成，OK-VQA 增益可能被高估；论文和代码中的 "GS-112K" 名称与实际语料不一致。是否用到答案字符串未能从仓库确认（推测：需查 Luo et al. EMNLP 2021 的收集方法）。
- 修复：二选一。（a）改用 `okvqa_train_corpus.csv`（112,724），并在论文中说明其与 OK-VQA test 题无交叠；（b）保留 168K 语料，但在论文和 README 明确"语料含评测题检索结果"，并把训练子集结果作为对照。无论哪种，都要统一名称。
- 推测：答案字符串是否参与了收集，**推测**，需要 Luo et al. 的收集说明。

### F02 严重：README 的 WebQA 开放池构建与旧池口径不同，且两者都写成 `text_documents.jsonl`
- 位置：`scripts/setup/prepare_webqa_open.py:42-44`（`text_key` = `_clean(title) + ": " + _clean(body)`）、`:57`（文档 ID 取该字符串的 sha256）、`:70-72`（写入 `text_documents.jsonl`）。旧池：`scripts/webqa_open_retrieval/docs/GLOBAL_POOL_REPORT.md:11`（"按正文精确相等去重……不做小写、空白或标点归一化"）、`scripts/webqa_open_retrieval/encode_global_text.py:62`（`document_input='exact text body; no title or instruction'`）。`src/omniaware/data.py:91` 的 docstring 引用的 "544,489 texts, 389,750 images" 来自旧池。
- 证据：在临时数据上运行 `build_pool`，同一正文、不同标题的两条事实得到两个文档：`['Paris: The tower is in Paris.', 'Eiffel: The tower is in Paris.']`（`verify_pool_and_skip.py` 输出）。按旧口径（精确正文）这两条应合并为一个文档。两条路径写出的文件名相同（`text_documents.jsonl`），但内容、文档 ID 和嵌入输入都不同。
- 影响：按 README 从零构建得到的 WebQA 池（文档数、ID、嵌入输入）与论文所依赖的池不同。两者都叫 `text_documents.jsonl`，若混用旧产物，加载器不会报错，结果会被静默污染。文档数的具体差异**推测**：未运行真实数据，无法给出数字。
- 修复：二选一。（a）把 README 的 `pool` 与旧池统一（精确正文、不做标题前缀，或两边都用标题前缀），然后用真实数据核对 544,489 / 389,750；（b）若论文以新口径为准，则删除 `data.py:91` 中的旧数字，并在论文附录注明池的构成。另外给不同口径的产物改不同文件名。

### F03 严重：`images` 步骤遇到一张截断的 JPEG 就整体中止，且不能自愈
- 位置：`scripts/setup/prepare_webqa_open.py:84-85`（`Image.open(...).convert("RGB").save(...)`，无异常处理，未设 `ImageFile.LOAD_TRUNCATED_IMAGES`）、`:102-104`（`ProcessPoolExecutor` 的 `map`，一个子进程异常会中断迭代）。
- 证据：在 `verify_images_e2e.py` 中，imgs.tsv 含一张截断图片时输出 `extract_images aborted: OSError Truncated File Read`，且 `images/` 下只写出了前一张（`files written: ['300.jpg']`）。PIL 版本 10.2.0，`LOAD_TRUNCATED_IMAGES = False`。旧文档 `scripts/webqa_open_retrieval/docs/README.md:43` 写明官方 imgs.tsv 中有 "Four truncated original JPEGs, including 30172755"，并按官方宽容加载方式处理。
- 影响：只要 imgs.tsv 含截断图，README 第 2 步就会失败；重跑时已存在的文件被跳过，但截断的那张仍会失败，因此无法靠重跑通过。
- 修复：与官方 loader 保持一致，设 `ImageFile.LOAD_TRUNCATED_IMAGES = True`，并把受影响 ID 写入清单、计数。或者逐张 try/except，失败记录到 `image_errors.jsonl` 并报告数量。
- 推测：官方 imgs.tsv 确实含截断图，**推测**（依据是旧文档的断言，本轮未下载 TSV 验证）。

### F04 严重：`extract_images` 一次性把所有待抽取图片的 base64 字符串塞进进程池，峰值内存可能很大
- 位置：`scripts/setup/prepare_webqa_open.py:95-104`（`jobs()` 生成器交给 `pool.map`）。
- 证据：Python 3.11.7 的 `concurrent.futures.Executor.map` 源码为 `fs = [self.submit(fn, *args) for args in zip(*iterables)]`，即在迭代开始前把所有参数全部提交。因此池中所有图片的 base64 串同时驻留在父进程内存中，直到各自被执行。
- 影响：峰值内存约等于 imgs.tsv 中属于池的那部分的总字节数（约 390k 张图）。imgs.tsv 的大小本轮未知，若是数十 GB，父进程会 OOM。
- 修复：按窗口分批提交（例如每批 2,000 行），或按行偏移读取。WebQA 官方仓库已提供 `imgs.lineidx`（其 README 第 23 行），可直接按偏移读取。
- 推测：imgs.tsv 大小与内存峰值，**推测**，需要实际测量文件大小。

### F05 严重：不在 imgs.tsv 中的池内图片被静默移出检索空间；打印的说明"loader 会计数"不成立
- 位置：`scripts/setup/prepare_webqa_open.py:110-112`（打印 "pool images are absent from imgs.tsv; the loader counts them"）、`:119`（编码只读 `assets.jsonl`）、`src/omniaware/data.py:127`、`:145-148`（只统计"被检索到但无文件"的图）。
- 证据：`verify_images_e2e.py` 的场景 A：池中 2 张图，imgs.tsv 只有 1 张。输出：`[images] 1 pool images are absent from imgs.tsv; the loader counts them`；CLIP 检索结果只含 `image:300`；加载器报告 `missing_images = 0`，图像候选只有 `i1`。缺失图从未被编码、从未被检索，所以加载器无从计数。
- 影响：图像候选池静默缩小，与论文中 "390k 图" 的口径不一致。旧池报告中明确提到原始 TSV 存在缺口（`scripts/webqa_open_retrieval/docs/README.md:8` 与 `fetch_global_images.py` 的状态名 `…ORIGINAL_TSV_GAPS_REMAIN`），因此按 README 构建的图像池会比旧池小。
- 修复：把缺失 ID 写入清单文件，并在运行配置里记录池的实际大小（索引向量数 vs 池大小）。删掉"加载器会计数"的错误说明。缺失图的补齐方式（镜像，见 F27）要在论文中写明。

### F06 严重：模型下载的"已完成"判断只看 `*.json` 是否存在，网络中断后的半成品会被跳过，且 check_assets 也放行
- 位置：`scripts/setup/download_models.py:47-48`（`if target.exists() and any(target.glob("*.json")): print("[skip]")`）、`:51`（`snapshot_download`，无完成标记）、`scripts/setup/check_assets.py:18-22`（模型只检查 `config.json` 是否存在）。README 第 92 行写"可重复执行，已下载的跳过"。
- 证据：在 `scratchpad\review\verify_pool_and_skip.py` 中，目录里只有 `config.json`、没有 `model.safetensors` 时，`hf_download` 打印 `[skip]`，`snapshot_download` 未被调用。
- 影响：一次网络中断后，重跑会永久跳过该模型（权重缺失），`check_assets` 仍报 OK，直到加载模型时才崩溃。`check_nli` 仅对 NLI 有补救。
- 修复：下载全部 `allow_patterns` 文件后写入 `.complete` 标记，以标记判断是否跳过。`check_assets` 检查权重文件（或 safetensors 索引与分片数）。

### F07 严重：引用指标（README 第 6 节 `omniaware.eval.citation.attribution`）依赖的 NLTK 资产没有任何生成步骤
- 位置：`src/omniaware/eval/citation/intake_answers.py:20-27`（读取 `cache/e5_citation_support/nltk_runtime`、断言 `nltk.__version__ == '3.9.2'`、读取 `sentence_splitter_assets.json`、设置 `nltk_data` 路径）；`tests/test_citation_intake.py:12-13`（资产缺失时跳过）；`src/omniaware/eval/citation/attribution.py:120`（`hashlib.file_digest`，要求 Python 3.11+）。README 第 256 行调用了该模块，但第 3 节、第 4 节和 `check_assets` 都没有提到 Punkt/NLTK。
- 证据：全仓库 grep `sentence_splitter_assets`、`nltk_runtime`、`nltk_data`，只有消费者（`intake_answers.py`、`test_citation_intake.py`），没有生成者。`docs/citation_support_protocol.md:43` 声称 "英文 Punkt 数据固定在 … revision `550b6625…`，仅提取英文四个资源文件并冻结 SHA256"，但仓库中找不到实现这一步的脚本。`pytest -rs` 的跳过理由是 "pinned NLTK Punkt assets not under the asset root"。
- 影响：Faith/AP/Inv 指标在新服务器上无法复现；pip 安装的 nltk 版本不是 3.9.2 时，断言直接失败。
- 修复：增加一个 setup 脚本（或并入 `download_models`），拉取指定 revision 的 nltk_data、写出 `sentence_splitter_assets.json`、准备 `nltk_runtime`，并在 `check_assets` 中列出。pyproject 中给 nltk 加显式版本。

### F08 严重：test 阶段在缺少或不匹配 `dev_decisions.json` 时静默使用硬编码的"冻结"选择
- 位置：`scripts/paper_runs.py:30-31`（`DEFAULTS` = `eab:g1.2:ans1:pos:gate`、`sum`、`last`、`okvqa_gs`）、`:35-40`（文件不存在，或 `d.get("model") != model`，都直接回落到 `DEFAULTS`）、`:260`（`--list` 打印 `frozen choices: …`）；`scripts/analysis/dev_select.py:123`（`okvqa_loader: okvqa or "okvqa_gs"`）。
- 证据：仓库中没有 `runs/` 目录（`ls runs` 报错），但 `python -m scripts.paper_runs --list` 输出 `frozen choices: {'eab_variant': 'eab:g1.2:ans1:pos:gate', 'u_combine': 'sum', 'attn_layer': 'last', 'okvqa_loader': 'okvqa_gs'}`，即在任何 dev 结果存在之前就"冻结"了。
- 影响：（a）test 阶段可能在没有 dev 决策的情况下跑，选择来源不可追溯；（b）`--model internvl3.5-8b`（README 第 246–247 行）与 8B 的 dev 决策 `model` 字段不同，因此会静默回落到 `DEFAULTS`，而不是沿用 8B 的冻结结果；（c）`--list` 把默认值显示为"冻结"，具有误导性。
- 修复：文件缺失或模型不匹配时直接报错；`DEFAULTS` 删除或只用于 dev 网格；`--list` 显示 "not frozen"。InternVL 迁移是否应沿用 8B 的决策，需要在 `docs/DEV_DECISIONS.md` 中明确。

### F09 中等：dev 决策规则文件的日期与提交时间不一致，"在看到 dev 结果前写定"无法核实
- 位置：`docs/DEV_DECISIONS.md:1`（标题写"2026-10-11，在看到任何 dev 结果之前写定"）。
- 证据：今天是 2026-10-10。git 中该文件最近的提交 `cd48ab8` 时间为 `2026-10-10 10:59:26 +0800`，提交信息为 "…pre-registered dev selection"，与代码改动（EAB-P 默认值、GS-112K 语料）同一次提交。仓库中没有 dev 结果，因此无法从仓库验证写定时点早于 dev 结果。
- 影响：预注册的可信度说明存在日期矛盾。规则 3 的 OK-VQA 选择（见 F10）也依赖于未构建的语料。
- 修复：修正日期；在附录写明写定时点与 commit hash；`dev_decisions.json` 中记录规则文件的 hash 与 dev 运行的 hash。

### F10 严重：OK-VQA 的 dev 语料选择（3 选 1）无法按 README 执行，实际退化为默认值
- 位置：`README.md:143`（"没有构建时，paper_runs 会跳过它们并给出提示"）、`scripts/paper_runs.py`（`--list` 输出 `skipped, candidates not built: okvqa, okvqa_dense`）、`scripts/analysis/dev_select.py:123`（默认回落 `okvqa_gs`）、`src/omniaware/okvqa_dense.py:27`（DPR 维基语料路径）、`scripts/data_prep/prepare_okvqa_knowledge.py:15-22`（需要 `cache/emnlp_source_20261004/self-aware-mrag-main/FlashRAG/tools/build_corpus_wiki_only_3m.py`，该文件不在仓库中；`ROOT` 依赖环境变量且无默认值）；`scripts/data_prep/prepare_okvqa_bm25_candidates.py`（BM25 候选，README 未提及）。论文附录 `paper/sec_appendix.tex:13` 写明 dev 网格包含三种语料。
- 影响：规则 3（"选 static−direct 差值最大的语料"）只能在 GS 一个选项上执行，选择是平凡的；`okvqa` 与 `okvqa_dense` 两个 test 对照臂也无法运行。论文中的 OK-VQA 语料选择无法从仓库复现。
- 修复：补充 DPR psgs_w100 的下载与 first-3M 构建步骤（URL 已核实可达，见第 0 节），或在论文中删除三选一的说法，改为只报告 GS。

### F11 严重：论文说 HybridQA/VideoRAG 用的是 UniversalRAG 套件的统一语料，实际构建的是缩小后的子集
- 位置：`paper/sec_experiments.tex:8`（"we use the query sets and unified corpora of the UniversalRAG suite"）；`README.md:157`（"build_subset 构建的候选池比完整套件小"）；`scripts/universalrag_retrieval/build_subset.py:3-15`（docstring 明确说只用 HybridQA 与 VideoRAG 两部分，完整套件的 `0_dataset.sh` 会下载十个数据集）。
- 影响：论文对语料的描述与实际实现不一致。UniversalRAG 基线与所有对照臂共享的是这个子集，因此结果是在更小的候选池上得到的。README 要求在论文中注明，但论文正文目前没有注明。
- 修复：要么用 `0_dataset.sh` 构建完整语料并重跑，要么在论文 sec_experiments 中改写为"子集"，并在附录说明候选池大小。

### F12 中等：UniversalRAG 路由器训练默认开启 wandb，README 的命令在无登录的服务器上会卡住或失败
- 位置：`external/UniversalRAG/route/train/train_qwen3_vl.py:298`（`--use-wandb` 默认 True）、`:305-306`（`wandb.init`）、`:391`（`report_to=["wandb"]`）；`README.md:182-183`（未传 `--no-use-wandb`）。
- 证据：`argparse.BooleanOptionalAction` 的默认值为 True，README 命令未覆盖。
- 修复：README 命令加 `--no-use-wandb`，或说明需要先 `wandb login`。

### F13 中等：GS-112K CSV 的列名为猜测，且 `--n-passages 168306` 写死；任何空行都会导致搜索失败
- 位置：`src/omniaware/okvqa_dense.py:45`（`field = "text" if "text" in reader.fieldnames else reader.fieldnames[-1]`）、`:47-49`（跳过空行，计数随之变化）、`:88`（`emb_{n}` 以实际行数命名）、`:118`（搜索时以 `n_passages` 打开 `emb_{n_passages}.f16`）；`README.md:140`（`--n-passages 168306`）；`tests/test_okvqa_gs.py:8`（测试假设表头为 `id,text`）。
- 证据：DPR 的 `CsvCtxSrc` 默认按位置读取：`text_col=1`、`title_col=2`（`external/DPR/dpr/data/retriever_data.py:253-262`）。若 GS CSV 的表头不含 `text`，回退取最后一列，在 DPR 布局下这是 title。CSV 的实际表头本轮无法核实（需要 Google Drive 下载）。
- 影响：（a）列名不符时，静默地把标题当作段落；（b）只要存在空行，`n` 就不等于 168306，`search` 会因找不到 `emb_168306.f16` 而失败。
- 修复：按位置读取并打印前几行与平均长度做校验；`--n-passages` 从 `corpus.jsonl` 行数读取，而非手填。
- 推测：列布局和表头，**推测**（需要真实文件）。

### F14 中等：多处"缺失即静默丢弃"，违反 CONVENTIONS §2"数据缺失必须计数并报告"
- 位置：`src/omniaware/data.py:203`、`:209`（`[e for e in examples if e["candidates"]]`，无计数）；`src/omniaware/data.py:244-253`（`_video_script` 找不到字幕时返回 `""`，无计数）；`scripts/universalrag_retrieval/prepare_frames.py:32-44`（循环中某个时间点没取到帧时，静默跳过，仍写出 `frames.json`，meta 记录的帧数少于 `--frames`）、`:23-24`（`frames.json` 存在即直接返回，残缺结果永远不会被重试）；`src/omniaware/data.py:310`（只统计 frames.json 不存在的情况，不统计帧数不足的情况）。
- 影响：OK-VQA 的 dense 问题会因没有候选而被整题删掉，删掉的题数不报告，不同对照臂的样本集可能不一致；VideoRAG 的字幕缺失会让视频只剩画面，文本信号被悄悄削弱。
- 修复：过滤处计数并打印；`prepare_frames` 帧数不足时报错或记录 `short=True`，并允许重跑补齐；字幕缺失计数写入运行日志。

### F15 中等：WebQA 开放池的图像与索引元数据写入绝对路径，移动资产目录后加载器崩溃；旧路径默认值会静默切换
- 位置：`scripts/setup/prepare_webqa_open.py:105-109`（`assets.jsonl` 与 `plan.json` 写入 `str(img_dir)` 绝对路径）；`src/omniaware/data.py:81-87`（`_asset_path` 只映射旧主机标记 `experiments/eab_first_token/`）、`:124`（打开 `plan.json` 指定目录下的 `assets.jsonl`，目录不存在时 FileNotFoundError）；`src/omniaware/paths.py:41-43`（若 `ASSET_ROOT/runs/phase1_dense_retrieval_codex_20261006` 存在，`WEBQA_OPEN` 静默指向它）。
- 影响：把资产复制到新挂载点后，加载器直接崩溃；旧目录存在时，默认路径静默切换，运行结果来源不明确。
- 修复：`plan.json` 与 `assets.jsonl` 中的路径改为相对 `OUT`，加载时拼接；去掉 `paths.py` 中基于旧目录存在与否的隐式切换，改为只读环境变量。

### F16 中等：WebQA"开放检索"的池由数据集自身的标注事实构成，每道评测题的 gold 都在池中；docstring 的"full public pool"说法偏大
- 位置：`scripts/setup/prepare_webqa_open.py:49-69`（池包含全部题的 `txt_posFacts`/`txt_negFacts`/`img_posFacts`/`img_negFacts`，其中包括 val 评测题的 gold）；`src/omniaware/data.py:91`（"full public pool"）；`scripts/webqa_open_retrieval/docs/README.md:1`（"newly reconstructed public WebQA corpus protocol"）。
- 证据：`run.py:573-579` 只把 gold 用于记录（不影响输入），因此不存在"用 gold 选证据"的泄漏；问题在于池的定义。
- 影响：池的规模和构成与真实网页检索不同（每道评测题的正例都在池中，召回的上界是 100%）。"开放检索"的结论只在这个数据集内部的池上成立。
- 修复：docstring 与论文写明池的来源与构成；如需更接近开放检索，补充无标注的外部事实作为干扰项，并报告召回。
- 推测：WebQA 原论文的 open-retrieval 设定是否与此一致，**推测**，需对照原论文的实验设置。

### F17 中等：模型版本不固定，NLI 下载会拉取不需要的 ONNX 与 .bin 文件
- 位置：`scripts/setup/download_models.py:29`（qwen8b，revision None）、`:31`（internvl，None）、`:34`（clip，None）、`:36`（reranker，None）、`:24`（`BART_CNN_REVISION = "main"`）、`:38`（NLI 的 `allow_patterns` 为 None）；`README.md:101`（只对 bge 写了"固定 revision"）。
- 证据：HF API（2026-10-10）：Qwen3-VL-8B-Instruct `main = 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；Qwen3-VL-2B-Instruct `89644892e4d85e24eaac8bacfd4f463576704203`；InternVL3_5-8B-HF `741a7d03020411e666c6109218ab71e08151ef86`；clip-vit-large-patch14 `32bd64288804d66eefd0ccbe215aa642df71cc41`；bge-reranker-v2-m3 `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`；bart-large-cnn `37f520fa929c961707657b28798b30c003dd100b`。NLI 仓库含 `onnx/model.onnx`（1.74 GB）、`onnx/model_quantized.onnx`（0.64 GB）、`pytorch_model.bin`（0.87 GB），全部会被下载，而代码只校验 manifest 中的 6 个文件。`facebook/bart-large-cnn` 没有 `tokenizer_config.json`，allow 列表中的该项不匹配（无害）。
- 修复：把上述 revision 写入 `download_models.py`，并同步 README；NLI 的 `allow_patterns` 设为 manifest 列出的文件。

### F18 中等：依赖声明不完整，且与代码实际需要的版本有出入
- 位置：`pyproject.toml:5`（`requires-python >=3.10`）、`:9`（`transformers>=4.57`，无上界）；`scripts/setup/download_models.py:60`、`src/omniaware/eval/citation/attribution.py:120`（`hashlib.file_digest`，需 3.11+）；`src/omniaware/eval/citation/intake_answers.py:23`（断言 nltk 3.9.2，但 nltk 未在 pyproject 中声明）；`scripts/webqa_open_retrieval/search_dense_queries.py:14`、`build_dense_index.py:112`（`faiss`，未声明，但不在 README 路径上）；`scripts/universalrag_retrieval/export_rankings.py:24`（`universalrag` 包，靠 `:47` 的 `sys.path` 注入 `external/UniversalRAG/src` 导入，可用）。
- 证据：Qwen3-VL-8B-Instruct 的 config 记录 `transformers_version 4.57.0.dev0`，支持 4.57 下限；InternVL3_5-8B-HF 的 config 记录 `4.55.0`，因此 `README.md:54` 中"InternVL3.5-HF 需要 4.57"没有证据支持。本仓库测试只在 transformers 5.5.3 上运行过（沙箱环境），没有任何 4.57 的测试。
- 修复：区分 Python 3.11（或把 `file_digest` 改为兼容实现）；nltk 与 faiss 按需声明为 extra；给 transformers 设定测试过的范围（例如 `>=4.57,<6`），并在 CI 或文档中注明测试版本。
- 推测：transformers 5.x 下 `qwen_eab.py` 的行为与 4.57 一致，**推测**，未测试。

### F19 中等：VideoRAG 下载不强制 mp4 容器，而后续脚本只匹配 `.mp4`，非 mp4 的视频会被当成缺失
- 位置：`external/UniversalRAG/dataset/videorag/download_video.py:32`（`'format': 'best[vcodec!=av01][height<=480]'`，无 `merge_output_format`，输出为 `{video_id}.%(ext)s`）；`scripts/universalrag_retrieval/build_subset.py:56-58`、`:15`（缺失视频写入 `videorag_missing.txt`，并解释为"YouTube 已下线"）；`scripts/universalrag_retrieval/prepare_frames.py:5`、`:65`（假定 `videos/<id>.mp4`）。
- 影响：若 yt-dlp 为某些视频选择了 webm 或 mkv，这些视频会被记为缺失，且原因描述与实际不符。
- 修复：下载时强制 mp4（`merge_output_format: 'mp4'`），或 `build_subset` 按 glob 匹配并统计扩展名。
- 推测：实际下载格式分布，**推测**，需要真实下载结果。

### F20 中等：WebQA 检索的显存峰值较高，且 `top` 大于库大小时直接报错
- 位置：`scripts/setup/prepare_webqa_open.py:186-188`（`chunk=4096` 查询 × 全部文档，`(q @ bank.T)` 为 fp16，再 `.float()`）；`:188` 的 `topk(top)` 在库小于 top 时报 `RuntimeError: selected index k out of range`（`verify_images_e2e.py` 场景中复现）。
- 证据：对 N = 544,489 的文本库，fp16 中间矩阵约 4.5 GB，float32 副本约 8.9 GB，峰值约 13 GB，再加上库本身约 1.1 GB。16 GB 显卡很可能 OOM。
- 修复：降低 `chunk`（如 512），或直接复用 `okvqa_dense.py:126-133` 的分块合并 top-k 逻辑。
- 推测：16 GB 显卡是否 OOM，**推测**（未在 GPU 上运行）。

### F21 中等：大文件下载没有超时、重试或断点续传
- 位置：`scripts/setup/prepare_okvqa.py:38`（`urllib.request.urlretrieve`，无超时，无重试）、`:65-67`（COCO val2014.zip 约 6.2 GB，`extractall`）；`fetch` 在 `.part` 残留时不续传。
- 影响：6 GB 下载中断后只能从头开始。
- 修复：改用 `requests` 带 `Range` 续传、超时与重试，并校验 Content-Length 或哈希。

### F22 中等：旧版池构建脚本无法在仓库内运行，但它们是"池数字"的唯一出处
- 位置：`scripts/webqa_open_retrieval/build_global_webqa_pool.py:17`（导入 `ASSET_ROOT/prepare_eab_recovery_confirmation.py`，仓库中不存在）、`:70`（硬编码 `/Users/futaoran/Downloads/WWW2027/experiments/eab_first_token`）、`:113`（`data/WebQA_data_first_release.7z`，且依赖 `bsdtar`）；`scripts/data_prep/*.py`（`ROOT = Path(os.environ['OMNIAWARE_ASSETS'] or os.environ['ERC_ASSET_ROOT'])`，两者都未设置时 KeyError）。
- 证据：`find` 在仓库中找不到 `prepare_eab_recovery_confirmation*`。
- 影响：`GLOBAL_POOL_REPORT.md` 中的 544,489 / 389,750 等数字无法从仓库重算，也无法与 F02 的新池对比。
- 修复：要么删除这些脚本并把数字来源改为"外部交付物"（附 hash），要么修成可运行。

### F23 中等：图像池的补齐来源是第三方镜像，不是官方 TSV
- 位置：`scripts/data_prep/fetch_webqa_parquet_images.py:22-23`（HF 数据集 `MrZilinXiao/mbeir_test_webqa_task2`，revision `bffbce53…`，HF API 核实存在）；`scripts/webqa_open_retrieval/fetch_global_images.py:1`（"from the pinned, ID-verified public mirror"）；`scripts/webqa_open_retrieval/docs/README.md:8`（"complete fetch does not imply all original TSV pixels were recovered"）。
- 影响：论文池中的部分图像可能来自镜像，README 的 `images` 步骤无法复现这一部分（与 F05 同源）。镜像图像与官方 TSV 字节不一定一致，文档也承认"未做 TSV 字节验证"。
- 修复：若采用镜像补齐，在 README 中给出完整步骤与 hash，并在论文中说明图像来源。

### F24 轻微：CLI 与帮助信息的小问题
- 位置：`scripts/setup/check_assets.py:46-67`（无 argparse，`--help` 直接执行检查，实测）；`scripts/launch.py:89-90`（`--help` 报用法后以码 1 退出，实测）；`scripts/analysis/routing_metrics.py:47`（在 GBK 控制台执行 `--help` 抛 `UnicodeEncodeError`，实测；Windows 控制台特有）。
- 修复：为 `check_assets` 加 argparse；`launch` 支持 `--help`；帮助输出用 UTF-8 或 `errors="replace"`。

### F25 轻微：文档与清单的漂移
- `README.md:11`：引用了仓库外的 `../Omni-Aware_全面审查报告_20261010.md`，对只拿到仓库的读者是悬空引用。
- `README.md:75`：表格写 `OMNIAWARE_ASSETS` 默认 `./assets`，实际 `src/omniaware/paths.py:17` 基于 `REPO_ROOT`，与当前工作目录无关。
- `third_party/README.md:13`：写 M2-RAG 源码在 `cache/official_m2rag/`，而 `scripts/setup/check_assets.py:40` 检查的是 `external/M2RAG/src`。
- `src/omniaware/eval/citation/checker_asset_manifest.json:6`：含绝对路径 `/Users/futaoran/...`。
- `MIGRATION_SOURCES.json`：`files` 与 `files_round2` 共 93 个目标，其中 36 个在仓库中不存在（例如 `src/omniaware/answer_format.py`、`table_render.py`、`eval/mmqa_official.py`）。存在的 57 个中，39 个的 `sha256` 与当前文件不一致；字段含义未说明，很可能是源文件的哈希，本仓库无法核对。
- `README.md:57`："约 10 秒"，沙箱实测约 38 秒（不是缺陷，但说明依据不足）。

### F26 轻微：WebQA 的 distractor 加载器依赖一个没有生成步骤的图像目录
- 位置：`src/omniaware/paths.py:31`（`WEBQA_IMAGES = CACHE/"webqa_full4966_images"/"images"`）、`src/omniaware/data.py:56`、`:76`；`src/omniaware/run.py:43` 把 `webqa_img`、`webqa_full` 列为可选数据集。
- 证据：全仓库只有 `scripts/webqa_open_retrieval/fetch_global_images.py:14`（旧脚本）引用该目录。`paper_runs.py` 未使用这两个数据集。缺失的事实会被丢弃并打印总数（`data.py:57-58`、`:74-76`），不是完全静默。
- 修复：若论文不用，删除这两个 loader；若要用，把生成步骤写进 README。

### F27 轻微：中断后的文件写入不是原子的，已存在即跳过
- 位置：`scripts/setup/prepare_webqa_open.py:84-85`（图片直接写入目标文件，`exists()` 即跳过，`:84`）；`:141`（`np.save` 直接写入）；`:128`（`npy.exists()` 即视为完成）。仓库内 `src/omniaware/relevance.py:192-194` 已有原子写入的写法，可以复用。
- 影响：进程在写入中途被杀，会留下截断的 `.npy` 或 `.jpg`，之后被当作完成；多数情况下 `_load` 会报错（响亮），但不保证。
- 修复：先写 `.tmp` 再 `os.replace`。

---

## 2. 按 README 从零跑：最可能卡住的地方（按步骤顺序）

1. **第 1 节 环境**：`pip install -e ".[all]"` 在 Linux 上一般可行。`python scripts/fetch_external.py` 需要能访问 GitHub，上游 commit 都存在（已核实）。UniversalRAG 环境要求 Python 3.12（其 `pyproject.toml` 的 `requires-python >=3.12`），并依赖 `flash-attn`、`deepspeed`，编译失败的风险较高（推测）。
2. **第 2 节 / check_assets**：首次运行会列出大量 MISSING，属正常。但它不能检测截断的权重（F06）。
3. **第 3 节 download_models**：约 45 GB，中断后重跑会跳过半成品（F06）。NLI 会多下约 3 GB 不需要的 ONNX/.bin（F17）。未固定 revision（F17）。
4. **第 4.1 节 WebQA**：
   - 需要本机有 `7z`（README 第 113 行），但 README 没有列出它。
   - `WebQA_train_val.json` 需要手动复制到 `$OMNIAWARE_ASSETS/data/`（README 第 114 行，`search` 直接读该路径，而不是 `--webqa-dir`）。
   - `images` 步骤遇到截断 JPEG 就中止（F03）；全部图片同时驻留内存（F04）；缺失图片静默缩小索引（F05）。
   - 即使一切成功，产出的池也很可能与旧池口径不同（F02）。
   - `encode` 在 CPU 上处理 54 万条文本，耗时很长（推测）。
5. **第 4.2 节 OK-VQA**：`prepare_okvqa --download` 需要约 6.5 GB，无续传（F21）。GS CSV 需要人工从 Google Drive 下载（README 未给出确切文件链接，无法自动化）。列名猜测（F13）。`--n-passages 168306` 写死（F13）。语料本身的评测题交叠问题（F01）。
6. **第 4.3 节 HybridQA / VideoRAG**：`get_videorag.sh` 通过 yt-dlp 下载约 8,620 条 YouTube 视频，需要 cookies 的可能性较大，大部分可能失败（推测）；mp4 匹配问题（F19）。`build_subset` 的候选池比套件小（F11）。`export_rankings.py` 必须在 UniversalRAG 环境中、且特征已由 `1_preprocess.sh` 生成，这一步依赖最多。
7. **第 4.4 节 UniversalRAG 路由器**：`train_qwen3_vl.py` 默认开 wandb，无登录会卡住（F12）。`prep_train_data.py` 从包括评测集在内的 3 个查询文件里抽样，再由 `filter-train` 按 index 剔除评测题；index 格式已核实一致，剔除逻辑正确（已验证）。
8. **第 5.1 节**：单条运行本身不在本轮审查范围内。
9. **第 5.2 节 dev 阶段**：OK-VQA 的 `okvqa` 与 `okvqa_dense` 候选未构建，dev 决策退化为 GS（F10）。`dev_select` 的输出与论文的 3 选 1 规则不一致（F10）。
10. **第 5.2 节 test 阶段**：若 `dev_decisions.json` 缺失或模型名不一致，`paper_runs` 静默使用硬编码选择（F08）。`--model internvl3.5-8b` 迁移不读取 8B 的冻结结果（F08）。
11. **第 6 节 评分与分析**：`score_official --bart` 需要 BARTScore 资产（`download_models --only bartscore` 通过 gdown 拉取 Google Drive，`python -m gdown` 的入口在上游存在，已核实）。`citation.attribution` 缺少 NLTK 资产，**会失败**（F07）。

---

## 3. 没来得及深入的方向

- `src/omniaware/run.py` 的分片续跑逻辑、`merge_shards.py` 的核对逻辑（本轮只看了 launch 与 README 的接口）。
- `scripts/analysis/paper_analyses.py`、`dev_select.py` 的统计细节，以及 `power_design.py`。
- `external/UniversalRAG/script/1_preprocess.sh` 生成的特征与 `export_rankings.py` 中 `source_id` 的类型（字符串还是整数）是否与 parquet 索引一致（`data.py:284-298` 的 `corpora[m][sid]`）。推测：若索引为整数而 sid 为字符串，会在 KeyError 处崩溃（响亮），但未验证。
- `build_dense_index.py` / `search_dense_queries.py`（依赖 faiss，本轮未运行）。
- 真实数据验证：GS CSV 表头、WebQA imgs.tsv 大小与截断图数量、池的实际计数（544,489 / 389,750）、yt-dlp 下载的容器格式分布。这些都需要下载，本轮按约束未做。
- `intake_answers.py` 在 nltk 3.9.2 下的行为（沙箱只有 3.8.1，未测试）。
- `prepare_webqa_open.py` 的 `encode` 在多 GPU 下的分片一致性（只看了代码，未在 GPU 上运行）。
- `scripts/data_prep/fetch_webqa_parquet_images.py`（366 行）的完整校验逻辑。

---

## 4. 对本轮的说明

- 沙箱默认的 `python` 是 `D:\Anaconda\python.exe`（不是 README 要求的环境）。`omniaware.*` 模块在没有 `pip install -e` 时无法被 `-m` 找到，因此 `--help` 检查时使用了 `PYTHONPATH=src`。`scripts.*` 模块从仓库根目录运行即可。
- 本轮没有修改仓库中的任何文件，也没有 push 或改 git 历史。`git status` 为干净状态。
- 所有临时脚本与输出都放在 `scratchpad\review\`。

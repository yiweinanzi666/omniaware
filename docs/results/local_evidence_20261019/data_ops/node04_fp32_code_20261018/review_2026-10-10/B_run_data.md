# 审查报告 B：主控制流、数据加载、校准、分片/续跑/缓存

仓库：`C:\Users\22688\Desktop\www_code\omniaware`（只读审查，未修改仓库任何文件；pytest 运行只更新了 gitignore 内的 `__pycache__` 与 `.pytest_cache`）。
验证脚本：`scratchpad/review/exp1_*.py` 至 `exp8_*.py`，全部使用临时目录或内存数据，未写入仓库。

## 一、摘要（每条一行）

1. [致命] VideoRAG-Synth 在默认参数下评测集为 0 题、校准吞掉全部 374 题，且不报错 — `src/omniaware/run.py:242-245`；`src/omniaware/data.py:369-371`；`scripts/paper_runs.py:112-121`
2. [严重] heldout 划分的 test 集随 `--n-calib` 改变；n_calib=100 的消融 test 含全部 256 道 dev 调参题，n_calib=1000 丢掉 500 道主测试题 — `src/omniaware/data.py:369-371`；`scripts/paper_runs.py:145-146`
3. [严重] relevance / CE / U 校准缓存只按 qid 或数据集名命中，候选文本或候选文件变化时静默复用旧分数 — `src/omniaware/run.py:25-26, 416-417`；`src/omniaware/rerank.py:102-105`；`src/omniaware/data.py:201-202`
4. [严重] `scripts/launch.py` 见到同名 `<dataset>.jsonl`（单进程半成品）即判定"已合并"并跳过，随后被评分 — `scripts/launch.py:57-59`；`src/omniaware/run.py:210-211`
5. [严重] OK-VQA 稠密候选的检索编码与 relevance 重打分前缀不一致，且检索得分被丢弃（推测影响大小）— `src/omniaware/okvqa_dense.py:56-57, 102, 123`；`src/omniaware/relevance.py:94, 98`；`src/omniaware/data.py:186-189`
6. [中等] 续跑遇到半写入行直接 JSONDecodeError 崩溃，且半行残留；ucal/SeaKR 缓存非原子写，中断后下次启动即崩溃 — `src/omniaware/run.py:503-504, 439, 486, 418, 481`
7. [中等] `merge_shards` 默认不检查覆盖（`--expected` 默认 None），文档称已检查 — `src/omniaware/merge_shards.py:6-8, 48-49, 67-68`
8. [中等] `runs/dev_decisions.json` 缺失时 test 阶段静默回退到默认设置；空 dev 数组使 `dev_select` 得到 NaN 并被 `max` 任意选择 — `scripts/paper_runs.py:34-40`；`scripts/analysis/dev_select.py:46-54, 70`
9. [中等] 文档与代码矛盾：默认 EAB 变体（PROTOCOL 写 ans2，代码为 ans1）；冷启动门控（PROTOCOL 写 U_pre，代码用 probe）；dev/test 互斥"运行时断言"只覆盖 calib 与 eval — `docs/PROTOCOL.md:17, 33, 35`；`src/omniaware/run.py:155, 559-560, 662`
10. [中等] "test 只跑一次"与实际：β×τ_b 网格、noprobe、噪声、校准大小、排列等消融都在 `--split test` 上运行；干预的反事实实体池取自评测题的 gold — `scripts/paper_runs.py:137-153`；`src/omniaware/run.py:498`；`src/omniaware/interventions.py:79-85`
11. [中等] 框架判定（framing）用同一 dev 上 6 种 EAB 形式取最大后的 bootstrap 区间，未做选择偏差校正（推测对结论有影响）— `scripts/analysis/dev_select.py:70, 114-117`；`docs/DEV_DECISIONS.md:20, 24, 26-32`
12. [中等] 配置一致性检查不完整：仅在输出已存在时比较参数；续跑时校准统计变化不被发现；ucal 缓存只按模型名 — `src/omniaware/run.py:212-222, 505-519, 416-417`
13. [中等] 分片下若跳过 `--calibrate-only`，每个分片都完整重算校准并并发写同一 ucal 文件 — `src/omniaware/run.py:420-439`
14. [中等] `Calibrator.ecdf` / `relevance` 对未见通道静默返回 0.5，影响 EAB 的 r_j — `src/omniaware/relevance.py:146-147, 163-165`；`src/omniaware/run.py:284-285`
15. [中等] 候选不足时中途抛 ValueError（perm 要求恰好 5 块；路由模态缺失），续跑会在同一题再次崩溃（推测触发频率）— `src/omniaware/run.py:53-55, 78-79`；`src/omniaware/data.py:308-311`
16. [轻微] BM25 `okvqa` 保留无候选的题（静态臂无证据）；OK-VQA 与 UniversalRAG 的图像路径不检查存在性 — `src/omniaware/data.py:186-194, 192, 320`
17. [轻微] relevance / CE 缓存只在全部计算完后写一次，docstring 称"可续跑"；SeaKR 缓存写入非原子 — `src/omniaware/rerank.py:98, 119-122`；`src/omniaware/run.py:32-33, 486`
18. [轻微] `last_answer_attention` 的行定义与注释不一致；`max_new_tokens=1` 时静默为 None — `src/omniaware/qwen_eab.py:279-293`；`src/omniaware/qwen_eab.py:127`
19. [轻微] NaN 写入 JSON；协议不变量使用 `assert`（`python -O` 下失效）— `src/omniaware/analysis.py:26-30`；`src/omniaware/run.py:254`；`src/omniaware/okvqa_dense.py:120`
20. [轻微] 溯源：DEV_DECISIONS 标注 2026-10-11 "在看到任何 dev 结果之前写定"，而其 git 提交为 2026-10-10 10:59，晚于今天的日期 — `docs/DEV_DECISIONS.md:1`；`docs/PROTOCOL.md:15`；`git log cd48ab8`
21. [轻微] 代码细节：`select_calibrated` 硬编码 `[:5]`；`raw_scores` 中 `startswith("f")` 条件无意义；`rank_sources` 字段名 `bge_cosine` 实际存的是校准后的 r — `src/omniaware/rerank.py:52-53, 84, 132`；`src/omniaware/relevance.py:115`
22. [轻微] 测试缺口：`test_splits` 不测 n_calib 稳定性与空评测集；分片测试 monkeypatch 了 `load_relevance`，未覆盖缓存、续跑截断、launch 判定、merge 默认路径 — `tests/test_splits.py`；`tests/test_run_smoke.py:388-415`

---

## 二、问题详情

### 1. [致命] VideoRAG-Synth 在默认参数下评测集为空，校准吞掉全部题目

- 位置：`src/omniaware/run.py:242-245`（划分与切片）、`src/omniaware/data.py:369-371`（heldout 划分）、`scripts/paper_runs.py:23-24`（CALIB 为 heldout）、`scripts/paper_runs.py:112-121, 126-134`（主表与 UniversalRAG 阶段未传 `--n-calib`）、`scripts/paper_runs.py:81, 88`（dev 阶段包含 videorag_synth）。
- 证据：
  - `docs/PROTOCOL.md:13` 与论文 `paper/sec_experiments.tex:8` 都写 VideoRAG-Synth 为 374 题。
  - 划分逻辑为 `calib = order[:n_calib]`，`rest = order[n_calib:]`，`dev = rest[:n_dev]`，`test = rest[n_dev:]`。默认 `n_calib=500`、`n_dev=256`，因此 374 题时 calib=374、dev=0、test=0。
  - 实测（`scratchpad/review/exp6_videorag_split.py`）：total=374，n_calib=500 → calib=374, dev=0, test=0；n_calib=100 → dev=256, test=18；n_calib=150 → dev=224, test=0。
  - 端到端实测（`exp7_videorag_default_run.py`，用 374 条合成样本，调用与 paper_runs 相同的参数）：输出 `[data] ... calib 374 (heldout), test 0`，`[done] okvqa 0`，`rows written: 0`，配置里 `evaluation_questions: 0`，进程正常退出。
- 论文与协议的设计在此数据集上不可实现：`paper/sec_experiments.tex:16` 要求每个基准 500 条校准题，`:18` 要求 256 条 dev 题，二者与评测题互不相交。374 < 500 + 256，因此不存在合法划分。
- 影响：所有 VideoRAG-Synth 数字（主表、dev 选择、消融、干预、顺序）无法由当前默认流程产生；流程不报错，只会得到空文件或在评分阶段得到 NaN。若手动改为 `--n-calib 100`，test 只剩 18 题，统计功效不足，也不符合协议。
- 修复建议：
  1. 在 `split_examples` 或 `run.main` 中加守卫：`len(evals)==0` 或 `len(calib)+n_dev > 总数` 时直接 `ap.error`。
  2. 为 VideoRAG-Synth 预先登记划分（例如 calib 100、dev 64、test 210，或其他满足 `n_calib+n_dev+test=374` 的方案），写入 `paper_runs.py` 与 `PROTOCOL.md`。由于 `DEV_DECISIONS.md` 规定规则不得在看到结果后修改，修改需在该文件注明时间与原因，并在附录披露。
  3. 论文 `sec_experiments.tex:16, 18` 的数字（500 校准、256 dev）对 VideoRAG-Synth 必须改写。
- 是否推测：否。已用代码算术和端到端运行证实。

### 2. [严重] heldout 划分的 test 集随 `--n-calib` 变化（消融污染 test）

- 位置：`src/omniaware/data.py:369-371`；`src/omniaware/run.py:242, 245`；`scripts/paper_runs.py:145-146`（`--n-calib 100/1000`，默认 `--split test`）。
- 证据（`exp1_split_vs_ncalib.py`，5,046 题，seed 2027，dev 256）：
  - n_calib=100：test=4,690；n_calib=500：test=4,290；n_calib=1000：test=3,790。
  - n_calib=100 的 test 与主实验 dev 重叠 256 题（即全部调参题被用作消融的测试题）。
  - n_calib=1000 的 test 丢掉主实验 test 的 500 题。
  - dev 集也随 n_calib 改变（n_calib=100 与 500 的 dev 互不相同）。
- 影响：`v321_calib100` 与 `v321_calib1000` 在不同的题目集合上计算，与主实验不可直接比较；n_calib=100 的消融把 dev 调参题纳入测试集，违反 `PROTOCOL.md:18-19`。仅影响 OK-VQA（`okvqa_gs`）的校准大小消融，主实验 n_calib=500 不受影响。
- 修复建议：把 dev/test 固定为与 n_calib 无关的索引（例如先切出 dev/test，再从剩余题中取 calib），或把 test 的 qid 列表冻结到文件并在运行时校验；消融比较只在交集上进行。
- 是否推测：否。

### 3. [严重] 缓存只按 qid 或数据集名命中，候选变化时静默复用

- 位置：
  - `src/omniaware/run.py:25-26`：`todo = [e for e in examples if e["qid"] not in raw]`，已有 qid 直接复用，不比较候选文本。
  - `src/omniaware/run.py:231-232`：`data_tag` 只包含 `top` 与 `modalities`，不包含候选文件路径或哈希。
  - `src/omniaware/run.py:416-417`：ucal 缓存文件名不含候选、模型权重路径的指纹。
  - `src/omniaware/rerank.py:102-105`：CE 缓存按 qid 命中，不含模型路径或候选内容。
  - `src/omniaware/data.py:201-202`：`ERC_OKVQA_DENSE_CANDIDATES` 可用环境变量切换候选文件，不进入任何缓存键或 config。
  - `src/omniaware/data.py:101, 122`：WebQA 开放检索的排名目录（`OMNIAWARE_WEBQA_OPEN_DIR`）同样不进入缓存键。
- 证据（`exp2_relevance_cache.py`）：同一 qid、同一 cid `k1`，候选文本由 "short passage" 改为一段更长的文本后，`load_relevance` 直接返回旧分数 `1.0075`，且 BGE 没有被再次调用。只有新增的 cid（`k6`）才会以 `KeyError` 崩溃。
- 影响：候选文件重新生成、更换语料（wiki3m ↔ gs112k 之外的同名重建）或改变 env 路径时，结果会在没有任何警告的情况下使用旧的相关度，直接改变 EAB 的 r_j、Top-k 选择和 ECDF 校准。这与旧仓库"复用旧缓存"的问题相同。
- 修复建议：缓存条目保存候选指纹（cid、文本、来源路径、文件 sha256），命中前比较；ucal/CE/relevance 的文件名加入数据版本与模型权重指纹；把候选文件路径与哈希写入 config。
- 是否推测：否（代码与实测均已核实）。若候选文件在本项目中从未重建，实际风险较小，但缓存机制本身不安全。

### 4. [严重] launch 将半成品单进程输出判定为"已合并"

- 位置：`scripts/launch.py:57-59`（`if out.exists(): print("already merged"); return out`，在多 GPU 分支中，且不检查行数）；`src/omniaware/run.py:210-211`（单进程输出 `"{dataset}.jsonl"`，与合并结果同名）。
- 证据（`exp3_launch_partial.py`）：在同 tag 下放置一个只含 3 行的 `okvqa_gs.jsonl`，`launch(..., gpus=["0","1"])` 直接返回，0 次子进程调用，只打印 "already merged; nothing to do"。
- 场景：先用单进程（或 `--gpus 0` 的单槽路径，`launch.py:52-56` 直接运行原命令）跑某个 tag，中途被中断；之后用同一 tag 的多 GPU 命令重跑，会静默跳过；`scripts/paper_runs.py` 随后评分这个半成品文件。
- 影响：论文表格可能基于部分题目的结果计算，且无任何报错。
- 修复建议：判定"已合并"需同时满足（a）没有 `.shard*` 文件，（b）行数等于 `config["evaluation_questions"]`，（c）qid 唯一。否则报错退出。单进程输出也应使用单独文件名，或在 `run.main` 结束时写完成标记。
- 是否推测：否。

### 5. [严重] OK-VQA 稠密候选的检索编码与 relevance 重打分不一致

- 位置：
  - `src/omniaware/okvqa_dense.py:56-57`：passage 前缀 `"passage: "`，query 前缀 `"query: "`（E5 风格，注释说明与旧索引一致）。
  - `src/omniaware/okvqa_dense.py:102`：passage 编码时加 `PASSAGE_PREFIX`。
  - `src/omniaware/okvqa_dense.py:123`：query 编码时加 `QUERY_PREFIX`，但 `bge.encode(...)` 未传 `query=True`，因此不加 BGE 的查询指令。
  - `src/omniaware/relevance.py:21, 94`：relevance 的 query 使用 `BGE_QUERY_PREFIX`（`"Represent this sentence..."`）。
  - `src/omniaware/relevance.py:98`：候选文本编码时不加任何 passage 前缀。
  - `src/omniaware/data.py:186-189`：候选的 `bge_score` 被丢弃，只保留 rank 与文本。
- 证据：代码对照即可确认两套编码方式不同。候选集合（top-20）来自 FAISS 检索时的编码；运行时的 top-5 选择与 EAB 的 r_j 却来自另一套编码（query 指令不同，passage 无前缀）。论文 `sec_experiments.tex:16` 称 BGE-large 为同一检索器。
- 影响：OK-VQA 的 static 与 gated 结果中，"Top-5"并不是检索器给出的 Top-5；EAB 的相关度偏置也基于不同的分数。DEV 阶段选出的语料（`okvqa_dense` 与 `okvqa_gs`）可能因此被误判。影响大小为推测，需要实测。
- 修复建议：统一编码函数，检索与重打分共用同一前缀；或直接使用检索时的 `bge_score` 作为 r 的原始分数。
- 需要的证据（推测部分）：在 50 道题上分别用 search 的编码与 `relevance.raw_scores` 的编码计算 top-5 重合率，并报告 static 的分数差。
- 是否推测：代码不一致已核实；对结果的影响为推测。

### 6. [中等] 续跑遇到半写入行即崩溃；缓存非原子写

- 位置：`src/omniaware/run.py:503-504`（`done = {json.loads(l)["qid"] for l in out_path.open()}`）；`src/omniaware/run.py:525-527`（`open("a")` 直接追加，没有检查末尾是否有换行）。
- 证据（`exp4_resume_truncated.py`）：完整运行 5 行后，将最后一行截断到一半（无换行），再次运行同一命令，报错 `JSONDecodeError: Expecting value`，且文件仍保留半行。`launch.py` 的续跑路径同样会在每次重跑时复现此错误。
- 缓存文件的非原子写：
  - `src/omniaware/run.py:439`：ucal 每题都 `write_text` 全量 JSON。中断时文件被截断，下次启动 `run.py:418` 的 `json.loads` 直接崩溃。
  - `src/omniaware/run.py:486`：SeaKR 缓存同样非原子（`run.py:481` 读取时崩溃）。
  - `src/omniaware/okvqa_dense.py:136`：candidates 文件以 "w" 直接写入，中断后 `load_okvqa` 读到截断行。
  - 对照：`relevance.save_raw`（`relevance.py:192-194`）与 `rerank.ce_logits`（`rerank.py:120-122`）使用了 tmp + `os.replace`，是正确的。
- 影响：崩溃是显式的，不是静默错误，但需要人工删除半行或缓存，且续跑语义在文档中被描述为"安全"（`merge_shards.py:8`）。
- 修复建议：读取时忽略末尾无换行的行并截断，追加前检查换行；所有缓存改为 tmp + `os.replace`；ucal 按题累积并定期原子写，而非每题全量重写。
- 是否推测：否。

### 7. [中等] merge_shards 默认不检查覆盖率

- 位置：`src/omniaware/merge_shards.py:6-8`（文档称"checks that every question of the evaluation split was answered"）；`:48-49`（只有 `expected_questions is not None` 时才检查）；`:67-68`（`--expected` 默认 None）。
- 证据（`exp4_resume_truncated.py` 最后一段）：两个分片各缺若干题的情况下，`merge_shards.main([stem])` 正常生成 5 行结果，没有报错。
- 说明：`launch.py:84` 传入了 `evaluation_questions`，因此多 GPU 路径是安全的；风险在于手动合并或其他脚本调用。
- 修复建议：默认从 `{dataset}_config.json` 读取 `evaluation_questions`；读不到则拒绝合并。
- 是否推测：否。

### 8. [中等] 预注册决策文件缺失时静默回退；空 dev 导致 NaN 选择

- 位置：
  - `scripts/paper_runs.py:34-40`：`decisions()` 找不到 `runs/dev_decisions.json` 或模型不匹配时直接返回 DEFAULTS，`--run` 时不报错、不打印警告。
  - `scripts/analysis/dev_select.py:46-54`：`pooled_gain` 遇到空数组返回 NaN。
  - `scripts/analysis/dev_select.py:70`：`max(results, key=...)` 遇到 NaN 的结果不可靠。
- 证据：`exp5_fallbacks.py`，本仓库 `runs/dev_decisions.json` 不存在；`stages("8b","test")` 的主阶段命令使用 `eab:g1.2:ans1:pos:gate`。`exp8_dev_select_empty.py`：`pooled_gain` 在 VideoRAG 空数组下返回 `nan`，并伴随 RuntimeWarning。空 dev 与问题 1 相关。
- 影响：test 阶段可能在没有任何 dev 选择的情况下运行，违背"设置在 dev 上冻结"的协议；dev 选择可能因 NaN 而随机化。
- 修复建议：test 阶段在 `dev_decisions.json` 不存在或模型不一致时直接报错；在 config 与结果文件中记录冻结的选择；`dev_select` 在任何数据集样本数为 0 时报错。
- 是否推测：否。

### 9. [中等] 文档与代码矛盾（三处）

- (a) 默认 EAB 变体：`docs/PROTOCOL.md:33` 写默认为 `eab:g1.2:ans2:pos:gate`（"即论文 decode-only"）。代码 `src/omniaware/run.py:155` 默认为 `ans1`，`src/omniaware/pipeline.py:180` 的 `parse_variant` 默认也是 ans1，`docs/DEV_DECISIONS.md:13` 与 `paper/sec_appendix.tex:13` 都称 ans1 为论文默认，`scripts/paper_runs.py:30` 的 DEFAULTS 为 ans1。
- (b) 冷启动门控：`docs/PROTOCOL.md:35` 写"U_pre ≤ τ 时用 direct"。`src/omniaware/run.py:559-560` 实际使用 `u_gate = u_probe if 存在 else u_pre`，并以此判定 `retrieve`；`run.py:662` 的注释也写 U_pre。`docs/DEV_DECISIONS.md:16` 与 `scripts/paper_runs.py:27, 61-62` 对 WebQA、HybridQA、VideoRAG 使用探测轮。论文 `paper/main.tex:203` 与代码一致，协议文档与 `run.py:662` 注释过时。
- (c) 协议 §1（`docs/PROTOCOL.md:17`）称"三者互不相交，运行时会断言检查"。实际只有 `run.py:254` 断言 calib 与 eval 不相交；dev 与 test 的不相交只靠切片保证，未断言。
- 影响：读者依据协议文档理解门控与默认设置，会得出与实际运行不同的结论。
- 修复建议：修正三处文档与注释；在 `split_examples` 中断言 dev∩test=∅。
- 是否推测：否。

### 10. [中等] "test 只跑一次"与消融实际在 test 上运行

- 位置：`scripts/paper_runs.py:137-153`（`abl` 阶段：noise、relevance-norm、nocal、calib100/1000、β×τ_b 的 5×3 网格，`split` 默认为 test）；`:151-153`（noprobe）；`:172-185`（排列，使用 `--n-eval 1000`）；`src/omniaware/run.py:498`（`substitute_pool(evals, ...)`，干预的反事实实体取自评测题的 gold）。
- 证据：`docs/PROTOCOL.md:18-19` 规定"调参只看 dev；test 在设置冻结后只跑一次"。上述阶段在 test 上执行大量变体。
- 影响：这些消融本身可以作为事后敏感性分析报告，但必须标注为 post-hoc，且不能为主张的选择提供依据。反事实实体池取自评测题的 gold 答案，属于使用测试标签构造输入（不用于决策，但应在协议中声明）。
- 修复建议：在协议中明确"消融 = 事后分析，不用于任何设置选择"；反事实实体池改为取自校准集或训练集的答案。
- 是否推测：否（事实层面）；是否造成偏差取决于报告方式。

### 11. [中等] 框架判定的选择偏差未校正（推测）

- 位置：`scripts/analysis/dev_select.py:70`（在 6 种 EAB 形式中取最大增益）、`:114`（`eab_effective` 用所选形式的 95% 区间下界 > 0）、`:117`（框架由上述判定决定）；`docs/DEV_DECISIONS.md:20-21, 24, 26-32`。
- 证据：规则 1 对所选形式的置信区间不做多重比较校正；规则 5 只约束正文数字不使用 dev 结果，但"框架 1/2/3"直接由同一 dev 统计量决定论文主张，并决定是否投 WWW 2027。
- 影响（推测）：由于在 6 个形式中取最大，区间下界被系统性地抬高，"EAB 有效"的判定可能偏乐观，从而使框架 1 更易成立。需要在独立 dev 切分或 bootstrap 最优选择偏差的模拟中验证。
- 修复建议：对所选形式的增益做 max-statistic 校正，或在 dev 的一半上选择、另一半上确认；将框架判定的置信水平写入规则。
- 是否推测：是。

### 12. [中等] 配置一致性检查不完整

- 位置：
  - `src/omniaware/run.py:212-222`：只有 `out_path` 或 `{dataset}.jsonl` 存在时才比较 args。`--calibrate-only` 之后的首次分片运行没有输出文件，因此不做比较，可以用不同参数的分片写入同一 config（`run.py:517-519` 覆盖）。
  - `src/omniaware/run.py:505-519`：config 每次运行都被重写，包括 norm、comb、nte_threshold。续跑时若校准统计变化（例如 relevance 缓存被更改，见问题 3），新 config 与早期已写入答案所用的统计不再一致，且不会被检测。
  - `src/omniaware/run.py:416-417`：ucal 只按 `args.model`（名称）区分，不包含 `paths.QWEN_8B` 等路径或权重哈希。
- 修复建议：只要 config 存在就比较参数；续跑时比较 `norm_pre/norm_post/comb_*/nte_threshold` 与已保存值，不一致则拒绝；config 中记录模型路径与哈希。
- 是否推测：否。

### 13. [中等] 分片下跳过 `--calibrate-only` 时的校准竞争

- 位置：`src/omniaware/run.py:420-439`（校准循环对全部 `calib` 执行，没有 `shard_id` 过滤，且每题 `write_text` 整个 ucal）。
- 证据：代码阅读。`scripts/launch.py:60` 之所以安全，是因为先执行了 `--calibrate-only`；若用户直接以 `--num-shards N --shard-id i` 启动，每个分片都会完整重算校准（约 500 次 prefill），并并发覆盖同一个 ucal 文件，且 `run.py:439` 不是原子写。
- 修复建议：分片模式下若 ucal 缺失则报错，要求先执行 `--calibrate-only`。
- 是否推测：否。

### 14. [中等] ECDF 与 relevance 对未见通道静默返回 0.5

- 位置：`src/omniaware/relevance.py:146-147`（`if not arr: return 0.5`）；`:163-165`（没有任何可用通道时返回 0.5）；`src/omniaware/run.py:284-285`（CE 的 `ce_rel_of` 在 calib 中缺少某模态时使用 0.5）。
- 证据：`exp5_fallbacks.py` 输出 `ecdf for unseen channel: 0.5`；仅有 caption 通道而 `image_mode=clip` 时 `relevance(...)` 返回 0.5。
- 影响：若校准集缺少某模态（例如某个 benchmark 的 calib 中没有视频或图像候选），该模态所有候选的 r_j 被置为 0.5，EAB 的偏置与 Top-k 排序被静默扭曲。发生概率取决于数据，推测较低。
- 修复建议：未见通道时报错，或在日志中计数并写入 config。
- 是否推测：发生频率是推测，代码行为已核实。

### 15. [中等] 候选不足导致中途异常（推测触发频率）

- 位置：`src/omniaware/run.py:53-55`（路由模态在候选中缺失时抛 ValueError）；`:78-79`（perm 要求恰好 5 块）。
- 证据：代码阅读。`src/omniaware/data.py:308-311` 会丢弃缺帧的视频候选，`data.py:325-326` 只打印计数。若某题的视频候选全部缺帧，路由到 video 即抛异常，整个运行中止；续跑会在同一题再次崩溃。`--order perm:*` 同理，候选少于 5 时直接报错。
- 修复建议：在加载阶段统一过滤或统计不可用的题，并在运行前报告；或在 record 中写入 skip 原因，而不是中止。
- 是否推测：是（需要真实数据统计候选缺失率）。

### 16. [轻微] 无候选题保留与缺图检查

- 位置：`src/omniaware/data.py:186-194`（`cands_by_q.get(qid, [])`，BM25 的 `okvqa` 不过滤无候选题，静态臂无证据，与其他 okvqa 加载器不一致）；`data.py:192`（`query_images` 的路径不检查存在性）；`data.py:320`（UniversalRAG 的 query_image 同样不检查）。
- 修复建议：统一过滤或统计；加载时检查图像存在性。
- 是否推测：路径存在性为推测（本仓库无数据）。

### 17. [轻微] 声称"可续跑"的缓存实际不增量写入

- 位置：`src/omniaware/rerank.py:98`（docstring："cached, resumable"）；`:114-122`（全部计算后才写一次）。`src/omniaware/run.py:32-33`（relevance 同样整批写）；`src/omniaware/run.py:486`（SeaKR 每题写整个 JSON，非原子）。
- 修复建议：改为增量写或按批写，并使用 tmp + replace。

### 18. [轻微] `last_answer_attention` 行定义与注释不一致

- 位置：`src/omniaware/qwen_eab.py:279-280`（注释："the first generated token's row (the first row EAB biases)"）；`:291-293`（捕获的是把第 1 个生成 token 作为输入时的行，该行预测第 2 个 token）；`:127`（Result 注释 "first answer row"）。
- 证据：`run()` 中 `grounding` 来自 `base_logits` 之前的后缀最后一行（预测第 1 个 token），而 `_decode` 中 `last_answer_attention` 是第 1 个生成 token 的行。对 `ans1` 变体，EAB 的第一个偏置行是预测第 1 个 token 的行（`pipeline.py:166`，`q_start = enc.length - 1`），与 ARD 使用的行不同。对 `ans2` 变体两者一致。
- 另：`max_new_tokens=1` 时 `_decode` 不执行捕获（`qwen_eab.py:289-293`），`last_answer_attention` 静默为 None，ARD 缺失。
- 影响：ARD 的数值解释取决于定义，推测影响较小。
- 修复建议：明确定义 ARD 所用行，并与 `ans1/ans2` 对齐；增加与显式注意力比较的单元测试。
- 是否推测：定义不一致已核实；数值影响为推测。

### 19. [轻微] NaN 写入 JSON；assert 作为协议不变量

- 位置：`src/omniaware/analysis.py:26-30`（无可用 cid 时返回 NaN，`run.py:610` 写入 `rec["ard"]`，`json.dumps` 产生非严格 JSON 的 `NaN`）；`src/omniaware/run.py:254`（calib 与 eval 不相交的唯一检查）；`src/omniaware/okvqa_dense.py:120`（编码未完成检查）。
- 修复建议：NaN 写为 null；协议检查改为显式 `if ...: raise`。

### 20. [轻微] 预注册日期与提交时间不一致

- 位置：`docs/DEV_DECISIONS.md:1`（"2026-10-11，在看到任何 dev 结果之前写定"）；`docs/PROTOCOL.md:15`（"2026-10-11 决定"）。
- 证据：`git log` 显示 `docs/DEV_DECISIONS.md` 对应提交 `cd48ab8` 的时间为 2026-10-10 10:59:26 +0800，而今天的日期是 2026-10-10，文件中的日期晚于今天。仓库中也没有 `runs/`，因此无法从本地验证 dev 结果是否在规则写定之后产生。
- 修复建议：更正日期，或说明为何日期与提交不一致；保留提交哈希作为预注册证据。
- 是否推测：否（日期事实已核实，动机不明）。

### 21. [轻微] 代码细节

- `src/omniaware/rerank.py:84`：`select_calibrated` 硬编码 `[:5]`，与 `TOP_K` 解耦。
- `src/omniaware/relevance.py:115`：`not c["cid"].startswith("f")` 对当前所有 cid 形式（`t*`、`i*`、`k*`、`v*`、`c*`）都为真，条件无意义（可能是旧版残留）。
- `src/omniaware/rerank.py:52-53, 132`：`rank_sources` 的 `bge_cosine` 字段存的是校准后的 r（`ce_select` 传入的 `rel`），字段名与含义不符。

### 22. [轻微] 测试缺口

- `tests/test_splits.py`：只在固定 n_calib 下测试不相交与确定性，未测 n_calib 变化时 dev/test 的稳定性，也未测空评测集。
- `tests/test_run_smoke.py:388-415`（分片测试）：monkeypatch 了 `load_relevance`，未覆盖缓存命中、续跑截断、`launch` 的"已合并"判定，以及 `merge_shards` 不带 `--expected` 的路径。
- 所有测试都用合成数据，未覆盖 VideoRAG 尺寸的划分。

---

## 三、已核实无问题（正面结论）

- 划分确定性：`data.py:351-354` 先按 qid 排序，再用固定种子打乱，结果与输入顺序无关（`tests/test_splits.py:23-26` 也覆盖）。
- 分片不依赖进程级哈希：分片按 `i % num_shards` 对同一有序列表切分（`run.py:527`）；随机过程使用 `random.Random(str)`（sha512 种子）或 sha256（`interventions.py:91-96`）；未发现依赖内置 `hash()` 的代码。`set` 只用于成员判断。
- 分片覆盖：`tests/test_run_smoke.py:388-415` 验证了 3 分片的并集等于 test 集、无重复。
- 校准标签无关：U 的 Norm/Combiner、ECDF、u_rate、nte_threshold、SeaKR 阈值匹配都只使用无标签的候选与分数（`pipeline.py:77-124`，`run.py:470-488`）；calib 中的 gold 字段未被读取。
- WebQA 开放检索池：`scripts/webqa_open_retrieval/build_global_webqa_pool.py:196-197` 写明 `query_fact_links_exported=False`、`gold_roles_exported=False`，池中不含题目与事实的对应关系；`load_webqa_open` 的 `gold` 仅用于诊断（`data.py:142-143, 150`）。未发现泄漏。
- 生成解码：全部臂使用贪心解码（`qwen_eab.py:278-294` argmax；`baselines/arms.py:29`、`m2rag_qwen.py:84` `do_sample=False`），同一输入的 max_new 相同（`run.py:208`）。
- 排列：`order.permutation_schedule()` 返回 14 个排列（2 个锚点 + 3 个种子 × 4），与协议一致（实测）。
- 配对分析按 qid 取交集（`scripts/analysis/paper_analyses.py:83, 390`），不会因题目集合不同而错位。
- 测试：`python -m pytest -x -q tests/test_run_smoke.py tests/test_splits.py` 全部通过（22 个用例）。
- 显存：EAB 通过自定义 sdpa 注入偏置，只在单层单行捕获注意力（`qwen_eab.py:54-58`），不使用 `output_attentions`，全层全头的显存问题不存在。

## 四、没来得及深入的方向

1. 真实模型上的 chat template 一致性：`pipeline.py:41, 71` 与 `qwen_eab.py:216` 手写 ChatML 且不加系统提示；需与 `processor.apply_chat_template` 的输出逐 token 对比（Qwen3-VL）。InternVL3.5 的官方模板可能带默认系统提示，推测不一致，需要真实权重核实。
2. `qwen_eab.py:253-260` 中对 `rope_deltas` 的手工置空与补零，依赖 transformers 5.5.3 的内部实现，需要在真实 Qwen3-VL 上比较有无缓存的 logits。
3. 吞吐与显存：未在 GPU 上测量；单卡 bf16 加载整模型（`qwen_eab.py:156`），`procs_per_gpu` 的估计（约 20 GB）未验证。
4. `data.py:285, 298`：`df[df.index.isin(...)]` 与 `corpora[m][sid]` 假设 parquet 索引即 source_id，未经真实数据验证（推测可能 KeyError）。
5. `data.py:80-87`（`_asset_path`）：只有 WebQA 开放检索使用了路径映射，OK-VQA 与 UniversalRAG 的 image_path / query_image 原样使用，若记录在其他机器上的绝对路径会失败（推测）。
6. `interventions.py` 的 `distractors` 与 `inject` 的位置保持逻辑、`mask_tokens` 的 pad 处理未逐行核对。
7. `scripts/analysis/dev_select.py` 的 `gate_rho` 对空数组与常数数组的 Spearman 结果（NaN）未在框架判定中做处理。
8. 基线臂（FiTM、M2-RAG、SeaKR 的 `choose_document`）的数据流与其他 agent 的分工重叠，未深入。

# v2/arbitration 验收意见（2026-10-13）

审查对象：`omniaware` 分支 `v2/arbitration`（HEAD `3f7516a`，相对 main 改动 48 个文件，+6425/−608）。
审查方式：通读核心源码和全部 diff，运行 `python -m pytest -q`（全部通过，1 个跳过），核对交付文档，并在同一批旧数据上用朴素基线复算了离线结果。

## 结论：**不通过，需要返工**

工程卫生和 P0 里的简单修复基本合格。但有三个问题让这次交付达不到目标：
1. **新方法的核心（选择器）在设计上就不可能起作用**，离线结果是 0 增益，而同一批数据上最朴素的规则有 +4.8%。
2. **各阶段没有接通**：GPU 跑完之后，没有任何脚本能把输出变成 go/no-go 判据、校准决定和主表。
3. **多处“看起来完成”实际上是空壳**：无效参数、同义反复的测试、静默替换的基线。

合格的部分：分支基于最新 main，main 和远端未动，`C:\rc` 未改；测试全通过；论文里没有编造数字，未跑的实验都用 `\ph{}`；交付文档如实写了“选择器离线效用为 0”，没有粉饰。

---

## A. 致命问题（必须改）

### A-1 风险控制的损失定义错了，选择器必然退回单一动作
位置：`src/omniaware/arbitration.py:283-330`（`ltt_select`）

- 现在的损失是“这道题上策略得分低于最强单一动作就记 1”，并要求这个比例 ≤ δ=0.1。
- 但任何真正在两路之间切换的策略，都必然在相当一部分题上比最强单臂差（WebQA 上 direct 比 static 好的题就占 23%），同时在更多的题上更好。按“逐题退化率 ≤10%”来检验，所有有用的策略都会被否决；只有“几乎不切换”的策略能通过。
- 实测印证：`runs/offline_v2/REPORT.md` 中 WebQA 的 v2-Rule 决定是 `{"name":"best_single","fallback":true}`，评测半上 2483 题全部选 static，效用为 0。
- **应改为对“平均效用”做风险控制**：原假设是 E[策略得分 − 最强单臂得分] < 0（或 < −ε），用配对差值的单侧检验（配对 bootstrap，或有界损失下的 Hoeffding / 经验 Bernstein 界）。这才对应“平均而言不差于最强单一动作”的保证。
- α 与 δ 被写成同一个参数（`_binom_cdf(..., delta)` 再与 `delta` 比较），两者含义不同，应分开。

### A-2 固定序列的起点是“始终直答”，最强单臂不是 direct 时第一步就失败
位置：`arbitration.py:28`（`_TAU1_GRID` 以 −1e9 开头）、`:341-356`、`:308-317`

- 序列的第一个候选是 τ₁=−1e9，即“永远直接作答”。当校准集上最强单臂是 static 时（WebQA 就是），“永远直答”相对 static 有大量退化，第一个检验就失败，`break` 之后整个序列作废。
- “保守”应该相对最强单一动作来定义：序列的起点应是“永远选最强单臂”，再逐步放宽切换条件。
- 规则里“直答不自信、且没有一致的证据动作时退回 direct”（`:89-90`）同样把 direct 当成了默认，应退回最强单臂。

### A-3 v2-Cal 没有动作身份，实际退化成“永远直答”
位置：`arbitration.py:95-119, 176-225`

- 所有动作共享一组权重，预测“这个动作答对的概率”，特征里没有动作类型，也没有“问题层面”的特征交互，所以学不到“图像题别用图像检索、文本题该检索”这类规律。
- 实测：WebQA 评测半上 v2-Cal 均值 0.41275，与 direct 均值逐位相同，即全部选了 direct。
- 缺失特征被填成 0（`kl_to_direct`、`top1_score` 为 None 时），把“没有这个信号”和“信号值为 0”混在一起。
- 建议改为直接预测**效用差**（证据动作得分 − 直答得分），每个证据动作一组小权重或加动作指示变量，并与 A-1 的风险控制接上。v2-Cal 目前完全不经过风险控制，也没有进入 `dev_select`。

### A-4 流水线没有接通：S2 跑完后无法得到判据、决定和主表
- `run.py` 把五个动作的答案和信号写进 `rec["actions"]`，但 `score_official` 只给 `rec["preds"]` 里的臂打分。`static_text`、`static_image`、`rerank_static` 三个动作的答案**从来不会被官方评分**。
- `dev_select.select_v2` 要求每条记录同时有 `actions` 和每个动作的 `scores`；`v2_gate.py` 要求 `{"questions":[{"qid","scores"}]}`。**仓库里没有任何脚本生成这两种输入。**
- `apply_frozen_v2`、`fit_v2_cal`、`select_v2_cal` 除了离线探索脚本之外没有任何调用方；S4 的 `--decisions` 只被算了个哈希，冻结的决定并没有被应用到测试输出上，也就没有产出主表的脚本。`docs/PAPER_TABLE_MAP.md` 里对应的命令因此不成立。
- 需要补一个明确的“打分并拼表”步骤：读运行输出 → 对每个动作的答案做官方评分 → 输出逐题的“动作 × 分数 + 信号”表 → 供 gate、dev_select、测试阶段应用冻结决定并出主表。

### A-5 `rerank_static` 被静默换成“词面重叠”
位置：`run.py:990-994`、`signals.py:76-87`、`scripts/v2_runs.py`

- 没加载 cross-encoder 时，`rerank_static` 用问题与候选的词重叠来“重排”。这不是 reranker。更严重的是 `v2_runs.py` 生成的命令**从不传 `--selector ce`**，所以按文档跑出来的 S1/S2/S4 里，`rerank_static` 全部是词重叠版本。
- 只在配置里记一个 `rerank_static_source` 不够。应改为：没有 cross-encoder 时直接报错，或者不产出这个动作。

---

## B. 严重问题

### B-1 无效的“契约参数”
`run.py:251-260`：`--save-signals`、`--empty-answer-accounting`、`--shared-prompt-registry` 三个参数不改变任何行为；`--v2-actions` 只允许等于完整列表，同样没有意义。应全部删除。

### B-2 v2 动作解码对所有运行强制开启
`run.py:983-1028` 的 v2 块在主循环里无条件执行。所有旧实验（顺序实验、消融等）每题都会多解码 2–3 次，成本成倍增加。应改成显式开关（如 `--v2`），并把这段逻辑从 `main()` 里抽成函数。

### B-3 A2 的测试是同义反复
`prompts.arm_sha256s` 对传入的任意臂名都返回同一个哈希，`tests/test_v2_prompts.py` 断言“所有臂哈希相同”恒成立，并没有检查各臂**实际解码时**用的指令。M2-RAG 是把共享指令拼在它自己的 `EXTRACTOR_INSTRUCTION` 前面，SeaKR 采样、FiTM 各自怎么取指令都没有被测试覆盖，`EXCEPTIONS` 却是空的。需要在真实的构造路径上断言（例如用 tiny 模型跑一题，抓各臂的 prompt 文本）。

### B-4 `paper_runs.py` 主表运行被整体关掉了探测和路由
`scripts/paper_runs.py` 把 `v321_main` 等运行改成 `probe=False, route=False`，理由写的是修 S-4。但 main 里基线本来就读 joint 输入（D4），S-4 在这层已经解决；这个改动实际上把**我们自己方法**的探测轮和路由也关了，属于超出要求的行为变更，理由也不成立。需要说明或回退。`run.py` 里新增的配置项 `"baseline_retrieval": "joint_topk_no_probe_no_route"` 是写死的字符串，不随实际参数变化，会在开了路由的运行里写入错误信息。

### B-5 空答案处理的细节问题
- `run.py:1018-1024` 用解析后的答案**覆盖** `rec["preds"]`，原始输出没有保留。`--answer-style cite` 的运行里，单独成行的 `[E1] [E2]` 引用会被剥掉，可能影响引用评估。应保留原始文本，另存解析结果。
- 空答案率只在单进程路径（`num_shards == 1`）里统计，多卡合并路径没有；统计也只覆盖 `preds`，不包括 `actions`。
- 模态候选池为空时，`static_text` / `static_image` 会静默复用 direct 的结果（`decoded[tuple()]`），应显式标记“该动作不可用”。

### B-6 `v2_runs.py` 的数据集与陈旧注释
- 数据集表里用的是 `okvqa`，而 `docs/DEV_DECISIONS.md` 明确写了 BM25 版 `okvqa` 候选无法复现、已移除，实际可用的是 `okvqa_gs`。
- 文件里留着已经过期的注释（“run.py --split is only dev|test”“run.py in this worktree does not define these flags yet”），与当前代码矛盾。
- S2 只生成解码命令，没有评分命令，也没有调用 gate。

### B-7 WebQA 子集划分的回退逻辑不对
`score_official.webqa_subset_name`：没有 `gold_modalities` 时按 `qcate in {'image','img'}` 判断。WebQA 的 qcate 取值是 text / YesNo / color / shape / number / choose / Others，图像题的 qcate 不会是 image，这个回退会把图像题全部归为文本题。应在缺少 `gold_modalities` 时直接报错。

---

## C. 命名、组织与代码质量

- **同一个常量定义了四次**：`V2_ACTIONS` 分别出现在 `signals.py`、`arbitration.py`、`v2_runs.py`（字符串）、`v2_gate.py`（`ACTION_ORDER`）。应只在一处定义。
- **重复实现**：`Sources` 正则在 `answers.py` 和 `offline_v2.py` 各一份；对半拆分有 `arbitration.split_half` 和 `offline_v2.local_split_half` 两份；“读取并校验冻结决定”在 `run.check_v2_flags` 和 `v2_runs.read_frozen_decisions` 各一份；`score_official` 里 `_mean_defined` 与 `_arm_means` 逻辑重复，输出里 `normalized_bart` / `normalized_bart_score`、`QA_approx` / `acc_approx_x_normalized_bart` 是同值别名，还在 `means` 和顶层各存一遍。
- **`offline_v2.py` 有 1263 行**，数据加载、信号代理、评估、报告渲染全部混在一个文件。里面有明显的并行开发残留：`_import_arbitration()` 用 `SimpleNamespace` 包一层并处理“模块不存在”的情况，还去取并不存在的 `ActionRecord`、`FirstToken`；`make_action(mod, ...)` 第一行就是 `del mod`。应直接正常 import，并拆成“加载 / 评估 / 报告”几个部分。
- **命名**：`_TAU1_GRID` 用 ±1e9 当哨兵，候选名因此成了 `margin_tau-1e9_theta1` 这种形式；`keepers`、`_format_threshold`、`N_PARAMS = 7` 加运行时检查 `N_PARAMS >= 10` 都很别扭；`split_half`、`assert_disjoint_qids` 放在选择器模块里不合适；`diagnostic_substring_accuracy` 是任务没有要求的新增宽松评分函数，且没有调用方，应删除。
- **小问题**：`token_f1` 只按空格切词，不做大小写和标点归一化（`Wood.` 与 `wood` 不匹配），一致度信号会被严重削弱；`signals.sequence_logprob` 在多 token 且没有逐 token logprob 时返回 None，而 `action_features` 直接对它做 `float()`，会抛异常；`run.py` 里 `selected = None if chosen is None else chosen`、`getattr(result, "token_logprobs", None)` 是多余写法；`ltt_select` 的文档字符串是中文而模块其他部分是英文。
- **文档不一致**：`docs/REVIEW_V2_STATUS.md` 写着 S-2、S-12、S-14“不修（本代理禁止改论文 / README）”，而 `DELIVERY_v2.md` 把它们写成“已完成 / 部分完成”。前者是子代理的局部视角，没有在合并后更新。`DELIVERY_v2.md` 里还有不少条目（S-8、S-9、S-10、S-11、S-19 等）标为“已完成”，实际是 main 里早就修好的，应注明“本次未改动”。

---

## D. 论文部分
- 没有编造数字，这一点合格。但第 69–78、275–290 行引用的数字来自旧管线（含已知 bug），正文应明确标注来源和性质。
- 风险控制的表述要随 A-1 重写。现在全文五处重复“does not say that the method is never worse”，而“逐题退化率 ≤10%”这个声明本身既弱又不是我们要的保证；改为平均效用的保证后，表述会更自然，也更有说服力。

---

## E. 返工要求（按优先级）
1. 重写风险控制：对平均效用做单侧检验，α 与 δ 分开；序列起点改为“永远选最强单臂”；规则的默认动作改为最强单臂。（A-1、A-2）
2. 重做 v2-Cal：预测效用差，带动作身份；缺失信号不能填 0；纳入风险控制和 `dev_select`。（A-3）
3. 补齐“动作评分与拼表”步骤，接通 S2 → gate → dev_select → S4 应用冻结决定 → 主表。（A-4）
4. `rerank_static` 没有 cross-encoder 时报错；`v2_runs.py` 传 `--selector ce`，数据集改为 `okvqa_gs`，补评分和 gate 命令，删掉陈旧注释。（A-5、B-6）
5. 删除无效参数；v2 解码改为显式开关并抽成函数。（B-1、B-2）
6. A2 的测试改为检查各臂真实使用的 prompt；逐个核对 M2-RAG、SeaKR、FiTM。（B-3）
7. 说明或回退 `paper_runs.py` 的 `probe=False, route=False`；修正写死的配置字符串。（B-4）
8. 保留原始输出；多卡路径也要统计空答案率；模态池为空时显式标记。（B-5）
9. 清理重复定义和 `offline_v2.py`；按 C 节整理命名；统一两份状态文档。（C）
10. **验收门槛（新增）**：在 `runs/offline_v2` 的同一批旧数据、同一个对半拆分上，新的选择器在评测半上的效用必须为正，并且不低于“按 `u_initial` 单阈值”这个朴素基线（WebQA 评测半 acc 43.31，始终 static 为 41.32）。报告里要并排列出这个朴素基线。该结果仍标注为探索性。

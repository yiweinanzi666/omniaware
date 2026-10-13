# v2 交接：拿到最新代码后要做什么

读者：在 GPU 服务器上执行实验的人。本文只讲顺序、判断标准和要回传的东西；每条命令的细节在 [V2_RUNS.md](V2_RUNS.md)，协议在 [PROTOCOL.md](PROTOCOL.md)，冻结的规则在 [DEV_DECISIONS.md](DEV_DECISIONS.md) 的“v2 预注册”一节。

## 0. 先知道这几件事

- **方向变了。** 主方法不再是 EAB（注意力偏置）。现在的方法是：每道题同时解码五个动作（`direct`、`static_text`、`static_image`、`static_mixed`、`rerank_static`），再由选择器逐题选一个。EAB 只留在论文的机制分析里。
- **旧结果全部作废。** 相对历史缓存基线的 +33% / +15%、训练式控制器的 +8.43%、32 题上的 +14.4%，都不能再引用。原因见 [reports/Omni-Aware_根因溯源报告_20261013.md](reports/Omni-Aware_根因溯源报告_20261013.md)；新方向的依据见 [reports/Omni-Aware_方向与改造方案_20261013.md](reports/Omni-Aware_方向与改造方案_20261013.md)。
- **目前没有任何 v2 的正式结果。** `runs/offline_v2/REPORT.md` 是在旧数据上做的探索，只说明代码能用、方向没错：去掉旧管线的空答案后，零训练规则相对“始终检索”是 +2.8%，逐题选对的上限是 +25%。能不能到 5%，要看这次 GPU 跑出来的新信号。
- **有两个地方必须停下来等判断**（第 3 步和第 5 步），不要一口气跑到底。
- **不要改任何规则和阈值。** 提示词、δ、ε、阈值网格、数据切分都已冻结。发现问题先反馈，不要自己调。

## 1. 取代码、装环境、自检（不需要 GPU，约 10 分钟）

```bash
git checkout main
git pull origin main
pip install -e ".[all]"
python -m pytest -p no:warnings
```

预期：汇总行里没有 `failed` 和 `error`。passed 与 skipped 的数量随环境变化：缺少 torchvision / spacy / NLTK 数据或资产目录的测试会跳过；这一步在下载模型之前做，`tests/test_m2rag_qwen.py` 读取本地 Qwen3-VL-8B 的处理器文件，模型没下载时整个文件跳过，下载后再跑一次它会参与运行。**有 failed 或 error 就停下来反馈，不要继续。**

数据和模型的准备方式没有变，见 README 的第 1–4 节。v2 用到三个数据集：

| 数据集 | 角色 | 需要的资产 |
|---|---|---|
| `webqa_full` | 主设置（WebQA distractor） | WebQA 标注与图片、BGE / CLIP 相关度 |
| `webqa_open_img` | 第二设置（开放检索，图像题） | 开放检索的全局池与索引 |
| `okvqa_gs` | 第三设置 | OK-VQA 标注、COCO 图片、GS-112K 语料的检索结果 |

另外需要 cross-encoder reranker（`bge-reranker-v2-m3`）的权重，`rerank_static` 这个动作和 CE 基线都要用。

## 2. S1 冒烟 + 图像证据诊断（1 张 GPU，估计半天）

```bash
python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50
python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --n 50
python scripts/analysis/image_evidence_check.py --n 50 --output runs/image_evidence_check.json
```

想先看会执行什么，加 `--dry-run`。

**S1 要确认四件事：**

1. **空答案率不超过 0.5%。** 这里的空答案指空白输出和只有引用行的输出（如 `Sources: none`）。命令里带了 `--max-empty-rate 0.005`，超过会以非零退出码结束并打印 `EMPTY ANSWER RATE`。旧管线里约 10% 的题只输出 `Sources: none`，这次必须确认它消失了。拒答（如 “I cannot determine the answer from the provided information.”）评分时同样按空答案计，但单独统计，不触发这个退出码：`<数据集>.merged.json` 里的 `refusal_rate` 是总的拒答率，`empty_by_arm` 按臂和动作（`action:<动作>`）列出空白、纯引用、拒答各多少。WebQA 的 `direct` 不带图片，图像题上有一定拒答属于正常；**如果某个证据动作（尤其 `action:static_image`）的拒答明显多于 `direct`，说明图片可能没进模型，停下来反馈。**
2. **各个动作的回答格式一致。** 打开 `runs/v2_s1_<数据集>/<数据集>.jsonl`，抽 20 题看 `actions` 里五个动作的 `answer`：WebQA 应当都是一个完整的句子，OK-VQA 应当都不超过 3 个词。旧管线的问题是带检索的臂答得很短、不带检索的臂答得很长，导致比较失真。**如果某个动作的回答明显更长或更短，停下来反馈。**
3. **图像证据真的进了模型。** 看 `runs/image_evidence_check.json`：每道图像题的 `static_mixed` 和 `static_image` 输入里有几张图、文件是否可读、缩放后的像素和视觉 token 数、gold 图片是否在选中的证据里。旧数据里图像题加了检索反而变差（即使回答格式相同也是如此），需要判断这是管线 bug 还是真实现象。可以再跑一下它打印的 GPU 探针命令（让模型描述每张检索到的图）。

4. **溯源记录写出来了，而且内容合理。** 这一条在本机只用假模型测过，真实模型和真实图片上要在冒烟时确认：
   - `runs/v2_s1_<数据集>/<数据集>.env.ndjson` 里 `git_commit` 不为空，`git_dirty` 是 `false`（有未提交的改动就先提交或还原再跑），`model.attn_implementation` 是 `sdpa`，`model.dtype` 和 `gpus` 符合预期；
   - `<数据集>.jsonl` 里 WebQA 图像题的 `actions.static_image.input.n_images` 大于 0，`image_tokens` 里每个数都大于 0；`actions.direct.input.n_images` 是 0（OK-VQA 的直答带问题图片，应为 1）；
   - `actions.<动作>.first_token.topk` 里每个候选都有 `text`；
   - `<数据集>.prompts.ndjson` 有 20 行，打开看几条，确认 prompt 的样子符合预期；
   - `<数据集>.log` 存在，里面有 `[env]` 和 `[done]` 两行。

**回传：** 两个 `runs/v2_s1_*` 目录整个提交，以及 `runs/image_evidence_check.json`。提交方法见第 9 节。

## 3. 第一个停顿点：S1 的结果要先看过

以下任何一条出现，都先停下来：

- 空答案率超过 0.5%；
- 各动作的回答格式不一致；
- 图像诊断显示图片没进输入、gold 图片经常不在选中的证据里，或像素预算明显过低（默认 `--max-pixels` 是 256×32×32）。

这三条都会让后面的全部结果失真，必须先修。

## 4. S2 校准集与开发集（1 张 A100，估计 4–8 小时，不含 SeaKR）

```bash
python scripts/v2_runs.py --stage s2 --dataset webqa_full
python scripts/v2_runs.py --stage s2 --dataset okvqa_gs
python scripts/v2_runs.py --stage s2 --dataset webqa_open_img
```

每条命令会依次做：解码校准集（500 题）→ 解码开发集（256 题，带 FiTM / M2-RAG / SeaKR 基线臂）→ 官方评分 → 拼逐题动作表 → 在开发集上做 go/no-go 判断 → 在校准集上冻结选择器的决定，写到 `runs/v2_decisions/<数据集>.json` → 把冻结决定应用到开发集，检查选择器相对最强单一动作是否至少 +5% 且 CI 下界大于 0，结果写到 `runs/v2_decisions/<数据集>.dev_check.json`。

- SeaKR 每题最多 125 次额外采样，很贵。算力不够时加 `--no-seakr`，主表会缺 SeaKR 一行，输出里会注明。
- 任何一步失败都会停下并打印原因。**go/no-go 不通过时不会生成决定文件，这是预期行为。** 开发集检查不通过时，决定文件和检查文件都会写出，命令以 `development check FAIL` 结束。
- 可以多卡分片，方法和以前一样（README 第 5 节）；同一个输出目录有运行锁，重复启动会被拒绝。

**回传：** 每个 `runs/v2_s2_*` 目录整个提交（方法见第 9 节），加上 `runs/v2_decisions/*.json`（决定文件和 `*.dev_check.json` 都在里面）。每个目录里各文件的用途见下面的“每次运行会留下什么”。

## 5. 第二个停顿点：go/no-go（最关键的一步）

`v2_table.py gate` 在开发集上算一个数：**每题都选对动作时的得分，比最强的单一动作高多少。** 判据是相对差距至少 +8%。

- **PASS**：这个方向在修好的管线上仍有空间，继续第 6 步。
- **FAIL**：修掉 bug 之后，直答和检索不再互补，这个方法没有足够的提升空间。**不要跑 S4。** 把结果回传，论文会改成“度量发现 + 机制分析”的定位。

不管 PASS 还是 FAIL，都把 `gate` 的完整输出和 `*.v2_build_summary.json` 回传。同时看一下决定文件里的 `fallback` 字段：如果是 `true`，说明选择器在校准集上没找到任何能通过风险控制的策略，退回了最强单一动作，这种情况也要先反馈再决定是否跑 S4。

gate 只说明有空间。S2 的最后一步 `devcheck` 检查冻结后的选择器在开发集上实际拿到了多少：相对最强单一动作至少 +5%，且配对 bootstrap 95% CI 下界大于 0（协议第 5 节）。

- **PASS**：可以进入第 6 步。
- **FAIL**：先回传 `runs/v2_decisions/<数据集>.dev_check.json` 和 `runs/v2_s2_<数据集>_dev/<数据集>.v2_dev_main.json`，等决定。退回单一动作（`fallback=true`）时这一步必然 FAIL。S4 会拒绝在未通过的检查上运行；只有确认要按协议照跑一次并照实报告时，才加 `--allow-failed-dev-check`。

## 6. S4 冻结测试（1–2 张 A100，估计 1–2 天）

只有第 5 步的 gate 和 devcheck 都通过、并且确认要跑之后才执行。**每个数据集只跑一次。** S4 会核对决定文件旁边的 `<数据集>.dev_check.json`：文件缺失、记录的 sha256 与决定文件不一致，或检查未通过时，都不会开始解码。

```bash
python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions runs/v2_decisions/webqa_full.json
python scripts/v2_runs.py --stage s4 --dataset okvqa_gs --decisions runs/v2_decisions/okvqa_gs.json
```

它会解码全部测试集（WebQA 是 4966 道 val），官方评分，拼表，再把冻结的决定应用上去，输出 `runs/v2_s4_<数据集>/<数据集>.v2_main.json`。这个文件就是论文主表的来源，里面有：

- 五个单一动作各自的得分、逐题选对的上限；
- v2-Rule（主方法）和 v2-Cal 的得分；
- 各基线的得分：调过阈值的 U 门控和 NTE、SeaKR、FiTM、M2-RAG；
- v2-Rule 相对每个基线的配对差、95% 置信区间、胜 / 负 / 平；
- WebQA 另外按图像题和文本题各报一遍。

**回传：** `runs/v2_s4_*` 整个目录。如果 `<数据集>.jsonl` 太大，先传其余文件，这个文件单独压缩后再传；**不要删掉它**，逐题的输入、解码和候选排序都在里面。

## 6.5 每次运行会留下什么（溯源用，全部要回传）

每个 `runs/<tag>/` 目录里，除了答案之外，还有下面这些记录。它们的作用是：结果出现反常时，能从最终分数一路追回到模型实际读到的输入，不需要重跑。**它们都是自动写的，不需要额外操作，但请不要删、不要改、不要只挑一部分回传。**

| 文件 | 内容 | 用来回答什么 |
|---|---|---|
| `<数据集>.env.ndjson` | 每次启动一行：代码的 git 提交号、是否有未提交的改动、python / torch / transformers 版本、GPU 型号、注意力实现、dtype、完整命令行 | 这次结果是哪版代码、哪个环境跑出来的 |
| `<数据集>.log` | 屏幕上的全部输出，包括警告和异常堆栈；续跑时追加，不覆盖 | 运行中途发生了什么，为什么失败 |
| `<数据集>_config.json` | 全部参数、三份题目列表的 sha256、缓存指纹、提示词及其哈希 | 用的是什么设置、哪些题 |
| `<数据集>.jsonl` | 逐题记录，见下 | 每道题每一步发生了什么 |
| `<数据集>.prompts.ndjson` | 前 20 道题、每个动作的完整 prompt 文本（图片位置标为 `<image:路径>`） | 人工检查模型到底读到了什么 |
| `<数据集>.merged.json` | 完成标记：题数、输出与配置的 sha256、空答案率 | 这次运行是否完整 |
| `<数据集>.official_scores.jsonl` / `.official_summary.json` | 逐题逐臂的官方分数、汇总 | 评分 |
| `<数据集>.v2_actions.jsonl` / `.v2_build_summary.json` | 逐题动作表、各动作均分与逐题上限 | 选择器的输入 |

逐题记录（`<数据集>.jsonl`）里与溯源有关的字段：

- `candidate_scores`：**全部**候选的相关度、重排分数、模态、是否为 gold，按相关度排序。可以看 gold 证据排第几、离入选差多少。
- `actions.<动作>.chunks`：这个动作实际送进模型的证据。
- `actions.<动作>.input`：prompt 的 token 数、token 序列的哈希、图片张数、**每张图实际占的视觉 token 数**（从送进模型的 token 序列里数出来的，不是按预期算的）。
- `actions.<动作>.decode`：生成的 token id、逐 token 的对数概率、耗时、是否撞到生成长度上限。
- `actions.<动作>.first_token`：首个答案 token 的前 10 个候选（id、文字、概率）、熵、领先幅度。
- `actions.<动作>.retriever`：该动作所用证据上的检索分数统计，以及整个候选池的统计。
- `preds` 与 `parsed`：各臂的原始输出，以及解析后的答案和空答案标记。

决定文件 `runs/v2_decisions/<数据集>.json` 里的 `trace` 字段记录了选择器检验过的**每一个**候选阈值：p 值、平均收益、改变了多少题、是否通过，以及在哪一步停下。如果选择器退回了单一动作，`fallback_reason` 会写明原因。

多卡分片运行时，每个分片有自己的 `.env.ndjson`、`.log` 和 `.prompts.ndjson`（文件名里带 `shardXX-of-NN`），合并答案时不会动它们，请一并回传。

## 7. 不要做的事

- 不要在测试集上调任何东西，不要重跑 S4 来“换一个更好的结果”。
- 不要改提示词、`--k`、`--max-pixels`、δ、ε、切分种子。要改必须先商量，并且要从 S2 重新开始。
- 不要用 `--answer-style cite` 跑主实验。引用格式会让模型弃权，只用于单独的引用评估。
- 不要把 `runs/offline_v2` 里的数字、或者任何旧管线的数字写进论文主表。
- v1 的 `scripts/paper_runs.py` 和 `scripts/run_all.sh` 还在，它们只用于机制分析和基线，**不产生 v2 主表**。

## 8. 已知的遗留问题

- **OK-VQA 的校准集和开发集取自评测数据里保留出来的部分**，不是 train（仓库里没有 OK-VQA 的 train 加载器）。三部分互不重叠，论文里会如实写明。
- **校准运行不带 SeaKR / FiTM / M2-RAG**，所以“校准集上的最强基线”只在单一动作、U 门控、NTE 之间确定；测试时仍会逐个报告 v2 相对这三个基线的比较。
- **FiTM 的温度是 5e-5**，没有对照原论文核实过。
- **真实权重下的对话模板**只用小模型测过，S1 时请留意生成的回答是否正常。
- **UniversalRAG 基线在 WebQA 上不可用**（WebQA 的候选没有它需要的 suite 分数），主表里这一格是空的。

## 9. 结果怎么提交回来

`.gitignore` 已经放行了 v2 的运行目录，直接用 git 提交即可，不需要另外打包：

```bash
git checkout -b results/v2-s1-<日期>          # 每个阶段一个分支，不要直接推 main
git add runs/v2_s1_* runs/image_evidence_check*.json
git commit -m "S1 smoke runs and image evidence check"
git push -u origin results/v2-s1-<日期>
```

S2 阶段把 `runs/v2_s2_*` 和 `runs/v2_decisions/` 加进去，S4 阶段加 `runs/v2_s4_*`。

- 会被提交的：这些目录里的答案、配置、环境记录、日志、prompt 样本、官方分数、动作表、决定文件。
- 自动被忽略的：锁文件，以及分片答案文件（`*.shardXX-of-NN.jsonl`，合并后的 `<数据集>.jsonl` 已经包含它们）。分片的日志、环境记录和 prompt 样本会保留。
- **单个文件不能超过 100 MB**，否则 GitHub 会拒收。全量测试的 `<数据集>.jsonl` 如果超了，先 `gzip` 再提交（`*.jsonl.gz` 不会被忽略），原文件留在服务器上不要删。
- 如果设置了 `OMNIAWARE_RUNS` 把输出放到了仓库外面，提交前把对应目录复制到仓库的 `runs/` 下，目录名保持不变。
- 提交前确认工作区里没有改过代码（`git status` 里只有 `runs/` 下的新文件）。如果为了跑通改了代码，把改动单独提交并说明原因。

## 10. 遇到问题时

反馈时请带上：执行的完整命令、报错的最后 30 行、对应 `runs/<tag>/` 下的 `*_config.json`。不要手动删除 `runs/<tag>/` 里的部分文件再续跑；要重来就删掉整个 tag 目录。

# v2 交接：拿到最新代码后要做什么

读者：在 GPU 服务器上执行实验的人。本文只讲顺序、判断标准和要回传的东西；每条命令的细节在 [V2_RUNS.md](V2_RUNS.md)，协议在 [PROTOCOL.md](PROTOCOL.md)，冻结的规则在 [DEV_DECISIONS.md](DEV_DECISIONS.md) 的“v2 预注册”一节。

## 0. 先知道这几件事

- **方向变了。** 主方法不再是 EAB（注意力偏置）。现在的方法是：每道题同时解码五个动作（`direct`、`static_text`、`static_image`、`static_mixed`、`rerank_static`），再由选择器逐题选一个。EAB 只留在论文的机制分析里。
- **旧结果全部作废。** 相对历史缓存基线的 +33% / +15%、训练式控制器的 +8.43%、32 题上的 +14.4%，都不能再引用。原因见仓库外的《根因溯源报告》。
- **目前没有任何 v2 的正式结果。** `runs/offline_v2/REPORT.md` 是在旧数据上做的探索，只说明代码能用、方向没错：去掉旧管线的空答案后，零训练规则相对“始终检索”是 +2.8%，逐题选对的上限是 +25%。能不能到 5%，要看这次 GPU 跑出来的新信号。
- **有两个地方必须停下来等判断**（第 3 步和第 5 步），不要一口气跑到底。
- **不要改任何规则和阈值。** 提示词、δ、ε、阈值网格、数据切分都已冻结。发现问题先反馈，不要自己调。

## 1. 取代码、装环境、自检（不需要 GPU，约 10 分钟）

```bash
git fetch origin
git checkout v2/arbitration
git pull
pip install -e ".[all]"
python -m pytest -p no:warnings
```

预期：`449 passed, 5 skipped`（跳过的是缺少 torchvision / spacy / NLTK 数据或资产目录的测试，数量可能因环境略有不同）。**有失败就停下来反馈，不要继续。**

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

**S1 要确认三件事：**

1. **空答案率不超过 0.5%。** 命令里带了 `--max-empty-rate 0.005`，超过会以非零退出码结束并打印 `EMPTY ANSWER RATE`。旧管线里约 10% 的题只输出 `Sources: none`，这次必须确认它消失了。
2. **各个动作的回答格式一致。** 打开 `runs/v2_s1_<数据集>/<数据集>.jsonl`，抽 20 题看 `actions` 里五个动作的 `answer`：WebQA 应当都是一个完整的句子，OK-VQA 应当都不超过 3 个词。旧管线的问题是带检索的臂答得很短、不带检索的臂答得很长，导致比较失真。**如果某个动作的回答明显更长或更短，停下来反馈。**
3. **图像证据真的进了模型。** 看 `runs/image_evidence_check.json`：每道图像题的 `static_mixed` 和 `static_image` 输入里有几张图、文件是否可读、缩放后的像素和视觉 token 数、gold 图片是否在选中的证据里。旧数据里图像题加了检索反而变差（即使回答格式相同也是如此），需要判断这是管线 bug 还是真实现象。可以再跑一下它打印的 GPU 探针命令（让模型描述每张检索到的图）。

**回传：** 两个 `runs/v2_s1_*` 目录下的 `*_config.json`、`*.merged.json`、`*.jsonl`，以及 `runs/image_evidence_check.json`。

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

每条命令会依次做：解码校准集（500 题）→ 解码开发集（256 题，带 FiTM / M2-RAG / SeaKR 基线臂）→ 官方评分 → 拼逐题动作表 → 在开发集上做 go/no-go 判断 → 在校准集上冻结选择器的决定，写到 `runs/v2_decisions/<数据集>.json`。

- SeaKR 每题最多 125 次额外采样，很贵。算力不够时加 `--no-seakr`，主表会缺 SeaKR 一行，输出里会注明。
- 任何一步失败都会停下并打印原因。**go/no-go 不通过时不会生成决定文件，这是预期行为。**
- 可以多卡分片，方法和以前一样（README 第 5 节）；同一个输出目录有运行锁，重复启动会被拒绝。

**回传：** 每个 `runs/v2_s2_*` 目录下的 `*_config.json`、`*.merged.json`、`*.official_summary.json`、`*.v2_build_summary.json`、`*.v2_actions.jsonl`，go/no-go 的终端输出，以及 `runs/v2_decisions/*.json`。

## 5. 第二个停顿点：go/no-go（最关键的一步）

`v2_table.py gate` 在开发集上算一个数：**每题都选对动作时的得分，比最强的单一动作高多少。** 判据是相对差距至少 +8%。

- **PASS**：这个方向在修好的管线上仍有空间，继续第 6 步。
- **FAIL**：修掉 bug 之后，直答和检索不再互补，这个方法没有足够的提升空间。**不要跑 S4。** 把结果回传，论文会改成“度量发现 + 机制分析”的定位。

不管 PASS 还是 FAIL，都把 `gate` 的完整输出和 `*.v2_build_summary.json` 回传。同时看一下决定文件里的 `fallback` 字段：如果是 `true`，说明选择器在校准集上没找到任何能通过风险控制的策略，退回了最强单一动作，这种情况也要先反馈再决定是否跑 S4。

## 6. S4 冻结测试（1–2 张 A100，估计 1–2 天）

只有第 5 步通过、并且确认要跑之后才执行。**每个数据集只跑一次。**

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

**回传：** `runs/v2_s4_*` 整个目录（`*.jsonl` 如果太大就先传其余文件）。

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

## 9. 遇到问题时

反馈时请带上：执行的完整命令、报错的最后 30 行、对应 `runs/<tag>/` 下的 `*_config.json`。不要手动删除 `runs/<tag>/` 里的部分文件再续跑；要重来就删掉整个 tag 目录。

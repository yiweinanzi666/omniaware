# Omni-Aware v2 交付说明

分支 `v2/arbitration`，起点是 `main` 的 `5c66f73`。没有推送到远端。`main` 与 `origin/main` 仍是 `5c66f73`。`C:\rc\` 只读，没有修改。

本文件之后的提交只加入本文件，所以下面的 `git log` 不包含这一次提交。

## 1. 总表

状态只取：已完成 / 部分完成 / 未做。测试列是锁定该项的测试；没有单独测试时写“无”。

| 条目 | 状态 | 改动文件 | 测试 | 备注 |
|---|---|---|---|---|
| R1 EAB 推不动答案 | 已完成 | `paper/main.tex`，`paper/sec_experiments.tex`，`src/omniaware/run.py` | `tests/test_v2_review.py` | EAB 退出主方法，论文改为机制分析。没有新的 GPU 数字。 |
| R1b 实际偏置小于上界 | 已完成 | 同上 | 无 | 作为负结果写进 “Attention steering moves attention, not answers”。数字只引用根因报告。 |
| R2 EAB 实现不是失效 bug | 已完成 | `src/omniaware/qwen_eab.py` | `tests/test_qwen_eab.py`，`tests/test_v2_review.py` | 维持原结论。非 sdpa 显式报错。 |
| R3 格式拖垮 static | 部分完成 | `src/omniaware/prompts.py`，`src/omniaware/pipeline.py` | `tests/test_v2_prompts.py` | 各臂共用冻结指令。措辞未在开发集上重选，本机没有 GPU。 |
| R4 QA 分被 BART 主导 | 已完成 | `src/omniaware/eval/score_official.py` | `tests/test_v2_scoring_path.py` | 主指标是 acc_approx。BART 与 QA_approx 同时输出；没有权重时这两项为 null，不编造。 |
| R5 组件没被真正测到 | 部分完成 | `src/omniaware/signals.py`，`src/omniaware/arbitration.py`，`src/omniaware/run.py` | `tests/test_v2_signals.py`，`tests/test_arbitration.py` | 信号和选择器已实现。没有新的全量运行。 |
| R6 正结果来自小样本或已消费的开发题 | 部分完成 | `src/omniaware/data.py`，`docs/DEV_DECISIONS.md`，`scripts/analysis/dev_select.py` | `tests/test_v2_splits.py`，`tests/test_arbitration.py` | 切分与预注册已写入。+8% 判据没有在新开发集上计算。 |
| R7 评分口径不统一 | 已完成 | `src/omniaware/eval/score_official.py` | `tests/test_v2_scoring_path.py` | 主表路径走官方评分。 |
| R8 论文声明与实现不符 | 已完成 | `paper/main.tex`，`paper/sec_experiments.tex`，`paper/sec_appendix.tex`，`README.md` | 无 | 旧声明改为撤回或 `\ph{}`。`latexmk -pdf main.tex` 退出码 0，8 页。 |
| P0-1 / A1 空答案与纯引用 | 已完成 | `src/omniaware/answers.py`，`src/omniaware/run.py` | `tests/test_v2_answers.py` | 主实验不用 cite。汇总写 `empty_answer_rate`，超过 0.5% 打印 `EMPTY ANSWER RATE`。 |
| P0-2 / A2 各臂同一指令 | 已完成 | `src/omniaware/prompts.py` | `tests/test_v2_prompts.py` | `EXCEPTIONS` 为空。指令 sha256 写入运行配置。 |
| P0-3 / A3 官方评分 | 已完成 | `src/omniaware/eval/score_official.py` | `tests/test_v2_scoring_path.py` | OK-VQA 只走 `score_okvqa`。WebQA 输出 acc_approx、归一化 BART、QA_approx，以及图像/文本子集。 |
| P0-4 / A4 EAB 与 ans1 | 已完成 | `src/omniaware/pipeline.py`，`src/omniaware/run.py`，`scripts/paper_runs.py` | `tests/test_v2_review.py` | 默认 `--variants` 不含 EAB。`--analysis-eab` 才加入 ans1。ans2 只作为 decode-only。 |
| P0-5 / A5 图像证据诊断 | 部分完成 | `scripts/analysis/image_evidence_check.py` | `tests/test_v2_review.py` | `--fixture` 可在无数据时跑。默认 50 题的真实 WebQA 诊断没有跑，GPU 描述命令只打印。 |
| P0-6 / A6 切分 | 已完成 | `src/omniaware/data.py`，`src/omniaware/run.py` | `tests/test_v2_splits.py`，`tests/test_splits.py` | WebQA 校准与开发来自 train，测试为全部 val。三者不交。空测试集报错。三份 qid 的 sha256 写入配置。 |
| P0-7 / A7 审查协议 | 部分完成 | `docs/REVIEW_V2_STATUS.md`，`scripts/paper_runs.py`，`src/omniaware/baselines/arms.py` | `tests/test_v2_review.py`，`tests/test_paper_runs.py` | F-1、F-2、S-4、S-5 已修。其余致命/严重条目见该状态表；标为不修的没有假装已修。 |
| F-1 VideoRAG 评测集为空 | 已完成 | `src/omniaware/data.py` | `tests/test_v2_splits.py` | 空 test 报错。默认 reserve 100、dev 50。 |
| F-2 M2-RAG 看不到问题图 | 已完成 | `src/omniaware/baselines/arms.py`，`src/omniaware/baselines/m2rag_webqa.py` | `tests/test_v2_review.py` | 问题图进入 summarizer 与 extractor。 |
| S-1 评测规模 | 已完成 | `src/omniaware/data.py`，`paper/sec_experiments.tex` | `tests/test_v2_splits.py` | 论文改为实际切分。 |
| S-2 测试集上选参与无来源数字 | 部分完成 | `paper/sec_appendix.tex`，`scripts/paper_runs.py` | `tests/test_dev_select.py` | 论文中的无来源数字已撤回。没有新的开发集网格。 |
| S-3 n_calib 改变 test | 已完成 | `src/omniaware/data.py` | `tests/test_splits.py` | test 不随 n_calib 移动。 |
| S-4 冷启动基线共用探测和路由 | 已完成 | `scripts/paper_runs.py`，`src/omniaware/run.py` | `tests/test_v2_review.py` | 主表基线用 joint top-k，不带 probe 与 first-round routing。 |
| S-5 SeaKR 只跑 1000 题 | 已完成 | `scripts/paper_runs.py` | `tests/test_v2_review.py`，`tests/test_paper_runs.py` | 主表 SeaKR 不传更小的 `--n-eval`。order 实验的 1000 题子集在 stage 说明里标明，不当成全量测试。 |
| S-6 UniversalRAG 静默退回 | 已完成 | `src/omniaware/baselines/universalrag.py`，`scripts/paper_runs.py` | 既有路由测试 | 空路由不再变成联合检索。OK-VQA 列没有可执行的路由器命令，stage 说明里写了。 |
| S-7 缺决定文件时用默认值 | 已完成 | `scripts/paper_runs.py`，`scripts/analysis/dev_select.py`，`src/omniaware/run.py` | `tests/test_paper_runs.py`，`tests/test_v2_review.py` | 缺文件或非法 JSON 直接失败。 |
| S-8 缓存指纹 | 已完成 | `src/omniaware/run.py`，`src/omniaware/runio.py` | `tests/test_cache_resume.py` | 审查时已在当前 main 中。本次未再改语义。 |
| S-9 半成品当已合并 | 已完成 | `scripts/launch.py`，`src/omniaware/runio.py` | 既有 launch 测试 | 同上，当前代码会拒绝没有完成标记的半成品。 |
| S-10 坏图中断抽取 | 已完成 | `scripts/setup/prepare_webqa_open.py` | 既有准备脚本测试 | 当前代码按批记录坏图。 |
| S-11 模型完成判定 | 已完成 | `scripts/setup/download_models.py` | 无新测试 | 当前代码检查权重分片。 |
| S-12 OK-VQA 三种语料的构建说明 | 部分完成 | `scripts/paper_runs.py` | `tests/test_paper_runs.py` | 缺候选文件时 dev 阶段失败。README 没有补上 BM25 语料的完整构建；该语料本仓库仍做不完。 |
| S-13 Punkt 下载 | 已完成 | `scripts/setup/download_models.py` | `tests/test_citation_intake.py` 仍会跳过 | 下载步骤在。本机没装 Punkt，所以测试跳过。 |
| S-14 缩小语料仍称 unified corpora | 部分完成 | `paper/sec_experiments.tex` | 无 | 代码仍用子集。论文已改写，不再把未跑的单元格写成结果。 |
| S-15 M2-RAG 像素预算 | 已完成 | `src/omniaware/baselines/arms.py` | `tests/test_v2_review.py` | 问题图与同一次运行的 `max_pixels` 会传进去。 |
| S-16 FiTM 温度 | 未做 | `src/omniaware/baselines/fitm_attention.py` | 无 | 温度仍是 5e-5。没有原文依据，没有改。 |
| S-17 真实权重的 chat template | 未做 | `src/omniaware/qwen_eab.py` | `tests/test_v2_signals.py` 只用小模型 | 禁止下载模型。 |
| S-18 EAB 选择偏差 | 已完成 | `scripts/analysis/dev_select.py` | `tests/test_dev_select.py` | 已是 max-statistic。EAB 不再是主方法。 |
| S-19 因果编辑 | 已完成 | `src/omniaware/interventions.py` | 既有 intervention 测试 | 当前 main 已修。本次未再改语义。 |
| S-20 引用指标不是门控答案 | 已完成 | `src/omniaware/eval/citation/attribution.py`，`src/omniaware/run.py` | 既有 citation 测试 | cite 仍是单独的 `--answer-style cite`。 |
| S-21 OK-VQA 全量语料 | 已完成 | `src/omniaware/okvqa_dense.py`，`scripts/analysis/dev_select.py` | `tests/test_okvqa_gs.py` | 默认 gs112k。含测试题的 full 语料不参与 dev 选择。 |
| B1 五个动作同批解码 | 已完成 | `src/omniaware/run.py`，`src/omniaware/signals.py` | `tests/test_v2_signals.py`，`tests/test_v2_review.py` | direct、static_text、static_image、static_mixed、rerank_static。没有 cross-encoder 时 rerank 用词面重叠，并在配置里写 `rerank_static_source`。 |
| B2 信号字段 | 已完成 | `src/omniaware/signals.py` | `tests/test_v2_signals.py` | CPU 小模型。熵非负，margin ≥ 0。`gold_in_topk` 只存储。 |
| B3 选择器与风险控制 | 已完成 | `src/omniaware/arbitration.py` | `tests/test_arbitration.py` | v2-Rule、7 参数 v2-Cal、按效用选择的固定序列 LTT。合成数据上的三项都在测试里。 |
| B4 预注册 | 已完成 | `docs/DEV_DECISIONS.md`，`scripts/analysis/dev_select.py` | `tests/test_arbitration.py`，`tests/test_dev_select.py` | 日期 2026-10-13。选择器提交 `9829fef`。测试阶段只读冻结决定。 |
| C 离线探索 | 已完成 | `scripts/analysis/offline_v2.py`，`runs/offline_v2/` | `tests/test_offline_v2.py` | 见第 2 节。不是论文主表。 |
| D S1/S2/S4 与门槛 | 部分完成 | `scripts/v2_runs.py`，`scripts/analysis/v2_gate.py`，`docs/V2_RUNS.md`，`README.md` | `tests/test_v2_runs.py`，`tests/test_v2_gate.py` | 只写脚本和干跑。没有 GPU 运行。`--split calib` 现在能被 `run.py` 接受。 |
| E 文档与论文 | 已完成 | `README.md`，`docs/PROTOCOL.md`，`docs/PAPER_TABLE_MAP.md`，`paper/*.tex` | 无 | 未跑实验的数字是 `\ph{}`。 |

## 2. 任务 C 摘要

命令：

```
python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2
```

退出码 0。全部数字都是探索性、旧管线、含已知 bug 的数据。

空答案：WebQA direct 410/4966（0.0826），always-static 447/4966（0.0900），都是精确的 `Sources: none`。方向文档 N1 的范围是每臂约 433–551。direct 低于下限，always-static 落在范围内。OK-VQA direct 与 static 都是 0/256，那份记录没有引用格式。`QA_approx` 不在 WebQA 文件里，没有补造。

评测半集上，风险控制退回最强单一动作，效用为 0，拿到的逐题选对上限比例为 0。

- WebQA：2483/2483。最强动作 `static_mixed`，acc_approx 0.413；逐题上限 0.545。v2-Cal 0.413，配对差 −0.0004，95% CI 跨过 0。
- OK-VQA：128/128。最强动作 direct，官方 VQA accuracy 58.59；逐题上限 60.63。v2-Cal 58.13，配对差 −0.47，95% CI 跨过 0。

WebQA 的 margin 是 `u_initial` 的代理，不是首 token logit。OK-VQA 的 margin 是 `-h_sem` 的代理。序列 log-prob 在这两份旧数据里不存在，特征填 0。

## 3. 没做或做不到

- 没有 GPU，没有下载模型或数据。S1、S2、S4 没有实际运行。开发集 +8% 门槛因此没有 PASS/FAIL。
- 50 道 WebQA 图像题的真实诊断没有跑。脚本的 GPU 描述探针只打印命令。
- WebQA 提示词没有在开发集上重选。注册表里是冻结的候选句。
- 归一化 BART 与 QA_approx 在没有 ParaBank / 相关权重时保持 null。
- FiTM 温度仍是 5e-5。真实权重上的 chat template 没有核对。
- 离线数字不能进入论文主表。

## 4. 提交

`git log --oneline main..HEAD`：

```
565cdab Accept the v2 run flags, including a calibration split decode.
9e8add3 Merge the v2 protocol fixes into v2/arbitration.
3e4135c Drop EAB from the main method, keep cold-start baselines on joint top-k, and write v2 action records (A4, A7).
541a872 Add a CPU WebQA image-evidence diagnostic and print the GPU probe without running it (A5).
e6c87b6 Record first-token entropy, margin, sequence logprob, and KL for each decoded action (B2).
ee0a7a8 Lock WebQA calibration, dev, and the full validation test split as pairwise disjoint (A6).
f3392df Score OK-VQA only with score_okvqa and emit WebQA acc_approx, normalized BART, and QA_approx (A3).
c1f698e Give every arm one frozen per-dataset instruction and record its sha256 (A2).
0714580 Merge the v2 arbitration selector into v2/arbitration.
5035ef5 Parse blank, citation-only, and refusal answers before they are scored (A1).
bb08c50 Merge the v2 protocol and paper draft into v2/arbitration.
9fec5cc Match the v2-Cal feature list to the seven-parameter selector.
982f97a Record the fixed L2 penalty used by the v2 calibrator (B4).
496f061 Merge the exploratory offline v2 check into v2/arbitration.
af80369 Score the old dumps with the v2 selector on the held-out half.
473f10e Preregister the v2 selector grid, utility, and go/no-go rule (B4).
9829fef Add the v2 arbitration selector and its dev selection step (B3).
3c96e4e Rewrite the paper draft around risk-controlled arbitration and retire EAB as analysis.
1513923 Map each v2 paper table to the command that produces it.
ff10ada Rewrite the experiment protocol around the v2 splits, metrics, and strong baseline.
777ae47 Document the v2 direction and withdraw historical-cache gains.
171ffc0 Record measured empty-answer rates and half-split sizes from the old dumps.
646f8cb Merge v2 GPU stage scripts into v2/arbitration.
1afcfa4 Add the exploratory offline v2 checker and its unit tests.
a773a06 Decode the calibration split in S2 so the selector has B2 signals.
27f07db Add S1, S2, and S4 dry-run GPU commands for task D.
a16ee0a Add the S2 dev-set oracle gap gate for task D.
```

新增或修改的主要文件：`src/omniaware/answers.py`，`prompts.py`，`signals.py`，`arbitration.py`，`run.py`，`pipeline.py`，`data.py`，`eval/score_official.py`，`baselines/arms.py`，`baselines/m2rag_webqa.py`，`scripts/v2_runs.py`，`scripts/analysis/v2_gate.py`，`scripts/analysis/offline_v2.py`，`scripts/analysis/image_evidence_check.py`，`scripts/analysis/dev_select.py`，`scripts/paper_runs.py`，`docs/PROTOCOL.md`，`docs/DEV_DECISIONS.md`，`docs/PAPER_TABLE_MAP.md`，`docs/V2_RUNS.md`，`docs/REVIEW_V2_STATUS.md`，`README.md`，`paper/main.tex`，`paper/sec_experiments.tex`，`paper/sec_appendix.tex`，`runs/offline_v2/`，以及上表里的测试。

## 5. pytest

`python -m pytest -q` 退出码 0。`pyproject.toml` 的 `addopts` 已经是 `-q`，所以这条命令不再打印 “N passed” 行。原样末尾：

```
C:\Users\22688\AppData\Roaming\Python\Python311\site-packages\requests\__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.1)/charset_normalizer (2.0.4) doesn't match a supported version!
  warnings.warn(
..........................................................s............. [ 19%]
..................................s............................... [ 37%]
........................................................................ [ 56%]
........................................................................ [ 75%]
........................................................................ [ 95%]
................s.                                                     [100%]
============================== warnings summary ===============================
tests/test_okvqa_gs.py::test_passages_have_no_prefix_and_questions_use_the_bge_instruction
  C:\Users\22688\Desktop\www_code\omniaware\src\omniaware\okvqa_dense.py:139: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:209.)
    block = torch.from_numpy(np.asarray(emb[s:s + chunk])).to(bge.device)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```

同一棵树、关掉 `addopts` 之后，`python -m pytest -o addopts= -q --tb=no` 的最后一行是：

```
369 passed, 5 skipped, 1 warning, 8 subtests passed in 16.76s
```

跳过原因（`python -m pytest -q -rs --tb=line`，同样退出码 0）：

- `tests/test_m2rag_qwen.py:5`：没有 `torchvision`
- `tests/test_webqa_acc.py:8`：没有 `spacy`
- `tests/test_citation_intake.py:33`：没有安装 NLTK Punkt 英文数据
- `tests/test_m2rag_official.py:81`：asset root 下没有 legacy T15 运行产物
- `tests/test_webqa_bart.py:105`：asset root 下没有官方 BARTScore 源和 tokenizer

## 6. 论文编译

在 `paper/` 中执行 `latexmk -pdf main.tex`，退出码 0。输出 `main.pdf`，8 页，559861 字节。该 PDF 被 `.gitignore` 忽略，没有提交。

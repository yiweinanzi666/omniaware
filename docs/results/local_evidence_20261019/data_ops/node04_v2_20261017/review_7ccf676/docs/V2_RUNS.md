# v2 GPU 阶段命令（S1 / S2 / S4）

## 数据集与设置（先读这里）

| 设置 | 数据集 | `--calib-source` | 角色 |
|---|---|---|---|
| 主设置 | `webqa_full`（WebQA distractor，每题自带正负事实池） | `train` | 主表。校准与开发来自 train，测试是全部 val（4966 题） |
| 第二设置 | `webqa_open_img`（WebQA 开放检索的图像子集，`data.IMAGE_SUBSET_OF`） | `train` | 开放检索，只有 S2 结果；S4 的测试结果不在本次计划内 |
| 第三设置 | `okvqa_gs`（GS-112K 检索语料的 OK-VQA） | `heldout` | 第三个数据集，校准 500 题来自保留区 |

所有命令传协议尺寸 `--n-calib 500 --n-dev 256 --split-seed 2027`，`--model 8b`。`okvqa`（BM25 维基候选）已从 v2 数据集中移除，见 `docs/DEV_DECISIONS.md`。

## 运行前提

- 在仓库根目录运行。`omniaware` 包需要可导入：`pip install -e ".[all]"`（README 第 162 行），否则 `python -m omniaware.*` 报 `ModuleNotFoundError`。
- 输出目录与缓存由 `OMNIAWARE_RUNS`、`OMNIAWARE_CACHE`、`OMNIAWARE_ASSETS` 决定（`src/omniaware/paths.py`），脚本打印的命令使用同一目录。
- Windows 下 `--dry-run` 打印的命令按 Windows 命令行规则加引号，可直接粘贴到 PowerShell 或 cmd。
- 本仓库不下载模型或数据。下面的时长全部是估计值，没有实测吞吐。

## 时长与成本（估计值）

| 阶段 | 内容 | GPU 时间（估计） |
|---|---|---|
| S1 | 每个数据集 50 题冒烟，五个 v2 动作 | 1 张 GPU，约半天 |
| S2 | 校准集与开发集（开发集带基线臂） | 1 张 A100，约 4–8 小时（8B 模型短答案；基线臂见下） |
| S4 | 全量测试集，带基线臂，每个数据集只跑一次 | 1–2 张 A100，约 1–2 天 |

### 基线臂的每题额外成本（S2 开发集与 S4 运行）

S2 开发集与 S4 运行带 `--fitm --m2rag --seakr-samples 20 --analysis-eab`。除 v2 动作本身的约 4 次额外证据解码外，每题的额外前向如下（计数，未计时）：

- FiTM：每题一次额外的 greedy 生成，加一次标定前向。
- M2-RAG：每题两次生成，一次官方摘要器（最多 384 词元）和一次答案抽取器。不适用于 VideoRAG。
- SeaKR（`--seakr-samples 20`）：直答的特征值需要 20 次采样续写（每次最多 100 词元）。当 SeaKR 判定检索时，还要对联合 top-5 的每个块做 1 次贪心解码加 20 次采样，即再 5 × 21 = 105 次。每题的期望额外采样数约为 20 + 105 × r，r 为 SeaKR 的检索率（未知，需用开发集实测）。上限为每题 125 次。
- EAB 分析臂（`--analysis-eab`，DEV_DECISIONS 2026-10-15 修订）：门控打开（U > τ_b）的题上，多一次与 `static` 共享证据前缀的贪心解码；门控关闭时直接复用 `static` 的结果。它只由 `score_official` 评分，`v2_table.py` 的各步不读它。
- SeaKR 的阈值在校准集上匹配 U 的检索率，因此 S2 开发集运行另外要对 500 道校准题各做 20 次采样（一次性，写入缓存）。

算力不足时，给 `scripts/v2_runs.py` 加 `--no-seakr`：S2 开发集与 S4 都不再带 SeaKR。该开关会在输出中打印一条 NOTE，说明主表缺 SeaKR 行。S1 不带基线臂，不接受该开关。

## v2 选项

每条 `omniaware.run` 命令都带 `--v2 --v2-reranker`：

- `--v2`：每题解码五个动作（`direct`、`static_text`、`static_image`、`static_mixed`、`rerank_static`），写入 `rec["actions"]`。每个证据动作记录送入模型的块 id（`chunks`）。
- `--v2-reranker`：只为 `rerank_static` 加载 cross-encoder 分数。`rerank_static` 与 CE 基线（`--selector ce`）选择并排序的证据完全相同。选择器保持 relevance，`static_mixed` 仍是按检索相关度的联合 top-k；其他臂不变。

`--v2` 的约束（`run.check_v2_args`）：不能与 `--selector ce` / `ce_calibrated` 同用；`--k` 必须为 5；不能用 `--order`（动作的证据顺序固定为相关度顺序）。候选池为空的模态动作记录为 `"available": false`，不解码，不复用 direct 的结果。

`rec["preds"]` 保存模型原始输出，解析结果（`answer`、`answer_empty`、`citation_only`）在 `rec["parsed"]`。空答案率在 `<dataset>.merged.json` 里覆盖 preds 与可用动作，多卡合并路径同样写出。

## 溯源记录

每次运行在输出目录里自动写出：`<dataset>.env.ndjson`（每次启动一行：git 提交号、库版本、GPU、注意力实现、dtype、命令行）、`<dataset>.log`（全部屏幕输出与异常堆栈，续跑时追加）、`<dataset>.prompts.ndjson`（`--v2` 下前 N 道题每个动作的完整 prompt，N 由 `--trace-prompts` 控制，默认 20，0 关闭）。分片运行时这三个文件名带 `shardXX-of-NN`。

`--v2` 下逐题记录另含 `candidate_scores`（全部候选的相关度、重排分数、gold 标记），每个动作记录另含 `input`（prompt 的 token 数与哈希、图片张数、每张图的视觉 token 数）与 `decode`（token id、逐 token 对数概率、耗时、是否到达长度上限）。决定文件的 `trace` 字段记录选择器检验过的每个候选。选择器不读取这些字段。各字段的含义见 [HANDOFF_v2.md](HANDOFF_v2.md) 的“每次运行会留下什么”。

## 部署路径（S4）

S4 带 `--decisions` 时，除了照常解码五个动作，还会把冻结的 v2-Rule 按部署方式再跑一遍：先对直答的 prompt 做一次前向（不生成）；直答的首 token 领先幅度达到阈值就只解码直答；否则，“校准集上最好的证据动作”这一规则直接解码该动作，“当前题上最自信的证据动作”这一规则先对每个可用的证据动作各做一次前向再解码被选中的那个。每题最多 5 次前向、恰好 1 次解码。结果写在逐题记录的 `v2_deploy` 字段，`v2_table.py apply` 的输出里有汇总（`deployment`）。v2-Cal 要读解码后的答案，没有这种运行方式。论文的成本表用这条路径的计数。

## 运行锁

同一输出目录（tag）内，每个数据集（或每个分片）在运行时持有一个锁文件 `<dataset>.lock`（分片为 `<dataset>.shardXX-of-NN.lock`）。锁文件记录进程号。另一个仍在运行的进程占用同一锁时，新进程拒绝启动并给出原因。进程崩溃后残留的锁会被识别为失效并自动接管。

## 冻结决定

S2 的 `select` 为每个数据集写一个决定文件：`runs/v2_decisions/<dataset>.json`（实际路径在 `OMNIAWARE_RUNS` 下）。一个文件对应一个数据集，避免三个数据集互相覆盖。S4 用 `--decisions` 指定该文件。文件缺失、不是 JSON 或根节点不是对象时，S4 直接失败，不用默认超参代替。

## 命令

### S1 冒烟（估计值：1 张 GPU，约半天）

```
python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50 --dry-run
python scripts/v2_runs.py --stage s1 --dataset webqa_open_img --n 50 --dry-run
python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --n 50 --dry-run
```

每个数据集打印一条 `omniaware.run` 命令（`--split dev --n-eval 50 --max-empty-rate 0.005`，`--tag v2_s1_<数据集>`）和一条格式检查 `python scripts/analysis/answer_slot.py format --run runs/v2_s1_<数据集>/<数据集>.jsonl --min-answer-first 0.9`（WebQA 的先答案格式：任一动作遵守率低于 90% 时以 `answer-first format FAIL` 结束），再打印图像诊断命令 `python scripts/analysis/image_evidence_check.py --n 50`。`--max-empty-rate 0.005` 使空答案率超过 0.5% 的冒烟运行以非零退出码结束；空答案率只算空白与纯引用输出，拒答单独记为 `refusal_rate` 和按臂的 `empty_by_arm`，不计入这一门槛。诊断命令只打印，本脚本不执行它。

### S2 校准集与开发集（估计值：1 张 A100，约 4–8 小时）

```
python scripts/v2_runs.py --stage s2 --dataset webqa_full --dry-run
python scripts/v2_runs.py --stage s2 --dataset webqa_full --no-seakr --dry-run
```

每个数据集按执行顺序打印命令：

1. 解码校准集：`--split calib --tag v2_s2_<数据集>_calib`
2. 解码开发集：`--split dev --tag v2_s2_<数据集>_dev --fitm --m2rag --seakr-samples 20 --analysis-eab`（`--no-seakr` 时去掉 `--seakr-samples 20`）
3. 评分：`python -m omniaware.eval.score_official runs/<tag>/<数据集>.jsonl`（两个 tag 各一条；WebQA 带 `--bart`）
4. 拼表：`python scripts/analysis/v2_table.py build --run runs/<tag>/<数据集>.jsonl`（两个 tag 各一条；WebQA 带 `--bart`），生成 `runs/<tag>/<数据集>.v2_actions.jsonl`
5. 开发集判据：先运行 `python scripts/analysis/answer_slot.py position --table runs/v2_s2_<数据集>_dev/<数据集>.v2_actions.jsonl`（第一个生成 token 与答案 token 上直答置信度的 AUROC，写到同目录的 `*.answer_slot.json`）；加 `--position-ablation` 时（只限 WebQA），再用 `--answer-style sentence` 在同一批开发题上解码、评分、拼表并做同样的分析（tag `v2_s2_<数据集>_dev_sentence`）。然后 `python scripts/analysis/v2_table.py gate --table runs/v2_s2_<数据集>_dev/<数据集>.v2_actions.jsonl`
6. 校准选择：`python scripts/analysis/v2_table.py select --table runs/v2_s2_<数据集>_calib/<数据集>.v2_actions.jsonl --out runs/v2_decisions/<数据集>.json`
7. 开发集检查：`python scripts/analysis/v2_table.py devcheck --table runs/v2_s2_<数据集>_dev/<数据集>.v2_actions.jsonl --decisions runs/v2_decisions/<数据集>.json --out runs/v2_decisions/<数据集>.dev_check.json`。把冻结决定应用到开发动作表（结果写到 `runs/v2_s2_<数据集>_dev/<数据集>.v2_dev_main.json`），主选择器相对最强单一动作 a\* 的相对提升至少 +5%，且配对 bootstrap 95% CI 下界大于 0，才输出 PASS。检查文件记录决定文件的 sha256。

不生成测试集，不传 `--n-eval`。`--n-calib` 只决定校准题的 id，所以校准集必须单独解码。非 dry-run 时，任一步非零退出即停止并打印原因；步骤 5 的 gate 失败（退出码 1）时打印 `gate FAIL`，不执行步骤 6、7；步骤 7 失败（退出码 1）时打印 `development check FAIL`，决定文件和检查文件都已写出。

### S4 冻结测试（估计值：1–2 张 A100，约 1–2 天）

```
python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions runs/v2_decisions/webqa_full.json --dry-run
```

打印四条命令：

1. 解码测试集：`--split test --tag v2_s4_<数据集> --fitm --m2rag --seakr-samples 20 --analysis-eab --decisions <决定文件>`
2. 评分：`python -m omniaware.eval.score_official runs/v2_s4_<数据集>/<数据集>.jsonl`（WebQA 带 `--bart`）
3. 拼表：`python scripts/analysis/v2_table.py build --run runs/v2_s4_<数据集>/<数据集>.jsonl`（WebQA 带 `--bart`）
4. 应用冻结决定并出主表：`python scripts/analysis/v2_table.py apply --table runs/v2_s4_<数据集>/<数据集>.v2_actions.jsonl --decisions <决定文件> --out runs/v2_s4_<数据集>/<数据集>.v2_main.json`

没有 `--decisions`，或文件不存在、不是 JSON 对象时，脚本以非零退出码结束，标准错误写明原因，不解码测试集。决定文件旁边必须有 S2 写出的 `<决定文件名>.dev_check.json`，且其中记录的 sha256 与决定文件一致；检查文件缺失、对不上，或记录为未通过时，S4 同样拒绝运行。协议允许在开发集检查未通过时仍跑一次冻结测试并照实报告：这时加 `--allow-failed-dev-check`，脚本在命令前打印一行 `NOTE`，写明未通过的数值。

## go/no-go 与主表的完整性

`v2_table.py gate` 在开发集上判断：逐题选择的上限相对最强单一动作是否至少 +8%。判据的实现与输出格式见 `scripts/analysis/v2_table.py`。不通过则不进入校准选择与 S4。

`v2_table.py devcheck` 是进入测试前的第二道判据（协议第 5 节）：冻结后的主选择器在开发集上相对 a\* 至少 +5%，且 CI 下界大于 0。上限判据只说明有空间，这一条检查选择器实际拿到了多少。

`score_official` 核对运行的题集与配置（`check_question_set`）。`v2_table.py` 的 `build` 与 `apply` 的完整性检查见该脚本的说明。

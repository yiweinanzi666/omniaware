# v2 GPU 阶段命令（S1 / S2 / S4）

`scripts/v2_runs.py` 只编排命令。`--dry-run` 把命令行打到标准输出，不写文件，不启动模型。去掉 `--dry-run` 时，`omniaware.run` 命令按顺序执行，之后的评分与 `v2_table` 命令也按顺序执行；S1 的图像诊断命令只打印，不执行。命令用当前解释器（`sys.executable`）执行，相对路径，需在仓库根目录运行。本机不跑 GPU，下面的时长全部是估计值，没有实测吞吐。

## 时长（估计值）

抄自方向文档第 6 节。S1 之后才按实测吞吐修正。

| 阶段 | 内容 | GPU 时间 |
|---|---|---|
| S1 | 每个数据集 50 题冒烟，五个动作，外加图像诊断命令 | 1 张 GPU，约半天（估计值） |
| S2 | WebQA distractor、`webqa_open_img`、OK-VQA（`okvqa_gs`）的校准集与开发集 | 1 张 A100，约 4–8 小时（估计值，8B 模型短答案） |
| S4 | 全量测试集，每个数据集只跑一次 | 1–2 张 A100，约 1–2 天（估计值） |

## 数据集与切分

| 数据集 | `--calib-source` | 说明 |
|---|---|---|
| `webqa_full` | `train` | distractor 设置；校准与开发来自 train，测试是全部 val |
| `webqa_open_img` | `train` | 开放检索的图像子集（`data.IMAGE_SUBSET_OF`） |
| `okvqa_gs` | `heldout` | GS-112K 检索语料；没有 train loader，校准 500 题来自保留区 |

所有命令传协议尺寸 `--n-calib 500 --n-dev 256 --split-seed 2027`，`--model 8b`。`okvqa`（BM25 维基候选）已从 v2 数据集中移除，见 `docs/DEV_DECISIONS.md`。

## v2 选项

每条 `omniaware.run` 命令都带 `--v2 --v2-reranker`：

- `--v2`：每题解码五个动作（`direct`、`static_text`、`static_image`、`static_mixed`、`rerank_static`），写入 `rec["actions"]`。
- `--v2-reranker`：只为 `rerank_static` 加载 cross-encoder 分数。选择器保持 relevance，`static_mixed` 仍是按检索相关度的联合 top-k；其他臂不变。

`--v2` 不能与 `--selector ce` / `ce_calibrated` 同用，那会把联合 top-k 换成 CE 选法。候选池为空的模态动作记录为 `"available": false`，不解码，不复用 direct 的结果。

`rec["preds"]` 保存模型原始输出，解析结果（`answer`、`answer_empty`、`citation_only`）在 `rec["parsed"]`。空答案率在 `<dataset>.merged.json` 里覆盖 preds 与可用动作，多卡合并路径同样写出。

## 冻结决定

S2 的 `select` 为每个数据集写一个决定文件：`runs/v2_decisions/<dataset>.json`。一个文件对应一个数据集，避免三个数据集互相覆盖。S4 用 `--decisions` 指定该文件。文件缺失、不是 JSON 或根节点不是对象时，S4 直接失败，不用默认超参代替。

## 命令

### S1 冒烟（估计值：1 张 GPU，约半天）

```
python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50 --dry-run
python scripts/v2_runs.py --stage s1 --dataset webqa_open_img --n 50 --dry-run
python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --n 50 --dry-run
```

每个数据集打印一条 `omniaware.run` 命令（`--split dev --n-eval 50`，`--tag v2_s1_<数据集>`），再打印图像诊断命令 `python scripts/analysis/image_evidence_check.py --n 50`。诊断只用于 WebQA 图像题（任务 A5），本脚本不执行它。

### S2 校准集与开发集（估计值：1 张 A100，约 4–8 小时）

```
python scripts/v2_runs.py --stage s2 --dataset webqa_full --dry-run
```

每个数据集打印 8 条命令，按执行顺序：

1. 解码校准集：`--split calib --tag v2_s2_<数据集>_calib`
2. 解码开发集：`--split dev --tag v2_s2_<数据集>_dev`
3. 评分：`python -m omniaware.eval.score_official runs/<tag>/<数据集>.jsonl`（两个 tag 各一条）
4. 拼表：`python scripts/analysis/v2_table.py build --run runs/<tag>/<数据集>.jsonl`（两个 tag 各一条），生成 `runs/<tag>/<数据集>.v2_actions.jsonl`
5. 开发集判据：`python scripts/analysis/v2_table.py gate --table runs/v2_s2_<数据集>_dev/<数据集>.v2_actions.jsonl`
6. 校准选择：`python scripts/analysis/v2_table.py select --table runs/v2_s2_<数据集>_calib/<数据集>.v2_actions.jsonl --out runs/v2_decisions/<数据集>.json`

不生成测试集，不传 `--n-eval`。`--n-calib` 只决定校准题的 id，所以校准集必须单独解码。

### S4 冻结测试（估计值：1–2 张 A100，约 1–2 天）

```
python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions runs/v2_decisions/webqa_full.json --dry-run
```

打印四条命令：

1. 解码测试集：`--split test --tag v2_s4_<数据集> --decisions <决定文件>`
2. 评分：`python -m omniaware.eval.score_official runs/v2_s4_<数据集>/<数据集>.jsonl`
3. 拼表：`python scripts/analysis/v2_table.py build --run runs/v2_s4_<数据集>/<数据集>.jsonl`
4. 应用冻结决定并出主表：`python scripts/analysis/v2_table.py apply --table runs/v2_s4_<数据集>/<数据集>.v2_actions.jsonl --decisions <决定文件> --out runs/v2_s4_<数据集>/<数据集>.v2_main.json`

没有 `--decisions`，或文件不存在、不是 JSON 对象时，脚本以非零退出码结束，标准错误写明原因，不解码测试集。

## S2 go/no-go

`v2_table.py gate` 在开发集上判断：逐题选择的上限相对最强单一动作是否至少 +8%。判据的实现与输出格式见 `scripts/analysis/v2_table.py`，不通过则不进入 S4。

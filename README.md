# Omni-Aware

Omni-Aware 是一个多模态 RAG 方法，用不确定性门控决定是否检索，并在检索到的证据上施加 Evidence Attention Bias（EAB）。本仓库是它唯一的实现与实验代码，从旧仓库 `EMNLP-reproduction` 整理而来（见 [docs/LEGACY_MAP.md](docs/LEGACY_MAP.md)）。

> **当前状态：** 本仓库**还没有任何正式结果**。旧仓库与论文 V3.20 中的数字不能当作本仓库的结果，原因见 `../Omni-Aware_全面审查报告_20261010.md`。今后所有结果都必须按 [docs/PROTOCOL.md](docs/PROTOCOL.md) 重新跑出来。

## 规范

先读 [CONVENTIONS.md](CONVENTIONS.md)。要点：
- 代码只有一份，不复制源码；
- 只用官方评测；
- 所有对照臂在同一输入下生成；
- calib/dev/test 三者分开；
- 每个数字都能重算。

## 目录

```
src/omniaware/      方法实现（数据、检索打分、Qwen3-VL + EAB、控制器、运行）
src/omniaware/eval/ 官方评测入口 score_official.py
third_party/        原样保存的官方评测脚本
scripts/data_prep/  资产准备脚本（OK-VQA 知识库与 BM25、Video-MME 帧、WebQA 图片）
tests/              CPU 测试（包括在随机初始化的小 Qwen3-VL 中验证 EAB 钩子）
docs/               实验协议、新旧代码对照
```

## 安装

```bash
pip install -e ".[data,webqa,dev]"
python -m spacy download en_core_web_sm
```

模型、数据和图片缓存放在资产目录下（布局与旧仓库的 `experiments/eab_first_token` 相同，可以直接指向它）：

```bash
export OMNIAWARE_ASSETS=/path/to/experiments/eab_first_token
```

## 运行与评分

一次运行会对每道题生成 direct、static、EAB 变体，以及 gated 系列的全部对照臂：

```bash
python -m omniaware.run --dataset okvqa --split dev --calib-source heldout --model 8b --tag okvqa_dev_v1
python -m omniaware.eval.score_official runs/okvqa_dev_v1/okvqa.jsonl
```

各数据集的常用参数：

| 数据集 | 示例 |
|---|---|
| OK-VQA | `--dataset okvqa --calib-source heldout --variants eab:g1.2:ans2:pos:gate,eab:g1.2:ans1:pos:gate` |
| MMQA 单表 | `--dataset mmqa_table --calib-source train --mmqa-provided-context --modality-routing --answer-style json_list` |
| Video-MME Short | `--dataset videomme --calib-source heldout --video-frames-dir $OMNIAWARE_ASSETS/data/videomme_short_frames32 --subtitle-words 0 --answer-style strict_letter --variants eab:g1.2:ans1:pos:gate` |
| WebQA（distractor 设置） | `--dataset webqa_img`（或 `webqa_full`）`--calib-source train`；评分时加 `--bart` 得到 acc_approx × BARTScore |

评分会生成 `<dataset>.official_summary.json`，内容包括每个臂的均值，以及与 static、direct 的配对 bootstrap 比较，并写出逐题分数。

## 测试

```bash
python -m pytest
```

需要资产的测试（MMQA 官方评测、BARTScore、spacy）在缺少资产时会自动跳过。其余测试都能在 CPU 上运行，不需要模型权重。

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
src/omniaware/                 方法实现（数据、检索打分、Qwen3-VL + EAB、控制器、运行）
src/omniaware/rerank.py        cross-encoder 选证
src/omniaware/order.py         顺序实验与复制对照
src/omniaware/baselines/       DRAGIN、Found-in-the-Middle、M2-RAG（模块与测试，尚未接入 run.py）
src/omniaware/eval/            官方答案评测入口 score_official.py
src/omniaware/eval/citation/   CiteRec 引用支持度
third_party/                   原样保存的官方评测脚本
scripts/data_prep/             资产准备（OK-VQA 知识库与 BM25、完整 DPR 索引、Video-MME 8/32 帧、WebQA 图片）
scripts/webqa_open_retrieval/  WebQA 全语料开放检索（54.4 万文本 + 39.0 万图片，BGE/CLIP + FAISS）
scripts/analysis/              统计功效、证据是否包含答案
tests/                         CPU 测试（包括在随机初始化的小 Qwen3-VL 中验证 EAB 钩子和 FITM）
docs/                          实验协议、新旧代码对照、引用支持度协议
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

一次运行会对每道题生成 direct、static、各 EAB 变体，以及 gated 系列的全部对照臂：

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
| WebQA（全语料开放检索） | 先生成检索结果（见下），再用 `--dataset webqa_open_img`（或 `webqa_open_full`） |

可选对照臂与选项（对所有臂同时生效，或增加一个对照臂）：

| 选项 | 作用 |
|---|---|
| `--replication-gamma 1.2` | 增加复制对照臂 `replication`：每块重复 ⌈exp(γ·U·r)⌉ 份 |
| `--order perm:<id>` | 对所有题使用同一个固定排列（`sorted`、`reverse`、`s17-p1`…，共 14 个，见 `order.py`） |
| `--selector ce` / `ce_calibrated` | 用 bge-reranker 选 top-5，后者按模态 CE 百分位选；EAB 的 r_j 随之改用 CE 百分位 |
| `--answer-style cite` | 证据编号为 `[E1]..[Ek]`，要求带引用作答，用于 CiteRec |

评分会生成 `<dataset>.official_summary.json`，内容包括每个臂的均值，以及与 static、direct 的配对 bootstrap 比较，并写出逐题分数。

### WebQA 全语料开放检索

实验主机上已经建好完整的 BGE 与 CLIP 索引（`OMNIAWARE_WEBQA_OPEN_DIR`，默认为旧目录 `runs/phase1_dense_retrieval_codex_20261006`）。为 val 生成检索结果需要两步（Linux）：

```bash
python -m scripts.webqa_open_retrieval.encode_webqa_queries --split val
python -m scripts.webqa_open_retrieval.search_dense_queries --index $OMNIAWARE_WEBQA_OPEN_DIR/faiss_bge_full --queries $OMNIAWARE_WEBQA_OPEN_DIR/webqa_val_queries_cpu --output $OMNIAWARE_WEBQA_OPEN_DIR/dense_top5_full/webqa_val/bge
```

把 `bge` 换成 `clip` 再运行一次第二条命令，就得到图像检索结果。如果要用训练集校准，对 `--split train` 重复以上步骤。

### 引用支持度（CiteRec）

```bash
python -m omniaware.run ... --answer-style cite --max-new-tokens 128 --tag cite_v1
python -m omniaware.eval.citation.from_run runs/cite_v1/okvqa.jsonl --output runs/cite_v1/citerec
```

之后依次运行 `intake_answers` → `run_nli` → `score_nli`，命令写在 `from_run.py` 的说明里。规则见 [docs/citation_support_protocol.md](docs/citation_support_protocol.md)。

### 分析

```bash
python -m scripts.analysis.power_design runs/x/okvqa.official_scores.jsonl --baseline static --candidate eab:g1.2:ans2:pos:gate --metric vqa_accuracy
python -m scripts.analysis.answer_recall runs/x/okvqa.jsonl
```

在决定 dev 集大小和候选方案数量之前，先看 `power_design` 的最小可检测效应和 best-of-K 噪声。

## 测试

```bash
python -m pytest
```

需要资产的测试（MMQA 官方评测、BARTScore、spacy、torchvision、官方 M2-RAG 与 DRAGIN 源码、NLI 资产）在缺少资产时会自动跳过。其余测试都能在 CPU 上运行，不需要模型权重。

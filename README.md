# Omni-Aware

Omni-Aware 是一个面向多模态 Web 问答的检索控制方法。它在证据 span 上做一次零偏置 prefill：已有输入证据时直接用，冷启动时先取一轮"探测"检索。由这一次 prefill 同时决定三件事：

- 是否检索：由不确定性 U_omni 决定；
- 去哪个模态检索：由 M_m·H_m 决定；
- 如何使用证据：在所有预测答案 token 的注意力行上，按相关度给证据加偏置，即 Evidence Attention Bias（EAB）。

本仓库是它唯一的实现与实验代码；论文 LaTeX 源码（V3.21）在 [`paper/`](paper/)，用 `cd paper && latexmk -pdf main.tex` 编译。

> **当前状态：** 本仓库**还没有任何正式结果**。旧仓库和论文 V3.20 中的数字都不能当作结果，原因见 `../Omni-Aware_全面审查报告_20261010.md`。所有结果都必须按 [docs/PROTOCOL.md](docs/PROTOCOL.md) 与 [docs/DEV_DECISIONS.md](docs/DEV_DECISIONS.md) 重新产生。

快速导航：[1 环境](#1-环境) · [2 资产目录](#2-资产目录与环境变量) · [3 模型](#3-下载模型) · [4 数据集与-benchmark](#4-数据集与-benchmark) · [5 实验](#5-实验多卡) · [6 评分与分析](#6-评分与分析) · [7 论文表格](#7-论文表格与代码的对应) · [8 测试](#8-测试)

## 目录

```
src/omniaware/
  run.py               一次运行：同一输入下解码所有对照臂，写逐题 JSONL；支持 --num-shards/--shard-id 多卡分片
  merge_shards.py      合并分片输出
  qwen_eab.py          Qwen3-VL / InternVL3.5 推理与 EAB 注意力钩子
  pipeline.py          提示词、U 的组合与归一化、EAB 变体
  routing.py           模态路由 M_m·H_m 与各消融规则
  relevance.py         BGE / CLIP 打分与按模态 ECDF 校准
  data.py              各数据集加载器与 calib/dev/test 划分
  okvqa_dense.py       OK-VQA 稠密检索（GS-112K 或 DPR 维基）
  rerank.py            cross-encoder 选证
  order.py             顺序实验与复制对照
  interventions.py     干预实验与相关度噪声
  analysis.py          ARD、AURC、Spearman
  baselines/           SeaKR、Found-in-the-Middle、M2-RAG、UniversalRAG
  eval/                官方评测入口 score_official.py；UniversalRAG 套件指标；citation/ 引用指标
scripts/
  setup/               新服务器准备：模型下载、OK-VQA、WebQA 开放检索、资产检查
  universalrag_retrieval/  UniversalRAG 套件：子集语料、候选池导出、视频帧采样
  launch.py            把一条 omniaware.run 命令分到多张 GPU 上
  paper_runs.py        论文全部实验命令（dev → 冻结 → test），支持 --gpus
  analysis/            论文各分析表、dev 决策、路由指标、统计功效
  fetch_external.py    下载并固定版本的官方仓库（external/）
  data_prep/, webqa_open_retrieval/   旧主机上的资产构建脚本（仅作记录，新服务器用 scripts/setup/）
third_party/           原样保存的官方评测脚本
tests/                 CPU 测试（小模型验证 EAB 钩子，合成数据跑通全流程）
docs/                  实验协议、dev 决策规则、论文表格映射、新旧代码对照
paper/                 论文 LaTeX 源码（V3.21，ACM sigconf）与各版修改说明
```

## 1. 环境

Linux 加 NVIDIA GPU。一个 8B 模型在 bf16 下约需 20 GB 显存，24 GB 及以上的卡每卡一个进程，80 GB 的卡可以每卡 2～3 个进程（`--procs-per-gpu`）。注意力使用 SDPA，不需要 FlashAttention：FlashAttention 会忽略 EAB 用的非二值 mask。

```bash
conda create -n omniaware python=3.11 -y && conda activate omniaware
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按本机 CUDA 版本选择
pip install -e ".[all]"                         # transformers>=4.57（Qwen3-VL、InternVL3.5-HF 需要）及全部可选依赖
python -m spacy download en_core_web_sm
python scripts/fetch_external.py                # 官方仓库（固定 commit）-> external/：SeaKR、M2RAG、UniversalRAG、BARTScore、WebQA、VQA ...
python -m pytest                                # CPU 上约 10 秒
```

UniversalRAG 的特征提取和路由器训练需要它自己的依赖（VLM2Vec、Qwen3-Embedding）。建议在 `external/UniversalRAG` 中按其 README 用 `uv sync` 另建一个环境，只在 4.3 节使用。

## 2. 资产目录与环境变量

所有大文件都放在一个资产根目录下，在 `~/.bashrc` 中设置：

```bash
export OMNIAWARE_ASSETS=/data/omniaware_assets      # 模型、数据集、检索产物
export OMNIAWARE_RUNS=$PWD/runs                     # 可选：运行输出（默认 ./runs）
export OMNIAWARE_CACHE=$PWD/cache                   # 可选：相关度、校准等缓存（默认 ./cache）
export HF_HOME=/data/hf_cache                       # 可选
```

| 变量 | 默认位置 | 内容 |
|---|---|---|
| `OMNIAWARE_ASSETS` | `./assets` | `cache/<模型>`、`data/...` |
| `OMNIAWARE_WEBQA_OPEN_DIR` | `$OMNIAWARE_ASSETS/webqa_open` | WebQA 全池、图片、编码、检索结果 |
| `OMNIAWARE_UNIVERSALRAG_DIR` | `external/UniversalRAG` | UniversalRAG 仓库及其 `dataset/` |
| `OMNIAWARE_UNIVERSALRAG_RANKINGS` | `$OMNIAWARE_ASSETS/universalrag/rankings` | 导出的共享候选池 |
| `OMNIAWARE_UNIVERSALRAG_FRAMES` | `$OMNIAWARE_ASSETS/universalrag/frames` | 视频与片段的采样帧 |
| `ERC_QWEN_8B`、`OMNIAWARE_INTERNVL_8B`、`ERC_BGE`、`ERC_CLIP`、`ERC_RERANKER`、`OMNIAWARE_BARTSCORE` | `$OMNIAWARE_ASSETS/cache/...` | 单独覆盖某个模型的位置 |

完整路径定义见 `src/omniaware/paths.py`。任何时候都可以用下面的命令检查缺了什么：

```bash
python -m scripts.setup.check_assets
```

## 3. 下载模型

```bash
python -m scripts.setup.download_models --list      # 列出模型、版本与用途
python -m scripts.setup.download_models             # 全部（约 45 GB）；可重复执行，已下载的跳过
# 国内网络：HF_ENDPOINT=https://hf-mirror.com python -m scripts.setup.download_models
```

| 名称 | 来源 | 用途 |
|---|---|---|
| qwen8b | Qwen/Qwen3-VL-8B-Instruct | 主骨干 |
| qwen2b | Qwen/Qwen3-VL-2B-Instruct | 冒烟测试、快速调试 |
| internvl | OpenGVLab/InternVL3_5-8B-HF | 骨干迁移 |
| bge | BAAI/bge-large-en-v1.5（固定 revision） | 文本相关度；WebQA、OK-VQA 文本检索 |
| clip | openai/clip-vit-large-patch14 | 图像相关度；WebQA 图像检索 |
| reranker | BAAI/bge-reranker-v2-m3 | cross-encoder 对照 |
| nli | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli@b3546ea | 引用支持度检查器（按 manifest 校验哈希） |
| bartscore | BARTScore ParaBank 权重（Google Drive，需 gdown）+ facebook/bart-large-cnn 配置 | WebQA QA_approx 的流畅度因子 |

## 4. 数据集与 benchmark

论文使用 4 个 benchmark：WebQA（开放检索）、OK-VQA、HybridQA、VideoRAG-Synth。下面每节都给出下载来源、准备命令和产物。

### 4.1 WebQA（开放检索：全池约 54 万文本、39 万图片）

1. 从 [WebQA 官方仓库](https://github.com/WebQnA/WebQA) 的 Google Drive 下载 `WebQA_train_val.json`、`WebQA_test.json` 和图片分卷 `imgs.7z.001…`。用 `7z x imgs.7z.001` 解出 `imgs.tsv`。
2. 把 `WebQA_train_val.json` 放到 `$OMNIAWARE_ASSETS/data/WebQA_train_val.json`，然后构建全池、编码并检索：

```bash
W=/data/webqa                                            # 含 WebQA_train_val.json、WebQA_test.json、imgs.tsv
python -m scripts.setup.prepare_webqa_open pool   --webqa-dir $W
python -m scripts.setup.prepare_webqa_open images --webqa-dir $W --workers 32
python -m scripts.setup.prepare_webqa_open encode --kind text  --gpus auto     # BGE，按 GPU 分片
python -m scripts.setup.prepare_webqa_open encode --kind image --gpus auto     # CLIP，按 GPU 分片
python -m scripts.setup.prepare_webqa_open search --split val
python -m scripts.setup.prepare_webqa_open search --split train                # 训练集问题用于无标签校准
```

产物格式与加载器 `webqa_open_full` / `webqa_open_img` 一致。检索是对全部归一化向量的精确内积 top-5，结果等同 FAISS `IndexFlatIP`。

### 4.2 OK-VQA（val 5,046 题）与 GS-112K 检索语料

```bash
python -m scripts.setup.prepare_okvqa --download        # 官方问题与标注 + COCO val2014 图片（约 6.5 GB）
# 已有 COCO：python -m scripts.setup.prepare_okvqa --coco-dir /data/coco/val2014（问题与标注 zip 仍需下载）
```

GS-112K（Luo et al., EMNLP 2021，OK-VQA 文献常用的 Google Search 语料）：从其官方发布（[luomancs/retriever_reader_for_okvqa](https://github.com/luomancs/retriever_reader_for_okvqa)）下载完整测试语料 CSV（168,306 条），然后：

```bash
python -m omniaware.okvqa_dense --corpus gs112k prepare --csv okvqa_full_corpus.csv
python -m omniaware.okvqa_dense --corpus gs112k encode
python -m omniaware.okvqa_dense --corpus gs112k search --n-passages 168306 --k 20     # -> 数据集 okvqa_gs
```

dev 网格里的另外两种 OK-VQA 语料（BM25 维基 `okvqa`、BGE 维基 `okvqa_dense`）需要 DPR 维基语料，属于可选。没有构建时，`paper_runs` 会跳过它们并给出提示。

### 4.3 HybridQA 与 VideoRAG-Synth（UniversalRAG 套件）

在 UniversalRAG 的环境中（见第 1 节）：

```bash
cd external/UniversalRAG
(cd dataset/HybridQA && bash get_hybridqa.sh)          # Web 表格 + 链接段落
(cd dataset/videorag && bash get_videorag.sh)          # HowTo100M 视频（yt-dlp，可能需要 cookies）+ 字幕
cd ../..
python -m scripts.universalrag_retrieval.build_subset  # 只用这两部分构建统一语料；失效视频记入 videorag_missing.txt
```

`build_subset` 构建的候选池比完整套件小。若要与 UniversalRAG 论文完全一致，改为运行其 `script/0_dataset.sh`（会下载十个数据集），并在论文中注明使用的是哪一种。

接着用 UniversalRAG 自己的检索器提取特征并导出共享候选池（所有对照臂读取同一个池）：

```bash
cd external/UniversalRAG
# 运行 script/1_preprocess.sh 中 query（qwen3、vlm2vec）、paragraph、table、clip、video 这几段（CUDA_DEVICES="0 1 2 3"）
for t in hybridqa videorag_synth; do
  python ../../scripts/universalrag_retrieval/export_rankings.py --target $t --top 10 --output $OMNIAWARE_ASSETS/universalrag/rankings
done
cd ../..
python -m scripts.universalrag_retrieval.prepare_frames --target videorag_synth --frames 8
```

### 4.4 UniversalRAG 路由器（对照方法）

路由器的训练数据会从包括我们评测题在内的查询中抽样，必须先剔除我们的评测题：

```bash
U=external/UniversalRAG; R=runs/v321_universalrag_8b; mkdir -p $R
(cd $U && python route/train/prep_train_data.py)
cp $U/route/train/data/train_data.json $R/train_data.json
for d in webqa_open_full hybridqa videorag_synth; do       # 依次剔除三个评测集的题目
  python -m omniaware.baselines.universalrag filter-train --train $R/train_data.json --dataset $d --output $R/train_data.json
done
(cd $U && python route/train/train_qwen3_vl.py --model-name Qwen/Qwen3-VL-2B-Instruct \
    --train-data ../../$R/train_data.json --output-dir route/train/checkpoints/qwen3_vl_2b_filtered)
for d in webqa_open_full hybridqa videorag_synth; do       # 导出查询并路由；paper_runs 的 universalrag 阶段读取 *.routes.json
  python -m omniaware.baselines.universalrag export --dataset $d --output $R/$d.queries.json
  python $U/route/train/route_qwen3_vl.py --model-path $U/route/train/checkpoints/qwen3_vl_2b_filtered \
      --input $R/$d.queries.json --output $R/$d.routes.json
done
```

### 4.5 检查

```bash
python -m scripts.setup.check_assets      # 所有必需资产都是 OK 才开始实验
```

## 5. 实验（多卡）

### 5.1 单条命令

一次运行在同一输入下为每道题生成全部对照臂：direct、static、各 EAB 变体、gated 系列，以及按选项加入的对照方法。

```bash
python -m omniaware.run --dataset okvqa_gs --split dev --calib-source heldout --model 8b --tag try_v1
```

多卡运行用 `scripts.launch`。它先在第一张卡上做一次 `--calibrate-only`，填好所有缓存并写出配置；然后每张卡起一个分片进程，问题按轮转分配；最后合并，并核对每道题恰好回答一次：

```bash
python -m scripts.launch --gpus auto -- --dataset okvqa_gs --split test --calib-source heldout --model 8b --tag v321_main_8b \
    --variants eab:g1.2:ans1:pos:gate --replication-gamma 1.2 --fitm --m2rag
python -m scripts.launch --gpus 0,1,2,3 --procs-per-gpu 2 -- <同上的 run 参数>    # 80 GB 卡每卡 2 个进程
```

- 中断后用原命令重跑即可续跑：每个分片跳过已完成的题，合并前会核对题数。
- 日志在 `runs/<tag>/logs/`，分片文件合并后移到 `runs/<tag>/<dataset>.shards/`。

常用选项：

| 选项 | 作用 |
|---|---|
| `--probe-uncertainty --first-round-routing` | 冷启动探测轮（WebQA、HybridQA、VideoRAG 的默认设置） |
| `--variants` | EAB 变体，格式 `eab:g<γ>:<ans1/ans2/q>:<pos/ctr/fixed/pos_nou>:<gate/nogate>[:layers:relevance:ctx]` |
| `--replication-gamma 1.2` | 复制对照臂 `replication` |
| `--selector ce` | cross-encoder 选证 |
| `--seakr-samples 20` / `--fitm` / `--m2rag` | SeaKR / Found-in-the-Middle / M2-RAG 对照臂（FITM、M2-RAG 仅支持 Qwen3-VL） |
| `--route-rule {mh,mass,entropy,retriever,all,oracle,query}` | 路由规则；`query` 为 UniversalRAG，需配合 `--query-routes` |
| `--order perm:<id>` | 固定证据顺序（14 个，见 `order.py`） |
| `--answer-style cite` | 带 `[E<j>]` 引用作答，用于引用指标 |
| `--intervention ...`、`--relevance-noise σ --noise-draw i` | 干预实验、相关度噪声 |
| `--u-combine zsum`、`--attn-layer mid` | dev 候选：U 的两项各自标准化；改用中间层注意力 |
| `--n-eval N` | 只跑前 N 题（SeaKR 与顺序实验各取 1,000 题） |

### 5.2 论文全流程

```bash
# 1. dev 网格：6 种 EAB 形式 × 3 种 U 设置 × 3 种 OK-VQA 语料，每个数据集 256 题
python -m scripts.paper_runs --stage dev --run --gpus auto
# 2. 按预先登记的规则冻结选择 -> runs/dev_decisions.json，并决定论文框架（docs/DEV_DECISIONS.md）
python -m scripts.analysis.dev_select
# 3. test，每个阶段只跑一次（自动读取冻结的选择）
for s in main universalrag ablation routing order citation interventions analysis; do
  python -m scripts.paper_runs --stage $s --run --gpus auto
done
# 4. 骨干迁移
python -m scripts.paper_runs --model internvl3.5-8b --stage main  --run --gpus auto
python -m scripts.paper_runs --model internvl3.5-8b --stage order --run --gpus auto
```

`python -m scripts.paper_runs --list` 列出各阶段的命令数及其对应的论文表格，`--stage X --print` 只打印命令。`universalrag` 阶段需要先按 4.4 节得到路由结果。

## 6. 评分与分析

```bash
python -m omniaware.eval.score_official runs/<tag>/<dataset>.jsonl --bart     # 官方指标 + 配对 bootstrap
python -m omniaware.eval.citation.attribution runs/<tag>/<dataset>.jsonl --device cuda   # Faith / AP / Inv
python -m scripts.analysis.paper_analyses <analysis> ...                     # 见下表
python -m scripts.analysis.routing_metrics runs/v321_route_*/webqa_open_full.jsonl --arm gated:<EAB> --metric acc_approx
```

各数据集的官方指标：

| 数据集 | 指标 |
|---|---|
| WebQA | acc_approx；加 `--bart` 得到 × ParaBank BARTScore 的 QA_approx |
| OK-VQA | VQA accuracy（官方 VQAEval） |
| HybridQA | EM / F1 |
| VideoRAG-Synth | ROUGE-L；加 `--bart` 得到 BERTScore；bootstrap 按源视频聚类 |

`paper_analyses` 包含以下分析：

- `order`：12 个排列的标准差、正序−逆序、翻转率
- `uncertainty` / `selective`：AURC 与 ρ；`selective` 用每个系统自己的答案
- `agreement`：NTE 与 U 的触发一致性
- `transitions`：错→对、对→错的比例，以及按被偏置最多的证据块分类的退化
- `paired`：跨运行的配对 bootstrap，用于主表的显著性标记
- `ard`、`interventions`、`gold25`、`noise`、`latency`、`stats`、`ranges`、`errors`

## 7. 论文表格与代码的对应

论文 V3.21 每张表、每个单元格来自哪次运行的哪个对照臂，见 [docs/PAPER_TABLE_MAP.md](docs/PAPER_TABLE_MAP.md)。规范见 [CONVENTIONS.md](CONVENTIONS.md)，实验协议见 [docs/PROTOCOL.md](docs/PROTOCOL.md)，dev 决策规则见 [docs/DEV_DECISIONS.md](docs/DEV_DECISIONS.md)。

## 8. 测试

```bash
python -m pytest
```

测试都在 CPU 上运行，不需要模型权重，包括：

- 在随机初始化的小 Qwen3-VL 和 InternVL 上验证 EAB 钩子；
- 用合成数据跑通整个流程，包括多卡分片与合并、WebQA 开放检索构建、dev 决策和各分析。

需要额外资产的测试（BARTScore、spaCy、NLI 检查器）在缺少资产时自动跳过。

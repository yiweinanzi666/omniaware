# Omni-Aware

Omni-Aware 是面向多模态 Web 问答的隐空间证据控制。它在一个冻结的生成模型上，用“不带证据和带各类证据做前向时，首个答案 token 的分布”这一套信号，同时决定三件事：是否检索、去哪个模态检索、怎么用证据。

控制证据有两种方式。**软控制**不改输入，只调整模型对各段证据的注意力，即 v1 的 Evidence Attention Bias（EAB）。实测它能把注意力推向支持证据，答案却几乎不变，所以它只留在论文的机制分析里。**硬控制**直接决定把哪一组证据交给模型，这是现在的主方法（v2）：每道题有五个动作 `direct`、`static_text`、`static_image`、`static_mixed`、`rerank_static`，共用同一个提示词、预算和候选池。选择器有两个：零训练的 v2-Rule（只读首 token 的领先幅度，部署时只需前向加一次解码），以及 v2-Cal（每个证据动作一个岭回归，预测相对最强单一动作的效用差）。两者都经过固定序列的单侧配对检验：对平均效用做风险控制，δ = 0.10，ε = 1.0。主比较的分母是校准集上得分最高的单一动作。

本仓库是实现与实验代码；论文 LaTeX 在 [`paper/`](paper/)，用 `cd paper && latexmk -pdf main.tex` 编译。

> **拿到代码后先读 [docs/HANDOFF_v2.md](docs/HANDOFF_v2.md)。** 它按顺序写了要做什么、在哪两处必须停下来等判断、每一步回传什么。
>
> **当前状态（v2）：** 方向是风险控制下的逐题仲裁。**正式 v2 实验还没有跑。** 论文里尚未跑出的实验数字都是占位符 `\ph`。旧仓库、论文 V3.20，以及相对历史缓存基线算出的提升，都不能当作结果。v1 EAB 降为分析。协议见 [docs/PROTOCOL.md](docs/PROTOCOL.md)，表与命令的对应见 [docs/PAPER_TABLE_MAP.md](docs/PAPER_TABLE_MAP.md)。预注册条文见 [docs/DEV_DECISIONS.md](docs/DEV_DECISIONS.md) 的 v2 预注册一节；测试命令只读冻结文件 `runs/v2_decisions/<dataset>.json`。

作废、不得进入论文主表的旧结果：

- 相对**历史缓存基线**的提升。根因报告写明，汇总表里的 **+33%（WebQA）** 和 **+15%（OK-VQA）** 比的就是这种基线：缓存 static 17.9，同批 512 题 static 23.4，相差 5.5 分，提示、预算和检索打分都不相同。
- 同一报告里经不起复核的其他正结果，包括训练式控制器的 **+8.43%**，以及 32 题开发集上选出的 **+14.4%**。
- 把 EAB 写成“提升证据利用”或“提升位置鲁棒性”的主贡献。
- 宽松口径的 WebQA 数字，以及把这种数字和官方 acc_approx 并列。
- “校准来自 validation”，同时测试集又是全部 validation。
- 贪心解码下的 seed 标准差。
- 在测试集上选出的网格。

`python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2` 的结果是**探索性**的，来自**旧管线**，含**已知 bug**，只用于检查代码和信号方向，**不是论文主表**。

快速导航：[v2 实验](#v2-实验) · [0 一键全流程](#0-一键全流程) · [1 环境](#1-环境) · [2 资产目录](#2-资产目录与环境变量) · [3 模型](#3-下载模型) · [4 数据集与-benchmark](#4-数据集与-benchmark) · [5 实验](#5-实验多卡) · [6 评分与分析](#6-评分与分析) · [7 论文表格](#7-论文表格与代码的对应) · [8 测试](#8-测试)

## v2 实验

执行顺序和判断标准见 [docs/HANDOFF_v2.md](docs/HANDOFF_v2.md)，每条命令的细节见 [docs/V2_RUNS.md](docs/V2_RUNS.md)。下面是 v2 的阶段命令，由 `scripts/v2_runs.py` 依次执行（加 `--dry-run` 只打印）。每个阶段内的评分、拼表、go/no-go 判断、冻结决定、开发集检查和出主表，由 `scripts/analysis/v2_table.py` 的 `build`、`gate`、`select`、`devcheck`、`apply` 完成。整条流程已在 CPU 上用小模型和合成数据走通；S1、S2、S4 还没有在真实模型和数据上跑过。

```bash
python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50
python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --n 50
python scripts/v2_runs.py --stage s2 --dataset webqa_full
python scripts/v2_runs.py --stage s2 --dataset webqa_open_img
python scripts/v2_runs.py --stage s2 --dataset okvqa_gs
python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions runs/v2_decisions/webqa_full.json
python scripts/v2_runs.py --stage s4 --dataset okvqa_gs --decisions runs/v2_decisions/okvqa_gs.json
python scripts/analysis/image_evidence_check.py --n 50
python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2
```

- **S1**：每个数据集 50 题冒烟，检查空答案、各臂格式和图像诊断命令（诊断命令只打印）。
- **S2**：解码校准集与开发集，官方评分，生成逐题动作表。开发集上做 go/no-go（`v2_table.py gate`，+8% 判据），校准集上生成冻结决定（`v2_table.py select`），再把冻结决定应用到开发集做检查（`v2_table.py devcheck`，相对最强单一动作 +5% 且 CI 下界大于 0；S4 只在这一检查通过时运行）。`webqa_full` 是 distractor 主设置；`webqa_open_img` 是开放检索的第二设置。
- **S4**：读取冻结的 `runs/v2_decisions/<dataset>.json`，全量测试集只跑一次，评分后用 `v2_table.py apply` 产出主表输入。不搜索网格。
- **image_evidence_check**：50 道图像题的证据管线诊断。
- **offline_v2**：`runs/offline_v2/REPORT.md`，由 `scripts/analysis/offline_v2.py` 生成。探索性结果，来自含已知 bug 的旧管线，不是主表。

GPU 时间都是**估计值**，不是实测，也不要写成测量结果：

| 阶段 | 估计值 |
|---|---|
| S1 | 约 0.5 天，1 张 GPU |
| S2 | 约 4–8 小时，1 张 A100 |
| S4 | 约 1–2 天，1–2 张 A100 |

## 0. 一键全流程

> 本节是 v1 的 `scripts/run_all.sh`。它不产生 v2 主表。v2 命令见 [v2 实验](#v2-实验)。

`scripts/run_all.sh` 把从一台新服务器到 v1 表格的流程串成 12 步。每步结束时在 `runs/.pipeline/` 写一个完成标记；中途失败或中断后，修好问题再执行同一条命令，会从未完成的那一步继续（各脚本在步内部也能续跑）。

### 0.1 准备（只做一次）

1. **机器**：Linux，NVIDIA GPU（每个 8B 进程约 20 GB 显存），磁盘约 1 TB（模型约 45 GB，WebQA 图片与编码、DPR 维基、COCO、HowTo100M 视频占大头）。
2. **主环境**（Python 3.11）：
   ```bash
   git clone https://github.com/yiweinanzi666/omniaware.git && cd omniaware
   conda create -n omniaware python=3.11 -y && conda activate omniaware
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124    # 按本机 CUDA 版本选择
   sudo apt install -y p7zip-full ffmpeg                                              # 解压 WebQA 图片；视频抽帧
   ```
3. **两份需要手动下载的数据**（许可原因，脚本无法自动下载；缺了脚本会停下并告诉你放在哪里）：
   - WebQA：从 [WebQA 官方仓库](https://github.com/WebQnA/WebQA) 的 Google Drive 下载 `WebQA_train_val.json`、`WebQA_test.json` 和全部 `imgs.7z.0*`，放到 `$WEBQA_DIR`；
   - OK-VQA 的 GS 语料：从 [luomancs/retriever_reader_for_okvqa](https://github.com/luomancs/retriever_reader_for_okvqa) README 的 Corpus 一节下载 `okvqa_train_corpus.csv`（必需）和 `okvqa_full_corpus.csv`（附录对照用），放到 `$GS_DIR`。
4. **UniversalRAG 环境**（Python 3.12，只用于 HybridQA / VideoRAG 的特征提取与路由器训练）：第一步 `env` 会把官方仓库取到 `external/UniversalRAG`，之后执行
   ```bash
   (cd external/UniversalRAG && pip install uv && uv sync)    # 路由器训练还需要与本机 CUDA 匹配的 flash-attn
   ```
5. **设置路径**（写进 `~/.bashrc`）：
   ```bash
   export OMNIAWARE_ASSETS=/data/omniaware_assets             # 所有大文件
   export WEBQA_DIR=/data/downloads/webqa                     # 第 3 条的 WebQA 文件
   export GS_DIR=/data/downloads/okvqa_gs                     # 第 3 条的 GS 语料
   export GPUS=auto                                           # 或 0,1,2,3；80 GB 卡可再设 PROCS_PER_GPU=2
   # 国内网络可加：export HF_ENDPOINT=https://hf-mirror.com
   ```

### 0.2 执行

```bash
bash scripts/run_all.sh --list      # 12 个步骤、各自的作用与完成状态
bash scripts/run_all.sh             # 从头跑到尾；中断后重跑同一条命令即可续跑
```

| 步骤 | 内容 | 主要产物 |
|---|---|---|
| `env` | `pip install -e ".[all]"`，取固定 commit 的官方仓库，跑 CPU 测试 | `external/` |
| `models` | 下载全部模型（固定 commit）、BARTScore、spaCy、NLTK Punkt，并校验完整性 | `$OMNIAWARE_ASSETS/cache/` |
| `webqa` | 解压图片、构建开放检索全池、BGE/CLIP 编码、val 与 train 的 top-5 检索 | `$OMNIAWARE_ASSETS/webqa_open/` |
| `okvqa` | OK-VQA 问题与 COCO 图片；GS-112K、GS-168K（附录）、DPR 维基 3M 三种稠密候选 | `$OMNIAWARE_CACHE/okvqa_*/` |
| `universalrag_data` | HybridQA 与 VideoRAG 下载、子集语料、套件特征、共享候选池、视频抽帧 | `$OMNIAWARE_ASSETS/universalrag/` |
| `universalrag_router` | 剔除我们的评测题后训练 UniversalRAG 路由器 | `external/UniversalRAG/route/train/checkpoints/` |
| `check` | `check_assets`：所有资产齐全且完整 | — |
| `verify` | 在真实 Qwen3-VL-8B 上核对提示模板、分段 prefill 与 EAB（任一项不符即停止） | `runs/verify_real_model_8b.json` |
| `dev` | dev 网格 → `dev_select --phase 1` → β×τ_b 网格 → `dev_select --phase 2`，冻结全部选择 | `runs/dev_decisions.json` |
| `main` | Qwen3-VL-8B 的全部 test 阶段：main、universalrag、ablation、routing、order、citation、interventions、analysis | `runs/v321_*` |
| `transfer` | InternVL3.5-8B 的骨干迁移（main、order） | `runs/v321_*_internvl3.5-8b` |
| `paper` | 编译 `paper/main.pdf` | `paper/main.pdf` |

常用操作：

```bash
bash scripts/run_all.sh --only check          # 只跑一步
bash scripts/run_all.sh --from dev            # 从某一步开始（资产已准备好时）
bash scripts/run_all.sh --redo webqa          # 清除某步的完成标记并重跑
```

- `dev` 结束后先看 `runs/dev_decisions.json` 中的 `framing`：它按 [docs/DEV_DECISIONS.md](docs/DEV_DECISIONS.md) 的预先登记规则决定论文能主张什么，`main` 阶段只读取这个冻结文件。
- 论文中的 `\ph` 需要按 [docs/PAPER_TABLE_MAP.md](docs/PAPER_TABLE_MAP.md) 从 `runs/` 的评分与分析输出手工填入；`paper` 一步只负责编译。
- 只想先在小规模上试跑：`python -m omniaware.run --dataset okvqa_gs --split dev --calib-source heldout --model 2b --n-eval 20 --tag smoke`（需要先完成 `env`、`models`、`okvqa`）。
- 下面第 1～8 节是每一步的展开说明，排查问题时查阅。

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
  okvqa_dense.py       OK-VQA 稠密检索（GS 训练语料 gs112k、可选 gs168k_full，或 DPR 维基 wiki3m）
  rerank.py            cross-encoder 选证
  order.py             顺序实验与复制对照
  interventions.py     干预实验与相关度噪声
  analysis.py          ARD、AURC、Spearman
  baselines/           SeaKR、Found-in-the-Middle、M2-RAG、UniversalRAG
  eval/                官方评测入口 score_official.py；UniversalRAG 套件指标；citation/ 引用指标
scripts/
  setup/               新服务器准备：模型与 NLTK 下载、OK-VQA、WebQA 开放检索、资产检查
  universalrag_retrieval/  UniversalRAG 套件：子集语料、候选池导出、视频帧采样
  launch.py            把一条 omniaware.run 命令分到多张 GPU 上
  paper_runs.py        论文全部实验命令（dev → 冻结 → test），支持 --gpus
  analysis/            论文各分析表、dev 决策、路由指标、统计功效
  fetch_external.py    下载并固定版本的官方仓库（external/）
  data_prep/           OK-VQA 维基语料与 BM25 索引（旧主机路径，见 4.2 节）
  webqa_open_retrieval/  旧主机的 WebQA 池与编码脚本；build_global_webqa_pool.py 为 LEGACY，不能在本仓库运行
third_party/           原样保存的官方评测脚本
tests/                 CPU 测试（小模型验证 EAB 钩子，合成数据跑通全流程）
docs/                  实验协议、dev 决策规则、论文表格映射、新旧代码对照
paper/                 论文 LaTeX 源码（V3.21，ACM sigconf）与各版修改说明
```

## 1. 环境

Linux 加 NVIDIA GPU。一个 8B 模型在 bf16 下约需 20 GB 显存，24 GB 及以上的卡每卡一个进程，80 GB 的卡可以每卡 2～3 个进程（`--procs-per-gpu`）。注意力使用 SDPA，不需要 FlashAttention：FlashAttention 会忽略 EAB 用的非二值 mask。

```bash
conda create -n omniaware python=3.11 -y && conda activate omniaware     # 需要 Python >= 3.11
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按本机 CUDA 版本选择
pip install -e ".[all]"                         # transformers==5.5.3（已测试版本）及全部可选依赖，含 nltk、faiss-cpu
python -m spacy download en_core_web_sm
python scripts/fetch_external.py                # 官方仓库（固定 commit）-> external/：SeaKR、M2RAG、UniversalRAG、BARTScore、WebQA、VQA ...
python -m pytest                                # CPU 上约 20 秒（2026-10-13 实测：468 通过、5 跳过）
```

需要 **7z**（p7zip）解压 WebQA 的图片分卷（4.1 节）：

```bash
sudo apt install p7zip-full                     # Debian/Ubuntu
conda install -c conda-forge p7zip              # 或在 conda 环境中安装
```

UniversalRAG 的特征提取和路由器训练需要它自己的依赖（VLM2Vec、Qwen3-Embedding）。建议在 `external/UniversalRAG` 中按其 README 用 `uv sync` 另建一个环境，只在 4.3、4.4 节使用。该环境要求 **Python 3.12**，路由器训练还依赖 **flash-attn**（需与本机 CUDA 和 torch 版本匹配的预编译轮子，或本地编译）。本仓库的 3.11 环境不需要 flash-attn。

## 2. 资产目录与环境变量

所有大文件都放在一个资产根目录下，在 `~/.bashrc` 中设置：

```bash
export OMNIAWARE_ASSETS=/data/omniaware_assets      # 模型、数据集、检索产物
export OMNIAWARE_RUNS=$PWD/runs                     # 可选：运行输出（默认 <仓库>/runs）
export OMNIAWARE_CACHE=$PWD/cache                   # 可选：相关度、校准、OK-VQA 候选等缓存（默认 <仓库>/cache）
export HF_HOME=/data/hf_cache                       # 可选
```

| 变量 | 默认位置 | 内容 |
|---|---|---|
| `OMNIAWARE_ASSETS` | `<仓库>/assets`（与当前工作目录无关） | `cache/<模型>`、`data/...` |
| `OMNIAWARE_WEBQA_OPEN_DIR` | `$OMNIAWARE_ASSETS/webqa_open` | WebQA 全池、图片、编码、检索结果 |
| `OMNIAWARE_UNIVERSALRAG_DIR` | `external/UniversalRAG` | UniversalRAG 仓库及其 `dataset/` |
| `OMNIAWARE_UNIVERSALRAG_RANKINGS` | `$OMNIAWARE_ASSETS/universalrag/rankings` | 导出的共享候选池 |
| `OMNIAWARE_UNIVERSALRAG_FRAMES` | `$OMNIAWARE_ASSETS/universalrag/frames` | 视频与片段的采样帧 |
| `ERC_QWEN_8B`、`ERC_QWEN_2B`、`OMNIAWARE_INTERNVL_8B`、`ERC_BGE`、`ERC_CLIP`、`ERC_RERANKER`、`OMNIAWARE_BARTSCORE` | `$OMNIAWARE_ASSETS/cache/...` | 单独覆盖某个模型的位置 |

完整路径定义见 `src/omniaware/paths.py`。

WebQA 开放检索的目录**只认环境变量或默认值**。如果 `experiments/eab_first_token/runs/phase1_dense_retrieval_codex_20261006` 这类旧主机目录存在，而没有设置 `OMNIAWARE_WEBQA_OPEN_DIR`，程序会给出警告且不使用它；要使用旧目录，显式设置该变量。

任何时候都可以用下面的命令检查缺了什么：

```bash
python -m scripts.setup.check_assets            # 逐项列出 OK / MISSING；模型按全部权重分片检查
```

## 3. 下载模型

```bash
python -m scripts.setup.download_models --list      # 列出模型、固定的 commit 与用途
python -m scripts.setup.download_models             # 全部（约 45 GB）
python -m scripts.setup.download_models --only nltk # NLTK 3.9.2 与英文 Punkt（引用指标，见下）
# 国内网络：HF_ENDPOINT=https://hf-mirror.com python -m scripts.setup.download_models
```

每次运行都会调用 `snapshot_download`，它会续传并校验已下载的文件；中断后直接重跑同一条命令即可。下载结束后，程序会按每个 safetensors 文件头中的长度检查权重是否完整，截断的文件会报错，而不是被当作完成。所有 Hub 仓库都固定到一个 commit（`scripts/setup/download_models.py` 中的 `REVISIONS`，即 2026-10-10 查询到的 `main` 的 commit）。

| 名称 | 来源（固定 commit） | 用途 |
|---|---|---|
| qwen8b | Qwen/Qwen3-VL-8B-Instruct | 主骨干 |
| qwen2b | Qwen/Qwen3-VL-2B-Instruct | 冒烟测试、快速调试 |
| internvl | OpenGVLab/InternVL3_5-8B-HF | 骨干迁移 |
| bge | BAAI/bge-large-en-v1.5 | 文本相关度；WebQA、OK-VQA 文本检索 |
| clip | openai/clip-vit-large-patch14 | 图像相关度；WebQA 图像检索 |
| reranker | BAAI/bge-reranker-v2-m3 | cross-encoder 对照 |
| nli | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | 引用支持度检查器；只下载 `checker_asset_manifest.json` 列出的 8 个文件，并按哈希校验 |
| bart_cnn_config | facebook/bart-large-cnn（只取配置与分词器） | BARTScore 结构 |
| bartscore | BARTScore ParaBank 权重（Google Drive，需 gdown） | WebQA QA_approx 的流畅度因子 |
| nltk | nltk==3.9.2；nltk_data@550b6625（只取英文 Punkt 的四个表） | 引用指标的分句（CiteRec） |

`nltk` 步骤会把 NLTK 3.9.2 装到 `cache/e5_citation_support/nltk_runtime`，下载 nltk_data 的 `punkt_tab.zip` 并核对其 git blob 哈希，然后只取 `punkt_tab/english/` 下的四个文件，把它们的 sha256 冻结在 `cache/e5_citation_support/sentence_splitter_assets.json` 中。之后每次使用都会校验这些哈希。

## 4. 数据集与 benchmark

论文使用 4 个 benchmark：WebQA（开放检索）、OK-VQA、HybridQA、VideoRAG-Synth。下面每节都给出下载来源、准备命令和产物。

### 4.1 WebQA（开放检索：全池的文本与图像数见 `pool_stats.json`）

1. 从 [WebQA 官方仓库](https://github.com/WebQnA/WebQA) 的 Google Drive 下载 `WebQA_train_val.json`、`WebQA_test.json` 和图片分卷 `imgs.7z.001…`。用 `7z x imgs.7z.001` 解出 `imgs.tsv`。
2. 把 `WebQA_train_val.json` 放到 `$OMNIAWARE_ASSETS/data/WebQA_train_val.json`（加载器与检索都读这个位置；检索会核对它与构建池时用的文件的 sha256 是否一致）。然后构建全池、编码并检索：

```bash
W=/data/webqa                                            # 含 WebQA_train_val.json、WebQA_test.json、imgs.tsv
python -m scripts.setup.prepare_webqa_open pool   --webqa-dir $W
python -m scripts.setup.prepare_webqa_open images --webqa-dir $W --workers 32 --batch-size 2000
python -m scripts.setup.prepare_webqa_open encode --kind text  --gpus auto --batch 256     # BGE，按 GPU 分片
python -m scripts.setup.prepare_webqa_open encode --kind image --gpus auto --batch 256     # CLIP，按 GPU 分片
python -m scripts.setup.prepare_webqa_open search --split val --top 5 --chunk 512          # 显存不足时减小 --chunk
python -m scripts.setup.prepare_webqa_open search --split train --top 5 --chunk 512        # 训练集问题用于无标签校准
```

- **池的数字只能取自 `$OMNIAWARE_ASSETS/webqa_open/pool_stats.json`**（`text_documents`、`image_corpus`）。论文中的池规模必须引用它；旧池（`scripts/webqa_open_retrieval/docs/GLOBAL_POOL_REPORT.md` 中的 544,489 / 389,750）来自无法在本仓库运行的 LEGACY 脚本，两者的文本键不同，不要混用。
- `images` 步骤逐批提交 imgs.tsv（每批 `--batch-size` 行），一张截断或损坏的图片只记录、不中止。运行结束时输出三个数：池中图像数、已抽取并建索引的图像数、缺失与损坏的数量。清单写在 `webqa_open/images/missing_images.jsonl`（imgs.tsv 中没有的池内图片）和 `webqa_open/images/bad_images.jsonl`（无法解码的图片）。这两类图片不进入图像索引，因此实际图像池比 `pool_stats.json` 的图像数少；论文中要同时报告两个数字。
- 编码每个分片写 `.done` 标记，清单 `emb/<kind>.manifest.json` 记录分片数、行数和 sha256。多卡运行中任一分片失败，其余分片会被停止，命令报错；按原命令重跑即可续跑。
- 检索是对全部归一化向量的精确内积 top-k，结果等同 FAISS `IndexFlatIP`。`--top` 不能超过索引大小。
- 索引中的路径相对于 `$OMNIAWARE_ASSETS` 记录，资产目录可以整体移动。

`webqa_full`（WebQA 的 distractor 设定）是 v2 的主设置；`webqa_img` 是它的图像题子集，v2 不使用。两者都从 `$OMNIAWARE_ASSETS/cache/webqa_full4966_images/images/<image_id>.jpg` 读图，val 用于评测、train 用于校准，缺任何一张都会报错。这个目录没有单独的生成步骤：`prepare_webqa_open images` 在 `webqa_open/images/` 下写出同名文件，覆盖全部图片事实，默认位置下建一个符号链接即可：

```bash
mkdir -p $OMNIAWARE_ASSETS/cache/webqa_full4966_images
ln -s $OMNIAWARE_ASSETS/webqa_open/images $OMNIAWARE_ASSETS/cache/webqa_full4966_images/images
```

`check_assets` 的 `WebQA distractor images` 一项核对 train 与 val 的每个图片事实都有图。`prepare_webqa_open images` 记为缺失或损坏的图片（`missing_images.jsonl`、`bad_images.jsonl`）不会写出文件，其中属于 train 或 val 图片事实的会让这一项不通过。

### 4.2 OK-VQA（val 5,046 题）与 dev 选型语料

```bash
python -m scripts.setup.prepare_okvqa --download        # 官方问题与标注 + COCO val2014 图片（约 6.5 GB；断点续传，失败会自动重试）
# 已有 COCO：python -m scripts.setup.prepare_okvqa --coco-dir /data/coco/val2014（问题与标注 zip 仍需下载）
```

**GS 训练语料 gs112k**（Luo et al., EMNLP 2021，OK-VQA 文献常用的 Google Search 语料）。上游仓库 [luomancs/retriever_reader_for_okvqa](https://github.com/luomancs/retriever_reader_for_okvqa) 的 Corpus 一节提供两份语料；需要手动从其 README 给出的 Google Drive 链接下载（仓库中未固定链接）：

- `okvqa_train_corpus.csv`：112,724 句，只由训练题收集。**本仓库默认使用它**，名称 `gs112k`、数据集 `okvqa_gs`，用于 dev 选型。
- `okvqa_full_corpus.csv`：168,306 句，由训练题和测试题收集，因此含有评测题的检索结果。仅作附录对照（语料名 `gs168k_full`，数据集 `okvqa_gs_full`，运行标签 `v321_gsfull`），不参与 dev 选型。

```bash
python -m omniaware.okvqa_dense --corpus gs112k prepare --csv okvqa_train_corpus.csv   # 需要 text 列，否则报错
python -m omniaware.okvqa_dense --corpus gs112k encode --batch 128                     # 段落不加前缀，fp16 memmap，可续跑
python -m omniaware.okvqa_dense --corpus gs112k search --k 20                          # 段数由 corpus.jsonl 的行数决定；问题用 BGE 查询指令
# -> $OMNIAWARE_CACHE/okvqa_gs112k/candidates_full.jsonl（数据集 okvqa_gs 读取）
```

`okvqa_dense` 的段落按 BGE 官方用法编码（无前缀），问题调用 `relevance.BGE.encode(query=True)`（即 BGE 查询指令），与重打分使用同一函数。

**dev 网格的 OK-VQA 语料**。参加选型的是 `okvqa_gs` 与 `okvqa_dense` 两种，运行 dev 阶段前两者都必须构建；缺任何一个，`paper_runs --stage dev --run` 直接报错（`--list` 会列出缺失的文件）。第 3 条 BM25 路线只为旧结果保留，不参加选型：

1. **okvqa_gs（GS 训练语料）**：如上。
2. **okvqa_dense（DPR 维基，3M 段落）**：

   ```bash
   export OMNIAWARE_ASSETS=/data/omniaware_assets
   python -m scripts.data_prep.prepare_okvqa_knowledge --action download --build-after-download
   #   下载 psgs_w100.tsv.gz（约 4.7 GB，校验 Content-Length 与 md5），取前 3,000,000 行，写入
   #   $OMNIAWARE_ASSETS/data/okvqa_knowledge_dpr_first3m/corpus_wiki_first3m.jsonl 和 manifest.json
   python -m omniaware.okvqa_dense --corpus wiki3m encode --batch 128                    # 3M 段落，需要数个 GPU 小时
   python -m omniaware.okvqa_dense --corpus wiki3m search --k 20                         # -> $OMNIAWARE_CACHE/okvqa_dense/candidates_full.jsonl
   ```

   构建规则（前 3,000,000 个数据行、去空行、行号从 0 开始、标题加换行加正文）写在脚本的 `build()` 中。旧主机上的规则源码 `cache/emnlp_source_…/build_corpus_wiki_only_3m.py` 不在仓库中；存在时脚本才会记录它的哈希。

3. **okvqa（BM25，DPR 3M）**：构建链如下，**目前无法从本仓库完整复现**。

   ```bash
   python -m scripts.data_prep.prepare_okvqa_knowledge --action build                    # 若已下载，只做 3M 构建（同上）
   python -m scripts.data_prep.prepare_okvqa_bm25_index \
       --corpus $OMNIAWARE_ASSETS/data/okvqa_knowledge_dpr_first3m/corpus_wiki_first3m.jsonl \
       --source-manifest $OMNIAWARE_ASSETS/data/okvqa_knowledge_dpr_first3m/manifest.json \
       --output $OMNIAWARE_ASSETS/data/okvqa_dpr_first3m_fts5/index.sqlite
   python -m scripts.data_prep.prepare_okvqa_bm25_candidates \
       --index $OMNIAWARE_ASSETS/data/okvqa_dpr_first3m_fts5/index.sqlite \
       --output $OMNIAWARE_ASSETS/data/okvqa_bm25_candidates
   ```

   缺口：`prepare_okvqa_bm25_candidates` 读取 `data/okvqa_local/{calibration,development}_inputs.jsonl`，这两个输入由旧主机的流程生成，仓库中没有；数据集 `okvqa` 读取的 `paths.OKVQA_CANDIDATES`（`validation_candidates.jsonl`）由 `prepare_okvqa_val_assets.py` 产生，而它的 `finalize` 步骤要求旧主机的冻结锁文件 `runs/local_m4_20261004/supplements_queue_lock.json`。因此 `okvqa`（BM25）**不参加** dev 语料选择（`scripts/analysis/dev_select.py` 的 `OKVQA_LOADERS` 只含 `okvqa_dense` 与 `okvqa_gs`），加载器仅为旧结果保留。

`scripts/data_prep/build_full_dpr_index.py`（无参数，读取 `cache/okvqa_knowledge_20261004/psgs_w100.tsv.gz`，即上面的下载）建立全部 2,100 万段落的 BM25 索引，它**不是** dev 语料，只用于附录的对照。`scripts/data_prep` 中的脚本都使用 `OMNIAWARE_ASSETS`（未设置时为 `<仓库>/assets`）作为根目录，路径为旧主机的布局。

### 4.3 HybridQA 与 VideoRAG-Synth（UniversalRAG 套件）

在 UniversalRAG 的环境中（见第 1 节）：

```bash
cd external/UniversalRAG
(cd dataset/HybridQA && bash get_hybridqa.sh)          # Web 表格 + 链接段落
(cd dataset/videorag && bash get_videorag.sh)          # HowTo100M 视频（yt-dlp，可能需要 cookies）+ 字幕
cd ../..
python -m scripts.universalrag_retrieval.build_subset  # 只用这两部分构建统一语料；失效视频记入 videorag_missing.txt
```

`build_subset` 构建的候选池比完整套件小。若要与 UniversalRAG 论文完全一致，改为运行其 `script/0_dataset.sh`（会下载十个数据集），并在论文中注明使用的是哪一种。yt-dlp 可能选择 `.mp4` 以外的容器（`.mkv`、`.webm`、`.mov`）；`build_subset` 与 `prepare_frames` 都会接受同名的这些文件。

接着用 UniversalRAG 自己的检索器提取特征并导出共享候选池（所有对照臂读取同一个池）：

```bash
cd external/UniversalRAG
CUDA_DEVICES="0 1 2 3" bash script/1_preprocess.sh      # 生成 eval/features/*.pkl；同时写出字幕特征 clip_clipscript.pkl、video_vidscript.pkl
# HybridQA：段落与表格（默认的 alpha 不作用于它们）
python ../../scripts/universalrag_retrieval/export_rankings.py --target hybridqa --modalities paragraph table --top 10 \
    --output $OMNIAWARE_ASSETS/universalrag/rankings
# VideoRAG-Synth：视频与片段（与 data.py 的默认模态一致；alpha 0.2 需要字幕特征）
python ../../scripts/universalrag_retrieval/export_rankings.py --target videorag_synth --modalities video clip --top 10 --alpha 0.2 \
    --output $OMNIAWARE_ASSETS/universalrag/rankings
cd ../..
python -m scripts.universalrag_retrieval.prepare_frames --target videorag_synth --frames 8
```

字幕特征的文件名：`1_preprocess.sh` 通过 `preprocess/extract_{clip,video}_feats.py` 写出 `clip_clipscript.pkl` 与 `video_vidscript.pkl`，而 `eval/eval.py` 读取 `clip_script.pkl` 与 `video_script.pkl`。`export_rankings` 两种文件名都接受；若两者都不存在而 `--alpha` 不为 0，它会报出该错误并给出处理办法（`--alpha 0` 可以跳过字幕特征）。

### 4.4 UniversalRAG 路由器（对照方法）

路由器的训练数据会从包括我们评测题在内的查询中抽样，必须先剔除我们的评测题。训练脚本默认启用 Weights & Biases（`--use-wandb` 默认为真），没有登录时会卡住，因此下面的命令设置 `WANDB_MODE=disabled` 并传 `--no-use-wandb`。

```bash
U=external/UniversalRAG; R=runs/v321_universalrag_8b; mkdir -p $R
(cd $U && python route/train/prep_train_data.py)
cp $U/route/train/data/train_data.json $R/train_data.json
for d in webqa_open_full hybridqa videorag_synth; do       # 依次剔除三个评测集的题目
  python -m omniaware.baselines.universalrag filter-train --train $R/train_data.json --dataset $d --output $R/train_data.json
done
(cd $U && WANDB_MODE=disabled python route/train/train_qwen3_vl.py --model-name Qwen/Qwen3-VL-2B-Instruct --no-use-wandb \
    --train-data ../../$R/train_data.json --output-dir route/train/checkpoints/qwen3_vl_2b_filtered)
for d in webqa_open_full hybridqa videorag_synth; do       # 导出查询并路由；paper_runs 的 universalrag 阶段读取 *.routes.json
  python -m omniaware.baselines.universalrag export --dataset $d --output $R/$d.queries.json
  python -m scripts.universalrag_retrieval.route_queries --queries $R/$d.queries.json \
      --model-path $U/route/train/checkpoints/qwen3_vl_2b_filtered --output $R/$d.routes.json
done
```

路由器训练需要 UniversalRAG 环境（Python 3.12，flash-attn，见第 1 节）。

### 4.5 检查

```bash
python -m scripts.setup.check_assets      # 所有必需资产都是 OK 才开始实验
```

## 5. 实验（多卡）

> 本节是 **v1** 管线（EAB、`paper_runs.py`、历史运行标签）。它的命令和任何已有输出都**不是** v2 论文主表。v2 的命令、作废结果和估计用时见 [v2 实验](#v2-实验)。

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
| `--probe-uncertainty --first-round-routing` | 冷启动探测轮（WebQA、HybridQA、VideoRAG 的默认设置）。去掉 `--probe-uncertainty` 即 noprobe 消融 |
| `--variants` | EAB 变体，格式 `eab:g<γ>:<ans1/ans2/q>:<pos/ctr/fixed/pos_nou>:<gate/nogate>[:layers:relevance:ctx]` |
| `--replication-gamma 1.2` | 复制对照臂 `replication` |
| `--selector ce` | cross-encoder 选证 |
| `--seakr-samples 20` / `--fitm` / `--m2rag` | SeaKR / Found-in-the-Middle / M2-RAG 对照臂（FITM、M2-RAG 仅支持 Qwen3-VL） |
| `--route-rule {mh,mass,entropy,retriever,all,oracle,query}` | 路由规则；`query` 为 UniversalRAG，需配合 `--query-routes` |
| `--order perm:<id>` | 固定证据顺序（14 个，见 `order.py`） |
| `--answer-style cite` | 带 `[E<j>]` 引用作答，用于引用指标 |
| `--intervention ...`、`--relevance-noise σ --noise-draw i` | 干预实验、相关度噪声 |
| `--u-combine zsum`、`--attn-layer mid` | dev 候选：U 的两项各自标准化；改用中间层注意力 |
| `--n-eval N` | 只跑前 N 题。顺序实验取前 1,000 题；SeaKR 在全量 test 上运行（不加 `--n-eval`） |

### 5.2 论文全流程

`bash scripts/run_all.sh --from dev` 依次执行下面的命令（第 0 节）。手动执行时：

```bash
# 1. dev 网格 phase 1：30 个 EAB 候选（形式 × γ）、3 种 U 设置、2 种 OK-VQA 语料（docs/DEV_DECISIONS.md）
python -m scripts.paper_runs --stage dev --run --gpus auto
python -m scripts.analysis.dev_select --phase 1            # -> runs/dev_decisions.json（phase 1）
# 2. phase 2：所选 EAB 候选上的 β × τ_b 网格；τ 由离线门控计算
python -m scripts.paper_runs --stage dev_beta --run --gpus auto
python -m scripts.analysis.dev_select --phase 2            # -> 完整的 runs/dev_decisions.json，含论文框架 framing
# 3. test：每个阶段只跑一次，自动读取冻结的选择；没有完整的 phase-2 文件时直接报错
for s in main universalrag ablation routing order citation interventions analysis; do
  python -m scripts.paper_runs --stage $s --run --gpus auto
done
# 4. 骨干迁移
python -m scripts.paper_runs --model internvl3.5-8b --stage main  --run --gpus auto
python -m scripts.paper_runs --model internvl3.5-8b --stage order --run --gpus auto
```

`python -m scripts.paper_runs --list` 列出各阶段的命令数、对应的论文表格，以及缺少的文件或冻结选择；`--stage X --print` 只打印命令。`universalrag` 阶段需要先完成 4.4 节的路由器训练（它自己导出查询并调用 `route_queries`）。

## 6. 评分与分析

v2 主表只使用官方指标：OK-VQA 为官方 VQA accuracy；WebQA 主指标为 acc_approx，同时报告归一化 BART 和 QA_approx，并分图像子集与文本子集。下面的命令和 `paper_analyses` 列表属于 v1 分析流程。

```bash
python -m omniaware.eval.score_official runs/<tag>/<dataset>.jsonl --bart     # 官方指标 + 配对 bootstrap
python -m omniaware.eval.citation.attribution runs/<tag>/<dataset>.jsonl --device cuda   # Faith / AP / Inv（需要 4.3 节 NLTK 步骤）
python -m scripts.analysis.paper_analyses <analysis> <runs...> [--metric ...]          # 见下表
python -m scripts.analysis.routing_metrics runs/v321_route_*/webqa_open_full.jsonl --arm gated:<EAB> --metric acc_approx
```

`attribution` 的分句依赖 NLTK 3.9.2 与英文 Punkt，需要先运行 `python -m scripts.setup.download_models --only nltk`。

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

v2 论文里每张表对应哪条命令，见 [docs/PAPER_TABLE_MAP.md](docs/PAPER_TABLE_MAP.md)。实验协议见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。规范见 [CONVENTIONS.md](CONVENTIONS.md)。dev 决策的原文见 [docs/DEV_DECISIONS.md](docs/DEV_DECISIONS.md)；v2 测试命令读取的是 `runs/v2_decisions/<dataset>.json`。

## 8. 测试

```bash
python -m pytest
```

测试都在 CPU 上运行，不需要模型权重，包括：

- 在随机初始化的小 Qwen3-VL 和 InternVL 上验证 EAB 钩子；
- 用合成数据跑通整个流程，包括多卡分片与合并、WebQA 开放检索构建（含截断 JPEG、分片缺失、续跑）、OK-VQA 语料格式、下载与续传、dev 决策和各分析。

需要额外资产的测试（BARTScore、spaCy、NLI 检查器、NLTK Punkt）在缺少资产时自动跳过。

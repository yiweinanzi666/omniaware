# OmniAware 数据补齐、完整性核验与实验交接报告

> 本报告给 Claude Code 使用。报告中的“完整”只表示已经通过大小、 SHA-256、行数、 .done 标记或仓库检查；没有通过检查的文件不会被写成完成。Qwen3-VL-2B 只能作为明确标注的 fallback，不能在任何结果表中写成 Qwen3-VL-8B。

## 1. 任务范围与结论口径

本次工作从 Windows 本地目录 D:\OmniAware_Data 出发，遵循“先在本地取得并校验，再上传到 node04:/home/user/Futaoran/omniaware”的路径。服务器连接使用跳板机 user@172.30.0.6，没有把认证信息写入仓库、脚本、日志或本报告。

截至本报告最终更新时间，已经具备一条可运行的 OK-VQA/开放 WebQA 工程路径：完整 OK-VQA COCO 图片与问题输入、GS-112K 候选、BGE 文本检索资产、Qwen3-VL-2B fallback、CLIP 权重，以及 WebQA 开放池。WebQA 图像向量和四个 top-5 ranking 在服务器上由后台流水线生成；最终版报告会在本节补上它们的 manifest、行数和日志状态。

GitHub 论文主实验所要求的完整资产仍然包含 Qwen3-VL-8B、InternVL3.5-8B、cross-encoder、NLI/BARTScore、UniversalRAG 的完整 HybridQA/VideoRAG 数据、VideoRAG 视频帧和官方 ranking 等大项。这些资产在当前网络和 10 小时窗口内无法全部取得并上传，因此本报告把“可开始 fallback smoke/工程实验”和“可声称论文完整复现实验”分开记录。

## 2. 服务器硬件与软件基线

2026-10-16 在 node04 实测：

| 项目 | 实测值 |
|---|---|
| CPU | 2 × Intel Xeon Platinum 8368Q，76 个物理核 |
| RAM | 251 GiB，总可用约 220 GiB |
| GPU | 2 × NVIDIA L20 |
| 单卡显存 | 46,068 MiB |
| 根盘 | 803G，总可用约 696G |
| /home | 7.3T，总可用约 6.9T |
| PyTorch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| Transformers | 5.17.0 |

两张 L20 已通过 CUDA 矩阵乘法测试。WebQA BGE 文本编码和 CLIP 图像编码均采用每卡一个进程的双卡分片；编码期间两张卡都能看到约 4.7 GiB 以上的模型显存占用。2B fallback 在显存上足够运行双卡分片；8B 主骨干仍未上传，不能用这条结论替代 8B 实测。

## 3. 本地文件与远端文件的完整资产表

### 3.1 已完成且已上传

| 资产 | 本地来源 | 远端位置 | 完整性证据 |
|---|---|---|---|
| Qwen3-VL-2B-Instruct 权重 | transfer/qwen2b_model.safetensors | assets/cache/qwen3-vl-2b-instruct/model.safetensors | 4,255,140,312 bytes；SHA-256 7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0 |
| CLIP ViT-L/14 权重 | transfer/clip_model.safetensors | assets/cache/clip-vit-large-patch14/model.safetensors | 1,710,540,580 bytes；SHA-256 a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb；远端与本地一致 |
| BGE-large-en-v1.5 | 本地配置/词表及原始权重 | assets/cache/bge-large-en-v1.5/ | 权重 1,340,616,616 bytes；SHA-256 45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7；配置、vocab.txt、tokenizer_config.json 已补齐 |
| OK-VQA metadata | transfer/okvqa_metadata.tar | assets/data/ | 129,127,424 bytes；SHA-256 11e4daaa45d67bcd08f9e67908f5dcfd6dd04c927ff67c3b71723a9968301933 |
| OK-VQA 完整核心包 | transfer/okvqa_core.tar | 解包到 assets/data/ | 本地 tar 2,454,410,752 bytes；SHA-256 cd7fd312f3816dddbbb54ffe1eb66c0fd295645f4029a2fb270b03baee86b168；远端分块会在最终版再次重组并逐块核对后记录 |
| OK-VQA GS-112K corpus | transfer 中的 prepared corpus | cache/okvqa_gs112k/corpus.jsonl | 112,724 行；emb_112724.f16 与 .done 均存在 |
| OK-VQA GS-112K candidates | 本地生成后上传 | cache/okvqa_gs112k/candidates_full.jsonl | 5,046 个问题；每题 20 个候选；约 50 MB |
| WebQA JSON | WebQA_train_val.json、WebQA_test.json | assets/data/；原始副本在 WebQA/ | pool_stats.json 记录 train/val SHA-256；文本池与原始 JSON 使用同一快照 |
| WebQA 开放池 | 服务器本地生成 | assets/webqa_open/ | 545,820 个去重文本文档；389,750 个图像事实；统计在 pool_stats.json |
| WebQA JPEG 解码 | imgs.tsv | assets/webqa_open/images/ | 389,740 个可解码 JPEG；缺失 0；不可解码 10；assets.jsonl 389,740 行，bad_images.jsonl 10 行 |
| WebQA 文本 BGE 向量 | 双 L20 编码 | assets/webqa_open/emb/text.* | 545,820 行，2 个 shard，每 shard 272,910 行、维度 1024；text.manifest.json 和两个 .done 存在 |
| M2-RAG 官方源码 | 本地 external/M2RAG | external/M2RAG/src/ 与 assets/cache/official_m2rag/src/ | 固定 commit ea6b9977f5a153215b38cd9a6aa3e4702ffbda22；源码 tar 已上传 |
| BARTScore 官方源码 | 本地 external/BARTScore/bart_score.py | external/BARTScore/bart_score.py | 固定 commit 248f511cb34ae3753fc81f7d7a945de5bfe33458；源码已上传 |
| UniversalRAG 代码与小型语料 | 本地 external/UniversalRAG 的代码、query、HybridQA parquet | external/UniversalRAG/ | hybridqa.json、table.parquet、paragraph.parquet 等小文件已上传；视频目录没有伪装成完整上传 |
| NLTK citation 文件 | 本地 assets/cache/e5_citation_support | 远端同路径 | nltk_runtime 使用 NLTK 3.9.2 纯 Python 包；Windows 的 regex 二进制已删除，使用服务器 Linux regex；English Punkt 四文件清单存在 |

### 3.2 已在本地取得但未形成远端完整资产

| 资产 | 本地状态 | 原因 |
|---|---|---|
| bge-reranker-v2-m3 | transfer/models/bge-reranker-v2-m3/model.safetensors 只取得了约 188 MB 的不完整前缀；配置和 tokenizer 文件已出现 | 完整权重约 2.27 GB，镜像下载速度很低；不完整文件不能上传或用于推理 |
| spaCy en_core_web_sm | 本地安装并打包为 transfer/models/en_core_web_sm_bundle.tar（约 15 MB） | 服务器 Python 环境没有 spaCy，服务器 DNS 无法访问 PyPI；已上传 bundle 但尚未把 spaCy 本体依赖安装到远端，因此 check_assets 仍需记录为缺失 |
| WebQA CLIP 图像向量 | 服务器后台编码中 | 389,740 张图片需要逐张通过 CLIP；编码完成后才会写 image.manifest.json |
| WebQA 四个 ranking | 后台流水线等待图像 manifest 后自动生成 | prepare_webqa_open search 同时读取 BGE 与 CLIP 两类完整索引，不能提前伪造单模态 ranking |

### 3.3 明确仍缺失、不能声称 GitHub 完整实验

- Qwen3-VL-8B-Instruct 主骨干；当前 ERC_QWEN_8B 为了让仓库路径可运行而指向 Qwen3-VL-2B，所有结果都必须标为 Qwen3-VL-2B fallback。
- InternVL3.5-8B-HF 及其骨干迁移实验。
- bge-reranker-v2-m3 完整权重。
- DeBERTa NLI citation checker 完整权重。
- BARTScore ParaBank bart_score.pth。
- spaCy runtime/model（本地模型已取得，远端本体尚未安装）。
- UniversalRAG 官方完整 checkout 中的大型数据、HybridQA/VideoRAG 官方 ranking、VideoRAG 全量视频和 frames。当前本地 VideoRAG 只有缩减子集，不能当作 8,621 视频全量。
- OK-VQA BM25 candidates 与 DPR-wiki 3M candidates（仓库将它们标为 dev grid 可选项；GS-112K 主路径已具备）。

## 4. 传输与核验过程中的关键修复

初始 HTTP Range 断点续传可以保持大小正确，但对若干 OK-VQA 分块出现“大小正确、SHA-256 不同”。我没有继续使用这些文件，而是逐块比较本地和远端 SHA-256，发现不一致的分块为 part_005、part_007、part_009、part_011–part_017。这些远端文件已删除，并从本地 transfer/okvqa_parts2/ 通过 Paramiko 的 SSH 跳板连接重新上传。最终版报告必须使用重新上传后的逐块 hash，再重组 tar；如果任何分块仍不一致，必须保持“未完成”状态。

重组只能使用精确的 part_000 到 part_023 文件名。此前 cat okvqa_parts/part_* 会把旁边的 .log 文件也拼进 tar，造成 960,350 bytes 的额外尾部；该错误 tar 已废弃，重组命令已改为显式三位编号循环。

## 5. 服务器后台任务与日志

当前使用的主要日志路径：

~~~text
/home/user/Futaoran/omniaware/logs/webqa_image_encode.log
/home/user/Futaoran/omniaware/logs/webqa_image_pipeline.log
/home/user/Futaoran/omniaware/logs/webqa_search_val.log
/home/user/Futaoran/omniaware/logs/webqa_search_train.log
/home/user/Futaoran/omniaware/logs/after_webqa_smoke.log
/home/user/Futaoran/omniaware/logs/qwen2b_fallback_dual_l20_smoke.launch.log
/home/user/Futaoran/omniaware/logs/check_assets_final.log
/home/user/Futaoran/omniaware/logs/okvqa_parts.sha256
/home/user/Futaoran/omniaware/logs/okvqa_core.sha256
~~~

后台流水线做了以下事情：

1. 等待 CLIP 权重达到 1,710,540,580 bytes，并记录远端 SHA-256。
2. 用两张 L20、每卡一个 shard 执行 prepare_webqa_open encode --kind image --gpus 0,1。
3. 图像 manifest 完成后自动生成 WebQA train/val 的 BGE 与 CLIP top-5 ranking。
4. ranking 完成后自动启动一个 50 题的 OK-VQA 工程 smoke，使用两张 L20、Qwen3-VL-2B、--selector relevance 和 10 个新 token 的短输出上限。该 smoke 没有 --v2，因为 v2 强制要求完整 bge-reranker-v2-m3；它不能被报告成正式 v2 主表结果。
5. 图像 manifest 完成后运行 scripts.setup.check_assets，完整输出保存在 check_assets_final.log。

## 6. 可复现的当前可运行命令

### 6.1 环境变量

~~~bash
cd /home/user/Futaoran/omniaware
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
export ERC_QWEN_2B=/home/user/Futaoran/omniaware/assets/cache/qwen3-vl-2b-instruct
export ERC_BGE=/home/user/Futaoran/omniaware/assets/cache/bge-large-en-v1.5
export ERC_CLIP=/home/user/Futaoran/omniaware/assets/cache/clip-vit-large-patch14
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
~~~

### 6.2 检查 OK-VQA GS 资产

~~~bash
/home/user/anaconda3/bin/python -m scripts.setup.check_assets
~~~

检查输出中即使有 Qwen3-VL-8B-Instruct OK，也要查看该行路径是否由 ERC_QWEN_8B 指向 2B fallback；不能把路径别名当成模型身份。

### 6.3 双 L20 fallback smoke

~~~bash
CUDA_VISIBLE_DEVICES=0,1 \
/home/user/anaconda3/bin/python -m scripts.launch \
  --gpus 0,1 -- \
  --dataset okvqa_gs \
  --split dev \
  --model 2b \
  --calib-source heldout \
  --n-calib 50 \
  --n-eval 50 \
  --selector relevance \
  --k 5 \
  --max-new-tokens 10 \
  --max-empty-rate 0.005 \
  --tag qwen2b_fallback_dual_l20_smoke
~~~

结果目录应为 runs/qwen2b_fallback_dual_l20_smoke/。只有同时存在合并 JSONL、completion marker、两个 shard 日志并且进程退出码为 0 时，才能在最终版报告中填写题数、空答案率或分数；未运行完成前不要写任何分数。

## 7. Claude Code harness 需要保留的事实

- 主骨干身份：Qwen3-VL-2B-Instruct fallback；主仓库参数名 --model 2b 与 ERC_QWEN_2B 路径一致。
- WebQA 图像池有 389,750 条事实，但只有 389,740 张图片可解码，10 张坏图必须在 harness 中作为明确的 bad-image 清单处理。
- WebQA 文本池、图像池、向量 shard 和 ranking 必须绑定同一份 pool_stats.json 与 WebQA_train_val.json SHA-256。
- OK-VQA 完整 tar 的本地 SHA-256 是 cd7fd312f3816dddbbb54ffe1eb66c0fd295645f4029a2fb270b03baee86b168；远端重组后必须重新核对此值。
- 论文 v2 运行强制 --v2 --v2-reranker，缺少 bge-reranker-v2-m3 时只能运行工程 smoke 或旧 selector 路径，不能生成 v2 主表。
- VideoRAG 缩减数据不能扩写为 full benchmark；没有完整 frames/rankings 时相关 stage 必须跳过并报告原因。

## 8. 最终收尾清单

最终提交前按以下顺序执行：

1. 读取 webqa_image_encode.log，记录两个 shard 的 .done、manifest 行数和 hash。
2. 读取 train/val ranking 文件的行数、最后一行格式和文件 hash。
3. 读取 okvqa_parts.sha256，逐行与本地 transfer/okvqa_parts2 比较；重建 okvqa_core.tar，记录远端 tar hash。
4. 读取 check_assets_final.log，把每个 OK、MISSING 和 optional 项逐项复制进报告。
5. 读取 smoke 日志、completion marker、GPU 记录和最终输出；没有成功完成就只写“未完成/失败原因”。
6. 更新本报告的状态、时间戳和上述占位信息。
7. 在 D:\OmniAware_Data\omniaware 执行 git diff --check、git status，提交本报告并推送到 origin/main。

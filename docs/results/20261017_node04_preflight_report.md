# node04 实验前置核验：2026-10-17

结论：本次已完成服务器实查、GitHub main 核对、完整 CPU 回归、失败项定位及下载测试的单独复测。正式 v2 S1/S2 尚不具备开跑条件。未运行 S4，未生成新的模型分数。

所有时间为北京时间（UTC+8）。本次只增加核验脚本、日志和本报告，保留原有源码修改与实验产物。

## 代码与环境

- GitHub API `https://api.github.com/repos/yiweinanzi666/omniaware/commits/main` 本次返回 `7ccf676e9804b2dfdf571d5b94fe8c9aba843acb`。
- CPU 回归工作区：`/home/user/Futaoran/omniaware_v2_exec`，提交与上述 main 一致。回归开始时 `git status --short` 输出为空。
- 资产根：`/home/user/Futaoran/omniaware/assets`；复用缓存：`/home/user/Futaoran/omniaware/cache`。
- 02:58:40 实查两张 NVIDIA L20，每卡总显存 46,068 MiB、已用 14 MiB、利用率 0%。本次 CPU 回归设置 `CUDA_VISIBLE_DEVICES=''`，没有加载 GPU 模型。
- 实际环境：Python 3.12.7，torch 2.14.0+cu130，transformers 5.17.0，pytest 7.4.4，pyflakes 3.2.0；未安装 spacy、faiss-cpu。仓库要求 `transformers==5.5.3`、开发依赖 `pytest>=8`，环境尚未满足。

## 本次执行与结果

完整执行脚本、命令、开始结束时间、退出码、JUnit XML 和日志存放于：

`/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242/`

本地取回副本与 SHA-256 清单：

`D:/WWW2027/data_ops/v2_preflight_20261017/remote_evidence/`

`D:/WWW2027/data_ops/v2_preflight_20261017/artifact_manifest.json`

| 检查 | 起止时间 | 退出码 | 结果及来源 |
| --- | --- | --- | --- |
| 完整 CPU pytest | 02:42:32–02:47:59 | 1 | `pytest.log`：528 passed、4 failed、5 skipped；pytest 报告用时 227.41 秒，包含启动在内的总墙钟时间以 start/end 文件为准 |
| 下载测试独立复测 | 02:51:20–02:53:06 | 0 | `targeted.log`：4 passed in 2.43s；命令显式设置 `no_proxy=127.0.0.1,localhost` 和 `NO_PROXY` |
| 原 2B smoke 完成标记核对 | 02:58:40 | 0 | 输出和配置的 SHA-256 均与原完成标记相同，详见本地 `final_evidence.log` |

完整回归命令：

```bash
cd /home/user/Futaoran/omniaware_v2_exec
export OMNIAWARE_ASSETS=/home/user/Futaoran/omniaware/assets
export OMNIAWARE_CACHE=/home/user/Futaoran/omniaware/cache
export OMNIAWARE_WEBQA_OPEN_DIR=/home/user/Futaoran/omniaware/assets/webqa_open
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/user/anaconda3/bin/python -u -m pytest -p no:warnings \
  --basetemp /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242/pytest_tmp \
  --junitxml /home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/preflight_0242/pytest.xml
```

独立复测采用相同资产与 CPU 设置，增加上述 `NO_PROXY`，限定 `tests/test_prepare_okvqa_fetch.py`，使用独立的 `targeted_tmp`、`targeted.xml` 和 `targeted.log`。独立复测不能改写完整回归的原始失败结果；下一轮完整回归仍需全部通过。

## 阻塞项与证据

1. **真实权重自检的 CUDA 设备错误。** GitHub main 的 `scripts/verify_real_model.py` 中，`reference_decode` 的 `q_pos`、`k_pos`、`col` 和零填充张量在 CPU 创建，但 `logits` 在 CUDA 上。服务器历史日志 `logs/verify_real_model_qwen2b_20261016.log` 保存了实际异常：`expected self and mask to be on the same device, but got mask on cpu and self on cuda:0`。用户已授权修复；修复已提交到本地分支 `fix/verify-real-model-cuda-device`，提交 `d93a076aba5930719d7be3753af6218bed9f00e0`，并同步到 node04 独立工作区同名分支，远端提交为 `da416a2266a0f3f01e0c394674d0e7b0ad0e5722`。修复把所有参考掩码和偏置张量放到 `query.device`，并保持 dtype 一致。服务器 `py_compile` 级别加载未报语法错误；真实权重自检尚未进入模型计算阶段，因为环境先在 scipy/numpy 导入处失败。

2. **M2-RAG 官方 checkout 缺失 Git 身份。** `tests/test_m2rag_official.py::OfficialM2RAGTests::test_vendor_commit_clean_and_loaded_source_is_official` 期望 `ea6b9977f5a153215b38cd9a6aa3e4702ffbda22`，实际在 `official_m2rag` 目录执行 git 返回父仓库提交 `7ccf676...`。这是现有源码副本缺少独立 Git 元数据导致的溯源验收失败，不能据此直接断言源码内容错误。本地 `transfer/models/m2rag_official.bundle` 已通过 `git bundle verify`，包含期望的官方 HEAD；仍需用它建立独立的固定版本 checkout、核对已有文件并复测。

3. **下载测试在完整回归中失败。** 三项恢复下载测试访问 `127.0.0.1` 时得到 HTTP 405，完整日志保留。显式绕过本机地址代理的独立复测全部通过。当前证据还不能区分代理设置影响与整套测试之间的状态影响；不能把独立复测通过写成完整回归通过。

4. **模型与运行依赖未完整就绪。** 后续复查已确认服务器 8B 四个权重分片均已存在，大小分别为 4,902,275,944、4,915,962,496、4,999,831,048、2,716,270,024 字节；这一事实只说明分片已落盘，仍需按 manifest/hash 做完整模型验收。reranker 在本轮早期检查时仍在传输；BARTScore 权重目录为空。当前基础环境为 numpy 2.2.6、scipy 1.13.1、scikit-learn 1.5.1、Transformers 5.17.0。scipy/scikit-learn 是按 NumPy 1.x ABI 构建的，导入时实际报 `numpy.core.multiarray failed to import`；因此修复后的真实 8B 自检退出码 1，但尚未执行模型前向。仓库要求 Transformers 5.5.3，固定环境仍需建立。其它会话正在处理模型和环境准备，本次没有覆盖其任务。

按 `docs/HANDOFF_v2.md` 第 1、1.5、3 节，pytest 有失败或真实权重自检报错时不能继续进入正式 S1。现有用户交接文件 `C:/Users/Administrator/Downloads/CODEX_FINAL_PROMPT_node04_with_login.md` 第 0 节要求源码修复先确认、再在独立分支提交；用户已确认，修复提交已完成。当前剩余阻塞是环境 ABI、固定 Transformers 版本、M2-RAG checkout 和模型/评分资产验收。

## 已核实的资产进展

WebQA 文本及图像 manifest 均已出现。四份 ranking 已重新统计行数、大小和 SHA-256：

| ranking（相对 `assets/webqa_open/dense_top5_full/`） | 行数 | 字节 | SHA-256 |
| --- | ---: | ---: | --- |
| webqa_val/bge/rankings.jsonl | 4,966 | 3,398,706 | 99e2a3483f87bb833adc00f3de41a346f355e6d6fcc9e15fabd9590b485060cd |
| webqa_val/clip/rankings.jsonl | 4,966 | 1,859,312 | 7d5d0fb7fd962478cabd4bc3583caedbadba1c2b64e87fe360afd746924bf988 |
| webqa_train/bge/rankings.jsonl | 36,766 | 25,162,332 | 32d989fabe72b6d2294b55399ec669d7138caf9bef20d543d569687b9bc3eafb |
| webqa_train/clip/rankings.jsonl | 36,766 | 13,766,094 | ee085fa8acc7de45f46ac2630166a11f311bb52af745aa07ec0c496a2a886b72 |

以上只确认文件和行数/hash，不等于问题 ID、候选内容和完整数据加载均已验收。本次全量加载尝试超过连接命令的 55 秒上限，没有取得完整加载结果，保留为未验证；后续应放入 tmux 并保存完整日志。

已完成的历史工程 smoke 位于 `runs/qwen2b_fallback_dual_l20_smoke/`：50 题、模型 `2b`、`v2=false`、`max_new_tokens=10`。完成标记记录空答案率和拒答率均为 0。输出和配置 hash 本次重新核对通过。它不是新生成的结果，也不能替代 8B v2 S1 或论文成绩。

## 后续执行顺序

先完成固定环境及模型 hash 验收、恢复 M2-RAG 固定版本 checkout，并修复服务器 NumPy/Scipy ABI 环境；随后重新运行完整 pytest 和真实 8B 自检。CUDA 修复分支已经准备好，但环境修复完成前不把自检退出码解释为代码失败或通过。全部通过后，才按冻结配置运行正式 S1，逐项核对格式、图片输入和溯源记录。S1 通过后再按既定关卡推进 S2，S2 后停下汇报；S4 仍需用户明确批准。

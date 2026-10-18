# node04 准备验收与停止报告

记录时间：2026-10-17 07:03（Asia/Shanghai）。这是准备阶段报告，不是 S1/S2 结果，也不代表完整代码 review 已通过。

实验执行目录 `/home/user/Futaoran/omniaware_v2_exec`；提交 `7ccf676e9804b2dfdf571d5b94fe8c9aba843acb`。`pytest_stop_diagnosis_0655.json` 的 `git.experiment.status.stdout` 为空。原始工作目录保留，未覆盖其已有改动。

## 已确认的完成项

- Qwen3-VL-8B、BGE、CLIP、reranker、NLI 在 `check_assets_065446_625185.log` 中均为 OK。新下载的 Qwen、reranker、NLI 另有各自 verified manifest。
- Python 3.11.17、torch 2.11.0+cu128、transformers 5.5.3，CUDA 可见 2 张 GPU；pip check、spaCy/faiss 导入和英语模型加载已通过。固定提交的外部仓库已取得。证据为远端 `environment_verify_061759_547144.log`、`pip_check_061754_186672.log`、`environment_complete.json`。这些只证明依赖安装检查通过，不代表运行验收通过。
- OK-VQA：`loader_acceptance_065119_463912.log` 的 `okvqa_gs.count=5046`、`meta.dropped={}`。资产报告 `checks.okvqa_images` 中 questions=5046、distinct_image_ids=5033、missing=[]、decode_errors=[]、pass=true。
- 现有候选文件：资产报告 `checks.candidates` 中 rows=5046、unique_question_ids=5046、bad_qids=[]、pass=true。embedding_bytes 与 embedding_done 均为 true。
- 开放检索 WebQA：加载日志中 universe=`webqa_open_full`、count=4966、dropped={}；train_count=36766、train_meta.dropped={}。这不替代 `webqa_full` 的图片验收。
- BARTScore 官方 Drive 续传在 06:58:48 退出 0，日志 `bart_resume_20261017_064737_1.json`。本地文件大小 1625565295 字节，SHA256 `4c076ee386795a831e0021788148becf57136127d6512ca350c7973f36c0e573`。原 watcher 因 Paramiko 文件打开模式无法写入而失败；运维修订版 `finish_bart_v2.py` 已用新临时文件重试，07:01 观察到远端临时文件 12288000 字节。远端最终校验未完成前不报资产就绪。失败的空临时文件仍保留。
- 两份缺失原始 CSV 已补传并校验，`source_csv_verified_0657.json.files` 记录最终 `/assets/downloads/okvqa_gs/` 路径、大小和 SHA256：train 为 `f6646ad30d05166cfd32c281566a0032656035761e7d33acaf3267bfbb9d4997`，full 为 `4ab2352a2bff6413ac74f1a18ef7ffcc8a15b5f77df97a5b6afd8375448128ae`。两者与任务指定值完全一致，existing_retrieval_cache_changed=false。仅补传缺失原始文件，未重新生成检索缓存。

## 阻塞 1：现有 GS-112K 语料验收失败

文件 `/home/user/Futaoran/omniaware/cache/okvqa_gs112k/corpus.jsonl`。

- `V2_ASSET_CHECK_20261017_initial.json` 中 `assets.gs_corpus.jsonl.lines=113019`，要求 112724；字节数 43613003；SHA256 `0ba5607c4733de5e94e644e92b4883f684b7cd37bcc4b23238eeb6abe86c96e6`。
- 独立只读解析 `blocked_assets_0657.json` 的 `corpus`：physical_lines=113019、json_rows=113018、blank_lines=0、unique_ids=112724；第 112725 行 JSON 解析失败。由此确认并非空行计数问题，且解析成功行存在重复 ID。
- embedding 和候选结构通过，不能抵消语料验收失败，也不能据此证明其与该文件的内容对应关系正确。
- 按用户任务 3.4 的既有缓存验收失败规则停止 OK-VQA 实验。保留语料、embedding、offsets 和 candidates 原样，未截断、去重或重建。异常产生原因尚未确认。

## 阻塞 2：WebQA distractor 图片缺失

`webqa_full` 加载器抛出 MissingDataError：val 中 4 个图片事实缺失，涉及 4 道题、3 个唯一文件。详见 `blocked_assets_0657.json.missing_webqa_val_images`：

- `30278219.jpg`，影响 2 道题。
- `30045608.jpg`，影响 1 道题。
- `30172755.jpg`，影响 1 道题。

均为 `img_negFacts`；目标目录 `/home/user/Futaoran/omniaware/assets/cache/webqa_full4966_images/images/`。未使用 allow-missing-assets，未丢弃题目。上述是 val 失败详情，不是 train 全量缺图清单。`check_assets_065446_625185.log` 也将 WebQA distractor images 标为 MISSING。

## 阻塞 3：pytest 失败

运行目录为干净执行工作树，完整命令：

```text
/home/user/Futaoran/omniaware_v2_python311/bin/python -m pytest -p no:warnings
```

日志 `pytest_064746_161869.log`。发现失败后向本次 pytest PID 2116189 发送 SIGINT，保留了错误汇总：**4 failed, 318 passed, 1 skipped, 8 subtests passed**，退出码 2；这是中断运行的结果，不是完整测试套件结果。

1. `tests/test_m2rag_official.py:61`，`test_vendor_commit_clean_and_loaded_source_is_official`。`src/omniaware/baselines/m2rag_official.py:20` 优先选取已存在的 legacy cache。该目录无独立仓库元数据，git 向上找到原始 omniaware 仓库，得到 `7ccf676...`，与要求 `ea6b9977f5a153215b38cd9a6aa3e4702ffbda22` 不一致。新下载的 `external/M2RAG` 固定提交正确且干净；诊断 JSON 的四个实际加载源文件 `same_sha256` 全为 true。因此目前证据是来源/路径元数据问题，不能称四个文件内容已损坏。未改源代码或旧缓存。
2. `tests/test_prepare_okvqa_fetch.py` 的三个下载测试分别在本机回环 HTTP 服务上失败，日志均报告 HTTP 405。准备脚本设置了 HTTP_PROXY，但没有设置 loopback 的 NO_PROXY。独立探针的 `proxy_bypass=false`；沿用代理环境失败，而直连同一回环服务 status=200，见 `pytest_stop_diagnosis_0655.json.loopback_proxy_probe`。应修正测试/运行环境的回环代理配置后再验证；尚未重跑或宣称解决。

## 执行状态与下一步

真实 8B 自检、S1、S2 均未启动；S4 禁止。完整代码 review 尚未完成，不能报通过。因上述明确失败，本轮只完成只读诊断、缺失原始 CSV 补传、BARTScore 下载上传收尾与记录。

后续须先处理资产失败：保留异常缓存，确认采用原始 CSV 在独立新目录恢复并校验缓存的方案；从官方图片源恢复缺失文件；隔离 legacy M2-RAG 缓存路径并修正回环代理环境。重新通过资产与 pytest 验收、完整 review 后，才有条件考虑真实权重自检和 S1。不得仅凭依赖安装完成或下载进度达到 100% 启动实验。

所有远端证据位于 `/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/`。后续每小时继续汇报下载/传输状态和停止原因；没有用户后续决定时，不能把已报告的失败忽略后进入 GPU 阶段。

# v2 S1 前置自检：失败，未启动冒烟

代码基线：`yiweinanzi666/omniaware@162115ecea8cdd9b2d1503f2d304e3066f4adb8c`。
按 `docs/HANDOFF_v2.md` 顺序执行了 main 的 fast-forward pull、独立环境里的 `pip install -e ".[all]"`、`python -m pytest -p no:warnings`。没有修改源码、提示词、阈值、切分、预算或其他实验参数。

最终结果：**1 failed, 473 passed, 1 skipped, 21 subtests passed in 13.11s**，退出码 1。测试数量因本机已提供 tokenizer、processor 与资产而高于交接文档的最小环境记录。完整测试输出在 `pytest.log`，最后 30 行在 `pytest_last30.txt`；环境、pip freeze、命令与时间记录一并保留。

## 剩余失败及证据

`tests/test_run_smoke.py::test_interventions_and_relevance_noise`，断言在 `tests/test_run_smoke.py:379`。该测试使用 FakeQwen 和合成 16 题，不是 Qwen3-VL-8B 的真实实验。

本机 `spaCy en_core_web_sm 3.8.0` 把 Paris/Rome/Berlin/Madrid/Vienna/Prague/Lisbon 判为 GPE，把 Oslo 判为 ORG。`run.py` 用真实 spaCy 分类构造同类型替换池；校准池里没有 Oslo 所需的 ORG 替换项。`interventions.pick_substitute` 因而返回 None，合成题 q006、q014 的 entity 均为 None，而测试要求每题都有一个不同城市。

失败样例的完整运行目录保留在 `failed_synthetic_test_fixture/`，包括 `okvqa_config.json`、合成预测、环境、日志与完成记录。这个 v1 单元测试没有生成 `.prompts.ndjson`，未补写不存在的 prompt 样本。这些只能用于追溯单元测试，不是 S1 冒烟输出或正式实验结果。

建议由仓库维护者确认测试是否应固定 NER 的 fixture，并独立验证找不到同类型替换项时的记录语义。这里没有修改测试、移除断言、隐藏失败或关闭 spaCy 来让测试通过。

## 环境设置过程

首次未指定资产根目录时，7 个 M2-RAG processor 测试因新 checkout 下没有模型配置而在 setup 报错；原始日志和收据为 `pytest_initial.log`、`pytest_receipt_initial.json`、`environment_initial.json`。

按 README 设置 `OMNIAWARE_ASSETS` 指向已有本地资产后，上述 7 个 setup 报错消失；两项 dry-run 测试随后因未明确指定 WebQA open 目录产生的旧目录警告而失败。该轮完整日志和收据为 `pytest_asset_paths.log` 与 `pytest_receipt_asset_paths.json`。

再按 README 设置 `OMNIAWARE_WEBQA_OPEN_DIR` 指向当前 open 目录后，这两个警告相关的测试通过。没有 suppress warnings，没有替换数据、模型或评分协议，也没有运行任何 GPU 实验。最新有效路径见 `pytest_receipt.json`。

独立环境为 Python 3.12.10 / PyTorch 2.14.1 / Transformers 5.5.3 / torchvision 0.29.1 / NLTK 3.9.2；本机没有 CUDA。环境通过 .pth 复用旧环境中已有依赖，独立安装的版本优先；原环境未被修改。未加载真实 8B，所以 environment.json 中真实模型的 dtype/attn_implementation 是 null，不能当成冒烟的溯源检查通过。

## 阶段状态与算力

遵守交接文档“有失败就停下来反馈”：S1 三个数据集、真实图像证据诊断、S2、S4 全部未启动，故四项 S1 检查尚无真实模型结果。没有 S1 的 *_config.json；保留的是失败单元测试的配置。

旧 A100 SSH 地址 `region-9.autodl.pro:32361` 直连返回 Connection refused，经本机 SOCKS 代理路径也未能连接。需要恢复实例访问或提供新的 SSH 命令，随后才能核实服务器 GPU、资产与环境。未使用历史 PID 操作服务器。

本目录整体提交到 `results/v2-s1-20261013`，只是 S1 前置自检失败的收据，不代表 S1 已完成；未推送 main。自检通过后才能跑三个数据集各 50 题的 S1，并在 S1 完成后等待用户确认；S2 与 S4 的确认点保持不变。

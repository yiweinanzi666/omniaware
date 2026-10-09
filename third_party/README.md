# 官方评测（原样保存）

本目录下的文件只做逐字节保存，**不允许修改**。需要适配接口时，在 `src/omniaware/eval/` 里包装调用，不改这里的源码。

| 目录 | 内容 | 上游来源 |
|---|---|---|
| `vqa_eval/vqaEval_py3.py` | OK-VQA / VQA 官方准确率（10 位标注者，min(#/3, 1)） | VQA 官方评测 `vqaEval.py` 的 Python 3 版本（GT-Vision-Lab/VQA） |
| `videomme/official_video_eval.py` | Video-MME 官方答案解析与统计 | Video-MME 官方仓库的结果评测脚本（函数 `extract_characters_regex` 与 `eval_your_results`），取自旧仓库 Pro6000 运行目录 |
| `webqa/eval_1022.py`、`eval_webqa.md`、两个 LICENSE | WebQA 排行榜评测脚本，以及 validation 上的 acc_approx 近似 | 见 `webqa/provenance.json`（含 commit 与 SHA256） |

以下官方文件不在 git 中，放在资产目录（`OMNIAWARE_ASSETS`）下：

- MMQA 官方 `evaluate.py` 和 `common_utils.py`：放在 `data/mmqa_official/official_eval/`（allenai/multimodalqa）；
- WebQA 的 ParaBank BARTScore：`cache/bartscore/`，包括 `official_source/bart_score.py`、`bart_score.pth`、`bart-large-cnn-config/` 和 provenance 文件。

缺少这些文件时，相关测试会自动跳过，相应数据集的评分会报错，不会退回到宽松匹配。

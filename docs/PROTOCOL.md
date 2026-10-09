# 实验协议

本协议针对旧实验中发现的问题制定（见 `../Omni-Aware_全面审查报告_20261010.md`）。论文里的任何结果都必须按本协议产生。

## 1. 数据划分

| 数据集（`--dataset`） | 评测数据 | 校准来源（`--calib-source`） | 说明 |
|---|---|---|---|
| `okvqa` / `okvqa_dense` | OK-VQA val 5,046 | `heldout`（训练集候选尚未生成） | BM25 或 BGE 维基段落候选 |
| `mmqa_table` | MMQA dev 单表题 369（`--mmqa-scope legacy299` 为旧口径） | `train` | 候选为该题自己的表格行块与文本、图片 |
| `videomme` | Video-MME Short 900 | `heldout` | 证据固定（全部帧 + 字幕），帧目录和字幕长度都是选项 |
| `webqa_img` / `webqa_full` | WebQA val 图像题 2,511 / 全部 4,966 | `train` | **distractor 设置**：每题自带的正负事实池，不是开放检索 |

- `--split dev|test`：先用固定种子 `--split-seed`（默认 2027）把数据打乱，然后依次取出 calib（仅 heldout 时）、dev（`--n-dev`，默认 256）和 test。三者互不相交，运行时会断言检查。
- 调参（τ、τ_bias、β、γ、k、提示词、帧数等）**只看 dev**。test 在设置冻结后只跑一次。
- 在 test 上看过结果再改设置，那么这一轮 test 只能算 dev，必须重新划分或换数据集确认。

## 2. 校准

- 校准用 `--n-calib` 个**无标签**问题（默认 500，与论文一致），估计 U 的 median/IQR 和各模态检索分数的 ECDF。
- 校准过程不读取答案，也不按准确率选阈值（旧的 `--select-tau` 已删除）。
- 校准缓存按数据集、模型、k、输入选项和校准来源分别存放。

## 3. 对照臂（每题同一输入）

| 臂 | 含义 |
|---|---|
| `direct` | 不加检索证据（只保留问题自带的图片或表格上下文） |
| `static` | top-k 证据静态拼接 |
| `<eab 变体>` | 在 static 的同一输入上加 EAB（默认 `eab:g1.2:ans2:pos:gate`，即论文 decode-only） |
| `pasta:*` | 不区分相关度的统一偏置（PASTA 风格对照） |
| `gated:static` | U_pre ≤ τ 时用 direct，否则用 static（论文的 "U_omni, static"） |
| `gated:<eab 变体>` | 完整系统 |

- 要判断 EAB 本身的效果，比较 `<eab>` 与 `static`，或 `gated:<eab>` 与 `gated:static`；要判断检索门控的效果，比较 `gated:static` 与 `static` 和 `direct`。
- 提示词、帧数、字幕、答案格式（`--answer-style`）对所有臂同时生效，所以这些改动带来的提升**不能**算作方法贡献。如果要报告这类工程改进，单独列一行。
- `ans2`（decode-only）在数学上无法改变第一个答案 token（见 `tests/test_qwen_eab.py`）。单 token 答案（选项字母、单词）的数据集应同时报告 `ans1`（EAB-P）。

## 4. 评分与统计

```bash
python -m omniaware.eval.score_official runs/<tag>/<dataset>.jsonl
```

- 只用官方指标（见 CONVENTIONS §5）。WebQA validation 没有隐藏测试集的关键词，只能报告 acc_approx（以及 × BARTScore 的近似综合分），不能称为排行榜分数。
- 对每个臂与参考臂（默认 `static` 和 `direct`）做配对 bootstrap（5,000 次），Video-MME 按视频聚类重采样。报告均值差、95% 区间和胜/负/平。
- greedy 解码下结果是确定的。"多个 seed"只有在 seed 确实控制了某个随机过程（例如顺序打乱或采样）时才有意义，要写明 seed 控制的是什么。

## 5. 顺序敏感性

`--order shuffle --seed s` 在固定证据集合上打乱顺序。12 种排列必须固定并对所有题目相同；报告各排列分数的标准差、正序与逆序之差，以及逐题翻转率。

## 6. 报告规则

- 论文中的每个数字都要附上 run tag、commit 和评分文件。不能重算的数字一律不写。
- 负结果和不显著的结果同样保留在 `runs/` 中，不按结果好坏挑选。

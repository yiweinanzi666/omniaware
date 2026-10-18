# 代码与实验规范

本仓库的所有代码、实验和结果都遵守本文件。规范的目的有三个：
- 每个数字都能追溯到一条命令和一份逐题输出；
- 每个比较都是公平的；
- 代码只有一份。

需要改规范时，直接修改本文件并单独提交。

## 1. 目录与职责

```
src/omniaware/        唯一的方法实现（每个概念只有一处代码）
  paths.py            资产与输出位置（环境变量覆盖）
  data.py             数据集加载与 calib/dev/test 划分
  relevance.py        BGE/CLIP 打分与按模态 ECDF 校准
  okvqa_dense.py      OK-VQA 稠密检索候选生成
  qwen_eab.py         Qwen3-VL 推理与 EAB 注意力钩子
  pipeline.py         提示词、U_omni 归一化、EAB 偏置形式
  routing.py          模态路由 M_m·H_m
  run.py              一次运行：解码所有对照臂，写逐题 JSONL
  interventions.py    干预实验（遮盖、反事实、干扰项、注入）与相关度噪声
  analysis.py         ARD、AURC、Spearman 等逐题分析量
  rerank.py           cross-encoder 选证（bge-reranker）与按模态 CE 百分位
  order.py            固定排列表、复制份数、顺序标准差统计
  baselines/          对照方法：SeaKR、Found-in-the-Middle、M2-RAG、UniversalRAG
  eval/               只放官方评测：score_official.py 是唯一答案评分入口
  eval/citation/      CiteRec 引用支持度（NLI），以及从运行结果导出的工具
third_party/          原样保存的官方评测脚本及来源、许可
scripts/data_prep/    生成资产的数据准备脚本（每个数据集一次）
scripts/webqa_open_retrieval/  WebQA 全语料池、BGE/CLIP 编码、FAISS 索引与检索
scripts/universalrag_retrieval/  UniversalRAG 套件候选池导出、视频帧采样
scripts/analysis/     基于运行输出的分析（论文各分析表、路由、统计功效、证据是否包含答案）
tests/                单元测试与端到端测试（CPU 即可运行）
docs/                 实验协议、新旧代码对照
runs/  cache/         运行输出与缓存，不进 git
```

- 新功能先找现有模块。只有确实没有归属时才新建模块，并在上面的目录表里登记。
- **禁止复制源码。** 不允许把 `erc/` 之类的包复制进运行目录、候选目录或交付目录。复现一次运行靠 git commit 加保存的配置，不靠源码快照。
- 废弃的代码直接删除，历史留在 git 里。不保留 `*_before.py`、`*_v2.py`、`candidate/` 这类并行副本。

## 2. 代码风格

- Python 3.11+（`requires-python>=3.11`，D13；`hashlib.file_digest` 需要 3.11），遵循 PEP 8，行宽 120；`python -m pyflakes src tests` 必须没有警告。依赖固定为 `transformers==5.5.3`，并声明 nltk、faiss 等运行时依赖（D13）。
- 模块开头用一段 docstring 写清楚：做什么、对应论文哪一节、有哪些已知限制。
- 函数只做一件事。命令行参数只在 `run.py` 和各评测入口里解析。
- 不用宽泛的 `except`，失败要明确报错。数据缺失必须计数并报告，不能静默跳过。
- 每个可调选项都要写进运行配置文件（`run.py` 会自动保存全部参数）。
- 实验阶段性试过的选项，没有采用就删掉，不要用开关永久保留。

## 3. 提交

- 一次提交只做一件事，提交信息写清楚**改了什么、为什么改**。
- 改动方法行为或评分的提交，要同时附上测试，或更新现有测试。
- 合并前必须全部测试通过：`python -m pytest`。
- 不提交数据、模型、运行输出和缓存（已在 `.gitignore` 中排除）。

## 4. 运行与结果

- 每次运行写到 `runs/<tag>/`，包括 `<dataset>.jsonl`（逐题、所有对照臂）、`<dataset>_config.json`（全部参数）和评分文件。配置一变就换新 `--tag`，旧结果不覆盖。
- 结果目录只放机器生成的文件。分析结论写在 `docs/results/`（或论文仓库）中，并引用对应 run 的 tag 与 commit。
- 不再新建 `*_RECEIPT.json`、`*_ACCEPTANCE.json`、`STATUS.md` 这类进度文件。进度看 git log 和运行日志即可。

## 5. 评分

- **只用官方评测**：OK-VQA VQAEval、WebQA acc_approx（可加 BARTScore）、HybridQA EM/F1 与 VideoRAG ROUGE-L/BERTScore（按 UniversalRAG 论文 Table 1）。入口只有一个：`python -m omniaware.eval.score_official`。
- **禁止**宽松匹配、同义词表、按题修补评分规则。如果某个官方评测确有缺陷，在论文中说明并单独报告，不改分数。
- **例外：引用支持度的 alias 判定。** `src/omniaware/eval/citation/attribution.py` 中的 alias 判定（某个 gold 别名出现在被引块中）是字符串匹配，字面上违反上一条。豁免理由：(1) 它只用于引用支持度（Faith、AP），**不改变任何答案分数**，答案分数仍只用官方指标；(2) 按 D9，alias 判定与 NLI 判定分别报告，论文两者都给出，不做短路组合；(3) alias 命中是可解释的快速信号，用于判断引用的块是否含答案词。该例外只适用于 `eval/citation/`，不得扩展到答案评分或其它指标。
- **例外：预注册的第二指标 `semantic_acc`。** `src/omniaware/eval/semantic.py` 用固定版本的 NLI 检查器判断预测与参考答案是否语义一致。豁免理由：(1) 它是第二指标，主指标仍只用官方评测，选择、关卡和结论只读主指标；(2) 它在任何 v2 运行之前写入 `docs/DEV_DECISIONS.md`，所有方法、所有表格都与主指标同时报告，不按结果选择；(3) 它在开发集上做盲标人工抽检，并报告与主指标的一致率。旧的宽松词面匹配仍然禁止。
- 评分不在推理时进行，不读取模型输入之外的任何信息去影响生成。

## 6. 实验协议（详见 docs/PROTOCOL.md）

- 所有对照臂（direct、static、EAB、gated 等）在**同一次运行、同一输入、同一提示词**下生成。不和旧运行的缓存预测比较。
- 校准集无标签，优先取训练集，并且永远不参与评测。
- 超参数只在 dev 上选（γ、β、τ_b、τ 的规则见 `docs/DEV_DECISIONS.md`，δ 为先验设定）；test 只在设置冻结后跑一次。
- 报告的数字必须能从 `runs/` 里的逐题文件用 `score_official` 重算出来。论文中不允许出现任何无法这样重算的数字。

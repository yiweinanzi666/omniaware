# E. 论文数字来源、统计方法与 dev 选参审查

审查对象：`C:\Users\22688\Desktop\www_code\omniaware`（只读，未修改仓库；git status 为空）。论文为 V3.21（`paper/`）。

## 0. 结论摘要

- 论文目前没有任何已填入的实验数字（全部是 `\ph` 或 `\tbd`）。所以"论文数字从哪来"要分两层看：(a) 每个表格单元有没有命令能产出；(b) 正文和附录中已经写死的数字和协议声明有没有来源。两层都有大量缺口。
- 致命问题：`videorag_synth` 在默认参数下评测集为空（F1），dev 阶段与 test 阶段都会在该数据集处中止，`runs/dev_decisions.json` 不会生成。
- 论文声称的评测规模与代码划分不符（F2）：WebQA 声称"全部 4,966 题"，代码实际 4,710 题；HybridQA 声称 2,000 题，实际 1,244 题。
- dev 选参的框架判据有明显假阳性（F3）：在零效应的零假设下，"EAB 有效"被判定为真的概率约 15–18%，名义值 2.5%。
- 论文声称 γ、β、τ_b 在 dev 上选定（F4），但代码没有 γ 的 dev 网格，β/τ_b 的网格实际跑在测试集上，与 `docs/DEV_DECISIONS.md` 的"不再调整"矛盾。
- 校准规模消融（`n_calib=100`）的测试集包含全部 256 道主 dev 题（F5）。
- 多处表格单元和附录实验没有任何命令能产出（F6、F7、F8、F14 相关条目）。
- 统计原语本身基本正确：配对 bootstrap、簇 bootstrap、AURC、Spearman、order 标准差 bootstrap 都已核实，覆盖率模拟与名义 95% 一致。问题主要在设计、子集、选择和报告口径。
- 测试：指定的 4 个测试文件共 11 个测试，全部通过。但测试没有检查区间、簇重采样、dev/test 划分、或表格字段是否存在。

## 1. 检查范围与做法

- 读过：`scripts/paper_runs.py`（全文）、`scripts/analysis/{paper_analyses,dev_select,power_design,routing_metrics,answer_recall}.py`、`src/omniaware/{analysis,run,data,pipeline,routing,order,interventions,paths}.py`、`src/omniaware/eval/score_official.py`、`src/omniaware/qwen_eab.py`（计时段）、`docs/{PAPER_TABLE_MAP,PAPER_COVERAGE,DEV_DECISIONS,PROTOCOL,citation_support_protocol}.md`、`paper/{main,sec_experiments,sec_appendix}.tex`、四个指定测试文件。
- 运行：`python -m pytest -q tests/test_paper_analyses.py tests/test_dev_select.py tests/test_analysis.py tests/test_paper_runs.py`，11 个测试全部通过。
- 一次性脚本（均在 `scratchpad\review\`）：
  - `split_check.py`：用仓库真实的 `data.split_examples` 检查 calib/dev/test 大小与交叉。
  - `sim_dev_select.py`：6 个 EAB 形式、零真实增益、4 个数据集、每个 256 题，用仓库的 `dev_select.pooled_gain` 判定 `eab_effective`，400 次仿真，每次 1000 次 bootstrap。
  - `sim_paired_iid.py`、`sim_paired.py`：用仓库的 `score_official.paired` 检查 95% 区间在零假设下的覆盖率（独立样本 / 按视频聚类 / 朴素重采样）。
  - 两段内联脚本（未单独保存）：检查 `selective_table` 的运行顺序依赖（10 题 SeaKR 运行 + 40 题主运行），以及 `paper_runs` 在无 `dev_decisions.json` 时的回退行为（`paper_runs.py --list` 与 `stages()`）。
- 不下载模型或数据，没有读取 `assets/`。论文数字的正确性因此无法实测，只能核实"有没有产出路径"和"协议是否自洽"。

## 2. 追踪表

命令前缀省略为 `python -m`；tag 省略模型后缀 `_8b`。"产出"指 `paper_runs.py` 中存在对应命令；"分析"指 `paper_runs.py` 中对该 tag 的 `scripts.analysis` 调用。

### 2.1 表 main（`sec_experiments.tex:20-47`）

| 单元 | 阶段 / tag / 臂 | 输出 | 分析 | 状态 |
|---|---|---|---|---|
| Direct（5 列） | main `v321_main` / `direct` | `runs/v321_main_8b/<d>.jsonl` → `.official_scores.jsonl` → `.official_summary.json` | `latency`（`direct_only`） | 命令存在；VideoRAG 列空（F1）；WebQA 列样本量与声明不符（F2） |
| Static（5 列） | main `v321_main` / `static` | 同上 | 同上 | 同上 |
| CE + static（5 列） | main `v321_ce`（`--selector ce`）/ `static` | 同上 | 无 latency（F8） | 指标可产出，延迟缺 |
| FITM（4 列，VideoRAG 也跑） | main `v321_main` / `fitm`（非 InternVL） | 同上 | `latency` 的 `baseline_seconds` | 可产出 |
| M2-RAG（4 列，VideoRAG 为"—"） | main `v321_main` / `m2rag` | 同上 | `latency` 的 `baseline_seconds` | 可产出 |
| SeaKR（5 列） | main `v321_seakr`（`--n-eval 1000`）/ `seakr` | 同上 | `latency(seakr)` | 1,000 题子集，论文未披露（F10） |
| UniversalRAG（5 列） | universalrag `v321_universalrag`，只跑 WebQA-full、HybridQA、VideoRAG（`paper_runs.py:125-134`） | 同上 + `.routes.json` 由 `echo` 占位的手工步骤产生（`:130-131`） | 无 latency | WebQA-img 与 OK-VQA 列无命令（F7）；路由文件不可复现（F7） |
| Omni-Aware（5 列） | main `v321_main` / `gated:EAB` | 同上 | `latency(v321_main)`、`uncertainty` 等 | 可产出；VideoRAG 受 F1 阻断 |
| Omni-Aware + CE | main `v321_ce` / `gated:EAB` | 同上 | 无 latency（F8） | 指标可产出，延迟缺 |
| † 显著性 | `analyse("paired", …)`，参照硬编码（`paper_runs.py:230-238`） | `paired` 字典，只有 CI 无 p 值 | `paper_analyses paired` | 参照不是"最强基线"（M1） |
| Latency 列 | `paper_analyses latency`（只对 v321_main 与 v321_seakr，`:221`） | JSON | — | CE、UniversalRAG、CE+EAB 无来源（F8）；口径不符（F9） |

### 2.2 表 uomni_ablation（`sec_experiments.tex:61-91`）

| 行 | 臂 / tag | 产出 | 备注 |
|---|---|---|---|
| Always retrieve, static | `v321_ablation` / `static` | 有 | R% = 100 |
| NTE, static | `v321_ablation` / `nte_shared:static` | 有 | 论文正文对 NTE 的描述与此映射不完全一致（L5） |
| Per-format thresholds, static | `v321_ablation` / `nte_trigger:static` | 有 | 阈值是按 U 的校准检索率匹配的分位数，不是"调出来的"（L5） |
| H_grd only, static | `v321_beta0_tb0.4` / `gated:static` | 有 | 注释写 WebQA-img 上 H_grd=0 而省略该行，与冷启动 gate 实现不符（M6） |
| U_omni, static | `v321_ablation` / `gated:static` | 有 | |
| U_omni, replication | `v321_ablation` / `gated:replication` | 有 | `run.py:616-629` |
| U_omni, EAB / decode-only | `v321_ablation` / `gated:EAB`、`gated:EAB_dec` | 有 | |
| Cold start without probe | `v321_ablation_noprobe` / `gated:EAB` | 有 | 只在 WebQA-img 与 HybridQA 上跑 |
| EAB + context bias | `v321_ablation` / `gated:eab:…:all:retriever:ctx` | 有 | |
| Raw retriever scores | `v321_ablation_raw` / `gated:EAB` | 有 | |
| Always retrieve, relevance bias | `v321_ablation` / `eab:…:nogate` | 有 | |
| Relevance bias without U | `v321_ablation` / `gated:eab:…:pos_nou:gate` | 有 | |
| Uniform (PASTA) bias | `v321_ablation` / `gated:eab:…:fixed:gate` | 有 | |
| Random relevance | `v321_ablation` / `gated:eab:…:gate:all:random` | 有 | |
| Acc 列 | `v321_ablation`（默认 prompt） | 有 | |
| AP 列 | `v321_cite`（`--answer-style cite`，`--max-new-tokens 128`） | 有 | Acc 与 AP 来自不同解码（M2） |
| R% 列 | `paper_analyses stats` | 有 | |

### 2.3 表 routing（`sec_experiments.tex:99-117`，WebQA open）

| 行 | tag | 产出 | 备注 |
|---|---|---|---|
| All modalities | `v321_route_all` | 有 | Mod./q = 2.00 来自 `routing_metrics` 的默认参数（`candidate_modalities` 从未写入，L2） |
| Retriever-side | `v321_route_retriever` | 有 | |
| UniversalRAG router | `v321_route_query`，依赖 `v321_universalrag` 的 routes 文件 | 有条件 | routes 文件靠手工步骤（F7） |
| M_m / H_m / M_m·H_m | `v321_route_mass` / `_entropy` / `_mh` | 有 | 均 `--first-round-routing` |
| Oracle | `v321_route_oracle` | 有 | WR = 0 为定义性结果（L2） |
| Score 列 | `paper_analyses`/`routing_metrics --arm gated:EAB` | 有 | |

### 2.4 表 order 与附录 order_full（`sec_experiments.tex:121-138`，`sec_appendix.tex:80-110`）

- 运行：`v321_perm_<id>`，14 个 `--order perm:<id>`（2 个锚点 + 12 个随机），每个数据集 `--n-eval 1000`（`paper_runs.py:172-187`）。
- 分析：`paper_analyses order` 给出 Std、flip、sorted−reverse；`paper_analyses ard` 对 `v321_perm_sorted` 给出 ARD。
- 状态：OK-VQA、WebQA-img、HybridQA 可产出；VideoRAG 受 F1 阻断（`n_calib=500` 下测试集为空）。1,000 题子集未披露（F10）。flip 阈值未在论文定义（M3）。order_full 的 "±" 第二个数字未由代码产出（L3）。

### 2.5 表 causal（`sec_experiments.tex:148-164`）

| 列 | 来源 | 状态 |
|---|---|---|
| Mask、CF | `v321_interventions`（WebQA-img，`--answer-style cite`）→ `paper_analyses interventions` | 有 |
| Dist. | `omniaware.eval.citation.attribution … --intervention distractors` | 有（需 GPU，见 `paper_runs.py:204-205`） |
| Gold@2–5 | `paper_analyses gold25 runs/v321_main/webqa_open_img.jsonl` | 有 |
| σ = 0.1 / 0.2 | `v321_noise<σ>_d{0,1,2}` → `paper_analyses noise` | 有（三次取平均） |
| Misled | `inject_answer`、`inject_instruction` → `paper_analyses interventions`（`misled`） | 有，但判定为词边界子串包含（`src/omniaware/interventions.py:38-44`），含注入实体即计为被误导 |

### 2.6 附录段落

| 段落（`sec_appendix.tex`） | 来源 | 状态 |
|---|---|---|
| β、τ_b 敏感性（:66） | `v321_beta<β>_tb<τ_b>`，OK-VQA 与 WebQA-img，**测试集** | 有，但与"在 dev 上选定"矛盾（F4） |
| 校准迁移（:68） | `v321_transfer_okvqa`（`--norm-from`），HybridQA、VideoRAG、WebQA-img | 有；VideoRAG 受 F1 阻断 |
| 校准规模（:68） | `v321_calib100/1000`（OK-VQA、WebQA-img） | calib100 泄漏 dev（F5） |
| 不校准（:68） | `v321_ablation_nocal`（`--no-calibration`） | 同时改相关度归一化（M4） |
| 分数归一化（:112） | `v321_ablation_minmax`、`_raw` 的 ARD 与 acc | 有 |
| 引用支持 Faith / AP / Inv（:70） | `attribution` 作用于 `v321_cite`（OK-VQA、WebQA-img、HybridQA） | 有；WebQA 的完整性见 F13 |
| 选择性风险 AURC、ρ（:72） | `paper_analyses selective`（seakr 在前） | 实际取自 1,000 题子集（F11） |
| 触发一致性（:114） | `paper_analyses agreement`（main） | 检索子集比较混入 EAB 效应（M9） |
| 计算开销（`sec_experiments.tex:175`） | `latency` 的 `prefill_direct_ms`、`prefill_probe_ms` | 有（只对 main） |
| 原始分数范围（:5） | `paper_analyses ranges cache/relevance_<d>*.json` | 有 |
| 检索阈值 τ 扫描（:74） | **无命令**。`paper_runs.py` 中没有 `--tau`（只有 `--tau-bias`，第 150 行） | 无来源（F6） |
| 骨干迁移（:76） | stage `main` 的 InternVL 分支与 `order`（`paper_runs.py:97-111`） | 有 |
| 次要指标（:78） | `score_official`（`--bart` 给 BERTScore） | 有 |
| 数据统计、首 token（`sec_experiments.tex:14` 提到首 token 占比） | `paper_analyses stats` | 有 |
| 错误分析（:127–131） | `paper_analyses errors`（`paper_runs.py:223`，无 `--attribution`） | 类别不可达与规则标注问题（F12） |
| 转移与损失（:131） | `paper_analyses transitions`（`paper_runs.py:226`） | 损失标签名与内容不符（M8） |
| 重排器召回（:131） | `paper_analyses stats runs/v321_ce/<d>.jsonl` 与 main 的 `gold_in_topk` | 有 |

### 2.7 论文中已写死的非占位数字与协议数字

| 数字 | 位置 | 来源 / 代码 | 结论 |
|---|---|---|---|
| 4,966；2,511；2,455（WebQA） | `sec_experiments.tex:8` | 数据统计，与 `PROTOCOL.md:10-11` 一致；仓库内无数据计数验证 | "全部 4,966 题"与代码 test=4,710 不符（F2）；2,511 的 test 为 2,255（F2） |
| 2,000（HybridQA）、374（VideoRAG） | `sec_experiments.tex:8` | 同上 | test 分别为 1,244 与 0（F1、F2） |
| 390k、540k、168,306 | `sec_experiments.tex:8` | 168,306 见 `README.md:140`、`src/omniaware/okvqa_dense.py:5`；390k、540k 与 `PROTOCOL.md` 的"39.0万/54.4万"一致 | 有出处，未由代码计数（推测：来自数据集文档） |
| 500 个无标签校准题 | `sec_experiments.tex:16`；默认 `run.py:96` | 代码默认值一致 | VideoRAG 不可行（F1） |
| 256 个 dev 题 | `sec_experiments.tex:18`；默认 `run.py:99` | 代码默认值一致 | VideoRAG dev 为 0（F1） |
| 5,000 次 bootstrap | `score_official.py:39-62`（20 × 250） | 一致 | 无问题 |
| 12 个排序、3 个种子 | `order.py:12,15-25` | 一致（2 锚点 + 3 种子各 4 个） | 无问题 |
| (τ, τ_b, β, γ) = (0.35, 0.40, 0.30, 1.2) | 默认值 `run.py:147-149`；`paper_runs.py:30`；`DEV_DECISIONS.md:23` | 仓库内无推导记录 | 来源不明（F14） |
| δ = 0.08 | `routing.py:15,53,59` | 默认值一致；`DEV_DECISIONS.md:23` 称"论文值" | 来源不明（F14） |
| γ 从 {0.6, 0.9, 1.2, 1.5, 2.0} 在 dev 上选 | `sec_appendix.tex:9` | 无 γ 网格（`paper_runs.py:28-56` 的 dev 阶段只用 γ=1.2） | 无来源（F6a） |
| 全部层优于最后四层 0.6 点 | `sec_appendix.tex:9` | 只有 `mid`（n/4–3n/4）与 `late`（n/2–n）选项（`pipeline.py:168-172`），无"最后四层"，且无命令产出 | 无来源（F6b） |
| "lowers the order-induced std by X% to 70%" | `main.tex:130` | 70% 不由任何命令产出 | 无来源（F6c） |
| 8 个均匀采样帧 | `sec_experiments.tex:16` | `src/` 中未找到帧数配置（grep `num_frames`、`n_frames` 无结果） | 来源不明（推测：在数据准备脚本中，未核实） |
| e^{1.2} ≈ 3.3 | `main.tex:248` | 算术正确（e^{1.2} = 3.32） | 无问题 |
| 20 次 SeaKR 采样 | `paper_runs.py:120` | `--seakr-samples 20` | 无问题 |
| 三次噪声 draw | `paper_runs.py:207-211` | 一致 | 无问题 |

## 3. 问题清单

严重程度：致命 / 严重 / 中等 / 轻微。"推测"表示结论依赖未核实的前提，已说明需要的证据。

### 致命

**F1（致命）`videorag_synth` 的评测集为空，dev 与 test 流水线在该数据集处中止**
- 位置：`scripts/paper_runs.py:24`（CALIB 为 heldout）、`:81-91`（dev 阶段包含 videorag）、`:112-123`、`:125-134`；`src/omniaware/run.py:96`（默认 `--n-calib 500`）、`:242-245`；`src/omniaware/data.py:370-371`；`src/omniaware/eval/score_official.py:113`。
- 证据：`split_check.py` 输出 `VideoRAG-Synth, heldout calib (500)  n=374 … calib=374 dev=0 test=0`。`run.py` 的 `evals` 为空，于是写出空的 jsonl；`score_run` 第 113 行的 `assert len(families) == 1` 在空文件上失败。dev 阶段的 `subprocess.run(..., check=True)` 因此中止，videorag 是 dev base 循环的最后一个数据集，后面的 zsum/mid 循环和 `dev_select`（`paper_runs.py:91`）都不会执行。
- 影响：论文表 main、uomni、order、routing（如果含 videorag）的 VideoRAG-Synth 列全部无法产出；`runs/dev_decisions.json` 不生成，整个"预先登记"框架失效。
- 修复：heldout 数据集须满足 N ≥ n_calib + n_dev + 最小测试量。VideoRAG 需要 train 查询，或减小 calib 并重新定义 dev/test。在 `run.py` 中对空 `evals` 直接 `ap.error`。
- 推测：否（已由代码阅读与划分运行验证）。

### 严重

**F2（严重）论文声称的评测规模与代码划分不符，且未披露 dev 扣除**
- 位置：`paper/sec_experiments.tex:8`（"evaluate all 4,966 validation questions"，"2,000 HybridQA and 374 VideoRAG-Synth questions"）；`src/omniaware/data.py:357-372`（`dev = rest[:n_dev]`，`test = rest[n_dev:]`）；`run.py:244-245`。
- 证据（`split_check.py`，使用仓库真实的 `split_examples`）：WebQA 全集（calib 来自 train）test=4,710（dev 256 被扣除）；OK-VQA（calib heldout 500）test=4,290；HybridQA test=1,244；VideoRAG test=0。WebQA-img（N=2,511，calib 来自 train）test=2,255。WebQA-text（N=2,455）按同一算术约为 2,199（未单独运行）。
- 影响：所有"full val"列与分母都与正文不符。论文同时写"no number in this paper comes from development questions"（`sec_experiments.tex:18`），因此正确的说法应该是"测试集为全集去掉 dev 与 calib"。
- 修复：在论文中写出每个数据集的 calib/dev/test 大小；或在冻结选择后把 dev 题也纳入最终测试并说明。
- 推测：否。

**F3（严重）dev 选择的 EAB"有效"判据假阳性高（best-of-6，CI 未校正）**
- 位置：`scripts/analysis/dev_select.py:57-71`（`select_eab` 取最大增益）、`:114`（`eab_effective` 直接用所选形式未校正的 95% 区间下界）、`:46-54`（`pooled_gain`）；`docs/DEV_DECISIONS.md:20`（规则 1）与 `:24`（规则 5 承认 best-of-K 乐观偏差，但代码未处理）。
- 证据：`sim_dev_select.py`，零真实增益，6 个形式、4 个数据集、n=256、每题 sd=25，400 次仿真，每次 1000 次 bootstrap。P(eab_effective | 零假设)：独立形式 0.180；ρ=0.5 时 0.145；ρ=0.8 时 0.147。单个预先指定的形式为 0.020–0.030，与名义 0.025 一致。
- 影响："EAB 有效"决定论文框架（`dev_select.py:117`，framing 1/2/3）。framing 1 对应完整主张（`DEV_DECISIONS.md:30`）。在零效应下，约 15–18% 的情况会错误地进入 framing 1 的主张。
- 修复：对 framing 判据使用 max-T 或 Bonferroni 校正的区间；或把 dev 判据仅作为建议，把确认留给一次性的 test 检验（且 test 只跑冻结配置）。
- 推测：仿真参数（sd=25、独立或 ρ 取值、1000 次 bootstrap）是假设值，需要真实 dev 的 per-question 差值方差来校准。假阳性的方向与量级本身是可靠的。

**F4（严重）γ、β、τ_b 的"dev 选定"声明与代码不符**
- 位置：`paper/sec_appendix.tex:9`（"γ is chosen from {0.6,…,2.0} on development queries"）、`:66`（"β and τ_b were selected on development queries"）；`scripts/paper_runs.py:28,84-90`（dev 阶段只用 γ=1.2，无 γ 网格）、`:147-150`（β×τ_b 网格在 `stages(…, split)` 的 ablation 阶段，默认 `--split test`）；`docs/DEV_DECISIONS.md:23`（"其余超参数不再调整 … β、τ_b 的敏感性只在附录报告"）。
- 证据：`paper_runs.py` 的 dev 阶段没有 γ、β、τ_b 的变化。β×τ_b 网格只出现在 ablation 阶段，且 `run()` 默认 `split` 为 test（`paper_runs.py:65-67`）。
- 影响：论文对超参数来源的叙述与代码、与 DEV_DECISIONS 三者互相矛盾。"β=0.30 scores highest at every τ_b on OK-VQA"（`sec_appendix.tex:66`）是在测试集上的事后比较。如果把它当作选择依据，就是在测试集上调参。
- 修复：二选一。(a) 把 γ、β、τ_b 的网格加入 dev 阶段并冻结，同时修正 DEV_DECISIONS；(b) 改写论文：这些值是事先给定的（需说明来源，见 F14），β×τ_b 网格是测试集上的敏感性分析，不构成选择。
- 推测：否。

**F5（严重）校准规模消融把全部主 dev 题放进测试集**
- 位置：`scripts/paper_runs.py:144-146`（`v321_calib{100,1000}`，OK-VQA 与 WebQA-img，`--split` 默认 test）；`src/omniaware/data.py:370-371`（`rest = order[n_calib:]`，dev 与 test 都从 `rest` 切出）。
- 证据：`split_check.py`：主配置（n_calib=500）的 256 道 dev 题，全部落在 n_calib=100 运行的测试集中（256/256）；n_calib=1000 的测试集与之无交（0/256）。WebQA-img 使用 train 校准，n_calib 不影响切分，因此只有 OK-VQA 的 calib100 受影响。
- 影响：`sec_experiments.tex:18` 声称"no number in this paper comes from development questions"，被 calib100 行违反。相关的 OK-VQA 准确率比较（`sec_appendix.tex:68`）也不是在相同题目上比较的。
- 修复：先固定 dev 与 test，再从剩余题中抽取 calib；所有 calib 规模在同一测试集上重跑。
- 推测：否。

**F6（严重）无来源或无命令产出的数字与实验**
- (a) γ 从 {0.6,…,2.0} 在 dev 上选择（`sec_appendix.tex:9`）：无命令（见 F4）。
- (b) "全部层优于最后四层 0.6 点"（`sec_appendix.tex:9`）：无命令。代码只有 `mid` 与 `late` 两个层选项（`src/omniaware/pipeline.py:168-172`），没有"最后四层"，`paper_runs.py` 也从未使用这两个选项。
- (c) "by X% to 70%"（`paper/main.tex:130`）：70% 不由任何命令产出，也没有表格单元对应。
- (d) "Retrieval threshold"段落（`sec_appendix.tex:74`）：描述"varying τ to lower retrieval from 100% to X%"，但 `paper_runs.py` 从不传 `--tau`（全文只有 `--tau-bias`，第 150 行）。
- (e) `sec_experiments.tex:12` 的"baselines receive \tbd{tuning budget}"是占位，基线调参预算未说明（与公平性相关）。
- 影响：这些数字或结论没有任何可重算的来源，按 `PROTOCOL.md:90` 的报告规则应删除。
- 修复：每个数字都要么有命令产出，要么删除。
- 推测：否。

**F7（严重）UniversalRAG 行缺少两列的命令，路由文件由手工占位步骤产生**
- 位置：`scripts/paper_runs.py:125-134`（只跑 `webqa_open_full`、`hybridqa`、`videorag_synth`）、`:130-131`（`echo` 占位）；`paper/sec_experiments.tex:41`（表中 UniversalRAG 行的 WebQA-img 与 OK-VQA 列为 `\ph`）。
- 证据：阶段中没有 `webqa_open_img` 或 `okvqa` 的 UniversalRAG 运行，但表格有对应单元。`--route-rule query` 需要 `--query-routes`（`run.py:195-196`），而 routes 文件只能靠手工运行官方路由器产生。
- 影响：WebQA-img 与 OK-VQA 的 UniversalRAG 单元无来源；其余单元依赖一个不可由命令复现的中间文件。
- 修复：补上缺失列的运行，或对 OK-VQA 标"—"（UniversalRAG 的语料不含 OK-VQA，需确认）；把路由步骤脚本化，或记录路由文件的哈希。
- 推测：UniversalRAG 不含 OK-VQA 语料是我的推测，未在仓库中核实。

**F8（严重）延迟列缺少 CE、UniversalRAG、Omni-Aware+CE 的分析命令**
- 位置：`scripts/paper_runs.py:220-221`（只对 `v321_main` 与 `v321_seakr` 调用 `latency`）；`paper/sec_experiments.tex:26,43-44`（表中延迟列对全部方法有单元）。
- 影响：`v321_ce` 与 `v321_universalrag` 的延迟列无来源。`latency_table` 的 `universalrag` 键只在 route rule 为 query 的运行中出现（`paper_analyses.py:216-220`），因此 UniversalRAG 的延迟需要单独调用。
- 修复：为 `v321_ce`、`v321_ce` 的 `gated:EAB`、`v321_universalrag` 增加 latency 调用。
- 推测：否。

**F9（严重）延迟的定义与论文描述不符**
- 位置：`paper/sec_experiments.tex:18`（"Latency covers uncertainty scoring, retrieval, evidence preparation, and generation … measured for all methods"）；`scripts/analysis/paper_analyses.py:205-207`（docstring："retrieval lookup is cached and excluded"）、`:221-228`（由各段计时合成）；`src/omniaware/run.py:630-661`（FITM、M2-RAG、SeaKR 为墙钟时间）；`src/omniaware/qwen_eab.py:320,327`（`last_prefill_seconds` 只计未偏置前向，`make_variants` 之后的不确定性与偏置计算不计时）。
- 证据：`latency_table` 的 `gated:*` 是 `direct_prefill + probe + (evidence_prefill + arm_decode 或 direct_decode)` 的合成，而非逐题端到端计时。
- 影响：门控系统的延迟是对成本的模型，不是测量；基线（FITM、M2-RAG、SeaKR）是墙钟时间。两者口径不同，相互比较无效。`paper/sec_experiments.tex:175` 的"uncertainty computation adds X ms"只包括前向，不包括偏置计算。
- 修复：要么对所有方法做逐题端到端计时（相同批大小、相同缓存策略、相同预热），要么把表头改为"模型合成延迟"并把基线单列。检索查表成本要单独报告。
- 推测：否（口径由代码直接确定）。

**F10（严重）未披露的 1,000 题子集（SeaKR、排序实验、显著性）**
- 位置：`scripts/paper_runs.py:29`（`SUBSAMPLE = "1000"`）、`:107,120,179`（排序与 SeaKR 使用）、`:232-233`（SeaKR 的 † 检验）；`paper/sec_experiments.tex` 与 `sec_appendix.tex` 全文无"1,000"或子集说明。`docs/PAPER_TABLE_MAP.md:16,61` 记录了子集，但论文没有。
- 影响：表 main 的 SeaKR 列、表 order 的全部数字（`paper_runs.py:186`，"1,000 questions per dataset"）、选择性风险（F11）都基于 1,000 题，而论文把它们与全测试集结果并列，并称所有方法"在同样的候选上回答每个问题"（`sec_experiments.tex:12`）。
- 修复：在表注中写明子集大小，或对完整测试集运行。
- 推测：否。

**F11（严重）选择性风险结果实际取自 1,000 题的 SeaKR 运行（参数顺序导致）**
- 位置：`scripts/analysis/paper_analyses.py:309-328`（`systems` 逐路径循环，第 320 行 `if name in out: continue`，先到者胜出）；`scripts/paper_runs.py:224`（传入顺序为 `seakr` 在前、`path` 在后）。
- 证据：内联检查，构造 10 题的 seakr 运行与 40 题的主运行，`selective_table` 的三个系统（U_omni、NTE、SeaKR）全部取自 10 题的 seakr 运行（`n=10`），主运行从未被用到。在真实命令中，这意味着 U、NTE、SeaKR 全部在 1,000 题子集上计算。
- 影响：`sec_appendix.tex:72` 的 AURC 与 ρ 数字都是子集结果，且 U 的数字不是全测试集上的 U。
- 修复：U 与 NTE 从主运行取，SeaKR 从 seakr 运行取，并在表中写明 n。
- 推测：否。

**F12（严重）错误分析：不可达的"unsupported citation"类别、规则标注未披露、OK-VQA 兜底**
- 位置：`scripts/paper_runs.py:223`（不传 `--attribution`）；`scripts/analysis/paper_analyses.py:279-280`（`faith == 0` 才归为 unsupported citation）、`:281`（有 query image 即归为 ambiguous visual grounding）、`:271-283`（类别判定顺序）；`src/omniaware/eval/citation/attribution.py:146-147`（attribution 只接受 `--answer-style cite` 的运行，即 `v321_cite`，不是主运行）；`paper/sec_appendix.tex:129-131`（列出 unsupported citations 类别，未说明是规则标注，也未说明样本量）。
- 证据：流水线中 `faith` 为空字典，所以 unsupported citation 类别在该命令中永远不会出现。OK-VQA 的每题都有 query image，因此所有未被前面类别覆盖的 OK-VQA 错误都归入 ambiguous visual grounding，`other` 对 OK-VQA 不可达。`answer_normalization` 用的是子串包含判定（`interventions.py:38-44`），含有金标答案词的错误会被误归为"归一化"。
- 影响：表中各类比例主要反映规则的先后顺序，而非错误的真实原因。`PAPER_TABLE_MAP.md:90` 说明是"规则预标注 + 抽样人工核对"，论文正文没有这一说明。
- 修复：把 attribution 对主运行的 arm 加入流水线（或在报告中注明 Faith 来自另一运行）；去掉或重写兜底规则；对样本做人工标注并报告一致率；在论文中说明方法。
- 推测：否。

**F13（严重）引用支持有两条不一致的管线，其中一条只完成了 WebQA 的 257/4,966 题（推测：论文引用数字的来源未知）**
- 位置：`docs/citation_support_protocol.md:13`（"WebQA val 4966 | 257 | 当前已完成正式 top5 的 257 题…其余 4709 题…pending"）；同文件"生成与 U 的边界"一节（"U 的观测不随 citation prompt 重算 … 复用原 query-only 初始观察和冻结 500 题校准"）；对比 `src/omniaware/run.py:539`（U 用当前 `args.answer_style` 的 prompt 重算）。
- 证据：E5 包声明 U 复用旧观测而不重算，而 `run.py` 的 `v321_cite` 运行会重算 U。两者的 U 与生成条件不同。
- 影响：论文正文 `sec_appendix.tex:70` 报告 WebQA 的 Faith 与 AP。如果这些数字来自 E5 包，WebQA 只有 257 题，并且 U 与原生成条件不一致。这一点无法在仓库中核实。
- 修复：明确论文的引用实验来自哪条管线；只保留一条；重新生成完整 WebQA 结果。
- 推测：是。需要知道论文中 Faith/AP 的生成命令与运行 tag。

**F14（严重，推测）超参数 (τ, τ_b, β, γ, δ) 来源无记录，若来自旧仓库在测试集上的调参则测试集已被污染**
- 位置：`src/omniaware/run.py:147-149`、`scripts/paper_runs.py:30`、`src/omniaware/routing.py:15`、`docs/DEV_DECISIONS.md:23`（"保持论文值"）。
- 证据：在 README、CONVENTIONS、MIGRATION_SOURCES、LEGACY_MAP 中搜索 0.35、0.40、0.30、1.2 均无推导记录。审查任务描述称旧仓库"在测试集上反复调参"。
- 影响：如果这些值来自旧仓库在测试集上的调参，那么它们的测试结果不是干净的，DEV_DECISIONS 的整个机制也无法补救。
- 修复：记录每个超参数的来源（commit、运行 tag、日期）。若来自旧测试集调参，必须在新的 dev/test 划分上重新确认。
- 推测：是。需要旧仓库的运行记录。

### 中等

**M1（中等）显著性：只输出 95% 区间而非 p 值；参照组按列硬编码；未做多重比较校正**
- 位置：`src/omniaware/eval/score_official.py:39-62`（`paired` 返回百分位区间，没有 p 值）；`paper/sec_experiments.tex:18,21`（"paired bootstrap tests … against the strongest baseline (p<0.05)"）；`scripts/paper_runs.py:230-238`（参照为 m2rag（WebQA、OK-VQA、HybridQA）、static（VideoRAG）、seakr、static（CE）、universalrag）。
- 证据：`sim_paired_iid.py`（对称零假设，n=256）覆盖率 0.955；`sim_paired.py` 簇重采样覆盖率 0.957，朴素重采样 0.757。所以区间本身是校准的，问题在报告层面。
- 影响："p<0.05 against the strongest baseline"与代码不符：没有 p 值，参照组不是按"最强"选出的，五个数据集各有三个未校正的比较（M2-RAG 或 static、SeaKR、CE），其中三个数据集另有 UniversalRAG 比较，合计 18 个未校正的 † 检验（`paper_runs.py:218-238`）。
- 修复：报告区间与 p 值；参照组按明确规则选定（如同数据集上 dev 表现最好的基线），或改写为描述性；对多重比较做 Holm 校正。
- 推测：否。

**M2（中等）Acc 与 AP 来自不同解码**
- 位置：`scripts/paper_runs.py:136-139`（ablation，默认 prompt）与 `:189-195`（cite，`--max-new-tokens 128`）；`docs/PAPER_TABLE_MAP.md:27,45`。
- 影响：同一行的准确率和 AP 不是同一批生成的答案，`sec_experiments.tex:146` 的"higher answer scores come with better citation support"无法由该表直接支持。
- 修复：在 cite 运行中同时报告 Acc，并量化 prompt 对准确率的影响；或说明两者分别来自哪次生成。
- 推测：否。

**M3（中等）"正确"阈值未在论文定义，且各处不一致**
- 位置：`scripts/analysis/paper_analyses.py:418`（`--correct-at` 默认 50）、`:93,99`（flip 使用）、`:357-358`（transitions 使用）；`scripts/analysis/answer_recall.py:42`（`> 0`）。
- 证据：VQA 准确率只取 {0, 33.3, 66.7, 100}，阈值 50 把 33.3 判为错误。`answer_recall` 的阈值是 `> 0`。
- 影响：flip rate（`sec_experiments.tex:10`）与错误转移的数字依赖一个未定义的阈值。
- 修复：在论文中定义阈值，并报告对阈值的敏感性。统一各脚本。
- 推测：否。

**M4（中等）"不校准"消融同时改变了相关度归一化**
- 位置：`src/omniaware/run.py:193-194`（`--no-calibration` 强制 `relevance_norm = "minmax"`）、`scripts/paper_runs.py:140-143`；`paper/sec_appendix.tex:68`（"Without calibration, the controller uses heuristic default statistics…"）。
- 影响：`v321_ablation_nocal` 的结果混合了两个改动，论文只说明了其中一个。
- 修复：拆成两个标志，分别做消融。
- 推测：否。

**M5（中等）dev 决策缺失时静默回退到硬编码默认；framing 不做门控；语料选择范围静默缩小**
- 位置：`scripts/paper_runs.py:34-40`（找不到 `runs/dev_decisions.json` 或模型不匹配时直接返回 DEFAULTS，无报错）、`:30`（DEFAULTS 的 EAB 形式为 ans1）；`scripts/analysis/dev_select.py:118-120`（无证据时仍写入默认 variant 与 `eab_effective=False`）；`scripts/paper_runs.py:79-82`（缺失候选的 OK-VQA 语料被跳过，只在 `--list` 中打印）；`src/omniaware/paths.py`（`RUNS` 由环境变量 `OMNIAWARE_RUNS` 决定）与 `paper_runs.py:35`（固定读取 `ROOT/"runs"`），两者可能不一致。
- 证据：`python -m scripts.paper_runs --list` 输出 `frozen choices: {'eab_variant': 'eab:g1.2:ans1:pos:gate', …}`，但 `runs/` 目录不存在，这些值是默认值而非 dev 选择结果。`--list` 同时显示 okvqa、okvqa_dense 被跳过。
- 影响：test 阶段可以在没有 dev 选择的情况下运行，且不会被发现；论文的"预先登记"只是口头声明。
- 修复：test 阶段在 `dev_decisions.json` 缺失时直接报错；framing 写入论文前必须核对；记录被跳过的语料；统一运行目录的解析方式。
- 推测：否。

**M6（中等）"H_grd = 0 on WebQA-img"与冷启动 gate 的实现不符（推测：数值待跑）**
- 位置：`paper/sec_experiments.tex:62`（表注）；`src/omniaware/run.py:559-560`（冷启动 gate 用 `u_probe`）、`:553`（probe 轮）；`src/omniaware/pipeline.py:127-131`（H_grd 在所给 spans 上计算）；`paper/main.tex:203`（"When no evidence is supplied, H_grd=0"，但 probe 轮提供了 spans）。
- 证据：WebQA-img 为冷启动，gate 使用 probe 轮的 H_grd，它在多个 span 上计算，通常非零。`paper_runs.py:147-150` 也确实在 webqa_open_img 上跑了 beta0 运行。
- 影响：表注给出的省略理由与实现不符。
- 修复：修正表注，或说明省略的真实理由。
- 推测：是。需要核对 `v321_beta0_tb0.4` 在 WebQA-img 上的 H_grd 分布。

**M7（中等）"a single zero-bias prefill … yields all three control decisions"与实际的多次前向不符**
- 位置：`paper/main.tex:84`（摘要）与 `:128`（贡献 2）；`src/omniaware/run.py:549`（direct 前向）、`:553`（冷启动 probe 前向）、`:585` 与 `:600`（evidence 前向）、`:624-626`（replication 前向）。
- 影响：完整系统至少两次前向（direct 与 evidence），冷启动再加一次 probe。正文 `sec_experiments.tex:175` 也承认"triggered queries need a second prefill"。摘要与正文矛盾。
- 修复：改写摘要为"决策读取同一次零偏置前向的注意力"，并注明 EAB 需要第二次前向。
- 推测：否。

**M8（中等）损失类别的名称与代码不符**
- 位置：`scripts/analysis/paper_analyses.py:362-375`（损失只按 top 块的模态分类：`top_chunk_image`、`top_chunk_table`、`top_chunk_video`，再按 chunk_has_answer 字符串判断）；`paper/sec_appendix.tex:131`（"an unrelated image region"、"a misaligned table cell"）。
- 影响："unrelated"与"misaligned"是内容判断，代码只看模态。论文中的比例实际是模态计数。
- 修复：改名为"top 块为图像/表格"，或做内容标注并报告一致率。
- 推测：否。

**M9（中等）触发一致性比较混入了 EAB 效应**
- 位置：`scripts/analysis/paper_analyses.py:349`（`"NTE (static)": score("static", nte)` 与 `"U (EAB)": score(eab_arm, u)` 在各自检索子集上比较）；`paper/sec_appendix.tex:114`（"retrieved questions reach X and Y in the respective complete configurations"）。
- 影响：两个数字的臂不同，差异既来自触发器，也来自 EAB，无法归因于触发器。
- 修复：用同一臂（如 static 对 static，或 EAB 对 EAB）比较检索子集。
- 推测：否。

### 轻微

**L1（轻微）功效分析未接入流水线；docstring 与实现的描述不完全一致**
- 位置：`scripts/analysis/power_design.py:29-33`（MDE = 2.8·SE，公式正确；best-of-K 以样本均值为中心）；`docs/PROTOCOL.md:72`（要求在 dev 比较前运行）。
- 证据：`paper_runs.py` 与 `dev_select.py` 都不调用 power_design，`docs` 中也没有输出结果。docstring 写"K candidates with true effect zero"，实现则以样本均值为中心，两者在数值上接近但描述不同。
- 修复：在 dev 阶段前运行并记录结果；修正 docstring。
- 推测：否。

**L2（轻微）routing 的"All modalities"Mod./q 与 Oracle 的 WR 不是实验测量**
- 位置：`scripts/analysis/routing_metrics.py:19`（`candidate_modalities` 从未被 `run.py` 写入，回退到 `--modalities` 默认值）；`src/omniaware/run.py:398`（Oracle 直接取 `gold_modalities`，所以 WR=0 为定义性结果）。
- 修复：在 `run.py` 中写入 `candidate_modalities`；在表注中说明 Oracle 的 WR 为定义性。

**L3（轻微）order_full 的"±"第二个数字未由代码产出**
- 位置：`scripts/analysis/paper_analyses.py:91-92`（只输出各种子组的标准差列表 `std_per_seed`，不输出跨种子的标准差）；`paper/sec_appendix.tex:81,86-92`。
- 修复：输出跨种子的均值与标准差，并在表注中注明含义。

**L4（轻微）默认 EAB 形式在不同文件中不一致**
- 位置：`docs/PROTOCOL.md:33`（写为 `ans2`，"即论文 decode-only"）；`docs/DEV_DECISIONS.md:13`（ans1 为论文默认）；`scripts/paper_runs.py:30` 与 `src/omniaware/run.py:155`（ans1）。
- 修复：统一为 ans1，修正 PROTOCOL。

**L5（轻微）NTE 与 nte_trigger 的描述不符**
- 位置：`paper/sec_experiments.tex:93`（"it needs one tuned threshold per modality"）；`src/omniaware/run.py:470-472`（阈值为 U 校准检索率匹配的分位数，不是调出来的）。
- 修复：改写为"按 U 的校准检索率匹配的分位数阈值"。

**L6（轻微）`answer_recall.py` 无调用方，论文未引用**
- 位置：`scripts/analysis/answer_recall.py`（全文）；`scripts/paper_runs.py` 与 `paper/` 中无引用。
- 影响：信息性，不影响论文数字。

**L7（轻微，推测）dev 阶段对 WebQA-open 的 train 候选不做检查**
- 位置：`scripts/paper_runs.py:81-91`（webqa_open_full 的 dev 运行需要 `TRAIN_LOADERS` 的 train 候选，`src/omniaware/data.py:343-347`）；`docs/PAPER_COVERAGE.md:25`（"索引已建，val 检索结果待在主机上生成"）。
- 推测：若候选缺失，dev 阶段会在 webqa 处中止。需要确认主机上的 train 候选是否存在。

## 4. 已核实无误的部分

- 配对 bootstrap（`score_official.paired`）：独立样本覆盖率 0.955（n=256，400 次）、0.965（n=1000）、0.958（n=64）。使用对称零假设，真实差值为 0。
- 按视频聚类的 bootstrap（ratio 估计 `sums[ids].sum/counts[ids].sum`）：覆盖率 0.957（300 次），而朴素重采样为 0.757。聚类实现正确。
- `order_std_statistics`（`order.py:33-54`）：按题重采样、跨排序重算标准差，配对正确。
- `order_table` 的标准差、flip 与 sorted−reverse 计算（`paper_analyses.py:71-103`）：与论文定义一致（阈值问题见 M3）。
- `analysis.aurc`（按不确定性升序、累计平均错误）与 `analysis.spearman`（平均秩处理并列）：与标准定义一致。
- `dev_select.pooled_gain`（按数据集分层的 bootstrap，每个数据集权重相同）：正确。
- `pipeline.Norm`（`pipeline.py:78-83`）与论文 Eq.(norm) 一致，IQR 下限 1e-6 也一致。
- `routing.route_modalities`（`routing.py:15-43`）：M_m·H_m 排序与 δ=0.08 的并列规则与论文描述一致。
- `permutation_schedule()`（`order.py:15-25`）：2 个锚点加 3 个种子各 4 个不重复的随机排列，共 14 个，与论文一致。
- 同一次运行内的 calib/dev/test 不相交（`data.py:357-372`，`run.py:254` 断言）。跨运行的问题见 F5。
- `power_design` 的 MDE 公式（2.8·SE，两侧 5%、80% 功效）正确。
- 11 个指定测试全部通过。

## 5. 没来得及深入的方向

- `src/omniaware/eval/webqa_acc.py`、`universalrag_metrics.py`（WebQA acc_approx、HybridQA EM/F1、VideoRAG ROUGE-L）的实现，未逐行核对。
- `src/omniaware/eval/citation/attribution.py` 的 Faith/AP/Inv 计算，未核对。
- `src/omniaware/baselines/universalrag.py`：`overlapping()` 函数存在，但没有核对是否真的用它剔除了评测题，也没有核对路由器训练脚本（不在仓库内）。
- `relevance.load_relevance`（`run.py:24-34`）只按 qid 复用缓存，不检查候选池是否变化。
- `src/omniaware/paths.py` 中 `WEBQA_OPEN` 默认指向旧的 `runs/phase1_dense_retrieval_codex_20261006`，其内容来源未核实。
- 延迟中 direct 与 static 是否都在相同的注意力捕获开销下计时，未核实。
- 数据计数（4,966、2,511、2,455、2,000、374、5,046）只能由数据文件验证，本次无数据，未验证。
- `tests/` 中的其他测试（如 `test_order.py`、`test_routing_rules.py`）不在本次范围内。

## 6. 验证脚本位置

- `C:\Users\22688\AppData\Local\Temp\claude\C--Users-22688-Desktop-www-code\8e3454d9-d7cd-4088-813b-e62b2bcd35b9\scratchpad\review\split_check.py`
- `…\scratchpad\review\sim_dev_select.py`
- `…\scratchpad\review\sim_paired_iid.py`
- `…\scratchpad\review\sim_paired.py`（第一版的 iid 检查有设计错误，已由 `sim_paired_iid.py` 替换；聚类部分仍有效）

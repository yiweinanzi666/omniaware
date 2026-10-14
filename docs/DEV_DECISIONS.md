# Dev 阶段预先登记的决策规则

写定日期：**2026-10-10**（提交 `cd48ab8`，"pre-registered dev selection"）。此前本文件标注的日期 2026-10-11 有误，它晚于该提交的时间（2026-10-10 10:59 +0800），以 2026-10-10 为准。

目的：避免在测试集上挑选设置，也避免事后按结果选择论文主张。规则由 `scripts/analysis/dev_select.py` 机械执行，结果写入 `runs/dev_decisions.json`（含所用 dev 运行文件的 sha256）。`scripts/paper_runs.py` 的 test 阶段只读取 **phase 2 完整** 的决定文件；缺失、模型不匹配或只有 phase 1 时直接报错退出。

## 运行（两阶段，因为 β、τ_b 会改变 EAB 偏置的强度）

```bash
python -m scripts.paper_runs --stage dev --run            # phase 1 的全部 dev 运行（--split dev）
python -m scripts.analysis.dev_select --phase 1           # -> runs/dev_decisions.json（phase 1）
python -m scripts.paper_runs --stage dev_beta --run       # β × τ_b 网格，使用 phase-1 选出的 EAB 候选
python -m scripts.analysis.dev_select --phase 2           # -> 完整的 runs/dev_decisions.json（phase 2）
python -m scripts.paper_runs --stage main --run           # test 阶段（只在 phase 2 完整时运行）
```

划分（D1、D2）：

| 数据集 | calib（保留区，不进 dev 和 test） | dev | test（评测） |
|---|---|---|---|
| OK-VQA（`okvqa*`） | 500（保留区大小 `CALIB_RESERVE`） | 256 | 5,046 − 500 − 256 = 4,290 |
| HybridQA | 500 | 256 | 2,000 − 500 − 256 = 1,244 |
| VideoRAG-Synth | 100 | 50 | 374 − 100 − 50 = 224 |
| WebQA | 训练集打乱后的前 `n_calib` 题 | 训练集打乱后接下来的 256 题 | 全部 4,966 道 val；图像题子集为 `webqa_open_img` |

dev 网格：
- **EAB 候选（30 个）**：起始行 {ans1（从预测第一个答案 token 的行开始，论文默认）、ans2（decode-only）、q（从问题开始）} × 偏置形式 {pos：γUr，ctr：γU(2r−1)} × **γ ∈ {0.6, 0.9, 1.2, 1.5, 2.0}**。在默认 β、τ_b 下于同一个 base 运行中全部计算。
- **β × τ_b（5 × 3 = 15 个运行）**：β ∈ {0, 0.15, 0.3, 0.5, 1.0}，τ_b ∈ {0.2, 0.4, 0.6}，只跑 phase-1 选出的那一个 EAB 候选。
- **τ ∈ {0.25, 0.35, 0.45}**：没有单独的运行。τ 只决定门控 `retrieve = (u_gate > τ)`，而每个臂对每题都解码，所以门控后的分数可以从所选 (β, τ_b) 的运行中离线精确算出（`paper_analyses tau` 会与运行自身的门控分数核对，不一致即报错）。
- **不确定性设置（3 种）**：sum + 末层（论文）、zsum（两项各自标准化）+ 末层、sum + 中间层。
- **OK-VQA 检索语料（2 种）**：BGE 维基 3M（`okvqa_dense`）、BGE GS-112K 训练语料（`okvqa_gs`，112,724 句）。原先的 BM25 维基（`okvqa`）候选依赖旧主机生成、仓库中没有的输入，无法复现（README 4.2），已在任何 dev 运行之前从候选中移除。`gs168k_full` 含测试题的检索结果，不参加选型，只用于附录。
- 冷启动数据集（WebQA、HybridQA、VideoRAG-Synth）一律使用探测轮（`--probe-uncertainty --first-round-routing`）。
- 缺少任一语料的候选文件时，dev 阶段直接报错，不跳过。

## 规则

选择顺序：语料 → 不确定性设置 → EAB 候选 → β、τ_b → τ。每一步只用前面步骤的结果。

1. **OK-VQA 语料。** 选 static − direct 差值最大的语料。"检索有用"当且仅当这个差值 ≥ 0；否则 OK-VQA 在论文中作为"检索可能有害、门控用来避免伤害"的场景报告，不作为增益来源。
2. **不确定性设置。** 指标是门控分数与直答错误的 Spearman ρ，跨 4 个 dev 数据集（WebQA、OK-VQA-GS、HybridQA、VideoRAG-Synth）取平均；选 ρ 最大的设置。"U 优于 NTE"当且仅当所选设置的平均 ρ 大于下一 token 熵的平均 ρ。
3. **EAB 候选（形式 × γ）只选一种，全局通用。** 指标是同一输入下 EAB 相对 static 的逐题增益：先在每个数据集内取平均，再对数据集取平均（每个数据集权重相同）。选平均增益最大的候选。不允许按数据集分别选择。
   **"EAB 有效"的判据做 30 选 1 的 max-statistic bootstrap 校正**：题目在数据集内分层、各候选共享同一组重采样（配对）；对每个 bootstrap 样本计算各候选的标准化偏差 t*_k = (g*_k − g_k)/se_k，取 M* = max_k t*_k；所选候选的单侧同时下界为 g − q_{0.95}(M*)·se。"EAB 有效"当且仅当该下界大于 0（等价于 p_max-stat < 0.05）。零效应仿真见 `tests/test_dev_select.py`。
4. **β、τ_b。** 在 phase-2 的 15 个运行上，对 phase-1 的 EAB 候选计算按数据集等权的平均增益，选最大者（网格中先出现者优先）。这一步不再做显著性检验：EAB 有效性已由规则 3 判定。
5. **τ。** 在所选 (β, τ_b) 的运行上，离线计算三个 τ 下按数据集等权的平均门控分数，选最大者；并列时取最接近论文值 0.35 的一个。
6. **其余超参数不再调整。** δ = 0.08（路由边际，固定先验，未调参）、k = 5 保持论文值。
7. **最佳候选的乐观偏差。** 所有选择都在 dev 上完成；论文中的数字只来自 test，dev 数字不进正文。β×τ_b 与 τ 在 test 上的扫描只作为**敏感性报告，不参与选择**（paper_runs 的命令注释和 fills 中已标明）。

## 论文框架（由规则 2、3 机械决定）

| framing | 条件 | 论文主张 |
|---|---|---|
| 1 | EAB 有效，且 U 优于 NTE | 完整主张：一次前向决定是否检索、往哪检索，以及如何用证据 |
| 2 | 两者只有一项成立 | 主张收缩到成立的部分；另一部分改为分析，如实报告负结果 |
| 3 | 两者都不成立 | 不投 WWW 2027 正会（改投短文或下一个会议） |

结果出来后，规则只能用于决定，不能修改。若需修改规则，必须在本文件中注明修改时间和原因，并在论文附录中披露。

## 修订记录

**2026-10-10（修复组 G5，对应审查 S-2、S-7、S-18）**。修订时仓库中没有任何 dev 结果（审查报告确认 dev 阶段从未运行），因此修订不是事后按结果调整。修订原因：
- 原规则没有 γ、β、τ_b、τ 的选择步骤，论文中的这四个值无来源（S-2）；新规则把它们纳入 dev 网格和选择，τ 离线计算。
- 原 EAB 有效性判据是对 6 个候选中最优者做普通 bootstrap 区间，零效应下的误判率约 15–18%（S-18）。现改为 max-statistic 校正，候选增至 30 个。
- 原 dev 选择在缺失或 NaN 时仍写出决定（S-7）；现在直接报错，决定文件记录所用文件的 sha256。
- 原规则 4 写"(τ, τ_b, β, γ, δ) 保持论文值"，已废止；论文值 (0.35, 0.40, 0.30, 1.2) 现在只是 dev 网格中的一格，也是并列时的优先值。
- OK-VQA 语料候选由 3 种减为 2 种：BM25 维基（`okvqa`）的候选依赖旧主机生成的输入，无法在本仓库复现（README 4.2）。同样在任何 dev 运行之前修订。

## v2 预注册

**本节在 2026-10-13 返工后改写，取代此前基于“逐题退化率二项检验”的版本。** 原因：那个定义把“得分低于最强单一动作的题”一律记为失败，而任何在两路之间真正切换的策略都必然在一部分题上比最强单一动作差（WebQA 上 direct 比 static 好的题就占 23%），于是只有几乎不切换的策略能通过，所有真正切换的策略都会被否决。

写定日期：2026-10-13（返工改写）。这些规则与实现它们的代码在同一批提交里写定，早于修复后管线的任何 GPU 运行；它们没有先于代码单独提交，所以严格说是“在看到新数据之前冻结”，不是独立的预注册。旧管线的离线探索结果（`runs/offline_v2`）在写定时已经可见，规则的形式参考了它。

### 修订：预注册第二指标 semantic_acc（2026-10-13，任何 v2 GPU 运行之前）

写定时没有任何 v2 运行的分数。规则：

1. **主指标不变**：OK-VQA 官方 VQA accuracy，WebQA 官方 acc_approx。v2-Rule / v2-Cal 的选择、开发集 +8% 关卡、+5% devcheck、强基线与相对提升都只读主指标。
2. **第二指标 `semantic_acc`**（`src/omniaware/eval/semantic.py`）：每题 0/100。前提句为"Question: 问题 / Answer: 预测答案"，假设句为"Question: 问题 / Answer: 参考答案"，固定版本的 NLI 检查器（引用评估用的 `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`，清单 revision，逐文件校验 sha256，float32，不截断，超出窗口即报错）给出的蕴含概率 ≥ 0.5 即记 100。参考答案：WebQA 为该题唯一的参考句；OK-VQA 为至少 3 位标注者给出的去重答案（VQA 满分门槛），没有答案达到 3 时取次数最多的答案。空答案记 0。预测文本与主指标所评分的文本相同（WebQA 为整句行）。
3. **报告规则**：所有方法、所有表格同时报告主指标与 `semantic_acc`，与结果的方向无关；不按结果在两者之间选择，任何结论以主指标为准。`semantic_acc` 在主表里使用主指标所做的同一组选择（`v2_main.json` 的 `secondary`），自己不参与任何选择。旧的宽松评分（"任一标注答案的子串命中"、`erc/scoring.py`）仍然作废。
4. **人工抽检**：S2 结束后，从开发动作表按"主指标对/错 × semantic_acc 对/错"四层，每层固定种子（20261013）抽至多 25 个"题 × 动作"，导出盲标表（只含问题、参考答案、预测答案；动作与分数在单独的 key 文件里，标注者不打开）。标注规则：预测答案与参考答案回答同一件事记 1，否则 0。`scripts/analysis/semantic_audit.py score` 报告两个指标与人工标注的分层一致率、按层大小加权的总体一致率、样本 Cohen's κ 与人工估计的准确率。抽检结果与两个指标一同报告（论文中的写法按论文修改计划另行审批）；它不改变主指标，也不决定是否报告第二指标。

### 修订：在答案 token 上测不确定性（2026-10-13，任何 v2 GPU 运行之前）

修订时仓库里没有任何 v2 的 GPU 运行，也没有修复后管线的任何数据；依据只来自旧管线的 WebQA 全量 4966 题（seed17 官方评分文件，`u_initial` 是直答 prompt 上第一个生成 token 的预测熵经 Norm 后的值）：

| 直答的第一个生成 token | 题数 | 直答正确的 AUROC [95% CI] | static 更好的 AUROC（两路得分不同的题）[95% CI] |
|---|---|---|---|
| 答案本身（Yes / No / 数字 / 实词） | 3345 | 0.623 [0.605, 0.642] | 0.618 [0.589, 0.647] |
| 句式词（The / A / There / Both …） | 1202 | 0.519 [0.486, 0.551] | 0.472 [0.433, 0.512] |

首 token 信号在第一个 token 承载答案时有效，落在句式词上时接近随机。原 WebQA 指令要求整句复述，会让多数非 YesNo 题以句式词开头。修订内容：

1. **WebQA 答案格式改为先答案后整句**（`prompts.ANSWER_FIRST`）：第一行只写短答案，第二行写复述关键对象的完整句。官方评分只读整句那一行（参考答案是句子）；短答案行用于信号与 token F1。整句行缺失时评分退回短答案行，记 `answer_first_ok=false`。OK-VQA 的指令（不超过 3 个词）不变。
2. **信号读在答案 token 上**。解码时记录每一步的归一化熵与 margin；答案 token 是第一个承载答案的生成 token（`signals.answer_slot`：先答案格式下是第一行里第一个非功能词；普通输出下还跳过复述问题的词）。每个可用动作记录 `answer_token`：位置、归一化熵 H_sem、margin、t_end 行上对本动作证据的 H_grd（直答为对所带上下文，如 OK-VQA 问题图；无上下文为 0）、β（运行的 `--beta`，0.30）与 `u_raw = H_grd + β·H_sem`，即论文 U_omni 的未归一化形式（Norm 单调，不改变阈值策略与同一阶段内的排序）。
3. **预注册的置信度由 `margin` 改为 `u_omni`**（`arbitration.PRIMARY_CONFIDENCE`，置信度为 −u_raw）。`margin` 与 `neg_entropy` 保留为消融，同样读答案 token。v2-Cal 的 conf 特征随之使用 u_omni；token F1 特征改为比较短答案。
4. **v2-Rule 增加 `route` 族**：直答不够确定时，按论文的模态路由 M_m·H_m（static_mixed 的 t_end 行，margin 0.08）选择 static_text 或 static_image，近似并列或没有注意力路由时选 static_mixed。只有当某道校准题的 static_mixed 证据含两种模态时才检验该族（与标签无关；OK-VQA 只有文本证据，不检验）。F 个族各用 δ/F，合起来水平仍为 δ。
5. **S1 增加格式关卡**：`scripts/analysis/answer_slot.py format`，先答案格式下任一可用动作的 `answer_first_ok` 低于 90% 即失败。S2 对开发表运行 `answer_slot.py position`，比较第一个生成 token 与答案 token 上的置信度；`--position-ablation` 另在 WebQA 开发集上用整句指令（`--answer-style sentence`）解码一次，作为测量位置的对照。

下文各节中与本修订冲突的旧条文以本修订为准。

### 修订：OK-VQA 指令改为单词或短语（2026-10-14，任何开发集与测试集结果之前）

本机 MPS 上第一次 okvqa_gs 校准运行（500 题，commit `bbead80`）中，直答的官方 VQA accuracy 只有 14.56，而 2026-10 旧管线在同 500 题上为 46.4。用 transformers 的原生 `generate`、同一张问题图，在这 500 题的前 40 题上做了 2 × 3 的对照（官方 VQA accuracy）：

| 指令 | transformers 5.5.3 | transformers 5.17.0 |
|---|---|---|
| 原 OK-VQA 指令 "Answer in no more than 3 words. Give the answer only." | 23.0 | 23.0 |
| 旧管线指令（system 提示 + `Question: …\nAnswer:`） | 55.5 | 55.5 |
| VQA 通用指令 "Answer the question using a single word or phrase." | 56.5 | — |

库版本对结果没有影响（S-17 关心的真实权重行为在两个版本下一致）。差距来自指令："不超过 3 个词"让模型给出两三个词的描述（"cormorant birds"、"Italian restaurant"），官方 VQA 评测按精确匹配记 0 分。

修订：OK-VQA 指令改为 VQA 的通用指令 "Answer the question using a single word or phrase."（`pipeline.SHORT_INSTRUCTION` 的同一句）。选它的理由是它是 VQA 评测的通用写法，不是因为它在 40 题上分数最高；与旧管线指令的差别（56.5 对 55.5）在 40 题上没有意义。修订时没有任何开发集或测试集的结果：开发集运行在解码第一道开发题之前就已停止。原指令下的校准运行作废，按新指令重跑。第 1 节修订中"OK-VQA 的指令（不超过 3 个词）不变"一句以本修订为准。

上文的 EAB / β / τ 规则不变。v2 是单独的决定，由 `scripts/analysis/v2_table.py select` 机械执行（`scripts/v2_runs.py --stage s2` 生成该命令），调用 `omniaware.arbitration` 的 `select_rule` 与 `select_cal`，结果写入 `runs/v2_decisions/<dataset>.json`。`scripts/analysis/dev_select.py` 的 `select_v2` 调用同样两个函数，是另一个入口，正式流程以 `v2_runs.py` 生成的命令为准。测试阶段（`v2_runs.py --stage s4`）只读取这个冻结文件，不搜索网格。

### 动作集合

五个动作，只从 `omniaware.actions` 导入（`V2_ACTIONS`）：

- `direct`：不检索。
- `static_text`：候选池中文本模态的 top-k。
- `static_image`：候选池中图像模态的 top-k。
- `static_mixed`：两种模态联合的 top-k，按校准相关度排序（即 v1 的 static 联合检索）。
- `rerank_static`：cross-encoder 对整个候选池重排后的 top-k。

证据动作集合 `EVIDENCE_ACTIONS` 是后四个。`rerank_static` 需要 cross-encoder 分数（`--v2-reranker`）；没有时运行报错，不用词面重叠代替。某个模态的候选池为空时，对应动作记为 `available=False`，不复用其他动作的结果，在任何策略里都不被选中。

### 信号

每个可用动作记录首 token 的 margin（top-1 与 top-2 logit 之差）、熵与归一化熵；答案 token 上的同样各量、H_grd 与 `u_raw = H_grd + β·H_sem`（`answer_token`，见上方修订）；贪心序列 log-prob、生成长度、与 direct 的首 token KL（direct 自身为空）、检索器 top-1 分数，以及 top-1 与 top-2 的分差；证据动作另记模态路由 M_m·H_m。

**决策只读取下面四类量**：
- 置信度：答案 token 上的 `u_omni`（−u_raw，直答的，以及每个证据动作的）；`route` 族另读 static_mixed 的模态路由；
- 动作短答案与 a* 短答案的 token F1（按归一化后的词多重集计算，空对空为 1，空对非空为 0）；
- 证据动作与 direct 的首 token KL；
- 证据动作所用候选的检索器 top-1 分数。

熵、归一化熵、序列 log-prob、生成长度、检索器的 top-1 与 top-2 之差、`gold_in_topk` 只存储，不进入任何决策。`gold_in_topk` 只用于离线分析。

预注册的置信度为答案 token 上的 `u_omni`（`confidence="u_omni"`，见上方修订）。`margin` 与 `neg_entropy` 只作消融。

### 1. v2-Rule（零训练）

策略由 `(evidence_choice, tau)` 决定：

- 若直答的置信度 ≥ tau，选 `direct`。
- 否则选证据动作：
  - `evidence_choice = "best"`：选 e\*，即证据动作中校准集平均得分最高者。若该题的 e\* 不可用，选该题可用证据动作中校准均分最高者；都不可用则选 `direct`。
  - `evidence_choice = "confident"`：选该题可用证据动作中 margin 最高者。

`tau = −∞` 为永远直答，`tau = +∞` 为永远用证据动作。

**阈值网格**：取校准集上直答 margin 的 0%、5%、…、100% 分位数（21 个），再加 −∞ 与 +∞，去掉相邻重复。网格只依赖校准集的信号分布，不依赖标签。

**固定序列的顺序（关键）**：必须从“等价于 a\* 的一端”开始，逐步远离。

- a\* 是 `direct`：tau 从 −∞ 递增（越来越多的题改用证据动作）。
- a\* 是证据动作：tau 从 +∞ 递减（越来越多的题改用直答）。此时 `evidence_choice="best"` 的起点策略恰好等于 a\*，因为证据动作为 a\* 时 e\* 就是 a\*。
- `evidence_choice="confident"` 族的起点不一定等于 a\*。它照样从同一端开始检验；起点不通过则整族被否决。

每个族各做一次固定序列检验。族为 `best`、`confident`，以及满足条件时的 `route`（见上方修订），F 个族各用 δ/F（Bonferroni），合起来的水平为 δ。

### 2. v2-Cal（轻量校准，预测效用差）

- 对每个非 a\* 的动作 b，单独拟合一个岭回归，预测 `score_b − score_{a*}`。每个动作一组权重，不共享。
- 特征（每个动作一组，`conf` 为 margin）：`[1, conf(direct), conf(a*), conf(b) − conf(a*), token_F1(answer_b, answer_a*), kl_to_direct(b), retriever.top1_score(b)]`，即截距加 6 个特征。截距不受惩罚；特征标准化后加岭惩罚（系数 1.0），返回的权重换回原始尺度。
- **缺失处理**：某特征在拟合集上全部缺失或方差为 0，丢弃该列，并在模型里记录被丢弃的列名。某特征只在部分行缺失，报错。**缺失不填 0。** 预测时若模型保留的某特征在该题缺失，报错。
- 不可用的动作行不参与拟合；预测时不可用的动作不参与比较。
- **校准集对半拆分**（固定种子 `seed = 0`，随机排列后取前一半）：前一半拟合回归，后一半做第 3 节的风险控制。两半不相交。
- 策略：该题上预测增益最大的可用动作 b̂；若其预测增益 > lam，选 b̂，否则选 a\*。
- `lam` 网格：拟合集上预测增益的 0%–100% 分位数（21 个），加 ±∞，从 +∞ 递减开始检验（`lam = +∞` 即永远 a\*）。
- 模型字典报告每个动作的参数个数和总参数个数。

v2-Cal 是单独的决定，单族，风险控制水平为 δ。

### 3. 风险控制（两个族共用的检验）

- 配对差：`d_i = score_policy(i) − score_{a*}(i)`，在风险控制用的校准题上计算。
- 容忍度 ε ≥ 0，与分数同量纲。0–100 量纲下预注册 **ε = 1.0**（分）。
- 原假设 H0：`E[d] < −ε`。检验统计量

  `t = (mean(d) + ε) / (sd(d, ddof=1) / sqrt(n))`，p 值为标准正态分布的上尾概率。

  代码对所有样本量 n 都用正态近似，不用 t 分布。对小 n 这偏乐观。协议中风险控制的题数为 500（v2-Rule）或 250（v2-Cal 的后半），在这个范围内可以接受。
- `sd = 0` 时：若 `mean(d) ≥ −ε`，p = 0；否则 p = 1。
- **固定序列**：按第 1 节或第 2 节的顺序逐个检验，`p ≤ level` 则接受并继续，否则停止，之后的候选全部不接受。`level` 对 v2-Rule 的每个族为 δ/2，对 v2-Cal 为 δ。
- 在被接受的候选里，选校准集平均效用 `mean(d)` 最大者。并列取先出现者（同族内更靠近起点；两族之间 `best` 先于 `confident`）。
- **回退**：若没有被接受的候选，或最大效用 ≤ 0，选择“永远 a\*”，记 `kind="single"`、`fallback=True`。效用恰好为 0 也回退。

**δ 与 ε 是两个独立参数**：δ 是失败概率（预注册 δ = 0.10），ε 是可容忍的期望分数损失（预注册 ε = 1.0）。改变其中一个不改变另一个的语义。原先 `alpha = delta` 的写法作废，本设计里不存在 α。

**保证的表述**：在题目 i.i.d.、t 检验渐近有效的假设下，以至少 1 − δ 的概率（对校准抽样而言），被选中的策略满足

  E[score_policy] ≥ E[score_{a\*}] − ε。

这是对题目分布取期望的陈述，不是逐题或逐切分的陈述。

**假设与边界**：

- 题目按 i.i.d. 处理。
- 检验是渐近有效的。正态近似在小样本时不保证。
- a\* 是在全部校准题上选出的，包括 v2-Cal 的风险控制半。它与用于检验的题有重叠，这带来轻微的乐观偏差。
- 本节的保证不声称该策略永不差于 a\*，也不声称退化率为 0。

### 4. 主决定与回退

主决定是 **v2-Rule**（冻结文件的 `primary = "rule"`）。v2-Cal 作为第二个臂报告，但其结果不改变主决定。

回退对两个族都一样：没有候选被接受，或最大效用 ≤ 0，则决定为“永远 a\*”。

### 5. 开发集 go/no-go 判据

在开发集上，记 O 为逐题选对的上限（每题五个动作中官方得分的最大者）的平均分，B 为最强单一动作（开发集上平均得分最高的动作）的平均分。相对差 `(O − B) / B`（B ≤ 0 时取绝对差 `O − B`，与 `scripts/analysis/v2_table.py gate` 的定义一致）。**相对差 ≥ +8%** 才继续把 v2 作为主方法做下去。

这是开发集上的上限判据，不是选择器已经实现的增益，也不在测试集上改规则来凑这个数字。本节只登记判据，不计算它。

# 实验协议（v2）

论文主表必须遵守本协议。协议对应方向文档第 5 节，并写明切分、主指标、强基线和风险控制的口径。**v2 实验还没有跑。** 下面出现的门槛（例如 +8%）是预注册判据，不是测量结果。

`runs/offline_v2`（命令 `python scripts/analysis/offline_v2.py --seed 20261013 --out runs/offline_v2`）是探索性结果，来自旧管线，含已知 bug，只用于检查代码和信号方向，**不是论文主表**。2026-10-13 返工后该脚本仍需重写，目录中的旧输出对应旧选择器，不得引用。

## 1. 数据与切分

主结果是 WebQA 与 OK-VQA。MMQA 与 VideoRAG 只作为泛化实验，题量按现有资产决定，不进入主表的分母定义。

| 数据集 | 校准 | 开发 | 测试 | 设置 |
|---|---|---|---|---|
| WebQA（`webqa_full`） | 至少 500 题，取自 **train** | 256 题，取自 **train** | 全部 **4,966** 道 val | **主设置：distractor**（每题自带候选池）。校准不从 validation 抽取 |
| WebQA 开放检索（`webqa_open_*`，当前接口里是 `webqa_open_img`） | 与主设置相同的规则：train 上的校准与开发，彼此不相交 | 同左 | 同一 val 划分下的对应子集 | **第二设置**。审稿人若问真实网页检索，用这一设置回答。当前命令接口只有 S2，没有单独的 S4 |
| OK-VQA（`okvqa_gs`，GS-112K 检索语料） | 500（保留区，`--calib-source heldout`） | 256 | **val 的剩余部分** | 三者两两不相交。val 为 5,046 题时，测试集为 4,290。测试集为空是错误。BM25 维基候选（`okvqa`）已从 v2 中移除，其输入无法在本仓库复现 |

规则：

- WebQA 的校准集和开发集都来自 train，并且与测试集不相交。删除“WebQA 校准来自 validation”的说法：以前在报告全部 4,966 道 val 的同时把校准说成来自 validation，二者不能同时成立。
- OK-VQA 的校准、开发、测试是 val 上两两不相交的三份切片。测试集是 val 去掉校准和开发之后的剩余部分，不是另一份独立打乱的 val。
- 三个切分两两不相交，由代码断言。测试集为空直接报错。
- 题号列表的 sha256 写入运行输出。测试阶段核对冻结决定文件的 sha256。
- 调参、提示词和阈值都不看测试集。测试集在决定冻结之后只跑一次。在测试集上看过结果再改设置，这一轮只能算开发，不能当作测试。

## 2. 主指标

只用官方评分。

- **OK-VQA**：只报告官方 VQA accuracy。不报告“10 个标注里任一子串命中”的宽松分，也不把宽松分和官方分并列。
- **WebQA**：主指标是 **acc_approx**。同时报告归一化 BART 和 **QA_approx**。图像子集和文本子集都要报告，子集由 `gold_modalities` 决定，缺少该字段时报错，不按 qcate 猜测。主结果注明是 distractor（`webqa_full`）还是开放检索（`webqa_open_*`）。不把宽松词面分和 acc_approx 并列。
- 主实验的提示词不要求输出引用行，也不把 `Sources: none` 当作答案。引用评估如果要做，单独运行，不混进主表。

同一运行里所有动作共用**同一个**提示词、解码预算和候选池。答案格式在开发集上选定后冻结：WebQA 输出一句与参考句式一致的完整句；OK-VQA 输出不超过 3 个词，并做 VQA 规范化。格式差异带来的分数变化不算作选择器的贡献。

## 3. 动作、基线与强基线

主方法的动作集合（EAB **不是**其中的臂）：

| 动作 | 含义 |
|---|---|
| `direct` | 不检索 |
| `static_text` | 只用文本模态候选的 top-k |
| `static_image` | 只用图像模态候选的 top-k |
| `static_mixed` | 两种模态联合的 top-k（按校准相关度排序），即以往的 static |
| `rerank_static` | cross-encoder 对整个候选池重排后的 top-k。需要 `--v2-reranker`；没有 cross-encoder 时报错，不用词面重叠替代 |

模态候选池为空时，对应动作标记为不可用（`available=false`），不复用其他动作的结果，也不参与任何选择。

对照（与上面的动作在可比的提示词和候选上比较；自适应方法不是“单一动作”）：

- 自适应检索：NTE-trigger、SeaKR、原来的 U 门控（Omni-Aware v1）、UniversalRAG 路由。
- **强基线**：校准集上得分最高的**单一动作**（上表五个动作之一）。身份在测试之前冻结。主表的**相对提升以强基线为分母**。同时报告相对 `static_mixed` 的提升。
- **上限（oracle）**：逐题在这五个动作里取得分最高者。

历史缓存基线作废。根因报告中的 +33%（WebQA）和 +15%（OK-VQA）比的就是这种基线，不能进入主表。+8.43% 和 +14.4% 同样作废。

`gold-in-top-k` 只用于离线分析，**不是**推理时特征，也不能进入 v2-Cal。

## 4. 统计

- 按题做配对 bootstrap，报告 **95% CI**。多重比较用 **Holm** 校正。
- 解码是贪心的。贪心解码下**不报 seed 标准差**。以前把贪心解码的 “3 seeds” 或 ±0.2 当作统计误差，这一做法作废；根因报告指出，±0.2 小于 5,046 题的抽样标准误（约 0.66 分）。seed 只有在它确实控制了某个随机过程时才报告，并写明它控制的是什么。
- 风险控制报告的是**平均效用**：配对差 `d = score_policy − score_{a*}` 的均值及其配对 bootstrap 95% CI。不报告“逐题退化比例”，那个定义在 2026-10-13 作废（见 `docs/DEV_DECISIONS.md` 的 v2 预注册）。
- 若门控或选择器的分数仍低于 `direct`，不得写成“优于无检索”。

相对提升的定义：\((\mathrm{方法}-\mathrm{强基线})/\mathrm{强基线}\)。相对 `static_mixed` 用同一个形式，并同时给出绝对差。这些数在跑完之前都是占位符，协议不填任何实测值。

## 5. 预注册与阶段命令

沿用 DEV_DECISIONS 机制。开发阶段的决定写入 `runs/v2_decisions/<dataset>.json` 并记录 sha256；测试阶段只读取该冻结文件，只跑一次。阶段命令由 `scripts/v2_runs.py` 生成（`--dry-run` 只打印命令）：

```bash
python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50
python scripts/v2_runs.py --stage s2 --dataset webqa_full
python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions runs/v2_decisions/webqa_full.json
python scripts/v2_runs.py --stage s2 --dataset okvqa_gs
python scripts/v2_runs.py --stage s4 --dataset okvqa_gs --decisions runs/v2_decisions/okvqa_gs.json
```

- **S2**（校准与开发）：分别解码校准集与开发集，对每个动作的答案做官方评分，`v2_table.py build` 生成逐题动作表；开发集上 `v2_table.py gate`，校准集上 `v2_table.py select` 写出冻结决定。
- **S4**（测试）：解码测试集，评分，`v2_table.py build`，再用 `v2_table.py apply` 把冻结决定应用到测试动作表，产出主表输入。测试阶段不搜索网格。

冻结文件至少记录：信号集合、v2-Rule 的阈值网格、δ、ε、种子、强基线（a\*）的身份、两个族的校准统计量与被接受的候选、v2-Cal 的模型（每个动作的特征、权重、被丢弃的列、参数个数）、lam 网格与选中的 lam、效用定义、答案格式提示词的 sha256、输入动作表的 sha256。网格和阈值只在校准集或开发集上决定。**不在测试集上选网格。** 以往 Table 20 的选定点落在测试集网格的最大值上，这一做法作废。

开发集关卡（判据，不是结果）：

- S2 之后，`v2_table.py gate` 检查：修管线之后，开发集上逐题选对的上限相对最强单一动作（开发集上平均分最高者）是否仍不低于 **+8%**（相对）。输出 PASS 或 FAIL。
- 进入测试之前，选择器在开发集上相对强基线的提升不低于 **+5%**（相对），且配对 bootstrap 的 CI 下界大于 0。不满足就不要把该选择器当作已经通过的测试结果；测试仍然只跑冻结后的那一次，结果好坏都照报。这一比较由 `v2_table.py devcheck` 计算（`scripts/v2_runs.py --stage s2` 的最后一步）：把冻结决定应用到开发动作表，结果写入 `runs/v2_decisions/<dataset>.dev_check.json`，并记录决定文件的 sha256。S4 读取该文件；未通过时拒绝运行，除非显式传入 `--allow-failed-dev-check`，此时冻结测试照常只跑一次，结果照报，选择器不算通过。

`docs/PROTOCOL.md` 原先的“退化比例不超过 δ”一条随风险控制的改写一并撤销。

## 6. 风险控制的声明

选择器的风险控制是对**平均效用**的固定序列单侧配对检验，完整定义见 `docs/DEV_DECISIONS.md` 的 v2 预注册。效用为 `score_policy − score_{a*}` 的校准均值。

保证的形式：在题目 i.i.d. 与检验渐近有效的假设下，以至少 1 − δ 的概率，被选中的策略满足

  E[score_policy] ≥ E[score_{a*}] − ε，

其中 δ = 0.10（失败概率），ε = 1.0（0–100 量纲下的容忍度，单位为分）。δ 与 ε 是两个独立参数，不写成 α = δ。

最强单一动作 a\* 在全部校准题上选出，因此略偏乐观。

v2-Rule 的阈值与 v2-Cal 的 lam 都按同一效用选择，不按准确率最大值选。v2-Cal 是每个证据动作一个岭回归，预测相对 a\* 的效用差；论文里写成校准，不写成训练出来的 router。v1 里 `original_retrieve or p_image<0.5` 用过训练过的问题分类器，和“不训练 router”冲突；v2 不使用该分类器。

## 7. 信号（推理时）

每个可用动作记录：首 token 的 top-1 与 top-2 logit margin；首 token 熵及熵除以 \(\log|V|\)；贪心序列的 log-prob 与生成长度；与 `direct` 的首 token KL（direct 自身为空）；检索器 top-1 分数，以及 top-1 与 top-2 的分差。`gold-in-top-k` 只用于离线分析。

决策只读取下面四类量：

- margin（直答与证据动作的置信度）；
- 动作答案与 a\* 答案的 token F1（归一化后按词计）；
- 证据动作与 `direct` 的首 token KL；
- 证据动作所用候选的检索器 top-1 分数。

熵、归一化熵、log-prob、生成长度与检索器分差只存储，不进入决策。`gold-in-top-k` 不是推理时特征。

v2-Cal 每个证据动作的特征为：截距、conf(direct)、conf(a\*)、conf(b) − conf(a\*)、token F1(answer_b, answer_{a\*})、KL(b 与 direct)、检索器 top-1(b)。某特征在拟合集上全部缺失或为常数则丢弃并记录，部分缺失则报错，不填 0。

## 8. 作废的旧声明

与根因报告第 3.3 节和方向文档第 7.4 节一致，论文里必须删除或改写：

1. “相对 static 提升 X%”作为主结论。主结论改成相对强基线，并另报相对 `static_mixed`。
2. “EAB 提升证据利用 / 位置鲁棒性”。在同协议正结果出现之前，EAB 只作为分析里的负结果：注意力质量被移动，答案不变。
3. WebQA 改用官方 acc_approx，并注明 distractor 或开放检索。不与宽松尺度并列。
4. EAB 的行集合以公式为准：\(a\ge t_{\mathrm{end}}\)（实现名 `ans1`）。把 `ans2` 标成“论文式”的数字要改标。旧门控实际读的是 \(U_{\mathrm{post}}\)，不是正文写的检索时的 \(U\)。
5. WebQA 校准来自 train，并写明至少 500 题、开发 256 题。删除“来自 validation”。
6. 贪心解码不报 seed 标准差，改报配对 bootstrap CI。
7. 选参只在开发集或校准集上进行。测试集网格不能用来挑选报告点。
8. 选择器若仍低于 `direct`，不能声称优于无检索。
9. v2-Rule 的“逐题退化率 ≤ δ”的 Learn-then-Test 声明（2026-10-13 作废）。它会否决所有真正切换的策略。改为对平均效用的单侧配对检验。
10. `α = δ = 0.10` 的写法（2026-10-13 作废）。δ 与 ε 是两个独立参数。
11. v2-Cal 的共享逻辑回归（7 个参数）与“按预测的正确概率选动作”（2026-10-13 作废）。改为每个证据动作一个岭回归，预测相对 a\* 的效用差。
12. 没有 cross-encoder 时用词面重叠代替的 `rerank_static`（2026-10-13 作废）。现在没有 cross-encoder 时报错。

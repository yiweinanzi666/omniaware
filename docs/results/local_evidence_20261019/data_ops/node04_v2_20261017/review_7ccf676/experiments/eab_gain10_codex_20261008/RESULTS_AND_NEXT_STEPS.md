> 2026-10-09更新：用户验收目标已改为相对至少5%，并采用Claude Code实现、Codex审核。当前七轮352条评分及梯度探针见[最新进度](/Users/futaoran/Desktop/WWW2027/experiments/eab_gain10_codex_20261008/GOAL_PROGRESS_20261009.md)；协作与训练启动失败见[实时审核交接](/Users/futaoran/Desktop/WWW2027/experiments/eab_gain10_codex_20261008/reviews/CLAUDE_HANDOFF_20261009.md)。以下内容保留为前三轮的历史快照。

# EAB 相对提升超过10%的实测结果

**当前状态：已完成三轮机制开发，超过10%的目标未达到。** 这一判断针对同输入 static 的 EAB 单项官方公开 WebQA QA_approx，不使用整体流水线收益代替组件收益。新候选尚未进入正式512题评测或跨样本确认，V3.19的历史数据未被替换。

本次固定32题 static 为27.062139689，按相对提升超过10%的要求，候选必须高于29.768353658。最佳实测候选 mass_initial 为27.121398958，相对提升0.218975%；差值95%区间包含0。

| 设置 | QA_approx（0–100） | 相对 static | QA差值95%区间／分 | QA增／降／平 |
|---|---:|---:|---|---|
| 同输入 static | 27.062140 | — | — | — |
| 当前 EAB：第二 token 起 | 27.062140 | +0.000000% | [+0.000000, +0.000000] | 0/0/32 |
| 当前 EAB：首 token 起 | 26.708720 | -1.305957% | [-1.060260, +0.000000] | 0/1/31 |
| 初始 U＋普通正偏置 | 26.432577 | -2.326361% | [-1.563502, +0.000000] | 0/3/29 |
| 后验 U＋块注意力质量校正 | 27.054801 | -0.027119% | [-0.022017, +0.000000] | 0/1/31 |
| 质量校正扩到问题区 | 23.937448 | -11.546357% | [-9.469601, +0.116473] | 2/3/27 |
| 初始 U＋块注意力质量校正 | 27.121399 | +0.218975% | [-0.014678, +0.191278] | 2/1/29 |

三个分阶段的完整逐题结果为 [首轮五臂](/Users/futaoran/Desktop/WWW2027/experiments/eab_gain10_codex_20261008/development32/OFFICIAL_RESULT.json)、[扩展问题区](/Users/futaoran/Desktop/WWW2027/experiments/eab_gain10_codex_20261008/question_rows32/OFFICIAL_RESULT.json)、[初始 U 质量校正](/Users/futaoran/Desktop/WWW2027/experiments/eab_gain10_codex_20261008/initial_mass32/OFFICIAL_RESULT.json)。同一题、同一配置的重复评分行已核为完全相同，总计32×7＝224条唯一配置评分；三轮继续使用同一批已消费的训练开发题，不能把三轮当成96个独立样本。

## 结果解释

当前实现的首答案 token 干预、无偏上下文重算、因果掩码与门控路径经源码与实际执行身份核验可用，未发现能把这批负结果解释为“EAB 根本没有启用”的错误。六项新分支性质测试通过；这只验证实现性质，不代替QA实测。

质量校正把每层每头当前的证据块条件注意力分配推向检索相关度分布，并保留局部总证据质量。第一轮后验 U 校正没有QA增益；扩展到问题区使已正确的 Eastern Indo-Aryan 改为 Bengali，该题2/2支持事实都已展示，单题贡献−3.125分，几乎解释全部净下降。

初始 U 质量校正的5条答案文本变化中，两条QA上升均来自原事实答案的字面表达变化：Chelsea Market仍回答有时钟；天使头饰仍回答花环，仅加入entitled并改变作品名引述格式。另两题原acc为0，修改后仍为0；Roman题仅首字母变小写，官方BART归一化没有转小写，导致小幅损失。当前没有新修正事实错误的证据。原始答案和评分完整保留。

只读支持标签诊断发现32题均至少展示1条标注支持，21题展示全部支持，11题缺少部分支持。Brenda题缺少Steve Sanders事实，IDA题缺少three FFRDCs事实。固定证据内的偏置不能加入这些漏掉的文档；这部分需在证据选择上解决，并单独计为检索/路由改进。反过来，支持齐全的Eastern Indo-Aryan仍被扩区偏置改错，说明支持完整性并非当前EAB唯一问题。

给已经生成的7种回答逐题使用标准答案选择最优，得分也仅27.130084158、相对提升0.251068%。这是不可部署的有限输出诊断，说明只挑这批既有回答达不到10%；它不限制未来新机制的可能收益。

既有CUDA512题×5臂的有限输出诊断也已独立复算：static 23.384771282，逐题用标准答案挑最优仅24.566377972，相对提升5.052890%；超过10%需高于25.723248410。既有MPS512同样仅+3.317826%。这是既有输出集合的约束，不能解释成未来所有EAB机制的上界。来源：[CUDA512官方逐题评分](/Users/futaoran/Desktop/WWW2027/experiments/v318_missing_experiments_5h_root_20261008/cuda_controls512/parallel/scoring/OFFICIAL_SCORES.jsonl)、[双backend对照](/Users/futaoran/Desktop/WWW2027/experiments/v318_missing_experiments_5h_root_20261008/mps_controls512/backend_comparison/REPORT.md)。

## 证据和验收边界

只读重建train64的320个所选块，与冻结相关度逐块核为一致：64题中43题最高r块是支持、18题最高r负例超过全部已选支持，另3题没有标注支持；当前32题分别为20题和12题，零题无支持。r是相关度的ECDF分位数，没有用支持标签拟合，不是支持概率；让所有层/头更强地追随该r分布未必改善作答。来源：[训练事实](/Users/futaoran/Desktop/WWW2027/experiments/emnlp_repro_codex_review/webqa_goal_official_20261008/full_routed4966/TRAIN_EXAMPLES.json)、[相关度实现](/Users/futaoran/Desktop/WWW2027/experiments/emnlp_repro_codex_review/pro6000_fresh_cuda_review_20261008/source/erc/relevance.py:145)。

当前没有足够依据继续追加同类QA臂或扩大这些候选。最小剩余工作是训练内机制观测：在固定输入的首答案行保存真实逐层/逐头证据总质量、块条件质量、裁剪比例和干预后的质量，用支持标签只做离线诊断，检验偏置是否把质量移向支持。三轮已有日志没有这些观测，无法离线补出。若观测确认有稳定、可改善的层/头，再研究稀疏层/头控制器，并冻结新的跨样本确认；该控制器尚未实现或训练，也没有>10%的证据。

32题按固定盐与qid哈希选取，模型推理没有使用参考答案或支持标签；同一Qwen3-VL-8B、MPS BF16、同一token/图像、最终五块证据及顺序、greedy seed17和384输出token上限。static逐题共享，追加轮次在预先指定第一题重新解码核对完整token与文本，并核对全32题无偏U和输入身份。gate-off复用同一static而非伪造新推理。

评分沿用冻结官方acc_approx与CPU FP32/eager ParaBank BARTScore，以逐题乘积再求均值。未改参考、答案、大小写、题目分母或指标。区间为10,000次整题配对bootstrap，开发后选择与多轮查看使这些区间不能充当独立泛化证据。独立复核详见 [INDEPENDENT_REVIEW.md](/Users/futaoran/Desktop/WWW2027/experiments/eab_gain10_codex_20261008/INDEPENDENT_REVIEW.md)。

正式验收仍未完成：应先形成有开发证据的新机制并冻结参数，再在EAB调参未接触的新题上进行同协议跨样本确认，之后评测固定512与公开全val。router已使用训练模态标签，新train题只能称EAB跨样本确认，不能称整个流水线完全未见训练数据的留出。10%点估计、区间是否跨0和是否达到区间下界也超过10%须分开报告。

## 实际成本和当前文件

三轮成功生成共1699次QA forward，成功生成账合计531.614797秒；4次QA模型加载合计61.598163秒。另保留1次资源争用中断，2次forward、失败账211.788548秒。相同失败在finally另有收尾计时，没有重复相加。三个CPU评分阶段共19次BART forward，未运行QA模型。上述计时边界各自不同，不把它们相加称为端到端耗时或性能提升。

汇总机器可读值与15份结果/计划/预测文件SHA256在 [RESULTS.json](/Users/futaoran/Desktop/WWW2027/experiments/eab_gain10_codex_20261008/RESULTS.json)；失败、加载、token与调用账在各轮目录。只创建隔离候选和报告，未改原实验、论文、历史结果、现有服务器任务或评分器。原V3.19审计见 [V3.19_数据来源与矛盾核查.md](/Users/futaoran/Desktop/WWW2027/experiments/eab_gain10_codex_20261008/V3.19_数据来源与矛盾核查.md)。

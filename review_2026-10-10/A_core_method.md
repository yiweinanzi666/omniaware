# A 核心方法实现 vs 论文公式 审查报告

审查对象：`C:\Users\22688\Desktop\www_code\omniaware`（只读，未修改仓库文件）。
论文：`paper/main.tex`（方法 L172–248）、`paper/sec_experiments.tex`、`paper/sec_appendix.tex`。
环境：Python 3.11.7、torch 2.7.1+cpu、transformers 5.5.3，全部在小型随机初始化模型和合成数据上完成，未下载任何真实权重或数据集。

---

## 0. 摘要

- **未发现致命问题。** EAB 注意力钩子与一个独立实现的无缓存参考逐 token、逐 logit 一致（文本、中间层、带图 prompt、多步解码），零偏置前缀与无缓存前向一致，图像 token 展开与 span 定位正确（真实 Qwen2VL 系图像处理器，148 = 148）。H_grd、H_pred、Norm、M_m·H_m、门控、冷启动探测的公式与论文一致。
- **严重（2 条）：** (A-01) 复制对照臂的拷贝数 `ceil(exp(γUr))` 不等价于 EAB 偏置，且没有论文所说的"同输入预算"截断；(A-02) 论文关于 β、τ_b、γ 在 dev 上选择的叙述与仓库不一致，β×τ_b 敏感性表在测试集上运行，并被用作"β=0.30 最优"的依据。
- **中等（7 条）：** 多通道百分位平均后 r_j 被压缩（A-03）、ARD 捕获排除末层偏置（A-04）、偏置对零偏置键的绝对重分配（A-05，推测幅度）、token 级熵混合模态结构（A-06，推测）、β 不控制两项权重（A-07，推测）、Static 基线在冷启动数据集上带路由（A-08）、单模态上下文路由未实现论文的"跳过"规则且会崩溃（A-09）。
- **轻微（8 条）：** A-10 至 A-17，主要是数值边界、静默失效风险、默认参数与文档不一致、提示模板未核实、死代码、视频帧输入方式、依赖版本。

---

## 1. 已验证为正确的部分（含证据）

| 检查 | 方法 | 结果 |
|---|---|---|
| 现有核心测试 | `pytest tests/test_qwen_eab.py tests/test_internvl_eab.py tests/test_controller.py tests/test_routing_rules.py tests/test_interventions.py -p no:cacheprovider` | 24 个用例全部通过 |
| 钩子 vs 独立参考（文本） | `scratchpad/review/verify_eab_ref.py`：参考实现是无缓存全序列前向，显式 causal 掩码和 bias 行掩码，逐步贪心解码，不复用 `qwen_eab._eab_sdpa` | static、ans1、ans2、中间层限定（layers=mid）四个臂：生成 token 完全一致，首 token logits 最大差 1.3e-7 |
| 钩子 vs 参考（带图） | 同上，prompt 含一张图（手工构造的 image 块，grid [1,4,4]，两个 span） | static、ans1、ans2、q 四个臂：token 一致，首 token logits 差 1.5e-7。ans1 确实改变了首 token 分布（与 static 不同） |
| 偏置行范围 | 代码检查 + 参考比对 | ans1 对应 `q_start = enc.length-1`（t_end），ans2 对应 `enc.length`，q 对应 `enc.split`；仅 `self_attn` 的文本层被标记（`tag_text_layers`，qwen_eab.py:130-136），视觉塔未受影响 |
| 零偏置 prefill | `test_shared_prefix_equals_full_forward` 通过；参考比对中 static 臂与无缓存前向一致 | 零偏置前缀 = 无偏置前向 |
| 解码-only 性质 | `test_decode_only_eab_cannot_change_first_token_but_eab_p_can` 通过 | ans2 的首 token 不受偏置影响，与论文"decode-only 不改第一个答案 token"一致 |
| 图像 token 展开（真实处理器） | `scratchpad/review/verify_encode_and_calib.py` (a)：`Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)`，两张真实 PNG，其中一张走 `max_pixels` 分支 | image_pad 插入 148 个，等于视觉输出 `prod(grid)//4` 之和 148；`__query__` 和 `c1` 的 span 覆盖各自的 pad；前向与贪心解码运行成功 |
| KV crop 与 rope_deltas | 参考比对包括多步解码与带图 prompt（rope_deltas = -2） | 与无缓存参考一致 |
| 公式 | 对照 `paper/main.tex` L172–248 与 sec_appendix：Eq. attn（qwen_eab.py:365–371 与 routing.py:32–34）、H_grd 用 log N 归一化且 N≤1 时为 0、H_pred 用 log 底数无关比值、Norm（pipeline.py:82–89）、M_m·H_m（routing.py:15–43，含 `len≤1 → H=0`）、门控（run.py:560）、冷启动 U 与 EAB 使用同一 U（run.py:559、594） | 与论文一致（除下列问题） |
| 数据划分 | `data.split_examples`（data.py:357–372）calib/dev/test 互斥；run.py:254 断言 calib 与 eval 不相交 | 无泄漏，已核实 |

---

## 2. 问题清单（每条一行）

| 编号 | 严重程度 | 位置 | 问题 |
|---|---|---|---|
| A-01 | 严重 | order.py:30；run.py:617–629；sec_appendix.tex:5、42–60 | 复制对照臂 `ceil(exp(γUr))` 偏置量化失真（γUr∈(0,ln2] 全部映射为 2 份），且无截断，不满足论文"同输入预算" |
| A-02 | 严重 | sec_appendix.tex:9、:66；sec_experiments.tex:16；docs/DEV_DECISIONS.md:23；scripts/paper_runs.py:28、147–150、250 | β、τ_b、γ 的"dev 选择"叙述与仓库不符；β×τ_b 敏感性在测试集上跑，且用作 β=0.30 最优的依据；τ=0.35、δ=0.08 无来源记录 |
| A-03 | 中等 | relevance.py:151–153、159–165 | 多通道 ECDF 取均值后不再是百分位（std 0.20 vs 0.29，p95 0.83 vs 0.95），跨模态排序与 EAB 偏置强度受影响 |
| A-04 | 中等 | qwen_eab.py:54–58、60–80、291–293；run.py:609–611 | 注意力捕获在加偏置之前进行，ARD 对 EAB 臂不含末层偏置；捕获行的选择论文未定义 |
| A-05 | 中等（推测） | qwen_eab.py:60–80；main.tex:221–248 | 命题只约束 chunk 之间的比值，对汇聚点、查询图像等零偏置键的绝对再分配可以很大（示例：汇聚点质量 0.60→0.35） |
| A-06 | 中等（推测） | qwen_eab.py:365–371；routing.py:15–43 | H_grd、H_m 是 token 级熵，混合了 patch 密度和模态 token 数；图像 chunk 的 256 个 patch 天然扩散 |
| A-07 | 中等（推测） | pipeline.py:102–108；run.py:443–448；main.tex:201 | β 不控制两项的相对权重，实际权重由各项在校准集上的 IQR 决定 |
| A-08 | 中等 | scripts/paper_runs.py:59–63、112–118；run.py:410–415、565–566；docs/PAPER_TABLE_MAP.md:12 | 冷启动数据集的 Static 臂已经带 M_m·H_m 路由，而 OK-VQA 没有，基线定义不统一 |
| A-09 | 中等 | routing.py:46–50、15–43；run.py:410–411；main.tex:217 | 单模态上下文（OK-VQA 查询图）路由未跳过，传入 --modality-routing 或 --first-round-routing 时 select_chunks 抛 ValueError |
| A-10 | 轻微–中等 | pipeline.py:82–83、106–108 | IQR≈0 时 Norm 退化为阶跃函数（U∈{0,1}）；zsum 与 Norm 的退化处理不一致 |
| A-11 | 轻微–中等 | qwen_eab.py:84、156、246–247 | EAB 依赖全局注册 "sdpa"，无运行时校验；回退 eager 会静默等同 static；_prefix 会清空 STATE |
| A-12 | 轻微 | run.py:115、157、410–415、553–560；README.md:222 | 冷启动 flag 的 CLI 默认值关闭，文档称为默认设置；单独开 --first-round-routing 时路由与门控证据不一致 |
| A-13 | 轻微（推测） | pipeline.py:41、67–71；qwen_eab.py:216–218 | 提示为手工拼接，无 system 段，每段单独 tokenize；未与官方 chat template 核对 |
| A-14 | 轻微（推测） | qwen_eab.py:362、173 | H_pred 归一化用 logits.numel()，可能是 padded 词表而非 \|V\| |
| A-15 | 轻微 | relevance.py:115、146–147；pipeline.py:39、52、58、128 | 死代码（startswith("f")、image_pixels 从未传入）；ecdf 对缺失通道静默返回 0.5；raw_uncertainty 静默丢弃不在 spans 的 cid |
| A-16 | 轻微（推测） | pipeline.py:55–58；qwen_eab.py:199–211 | 视频的 8 帧按独立图像（t=1）输入 Qwen3-VL，非原生视频路径，时间位置信息与 token 数均不同 |
| A-17 | 轻微 | pyproject.toml（transformers>=4.57）；qwen_eab.py:156 | 只在 transformers 5.5.3 上验证，依赖声明与实际不符 |

---

## 3. 详细说明

### A-01 复制对照臂的拷贝数与 EAB 偏置不等价，且没有预算控制（严重）

**位置**：`src/omniaware/order.py:28–30`；`src/omniaware/run.py:617–629`；`paper/sec_appendix.tex:5`（"chunk replication duplicates those chunks before truncation under the same input budget"）、`sec_appendix.tex:42–60`（局部等价命题）。

**证据**：
- `order.py:30`：`return [math.ceil(math.exp(gamma * uncertainty * value)) if gate else 1 for value in relevances]`。
- `verify_misc.py` 的输出（§2）：

```
U=0.41 r=0.05  EAB logit bias=0.025  copies=2  log(copies)=0.693
U=0.41 r=1.00  EAB logit bias=0.492  copies=2  log(copies)=0.693
U=0.60 r=0.60  EAB logit bias=0.432  copies=2  log(copies)=0.693
U=0.60 r=1.00  EAB logit bias=0.720  copies=3  log(copies)=1.099
U=1.00 r=1.00  EAB logit bias=1.200  copies=4  log(copies)=1.386
```

- 对任意 γUr∈(0, ln2]，拷贝数都是 2，等效对数偏置固定为 0.693。例如 U=0.41、r=0.05 时，EAB 偏置为 0.025，而复制臂相当于 0.693，约 28 倍。
- 命题（sec_appendix.tex:42–57）要求 n_j = e^{b_j}，一般不是整数，因此取整无法复现同样的重加权。
- `run.py:624`：`enc_r = qwen.encode(*evidence_segments(e, copies, rel))`，拷贝直接拼入，没有截断、没有 token 上限。全仓库 `grep -i truncat src/omniaware/*.py` 只命中 KV crop 的 docstring 与各嵌入器的 tokenizer 参数，没有预算逻辑。run.py:628 只记录 `replication_prompt_tokens`，没有据此约束。

**影响**：uomni_ablation 的 Replication 行、causal 表的 Replication 行，以及论文中"EAB 对复制"的比较（sec_appendix.tex:60 的数字位置），都是在一个既不同偏置、也不同预算的对照上做的。如果复制臂表现差，可能只是因为它过度偏置且输入更长；如果表现好，"EAB 不如复制"的结论也不成立。

**修复建议**：
1. 若保留命题，用 n_j = round(exp(γUr)) 并报告对数偏置残差，或直接按 n_j=ceil 的实际偏置做对照，并在论文中如实描述。
2. 在复制臂中强制与 EAB 相同的 token 预算（按 r 从低到高截断尾部，或限定总 token），并把 prompt 长度写入记录。
3. 在论文中删除"同输入预算"的说法，直到代码实现它。

**是否推测**：否。代码与数字可以直接核对。对论文具体数字的影响需要真实运行。

---

### A-02 超参数"dev 选择"的叙述与仓库不一致；敏感性表在测试集上运行（严重）

**位置**：
- `paper/sec_appendix.tex:9`："The strength γ is chosen from {0.6,…,2.0} on development queries"。
- `paper/sec_appendix.tex:66`："The values of β and τ_b were selected on development queries; we report their sensitivity on the evaluation sets." 同段："β=0.30 scores highest at every τ_b on OK-VQA"。
- `paper/sec_experiments.tex:16`：共享超参数 (τ, τ_b, β, γ, δ) = (0.35, 0.40, 0.30, 1.20, 0.08)。
- `paper/sec_experiments.tex:18`："no number in this paper comes from development questions"。
- `docs/DEV_DECISIONS.md:23`（规则 4）："其余超参数不再调整。(τ, τ_b, β, γ, δ) = (0.35, 0.40, 0.30, 1.2, 0.08) … 保持论文值"。
- `scripts/paper_runs.py:28`：`EAB_FORMS` 全部为 `eab:g1.2:…`，dev 阶段的 γ 固定为 1.2。
- `scripts/paper_runs.py:84–90`：dev 阶段只扫 EAB 形式、不确定性设置、OK-VQA 语料，没有 β、τ、τ_b、γ、δ 的扫描。
- `scripts/paper_runs.py:147–150`：β×τ_b 网格位于 ablation 阶段。
- `scripts/paper_runs.py:250`：`--split` 默认 `"test"`，ablation 阶段沿用它。

**证据**：
- dev 阶段的所有运行没有 `--beta`、`--tau`、`--tau-bias` 参数（paper_runs.py:84–90）。
- 仓库中没有 `runs/dev_decisions.json`（`runs/` 不在版本库中，且没有其他文件记录 τ=0.35、δ=0.08 的来源）。`docs/LEGACY_MAP.md` 提到旧的 `derive_tau.py` 已删除，但没有记录其结论或输入。
- `docs/PROTOCOL.md:18` 写的是"调参（τ、τ_bias、β、γ…）只看 dev"，与 DEV_DECISIONS 规则 4 的"不再调整"相矛盾。

**影响**：论文的 pre-registration 叙述不成立。β=0.30、τ_b=0.40 的"最优"是在测试集上观察到的，并以"选择"的形式写入附录。γ 的"dev 选择"在代码中没有对应的 dev 运行。τ=0.35、δ=0.08 没有任何来源记录，可能来自旧仓库中用标签选阈值的流程，而那正是任务说明中列出的旧问题之一。这会直接削弱论文的方法学声明，即使数字本身没有问题。

**修复建议**：
1. 改写论文：说明 γ、τ、δ、β、τ_b 的来源（沿用旧值须写明来源，或说明是人工设定）。删除"在 dev 上选择"的说法，除非有对应的 dev 记录。
2. 若保留 β×τ_b 表，明确标注为"测试集上的事后敏感性分析"，不用于支持"β=0.30 最优"的结论；或者把同一网格在 dev 上重跑并报告。
3. 把 dev 运行与 `dev_decisions.json` 纳入可复现的记录，或在 README 中说明它们不在仓库中。

**是否推测**：否。文本与脚本可直接核对。

---

### A-03 多通道百分位平均后不再是百分位，r_j 跨模态不可比（中等）

**位置**：`src/omniaware/relevance.py:151–153`（视频：`video_text`、`video_clip` 取均值）；`:159–162`（图像 mix：`image_clip`、`image_cap` 取均值）；`:163–165`（只有 clip 时回退）。论文 `paper/main.tex:217` 说 r_j "is comparable across modalities"。

**证据**：`verify_encode_and_calib.py` (c)，独立正态分布、校准集 5000 条：

```
text (single ECDF)                 mean=0.496 std=0.285 p05=0.051 p95=0.949
image clip-only (single ECDF)      mean=0.506 std=0.288 p05=0.049 p95=0.951
image mix (mean of two ECDFs)      mean=0.496 std=0.203 p05=0.160 p95=0.834
```

相关性（c2）：ρ=0.5 时 std=0.251、p95=0.912；ρ=0.9 时 std=0.277，压缩随相关性增加而减弱，但没有消失。

另一个模拟（scratchpad review 中的内联脚本）：文本与图像候选来自相同的原始分数分布，top-5 中图像的占比为 **0.456**（中性值为 0.5）。

另外，同一模态内部也不一致：`relevance.py:163–164` 对只有 clip 的图像用单个 ECDF，对有 caption 的图像用两个 ECDF 的均值，两者的分布不同。

**影响**：
1. 图像和视频候选的 r_j 被压缩到中间区间，跨模态 top-k 排序偏向文本（模拟中为 4.4 个百分点，真实数据的幅度未知）。
2. EAB 对图像/视频的偏置动态范围更小，因为 r_j 的 p95 只有约 0.83。
3. `routing.retriever_route`（routing.py:59–71）和 `--route-rule retriever` 的消融结论受影响。
4. 论文"r_j comparable across modalities"（main.tex:217）的前提不成立。

**修复建议**：
1. 在原始分数上融合（先按通道在校准集上标准化、再相加），然后对融合分数整体做一次 ECDF，使 r_j 近似均匀。
2. 对所有图像/视频统一规则，不再区分有无 caption。
3. 在校准集上报告每个模态 r_j 的分布（均值、std、p05、p95），作为论文的附录表。

**是否推测**：压缩本身已由模拟证实；对真实数据的排序和 EAB 效应的幅度是推测，需要真实通道分数。

---

### A-04 注意力捕获排除被捕获层自身的偏置；ARD 的行定义未说明（中等）

**位置**：`src/omniaware/qwen_eab.py:54–58`（捕获，在加偏置之前计算未偏置 logits）；`:60–80`（偏置在同一层之后加入）；`:291–293`（_decode 在第一个生成 token 的那一步捕获）；`src/omniaware/run.py:609–611`（`rec["ard"][name]` 使用 `r.attention`）。

**证据**：`verify_encode_and_calib.py` (b)，偏置只作用于最后一层（`layers={q2.layers[-1]}`）：

```
captured-layer-only bias: max |captured_biased - captured_clean| = 0.0
```

而同一设置下 logits 确实发生了变化（中间层与全层偏置的测试中，首 token logits 差 >1e-4）。捕获值不含末层自身的偏置。默认 `layers=all` 时，捕获值只包含上游层偏置引起的隐状态变化，末层偏置的效应完全缺席。

另外，`_decode` 捕获的是第一个生成 token 的查询行（位置 `enc.length`，test_qwen_eab.py:95 断言 key 长度为 `enc.length+1`），不是论文 U 所用的 t_end 行。论文附录 ARD 小节（sec_appendix.tex:17–21）没有规定使用哪一行。

**影响**：EAB 臂的 ARD 数值系统性地偏离"实际生效的注意力"。论文若据此解释 ARD 的变化，解释会有偏。行的选择也未在论文中写明。

**修复建议**：
1. 在 `_eab_sdpa` 中，于加偏置后的 softmax 上计算并捕获注意力，而不是加偏置之前。
2. 在论文中明确 ARD 使用的行（建议与 U 相同的 t_end 行，或分别报告两行）。
3. 增加回归测试：末层偏置开启时，捕获的注意力应当改变。

**是否推测**：否。最小化测试直接给出 0.0 的差值。

---

### A-05 偏置对零偏置键的绝对再分配（中等，推测幅度）

**位置**：论文 `paper/main.tex:221–248`（EAB 小节与"bounded relevance tilt"命题）；`src/omniaware/pipeline.py:135–172`（`Variant.spec`，L153 起），`src/omniaware/qwen_eab.py:60–80`（对所有头、所有文本层、所有 q≥t_end 的行加同一偏置）。

**分析**：命题的推导本身正确（我没有发现错误）。它只约束 chunk 之间的比值 e^{λ(r_j−r_l)}。但 chunk 与零偏置组（指令、系统 token、注意力汇聚点、查询图像）之间的比值变化为 e^{λr_j}，这一绝对变化没有上界意义下的"小"。

示例（假设的质量分配，非测量）：某一行中汇聚点 0.60、证据 0.30、查询 0.10，若证据的 r=1、λ=1.2（e^{1.2}≈3.32）：
- 证据：0.30 → 0.30·3.32 / (0.60 + 0.10 + 0.996) = **0.587**
- 汇聚点：0.60 → 0.60 / 1.696 = **0.354**（相对下降约 41%）

这个偏置作用于所有头和所有层，汇聚点主导的头会受到很大影响。此外，OK-VQA 中唯一的查询图像也属于零偏置组，因此 EAB 实际上也在把质量从查询图像转走（论文用 `ctx` 变体检验这一点，`sec_experiments` 中已有对照）。最后，span 的 pre-bias 质量随 token 数增加，256 token 的图像 chunk 获得的绝对位移大于文本 chunk，EAB 的绝对效应与 chunk 长度相关，而命题只描述比例。

**影响**："bounded"的表述容易被理解为"对注意力分布的扰动有界"，但实际上在汇聚点主导的头上扰动可能很大。这可以解释部分负面结果，也使"只是相对重标度"的解释不完整。

**修复建议**：
1. 在 dev 上测量汇聚点与查询图像的质量在 EAB 前后的变化，作为论文的诊断。
2. 论文 sec_appendix.tex:9 说"所有层优于末 4 层 0.6 点"，但没有对应的运行配置或结果文件，来源需要补充。
3. 讨论中把绝对位移与比值并列，而不只是比值。

**是否推测**：是。幅度依赖真实模型的注意力模式，示例数字是假设值。

---

### A-06 H_grd、H_m 是 token 级熵，混合了模态结构（中等，推测）

**位置**：`src/omniaware/qwen_eab.py:365–371`（grounding_entropy 在证据 token 上归一化，除以 log N）；`src/omniaware/routing.py:15–43`（每个模态内部的 token 级 H_m，routing.py:32–34）；论文 `main.tex:172–203`（不确定性小节，含"span length and image-patch density can still affect the score"的注记）。

**分析**：代码确实在 token 级计算，与论文的 Eq. (hgrd) 和 Eq. (routing) 一致。问题在于解释：论文说高 H_grd 表示"attention 分散在多个证据 chunk 上"（main.tex 不确定性小节），但实际计算的是跨 token 的分散。一个 256 token 的图像 chunk 中，注意力天然分布在许多 patch 上，因此即使注意力集中在一个 chunk 内，H_grd 也会偏高。冷启动时探测轮的 N 随模态组成剧烈变化（文本 chunk 几十 token，图像 256，视频 8 帧 × 256），log N 归一化后不在同一尺度上。

**影响**：U 与 M_m·H_m 的判定可能主要反映模态的 token 结构，而不是"证据是否被定位"。这与论文对路由的解释（"one with concentrated attention has already located its evidence"）不一致。

**修复建议**：
1. 在 chunk 级计算熵（先求每个 chunk 的注意力质量，再求熵），或在 dev 上比较两种定义。
2. 统计 H_grd、H_m 与 N、模态组成的关系，并在附录报告。

**是否推测**：是。需要在真实模型上统计 H_grd 与 N 和模态的相关性。

---

### A-07 β 不控制两项的相对权重（中等，推测）

**位置**：`src/omniaware/pipeline.py:102–104`（sum 模式直接相加）；`:82–89`（Norm 按整体 IQR 缩放）；`src/omniaware/run.py:443–448`（求和后再拟合 Norm）；论文 `main.tex:201`（"β sets its share of the sum"）。

**分析**：H_pred 按 log|V| 归一化，论文自己说它"small"。H_grd∈[0,1]，方差可能更大。Norm 使用整体的 IQR，因此两项的有效权重由各自在校准集上的方差决定。β=0.30 不等于"30% 的份额"。zsum（pipeline.py:106–108）正是为解决这个问题而设，但在 dev 中只是候选之一，没有看到两项方差的报告。

**影响**：β 敏感性表（β=0 与 β=0.3 之间的差）取决于两项的方差比。如果 H_pred 的方差远小于 H_grd，则 β 在 0–1 之间几乎没有作用，表中的差异可能只是噪声。

**修复建议**：从 `ucal` 缓存（run.py:416–439 写入）计算两项在校准集上的 std 与 IQR，报告有效份额；或把 zsum 设为默认并在论文中说明。

**是否推测**：是。需要真实的 ucal 数据。

---

### A-08 Static 基线在冷启动数据集上带路由，各数据集基线定义不统一（中等）

**位置**：`scripts/paper_runs.py:59–63`（COLD 数据集自动加 `--probe-uncertainty --first-round-routing`）；`:112–118`（main 阶段的 static 臂来自同一运行）；`src/omniaware/run.py:410–415`（first-round 路由），`:565–566`（静态臂使用路由后的 chunks）；`docs/PAPER_TABLE_MAP.md:12`（Static concatenation = `v321_main` 的 `static` 臂）。

**证据**：paper_runs.py:61–62 对 `webqa_open_full`、`webqa_open_img`、`hybridqa`、`videorag_synth` 自动附加 `--first-round-routing`。因此 main 表中这四个数据集的 Static 臂，实际是 M_m·H_m 路由之后的 top-k；OK-VQA 没有路由，所以各数据集的 Static 定义不同。Cross-encoder 行（`v321_ce`）同样带路由。

**影响**：Static 在冷启动数据集上比"朴素 top-k 拼接"强，对 Omni-Aware 的比较偏保守，但论文若写"Static concatenation（top-k）"就不准确。另外，路由带来的收益没有被单独归因。

**修复建议**：在论文表注中明确；或增加一个不带路由的 static 臂作为朴素基线，并在 routing 表中单独报告路由的收益。

**是否推测**：否。脚本可以直接核对。论文的表述只看了 PAPER_TABLE_MAP，未核对 main 表的正文定义。

---

### A-09 单模态上下文的路由未实现论文的"跳过"规则，传入相关标志时会崩溃（中等）

**位置**：`src/omniaware/routing.py:46–50`（initial_route 把查询图像映射为 image 模态）；`:15–43`（route_modalities 对单模态直接选中，没有"少于两个模态则跳过"的分支）；`src/omniaware/run.py:410–411`（非空的 initial route 被直接采用）；论文 `main.tex:217`（"when the supplied context covers fewer than two candidate modalities, as the query image of OK-VQA against a text corpus, routing is skipped and retrieval searches the full pool"）。

**证据**：`verify_misc.py` (1)：

```
initial_route -> {'mode': 'attention', 'modalities': ['image']}
select_chunks raises ValueError: q: no retrieval candidates for requested modalities ['image']
```

OK-VQA 的候选为纯文本，所以路由把检索限制在不存在的图像候选上，直接抛出异常。

**影响**：传入 `--modality-routing` 或 `--first-round-routing` 的 OK-VQA 运行会崩溃（显式错误，不是静默错误）。paper_runs.py 没有对 OK-VQA 添加这些标志（COLD 集合不含 OK-VQA），所以主表数字不受影响。但代码与论文描述矛盾，且 CLI 对 OK-VQA 实际不可用。

**修复建议**：在 `route_modalities` 或 `initial_route` 中加入"上下文覆盖的模态数少于两个则返回 skip"的分支，与论文一致；并为该分支增加测试。

**是否推测**：否。

---

### A-10 归一化的数值边界（轻微–中等）

**位置**：`src/omniaware/pipeline.py:82–83`（Norm 的 IQR 下限 1e-6）；`:106–108`（zsum 的 z() 在 IQR=0 时返回 0）。

**证据**（`verify_misc.py` (3)(5)）：
```
Norm.fit on near-constant H_pred: Norm(median=0.3, iqr=0.0)  U(0.31)= 1.0  U(0.29)= 0.0
zsum: grd norm iqr=0.0, returned term for h_grd=0.9 -> 0.0
```
Norm 在 IQR 很小时退化为阶跃函数，门控二值化，EAB 的 γU 缩放只剩 0 或 1。zsum 则直接把该项置 0，两者的退化行为不一致。

**影响**：只要某个 phase（pre、probe、post 中的任一个）在校准集上近乎常数，门控和 EAB 强度就会退化。当前没有检查或日志。

**修复建议**：对 IQR 设置相对下限（例如校准标准差的一个比例），或在 IQR 过小时报错；统一 Norm 与 zsum 的退化处理；在 config 中记录每个 phase 的 IQR。

**是否推测**：否。数值行为已验证；是否会在真实数据中出现需要查看 ucal。

---

### A-11 EAB 的静默失效风险与调用顺序陷阱（轻微–中等）

**位置**：`src/omniaware/qwen_eab.py:84`（全局注册 `"sdpa"`）；`:156`（`attn_implementation="sdpa"`）；`:246–247`（`_prefix` 调用 `STATE.reset()`）。

**分析**：所有 EAB 偏置都通过全局注册的 sdpa 钩子生效（我的参考比对证实 ans1 确实改变了首 token logits）。但代码中没有运行时检查钩子是否被调用。若 transformers 回退到 eager 或分发方式改变，`_eab_sdpa` 不会被调用，偏置臂与 static 完全一致，且不报错。另外，`_prefix` 会清空 `STATE`，所以在 `_prefix` 之前调用的 `_activate` 会被丢弃。`run()` 当前的调用顺序是正确的，但这是隐式约定，没有断言。

**修复建议**：
1. `QwenEAB.__init__` 后断言 `text_config._attn_implementation == "sdpa"`，并用计数器确认钩子在前向中被调用。
2. 在 `run()` 中，当 γ>0 且证据非空时，断言偏置臂的首 token logits 与 static 不同（至少在某些样本上）。
3. 把 `STATE.reset()` 移出 `_prefix`，或在 `_activate` 中检查状态。

**是否推测**：否。静默失效的可能性由代码结构直接给出，未在真实模型上复现。

---

### A-12 冷启动参数默认值与文档不一致（轻微）

**位置**：`src/omniaware/run.py:157`（`--probe-uncertainty` 默认关闭）；`:115`（`--first-round-routing` 默认关闭）；`README.md:222`（称 `--probe-uncertainty --first-round-routing` 为 WebQA、HybridQA、VideoRAG 的默认设置）；`run.py:410–415`（first-round 路由用 `probe(e, rel)`），`:553–560`（门控仅在 `probe_out` 存在时用探测 U，否则用 U_pre）。

**证据**：`resolve_route` 在没有 `--probe-uncertainty` 时仍会调用 `probe(e, rel)` 计算路由（run.py:412），但门控 `rec["u_gate"]` 仍用 U_pre（冷启动下 h_grd=0）。

**影响**：只靠 CLI 默认值运行冷启动数据集，得到的是"无探测、无路由"的设置（即 ablation 中的 noprobe），与论文主设定不同。单独开 `--first-round-routing` 得到的是"路由来自探测、门控来自 U_pre"的混合设定。

**修复建议**：让 `--first-round-routing` 隐含 `--probe-uncertainty`（或互相校验并报错）；把 README 的"默认设置"改为"paper_runs.py 对 COLD 数据集的设置"。

**是否推测**：否。

---

### A-13 提示构造与官方模板未核实（轻微，推测）

**位置**：`src/omniaware/pipeline.py:41`、`:67–71`（用户轮、问题与 assistant 头手工拼接，无 system 段）；`src/omniaware/qwen_eab.py:216–218`（每段单独 `tok.encode(..., add_special_tokens=False)`）。

**分析**：未与 Qwen3-VL-Instruct 或 InternVL3.5 的官方 chat template 比对（模型文件不在本地，且按要求不下载）。单独 tokenize 会在段边界处产生与联合 tokenize 不同的 BPE 切分。spans 本身是正确的（已验证）。

**影响**：提示分布可能与官方训练格式略有不同。static 与 EAB 使用同一提示，因此主要影响绝对准确率而非比较。InternVL 若官方模板含默认 system 提示，差异会更大（推测）。

**修复建议**：用 `processor.apply_chat_template` 生成完整文本，并在 token 序列上对齐 spans；或在 README 中说明并给出与官方模板的差异。

**是否推测**：是。

---

### A-14 H_pred 的归一化常数（轻微，推测）

**位置**：`src/omniaware/qwen_eab.py:362`（`h / math.log(logits.numel())`）；`:173`（`self.vocab_size` 取 `text_config.vocab_size`）。

**分析**：归一化使用的是 logits 的元素数（模型输出维度），而不是 tokenizer 的 |V|。Qwen3 系列的模型词表有时大于 tokenizer 词表（padding），这会使分母偏大。

**影响**：通常很小（log 比值为 1e-3 量级，推测），但与论文的 "log|V|" 文字不一致。

**修复建议**：统一用 tokenizer 的词表大小，或在文中说明。

**是否推测**：是。未读取真实 config，量级是推测。

---

### A-15 死代码与静默降级（轻微）

**位置与证据**：
- `src/omniaware/relevance.py:115`：`not c["cid"].startswith("f")`。`data.py` 的 loader 生成的 cid 前缀为 `p/d/t/i/v/c`（data.py:236、:296），以及 webqa 的 `tp/tn/ip/in`（data.py:50、:61）、`t{rank}/i{rank}`（data.py:142、:149）、`k{rank}`（data.py:188）。没有以 "f" 开头的 cid，该条件是死代码。若将来出现以 "f" 开头的图像 cid，它的 caption 分数会被静默丢弃。
- `src/omniaware/pipeline.py:39`、`:52`、`:58`：`image_pixels` 参数，run.py 从未传入，恒为 None。
- `src/omniaware/relevance.py:146–147`：`ecdf()` 对未见通道返回 0.5，不报错。若校准集缺某个通道，所有 r_j 都会变成 0.5。
- `src/omniaware/pipeline.py:128`：`raw_uncertainty` 静默忽略不在 `enc.spans` 中的 cid。

**影响**：当前没有数值影响，但 0.5 默认值与静默忽略会在数据异常时掩盖问题。

**修复建议**：删除死代码；`ecdf` 对缺失通道报错；`raw_uncertainty` 对不存在的 cid 报错。

**是否推测**：否。

---

### A-16 视频帧按独立图像输入 Qwen3-VL（轻微，推测）

**位置**：`src/omniaware/pipeline.py:55–58`（每帧一个 image Segment）；`src/omniaware/qwen_eab.py:199–211`（每个图像得到一个 grid，`t=1`）。

**分析**：8 帧各自作为独立图像输入，每帧 grid 的时间维为 1。Qwen3-VL 的原生视频路径会把相邻帧按 temporal_patch_size=2 合并，并赋予递增的时间位置索引。本实现的时间位置索引对所有帧相同（由 grid t=1 推断，未直接读取 `get_rope_index` 的输出）。论文说"视频帧保持时间顺序"，这只在序列位置层面成立。

**影响**：视频证据的 token 数约为原生视频路径的两倍，时间位置信息较弱，会影响 VideoRAG-Synth 的绝对数字与解释。

**修复建议**：若要忠实于原生视频输入，使用 Qwen3-VL 的视频处理器；否则在论文中说明这一选择。

**是否推测**：是。

---

### A-17 依赖声明与实际验证版本不符（轻微）

**位置**：`pyproject.toml`（`transformers>=4.57`）；`src/omniaware/qwen_eab.py:156`（`from_pretrained(..., dtype=...)`，并使用 transformers 5 的 `AttentionInterface`）。

**证据**：本环境只验证了 transformers 5.5.3，参考比对与全部测试均在该版本上完成。

**影响**：按 pyproject 安装 4.57 可能失败或行为不同。

**修复建议**：收紧依赖版本，或在 CI 中在两个版本上测试。

**是否推测**：否。

---

## 4. 已确认正确、可转为回归测试的项

- `verify_eab_ref.py` 的参考比对（文本、中间层、带图、多步解码）可以直接转为 pytest 回归测试。现有 `test_qwen_eab.py:18–38` 只比较 3 行的 toy 注意力，没有覆盖完整 `run()` 的早/晚臂与 KV crop 组合。
- `verify_encode_and_calib.py` (a) 的图像 token 数对齐检查可作为 `encode()` 的回归测试。

---

## 5. 没来得及深入的方向

1. **InternVL 真实路径**：`_internvl_images` 依赖的 processor 接口（`crop_to_patches`、`max_patches`、`num_patches`）在本环境中没有对应的 InternVL 图像处理器类可供离线验证；`processing_internvl.py:62` 与 `:59` 的属性存在，但 `image_seq_length` 与 `start_image_token_id` 的实际取值未核对。InternVL 的官方模板未核对（见 A-13）。
2. **真实权重下的数值**：bf16 下偏置量化、CUDA 上 sdpa 内核与显式 mask 的兼容、`_eab_sdpa` 在 `use_gqa_in_sdpa=True` 分支中的行为（hook 中 key 未重复时的路径）。
3. **论文"所有层优于末 4 层 0.6 点"的来源**（sec_appendix.tex:9）。
4. **A-04、A-06、A-07 的实证测量**：需要真实模型上的 ARD 差异、H_grd 与 N 的关系、两项方差比。
5. **未读的代码**：`scripts/analysis/dev_select.py`（dev 规则的机械执行）、`scripts/analysis/paper_analyses.py` 的 latency 计算、`rerank.py`（CE 选择器）、`interventions.py` 的 cf/inject/random_edit 细节、`order.py` 的 bootstrap 统计（只检查了函数级）。
6. **论文主张"EAB 降低数据集级顺序敏感性"**（main.tex §Limitations 与命题后段）未核实。

---

## 6. 产物与注意事项

- 本报告：`C:\Users\22688\AppData\Local\Temp\claude\C--Users-22688-Desktop-www-code\8e3454d9-d7cd-4088-813b-e62b2bcd35b9\scratchpad\review\A_core_method.md`
- 验证脚本（均在同一 scratchpad review 目录）：
  - `verify_eab_ref.py`：钩子 vs 无缓存参考（文本与带图，多步解码）。
  - `verify_misc.py`：单模态路由、复制取整、Norm 退化、捕获层偏置（首版，其中第 4 项因调用顺序错误无效，已由 `verify_encode_and_calib.py` (b) 修正）。
  - `verify_encode_and_calib.py`：真实图像处理器的 encode、捕获层专用偏置、多通道 ECDF 压缩。
- 仓库未修改。`pytest` 与 Python 导入可能更新了已存在的 `__pycache__` 目录（未新建被跟踪的文件）。
- 所有数值结论基于小型随机模型或合成数据，属于实现正确性与机制层面的结论，不是论文数字的复现。

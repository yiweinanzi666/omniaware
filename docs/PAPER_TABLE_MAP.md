# 论文各表格与代码的对应（2026-10-10 修订，按统一决策 D1–D18）

先跑 dev 阶段并冻结选择（[DEV_DECISIONS.md](DEV_DECISIONS.md)），再由 `python -m scripts.paper_runs --stage <stage> --print` 生成 test 命令（`--run` 直接执行）；下表中的 tag 省略了模型后缀 `_8b`。

**状态说明。** 本表的臂名与阶段名按 D4、D5、D6、D15、D16、D17 推断，并与 `scripts/paper_runs.py` 的当前命令逐项核对。标记 `TODO(G5)` 的地方是 `paper_runs.py` 目前尚未实现的部分（G5 正在修改该文件），在它实现前按本表执行会得到与论文描述不一致的结果。

`EAB` 指 dev 冻结的形式（默认 `eab:g1.2:ans1:pos:gate`，见 [DEV_DECISIONS.md](DEV_DECISIONS.md)），`EAB_dec` 指同形式的 decode-only 版本（`ans2`）；`gated:X` 是"U > τ 时用 X，否则用 direct"的臂。`γ`、`β`、`τ_b`、`τ` 的值以 `runs/dev_decisions.json` 为准（论文中以 \ph 占位）。分数文件是 `runs/<tag>/<dataset>.official_scores.jsonl`（由 `score_official` 生成）。

**臂名的含义（D4）。** `static_joint` 是不经探测轮和路由、直接取联合 top-k 的静态拼接，论文主表的 "Static concatenation" 行对应它。`static` 是路由后的静态拼接（Omni-Aware 的输入）。冷启动数据集上 `static` 与 `static_joint` 不同。`TODO(G5)`：当前 `paper_runs.py` 的 main 阶段对冷启动数据集一律加 `--probe-uncertainty --first-round-routing`，因此其中的 `static` 臂不是 `static_joint`，也不能用作论文的 Static 行。

## 表 main（主结果）——stage `main`、`universalrag`

| 行 | run tag | 臂 | 备注 |
|---|---|---|---|
| Direct | `v321_main` | `direct` | |
| Static concatenation（joint top-k） | `v321_main` | `static_joint` | `TODO(G5)`：`paper_runs.py` 目前取 `static`；需增加 `static_joint` 臂并保证该臂不加探测、不路由（D4） |
| Cross-encoder + static | `v321_ce` | `static_joint` | `TODO(G5)`：CE 运行不得带探测轮与路由参数（D4：外部基线用联合输入） |
| Found-in-the-Middle | `v321_main` | `fitm` | `TODO(G5)`：FiTM 输入须为联合 top-k（D4） |
| M2-RAG | `v321_main` | `m2rag` | VideoRAG 列为 "—"（运行器会拒绝 `--m2rag`）；输入须含问题图像（F-2）、像素上限与其它臂一致（M-43） |
| SeaKR | `v321_seakr` | `seakr` | **全量 test**（D5）。`TODO(G5)`：`paper_runs.py` 当前的 `--n-eval 1000` 须去掉 |
| UniversalRAG | `v321_universalrag` | `universalrag` | OK-VQA 列为 "—"（D6）；clip 与 video 为独立模态；路由器的推理步骤目前只是 `echo`，需固定 checker 并写成可执行脚本（S-6） |
| UniversalRAG，WebQA-img 列 | `v321_universalrag`（`webqa_open_full`） | `universalrag` | 按图像题过滤得到（D6）。`TODO(G5)`：目前无过滤命令 |
| Omni-Aware | `v321_main` | `gated:EAB` | |
| Omni-Aware + cross-encoder | `v321_ce` | `gated:EAB` | |
| Latency 列 | `paper_analyses latency` | 各臂的实测端到端 wall-clock | D15：`run.py` 记录每个臂的 wall-clock（包括 probe、路由、检索查表、编码、prefill、解码）；分析脚本读取，不再拼段。`static_joint` 不含 probe。`TODO`：`paper_analyses latency` 目前按段拼接，需改为读取实测值 |
| † 显著性 | `paper_analyses paired <基线运行> <本方法运行> --arms <基线臂> gated:EAB` | | D18：双侧配对 bootstrap 的 p 值，Holm 校正；"最强基线"由代码按列确定（列内平均分最高者）。跨运行配对只取 qid 交集，并在配置不一致时报错（M-10） |

列：WebQA 全量 = `webqa_open_full`；WebQA-img = `webqa_open_full` 中的图像题（论文中以同一 test 集的子集给出；D2）；OK-VQA = dev 冻结的语料（默认 `okvqa_gs`，指标 `vqa_accuracy`）；HybridQA = `hybridqa`（`em`，附录 `f1`）；VideoRAG-Synth = `videorag_synth`（`rouge_l`，附录 `bertscore_f1`，需 `--bart`）。

## 表 uomni_ablation（组件对照）——stage `ablation`、`citation`

| 行 | 臂（`v321_ablation`，AP 来自 `v321_cite`） |
|---|---|
| Always retrieve, static | `static`（OK-VQA 上与 `static_joint` 相同；WebQA-img 上为路由后的输入） |
| NTE, static | `nte_shared:static`（NTE 归一化后用共享 τ） |
| Per-format thresholds, static | `nte_trigger:static`（每个数据集的 NTE 阈值为与 U 检索率匹配的分位数） |
| H_grd only, static | 另跑 `--beta 0 --tau-bias <冻结的 τ_b>` 的 `gated:static`（tag `v321_beta0_tb<τ_b>`）。`TODO(G5)`：目前硬编码 `tb0.4`。WebQA-img 列在论文中不报告此行 |
| U_omni, static | `gated:static` |
| U_omni, replication | `gated:replication` |
| U_omni, EAB / decode-only EAB | `gated:EAB` / `gated:EAB_dec` |
| No probe round | `v321_ablation_noprobe` 的 `gated:EAB`。**去掉 `--probe-uncertainty`，保留 `--first-round-routing`**（D17）。论文中 OK-VQA 列为 "—"（无探测轮） |
| EAB + context bias | `gated:eab:g1.2:ans1:pos:gate:all:retriever:ctx` |
| Raw retriever scores | `v321_ablation_raw` 的 `gated:EAB`（`--relevance-norm raw`） |
| Always retrieve, relevance bias | `eab:g1.2:ans1:pos:nogate`（无 `gated:` 前缀 = 始终检索） |
| Relevance bias without U scaling | `gated:eab:g1.2:ans1:pos_nou:gate` |
| Uniform (PASTA-style) bias, gated | `gated:eab:g1.2:ans1:fixed:gate` |
| Random relevance, gated | `gated:eab:g1.2:ans1:pos:gate:all:random` |
| Replication（复制对照） | `gated:replication`（计数 `max(1, round(exp(γUr)))`，不截断，记录 prompt token 数，D11） |
| R% 列 | 记录中 `retrieve` 为真的比例（`paper_analyses stats` 的 `retrieval_rate`） |
| 说明 | 以上变体以默认形式 `ans1:pos` 为例；dev 若冻结为其他形式，`paper_runs.py` 自动替换为同一形式。`--no-calibration`（`v321_ablation_nocal`）只改变 Norm 的统计量，不改变 relevance 归一化（D16） |
| AP 列 | `python -m omniaware.eval.citation.attribution runs/v321_cite/<dataset>.jsonl`。AP 来自 `--answer-style cite` 的另一次解码（表注须写明）；门控臂的 AP 只在检索题上计算（D9）。`TODO(G5)`：`paper_runs.py` 的 citation 命令目前不传 `--arms`，须加入 `gated:*` 与 `replication`（S-20） |

## 表 routing（WebQA 开放检索）——stage `routing`

| 行 | tag | 说明 |
|---|---|---|
| All modalities | `v321_route_all` | |
| Retriever-side scores | `v321_route_retriever` | |
| UniversalRAG router | `v321_route_query` | 路由结果来自 stage `universalrag` |
| M_m only / H_m only / M_m·H_m | `v321_route_mass` / `_entropy` / `_mh` | 均加 `--first-round-routing`（冷启动第一轮联合检索） |
| Oracle | `v321_route_oracle` | |

Score 列为 `gated:EAB`；WR、Cov.、Mod./q 由 `scripts/analysis/routing_metrics.py` 计算。路由候选为 text、image 两种模态（WebQA）。`TODO`：`routing_metrics` 在空路由时不再把全部模态计为实际送入模型的模态（M-48 的同源问题）。

## 表 order 与附录 order_full——stage `order`

**14 个排列**（D5、`order.permutation_schedule()`）：2 个锚点（`sorted`、`reverse`）加 12 个随机排列（种子 17、23、29，每个种子 4 个）。每个运行 `--order perm:<id>`，tag `v321_perm_<id>`，**只用前 1,000 道 test 题**（VideoRAG-Synth 为全部 224 道，由 `--n-eval 1000` 自动截断）。论文必须披露该子集（D5）。

- `paper_analyses order` 给出 14 个排列上的 Std.、按种子分组的偏差、sorted − reverse 差值和翻转率（论文中的 "12 orderings" 指 12 个随机排列，另加两个锚点）。
- 臂为 `static`、`fitm`、`replication`、`EAB`、`EAB_dec`（非门控，检索证据固定）。`TODO(G5)`：表中的 Static 行须为 `static_joint`；目前 `paper_runs.py` 对冷启动数据集的顺序运行带探测轮与路由。
- `paper_analyses order` 对每个排列要求 qid 集合、config 与 manifest 完全一致，并报告每个排列的 n（M-46）。`TODO`：当前实现只取交集。
- 通配符参数须在 Python 中用 `glob` 展开为文件列表再传给分析器（M-45）。`TODO`：当前 `paper_runs.py` 对含 `*` 的参数先做 `shlex.quote`，随后又用 `shell=True` 执行，通配符不会被展开。
- ARD 来自 `paper_analyses ard`（D14：t_end 行、含偏置、末层头平均，所有臂用同一行）。

## 表 causal（干预与检索错误）——stage `interventions`

| 列 | 来源 |
|---|---|
| Mask、CF | `v321_interventions`，`paper_analyses interventions` 的 `changed` / `follows_entity`。Mask 为 attention 层面屏蔽（D10）；CF 替换全部 gold 别名（D10）；分母为固定的合格题，并报告失败数（D10） |
| 随机编辑 | 等长完整词替换，与 CF 用同一批题（D10）。`TODO`：`paper_analyses interventions` 须报告失败计数 |
| Dist. | `attribution --intervention distractors` 的 AP |
| Gold@2–5 | `paper_analyses gold25 runs/v321_main/webqa_open_img.jsonl`（rank 顺序运行，按 `chunk_gold` 选题） |
| σ = 0.1 / 0.2 | `v321_noise<σ>_d{0,1,2}`，`paper_analyses noise`（三次取平均）。`TODO`：`noise` 须校验 draw 身份、数量与题集（M-47） |
| Misled | `inject_answer`、`inject_instruction` 的 `misled` |

行：Static = `static`（`TODO(G5)`：表注须写明其为路由后的输入），Replication = `replication`，EAB = `EAB`。`intervene` 目前只重解码 static 与 EAB 变体，`TODO`：须加入 `replication`（D10）。

## 附录

| 段落 | 来源 |
|---|---|
| β 与 τ_bias 敏感性（仅报告，不用于选择） | `v321_beta<β>_tb<τ_b>`，网格 β ∈ {0, 0.15, 0.3, 0.5, 1.0}，τ_b ∈ {0.2, 0.4, 0.6}。`TODO(G5)`：dev 阶段须增加 (β, τ_b) 与 γ、τ 网格（D7） |
| τ 敏感性（仅报告） | `v321_tau<τ>`，τ ∈ {0.25, 0.35, 0.45}。`TODO(G5)`：目前无 `--tau` 命令 |
| 校准规模 | `v321_ablation_nocal`（不校准）、`v321_calib100`、与 `v321_ablation`（500）。**1,000 不再运行**（超过 OK-VQA 保留区，D1）。`TODO(G5)`：删去 `--n-calib 1000` |
| 校准迁移 | 只在冷启动数据集之间（`--norm-from` 的来源须带 `norm_probe`，否则报错；S-20）：HybridQA → VideoRAG-Synth、HybridQA → WebQA-img。`TODO(G5)`：目前的 `v321_transfer_okvqa`（OK-VQA 来源，无探测统计量）须删除 |
| 跨模态分数归一化 | `v321_ablation_minmax`（需 `--relevance-norm minmax`，D16）、`v321_ablation_raw` |
| 引用支持（Faith、AP、Inv） | `attribution` 的 `Faith`、`AP`、`Inv`，alias 与 NLI 判定分别报告（D9） |
| 选择性风险（AURC、ρ） | `paper_analyses selective`（每个系统用自己的答案：U→gated:EAB、NTE→nte_trigger:static、SeaKR→seakr） |
| 触发一致性 | `paper_analyses agreement` |
| 不确定性计算开销 | `paper_analyses latency` 的 `prefill_direct_ms`、`prefill_probe_ms`，以及各臂的实测 wall-clock（D15） |
| 原始检索分数范围 | `paper_analyses ranges cache/relevance_<dataset>*.json` |
| 骨干迁移（InternVL3.5-8B） | stage `main`，`--model internvl3.5-8b`。InternVL 图像为单个 448 tile，提示词手写（S-17，论文附录已披露） |
| 次要指标 | `score_official --bart` 的 `f1` / `bertscore_f1` |
| 完整语料对照（168,306 句） | `okvqa_gs_full`（`gs168k_full`），仅附录；`TODO(G5)`：目前未在 `paper_runs.py` 中 |
| 数据统计、首 token 占比 | `paper_analyses stats` |
| 错误分析 | `paper_analyses errors`（规则预标注 + 抽样人工核对）；EAB 的错→对 / 对→错与按偏置最多证据块分类的退化：`paper_analyses transitions --arms gated:static gated:EAB` |
| 重排器召回（gold 在 top-k 中） | `paper_analyses stats runs/v321_ce/<dataset>.jsonl` 与 `v321_main` 的 `gold_in_topk` |

## 多卡

上面每条 `omniaware.run` 命令都可以经 `scripts.launch` 在多卡上运行（`paper_runs --gpus auto` 会自动包装）；分片合并后的 `<dataset>.jsonl` 与单卡运行的格式完全一致，下游评分与分析不变。`TODO`：`launch` 须检查 `.merged` 标记并核对行数（S-9）；`merge_shards` 默认须从配置读取题数（M-8）。

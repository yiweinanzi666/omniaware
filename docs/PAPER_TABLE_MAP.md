# 论文 V3.21 各表格与代码的对应

所有命令由 `python -m scripts.paper_runs --stage <stage> --print` 生成（`--run` 直接执行）；下表中的 tag 省略了模型后缀 `_8b`。
`EAB` 指变体 `eab:g1.2:ans2:pos:gate`；`gated:X` 是"U > τ 时用 X，否则用 direct"的臂。分数文件是
`runs/<tag>/<dataset>.official_scores.jsonl`（由 `score_official` 生成）。

## 表 main（主结果）——stage `main`、`universalrag`

| 行 | run tag | 臂 |
|---|---|---|
| Direct | `v321_main` | `direct` |
| Static concatenation | `v321_main` | `static` |
| Cross-encoder + static | `v321_ce` | `static` |
| Found-in-the-Middle | `v321_main` | `fitm` |
| M2-RAG | `v321_main` | `m2rag`（VideoRAG 列为 "—"，运行器会拒绝 `--m2rag`） |
| SeaKR | `v321_main` | `seakr` |
| UniversalRAG | `v321_universalrag` | `universalrag` |
| Omni-Aware | `v321_main` | `gated:EAB` |
| Omni-Aware + cross-encoder | `v321_ce` | `gated:EAB` |
| Latency 列 | `paper_analyses latency` | 始终检索 / 门控臂由 prefill + 解码时间算出；FITM、M2-RAG、SeaKR 为各自的实测墙钟时间 |

列：WebQA 全量 = `webqa_open_full`；WebQA-img = `webqa_open_img`；OK-VQA = `okvqa`（指标 `vqa_accuracy`）；HybridQA = `hybridqa`（`em`，附录 `f1`）；VideoRAG-Synth = `videorag_synth`（`rouge_l`，附录 `bertscore_f1`，需 `--bart`）。显著性：`official_summary.json` 中的配对 bootstrap。

## 表 uomni_ablation（组件对照）——stage `ablation`、`citation`

| 行 | 臂（`v321_ablation`，AP 来自 `v321_cite`） |
|---|---|
| Always retrieve, static | `static` |
| NTE, static | `nte_shared:static`（NTE 归一化后用共享 τ） |
| Per-format thresholds, static | `nte_trigger:static`（每个数据集单独匹配检索率的阈值） |
| H_grd only, static | 另跑 `--beta 0`（tag `v321_beta0_tb0.4`）的 `gated:static` |
| U_omni, static | `gated:static` |
| U_omni, replication | `gated:replication` |
| U_omni, EAB / EAB-P | `gated:eab:g1.2:ans2:pos:gate` / `gated:eab:g1.2:ans1:pos:gate` |
| EAB + context bias | `gated:eab:g1.2:ans2:pos:gate:all:retriever:ctx` |
| Raw retriever scores | `v321_ablation_raw` 的 `gated:EAB` |
| Always retrieve, relevance bias | `eab:g1.2:ans2:pos:nogate`（无 `gated:` 前缀 = 始终检索） |
| Relevance bias without U scaling | `gated:eab:g1.2:ans2:pos_nou:gate` |
| Uniform (PASTA-style) bias, gated | `gated:eab:g1.2:ans2:fixed:gate` |
| Random relevance, gated | `gated:eab:g1.2:ans2:pos:gate:all:random` |
| R% 列 | 记录中 `retrieve` 为真的比例（`paper_analyses stats` 的 `retrieval_rate`） |
| AP 列 | `python -m omniaware.eval.citation.attribution runs/v321_cite/<dataset>.jsonl` |

## 表 routing（WebQA 开放检索）——stage `routing`

| 行 | tag | 说明 |
|---|---|---|
| All modalities | `v321_route_all` | |
| Retriever-side scores | `v321_route_retriever` | |
| UniversalRAG router | `v321_route_query` | 路由结果来自 stage `universalrag` |
| M_m only / H_m only / M_m·H_m | `v321_route_mass` / `_entropy` / `_mh` | 均加 `--first-round-routing`（冷启动第一轮联合检索） |
| Oracle | `v321_route_oracle` | |

Score 列为 `gated:EAB`；WR、Cov.、Mod./q 由 `scripts/analysis/routing_metrics.py` 计算。

## 表 order 与附录 order_full——stage `order`

14 个 `--order perm:<id>` 运行（tag `v321_perm_<id>`）。`paper_analyses order` 给出 12 个随机排列上的 Std.、按种子分组的偏差、sorted − reverse 差值和翻转率；臂为 `static`、`fitm`、`replication`、`EAB`（非门控，检索证据固定）。ARD 来自 `paper_analyses ard`。

## 表 causal（干预与检索错误）——stage `interventions`

| 列 | 来源 |
|---|---|
| Mask、CF | `v321_interventions`，`paper_analyses interventions` 的 `changed` / `follows_entity` |
| Dist. | `attribution --intervention distractors` 的 AP |
| Gold@2–5 | `paper_analyses gold25 runs/v321_main/webqa_open_img.jsonl` |
| σ = 0.1 / 0.2 | `v321_noise<σ>_d{0,1,2}`，`paper_analyses noise`（三次取平均） |
| Misled | `inject_answer`、`inject_instruction` 的 `misled` |

行：Static = `static`，Replication = `replication`，EAB = `EAB`。

## 附录

| 段落 | 来源 |
|---|---|
| β 与 τ_bias 敏感性 | `v321_beta<β>_tb<τ_b>` |
| 校准迁移、校准规模、不校准 | `v321_transfer_okvqa`（`--norm-from`）、`v321_calib<n>`、`v321_ablation_nocal` |
| 跨模态分数归一化 | `v321_ablation_minmax`、`v321_ablation_raw` |
| 引用支持（Faith、AP、Inv） | `attribution` 的 `Faith`、`AP`、`Inv` |
| 选择性风险（AURC、ρ） | `paper_analyses uncertainty`（U_omni、NTE、H_grd、SeaKR） |
| 骨干迁移 | stage `main`，`--model internvl3.5-8b` |
| 次要指标 | `score_official --bart` 的 `f1` / `bertscore_f1` |
| 数据统计、首 token 占比 | `paper_analyses stats` |
| 错误分析 | `paper_analyses errors`（规则预标注 + 抽样人工核对） |

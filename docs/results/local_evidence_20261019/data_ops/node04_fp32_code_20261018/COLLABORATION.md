# WWW2027 协作入口

本仓库同步当前论文、实验代码、协议、预注册、台账、Claude候选、独立审查与汇总结果。提交是工作阶段快照；代码候选、CPU回归、当前实测与历史服务器结果分别标明来源和比较范围。

## 先读什么

- 最新状态：`experiments/eab_first_token/runs/goal_diagnosis_codex_20261006/GOAL_STATUS.md`
- 负结果与真实差距：`experiments/eab_first_token/NEGATIVE_RESULTS_LEDGER.md`
- 完整实验记录：`experiments/eab_first_token/ledger.csv`
- 原要求：`experiments/eab_first_token/requirements.md`
- 实时分工：`deliveries/WWW2027_Claude_Codex_并行优化_20261005/CLAUDE_COORDINATION_会话分工_20261006.md`
- 新账号接续：同目录 `CODEX_ACCOUNT_SWITCH_HANDOFF_20261006.md`
- 原协议与本地差异：`deliveries/WWW2027_V3.8_阶段0核查与训练准备_20261006/PROTOCOL_COMPARISON.md`

## 如何修改

合作者从最新 `main` 建自己的分支，提交可审查的改动。2026-10-07 用户已切换到原EMNLP Ours恢复优先级，旧WebQA MPS续跑已暂停，AutoDL由用户报告关机，MMQA已完成；实际运行状态看最新Goal记录。正在运行的源码、冻结配置和输入哈希保持不变。需要修复时放新候选目录，附失败复现、文件范围、针对性回归和交付说明，交给Codex独立验证后排队。不要使用仓库里的历史PID直接操作进程。

当前Codex管理本地MPS资源、代码修改、独立验证、评分与原协议恢复。用户于2026-10-08明确要求“现在开始不允许使用 Claude”，此前的Claude交互授权已被取代。禁止启动、连接或调用Claude应用、CLI/API，不发送任务或反馈，不唤醒、重连或操作账号；已有落盘交付保留供Codex独立审查。Claude专属自动任务已删除。只有用户之后明确解除禁令才可恢复。

用户最新要求四个数据集采用官方评分，后续停止宽松评分开发和重算，已有评分与全部正负结果保留。WebQA公开`acc_approx`是组件，完整组合分需要ParaBank BARTScore；组件提升与完整成绩分别报告。已有baseline预测和分数直接复用；用户另行明确授权的Video Short900 static例外已经保存，不重复生成。18:12 UTC无CUDA设备是历史观察；19:49 UTC实际单张Pro6000正在运行已授权Video32帧候选。云端调度归现有执行会话“重新设置 EMNLP 复现实验”，root不另起重复worker、不自行开机或恢复旧强基线队列。最新额外一小时收尾截止2026-10-08T20:11:46Z，总180元，不自行续时。本地保持一个MPS模型任务。已完成输出不重跑；Git提交不自动上传或搬迁运行资产。

最新官方独立汇总见`experiments/emnlp_repro_codex_review/CURRENT_OFFICIAL_RESULT_SUMMARY_20261008.json`及同目录`CURRENT_EMNLP_COMPARISON_20261008.md`。其中保存内部基线比较、已消费开发迭代、原EMNLP协议缺口分别说明；独立重评分不等于独立样本确认，表头4题修复不解释全部方法收益。

## 同步范围和大型资产

用户于2026-10-06明确选择“协作文件全量，保留大型数据在本地”。`.gitignore`保留源码、协议、CSV台账、汇总JSON、审查、图表源文件、论文和交付文档；不上传模型权重、虚拟环境、张量、完整语料、逐调用原始输出、下载像素、数据库和重复全数据压缩包。`cache/`顶层的调度/审计Python与协议、配置也纳入Git，运行环境和下载资产仍忽略。

本地大文件及恢复入口见 `LOCAL_ASSETS.md` 和 `collaboration/local_assets_inventory.json`。被忽略的文件没有被删除；需要原始实验数据时按清单从实验主机或所记录的固定公开来源恢复。保存的绝对路径表示实验主机上的来源位置，不能直接假定在合作者电脑上存在。

必要的小型评分规则资源随源码同步：`experiments/emnlp_repro_claude/erc/resources/YESNO_REFERENCE_POLARITIES.json`，40,420字节，SHA256为`7647c040ec80665330252ec4afe9f9cadf05a71fa41c0053ed0bfb3b7dce5cd6`。它仅用于WebQA参考极性评分，未进入模型输入；缺失时旧scorer会静默改变部分分数。独立r2修复及回归在`emnlp_replication_recovery_20261007/incremental_acceptance_20261008T0155/missing_resource_candidate_r2/`，当前为已验证的隔离候选，主scorer是否采纳以最新交付记录为准。

## 定期同步

- Git同步：用户已要求停止提交，原每2小时自动任务已删除。Codex不再主动提交或推送；只有用户之后再次明确要求才恢复。已有提交保留。
- 每3小时：检查防御性编程、写作和声明，保留必要的科学限定和真实不确定性。
- Claude专属定时任务已删除；Codex按当前Goal接续已有交付的独立审查，禁止与Claude交互。

远端若出现合作者的新提交造成分叉，保留双方提交，通过隔离分支集成；不强推，也不在运行实验的工作树自动rebase或覆盖文件。定时任务依赖本机和Codex应用运行。

## 72小时目标窗口

Goal工具创建时间为2026-10-06 07:20:57 PDT（14:20:57 UTC，Unix1791296457），据此72小时截止2026-10-09 07:20:57 PDT。目标是完成必需实验和数据准备、检验原EMNLP效果及老师要求的5–10%提升，并保留机制和真实成本证据。到期按实际完成情况交付；负结果或未完成项不能改写成成功。

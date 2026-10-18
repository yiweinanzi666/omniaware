# node04 自检与代码复核报告（2026-10-18）

结论：**尚未全部通过，正式实验保持阻塞。** 当前源代码的 CPU 测试通过，真实 Qwen3-VL-8B 的 FP32 图文自检通过；正式实验使用的 BF16 数值自检仍失败。已有 S1 自动格式通过，但人工抽查发现 WebQA 的 short 字段包含整句解释，不能据此宣布 S1 全部验收。

## 1. 来源、范围与操作边界

- 本轮服务器隔离 checkout：`/home/user/Futaoran/omniaware_v2_selfcheck_20261018`，提交 `28e9514dc432d6891a492390bc4bdef176c519ea`。复核位置的行号均指此提交。
- Python 3.11.17、torch 2.11.0+cu128、transformers 5.5.3，NVIDIA L20。
- 真实权重：`/home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct`。
- 证据：本报告旁的 `node04_selfcheck_20261018/`；服务器原始日志在 `/home/user/Futaoran/omniaware_selfcheck_logs_20261018/`。
- 用户授权跨会话协调后，实验会话释放 GPU1，并保留 OK-VQA 开发集 54/256 条结果。WebQA 的既有 GPU0 作业仍由原会话负责；后续评分、选择及阶段推进保持暂停。不能把“后续暂停”说成两卡上的全部进程都已停止。
- 本轮没有修改 omniaware 源码、提示词、解析器、阈值或另一会话的环境。逐层 hook 和精度试验只作用于诊断进程。隔离 checkout 的 tracked files 无修改，`external` 是未跟踪的既有依赖软链。
- 未执行 S4，未重复下载或清理已经验收并清理的本地数据副本。

## 2. 实测结果

| 检查 | 结果 | 证据与边界 |
|---|---|---|
| `pip check` | 通过 | `No broken requirements found.` |
| 完整 pytest | 通过，无 failed/error | 546 passed、1 skipped、21 subtests passed，47.08 秒；`pytest.log`、`pytest.xml` |
| 唯一 skipped | 明确记录，不计作通过 | `test_all_128_saved_inputs_preserve_order_text_and_native_prompt` 缺少旧版 T15 历史输出；不是当前 8B/处理器缺失 |
| BF16 原始真实权重自检 | **失败** | 保留 `verify_real_model_8b_bfloat16_failed.json`；本轮 `precision_diagnostic.json` 在相同阈值下再次复现文本失败 |
| 原版脚本 FP32 文本＋图片自检 | **通过** | `verify_fp32_text_image.exit=0`、顶层及全部判定项 `ok=true`；仅证明 FP32 测试范围 |
| S1 输入、环境与日志溯源 | 本次检查项通过 | 两数据集各 20 行 prompt；env 有代码提交、dirty=false、sdpa、BF16、L20；log 同时有 `[env]` 和 `[done]` |
| S1 自动格式率 | 现有自动关卡通过 | WebQA 各动作 94%～100%；OK-VQA 为 plain，自动检查并不验证词/短语质量 |
| S1 人工格式抽查 | **尚未通过** | 前 20 题逐动作输出保存在 `s1_trace_and_format_review.json`，存在明显整句 short |

FP32 使用未修改的原版脚本，默认最多 8 个解码步骤，遇 EOS 会提前终止，不能声称每个臂实际生成了 8 个 token：

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.verify_real_model \
  --model /home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct \
  --dtype float32 \
  --image /home/user/Futaoran/omniaware/assets/data/coco/val2014/COCO_val2014_000000000042.jpg \
  --out /home/user/Futaoran/omniaware_selfcheck_logs_20261018/verify_fp32_text_image.json
```

运行前使用既有 Python 3.11 环境，并将 PYTHONPATH 指向隔离 checkout 的 src。冻结阈值未改：logits ≤ 0.05，attention ≤ 0.02。

| FP32 检查 | 文本最大误差 | 图片最大误差 |
|---|---:|---:|
| segmented logits | 0.0000705718994 | 0.0000696182251 |
| reference logits（三臂最大） | 0.0000782012939 | 0.0001339912415 |
| reference attention（三臂最大） | 0.0000001192093 | 0.0000014393590 |

模板文本完全匹配。图片模板按原脚本折叠 image pad 连续段后匹配，但图像 token 数为 234 对 300，`image_runs_equal=false`。这是该脚本单独报告的预处理网格差异，不是其模板通过条件；因此不能将 `ok=true` 描述为图片预处理逐 token 完全等同。实际 segmented/reference 对同一编码比较并通过。

## 3. 阻塞与影响解读的问题

### R1：正式 BF16 与通过的 FP32 自检不一致（阻塞）

位置：`src/omniaware/qwen_eab.py:187` 默认 `dtype=torch.bfloat16`；`src/omniaware/run.py:653` 构造 QwenEAB 时没有传 dtype；`scripts/verify_real_model.py:41` 冻结阈值。

BF16 文本 segmented 最大误差 0.375；reference static/ans2 为 0.404296875、ans1 为 0.5，超过 0.05。贪心 token 相同不能代替数值通过。原始失败报告与本轮诊断均保留。

本轮排查：关闭 BF16 reduced-precision reduction、切到 math SDPA、仅提高 attention 计算精度、仅提高线性层计算精度均未解决。flash-only 在该 mask/形状下没有可用 kernel。各诊断脚本的退出 0 只代表完成记录，不能覆盖 JSON 内的失败。

进一步捕获第一层 MLP 的输入及输出：gate_proj、up_proj 与 down_proj 的输入全等；gate_proj、up_proj 输出全等；down_proj 在整段 63-token 与分段 39+24-token 计算间有 457 个输出值不同，最大差 0.00390625。证据为 `mlp_input_diagnostic.json`。这定位了该样例、math SDPA 条件下的首个已测数值分歧，支持形状相关 BF16 线性运算舍入随后被深层放大的解释；并不证明所有样例只存在这一处问题。

同一权重整体 FP32 的单步诊断和完整原版图文验证都通过。但这仍不是正式 BF16 的修复。建议采用下一节明确的 FP32 配置方案，经确认后在独立代码分支实施并重验。

### R2：只更改模型 dtype 会复用错误的校准缓存（切换精度时阻塞）

位置：`src/omniaware/run.py:844`、`:849` 的 U 校准缓存路径和 fingerprint；`:921`、`:930` 的 SeaKR fingerprint/路径。

目前没有 dtype 字段。现有单一 BF16 配置下，这不能单独证明已发生混用；但若仅将模型切成 FP32，旧 BF16 entropy/attention/SeaKR 数值仍可能命中缓存。需要同时纳入显式 dtype 指纹、独立缓存命名和新运行目录，确保旧结果不覆盖、新精度重新校准。`run.py:535` 和 `:975` 已按参数保存/比较配置，新增 dtype 必须参与该比较，不能加入 EXECUTION_ARGS 排除项。

### R3：S1 自动格式通过不能替代人工短答案验收（阻塞）

位置：`src/omniaware/answers.py:107` 的格式判断只要求存在第二行、第一行词数不多于第二行；`scripts/analysis/answer_slot.py:95` 对非 answer_first 格式直接返回通过。这符合现有自动检测范围，却不证明第一行一定是短语。

WebQA 前 20 题中，qid `d5c50c840dba11ecb1e81171463288e9` 的 direct short 是：

> There is no single purple flower with the most petals, as petal counts vary widely by species and even individual blooms.

同题的 static_image/mixed short 也为整句。qid `d5de89020dba11ecb1e81171463288e9`、`d5dce9300dba11ecb1e81171463288e9` 等存在解释式第一行。因而不能宣布 HANDOFF 要求的人工短答案检查通过。OK-VQA 大多为短语，但 qid 2022015 的部分动作给出 `not specified`，当前解析未标为拒答，应作为解释限制留痕，不能悄悄修改答案或重算成另一种拒答规则。

WebQA template_start 为 direct 0.22、static_text 0.34、static_image 0.2653、static_mixed 0.28、rerank_static 0.24。这里只报告观察值；50 题样本不足以凭空给出“明显差异”的新阈值。正式精度统一后重跑 S1 并重新人工审查；如仍失败，应再提交具体提示词/格式方案，不能改变冻结规则来制造通过。

### R4：图像核验的覆盖边界（影响解读）

已有 CPU `image_evidence_check.json` 检查了 50 道图像题：static_image 250 图、gold 命中 50/50；static_mixed 167 图、gold 命中 48/50；不可读、配对错误、ID 错配均为 0。超预算图片均按预算缩放。报告明确写着 `gpu_probe_executed=false`，其中视觉 token 为估计值，不能称为模型逐图语义描述验证。

独立核对实际 S1 JSONL：WebQA direct 50/50 无图；22 道图像题的 static_image 均有输入图片，记录的 image_tokens 全部大于 0；OK-VQA 四个可用动作各 50/50 带 1 张问题图；first_token.topk 的候选均有 text。另做的完整 FP32 图片自检证明了该样例的数值路径，但未覆盖全部检索图片的语义理解或最大图像负载。

## 4. 代码复核与回归测试支持的性质

以下为定点源码复核与已执行测试的覆盖情况，不是所有真实数据、所有模型和所有长输入的形式化证明。

| 性质 | 源码和证据 |
|---|---|
| 五动作的候选池及共同指令、预算 | `run.py:155` 构造模态子池及共同候选，`:178` 共用 answer_style/max_new，`pipeline.py:49` 与 `prompts.py:14` 使用冻结指令；真实 S1 prompt 样例核对 |
| 切分互斥 | `data.py:490` 显式检查三个集合两两不交；`test_splits.py`、`test_v2_splits.py` 通过 |
| 选择只用校准表、开发集用于判定 | `v2_runs.py:223` 编排 calib/dev 到对应步骤，`v2_table.py:447` 将指定校准表交给 select_v2；相关流程测试通过。这不等于任意手工 CLI 都不能传错表 |
| 官方评分与来源校验 | `score_official.py:96`、`:116` 调用官方指标；`webqa_acc.py:43` 校验 pinned 源码 SHA256；`test_metrics_fixes.py`、`test_webqa_bart.py` 等参与本轮通过 |
| 冻结 gate 与 devcheck | `v2_table.py:59`、`:61` 分别为 0.08/0.05；`:361` gate，devcheck 另检查 CI 下界；对应流程与失败中断测试通过。当前没有新的 S2 分数/关卡通过结论 |
| 指纹及续跑保护 | `run.py:529` 起检查输出配置，`:858`/`:935` 按指纹过滤；`test_cache_resume.py` 通过。dtype 扩展需补 R2 所述测试 |
| 基线共同输入 | `run.py:1159` 起 FiTM 使用 joint encoding，M2-RAG 使用 joint 候选和同一 instruction；SeaKR 按其算法读取 direct/joint 单文档；适配器测试通过，不能声称不同算法拥有字节级相同内部提示词 |
| EAB 分析臂与主表隔离 | `run.py:139` 配置分析臂，`test_v2_trace.py:48` 验证不改变五动作；`v2_runs.py:227` 只对开发集命令加入分析臂；对应回归测试通过 |

## 5. 具体待确认方案

建议允许独立分支 `fix/node04-explicit-dtype-20261018` 的源码修复，并批准本轮正式运行改用 FP32。修改范围与验收计划见同目录 `20261018_node04_precision_proposal.md`。这将增加显存和耗时；完整多图负载能否稳定运行须实测，不能以单图通过作保证。

本报告不批准或放行后续 S2/S4。原任务文件第 1 节第 3 条要求“确实要改代码才能运行时，先停下汇报，我确认后在单独分支单独提交”；所以先提交证据和可审阅方案，等待该源码/实验配置变更确认。既有数据、源码和全部成功/失败结果继续保留。

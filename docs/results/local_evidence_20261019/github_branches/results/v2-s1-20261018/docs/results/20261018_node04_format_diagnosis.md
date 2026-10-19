# node04 原始回答格式诊断 — 2026-10-18

结论：已保存的两套 FP32 S1 共 449 个可用动作，其生成 token 用原模型 tokenizer 复原并重新调用原解析器后，六个解析字段全部一致，没有截断命中。未发现解析器把短答案错误扩成长句的证据。WebQA 的自动格式标记只检查结构和相对词数，不能证明第一行满足人工短答案要求。

## 本次实际执行

源码 a11e234，S1 归档 cb2d7b9。使用服务器现有 tokenizer 和存储的 `actions.*.decode.tokens`，没有加载 8B 权重进行新推理、没有重新评分，也没有修改已有 JSONL。命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/user/Futaoran/omniaware_v2_python311/bin/python /home/user/Futaoran/omniaware_selfcheck_logs_20261018/diagnose_saved_answer_format.py
```

命令退出 0。报告保存在 `docs/results/node04_format_diagnosis_20261018/saved_format_diagnosis.json`，含每条原始复原文本、解析值、差异列表和来源文件哈希；同目录保留诊断脚本和 manifest。

| 数据集 | 题数 | 可用动作数 | 重新解析不一致 | 生成长度上限命中 |
|---|---:|---:|---:|---:|
| WebQA | 50 | 249 | 0 | 0 |
| OK-VQA | 50 | 200 | 0 | 0 |

比对字段：answer、short、answer_first_ok、answer_empty、citation_only、refusal。此检验确认保存/解析路径一致，不证明模型答案正确，也不是新的语义评测。

## 根因证据

1. `src/omniaware/answers.py:97`：`form_ok = bool(sentence) and len(short.split()) <= len(sentence.split())`。第二行存在且第一行词数不更长即可通过；第一行是一句完整解释并不会自动失败。
2. `scripts/analysis/answer_slot.py:96`：非 answer_first 输出直接返回格式通过。因此 OK-VQA plain 模式的 passed=true 不是最短措辞认证。
3. 对固定的 WebQA 前 20 题逐动作查看，明确标出 14 个完整句第一行，11 个的 answer_first_ok=true。逐例证据在 `manual_sentence_examples.json`。这是已确认的例子集合，不是穷尽的自动语法分类器，也没有引入新通过阈值。
4. OK-VQA qid2022015 的原始生成确实为 `in a bakery`；static_text/mixed 原始生成确实为 `not specified`，后者现有解析器 refusal=false。它是拒答统计口径的限制，不能将零 refusal 描述为完全没有语义弃答。
5. 多词不等于格式失败。`cavalier king charles spaniel`、完整地名、数字与单位可能是必要答案。不得截短到 gold、机械删除问题重合词或以词数规则强行制造通过。

## 已核对的修复基础

当前 `run.py` 已将提示词哈希纳入 config、U/SeaKR 缓存指纹及文件名。已有 `tests/test_cache_resume.py::test_answer_instruction_is_part_of_the_calibration_key` 验证提示词变化拒绝复用同一答案目录，并保留旧校准文件；`tests/test_v2_prompts.py` 验证运行和基线使用同一指令。本次只审阅这些测试；没有改源码，所以没有重复执行完整测试。

拟修改的是两条生成指令，详见同目录 `20261018_node04_format_revision_proposal.md` 的全文、隔离运行方案和研究限制。当前没有改冻结提示词、解析器、阈值、切分或预算，没有启动新 S1/S2。新的真实验证必须等具体提示词方案获批后再开始，不能用本次只读诊断替代。

## 当前连接与协调

用户要求继续处理后，同一原 SSH 桥在 16:42:37 只读访问成功：GPU0 19416 MiB/49%，GPU1 14 MiB/0%；旧 WebQA BF16 进程仍由另一会话负责。此前自动审批拒绝的历史记录保留，本次没有改用其他连接路径。已协调保留 GPU1 给后续验证，不更改旧作业。

WebQA 人工格式 FAIL 与 OK-VQA 严格格式未明确通过的原结论不变，未获得新 S2 放行。S4 禁止。两条新提示词已完整列出并请求确认；对之前已批准且完成的 FP32 修复不再申请批准。

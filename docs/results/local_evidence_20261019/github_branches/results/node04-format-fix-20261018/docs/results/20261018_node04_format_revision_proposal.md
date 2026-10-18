# node04 回答格式修复方案（待批准；尚未修改冻结提示词）

用户要求继续解决 FP32 S1 的回答格式问题。本方案只解决模型原始输出的格式，保留既有数值修复和全部失败证据。当前运行代码为 a11e234，已归档结果为 cb2d7b9。

## 已定位的问题与边界

`answers.py:97` 的 WebQA `answer_first_ok` 只检查第二行非空、第一行词数不超过第二行；它不判断第一行是不是完整解释句。因此自动格式通过与人工短答案检查失败可以同时发生。OK-VQA 是 plain 格式，自动检查不审查措辞是否最小。不能把这两个自动检查当作完整语义验收。

OK-VQA 的 `in a bakery` 确实包含位置短语的框架词；但 `new york city`、狗的品种、完整机构名称等多词答案可能必要。不能按词数上限、是否多词或是否含问题词机械删词，不能按 gold 把输出截成答案。本方案不改变评分或解析器，也不把单纯答错判成格式失败。

`not specified` 未被现有拒答规则识别是已记录的统计口径限制。本轮保留它的原始值并在独立诊断中提示，不以修改拒答规则改变旧结果。

## 拟批准的唯一提示词候选

修改 `src/omniaware/prompts.py` 的两个默认字符串；所有同数据集动作使用完全相同的字符串，不注入本次样例、gold、qid 或答案词，不搜索多套提示词择优。

WebQA：

```text
Return exactly two lines. On the first line, give only the shortest word or phrase that directly answers the question. Keep words needed to identify the answer, including complete names, numbers and units. Do not restate the question, explain the answer, or add a label on the first line. On the second line, write one complete sentence that states the same answer and restates the key object in the question. Do not add any other lines.
```

OK-VQA：

```text
Answer the question using only the shortest word or phrase that directly answers it. Keep words needed to identify the answer, including complete names, numbers and units. Omit introductory wording and words that only repeat the question. Do not explain the answer, add a label, or write a complete sentence.
```

此修改不强制猜测，也不规定拒答措辞。WebQA 仍用第二行整句做官方评分、第一行用于信号；OK-VQA 仍按原始短答案评分。FP32、图像/生成预算、数据划分、阈值、候选池和解码策略均保持当前设置。

## 实施及验收

1. 在独立代码分支提交提示词修订和协议说明。回归验证所有动作收到同一指令，config 记录新哈希，旧运行目录续写被拒绝，新 U/SeaKR 指纹不会复用旧提示词校准。
2. 新建输出与缓存 namespace `fp32_format_20261018`。只放置已验收的不可变数据输入，不复制旧模型校准缓存；保留旧目录。
3. 运行完整 CPU 回归、原版真实权重数值自检后，GPU1 串行重跑两套各 50 题 S1。记录全部输出，不只保存成功样例。命令保持 FP32 与原始预算。
4. 自动数值和格式阈值保持不变。人工复核各前 20 题全部可用动作，逐条区分简洁答案、完整解释句、必要多词名称、额外框架词、拒答及单纯预测错误。保留证据和判断理由；不使用启发式长度规则假装语义判断。
5. 任一关卡失败或不明确则停止该数据集并报告；不得自动继续改提示词直到通过。全部明确通过后，才按原授权重新开始独立的 S2 校准和开发阶段。S4 始终禁止。SeaKR 不因耗时自行禁用。

## 研究解释限制

本次方案已看过原 S1 开发样例，因此是已观察到开发结果后的协议修订，不能声称新的提示词选择发生在任何开发数据之前。更新 DEV_DECISIONS 时明确披露日期、原失败和这一限制；重跑同一开发集不会恢复其未见属性。不读取测试集答案，不为提高开发分数设计答案专用规则。若要求严格的未见评估，需要另行批准数据协议，不能暗中改变切分。

## 为什么需要这次具体确认

原任务文件第 0 节第 1 条冻结提示词，第 0 节第 3 条要求源码改动先报告确认；仓库 `docs/HANDOFF_v2.md` 第 7 节要求提示词变更先商量并从 S2 重做。此前批准的是显式 FP32 和缓存隔离，不包含这里列出的新提示词。此次仅请确认上面两个具体字符串与隔离重验范围，不再请求已获准的 FP32 修复权限。

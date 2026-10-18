"""Build the final report only from completed artifacts and explicit manual review decisions."""
import pathlib,json,hashlib
root=pathlib.Path('D:/WWW2027/data_ops/node04_fp32_code_20261018')
base=root/'runs/fp32_format_20261018'
evidence=root/'docs/results/node04_format_validation_20261018'
review=json.loads((base/'s1_review.json').read_text(encoding='utf8'))
manual={ds:json.loads((evidence/f'format_{name}_manual_review.json').read_text(encoding='utf8')) for ds,name in [('webqa_full','webqa'),('okvqa_gs','okvqa')]}
checks=json.loads((base/'verify_real_model_8b.json').read_text(encoding='utf8'))
images=json.loads((base/'image_evidence_check.json').read_text(encoding='utf8'))
text='''# node04 已批准提示词修复与隔离重验报告 — 2026-10-18

本轮实施的是用户明确批准的唯一两条提示词，不自动搜索或迭代其他措辞。最终源码提交 `318ab6c59a9bba7d379e166d119074443bb68892`；提示词实现为 5056ef3，后续提交更新原提示词断言并保护官方评分文件原始字节。完整原始证据、失败及重试日志随本报告提交。

## 结论

'''
text+=f"完整 CPU 回归和原版真实权重图文数值自检通过，两套 S1 均完成 50/50。人工格式判定：WebQA **{manual['webqa_full']['manual_gate']}**；OK-VQA **{manual['okvqa_gs']['manual_gate']}**。具体逐动作理由见各 manual_review.json。不得将数值通过或自动格式 passed 当作全部人工关卡通过。\n"
text+='''
## 修改、环境及研究限制

- 两个批准字符串逐字核对方案全文；WebQA 两行、第一行最短必要答案、第二行完整句；OK-VQA 最短必要词/短语，保留完整名称、数字、单位。全部默认动作共享对应指令。
- `answers.py`、官方评分规则、拒答规则、预算、阈值、split seed 2027、题目切分保持原定义。原有 short 风格和句式消融不是本轮默认指令，不混入新主实验。
- `DEV_DECISIONS.md` 明确披露本次已观察 S1 开发样例，属于开发结果之后的协议修订；不能声称开发集未见，也不通过重跑恢复未见属性。未查看测试答案，没有按 gold 后处理输出。
- node04 两张 L20，本轮只用 GPU1；GPU0 旧 BF16 作业由另一会话负责。
- 远端独立工作树 `/home/user/Futaoran/omniaware_v2_format_20261018`，环境 `/home/user/Futaoran/omniaware_v2_python311/bin/python`；Python 3.11.17 / torch 2.11.0+cu128 / transformers 5.5.3 / CUDA 12.8，FP32、sdpa，env 记录源码干净。
- 从 `/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_resume_lf_1631.sh` 加载已验收环境，覆盖 PYTHONPATH 为新树 src，CUDA_VISIBLE_DEVICES=1，OMP_NUM_THREADS=4，MKL_NUM_THREADS=4，HF_HUB_OFFLINE=1，TRANSFORMERS_OFFLINE=1。
- `OMNIAWARE_RUNS=$PWD/runs/fp32_format_20261018`；`OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_cache_fp32_format_20261018`。8B 模型使用原服务器已验收目录。新提示词哈希隔离 U/SeaKR；不复制旧校准。仅复制不可变 GS 候选文件，原/新 SHA256 均为 a394b1e7bbbbb0db1247c9430ac58bf8a395660b9af893dd2ee8a03c64facede，新副本只读。

## 首轮失败与修复

初始 CPU 运行结果 544 passed、1 failed、1 skipped、5 errors、10 subtests，56.09 秒，退出 1；GPU 尚未开始。一个旧测试仍要求旧提示词中的 single word or phrase；五个错误来自新工作树 core.autocrlf=true 将官方 WebQA 文件转换为 CRLF，导致固定 SHA-256 不匹配。

更新已有提示词契约断言，给两个官方评分源文件添加 -text 检出规则。修复前逐字确认 CRLF→LF 后与固定 Git blob 完全相同，保留原 checkout 字节备份，再恢复固定哈希对应的文件；没有修改官方评分源码或哈希常量，没有修改其他会话的 Git 配置。父目录 validation.exit=1 是历史首轮失败，当前看 retry1/validation.exit。

## 本轮命令和自检结果

完整驱动 `run_format_validation_retry1.sh`：17:22:52（北京时间）启动，使用 flock 和启动标记防止重复运行；任何命令非零退出停止。审核驱动 `run_format_review.sh` 等同一锁释放且主驱动退出 0 后才运行图像/溯源检查。

| 步骤 | 命令 | 已验证结果 |
|---|---|---|
| 依赖 | python -m pip check | 通过 |
| CPU | python -m pytest -p no:warnings --junitxml=.../retry1/pytest.xml | 550 passed、1 skipped、21 subtests passed；49.38 秒，退出 0 |
| 真实权重 | python -m scripts.verify_real_model --model /home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct --dtype float32 --image /home/user/Futaoran/omniaware/assets/data/coco/val2014/COCO_val2014_000000000042.jpg --out "$OMNIAWARE_RUNS/verify_real_model_8b.json" | ok=true；驱动退出成功继续 S1 |
| WebQA S1 | python scripts/v2_runs.py --stage s1 --dataset webqa_full --dtype float32 --n 50 | 50/50；详见自动与人工检查 |
| OK-VQA S1 | python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --dtype float32 --n 50 | 50/50；详见自动与人工检查 |
| 溯源 / 图像 | python audit_format_results.py；python prepare_format_image_review.py | 原始 JSON 和审核退出记录随附 |

CPU 的 1 个 skip 为缺少历史 T15 固定输出，不是当前模型资产缺失。原版真实权重检查采用默认最多 8 步，部分 EOS 提前结束；最大 reference logits 差 0.00008392333984375，最大 attention 差 0.00000011920928955078125，分别小于 0.05 和 0.02。图像模板折叠检查通过，但 image_runs_equal=false、pad 数 234/300 的限制仍保留，不声称两种图片预处理逐 token 相同。

## 逐动作数字

来源为 s1_review.json 的 input_trace_counts 和 format.actions，以及原始 merged.json。拒答率使用现有解析器口径，不能把 semantic abstention 漏识别解释为完全没有弃答。自动格式通过比例排除解析器识别的拒答；OK-VQA 的 answer_first_ok=null，plain 的自动 passed 不替代人工审核。

| 数据集 / 动作 | 可用题 | 空答案数 | 拒答数 | answer_first_ok | template_start | 实际图总数 | 实际视觉 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for ds,v in review.items():
 for action,c in v['input_trace_counts'].items():
  f=v['format']['actions'][action];a=f['answer_first_ok']
  text+=f"| {ds}/{action} | {c['available']} | {c['empty_non_refusal']} | {c['refusal']} | {'N/A' if a is None else format(a,'.6f')} | {f['template_start']:.6f} | {c['images']} | {c['visual_tokens']} |\n"
text+='\n下列时间是 env 写入和合并完成的 UTC 时间，包含准备及生成；生成耗时和峰值显存直接引用 [done]，不是全部墙钟耗时。\n\n'
for ds,v in review.items():
 log=(base/f'v2_s1_{ds}'/f'{ds}.log').read_text(encoding='utf8')
 done=[line for line in log.splitlines() if line.startswith('[done]')]
 text+=f"- {ds}：{v['env'][0]['written_utc']} → {v['completion']['written_utc']}；`{done[-1]}`\n"
text+='\n## 图像和输入溯源\n\n'
for name,s in images['summary'].items():
 text+=f"- {name}：{s['questions_with_image_in_input']}/{s['questions']} 题含图，{s['images']} 张图，CPU 估算 {s['visual_tokens_total']} 个视觉 tokens；gold 入选 {s['gold_image_selected']}/{s['questions']}；打不开 {s['unreadable_images']}，配对错误 {s['images_with_pairing_error']}，ID 错配 {s['images_with_id_mismatch']}。\n"
text+='\nCPU 图像估算和模型实际 token 数不同口径；没有执行逐图语义描述 GPU 探针。图像诊断不是穷尽像素解码验收。\n\n'
for ds,v in review.items():
 text+=f"- {ds}：相同题目集合={v['same_qids_as_previous']}，批准指令完全匹配={v['instruction_matches_approved']}，prompt 哈希变化数={v['changed_prompt_hashes']}，证据/图片差异数={len(v['evidence_image_differences'])}，prompt记录={v['prompt_rows']}行，完成文件哈希校验={v['completion_hashes_match']}。冻结参数逐键比较见 s1_review.json。\n"
text+='''
## 人工格式与解释边界

WebQA 固定前 20 题所有 100 个动作中，机场距离问题 qid d5de89020dba11ecb1e81171463288e9 的 direct 第一行仍为 `Gainesville Regional Airport is closer.`，是整句，也没有回答 how much。因此该数据集人工格式 FAIL，不能用自动通过放行 S2。其他错误实体、必要多词名称不机械判成格式失败；矿物问题的第一行实体与第二行位置内容不对应，作为额外语义/信号解释风险记录，没有改写答案。

'''
text+=f"OK-VQA 人工结论：{manual['okvqa_gs']['manual_gate']}。{manual['okvqa_gs'].get('reason','逐条判断见附件。')}\n"
text+='\n## 原始样例（每套各 4 题，包含 gold 和所有可用动作）\n'
for ds in review:
 rows=[json.loads(x) for x in (base/f'v2_s1_{ds}'/f'{ds}.jsonl').read_text(encoding='utf8').splitlines()]
 for i in [0,2,14,18]:
  r=rows[i];sample={k:r[k] for k in ['qid','question','golds']}
  sample['actions']={k:{f:a.get(f) for f in ['short','answer','refusal']} for k,a in r['actions'].items() if a.get('available')}
  text+='\n```json\n'+json.dumps(sample,ensure_ascii=False,indent=2)+'\n```\n'
text+='\n## 日志尾部\n'
for ds in review:
 text+=f'\n### {ds}\n\n```text\n'+'\n'.join((base/f'v2_s1_{ds}'/f'{ds}.log').read_text(encoding='utf8').splitlines()[-30:])+'\n```\n'
text+='''
## 下一步及未完成事项

这一个批准候选已执行并如实验收；不自动继续改提示词，不通过删词、放宽判据或重写旧答案制造通过。WebQA 停在 S1；其他数据集是否进入 S2 必须依据其独立四项关卡、原协议分支/合并/命令核验要求。S4 未运行且继续禁止。

本报告不提供官方得分、选择器结论或盲标语义准确率。原始数据、模型、旧失败日志及当前结果保留；本地仅接收少量代码/日志/结果，没有恢复已清理的数据副本。原始传输证据的 manifest 对应服务器文件字节；本解释性报告在传输后生成，不属于原始 manifest。
'''
(root/'docs/results/20261018_node04_format_revalidation_report.md').write_text(text,encoding='utf8')
for p in [base,evidence]:(p/'.gitattributes').write_text('* -text\n',encoding='ascii')
print('Report written',len(text))

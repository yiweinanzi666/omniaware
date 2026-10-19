import json, pathlib, hashlib

root = pathlib.Path('D:/WWW2027/data_ops/node04_fp32_code_20261018')
runs = root/'runs/fp32_20261018'
review = json.loads((runs/'s1_review.json').read_text(encoding='utf8'))
manual = {
    'source_commit': 'a11e234fc2741fc122eb0a7bb6b5d7cd6f9f2ee8',
    'scope': 'First 20 saved S1 questions per dataset, every available action; format review only, not blinded semantic annotation.',
    'webqa_full': {'gate': 'FAIL', 'reason': 'Full explanatory sentences remain in short fields.',
        'examples': [{'qid': 'd5d034e20dba11ecb1e81171463288e9', 'action': 'static_image'},
                     {'qid': 'd5c50c840dba11ecb1e81171463288e9', 'action': 'direct'},
                     {'qid': 'd5de89020dba11ecb1e81171463288e9', 'action': 'direct'}]},
    'okvqa_gs': {'gate': 'NOT_CLEARLY_PASSED', 'reason': 'All sampled outputs are words/phrases, but the stricter minimal-answer requirement is not clearly satisfied: rerank_static gives in a bakery rather than the bare location. This exact pattern is documented in the repository as an exact-match concern. Incorrect entities alone are not format failures.',
        'examples': [{'qid': '2022015', 'action': 'rerank_static', 'answer': 'in a bakery'}],
        'parser_limitation': 'For qid2022015 static_text/static_mixed say not specified but refusal=false. Reported zero refusal is parser-defined, not absence of all semantic abstentions.'},
    'decision': 'No new FP32 S2 on either dataset; no S4. Frozen prompts, parsing, thresholds and answers remain unchanged.'
}
(runs/'manual_format_review.json').write_text(json.dumps(manual,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

text = '''# node04 FP32 修复与 S1 重验报告 — 2026-10-18

结论：已批准的精度修复完成，完整 CPU 测试、原版真实权重数值自检、多图显存探针和图像输入诊断通过；全部 S1 关卡尚未通过。WebQA 人工短答案格式失败；OK-VQA 虽然是词/短语输出，但仍有仓库明确指出的额外措辞，严格格式要求不能认定全部通过。两数据集新的 FP32 S2 均停在关卡前，未运行 S4。

## 环境与修改

- 节点 node04，2 张 NVIDIA L20。本会话串行使用 GPU1；GPU0 的旧 BF16 作业由另一会话负责。
- 源码提交 `a11e234fc2741fc122eb0a7bb6b5d7cd6f9f2ee8`，源码分支 `fix/node04-explicit-dtype-20261018`；实际运行 env 记录 `git_dirty=false`、`torch.float32`、`sdpa`。
- Python 3.11.17，torch 2.11.0+cu128，transformers 5.5.3，CUDA 12.8，numpy 2.4.6。
- 远端工作树 `/home/user/Futaoran/omniaware_v2_selfcheck_20261018`；解释器 `/home/user/Futaoran/omniaware_v2_python311/bin/python`。
- 环境从 `/home/user/Futaoran/omniaware/assets/manifests/v2_run_20261017/env_resume_lf_1631.sh` 加载，覆盖 `CUDA_VISIBLE_DEVICES=1`、`PYTHONPATH=$PWD/src`、`OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- `OMNIAWARE_RUNS=$PWD/runs/fp32_20261018`；`OMNIAWARE_CACHE=/home/user/Futaoran/omniaware_v2_cache_fp32_20261018`。8B 模型为 `/home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct`。
- `run.py` 与 `v2_runs.py` 显式传递 dtype，核对模型实际精度；U/SeaKR 缓存指纹及文件名包含 dtype；不同精度拒绝续写同一答案目录。使用独立缓存、运行与决定目录。新增回归覆盖精度传递、缓存隔离、续跑拒绝、实际 dtype 错配和各阶段命令传递。
- 提示词、解析规则、logits 0.05/attention 0.02 阈值、切分种子 2027、题目划分、图像预算 262144 和生成预算均未改变。旧 BF16 失败记录与资产保留。

## 执行与退出状态

完整启动脚本、命令和日志保存在 `docs/results/node04_fp32_validation_20261018/`，下列路径均相对该证据目录。命令在上述工作树和环境中执行。

| 步骤 | 命令 / 原始记录 | 结果 |
|---|---|---|
| CPU 全量 | `cpu_fp32_revision.sh`，`cpu_final/pytest.log`、`pytest.exit` | 550 passed、1 skipped、21 subtests passed，44.53 秒，退出 0；pip check 通过 |
| 真实权重 | `python -m scripts.verify_real_model --model /home/user/Futaoran/omniaware/assets/cache/qwen3-vl-8b-instruct --dtype float32 --image /home/user/Futaoran/omniaware/assets/data/coco/val2014/COCO_val2014_000000000042.jpg --out "$OMNIAWARE_RUNS/verify_real_model_8b.json"` | 原版检查全部通过，驱动继续下一步 |
| 多图 | `python /home/user/Futaoran/omniaware_selfcheck_logs_20261018/probe_fp32_multimage.py` | ok=true，驱动继续下一步 |
| WebQA S1 | `python scripts/v2_runs.py --stage s1 --dataset webqa_full --dtype float32 --n 50` | 50/50，生成 555.8 秒，自动格式退出 0；人工格式 FAIL |
| OK-VQA S1 重试 | `python scripts/v2_runs.py --stage s1 --dataset okvqa_gs --dtype float32 --n 50` | 50/50，生成 118.1 秒，自动格式退出 0；严格人工格式未明确通过 |
| 图像诊断 | `python /home/user/Futaoran/omniaware_selfcheck_logs_20261018/prepare_image_review.py` | 新缓存补齐诊断用 50 题相关度后调用未修改的 image_evidence_check；完成，重试驱动 `okvqa_retry.exit=0` |

CPU 首次运行 549 passed / 1 failed 的失败来自新增测试对 argparse SystemExit 的断言方式；运行时确实拒绝了精度不匹配。修正测试为检查退出码 2 和 stderr 后完整重跑，以上为最终结果。原失败日志保留，不覆盖。

首轮驱动 `validation.exit=1` 是 OK-VQA 缺少独立缓存根下 GS112K 候选文件的历史失败。GS112K 是不可变数据输入，并非 BF16 校准缓存。从服务器已验收目录复制到新根 `okvqa_gs112k/candidates_full.jsonl`，新副本只读，两端 SHA-256 均为 `a394b1e7bbbbb0db1247c9430ac58bf8a395660b9af893dd2ee8a03c64facede`。原失败日志 `s1_okvqa.log` 保留，重试日志另存 `s1_okvqa_retry.log`。未重新下载资产或恢复本地大数据。

各阶段开始/完成时间：驱动启动记录见 `validation.started`、`okvqa_retry.started`；下表是模型 env 写入和合并完成的 UTC 时间，包含加载、准备及生成，不把生成计时误当总墙钟耗时。

| 数据集 | env 写入 UTC | merged 完成 UTC |
|---|---|---|
'''
for ds,v in review.items():
    text+=f"| {ds} | {v['env'][0]['written_utc']} | {v['completion']['written_utc']} |\n"
text+='''
## 自检、显存与图像

真实权重文本和图片检查最大 reference logits 差为 0.00013399124145507812，最大 attention 差为 0.000001439359039068222，均在冻结阈值内。默认最多 8 个生成步骤，部分臂因 EOS 提前结束。`image_runs_equal=false` 仍被保留：两种模板路径 image pad 数为 234/300；折叠后的模板一致不表示图片预处理逐 token 相同。详见 `runs/fp32_20261018/verify_real_model_8b.json`，不能据此宣布 BF16 通过。

实际最长多图 S1 prompt 探针：qid `d5c8e0de0dba11ecb1e81171463288e9`、static_mixed、5 张图、1706 prompt tokens，输入哈希完全匹配。保持 262144 像素 / 384 max_new_tokens，生成 23 tokens，17.73 秒，峰值 allocated 34.208 GiB / reserved 34.424 GiB。S1 WebQA / OK-VQA 峰值分别 34.28 / 34.10 GiB（各自日志 `[done]`）。这是本轮样本测量，不是所有后续样本显存上界。

图像 CPU 检查覆盖 50 道 WebQA 图像题：static_image 250 张、估算 78388 visual tokens、gold 入选 50/50；static_mixed 167 张、估算 52375 tokens、gold 入选 48/50。两动作各 50/50 题都有图，打不开、标题/说明配对错误、ID 错配均 0。249/167 张原图超预算，全部缩放到预算内。此脚本检查可打开和输入构造，不是穷尽像素解码验收或逐图语义标注；`gpu_probe_executed=false`。本轮实际模型输入的 token 数另见下一表，不能和 CPU 估算混用。

## S1 逐动作关键数字

来源：`runs/fp32_20261018/s1_review.json` 的 `input_trace_counts`、`format.actions`，以及各数据集 `.merged.json.empty_by_arm`。空答案不含解析器单独记录的拒答。格式通过比例以非拒答项计算，以原工具字段为准；OK-VQA plain 格式的 answer_first_ok 为 null，不能把自动 passed=true 当人工审核。

| 数据集 / 动作 | 可用题数 | 空答案率 | 拒答数/可用题 | answer_first_ok | template_start | 实际图片总数 | 实际 visual tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for ds,v in review.items():
    for action,c in v['input_trace_counts'].items():
        f=v['format']['actions'][action];first=f['answer_first_ok']
        text+=f"| {ds}/{action} | {c['available']} | 0 | {c['refusal']}/{c['available']} | {'N/A' if first is None else format(first,'.6f')} | {f['template_start']:.6f} | {c['images']} | {c['visual_tokens']} |\n"
text+='''
WebQA 全部臂合并口径：549 个答案、空答案 0、拒答 4（0.7286%）。OK-VQA 合并口径：500 个答案、空答案和解析器记录的拒答均为 0。WebQA static_image 的 22 道图像题全部有正图片数及正视觉 token。OK-VQA 的四个可用动作各 50 题，每题有 1 张问题图片。WebQA direct 不带图。

## 人工格式关卡与溯源

检查两个数据集各前 20 题的所有可用动作。WebQA `short` 仍含整句解释，例如 quadrireme 船体建造技术问题 static_image 写出整段 frame-first construction 说明；花朵问题 direct 以 “There is no single purple flower ...” 开头。自动两行格式通过不足以满足短答案人工要求，明确 FAIL。

OK-VQA 抽样回答都是词/短语，没有把一般答错的实体自动判成格式失败；但 qid2022015 rerank_static 输出 `in a bakery`。仓库 `docs/reports/Omni-Aware_根因溯源报告_20261013.md` 和 HANDOFF 第 2 步第 2 条明确关注此类多余措辞导致的精确匹配风险。因此严格格式要求不认定明确通过，按原任务“任何一条不通过或判断不了则停止该数据集”执行。该题 static_text/static_mixed 的 `not specified` 被现有解析器标为 refusal=false，是本次报告的解析口径限制；没有改解析器。未运行本轮官方评分，不在此捏造分数。

两个 S1 题目集合、所有可用动作 prompt 哈希、冻结提示词和哈希、生成预算、像素预算、calib/dev/test 三份 split_qids_sha256 均与原输入记录一致；只核对输入一致性，不拿旧实验成绩比较。最初诊断使用了不存在的 calibration_qids_sha256 键，最终脚本已改用实际 split_qids_sha256；不以 null==null 充当验证。每套 prompt 文件 20 行；topk 候选均含 text；日志有 [env]/[done]；环境源提交干净且精度正确。原始前 20 题输出保存在 s1_review.json，人工决定另存 manual_format_review.json。

## 八个原始样例

下面每个样例包括 gold 和所有可用动作的原始 short/answer；只用于格式说明，未进行 S2 盲标。
'''
for ds,indices in [('webqa_full',[0,3,5,7]),('okvqa_gs',[0,3,10,18])]:
    rows=[json.loads(x) for x in (runs/f'v2_s1_{ds}'/f'{ds}.jsonl').read_text(encoding='utf8').splitlines()]
    for i in indices:
        q=rows[i]
        sample={k:q[k] for k in ['qid','question','golds']}
        sample['actions']={k:{f:a.get(f) for f in ['short','answer']} for k,a in q['actions'].items() if a.get('available')}
        text+=f"\n### {ds} / {q['qid']}\n\n```json\n{json.dumps(sample,ensure_ascii=False,indent=2)}\n```\n"
text+='\n## 停止时的日志尾部\n'
for ds in review:
    tail='\n'.join((runs/f'v2_s1_{ds}'/f'{ds}.log').read_text(encoding='utf8').splitlines()[-30:])
    text+=f'\n### {ds}（生成成功，人工关卡未放行）\n\n```text\n{tail}\n```\n'
text+='''
## 事实、限制与下一步

事实：精度实现与数值自检通过；S1 生成和输入溯源检查完成；人工格式仍阻塞。550 个 CPU 用例通过，1 个 skip 是缺少历史 T15 固定输出，不是当前 8B 权重缺失。初始失败和全部重试记录均保留。

推测：剩余问题可能来自冻结指令对最短答案约束不足或模型输出习惯；精度通过本身不能区分二者。验证需要另行提出并获准修改提示词/协议，采用独立运行与缓存重新做完整关卡；不能后处理重写当前答案或放宽格式判据。

未完成/未验证：全部 S1 关卡、逐图语义 GPU 描述、S2 官方评分/选择器/盲标、S4。本轮没有启动新的 S2；另一会话旧 BF16 作业不属于本报告的 FP32 验收。

建议：保留本次结果为失败证据，先审定具体格式问题及独立的提示词修订方案，再重新自检/S1，并重新校准 S2。本次已批准的精度方案不包含冻结提示词或解析规则变更，不能自行扩展。S4 继续禁止。

证据从服务器下载约 2.63 MiB，仅代码/日志/结果，无模型和数据集。MANIFEST.json 的 45 个服务器原始文件已逐个 SHA-256 校验；本报告和 manual_format_review.json 是随后生成的解释性文件，不在该原始传输 manifest 内。
'''
(root/'docs/results/20261018_node04_fp32_selfcheck_report.md').write_text(text,encoding='utf8')
for directory in [runs,root/'docs/results/node04_fp32_validation_20261018']:
    (directory/'.gitattributes').write_text('* -text\n',encoding='ascii')
print('Wrote report and manual review:',len(text),'characters')

# node04 已批准提示词修复与隔离重验报告 — 2026-10-18

本轮实施的是用户明确批准的唯一两条提示词，不自动搜索或迭代其他措辞。最终源码提交 `318ab6c59a9bba7d379e166d119074443bb68892`；提示词实现为 5056ef3，后续提交更新原提示词断言并保护官方评分文件原始字节。完整原始证据、失败及重试日志随本报告提交。

## 结论

完整 CPU 回归和原版真实权重图文数值自检通过，两套 S1 均完成 50/50。人工格式判定：WebQA **FAIL**（前 20 题 100 个动作中 1 个第一行仍为整句）；OK-VQA **FAIL**（前 20 题 80 个动作中 2 个保留多余引导措辞）。本轮两个数据集均停止在 S1，没有启动新 S2。具体逐动作理由见各 manual_review.json。不得将数值通过或自动格式 passed 当作全部人工关卡通过。

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
| webqa_full/direct | 50 | 0 | 0 | 1.000000 | 0.100000 | 0 | 0 |
| webqa_full/static_text | 50 | 0 | 1 | 1.000000 | 0.220000 | 0 | 0 |
| webqa_full/static_image | 49 | 0 | 0 | 1.000000 | 0.183673 | 242 | 55802 |
| webqa_full/static_mixed | 50 | 0 | 0 | 1.000000 | 0.160000 | 128 | 29487 |
| webqa_full/rerank_static | 50 | 0 | 0 | 1.000000 | 0.160000 | 44 | 10344 |
| okvqa_gs/direct | 50 | 0 | 0 | N/A | 0.040000 | 50 | 11568 |
| okvqa_gs/static_text | 50 | 0 | 0 | N/A | 0.080000 | 50 | 11568 |
| okvqa_gs/static_mixed | 50 | 0 | 0 | N/A | 0.080000 | 50 | 11568 |
| okvqa_gs/rerank_static | 50 | 0 | 0 | N/A | 0.060000 | 50 | 11568 |

下列时间是 env 写入和合并完成的 UTC 时间，包含准备及生成；生成耗时和峰值显存直接引用 [done]，不是全部墙钟耗时。

- webqa_full：2026-10-18T09:37:28.376228+00:00 → 2026-10-18T09:56:31.204113+00:00；`[done] webqa_full 50 in 545.6s -> /home/user/Futaoran/omniaware_v2_format_20261018/runs/fp32_format_20261018/v2_s1_webqa_full/webqa_full.jsonl; empty_answer_rate 0.000000; refusal_rate 0.001821; peak_gpu_memory_gib 34.28`
- okvqa_gs：2026-10-18T09:59:20.911226+00:00 → 2026-10-18T10:11:35.879114+00:00；`[done] okvqa_gs 50 in 125.9s -> /home/user/Futaoran/omniaware_v2_format_20261018/runs/fp32_format_20261018/v2_s1_okvqa_gs/okvqa_gs.jsonl; empty_answer_rate 0.000000; refusal_rate 0.000000; peak_gpu_memory_gib 34.10`

## 图像和输入溯源

- static_image：50/50 题含图，250 张图，CPU 估算 78388 个视觉 tokens；gold 入选 50/50；打不开 0，配对错误 0，ID 错配 0。
- static_mixed：50/50 题含图，167 张图，CPU 估算 52375 个视觉 tokens；gold 入选 48/50；打不开 0，配对错误 0，ID 错配 0。

CPU 图像估算和模型实际 token 数不同口径；没有执行逐图语义描述 GPU 探针。图像诊断不是穷尽像素解码验收。

- webqa_full：相同题目集合=True，批准指令完全匹配=True，prompt 哈希变化数=249，证据/图片差异数=0，prompt记录=20行，完成文件哈希校验={'output': True, 'config': True}。冻结参数逐键比较见 s1_review.json。
- okvqa_gs：相同题目集合=True，批准指令完全匹配=True，prompt 哈希变化数=200，证据/图片差异数=0，prompt记录=20行，完成文件哈希校验={'output': True, 'config': True}。冻结参数逐键比较见 s1_review.json。

## 人工格式与解释边界

WebQA 固定前 20 题所有 100 个动作中，机场距离问题 qid d5de89020dba11ecb1e81171463288e9 的 direct 第一行仍为 `Gainesville Regional Airport is closer.`，是整句，也没有回答 how much。因此该数据集人工格式 FAIL，不能用自动通过放行 S2。其他错误实体、必要多词名称不机械判成格式失败；矿物问题的第一行实体与第二行位置内容不对应，作为额外语义/信号解释风险记录，没有改写答案。

OK-VQA 人工结论：FAIL。qid2022015 static_text and static_mixed both output at a donut shop; unnecessary introductory at a remains. Complete names and meaningful multiword phrases are retained rather than truncated.

## 原始样例（每套各 4 题，包含 gold 和所有可用动作）

```json
{
  "qid": "d5d3d7f00dba11ecb1e81171463288e9",
  "question": "Which came first, the \"Red Fox Battallion\", or the British military officer later known as the \"Red Fox\"?",
  "golds": [
    "The \"Red Fox\" Battalion came into being in Sep.1917. Michael Sinclaire the \"Red Fox\" was born 5 months later."
  ],
  "actions": {
    "direct": {
      "short": "Red Fox Battalion",
      "answer": "The Red Fox Battalion came first, predating the British officer later known as the \"Red Fox.\"",
      "refusal": false
    },
    "static_text": {
      "short": "Red Fox Battalion",
      "answer": "The \"Red Fox Battalion\" came first, as it was formed on 22 September 1917, predating Lieutenant Albert Michael Sinclair, known as the \"Red Fox,\" who was a prisoner during World War II.",
      "refusal": false
    },
    "static_image": {
      "short": "British military officer",
      "answer": "The British military officer later known as the \"Red Fox\" came first, as there is no historical record of a \"Red Fox Battalion\" in military history.",
      "refusal": false
    },
    "static_mixed": {
      "short": "British military officer",
      "answer": "The British military officer, Lieutenant Albert Michael Sinclair, known as the \"Red Fox,\" predates the \"Red Fox Battalion\" by several decades.",
      "refusal": false
    },
    "rerank_static": {
      "short": "Red Fox Battalion",
      "answer": "The \"Red Fox Battalion\" came first, established on 22 September 1917, predating the British officer known as \"Red Fox\" who was active during World War II.",
      "refusal": false
    }
  }
}
```

```json
{
  "qid": "d5d034e20dba11ecb1e81171463288e9",
  "question": "What type of ship building technique might have been used to make quadriremes?",
  "golds": [
    "Frame-first construction"
  ],
  "actions": {
    "direct": {
      "short": "Clinker-built",
      "answer": "Quadriremes were likely constructed using clinker-built shipbuilding techniques, where overlapping planks are fastened together.",
      "refusal": false
    },
    "static_text": {
      "short": "Frame-first construction",
      "answer": "Quadriremes were likely built using frame-first construction, a technique originating from Mediterranean shipbuilding traditions.",
      "refusal": false
    },
    "static_image": {
      "short": "Keel and frame construction",
      "answer": "Quadriremes were built using keel and frame construction, a technique that provided structural integrity for large, oared warships.",
      "refusal": false
    },
    "static_mixed": {
      "short": "Frame-first construction",
      "answer": "Quadriremes, like other ancient warships, were likely built using frame-first construction, where the internal framework is assembled before attaching the planks.",
      "refusal": false
    },
    "rerank_static": {
      "short": "Frame-first construction",
      "answer": "Quadriremes were likely built using frame-first construction, a technique involving laying down the vessel’s framework before attaching planks.",
      "refusal": false
    }
  }
}
```

```json
{
  "qid": "d5c5b92c0dba11ecb1e81171463288e9",
  "question": "Is Lavenham High Street a one-way street or is it open to two-way traffic?",
  "golds": [
    "Lavenham High Street is a one-way street."
  ],
  "actions": {
    "direct": {
      "short": "Two-way",
      "answer": "Lavenham High Street is open to two-way traffic.",
      "refusal": false
    },
    "static_text": {
      "short": "Not specified",
      "answer": "The provided evidence does not indicate whether Lavenham High Street is one-way or open to two-way traffic.",
      "refusal": false
    },
    "static_image": {
      "short": "Two-way",
      "answer": "Lavenham High Street is open to two-way traffic.",
      "refusal": false
    },
    "static_mixed": {
      "short": "Two-way",
      "answer": "Lavenham High Street is open to two-way traffic.",
      "refusal": false
    },
    "rerank_static": {
      "short": "Two-way traffic",
      "answer": "Lavenham High Street is open to two-way traffic.",
      "refusal": false
    }
  }
}
```

```json
{
  "qid": "d5de89020dba11ecb1e81171463288e9",
  "question": "How much closer is Gainesville Regional Airport to Gainesville than Winchester Airport is to Winchester, Ontario, Canada?",
  "golds": [
    "It is .5 miles closer."
  ],
  "actions": {
    "direct": {
      "short": "Gainesville Regional Airport is closer.",
      "answer": "Gainesville Regional Airport is closer to Gainesville than Winchester Airport is to Winchester, Ontario, Canada.",
      "refusal": false
    },
    "static_text": {
      "short": "3 miles closer",
      "answer": "Gainesville Regional Airport is 3 miles closer to Gainesville than Winchester Airport is to Winchester, Ontario, Canada.",
      "refusal": false
    },
    "static_image": {
      "short": "Not applicable",
      "answer": "Gainesville Regional Airport is not in Ontario, Canada, and there is no Winchester Airport in Ontario, so the question cannot be answered with a distance.",
      "refusal": false
    },
    "static_mixed": {
      "short": "3 miles closer",
      "answer": "Gainesville Regional Airport is 3 miles closer to Gainesville than Winchester Airport is to Winchester, Ontario, Canada.",
      "refusal": false
    },
    "rerank_static": {
      "short": "3 miles",
      "answer": "Gainesville Regional Airport is 3 miles closer to Gainesville than Winchester Airport is to Winchester, Ontario, Canada.",
      "refusal": false
    }
  }
}
```

```json
{
  "qid": "3296875",
  "question": "What is the type of dog on the woman's lap?",
  "golds": [
    "cocker spaniel",
    "cocker spaniel",
    "cocker spaniel",
    "cocker spaniel",
    "cocker spaniel",
    "cocker spaniel",
    "poodle",
    "poodle",
    "sharpa",
    "sharpa"
  ],
  "actions": {
    "direct": {
      "short": "cavalier king charles spaniel",
      "answer": "cavalier king charles spaniel",
      "refusal": false
    },
    "static_text": {
      "short": "Cavalier King Charles Spaniel",
      "answer": "Cavalier King Charles Spaniel",
      "refusal": false
    },
    "static_mixed": {
      "short": "Cavalier King Charles Spaniel",
      "answer": "Cavalier King Charles Spaniel",
      "refusal": false
    },
    "rerank_static": {
      "short": "bichon frise",
      "answer": "bichon frise",
      "refusal": false
    }
  }
}
```

```json
{
  "qid": "5789675",
  "question": "How fast can this train go?",
  "golds": [
    "100 mph",
    "100 mph",
    "100 mph",
    "100 mph",
    "75mph",
    "75mph",
    "200mph",
    "200mph",
    "fast",
    "fast"
  ],
  "actions": {
    "direct": {
      "short": "250 mph",
      "answer": "250 mph",
      "refusal": false
    },
    "static_text": {
      "short": "311 mph",
      "answer": "311 mph",
      "refusal": false
    },
    "static_mixed": {
      "short": "311 mph",
      "answer": "311 mph",
      "refusal": false
    },
    "rerank_static": {
      "short": "300 mph",
      "answer": "300 mph",
      "refusal": false
    }
  }
}
```

```json
{
  "qid": "4812675",
  "question": "What activity is the man in the canoe about to do?",
  "golds": [
    "row",
    "row",
    "row",
    "row",
    "row",
    "row",
    "fish",
    "fish",
    "fish",
    "fish"
  ],
  "actions": {
    "direct": {
      "short": "row",
      "answer": "row",
      "refusal": false
    },
    "static_text": {
      "short": "canoeing",
      "answer": "canoeing",
      "refusal": false
    },
    "static_mixed": {
      "short": "canoeing",
      "answer": "canoeing",
      "refusal": false
    },
    "rerank_static": {
      "short": "paddle",
      "answer": "paddle",
      "refusal": false
    }
  }
}
```

```json
{
  "qid": "2022015",
  "question": "Where are these donuts being made?",
  "golds": [
    "bakery",
    "bakery",
    "bakery",
    "bakery",
    "bakery",
    "bakery",
    "dunkin donuts",
    "dunkin donuts",
    "ukraine",
    "ukraine"
  ],
  "actions": {
    "direct": {
      "short": "bakery",
      "answer": "bakery",
      "refusal": false
    },
    "static_text": {
      "short": "at a donut shop",
      "answer": "at a donut shop",
      "refusal": false
    },
    "static_mixed": {
      "short": "at a donut shop",
      "answer": "at a donut shop",
      "refusal": false
    },
    "rerank_static": {
      "short": "on premises",
      "answer": "on premises",
      "refusal": false
    }
  }
}
```

## 日志尾部

### webqa_full

```text
[22/50] d5c638980dba 11.8s P=900 U_pre=0.24 retrieve=False
[23/50] d5ce8c460dba 10.77s P=889 U_pre=1.00 retrieve=True
[24/50] d5d8972c0dba 14.76s P=581 U_pre=0.21 retrieve=False
[25/50] d5daefae0dba 10.98s P=1011 U_pre=0.32 retrieve=False
[26/50] d5cae5aa0dba 10.1s P=1025 U_pre=1.00 retrieve=True
[27/50] d5d7ef5c0dba 11.3s P=1069 U_pre=0.78 retrieve=True
[28/50] d5c703c20dba 9.67s P=788 U_pre=0.38 retrieve=True
[29/50] d5df37ee0dba 12.14s P=1079 U_pre=0.47 retrieve=True
[30/50] d5cd7c3e0dba 13.17s P=897 U_pre=0.21 retrieve=False
[31/50] d5d880f20dba 12.9s P=1029 U_pre=0.20 retrieve=False
[32/50] d5d0bb9c0dba 9.98s P=504 U_pre=0.20 retrieve=False
[33/50] d5db372a0dba 9.11s P=1096 U_pre=0.22 retrieve=False
[34/50] d5d6773a0dba 14.41s P=1117 U_pre=0.22 retrieve=False
[35/50] d5c569e00dba 10.14s P=880 U_pre=0.79 retrieve=True
[36/50] d5c8e0de0dba 9.17s P=1766 U_pre=1.00 retrieve=True
[37/50] d5bfc29c0dba 8.97s P=1507 U_pre=0.29 retrieve=False
[38/50] d5d9c4b20dba 12.83s P=815 U_pre=0.20 retrieve=False
[39/50] d5c747560dba 15.32s P=1084 U_pre=0.25 retrieve=False
[40/50] d5d7d0080dba 12.43s P=653 U_pre=1.00 retrieve=True
[41/50] d5d5b9120dba 5.87s P=412 U_pre=1.00 retrieve=True
[42/50] d5df295c0dba 12.95s P=1289 U_pre=0.20 retrieve=False
[43/50] d5d562aa0dba 10.05s P=832 U_pre=1.00 retrieve=True
[44/50] d5d44fa00dba 11.96s P=708 U_pre=1.00 retrieve=True
[45/50] d5c981a60dba 8.55s P=1520 U_pre=0.64 retrieve=True
[46/50] d5d318420dba 6.1s P=495 U_pre=0.41 retrieve=True
[47/50] d5cd51960dba 8.76s P=714 U_pre=1.00 retrieve=True
[48/50] d5c987320dba 9.65s P=1534 U_pre=0.89 retrieve=True
[49/50] d5d106560dba 12.46s P=1074 U_pre=1.00 retrieve=True
[50/50] d5d5b9da0dba 9.79s P=934 U_pre=0.25 retrieve=False
[done] webqa_full 50 in 545.6s -> /home/user/Futaoran/omniaware_v2_format_20261018/runs/fp32_format_20261018/v2_s1_webqa_full/webqa_full.jsonl; empty_answer_rate 0.000000; refusal_rate 0.001821; peak_gpu_memory_gib 34.28
```

### okvqa_gs

```text
[22/50] 2910485 2.07s P=1006 U_pre=0.21 retrieve=False
[23/50] 5100805 1.84s P=547 U_pre=0.55 retrieve=True
[24/50] 760295 2.49s P=1048 U_pre=0.01 retrieve=False
[25/50] 1538325 2.73s P=1053 U_pre=0.47 retrieve=True
[26/50] 4222125 2.12s P=681 U_pre=0.59 retrieve=True
[27/50] 1744575 2.2s P=811 U_pre=0.52 retrieve=True
[28/50] 4741645 1.74s P=820 U_pre=0.13 retrieve=False
[29/50] 5813945 1.64s P=604 U_pre=1.00 retrieve=True
[30/50] 802195 1.81s P=746 U_pre=1.00 retrieve=True
[31/50] 2600205 1.91s P=851 U_pre=0.78 retrieve=True
[32/50] 1413345 1.93s P=616 U_pre=0.54 retrieve=True
[33/50] 780325 1.96s P=621 U_pre=0.43 retrieve=True
[34/50] 2517525 4.16s P=1126 U_pre=0.10 retrieve=False
[35/50] 4246685 1.76s P=630 U_pre=0.22 retrieve=False
[36/50] 2804135 1.73s P=520 U_pre=0.00 retrieve=False
[37/50] 1576015 2.25s P=976 U_pre=0.62 retrieve=True
[38/50] 4751845 3.69s P=638 U_pre=0.52 retrieve=True
[39/50] 4273765 1.97s P=825 U_pre=0.76 retrieve=True
[40/50] 2689665 2.38s P=793 U_pre=0.77 retrieve=True
[41/50] 2907005 2.11s P=707 U_pre=0.70 retrieve=True
[42/50] 2248505 2.08s P=758 U_pre=0.69 retrieve=True
[43/50] 3294945 3.33s P=1114 U_pre=0.00 retrieve=False
[44/50] 5348455 1.92s P=749 U_pre=0.50 retrieve=True
[45/50] 5197165 2.23s P=956 U_pre=0.59 retrieve=True
[46/50] 3686485 2.0s P=889 U_pre=0.00 retrieve=False
[47/50] 5603325 1.73s P=521 U_pre=1.00 retrieve=True
[48/50] 2608085 2.79s P=1032 U_pre=0.69 retrieve=True
[49/50] 2162545 2.4s P=832 U_pre=0.66 retrieve=True
[50/50] 3249795 1.76s P=584 U_pre=0.41 retrieve=True
[done] okvqa_gs 50 in 125.9s -> /home/user/Futaoran/omniaware_v2_format_20261018/runs/fp32_format_20261018/v2_s1_okvqa_gs/okvqa_gs.jsonl; empty_answer_rate 0.000000; refusal_rate 0.000000; peak_gpu_memory_gib 34.10
```

## 下一步及未完成事项

这一个批准候选已执行并如实验收；不自动继续改提示词，不通过删词、放宽判据或重写旧答案制造通过。WebQA 停在 S1；其他数据集是否进入 S2 必须依据其独立四项关卡、原协议分支/合并/命令核验要求。S4 未运行且继续禁止。

本报告不提供官方得分、选择器结论或盲标语义准确率。原始数据、模型、旧失败日志及当前结果保留；本地仅接收少量代码/日志/结果，没有恢复已清理的数据副本。原始传输证据的 manifest 对应服务器文件字节；本解释性报告在传输后生成，不属于原始 manifest。

# node04 FP32 修复与 S1 重验报告 — 2026-10-18

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
| webqa_full | 2026-10-18T05:13:26.940225+00:00 | 2026-10-18T05:31:52.627144+00:00 |
| okvqa_gs | 2026-10-18T05:38:33.060210+00:00 | 2026-10-18T05:49:25.119110+00:00 |

## 自检、显存与图像

真实权重文本和图片检查最大 reference logits 差为 0.00013399124145507812，最大 attention 差为 0.000001439359039068222，均在冻结阈值内。默认最多 8 个生成步骤，部分臂因 EOS 提前结束。`image_runs_equal=false` 仍被保留：两种模板路径 image pad 数为 234/300；折叠后的模板一致不表示图片预处理逐 token 相同。详见 `runs/fp32_20261018/verify_real_model_8b.json`，不能据此宣布 BF16 通过。

实际最长多图 S1 prompt 探针：qid `d5c8e0de0dba11ecb1e81171463288e9`、static_mixed、5 张图、1706 prompt tokens，输入哈希完全匹配。保持 262144 像素 / 384 max_new_tokens，生成 23 tokens，17.73 秒，峰值 allocated 34.208 GiB / reserved 34.424 GiB。S1 WebQA / OK-VQA 峰值分别 34.28 / 34.10 GiB（各自日志 `[done]`）。这是本轮样本测量，不是所有后续样本显存上界。

图像 CPU 检查覆盖 50 道 WebQA 图像题：static_image 250 张、估算 78388 visual tokens、gold 入选 50/50；static_mixed 167 张、估算 52375 tokens、gold 入选 48/50。两动作各 50/50 题都有图，打不开、标题/说明配对错误、ID 错配均 0。249/167 张原图超预算，全部缩放到预算内。此脚本检查可打开和输入构造，不是穷尽像素解码验收或逐图语义标注；`gpu_probe_executed=false`。本轮实际模型输入的 token 数另见下一表，不能和 CPU 估算混用。

## S1 逐动作关键数字

来源：`runs/fp32_20261018/s1_review.json` 的 `input_trace_counts`、`format.actions`，以及各数据集 `.merged.json.empty_by_arm`。空答案不含解析器单独记录的拒答。格式通过比例以非拒答项计算，以原工具字段为准；OK-VQA plain 格式的 answer_first_ok 为 null，不能把自动 passed=true 当人工审核。

| 数据集 / 动作 | 可用题数 | 空答案率 | 拒答数/可用题 | answer_first_ok | template_start | 实际图片总数 | 实际 visual tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| webqa_full/direct | 50 | 0 | 0/50 | 0.940000 | 0.240000 | 0 | 0 |
| webqa_full/static_text | 50 | 0 | 2/50 | 0.958333 | 0.320000 | 0 | 0 |
| webqa_full/static_image | 49 | 0 | 1/49 | 0.979167 | 0.285714 | 242 | 55802 |
| webqa_full/static_mixed | 50 | 0 | 0/50 | 0.980000 | 0.280000 | 128 | 29487 |
| webqa_full/rerank_static | 50 | 0 | 1/50 | 0.979592 | 0.240000 | 44 | 10344 |
| okvqa_gs/direct | 50 | 0 | 0/50 | N/A | 0.060000 | 50 | 11568 |
| okvqa_gs/static_text | 50 | 0 | 0/50 | N/A | 0.100000 | 50 | 11568 |
| okvqa_gs/static_mixed | 50 | 0 | 0/50 | N/A | 0.100000 | 50 | 11568 |
| okvqa_gs/rerank_static | 50 | 0 | 0/50 | N/A | 0.100000 | 50 | 11568 |

WebQA 全部臂合并口径：549 个答案、空答案 0、拒答 4（0.7286%）。OK-VQA 合并口径：500 个答案、空答案和解析器记录的拒答均为 0。WebQA static_image 的 22 道图像题全部有正图片数及正视觉 token。OK-VQA 的四个可用动作各 50 题，每题有 1 张问题图片。WebQA direct 不带图。

## 人工格式关卡与溯源

检查两个数据集各前 20 题的所有可用动作。WebQA `short` 仍含整句解释，例如 quadrireme 船体建造技术问题 static_image 写出整段 frame-first construction 说明；花朵问题 direct 以 “There is no single purple flower ...” 开头。自动两行格式通过不足以满足短答案人工要求，明确 FAIL。

OK-VQA 抽样回答都是词/短语，没有把一般答错的实体自动判成格式失败；但 qid2022015 rerank_static 输出 `in a bakery`。仓库 `docs/reports/Omni-Aware_根因溯源报告_20261013.md` 和 HANDOFF 第 2 步第 2 条明确关注此类多余措辞导致的精确匹配风险。因此严格格式要求不认定明确通过，按原任务“任何一条不通过或判断不了则停止该数据集”执行。该题 static_text/static_mixed 的 `not specified` 被现有解析器标为 refusal=false，是本次报告的解析口径限制；没有改解析器。未运行本轮官方评分，不在此捏造分数。

两个 S1 题目集合、所有可用动作 prompt 哈希、冻结提示词和哈希、生成预算、像素预算、calib/dev/test 三份 split_qids_sha256 均与原输入记录一致；只核对输入一致性，不拿旧实验成绩比较。最初诊断使用了不存在的 calibration_qids_sha256 键，最终脚本已改用实际 split_qids_sha256；不以 null==null 充当验证。每套 prompt 文件 20 行；topk 候选均含 text；日志有 [env]/[done]；环境源提交干净且精度正确。原始前 20 题输出保存在 s1_review.json，人工决定另存 manual_format_review.json。

## 八个原始样例

下面每个样例包括 gold 和所有可用动作的原始 short/answer；只用于格式说明，未进行 S2 盲标。

### webqa_full / d5d3d7f00dba11ecb1e81171463288e9

```json
{
  "qid": "d5d3d7f00dba11ecb1e81171463288e9",
  "question": "Which came first, the \"Red Fox Battallion\", or the British military officer later known as the \"Red Fox\"?",
  "golds": [
    "The \"Red Fox\" Battalion came into being in Sep.1917. Michael Sinclaire the \"Red Fox\" was born 5 months later."
  ],
  "actions": {
    "direct": {
      "short": "The \"Red Fox Battalion\" came first.",
      "answer": "The \"Red Fox Battalion\" predates the British military officer later known as the \"Red Fox.\""
    },
    "static_text": {
      "short": "The \"Red Fox Battalion\" came first.",
      "answer": "The \"Red Fox Battalion\" was formed in 1917, while the British officer known as \"Red Fox\" (Lieutenant Albert Michael Sinclair) was active during World War II, which began in 1939."
    },
    "static_image": {
      "short": "The British military officer later known as the \"Red Fox\"",
      "answer": "The British military officer later known as the \"Red Fox\" came first, as there is no historical evidence of a \"Red Fox Battalion\" existing prior to the 20th century, while the title \"Red Fox\" was famously adopted by British Army officer Sir John \"Red Fox\" Hackett in the 1980s."
    },
    "static_mixed": {
      "short": "The British military officer later known as the \"Red Fox\".",
      "answer": "The British military officer later known as the \"Red Fox\" came first."
    },
    "rerank_static": {
      "short": "The \"Red Fox Battalion\"",
      "answer": "The \"Red Fox Battalion\" came first, as it was formed in 1917, while the British officer known as \"Red Fox\" was active during World War II (1939–1945)."
    }
  }
}
```

### webqa_full / d5db7faa0dba11ecb1e81171463288e9

```json
{
  "qid": "d5db7faa0dba11ecb1e81171463288e9",
  "question": "The mineral composed of iron (II) carbonate and is 48% iron is commonly found where?",
  "golds": [
    "Siderite is commonly found in hydrothermal veins."
  ],
  "actions": {
    "direct": {
      "short": "Siderite",
      "answer": "Siderite, the mineral composed of iron(II) carbonate and containing 48% iron, is commonly found in sedimentary rocks and coal seams."
    },
    "static_text": {
      "short": "Siderite",
      "answer": "The mineral siderite, composed of iron(II) carbonate and containing 48% iron, is commonly found in iron formations and sedimentary rocks as part of iron-bearing deposits."
    },
    "static_image": {
      "short": "Siderite",
      "answer": "Siderite, the mineral composed of iron (II) carbonate and containing 48% iron, is commonly found in sedimentary rocks and iron-rich deposits."
    },
    "static_mixed": {
      "short": "Siderite",
      "answer": "The mineral siderite, composed of iron (II) carbonate and containing 48% iron, is commonly found in iron ore deposits."
    },
    "rerank_static": {
      "short": "Siderite",
      "answer": "The mineral siderite, composed of iron (II) carbonate and containing 48% iron, is commonly found in iron ore deposits."
    }
  }
}
```

### webqa_full / d5db03fe0dba11ecb1e81171463288e9

```json
{
  "qid": "d5db03fe0dba11ecb1e81171463288e9",
  "question": "What part of the Limestone Massif of northwestern Syria are part of the four ethnic mountains of western Syria?",
  "golds": [
    "The Kurd Mountains are part of the four ethnic mountains."
  ],
  "actions": {
    "direct": {
      "short": "The Limestone Massif of northwestern Syria is not part of the four ethnic mountains of western Syria.",
      "answer": "The Limestone Massif of northwestern Syria is not included among the four ethnic mountains of western Syria."
    },
    "static_text": {
      "short": "Kurd Mountains",
      "answer": "The Kurd Mountains are part of the Limestone Massif of northwestern Syria and are one of the four \"ethnic mountains\" of western Syria."
    },
    "static_image": {
      "short": "",
      "answer": ""
    },
    "static_mixed": {
      "short": "Kurd Mountains",
      "answer": "The Kurd Mountains are part of the Limestone Massif of northwestern Syria and are one of the four \"ethnic mountains\" of western Syria."
    },
    "rerank_static": {
      "short": "Kurd Mountains",
      "answer": "The Kurd Mountains are part of the Limestone Massif of northwestern Syria and are one of the four \"ethnic mountains\" of western Syria."
    }
  }
}
```

### webqa_full / d5cc8e640dba11ecb1e81171463288e9

```json
{
  "qid": "d5cc8e640dba11ecb1e81171463288e9",
  "question": "Between an Amanita lividopallescens and a Small pink fungi, which mushroom has the larger cap?",
  "golds": [
    "Amanita lividopallescens has a larger cap than the small pink fungi."
  ],
  "actions": {
    "direct": {
      "short": "Amanita lividopallescens",
      "answer": "The Amanita lividopallescens has the larger cap compared to the Small pink fungi."
    },
    "static_text": {
      "short": "Small pink fungi",
      "answer": "The Small pink fungi has the larger cap."
    },
    "static_image": {
      "short": "Amanita lividopallescens",
      "answer": "The Amanita lividopallescens mushroom has the larger cap compared to the Small pink fungi."
    },
    "static_mixed": {
      "short": "Amanita lividopallescens",
      "answer": "The Amanita lividopallescens mushroom has the larger cap compared to the Small pink fungi."
    },
    "rerank_static": {
      "short": "Amanita lividopallescens",
      "answer": "The Amanita lividopallescens has the larger cap compared to the small pink fungi."
    }
  }
}
```

### okvqa_gs / 3296875

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
      "short": "poodle",
      "answer": "poodle"
    },
    "static_text": {
      "short": "cavalier king charles spaniel",
      "answer": "cavalier king charles spaniel"
    },
    "static_mixed": {
      "short": "cavalier king charles spaniel",
      "answer": "cavalier king charles spaniel"
    },
    "rerank_static": {
      "short": "bichon frise",
      "answer": "bichon frise"
    }
  }
}
```

### okvqa_gs / 1736935

```json
{
  "qid": "1736935",
  "question": "What derby is the horse apart of?",
  "golds": [
    "kentucky",
    "kentucky",
    "kentucky",
    "kentucky",
    "kentucky",
    "kentucky",
    "kentucky",
    "kentucky",
    "horse race",
    "horse race"
  ],
  "actions": {
    "direct": {
      "short": "kentucky",
      "answer": "kentucky"
    },
    "static_text": {
      "short": "kentucky derby",
      "answer": "kentucky derby"
    },
    "static_mixed": {
      "short": "kentucky derby",
      "answer": "kentucky derby"
    },
    "rerank_static": {
      "short": "kentucky derby",
      "answer": "kentucky derby"
    }
  }
}
```

### okvqa_gs / 1546075

```json
{
  "qid": "1546075",
  "question": "What kind of climate do those trees flourish in?",
  "golds": [
    "warm",
    "warm",
    "warm",
    "warm",
    "tropical",
    "tropical",
    "tropical",
    "tropical",
    "temperate",
    "temperate"
  ],
  "actions": {
    "direct": {
      "short": "warm",
      "answer": "warm"
    },
    "static_text": {
      "short": "warm and sunny",
      "answer": "warm and sunny"
    },
    "static_mixed": {
      "short": "warm and sunny",
      "answer": "warm and sunny"
    },
    "rerank_static": {
      "short": "tropical",
      "answer": "tropical"
    }
  }
}
```

### okvqa_gs / 2022015

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
      "answer": "bakery"
    },
    "static_text": {
      "short": "not specified",
      "answer": "not specified"
    },
    "static_mixed": {
      "short": "not specified",
      "answer": "not specified"
    },
    "rerank_static": {
      "short": "in a bakery",
      "answer": "in a bakery"
    }
  }
}
```

## 停止时的日志尾部

### webqa_full（生成成功，人工关卡未放行）

```text
[22/50] d5c638980dba 11.56s P=840 U_pre=0.28 retrieve=False
[23/50] d5ce8c460dba 12.67s P=829 U_pre=0.67 retrieve=True
[24/50] d5d8972c0dba 20.26s P=521 U_pre=0.49 retrieve=True
[25/50] d5daefae0dba 10.93s P=951 U_pre=0.42 retrieve=True
[26/50] d5cae5aa0dba 11.09s P=965 U_pre=0.95 retrieve=True
[27/50] d5d7ef5c0dba 11.23s P=1009 U_pre=0.66 retrieve=True
[28/50] d5c703c20dba 11.21s P=728 U_pre=0.93 retrieve=True
[29/50] d5df37ee0dba 12.32s P=1019 U_pre=0.34 retrieve=False
[30/50] d5cd7c3e0dba 8.12s P=837 U_pre=0.26 retrieve=False
[31/50] d5d880f20dba 12.07s P=969 U_pre=0.25 retrieve=False
[32/50] d5d0bb9c0dba 10.04s P=444 U_pre=0.25 retrieve=False
[33/50] d5db372a0dba 8.38s P=1036 U_pre=0.26 retrieve=False
[34/50] d5d6773a0dba 13.69s P=1057 U_pre=0.25 retrieve=False
[35/50] d5c569e00dba 9.83s P=820 U_pre=0.49 retrieve=True
[36/50] d5c8e0de0dba 8.29s P=1706 U_pre=1.00 retrieve=True
[37/50] d5bfc29c0dba 11.19s P=1447 U_pre=0.29 retrieve=False
[38/50] d5d9c4b20dba 12.23s P=755 U_pre=0.42 retrieve=True
[39/50] d5c747560dba 13.62s P=1024 U_pre=0.25 retrieve=False
[40/50] d5d7d0080dba 12.11s P=593 U_pre=1.00 retrieve=True
[41/50] d5d5b9120dba 6.67s P=352 U_pre=0.58 retrieve=True
[42/50] d5df295c0dba 10.36s P=1229 U_pre=0.26 retrieve=False
[43/50] d5d562aa0dba 10.05s P=772 U_pre=1.00 retrieve=True
[44/50] d5d44fa00dba 10.85s P=648 U_pre=1.00 retrieve=True
[45/50] d5c981a60dba 9.67s P=1460 U_pre=0.90 retrieve=True
[46/50] d5d318420dba 4.57s P=435 U_pre=0.32 retrieve=False
[47/50] d5cd51960dba 10.04s P=654 U_pre=1.00 retrieve=True
[48/50] d5c987320dba 9.95s P=1474 U_pre=0.35 retrieve=False
[49/50] d5d106560dba 13.81s P=1014 U_pre=1.00 retrieve=True
[50/50] d5d5b9da0dba 10.26s P=874 U_pre=0.34 retrieve=False
[done] webqa_full 50 in 555.8s -> /home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018/v2_s1_webqa_full/webqa_full.jsonl; empty_answer_rate 0.000000; refusal_rate 0.007286; peak_gpu_memory_gib 34.28
```

### okvqa_gs（生成成功，人工关卡未放行）

```text
[22/50] 2910485 2.15s P=957 U_pre=0.00 retrieve=False
[23/50] 5100805 1.8s P=498 U_pre=0.82 retrieve=True
[24/50] 760295 2.48s P=999 U_pre=0.24 retrieve=False
[25/50] 1538325 2.11s P=1004 U_pre=0.77 retrieve=True
[26/50] 4222125 1.59s P=632 U_pre=0.82 retrieve=True
[27/50] 1744575 2.28s P=762 U_pre=0.83 retrieve=True
[28/50] 4741645 1.9s P=771 U_pre=0.13 retrieve=False
[29/50] 5813945 1.7s P=555 U_pre=1.00 retrieve=True
[30/50] 802195 2.28s P=697 U_pre=0.69 retrieve=True
[31/50] 2600205 2.42s P=802 U_pre=0.57 retrieve=True
[32/50] 1413345 2.43s P=567 U_pre=0.51 retrieve=True
[33/50] 780325 2.46s P=572 U_pre=0.31 retrieve=False
[34/50] 2517525 3.68s P=1077 U_pre=0.03 retrieve=False
[35/50] 4246685 2.12s P=581 U_pre=0.16 retrieve=False
[36/50] 2804135 2.23s P=471 U_pre=0.06 retrieve=False
[37/50] 1576015 2.82s P=927 U_pre=0.69 retrieve=True
[38/50] 4751845 4.18s P=589 U_pre=0.55 retrieve=True
[39/50] 4273765 2.37s P=776 U_pre=0.61 retrieve=True
[40/50] 2689665 2.23s P=744 U_pre=0.58 retrieve=True
[41/50] 2907005 2.03s P=658 U_pre=0.73 retrieve=True
[42/50] 2248505 2.06s P=709 U_pre=0.92 retrieve=True
[43/50] 3294945 2.95s P=1065 U_pre=0.02 retrieve=False
[44/50] 5348455 1.84s P=700 U_pre=0.43 retrieve=True
[45/50] 5197165 2.2s P=907 U_pre=0.42 retrieve=True
[46/50] 3686485 2.23s P=840 U_pre=0.00 retrieve=False
[47/50] 5603325 1.64s P=472 U_pre=1.00 retrieve=True
[48/50] 2608085 2.82s P=983 U_pre=0.51 retrieve=True
[49/50] 2162545 1.86s P=783 U_pre=0.61 retrieve=True
[50/50] 3249795 1.76s P=535 U_pre=0.26 retrieve=False
[done] okvqa_gs 50 in 118.1s -> /home/user/Futaoran/omniaware_v2_selfcheck_20261018/runs/fp32_20261018/v2_s1_okvqa_gs/okvqa_gs.jsonl; empty_answer_rate 0.000000; refusal_rate 0.000000; peak_gpu_memory_gib 34.10
```

## 事实、限制与下一步

事实：精度实现与数值自检通过；S1 生成和输入溯源检查完成；人工格式仍阻塞。550 个 CPU 用例通过，1 个 skip 是缺少历史 T15 固定输出，不是当前 8B 权重缺失。初始失败和全部重试记录均保留。

推测：剩余问题可能来自冻结指令对最短答案约束不足或模型输出习惯；精度通过本身不能区分二者。验证需要另行提出并获准修改提示词/协议，采用独立运行与缓存重新做完整关卡；不能后处理重写当前答案或放宽格式判据。

未完成/未验证：全部 S1 关卡、逐图语义 GPU 描述、S2 官方评分/选择器/盲标、S4。本轮没有启动新的 S2；另一会话旧 BF16 作业不属于本报告的 FP32 验收。

建议：保留本次结果为失败证据，先审定具体格式问题及独立的提示词修订方案，再重新自检/S1，并重新校准 S2。本次已批准的精度方案不包含冻结提示词或解析规则变更，不能自行扩展。S4 继续禁止。

证据从服务器下载约 2.63 MiB，仅代码/日志/结果，无模型和数据集。MANIFEST.json 的 45 个服务器原始文件已逐个 SHA-256 校验；本报告和 manual_format_review.json 是随后生成的解释性文件，不在该原始传输 manifest 内。

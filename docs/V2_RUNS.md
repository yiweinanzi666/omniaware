# v2 GPU 阶段命令（S1 / S2 / S4）

`scripts/v2_runs.py` 只编排命令。`--dry-run` 把 `python -m omniaware.run` 的命令行打到标准输出，不写文件，不启动模型。去掉 `--dry-run` 才会执行那些 `omniaware.run` 行。本机不跑 GPU，下面的时长全部是估计值，没有实测吞吐。

S1 还会多打印一行图像诊断命令。那一行只打印，不执行。

## 时长（估计值）

抄自方向文档第 6 节。文档写明这些 GPU 时间都是估计值，S1 之后才按实测吞吐修正。这里没有测量，也不把估计拆成每条命令的吞吐。

| 阶段 | 内容 | GPU 时间 |
|---|---|---|
| S1 | 每个数据集 50 题冒烟，五个动作，外加图像诊断命令 | 1 张 GPU，约半天（估计值） |
| S2 | WebQA distractor、`webqa_open_img`、OK-VQA 的校准集与开发集 | 1 张 A100，约 4–8 小时（估计值，8B 模型短答案） |
| S4 | 全量测试集，每个数据集只跑一次 | 1–2 张 A100，约 1–2 天（估计值） |

## 切分（只用当前 `run.py` 已有的旗标）

`python -m omniaware.run --help` 里 `--split` 只有 `dev` 和 `test`。校准集不是第三个 split，而是：

- `--n-calib 500`（WebQA 要求不少于 500，协议默认就是 500）
- `--n-dev 256`
- `--split-seed 2027`（run.py 的默认值，这里写在命令里）
- WebQA（`webqa_full` 是 distractor；`webqa_open_img` 是开放检索）：`--calib-source train`。校准集和开发集都来自 train。测试集是全部 val。
- OK-VQA 没有 train loader，沿用已有协议：`--calib-source heldout`，校准 500，开发 256，测试集是 val 里这两段之后的剩余题。

`webqa_open_img` 的划分先在 `webqa_open_full` 上取 500 / 256，再切到图像题（`data.IMAGE_SUBSET_OF`）。命令仍传协议尺寸 500 和 256，不把图像子集的题数回写成另一个 `--n-calib`。

S1、S2 传 `--split dev`。S1 再用 `--n-eval 50` 取开发集前 50 题，不改 `--n-dev`，因此不挪动测试集。S2 不传 `--n-eval`，也不传 `--split test`：写出的评测 JSONL 是开发集；校准集由 `--n-calib` 和 `--calib-source` 在同一次运行里确定。S4 传 `--split test`，不传 `--n-eval`。脚本不发明 `--split calib`。

## v2 契约旗标（协议分支正在加）

本工作树的 `run.py` 还没有下面这些旗标。三个阶段的命令仍然会带上它们。它们属于 v2 契约，由协议分支加到 `omniaware.run`：

- `--v2-actions direct,static_text,static_image,static_mixed,rerank_static`
- `--save-signals`：保存 B2 信号（首 token 的 top-k 与 logprob、归一化和未归一化熵、margin、贪心答案的序列 log-prob 与长度、各证据动作相对 direct 的首 token KL、检索器信号）
- `--empty-answer-accounting`：空答案单独记账，并在运行汇总里给出空答案率
- `--shared-prompt-registry`：同一次运行的所有动作共用提示词注册表
- `--decisions PATH`：只在 S4 出现。脚本把冻结文件原样交给 runner，不把文件展开成网格，文件缺失或不是 JSON 对象时直接非零退出，不用默认超参顶上

模型是 `--model 8b`，与上面 S2 的 8B 短答案估计一致。`--tag` 为 `v2_<阶段>_<数据集>`。

## 命令

在仓库根目录执行。S1 / S2 的 `omniaware.run` 行是本机 `--dry-run` 的原样输出。

### S1 冒烟（估计值：1 张 GPU，约半天）

```
python scripts/v2_runs.py --stage s1 --dataset webqa_full --n 50 --dry-run
python scripts/v2_runs.py --stage s1 --dataset webqa_open_img --n 50 --dry-run
python scripts/v2_runs.py --stage s1 --dataset okvqa --n 50 --dry-run
```

`webqa_full` 与 `okvqa` 的实际打印：

```
python -m omniaware.run --dataset webqa_full --split dev --model 8b --tag v2_s1_webqa_full --calib-source train --n-calib 500 --n-dev 256 --split-seed 2027 --n-eval 50 --v2-actions direct,static_text,static_image,static_mixed,rerank_static --save-signals --empty-answer-accounting --shared-prompt-registry
python scripts/analysis/image_evidence_check.py --n 50
python -m omniaware.run --dataset okvqa --split dev --model 8b --tag v2_s1_okvqa --calib-source heldout --n-calib 500 --n-dev 256 --split-seed 2027 --n-eval 50 --v2-actions direct,static_text,static_image,static_mixed,rerank_static --save-signals --empty-answer-accounting --shared-prompt-registry
python scripts/analysis/image_evidence_check.py --n 50
```

`webqa_open_img` 与上面同一格式，`--dataset webqa_open_img`，`--calib-source train`，`--tag v2_s1_webqa_open_img`，同样再打印 `python scripts/analysis/image_evidence_check.py --n 50`。

图像诊断是 WebQA 图像题用的（任务 A5）。S1 每次都把这一行打出来，包括 OK-VQA；本脚本不调用它。

### S2 校准集与开发集（估计值：1 张 A100，约 4–8 小时，8B 短答案）

```
python scripts/v2_runs.py --stage s2 --dataset webqa_full --dry-run
python scripts/v2_runs.py --stage s2 --dataset webqa_open_img --dry-run
python scripts/v2_runs.py --stage s2 --dataset okvqa --dry-run
```

实际打印：

```
python -m omniaware.run --dataset webqa_full --split dev --model 8b --tag v2_s2_webqa_full --calib-source train --n-calib 500 --n-dev 256 --split-seed 2027 --v2-actions direct,static_text,static_image,static_mixed,rerank_static --save-signals --empty-answer-accounting --shared-prompt-registry
python -m omniaware.run --dataset webqa_open_img --split dev --model 8b --tag v2_s2_webqa_open_img --calib-source train --n-calib 500 --n-dev 256 --split-seed 2027 --v2-actions direct,static_text,static_image,static_mixed,rerank_static --save-signals --empty-answer-accounting --shared-prompt-registry
python -m omniaware.run --dataset okvqa --split dev --model 8b --tag v2_s2_okvqa --calib-source heldout --n-calib 500 --n-dev 256 --split-seed 2027 --v2-actions direct,static_text,static_image,static_mixed,rerank_static --save-signals --empty-answer-accounting --shared-prompt-registry
```

每条命令生成开发集（`--split dev`，256 题），并用 `--n-calib 500` 与 `--calib-source` 固定校准集。不生成测试集。`--save-signals` 要求保存 B2 信号；协议分支实现时，应对这次运行装入的校准集和开发集都落盘。

S2 结束后用开发集上各动作的官方分数做 go/no-go（见下）。相对最强单一动作不到 +8% 就不要进 S4。

### S4 冻结测试（估计值：1–2 张 A100，约 1–2 天）

```
python scripts/v2_runs.py --stage s4 --dataset webqa_full --decisions docs/dev_decisions.json --dry-run
python scripts/v2_runs.py --stage s4 --dataset webqa_open_img --decisions docs/dev_decisions.json --dry-run
python scripts/v2_runs.py --stage s4 --dataset okvqa --decisions docs/dev_decisions.json --dry-run
```

`docs/dev_decisions.json` 在本工作树里还不存在。上面三条现在会以退出码 1 结束，标准错误为：冻结决定文件找不到，并且没有用默认超参代替。文件必须是 JSON 对象；缺旗标、路径不存在、JSON 非法或根节点不是对象，都同样拒绝。

文件存在时每个数据集只打印一条命令，`--split test`，带上决定文件路径，没有网格：

```
python -m omniaware.run --dataset webqa_full --split test --model 8b --tag v2_s4_webqa_full --calib-source train --n-calib 500 --n-dev 256 --split-seed 2027 --v2-actions direct,static_text,static_image,static_mixed,rerank_static --save-signals --empty-answer-accounting --shared-prompt-registry --decisions docs/dev_decisions.json
python -m omniaware.run --dataset webqa_open_img --split test --model 8b --tag v2_s4_webqa_open_img --calib-source train --n-calib 500 --n-dev 256 --split-seed 2027 --v2-actions direct,static_text,static_image,static_mixed,rerank_static --save-signals --empty-answer-accounting --shared-prompt-registry --decisions docs/dev_decisions.json
python -m omniaware.run --dataset okvqa --split test --model 8b --tag v2_s4_okvqa --calib-source heldout --n-calib 500 --n-dev 256 --split-seed 2027 --v2-actions direct,static_text,static_image,static_mixed,rerank_static --save-signals --empty-answer-accounting --shared-prompt-registry --decisions docs/dev_decisions.json
```

脚本读文件只为了确认它在、且是 JSON 对象。它不读取里面的阈值列表，也不会为多组超参各跑一次。

## S2 go/no-go

开发集逐题选对上限相对最强单一动作至少 +8% 才 PASS。输入：

```json
{"questions": [{"qid": "q1", "scores": {"direct": 0.25, "static_text": 0.5, "static_image": 0.0, "static_mixed": 0.25, "rerank_static": 0.0}}]}
```

```
python scripts/analysis/v2_gate.py --scores tests/fixtures/v2_gate_example.json
```

标准输出恰好两行：第一行 `PASS` 或 `FAIL`，第二行是 oracle 均值、最强单一动作的名字、该动作的均值、相对差距。`mean(best) > 0` 时相对差距是 `(mean(oracle) - mean(best)) / mean(best)`，否则是 `mean(oracle) - mean(best)`。并列时按 `direct`、`static_text`、`static_image`、`static_mixed`、`rerank_static` 的顺序取最先的一个。退出码：PASS 为 0，FAIL 为 1，坏输入为 2。

示例夹具 `tests/fixtures/v2_gate_example.json` 为 PASS（相对差距 0.25）。`tests/fixtures/v2_gate_fail.json` 为 FAIL（相对差距 0.0625）。

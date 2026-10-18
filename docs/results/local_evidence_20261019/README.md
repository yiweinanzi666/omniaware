# 2026-10-19 实验分析交接索引

本目录补交 Git 工作树以外的小型实验材料，并汇集 GitHub `results/*`、`runs/*` 分支中已经公开的结果证据。它是供分析使用的文件快照，不代表重新运行或重新验收了实验。源码、模型身份、输入、参数、评分和完成状态应按每次运行自己的记录解释。

## 从哪里开始

1. `inventory_summary.json`：收集范围、文件数量、排除原因、外部源码版本，以及 GitHub 结果分支的固定提交。
2. `file_manifest.csv`：每个源文件的原路径、大小、SHA-256、归档路径和发布文件 SHA-256。`included_gzip` 表示无损压缩；带脱敏处理的文件不能当作源文件的逐字节副本。
3. `github_branches/results/v2-s1-20261018/docs/results/20261018_node04_format_revalidation_report.md`：已批准提示词修复后的隔离重验、自动与人工格式检查、失败及重试记录。
4. 同分支 `runs/fp32_format_20261018/`：对应的原始逐题输出、配置、环境、运行日志、完成文件和人工复核。
5. `data_ops/node04_selfcheck_20261018/`：本地已保存的 S1 逐题输出、FP32 数值检查、答案格式诊断与人工复核。
6. `data_ops/node04_v2_20261017/`、`data_ops/v2_preflight_20261017/`、`OmniAware_Data/manifests/`：资产校验、环境、测试日志和数据准备证据。
7. `data_ops/v2_s2_pause_20261018/` 及各时间戳日志：暂停、恢复、失败和部分运行记录；不要把部分输出当作完整正式结果。

## 如何解释不同时间的记录

根目录 `EXPERIMENT_COMPLETION_REPORT.md` 是历史交接记录，其中的“仍缺失”或“正在生成”只描述其记录时刻。2026-10-18 的其他 GitHub 结果分支已包含 8B 原始权重检查、FP32 诊断与 S1 运行证据，不能继续把历史报告中的缺失清单当作最新现状。最新分支和提交以本目录的 `inventory_summary.json` 为准。

提示词修复后的报告记录 WebQA 和 OK-VQA 两套 S1 均生成 50/50 题，但两套人工格式关卡均为 FAIL，该轮未进入新的 S2。分析时保留这个限制；不要把自动检查通过、数值检查通过或文件已上传解释为论文主实验全部通过。

各 `github_branches/<branch>/` 子目录保留原分支中的相对路径。关联代码请检出 `inventory_summary.json` 中记录的固定 commit。此处只是证据汇集，没有把不同分支的方法实现合并，也没有将多个不同协议的运行拼接成同一次实验。

## 上传与排除规则

- 单个原始文件达到 100 MiB 时排除；权重、张量、视频、音频、图片等体量大的资产类别、下载碎片、缓存和私有认证文件也排除。源目录中的 `.git` 历史和 Python 字节码不作为实验材料上传。
- 小型外部源码、许可、配置、查询、VideoRAG 转录、Parquet 文件及 NLTK 支持资源随本次提交保存。它们有助于检查工程与输入来源，但不意味着完整像素或权重都在 GitHub。
- 体量较大的文本日志、清单以及 VideoRAG `.info.json` 下载元数据使用 gzip 无损压缩。源文件保持原样。若原目录已有同名 `.gz`，新压缩副本使用 `.upload.gz`，两者分别登记。
- 识别到的认证字面量仅在发布副本中脱敏，源文件不修改；具体文件和数量见 `redacted_files`。
- 仓库外的日志可能仍由其他会话写入，本次保存的是收集时读取到的字节。源路径和文件名用于说明来源，不保证合作者机器上存在相同绝对路径。

gzip 文本可用标准工具读取，例如：

```python
import gzip
import json
from pathlib import Path

path = Path("path/from/file_manifest.csv.jsonl.gz")
with gzip.open(path, "rt", encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        # 按本次运行的配置、动作和题目 ID 进行分析。
```

在没有模型权重和原始像素的情况下，可以分析已保存预测、日志、配置、输入溯源与失败原因；重新生成模型答案或逐图复查仍需恢复被排除的大型资产。2026-10-19 尝试通过现有 SSH 密钥读取 node04 最新结果时，跳板机返回身份验证失败，本次未从服务器下载到新文件。清单没有记录的服务器文件是否存在、是否已完成，不能由此次 GitHub 上传推断。详见 `server_collection_attempt.json`。

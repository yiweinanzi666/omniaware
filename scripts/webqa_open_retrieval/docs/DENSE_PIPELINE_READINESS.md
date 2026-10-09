# 全库检索、百分位合并与来源打包接入

2026-10-06。本轮完成代码连接、三张图像的官方加载方式对齐和真实输入验证；全库编码尚未结束，没有新增 QA 准确率。

## 已完成

- `calibrate_dense_relevance.py` 按冻结的 `DENSE_ECDF_PROTOCOL.md`，用混合训练500题、每库2500个 top5 分数拟合右连续 ECDF；每题合并两库10条候选，选总共5条。模态内名次和固定哈希处理百分位平局。4项CPU回归通过。
- 真实1024向量工程索引上，独立逐分数计数核验2560个百分位，最大差0；256题全部来源次序一致，选中586条文本、694条图像。重复执行没有重新拟合、选择或改变文件。证据：`DENSE_RELEVANCE_INDEPENDENT_VALIDATION.json`。
- 开发时复现未完成输出被覆盖的缺陷，修复为已有内容一致才恢复完成记录，内容不同则保留现场。修复前后证据：`DENSE_RELEVANCE_REENTRY_REPRO.json`、`DENSE_RELEVANCE_PRE_FIX_TESTS.log`、`DENSE_RELEVANCE_TESTS.log`。这是新连接代码的缺陷，不是历史负准确率的已证实成因。
- `join_dense_source_metadata.py` 连接精确文本、全部去重caption及索引所用像素。工程256题共1280条来源、742个不同来源、435张不同图像，与原始元数据和检索排名逐项一致；保留20次额外caption出现。静态原排序输入343–882 tokens，四个最长输入的实际图文processor渲染一致。初始U尚未填入，不能作为正式执行清单。证据：`DENSE_SOURCE_METADATA_INDEPENDENT_VALIDATION.json`。

这里的 top5 定义沿用 `search_dense_queries.py`：FAISS原生截取候选，然后对返回项按分数和来源ID排序。ECDF单调性保证每库已有前五足够做本次合并；没有声称FAISS在第5名边界的同分项遵循全库词典序，也没有声称不同候选截取数的边界同分结果必然相同。

## 图像可用性勘误与协议附录

先前“3张JPEG不可用”的判断只对应严格PIL解码。已核对[官方README](https://github.com/WebQnA/WebQA)和固定版本的[官方CLIP脚本](https://github.com/WebQnA/WebQA_Baseline/blob/279d5ae47faee096f7d4171ba0bacdfaec27d30f/vlp/CLIP_retrieval_experiments/CLIP_img_encodings.py)：官方启用截断图像加载。三个原始TSV的SHA保持不变；严格方式均失败，官方方式均得到图像。

新增 `prepare_official_jpeg_assets.py` 仅对30039112、30039288、30298493的固定原始字节使用该方式，将所得RGB无损保存为独立PNG。CLIP和Qwen图像处理器的输出张量与直接官方方式解码逐元素相同；3条CLIP向量已实际完成。原始JPEG缺失的压缩字节没有被恢复，30298493底部可见解码器填充区。这三张无需采用其他公开版本；30172755后续同样恢复到原始TSV加载结果，见下文。

`TRUNCATED_JPEG_LOADER_REPRO.json`、`TRUNCATED_JPEG_OFFICIAL_LOADER_RECEIPT.json`、`TRUNCATED_JPEG_INDEPENDENT_VALIDATION.json`保存上述三张图的复现、版本、像素和处理器一致性，以及当时包含公开修订的完整ID并集。前两份批量编码仍未全完成。

随后对30172755继续同一加载方式检查，确认官方TSV payload与已保存的2008年原图逐字节相同，原始231×324图像也可以按官方方式解码。独立处理器核验及1条CLIP向量完成，见 `ORIGINAL_IMAGE_30172755_REPRO.json`、`ORIGINAL_IMAGE_30172755_OFFICIAL_LOADER_RECEIPT.json`、`ORIGINAL_IMAGE_30172755_INDEPENDENT_VALIDATION.json`。这消除了采用231×320公开新版裁剪的必要。新版资产、向量和历史记录全部保留，但不纳入后续正式索引。

**正式索引的四份图像目录**现在固定为：`clip_images_cpu_wave0`309793、`clip_images_cpu_wave1`79953、`clip_images_cpu_official_jpeg3`3、`clip_images_cpu_original_30172755`1。它们互不重叠，覆盖全部389750个ID；不得同时加入同ID的 `clip_images_public_revision_30172755`。四张均使用原始官方TSV的加载结果，未补造缺失压缩字节。

此附录更正冻结ECDF文档第1条的图像可用性描述，并更新 `SOURCE_METADATA_PROTOCOL.md` 第5条中30172755的资产选择为原始TSV加载结果。**ECDF公式、校准问题、排名及caption规则不变**。保留冻结原文与原工程产物，供追溯；正式清单要求完整544489文本和389750图像ID，并记录上述像素版本。工程前1024图像未包含30172755，其已验证输入不变；固定旧实验不受影响。

图像caption另按 `SOURCE_METADATA_PROTOCOL.md`：保留全部空白规范化后去重的描述，与题目、答案和标签无关。全库10007张图像有多个规范化描述，最多248条、最长合计28623字符。多版本描述可能冲突或增加噪声；这是明确的新输入协议，不冒充历史 `imgid2caption.pkl` 的恢复。

## 后续执行

1. 保持当前WebQA正式队列和两个CPU编码任务；CLIP wave0结束后接已有wave1，不重算任何完成分片。
2. 用所有完成的编码目录构建全库FAISS，BGE与CLIP对同一500校准题和256开发题分别搜索。正式CLIP输入包括四份不重叠资产目录。
3. 分别执行 `calibrate_dense_relevance fit`、`merge` 和 `join_dense_source_metadata`，使用新的 full 目录且省略 `--engineering-prefix`。工程百分位与来源不可充作正式结果。
4. 在可用设备槽位完成514条初始观测，复用242条已核验旧观测，再用完整500题拟合median/IQR。将真实U连接到来源清单，完成cohort消费检查。
5. 按正式完整提示计算共同输入上限，冻结seeds17/23/29各12排列及参考排序，执行static/replication/decode-only EAB。保留Direct及整体强基线比较，逐臂报告实际token和耗时。

2026-10-06T23:51:51Z，步骤1–3的CPU接续任务已实际启动，并通过实时时间、命令和上游进程核验。它按各路编码的实际完成情况自动接续，不启动MPS或问答模型；具体计划、恢复修复及8项测试见 `DENSE_CONTINUATION.md`。全库检索当前仍未完成，初始U和准确率仍待后续实验。

当前证据说明接入实现和来源身份通过检查，不说明检索证据有益。LPER128仍比Direct低9.0625点，老师要求的准确性提升尚未达到。

> **LEGACY（旧池）。** 本文记录的 544,489 文本与 389,750 图像来自 `build_global_webqa_pool.py`，该脚本无法在本仓库运行。论文池统计以 `scripts/setup/prepare_webqa_open.py pool` 写出的 `pool_stats.json` 为准；两者的文本键不同，两种产物不要混用。

# WebQA 全局公开事实池准备

**已完成全部train、val、test公开事实的全局池：544,489份精确正文去重文本＋389,750个图像ID，共934,239个检索文档。** 全部787,697个原snippet ID通过alias映射保留；尚未创建embedding或FAISS索引。

源包含train 36766、val 4966、test 7540题。train/val从既有JSON流式读入，test直接从原7z成员流式读取，未落盘包含答案的test原文件。只解码split和事实数组，跳过Q/A等值，输出不含正负标记或query→facts关联。

全局去重后：**787,697个文本ID、389,750个图像ID**。官方README给出的量级是544k文本＋390k图像，官方full-scale代码构造整个uniid2fact并追加全部imgid2caption；本目录没有把每题候选列表冒称全局检索。V3.8约390k text-and-image总池的写法与此官方规模不同，论文具体池仍需绑定。

## 全局文本去重与官方规模对齐

`text_corpus.jsonl`是787,697条按原snippet ID保存的完整来源库存。供后续文本编码使用的是`text_documents.jsonl`：按正文精确相等去重为544,489份，不做小写、空白或标点归一化；123,709份正文带有多个snippet ID，所有alias、title、URL均保留。检索返回的document_id能够映射回所有官方snippet IDs，没有丢弃原引用身份。

已读取固定官方BM25 notebook的代码字段，跳过所有输出值。其line33明确图像offset为544489；lines87–89使用fact2uniid及其逆映射，支持按正文事实去重的全局文本语义。此处独立得到完全相同的唯一正文数。原fact2uniid.pkl的行顺序未在本地验证，因此本交付采用明确的SHA256排序和自有row_index，不冒称原FAISS行号相同。

图像ID连续覆盖30000000–30389749，共389,750。已有公共镜像manifest的indexed_image_ids=389740与该原始公开图像ID数相差10；本池依据原数据保留所有ID，不为适配镜像而删图。V3.8约390k text-and-image总池与这里934,239个文档的总数不同，仍需论文实际corpus manifest解释。

## 冲突与不丢失策略

文本ID带多套元数据：0；真正同ID不同正文：0。后者保留所有text_variants，text=null、embedding_eligible=false。图像同ID不同caption或URL：12,417，完整写入id_conflicts.jsonl；元数据变体不自动等同像素冲突，也没有静默挑一个URL。

可明确编码的非空文本ID：787,697；需先处理的文本ID：0。所有官方ID仍保留，未按文本相似性跨ID合并。

## 现有图像资产

有manifest且实际文件字节数匹配：66,971；只有数值文件名匹配而缺乏manifest身份：1；扫描范围内没有本地文件：322,778。

reusable_image_assets.jsonl逐ID列出候选实路径、字节数、来源manifest及已记录SHA；missing_image_assets.jsonl列出每个缺图ID、公开URL/caption。已存在不等于本轮验证了像素正确性：本轮仅核存在和字节数，没有解码或重哈希全部图片。正式编码前仍需核验选定文件SHA、ID映射、processor和具体图像版本；官方TSV原字节与公共镜像图片的证据边界逐条保留。

扫描限定于本项目cache/webqa*目录和相应manifest，不声称其他磁盘完全没有缺失ID。没有下载图像，也没有用caption替代实际CLIP图片编码。

## 可复用交付

- `text_corpus.jsonl`：450,881,526 bytes；SHA256 `28fa97456525238b9870a3bfb19dab9dc492f660c082d06eea0c387ad1ff1ba2`。
- `image_corpus.jsonl`：281,117,948 bytes；SHA256 `e0bf4367349f49d65733da3d59d89b0937abf3a7fd47ee2b9657a48c13906061`。
- `id_conflicts.jsonl`：7,353,066 bytes；SHA256 `6c5c3575fd0937657f4d2e7f0d6c5c530bff88c47cde227cf2653c0329086971`。
- `missing_image_assets.jsonl`：110,350,072 bytes；SHA256 `9de45690f60f901717b671c4393dc3d9aa911dac5e9583d3c6cf387003e60f9e`。
- `reusable_image_assets.jsonl`：41,683,081 bytes；SHA256 `d83dd801f0d6a9a11945c92d06f5a916179113ad55d6b64fc3144328b3c76319`。
- `global_fact_variants.sqlite`：682,094,592 bytes；SHA256 `cd27575835e327bcbd9a846f3b475293394a4c3d37b9cc2fc7389f8480f15547`。

GLOBAL_POOL_MANIFEST.json保留原archive及两个source payload哈希、官方说明/代码固定revision、扫描范围、逐文件哈希和完整计数。SQLite只保存无标签事实变体，便于逐ID复查；JSONL可直接作为后续embedding输入的来源清单。

旧队列未停止或重启；本轮0模型调用、0图片下载、未进入MPS。下一步的真实编码/检索参数需依据新统一协议固定；这里没有把依赖测试、部分旧embedding或文本可编码数当成完整dense检索就绪。

最终校验：`GLOBAL_POOL_VALIDATION.json`验证7项输出SHA、所有JSONL行数、无答案/问题/正负角色键、全部snippet aliases覆盖、正文hash一致和图像ID完整连续范围。

`text_documents.jsonl`：335,267,438 bytes；SHA256 `57b2b79d24c0fbc7b36d065bef287642f3f8c6c2c048f29027d29c721053bdc8`。

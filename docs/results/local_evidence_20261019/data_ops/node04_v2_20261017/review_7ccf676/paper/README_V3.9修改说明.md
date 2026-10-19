# Omni-Aware V3.9：修改说明

对照依据：EMNLP 四位审稿人意见（`review-rag.txt`）、`WWW投稿_审稿意见对照与修改清单.md`、`科研全流程与论文写作指南.md`。
对比稿：`../diff_V3.8_V3.9/diff.pdf`（红色为删除，蓝色为新增；表格整体标记）。

## 一、编译状态

| 项目 | V3.8 | V3.9 |
|---|---|---|
| 正文结束页 | 第 8 页 | 第 8 页 |
| 全文页数 | 12 | **14**（超出 2 页，见第六节） |
| 编译错误 / 未定义引用 | 0 | 0 |

## 二、本版改了什么

### 写作与结构（按指南）

| 改动 | 对应 |
|---|---|
| 摘要按"任务、挑战、方法、效果"四个功能重写；第一次出现就定义 latent space | 指南 10.2；修改清单 3.5 |
| 引言第 2 段补上"现有方法为何不行"的文献（MMA-RAG、UniversalRAG、SAM-RAG、DRAGIN）；Challenge 1 改为现象证据（span 长度、分数尺度）加对照证据，不再引用附录表；Challenge 2 补上 sorted–reverse gap | 指南 4.2、10.3；sjbm W4、W5 |
| 贡献由 3 条改为 4 条，新增"识别两个挑战" | 指南 10.3（导师意见） |
| **图 1 图注统一为标准差**（3.53 / 2.03 / 1.07），删掉"10×" | 指南 13.3（导师点名） |
| 7 处待审措辞全部改写（therefore、thus、AURC thus reflects、bounded prior 等） | V3.8 的 `\ai` |
| `\hsem` 改名为 `\hpred`（predictive entropy） | 修改清单 3.3 |
| 缩写在首次出现时展开：IQR、NTE、JSD、NLI、AP、WR、SDPA、KV、RoPE、ECDF；附录中的 DCA 改为 rep | sjbm W5；修改清单 3.3 |
| 统一命名：WebQA-img（替代 WebQA T→MM、image-backed）、Video-MME-Short、MMQA-table、Chunk replication，表中方法名统一为 \method{} | 修改清单 3.3 |
| 问题定义补上 X_sys 的内容、检索器输出 s_j、模型输出 | sjbm C |
| 方法：说明 span 不压缩、图中只是示意；同模态多个 chunk 汇总计算 H_m；未路由的模态不删除；只有检索到的 chunk 加偏置；r_j 定义为按模态的 ECDF | sjbm W3a/b/c；修改清单 3.4 |
| 方法：新增 EAB-P 变体（首 token 也加偏置） | 修改清单 P0-2 |
| 实现细节（γ 网格、层数预实验、SDPA 开销）从方法节移到实验设置和附录 A | 指南 10.6 |
| 删掉流程表 Table 1（与图 3 和 4.1 节重复） | 修改清单 3.1 |
| 图 3 图注补充热力图横纵轴的含义 | sjbm C |
| 每个前向引用都按编号顺序（引言改为引用节号） | sjbm W5；指南 12.2 |
| 结论按指南结构重写：框架与挑战、模块、实验、未来工作 | 指南 10.11 |
| 新增附录 F Ethical Considerations；Limitations 改写为说明适用范围（白盒依赖、replication 作为黑盒近似、首 token） | YwiL 局限；fnp4 W1 |
| 4 张图补上 ACM 要求的 `\Description` | ACM 模板警告 |

### 协议表述的改动（**需要按此重跑或核实**）

| 位置 | 新写法 | 原写法 |
|---|---|---|
| 5.1 Implementation | WebQA 的 500 条校准查询**取自 train split**，与所有评测查询不相交 | 校准查询位于被评测的验证集内 |
| 5.1 Implementation | 超参数在**与所有评测集不相交的 development queries** 上选择 | "selected on these validation splits" |
| 5.1 Prompts | 新增各任务的答案格式，以及"首 token 决定答案的比例" | 无 |

这两条协议对应修改清单 P0-5。如果实际实验没有按这个协议做，要么重跑，要么把原文改回去。

## 三、新增实验（数字全部用 `\ph` 占位，显示为黄底 X.XX）

| 表 / 位置 | 内容 | 回应 |
|---|---|---|
| Table 1（新）评测设置表 | 各数据集的规模、初始上下文、路由候选模态 | 指南 6.2；sjbm W2 |
| 5.1 正文 | 各模态的平均 span 长度、检索分数范围 | Challenge 1 的现象证据 |
| Table 2 主表 新增 4 行 | DRAGIN、SeaKR（已填入已有的 OK-VQA 和 WebQA-img 数值）、Cross-encoder + static、Omni-Aware + cross-encoder | sjbm C；YwiL C2；修改清单 P1-3 |
| Table 3 消融表 | 新增 WebQA-img 三列；新增行：DRAGIN trigger、Per-format thresholds、EAB-P、Raw retriever scores（已填 64.25 / 67.10） | sjbm W1；P1-4；P1-5；P0-2 |
| Table 4（新）路由消融，MMQA-full | Random、All、Retriever-side、M only、H only、M·H、Oracle；指标为 Score / WR / 每题检索模态数 | sjbm W2；P0-3 |
| Table 5 顺序鲁棒性（改为紧凑版） | 新增 FITM 行和 Flip% 列；完整版连同 seed 标准差、Gap、PosBias 和 InternVL2-40B 一组移到附录 Table 14 | sjbm W4；P0-2；P1-7 |
| Table 6（新）干预与检索错误 | Mask / CF / Distractor AP；Gold@2–5、σ=0.1/0.2 噪声、注入网页误导率；OK-VQA 的干预结果在附录 Table 15 | fnp4 W2/W3；YwiL W1/W2/C1；P1-1；P1-2 |
| 5.4.5 节 | EAB 相对 static 的 win/loss 比例与失败类别；附录 Table 20 为案例表 | YwiL C4；指南 7.5 |
| 附录 Table 17 | OK-VQA 5046 题 BM25 全部 5 臂（已填 32.05、31.70、+4.35 [3.75, 4.96]，臂名待核对） | V3.8 README 第二节 |
| 附录 Table 12 | 选择性风险表新增 DRAGIN 行 | — |

## 四、需要核对结论方向的句子（`\ai{}`，黄底，共 7 处）

这些句子写成了论文的正常表述，但结论方向取决于实验数字。填完数字后逐句检查：方向一致就保留；不一致就改写，同时检查摘要、引言和结论里相应的说法。

1. `sec_experiments.tex:75` reranker 与 controller 互补
2. `:110` per-format thresholds 需要每个模态单独调阈值，U_omni 只用一个
3. `:112` EAB-P 与 decode-only 的差距在 Video-MME 上最大
4. `:136` 单独用 M 或 H 会提高 WR
5. `:162` FITM 降低了标准差，但仍高于 EAB
6. `:162` InternVL2-40B 上的顺序鲁棒性排序相同
7. `:187` 遮蔽、反事实、干扰项三项实验都显示 EAB 更依赖被强调的证据

全部审完后，把 `main.tex` 中的 `\reviewmarkstrue` 改为 `\reviewmarksfalse`。

## 五、需要填写的事实（`\tbd{}`）

| 位置 | 内容 |
|---|---|
| `main.tex:223` | 检索到的 chunk 插在哪里（目前写的是"supplied context 之后、question 之前"），第二次 prefill 是否整段重新编码 |
| Table 1 | OK-VQA、MMQA-table、Video-MME 的路由候选模态；MMQA-table、MMQA-full、Video-MME 的题数和初始上下文 |
| 5.1 | 各任务的答案格式；视频取 4 帧还是 8 帧的规则；GPU 型号和数量；匿名代码仓库 URL |
| 附录 C | 图文子集协议（V3.8 遗留）；Table 17 的第五个臂 |
| 附录 D | CE 图像确认实验中 EAB 的时机（V3.8 遗留）；案例表 4 行 |
| 附录 F | ACM 要求的生成式 AI 使用声明 |

**我根据上下文写的、需要你确认与实际实现一致的描述：**
- r_j 是"校准查询的候选中、该模态检索分数的 ECDF"（4.4 节）
- WR 的定义：被触发检索的查询中，所选模态不包含任何 gold 证据模态的比例（附录 A）。V3.8 中 18.6 / 4.2 等数值如果用的是别的定义，要统一
- 干预实验（Mask、CF、Distractor）和注入实验的构造方式（附录 A），实验时按这个做，或者按实际做法改写
- 附录 C 中"heuristic defaults 让几乎所有分数都高于 τ，因此全部检索"的解释

## 六、还没有处理的部分

**图**（按你的要求没动，图注里已有 `%% NOTE` 注释）：
- fig1 右栏仍标注方差和"10×"，需要改成标准差重新导出
- fig3 中的 H_sem 要改为 H_pred；两个热力图要加轴标签
- 修改清单 P0-6：图 2 和图 3 的风格（Comic Neue 字体、卡通图标）

**页数**：正文已压回 8 页（4 张图都没动）；全文 14 页，比 12 页多 2 页，多出的部分都在附录。附录里已经删掉重复的表并合并了两张表，再压就要删真实内容。可选方案：
1. 先查 WWW '27 的 CFP：如果参考文献和附录不计入页数限制，就不需要再删；
2. 移到补充材料或删除：长答案实验（Table 18）、A-OKVQA / MRAG-Bench 段落、附录 B 中的 quantization error 段、Table 17。

另外，`\ai` 和 `\tbd` 的黄底标记本身也会占一点宽度，审完关闭后大约能收回几行。

**数据问题**（上一轮指出的，本版没动数字）：
- 附录 Table 16 的 WebQA-img Skip Acc 80.45 与 cold start 下的排序一致性矛盾（上限约 77.2），已在源码中加 `%% NOTE`
- Table 2、Table 5 中 WebQA full、MMQA、Video-MME 的数值需要逐一对应到运行记录
- 标题和方法名仍为 Omni-Aware（指南 10.1、15.1 建议避开 "Omni"）。改名只需修改 `\method` 宏和 `\title`，需要你和导师决定

## 七、提交前检查

- 在项目里搜索 `\tbd{`、`\ph` 和 `\ai{`，结果应只剩宏定义
- `\reviewmarksfalse`
- 删除源码中的 `%% NOTE` 注释
- 摘要、引言、结论中的数字与表格一致

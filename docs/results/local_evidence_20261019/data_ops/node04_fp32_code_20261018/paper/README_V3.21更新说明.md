# V3.21 更新说明（2026-10-11）

按照 2026-10-10 / 10-11 确定的对照方法与评测范围修改论文。代码对应 `omniaware` 仓库提交 `ea47284` 及之后。

## 范围变化

| 项目 | V3.20 | V3.21 |
|---|---|---|
| 评测 | WebQA（全量 + 1,247 题图像子集）、OK-VQA、MMQA-table、MMQA-full、Video-MME-Short | WebQA（全量，另按图像题 2,511 / 文本题 2,455 分组）、OK-VQA、HybridQA、VideoRAG-Synth |
| 主表对照 | Direct、Static、NTE、PASTA 式偏置、SAM-RAG、ViDoRAG、Visual-RAG、M2-RAG、DRAGIN、SeaKR、Cross-encoder | Direct、Static、Cross-encoder、Found-in-the-Middle、M2-RAG（官方单阶段）、SeaKR、UniversalRAG |
| 消融 | 含 DRAGIN 触发 | 删除 DRAGIN；NTE 与 PASTA 式统一偏置只在消融表中 |
| 路由实验 | MMQA-full | WebQA 全量开放检索（新增 UniversalRAG 查询路由一行） |
| 骨干迁移 | InternVL2-40B、Llama-3.2-Vision-11B | InternVL3.5-8B |
| 引用支持 | DeBERTa + FLAN-T5-xxl + "Llama-3-8B-NLI" + 150 题人工审核 | 只保留 DeBERTa-v3-large |
| 附录 | 含长答案行为分析 | 删除 |
| 统计 | 3 个 seed 的均值 ± 标准差 | greedy 解码只跑一次；配对 bootstrap（5,000 次，VideoRAG 按源视频聚类）；seed 只用于抽取 12 个证据顺序 |

## 数字的处理（按审查报告 §1.1、§6.1）

- 正文、表格、附录中所有找不到逐题输出来源的结果数字（包括原来保留的 OK-VQA、WebQA 全量数字、延迟、顺序标准差、引用指标、错误分类比例）都已改为 `\ph`。
- 保留的数字只有：超参数、数据集规模、定义性数值（始终检索的 100% 检索率、oracle 行），以及附录中来自旧仓库真实运行的诊断（BM25 全量 OK-VQA、WebQA 文本流程、证据选择对比）。这些诊断使用旧代码和旧协议，正式投稿前建议用新代码重跑或删除。
- 依赖尚未产生结果的结论句已用 `\ai{}` 标出。
- `fig/fig1.pdf` 仍是旧数字（方差 12.45/4.12/1.15），`fig/fig3.pdf` 仍写 H_sem，两张图都要在拿到新结果后重新导出。

## 按代码补全的定义（原为 `\tbd`）

- Inv. 的定义与分母；别名匹配规则；DeBERTa 检查器的 checkpoint、revision 与判定规则（附录 A，Citation support）。
- 冷启动路由：触发后先做一轮联合 top-k，再用其 prefill 注意力计算 $M_m\cdot H_m$，第一轮的块不复用（第 4.3 节与附录 A）。这是为了让 WebQA 上的路由实验能检验 $M_m\cdot H_m$，属于方法描述的修改，已用 `\ai{}` 标出，请确认。
- 干预构造：目标块的选取、同类型实体的来源与类型规则、随机编辑、干扰项、注入段落的相关度与位置。
- Llama-3.2-Vision 的适配（删除，改为说明其不在接口范围内）。

## 编译

`latexmk -pdf main.tex` 编译通过，没有未定义的引用。两处 overfull box 在 V3.20 中就已存在。

## 2026-10-11 第二轮：方法修订与篇幅

**方法（与代码 `cd48ab8` 及之后一致）**
- EAB 从预测第一个答案 token 的行（t_end）开始加偏置，覆盖所有预测答案 token 的行；decode-only 降为消融（审查报告 §1.3、§3.1.1）。
- 冷启动探测轮：没有初始证据时，先用联合 top-k 做一次零偏置 prefill，在这些证据 span 上计算 U，并用同一次注意力做 M_m·H_m 路由（§3.1.3）。新增消融行"Cold start without probe round"。
- OK-VQA 检索改用 GS-112K（Luo et al., EMNLP 2021），以 dev 结果为准（§3.3.12）；OK-VQA 校准集写明为与评测不相交的验证集切片。
- 新增"Development protocol"段落和附录中的 dev 网格与决策规则（与 `omniaware/docs/DEV_DECISIONS.md` 一致）。

**新颖性定位**
- 摘要、引言、贡献改为突出"一次零偏置 prefill 同时得到是否检索、往哪检索、如何用证据三个决策；无需训练路由器或分类器，也不需要多次采样"。
- 相关工作新增并区分：Beyond the Query（检索信号不能可靠改进查询路由）、Lost at the End（多模态读者的首位偏好，重排无法消除）、Do RAG Systems Really Suffer From Positional Bias?（文本 RAG 中位置偏差较小）、MAD-RAG（检索文本分散 LVLM 的视觉注意力）。

**篇幅（WWW 2027：正文 8 页，总计 12 页）**
- 删除 Fig. 1（顺序敏感性示意图，内容与 Table 4 重复，且图中是旧数字）；数据集表并入正文；误差分析、校准迁移、部分统计移到附录。
- 删除附录中旧代码、旧协议下的诊断（BM25 全量 OK-VQA、WebQA 文本流程、distractor 池上的证据选择对比）：它们不再描述当前方法。
- 编译结果：正文和结论在第 8 页内结束，全文 12 页；只剩两处 V3.20 就有的轻微 overfull hbox。

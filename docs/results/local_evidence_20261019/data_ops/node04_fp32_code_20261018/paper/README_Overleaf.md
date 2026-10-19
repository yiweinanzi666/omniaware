# 在 Overleaf 上使用

## 上传

1. Overleaf → New Project → Upload Project，选这个 zip 文件。
2. 打开 Menu：
   - Compiler 选 **pdfLaTeX**；
   - TeX Live version 选最新；
   - Main document 选 `main.tex`。
3. 点 Recompile。

`latexmkrc` 已设为 pdfLaTeX 模式。acmart 会自动加载 ACM 官方字体（Linux Libertine + newtxmath）。

## 编译后检查

- **页数**：正文止于第 8 页，全文不超过 12 页。这份源码在本地用 Libertinus 字体排出 12 页；Libertinus 与 Libertine 的度量几乎相同，但行距仍可能有细微差异。如果超出，可以缩小图 4 的宽度（`sec_experiments.tex` 中的 `width=0.8\columnwidth`），或精简附录 C 的文字。
- **标记**：
  - 黄底红字共 10 处：3 处 `\tbd{}` 占位符，7 处 `\ai{}` 待审措辞。清单见 `README_V3.8审阅说明.md`。
  - 7 处待审措辞审完后，把 `main.tex` 中的 `\reviewmarkstrue` 改为 `\reviewmarksfalse`。
  - 3 处占位符必须用真实信息替换。提交前在项目里搜索 `\tbd{`，结果应为 0。
- **日志**：Logs 中不应有 undefined reference 或 citation。

## 文件

| 文件 | 内容 |
|---|---|
| `main.tex` | 主文件：导言、引言至第 4 节 |
| `sec_experiments.tex` | 第 5 节实验、第 6 节结论 |
| `sec_appendix.tex` | 附录 A–E |
| `references.bib` | 43 条参考文献 |
| `fig/fig1.pdf` – `fig/fig4.pdf` | 矢量图 |
| `optional_latest_acmart/` | acmart 官方仓库最新版的 `acmart.cls` 和 `ACM-Reference-Format.bst` |

默认使用 Overleaf 自带的 acmart。只有在会议要求最新版、或 Overleaf 自带版本报错时，才把 `optional_latest_acmart/` 里的两个文件移到项目根目录。

## 提交前

- 3 处 `\tbd{}` 已填写，`\reviewmarksfalse` 已设置。
- 图 2、图 3 中替换过的文字（Comic Neue 字体）建议从作图源文件重新导出。
- OpenReview 上的标题、摘要与 PDF 一致。

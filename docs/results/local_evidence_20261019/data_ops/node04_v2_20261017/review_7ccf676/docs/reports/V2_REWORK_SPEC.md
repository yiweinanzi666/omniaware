# v2 返工规格（所有修复者先读，必须严格按此实现）

背景：`C:\Users\22688\Desktop\www_code\CURSOR_REVIEW_v2_验收意见.md` 是对分支 `v2/arbitration` 的验收意见，本规格落实其中的返工要求。方向依据见 `Omni-Aware_方向与改造方案_20261013.md`。

## 0. 通用规则
- 仓库：`C:\Users\22688\Desktop\www_code\omniaware`，当前分支 `v2/arbitration`。**直接在这个工作目录里改，不要新建分支或 worktree，不要 git commit，不要 push**（由协调者统一检查后提交）。
- 只改分配给你的文件。需要别人文件里的东西时，按本规格的接口假定它存在。
- 不要改 `C:\rc\` 下任何文件；不要下载模型或数据；不要编造任何数字。
- 代码风格：英文 docstring，注释密度与周围一致；不要写“为了兼容模块不存在”的防御性导入、不要写不起作用的参数、不要留下描述开发过程的注释（如“这个分支还没有 xxx”）。
- 动作集合**只从** `omniaware.actions` 导入：`DIRECT`、`EVIDENCE_ACTIONS`、`V2_ACTIONS`。其他地方不得再定义。
- 完成后运行你负责的测试文件，并在最后运行一次 `python -m pytest -q`，在最终回复里原样贴出末尾几行。

## 1. 选择器接口与统计设计（`src/omniaware/arbitration.py`）

### 1.1 数据约定
- 一道题的 `actions: dict[str, dict]`，键是动作名。每个动作记录含：
  `answer: str`、`available: bool`（缺省视为 True；模态候选池为空时为 False）、
  `first_token: {"margin": float, "entropy": float, "entropy_norm": float}`、
  `seq_logprob: float | None`、`gen_len: int`、`kl_to_direct: float | None`、
  `retriever: {"top1_score": float | None, ...}`。
- 校准/评测行：`{"qid": str, "actions": {...}, "scores": {action: float}}`，分数是官方指标，量纲由调用方决定（0–100）。
- 不可用的动作（`available=False`）在任何策略里都不能被选中。

### 1.2 基本量
- `best_single_action(scores_by_action) -> str`：校准集上平均分最高的动作，记为 a*。并列按 `V2_ACTIONS` 顺序。
- `best_evidence_action(scores_by_action) -> str`：证据动作里平均分最高者，记为 e*。
- `normalize_answer(text) -> list[str]`：小写、去标点、按空白切词。`token_f1(a, b)` 基于它计算。
- 置信度 `confidence(record, kind)`：`kind="margin"` 返回 `first_token.margin`；`kind="neg_entropy"` 返回 `-first_token.entropy`。

### 1.3 v2-Rule（零训练，一维阈值族）
策略由 `(evidence_choice, tau)` 决定：
- 若直答的置信度 `confidence(direct) >= tau`，选 `direct`；
- 否则选证据动作：`evidence_choice="best"` 时选 e*（若该题 e* 不可用，选可用证据动作中校准均分最高者；都不可用则选 direct）；`evidence_choice="confident"` 时选该题可用证据动作中置信度最高者。
- `tau = -inf` 即“永远直答”，`tau = +inf` 即“永远用证据动作”。

阈值网格：取校准集上直答置信度的分位数（默认 0, 5%, …, 100% 共 21 个），再加上 ±inf。网格只依赖校准集的信号分布，不依赖标签。

**固定序列的顺序（关键）**：必须从“等价于 a*”的那一端开始，逐步远离。
- a* 是 direct：tau 从 −inf 递增（越来越多的题改用证据）。
- a* 是证据动作：tau 从 +inf 递减（越来越多的题改用直答）。此时 `evidence_choice="best"` 的起点策略恰好等于 a*（要求 e* == a*，证据动作为 a* 时必然成立）。
- `evidence_choice="confident"` 族的起点不一定等于 a*，照样从同一端开始检验，起点通不过则整族被否决。
- 两个族各自做固定序列检验，各用 `delta / 2`（Bonferroni）。

### 1.4 风险控制（对平均效用做单侧检验）
- 配对差 `d_i = score_policy(i) − score_{a*}(i)`，校准集上计算。
- 容忍度 `epsilon >= 0`（与分数同量纲，由调用方传入；0–100 量纲下默认 1.0）。
- 原假设 H0：`E[d] <= −epsilon`。检验：单侧配对 t 检验，统计量 `t = (mean(d) + epsilon) / (std(d, ddof=1) / sqrt(n))`，p 值取 t 分布上尾（用 `math`/`numpy` 实现，可用正态近似 n≥30 时，或实现 t 分布的生存函数；不要依赖 scipy）。`std == 0` 时：`mean(d) > −epsilon` 则 p=0，否则 p=1。
- 固定序列：按 1.3 的顺序逐个检验，`p <= level` 则接受并继续，否则停止，后面的全部不接受。
- 在被接受的候选里选校准集平均效用 `mean(d)` 最大的；并列取更靠近起点的。若没有被接受的候选，或最大效用 <= 0，返回回退决定：永远选 a*。
- 文档字符串里如实写明保证的形式和假设：在题目 i.i.d.、t 检验的渐近有效性下，以至少 `1 − delta` 的概率，被选中的策略满足 `E[score_policy] >= E[score_{a*}] − epsilon`；a* 是在同一份校准集上选出的，这会带来轻微的乐观偏差。不要写成“永远不差”。
- `alpha`（风险水平）这个概念在本设计里不存在，只有 `delta` 和 `epsilon` 两个参数，不要把它们混为一个。

### 1.5 v2-Cal（轻量校准，预测效用差）
- 对每个非 a* 的动作 b，单独拟合一个岭回归，预测 `score_b − score_{a*}`。特征（每个动作一组）：
  `[1, conf(direct), conf(a*), conf(b) − conf(a*), token_f1(answer_b, answer_{a*}), kl_to_direct(b), retriever.top1_score(b)]`
- 缺失处理：某个特征在拟合集上全部缺失或方差为 0 时，丢弃这一列并在模型里记录被丢弃的列名；某个特征只在部分行缺失时抛错。**不允许把缺失填成 0。**
- 不可用的动作行不参与拟合；预测时不可用的动作不参与比较。
- 策略带一个阈值 `lam`：该题上预测增益最大的可用动作 b̂，若预测增益 `> lam` 则选 b̂，否则选 a*。`lam = +inf` 即“永远 a*”。
- 网格：拟合集上预测增益的分位数，从 +inf 递减。
- **校准集对半拆分**（固定种子）：前一半拟合回归，后一半做 1.4 的风险控制。v2-Cal 也必须经过风险控制。
- 模型字典里报告每个动作的参数个数和总参数个数。

### 1.6 需要导出的函数（供其他模块调用，签名固定）
```python
def best_single_action(scores_by_action: dict[str, Sequence[float]]) -> str
def best_evidence_action(scores_by_action: dict[str, Sequence[float]]) -> str
def normalize_answer(text: str) -> list[str]
def token_f1(a: str, b: str) -> float
def apply_decision(decision: dict, actions: dict) -> dict        # -> {"action": str, "answer": str}
def select_rule(calib_rows: list[dict], delta: float = 0.1, epsilon: float = 1.0, confidence: str = "margin") -> dict
def select_cal(calib_rows: list[dict], delta: float = 0.1, epsilon: float = 1.0, seed: int = 0, confidence: str = "margin") -> dict
def policy_scores(decision: dict, rows: list[dict]) -> list[float]   # 每题：被选中动作的分数
```
- `select_rule` / `select_cal` 返回的 `decision` 是可 JSON 序列化的字典，至少包含：`kind`（"rule" / "cal" / "single"）、`best_single`、`fallback: bool`、`delta`、`epsilon`、`confidence`、`calib_utility`、`n_calib`、被接受的候选个数，以及应用该决定所需的全部参数（rule：`evidence_choice`、`tau`、`best_evidence`、各证据动作的校准均分排序；cal：各动作的权重、保留的特征名、`lam`）。回退时 `kind="single"`、`fallback=True`、`action=a*`。
- `apply_decision` 只读取 `decision` 和该题的 `actions`，**不读分数，不读 `gold_in_topk`**。
- 对半拆分、qid 不交检查这类通用工具不放在本模块里（放到调用它的脚本，或 `omniaware/data.py` 已有的工具）。

### 1.7 测试（`tests/test_arbitration.py`，重写）
合成数据，至少覆盖：
1. 直答置信度完美区分“直答对 / 证据对”时，`select_rule` 的评测集得分接近逐题上限，且不回退。
2. 信号为纯噪声时回退到 a*（`fallback=True`）。
3. a* 是证据动作时不会在第一步就失败（即 A-2 的回归测试）：构造 a*=static_mixed、直答置信度有信息量的数据，断言选出的策略在评测集上优于 a*。
4. 一个“在 30% 的题上变差、在 50% 的题上变好”的策略能被接受（A-1 的回归测试）。
5. `epsilon`、`delta` 是两个独立参数：改变其一不影响另一个的语义（例如 epsilon 变大时被接受的候选数不减少）。
6. v2-Cal：缺失特征被丢弃并记录；部分缺失抛错；不可用动作不会被选中；拟合与风险控制用的是不交的两半。
7. `token_f1("Wood.", "wood") == 1.0`。
8. `apply_decision` 的输入里即使有 `scores` 和 `gold_in_topk` 也不影响输出。

## 2. 逐题动作表（各脚本之间的唯一交换格式）
文件 `<run_dir>/<dataset>.v2_actions.jsonl`，每行：
```json
{"qid": "...", "subset": "image_question|text_question|null", "actions": {"direct": {...}, ...}, "scores": {"direct": 62.5, ...}, "metric": "acc_approx|vqa_accuracy"}
```
由 `scripts/analysis/v2_table.py build` 从 `omniaware.run` 的输出生成（见第 4 节）。`v2_gate`、`dev_select`、测试阶段的应用都只读这个文件。

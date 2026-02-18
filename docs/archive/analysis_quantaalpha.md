# QuantaAlpha 对当前项目的启发分析（Phase 1 / CSI1000）

## 1. 目标与范围（短）
本文档只做决策分析，不改代码、不改配置。  
重点是回答一个问题：**QuantaAlpha 的做法，哪些值得你当前项目在 Phase 1（CSI1000、单因子显著性、FDR<0.01）采纳，哪些不该照搬**。

边界说明：
- 本文结论基于源码与文档静态分析，不包含对 QuantaAlpha 的复现实验。
- 重点偏“你当前项目启发”（约 80%），仅保留必要的 QuantaAlpha 机制背景（约 20%）。

---

## 2. QuantaAlpha 关键逻辑速览（短）
QuantaAlpha 主链路可以概括为：
1. 输入研究方向并做并行方向规划（planning）。
2. 在 `original -> mutation -> crossover` 轨迹进化框架下运行多轮 5-step Loop。
3. 每轮固定 5 步：`hypothesis -> factor expression -> factor calculate -> backtest -> feedback`。
4. 用表达式质量门控（可解析、复杂度、冗余）减少无效因子。
5. 将每轮结果入统一因子库（JSON）并支持后续组合回测。

对应核心源码入口可见：
- `quantaalpha/pipeline/factor_mining.py`
- `quantaalpha/pipeline/loop.py`
- `quantaalpha/factors/proposal.py`
- `quantaalpha/factors/regulator/factor_regulator.py`
- `quantaalpha/backtest/runner.py`

---

## 3. 对当前项目的核心启发（主体）

### 启发 1：HEA Loop 标准化（减少轮次漂移）
问题/机会：  
你当前已有 HEA 思路，但执行层还偏“脚本驱动 + 人工组织”，轮次之间信息结构容易不一致，导致复盘成本上升。

QuantaAlpha 做法：  
把每轮流程固定为 5-step，并且每轮都形成结构化产物（假设、表达式、回测、反馈、轨迹元数据）。

对你项目的映射：  
- `/Users/jacobdu/Workspece/personal/agentic-alpha/AGENTS.md`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/test_factor_ic.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/test_new_factors.py`

建议动作：  
在不改代码前提下，先把每轮实验记录模板固化为统一 markdown/表格字段（HEA 编号、假设、表达式集合、统计结论、Promote/Iterate/Drop 决策）。

预期收益：  
- 减少“同类实验重复做”的概率。  
- 降低从结论回溯证据的时间。  
- 让后续 Top-N 组合阶段接入更顺滑。

风险与边界：  
如果模板字段过多，会增加记录负担并拖慢迭代；建议先最小字段集。  
不照搬条件：不需要复制 QuantaAlpha 的完整 Loop 框架，只需要复制其“结构化产出约束”。

### 启发 2：质量门控前移（先过滤再回测）
问题/机会：  
当前流程中，部分表达式在进入大规模评估前缺少“复杂度/冗余”预筛，容易浪费算力在低质量候选上。

QuantaAlpha 做法：  
在表达式进入回测前做三类约束：可解析性、复杂度（符号长度/基础特征数/自由参数比例）、重复子树冗余。

对你项目的映射：  
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/test_new_factors.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/test_factor_ic.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/src/project_qlib/factor_db.py`

建议动作：  
在“候选因子进入 IC 计算前”增加一层离线门控清单（可解析/复杂度/重复度），先做人工审查版，不急于自动化。

预期收益：  
- 减少无效因子评估批次。  
- 提高每轮可解释性与通过率。  
- 降低过拟合表达式进入 Top-N 候选的概率。

风险与边界：  
门槛过严会误杀有价值因子。建议先“软门控”（标记风险而非直接丢弃）。  
不照搬条件：不建议直接照抄其阈值（如长度阈值），应按你市场和样本期重新校准。

### 启发 3：轨迹化研究记录（不是只存最终指标）
问题/机会：  
你的因子库已很强（四张表），但“中间过程上下文”仍可能分散在日志/脚本输出中，影响复现实验与错误归因。

QuantaAlpha 做法：  
为每次探索维护 trajectory：记录父子关系、轮次、阶段（original/mutation/crossover）、反馈摘要与主指标。

对你项目的映射：  
- `/Users/jacobdu/Workspece/personal/agentic-alpha/src/project_qlib/factor_db.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/factor_db_cli.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/verify_all.py`

建议动作：  
基于你现有 DB，不新增复杂表，先补“实验轮次元信息字段规范”（hea_round、evidence、decision_reason、parent_factor_group）。

预期收益：  
- 从“结果数据库”升级为“研究过程数据库”。  
- 更快定位为什么某轮 Iterate 或 Drop。  
- 便于未来自动化报告与经验复用。

风险与边界：  
元数据字段失控会导致填报质量下降。  
不照搬条件：不需要复制其 JSON 轨迹池机制，你已有 SQLite，优先沿用现体系。

### 启发 4：评估分层（单因子信号与组合收益分开判）
问题/机会：  
项目已强调单因子优先，但实际讨论中仍容易把“单因子显著性”与“模型组合收益”混为同一层结论。

QuantaAlpha 做法：  
把挖掘阶段反馈指标与最终组合回测指标拆分；前者用于方向引导，后者用于真实策略评价。

对你项目的映射：  
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/test_factor_ic.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/run_official.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/run_custom_factor.py`

建议动作：  
在每轮结论中强制输出“双层结论”：  
`Layer A: 单因子统计显著性`；`Layer B: 组合后边际贡献（含成本）`。  
若 B 未验证，只允许给 A 层结论，不提前 Promote 到组合层。

预期收益：  
- 降低“统计显著但策略无增益”的误判。  
- 防止过早把候选因子推进到组合池。  
- 让 Phase 1 到 Phase 2 的衔接更可控。

风险与边界：  
流程会变得更“慢但稳”。  
不照搬条件：不用复制其全部回测模块，只需复制“评估口径分层”原则。

### 启发 5：LLM 使用边界（让 LLM 做提案，不做裁判）
问题/机会：  
当前你在因子研究中已具备较强统计与回测流程，LLM 若使用不当，容易把“主观解释”误当“客观验证”。

QuantaAlpha 做法：  
LLM 主要负责：方向规划、假设生成、表达式提案、反馈总结；真值由回测与统计指标裁定。

对你项目的映射：  
- `/Users/jacobdu/Workspece/personal/agentic-alpha/AGENTS.md`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/test_factor_ic.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/src/project_qlib/factor_db.py`

建议动作：  
为 LLM 定位三件事：  
1) 生成候选假设；2) 生成候选表达式草案；3) 生成复盘摘要。  
把“显著性判断、是否入库、是否 Promote”全部保留在统计规则中。

预期收益：  
- 用 LLM 提升搜索效率，同时不牺牲严谨性。  
- 降低“叙事驱动”替代“证据驱动”的风险。

风险与边界：  
如果给 LLM 过高自由度，会引入复杂但低鲁棒表达式。  
不照搬条件：不建议在你项目里直接引入“LLM 打分决定是否替换 SOTA”。

### 启发 6：因子库治理（从“存结果”到“存决策依据”）
问题/机会：  
你已有四表结构是优势，但“决策依据”字段还可以更系统化，便于跨轮次比较。

QuantaAlpha 做法：  
在因子条目中保留较完整的上下文（假设、反馈、回测摘要、阶段）。

对你项目的映射：  
- `/Users/jacobdu/Workspece/personal/agentic-alpha/src/project_qlib/factor_db.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/factor_db_cli.py`
- `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/import_factors_to_db.py`

建议动作：  
在现有入库流程中统一写入：  
`decision_basis`（为何 Accepted/Rejected）、`failure_mode`（失效类型）、`applicability`（适用市场/周期）、`next_action`（Iterate方向）。

预期收益：  
- 后续做 Top-N 组合时不只看分数，还能看“失效相关性”。  
- 提升跨市场迁移分析质量（csi1000 -> csiall/csi300）。

风险与边界：  
字段定义不统一会导致“有字段但不可用”。  
不照搬条件：无需照搬 QuantaAlpha 的 JSON 结构，保持你 SQLite 模型简洁优先。

---

## 4. 优先级建议（P0 / P1 / P2）

### P0（本周即可启动，不改代码）
1. 固化 HEA 记录模板，并要求每轮产出双层结论（单因子层/组合层）。  
2. 在候选评估前增加“软质量门控”清单（可解析、复杂度、冗余）。  
3. 因子决策时强制填写 `decision_basis` 与 `failure_mode`。

### P1（下周推进，仍可不改核心代码）
1. 在 `outputs/` 侧建立“每轮汇总页”，统一链接到 DB 证据与图表。  
2. 对现有 Accepted 因子补齐适用范围与失效条件。  
3. 为 Top-N 候选增加“低相关+不同失效模式”检查。

### P2（后续可选）
1. 再考虑引入自动化轨迹管理（若手工流程已成为瓶颈）。  
2. 再考虑引入 LLM 辅助提案流水线（前提是质量门控稳定运行）。

---

## 5. 风险与不建议照搬项
1. 不建议照搬 QuantaAlpha 的完整工程框架。  
原因：你当前项目已有清晰的单因子统计与因子库体系，整套迁移会引入高复杂度与维护负担。

2. 不建议让 LLM 参与“最终显著性判定”。  
原因：统计显著性与 FDR 判定必须保持可重复的规则系统。

3. 不建议直接复用其复杂度阈值和选择策略。  
原因：阈值强依赖市场、样本期、特征空间，你项目应做本地校准。

4. 不建议过早引入 mutation/crossover 自动进化。  
原因：在 Phase 1，优先提升“单因子证据质量”，先把基础筛选闭环做扎实。

---

## 6. 两周内可执行的最小行动清单（不改代码版）

### 第 1 周：标准化与门控
1. 建立 HEA 轮次模板并应用到所有新实验记录。  
2. 每轮实验前执行“软门控清单”并记录结果。  
3. 对过去两周因子决策补录 `decision_basis` 与 `failure_mode`。

交付物：
- 一份标准化 HEA 记录样例（含 Promote/Iterate/Drop）。  
- 一份候选因子门控台账。  
- 一份补录后的决策依据列表。

### 第 2 周：评估分层与组合前治理
1. 把 Top 候选按“单因子显著性层”与“组合验证层”分池。  
2. 对组合候选做“低相关 + 失效模式分散”人工筛查。  
3. 输出“下一轮优先实验方向”清单（不超过 10 条）。

交付物：
- 双层候选池清单。  
- 组合候选风险说明（何时失效、为何保留）。  
- 下一轮实验 backlog。

---

## 7. 证据映射（源码路径对照）

### 你当前项目（映射锚点）
1. `/Users/jacobdu/Workspece/personal/agentic-alpha/AGENTS.md`  
2. `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/test_factor_ic.py`  
3. `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/test_new_factors.py`  
4. `/Users/jacobdu/Workspece/personal/agentic-alpha/src/project_qlib/factor_db.py`  
5. `/Users/jacobdu/Workspece/personal/agentic-alpha/scripts/factor_db_cli.py`  
6. `/Users/jacobdu/Workspece/personal/agentic-alpha/src/project_qlib/workflow.py`  

### QuantaAlpha（对标证据）
1. 项目入口与主流程：  
   [launcher.py](https://github.com/QuantaAlpha/QuantaAlpha/blob/main/launcher.py)  
   [quantaalpha/pipeline/factor_mining.py](https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/pipeline/factor_mining.py)  
   [quantaalpha/pipeline/loop.py](https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/pipeline/loop.py)
2. 假设与表达式生成：  
   [quantaalpha/factors/proposal.py](https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/proposal.py)
3. 质量门控与表达式约束：  
   [quantaalpha/factors/regulator/factor_regulator.py](https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/regulator/factor_regulator.py)
4. 进化控制（original/mutation/crossover）：  
   [quantaalpha/pipeline/evolution/controller.py](https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/pipeline/evolution/controller.py)
5. 组合回测与指标：  
   [quantaalpha/backtest/runner.py](https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/backtest/runner.py)
6. 因子库写入：  
   [quantaalpha/factors/library.py](https://github.com/QuantaAlpha/QuantaAlpha/blob/main/quantaalpha/factors/library.py)

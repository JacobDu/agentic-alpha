# AGENTS.md

本文件只维护三类信息：
1. 核心工作流编排（R-G-E-D）
2. 全局硬门槛规则
3. 经验记忆与脚本治理

## 🎯 项目目标

通过 Agent 持续研究循环，沉淀可复用的高质量因子资产库。
默认市场：`csi1000`。

## 核心工作流（R-G-E-D）

`Retrieve -> Generate -> Evaluate -> Distill`

### 单因子挖掘（SFA）
1. Retrieve：调用 `$qlib-env-data-prep` 完成环境门禁；检索 `data/factor_library.db` 因子总表、历史 SFA 记录、经验记忆。
2. Generate：调用 `$qlib-single-factor-mining` 生成候选表达式，优先遵循“推荐方向”，避开“禁止方向”。
3. Evaluate：调用 `$qlib-single-factor-mining` 执行预检、显著性检验与相关性预算约束。
4. Distill：写入 `docs/workflows/single-factor/` 与 `data/factor_library.db`，并更新本文件经验记忆。

### 多因子组合（MFA）
1. Retrieve：调用 `$qlib-env-data-prep` 完成环境门禁；检索候选因子池、稳定因子集、历史 MFA 记录。
2. Generate：调用 `$qlib-multi-factor-backtest` 构造线性与非线性组合方案。
3. Evaluate：调用 `$qlib-multi-factor-backtest` 执行统一区间回测，输出含成本收益/风险指标与压力测试结果。
4. Distill：写入 `docs/workflows/multi-factor/` 与 `data/factor_library.db`，并更新本文件经验记忆。

## 全局硬门槛

### 决策集
仅允许：`Promote / Iterate / Drop`。

### SFA 显著性门槛
- `fdr_p < 0.01`
- `|rank_icir| >= 0.10`
- 证据完整（doc/output/db_query/run_id 至少一项）

### 正交性预算
- `max|rho| <= 0.50`

### 替换门槛（高相似因子）
- 触发条件：相关性 `> 0.80`
- 允许替换条件：新因子 `|ICIR|` 相对旧因子提升 `>= 20%`

## 因子资产与记录

1. 因子总表：`factors`（位于 `data/factor_library.db`）必须长期保留，记录所有因子定义与状态。
2. 结果表：`factor_test_results`、`factor_backtest_results`、`factor_ic_decay`。
3. 工作流表：`workflow_*`（SFA/MFA 记录、决策、证据、相似度、替换链路）。
4. 文档目录：
   - SFA：`docs/workflows/single-factor/`
   - MFA：`docs/workflows/multi-factor/`
   - 历史：`docs/heas/`（仅历史回填来源）

## 经验记忆（维护在本文件）

维护原则：
1. 仅记录可复用的结构化经验，不记录一次性日志。
2. 每次 Distill 最多新增 3 条，超过容量时按时间滚动淘汰旧项。
3. 每类最多保留 `50` 条。

### 推荐方向（最多50条）
1. 在 `csi1000` 多因子组合中，优先测试 `hold_thresh=40` 的低换手配置；当候选池为“最新显著Top30”时，交易成本下降带来的净值改善显著。
2. 线性加权（按 `rank_icir` 方向与绝对值权重）应作为 MFA 基准组合长期保留，用于快速筛掉退化的非线性配置。

### 禁止方向（最多50条）
1. 当候选池切换后，不要默认沿用旧模型结论；`LGB/Ensemble` 可能在 `hold<=20` 出现 IR 为负，必须先做统一口径回测再决策。

## 脚本治理

1. Skill 内 `scripts/` 只允许保留可复用脚本。
2. 单次研究临时脚本必须放在项目根目录 `./scripts/`。
3. 临时脚本使用完成后必须删除，不得沉淀到 `.agents/skills/*/scripts/`。
4. `__pycache__`、`.pyc` 不得留在 skills 目录。
5. 如临时脚本被复用 >=2 次，才评估升格为 skill 可复用脚本。

## Agent 行为约束

1. 不得编造数据、指标、run_id。
2. 任何结论必须可追溯到真实文件或数据库记录。
3. 若环境异常，先修复流程可用性，再继续因子研究。

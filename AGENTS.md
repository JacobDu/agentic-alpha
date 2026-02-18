# AGENTS.md

本文件维护本项目的核心工作流、判定标准、经验教训与 skill 路由规则。

## 🎯 项目目标

通过持续运行 Agent 驱动的研究循环，挖掘并验证有价值的量化因子（factor），最终形成可复用的高质量因子资产库。

默认市场为 `csi1000`（中证1000）。优先验证单因子预测能力，再推进 Top-N 多因子组合验证。

## 核心工作流

### 单因子挖掘工作流（SFA）

1. Hypothesis
- 生成轮次编号：`SFA-YYYY-MM-DD-XX`（当日递增两位序号）。
- 写明因子表达式、市场逻辑、预期方向。

2. Preflight Gate
- 调用 `$qlib-env-data-prep` 完成环境与数据门禁。
- 检查并记录：`parse_ok`、`complexity_level`、`redundancy_flag`、`data_availability`。

3. Experiment
- 调用 `$qlib-single-factor-mining` 执行单因子实验：
  - `uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_ic.py`
  - `uv run python .agents/skills/qlib-single-factor-mining/scripts/test_new_factors.py`
  - `uv run python .agents/skills/qlib-single-factor-mining/scripts/test_new_factor_batch.py`
  - `uv run python .agents/skills/qlib-single-factor-mining/scripts/test_ortho_factor_batch.py`
  - `uv run python .agents/skills/qlib-single-factor-mining/scripts/test_composite_factors.py`

4. Analysis
- 必填单因子指标：`IC/RankIC/ICIR/RankICIR/t/FDR`。

5. Decision
- 仅允许：`Promote / Iterate / Drop`。
- 结论必须引用门控 + 单因子指标证据。

6. Archive
- 单轮记录：`docs/workflows/single-factor/SFA-YYYY-MM-DD-XX.md`
- 索引更新：`docs/workflows/single-factor/INDEX.md`
- DB 留痕：`data/factor_library.db`（`workflow_*` 表 + `hea_round/evidence/notes` 兼容字段）
- 记录脚本：`uv run python .agents/skills/qlib-single-factor-mining/scripts/sfa_record_cli.py ...`

### 多因子组合工作流（MFA）

1. Hypothesis
- 生成轮次编号：`MFA-YYYY-MM-DD-XX`（当日递增两位序号）。
- 写明因子池来源、组合逻辑、收益风险目标。

2. Preflight Gate
- 调用 `$qlib-env-data-prep` 完成环境与数据门禁。
- 检查并记录：`parse_ok`、`complexity_level`、`redundancy_flag`、`data_availability`。

3. Experiment
- 调用 `$qlib-multi-factor-backtest` 执行多因子组合实验：
  - `uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_phase2_comparison.py`
  - `uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_topn_comparison.py`
  - `uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_optimization.py`

4. Analysis
- 必填多因子指标：`excess_return_with_cost`、`ir_with_cost`、`max_drawdown`。

5. Decision
- 仅允许：`Promote / Iterate / Drop`。
- 结论必须引用门控 + 多因子指标证据。

6. Archive
- 单轮记录：`docs/workflows/multi-factor/MFA-YYYY-MM-DD-XX.md`
- 索引更新：`docs/workflows/multi-factor/INDEX.md`
- DB 留痕：`data/factor_library.db`（`workflow_*` 表 + `hea_round/evidence/notes` 兼容字段）
- 记录脚本：`uv run python .agents/skills/qlib-multi-factor-backtest/scripts/mfa_record_cli.py ...`

## 记录模板来源

- 单因子模板：`.agents/skills/qlib-single-factor-mining/assets/templates/single_factor_experiment_record.md`
- 多因子模板：`.agents/skills/qlib-multi-factor-backtest/assets/templates/multi_factor_experiment_record.md`

历史 `docs/heas/` 保留，不删除；可通过回填脚本写入 `workflow_*` 新表。

## Skill 路由

1. 环境与数据准备 -> `$qlib-env-data-prep`
2. 单因子挖掘 -> `$qlib-single-factor-mining`
3. 多因子组合 -> `$qlib-multi-factor-backtest`

默认顺序：`$qlib-env-data-prep` -> (`$qlib-single-factor-mining` 或 `$qlib-multi-factor-backtest`) -> 归档到 docs 与数据库。

## Agent 行为约束

1. 不得编造数据、指标、run_id。
2. 任何结论必须可追溯到真实文件或数据库记录。
3. 若环境异常，先修复流程可用性，再继续因子挖掘。

## 经验速记

1. 单因子有信号不等于组合一定提升，必须看含成本结果。
2. macOS 下 mlflow 崩溃时，优先使用项目内 mlflow 启动脚本。
3. CSI1000 上常见显著负 RankIC（反转逻辑），均值回归更强。
4. 因子数量并非越多越好，过多会引入噪声退化。
5. Qlib 表达式引擎不支持一元 `-`，需用 `(0 - expr)`。
6. CSI1000 的可预测性显著强于 CSI300。
7. 低波动/均值回归类因子在本项目长期有效。
8. `hold_thresh` 对含成本收益影响通常最大。
9. 数据标准口径必须统一并定期交叉验证。
10. 关键结论必须双写入文档与因子库证据字段。

---
name: qlib-multi-factor-backtest
description: 负责 Top-N 多因子训练与含交易成本组合回测。用于 Phase 2/3 对比、参数优化、组合风险收益评估以及多因子证据产出。
---

# Qlib 多因子组合回测（MFA）

本 skill 负责 MFA 子工作流，按 `Retrieve -> Generate -> Evaluate -> Distill` 执行。

## 输入与输出

- 输入：因子池、模型配置、回测区间与交易参数。
- 输出：MFA 指标（含成本收益/IR/回撤）、决策、证据与文档。

## 默认时序切分（csi1000）

- Train：`2000-01-04 ~ 2023-12-31`
- Valid：`2024-01-01 ~ 2024-12-31`
- Test（OOS）：`2025-01-01 ~ 数据最新可用日`
- 目标：在长历史训练基础上，用最近一年验证超参数与模型选择，以提升对近期市场风格的适配。

## 指标命名标准

- 统一使用 `docs/METRIC_STANDARD_V1.md` 与 `src/project_qlib/metrics_standard.py`。
- 优先输出 canonical 字段：
  - `excess_return_annualized_with_cost`
  - `information_ratio_with_cost`
  - `max_drawdown_with_cost`
  - `excess_return_daily_with_cost`
  - `ic_mean` / `rank_ic_mean` / `ic_ir` / `rank_ic_ir`
- 旧字段名（如 `IR_with_cost`、`ann_ret_with_cost`）仅保留兼容读取，不作为新增文档主口径。

## R-G-E-D 执行定义

### 1) Retrieve
1. 从因子总表与 workflow 记录读取候选池与稳定因子集。
2. 明确本轮目标（增益、降回撤、降换手、稳健性）。

### 2) Generate
1. 构造至少两类组合器方案：线性 + 非线性。
2. 固定同一训练/验证/测试切分，保证可比性。

### 3) Evaluate
1. 统一回测口径输出 `excess_return_annualized_with_cost`、`information_ratio_with_cost`、`max_drawdown_with_cost`，并补充 `excess_return_daily_with_cost`。
2. 执行交易成本压力测试（不同成本/调仓强度参数）。
3. 判定是否相对基线产生稳定增量，而非单次偶然提升。

### 4) Distill
1. 用 `scripts/mfa_record_cli.py` 写入工作流记录与证据。
2. 同步文档到 `docs/workflows/multi-factor/` 并更新 `INDEX.md`。
3. 将有效组合模式回写 `AGENTS.md` 经验记忆。

## 线性/非线性对照要求

1. 线性方案：Lasso/线性加权类至少一套。
2. 非线性方案：树模型（如 LightGBM/XGBoost）至少一套。
3. 两类方案均需报告含成本指标，缺一不可。

## 可复用脚本

### 训练与对比
- `scripts/run_phase2_comparison.py`
- `scripts/run_topn_comparison.py`
- `scripts/run_optimization.py`
- `scripts/train_csi1000_lite.py`
- `scripts/train_csiall_lite.py`
- `scripts/train_csiall_topn.py`

### 可视化与记录
- `scripts/visualize_factors.py`
- `scripts/mfa_record_cli.py`

## 记录模板

- `assets/templates/multi_factor_experiment_record.md`

## 证据要求

- 至少包含一个可追溯证据：`output_path` / `db_query` / `run_id` / `doc`。
- `decision` 仅允许 `Promote / Iterate / Drop`。

## 临时脚本边界

1. 本 skill 的 `scripts/` 只保留可复用脚本。
2. 一次性参数扫描/临时对比脚本必须放在项目根目录 `./scripts/`。
3. 临时脚本完成后必须清理，不得沉淀到 skill。
4. skills 下不允许存在 `__pycache__` / `.pyc`。

## 参考资料

- `references/layer_b_metrics.md`（历史资料，按 MFA 口径解释）

# Metric Standard v1

本文件定义本仓库的统一指标术语与字段命名（`metric_schema_version = "v1"`）。

## 1. 日超额 vs 年化超额（核心区别）

- 日超额收益（含成本）：
  - `excess_return_daily_with_cost_t = return_t - bench_t - cost_t`
- 日超额收益（不含成本）：
  - `excess_return_daily_no_cost_t = return_t - bench_t`

- 年化超额收益（含成本）：
  - `excess_return_annualized_with_cost = Annualize(excess_return_daily_with_cost_t)`
- 年化超额收益（不含成本）：
  - `excess_return_annualized_no_cost = Annualize(excess_return_daily_no_cost_t)`

直观理解：
- `daily` 是“每天平均赢/亏多少”
- `annualized` 是把日序列按年化尺度（常用252交易日）折算后的年水平

## 2. Canonical 字段（统一输出）

### 信号质量
- `ic_mean`
- `ic_ir`
- `rank_ic_mean`
- `rank_ic_ir`
- `n_days`

### 组合表现
- `excess_return_daily_no_cost`
- `excess_return_daily_with_cost`
- `excess_return_annualized_no_cost`
- `excess_return_annualized_with_cost`
- `information_ratio_no_cost`
- `information_ratio_with_cost`
- `max_drawdown_no_cost`
- `max_drawdown_with_cost`
- `daily_turnover`
- `total_cost_pct`
- `benchmark_return_annualized`

## 3. 历史字段映射（向后兼容）

- `IR_with_cost` / `ir_with_cost` -> `information_ratio_with_cost`
- `ann_ret_with_cost` / `excess_ann_ret_with_cost` / `excess_return_with_cost` -> `excess_return_annualized_with_cost`
- `max_dd_with_cost` / `max_drawdown` -> `max_drawdown_with_cost`
- `IR_no_cost` -> `information_ratio_no_cost`
- `ann_ret_no_cost` / `excess_ann_ret_no_cost` -> `excess_return_annualized_no_cost`
- `max_dd_no_cost` -> `max_drawdown_no_cost`
- `IC` -> `ic_mean`
- `ICIR` -> `ic_ir`
- `Rank_IC` / `Rank IC` -> `rank_ic_mean`
- `Rank_ICIR` / `Rank ICIR` / `rank_icir` -> `rank_ic_ir`

## 4. 因子 IC 参考值（基于本库 csi1000 实际分布）

样本口径：
- 表：`factor_test_results`
- 条件：`market='csi1000' and significant=1`
- 样本数：`322`

复现命令（项目根目录）：
```bash
uv run python - <<'PY'
import sqlite3, pandas as pd
conn = sqlite3.connect("data/factor_library.db")
df = pd.read_sql_query(
    "select rank_ic_mean, rank_icir, rank_ic_t from factor_test_results "
    "where market='csi1000' and significant=1",
    conn,
)
for col in ["rank_ic_mean", "rank_icir", "rank_ic_t"]:
    q = df[col].abs().quantile([0.25, 0.5, 0.75, 0.9])
    print(col, q.to_dict())
PY
```

绝对值分位数（经验参考）：

### `|rank_ic_mean|`
- P25: `0.0145`
- P50: `0.0223`
- P75: `0.0282`
- P90: `0.0336`

### `|rank_ic_ir|`
- P25: `0.1408`
- P50: `0.1817`
- P75: `0.2399`
- P90: `0.2844`

### `|rank_ic_t|`
- P25: `5.33`
- P50: `7.19`
- P75: `9.33`
- P90: `11.34`

可操作分档（csi1000，单因子）：
- 弱：`|rank_ic_mean| < 0.01` 或 `|rank_ic_ir| < 0.10`
- 中：`|rank_ic_mean| in [0.01, 0.02)` 且 `|rank_ic_ir| in [0.10, 0.18)`
- 良：`|rank_ic_mean| in [0.02, 0.03)` 且 `|rank_ic_ir| in [0.18, 0.28)`
- 优：`|rank_ic_mean| >= 0.03` 或 `|rank_ic_ir| >= 0.28`

说明：
- 这是“经验参考区间”，不是替代显著性检验
- 最终仍以门槛与回测结果联合决策（例如 `fdr_p < 0.01`、成本后 IR）

## 5. 已接入脚本（本次）

- `.agents/skills/qlib-multi-factor-backtest/scripts/run_optimization.py`
- `.agents/skills/qlib-multi-factor-backtest/scripts/run_topn_comparison.py`
- `.agents/skills/qlib-multi-factor-backtest/scripts/run_phase2_comparison.py`

以上脚本现已通过 `src/project_qlib/metrics_standard.py` 统一指标字段。

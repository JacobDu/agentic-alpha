# 术语与指标说明（MFA/SFA）

本文档解释本项目常见术语、计算口径与字段含义，重点覆盖 `IR`、`excess` 等核心指标。
统一字段命名规范请以 `docs/METRIC_STANDARD_V1.md` 为准。

## 1. 收益与风险类指标（回测）

### 1.1 超额收益（Excess Return）

- 日度超额收益（不含成本）：
  - `excess_no_cost_t = return_t - bench_t`
- 日度超额收益（含成本）：
  - `excess_with_cost_t = return_t - bench_t - cost_t`

其中：
- `return_t`：策略组合当日收益
- `bench_t`：基准当日收益（本项目默认 `SH000852`）
- `cost_t`：交易成本（由开仓费率、平仓费率、最小费用等决定）

本项目脚本中的核心口径与上式一致：
- `excess_with_cost = report["return"] - report["bench"] - report["cost"]`

### 1.2 年化超额收益（`excess_return_with_cost`）

- 输入序列：`excess_with_cost_t`
- 输出：将日度超额收益序列做年化后的结果（字段名常见为 `annualized_return` 或 `excess_return_with_cost`）
- 交易日年化尺度通常按 `252` 天

### 1.3 信息比率（IR, Information Ratio）

- 定义：超额收益的“单位波动补偿”
- 常见表达：
  - `IR = Annualized(Excess Return) / Annualized(Tracking Error)`
- 在项目输出里：
  - `ir_with_cost`：基于 `excess_with_cost_t` 计算
  - `IR_no_cost`：基于 `excess_no_cost_t` 计算

解读：
- `IR > 0`：风险调整后跑赢基准
- `IR` 越高：同等风险下超额越稳定

### 1.4 最大回撤（Max Drawdown, `max_drawdown`）

- 设净值曲线为 `V_t`，历史峰值为 `Peak_t = max(V_1...V_t)`
- 回撤：`DD_t = V_t / Peak_t - 1`
- 最大回撤：`MDD = min(DD_t)`

解读：
- 数值通常为负，越接近 0 越好（例如 `-0.08` 好于 `-0.20`）

### 1.5 换手率（Turnover, `daily_turnover`）

- 表示组合日度调仓强度
- 数值越高，交易摩擦通常越高
- 在高费率环境下，高换手策略的 `excess_with_cost` 和 `IR` 更容易被侵蚀

## 2. 预测质量类指标（IC 系列）

### 2.1 IC（Information Coefficient）

- 在每个交易日的横截面上，计算“预测分数 vs 次日收益标签”的皮尔逊相关系数
- 日度 IC 序列取平均，得到 `IC`

### 2.2 Rank IC

- 与 IC 类似，但相关系数用 Spearman（秩相关）
- 对异常值更稳健，适合排序型策略

### 2.3 ICIR / Rank_ICIR

- 类似 IR 的“稳定性”度量：
  - `ICIR = mean(IC_t) / std(IC_t)`
  - `Rank_ICIR = mean(RankIC_t) / std(RankIC_t)`

解读：
- 绝对值越大，信号跨时间越稳定

## 3. 显著性与筛选指标（SFA 常用）

### 3.1 `rank_ic_t` / `rank_ic_p`

- 对 Rank IC 均值做统计检验得到 t 值与 p 值

### 3.2 `fdr_p`

- 多重检验校正后的 p 值（FDR）
- 本项目 SFA 硬门槛：
  - `fdr_p < 0.01`
  - `|rank_icir| >= 0.10`

## 4. 策略参数术语（TopkDropout）

### 4.1 `topk`

- 组合持仓股票数量（每期维持前 `topk` 个高分标的）

### 4.2 `n_drop`

- 每次调仓最多替换的持仓数量

### 4.3 `hold_thresh`

- 最短持有天数，未达到阈值时不允许卖出

### 4.4 有效换手周期（经验近似）

- `effective_cycle ~= hold_thresh * (topk / n_drop)`

含义：
- `hold_thresh` 越大、`n_drop` 越小，组合越“慢”，成本通常更低

## 5. 成本参数术语

- `open_cost`：开仓费率
- `close_cost`：平仓费率
- `min_cost`：单笔最小交易费用

这些参数直接进入 `cost_t`，进而影响：
- `excess_return_with_cost`
- `ir_with_cost`
- `max_drawdown`（间接）

## 6. 决策术语（Workflow）

项目只允许三种决策：
- `Promote`：达到可推广标准，作为候选默认方案
- `Iterate`：有价值但未达到替换标准，继续迭代
- `Drop`：证据不足或表现不达标，终止该方向

## 7. 字段对照（常见输出）

在 `outputs/*.json` 与 `workflow_*` 表中常见字段：

- `excess_return_with_cost`：含成本年化超额收益
- `ir_with_cost`：含成本信息比率
- `max_drawdown`：最大回撤
- `daily_turnover`：日均换手
- `IC` / `Rank_IC`：截面相关均值
- `ICIR` / `Rank_ICIR`：IC 稳定性指标

## 8. 快速解读建议

- 先看 `ir_with_cost` 和 `excess_return_with_cost`：是否真正“净收益有效”
- 再看 `max_drawdown`：收益是否以过高回撤换来
- 再看 `daily_turnover`：收益是否高度依赖高换手
- 最后看 `IC/Rank_IC/ICIR`：判断信号质量与可迁移性

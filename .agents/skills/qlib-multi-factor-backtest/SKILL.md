---
name: qlib-multi-factor-backtest
description: 负责 Top-N 多因子训练与含交易成本组合回测。用于 Phase 2/3 对比、参数优化、组合风险收益评估以及多因子证据产出。
---

# Qlib 多因子组合回测

负责 MFA 工作流：因子池建模、含成本回测、组合级证据留痕。

## 输入与输出

- 输入：因子池、模型配置、回测区间与交易参数。
- 输出：组合指标（excess_return_with_cost / IR / max_drawdown）、决策与证据链接。

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

## 标准执行顺序

1. 确定因子池与实验参数。
2. 运行训练与含成本回测。
3. 评估组合收益与风险。
4. 用 `mfa_record_cli.py` 写入 MFA 记录并同步索引。

## 记录模板

- `assets/templates/multi_factor_experiment_record.md`

## 证据要求

- 至少包含一个可追溯证据：`output_path` / `db_query` / `run_id` / `doc`。
- `decision` 仅允许 `Promote / Iterate / Drop`。

## 临时脚本边界

1. 本 skill 的 `scripts/` 只保留可复用脚本。
2. 一次性参数扫描/临时对比脚本必须放在项目根目录 `./scripts/`。
3. 临时脚本完成后必须清理，不得沉淀到 skill。

## 参考资料

- `references/layer_b_metrics.md`

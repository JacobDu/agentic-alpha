---
name: qlib-multi-factor-backtest
description: 负责 Top-N 多因子训练与含交易成本组合回测。用于 Phase 2/3 对比、参数优化、组合风险收益评估以及多因子证据产出。
---

# Qlib 多因子组合回测

执行多因子模型训练与组合验证。

## 使用脚本

- 使用 `scripts/run_phase2_comparison.py` 比较基线与 HEA 因子池。
- 使用 `scripts/run_topn_comparison.py` 比较不同 Top-N 方案。
- 使用 `scripts/run_optimization.py` 做系统化参数优化。
- 使用 `scripts/train_csi1000_lite.py`、`scripts/train_csiall_lite.py`、`scripts/train_csiall_topn.py` 执行训练变体。
- 使用 `scripts/visualize_factors.py` 输出可视化报告。
- 使用 `scripts/mfa_record_cli.py` 记录/查询 MFA 工作流轮次并同步文档索引。

## 执行流程

1. 选定因子池与市场。
2. 训练模型并生成信号。
3. 执行含成本组合回测。
4. 使用 `mfa_record_cli.py` 写入 MFA 决策证据并同步 `docs/workflows/multi-factor/INDEX.md`。

## 参考资料

- 阅读 `references/layer_b_metrics.md` 了解评估口径。
- 多因子记录模板：`assets/templates/multi_factor_experiment_record.md`。

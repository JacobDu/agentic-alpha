---
name: qlib-single-factor-mining
description: 负责在 csi1000 上进行单因子设计、IC/RankIC 显著性检验与 FDR 筛选。用于候选因子挖掘、单因子统计验证、冗余分析和单因子证据产出。
---

# Qlib 单因子挖掘

执行单因子实验与统计评估。

## 使用脚本

- 使用 `scripts/test_factor_ic.py` 做统一因子排名。
- 使用 `scripts/test_new_factors.py` 验证新候选因子。
- 使用 `scripts/test_new_factor_batch.py` 做批量候选测试。
- 使用 `scripts/test_ortho_factor_batch.py` 做正交因子测试。
- 使用 `scripts/test_composite_factors.py` 做复合因子测试。
- 使用 `scripts/test_financial_factors.py` 做财务因子验证。
- 使用 `scripts/analyze_factor_correlation.py` 做冗余相关性分析。
- 使用 `scripts/factor_db_cli.py` 查询因子库与证据记录。
- 使用 `scripts/import_factors_to_db.py` 执行因子定义入库初始化。
- 使用 `scripts/sfa_record_cli.py` 记录/查询 SFA 工作流轮次并同步文档索引。

## 执行流程

1. 先写假设与表达式。
2. 做解析与冗余预检。
3. 执行 IC 与 FDR 检验。
4. 使用 `sfa_record_cli.py` 写入 SFA 决策证据并同步 `docs/workflows/single-factor/INDEX.md`。

## 参考资料

- 阅读 `references/layer_a_thresholds.md` 了解 Promote 门槛。
- 单因子记录模板：`assets/templates/single_factor_experiment_record.md`。

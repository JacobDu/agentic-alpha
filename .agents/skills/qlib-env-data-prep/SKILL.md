---
name: qlib-env-data-prep
description: 负责本仓库的 Qlib 运行环境与数据准备、可用性校验和基线流程联调。用于依赖检查、数据下载与注入、环境故障排查、实验前就绪性验证等任务。
---

# Qlib 环境与数据准备

在因子实验前完成环境与数据门禁检查。

## 使用脚本

- 使用 `scripts/prepare_data.py` 准备基础 Qlib 数据。
- 使用 `scripts/download_financial_data.py` 注入估值与财务字段。
- 使用 `scripts/check_data.py` 检查数据可用性。
- 使用 `scripts/verify_all.py` 执行端到端就绪验证。
- 使用 `scripts/run_official.py` 与 `scripts/run_custom_factor.py` 做基线冒烟测试。
- 使用 `scripts/mlflow_ui.sh` 查看 MLflow 结果。
- 使用 `scripts/migrate_workflow_schema.py` 创建/升级 workflow 记录表结构。
- 使用 `scripts/backfill_workflow_runs.py` 将历史 `docs/heas/*.md` 回填到 workflow 记录表。

## 执行流程

1. 先准备数据。
2. 再检查数据状态。
3. 运行整体验证。
4. 若失败，先修复再进入因子挖掘。

## 参考资料

- 阅读 `references/data_prep_playbook.md` 了解执行顺序。
- 阅读 `references/feature_catalog.md` 了解字段覆盖。

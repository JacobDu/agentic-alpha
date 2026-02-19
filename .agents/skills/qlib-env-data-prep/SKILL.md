---
name: qlib-env-data-prep
description: 负责本仓库的 Qlib 运行环境与数据准备、可用性校验和基线流程联调。用于依赖检查、数据下载与注入、环境故障排查、实验前就绪性验证等任务。
---

# Qlib 环境与数据准备

保障实验前环境可用、数据完整、流程可运行。

## 输入与输出

- 输入：目标市场、数据目录、是否需要回填 workflow 结构。
- 输出：环境就绪结论、数据覆盖结果、可执行脚本日志与必要修复建议。

## 可复用脚本

### 数据准备与校验
- `scripts/prepare_data.py`
- `scripts/download_financial_data.py`
- `scripts/check_data.py`
- `scripts/verify_all.py`

### 基线冒烟
- `scripts/run_official.py`
- `scripts/run_custom_factor.py`
- `scripts/mlflow_ui.sh`

### Workflow DB 运维
- `scripts/migrate_workflow_schema.py`
- `scripts/backfill_workflow_runs.py`

## 标准执行顺序

1. 先执行数据准备与字段注入。
2. 再执行可用性检查与端到端验证。
3. 若失败，优先修复环境再交给下游 skill。
4. 需要 workflow 数据治理时，执行 schema 迁移与文档回填。

## 证据要求

- 输出目录：`outputs/`（检查结果、脚本日志）
- 关键结论应可回溯到具体脚本输出或 DB 记录。

## 临时脚本边界

1. 本 skill 的 `scripts/` 仅维护可复用脚本。
2. 单次排障/一次性检查脚本必须放在项目根目录 `./scripts/`。
3. 临时脚本执行完成后必须清理，不得留在 skill 目录。

## 参考资料

- `references/data_prep_playbook.md`
- `references/feature_catalog.md`

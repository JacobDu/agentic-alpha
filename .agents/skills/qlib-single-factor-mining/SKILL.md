---
name: qlib-single-factor-mining
description: 负责在 csi1000 上进行单因子设计、IC/RankIC 显著性检验与 FDR 筛选。用于候选因子挖掘、单因子统计验证、冗余分析和单因子证据产出。
---

# Qlib 单因子挖掘

负责 SFA 工作流：候选因子设计、统计检验、证据留痕。

## 输入与输出

- 输入：因子假设、表达式、市场与时间区间。
- 输出：单因子统计指标（IC/RankIC/FDR/ICIR）、决策与证据链接。

## 可复用脚本

### 单因子测试
- `scripts/test_factor_ic.py`
- `scripts/test_new_factors.py`
- `scripts/test_new_factor_batch.py`
- `scripts/test_ortho_factor_batch.py`
- `scripts/test_composite_factors.py`
- `scripts/test_financial_factors.py`

### 诊断与分析
- `scripts/analyze_factor_correlation.py`

### 数据库与记录
- `scripts/import_factors_to_db.py`
- `scripts/factor_db_cli.py`
- `scripts/sfa_record_cli.py`

## 标准执行顺序

1. 明确假设与表达式。
2. 执行解析/冗余预检。
3. 运行单因子统计检验。
4. 用 `sfa_record_cli.py` 写入 SFA 记录并同步索引。

## 记录模板

- `assets/templates/single_factor_experiment_record.md`

## 证据要求

- 至少包含一个可追溯证据：`output_path` / `db_query` / `run_id` / `doc`。
- `decision` 仅允许 `Promote / Iterate / Drop`。

## 临时脚本边界

1. 本 skill 的 `scripts/` 只保留可复用脚本。
2. 一次性研究脚本必须放在项目根目录 `./scripts/`。
3. 临时脚本完成后必须清理，不得长期保留在 skill 内。

## 参考资料

- `references/layer_a_thresholds.md`

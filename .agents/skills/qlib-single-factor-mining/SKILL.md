---
name: qlib-single-factor-mining
description: 负责在 csi1000 上进行单因子设计、IC/RankIC 显著性检验与 FDR 筛选。用于候选因子挖掘、单因子统计验证、冗余分析和单因子证据产出。
---

# Qlib 单因子挖掘（SFA）

本 skill 负责 SFA 子工作流，按 `Retrieve -> Generate -> Evaluate -> Distill` 执行。

## 输入与输出

- 输入：因子假设、表达式、市场与时间区间。
- 输出：SFA 指标（IC/RankIC/FDR/ICIR）、决策、证据与文档。

## R-G-E-D 执行定义

### 1) Retrieve
1. 读取 `AGENTS.md` 的经验记忆（推荐/禁止方向）。
2. 通过 `scripts/factor_db_cli.py` 查询因子总表、历史测试结果、历史工作流轮次。
3. 通过 `scripts/analyze_factor_correlation.py` 或 `factor_db_cli.py similarity show` 获取相似度快照。

### 2) Generate
1. 生成候选表达式与市场逻辑。
2. 避免与禁止方向重复的构造模式。
3. 明确候选预期方向（正/负/反转）。

### 3) Evaluate
1. 预检：表达式可解析、字段可用、数据完整。
2. 快筛：IC/RankIC 显著性初筛。
3. 正交预算：`max|rho| <= 0.50`。
4. 全量统计：`fdr_p < 0.01` 且 `|rank_icir| >= 0.10`。

### 4) Distill
1. 用 `scripts/sfa_record_cli.py` 写入工作流记录与证据。
2. 同步文档到 `docs/workflows/single-factor/` 并更新 `INDEX.md`。
3. 将可复用经验回写到 `AGENTS.md` 经验记忆区。

## 因子相似度评估标准

1. 主指标：每日截面 Spearman 相关，记录 `rho_mean_abs`、`rho_p95_abs`、`sample_days`。
2. 推荐阈值：
   - 通过：`max|rho| <= 0.50`
   - 警戒：`0.50 < max|rho| <= 0.80`
   - 高相似：`max|rho| > 0.80`
3. 记录入口：`factor_db_cli.py similarity calc/show`。

## 替换判定标准

1. 仅在高相似区域触发：相关性 `> 0.80`。
2. 新因子必须比旧因子 `|ICIR|` 提升 `>= 20%`。
3. 用 `factor_db_cli.py replace propose/confirm/history` 记录替换链路。

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

## 记录模板

- `assets/templates/single_factor_experiment_record.md`

## 证据要求

- 至少包含一个可追溯证据：`output_path` / `db_query` / `run_id` / `doc`。
- `decision` 仅允许 `Promote / Iterate / Drop`。

## 临时脚本边界

1. 本 skill 的 `scripts/` 只保留可复用脚本。
2. 一次性研究脚本必须放在项目根目录 `./scripts/`。
3. 临时脚本完成后必须清理，不得长期保留在 skill 内。
4. skills 下不允许存在 `__pycache__` / `.pyc`。

## 参考资料

- `references/layer_a_thresholds.md`（历史资料，按 SFA 口径解释）

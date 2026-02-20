# AGENTS.md

本文件只维护三类信息：
1. 核心工作流编排（R-G-E-D）
2. 全局硬门槛规则
3. 经验记忆与脚本治理

## 🎯 项目目标

通过 Agent 持续研究循环，沉淀可复用的高质量因子资产库。
默认市场：`csi1000`。

## 核心工作流（R-G-E-D）

`Retrieve -> Generate -> Evaluate -> Distill`

### 单因子挖掘（SFA）
1. Retrieve：调用 `$qlib-env-data-prep` 完成环境门禁；检索 `data/factor_library.db` 因子总表、历史 SFA 记录、经验记忆。
2. Generate：调用 `$qlib-single-factor-mining` 生成候选表达式，优先遵循“推荐方向”，避开“禁止方向”。
3. Evaluate：调用 `$qlib-single-factor-mining` 执行预检、显著性检验与相关性预算约束。
4. Distill：写入 `docs/workflows/single-factor/` 与 `data/factor_library.db`，并更新本文件经验记忆。

### 多因子组合（MFA）
1. Retrieve：调用 `$qlib-env-data-prep` 完成环境门禁；检索候选因子池、稳定因子集、历史 MFA 记录。
2. Generate：调用 `$qlib-multi-factor-backtest` 构造线性与非线性组合方案。
3. Evaluate：调用 `$qlib-multi-factor-backtest` 执行统一区间回测，输出含成本收益/风险指标与压力测试结果。
4. Distill：写入 `docs/workflows/multi-factor/` 与 `data/factor_library.db`，并更新本文件经验记忆。

## 全局硬门槛

### 决策集
仅允许：`Promote / Iterate / Drop`。

### SFA 显著性门槛
- `fdr_p < 0.01`
- `|rank_icir| >= 0.10`
- 证据完整（doc/output/db_query/run_id 至少一项）

### 正交性预算
- `max|rho| <= 0.50`

### 替换门槛（高相似因子）
- 触发条件：相关性 `> 0.80`
- 允许替换条件：新因子 `|ICIR|` 相对旧因子提升 `>= 20%`

### MFA 默认时间切分（风格优先）
- 训练集（Train）：`2000-01-04 ~ 2023-12-31`
- 验证集（Valid）：`2024-01-01 ~ 2024-12-31`
- OOS 测试集（Test/OOS）：`2025-01-01 ~ 数据最新可用日`
- 若数据最新日早于 `2026-12-31`，必须在文档中明确“2026 为年内截断 OOS”。

### 指标术语与字段标准
- 统一采用 `docs/METRIC_STANDARD_V1.md`。
- 强制区分：
  - 日超额收益：`excess_return_daily_with_cost` / `excess_return_daily_no_cost`
  - 年化超额收益：`excess_return_annualized_with_cost` / `excess_return_annualized_no_cost`
- 历史别名（如 `IR_with_cost`、`ann_ret_with_cost`）仅用于兼容读取，不作为新增文档主字段。

## 因子资产与记录

1. 因子总表：`factors`（位于 `data/factor_library.db`）必须长期保留，记录所有因子定义与状态。
2. 结果表：`factor_test_results`、`factor_backtest_results`、`factor_ic_decay`。
3. 工作流表：`workflow_*`（SFA/MFA 记录、决策、证据、相似度、替换链路）。
4. 文档目录：
   - SFA：`docs/workflows/single-factor/`
   - MFA：`docs/workflows/multi-factor/`
   - 历史：`docs/heas/`（仅历史回填来源）

## 经验记忆（维护在本文件）

### 当前 SOTA 基准（MFA-V4b, 2026-02-20）

| 维度 | 配置 |
|------|------|
| 模型 | Ensemble（XGBoost + LightGBM 均值集成） |
| 训练模式 | Rolling 3m 季度重训（5 个滚动窗口） |
| 训练起始 | 2018 年 |
| 特征 | Alpha158 + DB 因子 Top30（max_per_cat=5） |
| 预测目标 | 1d forward return |
| 组合策略 | TopkDropoutStrategy（topk=30, n_drop=5, hold_thresh=60） |
| 交易成本 | open=5bp, close=15bp, min_cost=5 |
| XGB 参数 | eta=0.05, max_depth=8, colsample_bytree=0.8879, subsample=0.8789, alpha=205.70, lambda=580.98, n_estimators=1000 |
| LGB 参数 | lr=0.05, max_depth=8, num_leaves=128, lambda_l1=205.70, lambda_l2=580.97, n_estimators=1000 |

**OOS 性能（2025-01-01 ~ 2026-02-13，年内截断 OOS）**：

| 指标 | 值 |
|------|-----|
| 年化超额收益（含成本） | **+29.10%** |
| IR（含成本） | **+1.920** |
| 最大回撤 | **-7.00%** |

> 证据：`outputs/mfa_v4b_results.json` → 实验 `C_roll3m_ens_tk30_h60`  
> 文档：`docs/workflows/multi-factor/MFA-V4b-2026-02-20.md`  
> DB：`data/factor_library.db` → `workflow_runs` / `workflow_mfa_metrics` (round_id=MFA-V4b-2026-02-20)

维护原则：
1. 仅记录可复用的结构化经验，不记录一次性日志。
2. 每次 Distill 最多新增 3 条，超过容量时按时间滚动淘汰旧项。
3. 每类最多保留 `50` 条。

### 推荐方向（最多50条）
1. 在 `csi1000` 多因子组合中，优先测试 `hold_thresh=40` 的低换手配置；当候选池为“最新显著Top30”时，交易成本下降带来的净值改善显著。
2. 线性加权（按 `rank_icir` 方向与绝对值权重）应作为 MFA 基准组合长期保留，用于快速筛掉退化的非线性配置。
3. Rolling 3m + XGB+LGB Ensemble 是当前 MFA 最佳训练范式（OOS +29.10%, IR=1.920），应作为默认配置。
4. `topk=30 + hold_thresh=60` 是 Rolling 模式下的最优组合参数；高 hold 阈值有效控制换手成本。
5. 因子数 n=30 (max_per_cat=5)、训练起始 2018 年是 "less is more" 最优点；不宜盲目扩充。

### 禁止方向（最多50条）
1. 当候选池切换后，不要默认沿用旧模型结论；`LGB/Ensemble` 可能在 `hold<=20` 出现 IR 为负，必须先做统一口径回测再决策。
2. 不要使用 2d/5d multi-day label 作为 TopkDropout 日频策略的预测目标 —— 1d label 是唯一有效目标（V4b 验证：label 工程使收益下降 54-82%）。
3. 不要使用 6m 滚动或更长训练起始(2010-)—— 因子时效性短，旧数据引入噪声（V4b 验证：6m 落后 3m 达 +14%、2010 起始落后 2018 达 -14%）。
4. 不要盲目增加因子数(n>30)—— 更多因子=更多噪声（V4b 验证：n40 比 n30 收益下降 -3.43%）。

## 脚本治理

1. Skill 内 `scripts/` 只允许保留可复用脚本。
2. 单次研究临时脚本必须放在项目根目录 `./scripts/`。
3. 临时脚本使用完成后必须删除，不得沉淀到 `.agents/skills/*/scripts/`。
4. `__pycache__`、`.pyc` 不得留在 skills 目录。
5. 如临时脚本被复用 >=2 次，才评估升格为 skill 可复用脚本。

## Agent 行为约束

1. 不得编造数据、指标、run_id。
2. 任何结论必须可追溯到真实文件或数据库记录。
3. 若环境异常，先修复流程可用性，再继续因子研究。

# AGENTS.md

本文件只维护三类信息：
1. 核心工作流编排（R-G-E-V-D）
2. 全局硬门槛规则
3. 经验记忆与脚本治理

## 🎯 项目目标

通过 Agent 持续研究循环，沉淀可复用的高质量因子资产库。
默认市场：`csi1000`。

## 核心工作流（R-G-E-V-D）

`Retrieve -> Generate -> Evaluate -> Validate -> Distill`

### 因子创意研究（Research）
1. 调用 `$factor-research-review` 从研报/论文中提取因子构造思路。
2. 生成因子假设卡片（YAML），维护 `docs/factor_ideas/backlog.md`。
3. 想法验证后分流：通过 → SFA Generate；失败 → `docs/factor_ideas/rejected_ideas.md`。

### 单因子挖掘（SFA）
1. Retrieve：调用 `$qlib-env-data-prep` 完成环境门禁；检索 `data/factor_library.db` 因子总表、历史 SFA 记录、经验记忆。
2. Generate：调用 `$qlib-single-factor-mining` 生成候选表达式，优先遵循“推荐方向”，避开“禁止方向”。
3. Evaluate：调用 `$qlib-single-factor-mining` 执行预检、显著性检验与相关性预算约束。
4. **Validate**：执行深度验证（新增）：
   - IC 衰减分析：`test_factor_ic_decay.py`，填充 `factor_ic_decay` 表
   - 稳定性分析：`test_factor_stability.py`，评估 rolling IC 稳定性
   - 相关性矩阵：`analyze_factor_correlation.py`，填充 `factor_similarity` 表
   - 周持仓适配：标记 `weekly_suitable`（基于 5d/1d ICIR 比率）
5. Distill：写入 `docs/workflows/single-factor/` 与 `data/factor_library.db`，并更新本文件经验记忆。

### 多因子组合（MFA）
1. Retrieve：调用 `$qlib-env-data-prep` 完成环境门禁；检索候选因子池、稳定因子集、历史 MFA 记录。
   - **因子池质量检查**：从 `factor_ic_decay` 表获取 5d ICIR、从稳定性分析获取 stability_score
   - 周频调仓时，仅允许 `weekly_suitable=True` 的因子进入组合
2. Generate：调用 `$qlib-multi-factor-backtest` 构造线性与非线性组合方案。
   - 必须同时测试日频（TopkDropout）与周频（Weekly Rebalance）两类调仓策略
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

### SFA Validate 门槛（新增）
- **IC 衰减**：`factor_ic_decay` 表必须有 1d/5d 两个 horizon 的记录
- **周适用性**：`5d_ICIR / 1d_ICIR >= 0.5` → 标记 `weekly_suitable=True`
- **稳定性**：`stability_score >= 0.3`（低于此值 Drop）
- **近期有效性**：`ic_recent_vs_full >= 0.5`（IC 近期严重退化的因子不入池）

### 替换门槛（高相似因子）
- 触发条件：相关性 `> 0.80`
- 允许替换条件：新因子 `|ICIR|` 相对旧因子提升 `>= 20%`

### MFA 默认时间切分（风格优先）
- 训练集（Train）：`2000-01-04 ~ 2023-12-31`
- 验证集（Valid）：`2024-01-01 ~ 2024-12-31`
- OOS 测试集（Test/OOS）：`2025-01-01 ~ 数据最新可用日`
- 若数据最新日早于 `2026-12-31`，必须在文档中明确“2026 为年内截断 OOS”。
### 周频调仓策略（Weekly Rebalance）
- **适用场景**：持股周期 5~10 交易日的组合策略
- **Label**：5d forward return（`Ref($close, -5)/Ref($close, -1) - 1`）
- **调仓频率**：每 5 个交易日调仓一次（可参数化 3/5/10）
- **因子筛选**：仅使用 `weekly_suitable=True` 的因子
- **策略差异**：
  - 日频（TopkDropout）：1d label + 每日调仓 + hold_thresh 控制换手
  - 周频（Weekly Rebalance）：5d label + 定期调仓 + 自然低换手
- **执行脚本**：`$qlib-multi-factor-backtest` → `scripts/train_weekly_rebal.py`
- **与 TopkDropout 的关系**：两种策略互补，周频不要求 `hold_thresh`；日频策略中 5d label 仍然无效（V4b 结论不变）
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

### 当前 SOTA 基准（MFA-V6, 2026-03-06）

| 维度 | 配置 |
|------|------|
| 模型 | Ensemble（XGBoost + LightGBM 均值集成） |
| 训练模式 | Rolling 3m 季度重训（5 个滚动窗口） |
| 训练起始 | 2018 年 |
| 特征 | Alpha158 + DB 因子 Top30（max_per_cat=5） |
| 预测目标 | 1d forward return |
| 组合策略 | TopkDropoutStrategy（topk=20, n_drop=2, hold_thresh=80） |
| 交易成本 | open=5bp, close=15bp, min_cost=5 |
| XGB 参数 | eta=0.05, max_depth=8, colsample_bytree=0.8879, subsample=0.8789, alpha=205.70, lambda=580.98, n_estimators=1000 |
| LGB 参数 | lr=0.05, max_depth=8, num_leaves=128, lambda_l1=205.70, lambda_l2=580.97, n_estimators=1000 |

**OOS 性能（2025-01-01 ~ 2026-03-04，年内截断 OOS）**：

| 指标 | 值 |
|------|-----|
| 年化超额收益（含成本） | **+33.28%** |
| IR（含成本） | **+1.847** |
| 最大回撤 | **-11.66%** |
| 日均换手 | **1.53%** |

> 证据：`outputs/mfa_v6_topk20_results.json`  
> 文档：`docs/workflows/multi-factor/MFA-V6-2026-03-06.md`  
> DB：`data/factor_library.db` → `workflow_runs` / `workflow_mfa_metrics` (round_id=MFA-V6-2026-03-06)
>
> **注**：V5 旧 SOTA (topk=30, IR=1.589, Ret=+21.75%) 已被取代。V6 同管线 baseline (topk=30) IR=0.907，V6 最优 (topk=20) IR=1.847，绝对 IR 相对 V5 提升 +16%。

维护原则：
1. 仅记录可复用的结构化经验，不记录一次性日志。
2. 每次 Distill 最多新增 3 条，超过容量时按时间滚动淘汰旧项。
3. 每类最多保留 `50` 条。

### 推荐方向（最多50条）
1. 在 `csi1000` 多因子组合中，优先测试 `hold_thresh=40` 的低换手配置；当候选池为“最新显著Top30”时，交易成本下降带来的净值改善显著。
2. 线性加权（按 `rank_icir` 方向与绝对值权重）应作为 MFA 基准组合长期保留，用于快速筛掉退化的非线性配置。
3. Rolling 3m + XGB+LGB Ensemble 是当前 MFA 最佳训练范式（OOS +21.75%, IR=1.589, 数据版本 2026-03-05），应作为默认配置。
4. `topk=30 + hold_thresh=60` 是 Rolling 模式下的最优组合参数；高 hold 阈值有效控制换手成本。
5. 因子数 n=30 (max_per_cat=5)、训练起始 2018 年是 "less is more" 最优点；不宜盲目扩充。
6. `topk=20 + n_drop=2 + hold_thresh=80` 是集中持仓的最优参数（V6 完整 5 窗口: IR=1.847, Ret=+33.28%, Turn=1.53%），高 hold 在困难市场（2026 Q1）中显著更稳健；已 Promote 为 SOTA。
7. topk=20 参数搜索时优先降 `n_drop`（2~3）而非沿用 topk=30 的 `n_drop=5`——高换仓比例在集中组合中放大了成本与波动。
8. MFA Rolling 流程内存优化三要素：(a) Ensemble 逐模型训练/预测，`del model` 后再训下一个；(b) 每个窗口结束后调用 `H.clear()` 清理 Qlib MemCache（特征表达式缓存）；(c) `gc.collect()` 配合 `del` 确保 Python 回收。峰值内存可降低 35-40%。
9. V7 rank-buffer 研究中，`buffer_rank=40~100` 在“任一持仓越界即卖 bottom x”表述下基本不生效；后续应优先测试更紧的 `buffer_rank=20~35`，或采用 `min_hold + buffer` 混合退出规则。
10. **V7b TrueBufferExit 验证成功**（2026-03-19）：`TrueBufferExitStrategy(buffer_rank=30, min_hold=60, n_drop=2)` 在 OOS 2025-01-01~2026-03-18 上 IR=1.667, Ann=+25.62%，显著优于同期 plain TopkDropout h80（IR=0.139）。关键是"只卖越界持仓本身"而非"任一越界则卖 bottom"。
11. V7b 中 `buffer_rank=25~40` 表现一致（IR=1.667），说明 `min_hold=60` 是主导约束；`buffer_rank=20` 过紧导致 IR 降至 1.455。
12. **短持仓最优配置**（2026-03-19）：`TopkDropout(topk=20, n_drop=1, hold_thresh=20)` 在短持仓约束下表现最佳：IR=1.732, Ann=+26.22%, MaxDD=-9.27%。关键发现：`n_drop=1` 是短持仓策略的核心——保守退出显著降低换手成本。
13. `hold_thresh=20 + n_drop=1` 是唯一在 2025H1 (+17.31%) 和 2025H2 (+10.27%) 均保持正向的配置，环境稳健性最佳。
### 禁止方向（最多50条）
1. 当候选池切换后，不要默认沿用旧模型结论；`LGB/Ensemble` 可能在 `hold<=20` 出现 IR 为负，必须先做统一口径回测再决策。
2. 不要使用 2d/5d multi-day label 作为 TopkDropout 日频策略的预测目标 —— 1d label 是唯一有效目标（V4b 验证：label 工程使收益下降 54-82%）。
3. 不要使用 6m 滚动或更长训练起始(2010-)—— 因子时效性短，旧数据引入噪声（V4b 验证：6m 落后 3m 达 +14%、2010 起始落后 2018 达 -14%）。
4. 不要盲目增加因子数(n>30)—— 更多因子=更多噪声（V4b 验证：n40 比 n30 收益下降 -3.43%）。
5. topk=20 时不要使用 `n_drop=5 + hold_thresh>=60`——V6 验证该组合 IR 仅 0.94~1.20，远逊于 n_drop=2 的 2.07（激进换仓在集中组合中严重拖累收益）。

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

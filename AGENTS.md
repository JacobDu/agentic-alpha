# AGENTS.md

本文件用于规范本项目内 Agent 的工作方式。默认所有 Agent 在开始工作前先阅读本文件，并按本文档执行。

## 🎯 项目目标

通过持续运行 Agent 驱动的研究循环，挖掘并验证有价值的量化因子（factor），最终形成可复用的高质量因子资产库。

## ✅ 成功标准（默认）

一个因子要进入“有价值因子库”，默认至少满足以下条件（可按阶段调整）：

1. 在测试区间具备正向预测能力（如 `IC` 或 `RankIC` 显著为正/绝对值足够高）。
2. 在含交易成本回测下不显著恶化组合表现。
3. 指标与结论可追溯到 `mlruns/` 和 `outputs/` 的真实结果文件。
4. 有明确市场直觉、适用范围与失效条件说明。

## 🛠️ 工作流：假设-实验-分析循环（HEA Loop）

每一轮必须按以下顺序执行，不可跳步：

1. **Hypothesis（假设）**
   - 明确因子定义（表达式或构造方式）。
   - 说明市场逻辑（为什么可能有效）。
   - 明确预期（提升哪类指标，预期方向是什么）。
   - 记录本轮编号（如 `HEA-2026-02-14-01`）。

2. **Experiment（实验）**
   - 基于当前工程运行实验，不得伪造结果。
   - 优先复用现有脚本：
     - `uv run python scripts/run_official.py`
     - `uv run python scripts/run_custom_factor.py`
     - `uv run python scripts/verify_all.py`
   - 记录关键 run 信息（run_id、配置、时间区间、模型、handler）。

3. **Analysis（分析）**
   - 分析 `mlruns/` 与 `outputs/` 指标：
     - `IC / RankIC / ICIR / RankICIR`
     - `excess_return_with_cost` 系列指标
   - 与基线（Alpha158）对比，判断增益或退化。
   - 给出结论：`Promote`（保留）/`Iterate`（继续迭代）/`Drop`（放弃）。

## 🔁 3. 自我进化

在**重要回测、策略迭代或习得新技能后**，必须更新本文档底部记忆库：

1. 记录成功参数与失败教训。
2. 记录市场洞察与适用边界。
3. 内容保持精简，避免冗长复述日志。

## 🧭 Agent 行为约束

1. 不得编造数据、指标、run_id。
2. 任何结论都要能定位到具体文件（`mlruns/` 或 `outputs/`）。
3. 每轮 HEA 结束后，必须更新“记忆库”中的至少两处：
   - `HEA 轮次日志`
   - `因子资产库` 或 `失败教训库`
4. 如遇环境异常（依赖冲突、服务启动失败），先修复流程可用性，再继续因子挖掘。

---

## 🧠 记忆库（简化版）

> 规则：每轮 HEA 结束后只更新以下 3 块；每块保持短小，禁止粘贴大段日志。

### 1) 优秀因子清单

| 因子 | 状态 | 一句话结论 | 证据 |
|---|---|---|---|
| Alpha158 Baseline | Baseline | 当前强基线（with_cost IR 约 2.18） | `mlruns/1/8d9baa735e9c452092d92b7414f77cf4` |
| CSTM_MOM_5 | Rejected | 加入后组合表现弱于基线，暂不采用 | `mlruns/1/f9fc1c736c3b4e9f858a0df6d7b53c36`, `outputs/factor_significance_test.csv` |

状态建议仅用：`Candidate / Tested / Accepted / Rejected / Baseline`

### 2) HEA 日志（最近 10 条）

| 轮次 | 因子/策略 | 决策 | 关键指标 | 日期 |
|---|---|---|---|---|
| HEA-2026-02-14-01 | `Alpha158PlusOne(CSTM_MOM_5)` | Rejected | with_cost IR 0.61（低于基线 2.18） | 2026-02-14 |

### 3) 经验速记（最多 10 条）

1. 单因子有信号不等于加入模型后会提升组合，必须和基线做含成本对比。
2. macOS 下 `mlflow ui` 若崩溃，使用 `scripts/mlflow_ui.sh`（已内置 `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`）。

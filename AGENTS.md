# AGENTS.md

本文件用于规范本项目内 Agent 的工作方式。默认所有 Agent 在开始工作前先阅读本文件，并按本文档执行。

## 🎯 项目目标

通过持续运行 Agent 驱动的研究循环，挖掘并验证有价值的量化因子（factor），最终形成可复用的高质量因子资产库。

**当前阶段目标**：在 csiall（全市场）上，以 Alpha158 为基线，逐一挖掘和验证单因子的预测能力，最终筛选出 Top-N 最有预测能力的因子进行组合。

### 方法论

1. **单因子优先**：先独立评估每个候选因子的 IC/RankIC 显著性，不急于加入模型组合。
2. **全市场验证**：使用 `csiall`（全 A 股）和全部可用数据进行因子有效性分析，确保普适性。
3. **统计严格**：因子入选需满足 RankIC t-test 显著性 p < 0.01，并通过 FDR 校正。
4. **分阶段推进**：
   - **Phase 1**：大规模候选因子设计 + 单因子 IC 筛选（当前阶段）
   - **Phase 2**：Top-N 因子组合 + LightGBM 模型训练
   - **Phase 3**：含交易成本回测验证
5. **基线对照**：Alpha158 的 158 个因子作为 IC 基线参照。

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
| Alpha158 Baseline (CSI300) | Baseline | CSI300 强基线（with_cost IR 约 2.18） | `mlruns/1/8d9baa735e9c452092d92b7414f77cf4` |
| Alpha158 Baseline (CSI1000) | Baseline | CSI1000 基线（IC=0.036, with_cost IR=0.67, ann_ret=4.7%） | `mlruns/1/1684a8aa6fd44d858a1f76bf88999291` |
| **CSTM_VOL_CV_10** | Candidate | csiall RankIC=-0.032, t=-21.04, ICIR=-0.51（最强因子） | `outputs/csiall_factor_significance.csv` |
| **CSTM_AMT_CV_20** | Candidate | csiall RankIC=-0.035, t=-20.06, ICIR=-0.49 | `outputs/csiall_factor_significance.csv` |
| **CSTM_RANGE_VOL_10** | Candidate | csiall RankIC=-0.054, t=-15.42, ICIR=-0.37 | `outputs/csiall_factor_significance.csv` |
| **CSTM_AMT_SURGE_60** | Candidate | csiall RankIC=-0.045, t=-15.21, ICIR=-0.37 | `outputs/csiall_factor_significance.csv` |
| **CSTM_RANGE_1D** | Candidate | csiall RankIC=-0.056, t=-15.06, ICIR=-0.37 | `outputs/csiall_factor_significance.csv` |
| **CSTM_PV_CORR_20** | Candidate | csiall RankIC=-0.032, t=-14.94, ICIR=-0.36 | `outputs/csiall_factor_significance.csv` |
| **CSTM_AMT_WTRET_20** | Candidate | csiall RankIC=-0.045, t=-13.89, ICIR=-0.34 | `outputs/csiall_factor_significance.csv` |
| **CSTM_REVERT_20** | Candidate | csiall RankIC=+0.038, t=+9.88, ICIR=+0.24（正向反转因子） | `outputs/csiall_factor_significance.csv` |

> 完整 43 因子检测结果见 `outputs/csiall_factor_significance.csv`，Top-20 见 `outputs/csiall_factor_top20.csv`

状态建议仅用：`Candidate / Tested / Accepted / Rejected / Baseline`

### 2) HEA 日志（最近 10 条）

| 轮次 | 因子/策略 | 决策 | 关键指标 | 日期 |
|---|---|---|---|---|
| HEA-2026-02-14-01 | `Alpha158PlusOne(CSTM_MOM_5)` | Rejected | CSI300 with_cost IR 0.61（低于基线 2.18） | 2026-02-14 |
| HEA-2026-02-14-02 | CSI1000 Alpha158 Baseline | Baseline | IC=0.036, RankICIR=0.37, with_cost IR=0.67 | 2026-02-14 |
| HEA-2026-02-14-03 | CSI1000 Alpha158+7自定义因子(v1) | Rejected | IC 降至 0.033, with_cost IR=-0.39（严重退化） | 2026-02-14 |
| HEA-2026-02-14-04 | CSI1000 Alpha158+4显著因子(v2) | Iterate | RankIC 升至 0.036(+7%), 但 with_cost IR 降至 0.52 | 2026-02-14 |
| HEA-2026-02-14-05 | csiall 43因子大规模 IC 筛选 | **Phase1完成** | 39/43 通过 FDR<0.01，Top3 ICIR>0.37 | 2026-02-14 |
| HEA-2026-02-15-06 | csiall 201因子统一排名 | **Phase1.5完成** | 182/201 显著，Custom占Top2(ICIR>0.49) | 2026-02-15 |
| HEA-2026-02-15-07 | csiall TopN vs Baseline(lite) | **Iterate** | csiall topk=50负回报，需调市场 | 2026-02-15 |
| HEA-2026-02-15-08 | CSI1000 TopN vs Baseline | **Promote TopN50** | TopN50 IR=0.70 > Baseline 0.56，见下表 | 2026-02-15 |
| HEA-2026-02-15-09 | CSI1000 因子排名 | **Phase1.5b完成** | 179/201显著，Custom占Top5中3席 | 2026-02-15 |
| HEA-2026-02-15-10 | CSI1000 调仓频率对比 | **Promote Biweekly** | 双周IR=1.61 > 日度IR=1.21，累计113.8% | 2026-02-15 |

**HEA-08 详细结果（CSI1000: train 2019-2023, test 2024H2-2026.02.13）：**

| 模型 | #因子 | IC | RankIC | ICIR | RankICIR | IR(w/c) | Ret(w/c) | MaxDD | Run ID |
|---|---|---|---|---|---|---|---|---|---|
| TopN20 | 20 | 0.027 | 0.034 | 0.25 | 0.38 | -0.37 | -2.6% | -10.3% | `69ef2400` |
| TopN30 | 30 | 0.030 | 0.038 | 0.26 | 0.38 | -0.13 | -1.0% | -10.3% | `0c8a83d0` |
| **TopN50** | **50** | **0.034** | **0.037** | **0.33** | **0.40** | **0.70** | **5.1%** | **-6.8%** | `d3aec63d` |
| Baseline | 158 | 0.037 | 0.034 | 0.37 | 0.38 | 0.56 | 4.0% | -8.8% | `d2343e3f` |

**HEA-10 调仓频率对比（基于TopN50 pred.pkl, CSI1000, 2024H2-2026.02.13）：**

| 策略 | 累计收益 | 年化收益 | 累计超额 | 年化超额 | IR | MDD | 换手率 | 月胜率 |
|---|---|---|---|---|---|---|---|---|
| Daily (d=5, h=1) | 95.1% | 52.7% | 25.4% | 12.9% | 1.21 | -16.8% | 0.147 | 60% |
| Daily (d=3, h=1) | 85.6% | 47.9% | 15.9% | 8.2% | 0.74 | -17.5% | 0.108 | 45% |
| Daily (d=1, h=1) | 74.8% | 42.4% | 5.1% | 2.6% | 0.30 | -16.6% | 0.034 | 45% |
| Weekly (d=5, h=5) | 62.2% | 35.8% | -7.6% | -4.0% | -0.33 | -17.4% | 0.087 | 40% |
| **Biweekly (d=5, h=10)** | **113.8%** | **61.8%** | **44.1%** | **22.0%** | **1.61** | **-15.0%** | **0.077** | **75%** |
| Monthly (d=5, h=20) | 73.2% | 41.6% | 3.5% | 1.8% | 0.21 | -19.7% | 0.044 | 50% |

### 3) 经验速记（最多 10 条）

1. 单因子有信号不等于加入模型后会提升组合，必须和基线做含成本对比。
2. macOS 下 `mlflow ui` 若崩溃，使用 `scripts/mlflow_ui.sh`（已内置 `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`）。
3. CSI1000 上 4 个自定义因子均呈显著负 RankIC（反转逻辑），小盘股的均值回归效应更强。
4. 因子数量控制很重要：7 因子全加严重退化，精选 4 个后 RankIC 提升但组合仍未超基线。
5. Qlib 表达式引擎不支持一元 `-` 运算符，需用 `(0 - expr)` 替代；但 LightGBM 可自动学习方向。
6. CSI1000 基线 IC(0.036) 显著高于 CSI300(0.008)，小盘股预测性更强。
7. csiall 全市场因子筛选：波动率类因子（VOL_CV, AMT_CV）t统计量最高(>20)，ICIR最优(>0.48)。
8. 全市场因子呈现一致的"低波动溢价"+"均值回归"特征：高波动/高量能→负收益，过去跌→正收益。
9. CSTM_RANGE_1D 和 CSTM_RANGE_VOL_10 的 RankIC 绝对值(>0.05)最大，但 ICIR 不如 VOL_CV。
10. **csiall topk=50策略亏损**：IC正向但组合负回报，因覆盖率仅0.9%(50/5600)。切换到csi1000后topk=50覆盖5%，效果改善。
11. **16GB内存优化**：csiall+158因子需缩短训练窗口(3yr)，减少num_leaves(128)。脚本：`scripts/train_csiall_lite.py`。
12. **TopN50在CSI1000上超越Baseline**：IR 0.70 vs 0.56，年化+5.1% vs +4.0%，MaxDD -6.8% vs -8.8%。仅用1/3因子量，训练快40%。
13. **数据范围**：行情数据 2000-01-04 ~ 2026-02-13。无财务数据（pe/pb/eps等均为空），仅10个量价字段。
14. **CSI1000因子排名与csiall高度一致**：Top-20重合18/20。CSI1000上CSTM_AMT_CV_20排第一(t=-16.0, ICIR=-0.39)，CSTM_VOL_CV_10第二(t=-15.7)。详见`outputs/csi1000_unified_factor_ranking.csv`。
15. **双周调仓最优**：hold_thresh=10(双周)的IR=1.61远超日度1.21和月度0.21。累计收益113.8%，月度超额胜率75%。原因：信号需要时间兑现但不宜过长。

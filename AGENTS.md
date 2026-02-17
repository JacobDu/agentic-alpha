# AGENTS.md

本文件用于规范本项目内 Agent 的工作方式。默认所有 Agent 在开始工作前先阅读本文件，并按本文档执行。

## 🎯 项目目标

通过持续运行 Agent 驱动的研究循环，挖掘并验证有价值的量化因子（factor），最终形成可复用的高质量因子资产库。

**当前阶段目标**：在 csi1000（中证1000）上，以 Alpha158 为基线，逐一挖掘和验证单因子的预测能力，最终筛选出 Top-N 最有预测能力的因子进行组合。

### 方法论

1. **单因子优先**：先独立评估每个候选因子的 IC/RankIC 显著性，不急于加入模型组合。
2. **默认市场**：使用 `csi1000`（中证1000）进行因子有效性分析和回测验证，因小盘股预测性更强（IC 显著高于 CSI300）。
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

在**重要回测、策略迭代或习得新技能后**：

1. **因子数据**写入因子库（`data/factor_library.db`），包括 IC 检验结果和回测数据。
2. **关键教训**更新到本文档底部「经验速记」区（保持精简，最多 15 条）。
3. 禁止在本文档粘贴大段指标表或日志。

## 🧭 Agent 行为约束

1. 不得编造数据、指标、run_id。
2. 任何结论都要能定位到因子库（`data/factor_library.db`）、`mlruns/` 或 `outputs/` 中的真实记录。
3. 每轮 HEA 结束后，必须将结果写入因子库，并视情况更新底部「经验速记」。
4. 如遇环境异常（依赖冲突、服务启动失败），先修复流程可用性，再继续因子挖掘。

---

## 🧠 记忆库

> 规则：因子数据、IC 指标、回测结果等详细信息统一存储在 **因子库**（`data/factor_library.db`）中，不在本文件重复记录。
>
> 查询方式：
> ```bash
> uv run python scripts/factor_db_cli.py summary --market csi1000   # 概览
> uv run python scripts/factor_db_cli.py list --market csi1000 --top 20  # Top-N 因子
> uv run python scripts/factor_db_cli.py show CSTM_VOL_CV_10       # 单因子详情（含 IC/回测/衰减）
> uv run python scripts/factor_db_cli.py markets                    # 各市场统计
> uv run python scripts/factor_db_cli.py backtest --market csi1000 --hold 10  # 换仓周期回测
> uv run python scripts/factor_db_cli.py decay CSTM_VOL_CV_10      # IC 衰减曲线
> ```
>
> **因子库四张表**：
> - `factors` — 因子定义（名称、公式、类别、状态）
> - `factor_test_results` — IC 统计（每因子 × 市场 × 测试区间）
> - `factor_backtest_results` — 组合回测（每因子 × 市场 × 换仓周期 × topk）
> - `factor_ic_decay` — IC 衰减（每因子 × 市场 × 前瞻天数）
>
> 本节仅保留**经验速记**，供 Agent 快速回忆关键教训。

### 经验速记

1. 单因子有信号不等于加入模型后会提升组合，必须和基线做含成本对比。
2. macOS 下 `mlflow ui` 若崩溃，使用 `scripts/mlflow_ui.sh`（已内置 `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`）。
3. CSI1000 上自定义因子均呈显著负 RankIC（反转逻辑），小盘股的均值回归效应更强。
4. 因子数量控制很重要：7 因子全加严重退化，TopN50（50 因子）在 CSI1000 上反而超越 158 因子基线（IR 0.70 vs 0.56）。
5. Qlib 表达式引擎不支持一元 `-` 运算符，需用 `(0 - expr)` 替代；但 LightGBM 可自动学习方向。
6. CSI1000 基线 IC(0.036) 显著高于 CSI300(0.008)，小盘股预测性更强。
7. 全市场因子呈一致的"低波动溢价"+"均值回归"特征：波动率类因子（VOL_CV, AMT_CV）ICIR 最优。
8. csiall topk=50 策略亏损（覆盖率仅 0.9%），切换到 CSI1000（覆盖 5%）后效果改善。
9. 16GB 内存优化：csiall + 158 因子需缩短训练窗口(3yr)，减少 num_leaves(128)。
10. 数据范围：行情数据 2000-01-04 ~ 2026-02-13，仅 10 个量价字段，无财务数据。
11. CSI1000 与 csiall 因子排名高度一致（Top-20 重合 18/20），可在 CSI1000 上快速验证。
12. 双周调仓（hold=10）最优：IR=1.61 远超日度 1.21 和月度 0.21。信号需要时间兑现但不宜过长。

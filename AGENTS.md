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

## 🛠️ 工作流：HEA v2（每轮强制留痕）

从本版本开始，**每次 HEA 都必须完成全量流程**，不得跳步：

1. **Hypothesis（假设）**
   - 明确因子定义（表达式或构造方式）、市场逻辑与预期方向。
   - 生成 round 编号：`HEA-YYYY-MM-DD-XX`。

2. **Preflight Gate（实验前门控）**
   - `parse_ok`：表达式可解析，可在当前 Qlib 表达式引擎执行。
   - `complexity_level`：复杂度评级（low/medium/high）。
   - `redundancy_flag`：是否与已测因子高度冗余（结构或经济含义）。
   - `data_availability`：所需字段在当前市场与时间区间可用。

3. **Experiment（实验）**
   - 基于当前工程运行实验，不得伪造结果。
   - 优先复用脚本：
     - `uv run python scripts/test_factor_ic.py`
     - `uv run python scripts/test_new_factors.py`
     - `uv run python scripts/run_official.py`
     - `uv run python scripts/run_custom_factor.py`
     - `uv run python scripts/verify_all.py`
   - 记录关键 run 信息（run_id、配置、时间区间、模型、handler）。

4. **Analysis（双层评估）**
   - `Layer A（单因子统计层）`：`IC/RankIC/ICIR/RankICIR/t/FDR`。
   - `Layer B（组合收益层）`：`excess_return_with_cost`、`IR(with cost)`、`max_drawdown`。
   - 口径约束：若未执行 Layer B，必须写 `not_run_reason`，且不得给出“组合有效”结论。

5. **Decision（硬门槛判定）**
   - 仅允许：`Promote` / `Iterate` / `Drop`。
   - 判定结果必须写入 `decision_basis`，并绑定可追溯证据。

6. **Archive（双写留痕）**
   - 写入当轮记录文件：`docs/heas/HEA-YYYY-MM-DD-XX.md`。
   - 更新索引：`docs/heas/INDEX.md`（新增一行摘要）。
   - 写入因子库：`data/factor_library.db`（含 `hea_round`、`evidence`、`notes`）。

## 🗂️ HEA 记录目录规范（docs/heas）

1. 目录固定为 `docs/heas/`。
2. 每轮文件命名固定为 `HEA-YYYY-MM-DD-XX.md`，其中 `XX` 为当日两位递增序号。
3. 必须维护 `docs/heas/INDEX.md`，按 `hea_round` 时间倒序排列。
4. 每轮结论必须双写：
   - 单轮文件（完整记录）
   - INDEX 摘要（快速检索）

## 🧾 HEA 记录模板（用于 docs/heas）

> 以下模板是唯一准则。每轮 HEA 文件必须包含全部字段，不可删减。

```md
# HEA-YYYY-MM-DD-XX

## 1) hea_round
- hea_round: HEA-YYYY-MM-DD-XX
- date: YYYY-MM-DD
- owner: <agent/user>

## 2) hypothesis
- hypothesis: <一句话假设>
- market_logic: <为何可能有效>
- expected_direction: <正/负/绝对值增强>

## 3) factor_expression_list
- factor_1: <name> = <expression>
- factor_2: <name> = <expression>

## 4) preflight_gate
- parse_ok: pass/fail
- complexity_level: low/medium/high
- redundancy_flag: low/medium/high
- data_availability: pass/fail
- gate_notes: <异常与修复建议>

## 5) experiment_config
- script: <scripts/...py>
- market: csi1000
- start: YYYY-MM-DD
- end: YYYY-MM-DD
- run_id: <mlflow_run_id 或 N/A>
- output_paths:
  - outputs/...
  - mlruns/...

## 6) layer_a_single_factor
- rank_ic_mean: <float>
- rank_ic_t: <float>
- fdr_p: <float>
- rank_icir: <float>
- n_days: <int>
- layer_a_result: pass/fail

## 7) layer_b_portfolio
- executed: yes/no
- not_run_reason: <若未执行必填>
- excess_return_with_cost: <float or N/A>
- ir_with_cost: <float or N/A>
- max_drawdown: <float or N/A>
- layer_b_result: pass/fail/not_run

## 8) decision
- decision: Promote/Iterate/Drop
- decision_basis: <必须引用 Layer A/B + 门控结果>
- failure_mode: <若 Iterate/Drop 必填>
- next_action: <下一轮动作>

## 9) evidence_links
- db_query: <factor_db_cli 命令或结果键>
- run_id: <mlflow run id>
- outputs:
  - outputs/...
```

## 📏 判定矩阵（Promote / Iterate / Drop）

| Decision | 必要条件 | 说明 |
|----------|----------|------|
| Promote | `fdr_p < 0.01` 且 `|rank_icir| >= 0.10` 且证据完整 | 进入候选保留池，可进入后续组合验证 |
| Iterate | 未达 Promote 但存在可修复空间 | 必须填写 `failure_mode` 与 `next_action` |
| Drop | 多轮不稳定或显著退化，且无明确修复路径 | 停止继续投入，保留证据用于反例库 |

补充规则：
1. 默认市场 `csi1000` 允许反向因子，方向以统计显著性与稳定性为准。
2. `FDR < 0.01` 为硬门槛，不得弱化为口头判断。
3. 任何结论若缺证据映射（DB/mlruns/outputs），视为无效结论。

## 🔁 自我进化

在重要回测、策略迭代或习得新技能后：

1. 因子数据写入 `data/factor_library.db`，至少包含 `hea_round`、关键指标与证据标识。
2. 每轮 HEA 同步写入 `docs/heas/` 与 `docs/heas/INDEX.md`。
3. 关键教训更新到本文档底部「经验速记」（最多 15 条）。
4. 禁止在本文档粘贴大段日志，详细结果放 `outputs/` 与 `mlruns/`。

## 🧭 Agent 行为约束

1. 不得编造数据、指标、run_id。
2. 任何结论都要能定位到 `data/factor_library.db`、`mlruns/` 或 `outputs/` 的真实记录。
3. 每轮 HEA 未完成“单轮文件 + INDEX + DB”三项留痕，视为流程未完成。
4. LLM 产出仅作为研究输入，不替代统计检验与实验真值。
5. 如遇环境异常（依赖冲突、服务启动失败），先修复流程可用性，再继续因子挖掘。

---

## 📊 可用数据字段（Features）

Qlib 数据目录 `data/qlib/cn_data/features/{stock}/` 中每个字段对应一个 `.day.bin` 文件。
在 Qlib 表达式中通过 `$field_name` 引用（如 `$close`、`$pe_ttm`）。

下载命令：`uv run python scripts/download_financial_data.py`（Phase 1 + Phase 2 一次性下载，支持增量更新）

### 量价 + 估值字段（Phase 1，来源：baostock 日频 K 线 + query_adjust_factor API）

| 字段 | 含义 | 说明 |
|------|------|------|
| `$open` | 开盘价 | 前复权 = raw × foreAdjustFactor |
| `$close` | 收盘价 | 前复权 = raw × foreAdjustFactor |
| `$high` | 最高价 | 前复权 = raw × foreAdjustFactor |
| `$low` | 最低价 | 前复权 = raw × foreAdjustFactor |
| `$volume` | 成交量 | 前复权手 = 股 / factor / 100 |
| `$amount` | 成交额 | 千元 = 元 / 1000 |
| `$vwap` | 成交均价 | 前复权 VWAP = (amount/volume) × factor |
| `$change` | 涨跌幅 | 日收益率 = pctChg / 100 |
| `$factor` | 复权因子 | foreAdjustFactor（来自 query_adjust_factor API） |
| `$pe_ttm` | 滚动市盈率 | 市值 / 过去 4 季度净利润 |
| `$pb_mrq` | 市净率 | 市值 / 最新季度净资产 |
| `$ps_ttm` | 滚动市销率 | 市值 / 过去 4 季度营收 |
| `$pcf_ttm` | 滚动市现率 | 市值 / 过去 4 季度经营现金流净额 |
| `$turnover_rate` | 换手率 | 当日成交量 / 流通股本 (%) |

### 季度财务指标（来源：baostock 季报 API，Phase 2，按 pubDate 前向填充）

#### 盈利能力（query_profit_data）

| 字段 | 含义 | baostock 原始字段 |
|------|------|------------------|
| `$roe` | 净资产收益率(平均) | roeAvg |
| `$npm` | 销售净利率 | npMargin |
| `$gpm` | 销售毛利率 | gpMargin |
| `$eps_ttm` | 每股收益(TTM) | epsTTM |

#### 成长能力（query_growth_data）

| 字段 | 含义 | baostock 原始字段 |
|------|------|------------------|
| `$yoy_ni` | 净利润同比增长率 | YOYNI |
| `$yoy_eps` | 基本 EPS 同比增长率 | YOYEPSBasic |
| `$yoy_pni` | 归母净利润同比增长率 | YOYPNI |
| `$yoy_equity` | 净资产同比增长率 | YOYEquity |
| `$yoy_asset` | 总资产同比增长率 | YOYAsset |

#### 偿债能力（query_balance_data）

| 字段 | 含义 | baostock 原始字段 |
|------|------|------------------|
| `$debt_ratio` | 资产负债率 | liabilityToAsset |
| `$eq_multiplier` | 权益乘数 | assetToEquity |
| `$yoy_liability` | 负债同比增长率 | YOYLiability |

#### 营运能力（query_operation_data）

| 字段 | 含义 | baostock 原始字段 |
|------|------|------------------|
| `$asset_turnover` | 总资产周转率 | AssetTurnRatio |

#### 现金流（query_cash_flow_data）

| 字段 | 含义 | baostock 原始字段 |
|------|------|------------------|
| `$cfo_to_or` | 经营现金流 / 营收 | CFOToOR |
| `$cfo_to_np` | 经营现金流 / 净利润（现金流质量） | CFOToNP |

#### 杜邦分析（query_dupont_data）

| 字段 | 含义 | baostock 原始字段 |
|------|------|------------------|
| `$dupont_roe` | 杜邦 ROE | dupontROE |
| `$dupont_asset_turn` | 杜邦 资产周转率 | dupontAssetTurn |
| `$dupont_nito_gr` | 杜邦 净利润率 | dupontNitogr |
| `$dupont_tax` | 杜邦 税收负担率 | dupontTaxBurden |

#### 分红（query_dividend_data）

| 字段 | 含义 | baostock 原始字段 |
|------|------|------------------|
| `$div_ps` | 每股税前现金股利 | dividCashPsBeforeTax，按股权登记日前向填充 |

### 行业分类

- 文件：`data/industry.parquet`（证监会一级行业分类，约 19 类）
- 来源：`bs.query_stock_industry()`
- 用途：行业中性化（IC 测试中默认启用）

### 数据范围

- **交易日历**：6331 天，2000-01-04 ~ 2026-02-13
- **股票数量**：6057 只（含 BJ），SH/SZ 约 5400+
- **日频估值**：5323 只 SH/SZ 股票，108 只无数据（已填 NaN）
- **季度财务**：2010 ~ 2025 年，按 pubDate 前向填充至日频
- **行业分类**：证监会一级行业，约 19 类

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
10. 数据源统一使用 baostock，支持增量更新。Phase 1 force 模式：raw K-line + query_adjust_factor（2 API calls/stock），增量模式：raw K-line（1 API call/stock）。Phase 2：19 个季度字段 + 分红 + 行业。详见「📊 可用数据字段」。
11. CSI1000 与 csiall 因子排名高度一致（Top-20 重合 18/20），可在 CSI1000 上快速验证。
12. 双周调仓（hold=10）最优：IR=1.61 远超日度 1.21 和月度 0.21。信号需要时间兑现但不宜过长。
13. 数据交叉验证（akshare）：OHLC 零误差，volume 单位=前复权手（baostock 股/factor/100），amount 单位=千元（baostock 元/1000）。财务指标 20/23 通过（87%），仅 ROE 定义差异（平均 vs 加权）和少数 EPS/增长率口径差异。
14. query_adjust_factor API 仅返回除权日事件（~10-30 行），forward-fill 到每日与 adj_close/raw_close 计算结果完全一致（14136 天零误差）。

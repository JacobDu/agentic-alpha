# SFA-LIQ-MICRO-01 — 流动性与微观结构因子挖掘

- **Round ID**: SFA-LIQ-MICRO-01
- **日期**: 2026-03-18
- **市场**: csi1000
- **测试区间**: 2020-01-01 ~ 2025-12-31

## 研究动机

当前因子库（112 个 curated 因子）中 **liquidity** 和 **microstructure** 类别为空白。参照 backlog 中的 Amihud 非流动性、Kyle's Lambda、订单不平衡等假设，设计 10 个候选因子填补空白。

## 候选因子

| # | 名称 | 类别 | 表达式 |
|---|------|------|--------|
| 1 | CSTM_AMIHUD_20 | liquidity | `Mean(Abs($close / Ref($close, 1) - 1) / ($amount + 1), 20)` |
| 2 | CSTM_AMIHUD_5 | liquidity | `Mean(Abs($close / Ref($close, 1) - 1) / ($amount + 1), 5)` |
| 3 | CSTM_ILLIQ_MOM | liquidity | 短期/长期 Amihud 比值 (5d/20d) |
| 4 | CSTM_TURN_MOM_5_20 | liquidity | 换手率动量 (5d vs 20d) |
| 5 | CSTM_TURN_CV_20 | liquidity | 换手率波动率 |
| 6 | CSTM_ILLIQ_TREND | liquidity | Amihud 斜率 (20d Slope) |
| 7 | CSTM_KYLE_LAMBDA | microstructure | `Mean(($high - $low) / ($amount + 1), 20)` |
| 8 | CSTM_VOL_AUTOCORR | microstructure | `Corr($volume, Ref($volume, 1), 20)` |
| 9 | CSTM_CLOSE_LOC | microstructure | 收盘价在日内位置 |
| 10 | CSTM_OI_AMOUNT | microstructure | 上涨日成交额占比 |

## Evaluate 结果

| 因子 | Rank IC | ICIR | t-stat | FDR p | 显著 |
|------|---------|------|--------|-------|------|
| CSTM_VOL_AUTOCORR | -0.0178 | **-0.212** | -8.2 | 0.0000 | ✅ |
| CSTM_OI_AMOUNT | -0.0230 | **-0.180** | -6.9 | 0.0000 | ✅ |
| CSTM_AMIHUD_5 | +0.0241 | **+0.167** | +6.4 | 0.0000 | ✅ |
| CSTM_KYLE_LAMBDA | +0.0142 | **+0.144** | +5.6 | 0.0000 | ✅ |
| CSTM_AMIHUD_20 | +0.0195 | **+0.131** | +5.1 | 0.0000 | ✅ |
| CSTM_ILLIQ_MOM | +0.0124 | **+0.115** | +4.4 | 0.0000 | ✅ |
| CSTM_ILLIQ_TREND | +0.0060 | +0.075 | +2.9 | 0.0050 | ✅ (但 \|ICIR\|<0.10) |
| CSTM_CLOSE_LOC | -0.0029 | -0.029 | -1.1 | 0.2663 | ❌ |
| CSTM_TURN_MOM_5_20 | NaN | NaN | NaN | NaN | ❌ (数据问题) |
| CSTM_TURN_CV_20 | NaN | NaN | NaN | NaN | ❌ (数据问题) |

## Validate 结果

### IC 衰减

| 因子 | 1d ICIR | 3d | 5d | 10d | 20d | 5d/1d |
|------|---------|-----|------|------|------|-------|
| CSTM_VOL_AUTOCORR | -0.212 | -0.308 | -0.361 | -0.425 | -0.466 | 1.70 |
| CSTM_AMIHUD_5 | +0.167 | +0.229 | +0.265 | +0.328 | +0.415 | 1.59 |
| CSTM_KYLE_LAMBDA | +0.144 | +0.208 | +0.243 | +0.304 | +0.382 | 1.70 |
| CSTM_AMIHUD_20 | +0.131 | +0.188 | +0.221 | +0.290 | +0.378 | 1.69 |
| CSTM_ILLIQ_MOM | +0.115 | +0.118 | +0.116 | +0.115 | +0.127 | 1.01 |
| CSTM_OI_AMOUNT | -0.180 | -0.205 | -0.211 | -0.177 | -0.234 | 1.17 |

> 所有因子 5d/1d ≥ 0.5 → 全部 `weekly_suitable = True`。
> 注：Amihud/Kyle 类因子随 horizon 增加 ICIR 显著上升，表现出慢衰减特性，非常适合周频调仓。

### 稳定性

| 因子 | stability_score | ic_recent_vs_full |
|------|----------------|-------------------|
| CSTM_OI_AMOUNT | **1.000** | 1.470 |
| CSTM_VOL_AUTOCORR | **0.991** | 1.335 |
| CSTM_AMIHUD_5 | **0.981** | 1.033 |
| CSTM_AMIHUD_20 | **0.941** | 0.958 |
| CSTM_KYLE_LAMBDA | **0.928** | 0.648 |
| CSTM_ILLIQ_MOM | **0.911** | 1.396 |

> 全部 stability ≥ 0.3 且 recent/full ≥ 0.5。

### 正交性（vs 现有 Top30 curated 因子）

| 因子 | max\|ρ\| | 最相关因子 | 通过 |
|------|---------|-----------|------|
| CSTM_AMIHUD_20 | 0.232 | CSTM_GAP_STD_10 | ✅ |
| CSTM_AMIHUD_5 | 0.274 | CSTM_GAP_STD_10 | ✅ |
| CSTM_ILLIQ_MOM | 0.275 | QTLD20 | ✅ |
| CSTM_KYLE_LAMBDA | 0.181 | CSTM_GAP_STD_10 | ✅ |
| CSTM_VOL_AUTOCORR | 0.493 | CSTM_AMT_CV_20 | ✅ (边界) |
| **CSTM_OI_AMOUNT** | **0.690** | CSTM_AMT_WTRET_10 | **❌** |

新因子间冗余：AMIHUD_20 vs AMIHUD_5 ρ=0.894 → 保留 ICIR 更高的 AMIHUD_5。

## 最终决策

| 因子 | 类别 | ICIR | 决策 | 原因 |
|------|------|------|------|------|
| **CSTM_VOL_AUTOCORR** | microstructure | -0.212 | **Promote → Accepted** | 最高\|ICIR\|, 稳定, 正交 |
| **CSTM_AMIHUD_5** | liquidity | +0.167 | **Promote → Accepted** | Amihud 最优窗口, 正交 |
| **CSTM_KYLE_LAMBDA** | microstructure | +0.144 | **Promote → Accepted** | 价格冲击因子, 正交 |
| **CSTM_ILLIQ_MOM** | liquidity | +0.115 | **Promote → Accepted** | 流动性动量, 独特 |
| CSTM_AMIHUD_20 | liquidity | +0.131 | Drop → Redundant | 与 AMIHUD_5 冗余 (ρ=0.894) |
| CSTM_OI_AMOUNT | microstructure | -0.180 | Drop → Rejected | 正交失败 (ρ=0.69 vs AMT_WTRET_10) |
| CSTM_ILLIQ_TREND | liquidity | +0.075 | Drop → Rejected | \|ICIR\| < 0.10 |

**新增 4 个 Accepted 因子（2 liquidity + 2 microstructure），成功填补空白类别。**

## 证据

- 测试输出: `outputs/sfa_liquidity_microstructure_results.csv`
- Validate 输出: `outputs/sfa_liq_micro_validate_results.csv`
- 相关性输出: `outputs/sfa_liq_micro_correlation_check.csv`
- DB: `data/factor_library.db` → `factors` / `factor_test_results` / `factor_ic_decay` / `factor_stability`

"""Large-scale single-factor IC significance test on csiall (all A-shares).

Evaluates a broad pool of candidate factors against Alpha158 baseline factors.
Uses daily cross-sectional Rank IC with t-test and FDR correction.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from scipy import stats

from project_qlib.runtime import init_qlib

# --------------------------------------------------------------------------- #
#  Candidate Factor Pool
#  Organized by alpha dimension. Each entry: (expression, name, category, logic)
# --------------------------------------------------------------------------- #
CANDIDATE_FACTORS = [
    # === 1. Volume / Turnover dynamics ===
    ("$amount / (Mean($amount, 20) + 1e-8)", "CSTM_AMT_SURGE_20",
     "volume", "量能急升(20日)→投机过热→反转"),
    ("$amount / (Mean($amount, 60) + 1e-8)", "CSTM_AMT_SURGE_60",
     "volume", "量能急升(60日)→长期量能偏离"),
    ("$volume / (Mean($volume, 5) + 1e-8)", "CSTM_VOL_SURGE_5",
     "volume", "短期放量→信息冲击"),
    ("Std($volume, 10) / (Mean($volume, 10) + 1e-8)", "CSTM_VOL_CV_10",
     "volume", "成交量变异系数→交易不确定性"),
    ("Std($amount, 20) / (Mean($amount, 20) + 1e-8)", "CSTM_AMT_CV_20",
     "volume", "成交额波动性→流动性风险"),
    ("Mean($volume, 5) / (Mean($volume, 20) + 1e-8)", "CSTM_VOL_RATIO_5_20",
     "volume", "短期vs长期成交量→量能趋势"),

    # === 2. VWAP-based factors ===
    ("$close / Mean($vwap, 5) - 1", "CSTM_VWAP_BIAS_5",
     "vwap", "收盘价偏离5日均VWAP→均值回归压力"),
    ("$close / Mean($vwap, 10) - 1", "CSTM_VWAP_BIAS_10",
     "vwap", "收盘价偏离10日均VWAP"),
    ("$close / Mean($vwap, 20) - 1", "CSTM_VWAP_BIAS_20",
     "vwap", "收盘价偏离20日均VWAP"),
    ("$close / $vwap - 1", "CSTM_VWAP_BIAS_1D",
     "vwap", "日内收盘偏离VWAP→当日尾盘情绪"),
    ("Corr($close/$vwap, $volume/Ref($volume, 1), 10)", "CSTM_VWAP_VOL_CORR_10",
     "vwap", "量价VWAP关联→机构行为痕迹"),
    ("Corr($close/$vwap, $volume/Ref($volume, 1), 20)", "CSTM_VWAP_VOL_CORR_20",
     "vwap", "20日量价VWAP关联"),

    # === 3. Intraday range & volatility ===
    ("($high - $low) / ($close + 1e-8)", "CSTM_RANGE_1D",
     "range", "日内振幅→当日波动强度"),
    ("Mean(($high - $low) / ($close + 1e-8), 5) / (Mean(($high - $low) / ($close + 1e-8), 20) + 1e-8)",
     "CSTM_RANGE_RATIO_5_20", "range", "波幅扩张比→突破/收缩信号"),
    ("Mean(($high - $low) / ($close + 1e-8), 5) / (Mean(($high - $low) / ($close + 1e-8), 60) + 1e-8)",
     "CSTM_RANGE_RATIO_5_60", "range", "波幅扩张比(vs60日)→长周期对比"),
    ("($close - $low) / ($high - $low + 1e-8)", "CSTM_CLOSE_POS",
     "range", "收盘位置→日内强弱"),
    ("Mean(($close - $low) / ($high - $low + 1e-8), 5)", "CSTM_CLOSE_POS_MA5",
     "range", "5日平均收盘位置→持续强弱"),
    ("($high - $close) / ($close - $low + 1e-8)", "CSTM_SHADOW_RATIO",
     "range", "上影/下影线比率→多空力量对比"),
    ("Std(($high-$low)/($close+1e-8), 10)", "CSTM_RANGE_VOL_10",
     "range", "振幅波动率→波动率的波动率"),

    # === 4. Overnight gap / Opening dynamics ===
    ("$open / Ref($close, 1) - 1", "CSTM_GAP_1D",
     "gap", "隔夜跳空→隔夜信息冲击"),
    ("Mean($open / Ref($close, 1) - 1, 5)", "CSTM_GAP_MA_5",
     "gap", "5日平均隔夜跳空→持续隔夜情绪"),
    ("Mean($open / Ref($close, 1) - 1, 10)", "CSTM_GAP_MA_10",
     "gap", "10日平均隔夜跳空"),
    ("Std($open / Ref($close, 1) - 1, 10)", "CSTM_GAP_STD_10",
     "gap", "隔夜跳空波动→隔夜风险"),

    # === 5. Return dynamics (beyond simple momentum) ===
    ("$close/Ref($close, 1) - Ref($close, 1)/Ref($close, 2)", "CSTM_RET_ACCEL_1",
     "momentum", "收益加速度(1日)→动量耗竭/启动"),
    ("Mean($close/Ref($close, 1) - 1, 5) - Mean($close/Ref($close, 1) - 1, 20)", "CSTM_MOM_DIFF_5_20",
     "momentum", "短期vs长期均收益率差→动量切换"),
    ("Ref($close, 1)/$close - 1", "CSTM_REVERT_1",
     "momentum", "1日反转→最短期均值回归"),
    ("Ref($close, 3)/$close - 1", "CSTM_REVERT_3",
     "momentum", "3日反转"),
    ("Ref($close, 5)/$close - 1", "CSTM_REVERT_5",
     "momentum", "5日反转"),
    ("Ref($close, 10)/$close - 1", "CSTM_REVERT_10",
     "momentum", "10日反转"),
    ("Ref($close, 20)/$close - 1", "CSTM_REVERT_20",
     "momentum", "20日反转"),

    # === 6. Price-Volume interaction ===
    ("Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 5)", "CSTM_PV_CORR_5",
     "price_vol", "5日量价相关→信息流方向"),
    ("Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 10)", "CSTM_PV_CORR_10",
     "price_vol", "10日量价相关"),
    ("Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 20)", "CSTM_PV_CORR_20",
     "price_vol", "20日量价相关"),
    ("Corr($close/Ref($close, 1) - 1, $amount/Ref($amount, 1) - 1, 10)", "CSTM_PA_CORR_10",
     "price_vol", "10日收益-成交额相关"),

    # === 7. Skewness / high-order moments ===
    ("Mean(Power($close/Ref($close, 1) - 1, 3), 20) / (Power(Std($close/Ref($close, 1) - 1, 20), 3) + 1e-12)",
     "CSTM_SKEW_20", "higher_moment", "20日收益率偏度→尾部风险"),
    ("Mean(Power($close/Ref($close, 1) - 1, 3), 60) / (Power(Std($close/Ref($close, 1) - 1, 60), 3) + 1e-12)",
     "CSTM_SKEW_60", "higher_moment", "60日收益率偏度"),

    # === 8. Amount-weighted return ===
    ("Mean(($close/Ref($close, 1) - 1) * $amount, 10) / (Mean($amount, 10) + 1e-8)",
     "CSTM_AMT_WTRET_10", "smart_money", "10日成交额加权收益→大单方向"),
    ("Mean(($close/Ref($close, 1) - 1) * $amount, 20) / (Mean($amount, 20) + 1e-8)",
     "CSTM_AMT_WTRET_20", "smart_money", "20日成交额加权收益→大单方向"),

    # === 9. Relative price position ===
    ("$close / Mean($close, 5) - 1", "CSTM_MA_BIAS_5",
     "trend", "5日均线偏离"),
    ("$close / Mean($close, 10) - 1", "CSTM_MA_BIAS_10",
     "trend", "10日均线偏离"),
    ("$close / Mean($close, 20) - 1", "CSTM_MA_BIAS_20",
     "trend", "20日均线偏离"),
    ("$close / Mean($close, 60) - 1", "CSTM_MA_BIAS_60",
     "trend", "60日均线偏离→中线趋势"),
    ("Mean($close, 5) / Mean($close, 20) - 1", "CSTM_MA_CROSS_5_20",
     "trend", "5/20日均线差→趋势方向"),
]


def compute_daily_rankic(factor_series: pd.Series, label_series: pd.Series, df_index) -> dict:
    """Compute daily cross-sectional Rank IC and run t-test."""
    dates = df_index.get_level_values(1).unique()
    daily_ic = []
    daily_rank_ic = []

    for dt in dates:
        mask = df_index.get_level_values(1) == dt
        f_day = factor_series[mask].dropna()
        l_day = label_series[mask].dropna()
        common = f_day.index.intersection(l_day.index)
        if len(common) < 50:  # need enough stocks
            continue
        f_vals = f_day.loc[common]
        l_vals = l_day.loc[common]

        ic = f_vals.corr(l_vals)
        rank_ic = f_vals.rank().corr(l_vals.rank())
        if not np.isnan(ic):
            daily_ic.append(ic)
        if not np.isnan(rank_ic):
            daily_rank_ic.append(rank_ic)

    if len(daily_rank_ic) < 30:
        return {"n_days": len(daily_rank_ic), "rank_ic_mean": np.nan, "rank_ic_t": np.nan, "rank_ic_p": 1.0}

    ic_arr = np.array(daily_ic)
    ric_arr = np.array(daily_rank_ic)
    n = len(ric_arr)

    ic_mean = np.nanmean(ic_arr) if len(ic_arr) > 0 else np.nan
    ic_std = np.nanstd(ic_arr, ddof=1) if len(ic_arr) > 1 else np.nan

    ric_mean = np.nanmean(ric_arr)
    ric_std = np.nanstd(ric_arr, ddof=1)
    ric_t = ric_mean / (ric_std / np.sqrt(n)) if ric_std > 0 else 0
    ric_p = 2 * (1 - stats.t.cdf(abs(ric_t), df=n - 1))

    return {
        "n_days": n,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "rank_ic_mean": ric_mean,
        "rank_ic_std": ric_std,
        "rank_ic_t": ric_t,
        "rank_ic_p": ric_p,
        "rank_icir": ric_mean / ric_std if ric_std > 0 else 0,
    }


def main():
    init_qlib()
    from qlib.data import D

    instruments = "csiall"
    # Use all available data for maximum statistical power
    start_time = "2019-01-01"
    end_time = "2025-12-31"

    # Label: next-day return
    label_expr = "Ref($close, -2)/Ref($close, -1) - 1"

    all_fields = [expr for expr, _, _, _ in CANDIDATE_FACTORS] + [label_expr]
    all_names = [name for _, name, _, _ in CANDIDATE_FACTORS] + ["LABEL"]

    print(f"Loading {len(CANDIDATE_FACTORS)} candidate factors on {instruments}...")
    print(f"Period: {start_time} to {end_time}")
    print("This may take a few minutes for full market...")

    df = D.features(
        instruments=D.instruments(instruments),
        fields=all_fields,
        start_time=start_time,
        end_time=end_time,
    )
    df.columns = all_names
    print(f"Data shape: {df.shape}")
    n_stocks = df.index.get_level_values(0).nunique()
    n_dates = df.index.get_level_values(1).nunique()
    print(f"Stocks: {n_stocks}, Dates: {n_dates}")
    print()

    label = df["LABEL"]
    results = []

    for i, (expr, name, category, logic) in enumerate(CANDIDATE_FACTORS):
        print(f"  [{i+1}/{len(CANDIDATE_FACTORS)}] {name}...", end="", flush=True)
        factor = df[name]
        non_null = int(factor.notna().sum())
        res = compute_daily_rankic(factor, label, df.index)
        res["factor"] = name
        res["category"] = category
        res["logic"] = logic
        res["non_null"] = non_null
        results.append(res)
        print(f" RankIC={res.get('rank_ic_mean', float('nan')):.5f}, t={res.get('rank_ic_t', float('nan')):.2f}")

    # FDR correction (Benjamini-Hochberg)
    res_df = pd.DataFrame(results)
    p_vals = res_df["rank_ic_p"].fillna(1.0).values
    n_tests = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    fdr_p = np.ones(n_tests)
    for rank, idx in enumerate(sorted_idx, 1):
        fdr_p[idx] = p_vals[idx] * n_tests / rank
    # Ensure monotonicity
    fdr_sorted = fdr_p[sorted_idx]
    for i in range(len(fdr_sorted) - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr_p[sorted_idx] = fdr_sorted
    fdr_p = np.minimum(fdr_p, 1.0)
    res_df["rank_ic_p_fdr"] = fdr_p

    # Sort by absolute t-stat
    res_df = res_df.sort_values("rank_ic_t", ascending=False, key=abs)

    # Print results
    print()
    print("=" * 120)
    print("  CSIALL Single-Factor Significance Test (全市场, 2019-2025)")
    print("=" * 120)
    print(f"{'Factor':<25} {'Cat':<12} {'Days':>5} {'IC':>10} {'RankIC':>10} {'RankIC_t':>10} {'p':>10} {'FDR_p':>10} {'RankICIR':>10} {'Sig':>5}")
    print("-" * 120)
    for _, row in res_df.iterrows():
        sig = "***" if row["rank_ic_p_fdr"] < 0.01 else "**" if row["rank_ic_p_fdr"] < 0.05 else "*" if row["rank_ic_p_fdr"] < 0.1 else ""
        print(f"{row['factor']:<25} {row['category']:<12} {row['n_days']:>5} {row.get('ic_mean', np.nan):>10.5f} {row['rank_ic_mean']:>10.5f} {row['rank_ic_t']:>10.3f} {row['rank_ic_p']:>10.6f} {row['rank_ic_p_fdr']:>10.6f} {row['rank_icir']:>10.4f} {sig:>5}")

    # Save full results
    out_path = PROJECT_ROOT / "outputs" / "csiall_factor_significance.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    # Print top-20 by absolute t-stat
    top20 = res_df.head(20)
    top_path = PROJECT_ROOT / "outputs" / "csiall_factor_top20.csv"
    top20.to_csv(top_path, index=False)
    print(f"Top-20 factors saved to {top_path}")

    # Summary
    sig_001 = (res_df["rank_ic_p_fdr"] < 0.01).sum()
    sig_005 = (res_df["rank_ic_p_fdr"] < 0.05).sum()
    print(f"\nSummary: {sig_001} factors significant at FDR<0.01, {sig_005} at FDR<0.05 (out of {len(res_df)})")


if __name__ == "__main__":
    main()

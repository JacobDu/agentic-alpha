"""Unified IC test: Alpha158 (158) + Custom (43) = 201 factors on csiall.

Loads factors in batches to manage memory, computes daily cross-sectional
Rank IC, then produces a unified ranking with FDR correction.
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

# ---- Custom factor definitions (same as test_factors_csiall.py) ----
CUSTOM_FACTORS = [
    ("$amount / (Mean($amount, 20) + 1e-8)", "CSTM_AMT_SURGE_20", "volume"),
    ("$amount / (Mean($amount, 60) + 1e-8)", "CSTM_AMT_SURGE_60", "volume"),
    ("$volume / (Mean($volume, 5) + 1e-8)", "CSTM_VOL_SURGE_5", "volume"),
    ("Std($volume, 10) / (Mean($volume, 10) + 1e-8)", "CSTM_VOL_CV_10", "volume"),
    ("Std($amount, 20) / (Mean($amount, 20) + 1e-8)", "CSTM_AMT_CV_20", "volume"),
    ("Mean($volume, 5) / (Mean($volume, 20) + 1e-8)", "CSTM_VOL_RATIO_5_20", "volume"),
    ("$close / Mean($vwap, 5) - 1", "CSTM_VWAP_BIAS_5", "vwap"),
    ("$close / Mean($vwap, 10) - 1", "CSTM_VWAP_BIAS_10", "vwap"),
    ("$close / Mean($vwap, 20) - 1", "CSTM_VWAP_BIAS_20", "vwap"),
    ("$close / $vwap - 1", "CSTM_VWAP_BIAS_1D", "vwap"),
    ("Corr($close/$vwap, $volume/Ref($volume, 1), 10)", "CSTM_VWAP_VOL_CORR_10", "vwap"),
    ("Corr($close/$vwap, $volume/Ref($volume, 1), 20)", "CSTM_VWAP_VOL_CORR_20", "vwap"),
    ("($high - $low) / ($close + 1e-8)", "CSTM_RANGE_1D", "range"),
    ("Mean(($high - $low) / ($close + 1e-8), 5) / (Mean(($high - $low) / ($close + 1e-8), 20) + 1e-8)",
     "CSTM_RANGE_RATIO_5_20", "range"),
    ("Mean(($high - $low) / ($close + 1e-8), 5) / (Mean(($high - $low) / ($close + 1e-8), 60) + 1e-8)",
     "CSTM_RANGE_RATIO_5_60", "range"),
    ("($close - $low) / ($high - $low + 1e-8)", "CSTM_CLOSE_POS", "range"),
    ("Mean(($close - $low) / ($high - $low + 1e-8), 5)", "CSTM_CLOSE_POS_MA5", "range"),
    ("($high - $close) / ($close - $low + 1e-8)", "CSTM_SHADOW_RATIO", "range"),
    ("Std(($high-$low)/($close+1e-8), 10)", "CSTM_RANGE_VOL_10", "range"),
    ("$open / Ref($close, 1) - 1", "CSTM_GAP_1D", "gap"),
    ("Mean($open / Ref($close, 1) - 1, 5)", "CSTM_GAP_MA_5", "gap"),
    ("Mean($open / Ref($close, 1) - 1, 10)", "CSTM_GAP_MA_10", "gap"),
    ("Std($open / Ref($close, 1) - 1, 10)", "CSTM_GAP_STD_10", "gap"),
    ("$close/Ref($close, 1) - Ref($close, 1)/Ref($close, 2)", "CSTM_RET_ACCEL_1", "momentum"),
    ("Mean($close/Ref($close, 1) - 1, 5) - Mean($close/Ref($close, 1) - 1, 20)", "CSTM_MOM_DIFF_5_20", "momentum"),
    ("Ref($close, 1)/$close - 1", "CSTM_REVERT_1", "momentum"),
    ("Ref($close, 3)/$close - 1", "CSTM_REVERT_3", "momentum"),
    ("Ref($close, 5)/$close - 1", "CSTM_REVERT_5", "momentum"),
    ("Ref($close, 10)/$close - 1", "CSTM_REVERT_10", "momentum"),
    ("Ref($close, 20)/$close - 1", "CSTM_REVERT_20", "momentum"),
    ("Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 5)", "CSTM_PV_CORR_5", "price_vol"),
    ("Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 10)", "CSTM_PV_CORR_10", "price_vol"),
    ("Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 20)", "CSTM_PV_CORR_20", "price_vol"),
    ("Corr($close/Ref($close, 1) - 1, $amount/Ref($amount, 1) - 1, 10)", "CSTM_PA_CORR_10", "price_vol"),
    ("Mean(Power($close/Ref($close, 1) - 1, 3), 20) / (Power(Std($close/Ref($close, 1) - 1, 20), 3) + 1e-12)",
     "CSTM_SKEW_20", "higher_moment"),
    ("Mean(Power($close/Ref($close, 1) - 1, 3), 60) / (Power(Std($close/Ref($close, 1) - 1, 60), 3) + 1e-12)",
     "CSTM_SKEW_60", "higher_moment"),
    ("Mean(($close/Ref($close, 1) - 1) * $amount, 10) / (Mean($amount, 10) + 1e-8)",
     "CSTM_AMT_WTRET_10", "smart_money"),
    ("Mean(($close/Ref($close, 1) - 1) * $amount, 20) / (Mean($amount, 20) + 1e-8)",
     "CSTM_AMT_WTRET_20", "smart_money"),
    ("$close / Mean($close, 5) - 1", "CSTM_MA_BIAS_5", "trend"),
    ("$close / Mean($close, 10) - 1", "CSTM_MA_BIAS_10", "trend"),
    ("$close / Mean($close, 20) - 1", "CSTM_MA_BIAS_20", "trend"),
    ("$close / Mean($close, 60) - 1", "CSTM_MA_BIAS_60", "trend"),
    ("Mean($close, 5) / Mean($close, 20) - 1", "CSTM_MA_CROSS_5_20", "trend"),
]

LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"


def get_alpha158_factors():
    """Extract Alpha158 factor definitions."""
    from qlib.contrib.data.handler import Alpha158
    h = Alpha158.__new__(Alpha158)
    fields, names = h.get_feature_config()
    return list(fields), list(names)


def compute_daily_rankic(factor: pd.Series, label: pd.Series, dates) -> dict:
    daily_ic = []
    daily_ric = []
    for dt in dates:
        mask_f = factor.index.get_level_values(1) == dt
        mask_l = label.index.get_level_values(1) == dt
        f_day = factor[mask_f].dropna()
        l_day = label[mask_l].dropna()
        common = f_day.index.intersection(l_day.index)
        if len(common) < 50:
            continue
        fv = f_day.loc[common]
        lv = l_day.loc[common]
        ic = fv.corr(lv)
        ric = fv.rank().corr(lv.rank())
        if not np.isnan(ic):
            daily_ic.append(ic)
        if not np.isnan(ric):
            daily_ric.append(ric)
    if len(daily_ric) < 30:
        return {"n_days": len(daily_ric), "ic_mean": np.nan, "rank_ic_mean": np.nan,
                "rank_ic_t": np.nan, "rank_ic_p": 1.0, "rank_icir": np.nan}
    ic_arr = np.array(daily_ic)
    ric_arr = np.array(daily_ric)
    n = len(ric_arr)
    ic_mean = np.nanmean(ic_arr)
    ric_mean = np.nanmean(ric_arr)
    ric_std = np.nanstd(ric_arr, ddof=1)
    ric_t = ric_mean / (ric_std / np.sqrt(n)) if ric_std > 0 else 0
    ric_p = 2 * (1 - stats.t.cdf(abs(ric_t), df=n - 1))
    return {
        "n_days": n,
        "ic_mean": ic_mean,
        "rank_ic_mean": ric_mean,
        "rank_ic_t": ric_t,
        "rank_ic_p": ric_p,
        "rank_icir": ric_mean / ric_std if ric_std > 0 else 0,
    }


def process_batch(instruments_obj, factor_list, factor_names, categories, label_data, dates,
                  start_time, end_time, batch_label):
    """Load a batch of factors and compute IC stats."""
    from qlib.data import D
    print(f"\n  Loading batch '{batch_label}' ({len(factor_list)} factors)...")
    df = D.features(instruments_obj, factor_list + [LABEL_EXPR],
                    start_time=start_time, end_time=end_time)
    df.columns = factor_names + ["LABEL"]
    if label_data[0] is None:
        label_data[0] = df["LABEL"]
        dates.extend(df.index.get_level_values(1).unique().tolist())
    label = label_data[0]

    results = []
    for i, name in enumerate(factor_names):
        print(f"    [{i+1}/{len(factor_names)}] {name}...", end="", flush=True)
        res = compute_daily_rankic(df[name], label, dates)
        res["factor"] = name
        res["category"] = categories[i]
        res["source"] = batch_label
        print(f" RankIC={res['rank_ic_mean']:.5f}, t={res['rank_ic_t']:.2f}")
        results.append(res)
    return results


def main():
    init_qlib()
    from qlib.data import D

    instruments = "csiall"
    start_time = "2019-01-01"
    end_time = "2025-12-31"

    # Get Alpha158 factors
    a158_fields, a158_names = get_alpha158_factors()
    a158_cats = ["alpha158"] * len(a158_names)

    # Custom factors
    cstm_fields = [f for f, _, _ in CUSTOM_FACTORS]
    cstm_names = [n for _, n, _ in CUSTOM_FACTORS]
    cstm_cats = [c for _, _, c in CUSTOM_FACTORS]

    total = len(a158_names) + len(cstm_names)
    print(f"Total factors to test: {total} (Alpha158: {len(a158_names)}, Custom: {len(cstm_names)})")
    print(f"Market: {instruments}, Period: {start_time} to {end_time}")

    instruments_obj = D.instruments(instruments)
    label_data = [None]
    dates = []
    all_results = []

    # Process Alpha158 in batches of 40
    batch_size = 40
    for start_idx in range(0, len(a158_fields), batch_size):
        end_idx = min(start_idx + batch_size, len(a158_fields))
        batch_results = process_batch(
            instruments_obj,
            a158_fields[start_idx:end_idx],
            a158_names[start_idx:end_idx],
            a158_cats[start_idx:end_idx],
            label_data, dates, start_time, end_time,
            f"Alpha158[{start_idx}:{end_idx}]"
        )
        all_results.extend(batch_results)

    # Process custom factors in one batch
    batch_results = process_batch(
        instruments_obj,
        cstm_fields, cstm_names, cstm_cats,
        label_data, dates, start_time, end_time,
        "Custom"
    )
    all_results.extend(batch_results)

    # Build DataFrame and apply FDR
    res_df = pd.DataFrame(all_results)
    p_vals = res_df["rank_ic_p"].fillna(1.0).values
    n_tests = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    fdr_p = np.ones(n_tests)
    for rank, idx in enumerate(sorted_idx, 1):
        fdr_p[idx] = p_vals[idx] * n_tests / rank
    fdr_sorted = fdr_p[sorted_idx]
    for i in range(len(fdr_sorted) - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr_p[sorted_idx] = fdr_sorted
    res_df["rank_ic_p_fdr"] = np.minimum(fdr_p, 1.0)

    # Sort by absolute t-stat
    res_df = res_df.sort_values("rank_ic_t", ascending=False, key=abs)

    # Print top-50
    print("\n" + "=" * 130)
    print("  UNIFIED FACTOR RANKING: Alpha158 + Custom on CSIALL (2019-2025)")
    print("=" * 130)
    print(f"{'Rank':>4} {'Factor':<25} {'Source':<12} {'Cat':<14} {'RankIC':>10} {'RankIC_t':>10} {'FDR_p':>12} {'ICIR':>10} {'Sig':>5}")
    print("-" * 130)
    for rank, (_, row) in enumerate(res_df.head(50).iterrows(), 1):
        sig = "***" if row["rank_ic_p_fdr"] < 0.01 else "**" if row["rank_ic_p_fdr"] < 0.05 else "*" if row["rank_ic_p_fdr"] < 0.1 else ""
        print(f"{rank:>4} {row['factor']:<25} {row['source']:<12} {row['category']:<14} {row['rank_ic_mean']:>10.5f} {row['rank_ic_t']:>10.3f} {row['rank_ic_p_fdr']:>12.8f} {row['rank_icir']:>10.4f} {sig:>5}")

    # Save all results
    out_full = PROJECT_ROOT / "outputs" / "csiall_unified_factor_ranking.csv"
    res_df.to_csv(out_full, index=False)
    print(f"\nFull ranking ({len(res_df)} factors) saved to {out_full}")

    # Save top-50
    out_top = PROJECT_ROOT / "outputs" / "csiall_unified_top50.csv"
    res_df.head(50).to_csv(out_top, index=False)
    print(f"Top-50 saved to {out_top}")

    # Summary stats
    sig_001 = (res_df["rank_ic_p_fdr"] < 0.01).sum()
    a158_in_top20 = res_df.head(20)["source"].str.contains("Alpha158").sum()
    cstm_in_top20 = res_df.head(20)["source"].str.contains("Custom").sum()
    print(f"\nSummary:")
    print(f"  Total significant (FDR<0.01): {sig_001}/{len(res_df)}")
    print(f"  Top-20 composition: Alpha158={a158_in_top20}, Custom={cstm_in_top20}")


if __name__ == "__main__":
    main()

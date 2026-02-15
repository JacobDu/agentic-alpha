"""Analyze individual factor IC significance on CSI1000."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from project_qlib.runtime import init_qlib


def main():
    init_qlib()
    from qlib.data import D

    config_path = PROJECT_ROOT / "configs" / "workflow_csi1000_custom_lightgbm.yaml"
    config = yaml.safe_load(config_path.read_text())

    # Use test period for factor evaluation
    instruments = "csi1000"
    start_time = "2024-07-01"
    end_time = "2025-12-31"

    # Custom factor definitions
    custom_factors = [
        ("$close / Mean($vwap, 5) - 1", "CSTM_VWAP_BIAS_5"),
        ("$amount / (Mean($amount, 20) + 1e-8)", "CSTM_AMT_SURGE"),
        ("($close - $low) / ($high - $low + 1e-8)", "CSTM_CLOSE_POS"),
        ("Mean($open / Ref($close, 1) - 1, 10)", "CSTM_GAP_MA_10"),
        (
            "Mean(($high - $low) / ($close + 1e-8), 5)"
            " / (Mean(($high - $low) / ($close + 1e-8), 20) + 1e-8)",
            "CSTM_RANGE_RATIO",
        ),
        ("$close/Ref($close, 1) - Ref($close, 1)/Ref($close, 2)", "CSTM_PRICE_ACCEL"),
        ("Corr($close/$vwap, $volume/Ref($volume, 1), 10)", "CSTM_VWAP_VOL_CORR"),
    ]

    # Also add the label (next-day return)
    label_expr = "Ref($close, -2)/Ref($close, -1) - 1"

    all_fields = [expr for expr, _ in custom_factors] + [label_expr]
    all_names = [name for _, name in custom_factors] + ["LABEL"]

    print("Loading factor data for CSI1000 test period...")
    df = D.features(
        instruments=D.instruments(instruments),
        fields=all_fields,
        start_time=start_time,
        end_time=end_time,
    )
    df.columns = all_names
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.get_level_values(1).min()} to {df.index.get_level_values(1).max()}")
    print()

    # Compute daily cross-sectional IC (rank IC)
    label = df["LABEL"]
    results = []

    for factor_name in [n for _, n in custom_factors]:
        factor = df[factor_name]
        # Daily cross-sectional Rank IC
        dates = df.index.get_level_values(1).unique()
        daily_ic = []
        daily_rank_ic = []

        for dt in dates:
            mask = df.index.get_level_values(1) == dt
            f_day = factor[mask].dropna()
            l_day = label[mask].dropna()
            common = f_day.index.intersection(l_day.index)
            if len(common) < 30:
                continue
            f_vals = f_day.loc[common]
            l_vals = l_day.loc[common]

            # Pearson IC
            ic = f_vals.corr(l_vals)
            daily_ic.append(ic)

            # Rank IC (Spearman)
            rank_ic = f_vals.rank().corr(l_vals.rank())
            daily_rank_ic.append(rank_ic)

        if not daily_ic:
            results.append({
                "factor": factor_name,
                "n_days": 0,
                "ic_mean": float("nan"),
                "rank_ic_mean": float("nan"),
            })
            continue

        ic_arr = np.array(daily_ic)
        ric_arr = np.array(daily_rank_ic)

        ic_mean = np.nanmean(ic_arr)
        ic_std = np.nanstd(ic_arr, ddof=1)
        n = len(ic_arr)
        ic_t = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0
        ic_p = 2 * (1 - stats.t.cdf(abs(ic_t), df=n - 1))

        ric_mean = np.nanmean(ric_arr)
        ric_std = np.nanstd(ric_arr, ddof=1)
        ric_t = ric_mean / (ric_std / np.sqrt(n)) if ric_std > 0 else 0
        ric_p = 2 * (1 - stats.t.cdf(abs(ric_t), df=n - 1))

        icir = ic_mean / ic_std if ic_std > 0 else 0
        ricir = ric_mean / ric_std if ric_std > 0 else 0

        non_null = int(factor.notna().sum())

        results.append({
            "factor": factor_name,
            "n_days": n,
            "non_null": non_null,
            "ic_mean": ic_mean,
            "ic_t": ic_t,
            "ic_p": ic_p,
            "icir": icir,
            "rank_ic_mean": ric_mean,
            "rank_ic_t": ric_t,
            "rank_ic_p": ric_p,
            "rank_icir": ricir,
        })

    # Print results
    print("=" * 100)
    print("  CSI1000 Custom Factor Significance (Test Period: 2024-07 to 2025-12)")
    print("=" * 100)
    print(f"{'Factor':<22} {'Days':>5} {'IC_mean':>10} {'IC_t':>8} {'IC_p':>8} {'ICIR':>8} {'RankIC':>10} {'RankIC_t':>10} {'RankIC_p':>10} {'RankICIR':>10}")
    print("-" * 100)

    res_df = pd.DataFrame(results).sort_values("rank_ic_t", ascending=False, key=abs)
    for _, row in res_df.iterrows():
        sig = "***" if row.get("rank_ic_p", 1) < 0.01 else "**" if row.get("rank_ic_p", 1) < 0.05 else "*" if row.get("rank_ic_p", 1) < 0.1 else ""
        print(f"{row['factor']:<22} {row.get('n_days', 0):>5} {row.get('ic_mean', float('nan')):>10.6f} {row.get('ic_t', float('nan')):>8.3f} {row.get('ic_p', float('nan')):>8.4f} {row.get('icir', float('nan')):>8.4f} {row.get('rank_ic_mean', float('nan')):>10.6f} {row.get('rank_ic_t', float('nan')):>10.3f} {row.get('rank_ic_p', float('nan')):>10.4f} {row.get('rank_icir', float('nan')):>10.4f} {sig}")

    # Save to CSV
    out_path = PROJECT_ROOT / "outputs" / "csi1000_factor_significance.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

"""HEA Round: Batch test new candidate factors.

Focus areas (based on gap analysis):
1. Valuation × Volume/Volatility interactions
2. Multi-scale momentum differences
3. Valuation quantile (relative position in history)
4. Turnover regime changes
5. Volume microstructure improvements
6. Price efficiency / impact metrics
7. Composite cross-field signals

All factors use Phase 1 fields only:
  Price: $open, $close, $high, $low
  Volume: $volume, $amount, $vwap
  Derived: $change, $factor, $turnover_rate
  Valuation: $pe_ttm, $pb_mrq, $ps_ttm, $pcf_ttm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Cannot locate project root (pyproject.toml not found)")

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------
# NEW CANDIDATE FACTORS
# -----------------------------------------------------------------------
NEW_FACTORS = [
    # ============================================================
    # A. Valuation × Volatility Interaction
    # ============================================================
    # PE stability: low PE + low PE volatility → quality value
    ("Div(Std($pe_ttm, 60), Mean(Abs($pe_ttm), 60) + 1e-8)",
     "NEW_PE_CV_60", "val_vol"),
    # PB stability over 60d
    ("Div(Std($pb_mrq, 60), Mean(Abs($pb_mrq), 60) + 1e-8)",
     "NEW_PB_CV_60", "val_vol"),
    # PS stability over 60d
    ("Div(Std($ps_ttm, 60), Mean(Abs($ps_ttm), 60) + 1e-8)",
     "NEW_PS_CV_60", "val_vol"),
    # PE volatility relative to price volatility
    ("Div(Std($pe_ttm, 20), Mean(Abs($pe_ttm), 20) + 1e-8) / (Std($close, 20) / (Mean($close, 20) + 1e-8) + 1e-8)",
     "NEW_PE_PRICE_VOL_RATIO", "val_vol"),

    # ============================================================
    # B. Valuation × Turnover Interaction
    # ============================================================
    # Cheap + High turnover: undervalued stocks attracting attention
    ("Mul(Div(1, $pb_mrq + 1e-8), $turnover_rate)",
     "NEW_BP_TURN", "val_turn"),
    # EP × Turnover surge (value + recent activity)
    ("Mul(Div(1, $pe_ttm + 1e-8), Div(Mean($turnover_rate, 5), Mean($turnover_rate, 60) + 1e-8))",
     "NEW_EP_TURN_SURGE", "val_turn"),
    # Low PS + high volume growth → revenue attractiveness
    ("Mul(Div(1, $ps_ttm + 1e-8), Div($amount, Mean($amount, 60) + 1e-8))",
     "NEW_SP_AMT_SURGE", "val_turn"),
    # PCF × turnover: cheap cash flow + active trading
    ("Mul(Div(1, Abs($pcf_ttm) + 1e-8), Mean($turnover_rate, 5))",
     "NEW_CFP_TURN", "val_turn"),

    # ============================================================
    # C. Multi-scale Momentum Differences
    # ============================================================
    # Short vs medium momentum difference (5d vs 20d)
    ("Div($close, Ref($close, 5)) - Div($close, Ref($close, 20))",
     "NEW_MOM_DIFF_5_20", "momentum_diff"),
    # Medium vs long momentum difference (20d vs 60d)
    ("Div($close, Ref($close, 20)) - Div($close, Ref($close, 60))",
     "NEW_MOM_DIFF_20_60", "momentum_diff"),
    # Short vs long momentum difference (5d vs 60d)
    ("Div($close, Ref($close, 5)) - Div($close, Ref($close, 60))",
     "NEW_MOM_DIFF_5_60", "momentum_diff"),
    # Momentum acceleration: (5d ret - Ref(5d ret, 5)) / 5d_std
    ("(Div($close, Ref($close, 5)) - Div(Ref($close, 5), Ref($close, 10))) / (Std(Div($close, Ref($close, 1)) - 1, 10) + 1e-8)",
     "NEW_MOM_ACCEL_5", "momentum_diff"),
    # 10d vs 60d: intermediate
    ("Div($close, Ref($close, 10)) - Div($close, Ref($close, 60))",
     "NEW_MOM_DIFF_10_60", "momentum_diff"),

    # ============================================================
    # D. Valuation Quantile / Relative Position
    # ============================================================
    # PE relative to 120d min-max range (where in the range)
    ("Div($pe_ttm - Min($pe_ttm, 120), Max($pe_ttm, 120) - Min($pe_ttm, 120) + 1e-8)",
     "NEW_PE_QUANTILE_120", "val_quantile"),
    # PB relative to 120d range
    ("Div($pb_mrq - Min($pb_mrq, 120), Max($pb_mrq, 120) - Min($pb_mrq, 120) + 1e-8)",
     "NEW_PB_QUANTILE_120", "val_quantile"),
    # PE relative to 240d range (annual)
    ("Div($pe_ttm - Min($pe_ttm, 240), Max($pe_ttm, 240) - Min($pe_ttm, 240) + 1e-8)",
     "NEW_PE_QUANTILE_240", "val_quantile"),
    # Close relative to 120d range (price quantile)
    ("Div($close - Min($close, 120), Max($close, 120) - Min($close, 120) + 1e-8)",
     "NEW_CLOSE_QUANTILE_120", "val_quantile"),
    # Turnover relative to 120d range
    ("Div($turnover_rate - Min($turnover_rate, 120), Max($turnover_rate, 120) - Min($turnover_rate, 120) + 1e-8)",
     "NEW_TURN_QUANTILE_120", "val_quantile"),

    # ============================================================
    # E. Turnover Regime Changes
    # ============================================================
    # Turnover persistence: autocorrelation proxy
    ("Corr($turnover_rate, Ref($turnover_rate, 1), 20)",
     "NEW_TURN_PERSIST_20", "turn_regime"),
    # Turnover trend: slope proxy (MA5/MA20 vs prior period)
    ("Div(Mean($turnover_rate, 5), Mean($turnover_rate, 20) + 1e-8) - Div(Ref(Mean($turnover_rate, 5), 10), Ref(Mean($turnover_rate, 20), 10) + 1e-8)",
     "NEW_TURN_TREND", "turn_regime"),
    # Turnover max / mean ratio (peakiness)
    ("Div(Max($turnover_rate, 20), Mean($turnover_rate, 20) + 1e-8)",
     "NEW_TURN_PEAK_20", "turn_regime"),
    # Amount skewness proxy: (max - mean) / std
    ("Div(Max($amount, 20) - Mean($amount, 20), Std($amount, 20) + 1e-8)",
     "NEW_AMT_SKEW_20", "turn_regime"),

    # ============================================================
    # F. Volume Microstructure
    # ============================================================
    # VWAP deviation from close (institutional vs retail)
    ("Div($vwap - $close, $close + 1e-8)",
     "NEW_VWAP_DEV", "microstructure"),
    # VWAP deviation MA (persistent institutional flow)
    ("Mean(Div($vwap - $close, $close + 1e-8), 10)",
     "NEW_VWAP_DEV_MA10", "microstructure"),
    # Amount per unit of high-low range (liquidity depth)
    ("Div($amount, ($high - $low) + 1e-8)",
     "NEW_AMT_PER_RANGE", "microstructure"),
    # Volume concentration: max_daily_vol / sum_vol over 5d
    ("Div(Max($volume, 5), Sum($volume, 5) + 1e-8)",
     "NEW_VOL_CONC_5", "microstructure"),
    # Close position in daily range (buying pressure)
    ("Div($close - $low, $high - $low + 1e-8)",
     "NEW_CLOSE_POS", "microstructure"),
    # Close position MA10 (persistent buying pressure)
    ("Mean(Div($close - $low, $high - $low + 1e-8), 10)",
     "NEW_CLOSE_POS_MA10", "microstructure"),

    # ============================================================
    # G. Price Efficiency / Impact
    # ============================================================
    # Amihud illiquidity: |return| / amount
    ("Mean(Div(Abs(Div($close, Ref($close, 1)) - 1), $amount + 1e-8), 20)",
     "NEW_AMIHUD_20", "efficiency"),
    # Price impact: |return| / turnover
    ("Mean(Div(Abs(Div($close, Ref($close, 1)) - 1), $turnover_rate + 1e-8), 20)",
     "NEW_PRICE_IMPACT_20", "efficiency"),
    # Return per unit volume (efficiency)
    ("Div(Abs(Div($close, Ref($close, 5)) - 1), Mean($volume, 5) + 1e-8)",
     "NEW_RET_PER_VOL_5", "efficiency"),

    # ============================================================
    # H. Mean Reversion Signals
    # ============================================================
    # Distance from 5d MA (short-term reversion)
    ("Div($close - Mean($close, 5), Mean($close, 5) + 1e-8)",
     "NEW_MA_DEV_5", "mean_reversion"),
    # Distance from 10d MA
    ("Div($close - Mean($close, 10), Mean($close, 10) + 1e-8)",
     "NEW_MA_DEV_10", "mean_reversion"),
    # Distance from 20d MA
    ("Div($close - Mean($close, 20), Mean($close, 20) + 1e-8)",
     "NEW_MA_DEV_20", "mean_reversion"),
    # Distance from 60d MA
    ("Div($close - Mean($close, 60), Mean($close, 60) + 1e-8)",
     "NEW_MA_DEV_60", "mean_reversion"),
    # Z-score: (close - MA20) / Std20
    ("Div($close - Mean($close, 20), Std($close, 20) + 1e-8)",
     "NEW_ZSCORE_20", "mean_reversion"),
    # Z-score 60d
    ("Div($close - Mean($close, 60), Std($close, 60) + 1e-8)",
     "NEW_ZSCORE_60", "mean_reversion"),

    # ============================================================
    # I. Composite Cross-field Signals
    # ============================================================
    # Value + Low Vol: BP × (1/RangeVol)
    ("Div(Div(1, $pb_mrq + 1e-8), Std(Div($high - $low, $close + 1e-8), 20) + 1e-8)",
     "NEW_VALUE_LOWVOL", "composite"),
    # Momentum-Volume divergence: positive return + declining volume
    ("Mul(Div($close, Ref($close, 10)) - 1, 0 - Div($volume, Mean($volume, 20) + 1e-8))",
     "NEW_MOM_VOL_DIV_10", "composite"),
    # PE change vs price change ratio (fundamental vs price momentum alignment)
    ("Div(Div($pe_ttm, Ref($pe_ttm, 20)) - 1, Div($close, Ref($close, 20)) - 1 + 1e-8)",
     "NEW_PE_PRICE_ALIGNMENT", "composite"),
    # Turnover-adjusted momentum: ret / sqrt(turnover)
    ("Div(Div($close, Ref($close, 20)) - 1, Std($turnover_rate, 20) + 1e-8)",
     "NEW_MOM_TURN_ADJ_20", "composite"),
]


def _load_industry() -> pd.Series | None:
    """Load industry classification from data/industry.parquet."""
    ind_file = PROJECT_ROOT / "data" / "industry.parquet"
    if not ind_file.exists():
        print("  [WARN] Industry file not found, skipping neutralization.")
        return None
    ind_df = pd.read_parquet(ind_file)
    return ind_df.set_index("qlib_code")["industry"]


def _neutralize_factor(factor_series: pd.Series,
                       industry_map: pd.Series) -> pd.Series:
    """Industry-neutralize factor values (vectorized Z-score within industry)."""
    result = factor_series.copy()
    dates = factor_series.index.get_level_values("datetime")
    instruments = factor_series.index.get_level_values("instrument")
    tmp = pd.DataFrame({
        "factor": factor_series.values,
        "datetime": dates,
        "instrument": instruments,
    })
    tmp["industry"] = tmp["instrument"].map(industry_map)
    tmp.loc[tmp["industry"].isna(), "factor"] = np.nan
    grouped = tmp.groupby(["datetime", "industry"])["factor"]
    g_mean = grouped.transform("mean")
    g_std = grouped.transform("std")
    g_std = g_std.replace(0, np.nan)
    tmp["neutral"] = (tmp["factor"] - g_mean) / g_std
    result[:] = tmp["neutral"].values
    return result


def compute_ic(factor_df: pd.DataFrame, label_df: pd.DataFrame,
               method: str = "spearman",
               industry_map: pd.Series | None = None) -> pd.Series:
    """Compute daily IC between factor and label."""
    common = factor_df.index.intersection(label_df.index)
    factor_aligned = factor_df.loc[common].iloc[:, 0]
    label_aligned = label_df.loc[common].iloc[:, 0]

    if industry_map is not None:
        factor_aligned = _neutralize_factor(factor_aligned, industry_map)

    dates = common.get_level_values("datetime")
    unique_dates = dates.unique()

    ics = {}
    for dt in unique_dates:
        mask = dates == dt
        f = factor_aligned[mask].values
        l = label_aligned[mask].values
        valid = ~(np.isnan(f) | np.isnan(l))
        if valid.sum() < 30:
            continue
        if method == "spearman":
            corr, _ = stats.spearmanr(f[valid], l[valid])
        else:
            corr, _ = stats.pearsonr(f[valid], l[valid])
        ics[dt] = corr

    return pd.Series(ics)


def test_new_factors(market: str = "csi1000",
                     start: str = "2020-01-01",
                     end: str = "2024-12-31",
                     backfill: bool = False,
                     neutralize: bool = True) -> pd.DataFrame:
    """Test all new candidate factors and return results."""
    from project_qlib.runtime import init_qlib
    init_qlib()
    from qlib.data import D

    instruments = D.instruments(market)

    industry_map = None
    if neutralize:
        industry_map = _load_industry()
        if industry_map is not None:
            print(f"Industry neutralization: ON ({len(industry_map)} stocks, "
                  f"{industry_map.nunique()} categories)")

    label_expr = "Ref($close, -2)/Ref($close, -1) - 1"
    label_df = D.features(instruments, [label_expr],
                          start_time=start, end_time=end)
    label_df.columns = ["label"]
    print(f"Label loaded: {label_df.shape}, NaN={label_df['label'].isna().mean():.2%}")
    print(f"\nTesting {len(NEW_FACTORS)} new candidate factors...")
    print(f"{'='*90}")

    results = []
    for i, (expr, name, category) in enumerate(NEW_FACTORS):
        try:
            factor_df = D.features(instruments, [expr],
                                   start_time=start, end_time=end)
            factor_df.columns = [name]

            nan_rate = factor_df[name].isna().mean()
            if nan_rate > 0.8:
                print(f"  [{i+1:2d}/{len(NEW_FACTORS)}] {name:<28} SKIP (NaN={nan_rate:.0%})")
                results.append({
                    "factor": name, "category": category,
                    "expression": expr,
                    "rank_ic": np.nan, "rank_icir": np.nan,
                    "ic": np.nan, "icir": np.nan,
                    "t_stat": np.nan, "p_value": np.nan,
                    "n_days": 0, "nan_rate": nan_rate,
                    "status": "Skip",
                })
                continue

            ic_series = compute_ic(factor_df, label_df, method="spearman",
                                   industry_map=industry_map)
            n_days = len(ic_series)

            if n_days < 50:
                print(f"  [{i+1:2d}/{len(NEW_FACTORS)}] {name:<28} SKIP (n_days={n_days})")
                continue

            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            icir = mean_ic / std_ic if std_ic > 0 else 0
            t_stat = mean_ic / (std_ic / np.sqrt(n_days)) if std_ic > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_days - 1))

            # Skip Pearson IC for speed — RankIC is sufficient for screening
            mean_ic_pearson = np.nan
            icir_pearson = np.nan

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            status = "Accepted" if p_value < 0.01 else ("Candidate" if p_value < 0.05 else "Rejected")

            print(f"  [{i+1:2d}/{len(NEW_FACTORS)}] {name:<28} "
                  f"RankIC={mean_ic:+.4f} ICIR={icir:+.3f} "
                  f"t={t_stat:+.1f} p={p_value:.4f} {sig:3s} [{category}]")

            results.append({
                "factor": name, "category": category,
                "expression": expr,
                "rank_ic": mean_ic, "rank_icir": icir,
                "ic": mean_ic_pearson, "icir": icir_pearson,
                "t_stat": t_stat, "p_value": p_value,
                "n_days": n_days, "nan_rate": nan_rate,
                "status": status,
            })

        except Exception as e:
            print(f"  [{i+1:2d}/{len(NEW_FACTORS)}] {name:<28} ERROR: {e}")
            results.append({
                "factor": name, "category": category,
                "expression": expr,
                "rank_ic": np.nan, "rank_icir": np.nan,
                "ic": np.nan, "icir": np.nan,
                "t_stat": np.nan, "p_value": np.nan,
                "n_days": 0, "nan_rate": 1.0,
                "status": "Error",
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("rank_icir", key=abs, ascending=False)

    # Summary
    print(f"\n{'='*90}")
    print(f"NEW FACTOR BATCH RESULTS ({market}, {start}~{end})")
    print(f"{'='*90}")
    n_total = len(df)
    n_sig = (df["p_value"] < 0.01).sum()
    n_cand = ((df["p_value"] >= 0.01) & (df["p_value"] < 0.05)).sum()
    n_rej = (df["p_value"] >= 0.05).sum()
    print(f"Total: {n_total} | Accepted (p<0.01): {n_sig} | Candidate (p<0.05): {n_cand} | Rejected: {n_rej}")

    print(f"\n--- Top 20 by |RankICIR| ---")
    for rank, (_, row) in enumerate(df.head(20).iterrows(), 1):
        if pd.isna(row["rank_ic"]):
            continue
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*"
        print(f"  {rank:2d}. {row['factor']:<28} ICIR={row['rank_icir']:+.3f} "
              f"RankIC={row['rank_ic']:+.4f} t={row['t_stat']:+.1f} "
              f"[{row['category']:<16}] {sig}")

    print(f"\n--- By Category ---")
    cat_stats = df.groupby("category").agg(
        count=("factor", "count"),
        n_sig=("p_value", lambda x: (x < 0.01).sum()),
        best_icir=("rank_icir", lambda x: x.iloc[x.abs().argmax()] if len(x) > 0 else np.nan),
    )
    for cat, row in cat_stats.iterrows():
        print(f"  {cat:<20} total={row['count']:2.0f}  sig={row['n_sig']:2.0f}  "
              f"best_ICIR={row['best_icir']:+.3f}")

    # Save results
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "new_factor_batch_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    if backfill and not df.empty:
        _backfill_to_db(df, market, start, end)

    return df


def _backfill_to_db(df: pd.DataFrame, market: str,
                    start: str, end: str) -> None:
    """Write results to the factor library."""
    from project_qlib.factor_db import FactorDB

    db = FactorDB()
    n_upsert = 0
    for _, row in df.iterrows():
        if pd.isna(row["rank_ic"]):
            continue
        db.upsert_factor(
            name=row["factor"],
            expression=row["expression"],
            category=row["category"],
            status=row["status"],
            notes=f"New batch factor ({row['category']})",
        )
        db.upsert_test_result(
            factor_name=row["factor"],
            market=market,
            test_start=start,
            test_end=end,
            rank_ic_mean=row["rank_ic"],
            rank_icir=row["rank_icir"],
            ic_mean=row["ic"],
            rank_ic_t=row["t_stat"],
            rank_ic_p=row["p_value"],
            n_days=int(row["n_days"]),
            significant=row["p_value"] < 0.01,
        )
        n_upsert += 1
    print(f"\nBackfilled {n_upsert} factors to factor library.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test new candidate factors")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--backfill", action="store_true",
                        help="Write results to factor library")
    parser.add_argument("--no-neutralize", action="store_true",
                        help="Disable industry neutralization")
    args = parser.parse_args()

    test_new_factors(
        market=args.market,
        start=args.start,
        end=args.end,
        backfill=args.backfill,
        neutralize=not args.no_neutralize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

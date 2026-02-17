"""Test financial factors on CSI1000.

Uses all injected financial data (daily valuation + quarterly fundamentals).
Evaluates single-factor IC/RankIC significance.
Industry neutralization is ON by default (use --no-neutralize to disable).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------
# Financial factor definitions: (Qlib expression, factor name, category)
# -----------------------------------------------------------------------
FINANCIAL_FACTORS = [
    # --- Valuation Level ---
    # EP (Earnings-to-Price, inverse of PE)
    ("Div($close, $pe_ttm)", "FIN_EP", "valuation"),
    # BP (Book-to-Price, inverse of PB)
    ("Div(1, $pb_mrq)", "FIN_BP", "valuation"),
    # SP (Sales-to-Price, inverse of PS)
    ("Div($close, $ps_ttm)", "FIN_SP", "valuation"),
    # CFP (CashFlow-to-Price, inverse of PCF)
    ("Div($close, $pcf_ttm)", "FIN_CFP", "valuation"),

    # --- Valuation Change (momentum on fundamentals) ---
    # PE change over 20 days: recent becoming cheaper or more expensive
    ("Div($pe_ttm, Ref($pe_ttm, 20)) - 1", "FIN_PE_CHG_20", "valuation_momentum"),
    # PB change over 20 days
    ("Div($pb_mrq, Ref($pb_mrq, 20)) - 1", "FIN_PB_CHG_20", "valuation_momentum"),
    # PE change over 60 days (quarterly)
    ("Div($pe_ttm, Ref($pe_ttm, 60)) - 1", "FIN_PE_CHG_60", "valuation_momentum"),
    # PB change over 60 days (quarterly)
    ("Div($pb_mrq, Ref($pb_mrq, 60)) - 1", "FIN_PB_CHG_60", "valuation_momentum"),

    # --- Valuation Deviation from MA ---
    # PE relative to 20d MA (how far from recent norm)
    ("Div($pe_ttm, Mean($pe_ttm, 20)) - 1", "FIN_PE_DEV_MA20", "valuation_deviation"),
    # PB relative to 60d MA
    ("Div($pb_mrq, Mean($pb_mrq, 60)) - 1", "FIN_PB_DEV_MA60", "valuation_deviation"),
    # PE relative to 120d MA
    ("Div($pe_ttm, Mean($pe_ttm, 120)) - 1", "FIN_PE_DEV_MA120", "valuation_deviation"),
    # PE relative to 240d MA (annual)
    ("Div($pe_ttm, Mean($pe_ttm, 240)) - 1", "FIN_PE_DEV_MA240", "valuation_deviation"),

    # --- Valuation Volatility ---
    # PE volatility (unstable valuation = uncertain earnings)
    ("Div(Std($pe_ttm, 20), Mean($pe_ttm, 20))", "FIN_PE_VOL_20", "valuation_vol"),
    # PB volatility
    ("Div(Std($pb_mrq, 20), Mean($pb_mrq, 20))", "FIN_PB_VOL_20", "valuation_vol"),

    # --- Turnover ---
    # Turnover rate level
    ("$turnover_rate", "FIN_TURN", "turnover"),
    # Turnover MA5/MA20 ratio (short-term turnover surge)
    ("Div(Mean($turnover_rate, 5), Mean($turnover_rate, 20))", "FIN_TURN_SURGE_5_20", "turnover"),
    # Turnover MA5/MA60 ratio
    ("Div(Mean($turnover_rate, 5), Mean($turnover_rate, 60))", "FIN_TURN_SURGE_5_60", "turnover"),
    # Turnover CV (coefficient of variation, 20d)
    ("Div(Std($turnover_rate, 20), Mean($turnover_rate, 20))", "FIN_TURN_CV_20", "turnover"),
    # Turnover change over 20 days
    ("Div(Mean($turnover_rate, 5), Ref(Mean($turnover_rate, 5), 20)) - 1",
     "FIN_TURN_CHG_20", "turnover"),

    # --- Valuation-Price Interaction ---
    # Price momentum with PE adjustment: if price rose but PE stable, real improvement
    ("Div(Div($close, Ref($close, 20)) - 1, Div($pe_ttm, Ref($pe_ttm, 20)))",
     "FIN_PRICE_PE_RATIO_20", "interaction"),
    # Return / Turnover ratio (price efficiency per unit of turnover)
    ("Div(Div($close, Ref($close, 20)) - 1, Mean($turnover_rate, 20))",
     "FIN_RET_TURN_20", "interaction"),

    # --- Composite Valuation Score ---
    # EP rank + BP rank composite (lower rank = cheaper)
    # Using a simple proxy: average of normalized EP and BP
    ("Add(Div($close, $pe_ttm), Div(1, $pb_mrq))", "FIN_EP_BP_SUM", "composite"),
    # EP * Turnover (cheap + active)
    ("Mul(Div($close, $pe_ttm), $turnover_rate)", "FIN_EP_TURN", "composite"),

    # --- ST Risk ---
    # Whether the stock is ST-flagged (risk marker)
    # Not a trading factor per se, but useful for filtering
    # ("$is_st", "FIN_IS_ST", "risk"),  # skip: binary, not useful for IC
]


def _load_industry() -> pd.Series | None:
    """Load industry classification from data/industry.parquet."""
    ind_file = PROJECT_ROOT / "data" / "industry.parquet"
    if not ind_file.exists():
        print("  [WARN] Industry file not found, skipping neutralization.")
        return None
    ind_df = pd.read_parquet(ind_file)
    # qlib_code -> industry mapping
    return ind_df.set_index("qlib_code")["industry"]


def _neutralize_factor(factor_series: pd.Series,
                       industry_map: pd.Series) -> pd.Series:
    """Industry-neutralize factor values (cross-section Z-score within industry).

    For each date: factor_neutral = (x - industry_mean) / industry_std
    """
    result = factor_series.copy()
    dates = factor_series.index.get_level_values("datetime")
    instruments = factor_series.index.get_level_values("instrument")

    # Build a temporary DF for groupby
    tmp = pd.DataFrame({
        "factor": factor_series.values,
        "datetime": dates,
        "instrument": instruments,
    })
    # Map industry
    tmp["industry"] = tmp["instrument"].map(industry_map)
    # Drop stocks without industry mapping
    tmp.loc[tmp["industry"].isna(), "factor"] = np.nan

    # Z-score within (datetime, industry)
    grouped = tmp.groupby(["datetime", "industry"])["factor"]
    tmp["neutral"] = grouped.transform(lambda x: (x - x.mean()) / x.std()
                                       if x.std() > 0 else 0)
    result[:] = tmp["neutral"].values
    return result


def compute_ic(factor_df: pd.DataFrame, label_df: pd.DataFrame,
               method: str = "spearman",
               industry_map: pd.Series | None = None) -> pd.Series:
    """Compute daily IC between factor and label.

    If industry_map is provided, factor is industry-neutralized first.
    """
    # Align indices
    common = factor_df.index.intersection(label_df.index)
    factor_aligned = factor_df.loc[common].iloc[:, 0]
    label_aligned = label_df.loc[common].iloc[:, 0]

    # Industry neutralization
    if industry_map is not None:
        factor_aligned = _neutralize_factor(factor_aligned, industry_map)

    # Group by date
    dates = common.get_level_values("datetime")
    unique_dates = dates.unique()

    ics = {}
    for dt in unique_dates:
        mask = dates == dt
        f = factor_aligned[mask].values
        l = label_aligned[mask].values

        # Remove NaN pairs
        valid = ~(np.isnan(f) | np.isnan(l))
        if valid.sum() < 30:
            continue

        if method == "spearman":
            corr, _ = stats.spearmanr(f[valid], l[valid])
        else:
            corr, _ = stats.pearsonr(f[valid], l[valid])
        ics[dt] = corr

    return pd.Series(ics)


def test_financial_factors(market: str = "csi1000",
                           start: str = "2020-01-01",
                           end: str = "2024-12-31",
                           backfill: bool = False,
                           neutralize: bool = True) -> pd.DataFrame:
    """Test all financial factors and return results sorted by |ICIR|."""
    from project_qlib.runtime import init_qlib
    init_qlib()
    from qlib.data import D

    instruments = D.instruments(market)

    # Load industry map for neutralization
    industry_map = None
    if neutralize:
        industry_map = _load_industry()
        if industry_map is not None:
            print(f"Industry neutralization: ON ({len(industry_map)} stocks, "
                  f"{industry_map.nunique()} categories)")
        else:
            print("Industry neutralization: OFF (no data)")
    else:
        print("Industry neutralization: OFF (disabled)")

    # Load label
    label_expr = "Ref($close, -2)/Ref($close, -1) - 1"
    label_df = D.features(instruments, [label_expr],
                          start_time=start, end_time=end)
    label_df.columns = ["label"]
    print(f"Label loaded: {label_df.shape}, NaN={label_df['label'].isna().mean():.2%}")

    results = []
    for i, (expr, name, category) in enumerate(FINANCIAL_FACTORS):
        try:
            factor_df = D.features(instruments, [expr],
                                   start_time=start, end_time=end)
            factor_df.columns = [name]

            # NaN coverage check
            nan_rate = factor_df[name].isna().mean()
            if nan_rate > 0.8:
                print(f"  [{i+1}/{len(FINANCIAL_FACTORS)}] {name}: "
                      f"skip (NaN={nan_rate:.0%})")
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

            # Compute RankIC (with industry neutralization if enabled)
            ic_series = compute_ic(factor_df, label_df, method="spearman",
                                   industry_map=industry_map)
            n_days = len(ic_series)

            if n_days < 50:
                print(f"  [{i+1}/{len(FINANCIAL_FACTORS)}] {name}: "
                      f"skip (n_days={n_days})")
                continue

            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            icir = mean_ic / std_ic if std_ic > 0 else 0
            t_stat = mean_ic / (std_ic / np.sqrt(n_days)) if std_ic > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_days - 1))

            # Also compute regular IC
            ic_series_pearson = compute_ic(factor_df, label_df, method="pearson",
                                           industry_map=industry_map)
            mean_ic_pearson = ic_series_pearson.mean()
            std_ic_pearson = ic_series_pearson.std()
            icir_pearson = mean_ic_pearson / std_ic_pearson if std_ic_pearson > 0 else 0

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  [{i+1}/{len(FINANCIAL_FACTORS)}] {name:<30} "
                  f"RankIC={mean_ic:+.4f} ICIR={icir:+.3f} "
                  f"t={t_stat:+.1f} p={p_value:.4f} {sig} "
                  f"(NaN={nan_rate:.1%}, days={n_days})")

            results.append({
                "factor": name, "category": category,
                "expression": expr,
                "rank_ic": mean_ic, "rank_icir": icir,
                "ic": mean_ic_pearson, "icir": icir_pearson,
                "t_stat": t_stat, "p_value": p_value,
                "n_days": n_days, "nan_rate": nan_rate,
                "status": "Accepted" if p_value < 0.01 else "Candidate",
            })

        except Exception as e:
            print(f"  [{i+1}/{len(FINANCIAL_FACTORS)}] {name}: ERROR {e}")
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
    print(f"\n{'='*80}")
    print(f"Financial Factor IC Test Summary ({market}, {start}~{end})")
    print(f"{'='*80}")
    print(f"Total: {len(df)}, Significant (p<0.01): {(df['p_value'] < 0.01).sum()}")
    print(f"\nTop 10 by |RankICIR|:")
    for _, row in df.head(10).iterrows():
        sig = "***" if row["p_value"] < 0.001 else ""
        print(f"  {row['factor']:<30} RankIC={row['rank_ic']:+.4f} "
              f"ICIR={row['rank_icir']:+.3f} "
              f"t={row['t_stat']:+.1f} [{row['category']}] {sig}")

    # Backfill to factor library
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

        # Upsert factor definition
        db.upsert_factor(
            name=row["factor"],
            expression=row["expression"],
            category=row["category"],
            status=row["status"],
            notes=f"Financial valuation factor ({row['category']})",
        )

        # Upsert test result
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
    parser = argparse.ArgumentParser(description="Test financial factors on CSI1000")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--backfill", action="store_true",
                        help="Write results to factor library")
    parser.add_argument("--no-neutralize", action="store_true",
                        help="Disable industry neutralization (default: ON)")
    args = parser.parse_args()

    test_financial_factors(
        market=args.market,
        start=args.start,
        end=args.end,
        backfill=args.backfill,
        neutralize=not args.no_neutralize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

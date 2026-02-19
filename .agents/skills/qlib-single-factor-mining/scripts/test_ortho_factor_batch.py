"""Orthogonal factor batch test.

Design principles:
1. Each factor targets a dimension INDEPENDENT of existing top-40 clusters
2. Avoid: volume CV, range/KLEN, price-vol correlation, extreme return, turnover quantile
3. Target: return asymmetry, overnight patterns, tail risk, path complexity,
   volume-weighted return, conditional reversal, valuation speed, liquidity timing

Orthogonality targets (existing clusters to avoid high corr with):
- Cluster A: AMT_CV / TURN_CV / TURN_PEAK (turnover volatility, ρ>0.88)
- Cluster B: KLEN / RANGE_1D (intraday range, ρ=0.999)
- Cluster C: CORD / CORR / PV_CORR (price-vol correlation, ρ>0.99)
- Cluster D: MAX_RET / RET_SPREAD / AMT_WTABS (extreme returns, ρ>0.82)
- Cluster E: VSUMP/VSUMD/VSUMN (volume sum decomposition, ρ=1.0)
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

ORTHO_FACTORS = [
    # ============================================================
    # A. Return Asymmetry / Skewness (unique axis: distribution shape)
    # ============================================================
    # Upside vs downside volatility ratio
    ("Div(Std(If($close > Ref($close, 1), Div($close, Ref($close, 1)) - 1, 0), 20), "
     "Std(If($close < Ref($close, 1), Div($close, Ref($close, 1)) - 1, 0), 20) + 1e-8)",
     "ORTH_UPDOWN_VOL_RATIO_20", "return_asym"),
    # Gain days ratio over 20d (win rate)
    ("Mean(If($close > Ref($close, 1), 1, 0), 20)",
     "ORTH_WIN_RATE_20", "return_asym"),
    # Gain days ratio over 60d
    ("Mean(If($close > Ref($close, 1), 1, 0), 60)",
     "ORTH_WIN_RATE_60", "return_asym"),
    # Average gain / average loss ratio (profit factor)
    ("Div(Mean(If($close > Ref($close, 1), Div($close, Ref($close, 1)) - 1, 0), 20), "
     "Mean(If($close < Ref($close, 1), 1 - Div($close, Ref($close, 1)), 0), 20) + 1e-8)",
     "ORTH_PROFIT_FACTOR_20", "return_asym"),

    # ============================================================
    # B. Overnight vs Intraday Decomposition (unique axis: time-of-day)
    # ============================================================
    # Overnight return: open/prev_close - 1
    ("Div($open, Ref($close, 1)) - 1",
     "ORTH_OVERNIGHT_RET", "overnight"),
    # Overnight return MA10
    ("Mean(Div($open, Ref($close, 1)) - 1, 10)",
     "ORTH_OVERNIGHT_MA10", "overnight"),
    # Intraday return: close/open - 1
    ("Div($close, $open) - 1",
     "ORTH_INTRADAY_RET", "intraday"),
    # Intraday return MA10
    ("Mean(Div($close, $open) - 1, 10)",
     "ORTH_INTRADAY_MA10", "intraday"),
    # Overnight-Intraday divergence: persistent overnight gap vs intraday reversal
    ("Mean(Div($open, Ref($close, 1)) - 1, 10) - Mean(Div($close, $open) - 1, 10)",
     "ORTH_OVN_INTRA_DIV", "overnight"),

    # ============================================================
    # C. Tail Risk / Drawdown Based (unique axis: loss distribution)
    # ============================================================
    # Max drawdown proxy over 20d: (close - max_close_20) / max_close_20
    ("Div($close - Max($close, 20), Max($close, 20) + 1e-8)",
     "ORTH_DRAWDOWN_20", "tail_risk"),
    # Max drawdown proxy over 60d
    ("Div($close - Max($close, 60), Max($close, 60) + 1e-8)",
     "ORTH_DRAWDOWN_60", "tail_risk"),
    # Distance from 20d low (recovery from bottom)
    ("Div($close - Min($close, 20), Min($close, 20) + 1e-8)",
     "ORTH_RECOVERY_20", "tail_risk"),
    # Downside deviation (semi-variance) over 20d
    ("Std(If($close < Ref($close, 1), Div($close, Ref($close, 1)) - 1, 0), 20)",
     "ORTH_DOWNSIDE_DEV_20", "tail_risk"),

    # ============================================================
    # D. Price Path Complexity (unique axis: trajectory shape)
    # ============================================================
    # Path efficiency: |net move| / sum of |daily moves| over 10d
    ("Div(Abs(Div($close, Ref($close, 10)) - 1), "
     "Sum(Abs(Div($close, Ref($close, 1)) - 1), 10) + 1e-8)",
     "ORTH_PATH_EFF_10", "path"),
    # Path efficiency 20d
    ("Div(Abs(Div($close, Ref($close, 20)) - 1), "
     "Sum(Abs(Div($close, Ref($close, 1)) - 1), 20) + 1e-8)",
     "ORTH_PATH_EFF_20", "path"),
    # Consecutive up/down days proxy: sign persistence
    ("Mean(If(Mul(Div($close, Ref($close, 1)) - 1, Div(Ref($close, 1), Ref($close, 2)) - 1) > 0, 1, 0), 20)",
     "ORTH_SIGN_PERSIST_20", "path"),
    # Gap frequency: abs(open - prev_close) / prev_close, averaged
    ("Mean(Abs(Div($open, Ref($close, 1)) - 1), 20)",
     "ORTH_GAP_FREQ_20", "path"),

    # ============================================================
    # E. Volume-Weighted Return Features (unique axis: where returns occur in volume)
    # ============================================================
    # VWAP-to-close bias, cumulative: how persistently VWAP differs from close
    ("Std(Div($vwap, $close) - 1, 20)",
     "ORTH_VWAP_BIAS_VOL_20", "vw_return"),
    # Volume-weighted return skew proxy: high vs low price relative to VWAP
    ("Mean(Div($high - $vwap, $vwap - $low + 1e-8), 10)",
     "ORTH_VW_SKEW_10", "vw_return"),
    # Amount-weighted absolute return change: recent vs past intensity
    ("Div(Mean(Mul(Abs(Div($close, Ref($close, 1)) - 1), $amount), 5), "
     "Mean(Mul(Abs(Div($close, Ref($close, 1)) - 1), $amount), 20) + 1e-8)",
     "ORTH_AMT_WT_RET_SURGE", "vw_return"),

    # ============================================================
    # F. Conditional Reversal (unique axis: reversal qualified by other signals)
    # ============================================================
    # Mean reversion × low volume: reversal signal stronger when volume is low
    ("Mul(Div($close - Mean($close, 20), Mean($close, 20) + 1e-8), "
     "0 - Div($volume, Mean($volume, 20) + 1e-8))",
     "ORTH_REVERT_LOWVOL_20", "cond_reversal"),
    # Drawdown × turnover surge: recovery when volume arrives after drawdown
    ("Mul(Div($close - Max($close, 20), Max($close, 20) + 1e-8), "
     "Div($turnover_rate, Mean($turnover_rate, 20) + 1e-8))",
     "ORTH_DD_TURN_SURGE", "cond_reversal"),
    # Overnight gap × intraday reversal: gap that gets reversed intraday
    ("Mul(Div($open, Ref($close, 1)) - 1, 0 - (Div($close, $open) - 1))",
     "ORTH_GAP_REVERSAL", "cond_reversal"),

    # ============================================================
    # G. Valuation Change Speed (unique axis: fundamental momentum velocity)
    # ============================================================
    # PE acceleration: PE_chg_5d - PE_chg_5d_lagged_5
    ("Div($pe_ttm, Ref($pe_ttm, 5)) - Div(Ref($pe_ttm, 5), Ref($pe_ttm, 10))",
     "ORTH_PE_ACCEL_5", "val_speed"),
    # PB mean reversion speed: how fast PB returns to MA
    ("(Div($pb_mrq, Mean($pb_mrq, 60)) - 1) - Ref(Div($pb_mrq, Mean($pb_mrq, 60)) - 1, 10)",
     "ORTH_PB_REVERT_SPEED", "val_speed"),
    # EP (1/PE) change vs return change divergence
    ("(Div(1, $pe_ttm + 1e-8) - Div(1, Ref($pe_ttm, 20) + 1e-8)) - (Div($close, Ref($close, 20)) - 1)",
     "ORTH_EP_RET_DIV_20", "val_speed"),

    # ============================================================
    # H. Liquidity Timing (unique axis: when liquidity appears/disappears)
    # ============================================================
    # Volume trend: MA5/MA60 ratio (short vs long volume)
    ("Div(Mean($volume, 5), Mean($volume, 60) + 1e-8)",
     "ORTH_VOL_TREND_5_60", "liq_timing"),
    # Amount acceleration: surge acceleration (d/dt of surge ratio)
    ("Div(Mean($amount, 5), Mean($amount, 20) + 1e-8) - "
     "Div(Ref(Mean($amount, 5), 5), Ref(Mean($amount, 20), 5) + 1e-8)",
     "ORTH_AMT_ACCEL", "liq_timing"),
    # High-volume day return: avg return on days when volume > MA20
    ("Mean(If($volume > Mean($volume, 20), Div($close, Ref($close, 1)) - 1, 0), 20)",
     "ORTH_HIGHVOL_RET_20", "liq_timing"),

    # ============================================================
    # I. Relative Strength (unique axis: momentum relative to benchmark proxy)
    # ============================================================
    # Return rank stability: consistency of ranking (proxy via MA vs current)
    ("Corr(Div($close, Ref($close, 1)) - 1, Ref(Div($close, Ref($close, 1)) - 1, 1), 20)",
     "ORTH_RET_AUTOCORR_20", "rel_strength"),
    # Return persistence: sign of 5d ret same as 20d ret
    ("If(Mul(Div($close, Ref($close, 5)) - 1, Div($close, Ref($close, 20)) - 1) > 0, "
     "Abs(Div($close, Ref($close, 5)) - 1), "
     "0 - Abs(Div($close, Ref($close, 5)) - 1))",
     "ORTH_MOM_CONSISTENCY", "rel_strength"),

    # ============================================================
    # J. Cross-Valuation Divergence (unique axis: when valuations disagree)
    # ============================================================
    # PE vs PB divergence: when PE says cheap but PB says expensive (or vice versa)
    ("Div($pe_ttm - Mean($pe_ttm, 60), Std($pe_ttm, 60) + 1e-8) - "
     "Div($pb_mrq - Mean($pb_mrq, 60), Std($pb_mrq, 60) + 1e-8)",
     "ORTH_PE_PB_ZSCORE_DIV", "val_divergence"),
    # PS vs PE divergence
    ("Div($ps_ttm - Mean($ps_ttm, 60), Std($ps_ttm, 60) + 1e-8) - "
     "Div($pe_ttm - Mean($pe_ttm, 60), Std($pe_ttm, 60) + 1e-8)",
     "ORTH_PS_PE_ZSCORE_DIV", "val_divergence"),
]


def _load_industry():
    ind_file = PROJECT_ROOT / "data" / "industry.parquet"
    if not ind_file.exists():
        return None
    ind_df = pd.read_parquet(ind_file)
    return ind_df.set_index("qlib_code")["industry"]


def _neutralize_factor(factor_series, industry_map):
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


def compute_ic(factor_df, label_df, method="spearman", industry_map=None):
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
        corr, _ = stats.spearmanr(f[valid], l[valid])
        ics[dt] = corr
    return pd.Series(ics)


def _apply_fdr(df: pd.DataFrame, p_col: str = "p_value", out_col: str = "fdr_p") -> pd.DataFrame:
    """Benjamini-Hochberg FDR correction."""
    result = df.copy()
    result[out_col] = np.nan
    valid = result[p_col].notna()
    if valid.sum() == 0:
        result["significant"] = False
        return result

    p_vals = result.loc[valid, p_col].astype(float).values
    n_tests = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]
    adj = np.zeros(n_tests)
    for i in range(n_tests - 1, -1, -1):
        rank = i + 1
        if i == n_tests - 1:
            adj[sorted_idx[i]] = sorted_p[i]
        else:
            adj[sorted_idx[i]] = min(sorted_p[i] * n_tests / rank, adj[sorted_idx[i + 1]])
    result.loc[valid, out_col] = np.clip(adj, 0.0, 1.0)
    result["significant"] = result[out_col] < 0.01
    return result


def test_ortho_factors(market="csi1000", start="2020-01-01", end="2025-12-31",
                       backfill=False, neutralize=True, round_id="SFA-BATCH-ORTHO",
                       source_tag="Custom"):
    import warnings
    warnings.filterwarnings("ignore")

    from project_qlib.runtime import init_qlib
    init_qlib()
    from qlib.data import D

    instruments = D.instruments(market)

    industry_map = None
    if neutralize:
        industry_map = _load_industry()
        if industry_map is not None:
            print(f"Industry neutralization: ON ({len(industry_map)} stocks)")

    label_expr = "Ref($close, -2)/Ref($close, -1) - 1"
    label_df = D.features(instruments, [label_expr], start_time=start, end_time=end)
    label_df.columns = ["label"]
    print(f"Label loaded: {label_df.shape}")
    print(f"\nTesting {len(ORTHO_FACTORS)} orthogonal factors...")
    print("=" * 90)

    results = []
    for i, (expr, name, category) in enumerate(ORTHO_FACTORS):
        try:
            factor_df = D.features(instruments, [expr], start_time=start, end_time=end)
            factor_df.columns = [name]
            nan_rate = factor_df[name].isna().mean()
            if nan_rate > 0.8:
                print(f"  [{i+1:2d}/{len(ORTHO_FACTORS)}] {name:<30} SKIP (NaN={nan_rate:.0%})")
                results.append({"factor": name, "category": category, "expression": expr,
                    "rank_ic": np.nan, "rank_icir": np.nan, "ic": np.nan, "icir": np.nan,
                    "t_stat": np.nan, "p_value": np.nan, "n_days": 0, "nan_rate": nan_rate,
                    "status": "Skip"})
                continue

            ic_series = compute_ic(factor_df, label_df, industry_map=industry_map)
            n_days = len(ic_series)
            if n_days < 50:
                print(f"  [{i+1:2d}/{len(ORTHO_FACTORS)}] {name:<30} SKIP (n_days={n_days})")
                continue

            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            icir = mean_ic / std_ic if std_ic > 0 else 0
            t_stat = mean_ic / (std_ic / np.sqrt(n_days)) if std_ic > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_days - 1))

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

            print(f"  [{i+1:2d}/{len(ORTHO_FACTORS)}] {name:<30} "
                  f"RankIC={mean_ic:+.4f} ICIR={icir:+.3f} "
                  f"t={t_stat:+.1f} p={p_value:.4f} {sig:3s} [{category}]")

            results.append({"factor": name, "category": category, "expression": expr,
                "rank_ic": mean_ic, "rank_icir": icir, "ic": np.nan, "icir": np.nan,
                "t_stat": t_stat, "p_value": p_value, "n_days": n_days, "nan_rate": nan_rate,
                "status": "Pending"})

        except Exception as e:
            print(f"  [{i+1:2d}/{len(ORTHO_FACTORS)}] {name:<30} ERROR: {e}")
            results.append({"factor": name, "category": category, "expression": expr,
                "rank_ic": np.nan, "rank_icir": np.nan, "ic": np.nan, "icir": np.nan,
                "t_stat": np.nan, "p_value": np.nan, "n_days": 0, "nan_rate": 1.0,
                "status": "Error"})

    df = pd.DataFrame(results)
    if not df.empty:
        df = _apply_fdr(df)
        df["status"] = np.where(
            df["significant"] == True,
            "Accepted",
            np.where(df["p_value"] < 0.05, "Candidate", "Rejected"),
        )
        df = df.sort_values("rank_icir", key=abs, ascending=False)

    print(f"\n{'='*90}")
    print(f"ORTHOGONAL FACTOR BATCH RESULTS ({market}, {start}~{end})")
    print(f"{'='*90}")
    n_total = len(df)
    n_sig = int(df["significant"].sum()) if "significant" in df.columns else 0
    n_cand = int(((df["p_value"] < 0.05) & (~df["significant"])).sum()) if "significant" in df.columns else 0
    n_rej = (df["p_value"] >= 0.05).sum()
    print(f"Total: {n_total} | Accepted (FDR<0.01): {n_sig} | Candidate (p<0.05): {n_cand} | Rejected: {n_rej}")

    print(f"\n--- Top 20 by |RankICIR| ---")
    for rank, (_, row) in enumerate(df.head(20).iterrows(), 1):
        if pd.isna(row["rank_ic"]):
            continue
        fdr_p = row.get("fdr_p", np.nan)
        sig = "***" if pd.notna(fdr_p) and fdr_p < 0.01 else ("*" if row["p_value"] < 0.05 else "")
        print(f"  {rank:2d}. {row['factor']:<30} ICIR={row['rank_icir']:+.3f} "
              f"RankIC={row['rank_ic']:+.4f} t={row['t_stat']:+.1f} fdr={fdr_p if pd.notna(fdr_p) else np.nan:.4f} "
              f"[{row['category']:<16}] {sig}")

    print(f"\n--- By Category ---")
    cat_stats = df.groupby("category").agg(
        count=("factor", "count"),
        n_sig=("significant", "sum"),
        best_icir=("rank_icir", lambda x: x.iloc[x.abs().argmax()] if len(x) > 0 else np.nan),
    )
    for cat, row in cat_stats.iterrows():
        print(f"  {cat:<20} total={row['count']:2.0f}  sig={row['n_sig']:2.0f}  "
              f"best_ICIR={row['best_icir']:+.3f}")

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "ortho_factor_batch_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    if backfill and not df.empty:
        _backfill_to_db(df, market, start, end, round_id=round_id, source_tag=source_tag)

    return df


def _backfill_to_db(df, market, start, end, round_id: str, source_tag: str):
    from project_qlib.factor_db import FactorDB
    db = FactorDB()
    n_upsert = 0
    for _, row in df.iterrows():
        if pd.isna(row["rank_ic"]):
            continue
        db.upsert_factor(name=row["factor"], expression=row["expression"],
            category=row["category"], source=source_tag, status=row["status"],
            notes=f"Orthogonal batch factor ({row['category']})")
        db.upsert_test_result(factor_name=row["factor"], market=market,
            test_start=start, test_end=end,
            rank_ic_mean=row["rank_ic"], rank_icir=row["rank_icir"],
            ic_mean=row.get("ic", np.nan), rank_ic_t=row["t_stat"],
            rank_ic_p=row["p_value"], fdr_p=row.get("fdr_p"), n_days=int(row["n_days"]),
            significant=bool(row.get("significant", False)), hea_round=round_id)
        n_upsert += 1
    print(f"\nBackfilled {n_upsert} factors to factor library.")


def main():
    parser = argparse.ArgumentParser(description="Test orthogonal candidate factors")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--no-neutralize", action="store_true")
    parser.add_argument("--round-id", default="SFA-BATCH-ORTHO",
                        help="Round id written to DB (hea_round compatibility field)")
    parser.add_argument("--source-tag", default="Custom",
                        help="Factor source tag when backfilling")
    args = parser.parse_args()
    test_ortho_factors(market=args.market, start=args.start, end=args.end,
                       backfill=args.backfill, neutralize=not args.no_neutralize,
                       round_id=args.round_id, source_tag=args.source_tag)


if __name__ == "__main__":
    raise SystemExit(main())

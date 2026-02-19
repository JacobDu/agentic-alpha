"""Design and test new candidate factors on CSI1000.

This script tests a batch of newly designed factors, computes Rank IC with
FDR correction, and backfills results into the factor library.

Hypothesis:  Based on analysis of existing 44 custom factors (strongest signals
are volume CV, range volatility, price-volume correlation), we design new
factors in under-explored categories:
  - Liquidity (Amihud illiquidity)
  - Volume distribution (up-day volume ratio)
  - Higher moments (kurtosis)
  - Extreme returns (max/min daily return)
  - Relative position (close vs N-day high/low)
  - Return asymmetry (gain/loss ratio, up-days ratio)
  - Price efficiency (variance ratio)
  - Downside risk
  - VWAP pattern

Market logic:
  - Amihud: illiquid stocks earn premium (documented in literature)
  - Up-vol ratio: stocks with volume concentrated on up days are "overheated"
  - Kurtosis: high tail risk → negative expected return (risk premium)
  - Max return: recent extreme positive returns → reversal
  - Relative position: stocks near highs tend to reverse (CSI1000 mean-reversion)
  - Variance ratio: deviations from random walk are predictive
"""
from __future__ import annotations

import argparse
import gc
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

from project_qlib.runtime import init_qlib

LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"

# ---- New candidate factors ----
NEW_FACTORS = [
    # -- Liquidity (Amihud illiquidity) --
    ("Mean(Abs($close/Ref($close, 1) - 1) / ($amount + 1e-8), 10)",
     "CSTM_AMIHUD_10", "liquidity"),
    ("Mean(Abs($close/Ref($close, 1) - 1) / ($amount + 1e-8), 20)",
     "CSTM_AMIHUD_20", "liquidity"),

    # -- Volume distribution (up-day volume ratio) --
    ("Sum(If(Gt($close, Ref($close, 1)), $volume, 0), 10) / (Sum($volume, 10) + 1e-8)",
     "CSTM_UP_VOL_RATIO_10", "volume_dist"),
    ("Sum(If(Gt($close, Ref($close, 1)), $volume, 0), 20) / (Sum($volume, 20) + 1e-8)",
     "CSTM_UP_VOL_RATIO_20", "volume_dist"),

    # -- Higher moments (kurtosis) --
    ("Mean(Power($close/Ref($close, 1) - 1, 4), 20) / (Power(Std($close/Ref($close, 1) - 1, 20), 4) + 1e-12)",
     "CSTM_KURT_20", "higher_moment"),
    ("Mean(Power($close/Ref($close, 1) - 1, 4), 60) / (Power(Std($close/Ref($close, 1) - 1, 60), 4) + 1e-12)",
     "CSTM_KURT_60", "higher_moment"),

    # -- Extreme returns --
    ("Max($close/Ref($close, 1) - 1, 20)",
     "CSTM_MAX_RET_20", "extreme"),
    ("Min($close/Ref($close, 1) - 1, 20)",
     "CSTM_MIN_RET_20", "extreme"),
    ("Max($close/Ref($close, 1) - 1, 20) - Min($close/Ref($close, 1) - 1, 20)",
     "CSTM_RET_SPREAD_20", "extreme"),

    # -- Relative position --
    ("($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-8)",
     "CSTM_RANGE_POS_20", "relative_pos"),
    ("($close - Min($low, 60)) / (Max($high, 60) - Min($low, 60) + 1e-8)",
     "CSTM_RANGE_POS_60", "relative_pos"),
    ("$close / Max($high, 20) - 1",
     "CSTM_REL_HIGH_20", "relative_pos"),

    # -- Return asymmetry --
    ("Sum(If(Gt($close, Ref($close, 1)), 1, 0), 20) / 20",
     "CSTM_UP_DAYS_RATIO_20", "return_asym"),
    ("Sum(If(Gt($close, Ref($close, 1)), $close/Ref($close, 1) - 1, 0), 20) / (Sum(If(Gt(Ref($close, 1), $close), Ref($close, 1)/$close - 1, 0), 20) + 1e-8)",
     "CSTM_GAIN_LOSS_20", "return_asym"),

    # -- Efficiency / Variance ratio --
    ("Power(Std($close/Ref($close, 5) - 1, 60), 2) / (5 * Power(Std($close/Ref($close, 1) - 1, 60), 2) + 1e-12)",
     "CSTM_VAR_RATIO_5_60", "efficiency"),

    # -- Downside risk --
    ("Mean(If(Gt(Ref($close, 1), $close), Power($close/Ref($close, 1) - 1, 2), 0), 20)",
     "CSTM_DOWNSIDE_VAR_20", "risk"),

    # -- VWAP pattern --
    ("Sum(If(Gt($close, $vwap), 1, 0), 20) / 20",
     "CSTM_ABOVE_VWAP_20", "vwap"),

    # -- Smart money v2: amount-weighted absolute return (activity) --
    ("Mean(Abs($close/Ref($close, 1) - 1) * $amount, 20) / (Mean($amount, 20) + 1e-8)",
     "CSTM_AMT_WTABS_20", "smart_money"),

    # -- Volume-price divergence: price up but volume down --
    ("Corr(Sign($close/Ref($close, 1) - 1), $volume/Ref($volume, 1) - 1, 20)",
     "CSTM_SIGN_VOL_CORR_20", "volume_dist"),
]


def fast_daily_rankic(factor: pd.Series, label: pd.Series, min_stocks: int = 30) -> dict:
    """Compute daily cross-sectional Rank IC using groupby."""
    combined = pd.DataFrame({"factor": factor, "label": label}).dropna()
    if len(combined) == 0:
        return {"n_days": 0, "ic_mean": np.nan, "rank_ic_mean": np.nan,
                "rank_ic_t": np.nan, "rank_ic_p": 1.0, "rank_icir": np.nan}

    dates = combined.index.get_level_values(1)

    def _spearman(g):
        return np.nan if len(g) < min_stocks else g["factor"].rank().corr(g["label"].rank())

    def _pearson(g):
        return np.nan if len(g) < min_stocks else g["factor"].corr(g["label"])

    daily_ric = combined.groupby(dates).apply(_spearman).dropna()
    daily_ic = combined.groupby(dates).apply(_pearson).dropna()

    n = len(daily_ric)
    if n < 30:
        return {"n_days": n, "ic_mean": np.nan, "rank_ic_mean": np.nan,
                "rank_ic_t": np.nan, "rank_ic_p": 1.0, "rank_icir": np.nan}

    ric_mean = daily_ric.mean()
    ric_std = daily_ric.std()
    ric_t = ric_mean / (ric_std / np.sqrt(n)) if ric_std > 0 else 0
    ric_p = 2 * (1 - stats.t.cdf(abs(ric_t), df=n - 1))

    return {
        "n_days": n,
        "ic_mean": daily_ic.mean(),
        "ic_std": daily_ic.std(),
        "rank_ic_mean": ric_mean,
        "rank_ic_std": ric_std,
        "rank_ic_t": ric_t,
        "rank_ic_p": ric_p,
        "rank_icir": ric_mean / ric_std if ric_std > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Test new candidate factors")
    parser.add_argument("--market", default="csi1000", help="Market (default: csi1000)")
    parser.add_argument("--start", default="2019-01-01", help="Test start date")
    parser.add_argument("--end", default="2025-12-31", help="Test end date")
    parser.add_argument("--backfill", action="store_true", help="Write results to factor library")
    parser.add_argument("--round-id", default="SFA-BATCH-NEW", help="Round id written to DB (hea_round compatibility field)")
    parser.add_argument("--source-tag", default="Custom", help="Factor source tag when backfilling")
    args = parser.parse_args()

    init_qlib()

    from qlib.data import D

    # Get instruments
    instruments = D.instruments(args.market)
    stock_list = D.list_instruments(instruments, start_time=args.start, end_time=args.end, as_list=True)
    print(f"Market: {args.market}, stocks: {len(stock_list)}")

    min_stocks = 30 if args.market in ("csi1000", "csi300") else 50

    # Build all expressions
    all_exprs = [LABEL_EXPR]
    factor_names = []
    factor_categories = []
    for expr, name, cat in NEW_FACTORS:
        all_exprs.append(expr)
        factor_names.append(name)
        factor_categories.append(cat)

    n_factors = len(factor_names)
    print(f"Testing {n_factors} new factors...")

    # Fetch data in batches to save memory
    BATCH = 5
    results = []

    for batch_start in range(0, n_factors, BATCH):
        batch_end = min(batch_start + BATCH, n_factors)
        batch_names = factor_names[batch_start:batch_end]
        batch_exprs = [LABEL_EXPR] + [all_exprs[1 + i] for i in range(batch_start, batch_end)]
        batch_fields = [f"f{i}" for i in range(len(batch_exprs))]

        print(f"\n  Batch {batch_start//BATCH + 1}: {', '.join(batch_names)}")

        try:
            data = D.features(
                stock_list, batch_exprs,
                start_time=args.start, end_time=args.end,
            )
            data.columns = batch_fields

            label = data["f0"]

            for j, name in enumerate(batch_names):
                factor = data[f"f{j+1}"]
                ic_result = fast_daily_rankic(factor, label, min_stocks=min_stocks)
                ic_result["name"] = name
                ic_result["category"] = factor_categories[batch_start + j]
                ic_result["expression"] = NEW_FACTORS[batch_start + j][0]
                results.append(ic_result)

                ric = ic_result["rank_ic_mean"]
                t = ic_result["rank_ic_t"]
                icir = ic_result["rank_icir"]
                print(f"    {name:30s}  RankIC={ric:+.4f}  t={t:+.2f}  ICIR={icir:+.4f}")

        except Exception as e:
            for name in batch_names:
                print(f"    {name:30s}  ERROR: {e}")
                results.append({"name": name, "rank_ic_mean": np.nan, "rank_ic_t": np.nan})

        gc.collect()

    # Summary
    df = pd.DataFrame(results)
    df = df.sort_values("rank_ic_t", key=abs, ascending=False)

    # FDR correction (Benjamini-Hochberg)
    valid = df["rank_ic_p"].notna()
    if valid.sum() > 0:
        p_vals = df.loc[valid, "rank_ic_p"].values
        n_tests = len(p_vals)
        sorted_idx = np.argsort(p_vals)
        sorted_p = p_vals[sorted_idx]
        # BH procedure
        fdr_adj = np.zeros(n_tests)
        for i in range(n_tests - 1, -1, -1):
            rank = i + 1
            if i == n_tests - 1:
                fdr_adj[sorted_idx[i]] = sorted_p[i]
            else:
                fdr_adj[sorted_idx[i]] = min(
                    sorted_p[i] * n_tests / rank,
                    fdr_adj[sorted_idx[i + 1]]
                )
        df.loc[valid, "fdr_p"] = fdr_adj
        df["significant"] = (df["fdr_p"] < 0.01).astype(int)

    print("\n" + "=" * 80)
    print(f"Results ({args.market}, {args.start} ~ {args.end}):")
    print("=" * 80)
    display_cols = ["name", "category", "rank_ic_mean", "rank_ic_t", "rank_icir", "fdr_p", "significant", "n_days"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))

    n_sig = df["significant"].sum() if "significant" in df.columns else 0
    print(f"\nSignificant (FDR < 0.01): {n_sig}/{len(df)}")

    # Save CSV
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"{args.market}_new_factors_ic.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    # Backfill to factor library
    if args.backfill:
        from project_qlib.factor_db import FactorDB
        db = FactorDB()
        for _, row in df.iterrows():
            name = row["name"]
            cat = row.get("category", "")
            expr = row.get("expression", "")

            # Upsert factor definition
            db.upsert_factor(
                name=name,
                expression=expr,
                category=cat,
                source=args.source_tag,
                status="Candidate",
            )

            # Upsert test result
            if pd.notna(row.get("rank_ic_mean")):
                sig = bool(row.get("significant", 0))
                db.upsert_test_result(
                    factor_name=name,
                    market=args.market,
                    test_start=args.start,
                    test_end=args.end,
                    n_days=int(row.get("n_days", 0)),
                    ic_mean=row.get("ic_mean"),
                    ic_std=row.get("ic_std"),
                    rank_ic_mean=row["rank_ic_mean"],
                    rank_ic_std=row.get("rank_ic_std"),
                    rank_ic_t=row["rank_ic_t"],
                    rank_ic_p=row.get("rank_ic_p"),
                    rank_icir=row.get("rank_icir"),
                    fdr_p=row.get("fdr_p"),
                    significant=sig,
                    hea_round=args.round_id,
                )

                # Update status based on significance
                new_status = "Accepted" if sig else "Rejected"
                db.upsert_factor(name=name, status=new_status)

        db.close()
        print(f"Backfilled {len(df)} factors to factor library.")


if __name__ == "__main__":
    main()

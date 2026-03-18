"""Multi-horizon IC decay analysis for factor library.

Computes RankIC at multiple forward-return horizons (1d, 3d, 5d, 10d, 20d)
to understand factor persistence / half-life. Results are saved to
`factor_ic_decay` table and CSV.

This is critical for:
- Identifying which factors are suitable for weekly (5d) holding
- Understanding the decay profile to set optimal rebalance frequency
- Filtering out factors whose alpha vanishes beyond 1d

Usage:
    uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_ic_decay.py --market csi1000
    uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_ic_decay.py --market csi1000 --top-n 50 --backfill
    uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_ic_decay.py --market csi1000 --horizons 1,3,5,10
"""
from __future__ import annotations

import argparse
import gc
import sqlite3
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


HORIZONS_DEFAULT = [1, 3, 5, 10, 20]


def _label_expr(horizon: int) -> str:
    """Build forward return expression for a given horizon."""
    return f"Ref($close, -{horizon - 1})/Ref($close, -1) - 1" if horizon > 1 else "Ref($close, -2)/Ref($close, -1) - 1"


def fast_daily_rankic(factor: pd.Series, label: pd.Series, min_stocks: int = 30) -> dict:
    """Compute daily cross-sectional Rank IC."""
    combined = pd.DataFrame({"factor": factor, "label": label}).dropna()
    if len(combined) == 0:
        return {"n_days": 0, "ic_mean": np.nan, "rank_ic_mean": np.nan,
                "rank_ic_std": np.nan, "rank_ic_t": np.nan, "rank_icir": np.nan}

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
                "rank_ic_std": np.nan, "rank_ic_t": np.nan, "rank_icir": np.nan}

    ric_mean = daily_ric.mean()
    ric_std = daily_ric.std()
    ric_t = ric_mean / (ric_std / np.sqrt(n)) if ric_std > 0 else 0

    return {
        "n_days": n,
        "ic_mean": float(daily_ic.mean()),
        "ic_std": float(daily_ic.std()),
        "rank_ic_mean": float(ric_mean),
        "rank_ic_std": float(ric_std),
        "rank_ic_t": float(ric_t),
        "rank_icir": float(ric_mean / ric_std) if ric_std > 0 else 0,
    }


def get_top_factors(db_path: Path, market: str, top_n: int) -> list[dict]:
    """Retrieve top factors by |rank_icir| from the factor library."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT DISTINCT f.name, f.expression, f.category, f.source,
               t.rank_icir, t.fdr_p
        FROM factors f
        JOIN factor_test_results t ON f.name = t.factor_name
        WHERE t.market = ? AND f.status IN ('Accepted', 'Baseline')
              AND f.expression IS NOT NULL AND f.expression != ''
        ORDER BY ABS(COALESCE(t.rank_icir, 0)) DESC
    """, (market,)).fetchall()
    conn.close()

    seen = set()
    result = []
    for row in rows:
        name = row[0]
        if name in seen:
            continue
        seen.add(name)
        result.append({
            "name": name, "expression": row[1], "category": row[2],
            "source": row[3], "rank_icir_1d": row[4], "fdr_p": row[5],
        })
        if len(result) >= top_n:
            break
    return result


def backfill_db(results: list[dict], market: str, start: str, end: str) -> None:
    """Write IC decay results to factor_ic_decay table."""
    from project_qlib.factor_db import FactorDB
    db = FactorDB()
    count = 0
    for r in results:
        db.upsert_ic_decay(
            factor_name=r["factor_name"],
            market=market,
            horizon_days=r["horizon"],
            test_start=start,
            test_end=end,
            n_days=r.get("n_days"),
            ic_mean=r.get("ic_mean"),
            ic_std=r.get("ic_std"),
            rank_ic_mean=r.get("rank_ic_mean"),
            rank_ic_std=r.get("rank_ic_std"),
            rank_ic_t=r.get("rank_ic_t"),
            rank_icir=r.get("rank_icir"),
        )
        count += 1
    print(f"\n[Backfill] Wrote {count} IC-decay records (market={market})")
    db.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-horizon IC decay analysis")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--top-n", type=int, default=50, help="Top N factors by |ICIR|")
    parser.add_argument("--horizons", default="1,3,5,10,20", help="Comma-separated horizons")
    parser.add_argument("--backfill", action="store_true", help="Write to DB")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    market = args.market
    db_path = PROJECT_ROOT / "data" / "factor_library.db"

    init_qlib()
    from qlib.data import D

    # 1. Get top factors
    factors = get_top_factors(db_path, market, args.top_n)
    print(f"Loaded {len(factors)} factors for IC decay analysis")
    print(f"Horizons: {horizons}")

    instruments = D.instruments(market)
    min_stocks = 30 if market in ("csi1000", "csi300") else 50

    # 2. Preload labels for all horizons
    print("\nLoading labels for all horizons...")
    labels = {}
    for h in horizons:
        expr = _label_expr(h)
        ldf = D.features(instruments, [expr], start_time=args.start, end_time=args.end)
        labels[h] = ldf.iloc[:, 0]
        print(f"  {h}d label: {len(labels[h])} rows")

    # 3. Compute IC at each horizon for each factor
    all_results = []
    for batch_start in range(0, len(factors), args.batch_size):
        batch = factors[batch_start:batch_start + args.batch_size]
        exprs = [f["expression"] for f in batch]
        names = [f["name"] for f in batch]

        print(f"\nBatch [{batch_start}:{batch_start+len(batch)}] loading {len(batch)} factors...")
        df = D.features(instruments, exprs, start_time=args.start, end_time=args.end)
        df.columns = names

        for fi, fname in enumerate(names):
            factor_series = df[fname]
            row_results = {"factor_name": fname, "category": batch[fi]["category"]}

            for h in horizons:
                ic_res = fast_daily_rankic(factor_series, labels[h], min_stocks=min_stocks)
                row_results[f"rank_ic_{h}d"] = ic_res["rank_ic_mean"]
                row_results[f"rank_icir_{h}d"] = ic_res["rank_icir"]
                row_results[f"rank_ic_t_{h}d"] = ic_res["rank_ic_t"]

                all_results.append({
                    "factor_name": fname,
                    "horizon": h,
                    **ic_res,
                })

            # Print compact summary
            decay_str = " | ".join(
                f"{h}d: {row_results.get(f'rank_icir_{h}d', 0):.3f}"
                for h in horizons
            )
            print(f"  [{batch_start+fi+1}/{len(factors)}] {fname}: {decay_str}")

        del df
        gc.collect()

    # 4. Build pivot table: factor × horizon
    pivot_rows = []
    for fname in dict.fromkeys(r["factor_name"] for r in all_results):
        row = {"factor_name": fname}
        for r in all_results:
            if r["factor_name"] == fname:
                h = r["horizon"]
                row[f"rank_ic_{h}d"] = r["rank_ic_mean"]
                row[f"rank_icir_{h}d"] = r["rank_icir"]
        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows)

    # Compute decay ratio: 5d ICIR / 1d ICIR
    if "rank_icir_1d" in pivot_df.columns and "rank_icir_5d" in pivot_df.columns:
        pivot_df["decay_5d_1d"] = pivot_df["rank_icir_5d"] / pivot_df["rank_icir_1d"].replace(0, np.nan)
        pivot_df["weekly_suitable"] = pivot_df["decay_5d_1d"].abs() >= 0.5  # 5d retains >= 50% of 1d ICIR

    # Sort by 5d ICIR
    sort_col = "rank_icir_5d" if "rank_icir_5d" in pivot_df.columns else "rank_icir_1d"
    pivot_df = pivot_df.sort_values(sort_col, ascending=False, key=abs)

    # 5. Print results
    print("\n" + "=" * 140)
    print(f"  IC DECAY ANALYSIS: Top {args.top_n} factors on {market.upper()} ({args.start} ~ {args.end})")
    print("=" * 140)

    cols = ["factor_name"] + [f"rank_icir_{h}d" for h in horizons]
    if "decay_5d_1d" in pivot_df.columns:
        cols += ["decay_5d_1d", "weekly_suitable"]

    print(pivot_df[cols].head(50).to_string(index=False, float_format="{:.4f}".format))

    # Weekly-suitable summary
    if "weekly_suitable" in pivot_df.columns:
        n_weekly = pivot_df["weekly_suitable"].sum()
        print(f"\nWeekly-suitable factors (5d/1d ICIR ratio >= 0.5): {n_weekly}/{len(pivot_df)}")
        top_weekly = pivot_df[pivot_df["weekly_suitable"]].head(20)
        if len(top_weekly) > 0:
            print("\nTop-20 weekly-suitable factors:")
            print(top_weekly[cols].to_string(index=False, float_format="{:.4f}".format))

    # 6. Save CSV
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{market}_ic_decay_analysis.csv"
    pivot_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # 7. Backfill to DB
    if args.backfill:
        backfill_db(all_results, market, args.start, args.end)


if __name__ == "__main__":
    main()

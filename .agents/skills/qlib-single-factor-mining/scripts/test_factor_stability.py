"""Factor regime stability analysis via rolling-window IC.

Computes RankIC in rolling windows to evaluate:
- Factor stability across different market regimes
- IC trend (improving vs degrading)
- Drawdown periods (how long does a factor go negative?)

Usage:
    uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_stability.py --market csi1000
    uv run python .agents/skills/qlib-single-factor-mining/scripts/test_factor_stability.py --market csi1000 --top-n 30 --window 60
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

from project_qlib.runtime import init_qlib


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
            "source": row[3], "rank_icir_full": row[4],
        })
        if len(result) >= top_n:
            break
    return result


def rolling_ic_series(
    factor: pd.Series,
    label: pd.Series,
    window: int = 60,
    min_stocks: int = 30,
) -> pd.Series:
    """Compute rolling-window cross-sectional Rank IC.

    Returns a time series of Rank IC values (one per trading day).
    """
    combined = pd.DataFrame({"factor": factor, "label": label}).dropna()
    if len(combined) == 0:
        return pd.Series(dtype=float)

    dates = combined.index.get_level_values(1)

    def _spearman(g):
        return np.nan if len(g) < min_stocks else g["factor"].rank().corr(g["label"].rank())

    daily_ric = combined.groupby(dates).apply(_spearman).dropna()
    # Rolling average IC
    rolling_mean = daily_ric.rolling(window=window, min_periods=window // 2).mean()
    return rolling_mean.dropna()


def compute_stability_metrics(rolling_ic: pd.Series) -> dict:
    """Compute stability metrics from a rolling IC series."""
    if len(rolling_ic) < 10:
        return {
            "ic_positive_pct": np.nan,
            "ic_max_drawdown_days": np.nan,
            "ic_trend_slope": np.nan,
            "ic_regime_std": np.nan,
            "ic_recent_vs_full": np.nan,
            "stability_score": np.nan,
        }

    # % of time IC is positive (directional consistency)
    positive_pct = float((rolling_ic > 0).mean()) if rolling_ic.mean() > 0 else float((rolling_ic < 0).mean())

    # Max consecutive days with wrong-sign IC
    target_sign = 1 if rolling_ic.mean() > 0 else -1
    wrong_sign = (rolling_ic * target_sign) < 0
    max_drawdown_days = 0
    current_run = 0
    for v in wrong_sign:
        if v:
            current_run += 1
            max_drawdown_days = max(max_drawdown_days, current_run)
        else:
            current_run = 0

    # Trend: linear regression slope of IC over time
    x = np.arange(len(rolling_ic))
    if len(x) > 1:
        slope = float(np.polyfit(x, rolling_ic.values, 1)[0])
    else:
        slope = 0.0

    # Regime volatility: std of rolling IC
    regime_std = float(rolling_ic.std())

    # Recent (last 20%) vs full period IC
    recent_n = max(1, len(rolling_ic) // 5)
    recent_mean = float(rolling_ic.iloc[-recent_n:].mean())
    full_mean = float(rolling_ic.mean())
    recent_vs_full = recent_mean / full_mean if abs(full_mean) > 1e-8 else np.nan

    # Composite stability score (0-1, higher = more stable)
    # Components: positive_pct, low regime_std, no long drawdowns, consistent recent
    s1 = min(positive_pct, 1.0)  # 0-1
    s2 = max(0, 1 - max_drawdown_days / max(len(rolling_ic), 1))  # 0-1
    s3 = max(0, 1 - regime_std / (abs(full_mean) + 1e-8))  # 0-1
    s4 = min(max(recent_vs_full if not np.isnan(recent_vs_full) else 0, 0), 2) / 2  # 0-1
    stability_score = float(np.mean([s1, s2, s3, s4]))

    return {
        "ic_positive_pct": positive_pct,
        "ic_max_drawdown_days": max_drawdown_days,
        "ic_trend_slope": slope,
        "ic_regime_std": regime_std,
        "ic_recent_vs_full": recent_vs_full,
        "stability_score": stability_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Factor regime stability analysis")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--window", type=int, default=60, help="Rolling window size in trading days")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon (1=daily, 5=weekly)")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    market = args.market
    db_path = PROJECT_ROOT / "data" / "factor_library.db"

    init_qlib()
    from qlib.data import D

    factors = get_top_factors(db_path, market, args.top_n)
    print(f"Loaded {len(factors)} factors for stability analysis")
    print(f"Window: {args.window}d, Horizon: {args.horizon}d")

    instruments = D.instruments(market)
    min_stocks = 30 if market in ("csi1000", "csi300") else 50

    # Load label
    if args.horizon == 1:
        label_expr = "Ref($close, -2)/Ref($close, -1) - 1"
    else:
        label_expr = f"Ref($close, -{args.horizon - 1})/Ref($close, -1) - 1"

    print(f"\nLoading {args.horizon}d label...")
    label_df = D.features(instruments, [label_expr], start_time=args.start, end_time=args.end)
    label = label_df.iloc[:, 0]

    all_results = []
    for batch_start in range(0, len(factors), args.batch_size):
        batch = factors[batch_start:batch_start + args.batch_size]
        exprs = [f["expression"] for f in batch]
        names = [f["name"] for f in batch]

        print(f"\nBatch [{batch_start}:{batch_start+len(batch)}]...")
        df = D.features(instruments, exprs, start_time=args.start, end_time=args.end)
        df.columns = names

        for fi, fname in enumerate(names):
            ric_series = rolling_ic_series(df[fname], label, window=args.window, min_stocks=min_stocks)
            metrics = compute_stability_metrics(ric_series)
            metrics["factor_name"] = fname
            metrics["category"] = batch[fi]["category"]
            metrics["rank_icir_full"] = batch[fi]["rank_icir_full"]
            all_results.append(metrics)

            print(f"  [{batch_start+fi+1}/{len(factors)}] {fname}: "
                  f"stability={metrics['stability_score']:.3f}, "
                  f"pos%={metrics['ic_positive_pct']:.1%}, "
                  f"maxDD={metrics['ic_max_drawdown_days']}d, "
                  f"recent/full={metrics['ic_recent_vs_full']:.2f}")

        del df
        gc.collect()

    # Build summary table
    res_df = pd.DataFrame(all_results)
    res_df = res_df.sort_values("stability_score", ascending=False)

    print("\n" + "=" * 140)
    print(f"  FACTOR STABILITY RANKING ({args.window}d rolling, {args.horizon}d horizon)")
    print("=" * 140)

    display_cols = ["factor_name", "category", "rank_icir_full", "stability_score",
                    "ic_positive_pct", "ic_max_drawdown_days", "ic_trend_slope",
                    "ic_recent_vs_full"]
    print(res_df[display_cols].head(50).to_string(index=False, float_format="{:.4f}".format))

    # Identify stable + strong factors
    if len(res_df) > 0:
        stable_strong = res_df[
            (res_df["stability_score"] >= 0.5) &
            (res_df["rank_icir_full"].abs() >= 0.10)
        ]
        print(f"\nStable & strong factors (stability>=0.5, |ICIR|>=0.10): {len(stable_strong)}/{len(res_df)}")

    # Save
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{market}_factor_stability_{args.horizon}d.csv"
    res_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""Analyze CSI1000 benchmark return and factor importance."""
import sys
sys.path.insert(0, "src")

from project_qlib.runtime import init_qlib
init_qlib()

import numpy as np
import pandas as pd
from qlib.data import D

# 1. CSI1000 benchmark (SH000852) return in test period
print("=== CSI1000 Index (SH000852) Performance ===")
bench = D.features(["sh000852"], ["$close"], start_time="2024-07-01", end_time="2026-02-13")
if len(bench) > 0:
    dates = bench.index.get_level_values(1)
    close = bench.iloc[:, 0]
    start_val = close.iloc[0]
    end_val = close.iloc[-1]
    total_ret = end_val / start_val - 1
    n_days = len(close)
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1
    print(f"  Period: {dates.min().strftime('%Y-%m-%d')} ~ {dates.max().strftime('%Y-%m-%d')}")
    print(f"  Start: {start_val:.2f}, End: {end_val:.2f}")
    print(f"  Total Return: {total_ret*100:.2f}%")
    print(f"  Annualized Return: {ann_ret*100:.2f}%")
    print(f"  Trading Days: {n_days}")
    
    # Key dates
    peak = close.max()
    peak_date = dates[close.argmax()]
    trough = close.min()
    trough_date = dates[close.argmin()]
    print(f"  Peak: {peak:.2f} ({peak_date.strftime('%Y-%m-%d')})")
    print(f"  Trough: {trough:.2f} ({trough_date.strftime('%Y-%m-%d')})")
    mdd = (close / close.cummax() - 1).min()
    print(f"  Max Drawdown: {mdd*100:.2f}%")

# 2. Our strategy's EXCESS return interpretation
# excess_return_with_cost = strategy_return - benchmark_return
# So if excess_return = 5.1%, and benchmark returned X%, strategy returned X%+5.1%
print(f"\n=== Strategy vs Benchmark ===")
print(f"  TopN50 excess return (with cost): +5.1% annualized")
print(f"  Baseline excess return (with cost): +4.0% annualized")
print(f"  CSI1000 benchmark annualized: {ann_ret*100:.2f}%")
print(f"  => TopN50 absolute return ≈ {(ann_ret + 0.051)*100:.2f}%")
print(f"  => Baseline absolute return ≈ {(ann_ret + 0.040)*100:.2f}%")

# 3. Factor ranking from unified ranking
print(f"\n=== Current Factor Rankings (Top-50, deduplicated) ===")
ranking = pd.read_csv("/Volumes/Workspace/agentic-alpha/outputs/csiall_unified_factor_ranking.csv")
ranking = ranking[~ranking["factor"].str.match(r"^VSUMP|^VSUMN")]
top50 = ranking.head(50)

print(f"{'Rank':>4} {'Factor':<28} {'Source':<10} {'Cat':<14} {'RankIC':>8} {'|t|':>8} {'ICIR':>8}")
print("-" * 90)
for i, (_, row) in enumerate(top50.iterrows(), 1):
    print(f"{i:>4} {row['factor']:<28} {row['source']:<10} {row['category']:<14} {row['rank_ic_mean']:>+8.4f} {abs(row['rank_ic_t']):>8.2f} {row['rank_icir']:>+8.4f}")

# Source distribution
a158_count = (top50.source == "Alpha158").sum()
cstm_count = (top50.source == "Custom").sum()
print(f"\nTop-50 分布: Alpha158={a158_count}, Custom={cstm_count}")

# Category distribution
print(f"\n类别分布:")
for cat, cnt in top50.groupby("category").size().sort_values(ascending=False).items():
    print(f"  {cat}: {cnt}")

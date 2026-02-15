"""Compare rebalancing frequencies using existing TopN50 predictions.

Loads pred.pkl from the existing TopN50 run (d3aec63d) and runs backtest
with different hold_thresh values to simulate different rebalancing frequencies.
"""
from __future__ import annotations

import gc
import json
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd


def run_backtest(pred_score, strategy_config: dict, label: str) -> dict:
    """Run backtest with given strategy config."""
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy

    print(f"\n--- Backtest: {label} ---")
    print(f"  topk={strategy_config['topk']}, n_drop={strategy_config['n_drop']}, hold_thresh={strategy_config.get('hold_thresh', 1)}")

    t0 = time.time()

    report_normal, positions = backtest_daily(
        start_time="2024-07-01",
        end_time="2026-02-13",
        strategy={
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": pred_score,
                "topk": strategy_config["topk"],
                "n_drop": strategy_config["n_drop"],
                "hold_thresh": strategy_config.get("hold_thresh", 1),
            },
        },
        account=100_000_000,
        benchmark="SH000852",
        exchange_kwargs={
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    df = report_normal
    cum_strat = (1 + df['return']).prod() - 1
    cum_bench = (1 + df['bench']).prod() - 1
    n_days = len(df)
    ann_strat = (1 + cum_strat) ** (252 / n_days) - 1
    ann_bench = (1 + cum_bench) ** (252 / n_days) - 1

    excess_daily = df['return'] - df['bench']
    ann_excess = ann_strat - ann_bench
    ir = excess_daily.mean() / excess_daily.std() * np.sqrt(252) if excess_daily.std() > 0 else 0

    cum_ret = (1 + df['return']).cumprod()
    mdd_strat = (cum_ret / cum_ret.cummax() - 1).min()

    cum_excess_ret = (1 + excess_daily).cumprod()
    mdd_excess = (cum_excess_ret / cum_excess_ret.cummax() - 1).min()

    avg_turnover = df['turnover'].mean()
    total_cost = df['cost'].sum()
    total_cost_pct = total_cost / 100_000_000 * 100

    # Monthly breakdown
    df_m = df.copy()
    df_m.index = pd.to_datetime(df_m.index)
    df_m['excess'] = df_m['return'] - df_m['bench']
    monthly = df_m.groupby(df_m.index.to_period('M')).agg({
        'return': lambda x: (1 + x).prod() - 1,
        'bench': lambda x: (1 + x).prod() - 1,
        'excess': lambda x: (1 + x).prod() - 1,
    })
    n_pos = (monthly['excess'] > 0).sum()
    n_neg = (monthly['excess'] <= 0).sum()
    win_rate = n_pos / (n_pos + n_neg) * 100

    metrics = {
        "label": label,
        "cum_return": float(cum_strat),
        "ann_return": float(ann_strat),
        "cum_bench": float(cum_bench),
        "ann_bench": float(ann_bench),
        "cum_excess": float(cum_strat - cum_bench),
        "ann_excess": float(ann_excess),
        "IR": float(ir),
        "mdd_strat": float(mdd_strat),
        "mdd_excess": float(mdd_excess),
        "avg_turnover": float(avg_turnover),
        "total_cost_pct": float(total_cost_pct),
        "win_rate": float(win_rate),
        "n_days": n_days,
    }

    print(f"  CumRet: {cum_strat*100:.1f}%, Bench: {cum_bench*100:.1f}%, Excess: {(cum_strat-cum_bench)*100:.1f}%")
    print(f"  AnnRet: {ann_strat*100:.1f}%, AnnExcess: {ann_excess*100:.1f}%")
    print(f"  IR: {ir:.2f}, MDD: {mdd_strat*100:.1f}%, Turnover: {avg_turnover:.4f}, Cost: {total_cost_pct:.2f}%")
    print(f"  Monthly WinRate: {win_rate:.0f}%")

    return metrics


def main():
    from project_qlib.runtime import init_qlib
    init_qlib()

    # Load predictions from existing TopN50 run
    run_dir = list(Path("mlruns/1").glob("d3aec63d*"))[0]
    pred_pkl = run_dir / "artifacts" / "pred.pkl"
    print(f"Loading predictions from {pred_pkl}")
    with open(pred_pkl, "rb") as f:
        pred = pickle.load(f)
    print(f"  Shape: {pred.shape}")

    # Convert to Series if DataFrame
    if isinstance(pred, pd.DataFrame):
        pred_score = pred.iloc[:, 0]
    else:
        pred_score = pred

    # Strategy configs to compare
    strategies = [
        {
            "name": "daily_d5",
            "label": "Daily (d=5, h=1)",
            "topk": 50, "n_drop": 5, "hold_thresh": 1,
        },
        {
            "name": "daily_d3",
            "label": "Daily (d=3, h=1)",
            "topk": 50, "n_drop": 3, "hold_thresh": 1,
        },
        {
            "name": "daily_d1",
            "label": "Daily (d=1, h=1)",
            "topk": 50, "n_drop": 1, "hold_thresh": 1,
        },
        {
            "name": "weekly",
            "label": "Weekly (d=5, h=5)",
            "topk": 50, "n_drop": 5, "hold_thresh": 5,
        },
        {
            "name": "biweekly",
            "label": "Biweekly (d=5, h=10)",
            "topk": 50, "n_drop": 5, "hold_thresh": 10,
        },
        {
            "name": "monthly",
            "label": "Monthly (d=5, h=20)",
            "topk": 50, "n_drop": 5, "hold_thresh": 20,
        },
    ]

    results = []
    for s in strategies:
        config = {"topk": s["topk"], "n_drop": s["n_drop"], "hold_thresh": s["hold_thresh"]}
        r = run_backtest(pred_score, config, s["label"])
        r["strategy"] = s["name"]
        results.append(r)
        gc.collect()

    # Summary table
    print(f"\n{'='*140}")
    print(f"  CSI1000 TopN50 — Rebalancing Frequency Comparison")
    print(f"  Test: 2024-07-01 ~ 2026-02-13, Benchmark: SH000852")
    print(f"{'='*140}")
    print(f"{'Strategy':<25} {'CumRet':>8} {'AnnRet':>8} {'CumExc':>8} {'AnnExc':>8} {'IR':>7} {'MDD':>8} {'ExcMDD':>8} {'Turnover':>9} {'Cost%':>7} {'WinR':>5}")
    print("-" * 140)
    for r in results:
        print(f"{r['label']:<25} {r['cum_return']*100:>7.1f}% {r['ann_return']*100:>7.1f}% "
              f"{r['cum_excess']*100:>7.1f}% {r['ann_excess']*100:>7.1f}% {r['IR']:>7.2f} "
              f"{r['mdd_strat']*100:>7.1f}% {r['mdd_excess']*100:>7.1f}% "
              f"{r['avg_turnover']:>9.4f} {r['total_cost_pct']:>6.2f}% {r['win_rate']:>4.0f}%")

    # Save
    out = PROJECT_ROOT / "outputs" / "csi1000_rebalance_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

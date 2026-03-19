"""MFA Weekly Rebalance with Plan B diversity factor pool.

Uses:
- Factor pool: outputs/mfa_factor_pool_v7_diverse.csv (30 factors, 20 categories)
- Label: 5d forward return
- Rebalance: every 5 trading days
- Model: XGB+LGB Ensemble (Rolling 3m, quarterly retrain)
- Topk: 20 (V6 SOTA config)
- Hold buffer: stocks held unless rank drops below buffer_rank
- OOS: 2025-01-01 ~ latest data
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

def _find_root(start: Path) -> Path:
    for c in (start, *start.parents):
        if (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("project root not found")

PROJECT_ROOT = _find_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from project_qlib.runtime import init_qlib


def load_pool_from_csv(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    return [{"name": r["name"], "expression": r["expression"], "category": r["category"]}
            for _, r in df.iterrows() if pd.notna(r["expression"])]


CACHE_PATH = PROJECT_ROOT / "outputs" / "mfa_weekly_v7_predictions.pkl"


def train_and_predict(
    market: str = "csi1000",
    label_horizon: int = 5,
    train_start: str = "2018-01-01",
    oos_start: str = "2025-01-01",
    oos_end: str = "2026-03-18",
) -> pd.Series:
    """Train ensemble model and return OOS predictions. Uses cache if available."""
    if CACHE_PATH.exists():
        print(f"Loading cached predictions from {CACHE_PATH.name}")
        return pd.read_pickle(CACHE_PATH)

    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158
    from dateutil.relativedelta import relativedelta

    label_expr = f"Ref($close, -{label_horizon})/Ref($close, -1) - 1"

    pool_csv = PROJECT_ROOT / "outputs" / "mfa_factor_pool_v7_diverse.csv"
    db_factors = load_pool_from_csv(pool_csv)
    print(f"Factor pool: {len(db_factors)} from {pool_csv.name}")

    h = Alpha158.__new__(Alpha158)
    a158_fields, a158_names = h.get_feature_config()
    print(f"Alpha158: {len(a158_names)} features")

    all_fields = list(a158_fields)
    all_names = list(a158_names)
    for f in db_factors:
        if f["name"] not in all_names:
            all_fields.append(f["expression"])
            all_names.append(f["name"])
    print(f"Total features: {len(all_names)}")

    instruments = D.instruments(market)

    oos_dt = datetime.strptime(oos_start, "%Y-%m-%d")
    end_dt = datetime.strptime(oos_end, "%Y-%m-%d")
    train_start_dt = datetime.strptime(train_start, "%Y-%m-%d")

    windows = []
    current = oos_dt
    while current < end_dt:
        w_end = min(current + relativedelta(months=3), end_dt)
        windows.append({
            "train_start": train_start_dt.strftime("%Y-%m-%d"),
            "train_end": (current - relativedelta(days=1)).strftime("%Y-%m-%d"),
            "valid_start": (current - relativedelta(years=1)).strftime("%Y-%m-%d"),
            "valid_end": (current - relativedelta(days=1)).strftime("%Y-%m-%d"),
            "test_start": current.strftime("%Y-%m-%d"),
            "test_end": w_end.strftime("%Y-%m-%d"),
        })
        current = w_end

    print(f"Rolling windows: {len(windows)}")

    all_predictions = []
    for wi, w in enumerate(windows):
        print(f"\n{'─'*60}")
        print(f"Window {wi+1}/{len(windows)}: test={w['test_start']}~{w['test_end']}")

        train_data = D.features(instruments, all_fields,
                                start_time=w["train_start"], end_time=w["train_end"])
        train_data.columns = all_names
        valid_data = D.features(instruments, all_fields,
                                start_time=w["valid_start"], end_time=w["valid_end"])
        valid_data.columns = all_names
        test_data = D.features(instruments, all_fields,
                               start_time=w["test_start"], end_time=w["test_end"])
        test_data.columns = all_names

        train_label = D.features(instruments, [label_expr],
                                 start_time=w["train_start"], end_time=w["train_end"]).iloc[:, 0]
        valid_label = D.features(instruments, [label_expr],
                                 start_time=w["valid_start"], end_time=w["valid_end"]).iloc[:, 0]

        for df_ in [train_data, valid_data, test_data]:
            for col in df_.columns:
                df_[col] = pd.to_numeric(df_[col], errors='coerce')

        ci = train_data.index.intersection(train_label.index)
        X_train, y_train = train_data.loc[ci].fillna(0), train_label.loc[ci].fillna(0)
        ci_v = valid_data.index.intersection(valid_label.index)
        X_valid, y_valid = valid_data.loc[ci_v].fillna(0), valid_label.loc[ci_v].fillna(0)
        X_test = test_data.fillna(0)

        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=1000, max_depth=8, learning_rate=0.05,
            colsample_bytree=0.8879, subsample=0.8789,
            reg_alpha=205.70, reg_lambda=580.98,
            tree_method="hist", n_jobs=-1, verbosity=0,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        pred_xgb = xgb.predict(X_test)
        del xgb; gc.collect()

        from lightgbm import LGBMRegressor
        lgb = LGBMRegressor(
            n_estimators=1000, max_depth=8, learning_rate=0.05,
            num_leaves=128, reg_alpha=205.70, reg_lambda=580.97,
            n_jobs=-1, verbosity=-1,
        )
        lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[])
        pred_lgb = lgb.predict(X_test)
        del lgb; gc.collect()

        pred = (pred_xgb + pred_lgb) / 2
        pred_s = pd.Series(pred, index=X_test.index, name="score")
        all_predictions.append(pred_s)

        del train_data, valid_data, test_data, train_label, valid_label
        del X_train, y_train, X_valid, y_valid, X_test
        gc.collect()
        try:
            from qlib.data.cache import H
            H.clear()
        except Exception:
            pass

    predictions = pd.concat(all_predictions).sort_index()
    predictions.to_pickle(CACHE_PATH)
    print(f"\nPredictions cached to {CACHE_PATH.name} ({len(predictions)} rows)")
    return predictions


def simulate_weekly_rebal(
    predictions: pd.Series,
    market: str = "csi1000",
    topk: int = 20,
    buffer_rank: int = 60,
    rebal_days: int = 5,
    oos_start: str = "2025-01-01",
    oos_end: str = "2026-03-18",
    cost_rate: float = 0.002,
) -> dict:
    """Weekly rebalance simulation with hold buffer.

    Hold buffer logic:
    - On rebalance day, rank all stocks by predicted score
    - Keep existing holdings whose rank <= buffer_rank
    - Drop holdings below buffer_rank
    - Fill remaining slots (up to topk) from top-ranked non-held stocks
    - Equal weight across held stocks
    """
    from qlib.data import D

    benchmark = "SH000852" if market == "csi1000" else "SH000300"
    instruments = D.instruments(market)

    close_df = D.features(instruments, ["$close"], start_time=oos_start, end_time=oos_end)
    close_df.columns = ["close"]

    bench_close = D.features([benchmark], ["$close"], start_time=oos_start, end_time=oos_end)
    bench_s = bench_close.iloc[:, 0]
    bench_s.index = bench_s.index.droplevel("instrument")
    bench_daily_ret = bench_s.pct_change(fill_method=None).dropna()

    dates = sorted(predictions.index.get_level_values("datetime").unique())
    print(f"Trading days: {len(dates)}, rebalance every {rebal_days}d, buffer_rank={buffer_rank}")

    portfolio = set()  # set of held stock codes
    daily_returns_with_cost = []
    daily_returns_no_cost = []
    daily_turnovers = []

    for di, date in enumerate(dates):
        try:
            day_pred = predictions.xs(date, level="datetime").sort_values(ascending=False)
        except KeyError:
            daily_returns_with_cost.append(0.0)
            daily_returns_no_cost.append(0.0)
            daily_turnovers.append(0.0)
            continue

        is_rebal = (di % rebal_days == 0)
        cost = 0.0
        turnover = 0.0

        if is_rebal and len(day_pred) >= topk:
            # Rank all stocks (1-based)
            ranks = pd.Series(range(1, len(day_pred) + 1), index=day_pred.index)

            # Keep held stocks within buffer
            kept = {s for s in portfolio if s in ranks.index and ranks[s] <= buffer_rank}

            # Fill remaining slots from top-ranked non-held
            n_fill = topk - len(kept)
            if n_fill > 0:
                candidates = [s for s in day_pred.index if s not in kept]
                new_picks = set(candidates[:n_fill])
            else:
                new_picks = set()
                # If kept > topk, trim from worst-ranked
                if len(kept) > topk:
                    kept_ranked = sorted(kept, key=lambda s: ranks.get(s, 9999))
                    kept = set(kept_ranked[:topk])
                    new_picks = set()

            new_portfolio = kept | new_picks

            sold = portfolio - new_portfolio
            bought = new_portfolio - portfolio
            turnover = (len(sold) + len(bought)) / topk if topk > 0 else 0
            cost = turnover * cost_rate

            portfolio = new_portfolio
            daily_turnovers.append(turnover)
        else:
            daily_turnovers.append(0.0)

        # Compute portfolio return
        if portfolio and di > 0:
            prev_date = dates[di - 1]
            try:
                today_close = close_df.xs(date, level="datetime")["close"]
                prev_close_vals = close_df.xs(prev_date, level="datetime")["close"]
            except KeyError:
                daily_returns_with_cost.append(-cost)
                daily_returns_no_cost.append(0.0)
                continue

            w = 1.0 / len(portfolio)
            port_ret = 0.0
            valid_count = 0
            for stk in portfolio:
                if stk in today_close.index and stk in prev_close_vals.index:
                    if prev_close_vals[stk] > 0:
                        port_ret += w * (today_close[stk] / prev_close_vals[stk] - 1)
                        valid_count += 1
            if valid_count > 0 and valid_count < len(portfolio):
                port_ret *= len(portfolio) / valid_count
            daily_returns_with_cost.append(port_ret - cost)
            daily_returns_no_cost.append(port_ret)
        else:
            daily_returns_with_cost.append(-cost)
            daily_returns_no_cost.append(0.0)

    # Metrics
    idx = dates[:len(daily_returns_with_cost)]
    ret_wc = pd.Series(daily_returns_with_cost, index=idx)
    ret_nc = pd.Series(daily_returns_no_cost, index=idx)

    n_days = len(ret_wc)
    ann_factor = 252 / n_days if n_days > 0 else 1

    def _metrics(ret_s, label):
        cum = (1 + ret_s).cumprod()
        total_ret = float(cum.iloc[-1] - 1) if len(cum) else 0
        ann_ret = float((1 + total_ret) ** ann_factor - 1)

        # Benchmark - normalize index types for proper alignment
        bench_dt = bench_daily_ret.copy()
        bench_dt.index = pd.to_datetime(bench_dt.index)
        ret_dt = ret_s.copy()
        ret_dt.index = pd.to_datetime(ret_dt.index)

        bench_aligned = bench_dt.reindex(ret_dt.index).fillna(0)
        bench_cum = (1 + bench_aligned).cumprod()
        bench_total = float(bench_cum.iloc[-1] - 1) if len(bench_cum) else 0
        bench_ann = float((1 + bench_total) ** ann_factor - 1)

        excess_ann = ann_ret - bench_ann
        excess_daily = np.array(ret_dt.values, dtype=float) - np.array(bench_aligned.values, dtype=float)
        ir = float(np.nanmean(excess_daily) / (np.nanstd(excess_daily) + 1e-10) * np.sqrt(252))

        cummax = cum.cummax()
        max_dd = float(((cum - cummax) / cummax).min())

        return {
            f"excess_return_annualized_{label}": round(excess_ann, 4),
            f"information_ratio_{label}": round(ir, 4),
            f"max_drawdown_{label}": round(max_dd, 4),
            f"annualized_return_{label}": round(ann_ret, 4),
            "benchmark_annualized_return": round(bench_ann, 4),
            f"total_return_{label}": round(total_ret, 4),
        }

    result = {}
    result.update(_metrics(ret_wc, "with_cost"))
    result.update(_metrics(ret_nc, "no_cost"))

    rebal_turns = [t for t in daily_turnovers if t > 0]
    result["daily_avg_turnover"] = round(float(np.sum(daily_turnovers) / n_days) if n_days else 0, 4)
    result["rebal_avg_turnover"] = round(float(np.mean(rebal_turns)) if rebal_turns else 0, 4)
    result["rebal_count"] = len(rebal_turns)
    result["n_trading_days"] = n_days

    return result


def run_sweep(market, oos_start, oos_end):
    """Run parameter sweep over buffer_rank and topk."""
    predictions = train_and_predict(
        market=market, oos_start=oos_start, oos_end=oos_end,
    )

    configs = []
    for topk in [20, 30]:
        for buffer in [40, 60, 80, 100]:
            for rebal in [5]:
                configs.append({"topk": topk, "buffer_rank": buffer, "rebal_days": rebal})

    results = []
    for cfg in configs:
        label = f"topk={cfg['topk']},buf={cfg['buffer_rank']},rebal={cfg['rebal_days']}d"
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        r = simulate_weekly_rebal(
            predictions, market=market, oos_start=oos_start, oos_end=oos_end, **cfg,
        )
        r.update(cfg)
        r["label"] = label
        results.append(r)
        print(f"  Excess(w/c): {r['excess_return_annualized_with_cost']:+.2%}, "
              f"IR: {r['information_ratio_with_cost']:+.3f}, "
              f"Turn: {r['daily_avg_turnover']:.2%}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--buffer-rank", type=int, default=60)
    parser.add_argument("--rebal-days", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    init_qlib()

    market = "csi1000"
    oos_start = "2025-01-01"
    oos_end = "2026-03-18"

    if args.sweep:
        results = run_sweep(market, oos_start, oos_end)
        out = PROJECT_ROOT / "outputs" / "mfa_weekly_v7_sweep_results.json"
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        print(f"\nSweep results saved: {out}")
    else:
        predictions = train_and_predict(
            market=market, oos_start=oos_start, oos_end=oos_end,
        )

        result = simulate_weekly_rebal(
            predictions, market=market, topk=args.topk,
            buffer_rank=args.buffer_rank, rebal_days=args.rebal_days,
            oos_start=oos_start, oos_end=oos_end,
        )
        result.update({
            "strategy": "weekly_rebalance",
            "factor_pool": "mfa_factor_pool_v7_diverse",
            "market": market,
            "topk": args.topk,
            "buffer_rank": args.buffer_rank,
            "rebal_days": args.rebal_days,
            "label_horizon": 5,
            "model": "ensemble_xgb_lgb",
            "oos_period": f"{oos_start} ~ {oos_end}",
        })

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        for k, v in result.items():
            print(f"  {k}: {v}")

        out = PROJECT_ROOT / "outputs" / "mfa_weekly_v7_diverse_results.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print(f"\nSaved: {out}")

    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

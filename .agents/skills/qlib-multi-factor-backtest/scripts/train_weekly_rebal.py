"""Weekly-rebalance multi-factor backtest.

Uses 5d forward return as prediction target and rebalances every N trading days
instead of daily TopkDropout. Designed for weekly-level holding periods.

Key differences from daily TopkDropout:
- Label: 5d forward return (not 1d)
- Rebalance: every 5 trading days (weekly)
- Strategy: weight-based TopK with forced weekly refresh
- Lower turnover by design

Usage:
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/train_weekly_rebal.py --market csi1000
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/train_weekly_rebal.py --market csi1000 --topk 20 --rebal-days 5
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
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


def build_rolling_config(
    market: str,
    topk: int,
    n_drop: int,
    hold_thresh: int,
    rebal_days: int,
    label_horizon: int,
    train_start: str,
    oos_start: str,
    oos_end: str,
    model_type: str = "ensemble",
    max_per_cat: int = 5,
    top_n_factors: int = 30,
) -> dict:
    """Build configuration dict for weekly rebalance experiment."""
    return {
        "market": market,
        "topk": topk,
        "n_drop": n_drop,
        "hold_thresh": hold_thresh,
        "rebal_days": rebal_days,
        "label_horizon": label_horizon,
        "train_start": train_start,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "model_type": model_type,
        "max_per_cat": max_per_cat,
        "top_n_factors": top_n_factors,
        "cost": {"open": 0.0005, "close": 0.0015, "min_cost": 5},
    }


def get_factor_features(db_path: Path, market: str, top_n: int, max_per_cat: int):
    """Load top factors with category budget from DB."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT f.name, f.expression, f.category,
               ABS(COALESCE(t.rank_icir, 0)) as abs_icir
        FROM factors f
        JOIN factor_test_results t ON f.name = t.factor_name
        WHERE t.market = ? AND f.status IN ('Accepted', 'Baseline')
              AND f.expression IS NOT NULL AND f.expression != ''
        ORDER BY abs_icir DESC
    """, (market,)).fetchall()
    conn.close()

    seen_names = set()
    cat_count = {}
    selected = []
    for name, expr, cat, icir in rows:
        if name in seen_names:
            continue
        seen_names.add(name)
        cc = cat_count.get(cat, 0)
        if cc >= max_per_cat:
            continue
        cat_count[cat] = cc + 1
        selected.append({"name": name, "expression": expr, "category": cat, "icir": icir})
        if len(selected) >= top_n:
            break

    return selected


def run_weekly_rebal_backtest(config: dict) -> dict:
    """Run a weekly-rebalance backtest using qlib infrastructure."""
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158
    import qlib

    market = config["market"]
    topk = config["topk"]
    rebal_days = config["rebal_days"]
    label_horizon = config["label_horizon"]
    oos_start = config["oos_start"]
    oos_end = config["oos_end"]

    benchmark = "SH000852" if market == "csi1000" else "SH000300"

    # Build label expression for multi-day forward return
    if label_horizon == 1:
        label_expr = "Ref($close, -2)/Ref($close, -1) - 1"
    else:
        label_expr = f"Ref($close, -{label_horizon})/Ref($close, -1) - 1"

    print(f"\n--- Weekly Rebalance Backtest ---")
    print(f"  Market: {market}, Label: {label_horizon}d return")
    print(f"  Rebalance: every {rebal_days} trading days")
    print(f"  TopK: {topk}, Model: {config['model_type']}")
    print(f"  OOS: {oos_start} ~ {oos_end}")

    # Load factor features from DB
    db_path = PROJECT_ROOT / "data" / "factor_library.db"
    db_factors = get_factor_features(
        db_path, market, config["top_n_factors"], config["max_per_cat"]
    )
    print(f"  DB factors: {len(db_factors)} (max_per_cat={config['max_per_cat']})")

    # Get Alpha158 features
    h = Alpha158.__new__(Alpha158)
    a158_fields, a158_names = h.get_feature_config()
    print(f"  Alpha158 features: {len(a158_names)}")

    # Combine feature configs
    all_fields = list(a158_fields)
    all_names = list(a158_names)
    for f in db_factors:
        if f["name"] not in all_names:
            all_fields.append(f["expression"])
            all_names.append(f["name"])

    print(f"  Total features: {len(all_names)}")

    instruments = D.instruments(market)

    # Rolling windows (3-month quarterly retrain)
    from dateutil.relativedelta import relativedelta
    from datetime import datetime

    oos_dt = datetime.strptime(oos_start, "%Y-%m-%d")
    end_dt = datetime.strptime(oos_end, "%Y-%m-%d")
    train_start_dt = datetime.strptime(config["train_start"], "%Y-%m-%d")

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

    print(f"  Rolling windows: {len(windows)}")

    # Train and predict for each window
    all_predictions = []
    for wi, w in enumerate(windows):
        print(f"\n  Window {wi+1}/{len(windows)}: test={w['test_start']}~{w['test_end']}")

        # Load data
        train_data = D.features(
            instruments, all_fields,
            start_time=w["train_start"], end_time=w["train_end"]
        )
        train_data.columns = all_names

        valid_data = D.features(
            instruments, all_fields,
            start_time=w["valid_start"], end_time=w["valid_end"]
        )
        valid_data.columns = all_names

        test_data = D.features(
            instruments, all_fields,
            start_time=w["test_start"], end_time=w["test_end"]
        )
        test_data.columns = all_names

        # Load labels
        train_label = D.features(
            instruments, [label_expr],
            start_time=w["train_start"], end_time=w["train_end"]
        ).iloc[:, 0]

        valid_label = D.features(
            instruments, [label_expr],
            start_time=w["valid_start"], end_time=w["valid_end"]
        ).iloc[:, 0]

        # Align and clean
        common_idx = train_data.index.intersection(train_label.index)
        X_train = train_data.loc[common_idx].fillna(0)
        y_train = train_label.loc[common_idx].fillna(0)

        common_idx_v = valid_data.index.intersection(valid_label.index)
        X_valid = valid_data.loc[common_idx_v].fillna(0)
        y_valid = valid_label.loc[common_idx_v].fillna(0)

        X_test = test_data.fillna(0)

        if config["model_type"] == "ensemble":
            # XGBoost
            from xgboost import XGBRegressor
            xgb = XGBRegressor(
                n_estimators=1000, max_depth=8, learning_rate=0.05,
                colsample_bytree=0.8879, subsample=0.8789,
                reg_alpha=205.70, reg_lambda=580.98,
                tree_method="hist", n_jobs=-1, verbosity=0,
            )
            xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                     verbose=False)
            pred_xgb = xgb.predict(X_test)
            del xgb
            gc.collect()

            # LightGBM
            from lightgbm import LGBMRegressor
            lgb = LGBMRegressor(
                n_estimators=1000, max_depth=8, learning_rate=0.05,
                num_leaves=128, reg_alpha=205.70, reg_lambda=580.97,
                n_jobs=-1, verbosity=-1,
            )
            lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                     callbacks=[])
            pred_lgb = lgb.predict(X_test)
            del lgb
            gc.collect()

            # Ensemble mean
            pred = (pred_xgb + pred_lgb) / 2
        else:
            from lightgbm import LGBMRegressor
            lgb = LGBMRegressor(
                n_estimators=1000, max_depth=8, learning_rate=0.05,
                num_leaves=128, reg_alpha=205.70, reg_lambda=580.97,
                n_jobs=-1, verbosity=-1,
            )
            lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            pred = lgb.predict(X_test)
            del lgb
            gc.collect()

        pred_series = pd.Series(pred, index=X_test.index, name="score")
        all_predictions.append(pred_series)

        del train_data, valid_data, test_data, train_label, valid_label
        del X_train, y_train, X_valid, y_valid, X_test
        gc.collect()
        try:
            from qlib.data.cache import H
            H.clear()
        except Exception:
            pass

    # Merge all predictions
    predictions = pd.concat(all_predictions).sort_index()
    print(f"\nTotal predictions: {len(predictions)}")

    # Weekly rebalance simulation
    result = simulate_weekly_rebal(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        topk=topk,
        rebal_days=rebal_days,
        oos_start=oos_start,
        oos_end=oos_end,
        open_cost=config["cost"]["open"],
        close_cost=config["cost"]["close"],
        min_cost=config["cost"]["min_cost"],
    )

    return result


def simulate_weekly_rebal(
    predictions: pd.Series,
    market: str,
    benchmark: str,
    topk: int,
    rebal_days: int,
    oos_start: str,
    oos_end: str,
    open_cost: float = 0.0005,
    close_cost: float = 0.0015,
    min_cost: float = 5,
) -> dict:
    """Simulate weekly rebalancing portfolio.

    On every rebal_day-th trading day, select top-K stocks by predicted score.
    Hold equal-weight until next rebalance.
    """
    from qlib.data import D

    # Get daily close prices for return calculation
    instruments = D.instruments(market)
    close_df = D.features(instruments, ["$close"], start_time=oos_start, end_time=oos_end)
    close_df.columns = ["close"]

    # Get benchmark returns
    try:
        bench_df = D.features([benchmark], ["$close"], start_time=oos_start, end_time=oos_end)
        bench_returns = bench_df.iloc[:, 0].groupby(level=1).pct_change().groupby(level=1).mean()
    except Exception:
        bench_returns = None

    dates = sorted(predictions.index.get_level_values(1).unique())
    print(f"  Simulation: {len(dates)} trading days, rebal every {rebal_days}d")

    portfolio = {}  # stock -> weight
    daily_returns = []
    daily_turnovers = []
    rebal_count = 0

    for di, date in enumerate(dates):
        # Get today's predictions
        try:
            day_pred = predictions.xs(date, level=1)
        except KeyError:
            continue

        # Rebalance check
        is_rebal_day = (di % rebal_days == 0)

        if is_rebal_day and len(day_pred) >= topk:
            # Select top-K
            new_picks = day_pred.nlargest(topk).index.tolist()
            new_portfolio = {s: 1.0 / topk for s in new_picks}

            # Compute turnover
            old_stocks = set(portfolio.keys())
            new_stocks = set(new_portfolio.keys())
            sold = old_stocks - new_stocks
            bought = new_stocks - old_stocks

            turnover = 0
            for s in sold:
                turnover += portfolio.get(s, 0)
            for s in bought:
                turnover += new_portfolio.get(s, 0)

            # Apply transaction cost
            cost = turnover * (open_cost + close_cost)
            daily_turnovers.append(turnover)

            portfolio = new_portfolio
            rebal_count += 1
        else:
            cost = 0
            daily_turnovers.append(0)

        # Compute portfolio return for today
        if portfolio:
            try:
                day_close = close_df.xs(date, level=1)["close"]
            except KeyError:
                daily_returns.append(0)
                continue

            # Get previous day's close
            if di > 0:
                prev_date = dates[di - 1]
                try:
                    prev_close = close_df.xs(prev_date, level=1)["close"]
                except KeyError:
                    daily_returns.append(0)
                    continue

                port_ret = 0
                valid_weight = 0
                for stock, weight in portfolio.items():
                    if stock in day_close.index and stock in prev_close.index:
                        if prev_close[stock] > 0:
                            stock_ret = day_close[stock] / prev_close[stock] - 1
                            port_ret += weight * stock_ret
                            valid_weight += weight

                if valid_weight > 0:
                    port_ret = port_ret / valid_weight
                daily_returns.append(port_ret - cost)
            else:
                daily_returns.append(-cost if cost > 0 else 0)
        else:
            daily_returns.append(0)

    # Compute benchmark returns
    bench_close = D.features([benchmark], ["$close"], start_time=oos_start, end_time=oos_end)
    if len(bench_close) > 0:
        bench_daily = bench_close.iloc[:, 0].pct_change().dropna()
        bench_cum = (1 + bench_daily).cumprod()
    else:
        bench_daily = pd.Series(0, index=range(len(daily_returns)))
        bench_cum = None

    ret_series = pd.Series(daily_returns, index=dates[:len(daily_returns)])
    cum = (1 + ret_series).cumprod()

    # Excess returns (approximate)
    total_days = len(ret_series)
    ann_factor = 252 / total_days if total_days > 0 else 1
    total_ret = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0
    ann_ret = float((1 + total_ret) ** ann_factor - 1)

    # Benchmark annualized return
    if bench_cum is not None and len(bench_cum) > 0:
        bench_total = float(bench_cum.iloc[-1] - 1)
        bench_ann = float((1 + bench_total) ** ann_factor - 1)
    else:
        bench_ann = 0

    excess_ann = ann_ret - bench_ann

    # Daily excess returns for IR calculation
    excess_daily = ret_series.values - bench_daily.values[:len(ret_series)] if bench_daily is not None and len(bench_daily) >= len(ret_series) else ret_series.values
    ir = float(np.mean(excess_daily) / (np.std(excess_daily) + 1e-10) * np.sqrt(252))

    # Max drawdown
    cummax = cum.cummax()
    drawdown = (cum - cummax) / cummax
    max_dd = float(drawdown.min())

    # Average turnover
    rebal_turnovers = [t for t in daily_turnovers if t > 0]
    avg_turnover = float(np.mean(rebal_turnovers)) if rebal_turnovers else 0
    daily_avg_turnover = float(np.sum(daily_turnovers) / total_days) if total_days > 0 else 0

    result = {
        "excess_return_annualized_with_cost": round(excess_ann, 4),
        "information_ratio_with_cost": round(ir, 4),
        "max_drawdown_with_cost": round(max_dd, 4),
        "annualized_return": round(ann_ret, 4),
        "benchmark_annualized_return": round(bench_ann, 4),
        "total_return": round(total_ret, 4),
        "daily_avg_turnover": round(daily_avg_turnover, 4),
        "rebal_avg_turnover": round(avg_turnover, 4),
        "rebal_count": rebal_count,
        "total_days": total_days,
        "topk": topk,
        "rebal_days": rebal_days,
    }

    print(f"\n  Results:")
    print(f"    Ann. excess (w/c): {excess_ann:+.2%}")
    print(f"    IR (w/c): {ir:+.3f}")
    print(f"    Max DD: {max_dd:.2%}")
    print(f"    Daily avg turnover: {daily_avg_turnover:.2%}")
    print(f"    Rebalances: {rebal_count}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Weekly rebalance MFA backtest")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--rebal-days", type=int, default=5, help="Rebalance every N trading days")
    parser.add_argument("--label-horizon", type=int, default=5, help="Forward return horizon for label")
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--oos-start", default="2025-01-01")
    parser.add_argument("--oos-end", default="2026-03-18")
    parser.add_argument("--model", default="ensemble", choices=["ensemble", "lgb"])
    parser.add_argument("--n-factors", type=int, default=30)
    parser.add_argument("--max-per-cat", type=int, default=5)
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    start_time = time.time()
    init_qlib()

    if args.sweep:
        # Parameter sweep: rebal_days × topk × label_horizon
        configs = []
        for rebal in [3, 5, 10]:
            for topk in [15, 20, 30]:
                for label_h in [3, 5, 10]:
                    configs.append(build_rolling_config(
                        market=args.market, topk=topk, n_drop=0, hold_thresh=0,
                        rebal_days=rebal, label_horizon=label_h,
                        train_start=args.train_start, oos_start=args.oos_start,
                        oos_end=args.oos_end, model_type=args.model,
                        max_per_cat=args.max_per_cat, top_n_factors=args.n_factors,
                    ))

        results = []
        for ci, cfg in enumerate(configs):
            print(f"\n{'='*80}")
            print(f"Config {ci+1}/{len(configs)}: topk={cfg['topk']}, "
                  f"rebal={cfg['rebal_days']}d, label={cfg['label_horizon']}d")
            try:
                res = run_weekly_rebal_backtest(cfg)
                res["config"] = cfg
                results.append(res)
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"config": cfg, "error": str(e)})

        # Save sweep results
        out_path = PROJECT_ROOT / "outputs" / f"mfa_weekly_sweep_{args.market}.json"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        print(f"\nSaved sweep: {out_path}")

        # Summary table
        print(f"\n{'='*100}")
        print("SWEEP SUMMARY (sorted by IR)")
        print(f"{'='*100}")
        valid = [r for r in results if "error" not in r]
        valid.sort(key=lambda x: x.get("information_ratio_with_cost", -999), reverse=True)
        for r in valid:
            c = r["config"]
            print(f"  topk={c['topk']:>2}, rebal={c['rebal_days']:>2}d, label={c['label_horizon']:>2}d | "
                  f"IR={r['information_ratio_with_cost']:+.3f}, "
                  f"Ret={r['excess_return_annualized_with_cost']:+.2%}, "
                  f"DD={r['max_drawdown_with_cost']:.2%}, "
                  f"Turn={r['daily_avg_turnover']:.2%}")
    else:
        config = build_rolling_config(
            market=args.market, topk=args.topk, n_drop=0, hold_thresh=0,
            rebal_days=args.rebal_days, label_horizon=args.label_horizon,
            train_start=args.train_start, oos_start=args.oos_start,
            oos_end=args.oos_end, model_type=args.model,
            max_per_cat=args.max_per_cat, top_n_factors=args.n_factors,
        )
        result = run_weekly_rebal_backtest(config)
        result["config"] = config

        out_path = PROJECT_ROOT / "outputs" / f"mfa_weekly_{args.market}_topk{args.topk}_rebal{args.rebal_days}d.json"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print(f"\nSaved: {out_path}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()

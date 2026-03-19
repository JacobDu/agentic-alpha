"""MFA V7 follow-up: true buffer exit variants."""
from __future__ import annotations

import copy
import gc
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RESULT_FILE = OUT / "mfa_v7_buffer_variants.json"

MARKET = "csi1000"
BENCHMARK = "SH000852"
ACCOUNT = 1e8
TRAIN_START = "2018-01-01"
OOS_START = "2025-01-01"
OOS_END = "2026-03-18"  # Latest available data
TOPK = 20
N_DROP = 2
GLOBAL_SEED = 3407
SUBPERIODS = {
    "2025H1": ("2025-01-01", "2025-06-30"),
    "2025H2": ("2025-07-01", "2025-12-31"),
    "2026YTD": ("2026-01-01", "2026-03-18"),
}
EXCHANGE = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}


def set_global_seed(seed=GLOBAL_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def init_qlib():
    import qlib

    try:
        qlib.init(provider_uri=str(ROOT / "data/qlib/cn_data"), region="cn")
    except Exception:
        pass


def make_label(days: int):
    horizon = days + 1
    return ([f"Ref($close, -{horizon}) / Ref($close, -1) - 1"], ["LABEL0"])


def create_dataset(train, valid, test, label_days=1, topn=30, max_per_cat=5):
    from qlib.data.dataset import DatasetH
    from project_qlib.factors.topn_db import DBAlpha158PlusTopN

    class Handler(DBAlpha158PlusTopN):
        TOPN = topn
        MARKET = MARKET
        MAX_PER_CAT = max_per_cat

    handler = Handler(
        instruments=MARKET,
        start_time=train[0],
        end_time=test[1],
        fit_start_time=train[0],
        fit_end_time=train[1],
        label=make_label(label_days),
    )
    return DatasetH(handler=handler, segments={"train": train, "valid": valid, "test": test})


def train_xgb(ds):
    from qlib.contrib.model.xgboost import XGBModel

    params = dict(
        objective="reg:squarederror",
        max_depth=8,
        eta=0.05,
        colsample_bytree=0.8879,
        subsample=0.8789,
        alpha=205.6999,
        reg_lambda=580.9768,
        nthread=8,
        seed=GLOBAL_SEED,
        random_state=GLOBAL_SEED,
    )
    model = XGBModel(**params)
    model.fit(ds, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=0)
    return model


def train_lgb(ds):
    from qlib.contrib.model.gbdt import LGBModel

    params = dict(
        loss="mse",
        colsample_bytree=0.8879,
        learning_rate=0.05,
        subsample=0.8789,
        lambda_l1=205.6999,
        lambda_l2=580.9768,
        max_depth=8,
        num_leaves=128,
        num_threads=8,
        n_estimators=1000,
        early_stopping_rounds=50,
        seed=GLOBAL_SEED,
        feature_fraction_seed=GLOBAL_SEED,
        bagging_seed=GLOBAL_SEED,
        data_random_seed=GLOBAL_SEED,
        deterministic=True,
        force_col_wise=True,
    )
    model = LGBModel(**params)
    model.fit(ds)
    return model


def predict(model, ds):
    pred = model.predict(ds)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred.name = "score"
    return pred


def clear_qlib_cache():
    try:
        from qlib.data.cache import H

        H.clear()
    except Exception:
        pass
    gc.collect()


def rolling_retrain_predict():
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    set_global_seed(GLOBAL_SEED)

    oos_dt = datetime.strptime(OOS_START, "%Y-%m-%d")
    oos_end_dt = datetime.strptime(OOS_END, "%Y-%m-%d")
    retrain_points = []
    point = oos_dt
    while point < oos_end_dt:
        retrain_points.append(point)
        point = point + relativedelta(months=3)
    retrain_points.append(oos_end_dt)

    preds = []
    for idx in range(len(retrain_points) - 1):
        window_start = retrain_points[idx]
        window_end = retrain_points[idx + 1]
        valid_start = window_start - relativedelta(years=1)
        train_end = valid_start - relativedelta(days=1)

        train_seg = (TRAIN_START, train_end.strftime("%Y-%m-%d"))
        valid_seg = (
            valid_start.strftime("%Y-%m-%d"),
            (window_start - relativedelta(days=1)).strftime("%Y-%m-%d"),
        )
        test_seg = (window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"))
        print(
            f"W{idx + 1}: train~{train_seg[1]} valid={valid_seg[0]}~{valid_seg[1]} "
            f"test={test_seg[0]}~{test_seg[1]}"
        )

        ds = create_dataset(train_seg, valid_seg, test_seg, label_days=1, topn=30, max_per_cat=5)
        lgb = train_lgb(ds)
        pred_lgb = predict(lgb, ds)
        del lgb
        gc.collect()

        xgb = train_xgb(ds)
        pred_xgb = predict(xgb, ds)
        del xgb, ds

        pred = pd.concat([pred_lgb, pred_xgb], axis=1).dropna().mean(axis=1)
        pred.name = "score"
        preds.append(pred)
        del pred_lgb, pred_xgb, pred
        clear_qlib_cache()

    combined = pd.concat(preds)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.name = "score"
    return combined


def _get_first_n(items, n):
    return list(items)[:n]


class TrueBufferExitStrategy(BaseSignalStrategy):
    """Sell only holdings that themselves breach the buffer rank."""

    def __init__(
        self,
        *,
        signal,
        topk,
        n_drop,
        buffer_rank,
        min_hold,
        risk_degree=0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
    ):
        super().__init__(
            signal=signal,
            risk_degree=risk_degree,
            trade_exchange=trade_exchange,
            level_infra=level_infra,
            common_infra=common_infra,
        )
        self.topk = topk
        self.n_drop = n_drop
        self.buffer_rank = buffer_rank
        self.min_hold = min_hold

    def generate_trade_decision(self, execute_result=None):
        from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
        from qlib.backtest.position import Position

        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        current_temp: Position = copy.deepcopy(self.trade_position)
        current_stock_list = current_temp.get_stock_list()
        cash = current_temp.get_cash()

        score_rank = pred_score.sort_values(ascending=False)
        rank_map = {code: idx + 1 for idx, code in enumerate(score_rank.index)}

        breached = []
        time_per_step = self.trade_calendar.get_freq()
        for code in current_stock_list:
            rank = rank_map.get(code, len(score_rank) + 1)
            hold_count = current_temp.get_stock_count(code, bar=time_per_step)
            if rank > self.buffer_rank and hold_count >= self.min_hold:
                breached.append((code, rank, pred_score.get(code, -np.inf)))

        breached.sort(key=lambda item: (item[2], item[1]))
        sell = [code for code, _, _ in breached[: self.n_drop]]

        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        candidate_buy = _get_first_n(
            pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
            len(sell) + self.topk - len(last),
        )
        buy = candidate_buy[: len(sell) + self.topk - len(last)]

        sell_orders = []
        buy_orders = []
        for code in current_stock_list:
            if code not in sell:
                continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
            ):
                continue
            sell_amount = current_temp.get_stock_amount(code=code)
            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if self.trade_exchange.check_order(sell_order):
                sell_orders.append(sell_order)
                trade_val, trade_cost, _ = self.trade_exchange.deal_order(sell_order, position=current_temp)
                cash += trade_val - trade_cost

        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0
        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            ):
                continue
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_orders.append(
                Order(
                    stock_id=code,
                    amount=buy_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.BUY,
                )
            )
        return TradeDecisionWO(sell_orders + buy_orders, self)


def backtest_strategy(strategy, start_time, end_time):
    from qlib.contrib.evaluate import backtest_daily, risk_analysis

    report, _ = backtest_daily(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        benchmark=BENCHMARK,
        account=ACCOUNT,
        exchange_kwargs=EXCHANGE,
    )
    ex_wc = report["return"] - report["bench"] - report["cost"]
    risk = risk_analysis(ex_wc, freq="day")["risk"]
    return {
        "ann_wc": round(float(risk.loc["annualized_return"]), 6),
        "ir_wc": round(float(risk.loc["information_ratio"]), 4),
        "max_dd_wc": round(float(risk.loc["max_drawdown"]), 6),
        "turnover": round(float(report["turnover"].mean()), 6),
        "cum_excess_wc_pct": round(float((1 + ex_wc).prod() - 1) * 100, 2),
    }


def make_plain_topk(signal, hold_thresh):
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    return TopkDropoutStrategy(signal=signal, topk=TOPK, n_drop=N_DROP, hold_thresh=hold_thresh)


def make_topk_custom(signal, topk, n_drop, hold_thresh):
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    return TopkDropoutStrategy(signal=signal, topk=topk, n_drop=n_drop, hold_thresh=hold_thresh)


def run_one(name, strategy):
    result = {
        "exp": name,
        "overall_oos": backtest_strategy(strategy, OOS_START, OOS_END),
        "subperiods": {},
    }
    for label, (start_time, end_time) in SUBPERIODS.items():
        result["subperiods"][label] = backtest_strategy(strategy, start_time, end_time)
    return result


def main():
    t0 = time.time()
    set_global_seed(GLOBAL_SEED)
    init_qlib()
    pred = rolling_retrain_predict()

    experiments = []
    # Plain TopkDropout sweep with short hold (user requirement: ~5 days effective)
    for hold in [5, 10, 15, 20, 25, 30, 35]:
        experiments.append((f"plain_h{hold}", make_plain_topk(pred, hold)))
    
    # Sweep n_drop with best hold candidates
    for hold in [20, 25, 30]:
        for n_drop in [1, 2, 3]:
            experiments.append(
                (f"plain_h{hold}_d{n_drop}", make_topk_custom(pred, TOPK, n_drop, hold))
            )
    
    # Test different topk with short hold
    for topk in [15, 20, 25, 30]:
        experiments.append(
            (f"tk{topk}_h25_d2", make_topk_custom(pred, topk, 2, 25))
        )

    results = []
    for name, strategy in experiments:
        print(f"\n=== {name} ===")
        results.append(run_one(name, strategy))

    results.sort(key=lambda row: row["overall_oos"]["ir_wc"], reverse=True)
    RESULT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved to {RESULT_FILE}")
    print(f"elapsed_sec={time.time() - t0:.1f}")
    for row in results:
        overall = row["overall_oos"]
        print(row["exp"], overall["ann_wc"], overall["ir_wc"], overall["max_dd_wc"], overall["turnover"])


if __name__ == "__main__":
    main()

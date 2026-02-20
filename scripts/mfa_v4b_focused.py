"""临时脚本：MFA V4b - 精简版综合管线 (OOS 2025-2026)
基于 V4 log 中已有的 A 组 static 数据，聚焦 Rolling Retrain + Label 工程。
每完成一个实验即增量保存，防止中断丢失。
"""
from __future__ import annotations
import gc, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RESULT_FILE = OUT / "mfa_v4b_results.json"

MARKET = "csi1000"
BENCHMARK = "SH000852"
ACCOUNT = 1e8
EXCHANGE = {
    "limit_threshold": 0.095, "deal_price": "close",
    "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5,
}


def init_qlib():
    import qlib
    try:
        qlib.init(provider_uri=str(ROOT / "data/qlib/cn_data"), region="cn")
    except:
        pass


def load_results():
    if RESULT_FILE.exists():
        return json.load(open(RESULT_FILE))
    return []


def save_results(results):
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def already_done(results, name):
    return any(r.get("exp") == name for r in results)


def make_label(days):
    d = days + 1
    return ([f"Ref($close, -{d}) / Ref($close, -1) - 1"], ["LABEL0"])


def create_dataset(train, valid, test, label_days=1, topn=30, max_per_cat=5):
    from qlib.data.dataset import DatasetH
    from project_qlib.factors.topn_db import DBAlpha158PlusTopN

    class Handler(DBAlpha158PlusTopN):
        TOPN = topn
        MARKET = MARKET
        MAX_PER_CAT = max_per_cat

    handler = Handler(
        instruments=MARKET,
        start_time=train[0], end_time=test[1],
        fit_start_time=train[0], fit_end_time=train[1],
        label=make_label(label_days),
    )
    return DatasetH(handler=handler,
                    segments={"train": train, "valid": valid, "test": test})


def train_xgb(ds, **kw):
    from qlib.contrib.model.xgboost import XGBModel
    params = dict(
        objective="reg:squarederror", max_depth=8, eta=0.05,
        colsample_bytree=0.8879, subsample=0.8789,
        alpha=205.6999, reg_lambda=580.9768, nthread=8,
    )
    params.update(kw)
    model = XGBModel(**params)
    model.fit(ds, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=0)
    return model


def train_lgb(ds):
    from qlib.contrib.model.gbdt import LGBModel
    params = dict(
        loss="mse", colsample_bytree=0.8879, learning_rate=0.05,
        subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768,
        max_depth=8, num_leaves=128, num_threads=8,
        n_estimators=1000, early_stopping_rounds=50,
    )
    model = LGBModel(**params)
    model.fit(ds)
    return model


def predict(model, ds):
    p = model.predict(ds)
    if isinstance(p, pd.DataFrame):
        p = p.iloc[:, 0]
    p.name = "score"
    return p


def backtest(pred, topk=50, n_drop=5, hold=40, test_start=None, test_end=None):
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    strat = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": pred, "topk": topk, "n_drop": n_drop, "hold_thresh": hold},
    }
    report, _ = backtest_daily(
        start_time=test_start, end_time=test_end,
        strategy=strat, benchmark=BENCHMARK, account=ACCOUNT, exchange_kwargs=EXCHANGE,
    )
    ex_nc = report["return"] - report["bench"]
    ex_wc = report["return"] - report["bench"] - report["cost"]
    from qlib.contrib.evaluate import risk_analysis
    anc = risk_analysis(ex_nc, freq="day")
    awc = risk_analysis(ex_wc, freq="day")

    def _e(a, lab):
        r = a["risk"]
        return {
            f"ann_ret_{lab}": round(float(r.loc["annualized_return"]), 6),
            f"IR_{lab}": round(float(r.loc["information_ratio"]), 4),
            f"max_dd_{lab}": round(float(r.loc["max_drawdown"]), 6),
        }

    m = {}
    m.update(_e(anc, "no_cost"))
    m.update(_e(awc, "with_cost"))
    m["daily_turnover"] = round(float(report["turnover"].mean()), 6) if "turnover" in report.columns else None
    return m


def rolling_retrain_predict(
    train_start, oos_start, oos_end,
    retrain_freq_months=3,
    label_days=1, topn=30, max_per_cat=5,
    model_type="xgb",
):
    from dateutil.relativedelta import relativedelta
    from datetime import datetime

    oos_dt = datetime.strptime(oos_start, "%Y-%m-%d")
    oos_end_dt = datetime.strptime(oos_end, "%Y-%m-%d")

    retrain_points = []
    point = oos_dt
    while point < oos_end_dt:
        retrain_points.append(point)
        point = point + relativedelta(months=retrain_freq_months)
    retrain_points.append(oos_end_dt)

    print(f"    Rolling: {len(retrain_points)-1} windows, freq={retrain_freq_months}m")

    all_preds = []
    for i in range(len(retrain_points) - 1):
        window_start = retrain_points[i]
        window_end = retrain_points[i + 1]

        valid_start = window_start - relativedelta(years=1)
        train_end = valid_start - relativedelta(days=1)

        tr = (train_start, train_end.strftime("%Y-%m-%d"))
        va = (valid_start.strftime("%Y-%m-%d"), (window_start - relativedelta(days=1)).strftime("%Y-%m-%d"))
        te = (window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"))

        print(f"      W{i+1}: train~{tr[1]} valid={va[0]}~{va[1]} test={te[0]}~{te[1]}")

        try:
            ds = create_dataset(tr, va, te, label_days=label_days, topn=topn, max_per_cat=max_per_cat)

            if model_type == "xgb":
                model = train_xgb(ds)
            elif model_type == "lgb":
                model = train_lgb(ds)
            elif model_type == "ensemble":
                m1 = train_lgb(ds)
                p1 = predict(m1, ds)
                m2 = train_xgb(ds)
                p2 = predict(m2, ds)
                idx = p1.index.intersection(p2.index)
                p = (p1.loc[idx] + p2.loc[idx]) / 2
                p.name = "score"
                all_preds.append(p)
                del m1, m2, p1, p2, ds
                gc.collect()
                continue

            p = predict(model, ds)
            all_preds.append(p)
            del model, ds
            gc.collect()
        except Exception as e:
            print(f"      W{i+1} ERROR: {e}")
            continue

    if not all_preds:
        return None
    combined = pd.concat(all_preds)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined.name = "score"
    return combined


def run_one(name, mode, results, **kw):
    """Run a single experiment and incrementally save."""
    if already_done(results, name):
        print(f"  [SKIP] {name}")
        return results

    print(f"  [{mode.upper()}] {name}")
    t0 = time.time()

    try:
        if mode == "static":
            ds = create_dataset(
                kw["train"], kw["valid"], kw["test"],
                label_days=kw.get("label_days", 1),
                topn=kw.get("topn", 30),
                max_per_cat=kw.get("mpc", 5),
            )
            if kw.get("model", "xgb") == "xgb":
                model = train_xgb(ds)
            else:
                model = train_lgb(ds)
            pred = predict(model, ds)
            del model

            bt = backtest(pred, topk=kw["topk"], n_drop=kw.get("n_drop", 5),
                          hold=kw["hold"], test_start=kw["test"][0], test_end=kw["test"][1])
            del ds, pred
            gc.collect()

            r = {"exp": name, "mode": "static", "model": kw.get("model", "xgb"),
                 "topn": kw.get("topn", 30), "max_per_cat": kw.get("mpc", 5),
                 "label_days": kw.get("label_days", 1),
                 "topk": kw["topk"], "n_drop": kw.get("n_drop", 5), "hold_thresh": kw["hold"],
                 "train": f"{kw['train'][0]}~{kw['train'][1]}",
                 "test": f"{kw['test'][0]}~{kw['test'][1]}",
                 **bt, "elapsed": round(time.time() - t0, 1)}

        elif mode == "rolling":
            pred = rolling_retrain_predict(
                kw["train_start"], kw["oos_start"], kw["oos_end"],
                retrain_freq_months=kw.get("freq", 3),
                label_days=kw.get("label_days", 1),
                topn=kw.get("topn", 30),
                max_per_cat=kw.get("mpc", 5),
                model_type=kw.get("model", "xgb"),
            )
            if pred is None:
                results.append({"exp": name, "error": "no predictions"})
                save_results(results)
                return results

            bt = backtest(pred, topk=kw["topk"], n_drop=kw.get("n_drop", 5),
                          hold=kw["hold"], test_start=kw["oos_start"], test_end=kw["oos_end"])
            del pred
            gc.collect()

            r = {"exp": name, "mode": "rolling", "retrain_freq": kw.get("freq", 3),
                 "model": kw.get("model", "xgb"),
                 "topn": kw.get("topn", 30), "max_per_cat": kw.get("mpc", 5),
                 "label_days": kw.get("label_days", 1),
                 "topk": kw["topk"], "n_drop": kw.get("n_drop", 5), "hold_thresh": kw["hold"],
                 "oos": f"{kw['oos_start']}~{kw['oos_end']}",
                 **bt, "elapsed": round(time.time() - t0, 1)}

        ret = r.get("ann_ret_with_cost", 0)
        ir = r.get("IR_with_cost", 0)
        dd = r.get("max_dd_with_cost", 0)
        marker = " ★★★" if ret > 0.5 else " ★★" if ret > 0.3 else " ★" if ret > 0.2 else " ★" if ret > 0.1 else ""
        print(f"    IR={ir:+.3f} Ret={ret*100:+7.2f}% DD={dd*100:+6.2f}% [{time.time()-t0:.0f}s]{marker}")

        results.append(r)
        save_results(results)

    except Exception as e:
        print(f"    ERROR: {e}")
        results.append({"exp": name, "error": str(e)})
        save_results(results)

    return results


def main():
    init_qlib()
    results = load_results()
    print(f"Loaded {len(results)} existing results")

    OOS = ("2025-01-01", "2026-02-13")
    TRN = ("2018-01-01", "2023-12-31")
    VLD = ("2024-01-01", "2024-12-31")

    print("\n" + "=" * 100)
    print("MFA V4b: Focused Comprehensive Pipeline")
    print("  OOS: 2025-01-01 ~ 2026-02-13 | Rolling + Label + Concentrated + Ensemble")
    print("=" * 100)

    # =======================================================
    # A. Static Baselines — top 4 configs from V3b/V3d
    # =======================================================
    print("\n--- A. STATIC BASELINES ---")
    for topk, hold in [(50, 40), (30, 60), (50, 60), (30, 40)]:
        name = f"A_static_xgb_n30_mpc5_tk{topk}_h{hold}"
        results = run_one(name, "static", results,
                          train=TRN, valid=VLD, test=OOS, topk=topk, hold=hold)

    # Static with label engineering
    print("\n--- A2. Static + Label Engineering ---")
    for label in [2, 5]:
        for topk, hold in [(50, 40), (30, 60)]:
            name = f"A2_label{label}d_xgb_n30_mpc5_tk{topk}_h{hold}"
            results = run_one(name, "static", results,
                              train=TRN, valid=VLD, test=OOS, topk=topk, hold=hold, label_days=label)

    # =======================================================
    # B. Rolling Retrain — core experiments
    # =======================================================
    print("\n--- B. ROLLING RETRAIN (XGB) ---")
    for freq in [3, 6]:
        for topk, hold in [(50, 40), (30, 60), (50, 60)]:
            name = f"B_roll{freq}m_xgb_n30_mpc5_tk{topk}_h{hold}"
            results = run_one(name, "rolling", results,
                              train_start="2018-01-01", oos_start=OOS[0], oos_end=OOS[1],
                              freq=freq, topk=topk, hold=hold)

    # =======================================================
    # B2. Rolling + Label Engineering
    # =======================================================
    print("\n--- B2. ROLLING + LABEL ---")
    for label in [2, 5]:
        for topk, hold in [(50, 40), (30, 60)]:
            name = f"B2_roll3m_l{label}d_xgb_tk{topk}_h{hold}"
            results = run_one(name, "rolling", results,
                              train_start="2018-01-01", oos_start=OOS[0], oos_end=OOS[1],
                              freq=3, topk=topk, hold=hold, label_days=label)

    # =======================================================
    # C. Ensemble Rolling (XGB+LGB avg)
    # =======================================================
    print("\n--- C. ENSEMBLE ROLLING ---")
    for topk, hold in [(50, 40), (30, 60)]:
        name = f"C_roll3m_ens_tk{topk}_h{hold}"
        results = run_one(name, "rolling", results,
                          train_start="2018-01-01", oos_start=OOS[0], oos_end=OOS[1],
                          freq=3, topk=topk, hold=hold, model="ensemble")

    # =======================================================
    # D. Concentrated Portfolio Rolling
    # =======================================================
    print("\n--- D. CONCENTRATED ROLLING ---")
    for topk, n_drop in [(25, 3), (20, 3)]:
        for hold in [40, 60]:
            name = f"D_roll3m_xgb_tk{topk}_d{n_drop}_h{hold}"
            results = run_one(name, "rolling", results,
                              train_start="2018-01-01", oos_start=OOS[0], oos_end=OOS[1],
                              freq=3, topk=topk, n_drop=n_drop, hold=hold)

    # =======================================================
    # E. Longer Train (from 2010)
    # =======================================================
    print("\n--- E. LONGER TRAIN (from 2010) ---")
    for topk, hold in [(50, 40), (30, 60)]:
        name = f"E_long2010_roll3m_xgb_tk{topk}_h{hold}"
        results = run_one(name, "rolling", results,
                          train_start="2010-01-01", oos_start=OOS[0], oos_end=OOS[1],
                          freq=3, topk=topk, hold=hold)

    # =======================================================
    # F. More Factors
    # =======================================================
    print("\n--- F. MORE FACTORS + ROLLING ---")
    for topn, mpc in [(40, 5), (50, 7)]:
        name = f"F_roll3m_xgb_n{topn}_mpc{mpc}_tk50_h40"
        results = run_one(name, "rolling", results,
                          train_start="2018-01-01", oos_start=OOS[0], oos_end=OOS[1],
                          freq=3, topn=topn, mpc=mpc, topk=50, hold=40)

    # =======================================================
    # Summary
    # =======================================================
    valid = [r for r in results if "error" not in r]
    print(f"\n{'=' * 110}")
    print(f"FINAL RANKING: {len(valid)} valid experiments")
    print(f"{'=' * 110}")

    valid.sort(key=lambda x: x.get("ann_ret_with_cost", -999), reverse=True)
    print(f"{'#':>3} {'Experiment':<52} {'Mode':>7} {'TK':>3} {'H':>3} "
          f"{'IR':>7} {'Ret%':>8} {'DD%':>7}")
    print("-" * 100)
    for i, r in enumerate(valid, 1):
        ret = r.get("ann_ret_with_cost", 0) * 100
        ir = r.get("IR_with_cost", 0)
        dd = r.get("max_dd_with_cost", 0) * 100
        mode = r.get("mode", "?")
        print(f"{i:3d} {r['exp']:<52} {mode:>7} {r['topk']:3d} {r['hold_thresh']:3d} "
              f"{ir:+7.3f} {ret:+7.2f}% {dd:+6.2f}%")

    # Group comparison
    print(f"\n--- MODE COMPARISON ---")
    for m in ["static", "rolling"]:
        grp = [r for r in valid if r.get("mode") == m]
        if grp:
            rets = [r.get("ann_ret_with_cost", 0) * 100 for r in grp]
            print(f"  {m:>8}: n={len(grp):2d} avg={sum(rets)/len(rets):+.2f}% best={max(rets):+.2f}%")

    best = valid[0] if valid else None
    if best:
        print(f"\nBEST: {best['exp']}")
        print(f"  Ret={best['ann_ret_with_cost']*100:+.2f}% IR={best['IR_with_cost']:+.3f} DD={best['max_dd_with_cost']*100:.2f}%")


if __name__ == "__main__":
    main()

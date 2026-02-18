"""Comprehensive strategy optimization for CSI1000 TopN30.

Systematically tests all optimization dimensions:
  Phase 1: Rebalance frequency (hold_thresh=1/5/10/20) — backtest-only, shared model
  Phase 2: Label horizon alignment (1d/2d/5d/10d returns) — requires retraining
  Phase 3: TopK/n_drop tuning — backtest-only, shared model
  Phase 4: Loss function (mse/huber/mae) — requires retraining
  Phase 5: Model ensemble (LightGBM + XGBoost average)

Uses Qlib Python API directly for flexibility (train once, backtest many).

Usage:
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_optimization.py                    # all phases
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_optimization.py --phase 1          # only Phase 1
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_optimization.py --phase 1,2        # Phase 1 + 2
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_optimization.py --phase 1 --quick  # fast mode (fewer combos)
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Cannot locate project root (pyproject.toml not found)")

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Shared Constants ─────────────────────────────────────────────────────────

MARKET = "csi1000"
BENCHMARK = "SH000852"
TRAIN_SEG = ("2018-01-01", "2022-12-31")
VALID_SEG = ("2023-01-01", "2023-12-31")
TEST_SEG = ("2024-01-01", "2024-12-31")
ACCOUNT = 1e8
TOPN = 30

EXCHANGE_KWARGS = {
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}

LGB_PARAMS = dict(
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
)


# ─── Helper Functions ─────────────────────────────────────────────────────────


def init_qlib_once():
    """Initialize Qlib (safe to call multiple times)."""
    import qlib
    provider_uri = str((PROJECT_ROOT / "data" / "qlib" / "cn_data").resolve())
    try:
        qlib.init(provider_uri=provider_uri, region="cn")
    except Exception:
        # Already initialized
        pass


def make_label_expr(days: int) -> tuple[list[str], list[str]]:
    """Create label expression for N-day forward return.

    Label represents the return from T+1 close to T+1+days close.
    In Qlib's Ref convention (negative = future):
      - Buy at T+1 close = Ref($close, -1)
      - Sell at T+1+days close = Ref($close, -(1+days))
      - Return = Ref($close, -(1+days)) / Ref($close, -1) - 1
    """
    d = days + 1  # offset for sell day
    expr = f"Ref($close, -{d}) / Ref($close, -1) - 1"
    return ([expr], ["LABEL0"])


def create_dataset(label_days: int = 1, topn: int = TOPN):
    """Create DatasetH with DBTopN handler and custom label."""
    from project_qlib.factors.topn_db import DBTopNBase
    from qlib.data.dataset import DatasetH

    # Create a dynamic handler class
    class CustomTopN(DBTopNBase):
        TOPN = topn
        MARKET = MARKET

    label = make_label_expr(label_days)
    handler = CustomTopN(
        instruments=MARKET,
        start_time=TRAIN_SEG[0],
        end_time=TEST_SEG[1],
        fit_start_time=TRAIN_SEG[0],
        fit_end_time=TRAIN_SEG[1],
        label=label,
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": TRAIN_SEG,
            "valid": VALID_SEG,
            "test": TEST_SEG,
        },
    )
    return dataset


def train_lgb(dataset, loss: str = "mse", **extra_params) -> object:
    """Train LightGBM model and return it.

    Note: Qlib's LGBModel only accepts loss in {"mse", "binary"}.
    For other objectives (huber, mae, etc.), we construct with mse
    then override the objective in params before fitting.
    """
    from qlib.contrib.model.gbdt import LGBModel

    params = dict(LGB_PARAMS)
    actual_loss = loss
    # Map user-friendly names to LightGBM objective names
    loss_map = {
        "mse": "mse",
        "huber": "huber",
        "mae": "regression_l1",
        "fair": "fair",
    }
    lgb_objective = loss_map.get(loss, loss)

    # LGBModel only accepts "mse" or "binary"
    params["loss"] = "mse"
    params.update(extra_params)

    model = LGBModel(**params)
    # Override objective after construction if needed
    if lgb_objective != "mse":
        model.params["objective"] = lgb_objective
    model.fit(dataset)
    return model


def train_xgb(dataset) -> object:
    """Train XGBoost model and return it.

    XGBModel in Qlib uses xgb.train() API directly (not sklearn API),
    so parameters must be in xgboost native format.
    """
    from qlib.contrib.model.xgboost import XGBModel

    model = XGBModel(
        objective="reg:squarederror",
        max_depth=8,
        eta=0.05,
        colsample_bytree=0.8879,
        subsample=0.8789,
        alpha=205.6999,    # L1 reg
        reg_lambda=580.9768,  # L2 reg
        nthread=8,
    )
    model.fit(dataset, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=0)
    return model


def get_predictions(model, dataset) -> pd.Series:
    """Get model predictions on test set."""
    pred = model.predict(dataset)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred.name = "score"
    return pred


def compute_ic_metrics(pred: pd.Series, dataset) -> dict:
    """Compute IC/RankIC metrics between predictions and labels."""
    from qlib.contrib.eva.alpha import calc_ic

    label = dataset.prepare("test", col_set="label")
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    # Align index
    common_idx = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_idx]
    label_aligned = label.loc[common_idx]

    # Compute IC per day
    df = pd.DataFrame({"pred": pred_aligned, "label": label_aligned})
    daily_ic = df.groupby(level="datetime").apply(
        lambda x: x["pred"].corr(x["label"])
    )
    daily_rank_ic = df.groupby(level="datetime").apply(
        lambda x: x["pred"].corr(x["label"], method="spearman")
    )

    ic_mean = daily_ic.mean()
    icir = daily_ic.mean() / daily_ic.std() if daily_ic.std() > 0 else 0
    rank_ic_mean = daily_rank_ic.mean()
    rank_icir = daily_rank_ic.mean() / daily_rank_ic.std() if daily_rank_ic.std() > 0 else 0

    return {
        "IC": round(float(ic_mean), 6),
        "ICIR": round(float(icir), 4),
        "Rank_IC": round(float(rank_ic_mean), 6),
        "Rank_ICIR": round(float(rank_icir), 4),
        "n_days": int(len(daily_ic)),
    }


def run_backtest(pred: pd.Series, topk: int = 50, n_drop: int = 5,
                 hold_thresh: int = 1) -> dict:
    """Run backtest with given strategy parameters and return metrics."""
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.evaluate import risk_analysis

    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": pred,
            "topk": topk,
            "n_drop": n_drop,
            "hold_thresh": hold_thresh,
        },
    }

    report, positions = backtest_daily(
        start_time=TEST_SEG[0],
        end_time=TEST_SEG[1],
        strategy=strategy_config,
        benchmark=BENCHMARK,
        account=ACCOUNT,
        exchange_kwargs=EXCHANGE_KWARGS,
    )

    # Risk analysis
    excess_no_cost = report["return"] - report["bench"]
    excess_with_cost = report["return"] - report["bench"] - report["cost"]

    analysis_no_cost = risk_analysis(excess_no_cost, freq="day")
    analysis_with_cost = risk_analysis(excess_with_cost, freq="day")

    def _extract(analysis_df, label):
        risk = analysis_df["risk"]
        return {
            f"ann_ret_{label}": round(float(risk.loc["annualized_return"]), 6),
            f"IR_{label}": round(float(risk.loc["information_ratio"]), 4),
            f"max_dd_{label}": round(float(risk.loc["max_drawdown"]), 6),
        }

    metrics = {}
    metrics.update(_extract(analysis_no_cost, "no_cost"))
    metrics.update(_extract(analysis_with_cost, "with_cost"))

    # Turnover: daily cost / total portfolio value proxy
    total_cost = report["cost"].sum()
    total_days = len(report)
    metrics["daily_turnover"] = round(float(report["turnover"].mean()), 6) if "turnover" in report.columns else None
    metrics["total_cost_pct"] = round(float(total_cost / ACCOUNT * 100), 4)

    return metrics


# ─── Phase Runners ────────────────────────────────────────────────────────────


def phase1_rebalance_freq(pred: pd.Series, ic_metrics: dict, quick: bool = False) -> list[dict]:
    """Phase 1: Test different hold_thresh values (backtest-only, no retraining)."""
    print("\n" + "=" * 80)
    print("  PHASE 1: REBALANCE FREQUENCY OPTIMIZATION")
    print("  (hold_thresh controls min holding period, no retraining needed)")
    print("=" * 80)

    hold_values = [1, 3, 5, 10, 15, 20, 30, 40] if not quick else [1, 5, 10, 20]
    results = []

    for hold in hold_values:
        print(f"\n  Testing hold_thresh={hold}...", end=" ", flush=True)
        t0 = time.time()
        bt = run_backtest(pred, topk=50, n_drop=5, hold_thresh=hold)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        result = {
            "experiment": f"hold_{hold}",
            "phase": 1,
            "hold_thresh": hold,
            "topk": 50,
            "n_drop": 5,
            "label_days": 1,
            "loss": "mse",
            "model": "lgb",
            "elapsed": round(elapsed, 1),
            **ic_metrics,
            **bt,
        }
        results.append(result)
        _print_result(result)

    return results


def phase2_label_horizon(quick: bool = False) -> list[dict]:
    """Phase 2: Test different label horizons (requires retraining)."""
    print("\n" + "=" * 80)
    print("  PHASE 2: LABEL HORIZON ALIGNMENT")
    print("  (longer labels to match longer holding periods)")
    print("=" * 80)

    label_days_list = [2, 5, 10] if not quick else [2, 5]
    # Test with hold_thresh=1 (daily) and hold_thresh=20 (best from Phase 1)
    results = []

    for label_days in label_days_list:
        print(f"\n  Training with {label_days}-day label...", flush=True)
        t0 = time.time()

        dataset = create_dataset(label_days=label_days)
        model = train_lgb(dataset, loss="mse")
        pred = get_predictions(model, dataset)
        ic = compute_ic_metrics(pred, dataset)
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.1f}s. IC={ic['IC']:.4f}, RankIC={ic['Rank_IC']:.4f}")

        # Test with hold_thresh=1, 10, and 20
        hold_values = [1, 10, 20] if not quick else [1, 20]

        for hold in hold_values:
            bt_t0 = time.time()
            bt = run_backtest(pred, topk=50, n_drop=5, hold_thresh=hold)
            bt_elapsed = time.time() - bt_t0

            result = {
                "experiment": f"label{label_days}d_hold{hold}",
                "phase": 2,
                "hold_thresh": hold,
                "topk": 50,
                "n_drop": 5,
                "label_days": label_days,
                "loss": "mse",
                "model": "lgb",
                "elapsed": round(train_time + bt_elapsed, 1),
                **ic,
                **bt,
            }
            results.append(result)
            _print_result(result)

        del model, dataset, pred
        gc.collect()

    return results


def phase3_topk_ndrop(pred: pd.Series, ic_metrics: dict, quick: bool = False) -> list[dict]:
    """Phase 3: Optimize topk and n_drop (backtest-only)."""
    print("\n" + "=" * 80)
    print("  PHASE 3: TOPK / N_DROP OPTIMIZATION")
    print("  (portfolio concentration, no retraining needed)")
    print("=" * 80)

    if quick:
        combos = [(30, 3), (50, 5), (20, 2)]
    else:
        combos = [(20, 2), (30, 3), (30, 5), (50, 3), (50, 5), (50, 10), (80, 5)]

    # Use hold_thresh=1 (default) and hold_thresh=20 (best from Phase 1)
    hold_values = [1, 20] if not quick else [20]
    results = []

    for topk, n_drop in combos:
        for hold in hold_values:
            name = f"top{topk}_drop{n_drop}_hold{hold}"
            print(f"\n  Testing {name}...", end=" ", flush=True)
            t0 = time.time()
            bt = run_backtest(pred, topk=topk, n_drop=n_drop, hold_thresh=hold)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")

            result = {
                "experiment": name,
                "phase": 3,
                "hold_thresh": hold,
                "topk": topk,
                "n_drop": n_drop,
                "label_days": 1,
                "loss": "mse",
                "model": "lgb",
                "elapsed": round(elapsed, 1),
                **ic_metrics,
                **bt,
            }
            results.append(result)
            _print_result(result)

    return results


def phase4_loss_function(quick: bool = False) -> list[dict]:
    """Phase 4: Test different loss functions (requires retraining)."""
    print("\n" + "=" * 80)
    print("  PHASE 4: LOSS FUNCTION COMPARISON")
    print("  (mse vs huber vs mae)")
    print("=" * 80)

    losses = ["huber", "mae"] if not quick else ["huber"]
    results = []

    for loss in losses:
        print(f"\n  Training with loss={loss}...", flush=True)
        t0 = time.time()

        dataset = create_dataset(label_days=1)
        model = train_lgb(dataset, loss=loss)
        pred = get_predictions(model, dataset)
        ic = compute_ic_metrics(pred, dataset)
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.1f}s. IC={ic['IC']:.4f}, RankIC={ic['Rank_IC']:.4f}")

        # Test with hold_thresh=1 and 20
        for hold in [1, 20]:
            bt_t0 = time.time()
            bt = run_backtest(pred, topk=50, n_drop=5, hold_thresh=hold)
            bt_elapsed = time.time() - bt_t0

            result = {
                "experiment": f"loss_{loss}_hold{hold}",
                "phase": 4,
                "hold_thresh": hold,
                "topk": 50,
                "n_drop": 5,
                "label_days": 1,
                "loss": loss,
                "model": "lgb",
                "elapsed": round(train_time + bt_elapsed, 1),
                **ic,
                **bt,
            }
            results.append(result)
            _print_result(result)

        del model, dataset, pred
        gc.collect()

    return results


def phase5_model_ensemble(quick: bool = False) -> list[dict]:
    """Phase 5: XGBoost + LightGBM ensemble."""
    print("\n" + "=" * 80)
    print("  PHASE 5: MODEL ENSEMBLE (LightGBM + XGBoost)")
    print("=" * 80)

    results = []
    dataset = create_dataset(label_days=1)

    # Train LightGBM
    print("\n  Training LightGBM...", flush=True)
    t0 = time.time()
    lgb_model = train_lgb(dataset, loss="mse")
    lgb_pred = get_predictions(lgb_model, dataset)
    lgb_time = time.time() - t0
    lgb_ic = compute_ic_metrics(lgb_pred, dataset)
    print(f"  LGB done ({lgb_time:.1f}s): IC={lgb_ic['IC']:.4f}")

    # Train XGBoost
    print("  Training XGBoost...", flush=True)
    t0 = time.time()
    xgb_model = train_xgb(dataset)
    xgb_pred = get_predictions(xgb_model, dataset)
    xgb_time = time.time() - t0
    xgb_ic = compute_ic_metrics(xgb_pred, dataset)
    print(f"  XGB done ({xgb_time:.1f}s): IC={xgb_ic['IC']:.4f}")

    # XGBoost standalone
    for hold in [1, 20]:
        bt = run_backtest(xgb_pred, topk=50, n_drop=5, hold_thresh=hold)
        result = {
            "experiment": f"xgb_hold{hold}",
            "phase": 5,
            "hold_thresh": hold,
            "topk": 50,
            "n_drop": 5,
            "label_days": 1,
            "loss": "mse",
            "model": "xgb",
            "elapsed": round(xgb_time, 1),
            **xgb_ic,
            **bt,
        }
        results.append(result)
        _print_result(result)

    # Ensemble: average predictions (rank-normalized)
    print("\n  Creating ensemble (rank-average)...", flush=True)

    # Rank-normalize within each day then average
    def rank_normalize(s: pd.Series) -> pd.Series:
        return s.groupby(level="datetime").rank(pct=True)

    lgb_rank = rank_normalize(lgb_pred)
    xgb_rank = rank_normalize(xgb_pred)

    # Align
    common_idx = lgb_rank.index.intersection(xgb_rank.index)
    ensemble_pred = (lgb_rank.loc[common_idx] + xgb_rank.loc[common_idx]) / 2
    ensemble_pred.name = "score"

    ensemble_ic = compute_ic_metrics(ensemble_pred, dataset)
    print(f"  Ensemble IC={ensemble_ic['IC']:.4f}, RankIC={ensemble_ic['Rank_IC']:.4f}")

    for hold in [1, 20]:
        bt = run_backtest(ensemble_pred, topk=50, n_drop=5, hold_thresh=hold)
        result = {
            "experiment": f"ensemble_hold{hold}",
            "phase": 5,
            "hold_thresh": hold,
            "topk": 50,
            "n_drop": 5,
            "label_days": 1,
            "loss": "ensemble",
            "model": "lgb+xgb",
            "elapsed": round(lgb_time + xgb_time, 1),
            **ensemble_ic,
            **bt,
        }
        results.append(result)
        _print_result(result)

    del lgb_model, xgb_model, dataset
    gc.collect()

    return results


# ─── Output ───────────────────────────────────────────────────────────────────


def _print_result(result: dict):
    """Print one experiment result as a compact line."""
    name = result["experiment"]
    ic = result.get("IC", float("nan"))
    ric = result.get("Rank_IC", float("nan"))
    ir_wc = result.get("IR_with_cost", float("nan"))
    ret_wc = result.get("ann_ret_with_cost", float("nan"))
    mdd_wc = result.get("max_dd_with_cost", float("nan"))
    print(f"    {name:<30s}  IC={ic:+.4f}  RkIC={ric:+.4f}  "
          f"IR(w/c)={ir_wc:+.3f}  Ret(w/c)={ret_wc:+.4f}  MaxDD={mdd_wc:+.4f}")


def print_full_comparison(all_results: list[dict]):
    """Print final comparison table sorted by IR with cost."""
    print("\n" + "=" * 140)
    print("  FULL OPTIMIZATION RESULTS — SORTED BY IR(with cost)")
    print("=" * 140)

    header = (f"{'#':<3} {'Experiment':<35} {'Phase':>5} {'Label':>5} {'Loss':>6} "
              f"{'TopK':>5} {'Drop':>5} {'Hold':>5}  "
              f"{'IC':>8} {'RkIC':>8} {'ICIR':>7} "
              f"{'Ret(w/c)':>10} {'IR(w/c)':>9} {'MaxDD':>8}")
    print(header)
    print("-" * 140)

    sorted_results = sorted(all_results, key=lambda x: x.get("IR_with_cost", -999), reverse=True)
    for i, r in enumerate(sorted_results, 1):
        name = r["experiment"]
        phase = r["phase"]
        label = r.get("label_days", 1)
        loss = r.get("loss", "mse")
        topk = r.get("topk", 50)
        n_drop = r.get("n_drop", 5)
        hold = r.get("hold_thresh", 1)
        ic = r.get("IC", float("nan"))
        ric = r.get("Rank_IC", float("nan"))
        icir = r.get("ICIR", float("nan"))
        ir = r.get("IR_with_cost", float("nan"))
        ret = r.get("ann_ret_with_cost", float("nan"))
        mdd = r.get("max_dd_with_cost", float("nan"))

        marker = " ***" if i <= 3 else ""
        print(f"{i:<3} {name:<35} {phase:>5} {label:>5}d {loss:>6} "
              f"{topk:>5} {n_drop:>5} {hold:>5}  "
              f"{ic:>+8.4f} {ric:>+8.4f} {icir:>+7.3f} "
              f"{ret:>+10.4f} {ir:>+9.3f} {mdd:>+8.4f}{marker}")

    print("-" * 140)
    best = sorted_results[0]
    print(f"\n  BEST CONFIG: {best['experiment']}")
    print(f"    Phase {best['phase']} | label={best.get('label_days',1)}d | loss={best.get('loss','mse')} | "
          f"topk={best.get('topk',50)} | n_drop={best.get('n_drop',5)} | hold_thresh={best.get('hold_thresh',1)}")
    print(f"    IR(with cost)={best.get('IR_with_cost', 0):+.4f} | "
          f"Return(with cost)={best.get('ann_ret_with_cost', 0):+.4f} | "
          f"MaxDD={best.get('max_dd_with_cost', 0):+.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Comprehensive strategy optimization")
    parser.add_argument("--phase", default="1,2,3,4,5",
                        help="Comma-separated phases to run (default: 1,2,3,4,5)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer combinations per phase")
    args = parser.parse_args()

    phases = [int(p.strip()) for p in args.phase.split(",")]
    quick = args.quick

    print("=" * 80)
    print("  COMPREHENSIVE STRATEGY OPTIMIZATION")
    print(f"  Market: {MARKET} | TopN: {TOPN} | Train: {TRAIN_SEG} | Test: {TEST_SEG}")
    print(f"  Phases: {phases} | Quick: {quick}")
    print("=" * 80)

    init_qlib_once()

    all_results = []
    total_t0 = time.time()

    # Load existing results if available (for incremental runs)
    existing_path = OUTPUT_DIR / "optimization_results.json"
    existing_results = []
    if existing_path.exists():
        try:
            existing_results = json.loads(existing_path.read_text())
            print(f"  Loaded {len(existing_results)} existing results from previous run")
        except Exception:
            pass

    # ── Baseline: train default model (used by Phase 1, 3) ──
    if any(p in phases for p in [1, 3]):
        print("\n  Training baseline model (TopN30, label=1d, loss=mse)...", flush=True)
        t0 = time.time()
        dataset = create_dataset(label_days=1)
        model = train_lgb(dataset, loss="mse")
        pred = get_predictions(model, dataset)
        ic_metrics = compute_ic_metrics(pred, dataset)
        baseline_time = time.time() - t0
        print(f"  Baseline trained in {baseline_time:.1f}s")
        print(f"  IC={ic_metrics['IC']:.4f}, RankIC={ic_metrics['Rank_IC']:.4f}, "
              f"ICIR={ic_metrics['ICIR']:.4f}, RankICIR={ic_metrics['Rank_ICIR']:.4f}")

        # Add baseline result (hold=1, topk=50, n_drop=5)
        bt_baseline = run_backtest(pred, topk=50, n_drop=5, hold_thresh=1)
        all_results.append({
            "experiment": "baseline_topn30",
            "phase": 0,
            "hold_thresh": 1,
            "topk": 50,
            "n_drop": 5,
            "label_days": 1,
            "loss": "mse",
            "model": "lgb",
            "elapsed": round(baseline_time, 1),
            **ic_metrics,
            **bt_baseline,
        })
        _print_result(all_results[-1])

    # Phase 1: Rebalance frequency
    if 1 in phases:
        results = phase1_rebalance_freq(pred, ic_metrics, quick=quick)
        all_results.extend(results)

    # Phase 3 (before Phase 2 since it uses same model)
    if 3 in phases:
        results = phase3_topk_ndrop(pred, ic_metrics, quick=quick)
        all_results.extend(results)

    # Free baseline model
    if any(p in phases for p in [1, 3]):
        del model, dataset
        gc.collect()

    # Phase 2: Label horizon (requires retraining)
    if 2 in phases:
        results = phase2_label_horizon(quick=quick)
        all_results.extend(results)

    # Phase 4: Loss function (requires retraining)
    if 4 in phases:
        results = phase4_loss_function(quick=quick)
        all_results.extend(results)

    # Phase 5: Model ensemble (requires retraining)
    if 5 in phases:
        results = phase5_model_ensemble(quick=quick)
        all_results.extend(results)

    total_elapsed = time.time() - total_t0

    # Merge with existing results (avoid duplicates by experiment name)
    seen = {r["experiment"] for r in all_results}
    for r in existing_results:
        if r["experiment"] not in seen:
            all_results.append(r)
            seen.add(r["experiment"])

    # ── Final comparison ──
    print_full_comparison(all_results)
    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Save results
    out_path = OUTPUT_DIR / "optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

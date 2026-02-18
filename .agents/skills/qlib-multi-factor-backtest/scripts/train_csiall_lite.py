"""Memory-optimized training: TopN + Baseline on csiall.

Optimizations for 16GB RAM:
1. Shorter train window: 2021-2023 (3yr instead of 5yr) — ~40% less data
2. Shorter end_time: 2025-06-30 (avoid loading future dates)
3. Reduced LightGBM: max_depth=6, num_leaves=128, num_threads=4
4. Sequential runs with gc.collect() between each

Usage:
    uv run python scripts/train_csiall_lite.py
    uv run python scripts/train_csiall_lite.py --only baseline
    uv run python scripts/train_csiall_lite.py --only topn20
    uv run python scripts/train_csiall_lite.py --only topn20,topn30,topn50,baseline
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_qlib.runtime import PROJECT_ROOT, init_qlib
from project_qlib.workflow import run_qrun

# --- Shared config template (lite = memory optimized) ---
LITE_CONFIG_TEMPLATE = """qlib_init:
  provider_uri: "{provider_uri}"
  region: cn

market: &market csiall
benchmark: &benchmark SH000985

data_handler_config: &data_handler_config
  start_time: 2021-01-01
  end_time: 2025-06-30
  fit_start_time: 2021-01-01
  fit_end_time: 2023-12-31
  instruments: *market

port_analysis_config: &port_analysis_config
  strategy:
    class: TopkDropoutStrategy
    module_path: qlib.contrib.strategy
    kwargs:
      signal: <PRED>
      topk: 50
      n_drop: 5
  backtest:
    start_time: 2024-07-01
    end_time: 2025-06-30
    account: 100000000
    benchmark: *benchmark
    exchange_kwargs:
      limit_threshold: 0.095
      deal_price: close
      open_cost: 0.0005
      close_cost: 0.0015
      min_cost: 5

task:
  model:
    class: LGBModel
    module_path: qlib.contrib.model.gbdt
    kwargs:
      loss: mse
      colsample_bytree: 0.8879
      learning_rate: 0.2
      subsample: 0.8789
      lambda_l1: 205.6999
      lambda_l2: 580.9768
      max_depth: 6
      num_leaves: 128
      num_threads: 4
  dataset:
    class: DatasetH
    module_path: qlib.data.dataset
    kwargs:
      handler:
        class: {handler_class}
        module_path: {handler_module}
        kwargs: *data_handler_config
      segments:
        train: [2021-01-01, 2023-12-31]
        valid: [2024-01-01, 2024-06-30]
        test: [2024-07-01, 2025-06-30]
  record:
    - class: SignalRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        model: <MODEL>
        dataset: <DATASET>
    - class: SigAnaRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        ana_long_short: false
        ann_scaler: 252
    - class: PortAnaRecord
      module_path: qlib.workflow.record_temp
      kwargs:
        config: *port_analysis_config
"""

# Define all experiments
EXPERIMENTS = {
    "baseline": {
        "handler_class": "Alpha158",
        "handler_module": "qlib.contrib.data.handler",
        "n_factors": 158,
        "label": "Alpha158 Baseline",
    },
    "topn20": {
        "handler_class": "TopN20",
        "handler_module": "project_qlib.factors.topn_alpha",
        "n_factors": 20,
        "label": "Top-20 Factors",
    },
    "topn30": {
        "handler_class": "TopN30",
        "handler_module": "project_qlib.factors.topn_alpha",
        "n_factors": 30,
        "label": "Top-30 Factors",
    },
    "topn50": {
        "handler_class": "TopN50",
        "handler_module": "project_qlib.factors.topn_alpha",
        "n_factors": 50,
        "label": "Top-50 Factors",
    },
}


def write_config(name: str) -> Path:
    exp = EXPERIMENTS[name]
    config_path = PROJECT_ROOT / "configs" / f"workflow_csiall_{name}_lite.yaml"
    content = LITE_CONFIG_TEMPLATE.format(
        provider_uri=str((PROJECT_ROOT / "data" / "qlib" / "cn_data").resolve()),
        handler_class=exp["handler_class"],
        handler_module=exp["handler_module"],
    )
    config_path.write_text(content)
    return config_path


def read_mlflow_latest(experiment_id: str = "1") -> dict:
    """Read the latest MLflow run metrics."""
    mlruns_dir = PROJECT_ROOT / "mlruns" / experiment_id
    runs = sorted(mlruns_dir.glob("*/meta.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return {}
    run_dir = runs[0].parent
    run_id = run_dir.name
    metrics = {"run_id": run_id}
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        for f in metrics_dir.iterdir():
            try:
                lines = f.read_text().strip().split("\n")
                val = float(lines[-1].split()[1])
                metrics[f.name] = val
            except (IndexError, ValueError):
                pass
    return metrics


KEY_METRICS = [
    "IC", "ICIR", "Rank IC", "Rank ICIR",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
]


def run_experiment(name: str) -> dict:
    exp = EXPERIMENTS[name]
    print(f"\n{'='*80}")
    print(f"  {exp['label']} ({exp['n_factors']} factors) on csiall [LITE]")
    print(f"  Train: 2021-2023, Valid: 2024H1, Test: 2024H2-2025H1")
    print(f"{'='*80}")

    config_path = write_config(name)
    log_path = PROJECT_ROOT / "outputs" / "logs" / f"csiall_{name}_lite.log"

    t0 = time.time()
    result = run_qrun(config_path, log_path)
    elapsed = time.time() - t0

    status = "OK" if result["success"] else "FAILED"
    print(f"  Status: {status} ({elapsed:.0f}s)")
    if not result["success"]:
        print(f"  Error: {result['error_tail'][:500]}")
        gc.collect()
        return {"name": name, "n_factors": exp["n_factors"], "success": False, "elapsed": round(elapsed, 1)}

    metrics = read_mlflow_latest()
    print(f"  Run ID: {metrics.get('run_id', 'N/A')}")
    for k in KEY_METRICS:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")

    gc.collect()
    return {
        "name": name,
        "n_factors": exp["n_factors"],
        "success": True,
        "elapsed": round(elapsed, 1),
        **metrics,
    }


def print_summary(results: list[dict]):
    print(f"\n{'='*110}")
    print("  COMPARISON SUMMARY (csiall, LITE config: train 2021-2023, test 2024H2-2025H1)")
    print(f"{'='*110}")
    header = f"{'Model':<18} {'#Fac':>5} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'RkICIR':>8} {'IR(w/c)':>9} {'Ret(w/c)':>10} {'Time':>6}"
    print(header)
    print("-" * 110)
    for r in results:
        if not r.get("success"):
            print(f"{r['name']:<18} {'FAILED':>5}")
            continue
        ic = r.get("IC", float("nan"))
        ric = r.get("Rank IC", float("nan"))
        icir = r.get("ICIR", float("nan"))
        ricir = r.get("Rank ICIR", float("nan"))
        ir = r.get("1day.excess_return_with_cost.information_ratio", float("nan"))
        ret = r.get("1day.excess_return_with_cost.annualized_return", float("nan"))
        t = r.get("elapsed", 0)
        print(f"{r['name']:<18} {r['n_factors']:>5d} {ic:>8.4f} {ric:>8.4f} {icir:>8.4f} {ricir:>8.4f} {ir:>9.4f} {ret:>10.4f} {t:>5.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Comma-separated experiment names (baseline,topn20,topn30,topn50)")
    args = parser.parse_args()

    if args.only:
        exp_names = [x.strip() for x in args.only.split(",")]
    else:
        # Default order: smallest first (to fail fast if OOM), baseline last
        exp_names = ["topn20", "topn30", "topn50", "baseline"]

    results = []
    for name in exp_names:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            continue
        r = run_experiment(name)
        results.append(r)

    print_summary(results)

    # Save
    out_path = PROJECT_ROOT / "outputs" / "csiall_lite_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

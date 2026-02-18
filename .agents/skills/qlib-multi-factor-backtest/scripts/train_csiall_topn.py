"""Train and compare TopN factor models on csiall market.

Runs experiments with N=20, N=30, N=50 factor sets, plus Alpha158 baseline.
All on csiall (全A股) market, with LightGBM.

Usage:
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/train_csiall_topn.py [--skip-baseline] [--n 20,30,50]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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

from project_qlib.runtime import PROJECT_ROOT, init_qlib
from project_qlib.workflow import run_qrun


def create_topn_config(n: int) -> Path:
    """Create a YAML config for TopN handler."""
    config_path = PROJECT_ROOT / "configs" / f"workflow_csiall_topn{n}_lightgbm.yaml"
    provider_uri = str((PROJECT_ROOT / "data" / "qlib" / "cn_data").resolve())
    content = f"""qlib_init:
  provider_uri: "{provider_uri}"
  region: cn

market: &market csiall
benchmark: &benchmark SH000985

data_handler_config: &data_handler_config
  start_time: 2019-01-01
  end_time: 2025-12-31
  fit_start_time: 2019-01-01
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
    end_time: 2025-12-31
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
      max_depth: 8
      num_leaves: 210
      num_threads: 8
  dataset:
    class: DatasetH
    module_path: qlib.data.dataset
    kwargs:
      handler:
        class: TopN{n}
        module_path: project_qlib.factors.topn_alpha
        kwargs: *data_handler_config
      segments:
        train: [2019-01-01, 2023-12-31]
        valid: [2024-01-01, 2024-06-30]
        test: [2024-07-01, 2025-12-31]
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
    config_path.write_text(content)
    return config_path


def read_mlflow_metrics(experiment_id: str = "1") -> dict:
    """Read the latest MLflow run metrics."""
    import glob
    mlruns_dir = PROJECT_ROOT / "mlruns" / experiment_id
    # Find latest run
    runs = sorted(mlruns_dir.glob("*/meta.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return {}
    run_dir = runs[0].parent
    run_id = run_dir.name

    metrics = {}
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        for f in metrics_dir.iterdir():
            try:
                lines = f.read_text().strip().split("\n")
                last_line = lines[-1]
                val = float(last_line.split()[1])
                metrics[f.name] = val
            except (IndexError, ValueError):
                pass
    metrics["run_id"] = run_id
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true", help="Skip Alpha158 baseline run")
    parser.add_argument("--n", default="20,30,50", help="Comma-separated N values (default: 20,30,50)")
    args = parser.parse_args()

    n_values = [int(x) for x in args.n.split(",")]

    results = {}

    # Step 1: Baseline (Alpha158 with 158 factors)
    if not args.skip_baseline:
        print("=" * 80)
        print("  BASELINE: Alpha158 (158 factors) on csiall")
        print("=" * 80)
        baseline_config = PROJECT_ROOT / "configs" / "workflow_csiall_baseline_lightgbm.yaml"
        t0 = time.time()
        result = run_qrun(baseline_config, PROJECT_ROOT / "outputs" / "logs" / "csiall_baseline.log")
        elapsed = time.time() - t0
        print(f"  Baseline: {'OK' if result['success'] else 'FAILED'} ({elapsed:.0f}s)")
        if not result['success']:
            print(f"  Error: {result['error_tail'][:500]}")
        metrics = read_mlflow_metrics()
        results["baseline_158"] = {
            "n_factors": 158,
            "success": result["success"],
            "elapsed": round(elapsed, 1),
            **metrics,
        }
        key_names = ["IC", "ICIR", "Rank IC", "Rank ICIR",
                     "1day.excess_return_with_cost.annualized_return",
                     "1day.excess_return_with_cost.information_ratio"]
        print(f"  Run ID: {metrics.get('run_id', 'N/A')}")
        for k in key_names:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.4f}")
        gc.collect()

    # Step 2: TopN experiments
    for n in n_values:
        print()
        print("=" * 80)
        print(f"  TOP-{n} FACTORS on csiall")
        print("=" * 80)

        config_path = create_topn_config(n)
        print(f"  Config: {config_path.name}")

        t0 = time.time()
        result = run_qrun(config_path, PROJECT_ROOT / "outputs" / "logs" / f"csiall_topn{n}.log")
        elapsed = time.time() - t0
        print(f"  TopN{n}: {'OK' if result['success'] else 'FAILED'} ({elapsed:.0f}s)")
        if not result['success']:
            print(f"  Error: {result['error_tail'][:500]}")

        metrics = read_mlflow_metrics()
        results[f"topn_{n}"] = {
            "n_factors": n,
            "success": result["success"],
            "elapsed": round(elapsed, 1),
            **metrics,
        }
        key_names = ["IC", "ICIR", "Rank IC", "Rank ICIR",
                     "1day.excess_return_with_cost.annualized_return",
                     "1day.excess_return_with_cost.information_ratio"]
        print(f"  Run ID: {metrics.get('run_id', 'N/A')}")
        for k in key_names:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.4f}")
        gc.collect()

    # Step 3: Summary
    print()
    print("=" * 100)
    print("  COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Model':<20} {'#Factors':>8} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'RankICIR':>9} {'IR(w/cost)':>11} {'AnnRet(w/c)':>12} {'Time':>6}")
    print("-" * 100)
    for name, r in results.items():
        if not r.get("success"):
            print(f"{name:<20} {'FAILED':>8}")
            continue
        ic = r.get("IC", float("nan"))
        ric = r.get("Rank IC", float("nan"))
        icir = r.get("ICIR", float("nan"))
        ricir = r.get("Rank ICIR", float("nan"))
        ir_wc = r.get("1day.excess_return_with_cost.information_ratio", float("nan"))
        ret_wc = r.get("1day.excess_return_with_cost.annualized_return", float("nan"))
        t = r.get("elapsed", 0)
        print(f"{name:<20} {r['n_factors']:>8d} {ic:>8.4f} {ric:>8.4f} {icir:>8.4f} {ricir:>9.4f} {ir_wc:>11.4f} {ret_wc:>12.4f} {t:>5.0f}s")

    # Save results
    out_path = PROJECT_ROOT / "outputs" / "csiall_topn_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

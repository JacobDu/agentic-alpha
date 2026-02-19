"""Memory-optimized training on CSI1000 market.

Runs TopN20/30/50 + Baseline comparison on CSI1000 (~1000 stocks).
Uses extended date range including 2026 data.

Usage:
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/train_csi1000_lite.py
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/train_csi1000_lite.py --only baseline
    uv run python .agents/skills/qlib-multi-factor-backtest/scripts/train_csi1000_lite.py --only topn20,topn30,topn50,baseline
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

from project_qlib.metrics_standard import canonicalize_metrics, get_metric
from project_qlib.runtime import PROJECT_ROOT, init_qlib
from project_qlib.workflow import run_qrun

# --- Config template for CSI1000 ---
CONFIG_TEMPLATE = """qlib_init:
  provider_uri: "{provider_uri}"
  region: cn

market: &market csi1000
benchmark: &benchmark SH000852

data_handler_config: &data_handler_config
  start_time: 2019-01-01
  end_time: 2026-02-13
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
    end_time: 2026-02-13
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
        class: {handler_class}
        module_path: {handler_module}
        kwargs: *data_handler_config
      segments:
        train: [2019-01-01, 2023-12-31]
        valid: [2024-01-01, 2024-06-30]
        test: [2024-07-01, 2026-02-13]
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
        "label": "Top-20 (unified ranking)",
    },
    "topn30": {
        "handler_class": "TopN30",
        "handler_module": "project_qlib.factors.topn_alpha",
        "n_factors": 30,
        "label": "Top-30 (unified ranking)",
    },
    "topn50": {
        "handler_class": "TopN50",
        "handler_module": "project_qlib.factors.topn_alpha",
        "n_factors": 50,
        "label": "Top-50 (unified ranking)",
    },
}


def write_config(name: str) -> Path:
    exp = EXPERIMENTS[name]
    config_path = PROJECT_ROOT / "configs" / f"workflow_csi1000_{name}_lite.yaml"
    content = CONFIG_TEMPLATE.format(
        provider_uri=str((PROJECT_ROOT / "data" / "qlib" / "cn_data").resolve()),
        handler_class=exp["handler_class"],
        handler_module=exp["handler_module"],
    )
    config_path.write_text(content)
    return config_path


def read_mlflow_latest(experiment_id: str = "1") -> dict:
    mlruns_dir = PROJECT_ROOT / "mlruns" / experiment_id
    runs = sorted(mlruns_dir.glob("*/meta.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return {}
    run_dir = runs[0].parent
    metrics = {"run_id": run_dir.name}
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
    ("ic_mean", "ic_mean"),
    ("ic_ir", "ic_ir"),
    ("rank_ic_mean", "rank_ic_mean"),
    ("rank_ic_ir", "rank_ic_ir"),
    ("excess_return_annualized_with_cost", "excess_return_annualized_with_cost"),
    ("information_ratio_with_cost", "information_ratio_with_cost"),
    ("max_drawdown_with_cost", "max_drawdown_with_cost"),
]


def run_experiment(name: str) -> dict:
    exp = EXPERIMENTS[name]
    print(f"\n{'='*80}")
    print(f"  {exp['label']} ({exp['n_factors']} factors) on CSI1000")
    print(f"  Train: 2019-2023, Valid: 2024H1, Test: 2024H2-2026.02.13")
    print(f"{'='*80}")

    config_path = write_config(name)
    log_path = PROJECT_ROOT / "outputs" / "logs" / f"csi1000_{name}_lite.log"

    t0 = time.time()
    result = run_qrun(config_path, log_path)
    elapsed = time.time() - t0

    status = "OK" if result["success"] else "FAILED"
    print(f"  Status: {status} ({elapsed:.0f}s)")
    if not result["success"]:
        print(f"  Error: {result['error_tail'][:500]}")
        gc.collect()
        return {"name": name, "n_factors": exp["n_factors"], "success": False, "elapsed": round(elapsed, 1)}

    metrics = canonicalize_metrics(read_mlflow_latest(), keep_unknown=True)
    print(f"  Run ID: {metrics.get('run_id', 'N/A')}")
    for key, label in KEY_METRICS:
        val = get_metric(metrics, key)
        if val is not None:
            print(f"  {label}: {val:.4f}")

    gc.collect()
    return {"name": name, "n_factors": exp["n_factors"], "success": True, "elapsed": round(elapsed, 1), **metrics}


def print_summary(results: list[dict]):
    print(f"\n{'='*120}")
    print("  CSI1000 COMPARISON (train 2019-2023, test 2024H2-2026.02.13)")
    print(f"{'='*120}")
    header = f"{'Model':<18} {'#Fac':>5} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'RkICIR':>8} {'IR(w/c)':>9} {'Ret(w/c)':>10} {'MaxDD':>8} {'Time':>6}"
    print(header)
    print("-" * 120)
    for r in results:
        if not r.get("success"):
            print(f"{r['name']:<18} FAILED")
            continue
        ic = get_metric(r, "ic_mean", float("nan"))
        ric = get_metric(r, "rank_ic_mean", float("nan"))
        icir = get_metric(r, "ic_ir", float("nan"))
        ricir = get_metric(r, "rank_ic_ir", float("nan"))
        ir = get_metric(r, "information_ratio_with_cost", float("nan"))
        ret = get_metric(r, "excess_return_annualized_with_cost", float("nan"))
        mdd = get_metric(r, "max_drawdown_with_cost", float("nan"))
        t = r.get("elapsed", 0)
        print(f"{r['name']:<18} {r['n_factors']:>5d} {ic:>8.4f} {ric:>8.4f} {icir:>8.4f} {ricir:>8.4f} {ir:>9.4f} {ret:>10.4f} {mdd:>8.4f} {t:>5.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Comma-separated: baseline,topn20,topn30,topn50")
    args = parser.parse_args()

    if args.only:
        exp_names = [x.strip() for x in args.only.split(",")]
    else:
        exp_names = ["topn20", "topn30", "topn50", "baseline"]

    results = []
    for name in exp_names:
        if name not in EXPERIMENTS:
            print(f"Unknown: {name}")
            continue
        results.append(run_experiment(name))

    print_summary(results)

    out_path = PROJECT_ROOT / "outputs" / "csi1000_lite_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

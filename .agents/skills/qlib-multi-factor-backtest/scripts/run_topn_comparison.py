"""Phase 2/3: TopN Factor Combination with LightGBM.

Trains LightGBM models using only the Top-N factors from the factor library DB,
and compares them against the Alpha158 baseline.

Uses YAML configs + qlib.cli.run (subprocess) approach for reliable execution.

Test configurations:
  - Alpha158 baseline (158 factors)
  - TopN=20 (top 20 by |RankICIR|)
  - TopN=30
  - TopN=50
  - TopN=80

All on CSI1000, train 2018-2022, valid 2023, test 2024.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

if "--help" in sys.argv or "-h" in sys.argv:
    print("Usage: uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_topn_comparison.py")
    raise SystemExit(0)

def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Cannot locate project root (pyproject.toml not found)")

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_qlib.workflow import run_qrun


# ─── Config Templates ───

BASELINE_YAML = """
qlib_init:
  provider_uri: "data/qlib/cn_data"
  region: cn

market: &market csi1000
benchmark: &benchmark SH000852

data_handler_config: &data_handler_config
  start_time: 2018-01-01
  end_time: 2024-12-31
  fit_start_time: 2018-01-01
  fit_end_time: 2022-12-31
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
    start_time: 2024-01-01
    end_time: 2024-12-31
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
      learning_rate: 0.05
      subsample: 0.8789
      lambda_l1: 205.6999
      lambda_l2: 580.9768
      max_depth: 8
      num_leaves: 128
      num_threads: 8
      n_estimators: 1000
      early_stopping_rounds: 50
  dataset:
    class: DatasetH
    module_path: qlib.data.dataset
    kwargs:
      handler:
        class: Alpha158
        module_path: qlib.contrib.data.handler
        kwargs: *data_handler_config
      segments:
        train: [2018-01-01, 2022-12-31]
        valid: [2023-01-01, 2023-12-31]
        test: [2024-01-01, 2024-12-31]
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
""".strip()


def make_topn_yaml(n: int) -> str:
    """Generate YAML for a TopN experiment using DBTopN handler."""
    class_name = f"DBTopN{n}"
    return f"""
qlib_init:
  provider_uri: "data/qlib/cn_data"
  region: cn

market: &market csi1000
benchmark: &benchmark SH000852

data_handler_config: &data_handler_config
  start_time: 2018-01-01
  end_time: 2024-12-31
  fit_start_time: 2018-01-01
  fit_end_time: 2022-12-31
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
    start_time: 2024-01-01
    end_time: 2024-12-31
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
      learning_rate: 0.05
      subsample: 0.8789
      lambda_l1: 205.6999
      lambda_l2: 580.9768
      max_depth: 8
      num_leaves: 128
      num_threads: 8
      n_estimators: 1000
      early_stopping_rounds: 50
  dataset:
    class: DatasetH
    module_path: qlib.data.dataset
    kwargs:
      handler:
        class: {class_name}
        module_path: project_qlib.factors.topn_db
        kwargs: *data_handler_config
      segments:
        train: [2018-01-01, 2022-12-31]
        valid: [2023-01-01, 2023-12-31]
        test: [2024-01-01, 2024-12-31]
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
""".strip()


def parse_metrics(log_path: str) -> dict:
    """Parse IC and portfolio metrics from qlib log output."""
    metrics = {}
    log_text = Path(log_path).read_text(encoding="utf-8", errors="ignore")

    for pattern, key in [
        (r"'IC':\s*([-\d.e+]+)", "IC"),
        (r"'ICIR':\s*([-\d.e+]+)", "ICIR"),
        (r"'Rank IC':\s*([-\d.e+]+)", "Rank_IC"),
        (r"'Rank ICIR':\s*([-\d.e+]+)", "Rank_ICIR"),
    ]:
        m = re.search(pattern, log_text)
        if m:
            metrics[key] = float(m.group(1))

    for section, prefix in [
        ("excess return without cost", "no_cost"),
        ("excess return with cost", "with_cost"),
    ]:
        pat = (
            rf"{section}.*?annualized_return\s+([-\d.e+]+).*?"
            rf"information_ratio\s+([-\d.e+]+).*?"
            rf"max_drawdown\s+([-\d.e+]+)"
        )
        m = re.search(pat, log_text, re.DOTALL)
        if m:
            metrics[f"excess_ann_ret_{prefix}"] = float(m.group(1))
            metrics[f"IR_{prefix}"] = float(m.group(2))
            metrics[f"max_dd_{prefix}"] = float(m.group(3))

    # Extract feature count
    feat_m = re.search(r"number of features:\s*(\d+)", log_text, re.IGNORECASE)
    if feat_m:
        metrics["n_features"] = int(feat_m.group(1))

    return metrics


def print_comparison(results: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 130)
    print("PHASE 2/3: TOP-N FACTOR COMBINATION COMPARISON (CSI1000, LightGBM)")
    print("=" * 130)

    names = list(results.keys())
    col_w = 18
    header = f"{'Metric':<35}"
    for name in names:
        header += f" {name:>{col_w}}"
    print(header)
    print("-" * (35 + (col_w + 1) * len(names)))

    row = f"  {'Features':.<33}"
    for name in names:
        n = results[name].get("metrics", {}).get("n_features", "?")
        row += f" {str(n):>{col_w}}"
    print(row)

    row = f"  {'Training time (s)':.<33}"
    for name in names:
        t = results[name].get("seconds", 0)
        row += f" {t:>{col_w}.0f}"
    print(row)

    row = f"  {'Status':.<33}"
    for name in names:
        s = "OK" if results[name].get("success") else "FAIL"
        row += f" {s:>{col_w}}"
    print(row)
    print("-" * (35 + (col_w + 1) * len(names)))

    metric_groups = [
        ("Signal Quality", [
            ("IC", "IC (mean)"),
            ("ICIR", "ICIR"),
            ("Rank_IC", "Rank IC (mean)"),
            ("Rank_ICIR", "Rank ICIR"),
        ]),
        ("Portfolio (no cost)", [
            ("excess_ann_ret_no_cost", "Excess Ann Ret"),
            ("IR_no_cost", "IR"),
            ("max_dd_no_cost", "Max Drawdown"),
        ]),
        ("Portfolio (with cost)", [
            ("excess_ann_ret_with_cost", "Excess Ann Ret"),
            ("IR_with_cost", "IR"),
            ("max_dd_with_cost", "Max Drawdown"),
        ]),
    ]

    for group_name, metrics_list in metric_groups:
        print(f"\n  [{group_name}]")
        for key, label in metrics_list:
            row = f"    {label:.<31}"
            for name in names:
                val = results[name].get("metrics", {}).get(key)
                if val is not None:
                    if abs(val) < 0.01:
                        row += f" {val:>{col_w}.6f}"
                    else:
                        row += f" {val:>{col_w}.4f}"
                else:
                    row += f" {'N/A':>{col_w}}"
            print(row)

    # Best model
    print("\n" + "-" * (35 + (col_w + 1) * len(names)))
    best_name = None
    best_ir = -999
    for name, r in results.items():
        ir = r.get("metrics", {}).get("IR_with_cost")
        if ir is not None and ir > best_ir:
            best_ir = ir
            best_name = name
    if best_name:
        n_feat = results[best_name].get("metrics", {}).get("n_features", "?")
        print(f"  BEST MODEL: {best_name} (IR with cost = {best_ir:.4f}, {n_feat} features)")

    print("=" * 130)


def main():
    config_dir = PROJECT_ROOT / "configs"
    config_dir.mkdir(exist_ok=True)

    # Generate configs
    experiments = {}

    # Baseline
    baseline_path = config_dir / "workflow_topn_baseline.yaml"
    baseline_path.write_text(BASELINE_YAML, encoding="utf-8")
    experiments["baseline_158"] = baseline_path

    # TopN variants
    for n in [20, 30, 50, 80]:
        yaml_content = make_topn_yaml(n)
        yaml_path = config_dir / f"workflow_topn_{n}.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        experiments[f"topn_{n}"] = yaml_path

    results = {}
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for name, config_path in experiments.items():
        print(f"\n{'='*80}")
        print(f"RUNNING: {name}")
        print(f"Config: {config_path}")
        print(f"{'='*80}")

        log_path = log_dir / f"topn_{name}.log"
        result = run_qrun(config_path, log_path)

        print(f"  Return code: {result['returncode']}")
        print(f"  Time: {result['seconds']:.0f}s")

        if not result["success"]:
            print(f"  ERROR (last 10 lines):")
            for line in result["error_tail"].split("\n")[-10:]:
                print(f"    {line}")

        # Parse metrics
        result["metrics"] = parse_metrics(str(log_path)) if result["success"] else {}
        results[name] = result

        # Save intermediate
        save_path = PROJECT_ROOT / "outputs" / "topn_comparison.json"
        save_data = {
            k: {
                "success": v.get("success"),
                "seconds": v.get("seconds"),
                "metrics": v.get("metrics", {}),
            }
            for k, v in results.items()
        }
        save_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False))

    # Print comparison
    print_comparison(results)

    # Clean up generated configs
    for name, path in experiments.items():
        if path.name.startswith("workflow_topn_"):
            path.unlink(missing_ok=True)

    print(f"\nResults: {save_path}")
    print(f"Logs: {log_dir}/topn_*.log")


if __name__ == "__main__":
    main()

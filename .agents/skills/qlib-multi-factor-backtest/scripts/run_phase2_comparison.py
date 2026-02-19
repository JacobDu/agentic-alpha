"""Phase 2 & 3: Train LightGBM models and backtest on CSI1000.

Runs two configs side-by-side:
1. Baseline: Alpha158 only (158 factors)
2. Candidate Pool: Alpha158 + selected SFA top factors (~188 factors)

Compares IC, model performance, and portfolio returns with transaction costs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if "--help" in sys.argv or "-h" in sys.argv:
    print("Usage: uv run python .agents/skills/qlib-multi-factor-backtest/scripts/run_phase2_comparison.py")
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

from project_qlib.metrics_standard import canonicalize_metrics
from project_qlib.runtime import init_qlib
from project_qlib.workflow import run_qrun


def run_experiment(name: str, config_path: Path) -> dict:
    """Run a single experiment and return results."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {name}")
    print(f"Config: {config_path}")
    print(f"{'='*80}")

    log_path = PROJECT_ROOT / "outputs" / "logs" / f"phase2_{name}.log"
    result = run_qrun(config_path, log_path)

    print(f"  Return code: {result['returncode']}")
    print(f"  Time: {result['seconds']:.0f}s")
    print(f"  Log: {result['log']}")

    if not result["success"]:
        print(f"  ERROR - last 20 lines:")
        for line in result["error_tail"].split("\n")[-20:]:
            print(f"    {line}")

    return result


def parse_results(log_path: str) -> dict:
    """Parse model metrics from qlib log output."""
    metrics = {}
    log_text = Path(log_path).read_text(encoding="utf-8", errors="ignore")

    import re

    # Parse IC dict: {'IC': 0.036, 'ICIR': 0.302, 'Rank IC': 0.024, 'Rank ICIR': 0.177}
    ic_match = re.search(r"'IC':\s*([-\d.e+]+)", log_text)
    if ic_match:
        metrics["IC"] = float(ic_match.group(1))
    icir_match = re.search(r"'ICIR':\s*([-\d.e+]+)", log_text)
    if icir_match:
        metrics["ICIR"] = float(icir_match.group(1))
    ric_match = re.search(r"'Rank IC':\s*([-\d.e+]+)", log_text)
    if ric_match:
        metrics["Rank_IC"] = float(ric_match.group(1))
    ricir_match = re.search(r"'Rank ICIR':\s*([-\d.e+]+)", log_text)
    if ricir_match:
        metrics["Rank_ICIR"] = float(ricir_match.group(1))

    # Parse excess return sections
    # "excess return without cost" section
    no_cost = re.search(
        r"excess return without cost.*?annualized_return\s+([-\d.e+]+).*?"
        r"information_ratio\s+([-\d.e+]+).*?"
        r"max_drawdown\s+([-\d.e+]+)",
        log_text, re.DOTALL
    )
    if no_cost:
        metrics["excess_ann_ret_no_cost"] = float(no_cost.group(1))
        metrics["IR_no_cost"] = float(no_cost.group(2))
        metrics["max_dd_no_cost"] = float(no_cost.group(3))

    # "excess return with cost" section
    with_cost = re.search(
        r"excess return with cost.*?annualized_return\s+([-\d.e+]+).*?"
        r"information_ratio\s+([-\d.e+]+).*?"
        r"max_drawdown\s+([-\d.e+]+)",
        log_text, re.DOTALL
    )
    if with_cost:
        metrics["excess_ann_ret_with_cost"] = float(with_cost.group(1))
        metrics["IR_with_cost"] = float(with_cost.group(2))
        metrics["max_dd_with_cost"] = float(with_cost.group(3))

    # Benchmark return
    bench = re.search(
        r"benchmark return.*?annualized_return\s+([-\d.e+]+).*?"
        r"information_ratio\s+([-\d.e+]+).*?"
        r"max_drawdown\s+([-\d.e+]+)",
        log_text, re.DOTALL
    )
    if bench:
        metrics["bench_ann_return"] = float(bench.group(1))

    return canonicalize_metrics(metrics, keep_unknown=True)


def main():
    configs = {
        "baseline_alpha158": PROJECT_ROOT / "configs" / "workflow_csi1000_baseline_alpha158.yaml",
        "sfa_top_factors": PROJECT_ROOT / "configs" / "workflow_csi1000_hea_lightgbm.yaml",
    }

    results = {}
    for name, config_path in configs.items():
        result = run_experiment(name, config_path)
        result["metrics"] = parse_results(result["log"]) if result["success"] else {}
        results[name] = result

    # ─── Summary ───
    print("\n" + "=" * 100)
    print("PHASE 2/3 RESULTS COMPARISON")
    print("=" * 100)

    header = f"{'Metric':<40} {'Baseline (Alpha158)':<25} {'SFA Top Factors':<25} {'Delta':<15}"
    print(header)
    print("-" * 105)

    baseline_metrics = results.get("baseline_alpha158", {}).get("metrics", {})
    hea_metrics = results.get("sfa_top_factors", {}).get("metrics", {})

    key_order = [
        "ic_mean",
        "ic_ir",
        "rank_ic_mean",
        "rank_ic_ir",
        "excess_return_annualized_no_cost",
        "information_ratio_no_cost",
        "max_drawdown_no_cost",
        "excess_return_annualized_with_cost",
        "information_ratio_with_cost",
        "max_drawdown_with_cost",
        "benchmark_return_annualized",
    ]
    all_keys = sorted(set(list(baseline_metrics.keys()) + list(hea_metrics.keys())))
    ordered_keys = [k for k in key_order if k in all_keys] + [k for k in all_keys if k not in key_order]
    for key in ordered_keys:
        bv = baseline_metrics.get(key)
        hv = hea_metrics.get(key)
        bv_str = f"{bv:.6f}" if bv is not None else "N/A"
        hv_str = f"{hv:.6f}" if hv is not None else "N/A"
        delta_str = ""
        if bv is not None and hv is not None:
            delta = hv - bv
            pct = (delta / abs(bv) * 100) if bv != 0 else 0
            delta_str = f"{delta:+.6f} ({pct:+.1f}%)"
        print(f"  {key:<38} {bv_str:<25} {hv_str:<25} {delta_str}")

    # Timing
    print(f"\n  Training time:")
    for name in configs:
        t = results.get(name, {}).get("seconds", 0)
        s = "OK" if results.get(name, {}).get("success") else "FAIL"
        print(f"    {name}: {t:.0f}s [{s}]")

    # Save results
    output = {
        name: {
            "success": r.get("success", False),
            "seconds": r.get("seconds", 0),
            "metrics": r.get("metrics", {}),
            "log": r.get("log", ""),
        }
        for name, r in results.items()
    }
    out_path = PROJECT_ROOT / "outputs" / "phase2_comparison.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

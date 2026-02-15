"""Analyze MLflow runs and compare baseline vs custom factors."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml

mlruns = PROJECT_ROOT / "mlruns" / "1"
runs = []
for d in sorted(mlruns.iterdir()):
    if d.is_dir() and (d / "meta.yaml").exists() and d.name != "meta.yaml":
        meta = yaml.safe_load((d / "meta.yaml").read_text())

        metrics = {}
        metrics_dir = d / "metrics"
        if metrics_dir.exists():
            for mf in metrics_dir.iterdir():
                lines = mf.read_text().strip().split("\n")
                if lines:
                    parts = lines[0].split()
                    if parts:
                        metrics[mf.name] = float(parts[1])

        tags = {}
        tags_dir = d / "tags"
        if tags_dir.exists():
            for tf in tags_dir.iterdir():
                tags[tf.name] = tf.read_text().strip()

        runs.append({
            "run_id": d.name,
            "start_time": meta.get("start_time", 0),
            "run_name": tags.get("mlflow.runName", "unknown"),
            "metrics": metrics,
        })

runs.sort(key=lambda x: x["start_time"])

print("=" * 80)
print(f"{'Run Name':<25} {'Run ID':<14} {'IC':>10} {'RankIC':>10} {'ICIR':>10} {'RankICIR':>10}")
print("-" * 80)
for r in runs:
    m = r["metrics"]
    ic = m.get("IC", float("nan"))
    ric = m.get("RankIC", float("nan"))
    icir = m.get("ICIR", float("nan"))
    ricir = m.get("RankICIR", float("nan"))
    print(f"{r['run_name']:<25} {r['run_id'][:12]:<14} {ic:>10.6f} {ric:>10.6f} {icir:>10.6f} {ricir:>10.6f}")

print()
print(f"{'Run Name':<25} {'ann_w_cost':>12} {'ir_w_cost':>12} {'ann_wo_cost':>12} {'ir_wo_cost':>12}")
print("-" * 80)
for r in runs:
    m = r["metrics"]
    awc = m.get("excess_return_with_cost.annualized_return", float("nan"))
    iwc = m.get("excess_return_with_cost.information_ratio", float("nan"))
    awo = m.get("excess_return_without_cost.annualized_return", float("nan"))
    iwo = m.get("excess_return_without_cost.information_ratio", float("nan"))
    print(f"{r['run_name']:<25} {awc:>12.6f} {iwc:>12.6f} {awo:>12.6f} {iwo:>12.6f}")

print()
# Highlight comparison of last 2 runs (CSI1000 baseline vs custom)
csi1000_runs = [r for r in runs if "csi1000" in r["run_name"]]
if len(csi1000_runs) >= 2:
    baseline = csi1000_runs[-2]
    custom = csi1000_runs[-1]
    bm = baseline["metrics"]
    cm = custom["metrics"]
    
    print("=" * 80)
    print("  CSI1000 COMPARISON: Baseline vs Custom Factors")
    print("=" * 80)
    for key in ["IC", "RankIC", "ICIR", "RankICIR",
                "excess_return_with_cost.annualized_return",
                "excess_return_with_cost.information_ratio",
                "excess_return_without_cost.annualized_return",
                "excess_return_without_cost.information_ratio"]:
        bv = bm.get(key, float("nan"))
        cv = cm.get(key, float("nan"))
        diff = cv - bv
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        short_key = key.replace("excess_return_", "").replace(".", "_")
        print(f"  {short_key:<35} Baseline: {bv:>10.6f}  Custom: {cv:>10.6f}  Diff: {diff:>+10.6f} {arrow}")

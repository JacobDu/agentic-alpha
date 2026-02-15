"""HEA experiment runner for CSI1000 factor mining.

Runs baseline (Alpha158) and custom factor (Alpha158CSI1000) experiments
on the CSI1000 universe, then compares results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_qlib.workflow import run_with_fallback


def run_baseline() -> dict:
    """Run Alpha158 baseline on CSI1000."""
    print("=" * 60)
    print("  [1/2] Running CSI1000 Baseline (Alpha158)")
    print("=" * 60)
    result = run_with_fallback(
        primary_model="LightGBM",
        primary_config=PROJECT_ROOT / "configs" / "workflow_csi1000_baseline_lightgbm.yaml",
        fallback_model="XGBoost",
        fallback_config=PROJECT_ROOT / "configs" / "workflow_csi1000_baseline_xgboost.yaml",
        output_json=PROJECT_ROOT / "outputs" / "csi1000_baseline_result.json",
        run_name="csi1000_baseline",
    )
    print(f"  Baseline success: {result['success']}")
    print(f"  Selected model: {result.get('selected_model')}")
    return result


def run_custom() -> dict:
    """Run Alpha158CSI1000 (with custom factors) on CSI1000."""
    print("=" * 60)
    print("  [2/2] Running CSI1000 Custom Factors (Alpha158CSI1000)")
    print("=" * 60)
    result = run_with_fallback(
        primary_model="LightGBM",
        primary_config=PROJECT_ROOT / "configs" / "workflow_csi1000_custom_lightgbm.yaml",
        fallback_model="XGBoost",
        fallback_config=PROJECT_ROOT / "configs" / "workflow_csi1000_custom_xgboost.yaml",
        output_json=PROJECT_ROOT / "outputs" / "csi1000_custom_result.json",
        run_name="csi1000_custom",
    )
    print(f"  Custom success: {result['success']}")
    print(f"  Selected model: {result.get('selected_model')}")
    return result


def main() -> int:
    results = {}

    # Step 1: Baseline
    results["baseline"] = run_baseline()
    if not results["baseline"]["success"]:
        print("\n[ERROR] Baseline failed. Check logs.")
        (PROJECT_ROOT / "outputs" / "csi1000_hea_result.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 1

    # Step 2: Custom factors
    results["custom"] = run_custom()

    # Save combined results
    output_path = PROJECT_ROOT / "outputs" / "csi1000_hea_result.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  HEA Experiment Complete")
    print("=" * 60)
    print(f"  Baseline: {'OK' if results['baseline']['success'] else 'FAIL'}")
    print(f"  Custom:   {'OK' if results['custom']['success'] else 'FAIL'}")
    print(f"  Results:  {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

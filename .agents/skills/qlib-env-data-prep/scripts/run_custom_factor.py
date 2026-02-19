from __future__ import annotations

import argparse
import json
import sys
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

from project_qlib.workflow import check_custom_factor, run_with_fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Run custom-factor workflow with fallback model")
    parser.add_argument(
        "--primary-config",
        default=str(PROJECT_ROOT / "configs" / "workflow_custom_factor_lightgbm_quick.yaml"),
    )
    parser.add_argument(
        "--fallback-config",
        default=str(PROJECT_ROOT / "configs" / "workflow_custom_factor_xgboost_quick.yaml"),
    )
    parser.add_argument("--primary-model", default="LightGBM")
    parser.add_argument("--fallback-model", default="XGBoost")
    parser.add_argument("--run-name", default="custom_factor")
    parser.add_argument("--factor-name", default="CSTM_MOM_5")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "custom_factor_result.json"),
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_with_fallback(
        primary_model=args.primary_model,
        primary_config=Path(args.primary_config),
        fallback_model=args.fallback_model,
        fallback_config=Path(args.fallback_config),
        output_json=output_path,
        run_name=args.run_name,
    )

    if result["success"]:
        factor_check = check_custom_factor(Path(args.primary_config), factor_name=args.factor_name)
        result["factor_check"] = factor_check
        if not factor_check["present"]:
            result["success"] = False

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

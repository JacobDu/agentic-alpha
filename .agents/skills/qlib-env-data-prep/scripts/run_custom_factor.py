from __future__ import annotations

import json
import sys
from pathlib import Path

if "--help" in sys.argv or "-h" in sys.argv:
    print("Usage: uv run python .agents/skills/qlib-env-data-prep/scripts/run_custom_factor.py")
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

from project_qlib.workflow import check_custom_factor, run_with_fallback


def main() -> int:
    output_path = PROJECT_ROOT / "outputs" / "custom_factor_result.json"
    result = run_with_fallback(
        primary_model="LightGBM",
        primary_config=PROJECT_ROOT / "configs" / "workflow_custom_factor_lightgbm_quick.yaml",
        fallback_model="XGBoost",
        fallback_config=PROJECT_ROOT / "configs" / "workflow_custom_factor_xgboost_quick.yaml",
        output_json=output_path,
        run_name="custom_factor",
    )

    if result["success"]:
        factor_check = check_custom_factor(
            PROJECT_ROOT / "configs" / "workflow_custom_factor_lightgbm_quick.yaml",
            factor_name="CSTM_MOM_5",
        )
        result["factor_check"] = factor_check
        if not factor_check["present"]:
            result["success"] = False

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_qlib.data import prepare_investment_data
from project_qlib.runtime import init_qlib
from project_qlib.workflow import check_custom_factor, run_with_fallback


def _run_cmd(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "success": proc.returncode == 0,
    }


def _check_data_access() -> dict:
    from qlib.data import D

    init_qlib()
    sample = D.features(["SH600000"], ["$close", "Ref($close, -1)/$close-1"], start_time="2020-01-02", end_time="2020-03-31")
    return {
        "rows": int(len(sample)),
        "columns": [str(c) for c in sample.columns],
        "success": len(sample) > 0,
    }


def _build_markdown(report: dict) -> str:
    lines = [
        "# Qlib Verification Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Overall success: {report['overall_success']}",
        "",
        "## Environment Checks",
    ]
    for item in report["environment_checks"]:
        lines.append(f"- `{item['cmd']}` => success={item['success']}")

    lines.extend(
        [
            "",
            "## Data Preparation",
            f"- status: {report['data_preparation'].get('status')}",
            f"- used_proxy: {report['data_preparation'].get('used_proxy')}",
            "",
            "## Data Access Check",
            f"- success: {report['data_access']['success']}",
            f"- rows: {report['data_access']['rows']}",
            "",
            "## Official Workflow",
            f"- success: {report['official_workflow']['success']}",
            f"- selected_model: {report['official_workflow']['selected_model']}",
            "",
            "## Custom Factor Workflow",
            f"- success: {report['custom_workflow']['success']}",
            f"- selected_model: {report['custom_workflow']['selected_model']}",
            f"- factor_present: {report['custom_workflow'].get('factor_check', {}).get('present')}",
            f"- factor_non_null_count: {report['custom_workflow'].get('factor_check', {}).get('non_null_count')}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    environment_checks = [
        _run_cmd(["uv", "--version"]),
        _run_cmd(["python3.12", "--version"]),
        _run_cmd(["uv", "run", "python", "-c", "import qlib; print(qlib.__version__)"]),
    ]

    data_preparation = prepare_investment_data(force=False)
    data_access = _check_data_access()

    official_result = run_with_fallback(
        primary_model="LightGBM",
        primary_config=PROJECT_ROOT / "configs" / "workflow_official_lightgbm_quick.yaml",
        fallback_model="XGBoost",
        fallback_config=PROJECT_ROOT / "configs" / "workflow_official_xgboost_quick.yaml",
        output_json=PROJECT_ROOT / "outputs" / "official_result.json",
        run_name="official",
    )

    custom_result = run_with_fallback(
        primary_model="LightGBM",
        primary_config=PROJECT_ROOT / "configs" / "workflow_custom_factor_lightgbm_quick.yaml",
        fallback_model="XGBoost",
        fallback_config=PROJECT_ROOT / "configs" / "workflow_custom_factor_xgboost_quick.yaml",
        output_json=PROJECT_ROOT / "outputs" / "custom_factor_result.json",
        run_name="custom_factor",
    )
    if custom_result["success"]:
        custom_result["factor_check"] = check_custom_factor(
            PROJECT_ROOT / "configs" / "workflow_custom_factor_lightgbm_quick.yaml",
            factor_name="CSTM_MOM_5",
        )
        if not custom_result["factor_check"]["present"]:
            custom_result["success"] = False

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment_checks": environment_checks,
        "data_preparation": data_preparation,
        "data_access": data_access,
        "official_workflow": official_result,
        "custom_workflow": custom_result,
    }

    report["overall_success"] = all(item["success"] for item in environment_checks) and data_access["success"] and official_result["success"] and custom_result["success"]

    json_path = PROJECT_ROOT / "outputs" / "verification_report.json"
    md_path = PROJECT_ROOT / "outputs" / "verification_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

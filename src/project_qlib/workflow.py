from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from qlib.utils import init_instance_by_config

from project_qlib.runtime import PROJECT_ROOT, init_qlib


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = str((PROJECT_ROOT / "src").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing else f"{src_dir}{os.pathsep}{existing}"
    return env


def _resolve_config(config_path: Path) -> Path:
    """Resolve provider_uri placeholder in config YAML.

    If provider_uri is a relative path (no leading /), resolve it relative to PROJECT_ROOT.
    Returns the original path if no changes needed, or a temp file with resolved paths.
    """
    text = config_path.read_text(encoding="utf-8")
    provider_uri_str = str((PROJECT_ROOT / "data" / "qlib" / "cn_data").resolve())

    # Replace relative provider_uri (e.g. "data/qlib/cn_data") with absolute path
    import re
    match = re.search(r'provider_uri:\s*["\']?([^"\'\n]+)["\']?', text)
    if match:
        uri = match.group(1).strip()
        if not uri.startswith("/"):
            # Relative path — resolve against PROJECT_ROOT
            resolved = str((PROJECT_ROOT / uri).resolve())
            text = text.replace(uri, resolved)
        elif uri != provider_uri_str and "data/qlib/cn_data" in uri:
            # Absolute path pointing to wrong location — fix it
            text = text.replace(uri, provider_uri_str)

    if text != config_path.read_text(encoding="utf-8"):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", dir=config_path.parent, delete=False
        )
        tmp.write(text)
        tmp.close()
        return Path(tmp.name)
    return config_path


def run_qrun(config_path: Path, log_path: Path) -> dict[str, Any]:
    start = time.time()
    env = _build_env()
    resolved_config = _resolve_config(config_path)
    cmd = [sys.executable, "-m", "qlib.cli.run", str(resolved_config)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Clean up temp config if created
    if resolved_config != config_path:
        resolved_config.unlink(missing_ok=True)
    elapsed = time.time() - start

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_content = [
        f"$ {' '.join(cmd)}",
        "\n===== STDOUT =====\n",
        proc.stdout,
        "\n===== STDERR =====\n",
        proc.stderr,
    ]
    log_path.write_text("\n".join(log_content), encoding="utf-8")

    error_source = proc.stderr.strip() or proc.stdout.strip()
    error_tail = "\n".join(error_source.splitlines()[-30:]) if error_source else ""
    return {
        "config": str(config_path),
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "seconds": round(elapsed, 3),
        "log": str(log_path),
        "error_tail": error_tail,
    }


def run_with_fallback(
    *,
    primary_model: str,
    primary_config: Path,
    fallback_model: str,
    fallback_config: Path,
    output_json: Path,
    run_name: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    selected_model: str | None = None

    for model_name, config in (
        (primary_model, primary_config),
        (fallback_model, fallback_config),
    ):
        attempt = run_qrun(
            config_path=config,
            log_path=PROJECT_ROOT / "outputs" / "logs" / f"{run_name}_{model_name.lower()}.log",
        )
        attempt["model"] = model_name
        attempts.append(attempt)
        if attempt["success"]:
            selected_model = model_name
            break

    result = {
        "run_name": run_name,
        "success": selected_model is not None,
        "selected_model": selected_model,
        "attempts": attempts,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def check_custom_factor(config_path: Path, factor_name: str = "CSTM_MOM_5") -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    init_qlib()
    dataset = init_instance_by_config(config["task"]["dataset"])
    train_df = dataset.prepare("train")

    factor_present = False
    non_null_count = 0
    if isinstance(train_df, pd.DataFrame):
        if isinstance(train_df.columns, pd.MultiIndex):
            level_values = list(train_df.columns.get_level_values(-1))
            factor_present = factor_name in level_values
            if factor_present:
                factor_df = train_df.xs(factor_name, axis=1, level=-1, drop_level=False)
                non_null_count = int(factor_df.notna().sum().sum())
        else:
            factor_present = factor_name in train_df.columns
            if factor_present:
                non_null_count = int(train_df[factor_name].notna().sum())

    return {
        "factor_name": factor_name,
        "present": factor_present,
        "non_null_count": non_null_count,
        "rows": int(len(train_df)),
    }

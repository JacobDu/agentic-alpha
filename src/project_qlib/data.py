from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from project_qlib.network import download_file, fetch_json
from project_qlib.runtime import PROJECT_ROOT

INVESTMENT_RELEASE_API = "https://api.github.com/repos/chenditc/investment_data/releases/latest"
REQUIRED_DATA_DIRS = ["calendars", "features", "instruments"]


def _find_asset_url(release_json: dict[str, Any], filename: str) -> str:
    assets = release_json.get("assets", [])
    for asset in assets:
        if asset.get("name") == filename and asset.get("browser_download_url"):
            return str(asset["browser_download_url"])
    raise RuntimeError(f"Asset {filename} not found in latest release")


def _validate_data_root(data_root: Path) -> None:
    missing = [name for name in REQUIRED_DATA_DIRS if not (data_root / name).exists()]
    if missing:
        raise RuntimeError(f"Qlib data validation failed, missing: {missing}")


def prepare_investment_data(force: bool = False) -> dict[str, Any]:
    data_dir = PROJECT_ROOT / "data"
    archive_path = data_dir / "qlib_bin.tar.gz"
    qlib_data_root = data_dir / "qlib" / "cn_data"

    if qlib_data_root.exists() and not force:
        _validate_data_root(qlib_data_root)
        return {
            "status": "skipped",
            "message": "Data already exists",
            "archive": str(archive_path),
            "target_dir": str(qlib_data_root),
            "used_proxy": False,
        }

    release_json, release_meta = fetch_json(INVESTMENT_RELEASE_API, timeout=120)
    asset_url = _find_asset_url(release_json, "qlib_bin.tar.gz")
    download_meta = download_file(asset_url, archive_path, timeout=1800)

    if qlib_data_root.exists():
        shutil.rmtree(qlib_data_root)
    qlib_data_root.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "tar",
            "-zxf",
            str(archive_path),
            "-C",
            str(qlib_data_root),
            "--strip-components=1",
        ],
        check=True,
    )
    _validate_data_root(qlib_data_root)

    result = {
        "status": "ok",
        "release_tag": release_json.get("tag_name"),
        "asset_url": asset_url,
        "archive": str(archive_path),
        "target_dir": str(qlib_data_root),
        "release_used_proxy": release_meta.used_proxy,
        "download_used_proxy": download_meta.used_proxy,
        "used_proxy": release_meta.used_proxy or download_meta.used_proxy,
    }
    return result

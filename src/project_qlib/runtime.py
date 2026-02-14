from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROVIDER_URI = (PROJECT_ROOT / "data" / "qlib" / "cn_data").resolve()
REGION = "cn"


def ensure_provider_uri() -> Path:
    if not PROVIDER_URI.exists():
        raise FileNotFoundError(f"Qlib data directory not found: {PROVIDER_URI}")
    return PROVIDER_URI


def init_qlib() -> None:
    import qlib

    ensure_provider_uri()
    qlib.init(provider_uri=str(PROVIDER_URI), region=REGION)

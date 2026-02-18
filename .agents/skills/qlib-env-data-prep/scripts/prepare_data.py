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

from project_qlib.data import prepare_investment_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare qlib data from investment_data release")
    parser.add_argument("--force", action="store_true", help="Redownload and re-extract data")
    args = parser.parse_args()

    result = prepare_investment_data(force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

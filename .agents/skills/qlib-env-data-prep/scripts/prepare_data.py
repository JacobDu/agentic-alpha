from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

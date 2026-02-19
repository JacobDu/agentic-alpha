"""Create/upgrade workflow schema in factor_library.db.

Usage:
    uv run python .agents/skills/qlib-env-data-prep/scripts/migrate_workflow_schema.py
    uv run python .agents/skills/qlib-env-data-prep/scripts/migrate_workflow_schema.py --db data/factor_library.db
"""
from __future__ import annotations

import argparse
import sqlite3
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

from project_qlib.factor_db import FactorDB
from project_qlib.workflow_db import WorkflowDB


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate workflow schema")
    parser.add_argument("--db", help="SQLite db path", default=str(PROJECT_ROOT / "data" / "factor_library.db"))
    args = parser.parse_args()

    db_path = Path(args.db)

    # Ensure base schema exists.
    factor_db = FactorDB(db_path=db_path)
    factor_db.close()

    # Ensure workflow schema exists.
    workflow_db = WorkflowDB(db_path=db_path)
    conn = workflow_db._conn  # noqa: SLF001

    required_objects = [
        "workflow_runs",
        "workflow_sfa_metrics",
        "workflow_mfa_metrics",
        "workflow_decisions",
        "workflow_evidence",
        "factor_similarity",
        "factor_replacements",
        "v_workflow_sfa_rounds",
        "v_workflow_mfa_rounds",
        "v_workflow_round_summary",
    ]

    missing = [name for name in required_objects if not _table_exists(conn, name)]
    workflow_db.close()

    if missing:
        print("[FAIL] Missing workflow objects:")
        for name in missing:
            print(f"  - {name}")
        return 1

    print(f"[OK] Workflow schema ready: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

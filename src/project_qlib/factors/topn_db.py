"""TopN Factor Handler from DB — selects top-N factors by |RankICIR| from factor_library.db.

Unlike the legacy TopNBase which reads from a CSV file, this handler queries
the factor library DB directly, ensuring it includes all HEA-discovered factors.

Usage:
    - DBTopN20: Top 20 factors only
    - DBTopN30: Top 30 factors only
    - DBTopN50: Top 50 factors only
    - DBTopN(custom): set TOPN class attribute
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from qlib.contrib.data.handler import Alpha158

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "factor_library.db"


def _load_topn_from_db(n: int, market: str = "csi1000") -> tuple[list[str], list[str]]:
    """Load top-N factors from DB ranked by |RankICIR|.

    De-duplicates VSUMP/VSUMN (keeps VSUMD only, same information).
    Returns (fields, names) where fields are Qlib expressions.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Factor DB not found: {DB_PATH}")

    db = sqlite3.connect(str(DB_PATH))
    rows = db.execute("""
        SELECT f.name, f.expression, abs(t.rank_icir) as abs_icir
        FROM factors f
        JOIN factor_test_results t ON f.name = t.factor_name
        WHERE t.market = ?
          AND f.expression IS NOT NULL AND f.expression != ''
          AND f.name NOT LIKE 'VSUMP%'
          AND f.name NOT LIKE 'VSUMN%'
        ORDER BY abs(t.rank_icir) DESC
        LIMIT ?
    """, (market, n)).fetchall()
    db.close()

    fields = [r[1] for r in rows]
    names = [r[0] for r in rows]
    return fields, names


class DBTopNBase(Alpha158):
    """Base class for DB-based TopN factor handlers.

    Overrides get_feature_config() to return ONLY the top-N factors,
    NOT the full Alpha158 set. This lets LightGBM combine only the
    most predictive factors.
    """

    TOPN: int = 30
    MARKET: str = "csi1000"

    def get_feature_config(self):
        # Do NOT call super() — we only want TopN factors
        fields, names = _load_topn_from_db(self.TOPN, self.MARKET)
        return fields, names


class DBTopN20(DBTopNBase):
    """Top 20 factors from DB."""
    TOPN = 20


class DBTopN30(DBTopNBase):
    """Top 30 factors from DB."""
    TOPN = 30


class DBTopN50(DBTopNBase):
    """Top 50 factors from DB."""
    TOPN = 50


class DBTopN80(DBTopNBase):
    """Top 80 factors from DB."""
    TOPN = 80


class DBTopN100(DBTopNBase):
    """Top 100 factors from DB."""
    TOPN = 100

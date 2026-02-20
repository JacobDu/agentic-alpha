"""TopN Factor Handler from DB — selects top-N factors by |RankICIR| from factor_library.db.

Unlike the legacy TopNBase which reads from a CSV file, this handler queries
the factor library DB directly, ensuring it includes all HEA-discovered factors.

Usage:
    - DBTopN20: Top 20 factors only
    - DBTopN30: Top 30 factors only
    - DBTopN50: Top 50 factors only
    - DBTopN(custom): set TOPN class attribute
    - DBAlpha158PlusTopN: Alpha158 base + top-N diverse custom factors
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


def _load_diverse_topn_from_db(n: int, market: str = "csi1000",
                                max_per_cat: int = 3) -> tuple[list[str], list[str]]:
    """Load top-N factors with category diversity constraint.

    Each category can contribute at most max_per_cat factors.
    Baseline factors are excluded (they're already in Alpha158).
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Factor DB not found: {DB_PATH}")

    db = sqlite3.connect(str(DB_PATH))
    rows = db.execute("""
        SELECT f.name, f.expression, f.category, abs(t.rank_icir) as abs_icir
        FROM factors f
        JOIN factor_test_results t ON f.name = t.factor_name
        WHERE t.market = ?
          AND f.expression IS NOT NULL AND f.expression != ''
          AND f.name NOT LIKE 'VSUMP%'
          AND f.name NOT LIKE 'VSUMN%'
          AND f.status != 'Baseline'
        ORDER BY abs(t.rank_icir) DESC
    """, (market,)).fetchall()
    db.close()

    fields, names = [], []
    cat_count: dict[str, int] = {}
    for name, expr, cat, _ in rows:
        c = cat_count.get(cat, 0)
        if c >= max_per_cat:
            continue
        fields.append(expr)
        names.append(name)
        cat_count[cat] = c + 1
        if len(fields) >= n:
            break

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


class DBAlpha158PlusTopN(Alpha158):
    """Alpha158 (158 baseline factors) + top-N diverse custom factors.

    This handler combines the full Alpha158 feature set as a broad base,
    plus the best custom factors selected with category diversity constraints.
    This avoids the multicollinearity problem of using only custom factors.
    """

    TOPN: int = 30
    MARKET: str = "csi1000"
    MAX_PER_CAT: int = 3

    def get_feature_config(self):
        # Get Alpha158 base features
        base_fields, base_names = super().get_feature_config()

        # Get diverse custom factors
        custom_fields, custom_names = _load_diverse_topn_from_db(
            self.TOPN, self.MARKET, self.MAX_PER_CAT
        )

        # Combine (deduplicate by name)
        existing_names = set(base_names)
        for f, n in zip(custom_fields, custom_names):
            if n not in existing_names:
                base_fields.append(f)
                base_names.append(n)
                existing_names.add(n)

        return base_fields, base_names

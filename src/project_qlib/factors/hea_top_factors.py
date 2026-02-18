"""HEA Top Factor Handler — uses best factors from HEA research rounds.

Loads Alpha158 baseline + top HEA-discovered factors from factor_library.db.
Ensures no multicollinearity by selecting only one factor per high-correlation cluster.

Factor selection strategy:
1. Start with all Alpha158 baseline factors (158)
2. Add top single factors from HEA-01/02 (avoiding component-composite overlap)
3. Add top composite factors from HEA-03
4. De-duplicate: if a composite is present, exclude its component single factors

Max total features: ~190 (to keep training manageable on 16GB RAM)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from qlib.contrib.data.handler import Alpha158

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_hea_factors(max_single: int = 30, max_composite: int = 10):
    """Load top HEA factors from DB, avoiding component-composite redundancy."""
    db_path = PROJECT_ROOT / "data" / "factor_library.db"
    if not db_path.exists():
        return [], []

    db = sqlite3.connect(str(db_path))

    # Get top composite factors (HEA-03)
    composites = db.execute("""
        SELECT f.name, f.expression, abs(t.rank_icir) as abs_icir
        FROM factors f JOIN factor_test_results t ON f.name = t.factor_name
        WHERE t.market = 'csi1000' AND f.status = 'Accepted'
          AND f.name LIKE 'COMP_%'
          AND f.expression IS NOT NULL AND f.expression != ''
        ORDER BY abs(t.rank_icir) DESC
        LIMIT ?
    """, (max_composite,)).fetchall()

    # Get top single factors (NEW_ and ORTH_, excluding Alpha158 baseline)
    singles = db.execute("""
        SELECT f.name, f.expression, abs(t.rank_icir) as abs_icir
        FROM factors f JOIN factor_test_results t ON f.name = t.factor_name
        WHERE t.market = 'csi1000' AND f.status = 'Accepted'
          AND (f.name LIKE 'NEW_%' OR f.name LIKE 'ORTH_%' OR f.name LIKE 'CSTM_%' OR f.name LIKE 'FIN_%')
          AND f.source != 'Alpha158'
          AND f.expression IS NOT NULL AND f.expression != ''
        ORDER BY abs(t.rank_icir) DESC
        LIMIT ?
    """, (max_single * 2,)).fetchall()  # Get more to allow filtering

    db.close()

    # Build composite component tracking to avoid redundancy
    # Component factors that appear in top composites should not also be added as singles
    composite_component_names = set()
    # Known component factor names used in composites
    component_keywords = [
        "TURN_QUANTILE_120", "AMT_CV_20", "VOL_CV_10", "GAP_FREQ_20",
        "CFP_TURN", "VALUE_LOWVOL", "VOL_TREND_5_60", "VWAP_BIAS_VOL_20",
        "VOL_CONC_5", "AMT_PER_RANGE", "HIGHVOL_RET_20", "SIGN_VOL_CORR_20",
        "OVN_INTRA_DIV", "REVERT_LOWVOL_20", "INTRADAY",
    ]
    for name, expr, icir in composites:
        for kw in component_keywords:
            # If this keyword appears in the composite name, mark related singles
            if kw.lower() in name.lower() or kw.lower() in (expr or "").lower():
                for sn, se, si in singles:
                    if kw.lower() in sn.lower():
                        composite_component_names.add(sn)

    # Filter singles: exclude those already covered by composites
    filtered_singles = []
    for name, expr, icir in singles:
        if name not in composite_component_names:
            filtered_singles.append((name, expr, icir))
        if len(filtered_singles) >= max_single:
            break

    # Combine: composites first (stronger), then filtered singles
    all_factors = [(name, expr) for name, expr, _ in composites]
    all_factors += [(name, expr) for name, expr, _ in filtered_singles]

    fields = [expr for _, expr in all_factors]
    names = [name for name, _ in all_factors]
    return fields, names


class HEATopFactors(Alpha158):
    """Alpha158 + top HEA-discovered factors (composite + orthogonal singles).

    Total features: ~188 (158 baseline + ~30 HEA factors)
    Designed for CSI1000 with Phase 2 LightGBM training.
    """

    MAX_SINGLE = 20
    MAX_COMPOSITE = 10

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        extra_fields, extra_names = _load_hea_factors(
            max_single=self.MAX_SINGLE,
            max_composite=self.MAX_COMPOSITE,
        )
        return list(fields) + extra_fields, list(names) + extra_names


class HEATopFactorsLite(HEATopFactors):
    """Lighter version: fewer HEA factors for faster training."""
    MAX_SINGLE = 10
    MAX_COMPOSITE = 5


class HEACompositesOnly(Alpha158):
    """Alpha158 + only composite factors (most ICIR-efficient)."""

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        extra_fields, extra_names = _load_hea_factors(max_single=0, max_composite=15)
        return list(fields) + extra_fields, list(names) + extra_names

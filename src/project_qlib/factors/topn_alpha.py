"""TopN Factor Handler — selects top-N factors from unified ranking.

Reads the pre-computed csiall_unified_factor_ranking.csv, deduplicates
VSUMP/VSUMN (keeping VSUMD), and returns only the top-N factor expressions.
Supports N=20, N=30, N=50 via subclasses.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from qlib.contrib.data.handler import Alpha158

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RANKING_CSV = PROJECT_ROOT / "outputs" / "csiall_unified_factor_ranking.csv"

# Custom factor name -> Qlib expression mapping
CUSTOM_EXPR_MAP = {
    "CSTM_AMT_SURGE_20": "$amount / (Mean($amount, 20) + 1e-8)",
    "CSTM_AMT_SURGE_60": "$amount / (Mean($amount, 60) + 1e-8)",
    "CSTM_VOL_SURGE_5": "$volume / (Mean($volume, 5) + 1e-8)",
    "CSTM_VOL_CV_10": "Std($volume, 10) / (Mean($volume, 10) + 1e-8)",
    "CSTM_AMT_CV_20": "Std($amount, 20) / (Mean($amount, 20) + 1e-8)",
    "CSTM_VOL_RATIO_5_20": "Mean($volume, 5) / (Mean($volume, 20) + 1e-8)",
    "CSTM_VWAP_BIAS_5": "$close / Mean($vwap, 5) - 1",
    "CSTM_VWAP_BIAS_10": "$close / Mean($vwap, 10) - 1",
    "CSTM_VWAP_BIAS_20": "$close / Mean($vwap, 20) - 1",
    "CSTM_VWAP_BIAS_1D": "$close / $vwap - 1",
    "CSTM_VWAP_VOL_CORR_10": "Corr($close/$vwap, $volume/Ref($volume, 1), 10)",
    "CSTM_VWAP_VOL_CORR_20": "Corr($close/$vwap, $volume/Ref($volume, 1), 20)",
    "CSTM_RANGE_1D": "($high - $low) / ($close + 1e-8)",
    "CSTM_RANGE_RATIO_5_20": "Mean(($high - $low) / ($close + 1e-8), 5) / (Mean(($high - $low) / ($close + 1e-8), 20) + 1e-8)",
    "CSTM_RANGE_RATIO_5_60": "Mean(($high - $low) / ($close + 1e-8), 5) / (Mean(($high - $low) / ($close + 1e-8), 60) + 1e-8)",
    "CSTM_CLOSE_POS": "($close - $low) / ($high - $low + 1e-8)",
    "CSTM_CLOSE_POS_MA5": "Mean(($close - $low) / ($high - $low + 1e-8), 5)",
    "CSTM_SHADOW_RATIO": "($high - $close) / ($close - $low + 1e-8)",
    "CSTM_RANGE_VOL_10": "Std(($high-$low)/($close+1e-8), 10)",
    "CSTM_GAP_1D": "$open / Ref($close, 1) - 1",
    "CSTM_GAP_MA_5": "Mean($open / Ref($close, 1) - 1, 5)",
    "CSTM_GAP_MA_10": "Mean($open / Ref($close, 1) - 1, 10)",
    "CSTM_GAP_STD_10": "Std($open / Ref($close, 1) - 1, 10)",
    "CSTM_RET_ACCEL_1": "$close/Ref($close, 1) - Ref($close, 1)/Ref($close, 2)",
    "CSTM_MOM_DIFF_5_20": "Mean($close/Ref($close, 1) - 1, 5) - Mean($close/Ref($close, 1) - 1, 20)",
    "CSTM_REVERT_1": "Ref($close, 1)/$close - 1",
    "CSTM_REVERT_3": "Ref($close, 3)/$close - 1",
    "CSTM_REVERT_5": "Ref($close, 5)/$close - 1",
    "CSTM_REVERT_10": "Ref($close, 10)/$close - 1",
    "CSTM_REVERT_20": "Ref($close, 20)/$close - 1",
    "CSTM_PV_CORR_5": "Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 5)",
    "CSTM_PV_CORR_10": "Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 10)",
    "CSTM_PV_CORR_20": "Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 20)",
    "CSTM_PA_CORR_10": "Corr($close/Ref($close, 1) - 1, $amount/Ref($amount, 1) - 1, 10)",
    "CSTM_SKEW_20": "Mean(Power($close/Ref($close, 1) - 1, 3), 20) / (Power(Std($close/Ref($close, 1) - 1, 20), 3) + 1e-12)",
    "CSTM_SKEW_60": "Mean(Power($close/Ref($close, 1) - 1, 3), 60) / (Power(Std($close/Ref($close, 1) - 1, 60), 3) + 1e-12)",
    "CSTM_AMT_WTRET_10": "Mean(($close/Ref($close, 1) - 1) * $amount, 10) / (Mean($amount, 10) + 1e-8)",
    "CSTM_AMT_WTRET_20": "Mean(($close/Ref($close, 1) - 1) * $amount, 20) / (Mean($amount, 20) + 1e-8)",
    "CSTM_MA_BIAS_5": "$close / Mean($close, 5) - 1",
    "CSTM_MA_BIAS_10": "$close / Mean($close, 10) - 1",
    "CSTM_MA_BIAS_20": "$close / Mean($close, 20) - 1",
    "CSTM_MA_BIAS_60": "$close / Mean($close, 60) - 1",
    "CSTM_MA_CROSS_5_20": "Mean($close, 5) / Mean($close, 20) - 1",
}


def _get_topn_factors(n: int) -> tuple[list[str], list[str]]:
    """Read unified ranking, deduplicate, return top-N (fields, names)."""
    ranking = pd.read_csv(RANKING_CSV)

    # Deduplicate: remove VSUMP and VSUMN (keep VSUMD which encodes the same info)
    ranking = ranking[~ranking["factor"].str.match(r"^VSUMP|^VSUMN")]
    ranking = ranking.head(n)

    # Build Alpha158 name->expression lookup
    h = Alpha158.__new__(Alpha158)
    a158_fields, a158_names = h.get_feature_config()
    a158_expr_map = dict(zip(a158_names, a158_fields))

    fields = []
    names = []
    for _, row in ranking.iterrows():
        fname = row["factor"]
        if row["source"] == "Alpha158":
            if fname in a158_expr_map:
                fields.append(a158_expr_map[fname])
                names.append(fname)
            else:
                print(f"WARNING: Alpha158 factor {fname} not found in expression map")
        else:  # Custom
            if fname in CUSTOM_EXPR_MAP:
                fields.append(CUSTOM_EXPR_MAP[fname])
                names.append(fname)
            else:
                print(f"WARNING: Custom factor {fname} not found in expression map")

    return fields, names


class TopNBase(Alpha158):
    """Base class for TopN factor handlers. Subclass and set TOPN."""

    TOPN: int = 20

    def get_feature_config(self):
        fields, names = _get_topn_factors(self.TOPN)
        return fields, names


class TopN20(TopNBase):
    """Top 20 factors from unified ranking (deduplicated)."""
    TOPN = 20


class TopN30(TopNBase):
    """Top 30 factors from unified ranking (deduplicated)."""
    TOPN = 30


class TopN50(TopNBase):
    """Top 50 factors from unified ranking (deduplicated)."""
    TOPN = 50

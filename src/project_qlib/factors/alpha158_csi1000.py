"""Custom factors designed for CSI1000 (small/mid-cap) universe.

Design rationale:
- CSI1000 stocks exhibit stronger mean-reversion, wider intraday ranges,
  and more pronounced volume-price patterns than large-cap indices.
- We add factors capturing VWAP divergence, amount surges, intraday
  positioning, overnight gaps, range dynamics, return acceleration,
  and volume-price interactions around VWAP.
"""

from __future__ import annotations

from qlib.contrib.data.handler import Alpha158


class Alpha158CSI1000(Alpha158):
    """Alpha158 + 7 custom factors targeting small/mid-cap characteristics."""

    # ---- custom factor definitions ------------------------------------------------
    # Each tuple: (expression, name)
    CUSTOM_FACTORS = [
        # 1. VWAP Bias (5d): close vs recent avg VWAP – mean-reversion pressure
        ("$close / Mean($vwap, 5) - 1", "CSTM_VWAP_BIAS_5"),
        # 2. Amount Surge: relative daily trading amount vs 20d mean – unusual activity
        ("$amount / (Mean($amount, 20) + 1e-8)", "CSTM_AMT_SURGE"),
        # 3. Close Position: where close sits in today's range – intraday sentiment
        ("($close - $low) / ($high - $low + 1e-8)", "CSTM_CLOSE_POS"),
        # 4. Overnight Gap MA (10d): avg opening gap – overnight information arrival
        ("Mean($open / Ref($close, 1) - 1, 10)", "CSTM_GAP_MA_10"),
        # 5. Range Expansion Ratio: 5d vs 20d avg relative range – breakout signal
        (
            "Mean(($high - $low) / ($close + 1e-8), 5)"
            " / (Mean(($high - $low) / ($close + 1e-8), 20) + 1e-8)",
            "CSTM_RANGE_RATIO",
        ),
        # 6. Return Acceleration: today's return minus 5-day-ago return – momentum exhaustion
        ("$close/Ref($close, 1) - Ref($close, 1)/Ref($close, 2)", "CSTM_PRICE_ACCEL"),
        # 7. VWAP-Volume Correlation (10d): price position around VWAP vs volume change
        ("Corr($close/$vwap, $volume/Ref($volume, 1), 10)", "CSTM_VWAP_VOL_CORR"),
    ]

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        extra_fields = [expr for expr, _ in self.CUSTOM_FACTORS]
        extra_names = [name for _, name in self.CUSTOM_FACTORS]
        return list(fields) + extra_fields, list(names) + extra_names

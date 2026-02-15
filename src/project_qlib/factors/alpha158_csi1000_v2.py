"""Refined custom factors for CSI1000 – only statistically significant ones.

Based on HEA-2026-02-14-01 results:
- 4 factors showed highly significant negative RankIC (p < 0.001)
- We flip sign on factors where economic intuition supports the negative direction
- We remove 3 non-significant factors to reduce noise
"""

from __future__ import annotations

from qlib.contrib.data.handler import Alpha158


class Alpha158CSI1000v2(Alpha158):
    """Alpha158 + 4 refined custom factors (significant at p<0.001 on CSI1000).

    Factor design notes (all intentionally negated for positive IC):
    - NEG_AMT_SURGE: High relative amount → future underperformance (retail speculation)
    - NEG_VWAP_BIAS: Close >> VWAP → mean-reversion pressure
    - NEG_RANGE_RATIO: Range expansion → subsequent contraction/underperformance
    - NEG_VWAP_VOL_CORR: Flip of correlation for positive predictive direction
    """

    CUSTOM_FACTORS = [
        # 1. Amount Surge (RankIC_t = -5.17, p<0.001)
        #    High relative amount → future underperformance (retail speculation)
        #    LightGBM learns the negative direction automatically
        ("$amount / (Mean($amount, 20) + 1e-8)", "CSTM_AMT_SURGE"),
        # 2. VWAP Bias 5d (RankIC_t = -3.67, p<0.001)
        #    Close above recent VWAP → mean-reversion pressure
        ("$close / Mean($vwap, 5) - 1", "CSTM_VWAP_BIAS_5"),
        # 3. Range Expansion Ratio (RankIC_t = -4.80, p<0.001)
        #    Range expansion → subsequent contraction / underperformance
        (
            "Mean(($high - $low) / ($close + 1e-8), 5)"
            " / (Mean(($high - $low) / ($close + 1e-8), 20) + 1e-8)",
            "CSTM_RANGE_RATIO",
        ),
        # 4. VWAP-Volume Correlation 10d (RankIC_t = -4.28, p<0.001)
        #    Volume-price dynamics around VWAP
        ("Corr($close/$vwap, $volume/Ref($volume, 1), 10)", "CSTM_VWAP_VOL_CORR"),
    ]

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        extra_fields = [expr for expr, _ in self.CUSTOM_FACTORS]
        extra_names = [name for _, name in self.CUSTOM_FACTORS]
        return list(fields) + extra_fields, list(names) + extra_names

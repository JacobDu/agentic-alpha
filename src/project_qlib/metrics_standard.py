"""Canonical metric schema and alias helpers.

This module defines one canonical naming standard for research metrics.
Legacy keys are still supported via alias mapping for backward compatibility.
"""
from __future__ import annotations

from typing import Any, Mapping

METRIC_SCHEMA_VERSION = "v1"

# Canonical metric names used across scripts/docs.
CANONICAL_KEYS: tuple[str, ...] = (
    "ic_mean",
    "ic_ir",
    "rank_ic_mean",
    "rank_ic_ir",
    "n_days",
    "n_features",
    "excess_return_daily_no_cost",
    "excess_return_daily_with_cost",
    "excess_return_annualized_no_cost",
    "excess_return_annualized_with_cost",
    "information_ratio_no_cost",
    "information_ratio_with_cost",
    "max_drawdown_no_cost",
    "max_drawdown_with_cost",
    "daily_turnover",
    "total_cost_pct",
    "benchmark_return_annualized",
)

# canonical -> accepted aliases (legacy output keys)
CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "ic_mean": ("IC",),
    "ic_ir": ("ICIR",),
    "rank_ic_mean": ("Rank_IC", "Rank IC", "rank_ic_mean"),
    "rank_ic_ir": ("Rank_ICIR", "Rank ICIR", "rank_icir"),
    "n_days": ("n_days",),
    "n_features": ("n_features",),
    "excess_return_daily_no_cost": ("excess_return_daily_no_cost",),
    "excess_return_daily_with_cost": ("excess_return_daily_with_cost",),
    "excess_return_annualized_no_cost": (
        "excess_return_annualized_no_cost",
        "excess_ann_ret_no_cost",
        "ann_ret_no_cost",
        "1day.excess_return_without_cost.annualized_return",
    ),
    "excess_return_annualized_with_cost": (
        "excess_return_annualized_with_cost",
        "excess_return_with_cost",  # workflow table historical naming
        "excess_ann_ret_with_cost",
        "ann_ret_with_cost",
        "1day.excess_return_with_cost.annualized_return",
    ),
    "information_ratio_no_cost": (
        "information_ratio_no_cost",
        "IR_no_cost",
        "1day.excess_return_without_cost.information_ratio",
    ),
    "information_ratio_with_cost": (
        "information_ratio_with_cost",
        "ir_with_cost",  # workflow table historical naming
        "IR_with_cost",
        "1day.excess_return_with_cost.information_ratio",
    ),
    "max_drawdown_no_cost": (
        "max_drawdown_no_cost",
        "max_dd_no_cost",
        "1day.excess_return_without_cost.max_drawdown",
    ),
    "max_drawdown_with_cost": (
        "max_drawdown_with_cost",
        "max_drawdown",  # workflow table historical naming
        "max_dd_with_cost",
        "1day.excess_return_with_cost.max_drawdown",
    ),
    "daily_turnover": ("daily_turnover", "turnover"),
    "total_cost_pct": ("total_cost_pct",),
    "benchmark_return_annualized": (
        "benchmark_return_annualized",
        "bench_ann_return",
        "1day.benchmark_return.annualized_return",
    ),
}


def _build_alias_to_canonical() -> dict[str, str]:
    out: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        out[canonical] = canonical
        for alias in aliases:
            out[alias] = canonical
    return out


ALIAS_TO_CANONICAL = _build_alias_to_canonical()


def canonicalize_metrics(
    raw: Mapping[str, Any],
    *,
    keep_unknown: bool = True,
    include_schema_version: bool = True,
) -> dict[str, Any]:
    """Return a canonicalized metric dict.

    - Keeps only canonical keys (+ optional unknown keys)
    - Resolves legacy aliases into canonical names
    - Last-write-wins if both canonical and alias exist
    """
    canonical: dict[str, Any] = {}
    unknown: dict[str, Any] = {}

    for key, value in raw.items():
        mapped = ALIAS_TO_CANONICAL.get(key)
        if mapped is None:
            if keep_unknown:
                unknown[key] = value
            continue
        canonical[mapped] = value

    out = {}
    if keep_unknown:
        out.update(unknown)
    out.update(canonical)
    if include_schema_version:
        out["metric_schema_version"] = METRIC_SCHEMA_VERSION
    return out


def get_metric(metrics: Mapping[str, Any], canonical_key: str, default: Any = None) -> Any:
    """Read one metric using canonical key with alias fallback."""
    if canonical_key in metrics:
        return metrics[canonical_key]
    for alias in CANONICAL_ALIASES.get(canonical_key, ()):
        if alias in metrics:
            return metrics[alias]
    return default

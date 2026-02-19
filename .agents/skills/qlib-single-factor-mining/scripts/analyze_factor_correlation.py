"""Analyze top-factor correlations and optionally persist similarity snapshots.

This script is reusable for SFA Retrieve/Evaluate steps:
1. Pull top factors from factor_library.db.
2. Compute average cross-sectional Spearman correlations over sampled dates.
3. Export matrix + pairwise similarity stats.
4. Optionally upsert stats to workflow factor_similarity table.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Cannot locate project root (pyproject.toml not found)")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_qlib.runtime import init_qlib
from project_qlib.workflow_db import WorkflowDB
from qlib.data import D


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_top_factors(
    db_path: Path,
    market: str,
    top_n: int,
    statuses: tuple[str, ...] = ("Accepted", "Baseline"),
) -> list[dict[str, Any]]:
    """Load top factors by |rank_icir| and deduplicate by factor name."""
    placeholders = ",".join("?" for _ in statuses)
    query = f"""
        SELECT
            f.name,
            f.expression,
            f.category,
            f.status,
            t.rank_ic_mean,
            t.rank_icir,
            t.rank_ic_t,
            t.test_end
        FROM factors f
        JOIN factor_test_results t ON f.name = t.factor_name
        WHERE t.market = ? AND f.status IN ({placeholders})
        ORDER BY ABS(COALESCE(t.rank_icir, 0)) DESC,
                 COALESCE(t.test_end, '') DESC
    """
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(query, (market, *statuses)).fetchall()
    conn.close()

    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row[0]
        if name in deduped:
            continue
        deduped[name] = {
            "name": row[0],
            "expression": row[1],
            "category": row[2],
            "status": row[3],
            "rank_ic_mean": row[4],
            "rank_icir": row[5],
            "rank_ic_t": row[6],
            "test_end": row[7],
        }
        if len(deduped) >= top_n:
            break
    return list(deduped.values())


def _load_factor_panel(
    factors: list[dict[str, Any]],
    market: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    instruments = D.instruments(market)
    series_by_factor: dict[str, pd.DataFrame] = {}

    for factor in factors:
        name = factor["name"]
        expr = factor.get("expression")
        if not expr:
            continue
        try:
            df = D.features(instruments, [expr], start_time=start, end_time=end)
            df.columns = [name]
            series_by_factor[name] = df
        except Exception as exc:
            print(f"[WARN] Skip factor {name}: {exc}")

    if len(series_by_factor) < 2:
        return pd.DataFrame()

    panel = pd.concat(series_by_factor.values(), axis=1)
    return panel


def compute_pairwise_similarity(
    panel: pd.DataFrame,
    sample_days: int,
    min_cross_section: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Compute avg correlation matrix + per-pair similarity snapshots."""
    if panel.empty:
        return pd.DataFrame(), [], 0

    factor_names = panel.columns.tolist()
    n = len(factor_names)
    dates = panel.index.get_level_values("datetime").unique()
    if len(dates) == 0:
        return pd.DataFrame(), [], 0

    rng = np.random.default_rng(seed)
    if sample_days > 0 and len(dates) > sample_days:
        sampled_dates = rng.choice(np.array(dates), size=sample_days, replace=False)
    else:
        sampled_dates = np.array(dates)

    signed_sum = np.zeros((n, n), dtype=float)
    signed_count = np.zeros((n, n), dtype=int)
    abs_samples: dict[tuple[int, int], list[float]] = {}
    valid_days = 0

    for dt in sampled_dates:
        try:
            cross = panel.xs(dt, level="datetime")
        except KeyError:
            continue
        if len(cross) < min_cross_section:
            continue

        ranked = cross.rank(method="average")
        corr = ranked.corr(method="spearman").to_numpy()
        if corr.shape != (n, n):
            continue

        any_valid = False
        for i in range(n):
            for j in range(i + 1, n):
                value = corr[i, j]
                if np.isnan(value):
                    continue
                any_valid = True
                signed_sum[i, j] += value
                signed_sum[j, i] += value
                signed_count[i, j] += 1
                signed_count[j, i] += 1
                abs_samples.setdefault((i, j), []).append(abs(float(value)))

        if any_valid:
            valid_days += 1

    if valid_days == 0:
        return pd.DataFrame(), [], 0

    avg = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(avg, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            if signed_count[i, j] <= 0:
                continue
            mean_signed = signed_sum[i, j] / signed_count[i, j]
            avg[i, j] = mean_signed
            avg[j, i] = mean_signed

    corr_matrix = pd.DataFrame(avg, index=factor_names, columns=factor_names)

    pairs: list[dict[str, Any]] = []
    for (i, j), values in abs_samples.items():
        if not values:
            continue
        rho_mean_abs = float(np.mean(values))
        rho_p95_abs = float(np.percentile(values, 95))
        sample_count = len(values)
        rho_signed_mean = float(avg[i, j]) if not np.isnan(avg[i, j]) else None
        pairs.append(
            {
                "factor_a": factor_names[i],
                "factor_b": factor_names[j],
                "rho_mean_abs": rho_mean_abs,
                "rho_p95_abs": rho_p95_abs,
                "rho_signed_mean": rho_signed_mean,
                "sample_days": sample_count,
            }
        )

    pairs.sort(key=lambda row: row["rho_mean_abs"], reverse=True)
    return corr_matrix, pairs, valid_days


def _print_summary(factors: list[dict[str, Any]], pairs: list[dict[str, Any]], min_report_rho: float) -> None:
    print(f"Loaded factors: {len(factors)}")
    if not factors:
        return

    category_counts: dict[str, int] = {}
    for factor in factors:
        category = factor.get("category") or "Unknown"
        category_counts[category] = category_counts.get(category, 0) + 1

    print("Category distribution:")
    for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  - {category}: {count}")

    high_pairs = [row for row in pairs if row["rho_mean_abs"] >= min_report_rho]
    print(f"Pairs with mean |rho| >= {min_report_rho:.2f}: {len(high_pairs)}")

    if pairs:
        print("Top correlated pairs (by mean |rho|):")
        for row in pairs[:15]:
            signed = row["rho_signed_mean"]
            signed_text = "N/A" if signed is None else f"{signed:+.3f}"
            print(
                "  - "
                f"{row['factor_a']} vs {row['factor_b']}: "
                f"mean|rho|={row['rho_mean_abs']:.3f}, "
                f"p95|rho|={row['rho_p95_abs']:.3f}, "
                f"signed_mean={signed_text}, "
                f"days={row['sample_days']}"
            )


def _write_outputs(corr_matrix: pd.DataFrame, pairs: list[dict[str, Any]], corr_out: Path, pairs_out: Path) -> None:
    _ensure_parent(corr_out)
    _ensure_parent(pairs_out)

    corr_matrix.to_csv(corr_out)
    pd.DataFrame(pairs).to_csv(pairs_out, index=False)

    print(f"[OK] Correlation matrix: {corr_out}")
    print(f"[OK] Pair similarity stats: {pairs_out}")


def _write_similarity_db(
    db_path: Path,
    pairs: list[dict[str, Any]],
    market: str,
    window: str,
    source_round_id: str | None,
    notes: str | None,
    min_db_rho: float,
) -> int:
    wdb = WorkflowDB(db_path=str(db_path))
    written = 0
    try:
        for row in pairs:
            if row["rho_mean_abs"] < min_db_rho:
                continue
            wdb.upsert_similarity(
                factor_a=row["factor_a"],
                factor_b=row["factor_b"],
                market=market,
                window=window,
                rho_mean_abs=row["rho_mean_abs"],
                rho_p95_abs=row["rho_p95_abs"],
                sample_days=row["sample_days"],
                source_round_id=source_round_id,
                notes=notes,
            )
            written += 1
    finally:
        wdb.close()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze factor correlations and optionally persist similarity snapshots")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "factor_library.db"), help="Path to factor library DB")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--sample-days", type=int, default=120, help="Max dates sampled for correlation")
    parser.add_argument("--min-cross-section", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-report-rho", type=float, default=0.60)
    parser.add_argument(
        "--corr-output",
        default=str(PROJECT_ROOT / "outputs" / "factor_correlation_matrix.csv"),
        help="CSV path for correlation matrix",
    )
    parser.add_argument(
        "--pairs-output",
        default=str(PROJECT_ROOT / "outputs" / "factor_similarity_pairs.csv"),
        help="CSV path for pairwise similarity",
    )
    parser.add_argument("--write-db", action="store_true", help="Persist similarity snapshots into factor_similarity table")
    parser.add_argument("--window", default="global")
    parser.add_argument("--round-id", help="Optional source round id")
    parser.add_argument("--notes", help="Optional note saved with similarity records")
    parser.add_argument("--min-db-rho", type=float, default=0.50, help="Only write pairs with rho_mean_abs >= threshold")
    args = parser.parse_args()

    db_path = Path(args.db)
    corr_out = Path(args.corr_output)
    pairs_out = Path(args.pairs_output)

    factors = get_top_factors(db_path=db_path, market=args.market, top_n=args.top_n)
    if len(factors) < 2:
        print("Insufficient factors to compute correlation (need >=2).")
        return 1

    init_qlib()
    panel = _load_factor_panel(factors=factors, market=args.market, start=args.start, end=args.end)
    if panel.empty:
        print("No factor panel loaded. Check expressions/data availability.")
        return 1

    corr_matrix, pairs, valid_days = compute_pairwise_similarity(
        panel=panel,
        sample_days=args.sample_days,
        min_cross_section=args.min_cross_section,
        seed=args.seed,
    )
    if corr_matrix.empty or not pairs:
        print("Failed to compute valid correlation snapshots.")
        return 1

    print(f"Valid sampled days: {valid_days}")
    _print_summary(factors=factors, pairs=pairs, min_report_rho=args.min_report_rho)
    _write_outputs(corr_matrix=corr_matrix, pairs=pairs, corr_out=corr_out, pairs_out=pairs_out)

    if args.write_db:
        written = _write_similarity_db(
            db_path=db_path,
            pairs=pairs,
            market=args.market,
            window=args.window,
            source_round_id=args.round_id,
            notes=args.notes,
            min_db_rho=args.min_db_rho,
        )
        print(f"[OK] Similarity rows written: {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

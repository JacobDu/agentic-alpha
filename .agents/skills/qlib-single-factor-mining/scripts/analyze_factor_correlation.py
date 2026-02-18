"""Analyze factor distribution and cross-correlation among top factors.

Computes:
1. Category distribution of Accepted factors
2. Pairwise rank correlation matrix of top factors (cross-sectional average)
3. Factor cluster analysis to identify redundancy and orthogonal gaps
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster


def get_top_factors(n: int = 40, market: str = "csi1000") -> list[dict]:
    """Get top N factors by |ICIR| from factor library."""
    db = sqlite3.connect(str(PROJECT_ROOT / "data" / "factor_library.db"))
    rows = db.execute("""
        SELECT f.name, f.expression, f.category, f.status,
               t.rank_ic_mean, t.rank_icir, t.rank_ic_t
        FROM factors f JOIN factor_test_results t ON f.name = t.factor_name
        WHERE t.market = ? AND f.status IN ('Accepted', 'Baseline')
        ORDER BY abs(t.rank_icir) DESC
        LIMIT ?
    """, (market, n)).fetchall()
    db.close()
    return [{"name": r[0], "expression": r[1], "category": r[2],
             "status": r[3], "rank_ic": r[4], "icir": r[5], "t": r[6]}
            for r in rows]


def compute_factor_correlations(factors: list[dict],
                                market: str = "csi1000",
                                start: str = "2023-01-01",
                                end: str = "2024-12-31") -> pd.DataFrame:
    """Compute pairwise rank correlation matrix among factors."""
    from project_qlib.runtime import init_qlib
    init_qlib()
    from qlib.data import D

    instruments = D.instruments(market)

    # Load all factor values
    names = [f["name"] for f in factors]
    exprs = [f["expression"] for f in factors]

    # Load one at a time (some expressions may fail)
    factor_dfs = {}
    for name, expr in zip(names, exprs):
        if not expr:
            continue
        try:
            df = D.features(instruments, [expr],
                            start_time=start, end_time=end)
            df.columns = [name]
            factor_dfs[name] = df
        except Exception as e:
            print(f"  Skip {name}: {e}")

    if len(factor_dfs) < 2:
        print("Too few factors loaded")
        return pd.DataFrame()

    # Merge into single DataFrame
    merged = pd.concat(factor_dfs.values(), axis=1)
    valid_names = list(merged.columns)
    print(f"Loaded {len(valid_names)} factors, shape={merged.shape}")

    # Compute average cross-sectional rank correlation
    dates = merged.index.get_level_values("datetime").unique()
    # Sample dates for efficiency
    if len(dates) > 100:
        sample_dates = np.random.choice(dates, 100, replace=False)
    else:
        sample_dates = dates

    corr_sum = np.zeros((len(valid_names), len(valid_names)))
    n_valid = 0

    for dt in sample_dates:
        try:
            cs = merged.xs(dt, level="datetime")
            if len(cs) < 50:
                continue
            # Rank within cross-section
            ranked = cs.rank(method="average")
            c = ranked.corr(method="spearman").values
            if not np.any(np.isnan(c)):
                corr_sum += c
                n_valid += 1
        except Exception:
            continue

    if n_valid == 0:
        print("No valid dates for correlation")
        return pd.DataFrame()

    avg_corr = pd.DataFrame(corr_sum / n_valid,
                            index=valid_names, columns=valid_names)
    print(f"Computed avg cross-sectional rank correlation over {n_valid} dates")
    return avg_corr


def analyze_distribution(factors: list[dict]) -> None:
    """Analyze factor category distribution."""
    print("\n" + "=" * 70)
    print("FACTOR CATEGORY DISTRIBUTION (Accepted + Baseline, Top 40)")
    print("=" * 70)

    from collections import Counter
    cats = Counter(f["category"] for f in factors)
    for cat, cnt in cats.most_common():
        members = [f["name"] for f in factors if f["category"] == cat]
        icirs = [f["icir"] for f in factors if f["category"] == cat]
        print(f"  {cat:<20} n={cnt:2d}  ICIR=[{min(icirs):+.3f}, {max(icirs):+.3f}]")
        for m, ic in sorted(zip(members, icirs), key=lambda x: -abs(x[1])):
            print(f"    {m:<30} ICIR={ic:+.3f}")


def analyze_clusters(corr_matrix: pd.DataFrame, factors: list[dict]) -> None:
    """Cluster factors by correlation and identify redundant groups."""
    print("\n" + "=" * 70)
    print("FACTOR CORRELATION CLUSTERS")
    print("=" * 70)

    names = corr_matrix.index.tolist()
    # Distance = 1 - |correlation|
    dist = 1 - corr_matrix.abs().values
    np.fill_diagonal(dist, 0)
    # Make symmetric and positive
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, 2)

    # Hierarchical clustering
    from scipy.spatial.distance import squareform
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=0.3, criterion="distance")  # |corr| > 0.7 = same cluster

    # Get factor info map
    fmap = {f["name"]: f for f in factors}

    # Print clusters
    cluster_map = {}
    for name, cl in zip(names, clusters):
        cluster_map.setdefault(cl, []).append(name)

    for cl_id, members in sorted(cluster_map.items(), key=lambda x: -len(x[1])):
        if len(members) == 1:
            f = fmap.get(members[0], {})
            print(f"\n  Cluster {cl_id} (SINGLETON): {members[0]} "
                  f"[{f.get('category', '?')}] ICIR={f.get('icir', 0):+.3f}")
        else:
            print(f"\n  Cluster {cl_id} ({len(members)} factors):")
            for m in members:
                f = fmap.get(m, {})
                print(f"    {m:<30} [{f.get('category', '?'):<16}] ICIR={f.get('icir', 0):+.3f}")

            # Show intra-cluster correlations
            sub_corr = corr_matrix.loc[members, members]
            triu = np.triu(sub_corr.values, k=1)
            nonzero = triu[triu != 0]
            if len(nonzero) > 0:
                print(f"    → avg intra-corr: {nonzero.mean():.3f} "
                      f"(min={nonzero.min():.3f}, max={nonzero.max():.3f})")

    # Summary: highly correlated pairs
    print("\n" + "-" * 50)
    print("TOP 15 CORRELATED PAIRS (|ρ| > 0.6):")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = abs(corr_matrix.iloc[i, j])
            if c > 0.6:
                pairs.append((names[i], names[j], corr_matrix.iloc[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))
    for n1, n2, c in pairs[:15]:
        f1 = fmap.get(n1, {})
        f2 = fmap.get(n2, {})
        print(f"  {c:+.3f}  {n1:<28} [{f1.get('category', '?'):<14}] "
              f"↔ {n2:<28} [{f2.get('category', '?')}]")

    # Factors with low max correlation to others (most independent)
    print("\n" + "-" * 50)
    print("MOST INDEPENDENT FACTORS (lowest max |corr| to any other):")
    for name in names:
        others = corr_matrix.loc[name].drop(name)
        max_corr = others.abs().max()
        pairs.append((name, max_corr))
    independence = [(name, corr_matrix.loc[name].drop(name).abs().max())
                    for name in names]
    independence.sort(key=lambda x: x[1])
    for name, mc in independence[:10]:
        f = fmap.get(name, {})
        print(f"  {name:<30} max|corr|={mc:.3f} [{f.get('category', '?')}] "
              f"ICIR={f.get('icir', 0):+.3f}")


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print("Loading top 40 factors from factor library...")
    factors = get_top_factors(n=40, market="csi1000")
    print(f"Got {len(factors)} factors")

    # 1. Distribution analysis
    analyze_distribution(factors)

    # 2. Correlation analysis
    print("\nComputing cross-sectional correlations (2023-2024)...")
    corr_matrix = compute_factor_correlations(factors, start="2023-01-01", end="2024-12-31")

    if not corr_matrix.empty:
        # Save correlation matrix
        out_path = PROJECT_ROOT / "outputs" / "factor_correlation_matrix.csv"
        corr_matrix.to_csv(out_path)
        print(f"Correlation matrix saved to {out_path}")

        # 3. Cluster analysis
        analyze_clusters(corr_matrix, factors)


if __name__ == "__main__":
    main()

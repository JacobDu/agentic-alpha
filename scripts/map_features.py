"""Map Column_X feature names to actual factor names for TopN50."""
import sys
sys.path.insert(0, "src")

# Get the ordered factor list from TopN50 handler
from project_qlib.factors.topn_alpha import TopN50

handler = TopN50.__new__(TopN50)
# The factor order comes from _get_topn_factors
from project_qlib.factors.topn_alpha import _get_topn_factors
fields, names = _get_topn_factors(50)

print(f"TopN50 has {len(names)} factors")
print(f"\nColumn -> Factor mapping:")
print(f"{'ColumnIdx':>10} {'Factor Name':<35}")
print("-" * 50)
for i, name in enumerate(names):
    print(f"Column_{i:>3} -> {name}")

# Now map the feature importance 
import pandas as pd
imp = pd.read_csv("outputs/topn50_feature_importance.csv")
imp["factor_name"] = imp["feature"].apply(lambda x: names[int(x.split("_")[1])])

# Re-sort by importance
imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
print(f"\n=== TopN50 LightGBM Feature Importance (Actual Factor Names) ===")
print(f"{'Rank':>4} {'Factor':<35} {'Gain':>8} {'Pct':>7} {'CumPct':>7}")
print("-" * 65)
for j, (_, r) in enumerate(imp.iterrows(), 1):
    print(f"{j:>4} {r['factor_name']:<35} {r['importance']:>8.0f} {r['pct']:>6.2f}% {r['cum_pct']:>6.1f}%")

# Save enhanced version
imp.to_csv("outputs/topn50_feature_importance.csv", index=False)

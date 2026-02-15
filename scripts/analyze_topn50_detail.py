"""Complete analysis of TopN50 model: benchmark return, feature importance, daily returns."""
import sys, pickle
sys.path.insert(0, "src")
from project_qlib.runtime import init_qlib
init_qlib()

import pandas as pd
import numpy as np
from pathlib import Path

run_dir = list(Path("mlruns/1").glob("d3aec63d*"))[0]

# ==========================================
# 1. Portfolio returns vs benchmark
# ==========================================
report_pkl = run_dir / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl"
with open(report_pkl, "rb") as f:
    report = pickle.load(f)

print("=== Portfolio vs Benchmark Daily Returns ===")
print(f"Columns: {list(report.columns)}")
print(f"Date range: {report.index.min()} ~ {report.index.max()}")
print(f"Trading days: {len(report)}")

# Cumulative returns
cum_strat = (1 + report['return']).prod() - 1
cum_bench = (1 + report['bench']).prod() - 1
cum_excess = cum_strat - cum_bench
n_days = len(report)
ann_strat = (1 + cum_strat) ** (252 / n_days) - 1
ann_bench = (1 + cum_bench) ** (252 / n_days) - 1

print(f"\n  策略累计收益: {cum_strat*100:.2f}%")
print(f"  基准累计收益: {cum_bench*100:.2f}%")
print(f"  超额累计收益: {cum_excess*100:.2f}%")
print(f"  策略年化收益: {ann_strat*100:.2f}%")
print(f"  基准年化收益: {ann_bench*100:.2f}%")
print(f"  超额年化收益: {(ann_strat - ann_bench)*100:.2f}%")

# Max drawdown
cum_ret = (1 + report['return']).cumprod()
mdd_strat = (cum_ret / cum_ret.cummax() - 1).min()
cum_bm = (1 + report['bench']).cumprod()
mdd_bench = (cum_bm / cum_bm.cummax() - 1).min()
print(f"\n  策略最大回撤: {mdd_strat*100:.2f}%")
print(f"  基准最大回撤: {mdd_bench*100:.2f}%")

# Sharpe ratio (excess return / std)
excess_daily = report['return'] - report['bench']
sharpe = excess_daily.mean() / excess_daily.std() * np.sqrt(252)
print(f"  信息比率 (IR): {sharpe:.2f}")

# ==========================================
# 2. LightGBM Feature Importance
# ==========================================
params_pkl = run_dir / "artifacts" / "params.pkl"
with open(params_pkl, "rb") as f:
    model = pickle.load(f)

print(f"\n=== LightGBM Feature Importance (TopN50) ===")
if hasattr(model, "model") and model.model is not None:
    lgb_model = model.model
    importance = lgb_model.feature_importance(importance_type="gain")
    feature_names = lgb_model.feature_name()
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df["pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
    imp_df["cum_pct"] = imp_df["pct"].cumsum()
    
    print(f"Total features: {len(imp_df)}")
    print(f"\n{'Rank':>4} {'Feature':<45} {'Gain':>12} {'Pct':>7} {'CumPct':>7}")
    print("-" * 80)
    for j, (_, r) in enumerate(imp_df.iterrows(), 1):
        if j <= 50 or r['pct'] > 0.5:
            print(f"{j:>4} {r['feature']:<45} {r['importance']:>12.0f} {r['pct']:>6.2f}% {r['cum_pct']:>6.1f}%")
    
    # Save to file
    imp_df.to_csv("outputs/topn50_feature_importance.csv", index=False)
    print(f"\nSaved to outputs/topn50_feature_importance.csv")
    
    # Map feature importance back to factor ranking
    # Feature names in Qlib handler are like FEATURE_0, FEATURE_1 etc or the actual names
    print(f"\nSample feature names: {feature_names[:5]}")
else:
    print("  Model not available (model attribute is None)")
    # Try accessing through internal model
    print(f"  Model attributes: {[a for a in dir(model) if not a.startswith('_')]}")

# ==========================================
# 3. Monthly return breakdown
# ==========================================
print(f"\n=== Monthly Return Breakdown ===")
report_m = report.copy()
report_m.index = pd.to_datetime(report_m.index)
report_m['month'] = report_m.index.to_period('M')
report_m['excess'] = report_m['return'] - report_m['bench']

monthly = report_m.groupby('month').agg({
    'return': lambda x: (1 + x).prod() - 1,
    'bench': lambda x: (1 + x).prod() - 1,
    'excess': lambda x: (1 + x).prod() - 1,
})
monthly.columns = ['Strategy', 'Benchmark', 'Excess']

print(f"{'Month':>10} {'Strategy':>10} {'Benchmark':>10} {'Excess':>10}")
print("-" * 45)
for period, row in monthly.iterrows():
    print(f"{str(period):>10} {row['Strategy']*100:>9.2f}% {row['Benchmark']*100:>9.2f}% {row['Excess']*100:>9.2f}%")

n_pos = (monthly['Excess'] > 0).sum()
n_neg = (monthly['Excess'] <= 0).sum()
print(f"\n超额正月份: {n_pos}, 超额负月份: {n_neg}, 胜率: {n_pos/(n_pos+n_neg)*100:.0f}%")

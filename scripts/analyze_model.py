"""Extract feature importance from TopN50 model and portfolio analysis."""
import sys, pickle
sys.path.insert(0, "src")
from project_qlib.runtime import init_qlib
init_qlib()

import pandas as pd
import numpy as np
from pathlib import Path

run_dir = list(Path("mlruns/1").glob("d3aec63d*"))[0]

# 1. Try to get feature importance from model (task is a single pkl file, not dir)
task_file = run_dir / "artifacts" / "task"
if task_file.exists() and task_file.is_file():
    with open(task_file, "rb") as f:
        task_data = pickle.load(f)
    print(f"Task type: {type(task_data)}")
    if isinstance(task_data, dict):
        for k in task_data.keys():
            print(f"  Key: {k}")
    elif isinstance(task_data, list):
        for i, v in enumerate(task_data):
            print(f"  [{i}] type: {type(v)}")
            if isinstance(v, dict):
                for k in v.keys():
                    print(f"       Key: {k}")

# Try loading the model params to find model pickle
params_pkl = run_dir / "artifacts" / "params.pkl"
if params_pkl.exists():
    with open(params_pkl, "rb") as f:
        params = pickle.load(f)
    # Check if it's a dict or list
    print(f"\nParams type: {type(params)}")
    if isinstance(params, dict):
        for k, v in params.items():
            if isinstance(v, (str, int, float, bool)):
                print(f"  {k}: {v}")

# 2. Load portfolio daily returns for more detailed analysis
port_analysis = run_dir / "artifacts" / "portfolio_analysis" / "port_analysis_1day.pkl"
if port_analysis.exists():
    with open(port_analysis, "rb") as f:
        port_data = pickle.load(f)
    print(f"\nPortfolio analysis type: {type(port_data)}")
    if isinstance(port_data, dict):
        for k in port_data.keys():
            print(f"  Key: {k}")
    if isinstance(port_data, (list, tuple)):
        for i, v in enumerate(port_data):
            print(f"  [{i}] type: {type(v)}")

# 3. Load IC/RankIC time series
ic_pkl = run_dir / "artifacts" / "sig_analysis" / "ic.pkl"
ric_pkl = run_dir / "artifacts" / "sig_analysis" / "ric.pkl"
if ic_pkl.exists():
    with open(ic_pkl, "rb") as f:
        ic_data = pickle.load(f)
    print(f"\nIC data type: {type(ic_data)}, shape: {ic_data.shape if hasattr(ic_data, 'shape') else 'N/A'}")
    if isinstance(ic_data, (pd.Series, pd.DataFrame)):
        print(f"  Mean IC: {ic_data.mean():.4f}")
        print(f"  Std IC: {ic_data.std():.4f}")
        print(f"  ICIR: {ic_data.mean() / ic_data.std():.4f}")

if ric_pkl.exists():
    with open(ric_pkl, "rb") as f:
        ric_data = pickle.load(f)
    print(f"\nRankIC data type: {type(ric_data)}, shape: {ric_data.shape if hasattr(ric_data, 'shape') else 'N/A'}")
    if isinstance(ric_data, (pd.Series, pd.DataFrame)):
        print(f"  Mean RankIC: {ric_data.mean():.4f}")
        print(f"  Std RankIC: {ric_data.std():.4f}")
        print(f"  RankICIR: {ric_data.mean() / ric_data.std():.4f}")

# 4. Load report for cumulative positions/returns
report_pkl = run_dir / "artifacts" / "portfolio_analysis" / "report_normal_1day.pkl"
if report_pkl.exists():
    with open(report_pkl, "rb") as f:
        report = pickle.load(f)
    print(f"\nReport type: {type(report)}")
    if isinstance(report, (list, tuple)):
        for i, df in enumerate(report):
            if isinstance(df, pd.DataFrame):
                print(f"  [{i}] DataFrame cols: {list(df.columns)}, shape: {df.shape}")
                print(f"       Date range: {df.index.min()} ~ {df.index.max()}")
                if "return" in df.columns:
                    cum_ret = (1 + df['return']).prod() - 1
                    print(f"       Cumulative return: {cum_ret*100:.2f}%")
                if "excess_return" in df.columns:
                    cum_excess = (1 + df['excess_return']).prod() - 1
                    print(f"       Cumulative excess return: {cum_excess*100:.2f}%")
                if "bench" in df.columns:
                    cum_bench = (1 + df['bench']).prod() - 1
                    print(f"       Cumulative bench return: {cum_bench*100:.2f}%")
    elif isinstance(report, pd.DataFrame):
        print(f"  Columns: {list(report.columns)}")
        print(f"  Shape: {report.shape}")

# 5. Try to load the Qlib model itself to get feature importance
label_pkl = run_dir / "artifacts" / "label.pkl"
if label_pkl.exists():
    with open(label_pkl, "rb") as fp:
        label = pickle.load(fp)
    print(f"\nLabel type: {type(label)}")
    if hasattr(label, 'shape'):
        print(f"  Shape: {label.shape}")
    if isinstance(label, dict):
        for k in label: print(f"  {k}")

# Try pred.pkl for model predictions
pred_pkl = run_dir / "artifacts" / "pred.pkl"
if pred_pkl.exists():
    with open(pred_pkl, "rb") as fp:
        pred = pickle.load(fp)
    print(f"\nPred type: {type(pred)}, shape: {pred.shape if hasattr(pred, 'shape') else 'N/A'}")
    if isinstance(pred, pd.DataFrame):
        print(f"  Columns: {list(pred.columns)}")
        print(f"  Date range: {pred.index.get_level_values(1).min()} ~ {pred.index.get_level_values(1).max()}")

# Try to load model from Qlib recorder
try:
    from qlib.workflow import R
    recorders = R.list_recorders(experiment_name="workflow")
    for rid, rec in recorders.items():
        if rid.startswith("d3aec63d"):
            print(f"\nFound recorder: {rid}")
            model = rec.load_object("params.pkl")
            print(f"  params.pkl type: {type(model)}")
            if isinstance(model, dict) and "model" in model:
                m = model["model"]
                if hasattr(m, "model") and hasattr(m.model, "feature_importance"):
                    importance = m.model.feature_importance(importance_type="gain")
                    feature_names = m.model.feature_name()
                    imp_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": importance
                    }).sort_values("importance", ascending=False)
                    imp_df["pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
                    print(f"\n=== LightGBM Feature Importance (Top-30) ===")
                    print(f"{'Rank':>4} {'Feature':<40} {'Importance':>12} {'Pct':>7}")
                    print("-" * 65)
                    for j, (_, r) in enumerate(imp_df.head(30).iterrows(), 1):
                        print(f"{j:>4} {r['feature']:<40} {r['importance']:>12.1f} {r['pct']:>6.2f}%")
            break
except Exception as e:
    print(f"Error loading from Qlib: {e}")

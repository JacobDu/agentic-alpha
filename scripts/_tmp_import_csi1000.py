"""Temporary: Import csi1000 factor test results into DB."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from project_qlib.factor_db import FactorDB

db = FactorDB()
df = pd.read_csv(ROOT / "outputs" / "csi1000_unified_factor_ranking.csv")
count = 0
for _, row in df.iterrows():
    name = row["factor"]
    existing = db.get_factor(name)
    if not existing:
        continue
    fdr_p = row.get("rank_ic_p_fdr", 1.0)
    is_sig = bool(not pd.isna(fdr_p) and fdr_p < 0.01)
    db.upsert_test_result(
        factor_name=name,
        market="csi1000",
        test_start="2019-01-01",
        test_end="2025-12-31",
        n_days=int(row["n_days"]) if not pd.isna(row.get("n_days")) else None,
        ic_mean=float(row["ic_mean"]) if not pd.isna(row.get("ic_mean")) else None,
        rank_ic_mean=float(row["rank_ic_mean"]) if not pd.isna(row.get("rank_ic_mean")) else None,
        rank_icir=float(row["rank_icir"]) if not pd.isna(row.get("rank_icir")) else None,
        fdr_p=float(fdr_p) if not pd.isna(fdr_p) else None,
        significant=is_sig,
        evidence="outputs/csi1000_unified_factor_ranking.csv",
        hea_round="SFA-BOOTSTRAP-IMPORT",
    )
    count += 1

print(f"Imported {count} csi1000 test results")
print(db.summary())
db.close()

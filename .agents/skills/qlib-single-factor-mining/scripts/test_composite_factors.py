"""Test composite factors from orthogonal pairs.

Composite methods:
1. ADD: A + B (normalized via z-score style)
2. MUL: Mul(A, B) — interaction effect
3. COND_MUL: Mul(A, f(B)) — one factor modulates another

All pairs selected from orthogonal analysis (|rho| < 0.35, both |ICIR| > 0.15).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Cannot locate project root (pyproject.toml not found)")

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats

# ─── Factor expressions from DB ───
def get_expr(name):
    """Get expression for a factor name from DB."""
    db = sqlite3.connect(str(PROJECT_ROOT / "data" / "factor_library.db"))
    row = db.execute("SELECT expression FROM factors WHERE name=?", (name,)).fetchone()
    db.close()
    return row[0] if row else None

# ─── Composite factor definitions ───
# Format: (expression, name, category, description)
# We build composites from the top orthogonal pairs

COMPOSITE_FACTORS = []

# Helper: wrap expr for Qlib. For z-score normalization within expression:
# Use (expr - Mean(expr, N)) / (Std(expr, N) + 1e-8) to normalize before combining

# ═══════════════════════════════════════════════════════════════
# GROUP 1: TURN_QUANTILE × VOL_CV — the top two independent factors
# Pair: NEW_TURN_QUANTILE_120 (ICIR=-0.394) × CSTM_AMT_CV_20 (ICIR=-0.388), |ρ|=0.20
# ═══════════════════════════════════════════════════════════════

# A = TURN_QUANTILE_120 = Div($turnover_rate - Min($turnover_rate, 120), Max($turnover_rate, 120) - Min($turnover_rate, 120) + 1e-8)
# B = AMT_CV_20 = Div(Std($amount, 20), Mean($amount, 20))
A_TQ120 = "Div($turnover_rate - Min($turnover_rate, 120), Max($turnover_rate, 120) - Min($turnover_rate, 120) + 1e-8)"
B_AMTCV20 = "Div(Std($amount, 20), Mean($amount, 20))"

# MUL: Low turnover quantile × Low amt CV → both low = strong positive signal
COMPOSITE_FACTORS.append((
    f"Mul({A_TQ120}, {B_AMTCV20})",
    "COMP_TQ120_x_AMTCV20", "composite_mul",
    "TURN_QUANTILE_120 × AMT_CV_20 interaction"
))

# ADD: Simple addition (both negative IC, lower = better)
COMPOSITE_FACTORS.append((
    f"Add({A_TQ120}, {B_AMTCV20})",
    "COMP_TQ120_p_AMTCV20", "composite_add",
    "TURN_QUANTILE_120 + AMT_CV_20 additive"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 2: TURN_QUANTILE × GAP_FREQ — quantile × path
# Pair: NEW_TURN_QUANTILE_120 (ICIR=-0.394) × ORTH_GAP_FREQ_20 (ICIR=-0.364), |ρ|=0.185
# ═══════════════════════════════════════════════════════════════

C_GAPFREQ = "Mean(Abs(Div($open, Ref($close, 1)) - 1), 20)"

COMPOSITE_FACTORS.append((
    f"Mul({A_TQ120}, {C_GAPFREQ})",
    "COMP_TQ120_x_GAPFREQ", "composite_mul",
    "TURN_QUANTILE_120 × GAP_FREQ_20 interaction"
))

COMPOSITE_FACTORS.append((
    f"Add({A_TQ120}, {C_GAPFREQ})",
    "COMP_TQ120_p_GAPFREQ", "composite_add",
    "TURN_QUANTILE_120 + GAP_FREQ_20 additive"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 3: VOL_CV × CFP_TURN — volume stability × fundamental valuation flow
# Pair: CSTM_VOL_CV_10 (ICIR=-0.380) × NEW_CFP_TURN (ICIR=-0.304), |ρ|=0.12
# ═══════════════════════════════════════════════════════════════

D_VOLCV10 = "Div(Std($volume, 10), Mean($volume, 10))"
E_CFPTURN = "Mul(Div(1, Abs($pcf_ttm) + 1e-8), Mean($turnover_rate, 5))"

COMPOSITE_FACTORS.append((
    f"Mul({D_VOLCV10}, {E_CFPTURN})",
    "COMP_VOLCV10_x_CFP", "composite_mul",
    "VOL_CV_10 × CFP_TURN interaction"
))

COMPOSITE_FACTORS.append((
    f"Add({D_VOLCV10}, {E_CFPTURN})",
    "COMP_VOLCV10_p_CFP", "composite_add",
    "VOL_CV_10 + CFP_TURN additive"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 4: VALUE_LOWVOL × CFP_TURN — quality-value × cash flow
# Pair: NEW_VALUE_LOWVOL (ICIR=+0.322) × NEW_CFP_TURN (ICIR=-0.304), |ρ|=0.045
# Most orthogonal pair! ρ=0.045
# ═══════════════════════════════════════════════════════════════

F_VALUELVOL = "Div(Div(1, $pb_mrq + 1e-8), Std(Div($high - $low, $close + 1e-8), 20) + 1e-8)"

# VALUE_LOWVOL has positive IC (higher = better), CFP_TURN has negative IC (lower = better)
# For MUL to work well, flip CFP_TURN: Div(1, CFP_TURN)
COMPOSITE_FACTORS.append((
    f"Div({F_VALUELVOL}, {E_CFPTURN} + 1e-8)",
    "COMP_VALLV_div_CFP", "composite_ratio",
    "VALUE_LOWVOL / CFP_TURN — quality value when cash flow is quiet"
))

# MUL: VALUE_LOWVOL × (inverse of CFP_TURN — want high value AND low CFP)
COMPOSITE_FACTORS.append((
    f"Mul({F_VALUELVOL}, 0 - {E_CFPTURN})",
    "COMP_VALLV_x_negCFP", "composite_mul",
    "VALUE_LOWVOL × (-CFP_TURN) interaction"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 5: VOL_CV × AMT_PER_RANGE — stability × microstructure
# Pair: CSTM_VOL_CV_10 (ICIR=-0.380) × NEW_AMT_PER_RANGE (ICIR=-0.275), |ρ|=0.09
# ═══════════════════════════════════════════════════════════════

G_AMTRANGE = "Div($amount, ($high - $low) + 1e-8)"

COMPOSITE_FACTORS.append((
    f"Mul({D_VOLCV10}, {G_AMTRANGE})",
    "COMP_VOLCV10_x_AMTRANGE", "composite_mul",
    "VOL_CV_10 × AMT_PER_RANGE interaction"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 6: GAP_FREQ × VOL_TREND — path × liquidity timing
# Pair: ORTH_GAP_FREQ_20 (ICIR=-0.364) × ORTH_VOL_TREND_5_60 (ICIR=-0.299), |ρ|=0.157
# ═══════════════════════════════════════════════════════════════

H_VOLTREND = "Div(Mean($volume, 5), Mean($volume, 60) + 1e-8)"

COMPOSITE_FACTORS.append((
    f"Mul({C_GAPFREQ}, {H_VOLTREND})",
    "COMP_GAPFREQ_x_VOLTREND", "composite_mul",
    "GAP_FREQ × VOL_TREND_5_60 — gaps when volume is rising"
))

COMPOSITE_FACTORS.append((
    f"Add({C_GAPFREQ}, {H_VOLTREND})",
    "COMP_GAPFREQ_p_VOLTREND", "composite_add",
    "GAP_FREQ + VOL_TREND additive"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 7: VWAP_BIAS × VOL_CONC — where returns happen × volume concentration
# Pair: ORTH_VWAP_BIAS_VOL_20 (ICIR=-0.342) × NEW_VOL_CONC_5 (ICIR=-0.268), |ρ|=0.116
# ═══════════════════════════════════════════════════════════════

I_VWAPBIAS = "Std(Div($vwap, $close) - 1, 20)"
J_VOLCONC = "Div(Max($volume, 5), Sum($volume, 5) + 1e-8)"

COMPOSITE_FACTORS.append((
    f"Mul({I_VWAPBIAS}, {J_VOLCONC})",
    "COMP_VWAPBIAS_x_VOLCONC", "composite_mul",
    "VWAP_BIAS_VOL × VOL_CONC_5 — VWAP dispersion when volume is concentrated"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 8: OVN_INTRA_DIV × REVERT_LOWVOL — overnight divergence × conditional reversal
# Two highly orthogonal factors from prior orthogonal batch
# Both max|ρ| < 0.35 vs all existing top factors
# ═══════════════════════════════════════════════════════════════

K_OVNINTRA = "Mean(Div($open, Ref($close, 1)) - 1, 10) - Mean(Div($close, $open) - 1, 10)"
L_REVERT = "Mul(Div($close - Mean($close, 20), Mean($close, 20) + 1e-8), 0 - Div($volume, Mean($volume, 20) + 1e-8))"

# OVN has positive IC, REVERT has positive IC → MUL directly
COMPOSITE_FACTORS.append((
    f"Mul({K_OVNINTRA}, {L_REVERT})",
    "COMP_OVNINTRA_x_REVERT", "composite_mul",
    "OVN_INTRA_DIV × REVERT_LOWVOL — overnight divergence in reversal regime"
))

COMPOSITE_FACTORS.append((
    f"Add({K_OVNINTRA}, {L_REVERT})",
    "COMP_OVNINTRA_p_REVERT", "composite_add",
    "OVN_INTRA_DIV + REVERT_LOWVOL additive"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 9: INTRADAY_MA10 × AMT_PER_RANGE — intraday return × market depth
# ORTH_INTRADAY_MA10 (ICIR=-0.242) is highly orthogonal (max|ρ|=0.304)
# ═══════════════════════════════════════════════════════════════

M_INTRARET = "Mean(Div($close, $open) - 1, 10)"

COMPOSITE_FACTORS.append((
    f"Mul({M_INTRARET}, {G_AMTRANGE})",
    "COMP_INTRARET_x_AMTRANGE", "composite_mul",
    "INTRADAY_MA10 × AMT_PER_RANGE — intraday return × market depth"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 10: HIGHVOL_RET × SIGN_VOL_CORR — smart money timing
# ORTH_HIGHVOL_RET_20 (ICIR=-0.337) × CSTM_SIGN_VOL_CORR_20 (ICIR=-0.280)
# ═══════════════════════════════════════════════════════════════

N_HIGHVRET = "Mean(If($volume > Mean($volume, 20), Div($close, Ref($close, 1)) - 1, 0), 20)"
O_SIGNVCORR = "Corr(If(Div($close, Ref($close, 1)) > 1, $volume, 0 - $volume), Ref($close, 1), 20)"

COMPOSITE_FACTORS.append((
    f"Mul({N_HIGHVRET}, {O_SIGNVCORR})",
    "COMP_HIGHVRET_x_SIGNCORR", "composite_mul",
    "HIGHVOL_RET × SIGN_VOL_CORR — high-vol return quality"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 11: TQ120 × OVN_INTRA_DIV — quantile × time structure
# Very different dimensions: position in turnover range × overnight-intraday pattern
# ═══════════════════════════════════════════════════════════════

COMPOSITE_FACTORS.append((
    f"Mul({A_TQ120}, 0 - ({K_OVNINTRA}))",
    "COMP_TQ120_x_negOVN", "composite_mul",
    "TURN_QUANTILE × (-OVN_INTRA_DIV) — cold turnover + intraday strength"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 12: Triple composite — TQ120 × AMTCV × GAPFREQ
# Three of the most independent top factors combined
# ═══════════════════════════════════════════════════════════════

COMPOSITE_FACTORS.append((
    f"Mul(Mul({A_TQ120}, {B_AMTCV20}), {C_GAPFREQ})",
    "COMP_TRIPLE_TQ_CV_GAP", "composite_triple",
    "TURN_Q × AMT_CV × GAP_FREQ triple interaction"
))

# ═══════════════════════════════════════════════════════════════
# GROUP 13: Rank-style normalization composites
# Using z-score within rolling window as rank proxy
# ═══════════════════════════════════════════════════════════════

# Z-scored TURN_QUANTILE + Z-scored AMT_CV
COMPOSITE_FACTORS.append((
    f"Add(Div({A_TQ120} - Mean({A_TQ120}, 20), Std({A_TQ120}, 20) + 1e-8), "
    f"Div({B_AMTCV20} - Mean({B_AMTCV20}, 20), Std({B_AMTCV20}, 20) + 1e-8))",
    "COMP_ZSCORE_TQ_AMTCV", "composite_zscore",
    "Z(TURN_Q_120) + Z(AMT_CV_20) — z-score normalized addition"
))

# Z-scored TQ + Z-scored GAP_FREQ + Z-scored VOL_CV
COMPOSITE_FACTORS.append((
    f"Add(Add(Div({A_TQ120} - Mean({A_TQ120}, 20), Std({A_TQ120}, 20) + 1e-8), "
    f"Div({C_GAPFREQ} - Mean({C_GAPFREQ}, 20), Std({C_GAPFREQ}, 20) + 1e-8)), "
    f"Div({D_VOLCV10} - Mean({D_VOLCV10}, 20), Std({D_VOLCV10}, 20) + 1e-8))",
    "COMP_ZSCORE_TQ_GAP_CV", "composite_zscore",
    "Z(TURN_Q) + Z(GAP_FREQ) + Z(VOL_CV) — three-way z-score"
))

print(f"Total composite factors: {len(COMPOSITE_FACTORS)}")
for i, (expr, name, cat, desc) in enumerate(COMPOSITE_FACTORS):
    print(f"  {i+1:2d}. {name:<35} [{cat}]")


# ─── Test Infrastructure ───

def _load_industry():
    ind_file = PROJECT_ROOT / "data" / "industry.parquet"
    if not ind_file.exists():
        return None
    ind_df = pd.read_parquet(ind_file)
    return ind_df.set_index("qlib_code")["industry"]


def _neutralize_factor(factor_series, industry_map):
    """Neutralize factor by industry (vectorized)."""
    result = factor_series.copy()
    dates = factor_series.index.get_level_values("datetime")
    instruments = factor_series.index.get_level_values("instrument")
    tmp = pd.DataFrame({
        "factor": factor_series.values,
        "datetime": dates,
        "instrument": instruments,
    })
    tmp["industry"] = tmp["instrument"].map(industry_map)
    tmp = tmp.dropna(subset=["industry", "factor"])
    
    grouped = tmp.groupby(["datetime", "industry"])["factor"]
    g_mean = grouped.transform("mean")
    g_std = grouped.transform("std")
    neutral = (tmp["factor"] - g_mean) / (g_std + 1e-8)
    
    result.loc[:] = np.nan
    result.iloc[neutral.index] = neutral.values
    return result


def test_factor_ic(factor_values, label_values, n_days_min=200):
    """Compute RankIC statistics for a factor."""
    merged = pd.DataFrame({"factor": factor_values, "label": label_values}).dropna()
    
    dates = merged.index.get_level_values("datetime").unique()
    
    rank_ics = []
    for dt in dates:
        try:
            cs = merged.xs(dt, level="datetime")
            if len(cs) < 50:
                continue
            ric = cs["factor"].rank().corr(cs["label"].rank(), method="spearman")
            if not np.isnan(ric):
                rank_ics.append(ric)
        except Exception:
            continue
    
    if len(rank_ics) < n_days_min:
        return None
    
    rank_ics = np.array(rank_ics)
    rank_ic_mean = np.mean(rank_ics)
    rank_ic_std = np.std(rank_ics, ddof=1)
    rank_icir = rank_ic_mean / (rank_ic_std + 1e-8)
    t_stat = rank_ic_mean / (rank_ic_std / np.sqrt(len(rank_ics)) + 1e-8)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(rank_ics) - 1))
    
    return {
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_icir": rank_icir,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_days": len(rank_ics),
    }


def main():
    parser = argparse.ArgumentParser(description="Test composite factors from orthogonal candidates")
    parser.add_argument("--market", default="csi1000")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--source-tag", default="SFA-COMPOSITE")
    parser.add_argument("--round-id", default="SFA-COMPOSITE-ROUND")
    args = parser.parse_args()

    from project_qlib.runtime import init_qlib
    init_qlib()
    from qlib.data import D
    
    instruments = D.instruments(args.market)
    
    # Load label
    print("\nLoading label: Ref($close, -2)/Ref($close, -1) - 1")
    label_expr = "Ref($close, -2)/Ref($close, -1) - 1"
    label_df = D.features(instruments, [label_expr], start_time=args.start, end_time=args.end)
    label_df.columns = ["label"]
    label = label_df["label"]
    
    # Load industry
    ind_map = _load_industry()
    print(f"Industry map loaded: {len(ind_map)} stocks")
    
    results = []
    
    for i, (expr, name, cat, desc) in enumerate(COMPOSITE_FACTORS):
        print(f"\n[{i+1}/{len(COMPOSITE_FACTORS)}] Testing {name}...")
        try:
            factor_df = D.features(instruments, [expr], start_time=args.start, end_time=args.end)
            factor_df.columns = ["factor"]
            factor = factor_df["factor"]
            
            # Neutralize
            if ind_map is not None:
                factor = _neutralize_factor(factor, ind_map)
            
            # Test IC
            res = test_factor_ic(factor, label)
            if res is None:
                print(f"  SKIP: insufficient data")
                results.append({"name": name, "category": cat, "status": "Insufficient"})
                continue
            
            sig = "***" if res["p_value"] < 0.01 else ("**" if res["p_value"] < 0.05 else "ns")
            status = "Accepted" if res["p_value"] < 0.01 else ("Candidate" if res["p_value"] < 0.05 else "Rejected")
            
            print(f"  RankIC={res['rank_ic_mean']:+.4f}  ICIR={res['rank_icir']:+.3f}  "
                  f"t={res['t_stat']:+.1f}  p={res['p_value']:.4f}  n={res['n_days']}  {sig}")
            
            results.append({
                "name": name,
                "category": cat,
                "description": desc,
                "expression": expr,
                "rank_ic_mean": res["rank_ic_mean"],
                "rank_icir": res["rank_icir"],
                "t_stat": res["t_stat"],
                "p_value": res["p_value"],
                "n_days": res["n_days"],
                "status": status,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": name, "category": cat, "status": "Error", "description": str(e)})
    
    # ─── Summary ───
    print("\n" + "=" * 100)
    print("COMPOSITE FACTOR RESULTS SUMMARY")
    print("=" * 100)
    
    df = pd.DataFrame(results)
    valid = df[df["status"].isin(["Accepted", "Candidate", "Rejected"])].copy()
    valid = valid.sort_values("rank_icir", key=abs, ascending=False)
    
    print(f"\nTotal: {len(COMPOSITE_FACTORS)}")
    print(f"Accepted (p<0.01): {len(valid[valid['status']=='Accepted'])}")
    print(f"Candidate (p<0.05): {len(valid[valid['status']=='Candidate'])}")
    print(f"Rejected: {len(valid[valid['status']=='Rejected'])}")
    
    print(f"\n{'Rank':>4} {'Name':<35} {'Category':<18} {'RankIC':>8} {'ICIR':>8} {'t':>7} {'Status':<10}")
    print("-" * 100)
    for idx, row in valid.iterrows():
        sig_mark = "***" if row["status"] == "Accepted" else ("**" if row["status"] == "Candidate" else "")
        print(f"  {'':<2} {row['name']:<35} {row['category']:<18} {row['rank_ic_mean']:+.4f} "
              f"{row['rank_icir']:+.3f}  {row['t_stat']:+.1f}  {row['status']:<10} {sig_mark}")
    
    # Save results
    out_path = PROJECT_ROOT / "outputs" / "composite_factor_results.csv"
    df.to_csv(str(out_path), index=False)
    print(f"\nResults saved to {out_path}")
    
    # ─── Compare with component factors ───
    print("\n" + "=" * 100)
    print("COMPOSITE vs COMPONENT COMPARISON")
    print("=" * 100)
    
    # Key component ICIRs for reference
    component_icirs = {
        "NEW_TURN_QUANTILE_120": -0.394,
        "CSTM_AMT_CV_20": -0.388,
        "CSTM_VOL_CV_10": -0.380,
        "ORTH_GAP_FREQ_20": -0.364,
        "ORTH_VWAP_BIAS_VOL_20": -0.342,
        "ORTH_HIGHVOL_RET_20": -0.337,
        "NEW_VALUE_LOWVOL": 0.322,
        "NEW_CFP_TURN": -0.304,
        "NEW_AMT_PER_RANGE": -0.275,
        "NEW_VOL_CONC_5": -0.268,
    }
    
    for _, row in valid.head(10).iterrows():
        desc = row.get("description", "")
        print(f"\n  {row['name']} (ICIR={row['rank_icir']:+.3f})  [{row['category']}]")
        print(f"    {desc}")
    
    # ─── Backfill to DB ───
    print("\n" + "=" * 100)
    print("BACKFILLING TO FACTOR LIBRARY DB")
    print("=" * 100)
    
    db = sqlite3.connect(str(PROJECT_ROOT / "data" / "factor_library.db"))
    n_inserted = 0
    for _, row in df.iterrows():
        if row["status"] in ("Error", "Insufficient"):
            continue
        
        # Check if already exists
        existing = db.execute("SELECT name FROM factors WHERE name=?", (row["name"],)).fetchone()
        if existing:
            continue
        
        db.execute("""
            INSERT INTO factors (name, expression, source, category, status, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row["name"], row.get("expression", ""), args.source_tag, row["category"],
              row["status"], row.get("description", "")))
        
        if row["status"] in ("Accepted", "Candidate", "Rejected"):
            rank_ic_std = abs(row.get("rank_ic_mean", 0)) / (abs(row.get("rank_icir", 1)) + 1e-8)
            sig = 1 if row["status"] == "Accepted" else 0
            db.execute("""
                INSERT OR REPLACE INTO factor_test_results 
                (factor_name, market, test_start, test_end, n_days,
                 rank_ic_mean, rank_ic_std, rank_icir, rank_ic_t, rank_ic_p,
                 significant, hea_round)
                VALUES (?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?)
            """, (row["name"], args.market, args.start, args.end, row.get("n_days", 0),
                  row.get("rank_ic_mean", 0), rank_ic_std,
                  row.get("rank_icir", 0), row.get("t_stat", 0), row.get("p_value", 1),
                  sig, args.round_id))
        
        n_inserted += 1
    
    db.commit()
    db.close()
    print(f"Backfilled {n_inserted} composite factors to factor library.")


if __name__ == "__main__":
    main()

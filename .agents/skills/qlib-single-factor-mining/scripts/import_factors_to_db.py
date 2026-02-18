"""Import Alpha158 baseline factors and tested custom factors into the SQLite factor library.

Run once to bootstrap the factor library:
    uv run python scripts/import_factors_to_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_qlib.factor_db import FactorDB


def _get_alpha158_factors() -> list[tuple[str, str, str]]:
    """Return (name, expression, category) for all Alpha158 factors."""
    from qlib.contrib.data.handler import Alpha158

    h = Alpha158.__new__(Alpha158)
    fields, names = h.get_feature_config()

    # Categorize Alpha158 factors by prefix
    def _categorize(name: str) -> str:
        prefix = name.rstrip("0123456789")
        cats = {
            "KMID": "kbar", "KLEN": "kbar", "KUP": "kbar", "KLOW": "kbar", "KSFT": "kbar",
            "OPEN": "kbar", "HIGH": "kbar", "LOW": "kbar", "CLOSE": "kbar", "VWAP": "kbar",
            "ROC": "momentum", "MA": "trend", "STD": "volatility", "BETA": "beta",
            "RSQR": "beta", "RESI": "beta", "MAX": "range", "MIN": "range",
            "QTLU": "quantile", "QTLD": "quantile", "RANK": "rank",
            "RSV": "range", "IMAX": "timing", "IMIN": "timing", "IMXD": "timing",
            "CORR": "correlation", "CORD": "correlation",
            "CNTP": "count", "CNTD": "count", "CNTN": "count",
            "SUMP": "sum", "SUMD": "sum", "SUMN": "sum",
            "VMA": "volume", "VSTD": "volume", "VSUMP": "volume",
            "VSUMD": "volume", "VSUMN": "volume",
            "WVMA": "volume",
        }
        return cats.get(prefix, "alpha158_other")

    result = []
    for name, field in zip(names, fields):
        cat = _categorize(name)
        result.append((name, str(field), cat))
    return result


# Custom factor pool — same as test_factors_csiall.py
CUSTOM_FACTORS = [
    ("CSTM_AMT_SURGE_20", "$amount / (Mean($amount, 20) + 1e-8)", "volume", "量能急升(20日)→投机过热→反转"),
    ("CSTM_AMT_SURGE_60", "$amount / (Mean($amount, 60) + 1e-8)", "volume", "量能急升(60日)→长期量能偏离"),
    ("CSTM_VOL_SURGE_5", "$volume / (Mean($volume, 5) + 1e-8)", "volume", "短期放量→信息冲击"),
    ("CSTM_VOL_CV_10", "Std($volume, 10) / (Mean($volume, 10) + 1e-8)", "volume", "成交量变异系数→交易不确定性"),
    ("CSTM_AMT_CV_20", "Std($amount, 20) / (Mean($amount, 20) + 1e-8)", "volume", "成交额波动性→流动性风险"),
    ("CSTM_VOL_RATIO_5_20", "Mean($volume, 5) / (Mean($volume, 20) + 1e-8)", "volume", "短期vs长期成交量→量能趋势"),
    ("CSTM_VWAP_BIAS_5", "$close / Mean($vwap, 5) - 1", "vwap", "收盘价偏离5日均VWAP→均值回归压力"),
    ("CSTM_VWAP_BIAS_10", "$close / Mean($vwap, 10) - 1", "vwap", "收盘价偏离10日均VWAP"),
    ("CSTM_VWAP_BIAS_20", "$close / Mean($vwap, 20) - 1", "vwap", "收盘价偏离20日均VWAP"),
    ("CSTM_VWAP_BIAS_1D", "$close / $vwap - 1", "vwap", "日内收盘偏离VWAP→当日尾盘情绪"),
    ("CSTM_VWAP_VOL_CORR_10", "Corr($close/$vwap, $volume/Ref($volume, 1), 10)", "vwap", "量价VWAP关联→机构行为痕迹"),
    ("CSTM_VWAP_VOL_CORR_20", "Corr($close/$vwap, $volume/Ref($volume, 1), 20)", "vwap", "20日量价VWAP关联"),
    ("CSTM_RANGE_1D", "($high - $low) / ($close + 1e-8)", "range", "日内振幅→当日波动强度"),
    ("CSTM_RANGE_RATIO_5_20", "Mean(($high - $low) / ($close + 1e-8), 5) / (Mean(($high - $low) / ($close + 1e-8), 20) + 1e-8)", "range", "波幅扩张比→突破/收缩信号"),
    ("CSTM_RANGE_RATIO_5_60", "Mean(($high - $low) / ($close + 1e-8), 5) / (Mean(($high - $low) / ($close + 1e-8), 60) + 1e-8)", "range", "波幅扩张比(vs60日)→长周期对比"),
    ("CSTM_CLOSE_POS", "($close - $low) / ($high - $low + 1e-8)", "range", "收盘位置→日内强弱"),
    ("CSTM_CLOSE_POS_MA5", "Mean(($close - $low) / ($high - $low + 1e-8), 5)", "range", "5日平均收盘位置→持续强弱"),
    ("CSTM_SHADOW_RATIO", "($high - $close) / ($close - $low + 1e-8)", "range", "上影/下影线比率→多空力量对比"),
    ("CSTM_RANGE_VOL_10", "Std(($high-$low)/($close+1e-8), 10)", "range", "振幅波动率→波动率的波动率"),
    ("CSTM_GAP_1D", "$open / Ref($close, 1) - 1", "gap", "隔夜跳空→隔夜信息冲击"),
    ("CSTM_GAP_MA_5", "Mean($open / Ref($close, 1) - 1, 5)", "gap", "5日平均隔夜跳空→持续隔夜情绪"),
    ("CSTM_GAP_MA_10", "Mean($open / Ref($close, 1) - 1, 10)", "gap", "10日平均隔夜跳空"),
    ("CSTM_GAP_STD_10", "Std($open / Ref($close, 1) - 1, 10)", "gap", "隔夜跳空波动→隔夜风险"),
    ("CSTM_RET_ACCEL_1", "$close/Ref($close, 1) - Ref($close, 1)/Ref($close, 2)", "momentum", "收益加速度(1日)→动量耗竭/启动"),
    ("CSTM_MOM_DIFF_5_20", "Mean($close/Ref($close, 1) - 1, 5) - Mean($close/Ref($close, 1) - 1, 20)", "momentum", "短期vs长期均收益率差→动量切换"),
    ("CSTM_REVERT_1", "Ref($close, 1)/$close - 1", "momentum", "1日反转→最短期均值回归"),
    ("CSTM_REVERT_3", "Ref($close, 3)/$close - 1", "momentum", "3日反转"),
    ("CSTM_REVERT_5", "Ref($close, 5)/$close - 1", "momentum", "5日反转"),
    ("CSTM_REVERT_10", "Ref($close, 10)/$close - 1", "momentum", "10日反转"),
    ("CSTM_REVERT_20", "Ref($close, 20)/$close - 1", "momentum", "20日反转"),
    ("CSTM_PV_CORR_5", "Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 5)", "price_vol", "5日量价相关→信息流方向"),
    ("CSTM_PV_CORR_10", "Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 10)", "price_vol", "10日量价相关"),
    ("CSTM_PV_CORR_20", "Corr($close/Ref($close, 1) - 1, $volume/Ref($volume, 1) - 1, 20)", "price_vol", "20日量价相关"),
    ("CSTM_PA_CORR_10", "Corr($close/Ref($close, 1) - 1, $amount/Ref($amount, 1) - 1, 10)", "price_vol", "10日收益-成交额相关"),
    ("CSTM_SKEW_20", "Mean(Power($close/Ref($close, 1) - 1, 3), 20) / (Power(Std($close/Ref($close, 1) - 1, 20), 3) + 1e-12)", "higher_moment", "20日收益率偏度→尾部风险"),
    ("CSTM_SKEW_60", "Mean(Power($close/Ref($close, 1) - 1, 3), 60) / (Power(Std($close/Ref($close, 1) - 1, 60), 3) + 1e-12)", "higher_moment", "60日收益率偏度"),
    ("CSTM_AMT_WTRET_10", "Mean(($close/Ref($close, 1) - 1) * $amount, 10) / (Mean($amount, 10) + 1e-8)", "smart_money", "10日成交额加权收益→大单方向"),
    ("CSTM_AMT_WTRET_20", "Mean(($close/Ref($close, 1) - 1) * $amount, 20) / (Mean($amount, 20) + 1e-8)", "smart_money", "20日成交额加权收益→大单方向"),
    ("CSTM_MA_BIAS_5", "$close / Mean($close, 5) - 1", "trend", "5日均线偏离"),
    ("CSTM_MA_BIAS_10", "$close / Mean($close, 10) - 1", "trend", "10日均线偏离"),
    ("CSTM_MA_BIAS_20", "$close / Mean($close, 20) - 1", "trend", "20日均线偏离"),
    ("CSTM_MA_BIAS_60", "$close / Mean($close, 60) - 1", "trend", "60日均线偏离→中线趋势"),
    ("CSTM_MA_CROSS_5_20", "Mean($close, 5) / Mean($close, 20) - 1", "trend", "5/20日均线差→趋势方向"),
    # Original single custom factor
    ("CSTM_MOM_5", "Ref($close, 1)/Ref($close, 5)-1", "momentum", "5日动量"),
]

# Previous test results from outputs/csiall_factor_significance.csv (HEA-2026-02-14-05)
# Only the custom factors that were tested. We'll import these historical results.
HISTORICAL_CSIALL_RESULTS: dict[str, dict] = {
    "CSTM_VOL_CV_10": {"rank_ic_mean": -0.032, "rank_ic_t": -21.04, "rank_icir": -0.51, "status": "Accepted"},
    "CSTM_AMT_CV_20": {"rank_ic_mean": -0.035, "rank_ic_t": -20.06, "rank_icir": -0.49, "status": "Accepted"},
    "CSTM_RANGE_VOL_10": {"rank_ic_mean": -0.054, "rank_ic_t": -15.42, "rank_icir": -0.37, "status": "Accepted"},
    "CSTM_AMT_SURGE_60": {"rank_ic_mean": -0.045, "rank_ic_t": -15.21, "rank_icir": -0.37, "status": "Accepted"},
    "CSTM_RANGE_1D": {"rank_ic_mean": -0.056, "rank_ic_t": -15.06, "rank_icir": -0.37, "status": "Accepted"},
    "CSTM_PV_CORR_20": {"rank_ic_mean": -0.032, "rank_ic_t": -14.94, "rank_icir": -0.36, "status": "Accepted"},
    "CSTM_AMT_WTRET_20": {"rank_ic_mean": -0.045, "rank_ic_t": -13.89, "rank_icir": -0.34, "status": "Accepted"},
    "CSTM_REVERT_20": {"rank_ic_mean": 0.038, "rank_ic_t": 9.88, "rank_icir": 0.24, "status": "Accepted"},
}


def import_alpha158(db: FactorDB) -> int:
    """Import all Alpha158 factors as Baseline."""
    factors = _get_alpha158_factors()
    count = 0
    for name, expr, cat in factors:
        db.upsert_factor(
            name=name,
            expression=expr,
            category=cat,
            source="Alpha158",
            status="Baseline",
            market_logic="Alpha158 standard factor",
        )
        count += 1
    return count


def import_custom_factors(db: FactorDB) -> int:
    """Import all custom candidate factors."""
    count = 0
    for name, expr, cat, logic in CUSTOM_FACTORS:
        hist = HISTORICAL_CSIALL_RESULTS.get(name)
        status = hist["status"] if hist else "Candidate"
        db.upsert_factor(
            name=name,
            expression=expr,
            category=cat,
            source="Custom",
            market_logic=logic,
            status=status,
        )
        # If we have historical csiall results, add as test result
        if hist:
            db.upsert_test_result(
                factor_name=name,
                market="csiall",
                test_start="2019-01-01",
                test_end="2025-12-31",
                rank_ic_mean=hist["rank_ic_mean"],
                rank_ic_t=hist["rank_ic_t"],
                rank_icir=hist["rank_icir"],
                significant=True,
                hea_round="HEA-2026-02-14-05",
                evidence="outputs/csiall_factor_significance.csv",
            )
        count += 1
    return count


def try_import_csv_results(db: FactorDB) -> int:
    """Try to import detailed results from csiall_factor_significance.csv if it exists."""
    import pandas as pd

    csv_path = PROJECT_ROOT / "outputs" / "csiall_factor_significance.csv"
    if not csv_path.exists():
        print(f"  (no CSV found at {csv_path}, skipping detailed import)")
        return 0

    df = pd.read_csv(csv_path)
    count = 0
    for _, row in df.iterrows():
        name = row["factor"]
        fdr_p = row.get("rank_ic_p_fdr", 1.0)
        is_sig = bool(not pd.isna(fdr_p) and fdr_p < 0.01)

        # Update factor status based on significance
        existing = db.get_factor(name)
        if existing and existing["status"] != "Baseline":
            db.upsert_factor(name=name, status="Accepted" if is_sig else "Rejected")

        db.upsert_test_result(
            factor_name=name,
            market="csiall",
            test_start="2019-01-01",
            test_end="2025-12-31",
            n_days=int(row["n_days"]) if "n_days" in row and not pd.isna(row.get("n_days")) else None,
            ic_mean=float(row["ic_mean"]) if "ic_mean" in row and not pd.isna(row.get("ic_mean")) else None,
            ic_std=float(row["ic_std"]) if "ic_std" in row and not pd.isna(row.get("ic_std")) else None,
            rank_ic_mean=float(row["rank_ic_mean"]) if not pd.isna(row.get("rank_ic_mean")) else None,
            rank_ic_std=float(row["rank_ic_std"]) if "rank_ic_std" in row and not pd.isna(row.get("rank_ic_std")) else None,
            rank_ic_t=float(row["rank_ic_t"]) if not pd.isna(row.get("rank_ic_t")) else None,
            rank_ic_p=float(row["rank_ic_p"]) if not pd.isna(row.get("rank_ic_p")) else None,
            rank_icir=float(row["rank_icir"]) if not pd.isna(row.get("rank_icir")) else None,
            fdr_p=float(fdr_p) if not pd.isna(fdr_p) else None,
            significant=is_sig,
            evidence="outputs/csiall_factor_significance.csv",
            hea_round="HEA-2026-02-14-05",
        )
        count += 1
    return count


def main():
    db = FactorDB()
    print("=== Importing factors into SQLite factor library ===")
    print(f"Database: {db.db_path}")
    print()

    # 1. Import Alpha158
    n158 = import_alpha158(db)
    print(f"[1/3] Imported {n158} Alpha158 baseline factors")

    # 2. Import custom factors
    n_custom = import_custom_factors(db)
    print(f"[2/3] Imported {n_custom} custom candidate factors")

    # 3. Try importing detailed CSV results
    n_csv = try_import_csv_results(db)
    if n_csv:
        print(f"[3/3] Updated {n_csv} factors with detailed test results from CSV")
    else:
        print(f"[3/3] No CSV results to import (will be populated during future HEA rounds)")

    print()
    print(db.summary())
    db.close()


if __name__ == "__main__":
    main()

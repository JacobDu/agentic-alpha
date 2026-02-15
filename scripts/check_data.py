"""Check available data fields and date range."""
import sys
import multiprocessing
sys.path.insert(0, "src")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

from project_qlib.runtime import init_qlib
init_qlib()
from qlib.data import D

# Check available fields
print("=== Available Fields ===")
fields = ["close", "open", "high", "low", "volume", "vwap", "amount", "change", "factor", "adjclose"]
for f in fields:
    try:
        df = D.features(["sh600000"], [f"${f}"], start_time="2026-02-10", end_time="2026-02-13")
        if len(df) > 0:
            print(f"  ${f}: {len(df)} rows, last={df.iloc[-1,0]:.4f}")
        else:
            print(f"  ${f}: empty")
    except Exception as e:
        print(f"  ${f}: ERROR {e}")

# Date range
print("\n=== Date Range (sh600000) ===")
df = D.features(["sh600000"], ["$close"], start_time="2000-01-01", end_time="2030-12-31")
dates = df.index.get_level_values(1)
print(f"  Start: {dates.min()}")
print(f"  End: {dates.max()}")
print(f"  Total days: {len(dates)}")

# 2026 data
df26 = D.features(["sh600000"], ["$close"], start_time="2026-01-01", end_time="2026-12-31")
print(f"\n  2026: {len(df26)} days")
if len(df26) > 0:
    d26 = df26.index.get_level_values(1)
    print(f"  2026 range: {d26.min()} ~ {d26.max()}")

# Check for financial data fields (common: pe, pb, ps, etc.)
print("\n=== Testing Financial Fields ===")
fin_fields = ["pe", "pb", "ps", "pcf", "market_cap", "total_mv", "circ_mv",
              "turnover_rate", "pe_ttm", "eps", "roe", "roa", "bps"]
for f in fin_fields:
    try:
        df = D.features(["sh600000"], [f"${f}"], start_time="2025-01-01", end_time="2026-02-13")
        if len(df) > 0 and df.iloc[:, 0].notna().any():
            print(f"  ${f}: OK ({df.iloc[:, 0].notna().sum()} non-null)")
        else:
            print(f"  ${f}: all NaN or empty")
    except Exception as e:
        print(f"  ${f}: NOT AVAILABLE")

# Market coverage
if __name__ == "__main__":
    print("\n=== Market Coverage ===")
    for market in ["csi300", "csi500", "csi1000", "csiall"]:
        try:
            inst = D.instruments(market)
            df = D.features(inst, ["$close"], start_time="2026-02-13", end_time="2026-02-13")
            print(f"  {market}: {len(df)} stocks on 2026-02-13")
        except Exception as e:
            err = str(e).split("\n")[0]
            print(f"  {market}: ERROR {err}")

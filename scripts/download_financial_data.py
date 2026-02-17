"""Download financial data from baostock and inject into Qlib binary format.

Phase 1: Daily valuation data (PE, PB, PS, PCF, turnover) via K-line API.
  - Fast: ~0.2s per stock, ~10 min for 1900+ CSI1000 stocks
  - Daily frequency: maps directly to Qlib binary format
  - Fields injected: $pe_ttm, $pb_mrq, $ps_ttm, $pcf_ttm, $turnover_rate

Phase 2 (optional): Quarterly financial data (ROE, growth, margins).
  - Slow: ~7-8 hours for all stocks
  - Quarterly data forward-filled to daily using publish date

Usage:
    # Phase 1: daily valuation (fast)
    uv run python scripts/download_financial_data.py --phase 1

    # Phase 2: quarterly fundamentals (slow, resumable)
    uv run python scripts/download_financial_data.py --phase 2

    # Check status
    uv run python scripts/download_financial_data.py --status

    # Specific stocks only
    uv run python scripts/download_financial_data.py --phase 1 --market csi1000
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
QLIB_DATA_DIR = PROJECT_ROOT / "data" / "qlib" / "cn_data"
FEATURES_DIR = QLIB_DATA_DIR / "features"
CALENDAR_FILE = QLIB_DATA_DIR / "calendars" / "day.txt"
CACHE_DIR = PROJECT_ROOT / "data" / "financial_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Baostock K-line valuation fields (daily frequency)
VALUATION_FIELDS = "date,code,peTTM,pbMRQ,psTTM,pcfNcfTTM,turn,isST"

# Mapping: baostock field -> Qlib feature name
VALUATION_MAP = {
    "peTTM": "pe_ttm",
    "pbMRQ": "pb_mrq",
    "psTTM": "ps_ttm",
    "pcfNcfTTM": "pcf_ttm",
    "turn": "turnover_rate",
}

# Quarterly financial data field config
QUARTERLY_APIS = {
    "profit": {
        "fields": ["roeAvg", "npMargin", "gpMargin", "epsTTM", "netProfit"],
        "qlib_names": ["roe", "net_profit_margin", "gross_profit_margin", "eps_ttm", "net_profit"],
    },
    "growth": {
        "fields": ["YOYNI", "YOYEPSBasic", "YOYPNI", "YOYEquity"],
        "qlib_names": ["yoy_ni", "yoy_eps", "yoy_pni", "yoy_equity"],
    },
    "balance": {
        "fields": ["liabilityToAsset", "assetToEquity"],
        "qlib_names": ["debt_ratio", "equity_multiplier"],
    },
}


def _load_calendar() -> list[str]:
    """Load Qlib trading calendar dates."""
    with open(CALENDAR_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def _calendar_index(calendar: list[str]) -> dict[str, int]:
    """Map date string -> index in calendar."""
    return {d: i for i, d in enumerate(calendar)}


def _qlib_code_to_baostock(qlib_code: str) -> str:
    """Convert Qlib code (SH600000) to baostock (sh.600000)."""
    return qlib_code[:2].lower() + "." + qlib_code[2:]


def _qlib_code_to_dir(qlib_code: str) -> str:
    """Convert Qlib code (SH600000) to feature dir name (sh600000)."""
    return qlib_code.lower()


def _get_stock_list(market: str = "all") -> list[str]:
    """Get list of Qlib stock codes for a market."""
    from project_qlib.runtime import init_qlib
    init_qlib()
    from qlib.data import D

    instruments = D.instruments(market)
    stocks = D.list_instruments(instruments, start_time="2015-01-01",
                                end_time="2026-12-31", as_list=True)
    # Only SH/SZ (skip BJ stocks - baostock has limited BJ support)
    return [s for s in stocks if s.startswith("SH") or s.startswith("SZ")]


def _read_bin_start_idx(feature_dir: Path) -> int | None:
    """Read the calendar start index from an existing .day.bin file.

    Qlib binary format: arr[0] = calendar_start_index (float32),
                        arr[1:] = values for each calendar day.
    """
    close_bin = feature_dir / "close.day.bin"
    if not close_bin.exists():
        return None
    arr = np.fromfile(str(close_bin), dtype=np.float32, count=1)
    return int(arr[0])


def _write_qlib_bin(feature_dir: Path, field_name: str,
                    date_values: dict[str, float],
                    calendar: list[str], cal_idx: dict[str, int]) -> int:
    """Write a single feature as Qlib binary (.day.bin) file.

    Qlib binary format:
      arr[0] = calendar start index (float32 header)
      arr[1:] = float32 values, one per calendar day from start_idx

    Aligns to the same range as the stock's close.day.bin file.
    Returns number of valid (non-NaN) values written.
    """
    if not date_values:
        return 0

    close_bin = feature_dir / "close.day.bin"
    if not close_bin.exists():
        return 0

    # Read existing bin metadata
    start_idx = _read_bin_start_idx(feature_dir)
    if start_idx is None:
        return 0

    # Number of data days (excluding the header)
    n_data = close_bin.stat().st_size // 4 - 1

    # Build output array: [start_idx, val0, val1, ..., valN]
    output = np.full(n_data + 1, np.nan, dtype=np.float32)
    output[0] = float(start_idx)

    n_valid = 0
    for date_str, val in date_values.items():
        if date_str in cal_idx:
            data_idx = cal_idx[date_str] - start_idx
            if 0 <= data_idx < n_data:
                output[data_idx + 1] = val  # +1 for header
                n_valid += 1

    out_path = feature_dir / f"{field_name}.day.bin"
    output.tofile(str(out_path))
    return n_valid


def download_phase1(market: str = "all", start: str = "2015-01-01",
                    end: str = "2026-02-13") -> None:
    """Phase 1: Download daily valuation data and inject into Qlib."""
    import baostock as bs

    stocks = _get_stock_list(market)
    print(f"Phase 1: Downloading daily valuation for {len(stocks)} stocks ({market})")
    print(f"Period: {start} ~ {end}")
    print(f"Fields: {list(VALUATION_MAP.values())}")

    calendar = _load_calendar()
    cal_idx = _calendar_index(calendar)

    bs.login()
    success = 0
    skipped = 0
    errors = 0
    t_start = time.time()

    for i, qlib_code in enumerate(stocks):
        bs_code = _qlib_code_to_baostock(qlib_code)
        dir_name = _qlib_code_to_dir(qlib_code)
        feature_dir = FEATURES_DIR / dir_name

        if not feature_dir.exists():
            skipped += 1
            continue

        # Check if already downloaded (skip if pe_ttm.day.bin exists and recent)
        pe_bin = feature_dir / "pe_ttm.day.bin"
        if pe_bin.exists() and pe_bin.stat().st_size > 1000:
            skipped += 1
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(stocks)}] {elapsed:.0f}s - skipped (already exists)")
            continue

        try:
            rs = bs.query_history_k_data_plus(
                bs_code, VALUATION_FIELDS,
                start_date=start, end_date=end,
                frequency="d", adjustflag="3",
            )

            # Use get_data() for fast batch retrieval
            df = rs.get_data()

            if df.empty:
                skipped += 1
                continue

            # Write each valuation field to Qlib binary
            for bs_field, qlib_field in VALUATION_MAP.items():
                date_values = {}
                for date_str, val_str in zip(df["date"], df[bs_field]):
                    if val_str and val_str != "":
                        try:
                            date_values[date_str] = float(val_str)
                        except (ValueError, TypeError):
                            pass

                _write_qlib_bin(feature_dir, qlib_field, date_values, calendar, cal_idx)

            success += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error for {qlib_code}: {e}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(stocks) - i - 1) / rate
            print(f"  [{i+1}/{len(stocks)}] {elapsed:.0f}s elapsed, "
                  f"ETA {eta:.0f}s, success={success}, skip={skipped}, err={errors}")

    bs.logout()
    elapsed = time.time() - t_start
    print(f"\nPhase 1 complete: {success} stocks written, "
          f"{skipped} skipped, {errors} errors, {elapsed:.0f}s total")
    print(f"New Qlib fields available: {list(VALUATION_MAP.values())}")
    print(f"Use in expressions: $pe_ttm, $pb_mrq, $ps_ttm, $pcf_ttm, $turnover_rate")


def download_phase2(market: str = "all", start_year: int = 2015,
                    end_year: int = 2024) -> None:
    """Phase 2: Download quarterly financial data and inject into Qlib.

    Uses publish date (pubDate) for forward-fill to avoid lookahead bias.
    This is slow (~2-3 seconds per stock) but resumable via cache.
    """
    import baostock as bs

    stocks = _get_stock_list(market)
    print(f"Phase 2: Downloading quarterly financial data for {len(stocks)} stocks")
    print(f"Period: {start_year} ~ {end_year}")

    calendar = _load_calendar()
    cal_idx = _calendar_index(calendar)

    # Cache file for resume support
    cache_file = CACHE_DIR / f"quarterly_{market}_{start_year}_{end_year}.parquet"
    processed_stocks: set[str] = set()

    if cache_file.exists():
        cached_df = pd.read_parquet(cache_file)
        processed_stocks = set(cached_df["qlib_code"].unique())
        print(f"  Resuming: {len(processed_stocks)} stocks already cached")
    else:
        cached_df = pd.DataFrame()

    remaining = [s for s in stocks if s not in processed_stocks]
    if not remaining:
        print("All stocks already cached. Injecting into Qlib...")
        _inject_quarterly_to_qlib(cached_df, calendar, cal_idx)
        return

    print(f"  Remaining: {len(remaining)} stocks to download")

    bs.login()
    all_records = []
    t_start = time.time()

    for i, qlib_code in enumerate(remaining):
        bs_code = _qlib_code_to_baostock(qlib_code)
        dir_name = _qlib_code_to_dir(qlib_code)
        feature_dir = FEATURES_DIR / dir_name

        if not feature_dir.exists():
            continue

        try:
            for yr in range(start_year, end_year + 1):
                for q in range(1, 5):
                    record = {"qlib_code": qlib_code, "year": yr, "quarter": q}

                    # Profit data
                    rs = bs.query_profit_data(code=bs_code, year=yr, quarter=q)
                    if rs.error_code == "0" and rs.next():
                        row = rs.get_row_data()
                        data = dict(zip(rs.fields, row))
                        record["pub_date"] = data.get("pubDate", "")
                        record["stat_date"] = data.get("statDate", "")
                        for f in QUARTERLY_APIS["profit"]["fields"]:
                            record[f] = data.get(f, "")

                    # Growth data
                    rs = bs.query_growth_data(code=bs_code, year=yr, quarter=q)
                    if rs.error_code == "0" and rs.next():
                        row = rs.get_row_data()
                        data = dict(zip(rs.fields, row))
                        if not record.get("pub_date"):
                            record["pub_date"] = data.get("pubDate", "")
                        for f in QUARTERLY_APIS["growth"]["fields"]:
                            record[f] = data.get(f, "")

                    # Balance data
                    rs = bs.query_balance_data(code=bs_code, year=yr, quarter=q)
                    if rs.error_code == "0" and rs.next():
                        row = rs.get_row_data()
                        data = dict(zip(rs.fields, row))
                        for f in QUARTERLY_APIS["balance"]["fields"]:
                            record[f] = data.get(f, "")

                    all_records.append(record)

        except Exception as e:
            print(f"  Error for {qlib_code}: {e}")

        # Save checkpoint every 100 stocks
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(remaining) - i - 1) / rate

            # Append to cache
            new_df = pd.DataFrame(all_records)
            combined = pd.concat([cached_df, new_df], ignore_index=True)
            combined.to_parquet(cache_file)
            cached_df = combined
            all_records = []

            print(f"  [{i+1}/{len(remaining)}] {elapsed:.0f}s elapsed, "
                  f"ETA {eta:.0f}s ({eta/60:.0f}min), saved checkpoint")

    bs.logout()

    # Final save
    if all_records:
        new_df = pd.DataFrame(all_records)
        combined = pd.concat([cached_df, new_df], ignore_index=True)
        combined.to_parquet(cache_file)
        cached_df = combined

    elapsed = time.time() - t_start
    print(f"\nPhase 2 download complete: {len(cached_df)} records, {elapsed:.0f}s")

    # Inject into Qlib
    _inject_quarterly_to_qlib(cached_df, calendar, cal_idx)


def _inject_quarterly_to_qlib(df: pd.DataFrame, calendar: list[str],
                               cal_idx: dict[str, int]) -> None:
    """Convert quarterly data to daily and write Qlib binary files.

    Uses pubDate for forward-fill (point-in-time, no lookahead).
    """
    if df.empty:
        print("No quarterly data to inject.")
        return

    # All fields to inject
    all_fields = {}
    for api_cfg in QUARTERLY_APIS.values():
        for bs_f, q_f in zip(api_cfg["fields"], api_cfg["qlib_names"]):
            all_fields[bs_f] = q_f

    stocks = df["qlib_code"].unique()
    print(f"Injecting quarterly data for {len(stocks)} stocks, {len(all_fields)} fields")

    success = 0
    for i, qlib_code in enumerate(stocks):
        dir_name = _qlib_code_to_dir(qlib_code)
        feature_dir = FEATURES_DIR / dir_name

        if not feature_dir.exists():
            continue

        stock_df = df[df["qlib_code"] == qlib_code].copy()

        # Convert pub_date to proper format, sort by it
        stock_df = stock_df[stock_df["pub_date"].astype(str).str.len() >= 10]
        if stock_df.empty:
            continue

        stock_df = stock_df.sort_values("pub_date")

        for bs_field, qlib_field in all_fields.items():
            if bs_field not in stock_df.columns:
                continue

            # Build date->value mapping using pubDate (forward-fill)
            date_values = {}
            pub_val_pairs = []

            for _, row in stock_df.iterrows():
                pub_date = str(row.get("pub_date", ""))
                val_str = str(row.get(bs_field, ""))
                if pub_date and val_str and val_str != "" and val_str != "nan":
                    try:
                        pub_val_pairs.append((pub_date, float(val_str)))
                    except (ValueError, TypeError):
                        pass

            if not pub_val_pairs:
                continue

            # Forward fill: for each calendar day, use the most recent published value
            current_val = None
            pair_idx = 0
            for cal_date in calendar:
                while pair_idx < len(pub_val_pairs) and pub_val_pairs[pair_idx][0] <= cal_date:
                    current_val = pub_val_pairs[pair_idx][1]
                    pair_idx += 1
                if current_val is not None:
                    date_values[cal_date] = current_val

            _write_qlib_bin(feature_dir, qlib_field, date_values, calendar, cal_idx)

        success += 1
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(stocks)}] injected")

    print(f"Injected quarterly data for {success} stocks")
    print(f"New Qlib fields: {list(all_fields.values())}")


def check_status() -> None:
    """Check which financial features are available in Qlib data."""
    print("=== Financial Data Status ===\n")

    all_fields = list(VALUATION_MAP.values())
    for api_cfg in QUARTERLY_APIS.values():
        all_fields.extend(api_cfg["qlib_names"])

    # Sample a few stocks
    sample_dirs = sorted(FEATURES_DIR.iterdir())[:20]
    sh_sz_dirs = [d for d in sample_dirs if d.name.startswith("sh") or d.name.startswith("sz")]

    if not sh_sz_dirs:
        # Try to find some SH/SZ stocks
        for d in FEATURES_DIR.iterdir():
            if d.name.startswith("sh") or d.name.startswith("sz"):
                sh_sz_dirs.append(d)
                if len(sh_sz_dirs) >= 10:
                    break

    print(f"Checking {len(sh_sz_dirs)} sample stocks...\n")

    field_stats = {f: {"exists": 0, "total": len(sh_sz_dirs)} for f in all_fields}

    for d in sh_sz_dirs:
        for field in all_fields:
            bin_path = d / f"{field}.day.bin"
            if bin_path.exists() and bin_path.stat().st_size > 100:
                field_stats[field]["exists"] += 1

    print(f"{'Field':<20} {'Available':>10} {'Coverage':>10}")
    print("-" * 42)
    for field, stat in field_stats.items():
        pct = stat["exists"] / stat["total"] * 100 if stat["total"] > 0 else 0
        status = "✓" if pct > 80 else ("◐" if pct > 0 else "✗")
        print(f"{field:<20} {stat['exists']:>5}/{stat['total']:<5} {pct:>6.0f}% {status}")

    # Check cache files
    print(f"\n=== Cache Files ===")
    for f in sorted(CACHE_DIR.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download financial data from baostock and inject into Qlib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", type=int, choices=[1, 2],
                        help="1=daily valuation (fast), 2=quarterly fundamentals (slow)")
    parser.add_argument("--market", default="all",
                        help="Stock universe (all, csi1000, csi300, etc.)")
    parser.add_argument("--start", default="2000-01-01",
                        help="Start date for Phase 1 (default: 2000-01-01)")
    parser.add_argument("--end", default="2026-02-13",
                        help="End date for Phase 1")
    parser.add_argument("--start-year", type=int, default=2015,
                        help="Start year for Phase 2")
    parser.add_argument("--end-year", type=int, default=2024,
                        help="End year for Phase 2")
    parser.add_argument("--status", action="store_true",
                        help="Check which financial features are available")

    args = parser.parse_args()

    if args.status:
        check_status()
        return 0

    if args.phase is None:
        parser.print_help()
        return 1

    if args.phase == 1:
        download_phase1(market=args.market, start=args.start, end=args.end)
    elif args.phase == 2:
        download_phase2(market=args.market, start_year=args.start_year,
                        end_year=args.end_year)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

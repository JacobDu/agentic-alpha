"""Download ALL financial data from baostock and inject into Qlib binary format.

Unified data source: all market data comes from baostock, supporting
incremental updates. No dependency on chenditc/investment_data for ongoing use.

Usage:
    # Download all data (daily + quarterly)
    uv run python scripts/download_financial_data.py

    # Phase 1 only: daily OHLCV + valuation (incremental by default)
    uv run python scripts/download_financial_data.py --phase 1

    # Phase 2 only: quarterly fundamentals + dividends + industry (resumable)
    uv run python scripts/download_financial_data.py --phase 2

    # Force full re-download from baostock
    uv run python scripts/download_financial_data.py --force

    # Check status of all features
    uv run python scripts/download_financial_data.py --status

Injected Qlib features:
    Phase 1 (daily, from K-line API + query_adjust_factor, 2 API calls per stock):
        Price:     $open, $close, $high, $low, $volume, $amount
        Derived:   $vwap, $change, $factor, $turnover_rate
        Valuation: $pe_ttm, $pb_mrq, $ps_ttm, $pcf_ttm
    Phase 2 (quarterly -> daily via forward-fill by pubDate):
        Profitability: $roe, $npm, $gpm, $eps_ttm
        Growth:        $yoy_ni, $yoy_eps, $yoy_pni, $yoy_equity, $yoy_asset
        Balance:       $debt_ratio, $eq_multiplier, $yoy_liability
        Operation:     $asset_turnover
        CashFlow:      $cfo_to_or, $cfo_to_np
        DuPont:        $dupont_roe, $dupont_asset_turn, $dupont_nito_gr, $dupont_tax
        Dividend:      $div_ps (per-share pre-tax cash dividend, ffill by regist date)
    Industry: data/industry.parquet
"""
from __future__ import annotations

import argparse
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

# --- Phase 1: Daily data from baostock K-line API ---
# All K-line fields we need (used with adjustflag="3" for raw data)
DAILY_KLINE_FIELDS = (
    "date,open,high,low,close,volume,amount,"
    "turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
)

# Direct mapping: baostock field -> Qlib field (from adjustflag="2" query)
DAILY_DIRECT_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "peTTM": "pe_ttm",
    "pbMRQ": "pb_mrq",
    "psTTM": "ps_ttm",
    "pcfNcfTTM": "pcf_ttm",
}

# Computed fields: turnover_rate, factor, vwap, change
# All Phase 1 Qlib field names
PHASE1_FIELDS = (
    list(DAILY_DIRECT_MAP.values())
    + ["turnover_rate", "factor", "vwap", "change"]
)

# --- Phase 2: Quarterly financial data configuration ---
QUARTERLY_APIS_CONFIG = {
    "profit": {
        "fn": "query_profit_data",
        "fields": {
            "roeAvg": "roe",
            "npMargin": "npm",
            "gpMargin": "gpm",
            "epsTTM": "eps_ttm",
        },
    },
    "growth": {
        "fn": "query_growth_data",
        "fields": {
            "YOYNI": "yoy_ni",
            "YOYEPSBasic": "yoy_eps",
            "YOYPNI": "yoy_pni",
            "YOYEquity": "yoy_equity",
            "YOYAsset": "yoy_asset",
        },
    },
    "balance": {
        "fn": "query_balance_data",
        "fields": {
            "liabilityToAsset": "debt_ratio",
            "assetToEquity": "eq_multiplier",
            "YOYLiability": "yoy_liability",
        },
    },
    "operation": {
        "fn": "query_operation_data",
        "fields": {
            "AssetTurnRatio": "asset_turnover",
        },
    },
    "cash_flow": {
        "fn": "query_cash_flow_data",
        "fields": {
            "CFOToOR": "cfo_to_or",
            "CFOToNP": "cfo_to_np",
        },
    },
    "dupont": {
        "fn": "query_dupont_data",
        "fields": {
            "dupontROE": "dupont_roe",
            "dupontAssetTurn": "dupont_asset_turn",
            "dupontNitogr": "dupont_nito_gr",
            "dupontTaxBurden": "dupont_tax",
        },
    },
}

# Collect all quarterly field mappings
ALL_QUARTERLY_FIELDS: dict[str, str] = {}
for _cfg in QUARTERLY_APIS_CONFIG.values():
    ALL_QUARTERLY_FIELDS.update(_cfg["fields"])

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _load_calendar() -> list[str]:
    if not CALENDAR_FILE.exists():
        return []
    with open(CALENDAR_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def _calendar_index(calendar: list[str]) -> dict[str, int]:
    return {d: i for i, d in enumerate(calendar)}


def _qlib_code_to_baostock(qlib_code: str) -> str:
    return qlib_code[:2].lower() + "." + qlib_code[2:]


def _get_stock_list(market: str = "all") -> list[str]:
    from project_qlib.runtime import init_qlib
    init_qlib()
    from qlib.data import D
    instruments = D.instruments(market)
    stocks = D.list_instruments(instruments, start_time="2000-01-01",
                                end_time="2026-12-31", as_list=True)
    return [s for s in stocks if s.startswith("SH") or s.startswith("SZ")]


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None if invalid."""
    if val is None:
        return None
    s = str(val).strip()
    if s in ("", "nan", "None"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _extend_calendar(end_date: str) -> list[str]:
    """Extend the trading calendar using baostock trade dates API.

    Returns the full calendar (extended if new trading days found).
    baostock must already be logged in before calling this.
    """
    import baostock as bs

    calendar = _load_calendar()
    last_cal_date = calendar[-1] if calendar else "2000-01-01"

    if end_date <= last_cal_date:
        return calendar

    rs = bs.query_trade_dates(start_date=last_cal_date, end_date=end_date)
    df = rs.get_data()

    new_dates = []
    for _, row in df.iterrows():
        d = row["calendar_date"]
        is_trading = str(row.get("is_trading_day", "0"))
        if d > last_cal_date and is_trading == "1":
            new_dates.append(d)

    if new_dates:
        with open(CALENDAR_FILE, "a") as f:
            for d in new_dates:
                f.write(f"\n{d}")
        calendar.extend(new_dates)
        print(f"  Calendar extended: +{len(new_dates)} days "
              f"({new_dates[0]} ~ {new_dates[-1]})")

    return calendar


def _write_qlib_bin(feature_dir: Path, field_name: str,
                    date_values: dict[str, float],
                    calendar: list[str], cal_idx: dict[str, int],
                    merge: bool = False) -> int:
    """Write a single feature as Qlib .day.bin file.

    Format: arr[0] = calendar start index (header), arr[1:] = float32 values.
    If merge=True, read existing file and overlay new data on top.
    Uses close.day.bin as reference for start_idx and data length.
    """
    close_bin = feature_dir / "close.day.bin"
    if not close_bin.exists():
        return 0

    start_idx = int(np.fromfile(str(close_bin), dtype=np.float32, count=1)[0])
    n_data = close_bin.stat().st_size // 4 - 1

    target = feature_dir / f"{field_name}.day.bin"

    # In merge mode, load existing data as base
    if merge and target.exists() and target.stat().st_size == (n_data + 1) * 4:
        output = np.fromfile(str(target), dtype=np.float32)
    else:
        output = np.full(n_data + 1, np.nan, dtype=np.float32)
        output[0] = float(start_idx)

    n_valid = 0
    for date_str, val in date_values.items():
        if date_str in cal_idx:
            data_idx = cal_idx[date_str] - start_idx
            if 0 <= data_idx < n_data:
                output[data_idx + 1] = val
                n_valid += 1

    output.tofile(str(target))
    return n_valid


def _write_daily_data(feature_dir: Path,
                      fields: dict[str, dict[str, float]],
                      calendar: list[str], cal_idx: dict[str, int],
                      merge: bool = False) -> int:
    """Write multiple daily fields for base price data.

    Unlike _write_qlib_bin (which references close.day.bin), this function
    determines sizing from the data itself for new stocks, or extends
    existing binary files for incremental/calendar-extension updates.

    Args:
        fields: {qlib_field_name: {date_str: float_value}}
        merge: if True, overlay new data on existing binary files.

    Returns:
        Total number of data points written across all fields.
    """
    feature_dir.mkdir(parents=True, exist_ok=True)
    close_bin = feature_dir / "close.day.bin"

    # Determine start_idx
    if close_bin.exists() and close_bin.stat().st_size >= 4:
        start_idx = int(np.fromfile(str(close_bin), dtype=np.float32,
                                    count=1)[0])
    else:
        # New stock: find earliest date across all fields
        min_idx = len(calendar)
        for dv in fields.values():
            for d in dv:
                if d in cal_idx:
                    min_idx = min(min_idx, cal_idx[d])
        if min_idx >= len(calendar):
            return 0
        start_idx = min_idx

    # Determine n_data: max of existing coverage and new data coverage
    existing_n = 0
    if merge and close_bin.exists():
        existing_n = close_bin.stat().st_size // 4 - 1

    max_cal_idx = start_idx
    for dv in fields.values():
        for d in dv:
            if d in cal_idx:
                max_cal_idx = max(max_cal_idx, cal_idx[d])

    n_data = max(existing_n, max_cal_idx - start_idx + 1)
    if n_data <= 0:
        return 0

    total_written = 0
    # Write 'close' first (it's the reference for other functions)
    ordered = sorted(fields.keys(), key=lambda f: (0 if f == "close" else 1, f))

    for field_name in ordered:
        date_values = fields[field_name]
        target = feature_dir / f"{field_name}.day.bin"

        if merge and target.exists() and target.stat().st_size >= 4:
            existing = np.fromfile(str(target), dtype=np.float32)
            output = np.full(n_data + 1, np.nan, dtype=np.float32)
            output[0] = float(start_idx)
            copy_len = min(len(existing) - 1, n_data)
            if copy_len > 0:
                output[1:copy_len + 1] = existing[1:copy_len + 1]
        else:
            output = np.full(n_data + 1, np.nan, dtype=np.float32)
            output[0] = float(start_idx)

        for date_str, val in date_values.items():
            if date_str in cal_idx:
                data_idx = cal_idx[date_str] - start_idx
                if 0 <= data_idx < n_data:
                    output[data_idx + 1] = val
                    total_written += 1

        output.tofile(str(target))

    return total_written


def _fix_nan_headers() -> int:
    """Fix .day.bin files that have NaN as start index header."""
    all_fields = PHASE1_FIELDS + list(ALL_QUARTERLY_FIELDS.values()) + ["div_ps"]
    fixed = 0
    for d in FEATURES_DIR.iterdir():
        if not (d.name.startswith("sh") or d.name.startswith("sz")):
            continue
        close_bin = d / "close.day.bin"
        if not close_bin.exists():
            continue
        start_idx = np.fromfile(str(close_bin), dtype=np.float32, count=1)[0]
        if np.isnan(start_idx):
            continue

        for field in all_fields:
            if field == "close":
                continue
            fbin = d / f"{field}.day.bin"
            if not fbin.exists():
                continue
            arr = np.fromfile(str(fbin), dtype=np.float32, count=1)
            if np.isnan(arr[0]):
                full = np.fromfile(str(fbin), dtype=np.float32)
                full[0] = start_idx
                full.tofile(str(fbin))
                fixed += 1
    return fixed


def _get_last_valid_date(feature_dir: Path, field_name: str,
                         calendar: list[str]) -> str | None:
    """Find the last date with valid (non-NaN) data in a .day.bin file."""
    fbin = feature_dir / f"{field_name}.day.bin"
    if not fbin.exists() or fbin.stat().st_size < 100:
        return None
    arr = np.fromfile(str(fbin), dtype=np.float32)
    start_idx = int(arr[0])
    data = arr[1:]
    valid_mask = ~np.isnan(data)
    if not valid_mask.any():
        return None
    last_offset = np.where(valid_mask)[0][-1]
    cal_idx_val = start_idx + last_offset
    if cal_idx_val < len(calendar):
        return calendar[cal_idx_val]
    return None


# -----------------------------------------------------------------------
# Phase 1: Daily OHLCV + valuation from baostock K-line API
# -----------------------------------------------------------------------

def _read_last_factor(feature_dir: Path) -> float | None:
    """Read the last valid factor value from existing factor.day.bin."""
    fbin = feature_dir / "factor.day.bin"
    if not fbin.exists() or fbin.stat().st_size < 100:
        return None
    arr = np.fromfile(str(fbin), dtype=np.float32)
    data = arr[1:]
    valid = ~np.isnan(data)
    if not valid.any():
        return None
    return float(data[np.where(valid)[0][-1]])


def _process_force_mode(bs, bs_code: str, dl_start: str, end: str,
                        field_data: dict[str, dict[str, float]]) -> bool:
    """Force mode: download raw K-line + adjust factor events.

    2 API calls per stock:
      1. Raw K-line (adjustflag="3") — all fields (volume/amount/valuation)
      2. query_adjust_factor — dividend events only (~10-30 rows, very fast)

    Factor is forward-filled from dividend events to daily granularity.
    Prices are computed as: adj_price = raw_price × foreAdjustFactor.

    Returns True if data was successfully downloaded.
    """
    # Query 1: Raw K-line (不复权) — all fields
    rs_raw = bs.query_history_k_data_plus(
        bs_code, DAILY_KLINE_FIELDS,
        start_date=dl_start, end_date=end,
        frequency="d", adjustflag="3",
    )
    df_raw = rs_raw.get_data()
    if df_raw.empty:
        return False

    # Query 2: Adjust factor events (from 1990 for complete history)
    rs_fac = bs.query_adjust_factor(
        code=bs_code, start_date="1990-01-01", end_date=end,
    )
    df_fac = rs_fac.get_data()

    # Build daily factor lookup via forward-fill
    if df_fac.empty:
        # No dividends/splits: factor = 1.0 for all dates
        daily_factor: dict[str, float] = {
            d: 1.0 for d in df_raw["date"]
        }
    else:
        factor_events = sorted(
            [(row["dividOperateDate"], float(row["foreAdjustFactor"]))
             for _, row in df_fac.iterrows()],
            key=lambda x: x[0],
        )
        daily_factor = {}
        fi = 0
        for d in sorted(df_raw["date"].values):
            while (fi < len(factor_events) - 1
                   and factor_events[fi + 1][0] <= d):
                fi += 1
            if factor_events[fi][0] <= d:
                daily_factor[d] = factor_events[fi][1]
            else:
                # Date before first factor event (shouldn't happen in
                # practice as first event ≈ IPO date). Use first event's
                # factor as safe default.
                daily_factor[d] = factor_events[0][1]

    for _, row in df_raw.iterrows():
        d = row["date"]
        factor = daily_factor.get(d, 1.0)

        # OHLC: raw × foreAdjustFactor = 前复权 price
        for bs_f in ("open", "high", "low", "close"):
            v = _safe_float(row.get(bs_f))
            if v is not None:
                field_data[bs_f][d] = v * factor

        # Raw volume/amount from baostock (same regardless of adjustflag)
        raw_vol = _safe_float(row.get("volume"))  # 股
        raw_amt = _safe_float(row.get("amount"))  # 元

        # Valuation: same regardless of adjustflag
        for bs_f, qlib_f in [("peTTM", "pe_ttm"), ("pbMRQ", "pb_mrq"),
                              ("psTTM", "ps_ttm"),
                              ("pcfNcfTTM", "pcf_ttm")]:
            v = _safe_float(row.get(bs_f))
            if v is not None:
                field_data[qlib_f][d] = v

        # turnover_rate = turn
        v = _safe_float(row.get("turn"))
        if v is not None:
            field_data["turnover_rate"][d] = v

        # change = pctChg / 100
        v = _safe_float(row.get("pctChg"))
        if v is not None:
            field_data["change"][d] = v / 100.0

        # factor = foreAdjustFactor (already computed via forward-fill)
        field_data["factor"][d] = factor

        # Volume/Amount unit conversion (match chenditc format):
        #   volume: 股 → 前复权手 = raw_vol / factor / 100
        #   amount: 元 → 千元 = raw_amt / 1000
        if raw_vol is not None and factor > 0:
            field_data["volume"][d] = raw_vol / factor / 100.0
        if raw_amt is not None:
            field_data["amount"][d] = raw_amt / 1000.0

        # vwap = raw_vwap × factor (前复权 price)
        if (raw_amt is not None and raw_vol is not None
                and raw_vol > 0):
            field_data["vwap"][d] = (raw_amt / raw_vol) * factor

    return True


def _process_incremental_mode(bs, bs_code: str, dl_start: str, end: str,
                              existing_factor: float,
                              field_data: dict[str, dict[str, float]]
                              ) -> bool:
    """Incremental mode: download raw prices, apply existing factor.

    Only 1 API call. Prices are scaled to be consistent with existing
    chenditc/previous data using the last known factor.

    Returns True if data was successfully downloaded.
    """
    rs = bs.query_history_k_data_plus(
        bs_code, DAILY_KLINE_FIELDS,
        start_date=dl_start, end_date=end,
        frequency="d", adjustflag="3",  # 不复权
    )
    df = rs.get_data()
    if df.empty:
        return False

    factor = existing_factor

    for _, row in df.iterrows():
        d = row["date"]

        # Raw OHLC → 前复权 by applying existing factor
        raw_open = _safe_float(row.get("open"))
        raw_high = _safe_float(row.get("high"))
        raw_low = _safe_float(row.get("low"))
        raw_close = _safe_float(row.get("close"))
        raw_vol = _safe_float(row.get("volume"))  # 股
        raw_amt = _safe_float(row.get("amount"))  # 元

        if raw_open is not None:
            field_data["open"][d] = raw_open * factor
        if raw_high is not None:
            field_data["high"][d] = raw_high * factor
        if raw_low is not None:
            field_data["low"][d] = raw_low * factor
        if raw_close is not None:
            field_data["close"][d] = raw_close * factor

        # Volume/Amount unit conversion (match chenditc format):
        #   volume: 股 → 前复权手 = raw_vol / factor / 100
        #   amount: 元 → 千元 = raw_amt / 1000
        if raw_vol is not None and factor > 0:
            field_data["volume"][d] = raw_vol / factor / 100.0
        if raw_amt is not None:
            field_data["amount"][d] = raw_amt / 1000.0

        # Valuation: same regardless of adjustflag
        for bs_f, qlib_f in [("peTTM", "pe_ttm"), ("pbMRQ", "pb_mrq"),
                              ("psTTM", "ps_ttm"),
                              ("pcfNcfTTM", "pcf_ttm")]:
            v = _safe_float(row.get(bs_f))
            if v is not None:
                field_data[qlib_f][d] = v

        # turnover_rate = turn
        v = _safe_float(row.get("turn"))
        if v is not None:
            field_data["turnover_rate"][d] = v

        # change = pctChg / 100
        v = _safe_float(row.get("pctChg"))
        if v is not None:
            field_data["change"][d] = v / 100.0

        # factor: same as existing
        field_data["factor"][d] = factor

        # vwap = raw_vwap × factor (前复权 price)
        if (raw_amt is not None and raw_vol is not None
                and raw_vol > 0):
            field_data["vwap"][d] = (raw_amt / raw_vol) * factor

    return True


def download_phase1(stocks: list[str], start: str = "2000-01-01",
                    end: str | None = None, force: bool = False) -> None:
    """Download daily OHLCV + valuation from baostock K-line API.

    Two modes:
      Force mode (--force): 2 API calls per stock.
        1) Raw K-line (adjustflag="3") — all fields
        2) query_adjust_factor — dividend events (~10-30 rows, very fast)
        Factor is forward-filled from dividend events to daily granularity.
        Replaces all existing data (baostock's 前复权 scale, base=today).

      Incremental mode (default): 1 API call per stock.
        Downloads raw (不复权) data, applies last known factor from existing
        binary files. Prices remain on the SAME scale as existing data
        (compatible with chenditc legacy data).

    Fields written: $open, $close, $high, $low, $volume, $amount,
                    $vwap, $change, $factor, $turnover_rate,
                    $pe_ttm, $pb_mrq, $ps_ttm, $pcf_ttm
    """
    import baostock as bs
    from datetime import date

    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    print(f"\n--- Phase 1: Daily OHLCV + Valuation for {len(stocks)} stocks ---")
    print(f"Period: {start} ~ {end}")
    if force:
        print(f"Mode: force re-download (2 API calls/stock, baostock scale)")
    else:
        print(f"Mode: incremental (1 API call/stock, existing scale)")

    bs.login()

    # Extend calendar to cover the requested end date
    calendar = _extend_calendar(end)
    cal_idx = _calendar_index(calendar)
    cal_end = calendar[-1] if calendar else ""

    success = skipped = updated = errors = 0
    t0 = time.time()

    for i, qlib_code in enumerate(stocks):
        bs_code = _qlib_code_to_baostock(qlib_code)
        feature_dir = FEATURES_DIR / qlib_code.lower()

        # Determine download start date (incremental)
        dl_start = start
        merge = False
        if not force:
            if not feature_dir.exists():
                skipped += 1
                continue

            # Check close.day.bin first, fallback to pe_ttm for legacy
            last_date = _get_last_valid_date(feature_dir, "close", calendar)
            if last_date is None:
                last_date = _get_last_valid_date(
                    feature_dir, "pe_ttm", calendar)

            if last_date:
                if last_date >= end or last_date >= cal_end:
                    skipped += 1
                    continue
                dl_start = last_date
                merge = True

        try:
            field_data: dict[str, dict[str, float]] = {
                f: {} for f in PHASE1_FIELDS
            }

            if force or not merge:
                # Force/new stock: 2 API calls, compute factor
                ok = _process_force_mode(bs, bs_code, dl_start, end,
                                         field_data)
            else:
                # Incremental: 1 API call, apply existing factor
                existing_factor = _read_last_factor(feature_dir)
                if existing_factor is None or existing_factor <= 0:
                    # Fallback to force mode for this stock
                    ok = _process_force_mode(bs, bs_code, dl_start, end,
                                             field_data)
                else:
                    ok = _process_incremental_mode(
                        bs, bs_code, dl_start, end,
                        existing_factor, field_data)

            if not ok:
                skipped += 1
                continue

            _write_daily_data(feature_dir, field_data, calendar, cal_idx,
                              merge=merge)

            if merge:
                updated += 1
            else:
                success += 1

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Error {qlib_code}: {e}")

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 1
            eta = (len(stocks) - i - 1) / rate
            print(f"  Phase1 [{i+1}/{len(stocks)}] {el:.0f}s, "
                  f"ETA {eta/60:.1f}min, new={success} upd={updated} "
                  f"skip={skipped} err={errors}")

    bs.logout()
    el = time.time() - t0
    print(f"Phase 1 done: {success} new, {updated} updated, "
          f"{skipped} skipped, {errors} errors, {el:.0f}s")


# -----------------------------------------------------------------------
# Phase 2: Quarterly financials + dividends + industry
# -----------------------------------------------------------------------

def download_phase2(stocks: list[str], start_year: int = 2010,
                    end_year: int | None = None, force: bool = False) -> None:
    """Download quarterly financials, dividends, and industry.

    Incremental by default: reads cached data, finds the latest quarter
    per stock, and only downloads newer quarters. Use force=True to
    clear cache and re-download everything.
    """
    import baostock as bs
    from datetime import date

    if end_year is None:
        end_year = date.today().year

    calendar = _load_calendar()
    cal_idx = _calendar_index(calendar)

    cache_file = CACHE_DIR / "phase2_quarterly.parquet"
    div_cache = CACHE_DIR / "phase2_dividends.parquet"
    industry_file = PROJECT_ROOT / "data" / "industry.parquet"

    # Clear cache if force mode
    if force:
        for f in [cache_file, div_cache, industry_file]:
            if f.exists():
                f.unlink()
                print(f"  Cleared: {f.name}")

    # Resume support — build set of (stock, year, quarter) already in cache
    cached_quarterly = pd.DataFrame()
    cached_dividends = pd.DataFrame()
    cached_yq: set[tuple[str, int, int]] = set()
    cached_div_years: set[tuple[str, int]] = set()

    if cache_file.exists():
        cached_quarterly = pd.read_parquet(cache_file)
        for _, row in cached_quarterly.iterrows():
            cached_yq.add((row["qlib_code"], int(row["year"]),
                           int(row["quarter"])))
    if div_cache.exists():
        cached_dividends = pd.read_parquet(div_cache)
        if "year" in cached_dividends.columns:
            for _, row in cached_dividends.iterrows():
                cached_div_years.add((row["qlib_code"], int(row["year"])))

    # Count what needs downloading
    all_yq_pairs = [(yr, q) for yr in range(start_year, end_year + 1)
                    for q in range(1, 5)]
    total_todo = sum(
        1 for s in stocks for yr, q in all_yq_pairs
        if (s, yr, q) not in cached_yq
    )
    stocks_todo = [s for s in stocks
                   if any((s, yr, q) not in cached_yq
                          for yr, q in all_yq_pairs)]

    print(f"\n--- Phase 2: Quarterly + Dividends + Industry ---")
    print(f"  Stocks: {len(stocks)} total, {len(stocks_todo)} need update")
    print(f"  Period: {start_year} ~ {end_year} ({len(all_yq_pairs)} quarters)")
    print(f"  Cached: {len(cached_yq)} stock-quarters, todo: {total_todo}")
    print(f"  Quarterly fields: {len(ALL_QUARTERLY_FIELDS)}")

    # If all downloaded, just inject
    if not stocks_todo:
        if not industry_file.exists():
            print("  All quarterly cached. Downloading industry...")
        else:
            print("  All data cached. Injecting to Qlib...")
            _inject_quarterly_to_qlib(cached_quarterly, calendar, cal_idx)
            _inject_dividends_to_qlib(cached_dividends, calendar, cal_idx)
            return

    bs.login()

    # --- Industry classification (fast, ~30s for all stocks) ---
    if not industry_file.exists():
        print("  Downloading industry classification...")
        ind_records = []
        for j, s in enumerate(stocks):
            bs_code = _qlib_code_to_baostock(s)
            rs = bs.query_stock_industry(code=bs_code)
            df = rs.get_data()
            if not df.empty:
                ind_records.append({
                    "qlib_code": s,
                    "industry": df["industry"].values[0],
                })
            if (j + 1) % 1000 == 0:
                print(f"    Industry [{j+1}/{len(stocks)}]")
        if ind_records:
            ind_df = pd.DataFrame(ind_records)
            ind_df.to_parquet(industry_file, index=False)
            print(f"  Industry: {len(ind_df)} stocks, "
                  f"{ind_df['industry'].nunique()} categories")

    # --- Quarterly data + dividends ---
    quarterly_records: list[dict] = []
    dividend_records: list[dict] = []
    t0 = time.time()

    for i, qlib_code in enumerate(stocks_todo):
        bs_code = _qlib_code_to_baostock(qlib_code)
        feature_dir = FEATURES_DIR / qlib_code.lower()
        if not feature_dir.exists():
            continue

        # Quarterly financials — skip already-cached (year, quarter) pairs
        for yr in range(start_year, end_year + 1):
            for q in range(1, 5):
                if (qlib_code, yr, q) in cached_yq:
                    continue

                record: dict = {"qlib_code": qlib_code, "year": yr,
                                "quarter": q}

                for api_name, api_cfg in QUARTERLY_APIS_CONFIG.items():
                    fn = getattr(bs, api_cfg["fn"])
                    try:
                        rs = fn(code=bs_code, year=yr, quarter=q)
                        if rs.error_code == "0" and rs.next():
                            row_data = dict(zip(rs.fields,
                                                rs.get_row_data()))
                            if not record.get("pub_date"):
                                record["pub_date"] = row_data.get(
                                    "pubDate", "")
                            for bs_f in api_cfg["fields"]:
                                record[bs_f] = row_data.get(bs_f, "")
                    except Exception:
                        pass

                quarterly_records.append(record)

            # Dividends — skip already-cached years
            if (qlib_code, yr) not in cached_div_years:
                try:
                    rs = bs.query_dividend_data(
                        code=bs_code, year=str(yr), yearType="report"
                    )
                    div_df = rs.get_data()
                    if not div_df.empty:
                        for _, drow in div_df.iterrows():
                            cash_ps = drow.get("dividCashPsBeforeTax", "")
                            reg_date = drow.get("dividRegistDate", "")
                            if (cash_ps and cash_ps != ""
                                    and reg_date and reg_date != ""):
                                try:
                                    dividend_records.append({
                                        "qlib_code": qlib_code,
                                        "year": yr,
                                        "regist_date": reg_date,
                                        "cash_ps": float(cash_ps),
                                    })
                                except (ValueError, TypeError):
                                    pass
                except Exception:
                    pass

        # Checkpoint every 50 stocks
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 1
            eta = (len(stocks_todo) - i - 1) / rate

            new_q = pd.DataFrame(quarterly_records)
            combined_q = pd.concat([cached_quarterly, new_q],
                                   ignore_index=True)
            combined_q.to_parquet(cache_file)
            cached_quarterly = combined_q
            quarterly_records = []

            if dividend_records:
                new_d = pd.DataFrame(dividend_records)
                combined_d = pd.concat([cached_dividends, new_d],
                                       ignore_index=True)
                combined_d.to_parquet(div_cache)
                cached_dividends = combined_d
                dividend_records = []

            print(f"  Phase2 [{i+1}/{len(stocks_todo)}] {el:.0f}s, "
                  f"ETA {eta/60:.0f} min")

    bs.logout()

    # Final save
    if quarterly_records:
        new_q = pd.DataFrame(quarterly_records)
        cached_quarterly = pd.concat([cached_quarterly, new_q],
                                     ignore_index=True)
        cached_quarterly.to_parquet(cache_file)
    if dividend_records:
        new_d = pd.DataFrame(dividend_records)
        cached_dividends = pd.concat([cached_dividends, new_d],
                                     ignore_index=True)
        cached_dividends.to_parquet(div_cache)

    el = time.time() - t0
    print(f"Phase 2 download done: {len(cached_quarterly)} quarterly records, "
          f"{len(cached_dividends)} dividend records, {el:.0f}s")

    # Inject to Qlib
    _inject_quarterly_to_qlib(cached_quarterly, calendar, cal_idx)
    _inject_dividends_to_qlib(cached_dividends, calendar, cal_idx)


def _inject_quarterly_to_qlib(df: pd.DataFrame, calendar: list[str],
                               cal_idx: dict[str, int]) -> None:
    """Forward-fill quarterly data by pubDate and write to Qlib binary."""
    if df.empty:
        print("No quarterly data to inject.")
        return

    stocks = df["qlib_code"].unique()
    print(f"Injecting {len(ALL_QUARTERLY_FIELDS)} quarterly fields "
          f"for {len(stocks)} stocks...")

    success = 0
    for i, qlib_code in enumerate(stocks):
        feature_dir = FEATURES_DIR / qlib_code.lower()
        if not feature_dir.exists():
            continue

        stock_df = df[df["qlib_code"] == qlib_code].copy()
        stock_df = stock_df[stock_df["pub_date"].astype(str).str.len() >= 10]
        if stock_df.empty:
            continue
        stock_df = stock_df.sort_values("pub_date")

        for bs_field, qlib_field in ALL_QUARTERLY_FIELDS.items():
            if bs_field not in stock_df.columns:
                continue

            pub_val_pairs: list[tuple[str, float]] = []
            for _, row in stock_df.iterrows():
                pub_date = str(row.get("pub_date", ""))
                val_str = str(row.get(bs_field, ""))
                if pub_date and val_str and val_str not in ("", "nan"):
                    try:
                        pub_val_pairs.append((pub_date, float(val_str)))
                    except (ValueError, TypeError):
                        pass

            if not pub_val_pairs:
                continue

            # Forward-fill: from pubDate to next pubDate
            date_values: dict[str, float] = {}
            current_val: float | None = None
            pair_idx = 0
            for cal_date in calendar:
                while (pair_idx < len(pub_val_pairs)
                       and pub_val_pairs[pair_idx][0] <= cal_date):
                    current_val = pub_val_pairs[pair_idx][1]
                    pair_idx += 1
                if current_val is not None:
                    date_values[cal_date] = current_val

            _write_qlib_bin(feature_dir, qlib_field, date_values,
                            calendar, cal_idx)

        success += 1
        if (i + 1) % 500 == 0:
            print(f"  Injected [{i+1}/{len(stocks)}]")

    print(f"Injected quarterly data for {success} stocks")


def _inject_dividends_to_qlib(df: pd.DataFrame, calendar: list[str],
                               cal_idx: dict[str, int]) -> None:
    """Inject dividend per share, forward-filled from registration date."""
    if df.empty:
        print("No dividend data to inject.")
        return

    stocks = df["qlib_code"].unique()
    print(f"Injecting dividends for {len(stocks)} stocks...")

    success = 0
    for qlib_code in stocks:
        feature_dir = FEATURES_DIR / qlib_code.lower()
        if not feature_dir.exists():
            continue

        stock_df = df[df["qlib_code"] == qlib_code].sort_values("regist_date")
        pub_val_pairs = [
            (row["regist_date"], row["cash_ps"])
            for _, row in stock_df.iterrows()
            if row["regist_date"] and row["cash_ps"] > 0
        ]

        if not pub_val_pairs:
            continue

        date_values: dict[str, float] = {}
        current_val: float | None = None
        pair_idx = 0
        for cal_date in calendar:
            while (pair_idx < len(pub_val_pairs)
                   and pub_val_pairs[pair_idx][0] <= cal_date):
                current_val = pub_val_pairs[pair_idx][1]
                pair_idx += 1
            if current_val is not None:
                date_values[cal_date] = current_val

        _write_qlib_bin(feature_dir, "div_ps", date_values,
                        calendar, cal_idx)
        success += 1

    print(f"Injected dividends for {success} stocks (field: div_ps)")


# -----------------------------------------------------------------------
# Status check
# -----------------------------------------------------------------------

def check_status() -> None:
    """Check which financial features are available in Qlib data."""
    print("=== Financial Data Status ===\n")

    base_fields = ["open", "close", "high", "low", "volume", "amount",
                   "vwap", "change", "factor"]
    val_fields = ["turnover_rate", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]
    phase2_fields = list(ALL_QUARTERLY_FIELDS.values()) + ["div_ps"]

    # Sample stocks
    sh_sz_dirs = []
    for d in sorted(FEATURES_DIR.iterdir()):
        if d.name.startswith("sh") or d.name.startswith("sz"):
            sh_sz_dirs.append(d)
            if len(sh_sz_dirs) >= 50:
                break

    print(f"Sampling {len(sh_sz_dirs)} stocks...\n")

    all_fields = base_fields + val_fields + phase2_fields
    stats = {f: 0 for f in all_fields}
    for d in sh_sz_dirs:
        for field in all_fields:
            fbin = d / f"{field}.day.bin"
            if fbin.exists() and fbin.stat().st_size > 100:
                arr = np.fromfile(str(fbin), dtype=np.float32, count=2)
                if not np.isnan(arr[0]):
                    stats[field] += 1

    total = len(sh_sz_dirs)

    print("Phase 1 — Base Price (from baostock K-line):")
    for f in base_fields:
        pct = stats[f] / total * 100 if total else 0
        mark = "OK" if pct > 80 else ("~" if pct > 0 else "X")
        print(f"  {f:<20} {stats[f]:>3}/{total}  {pct:>5.0f}%  {mark}")

    print("\nPhase 1 — Daily Valuation (from baostock K-line):")
    for f in val_fields:
        pct = stats[f] / total * 100 if total else 0
        mark = "OK" if pct > 80 else ("~" if pct > 0 else "X")
        print(f"  {f:<20} {stats[f]:>3}/{total}  {pct:>5.0f}%  {mark}")

    print("\nPhase 2 — Quarterly Financials:")
    for f in phase2_fields:
        pct = stats[f] / total * 100 if total else 0
        mark = "OK" if pct > 80 else ("~" if pct > 0 else "X")
        print(f"  {f:<20} {stats[f]:>3}/{total}  {pct:>5.0f}%  {mark}")

    # Calendar info
    calendar = _load_calendar()
    if calendar:
        print(f"\nCalendar: {len(calendar)} days "
              f"({calendar[0]} ~ {calendar[-1]})")

    # Industry
    ind_file = PROJECT_ROOT / "data" / "industry.parquet"
    if ind_file.exists():
        ind_df = pd.read_parquet(ind_file)
        n_ind = ind_df["industry"].nunique()
        print(f"Industry: {len(ind_df)} stocks, {n_ind} categories  OK")
    else:
        print(f"Industry: not downloaded  X")

    # Cache
    print(f"\nCache files:")
    if CACHE_DIR.exists():
        for f in sorted(CACHE_DIR.iterdir()):
            mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {mb:.1f} MB")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download baostock financial data -> Qlib features",
    )
    parser.add_argument("--phase", type=int, choices=[1, 2],
                        help="1=daily OHLCV+valuation, 2=quarterly+dividends"
                             "+industry. Default: both.")
    parser.add_argument("--market", default="all",
                        help="Stock universe (default: all)")
    parser.add_argument("--start", default="2000-01-01",
                        help="Start date for Phase 1 (default: 2000-01-01)")
    parser.add_argument("--end", default=None,
                        help="End date for Phase 1 (default: today)")
    parser.add_argument("--start-year", type=int, default=2010,
                        help="Start year for Phase 2 (default: 2010)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="End year for Phase 2 (default: current year)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download (clear cache, ignore existing)")
    parser.add_argument("--status", action="store_true",
                        help="Check status of injected features")
    parser.add_argument("--fix-headers", action="store_true",
                        help="Fix .day.bin files with NaN headers")

    args = parser.parse_args()

    if args.status:
        check_status()
        return 0

    if args.fix_headers:
        n = _fix_nan_headers()
        print(f"Fixed {n} files with NaN headers")
        return 0

    stocks = _get_stock_list(args.market)
    print(f"Total SH/SZ stocks: {len(stocks)}")

    if args.phase is None:
        download_phase1(stocks, start=args.start, end=args.end,
                        force=args.force)
        download_phase2(stocks, start_year=args.start_year,
                        end_year=args.end_year, force=args.force)
    elif args.phase == 1:
        download_phase1(stocks, start=args.start, end=args.end,
                        force=args.force)
    elif args.phase == 2:
        download_phase2(stocks, start_year=args.start_year,
                        end_year=args.end_year, force=args.force)

    n_fixed = _fix_nan_headers()
    if n_fixed:
        print(f"Fixed {n_fixed} files with NaN headers")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

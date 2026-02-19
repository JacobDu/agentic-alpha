"""Reusable Qlib data availability and coverage checker.

Checks:
1. Base price/volume fields availability.
2. Optional financial fields availability.
3. Date range coverage for one sample instrument.
4. Market universe coverage on an as-of date.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Cannot locate project root (pyproject.toml not found)")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_qlib.runtime import init_qlib
from qlib.data import D

DEFAULT_BASE_FIELDS = [
    "close",
    "open",
    "high",
    "low",
    "volume",
    "vwap",
    "amount",
    "change",
    "factor",
    "adjclose",
]

DEFAULT_FINANCIAL_FIELDS = [
    "pe",
    "pb",
    "ps",
    "pcf",
    "market_cap",
    "total_mv",
    "circ_mv",
    "turnover_rate",
    "pe_ttm",
    "eps",
    "roe",
    "roa",
    "bps",
]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _check_fields(instrument: str, fields: list[str], start: str, end: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in fields:
        item: dict[str, Any] = {"field": field, "ok": False, "rows": 0}
        try:
            df = D.features([instrument], [f"${field}"], start_time=start, end_time=end)
            item["rows"] = int(len(df))
            if len(df) > 0:
                non_null = int(df.iloc[:, 0].notna().sum())
                item["non_null"] = non_null
                item["ok"] = non_null > 0
                if non_null > 0:
                    item["last_value"] = float(df.iloc[:, 0].dropna().iloc[-1])
            else:
                item["non_null"] = 0
        except Exception as exc:
            item["error"] = str(exc)
        rows.append(item)
    return rows


def _check_date_range(instrument: str, start: str, end: str) -> dict[str, Any]:
    try:
        df = D.features([instrument], ["$close"], start_time=start, end_time=end)
        if len(df) == 0:
            return {"ok": False, "rows": 0}
        dates = df.index.get_level_values("datetime")
        return {
            "ok": True,
            "rows": int(len(df)),
            "start": str(dates.min()),
            "end": str(dates.max()),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_market_coverage(markets: list[str], asof_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market in markets:
        item: dict[str, Any] = {"market": market, "ok": False, "asof_date": asof_date}
        try:
            instruments = D.instruments(market)
            df = D.features(instruments, ["$close"], start_time=asof_date, end_time=asof_date)
            item["rows"] = int(len(df))
            item["ok"] = len(df) > 0
        except Exception as exc:
            item["error"] = str(exc).split("\n")[0]
        rows.append(item)
    return rows


def _render_console(report: dict[str, Any]) -> None:
    print("=== Data Field Check ===")
    print(f"Sample instrument: {report['sample_instrument']}")
    print(f"Field window: {report['field_window']['start']} ~ {report['field_window']['end']}")

    print("\n[Base Fields]")
    for row in report["base_fields"]:
        if row.get("ok"):
            print(
                f"  - ${row['field']}: ok rows={row.get('rows', 0)} "
                f"non_null={row.get('non_null', 0)}"
            )
        else:
            err = row.get("error") or "empty/all NaN"
            print(f"  - ${row['field']}: fail ({err})")

    if report.get("financial_fields") is not None:
        print("\n[Financial Fields]")
        for row in report["financial_fields"]:
            if row.get("ok"):
                print(
                    f"  - ${row['field']}: ok rows={row.get('rows', 0)} "
                    f"non_null={row.get('non_null', 0)}"
                )
            else:
                err = row.get("error") or "empty/all NaN"
                print(f"  - ${row['field']}: fail ({err})")

    print("\n[Date Range]")
    dr = report["date_range"]
    if dr.get("ok"):
        print(f"  - rows={dr.get('rows')} start={dr.get('start')} end={dr.get('end')}")
    else:
        print(f"  - fail ({dr.get('error', 'empty')})")

    print("\n[Market Coverage]")
    for row in report["market_coverage"]:
        if row.get("ok"):
            print(f"  - {row['market']}: ok stocks={row.get('rows', 0)} on {row['asof_date']}")
        else:
            print(f"  - {row['market']}: fail ({row.get('error', 'empty')})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Qlib data availability and coverage")
    parser.add_argument("--sample-instrument", default="sh600000")
    parser.add_argument("--field-start", default="2020-01-01")
    parser.add_argument("--field-end", default="2025-12-31")
    parser.add_argument("--range-start", default="2000-01-01")
    parser.add_argument("--range-end", default="2030-12-31")
    parser.add_argument("--asof-date", help="Date for market coverage check; default uses --field-end")
    parser.add_argument("--base-fields", default=",".join(DEFAULT_BASE_FIELDS))
    parser.add_argument("--financial-fields", default=",".join(DEFAULT_FINANCIAL_FIELDS))
    parser.add_argument("--skip-financial", action="store_true")
    parser.add_argument("--markets", default="csi300,csi500,csi1000,csiall")
    parser.add_argument("--output-json", help="Optional path to save JSON report")
    args = parser.parse_args()

    multiprocessing.set_start_method("fork", force=True)
    init_qlib()

    base_fields = _split_csv(args.base_fields)
    financial_fields = _split_csv(args.financial_fields)
    markets = _split_csv(args.markets)
    asof_date = args.asof_date or args.field_end

    report: dict[str, Any] = {
        "sample_instrument": args.sample_instrument,
        "field_window": {"start": args.field_start, "end": args.field_end},
        "date_range_window": {"start": args.range_start, "end": args.range_end},
        "base_fields": _check_fields(args.sample_instrument, base_fields, args.field_start, args.field_end),
        "date_range": _check_date_range(args.sample_instrument, args.range_start, args.range_end),
        "market_coverage": _check_market_coverage(markets, asof_date),
    }

    if args.skip_financial:
        report["financial_fields"] = None
    else:
        report["financial_fields"] = _check_fields(
            args.sample_instrument,
            financial_fields,
            args.field_start,
            args.field_end,
        )

    _render_console(report)

    if args.output_json:
        output = Path(args.output_json)
    else:
        output = PROJECT_ROOT / "outputs" / "data_check_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Report written: {output}")

    base_ok = all(row.get("ok") for row in report["base_fields"])
    date_ok = bool(report["date_range"].get("ok"))
    market_ok = all(row.get("ok") for row in report["market_coverage"])
    fin_ok = True
    if report["financial_fields"] is not None:
        fin_ok = all(row.get("ok") for row in report["financial_fields"])

    return 0 if (base_ok and date_ok and market_ok and fin_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

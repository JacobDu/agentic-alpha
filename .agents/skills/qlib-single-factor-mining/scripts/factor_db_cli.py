"""Factor library CLI tool for querying and managing the SQLite factor database.

Usage:
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py list [--status STATUS] [--source SOURCE] [--market MARKET] [--top N]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py show FACTOR_NAME [--market MARKET]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py summary [--market MARKET]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py export [--output PATH] [--market MARKET]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py history FACTOR_NAME [--market MARKET]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py markets
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py backtest [FACTOR_NAME] [--market MARKET] [--hold N] [--top N]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py decay [FACTOR_NAME] [--market MARKET] [--horizon N] [--top N]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py runs list [--type sfa|mfa] [--market MARKET] [--decision Promote|Iterate|Drop] [--top N]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py runs show ROUND_ID
"""
from __future__ import annotations

import argparse
import json
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

from project_qlib.factor_db import FactorDB
from project_qlib.workflow_db import WorkflowDB


def cmd_list(db: FactorDB, args: argparse.Namespace) -> None:
    df = db.list_factors(
        status=args.status,
        source=args.source,
        category=args.category,
        market=args.market,
        significant_only=args.significant,
        limit=args.top,
    )
    if df.empty:
        print("No factors found matching criteria.")
        return

    # Display compact table — columns depend on whether market join happened
    if args.market:
        cols = ["name", "source", "category", "status",
                "rank_ic_mean", "rank_ic_t", "rank_icir", "market"]
    else:
        cols = ["name", "source", "category", "status"]
    cols = [c for c in cols if c in df.columns]
    display = df[cols].copy()

    for col in ["rank_ic_mean", "rank_ic_t", "rank_icir"]:
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f"{x:+.4f}" if x == x else "N/A"
            )

    print(display.to_string(index=False))
    print(f"\n({len(df)} factors)")


def cmd_show(db: FactorDB, args: argparse.Namespace) -> None:
    factor = db.get_factor(args.name)
    if not factor:
        print(f"Factor '{args.name}' not found.")
        return

    print(f"{'='*60}")
    print(f"Factor: {factor['name']}")
    print(f"{'='*60}")
    for key, val in factor.items():
        if val is not None and val != "":
            print(f"  {key:20s}: {val}")

    # Show test results per market
    results = db.get_test_results(args.name, market=args.market)
    if not results.empty:
        print(f"\n--- IC Test Results ---")
        for _, r in results.iterrows():
            icir = f"{r['rank_icir']:+.3f}" if r["rank_icir"] == r["rank_icir"] else "N/A"
            t = f"{r['rank_ic_t']:+.1f}" if r["rank_ic_t"] == r["rank_ic_t"] else "N/A"
            print(f"  [{r['market']}] {r['test_start']}~{r['test_end']}  ICIR={icir}  t={t}  sig={r['significant']}")

    # Show IC decay profile
    decay = db.get_ic_decay(args.name, market=args.market)
    if not decay.empty:
        print(f"\n--- IC Decay Profile ---")
        for _, r in decay.iterrows():
            icir = f"{r['rank_icir']:+.3f}" if r["rank_icir"] == r["rank_icir"] else "N/A"
            ric = f"{r['rank_ic_mean']:+.4f}" if r["rank_ic_mean"] == r["rank_ic_mean"] else "N/A"
            print(f"  [{r['market']}] horizon={r['horizon_days']:2d}d  RankIC={ric}  ICIR={icir}")

    # Show backtest results
    bt = db.get_backtest_results(args.name, market=args.market)
    if not bt.empty:
        print(f"\n--- Backtest Results ---")
        for _, r in bt.iterrows():
            ir_s = f"{r['ir']:+.2f}" if r["ir"] == r["ir"] else "N/A"
            ret_s = f"{r['annual_return']*100:+.1f}%" if r["annual_return"] == r["annual_return"] else "N/A"
            exr_s = f"{r['excess_return']*100:+.1f}%" if r["excess_return"] == r["excess_return"] else "N/A"
            mdd_s = f"{r['max_drawdown']*100:.1f}%" if r["max_drawdown"] == r["max_drawdown"] else "N/A"
            topk_s = f"topk={r['topk']}" if r["topk"] else ""
            print(f"  [{r['market']}] hold={r['holding_period']:2d}d {topk_s}  IR={ir_s}  Ann={ret_s}  Excess={exr_s}  MDD={mdd_s}")


def cmd_summary(db: FactorDB, args: argparse.Namespace) -> None:
    print(db.summary(market=getattr(args, "market", None)))


def cmd_export(db: FactorDB, args: argparse.Namespace) -> None:
    out = db.export_csv(args.output, market=args.market)
    print(f"Exported to {out}")


def cmd_history(db: FactorDB, args: argparse.Namespace) -> None:
    df = db.get_test_results(args.name, market=args.market)
    if df.empty:
        print(f"No test results for '{args.name}'.")
        return
    print(df.to_string(index=False))


def cmd_markets(db: FactorDB, _args: argparse.Namespace) -> None:
    ic_counts = db.count_results_by_market()
    bt_counts = db.count_backtest_by_market()
    decay_counts = db.count_ic_decay_by_market()

    if not ic_counts and not bt_counts and not decay_counts:
        print("No test results yet.")
        return

    if ic_counts:
        print("IC test results:")
        for mkt, cnt in sorted(ic_counts.items()):
            print(f"  {mkt}: {cnt} factors")

    if bt_counts:
        print("\nBacktest results (market × holding_period):")
        for mkt in sorted(bt_counts):
            hp_str = ", ".join(
                f"h={hp}d:{cnt}" for hp, cnt in sorted(bt_counts[mkt].items(), key=lambda x: int(x[0]))
            )
            print(f"  {mkt}: {hp_str}")

    if decay_counts:
        print("\nIC decay profiles (market × horizon):")
        for mkt in sorted(decay_counts):
            h_str = ", ".join(
                f"{h}d:{cnt}" for h, cnt in sorted(decay_counts[mkt].items(), key=lambda x: int(x[0]))
            )
            print(f"  {mkt}: {h_str}")


def cmd_backtest(db: FactorDB, args: argparse.Namespace) -> None:
    if args.name:
        # Show backtest for a specific factor
        df = db.get_backtest_results(args.name, market=args.market, holding_period=args.hold)
        if df.empty:
            print(f"No backtest results for '{args.name}'.")
            return
        cols = ["market", "holding_period", "topk", "annual_return", "excess_return",
                "ir", "max_drawdown", "turnover", "win_rate"]
        cols = [c for c in cols if c in df.columns]
        display = df[cols].copy()
        for col in ["annual_return", "excess_return", "max_drawdown", "turnover", "win_rate"]:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda x: f"{x*100:.1f}%" if x == x else "N/A"
                )
        for col in ["ir"]:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda x: f"{x:+.2f}" if x == x else "N/A"
                )
        print(f"Backtest results for {args.name}:")
        print(display.to_string(index=False))
    else:
        # List all backtest results
        df = db.list_backtest_results(
            market=args.market, holding_period=args.hold, limit=args.top
        )
        if df.empty:
            print("No backtest results found.")
            return
        cols = ["factor_name", "market", "holding_period", "topk",
                "annual_return", "excess_return", "ir", "max_drawdown"]
        cols = [c for c in cols if c in df.columns]
        display = df[cols].copy()
        for col in ["annual_return", "excess_return", "max_drawdown"]:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda x: f"{x*100:.1f}%" if x == x else "N/A"
                )
        for col in ["ir"]:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda x: f"{x:+.2f}" if x == x else "N/A"
                )
        print(display.to_string(index=False))
        print(f"\n({len(df)} results)")


def cmd_decay(db: FactorDB, args: argparse.Namespace) -> None:
    if args.name:
        # Show IC decay for a specific factor
        df = db.get_ic_decay(args.name, market=args.market)
        if df.empty:
            print(f"No IC decay data for '{args.name}'.")
            return
        cols = ["market", "horizon_days", "rank_ic_mean", "rank_icir", "rank_ic_t", "n_days"]
        cols = [c for c in cols if c in df.columns]
        display = df[cols].copy()
        for col in ["rank_ic_mean", "rank_icir"]:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda x: f"{x:+.4f}" if x == x else "N/A"
                )
        print(f"IC decay profile for {args.name}:")
        print(display.to_string(index=False))
    else:
        # List all factors at a specific horizon
        horizon = args.horizon or 5
        market = args.market or "csi1000"
        df = db.list_ic_decay(
            market=market, horizon_days=horizon, limit=args.top
        )
        if df.empty:
            print(f"No IC decay data for {market} at horizon={horizon}d.")
            return
        cols = ["factor_name", "market", "horizon_days", "rank_ic_mean", "rank_icir", "status"]
        cols = [c for c in cols if c in df.columns]
        display = df[cols].copy()
        for col in ["rank_ic_mean", "rank_icir"]:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda x: f"{x:+.4f}" if x == x else "N/A"
                )
        print(display.to_string(index=False))
        print(f"\n({len(df)} factors)")


def cmd_runs_list(args: argparse.Namespace) -> None:
    wdb = WorkflowDB(db_path=args.db)
    try:
        rows = wdb.list_runs(
            round_type=args.type,
            market=args.market,
            decision=args.decision,
            limit=args.top,
        )
        if not rows:
            print("No workflow runs found.")
            return
        display_cols = [
            "round_id",
            "round_type",
            "date",
            "market",
            "rank_icir",
            "ir_with_cost",
            "workflow_result",
            "decision",
            "doc_path",
        ]
        for row in rows:
            compact = {k: row.get(k) for k in display_cols}
            print(json.dumps(compact, ensure_ascii=False))
        print(f"\n({len(rows)} runs)")
    finally:
        wdb.close()


def cmd_runs_show(args: argparse.Namespace) -> None:
    wdb = WorkflowDB(db_path=args.db)
    try:
        row = wdb.get_run(args.round_id)
        if row is None:
            print(f"Workflow round '{args.round_id}' not found.")
            return
        print(json.dumps(row, ensure_ascii=False, indent=2))
    finally:
        wdb.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Factor library CLI")
    parser.add_argument("--db", help="SQLite db path", default=str(PROJECT_ROOT / "data" / "factor_library.db"))
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List factors")
    p_list.add_argument("--status", help="Filter by status")
    p_list.add_argument("--source", help="Filter by source (Alpha158/Custom)")
    p_list.add_argument("--category", help="Filter by category")
    p_list.add_argument("--market", help="Join with test results for this market")
    p_list.add_argument("--significant", action="store_true", help="Only significant factors")
    p_list.add_argument("--top", type=int, help="Limit to top N results")

    # show
    p_show = sub.add_parser("show", help="Show factor details + test results")
    p_show.add_argument("name", help="Factor name")
    p_show.add_argument("--market", help="Filter test results by market")

    # summary
    p_summary = sub.add_parser("summary", help="Show summary")
    p_summary.add_argument("--market", help="Show top factors for this market")

    # export
    p_export = sub.add_parser("export", help="Export to CSV")
    p_export.add_argument("--output", help="Output file path")
    p_export.add_argument("--market", help="Join with test results for this market")

    # history
    p_hist = sub.add_parser("history", help="Show test results history")
    p_hist.add_argument("name", help="Factor name")
    p_hist.add_argument("--market", help="Filter by market")

    # markets
    sub.add_parser("markets", help="List markets with test results")

    # backtest
    p_bt = sub.add_parser("backtest", help="Show backtest results by holding period")
    p_bt.add_argument("name", nargs="?", help="Factor name (omit to list all)")
    p_bt.add_argument("--market", help="Filter by market")
    p_bt.add_argument("--hold", type=int, help="Filter by holding period (days)")
    p_bt.add_argument("--top", type=int, help="Limit to top N results")

    # decay
    p_decay = sub.add_parser("decay", help="Show IC decay profiles")
    p_decay.add_argument("name", nargs="?", help="Factor name (omit to list all at given horizon)")
    p_decay.add_argument("--market", help="Market (default: csi1000)")
    p_decay.add_argument("--horizon", type=int, help="Forward return horizon in days (default: 5)")
    p_decay.add_argument("--top", type=int, help="Limit to top N results")

    # runs
    p_runs = sub.add_parser("runs", help="Workflow runs query interface")
    runs_sub = p_runs.add_subparsers(dest="runs_command", required=True)

    p_runs_list = runs_sub.add_parser("list", help="List workflow rounds")
    p_runs_list.add_argument("--type", choices=["sfa", "mfa"], help="Filter by workflow type")
    p_runs_list.add_argument("--market", help="Filter by market")
    p_runs_list.add_argument("--decision", choices=["Promote", "Iterate", "Drop"], help="Filter by decision")
    p_runs_list.add_argument("--top", type=int, help="Limit to top N runs")

    p_runs_show = runs_sub.add_parser("show", help="Show a workflow round")
    p_runs_show.add_argument("round_id", help="Round id, e.g. SFA-2026-02-18-01")

    args = parser.parse_args()
    if args.command == "runs":
        if args.runs_command == "list":
            cmd_runs_list(args)
        elif args.runs_command == "show":
            cmd_runs_show(args)
        return 0

    db = FactorDB(db_path=args.db)

    cmds = {
        "list": cmd_list,
        "show": cmd_show,
        "summary": cmd_summary,
        "export": cmd_export,
        "history": cmd_history,
        "markets": cmd_markets,
        "backtest": cmd_backtest,
        "decay": cmd_decay,
    }
    cmds[args.command](db, args)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

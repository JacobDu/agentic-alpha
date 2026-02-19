"""Factor library CLI tool for querying and managing the SQLite factor database.

Usage:
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py list [--status STATUS] [--source SOURCE] [--market MARKET] [--top N]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py show FACTOR_NAME [--market MARKET]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py runs list [--type sfa|mfa] [--market MARKET] [--decision Promote|Iterate|Drop] [--top N]
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py similarity calc FACTOR_A FACTOR_B --market csi1000 --window 252d --rho-mean-abs 0.42
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py similarity show --market csi1000 --top 20
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py replace propose OLD NEW --market csi1000 --corr-value 0.86 --old-icir 0.25 --new-icir 0.33
    uv run python .agents/skills/qlib-single-factor-mining/scripts/factor_db_cli.py replace confirm OLD NEW --market csi1000 --corr-value 0.86 --old-icir 0.25 --new-icir 0.33 --round-id SFA-YYYY-MM-DD-XX
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

REPLACE_CORR_THRESHOLD = 0.80
REPLACE_IMPROVE_THRESHOLD = 0.20


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

    if args.market:
        cols = ["name", "source", "category", "status", "rank_ic_mean", "rank_ic_t", "rank_icir", "market"]
    else:
        cols = ["name", "source", "category", "status"]
    cols = [c for c in cols if c in df.columns]
    display = df[cols].copy()

    for col in ["rank_ic_mean", "rank_ic_t", "rank_icir"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:+.4f}" if x == x else "N/A")

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

    results = db.get_test_results(args.name, market=args.market)
    if not results.empty:
        print("\n--- IC Test Results ---")
        for _, r in results.iterrows():
            icir = f"{r['rank_icir']:+.3f}" if r["rank_icir"] == r["rank_icir"] else "N/A"
            t_val = f"{r['rank_ic_t']:+.1f}" if r["rank_ic_t"] == r["rank_ic_t"] else "N/A"
            print(f"  [{r['market']}] {r['test_start']}~{r['test_end']}  ICIR={icir}  t={t_val}  sig={r['significant']}")

    decay = db.get_ic_decay(args.name, market=args.market)
    if not decay.empty:
        print("\n--- IC Decay Profile ---")
        for _, r in decay.iterrows():
            icir = f"{r['rank_icir']:+.3f}" if r["rank_icir"] == r["rank_icir"] else "N/A"
            ric = f"{r['rank_ic_mean']:+.4f}" if r["rank_ic_mean"] == r["rank_ic_mean"] else "N/A"
            print(f"  [{r['market']}] horizon={r['horizon_days']:2d}d  RankIC={ric}  ICIR={icir}")

    bt = db.get_backtest_results(args.name, market=args.market)
    if not bt.empty:
        print("\n--- Backtest Results ---")
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
            hp_str = ", ".join(f"h={hp}d:{cnt}" for hp, cnt in sorted(bt_counts[mkt].items(), key=lambda x: int(x[0])))
            print(f"  {mkt}: {hp_str}")

    if decay_counts:
        print("\nIC decay profiles (market × horizon):")
        for mkt in sorted(decay_counts):
            h_str = ", ".join(f"{h}d:{cnt}" for h, cnt in sorted(decay_counts[mkt].items(), key=lambda x: int(x[0])))
            print(f"  {mkt}: {h_str}")


def cmd_backtest(db: FactorDB, args: argparse.Namespace) -> None:
    if args.name:
        df = db.get_backtest_results(args.name, market=args.market, holding_period=args.hold)
        if df.empty:
            print(f"No backtest results for '{args.name}'.")
            return
        cols = ["market", "holding_period", "topk", "annual_return", "excess_return", "ir", "max_drawdown", "turnover", "win_rate"]
        cols = [c for c in cols if c in df.columns]
        display = df[cols].copy()
        for col in ["annual_return", "excess_return", "max_drawdown", "turnover", "win_rate"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x*100:.1f}%" if x == x else "N/A")
        if "ir" in display.columns:
            display["ir"] = display["ir"].apply(lambda x: f"{x:+.2f}" if x == x else "N/A")
        print(f"Backtest results for {args.name}:")
        print(display.to_string(index=False))
    else:
        df = db.list_backtest_results(market=args.market, holding_period=args.hold, limit=args.top)
        if df.empty:
            print("No backtest results found.")
            return
        cols = ["factor_name", "market", "holding_period", "topk", "annual_return", "excess_return", "ir", "max_drawdown"]
        cols = [c for c in cols if c in df.columns]
        display = df[cols].copy()
        for col in ["annual_return", "excess_return", "max_drawdown"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x*100:.1f}%" if x == x else "N/A")
        if "ir" in display.columns:
            display["ir"] = display["ir"].apply(lambda x: f"{x:+.2f}" if x == x else "N/A")
        print(display.to_string(index=False))
        print(f"\n({len(df)} results)")


def cmd_decay(db: FactorDB, args: argparse.Namespace) -> None:
    if args.name:
        df = db.get_ic_decay(args.name, market=args.market)
        if df.empty:
            print(f"No IC decay data for '{args.name}'.")
            return
        cols = ["market", "horizon_days", "rank_ic_mean", "rank_icir", "rank_ic_t", "n_days"]
        cols = [c for c in cols if c in df.columns]
        display = df[cols].copy()
        for col in ["rank_ic_mean", "rank_icir"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:+.4f}" if x == x else "N/A")
        print(f"IC decay profile for {args.name}:")
        print(display.to_string(index=False))
    else:
        horizon = args.horizon or 5
        market = args.market or "csi1000"
        df = db.list_ic_decay(market=market, horizon_days=horizon, limit=args.top)
        if df.empty:
            print(f"No IC decay data for {market} at horizon={horizon}d.")
            return
        cols = ["factor_name", "market", "horizon_days", "rank_ic_mean", "rank_icir", "status"]
        cols = [c for c in cols if c in df.columns]
        display = df[cols].copy()
        for col in ["rank_ic_mean", "rank_icir"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:+.4f}" if x == x else "N/A")
        print(display.to_string(index=False))
        print(f"\n({len(df)} factors)")


def cmd_runs_list(args: argparse.Namespace) -> None:
    wdb = WorkflowDB(db_path=args.db)
    try:
        rows = wdb.list_runs(round_type=args.type, market=args.market, decision=args.decision, limit=args.top)
        if not rows:
            print("No workflow runs found.")
            return
        display_cols = [
            "round_id", "round_type", "date", "market", "rank_icir", "ir_with_cost", "workflow_result", "decision", "doc_path"
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


def cmd_similarity_calc(args: argparse.Namespace) -> int:
    wdb = WorkflowDB(db_path=args.db)
    try:
        wdb.upsert_similarity(
            factor_a=args.factor_a,
            factor_b=args.factor_b,
            market=args.market,
            window=args.window,
            rho_mean_abs=args.rho_mean_abs,
            rho_p95_abs=args.rho_p95_abs,
            sample_days=args.sample_days,
            source_round_id=args.source_round_id,
            notes=args.notes,
        )
    finally:
        wdb.close()

    print(
        json.dumps(
            {
                "factor_a": args.factor_a,
                "factor_b": args.factor_b,
                "market": args.market,
                "window": args.window,
                "rho_mean_abs": args.rho_mean_abs,
                "rho_p95_abs": args.rho_p95_abs,
                "sample_days": args.sample_days,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_similarity_show(args: argparse.Namespace) -> int:
    wdb = WorkflowDB(db_path=args.db)
    try:
        rows = wdb.list_similarity(
            factor=args.factor,
            market=args.market,
            window=args.window,
            min_rho=args.min_rho,
            limit=args.top,
        )
    finally:
        wdb.close()

    if not rows:
        print("No similarity records found.")
        return 0

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"\n({len(rows)} rows)")
    return 0


def _replacement_decision(corr_value: float, old_icir: float, new_icir: float) -> tuple[bool, float]:
    old_abs = abs(old_icir)
    new_abs = abs(new_icir)
    if old_abs > 0:
        improve_ratio = (new_abs - old_abs) / old_abs
    else:
        improve_ratio = float("inf") if new_abs > 0 else 0.0
    passed = corr_value > REPLACE_CORR_THRESHOLD and improve_ratio >= REPLACE_IMPROVE_THRESHOLD
    return passed, improve_ratio


def cmd_replace_propose(args: argparse.Namespace) -> int:
    passed, improve_ratio = _replacement_decision(args.corr_value, args.old_icir, args.new_icir)
    payload = {
        "old_factor": args.old_factor,
        "new_factor": args.new_factor,
        "market": args.market,
        "corr_value": args.corr_value,
        "old_icir": args.old_icir,
        "new_icir": args.new_icir,
        "improve_ratio": improve_ratio,
        "gate": {
            "corr_threshold": REPLACE_CORR_THRESHOLD,
            "improve_threshold": REPLACE_IMPROVE_THRESHOLD,
            "passed": passed,
        },
        "decision": "confirmable" if passed else "blocked",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_replace_confirm(args: argparse.Namespace) -> int:
    db = FactorDB(db_path=args.db)
    try:
        if db.get_factor(args.old_factor) is None:
            print(f"Old factor not found: {args.old_factor}")
            return 1
        if db.get_factor(args.new_factor) is None:
            print(f"New factor not found: {args.new_factor}")
            return 1

        wdb = WorkflowDB(db_path=args.db)
        try:
            result = wdb.record_replacement(
                old_factor=args.old_factor,
                new_factor=args.new_factor,
                market=args.market,
                corr_value=args.corr_value,
                old_icir=args.old_icir,
                new_icir=args.new_icir,
                decided_in_round=args.round_id,
                reason=args.reason,
                enforce_gate=True,
            )
        except ValueError as exc:
            print(str(exc))
            return 1
        finally:
            wdb.close()

        db.upsert_factor(
            name=args.old_factor,
            notes=f"Replaced by {args.new_factor} in {args.round_id}",
        )
        db.upsert_factor(name=args.new_factor, status="Accepted")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        db.close()


def cmd_replace_history(args: argparse.Namespace) -> int:
    wdb = WorkflowDB(db_path=args.db)
    try:
        rows = wdb.list_replacements(market=args.market, factor=args.factor, limit=args.top)
    finally:
        wdb.close()

    if not rows:
        print("No replacement records found.")
        return 0

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"\n({len(rows)} rows)")
    return 0


def _register_factor_list_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--status", help="Filter by status")
    parser.add_argument("--source", help="Filter by source (Alpha158/Custom)")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--market", help="Join with test results for this market")
    parser.add_argument("--significant", action="store_true", help="Only significant factors")
    parser.add_argument("--top", type=int, help="Limit to top N results")


def main() -> int:
    parser = argparse.ArgumentParser(description="Factor library CLI")
    parser.add_argument("--db", help="SQLite db path", default=str(PROJECT_ROOT / "data" / "factor_library.db"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List factors")
    _register_factor_list_parser(p_list)

    p_show = sub.add_parser("show", help="Show factor details + test results")
    p_show.add_argument("name", help="Factor name")
    p_show.add_argument("--market", help="Filter test results by market")

    p_summary = sub.add_parser("summary", help="Show summary")
    p_summary.add_argument("--market", help="Show top factors for this market")

    p_export = sub.add_parser("export", help="Export to CSV")
    p_export.add_argument("--output", help="Output file path")
    p_export.add_argument("--market", help="Join with test results for this market")

    p_hist = sub.add_parser("history", help="Show test results history")
    p_hist.add_argument("name", help="Factor name")
    p_hist.add_argument("--market", help="Filter by market")

    sub.add_parser("markets", help="List markets with test results")

    p_bt = sub.add_parser("backtest", help="Show backtest results by holding period")
    p_bt.add_argument("name", nargs="?", help="Factor name (omit to list all)")
    p_bt.add_argument("--market", help="Filter by market")
    p_bt.add_argument("--hold", type=int, help="Filter by holding period (days)")
    p_bt.add_argument("--top", type=int, help="Limit to top N results")

    p_decay = sub.add_parser("decay", help="Show IC decay profiles")
    p_decay.add_argument("name", nargs="?", help="Factor name (omit to list all at given horizon)")
    p_decay.add_argument("--market", help="Market (default: csi1000)")
    p_decay.add_argument("--horizon", type=int, help="Forward return horizon in days (default: 5)")
    p_decay.add_argument("--top", type=int, help="Limit to top N results")

    p_factors = sub.add_parser("factors", help="Factor query interface")
    factors_sub = p_factors.add_subparsers(dest="factors_command", required=True)
    p_factors_list = factors_sub.add_parser("list", help="List factors")
    _register_factor_list_parser(p_factors_list)
    p_factors_show = factors_sub.add_parser("show", help="Show factor details")
    p_factors_show.add_argument("name", help="Factor name")
    p_factors_show.add_argument("--market", help="Filter test results by market")

    p_runs = sub.add_parser("runs", help="Workflow runs query interface")
    runs_sub = p_runs.add_subparsers(dest="runs_command", required=True)
    p_runs_list = runs_sub.add_parser("list", help="List workflow rounds")
    p_runs_list.add_argument("--type", choices=["sfa", "mfa"], help="Filter by workflow type")
    p_runs_list.add_argument("--market", help="Filter by market")
    p_runs_list.add_argument("--decision", choices=["Promote", "Iterate", "Drop"], help="Filter by decision")
    p_runs_list.add_argument("--top", type=int, help="Limit to top N runs")
    p_runs_show = runs_sub.add_parser("show", help="Show a workflow round")
    p_runs_show.add_argument("round_id", help="Round id, e.g. SFA-YYYY-MM-DD-XX")

    p_similarity = sub.add_parser("similarity", help="Factor similarity operations")
    sim_sub = p_similarity.add_subparsers(dest="similarity_command", required=True)
    p_sim_calc = sim_sub.add_parser("calc", help="Upsert one similarity snapshot")
    p_sim_calc.add_argument("factor_a")
    p_sim_calc.add_argument("factor_b")
    p_sim_calc.add_argument("--market", default="csi1000")
    p_sim_calc.add_argument("--window", default="252d")
    p_sim_calc.add_argument("--rho-mean-abs", type=float, required=True)
    p_sim_calc.add_argument("--rho-p95-abs", type=float)
    p_sim_calc.add_argument("--sample-days", type=int)
    p_sim_calc.add_argument("--source-round-id")
    p_sim_calc.add_argument("--notes")

    p_sim_show = sim_sub.add_parser("show", help="Show similarity snapshots")
    p_sim_show.add_argument("--factor")
    p_sim_show.add_argument("--market")
    p_sim_show.add_argument("--window")
    p_sim_show.add_argument("--min-rho", type=float)
    p_sim_show.add_argument("--top", type=int)

    p_replace = sub.add_parser("replace", help="Factor replacement operations")
    rep_sub = p_replace.add_subparsers(dest="replace_command", required=True)

    p_rep_propose = rep_sub.add_parser("propose", help="Evaluate replacement gate")
    p_rep_propose.add_argument("old_factor")
    p_rep_propose.add_argument("new_factor")
    p_rep_propose.add_argument("--market", default="csi1000")
    p_rep_propose.add_argument("--corr-value", type=float, required=True)
    p_rep_propose.add_argument("--old-icir", type=float, required=True)
    p_rep_propose.add_argument("--new-icir", type=float, required=True)

    p_rep_confirm = rep_sub.add_parser("confirm", help="Record replacement")
    p_rep_confirm.add_argument("old_factor")
    p_rep_confirm.add_argument("new_factor")
    p_rep_confirm.add_argument("--market", default="csi1000")
    p_rep_confirm.add_argument("--corr-value", type=float, required=True)
    p_rep_confirm.add_argument("--old-icir", type=float, required=True)
    p_rep_confirm.add_argument("--new-icir", type=float, required=True)
    p_rep_confirm.add_argument("--round-id", required=True)
    p_rep_confirm.add_argument("--reason")

    p_rep_history = rep_sub.add_parser("history", help="Show replacement history")
    p_rep_history.add_argument("--market")
    p_rep_history.add_argument("--factor")
    p_rep_history.add_argument("--top", type=int)

    args = parser.parse_args()

    if args.command == "runs":
        if args.runs_command == "list":
            cmd_runs_list(args)
        elif args.runs_command == "show":
            cmd_runs_show(args)
        return 0

    if args.command == "similarity":
        if args.similarity_command == "calc":
            return cmd_similarity_calc(args)
        if args.similarity_command == "show":
            return cmd_similarity_show(args)
        return 0

    if args.command == "replace":
        if args.replace_command == "propose":
            return cmd_replace_propose(args)
        if args.replace_command == "confirm":
            return cmd_replace_confirm(args)
        if args.replace_command == "history":
            return cmd_replace_history(args)
        return 0

    db = FactorDB(db_path=args.db)
    try:
        if args.command == "factors":
            if args.factors_command == "list":
                cmd_list(db, args)
            elif args.factors_command == "show":
                cmd_show(db, args)
            return 0

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
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""MFA workflow record CLI."""
from __future__ import annotations

import argparse
import json
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

from project_qlib.workflow_db import WorkflowDB

DEFAULT_DOCS_DIR = PROJECT_ROOT / "docs" / "workflows" / "multi-factor"
DEFAULT_INDEX_PATH = DEFAULT_DOCS_DIR / "INDEX.md"


def _val(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _bool_from_text(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.lower()
    if v in {"yes", "true", "1"}:
        return True
    if v in {"no", "false", "0"}:
        return False
    return None


def _upsert_from_args(db: WorkflowDB, args: argparse.Namespace) -> None:
    db.upsert_run(
        round_id=args.round_id,
        round_type="mfa",
        date=_val(args.date),
        owner=_val(args.owner),
        market=_val(args.market),
        hypothesis=_val(args.hypothesis),
        market_logic=_val(args.market_logic),
        expected_direction=_val(args.expected_direction),
        parse_ok=_val(args.parse_ok),
        complexity_level=_val(args.complexity_level),
        redundancy_flag=_val(args.redundancy_flag),
        data_availability=_val(args.data_availability),
        gate_notes=_val(args.gate_notes),
        script_path=_val(args.script_path),
        test_start=_val(args.test_start),
        test_end=_val(args.test_end),
        run_id=_val(args.run_id),
        doc_path=_val(args.doc_path),
        legacy_round_id=_val(args.legacy_round_id),
    )

    if any(
        getattr(args, name) is not None
        for name in [
            "executed",
            "not_run_reason",
            "excess_return_with_cost",
            "ir_with_cost",
            "max_drawdown",
            "mfa_result",
        ]
    ):
        db.upsert_mfa_metrics(
            round_id=args.round_id,
            executed=_bool_from_text(args.executed),
            not_run_reason=_val(args.not_run_reason),
            excess_return_with_cost=args.excess_return_with_cost,
            ir_with_cost=args.ir_with_cost,
            max_drawdown=args.max_drawdown,
            mfa_result=_val(args.mfa_result),
        )

    if any(getattr(args, name) is not None for name in ["decision", "decision_basis", "failure_mode", "next_action"]):
        db.upsert_decision(
            round_id=args.round_id,
            decision=_val(args.decision),
            decision_basis=_val(args.decision_basis),
            failure_mode=_val(args.failure_mode),
            next_action=_val(args.next_action),
        )


def cmd_create(db: WorkflowDB, args: argparse.Namespace) -> None:
    _upsert_from_args(db, args)
    print(f"[OK] Created/updated MFA round: {args.round_id}")


def cmd_update(db: WorkflowDB, args: argparse.Namespace) -> None:
    _upsert_from_args(db, args)
    print(f"[OK] Updated MFA round: {args.round_id}")


def cmd_list(db: WorkflowDB, args: argparse.Namespace) -> None:
    rows = db.list_runs(round_type="mfa", market=args.market, decision=args.decision, limit=args.top)
    if not rows:
        print("No MFA rounds found.")
        return
    for row in rows:
        compact = {
            "round_id": row.get("round_id"),
            "date": row.get("date"),
            "market": row.get("market"),
            "ir_with_cost": row.get("ir_with_cost"),
            "mfa_result": row.get("mfa_result"),
            "decision": row.get("decision"),
            "doc_path": row.get("doc_path"),
        }
        print(json.dumps(compact, ensure_ascii=False))
    print(f"\n({len(rows)} rounds)")


def cmd_show(db: WorkflowDB, args: argparse.Namespace) -> None:
    row = db.get_run(args.round_id)
    if row is None:
        print(f"Round not found: {args.round_id}")
        return
    print(json.dumps(row, ensure_ascii=False, indent=2))


def cmd_link_evidence(db: WorkflowDB, args: argparse.Namespace) -> None:
    db.add_evidence(
        round_id=args.round_id,
        evidence_type=args.evidence_type,
        evidence_value=args.evidence_value,
        notes=_val(args.notes),
    )
    print(f"[OK] Evidence linked: {args.round_id} {args.evidence_type}={args.evidence_value}")


def cmd_write_doc(db: WorkflowDB, args: argparse.Namespace) -> None:
    output = Path(args.output) if args.output else DEFAULT_DOCS_DIR / f"{args.round_id}.md"
    path = db.sync_round_to_markdown(args.round_id, output)
    print(f"[OK] Wrote markdown: {path}")


def _short_text(value: Any, max_len: int = 48) -> str:
    if value is None:
        return "N/A"
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def cmd_sync_index(db: WorkflowDB, args: argparse.Namespace) -> None:
    rows = db.list_runs(round_type="mfa", limit=args.top)
    index_path = Path(args.output) if args.output else DEFAULT_INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Multi-Factor Workflow Index",
        "",
        "| round_id | date | hypothesis_short | market | ir_with_cost | mfa_result | decision | doc |",
        "|----------|------|------------------|--------|--------------|------------|----------|-----|",
    ]
    for row in rows:
        lines.append(
            "| {round_id} | {date} | {hypothesis_short} | {market} | {ir_with_cost} | {mfa_result} | {decision} | {doc_path} |".format(
                round_id=row.get("round_id") or "N/A",
                date=row.get("date") or "N/A",
                hypothesis_short=_short_text(row.get("hypothesis"), max_len=64),
                market=row.get("market") or "N/A",
                ir_with_cost=row.get("ir_with_cost") if row.get("ir_with_cost") is not None else "N/A",
                mfa_result=row.get("mfa_result") or "N/A",
                decision=row.get("decision") or "N/A",
                doc_path=row.get("doc_path") or "N/A",
            )
        )

    lines.extend(
        [
            "",
            "维护规则：",
            "1. 每新增一轮 MFA 记录，需同步更新本索引。",
            "2. evidence 至少包含 doc/output_path/db_query/run_id 之一。",
            "3. decision 仅允许 Promote / Iterate / Drop。",
        ]
    )

    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Synced index: {index_path}")


def _add_common_round_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--date")
    parser.add_argument("--owner")
    parser.add_argument("--market")
    parser.add_argument("--hypothesis")
    parser.add_argument("--market-logic")
    parser.add_argument("--expected-direction")
    parser.add_argument("--parse-ok")
    parser.add_argument("--complexity-level")
    parser.add_argument("--redundancy-flag")
    parser.add_argument("--data-availability")
    parser.add_argument("--gate-notes")
    parser.add_argument("--script-path")
    parser.add_argument("--test-start")
    parser.add_argument("--test-end")
    parser.add_argument("--run-id")
    parser.add_argument("--doc-path")
    parser.add_argument("--legacy-round-id")

    parser.add_argument("--executed", choices=["yes", "no"])
    parser.add_argument("--not-run-reason")
    parser.add_argument("--excess-return-with-cost", type=float)
    parser.add_argument("--ir-with-cost", type=float)
    parser.add_argument("--max-drawdown", type=float)
    parser.add_argument("--mfa-result")

    parser.add_argument("--decision", choices=["Promote", "Iterate", "Drop"])
    parser.add_argument("--decision-basis")
    parser.add_argument("--failure-mode")
    parser.add_argument("--next-action")


def main() -> int:
    parser = argparse.ArgumentParser(description="MFA workflow record CLI")
    parser.add_argument("--db", help="SQLite db path", default=str(PROJECT_ROOT / "data" / "factor_library.db"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create MFA record")
    _add_common_round_args(p_create)

    p_update = sub.add_parser("update", help="Update MFA record")
    _add_common_round_args(p_update)

    p_list = sub.add_parser("list", help="List MFA records")
    p_list.add_argument("--market")
    p_list.add_argument("--decision", choices=["Promote", "Iterate", "Drop"])
    p_list.add_argument("--top", type=int)

    p_show = sub.add_parser("show", help="Show one MFA record")
    p_show.add_argument("round_id")

    p_link = sub.add_parser("link-evidence", help="Add one evidence entry")
    p_link.add_argument("round_id")
    p_link.add_argument("evidence_type", choices=["output_path", "db_query", "run_id", "script", "doc"])
    p_link.add_argument("evidence_value")
    p_link.add_argument("--notes")

    p_write = sub.add_parser("write-doc", help="Render record as markdown")
    p_write.add_argument("round_id")
    p_write.add_argument("--output")

    p_sync = sub.add_parser("sync-index", help="Sync MFA index markdown")
    p_sync.add_argument("--output")
    p_sync.add_argument("--top", type=int)

    args = parser.parse_args()
    db = WorkflowDB(db_path=args.db)

    try:
        commands = {
            "create": cmd_create,
            "update": cmd_update,
            "list": cmd_list,
            "show": cmd_show,
            "link-evidence": cmd_link_evidence,
            "write-doc": cmd_write_doc,
            "sync-index": cmd_sync_index,
        }
        commands[args.command](db, args)
    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

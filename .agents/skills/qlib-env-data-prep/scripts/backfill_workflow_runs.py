"""Backfill historical HEA markdown docs into workflow tables.

Usage:
    uv run python .agents/skills/qlib-env-data-prep/scripts/backfill_workflow_runs.py
    uv run python .agents/skills/qlib-env-data-prep/scripts/backfill_workflow_runs.py --db /tmp/factor_library.db --docs-dir docs/heas
"""
from __future__ import annotations

import argparse
import re
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

from project_qlib.workflow_db import WorkflowDB


def _extract_scalar(text: str, key: str) -> str | None:
    pattern = rf"^\s*-\s*{re.escape(key)}\s*:\s*(.+?)\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _clean_value(raw: str | None) -> str | None:
    if raw is None:
        return None
    val = raw.strip().replace("**", "")
    if not val or val in {"N/A", "n/a", "null", "None", "无"}:
        return None
    return val


def _normalize_decision(raw: str | None) -> str | None:
    val = _clean_value(raw)
    if val is None:
        return None
    low = val.lower()
    if low.startswith("promote"):
        return "Promote"
    if low.startswith("iterate"):
        return "Iterate"
    if low.startswith("drop"):
        return "Drop"
    return None


def _to_float(raw: str | None) -> float | None:
    val = _clean_value(raw)
    if val is None:
        return None
    val = val.replace("**", "").replace(",", "").strip()
    pct = val.endswith("%")
    if pct:
        val = val[:-1].strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
    if not m:
        return None
    num = float(m.group(0))
    return num / 100.0 if pct else num


def _to_int(raw: str | None) -> int | None:
    val = _clean_value(raw)
    if val is None:
        return None
    m = re.search(r"\d+", val)
    if not m:
        return None
    return int(m.group(0))


def _extract_output_paths(text: str) -> list[str]:
    results: list[str] = []
    for match in re.finditer(r"^\s*-\s*(outputs/[^\s]+.*?)\s*$", text, flags=re.MULTILINE):
        path = match.group(1).strip()
        if path and path not in results:
            results.append(path)
    return results


def _extract_script_paths(text: str) -> list[str]:
    results: list[str] = []
    for match in re.finditer(r"^\s*-\s*(scripts/[^\s]+.*?)\s*$", text, flags=re.MULTILINE):
        path = match.group(1).strip()
        if path and path not in results:
            results.append(path)
    return results


def backfill_file(db: WorkflowDB, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    round_id = path.stem

    doc_rel = str(path.relative_to(PROJECT_ROOT)) if path.is_absolute() else str(path)

    db.upsert_run(
        round_id=round_id,
        round_type="sfa",  # locked by requirement
        date=_clean_value(_extract_scalar(text, "date")),
        owner=_clean_value(_extract_scalar(text, "owner")),
        market=_clean_value(_extract_scalar(text, "market")) or "csi1000",
        hypothesis=_clean_value(_extract_scalar(text, "hypothesis")),
        market_logic=_clean_value(_extract_scalar(text, "market_logic")),
        expected_direction=_clean_value(_extract_scalar(text, "expected_direction")),
        parse_ok=_clean_value(_extract_scalar(text, "parse_ok")),
        complexity_level=_clean_value(_extract_scalar(text, "complexity_level")),
        redundancy_flag=_clean_value(_extract_scalar(text, "redundancy_flag")),
        data_availability=_clean_value(_extract_scalar(text, "data_availability")),
        gate_notes=_clean_value(_extract_scalar(text, "gate_notes")),
        script_path=_clean_value(_extract_scalar(text, "script")),
        test_start=_clean_value(_extract_scalar(text, "start")),
        test_end=_clean_value(_extract_scalar(text, "end")),
        run_id=_clean_value(_extract_scalar(text, "run_id")),
        doc_path=doc_rel,
        legacy_round_id=None,
    )

    # Single-factor metrics (SFA)
    if any(
        _extract_scalar(text, k) is not None
        for k in ["rank_ic_mean", "rank_ic_t", "fdr_p", "rank_icir", "n_days", "layer_a_result"]
    ):
        db.upsert_sfa_metrics(
            round_id=round_id,
            rank_ic_mean=_to_float(_extract_scalar(text, "rank_ic_mean")),
            rank_ic_t=_to_float(_extract_scalar(text, "rank_ic_t")),
            fdr_p=_to_float(_extract_scalar(text, "fdr_p")),
            rank_icir=_to_float(_extract_scalar(text, "rank_icir")),
            n_days=_to_int(_extract_scalar(text, "n_days")),
            sfa_result=_clean_value(_extract_scalar(text, "layer_a_result")),
        )

    # Decision
    if _extract_scalar(text, "decision") is not None:
        db.upsert_decision(
            round_id=round_id,
            decision=_normalize_decision(_extract_scalar(text, "decision")),
            decision_basis=_clean_value(_extract_scalar(text, "decision_basis")),
            failure_mode=_clean_value(_extract_scalar(text, "failure_mode")),
            next_action=_clean_value(_extract_scalar(text, "next_action")),
        )

    # Evidence
    db.clear_evidence(round_id)
    db.add_evidence(round_id=round_id, evidence_type="doc", evidence_value=doc_rel)

    db_query = _clean_value(_extract_scalar(text, "db_query"))
    if db_query:
        db.add_evidence(round_id=round_id, evidence_type="db_query", evidence_value=db_query)

    run_id = _clean_value(_extract_scalar(text, "run_id"))
    if run_id:
        db.add_evidence(round_id=round_id, evidence_type="run_id", evidence_value=run_id)

    script_value = _clean_value(_extract_scalar(text, "script"))
    if script_value:
        db.add_evidence(round_id=round_id, evidence_type="script", evidence_value=script_value)

    for script in _extract_script_paths(text):
        db.add_evidence(round_id=round_id, evidence_type="script", evidence_value=script)

    for output_path in _extract_output_paths(text):
        db.add_evidence(round_id=round_id, evidence_type="output_path", evidence_value=output_path)



def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill workflow runs from docs/heas")
    parser.add_argument("--db", default=str(PROJECT_ROOT / "data" / "factor_library.db"), help="SQLite db path")
    parser.add_argument("--docs-dir", default=str(PROJECT_ROOT / "docs" / "heas"), help="Directory for HEA markdown files")
    args = parser.parse_args()

    db = WorkflowDB(db_path=args.db)
    docs_dir = Path(args.docs_dir)
    files = sorted(docs_dir.glob("HEA-*.md"))

    if not files:
        print(f"No HEA files found in: {docs_dir}")
        db.close()
        return 0

    for file_path in files:
        backfill_file(db, file_path)
        print(f"[OK] Backfilled: {file_path.name}")

    print(f"[DONE] Backfilled {len(files)} files into workflow tables.")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

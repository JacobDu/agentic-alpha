from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from project_qlib.runtime import PROJECT_ROOT

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "factor_library.db"

_WORKFLOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_runs (
    round_id         TEXT PRIMARY KEY,
    round_type       TEXT NOT NULL CHECK(round_type IN ('sfa','mfa')),
    date             TEXT,
    owner            TEXT,
    market           TEXT,
    hypothesis       TEXT,
    market_logic     TEXT,
    expected_direction TEXT,
    parse_ok         TEXT,
    complexity_level TEXT,
    redundancy_flag  TEXT,
    data_availability TEXT,
    gate_notes       TEXT,
    script_path      TEXT,
    test_start       TEXT,
    test_end         TEXT,
    run_id           TEXT,
    doc_path         TEXT,
    legacy_round_id  TEXT,
    created_at       TEXT DEFAULT (datetime('now')),
    updated_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS workflow_sfa_metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id         TEXT NOT NULL,
    rank_ic_mean     REAL,
    rank_ic_t        REAL,
    fdr_p            REAL,
    rank_icir        REAL,
    n_days           INTEGER,
    sfa_result       TEXT,
    FOREIGN KEY (round_id) REFERENCES workflow_runs(round_id) ON DELETE CASCADE,
    UNIQUE(round_id)
);

CREATE TABLE IF NOT EXISTS workflow_mfa_metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id         TEXT NOT NULL,
    executed         INTEGER,
    not_run_reason   TEXT,
    excess_return_with_cost REAL,
    ir_with_cost     REAL,
    max_drawdown     REAL,
    mfa_result       TEXT,
    FOREIGN KEY (round_id) REFERENCES workflow_runs(round_id) ON DELETE CASCADE,
    UNIQUE(round_id)
);

CREATE TABLE IF NOT EXISTS workflow_decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id         TEXT NOT NULL,
    decision         TEXT CHECK(decision IN ('Promote','Iterate','Drop')),
    decision_basis   TEXT,
    failure_mode     TEXT,
    next_action      TEXT,
    decided_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (round_id) REFERENCES workflow_runs(round_id) ON DELETE CASCADE,
    UNIQUE(round_id)
);

CREATE TABLE IF NOT EXISTS workflow_evidence (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id         TEXT NOT NULL,
    evidence_type    TEXT CHECK(evidence_type IN ('output_path','db_query','run_id','script','doc')),
    evidence_value   TEXT NOT NULL,
    notes            TEXT,
    created_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (round_id) REFERENCES workflow_runs(round_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS factor_similarity (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_a         TEXT NOT NULL,
    factor_b         TEXT NOT NULL,
    market           TEXT NOT NULL,
    window           TEXT NOT NULL,
    rho_mean_abs     REAL NOT NULL,
    rho_p95_abs      REAL,
    sample_days      INTEGER,
    source_round_id  TEXT,
    notes            TEXT,
    calculated_at    TEXT DEFAULT (datetime('now')),
    UNIQUE(factor_a, factor_b, market, window)
);

CREATE TABLE IF NOT EXISTS factor_replacements (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    old_factor       TEXT NOT NULL,
    new_factor       TEXT NOT NULL,
    market           TEXT NOT NULL,
    reason           TEXT,
    corr_value       REAL,
    old_icir         REAL,
    new_icir         REAL,
    improve_ratio    REAL,
    decided_in_round TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_workflow_runs_type_date ON workflow_runs(round_type, date DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_market ON workflow_runs(market);
CREATE INDEX IF NOT EXISTS idx_workflow_decisions_decision ON workflow_decisions(decision);
CREATE INDEX IF NOT EXISTS idx_workflow_evidence_round ON workflow_evidence(round_id);
CREATE INDEX IF NOT EXISTS idx_factor_similarity_market ON factor_similarity(market, window);
CREATE INDEX IF NOT EXISTS idx_factor_similarity_factor_a ON factor_similarity(factor_a);
CREATE INDEX IF NOT EXISTS idx_factor_similarity_factor_b ON factor_similarity(factor_b);
CREATE INDEX IF NOT EXISTS idx_factor_replacements_market ON factor_replacements(market, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_factor_replacements_old ON factor_replacements(old_factor);
CREATE INDEX IF NOT EXISTS idx_factor_replacements_new ON factor_replacements(new_factor);
CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_evidence_unique
    ON workflow_evidence(round_id, evidence_type, evidence_value);

DROP VIEW IF EXISTS v_workflow_sfa_rounds;
DROP VIEW IF EXISTS v_workflow_mfa_rounds;
DROP VIEW IF EXISTS v_workflow_round_summary;

CREATE VIEW v_workflow_sfa_rounds AS
SELECT
    r.round_id,
    r.round_type,
    r.date,
    r.owner,
    r.market,
    r.hypothesis,
    r.script_path,
    r.run_id,
    r.doc_path,
    s.rank_ic_mean,
    s.rank_ic_t,
    s.fdr_p,
    s.rank_icir,
    s.n_days,
    s.sfa_result,
    d.decision,
    d.decision_basis,
    d.failure_mode,
    d.next_action,
    d.decided_at,
    r.updated_at
FROM workflow_runs r
LEFT JOIN workflow_sfa_metrics s ON s.round_id = r.round_id
LEFT JOIN workflow_decisions d ON d.round_id = r.round_id
WHERE r.round_type = 'sfa';

CREATE VIEW v_workflow_mfa_rounds AS
SELECT
    r.round_id,
    r.round_type,
    r.date,
    r.owner,
    r.market,
    r.hypothesis,
    r.script_path,
    r.run_id,
    r.doc_path,
    m.executed,
    m.not_run_reason,
    m.excess_return_with_cost,
    m.ir_with_cost,
    m.max_drawdown,
    m.mfa_result,
    d.decision,
    d.decision_basis,
    d.failure_mode,
    d.next_action,
    d.decided_at,
    r.updated_at
FROM workflow_runs r
LEFT JOIN workflow_mfa_metrics m ON m.round_id = r.round_id
LEFT JOIN workflow_decisions d ON d.round_id = r.round_id
WHERE r.round_type = 'mfa';

CREATE VIEW v_workflow_round_summary AS
SELECT
    r.round_id,
    r.round_type,
    r.date,
    r.owner,
    r.market,
    r.hypothesis,
    r.script_path,
    r.run_id,
    r.doc_path,
    s.rank_ic_mean,
    s.rank_ic_t,
    s.fdr_p,
    s.rank_icir,
    s.n_days,
    s.sfa_result,
    m.executed,
    m.not_run_reason,
    m.excess_return_with_cost,
    m.ir_with_cost,
    m.max_drawdown,
    m.mfa_result,
    CASE
        WHEN r.round_type = 'sfa' THEN s.sfa_result
        WHEN r.round_type = 'mfa' THEN m.mfa_result
        ELSE NULL
    END AS workflow_result,
    d.decision,
    d.decision_basis,
    d.failure_mode,
    d.next_action,
    d.decided_at,
    r.updated_at
FROM workflow_runs r
LEFT JOIN workflow_sfa_metrics s ON s.round_id = r.round_id AND r.round_type = 'sfa'
LEFT JOIN workflow_mfa_metrics m ON m.round_id = r.round_id AND r.round_type = 'mfa'
LEFT JOIN workflow_decisions d ON d.round_id = r.round_id;
"""


def ensure_workflow_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_WORKFLOW_SCHEMA)
    conn.commit()


class WorkflowDB:
    def __init__(
        self,
        db_path: Path | str | None = None,
        conn: sqlite3.Connection | None = None,
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._owns_conn = conn is None
        if conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        ensure_workflow_schema(self._conn)

    def close(self) -> None:
        if self._owns_conn:
            self._conn.close()

    def upsert_run(
        self,
        *,
        round_id: str,
        round_type: str,
        date: str | None = None,
        owner: str | None = None,
        market: str | None = None,
        hypothesis: str | None = None,
        market_logic: str | None = None,
        expected_direction: str | None = None,
        parse_ok: str | None = None,
        complexity_level: str | None = None,
        redundancy_flag: str | None = None,
        data_availability: str | None = None,
        gate_notes: str | None = None,
        script_path: str | None = None,
        test_start: str | None = None,
        test_end: str | None = None,
        run_id: str | None = None,
        doc_path: str | None = None,
        legacy_round_id: str | None = None,
    ) -> str:
        rt = round_type.lower()
        if rt not in {"sfa", "mfa"}:
            raise ValueError("round_type must be 'sfa' or 'mfa'")

        self._conn.execute(
            """INSERT INTO workflow_runs
               (round_id, round_type, date, owner, market, hypothesis, market_logic,
                expected_direction, parse_ok, complexity_level, redundancy_flag,
                data_availability, gate_notes, script_path, test_start, test_end,
                run_id, doc_path, legacy_round_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(round_id)
               DO UPDATE SET
                 round_type=excluded.round_type,
                 date=excluded.date,
                 owner=excluded.owner,
                 market=excluded.market,
                 hypothesis=excluded.hypothesis,
                 market_logic=excluded.market_logic,
                 expected_direction=excluded.expected_direction,
                 parse_ok=excluded.parse_ok,
                 complexity_level=excluded.complexity_level,
                 redundancy_flag=excluded.redundancy_flag,
                 data_availability=excluded.data_availability,
                 gate_notes=excluded.gate_notes,
                 script_path=excluded.script_path,
                 test_start=excluded.test_start,
                 test_end=excluded.test_end,
                 run_id=excluded.run_id,
                 doc_path=excluded.doc_path,
                 legacy_round_id=excluded.legacy_round_id,
                 updated_at=datetime('now')
            """,
            (
                round_id,
                rt,
                date,
                owner,
                market,
                hypothesis,
                market_logic,
                expected_direction,
                parse_ok,
                complexity_level,
                redundancy_flag,
                data_availability,
                gate_notes,
                script_path,
                test_start,
                test_end,
                run_id,
                doc_path,
                legacy_round_id,
            ),
        )
        self._conn.commit()
        return round_id

    def upsert_sfa_metrics(
        self,
        *,
        round_id: str,
        rank_ic_mean: float | None = None,
        rank_ic_t: float | None = None,
        fdr_p: float | None = None,
        rank_icir: float | None = None,
        n_days: int | None = None,
        sfa_result: str | None = None,
    ) -> None:
        # SFA/MFA are mutually exclusive for the same round.
        self._conn.execute("DELETE FROM workflow_mfa_metrics WHERE round_id = ?", (round_id,))
        self._conn.execute(
            """INSERT INTO workflow_sfa_metrics
               (round_id, rank_ic_mean, rank_ic_t, fdr_p, rank_icir, n_days, sfa_result)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(round_id)
               DO UPDATE SET
                 rank_ic_mean=excluded.rank_ic_mean,
                 rank_ic_t=excluded.rank_ic_t,
                 fdr_p=excluded.fdr_p,
                 rank_icir=excluded.rank_icir,
                 n_days=excluded.n_days,
                 sfa_result=excluded.sfa_result
            """,
            (round_id, rank_ic_mean, rank_ic_t, fdr_p, rank_icir, n_days, sfa_result),
        )
        self._conn.commit()

    def upsert_mfa_metrics(
        self,
        *,
        round_id: str,
        executed: bool | None = None,
        not_run_reason: str | None = None,
        excess_return_with_cost: float | None = None,
        ir_with_cost: float | None = None,
        max_drawdown: float | None = None,
        mfa_result: str | None = None,
    ) -> None:
        executed_int = None
        if executed is not None:
            executed_int = 1 if executed else 0
        # SFA/MFA are mutually exclusive for the same round.
        self._conn.execute("DELETE FROM workflow_sfa_metrics WHERE round_id = ?", (round_id,))
        self._conn.execute(
            """INSERT INTO workflow_mfa_metrics
               (round_id, executed, not_run_reason, excess_return_with_cost,
                ir_with_cost, max_drawdown, mfa_result)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(round_id)
               DO UPDATE SET
                 executed=excluded.executed,
                 not_run_reason=excluded.not_run_reason,
                 excess_return_with_cost=excluded.excess_return_with_cost,
                 ir_with_cost=excluded.ir_with_cost,
                 max_drawdown=excluded.max_drawdown,
                 mfa_result=excluded.mfa_result
            """,
            (
                round_id,
                executed_int,
                not_run_reason,
                excess_return_with_cost,
                ir_with_cost,
                max_drawdown,
                mfa_result,
            ),
        )
        self._conn.commit()

    # Backward-compatible aliases
    upsert_layer_a = upsert_sfa_metrics
    upsert_layer_b = upsert_mfa_metrics

    def upsert_decision(
        self,
        *,
        round_id: str,
        decision: str | None = None,
        decision_basis: str | None = None,
        failure_mode: str | None = None,
        next_action: str | None = None,
        decided_at: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO workflow_decisions
               (round_id, decision, decision_basis, failure_mode, next_action, decided_at)
               VALUES (?,?,?,?,?,COALESCE(?, datetime('now')))
               ON CONFLICT(round_id)
               DO UPDATE SET
                 decision=excluded.decision,
                 decision_basis=excluded.decision_basis,
                 failure_mode=excluded.failure_mode,
                 next_action=excluded.next_action,
                 decided_at=excluded.decided_at
            """,
            (round_id, decision, decision_basis, failure_mode, next_action, decided_at),
        )
        self._conn.commit()

    def add_evidence(
        self,
        *,
        round_id: str,
        evidence_type: str,
        evidence_value: str,
        notes: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO workflow_evidence
               (round_id, evidence_type, evidence_value, notes)
               VALUES (?,?,?,?)
            """,
            (round_id, evidence_type, evidence_value, notes),
        )
        self._conn.commit()

    def clear_evidence(self, round_id: str) -> None:
        self._conn.execute("DELETE FROM workflow_evidence WHERE round_id = ?", (round_id,))
        self._conn.commit()

    @staticmethod
    def _normalize_pair(factor_a: str, factor_b: str) -> tuple[str, str]:
        fa = factor_a.strip()
        fb = factor_b.strip()
        if not fa or not fb:
            raise ValueError("factor_a and factor_b must be non-empty")
        if fa == fb:
            raise ValueError("factor_a and factor_b must be different")
        return (fa, fb) if fa < fb else (fb, fa)

    def upsert_similarity(
        self,
        *,
        factor_a: str,
        factor_b: str,
        market: str,
        window: str,
        rho_mean_abs: float,
        rho_p95_abs: float | None = None,
        sample_days: int | None = None,
        source_round_id: str | None = None,
        notes: str | None = None,
        calculated_at: str | None = None,
    ) -> None:
        fa, fb = self._normalize_pair(factor_a, factor_b)
        self._conn.execute(
            """INSERT INTO factor_similarity
               (factor_a, factor_b, market, window, rho_mean_abs, rho_p95_abs,
                sample_days, source_round_id, notes, calculated_at)
               VALUES (?,?,?,?,?,?,?,?,?,COALESCE(?, datetime('now')))
               ON CONFLICT(factor_a, factor_b, market, window)
               DO UPDATE SET
                 rho_mean_abs=excluded.rho_mean_abs,
                 rho_p95_abs=excluded.rho_p95_abs,
                 sample_days=excluded.sample_days,
                 source_round_id=excluded.source_round_id,
                 notes=excluded.notes,
                 calculated_at=excluded.calculated_at
            """,
            (
                fa,
                fb,
                market,
                window,
                rho_mean_abs,
                rho_p95_abs,
                sample_days,
                source_round_id,
                notes,
                calculated_at,
            ),
        )
        self._conn.commit()

    def list_similarity(
        self,
        *,
        factor: str | None = None,
        market: str | None = None,
        window: str | None = None,
        min_rho: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM factor_similarity WHERE 1=1"
        params: list[Any] = []
        if factor:
            query += " AND (factor_a = ? OR factor_b = ?)"
            params.extend([factor, factor])
        if market:
            query += " AND market = ?"
            params.append(market)
        if window:
            query += " AND window = ?"
            params.append(window)
        if min_rho is not None:
            query += " AND rho_mean_abs >= ?"
            params.append(min_rho)
        query += " ORDER BY rho_mean_abs DESC, calculated_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _replacement_gate(corr_value: float, old_icir: float, new_icir: float) -> tuple[bool, float]:
        old_abs = abs(old_icir)
        new_abs = abs(new_icir)
        if old_abs > 0:
            improve_ratio = (new_abs - old_abs) / old_abs
        else:
            improve_ratio = float("inf") if new_abs > 0 else 0.0
        passed = corr_value > 0.80 and improve_ratio >= 0.20
        return passed, improve_ratio

    def record_replacement(
        self,
        *,
        old_factor: str,
        new_factor: str,
        market: str,
        corr_value: float,
        old_icir: float,
        new_icir: float,
        decided_in_round: str | None = None,
        reason: str | None = None,
        created_at: str | None = None,
        enforce_gate: bool = True,
    ) -> dict[str, Any]:
        if old_factor.strip() == new_factor.strip():
            raise ValueError("old_factor and new_factor must be different")
        passed, improve_ratio = self._replacement_gate(corr_value, old_icir, new_icir)
        if enforce_gate and not passed:
            raise ValueError(
                "replacement gate failed: requires corr_value > 0.80 and ICIR improvement >= 20%"
            )

        self._conn.execute(
            """INSERT INTO factor_replacements
               (old_factor, new_factor, market, reason, corr_value,
                old_icir, new_icir, improve_ratio, decided_in_round, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,COALESCE(?, datetime('now')))
            """,
            (
                old_factor,
                new_factor,
                market,
                reason,
                corr_value,
                old_icir,
                new_icir,
                improve_ratio,
                decided_in_round,
                created_at,
            ),
        )
        self._conn.commit()
        return {
            "passed": passed,
            "corr_value": corr_value,
            "old_icir": old_icir,
            "new_icir": new_icir,
            "improve_ratio": improve_ratio,
            "decided_in_round": decided_in_round,
        }

    def list_replacements(
        self,
        *,
        market: str | None = None,
        factor: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM factor_replacements WHERE 1=1"
        params: list[Any] = []
        if market:
            query += " AND market = ?"
            params.append(market)
        if factor:
            query += " AND (old_factor = ? OR new_factor = ?)"
            params.extend([factor, factor])
        query += " ORDER BY created_at DESC, id DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def list_runs(
        self,
        *,
        round_type: str | None = None,
        market: str | None = None,
        decision: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM v_workflow_round_summary WHERE 1=1"
        params: list[Any] = []
        if round_type:
            query += " AND round_type = ?"
            params.append(round_type.lower())
        if market:
            query += " AND market = ?"
            params.append(market)
        if decision:
            query += " AND decision = ?"
            params.append(decision)
        query += " ORDER BY date DESC, round_id DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, round_id: str) -> dict[str, Any] | None:
        run = self._conn.execute(
            "SELECT * FROM workflow_runs WHERE round_id = ?", (round_id,)
        ).fetchone()
        if run is None:
            return None

        sfa_metrics = self._conn.execute(
            "SELECT * FROM workflow_sfa_metrics WHERE round_id = ?", (round_id,)
        ).fetchone()
        mfa_metrics = self._conn.execute(
            "SELECT * FROM workflow_mfa_metrics WHERE round_id = ?", (round_id,)
        ).fetchone()
        decision = self._conn.execute(
            "SELECT * FROM workflow_decisions WHERE round_id = ?", (round_id,)
        ).fetchone()
        evidence = self._conn.execute(
            "SELECT * FROM workflow_evidence WHERE round_id = ? ORDER BY id ASC", (round_id,)
        ).fetchall()

        return {
            "run": dict(run),
            "sfa_metrics": dict(sfa_metrics) if sfa_metrics else None,
            "mfa_metrics": dict(mfa_metrics) if mfa_metrics else None,
            "decision": dict(decision) if decision else None,
            "evidence": [dict(r) for r in evidence],
        }

    def sync_round_to_markdown(
        self,
        round_id: str,
        output_path: Path | str,
    ) -> Path:
        payload = self.get_run(round_id)
        if payload is None:
            raise ValueError(f"Round not found: {round_id}")

        run = payload["run"]
        sfa_metrics = payload["sfa_metrics"] or {}
        mfa_metrics = payload["mfa_metrics"] or {}
        decision = payload["decision"] or {}
        evidence = payload["evidence"]

        def _fmt(value: Any, default: str = "N/A") -> str:
            if value is None:
                return default
            if isinstance(value, float):
                return f"{value:.6g}"
            return str(value)

        executed_val = mfa_metrics.get("executed")
        executed_text = "N/A"
        if executed_val is not None:
            executed_text = "yes" if int(executed_val) == 1 else "no"

        by_type: dict[str, list[str]] = {}
        for item in evidence:
            key = str(item.get("evidence_type") or "")
            val = str(item.get("evidence_value") or "").strip()
            if not key or not val:
                continue
            by_type.setdefault(key, []).append(val)

        output_paths = by_type.get("output_path", [])
        db_queries = by_type.get("db_query", [])
        run_ids = by_type.get("run_id", [])
        doc_paths = by_type.get("doc", [])

        lines = [
            f"# {run['round_id']}",
            "",
            "## round_meta",
            f"- round_id: {_fmt(run.get('round_id'))}",
            f"- round_type: {_fmt(run.get('round_type'))}",
            f"- date: {_fmt(run.get('date'))}",
            f"- owner: {_fmt(run.get('owner'))}",
            f"- market: {_fmt(run.get('market'))}",
            "",
            "## retrieve",
            "- memory_refs:",
            "  - AGENTS.md#推荐方向",
            "  - AGENTS.md#禁止方向",
            f"- db_snapshot_query: {_fmt(db_queries[0] if db_queries else None)}",
            f"- similarity_snapshot_query: {_fmt(db_queries[1] if len(db_queries) > 1 else None)}",
            "",
        ]

        if run.get("round_type") == "mfa":
            lines.extend(
                [
                    "## generate",
                    f"- hypothesis: {_fmt(run.get('hypothesis'))}",
                    f"- market_logic: {_fmt(run.get('market_logic'))}",
                    f"- expected_direction: {_fmt(run.get('expected_direction'))}",
                    "- model_families:",
                    "  - linear: N/A",
                    "  - nonlinear: N/A",
                    "- factor_pool_notes: N/A",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## generate",
                    f"- hypothesis: {_fmt(run.get('hypothesis'))}",
                    f"- market_logic: {_fmt(run.get('market_logic'))}",
                    f"- expected_direction: {_fmt(run.get('expected_direction'))}",
                    "- factor_expression_list:",
                    "  - factor_1: N/A",
                    "",
                ]
            )

        lines.extend(
            [
                "## evaluate_preflight",
                f"- parse_ok: {_fmt(run.get('parse_ok'))}",
                f"- complexity_level: {_fmt(run.get('complexity_level'))}",
                f"- redundancy_flag: {_fmt(run.get('redundancy_flag'))}",
                f"- data_availability: {_fmt(run.get('data_availability'))}",
                f"- gate_notes: {_fmt(run.get('gate_notes'))}",
                "",
                "## evaluate_metrics",
                f"- script: {_fmt(run.get('script_path'))}",
                f"- start: {_fmt(run.get('test_start'))}",
                f"- end: {_fmt(run.get('test_end'))}",
                f"- run_id: {_fmt(run.get('run_id') or (run_ids[0] if run_ids else None))}",
                "- output_paths:",
            ]
        )
        if output_paths:
            for p in output_paths:
                lines.append(f"  - {p}")
        else:
            lines.append("  - N/A")
        lines.append("")

        if run.get("round_type") == "mfa":
            lines.extend(
                [
                    f"- executed: {executed_text}",
                    f"- not_run_reason: {_fmt(mfa_metrics.get('not_run_reason'))}",
                    f"- excess_return_with_cost: {_fmt(mfa_metrics.get('excess_return_with_cost'))}",
                    f"- ir_with_cost: {_fmt(mfa_metrics.get('ir_with_cost'))}",
                    f"- max_drawdown: {_fmt(mfa_metrics.get('max_drawdown'))}",
                    "- stress_test_notes: N/A",
                    f"- mfa_result: {_fmt(mfa_metrics.get('mfa_result'))}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"- rank_ic_mean: {_fmt(sfa_metrics.get('rank_ic_mean'))}",
                    f"- rank_ic_t: {_fmt(sfa_metrics.get('rank_ic_t'))}",
                    f"- fdr_p: {_fmt(sfa_metrics.get('fdr_p'))}",
                    f"- rank_icir: {_fmt(sfa_metrics.get('rank_icir'))}",
                    f"- n_days: {_fmt(sfa_metrics.get('n_days'))}",
                    "- max_rho_abs: N/A",
                    f"- sfa_result: {_fmt(sfa_metrics.get('sfa_result'))}",
                    "",
                ]
            )

        lines.extend(
            [
                "## distill_decision",
                f"- decision: {_fmt(decision.get('decision'))}",
                f"- decision_basis: {_fmt(decision.get('decision_basis'))}",
                f"- failure_mode: {_fmt(decision.get('failure_mode'))}",
                f"- next_action: {_fmt(decision.get('next_action'))}",
                "",
                "## distill_evidence",
                f"- db_query: {_fmt(db_queries[0] if db_queries else None)}",
                f"- run_id: {_fmt(run.get('run_id') or (run_ids[0] if run_ids else None))}",
                "- outputs:",
            ]
        )
        if output_paths:
            for p in output_paths:
                lines.append(f"  - {p}")
        else:
            lines.append("  - N/A")
        lines.append(f"- doc: {_fmt(run.get('doc_path') or (doc_paths[0] if doc_paths else None))}")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

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

CREATE INDEX IF NOT EXISTS idx_workflow_runs_type_date ON workflow_runs(round_type, date DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_market ON workflow_runs(market);
CREATE INDEX IF NOT EXISTS idx_workflow_decisions_decision ON workflow_decisions(decision);
CREATE INDEX IF NOT EXISTS idx_workflow_evidence_round ON workflow_evidence(round_id);
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
        ]

        if run.get("round_type") == "mfa":
            lines.extend(
                [
                    "## portfolio_hypothesis",
                    f"- hypothesis: {_fmt(run.get('hypothesis'))}",
                    f"- market_logic: {_fmt(run.get('market_logic'))}",
                    f"- expected_direction: {_fmt(run.get('expected_direction'))}",
                    "",
                    "## factor_pool_source",
                    "- source_query: N/A",
                    "- n_factors: N/A",
                    "- factor_pool_notes: N/A",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## hypothesis",
                    f"- hypothesis: {_fmt(run.get('hypothesis'))}",
                    f"- market_logic: {_fmt(run.get('market_logic'))}",
                    f"- expected_direction: {_fmt(run.get('expected_direction'))}",
                    "",
                    "## factor_expression_list",
                    "- factor_1: N/A",
                    "",
                ]
            )

        lines.extend(
            [
                "## preflight_gate",
                f"- parse_ok: {_fmt(run.get('parse_ok'))}",
                f"- complexity_level: {_fmt(run.get('complexity_level'))}",
                f"- redundancy_flag: {_fmt(run.get('redundancy_flag'))}",
                f"- data_availability: {_fmt(run.get('data_availability'))}",
                f"- gate_notes: {_fmt(run.get('gate_notes'))}",
                "",
                "## experiment_config",
                f"- script: {_fmt(run.get('script_path'))}",
                f"- start: {_fmt(run.get('test_start'))}",
                f"- end: {_fmt(run.get('test_end'))}",
                f"- run_id: {_fmt(run.get('run_id'))}",
                "",
            ]
        )

        if run.get("round_type") == "mfa":
            lines.extend(
                [
                    "## multi_factor_metrics",
                    f"- executed: {executed_text}",
                    f"- not_run_reason: {_fmt(mfa_metrics.get('not_run_reason'))}",
                    f"- excess_return_with_cost: {_fmt(mfa_metrics.get('excess_return_with_cost'))}",
                    f"- ir_with_cost: {_fmt(mfa_metrics.get('ir_with_cost'))}",
                    f"- max_drawdown: {_fmt(mfa_metrics.get('max_drawdown'))}",
                    f"- mfa_result: {_fmt(mfa_metrics.get('mfa_result'))}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "## single_factor_metrics",
                    f"- rank_ic_mean: {_fmt(sfa_metrics.get('rank_ic_mean'))}",
                    f"- rank_ic_t: {_fmt(sfa_metrics.get('rank_ic_t'))}",
                    f"- fdr_p: {_fmt(sfa_metrics.get('fdr_p'))}",
                    f"- rank_icir: {_fmt(sfa_metrics.get('rank_icir'))}",
                    f"- n_days: {_fmt(sfa_metrics.get('n_days'))}",
                    f"- sfa_result: {_fmt(sfa_metrics.get('sfa_result'))}",
                    "",
                ]
            )

        lines.extend(
            [
                "## decision",
                f"- decision: {_fmt(decision.get('decision'))}",
                f"- decision_basis: {_fmt(decision.get('decision_basis'))}",
                f"- failure_mode: {_fmt(decision.get('failure_mode'))}",
                f"- next_action: {_fmt(decision.get('next_action'))}",
                "",
                "## evidence_links",
            ]
        )

        if evidence:
            for item in evidence:
                lines.append(f"- {item.get('evidence_type')}: {item.get('evidence_value')}")
        else:
            lines.append("- evidence: N/A")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

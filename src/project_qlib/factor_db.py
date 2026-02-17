"""Factor Library — SQLite-backed persistent storage for factor research results.

Two-table design:
  - ``factors``          : factor *definitions* (name, expression, category, …)
  - ``factor_test_results`` : one row per (factor × market × test_period)

This allows the same factor to carry independent IC statistics for every
market (csi1000, csiall, csi300 …) and every test window.

Usage::

    from project_qlib.factor_db import FactorDB
    db = FactorDB()
    db.upsert_factor(name="CSTM_VOL_CV_10", expression="...", category="volume",
                     source="Custom", status="Accepted")
    db.upsert_test_result("CSTM_VOL_CV_10", market="csi1000",
                          test_start="2019-01-01", test_end="2025-12-31",
                          rank_ic_mean=-0.027, rank_ic_t=-15.7, rank_icir=-0.38)
    df = db.list_factors(market="csi1000", status="Accepted")
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from project_qlib.runtime import PROJECT_ROOT

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "factor_library.db"

# --------------------------------------------------------------------------- #
#  Schema  (v2 — definition / results split)
# --------------------------------------------------------------------------- #

_SCHEMA_V2 = """
-- Factor definitions (one row per factor)
CREATE TABLE IF NOT EXISTS factors (
    name            TEXT PRIMARY KEY,
    expression      TEXT,
    category        TEXT,       -- volume, vwap, range, gap, momentum, …
    source          TEXT,       -- Alpha158 / Custom
    market_logic    TEXT,       -- 市场直觉
    status          TEXT DEFAULT 'Candidate',  -- Candidate / Accepted / Rejected / Baseline
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

-- Test results (one row per factor × market × test_period)
CREATE TABLE IF NOT EXISTS factor_test_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name     TEXT NOT NULL,
    market          TEXT NOT NULL,      -- csiall / csi1000 / csi300
    test_start      TEXT NOT NULL,
    test_end        TEXT NOT NULL,
    n_days          INTEGER,
    ic_mean         REAL,
    ic_std          REAL,
    rank_ic_mean    REAL,
    rank_ic_std     REAL,
    rank_ic_t       REAL,
    rank_ic_p       REAL,
    rank_icir       REAL,
    fdr_p           REAL,
    significant     INTEGER,   -- 1 if passes FDR < 0.01
    hea_round       TEXT,
    evidence        TEXT,
    notes           TEXT,
    tested_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (factor_name) REFERENCES factors(name),
    UNIQUE(factor_name, market, test_start, test_end)
);

-- Portfolio backtest results (one row per factor × market × holding_period × topk)
CREATE TABLE IF NOT EXISTS factor_backtest_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name     TEXT NOT NULL,
    market          TEXT NOT NULL,          -- csiall / csi1000 / csi300
    holding_period  INTEGER NOT NULL,       -- holding days: 1, 5, 10, 20
    topk            INTEGER,               -- number of stocks in portfolio
    cost_rate       REAL DEFAULT 0.0015,   -- one-way transaction cost rate
    test_start      TEXT NOT NULL,
    test_end        TEXT NOT NULL,
    -- Return metrics
    cumulative_return REAL,                -- total return (e.g. 1.138 = 113.8%)
    annual_return   REAL,                  -- annualized return
    excess_return   REAL,                  -- annualized excess return over benchmark
    -- Risk metrics
    ir              REAL,                  -- Information Ratio (with cost)
    sharpe          REAL,                  -- Sharpe Ratio
    max_drawdown    REAL,                  -- maximum drawdown (negative value)
    -- Trading metrics
    turnover        REAL,                  -- average daily turnover
    win_rate        REAL,                  -- monthly win rate (0-1)
    -- Meta
    benchmark       TEXT,                  -- benchmark name (e.g. 'SH000852')
    model_type      TEXT,                  -- LightGBM / XGBoost / etc.
    n_factors       INTEGER,               -- number of factors used in model
    hea_round       TEXT,
    run_id          TEXT,                  -- mlflow run id
    evidence        TEXT,
    notes           TEXT,
    tested_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (factor_name) REFERENCES factors(name),
    UNIQUE(factor_name, market, holding_period, topk, test_start, test_end)
);

-- IC decay analysis (one row per factor × market × horizon)
CREATE TABLE IF NOT EXISTS factor_ic_decay (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    factor_name     TEXT NOT NULL,
    market          TEXT NOT NULL,          -- csiall / csi1000 / csi300
    horizon_days    INTEGER NOT NULL,       -- forward return horizon: 1, 3, 5, 10, 20
    test_start      TEXT NOT NULL,
    test_end        TEXT NOT NULL,
    n_days          INTEGER,
    ic_mean         REAL,
    ic_std          REAL,
    rank_ic_mean    REAL,
    rank_ic_std     REAL,
    rank_ic_t       REAL,
    rank_icir       REAL,
    notes           TEXT,
    tested_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (factor_name) REFERENCES factors(name),
    UNIQUE(factor_name, market, horizon_days, test_start, test_end)
);
"""


class FactorDB:
    """SQLite-backed factor library (v2 — split definition / results)."""

    DB_VERSION = 2

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    #  Schema management
    # ------------------------------------------------------------------ #

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist, and migrate from v1 if needed."""
        # Check if we still have the old single-table layout
        old_cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(factors)").fetchall()
        }
        need_migrate = "rank_ic_mean" in old_cols  # v1 had IC cols in factors

        if need_migrate:
            self._migrate_v1_to_v2()
        else:
            self._conn.executescript(_SCHEMA_V2)
            self._conn.commit()

    def _migrate_v1_to_v2(self) -> None:
        """Migrate from v1 (IC cols in factors) to v2 (separate tables)."""
        print("[FactorDB] Migrating schema v1 → v2 …")

        # 1. Read old data
        old_factors = pd.read_sql_query("SELECT * FROM factors", self._conn)
        old_history = pd.DataFrame()
        tables = [
            r[0]
            for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        if "factor_test_history" in tables:
            old_history = pd.read_sql_query(
                "SELECT * FROM factor_test_history", self._conn
            )

        # 2. Drop old tables
        self._conn.execute("DROP TABLE IF EXISTS factor_test_history")
        self._conn.execute("DROP TABLE IF EXISTS factors")
        self._conn.commit()

        # 3. Create new tables
        self._conn.executescript(_SCHEMA_V2)
        self._conn.commit()

        # 4. Insert factor definitions
        now = datetime.now(timezone.utc).isoformat()
        for _, row in old_factors.iterrows():
            self._conn.execute(
                """INSERT OR IGNORE INTO factors
                   (name, expression, category, source, market_logic, status, notes,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    row["name"],
                    row.get("expression"),
                    row.get("category"),
                    row.get("source"),
                    row.get("market_logic"),
                    row.get("status", "Candidate"),
                    row.get("notes"),
                    row.get("created_at", now),
                    now,
                ),
            )

        # 5. Insert test results from old factors table (if IC cols present)
        ic_cols = {"market", "test_start", "test_end", "rank_ic_mean"}
        if ic_cols.issubset(set(old_factors.columns)):
            for _, row in old_factors.iterrows():
                mkt = row.get("market")
                ts = row.get("test_start")
                te = row.get("test_end")
                if pd.isna(mkt) or pd.isna(ts) or pd.isna(te):
                    continue
                self._insert_test_result_row(row["name"], row)

        # 6. Insert from old history table
        if not old_history.empty:
            for _, row in old_history.iterrows():
                fn = row.get("factor_name")
                mkt = row.get("market")
                ts = row.get("test_start")
                te = row.get("test_end")
                if pd.isna(fn) or pd.isna(mkt) or pd.isna(ts) or pd.isna(te):
                    continue
                self._insert_test_result_row(fn, row)

        self._conn.commit()
        n_factors = self._conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0]
        n_results = self._conn.execute(
            "SELECT COUNT(*) FROM factor_test_results"
        ).fetchone()[0]
        print(
            f"[FactorDB] Migration done: {n_factors} factors, {n_results} test results."
        )

    def _insert_test_result_row(self, factor_name: str, row: Any) -> None:
        """Helper: insert a single test-result row from a pandas Series/dict."""
        def _g(key: str) -> Any:
            val = row.get(key) if hasattr(row, "get") else row[key]
            return None if (val is None or (isinstance(val, float) and pd.isna(val))) else val

        self._conn.execute(
            """INSERT OR REPLACE INTO factor_test_results
               (factor_name, market, test_start, test_end, n_days,
                ic_mean, ic_std, rank_ic_mean, rank_ic_std,
                rank_ic_t, rank_ic_p, rank_icir,
                fdr_p, significant, hea_round, evidence, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                factor_name,
                _g("market"),
                _g("test_start"),
                _g("test_end"),
                _g("n_days"),
                _g("ic_mean"),
                _g("ic_std"),
                _g("rank_ic_mean"),
                _g("rank_ic_std"),
                _g("rank_ic_t"),
                _g("rank_ic_p"),
                _g("rank_icir"),
                _g("fdr_p"),
                _g("significant"),
                _g("hea_round"),
                _g("evidence"),
                _g("notes"),
            ),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ #
    #  Factor definitions
    # ------------------------------------------------------------------ #

    def upsert_factor(
        self,
        name: str,
        expression: str | None = None,
        category: str | None = None,
        source: str | None = None,
        market_logic: str | None = None,
        status: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Insert or update a factor *definition*. Only non-None fields are updated."""
        now = datetime.now(timezone.utc).isoformat()
        existing = self._conn.execute(
            "SELECT name FROM factors WHERE name = ?", (name,)
        ).fetchone()

        if existing:
            updates: dict[str, Any] = {}
            for field, value in [
                ("expression", expression),
                ("category", category),
                ("source", source),
                ("market_logic", market_logic),
                ("status", status),
                ("notes", notes),
            ]:
                if value is not None:
                    updates[field] = value
            updates["updated_at"] = now
            if updates:
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                vals = list(updates.values()) + [name]
                self._conn.execute(
                    f"UPDATE factors SET {set_clause} WHERE name = ?", vals
                )
        else:
            self._conn.execute(
                """INSERT INTO factors
                   (name, expression, category, source, market_logic, status, notes,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    name, expression, category, source, market_logic,
                    status or "Candidate", notes, now, now,
                ),
            )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    #  Test results
    # ------------------------------------------------------------------ #

    def upsert_test_result(
        self,
        factor_name: str,
        market: str,
        test_start: str,
        test_end: str,
        n_days: int | None = None,
        ic_mean: float | None = None,
        ic_std: float | None = None,
        rank_ic_mean: float | None = None,
        rank_ic_std: float | None = None,
        rank_ic_t: float | None = None,
        rank_ic_p: float | None = None,
        rank_icir: float | None = None,
        fdr_p: float | None = None,
        significant: bool | None = None,
        hea_round: str | None = None,
        evidence: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Insert or update a test result for (factor × market × period).

        Returns the row id.
        """
        sig_val = 1 if significant else (0 if significant is not None else None)
        cur = self._conn.execute(
            """INSERT INTO factor_test_results
               (factor_name, market, test_start, test_end, n_days,
                ic_mean, ic_std, rank_ic_mean, rank_ic_std,
                rank_ic_t, rank_ic_p, rank_icir,
                fdr_p, significant, hea_round, evidence, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(factor_name, market, test_start, test_end)
               DO UPDATE SET
                 n_days=excluded.n_days,
                 ic_mean=excluded.ic_mean, ic_std=excluded.ic_std,
                 rank_ic_mean=excluded.rank_ic_mean, rank_ic_std=excluded.rank_ic_std,
                 rank_ic_t=excluded.rank_ic_t, rank_ic_p=excluded.rank_ic_p,
                 rank_icir=excluded.rank_icir,
                 fdr_p=excluded.fdr_p, significant=excluded.significant,
                 hea_round=excluded.hea_round, evidence=excluded.evidence,
                 notes=excluded.notes,
                 tested_at=datetime('now')
            """,
            (
                factor_name, market, test_start, test_end, n_days,
                ic_mean, ic_std, rank_ic_mean, rank_ic_std,
                rank_ic_t, rank_ic_p, rank_icir,
                fdr_p, sig_val, hea_round, evidence, notes,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # Backward-compatible alias
    add_test_record = upsert_test_result

    # ------------------------------------------------------------------ #
    #  Backtest results  (portfolio-level, per holding period)
    # ------------------------------------------------------------------ #

    def upsert_backtest_result(
        self,
        factor_name: str,
        market: str,
        holding_period: int,
        test_start: str,
        test_end: str,
        topk: int | None = None,
        cost_rate: float = 0.0015,
        cumulative_return: float | None = None,
        annual_return: float | None = None,
        excess_return: float | None = None,
        ir: float | None = None,
        sharpe: float | None = None,
        max_drawdown: float | None = None,
        turnover: float | None = None,
        win_rate: float | None = None,
        benchmark: str | None = None,
        model_type: str | None = None,
        n_factors: int | None = None,
        hea_round: str | None = None,
        run_id: str | None = None,
        evidence: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Insert or update a backtest result for (factor × market × holding_period × topk)."""
        cur = self._conn.execute(
            """INSERT INTO factor_backtest_results
               (factor_name, market, holding_period, topk, cost_rate,
                test_start, test_end,
                cumulative_return, annual_return, excess_return,
                ir, sharpe, max_drawdown,
                turnover, win_rate,
                benchmark, model_type, n_factors,
                hea_round, run_id, evidence, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(factor_name, market, holding_period, topk, test_start, test_end)
               DO UPDATE SET
                 cost_rate=excluded.cost_rate,
                 cumulative_return=excluded.cumulative_return,
                 annual_return=excluded.annual_return,
                 excess_return=excluded.excess_return,
                 ir=excluded.ir, sharpe=excluded.sharpe,
                 max_drawdown=excluded.max_drawdown,
                 turnover=excluded.turnover, win_rate=excluded.win_rate,
                 benchmark=excluded.benchmark, model_type=excluded.model_type,
                 n_factors=excluded.n_factors,
                 hea_round=excluded.hea_round, run_id=excluded.run_id,
                 evidence=excluded.evidence, notes=excluded.notes,
                 tested_at=datetime('now')
            """,
            (
                factor_name, market, holding_period, topk, cost_rate,
                test_start, test_end,
                cumulative_return, annual_return, excess_return,
                ir, sharpe, max_drawdown,
                turnover, win_rate,
                benchmark, model_type, n_factors,
                hea_round, run_id, evidence, notes,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_backtest_results(
        self,
        factor_name: str,
        market: str | None = None,
        holding_period: int | None = None,
    ) -> pd.DataFrame:
        """Get backtest results, optionally filtered by market and/or holding period."""
        query = "SELECT * FROM factor_backtest_results WHERE factor_name = ?"
        params: list[Any] = [factor_name]
        if market:
            query += " AND market = ?"
            params.append(market)
        if holding_period is not None:
            query += " AND holding_period = ?"
            params.append(holding_period)
        query += " ORDER BY holding_period, tested_at DESC"
        return pd.read_sql_query(query, self._conn, params=params)

    def list_backtest_results(
        self,
        market: str | None = None,
        holding_period: int | None = None,
        order_by: str = "ir",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """List all backtest results, optionally filtered."""
        query = """
            SELECT b.*, f.category, f.source, f.status
            FROM factor_backtest_results b
            JOIN factors f ON b.factor_name = f.name
            WHERE 1=1
        """
        params: list[Any] = []
        if market:
            query += " AND b.market = ?"
            params.append(market)
        if holding_period is not None:
            query += " AND b.holding_period = ?"
            params.append(holding_period)
        if order_by in ("ir", "sharpe", "annual_return", "excess_return"):
            query += f" ORDER BY b.{order_by} DESC"
        elif order_by == "max_drawdown":
            query += " ORDER BY b.max_drawdown DESC"  # less negative = better
        else:
            query += f" ORDER BY b.{order_by}"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        return pd.read_sql_query(query, self._conn, params=params)

    # ------------------------------------------------------------------ #
    #  IC decay  (multi-horizon IC analysis)
    # ------------------------------------------------------------------ #

    def upsert_ic_decay(
        self,
        factor_name: str,
        market: str,
        horizon_days: int,
        test_start: str,
        test_end: str,
        n_days: int | None = None,
        ic_mean: float | None = None,
        ic_std: float | None = None,
        rank_ic_mean: float | None = None,
        rank_ic_std: float | None = None,
        rank_ic_t: float | None = None,
        rank_icir: float | None = None,
        notes: str | None = None,
    ) -> int:
        """Insert or update IC decay for a specific horizon."""
        cur = self._conn.execute(
            """INSERT INTO factor_ic_decay
               (factor_name, market, horizon_days, test_start, test_end,
                n_days, ic_mean, ic_std, rank_ic_mean, rank_ic_std,
                rank_ic_t, rank_icir, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(factor_name, market, horizon_days, test_start, test_end)
               DO UPDATE SET
                 n_days=excluded.n_days,
                 ic_mean=excluded.ic_mean, ic_std=excluded.ic_std,
                 rank_ic_mean=excluded.rank_ic_mean, rank_ic_std=excluded.rank_ic_std,
                 rank_ic_t=excluded.rank_ic_t, rank_icir=excluded.rank_icir,
                 notes=excluded.notes,
                 tested_at=datetime('now')
            """,
            (
                factor_name, market, horizon_days, test_start, test_end,
                n_days, ic_mean, ic_std, rank_ic_mean, rank_ic_std,
                rank_ic_t, rank_icir, notes,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_ic_decay(
        self,
        factor_name: str,
        market: str | None = None,
    ) -> pd.DataFrame:
        """Get IC decay profile for a factor across different horizons."""
        query = "SELECT * FROM factor_ic_decay WHERE factor_name = ?"
        params: list[Any] = [factor_name]
        if market:
            query += " AND market = ?"
            params.append(market)
        query += " ORDER BY horizon_days"
        return pd.read_sql_query(query, self._conn, params=params)

    def list_ic_decay(
        self,
        market: str,
        horizon_days: int = 5,
        order_by: str = "rank_icir",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """List IC decay for all factors at a specific horizon."""
        query = """
            SELECT d.*, f.category, f.source, f.status
            FROM factor_ic_decay d
            JOIN factors f ON d.factor_name = f.name
            WHERE d.market = ? AND d.horizon_days = ?
        """
        params: list[Any] = [market, horizon_days]
        if order_by in ("rank_icir", "rank_ic_t", "rank_ic_mean", "ic_mean"):
            query += f" ORDER BY ABS(d.{order_by}) DESC"
        else:
            query += f" ORDER BY d.{order_by}"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        return pd.read_sql_query(query, self._conn, params=params)

    # ------------------------------------------------------------------ #
    #  Query — factor definitions
    # ------------------------------------------------------------------ #

    def get_factor(self, name: str) -> dict[str, Any] | None:
        """Get a factor definition by name."""
        row = self._conn.execute(
            "SELECT * FROM factors WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    #  Query — test results
    # ------------------------------------------------------------------ #

    def get_test_results(
        self, factor_name: str, market: str | None = None
    ) -> pd.DataFrame:
        """Get all test results for a factor, optionally filtered by market."""
        if market:
            return pd.read_sql_query(
                "SELECT * FROM factor_test_results "
                "WHERE factor_name = ? AND market = ? ORDER BY tested_at DESC",
                self._conn,
                params=[factor_name, market],
            )
        return pd.read_sql_query(
            "SELECT * FROM factor_test_results "
            "WHERE factor_name = ? ORDER BY tested_at DESC",
            self._conn,
            params=[factor_name],
        )

    # Backward-compatible alias
    get_test_history = get_test_results

    # ------------------------------------------------------------------ #
    #  Query — joined views
    # ------------------------------------------------------------------ #

    def list_factors(
        self,
        status: str | None = None,
        source: str | None = None,
        category: str | None = None,
        market: str | None = None,
        significant_only: bool = False,
        order_by: str = "rank_icir",
        limit: int | None = None,
    ) -> pd.DataFrame:
        """List factors, optionally joined with test results for *market*.

        When *market* is given the returned DataFrame includes IC columns
        from ``factor_test_results`` for that market.  When *market* is
        ``None``, only factor definitions are returned.
        """
        if market:
            query = """
                SELECT f.*, r.market, r.test_start, r.test_end, r.n_days,
                       r.ic_mean, r.ic_std, r.rank_ic_mean, r.rank_ic_std,
                       r.rank_ic_t, r.rank_ic_p, r.rank_icir,
                       r.fdr_p, r.significant, r.hea_round AS result_hea,
                       r.evidence AS result_evidence, r.tested_at
                FROM factors f
                LEFT JOIN factor_test_results r
                  ON f.name = r.factor_name AND r.market = ?
                WHERE 1=1
            """
            params: list[Any] = [market]
        else:
            query = "SELECT f.* FROM factors f WHERE 1=1"
            params = []

        if status:
            query += " AND f.status = ?"
            params.append(status)
        if source:
            query += " AND f.source = ?"
            params.append(source)
        if category:
            query += " AND f.category = ?"
            params.append(category)
        if significant_only and market:
            query += " AND r.significant = 1"

        abs_cols = ("rank_icir", "rank_ic_t", "rank_ic_mean", "ic_mean")
        if order_by in abs_cols:
            col = f"r.{order_by}" if market else order_by
            query += f" ORDER BY ABS({col}) DESC"
        else:
            query += f" ORDER BY {order_by}"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        return pd.read_sql_query(query, self._conn, params=params)

    def count_by_status(self) -> dict[str, int]:
        """Count factors by status."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM factors GROUP BY status"
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def count_results_by_market(self) -> dict[str, int]:
        """Count test result rows by market."""
        rows = self._conn.execute(
            "SELECT market, COUNT(*) as cnt FROM factor_test_results GROUP BY market"
        ).fetchall()
        return {row["market"]: row["cnt"] for row in rows}

    def count_backtest_by_market(self) -> dict[str, dict[str, int]]:
        """Count backtest result rows by market × holding_period."""
        rows = self._conn.execute(
            "SELECT market, holding_period, COUNT(*) as cnt "
            "FROM factor_backtest_results GROUP BY market, holding_period"
        ).fetchall()
        result: dict[str, dict[str, int]] = {}
        for row in rows:
            mkt = row["market"]
            hp = str(row["holding_period"])
            result.setdefault(mkt, {})[hp] = row["cnt"]
        return result

    def count_ic_decay_by_market(self) -> dict[str, dict[str, int]]:
        """Count IC decay rows by market × horizon."""
        rows = self._conn.execute(
            "SELECT market, horizon_days, COUNT(*) as cnt "
            "FROM factor_ic_decay GROUP BY market, horizon_days"
        ).fetchall()
        result: dict[str, dict[str, int]] = {}
        for row in rows:
            mkt = row["market"]
            h = str(row["horizon_days"])
            result.setdefault(mkt, {})[h] = row["cnt"]
        return result

    # ------------------------------------------------------------------ #
    #  Export & convenience
    # ------------------------------------------------------------------ #

    def export_csv(
        self,
        output_path: Path | str | None = None,
        market: str | None = None,
    ) -> Path:
        """Export factors (with optional market results) to CSV."""
        df = self.list_factors(market=market)
        if output_path is None:
            suffix = f"_{market}" if market else ""
            output_path = PROJECT_ROOT / "outputs" / f"factor_library_export{suffix}.csv"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

    def get_accepted_factors(self, market: str | None = None) -> pd.DataFrame:
        """Convenience: get all Accepted factors, optionally with IC stats for *market*."""
        return self.list_factors(
            status="Accepted", market=market, significant_only=bool(market)
        )

    def summary(self, market: str | None = None) -> str:
        """Return a brief text summary of the factor library."""
        counts = self.count_by_status()
        total = sum(counts.values())
        lines = [f"Factor Library: {total} factors total"]
        for status, cnt in sorted(counts.items()):
            lines.append(f"  {status}: {cnt}")

        # IC test results by market
        mkt_counts = self.count_results_by_market()
        if mkt_counts:
            lines.append("\nIC test results by market:")
            for mkt, cnt in sorted(mkt_counts.items()):
                lines.append(f"  {mkt}: {cnt} results")

        # Backtest results by market × holding period
        bt_counts = self.count_backtest_by_market()
        if bt_counts:
            lines.append("\nBacktest results (market × holding_period):")
            for mkt in sorted(bt_counts):
                hp_str = ", ".join(
                    f"h={hp}d:{cnt}" for hp, cnt in sorted(bt_counts[mkt].items(), key=lambda x: int(x[0]))
                )
                lines.append(f"  {mkt}: {hp_str}")

        # IC decay by market × horizon
        decay_counts = self.count_ic_decay_by_market()
        if decay_counts:
            lines.append("\nIC decay profiles (market × horizon):")
            for mkt in sorted(decay_counts):
                h_str = ", ".join(
                    f"{h}d:{cnt}" for h, cnt in sorted(decay_counts[mkt].items(), key=lambda x: int(x[0]))
                )
                lines.append(f"  {mkt}: {h_str}")

        # Top 5 by |ICIR| for specified market (or first available)
        show_mkt = market or (list(mkt_counts.keys())[0] if mkt_counts else None)
        if show_mkt:
            top = self.list_factors(
                market=show_mkt, significant_only=True,
                order_by="rank_icir", limit=5,
            )
            if not top.empty:
                lines.append(f"\nTop 5 by |RankICIR| ({show_mkt}):")
                for _, row in top.iterrows():
                    icir = row.get("rank_icir")
                    t = row.get("rank_ic_t")
                    icir_s = f"{icir:+.3f}" if icir == icir else "N/A"
                    t_s = f"{t:+.1f}" if t == t else "N/A"
                    lines.append(
                        f"  {row['name']:30s} | ICIR={icir_s} | t={t_s} | {row['status']}"
                    )

        return "\n".join(lines)

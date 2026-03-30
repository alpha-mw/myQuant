"""SQLite persistence for run history, presets, and trade records."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_DB_PATH = Path("data") / "web_runs.db"
_WORKSPACE_LEARNING_DIR = Path("data") / "workspace_learning"


class RunHistoryStore:
    """Thread-safe SQLite store for research run history."""

    def __init__(self, db_path: Path = _DB_PATH):
        self._db_path = db_path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return self._local.conn

    def init_db(self) -> None:
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                job_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                report_markdown TEXT DEFAULT '',
                result_summary_json TEXT DEFAULT '{}',
                total_time REAL,
                market TEXT DEFAULT 'CN',
                stock_pool TEXT DEFAULT '[]',
                risk_level TEXT DEFAULT '中等',
                preset_id TEXT DEFAULT '',
                error TEXT DEFAULT '',
                selection_meta_json TEXT DEFAULT '{}',
                recall_context_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS presets (
                preset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_records (
                trade_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                rationale TEXT DEFAULT '',
                status TEXT DEFAULT 'suggested',
                outcome_status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES runs(job_id)
            );
            """
        )
        # Migrate existing DBs
        for col, sql in [
            ("error",               "TEXT DEFAULT ''"),
            ("selection_meta_json", "TEXT DEFAULT '{}'"),
            ("recall_context_json", "TEXT DEFAULT '{}'"),
        ]:
            self._ensure_column(conn, "runs", col, sql)
        conn.commit()

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_sql: str,
    ) -> None:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        columns = {row["name"] for row in rows}
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

    # ------------------------------------------------------------------ runs

    def save_run(
        self,
        job_id: str,
        created_at: str,
        status: str,
        request_json: str,
        report_markdown: str = "",
        result_summary_json: str = "{}",
        total_time: Optional[float] = None,
        market: str = "CN",
        stock_pool: str = "[]",
        risk_level: str = "中等",
        preset_id: str = "",
        error: str = "",
        selection_meta_json: str = "{}",
        recall_context_json: str = "{}",
    ) -> None:
        conn = self._conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (job_id, created_at, status, request_json, report_markdown,
                 result_summary_json, total_time, market, stock_pool, risk_level,
                 preset_id, error, selection_meta_json, recall_context_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                created_at,
                status,
                request_json,
                report_markdown,
                result_summary_json,
                total_time,
                market,
                stock_pool,
                risk_level,
                preset_id,
                error,
                selection_meta_json,
                recall_context_json,
            ),
        )
        conn.commit()

    def get_run(self, job_id: str) -> Optional[dict[str, Any]]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM runs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def get_report(self, job_id: str) -> Optional[str]:
        conn = self._conn()
        row = conn.execute(
            "SELECT report_markdown FROM runs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return row["report_markdown"] if row else None

    def get_history(
        self,
        page: int = 1,
        per_page: int = 20,
        market: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        conn = self._conn()
        where = ""
        params: list[Any] = []
        if market:
            where = "WHERE market = ?"
            params.append(market)

        total_row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM runs {where}", params
        ).fetchone()
        total = total_row["cnt"] if total_row else 0

        offset = (page - 1) * per_page
        rows = conn.execute(
            f"""
            SELECT job_id, created_at, status, market, stock_pool,
                   total_time, risk_level, preset_id
            FROM runs {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            [*params, per_page, offset],
        ).fetchall()

        items = []
        for row in rows:
            item = dict(row)
            try:
                item["stock_pool"] = json.loads(item.get("stock_pool", "[]"))
            except (json.JSONDecodeError, TypeError):
                item["stock_pool"] = []
            items.append(item)

        return items, total

    def delete_run(self, job_id: str) -> bool:
        conn = self._conn()
        run_exists = conn.execute(
            "SELECT 1 FROM runs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if run_exists is None:
            return False
        conn.execute("DELETE FROM trade_records WHERE job_id = ?", (job_id,))
        cursor = conn.execute("DELETE FROM runs WHERE job_id = ?", (job_id,))
        conn.commit()
        self._delete_workspace_learning_files(job_id)
        return cursor.rowcount > 0

    # ---------------------------------------------------------- trade records

    def save_trade_records(
        self,
        job_id: str,
        trades: list[dict[str, Any]],
    ) -> None:
        """Persist suggested trade records from a completed run."""
        if not trades:
            return
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()
        for t in trades:
            conn.execute(
                """
                INSERT OR IGNORE INTO trade_records
                    (trade_id, job_id, symbol, direction, rationale, status, outcome_status, created_at)
                VALUES (?, ?, ?, ?, ?, 'suggested', 'pending', ?)
                """,
                (
                    t.get("trade_id") or uuid.uuid4().hex[:16],
                    job_id,
                    t.get("symbol", ""),
                    t.get("direction", "buy"),
                    t.get("rationale", ""),
                    now,
                ),
            )
        conn.commit()

    def get_recent_trade_records(self, limit: int = 20) -> list[dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT tr.*, r.market, r.created_at as run_created_at
            FROM trade_records tr
            JOIN runs r ON tr.job_id = r.job_id
            ORDER BY tr.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def build_review_recall_context(
        self,
        *,
        market: str,
        stock_pool: list[str],
        recent_n: int = 5,
    ) -> dict[str, Any]:
        """Build a bounded structured recall packet for the next review pass."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT job_id, created_at, market, stock_pool, recall_context_json
            FROM runs
            WHERE status = 'completed' AND market = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (market, recent_n),
        ).fetchall()

        recent_runs: list[dict[str, Any]] = []
        recent_symbols: list[str] = []
        recent_markets: list[str] = []
        recent_convictions: list[dict[str, Any]] = []
        top_picks: list[dict[str, Any]] = []

        target_symbols = {str(symbol).strip() for symbol in stock_pool if str(symbol).strip()}
        seen_symbols: set[str] = set()
        seen_pick_pairs: set[tuple[str, str]] = set()

        for row in rows:
            item = dict(row)
            try:
                run_symbols = json.loads(item.get("stock_pool", "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                run_symbols = []
            try:
                recall_context = json.loads(item.get("recall_context_json", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                recall_context = {}

            conviction = str(recall_context.get("conviction", "")).strip()
            run_top_picks = recall_context.get("top_picks", [])
            recent_runs.append(
                {
                    "job_id": item["job_id"],
                    "created_at": item["created_at"],
                    "market": item["market"],
                    "symbols": list(run_symbols)[:8],
                    "conviction": conviction,
                    "top_picks": list(run_top_picks)[:5] if isinstance(run_top_picks, list) else [],
                }
            )

            if item["market"] and item["market"] not in recent_markets:
                recent_markets.append(item["market"])

            if conviction:
                recent_convictions.append(
                    {
                        "job_id": item["job_id"],
                        "conviction": conviction,
                        "created_at": item["created_at"],
                    }
                )

            candidate_symbols = []
            candidate_symbols.extend(str(symbol).strip() for symbol in run_symbols if str(symbol).strip())
            if isinstance(run_top_picks, list):
                candidate_symbols.extend(
                    str(pick.get("symbol", "")).strip()
                    for pick in run_top_picks
                    if isinstance(pick, dict)
                )
            for symbol in candidate_symbols:
                if (
                    symbol
                    and symbol not in seen_symbols
                    and (not target_symbols or symbol in target_symbols or len(recent_symbols) < 6)
                ):
                    seen_symbols.add(symbol)
                    recent_symbols.append(symbol)
                if len(recent_symbols) >= 12:
                    break

            if isinstance(run_top_picks, list):
                for pick in run_top_picks:
                    if not isinstance(pick, dict):
                        continue
                    symbol = str(pick.get("symbol", "")).strip()
                    action = str(pick.get("action", "")).strip().lower()
                    rationale = str(pick.get("rationale", "")).strip()
                    pair = (symbol, action)
                    if not symbol or not action or pair in seen_pick_pairs:
                        continue
                    if target_symbols and symbol not in target_symbols and len(top_picks) >= 4:
                        continue
                    seen_pick_pairs.add(pair)
                    top_picks.append(
                        {
                            "symbol": symbol,
                            "action": action,
                            "rationale": rationale[:160],
                            "job_id": item["job_id"],
                        }
                    )
                    if len(top_picks) >= 8:
                        break

        trades = self.get_recent_trade_records(limit=20)
        pending_buy = 0
        pending_sell = 0
        for trade in trades:
            if trade.get("market") != market or trade.get("outcome_status") != "pending":
                continue
            direction = str(trade.get("direction", "")).strip().lower()
            if direction == "buy":
                pending_buy += 1
            elif direction == "sell":
                pending_sell += 1
            pair = (str(trade.get("symbol", "")).strip(), direction)
            if (
                pair[0]
                and pair not in seen_pick_pairs
                and (not target_symbols or pair[0] in target_symbols or len(top_picks) < 4)
            ):
                seen_pick_pairs.add(pair)
                top_picks.append(
                    {
                        "symbol": pair[0],
                        "action": direction,
                        "rationale": str(trade.get("rationale", "")).strip()[:160],
                        "job_id": trade.get("job_id", ""),
                    }
                )
                if len(top_picks) >= 8:
                    break

        return {
            "recent_markets": recent_markets[:3],
            "recent_symbols": recent_symbols[:12],
            "recent_convictions": recent_convictions[:5],
            "top_picks": top_picks[:8],
            "pending_trade_counts": {
                "total": pending_buy + pending_sell,
                "buy": pending_buy,
                "sell": pending_sell,
            },
            "recent_runs": recent_runs[:5],
        }

    # -------------------------------------------------------- startup context

    def get_startup_context(self, recent_n: int = 5) -> dict[str, Any]:
        """Return a bounded structured recall summary for startup display."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT job_id, created_at, status, market, stock_pool,
                   total_time, recall_context_json, selection_meta_json
            FROM runs
            WHERE status IN ('completed', 'failed')
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (recent_n,),
        ).fetchall()

        recent_runs = []
        for row in rows:
            item = dict(row)
            try:
                item["stock_pool"] = json.loads(item.get("stock_pool", "[]"))
            except (json.JSONDecodeError, TypeError):
                item["stock_pool"] = []
            try:
                item["recall_context"] = json.loads(item.pop("recall_context_json", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                item["recall_context"] = {}
            try:
                item["selection_meta"] = json.loads(item.pop("selection_meta_json", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                item["selection_meta"] = {}
            recent_runs.append(item)

        trades = self.get_recent_trade_records(limit=10)

        # Build a lightweight recall_summary bounded to recent runs
        recall_summary: dict[str, Any] = {
            "run_count": len(recent_runs),
            "markets": sorted({r["market"] for r in recent_runs if r.get("market")}),
            "last_run_at": recent_runs[0]["created_at"] if recent_runs else None,
            "pending_trades": sum(1 for t in trades if t.get("outcome_status") == "pending"),
        }

        return {
            "recent_runs": recent_runs,
            "suggested_trades": trades,
            "recall_summary": recall_summary,
        }

    def _delete_workspace_learning_files(self, job_id: str) -> None:
        if not _WORKSPACE_LEARNING_DIR.exists():
            return
        for path in _WORKSPACE_LEARNING_DIR.glob(f"{job_id}_*.json"):
            try:
                path.unlink()
            except OSError:
                continue


history_store = RunHistoryStore()

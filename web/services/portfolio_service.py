"""Service layer for user holdings and watchlist management."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from web.config import APP_DB_PATH
from web.services import data_service

WATCHLIST_INIT_SQL = """
CREATE TABLE IF NOT EXISTS user_watchlist (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    market TEXT NOT NULL,
    priority TEXT DEFAULT 'normal',
    notes TEXT DEFAULT '',
    created_at TEXT,
    updated_at TEXT
);
"""

HOLDINGS_INIT_SQL = """
CREATE TABLE IF NOT EXISTS user_holdings (
    holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL DEFAULT '默认账户',
    symbol TEXT NOT NULL,
    name TEXT,
    market TEXT NOT NULL,
    quantity REAL NOT NULL,
    cost_basis REAL,
    notes TEXT DEFAULT '',
    created_at TEXT,
    updated_at TEXT,
    UNIQUE(account_name, symbol)
);
"""


def _connect_app_db() -> sqlite3.Connection:
    db_path = Path(APP_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    _ensure_holdings_schema(conn)
    conn.executescript(WATCHLIST_INIT_SQL)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]


def _ensure_holdings_schema(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "user_holdings"):
        conn.executescript(HOLDINGS_INIT_SQL)
        return

    columns = _table_columns(conn, "user_holdings")
    if {"holding_id", "account_name"}.issubset(columns):
        conn.executescript(HOLDINGS_INIT_SQL)
        return

    conn.execute("ALTER TABLE user_holdings RENAME TO user_holdings_legacy")
    conn.executescript(HOLDINGS_INIT_SQL)

    legacy_columns = _table_columns(conn, "user_holdings_legacy")
    if {"symbol", "market", "quantity"}.issubset(legacy_columns):
        conn.execute(
            """
            INSERT INTO user_holdings (
                account_name, symbol, name, market, quantity, cost_basis, notes, created_at, updated_at
            )
            SELECT
                '默认账户',
                symbol,
                name,
                market,
                quantity,
                cost_basis,
                COALESCE(notes, ''),
                created_at,
                updated_at
            FROM user_holdings_legacy
            """
        )
    conn.execute("DROP TABLE user_holdings_legacy")
    conn.commit()


def _infer_market(symbol: str) -> str:
    normalized = symbol.upper()
    if "." in normalized and normalized.rsplit(".", 1)[-1] in {"SH", "SZ", "BJ"}:
        return "CN"
    return "US"


def _resolve_symbol_context(symbol: str, name: str | None, market: str | None) -> tuple[str, str | None, str]:
    normalized_symbol = symbol.strip().upper()
    resolved_name = name.strip() if isinstance(name, str) and name.strip() else None
    resolved_market = (market or "").strip().upper() or _infer_market(normalized_symbol)

    try:
        detail = data_service.get_stock_detail(normalized_symbol)
    except Exception:
        detail = None

    if detail:
        resolved_name = resolved_name or detail.get("name")
        resolved_market = str(detail.get("market") or resolved_market).upper()

    return normalized_symbol, resolved_name, resolved_market


def _holding_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "holding_id": int(row["holding_id"]),
        "account_name": str(row["account_name"] or "默认账户"),
        "symbol": str(row["symbol"]),
        "name": row["name"],
        "market": str(row["market"] or "CN"),
        "quantity": float(row["quantity"] or 0.0),
        "cost_basis": float(row["cost_basis"]) if row["cost_basis"] is not None else None,
        "notes": str(row["notes"] or ""),
        "created_at": str(row["created_at"] or ""),
        "updated_at": str(row["updated_at"] or ""),
    }


def _watchlist_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "symbol": str(row["symbol"]),
        "name": row["name"],
        "market": str(row["market"] or "CN"),
        "priority": str(row["priority"] or "normal"),
        "notes": str(row["notes"] or ""),
        "created_at": str(row["created_at"] or ""),
        "updated_at": str(row["updated_at"] or ""),
    }


def get_portfolio_state() -> dict[str, Any]:
    conn = _connect_app_db()
    try:
        holdings_rows = conn.execute(
            """
            SELECT holding_id, account_name, symbol, name, market, quantity, cost_basis, notes, created_at, updated_at
            FROM user_holdings
            ORDER BY account_name ASC, updated_at DESC, symbol ASC
            """
        ).fetchall()
        watchlist_rows = conn.execute(
            """
            SELECT symbol, name, market, priority, notes, created_at, updated_at
            FROM user_watchlist
            ORDER BY updated_at DESC, symbol ASC
            """
        ).fetchall()
    finally:
        conn.close()

    holdings = [_holding_from_row(row) for row in holdings_rows]
    watchlist = [_watchlist_from_row(row) for row in watchlist_rows]
    accounts = sorted({item["account_name"] for item in holdings})
    holdings_by_account = {
        account_name: sum(1 for item in holdings if item["account_name"] == account_name)
        for account_name in accounts
    }

    return {
        "holdings": holdings,
        "watchlist": watchlist,
        "summary": {
            "account_count": len(accounts),
            "accounts": accounts,
            "holdings_count": len(holdings),
            "watchlist_count": len(watchlist),
            "holdings_by_account": holdings_by_account,
            "holding_symbols": [item["symbol"] for item in holdings],
            "watchlist_symbols": [item["symbol"] for item in watchlist],
        },
    }


def upsert_holding(payload: dict[str, Any]) -> dict[str, Any]:
    holding_id = payload.get("holding_id")
    account_name = str(payload.get("account_name") or "默认账户").strip() or "默认账户"
    symbol, name, market = _resolve_symbol_context(
        str(payload.get("symbol", "")),
        payload.get("name"),
        payload.get("market"),
    )
    quantity = float(payload.get("quantity", 0.0))
    cost_basis = payload.get("cost_basis")
    notes = str(payload.get("notes", "") or "")
    timestamp = datetime.now().isoformat(timespec="seconds")

    conn = _connect_app_db()
    try:
        if holding_id is not None:
            existing = conn.execute(
                """
                SELECT created_at
                FROM user_holdings
                WHERE holding_id = ?
                LIMIT 1
                """,
                (holding_id,),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    """
                    UPDATE user_holdings
                    SET account_name = ?, symbol = ?, name = ?, market = ?, quantity = ?, cost_basis = ?, notes = ?, updated_at = ?
                    WHERE holding_id = ?
                    """,
                    (account_name, symbol, name, market, quantity, cost_basis, notes, timestamp, holding_id),
                )
                conn.commit()
                return {
                    "ok": True,
                    "message": f"{account_name} · {symbol} 持仓已更新",
                    "state": get_portfolio_state(),
                }

        conn.execute(
            """
            INSERT INTO user_holdings (account_name, symbol, name, market, quantity, cost_basis, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_name, symbol) DO UPDATE SET
                name = excluded.name,
                market = excluded.market,
                quantity = excluded.quantity,
                cost_basis = excluded.cost_basis,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (account_name, symbol, name, market, quantity, cost_basis, notes, timestamp, timestamp),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "ok": True,
        "message": f"{account_name} · {symbol} 持仓已保存",
        "state": get_portfolio_state(),
    }


def delete_holding(holding_id: int) -> dict[str, Any]:
    conn = _connect_app_db()
    try:
        row = conn.execute(
            """
            SELECT account_name, symbol
            FROM user_holdings
            WHERE holding_id = ?
            LIMIT 1
            """,
            (holding_id,),
        ).fetchone()
        cursor = conn.execute("DELETE FROM user_holdings WHERE holding_id = ?", (holding_id,))
        conn.commit()
        deleted = cursor.rowcount
    finally:
        conn.close()

    account_name = str(row["account_name"]) if row is not None else "账户"
    symbol = str(row["symbol"]) if row is not None else ""
    return {
        "ok": deleted > 0,
        "message": f"{account_name} · {symbol} 持仓已删除" if deleted > 0 else "未找到对应持仓",
        "state": get_portfolio_state(),
    }


def upsert_watchlist(payload: dict[str, Any]) -> dict[str, Any]:
    symbol, name, market = _resolve_symbol_context(
        str(payload.get("symbol", "")),
        payload.get("name"),
        payload.get("market"),
    )
    priority = str(payload.get("priority", "normal") or "normal")
    notes = str(payload.get("notes", "") or "")
    timestamp = datetime.now().isoformat(timespec="seconds")

    conn = _connect_app_db()
    try:
        conn.execute(
            """
            INSERT INTO user_watchlist (symbol, name, market, priority, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                name = excluded.name,
                market = excluded.market,
                priority = excluded.priority,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (symbol, name, market, priority, notes, timestamp, timestamp),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "ok": True,
        "message": f"{symbol} 已加入自选池",
        "state": get_portfolio_state(),
    }


def delete_watchlist(symbol: str) -> dict[str, Any]:
    normalized_symbol = symbol.strip().upper()
    conn = _connect_app_db()
    try:
        cursor = conn.execute("DELETE FROM user_watchlist WHERE symbol = ?", (normalized_symbol,))
        conn.commit()
        deleted = cursor.rowcount
    finally:
        conn.close()

    return {
        "ok": deleted > 0,
        "message": f"{normalized_symbol} 已从自选池移除" if deleted > 0 else "未找到对应自选池标的",
        "state": get_portfolio_state(),
    }

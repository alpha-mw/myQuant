"""Service layer wrapping local market data for the web API."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from web.config import APP_DB_PATH, PROJECT_ROOT, RESULTS_DIR, STOCK_DB_PATH

logger = logging.getLogger(__name__)

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS stock_list (
    ts_code TEXT PRIMARY KEY,
    name TEXT,
    industry TEXT,
    market TEXT,
    list_date TEXT,
    is_hs300 INTEGER DEFAULT 0,
    is_zz500 INTEGER DEFAULT 0,
    is_zz1000 INTEGER DEFAULT 0,
    last_update TEXT
);
CREATE TABLE IF NOT EXISTS daily_data (
    ts_code TEXT,
    trade_date TEXT,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL,
    PRIMARY KEY (ts_code, trade_date)
);
CREATE TABLE IF NOT EXISTS factor_data (
    ts_code TEXT,
    trade_date TEXT,
    factor_name TEXT,
    factor_value REAL,
    PRIMARY KEY (ts_code, trade_date, factor_name)
);
CREATE TABLE IF NOT EXISTS download_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_code TEXT, start_date TEXT, end_date TEXT,
    records_count INTEGER, status TEXT, message TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS stock_profiles (
    ts_code TEXT PRIMARY KEY,
    market TEXT,
    sector TEXT,
    industry TEXT,
    exchange TEXT,
    summary TEXT,
    description TEXT,
    products_json TEXT,
    business_lines_json TEXT,
    website TEXT,
    city TEXT,
    region TEXT,
    country TEXT,
    employees INTEGER,
    source TEXT,
    fetched_at TEXT,
    updated_at TEXT,
    raw_json TEXT
);
CREATE TABLE IF NOT EXISTS fundamental_snapshots (
    ts_code TEXT PRIMARY KEY,
    market TEXT,
    report_period TEXT,
    currency TEXT,
    revenue REAL,
    net_income REAL,
    gross_margin REAL,
    operating_margin REAL,
    roe REAL,
    roa REAL,
    debt_to_asset REAL,
    pe_ttm REAL,
    pb REAL,
    ps REAL,
    market_cap REAL,
    total_assets REAL,
    total_liabilities REAL,
    shareholder_equity REAL,
    operating_cashflow REAL,
    free_cashflow REAL,
    source TEXT,
    fetched_at TEXT,
    updated_at TEXT,
    raw_json TEXT
);
CREATE TABLE IF NOT EXISTS fundamental_series (
    ts_code TEXT,
    metric_name TEXT,
    label TEXT,
    period TEXT,
    value REAL,
    source TEXT,
    fetched_at TEXT,
    PRIMARY KEY (ts_code, metric_name, period)
);
CREATE TABLE IF NOT EXISTS peer_relationships (
    ts_code TEXT,
    peer_ts_code TEXT,
    relation_type TEXT,
    similarity_score REAL,
    reason TEXT,
    source TEXT,
    updated_at TEXT,
    PRIMARY KEY (ts_code, peer_ts_code, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_daily_ts_code ON daily_data(ts_code);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_data(trade_date);
CREATE INDEX IF NOT EXISTS idx_profile_market ON stock_profiles(market);
CREATE INDEX IF NOT EXISTS idx_snapshot_market ON fundamental_snapshots(market);
CREATE INDEX IF NOT EXISTS idx_series_ts_code ON fundamental_series(ts_code);
CREATE INDEX IF NOT EXISTS idx_peer_ts_code ON peer_relationships(ts_code);
"""

_STOCK_UPSERT_SQL = """
INSERT INTO stock_list (
    ts_code, name, industry, market, list_date,
    is_hs300, is_zz500, is_zz1000, last_update
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(ts_code) DO UPDATE SET
    name = COALESCE(excluded.name, stock_list.name),
    industry = COALESCE(excluded.industry, stock_list.industry),
    market = COALESCE(excluded.market, stock_list.market),
    list_date = COALESCE(excluded.list_date, stock_list.list_date),
    is_hs300 = excluded.is_hs300,
    is_zz500 = excluded.is_zz500,
    is_zz1000 = excluded.is_zz1000,
    last_update = excluded.last_update
"""

_CURATED_STOCK_METADATA: dict[str, dict[str, str]] = {
    "000001.SZ": {"name": "平安银行", "industry": "银行"},
    "000333.SZ": {"name": "美的集团", "industry": "家电"},
    "000858.SZ": {"name": "五粮液", "industry": "白酒"},
    "002475.SZ": {"name": "立讯精密", "industry": "电子"},
    "002594.SZ": {"name": "比亚迪", "industry": "新能源汽车"},
    "300750.SZ": {"name": "宁德时代", "industry": "动力电池"},
    "600018.SH": {"name": "上港集团", "industry": "港口航运"},
    "600036.SH": {"name": "招商银行", "industry": "银行"},
    "600276.SH": {"name": "恒瑞医药", "industry": "创新药"},
    "600519.SH": {"name": "贵州茅台", "industry": "白酒"},
    "600803.SH": {"name": "新奥股份", "industry": "燃气"},
    "600900.SH": {"name": "长江电力", "industry": "水电"},
    "601318.SH": {"name": "中国平安", "industry": "保险"},
    "601618.SH": {"name": "中国中冶", "industry": "建筑工程"},
    "601633.SH": {"name": "长城汽车", "industry": "乘用车"},
    "601669.SH": {"name": "中国电建", "industry": "建筑工程"},
    "601800.SH": {"name": "中国交建", "industry": "建筑工程"},
    "601838.SH": {"name": "成都银行", "industry": "银行"},
    "AAPL": {"name": "Apple"},
    "AMZN": {"name": "Amazon"},
    "GOOGL": {"name": "Alphabet"},
    "META": {"name": "Meta"},
    "MSFT": {"name": "Microsoft"},
    "NVDA": {"name": "NVIDIA"},
    "TSLA": {"name": "Tesla"},
}

_sync_lock = threading.Lock()
_SYNC_IN_PROGRESS = False
_LAST_SYNC_SIGNATURE: tuple[int, int, float] | None = None
_CN_METADATA_SYNCED = False
_missing_sync_lock = threading.Lock()
_MISSING_SYNC_IN_PROGRESS = False
_LAST_MISSING_SYNC_AT = 0.0
_MISSING_SYNC_COOLDOWN_SECONDS = 15 * 60
_DEFAULT_MISSING_DOWNLOAD_START = "20200101"


def _configure_connection(conn: sqlite3.Connection, label: str) -> sqlite3.Connection:
    try:
        conn.execute("PRAGMA busy_timeout=60000")
    except sqlite3.DatabaseError:
        logger.warning("Failed to set busy_timeout for %s", label)

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        logger.warning("SQLite WAL unavailable for %s, falling back to default journal mode", label)

    return conn


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(STOCK_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(STOCK_DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    _configure_connection(conn, STOCK_DB_PATH)
    try:
        conn.executescript(_INIT_SQL)
    except sqlite3.DatabaseError:
        logger.warning("SQLite schema init skipped for %s due to database errors", STOCK_DB_PATH)
    return conn


def _data_directories() -> list[Path]:
    candidates = [
        PROJECT_ROOT / "data" / "cn_market_full" / "hs300",
        PROJECT_ROOT / "data" / "cn_market_full" / "zz500",
        PROJECT_ROOT / "data" / "cn_market_full" / "zz1000",
        PROJECT_ROOT / "data" / "us_market" / "large_cap",
        PROJECT_ROOT / "data" / "us_market" / "mid_cap",
        PROJECT_ROOT / "data" / "us_market" / "small_cap",
        PROJECT_ROOT / "data" / "us_market_full" / "large_cap",
        PROJECT_ROOT / "data" / "us_market_full" / "mid_cap",
        PROJECT_ROOT / "data" / "us_market_full" / "small_cap",
    ]
    return [path for path in candidates if path.exists()]


def _data_signature() -> tuple[int, int, float]:
    directories = _data_directories()
    file_count = 0
    latest_mtime = 0.0
    for directory in directories:
        for csv_file in directory.glob("*.csv"):
            file_count += 1
            try:
                latest_mtime = max(latest_mtime, csv_file.stat().st_mtime)
            except OSError:
                continue
    return len(directories), file_count, latest_mtime


def _ensure_local_data_indexed(force: bool = False) -> None:
    global _SYNC_IN_PROGRESS, _LAST_SYNC_SIGNATURE

    signature = _data_signature()
    if signature[1] == 0:
        return
    if not force and _LAST_SYNC_SIGNATURE == signature:
        return

    with _sync_lock:
        # Re-check after acquiring lock
        if _SYNC_IN_PROGRESS:
            return
        if not force and _LAST_SYNC_SIGNATURE == signature:
            return

        conn = _connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stock_list")
        stock_count = int(cursor.fetchone()[0])
        cursor.execute("SELECT COUNT(*) FROM daily_data")
        record_count = int(cursor.fetchone()[0])
        conn.close()

        db_mtime = 0.0
        if os.path.exists(STOCK_DB_PATH):
            try:
                db_mtime = os.path.getmtime(STOCK_DB_PATH)
            except OSError:
                db_mtime = 0.0

        should_sync = force or stock_count == 0 or record_count == 0 or db_mtime < signature[2]
        if not should_sync:
            _LAST_SYNC_SIGNATURE = signature
            _ensure_cn_metadata_populated()
            return

        _SYNC_IN_PROGRESS = True

    try:
        logger.info("Starting CSV data sync (signature=%s)", signature)
        import_csv_data()
        with _sync_lock:
            _LAST_SYNC_SIGNATURE = signature
        logger.info("CSV data sync complete")
    finally:
        with _sync_lock:
            _SYNC_IN_PROGRESS = False

    _ensure_cn_metadata_populated(force=True)


def _count_missing_stock_rows(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM stock_list s
        WHERE NOT EXISTS (
            SELECT 1
            FROM daily_data d
            WHERE d.ts_code = s.ts_code
        )
        """
    ).fetchone()
    return int(row[0]) if row else 0


def _default_missing_download_start(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT MIN(trade_date) FROM daily_data").fetchone()
    trade_date = str(row[0]).strip() if row and row[0] else ""
    if trade_date:
        return trade_date
    return _DEFAULT_MISSING_DOWNLOAD_START


def _ensure_missing_stock_data_downloaded(force: bool = False) -> None:
    """
    若 stock_list 中存在尚未下载任何日线的股票，则在后台触发补齐。

    这里采用异步触发，避免首次打开列表时被全市场补数据阻塞。
    """
    global _MISSING_SYNC_IN_PROGRESS, _LAST_MISSING_SYNC_AT

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM stock_list")
    stock_count = int(cursor.fetchone()[0])
    missing_count = _count_missing_stock_rows(conn)
    start_date = _default_missing_download_start(conn)
    conn.close()

    if stock_count == 0 or missing_count == 0:
        return

    now = time.time()
    with _missing_sync_lock:
        if _MISSING_SYNC_IN_PROGRESS:
            return
        if not force and now - _LAST_MISSING_SYNC_AT < _MISSING_SYNC_COOLDOWN_SECONDS:
            return
        _MISSING_SYNC_IN_PROGRESS = True
        _LAST_MISSING_SYNC_AT = now

    def runner() -> None:
        global _MISSING_SYNC_IN_PROGRESS

        try:
            from quant_investor.stock_database import StockDatabase

            logger.info(
                "Detected %s stocks without daily_data; start background backfill from %s",
                missing_count,
                start_date,
            )
            db = StockDatabase(db_path=STOCK_DB_PATH, verbose=False, init_universe=False)
            progress = db.download_missing_stocks(
                start_date=start_date,
                max_workers=3,
            )
            logger.info(
                "Missing-stock backfill finished: success=%s failed=%s",
                progress.completed_stocks,
                len(progress.failed_stocks),
            )
        except Exception:
            logger.exception("Failed to backfill missing stocks from stock_list")
        finally:
            with _missing_sync_lock:
                _MISSING_SYNC_IN_PROGRESS = False

    threading.Thread(target=runner, name="missing-stock-backfill", daemon=True).start()


def _load_json(path: os.PathLike[str] | str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _blank(value: Any) -> bool:
    return not str(value or "").strip()


def _normalize_symbol(value: str) -> str:
    return value.strip().upper().replace("-", ".")


def _update_stock_metadata(updates: dict[str, dict[str, Any]]) -> int:
    if not updates:
        return 0

    conn = _connect()
    cursor = conn.cursor()
    updated = 0
    last_update = datetime.now().isoformat(timespec="seconds")

    for ts_code, meta in updates.items():
        name = str(meta.get("name") or "").strip()
        industry = str(meta.get("industry") or "").strip()
        market = str(meta.get("market") or "").strip() or _guess_market(ts_code)
        list_date = str(meta.get("list_date") or "").strip()

        cursor.execute(
            """
            UPDATE stock_list
            SET
                name = COALESCE(NULLIF(?, ''), name),
                industry = COALESCE(NULLIF(?, ''), industry),
                market = COALESCE(NULLIF(?, ''), market),
                list_date = COALESCE(NULLIF(?, ''), list_date),
                last_update = ?
            WHERE ts_code = ?
            """,
            (name, industry, market, list_date, last_update, ts_code),
        )
        if cursor.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()
    return updated


def _fetch_cn_metadata(codes: list[str]) -> dict[str, dict[str, Any]]:
    if not codes:
        return {}

    try:
        import tushare as ts
        from quant_investor.config import config
        from quant_investor.credential_utils import create_tushare_pro
    except Exception:
        return {}

    pro = create_tushare_pro(
        ts,
        getattr(config, "TUSHARE_TOKEN", ""),
        os.environ.get("TUSHARE_URL", "http://lianghua.nanyangqiankun.top"),
    )
    if pro is None:
        return {}

    updates: dict[str, dict[str, Any]] = {}
    for batch in _chunked(sorted(set(codes)), 100):
        try:
            df = pro.stock_basic(
                ts_code=",".join(batch),
                fields="ts_code,name,industry,market,list_date",
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue
        for row in df.to_dict("records"):
            ts_code = str(row.get("ts_code", "")).strip().upper()
            if not ts_code:
                continue
            updates[ts_code] = {
                "name": str(row.get("name") or "").strip(),
                "industry": str(row.get("industry") or "").strip(),
                "market": "CN",
                "list_date": str(row.get("list_date") or "").strip(),
            }
    return updates


def _fetch_us_metadata(codes: list[str]) -> dict[str, dict[str, Any]]:
    if not codes:
        return {}

    import requests

    def fetch_one(symbol: str) -> tuple[str, dict[str, Any] | None]:
        try:
            response = requests.get(
                "https://query1.finance.yahoo.com/v1/finance/search",
                params={"q": symbol, "quotesCount": 8, "newsCount": 0},
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return symbol, None

        quotes = payload.get("quotes", [])
        normalized = _normalize_symbol(symbol)
        candidates: list[dict[str, Any]] = [item for item in quotes if isinstance(item, dict)]

        def score(item: dict[str, Any]) -> tuple[int, int]:
            quote_symbol = _normalize_symbol(str(item.get("symbol", "")))
            is_equity = 1 if str(item.get("quoteType", "")).upper() == "EQUITY" else 0
            exact = 1 if quote_symbol == normalized else 0
            return exact, is_equity

        candidates.sort(key=score, reverse=True)
        for item in candidates:
            quote_symbol = _normalize_symbol(str(item.get("symbol", "")))
            if quote_symbol != normalized:
                continue
            name = str(item.get("shortname") or item.get("longname") or "").strip()
            if not name:
                continue
            return symbol, {"name": name, "market": "US"}

        return symbol, None

    updates: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_one, code) for code in sorted(set(codes))]
        for future in as_completed(futures):
            try:
                symbol, meta = future.result()
            except Exception:
                continue
            if meta:
                updates[symbol] = meta

    return updates


def _ensure_cn_metadata_populated(force: bool = False) -> None:
    global _CN_METADATA_SYNCED

    if _CN_METADATA_SYNCED and not force:
        return

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ts_code
        FROM stock_list
        WHERE market = 'CN'
          AND (
            name IS NULL OR TRIM(name) = ''
            OR industry IS NULL OR TRIM(industry) = ''
          )
        """
    )
    codes = [str(row["ts_code"]).upper() for row in cursor.fetchall()]
    conn.close()

    if not codes:
        _CN_METADATA_SYNCED = True
        return

    updates = _fetch_cn_metadata(codes)
    _update_stock_metadata(updates)

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM stock_list
        WHERE market = 'CN'
          AND (
            name IS NULL OR TRIM(name) = ''
            OR industry IS NULL OR TRIM(industry) = ''
          )
        """
    )
    remaining = int(cursor.fetchone()[0])
    conn.close()
    _CN_METADATA_SYNCED = remaining == 0


def _ensure_metadata_for_codes(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cn_codes = sorted(
        {
            item["ts_code"]
            for item in items
            if item.get("market") == "CN"
            and (_blank(item.get("name")) or _blank(item.get("industry")))
        }
    )
    us_codes = sorted(
        {
            item["ts_code"]
            for item in items
            if item.get("market") == "US" and _blank(item.get("name"))
        }
    )

    updates: dict[str, dict[str, Any]] = {}
    if cn_codes:
        updates.update(_fetch_cn_metadata(cn_codes))
    if us_codes:
        updates.update(_fetch_us_metadata(us_codes))

    _update_stock_metadata(updates)
    return updates


@lru_cache(maxsize=1)
def _local_metadata_cache() -> dict[str, dict[str, str]]:
    metadata = {code: dict(values) for code, values in _CURATED_STOCK_METADATA.items()}
    recommendation_globs = [
        RESULTS_DIR / "cn_analysis" / "最新交易建议_*.json",
        RESULTS_DIR / "cn_analysis_full" / "交易建议数据_*.json",
    ]

    for pattern in recommendation_globs:
        for path in pattern.parent.glob(pattern.name):
            payload = _load_json(path)
            if not isinstance(payload, list):
                continue
            for item in payload:
                if not isinstance(item, dict):
                    continue
                ts_code = str(item.get("ts_code", "")).strip().upper()
                if not ts_code:
                    continue
                entry = metadata.setdefault(ts_code, {})
                if item.get("name"):
                    entry.setdefault("name", str(item["name"]))
                if item.get("industry"):
                    entry.setdefault("industry", str(item["industry"]))
                if item.get("category"):
                    entry.setdefault("category", str(item["category"]))

    return metadata


def _guess_market(ts_code: str) -> str:
    code = ts_code.upper()
    if code.endswith(".SH") or code.endswith(".SZ") or code.endswith(".BJ"):
        return "CN"
    return "US"


def _normalize_symbol_for_market(value: str, market: str) -> str:
    normalized = _normalize_symbol(value)
    normalized_market = str(market or "CN").upper()

    if normalized_market == "CN":
        exact_match = re.fullmatch(r"(\d{6})\.(SH|SZ|BJ)", normalized)
        if exact_match:
            return f"{exact_match.group(1)}.{exact_match.group(2)}"

        prefixed_match = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", normalized)
        if prefixed_match:
            return f"{prefixed_match.group(2)}.{prefixed_match.group(1)}"

        raw_digits = re.sub(r"\D", "", normalized)
        if len(raw_digits) == 6:
            if raw_digits.startswith(("6", "9", "5")):
                suffix = "SH"
            elif raw_digits.startswith(("4", "8")):
                suffix = "BJ"
            else:
                suffix = "SZ"
            return f"{raw_digits}.{suffix}"

    return normalized.replace(" ", "")


def _upsert_stock_list_entries(symbols: list[str], market: str, metadata: dict[str, dict[str, Any]]) -> None:
    if not symbols:
        return

    conn = _connect()
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for symbol in symbols:
        meta = metadata.get(symbol, {})
        rows.append(
            (
                symbol,
                str(meta.get("name") or symbol).strip(),
                str(meta.get("industry") or "").strip() or None,
                market,
                str(meta.get("list_date") or "").strip() or None,
                0,
                0,
                0,
                now,
            )
        )
    conn.executemany(_STOCK_UPSERT_SQL, rows)
    conn.commit()
    conn.close()


def ensure_symbols_available(symbols: list[str], market: str) -> list[str]:
    """
    确保给定股票已经写入 stock_list，且具备可分析的本地日线数据。

    适用于分析中心手工输入库外股票代码的场景。
    """
    normalized_market = str(market or "CN").upper()
    normalized_symbols = list(
        dict.fromkeys(
            _normalize_symbol_for_market(symbol, normalized_market)
            for symbol in symbols
            if str(symbol or "").strip()
        )
    )
    if not normalized_symbols:
        return []

    conn = _connect()
    start_date = _default_missing_download_start(conn)
    placeholders = ",".join("?" for _ in normalized_symbols)
    rows_with_data = conn.execute(
        f"""
        SELECT DISTINCT ts_code
        FROM daily_data
        WHERE ts_code IN ({placeholders})
        """,
        normalized_symbols,
    ).fetchall()
    conn.close()

    existing_with_data = {str(row["ts_code"]).upper() for row in rows_with_data}
    missing_symbols = [symbol for symbol in normalized_symbols if symbol not in existing_with_data]
    if not missing_symbols:
        return normalized_symbols

    metadata = (
        _fetch_cn_metadata(missing_symbols)
        if normalized_market == "CN"
        else _fetch_us_metadata(missing_symbols)
    )
    _upsert_stock_list_entries(missing_symbols, normalized_market, metadata)

    from quant_investor.stock_database import DownloadTask, StockDatabase

    db = StockDatabase(db_path=STOCK_DB_PATH, verbose=False, init_universe=False)
    end_date = datetime.now().strftime("%Y%m%d")
    failed_symbols: list[str] = []

    for symbol in missing_symbols:
        meta = metadata.get(symbol, {})
        success = db.download_task(
            DownloadTask(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date,
                reason="analysis_autoload",
                market=normalized_market,
                list_date=str(meta.get("list_date") or "").strip() or None,
            )
        )
        if not success:
            failed_symbols.append(symbol)

    conn = _connect()
    verified_rows = conn.execute(
        f"""
        SELECT DISTINCT ts_code
        FROM daily_data
        WHERE ts_code IN ({placeholders})
        """,
        normalized_symbols,
    ).fetchall()
    conn.close()
    verified_symbols = {str(row["ts_code"]).upper() for row in verified_rows}
    unresolved = [symbol for symbol in normalized_symbols if symbol not in verified_symbols]
    if unresolved:
        symbol_text = "、".join(unresolved[:5])
        raise ValueError(f"以下股票暂时无法自动下载入库：{symbol_text}")

    return normalized_symbols


_RESEARCH_STALE_DAYS = 7


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _safe_json_loads(value: Any) -> Any:
    if value in {None, ""}:
        return None
    try:
        return json.loads(str(value))
    except Exception:
        return None


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for parser in (
        lambda item: datetime.fromisoformat(item),
        lambda item: datetime.strptime(item, "%Y-%m-%d %H:%M:%S"),
        lambda item: datetime.strptime(item, "%Y%m%d"),
    ):
        try:
            return parser(value)
        except Exception:
            continue
    return None


def _is_stale(value: Optional[str], days: int = _RESEARCH_STALE_DAYS) -> bool:
    parsed = _parse_datetime(value)
    if parsed is None:
        return True
    return parsed < datetime.now() - timedelta(days=days)


def _ratio_from_percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) > 1:
        return number / 100.0
    return number


def _clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _split_keywords(value: Any, limit: int = 6) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    parts = [
        item.strip(" ;；,，。")
        for item in re.split(r"[；;、\n]|(?<=[。.!?])", text)
        if item.strip(" ;；,，。")
    ]
    deduped: list[str] = []
    for item in parts:
        if item not in deduped:
            deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _coalesce_numeric(*values: Any) -> Optional[float]:
    for value in values:
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _snapshot_metric(frame: Optional[pd.DataFrame], *keys: str) -> Optional[float]:
    if frame is None or frame.empty:
        return None
    first = frame.iloc[0].to_dict()
    return _coalesce_numeric(*(first.get(key) for key in keys))


def _profile_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["products"] = _safe_json_loads(payload.pop("products_json", None)) or []
    payload["business_lines"] = _safe_json_loads(payload.pop("business_lines_json", None)) or []
    payload["raw"] = _safe_json_loads(payload.pop("raw_json", None))
    return payload


def _snapshot_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["raw"] = _safe_json_loads(payload.pop("raw_json", None))
    return payload


def _cached_profile(ts_code: str) -> dict[str, Any] | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM stock_profiles WHERE ts_code = ?", (ts_code.upper(),)).fetchone()
    conn.close()
    return _profile_from_row(row)


def _cached_snapshot(ts_code: str) -> dict[str, Any] | None:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM fundamental_snapshots WHERE ts_code = ?",
        (ts_code.upper(),),
    ).fetchone()
    conn.close()
    return _snapshot_from_row(row)


def _cached_series(ts_code: str) -> list[dict[str, Any]]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT metric_name, label, period, value, source, fetched_at
        FROM fundamental_series
        WHERE ts_code = ?
        ORDER BY period DESC, metric_name ASC
        """,
        (ts_code.upper(),),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _store_profile(ts_code: str, market: str, payload: dict[str, Any]) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO stock_profiles (
            ts_code, market, sector, industry, exchange, summary, description,
            products_json, business_lines_json, website, city, region, country,
            employees, source, fetched_at, updated_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ts_code) DO UPDATE SET
            market = excluded.market,
            sector = COALESCE(excluded.sector, stock_profiles.sector),
            industry = COALESCE(excluded.industry, stock_profiles.industry),
            exchange = COALESCE(excluded.exchange, stock_profiles.exchange),
            summary = COALESCE(excluded.summary, stock_profiles.summary),
            description = COALESCE(excluded.description, stock_profiles.description),
            products_json = COALESCE(excluded.products_json, stock_profiles.products_json),
            business_lines_json = COALESCE(excluded.business_lines_json, stock_profiles.business_lines_json),
            website = COALESCE(excluded.website, stock_profiles.website),
            city = COALESCE(excluded.city, stock_profiles.city),
            region = COALESCE(excluded.region, stock_profiles.region),
            country = COALESCE(excluded.country, stock_profiles.country),
            employees = COALESCE(excluded.employees, stock_profiles.employees),
            source = COALESCE(excluded.source, stock_profiles.source),
            fetched_at = COALESCE(excluded.fetched_at, stock_profiles.fetched_at),
            updated_at = excluded.updated_at,
            raw_json = COALESCE(excluded.raw_json, stock_profiles.raw_json)
        """,
        (
            ts_code.upper(),
            market,
            payload.get("sector"),
            payload.get("industry"),
            payload.get("exchange"),
            payload.get("summary"),
            payload.get("description"),
            _safe_json_dumps(payload.get("products", [])),
            _safe_json_dumps(payload.get("business_lines", [])),
            payload.get("website"),
            payload.get("city"),
            payload.get("region"),
            payload.get("country"),
            payload.get("employees"),
            payload.get("source"),
            payload.get("fetched_at"),
            _now_iso(),
            _safe_json_dumps(payload.get("raw")),
        ),
    )
    conn.commit()
    conn.close()


def _store_snapshot(ts_code: str, market: str, payload: dict[str, Any]) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO fundamental_snapshots (
            ts_code, market, report_period, currency, revenue, net_income, gross_margin,
            operating_margin, roe, roa, debt_to_asset, pe_ttm, pb, ps, market_cap,
            total_assets, total_liabilities, shareholder_equity, operating_cashflow,
            free_cashflow, source, fetched_at, updated_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ts_code) DO UPDATE SET
            market = excluded.market,
            report_period = COALESCE(excluded.report_period, fundamental_snapshots.report_period),
            currency = COALESCE(excluded.currency, fundamental_snapshots.currency),
            revenue = COALESCE(excluded.revenue, fundamental_snapshots.revenue),
            net_income = COALESCE(excluded.net_income, fundamental_snapshots.net_income),
            gross_margin = COALESCE(excluded.gross_margin, fundamental_snapshots.gross_margin),
            operating_margin = COALESCE(excluded.operating_margin, fundamental_snapshots.operating_margin),
            roe = COALESCE(excluded.roe, fundamental_snapshots.roe),
            roa = COALESCE(excluded.roa, fundamental_snapshots.roa),
            debt_to_asset = COALESCE(excluded.debt_to_asset, fundamental_snapshots.debt_to_asset),
            pe_ttm = COALESCE(excluded.pe_ttm, fundamental_snapshots.pe_ttm),
            pb = COALESCE(excluded.pb, fundamental_snapshots.pb),
            ps = COALESCE(excluded.ps, fundamental_snapshots.ps),
            market_cap = COALESCE(excluded.market_cap, fundamental_snapshots.market_cap),
            total_assets = COALESCE(excluded.total_assets, fundamental_snapshots.total_assets),
            total_liabilities = COALESCE(excluded.total_liabilities, fundamental_snapshots.total_liabilities),
            shareholder_equity = COALESCE(excluded.shareholder_equity, fundamental_snapshots.shareholder_equity),
            operating_cashflow = COALESCE(excluded.operating_cashflow, fundamental_snapshots.operating_cashflow),
            free_cashflow = COALESCE(excluded.free_cashflow, fundamental_snapshots.free_cashflow),
            source = COALESCE(excluded.source, fundamental_snapshots.source),
            fetched_at = COALESCE(excluded.fetched_at, fundamental_snapshots.fetched_at),
            updated_at = excluded.updated_at,
            raw_json = COALESCE(excluded.raw_json, fundamental_snapshots.raw_json)
        """,
        (
            ts_code.upper(),
            market,
            payload.get("report_period"),
            payload.get("currency"),
            payload.get("revenue"),
            payload.get("net_income"),
            payload.get("gross_margin"),
            payload.get("operating_margin"),
            payload.get("roe"),
            payload.get("roa"),
            payload.get("debt_to_asset"),
            payload.get("pe_ttm"),
            payload.get("pb"),
            payload.get("ps"),
            payload.get("market_cap"),
            payload.get("total_assets"),
            payload.get("total_liabilities"),
            payload.get("shareholder_equity"),
            payload.get("operating_cashflow"),
            payload.get("free_cashflow"),
            payload.get("source"),
            payload.get("fetched_at"),
            _now_iso(),
            _safe_json_dumps(payload.get("raw")),
        ),
    )
    conn.commit()
    conn.close()


def _store_series(ts_code: str, series: list[dict[str, Any]]) -> None:
    if not series:
        return
    conn = _connect()
    conn.executemany(
        """
        INSERT OR REPLACE INTO fundamental_series (
            ts_code, metric_name, label, period, value, source, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                ts_code.upper(),
                item.get("metric_name"),
                item.get("label"),
                item.get("period"),
                item.get("value"),
                item.get("source"),
                item.get("fetched_at"),
            )
            for item in series
            if item.get("metric_name") and item.get("period")
        ],
    )
    conn.commit()
    conn.close()


def _store_peer_relationships(
    ts_code: str,
    relation_type: str,
    peers: list[dict[str, Any]],
    source: str,
) -> None:
    conn = _connect()
    now = _now_iso()
    conn.execute(
        "DELETE FROM peer_relationships WHERE ts_code = ? AND relation_type = ?",
        (ts_code.upper(), relation_type),
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO peer_relationships (
            ts_code, peer_ts_code, relation_type, similarity_score, reason, source, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                ts_code.upper(),
                str(item.get("ts_code", "")).upper(),
                relation_type,
                item.get("similarity_score"),
                item.get("reason"),
                source,
                now,
            )
            for item in peers
            if item.get("ts_code")
        ],
    )
    conn.commit()
    conn.close()


def _fetch_tushare_client() -> Any | None:
    try:
        import tushare as ts
        from quant_investor.config import config
        from quant_investor.credential_utils import create_tushare_pro
    except Exception:
        return None

    return create_tushare_pro(
        ts,
        getattr(config, "TUSHARE_TOKEN", ""),
        os.environ.get("TUSHARE_URL", "http://lianghua.nanyangqiankun.top"),
    )


def _fetch_cn_research_bundle(ts_code: str) -> dict[str, Any]:
    pro = _fetch_tushare_client()
    if pro is None:
        return {}

    try:
        company_df = pro.stock_company(ts_code=ts_code)
    except Exception:
        logger.warning("Failed to fetch company info for %s", ts_code, exc_info=True)
        company_df = pd.DataFrame()
    try:
        fina_df = pro.fina_indicator(ts_code=ts_code, limit=4)
    except Exception:
        logger.warning("Failed to fetch fina_indicator for %s", ts_code, exc_info=True)
        fina_df = pd.DataFrame()
    try:
        income_df = pro.income(ts_code=ts_code, limit=4)
    except Exception:
        logger.warning("Failed to fetch income for %s", ts_code, exc_info=True)
        income_df = pd.DataFrame()
    try:
        balance_df = pro.balancesheet(ts_code=ts_code, limit=4)
    except Exception:
        logger.warning("Failed to fetch balancesheet for %s", ts_code, exc_info=True)
        balance_df = pd.DataFrame()
    try:
        cashflow_df = pro.cashflow(ts_code=ts_code, limit=4)
    except Exception:
        logger.warning("Failed to fetch cashflow for %s", ts_code, exc_info=True)
        cashflow_df = pd.DataFrame()

    company = company_df.iloc[0].to_dict() if not company_df.empty else {}
    fetched_at = _now_iso()
    profile_summary = _clean_text(company.get("introduction") or company.get("main_business"))
    business_scope = _clean_text(company.get("business_scope"))
    summary = profile_summary or business_scope
    products = _split_keywords(company.get("main_business"))
    business_lines = _split_keywords(company.get("business_scope"))

    profile = {
        "sector": _clean_text(company.get("industry")) or None,
        "industry": _clean_text(company.get("industry")) or None,
        "exchange": _clean_text(company.get("exchange")) or None,
        "summary": summary or None,
        "description": business_scope or summary or None,
        "products": products,
        "business_lines": business_lines,
        "website": _clean_text(company.get("website")) or None,
        "city": _clean_text(company.get("city")) or None,
        "region": _clean_text(company.get("province")) or None,
        "country": "中国",
        "employees": int(company.get("employees")) if company.get("employees") not in {None, ""} else None,
        "source": "tushare",
        "fetched_at": fetched_at,
        "raw": company,
    }

    report_period = None
    if not fina_df.empty:
        report_period = str(fina_df.iloc[0].get("end_date") or fina_df.iloc[0].get("ann_date") or "")
    elif not income_df.empty:
        report_period = str(income_df.iloc[0].get("end_date") or income_df.iloc[0].get("ann_date") or "")

    total_assets = _snapshot_metric(balance_df, "total_assets")
    total_liabilities = _snapshot_metric(balance_df, "total_liab", "total_liabilities")
    shareholder_equity = _snapshot_metric(
        balance_df,
        "total_hldr_eqy_exc_min_int",
        "total_hldr_eqy_inc_min_int",
        "total_assets",
    )
    operating_cashflow = _snapshot_metric(cashflow_df, "n_cashflow_act")

    snapshot = {
        "report_period": report_period or None,
        "currency": "CNY",
        "revenue": _snapshot_metric(income_df, "total_revenue", "revenue"),
        "net_income": _snapshot_metric(income_df, "n_income_attr_p", "n_income"),
        "gross_margin": _ratio_from_percent(_snapshot_metric(fina_df, "grossprofit_margin")),
        "operating_margin": _ratio_from_percent(_snapshot_metric(fina_df, "op_of_gr")),
        "roe": _ratio_from_percent(_snapshot_metric(fina_df, "roe", "roe_dt")),
        "roa": _ratio_from_percent(_snapshot_metric(fina_df, "roa")),
        "debt_to_asset": _ratio_from_percent(_snapshot_metric(fina_df, "debt_to_assets")),
        "pe_ttm": None,
        "pb": None,
        "ps": None,
        "market_cap": None,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "shareholder_equity": shareholder_equity if shareholder_equity != total_assets else None,
        "operating_cashflow": operating_cashflow,
        "free_cashflow": None,
        "source": "tushare",
        "fetched_at": fetched_at,
        "raw": {
            "fina_indicator": fina_df.head(4).to_dict(orient="records"),
            "income": income_df.head(4).to_dict(orient="records"),
            "balancesheet": balance_df.head(4).to_dict(orient="records"),
            "cashflow": cashflow_df.head(4).to_dict(orient="records"),
        },
    }

    series: list[dict[str, Any]] = []
    for metric_name, label, frame, keys, ratio in [
        ("revenue", "营业收入", income_df, ("total_revenue", "revenue"), False),
        ("net_income", "净利润", income_df, ("n_income_attr_p", "n_income"), False),
        ("roe", "ROE", fina_df, ("roe", "roe_dt"), True),
        ("gross_margin", "毛利率", fina_df, ("grossprofit_margin",), True),
        ("operating_cashflow", "经营现金流", cashflow_df, ("n_cashflow_act",), False),
    ]:
        if frame.empty:
            continue
        for _, row in frame.head(4).iterrows():
            period = str(row.get("end_date") or row.get("ann_date") or "")
            value = _coalesce_numeric(*(row.get(key) for key in keys))
            if ratio:
                value = _ratio_from_percent(value)
            series.append(
                {
                    "metric_name": metric_name,
                    "label": label,
                    "period": period,
                    "value": value,
                    "source": "tushare",
                    "fetched_at": fetched_at,
                }
            )

    return {"profile": profile, "snapshot": snapshot, "series": series}


def _safe_yfinance_frame(frame: Any) -> pd.DataFrame:
    try:
        if frame is None:
            return pd.DataFrame()
        if isinstance(frame, pd.DataFrame):
            return frame.copy()
        return pd.DataFrame(frame)
    except Exception:
        return pd.DataFrame()


def _fetch_us_research_bundle(ts_code: str) -> dict[str, Any]:
    try:
        import yfinance as yf
    except Exception:
        logger.debug("yfinance not available, skipping US research for %s", ts_code)
        return {}

    fetched_at = _now_iso()
    try:
        ticker = yf.Ticker(ts_code)
        info = ticker.get_info() if hasattr(ticker, "get_info") else dict(ticker.info)
    except Exception:
        logger.warning("Failed to fetch yfinance info for %s", ts_code, exc_info=True)
        info = {}
        ticker = None

    financials = _safe_yfinance_frame(getattr(ticker, "quarterly_financials", None) if ticker else None)
    if financials.empty:
        financials = _safe_yfinance_frame(getattr(ticker, "financials", None) if ticker else None)
    balance_sheet = _safe_yfinance_frame(getattr(ticker, "quarterly_balance_sheet", None) if ticker else None)
    if balance_sheet.empty:
        balance_sheet = _safe_yfinance_frame(getattr(ticker, "balance_sheet", None) if ticker else None)
    cashflow = _safe_yfinance_frame(getattr(ticker, "quarterly_cashflow", None) if ticker else None)
    if cashflow.empty:
        cashflow = _safe_yfinance_frame(getattr(ticker, "cashflow", None) if ticker else None)

    summary = _clean_text(info.get("longBusinessSummary"))
    profile = {
        "sector": _clean_text(info.get("sector")) or None,
        "industry": _clean_text(info.get("industry")) or None,
        "exchange": _clean_text(info.get("exchange")) or _clean_text(info.get("fullExchangeName")) or None,
        "summary": summary or None,
        "description": summary or None,
        "products": _split_keywords(summary),
        "business_lines": _split_keywords(summary, limit=4),
        "website": _clean_text(info.get("website")) or None,
        "city": _clean_text(info.get("city")) or None,
        "region": _clean_text(info.get("state")) or None,
        "country": _clean_text(info.get("country")) or None,
        "employees": int(info.get("fullTimeEmployees")) if info.get("fullTimeEmployees") not in {None, ""} else None,
        "source": "yfinance",
        "fetched_at": fetched_at,
        "raw": info,
    }

    snapshot = {
        "report_period": None,
        "currency": _clean_text(info.get("currency")) or "USD",
        "revenue": _coalesce_numeric(info.get("totalRevenue")),
        "net_income": _coalesce_numeric(info.get("netIncomeToCommon"), info.get("netIncome")),
        "gross_margin": _coalesce_numeric(info.get("grossMargins")),
        "operating_margin": _coalesce_numeric(info.get("operatingMargins")),
        "roe": _coalesce_numeric(info.get("returnOnEquity")),
        "roa": _coalesce_numeric(info.get("returnOnAssets")),
        "debt_to_asset": None,
        "pe_ttm": _coalesce_numeric(info.get("trailingPE")),
        "pb": _coalesce_numeric(info.get("priceToBook")),
        "ps": _coalesce_numeric(info.get("priceToSalesTrailing12Months")),
        "market_cap": _coalesce_numeric(info.get("marketCap")),
        "total_assets": None,
        "total_liabilities": None,
        "shareholder_equity": None,
        "operating_cashflow": _coalesce_numeric(info.get("operatingCashflow")),
        "free_cashflow": _coalesce_numeric(info.get("freeCashflow")),
        "source": "yfinance",
        "fetched_at": fetched_at,
        "raw": info,
    }

    series: list[dict[str, Any]] = []
    for metric_name, label, frame, candidates in [
        ("revenue", "营业收入", financials, ("Total Revenue", "Revenue")),
        ("net_income", "净利润", financials, ("Net Income", "Net Income Common Stockholders")),
        ("operating_cashflow", "经营现金流", cashflow, ("Operating Cash Flow", "Total Cash From Operating Activities")),
        ("total_assets", "总资产", balance_sheet, ("Total Assets",)),
        ("shareholder_equity", "股东权益", balance_sheet, ("Stockholders Equity", "Common Stock Equity")),
    ]:
        if frame.empty:
            continue
        working = frame.transpose()
        for period, row in working.head(4).iterrows():
            value = _coalesce_numeric(*(row.get(key) for key in candidates))
            if value is None:
                continue
            series.append(
                {
                    "metric_name": metric_name,
                    "label": label,
                    "period": getattr(period, "strftime", lambda fmt: str(period))("%Y-%m-%d"),
                    "value": value,
                    "source": "yfinance",
                    "fetched_at": fetched_at,
                }
            )

    if not balance_sheet.empty:
        latest_balance = balance_sheet.transpose().head(1)
        snapshot["total_assets"] = _snapshot_metric(latest_balance, "Total Assets")
        snapshot["total_liabilities"] = _snapshot_metric(
            latest_balance,
            "Total Liabilities Net Minority Interest",
            "Total Liabilities",
        )
        snapshot["shareholder_equity"] = _snapshot_metric(
            latest_balance,
            "Stockholders Equity",
            "Common Stock Equity",
        )
        if snapshot["total_assets"] and snapshot["total_liabilities"]:
            snapshot["debt_to_asset"] = snapshot["total_liabilities"] / snapshot["total_assets"]

    if series:
        snapshot["report_period"] = series[0]["period"]

    return {"profile": profile, "snapshot": snapshot, "series": series}


def _fetch_research_bundle(ts_code: str, market: str) -> dict[str, Any]:
    if market == "CN":
        return _fetch_cn_research_bundle(ts_code)
    return _fetch_us_research_bundle(ts_code)


def _ensure_research_cache(ts_code: str, market: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    ts_code = ts_code.upper()
    profile = _cached_profile(ts_code)
    snapshot = _cached_snapshot(ts_code)
    series = _cached_series(ts_code)

    needs_refresh = (
        profile is None
        or snapshot is None
        or _is_stale(profile.get("updated_at") if profile else None)
        or _is_stale(snapshot.get("updated_at") if snapshot else None)
    )
    if not needs_refresh:
        return profile, snapshot, series

    bundle = _fetch_research_bundle(ts_code, market)
    profile_payload = bundle.get("profile")
    snapshot_payload = bundle.get("snapshot")
    series_payload = bundle.get("series") or []

    if isinstance(profile_payload, dict):
        _store_profile(ts_code, market, profile_payload)
        if profile_payload.get("industry"):
            _update_stock_metadata(
                {
                    ts_code: {
                        "industry": profile_payload.get("industry"),
                        "market": market,
                    }
                }
            )
    if isinstance(snapshot_payload, dict):
        _store_snapshot(ts_code, market, snapshot_payload)
    if series_payload:
        _store_series(ts_code, series_payload)

    return _cached_profile(ts_code), _cached_snapshot(ts_code), _cached_series(ts_code)


def _format_trade_date(value: Optional[str]) -> str:
    if not value:
        return "-"
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:8]}"
    return value


def _pct_to_text(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.2f}%"


def _float_to_text(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}{suffix}"


def _tone_from_value(value: Optional[float], positive_good: bool = True) -> str:
    if value is None:
        return "neutral"
    if abs(value) < 1e-9:
        return "neutral"
    is_positive = value > 0
    if positive_good:
        return "positive" if is_positive else "negative"
    return "negative" if is_positive else "positive"


def _enrich_stock_payload(payload: dict[str, Any]) -> dict[str, Any]:
    code = str(payload.get("ts_code", "")).upper()
    meta = _local_metadata_cache().get(code, {})
    payload = dict(payload)
    payload["ts_code"] = code
    payload["name"] = payload.get("name") or meta.get("name")
    payload["industry"] = payload.get("industry") or meta.get("industry")
    payload["market"] = payload.get("market") or _guess_market(code)
    return payload


def _search_codes_from_local_metadata(search: str) -> list[str]:
    keyword = search.strip().lower()
    if not keyword:
        return []
    matched = []
    for code, meta in _local_metadata_cache().items():
        name = str(meta.get("name", "")).lower()
        if keyword in code.lower() or (name and keyword in name):
            matched.append(code)
    return matched[:50]


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).copy()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    return df


def _price_snapshot(df: pd.DataFrame) -> dict[str, Optional[float]]:
    if df.empty or "close" not in df.columns:
        return {
            "latest_close": None,
            "previous_close": None,
            "change_pct": None,
            "return_20d": None,
            "return_60d": None,
            "volatility_20d": None,
            "avg_volume_20d": None,
            "high_52w": None,
            "low_52w": None,
            "support_level": None,
            "resistance_level": None,
        }

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(dtype=float)
    returns = close.pct_change()

    def _relative_return(days: int) -> Optional[float]:
        if len(close) <= days:
            return None
        anchor = float(close.iloc[-days - 1])
        if anchor == 0:
            return None
        return float(close.iloc[-1] / anchor - 1)

    previous_close = float(close.iloc[-2]) if len(close) > 1 else None
    latest_close = float(close.iloc[-1])
    change_pct = None if previous_close in {None, 0} else latest_close / previous_close - 1

    return {
        "latest_close": latest_close,
        "previous_close": previous_close,
        "change_pct": change_pct,
        "return_20d": _relative_return(20),
        "return_60d": _relative_return(60),
        "volatility_20d": float(returns.tail(20).std() * (252 ** 0.5)) if len(returns.dropna()) >= 10 else None,
        "avg_volume_20d": float(volume.tail(20).mean()) if not volume.empty else None,
        "high_52w": float(high.tail(252).max()) if len(high) else None,
        "low_52w": float(low.tail(252).min()) if len(low) else None,
        "support_level": float(low.tail(20).min()) if len(low) >= 5 else None,
        "resistance_level": float(high.tail(20).max()) if len(high) >= 5 else None,
    }


def _build_factor_signals(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or len(df) < 20:
        return []

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = (-diff.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    volume_ratio = volume.iloc[-1] / volume.tail(20).mean() if volume.tail(20).mean() else None

    latest_close = float(close.iloc[-1])
    ma20_gap = (latest_close / float(ma20.iloc[-1]) - 1) if pd.notna(ma20.iloc[-1]) and float(ma20.iloc[-1]) else None
    ma60_gap = (latest_close / float(ma60.iloc[-1]) - 1) if pd.notna(ma60.iloc[-1]) and float(ma60.iloc[-1]) else None
    macd = float(ema12.iloc[-1] - ema26.iloc[-1])
    rsi_value = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
    momentum_20 = latest_close / float(close.iloc[-21]) - 1 if len(close) > 20 and float(close.iloc[-21]) else None

    return [
        {
            "key": "ma20_gap",
            "label": "相对20日均线",
            "value": ma20_gap,
            "display_value": _pct_to_text(ma20_gap),
            "signal": _tone_from_value(ma20_gap),
            "description": "衡量股价相对短期趋势线的偏离程度。",
        },
        {
            "key": "ma60_gap",
            "label": "相对60日均线",
            "value": ma60_gap,
            "display_value": _pct_to_text(ma60_gap),
            "signal": _tone_from_value(ma60_gap),
            "description": "衡量中期趋势位置，适合判断是否处于上行通道。",
        },
        {
            "key": "rsi14",
            "label": "RSI(14)",
            "value": rsi_value,
            "display_value": _float_to_text(rsi_value, digits=1),
            "signal": "negative" if rsi_value and rsi_value >= 70 else "positive" if rsi_value and rsi_value <= 30 else "neutral",
            "description": "高于70偏热，低于30偏冷。",
        },
        {
            "key": "macd",
            "label": "MACD 差值",
            "value": macd,
            "display_value": _float_to_text(macd, digits=2),
            "signal": _tone_from_value(macd),
            "description": "正值代表短期均线强于中期均线。",
        },
        {
            "key": "volume_ratio",
            "label": "量比(对20日均量)",
            "value": float(volume_ratio) if volume_ratio is not None else None,
            "display_value": _float_to_text(float(volume_ratio), digits=2) if volume_ratio is not None else "-",
            "signal": "positive" if volume_ratio and volume_ratio >= 1.2 else "neutral",
            "description": "衡量近期成交活跃度是否明显放大。",
        },
        {
            "key": "momentum_20",
            "label": "20日动量",
            "value": momentum_20,
            "display_value": _pct_to_text(momentum_20),
            "signal": _tone_from_value(momentum_20),
            "description": "过去20个交易日的价格趋势。",
        },
    ]


def _build_profile_summary(stock: dict[str, Any], snapshot: dict[str, Optional[float]]) -> str:
    display_name = stock.get("name") or stock["ts_code"]
    market_label = "A股" if stock.get("market") == "CN" else "美股"
    parts = [f"{display_name} 是一只 {market_label} 标的。"]

    if stock.get("industry"):
        parts.append(f"当前归类为“{stock['industry']}”方向。")
    if stock.get("list_date"):
        parts.append(f"上市日期为 {_format_trade_date(stock['list_date'])}。")
    if stock.get("record_count"):
        parts.append(f"本地已收录 {stock['record_count']:,} 条日线记录。")
    if snapshot.get("latest_close") is not None:
        parts.append(f"最新收盘价约 {_float_to_text(snapshot['latest_close'])}。")
    if snapshot.get("return_20d") is not None:
        parts.append(f"近20日涨跌幅 {_pct_to_text(snapshot['return_20d'])}。")

    return " ".join(parts)


def _build_tags(stock: dict[str, Any]) -> list[str]:
    tags = []
    if stock.get("market") == "CN":
        tags.append("A股")
    elif stock.get("market") == "US":
        tags.append("美股")
    if stock.get("industry"):
        tags.append(str(stock["industry"]))
    if stock.get("is_hs300"):
        tags.append("HS300")
    if stock.get("is_zz500"):
        tags.append("中证500")
    if stock.get("is_zz1000"):
        tags.append("中证1000")
    category = _local_metadata_cache().get(stock["ts_code"], {}).get("category")
    if category:
        tags.append(category)
    return tags[:6]


def _build_key_metrics(stock: dict[str, Any], snapshot: dict[str, Optional[float]]) -> list[dict[str, str]]:
    return [
        {
            "label": "最新收盘",
            "value": _float_to_text(snapshot.get("latest_close")),
            "tone": "neutral",
        },
        {
            "label": "当日涨跌",
            "value": _pct_to_text(snapshot.get("change_pct")),
            "tone": _tone_from_value(snapshot.get("change_pct")),
        },
        {
            "label": "20日收益",
            "value": _pct_to_text(snapshot.get("return_20d")),
            "tone": _tone_from_value(snapshot.get("return_20d")),
        },
        {
            "label": "60日收益",
            "value": _pct_to_text(snapshot.get("return_60d")),
            "tone": _tone_from_value(snapshot.get("return_60d")),
        },
        {
            "label": "20日波动率",
            "value": _pct_to_text(snapshot.get("volatility_20d")),
            "tone": _tone_from_value(snapshot.get("volatility_20d"), positive_good=False),
        },
        {
            "label": "20日均量",
            "value": _float_to_text(snapshot.get("avg_volume_20d"), digits=0),
            "tone": "neutral",
        },
    ]


def _build_company_metrics(stock: dict[str, Any], snapshot: dict[str, Optional[float]]) -> list[dict[str, str]]:
    return [
        {
            "label": "52周高点",
            "value": _float_to_text(snapshot.get("high_52w")),
            "tone": "neutral",
        },
        {
            "label": "52周低点",
            "value": _float_to_text(snapshot.get("low_52w")),
            "tone": "neutral",
        },
        {
            "label": "短线支撑",
            "value": _float_to_text(snapshot.get("support_level")),
            "tone": "neutral",
        },
        {
            "label": "短线阻力",
            "value": _float_to_text(snapshot.get("resistance_level")),
            "tone": "neutral",
        },
        {
            "label": "数据区间",
            "value": (
                f"{_format_trade_date(stock.get('date_start'))} ~ {_format_trade_date(stock.get('date_end'))}"
                if stock.get("date_start") and stock.get("date_end")
                else "-"
            ),
            "tone": "neutral",
        },
        {
            "label": "本地记录数",
            "value": f"{int(stock.get('record_count', 0)):,}",
            "tone": "neutral",
        },
    ]


def _availability(
    ready: bool,
    *,
    updated_at: Optional[str] = None,
    source: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "ready": bool(ready),
        "updated_at": updated_at,
        "source": source,
        "note": note,
    }


def _load_recent_analysis_context(limit: int = 120) -> tuple[set[str], list[dict[str, Any]]]:
    if not os.path.exists(APP_DB_PATH):
        return set(), []

    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT analysis_id, created_at, result_json
            FROM analysis_sessions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return set(), []
    conn.close()

    symbols: set[str] = set()
    watch_candidates: list[dict[str, Any]] = []
    for row in rows:
        result = _safe_json_loads(row["result_json"])
        if not isinstance(result, dict):
            continue
        candidate_symbols = [
            str(item).upper()
            for item in result.get("candidate_symbols", [])
            if str(item).strip()
        ]
        symbols.update(candidate_symbols)
        request = result.get("request", {})
        symbols.update(
            str(item).upper()
            for item in request.get("targets", request.get("stocks", []))
            if str(item).strip()
        )
        if candidate_symbols:
            watch_candidates.append(
                {
                    "symbol": candidate_symbols[0],
                    "title": f"{request.get('preset', 'quick_scan')} · {request.get('mode', 'single')}",
                    "created_at": str(row["created_at"]),
                    "summary": str(result.get("final_decision", "")) or "最近分析中进入候选池。",
                }
            )

    deduped_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in watch_candidates:
        if item["symbol"] in seen:
            continue
        seen.add(item["symbol"])
        deduped_candidates.append(item)
    return symbols, deduped_candidates[:8]


def _build_completeness(
    stock: dict[str, Any],
    snapshot: dict[str, Any],
    profile: dict[str, Any] | None,
    fundamentals: dict[str, Any] | None,
    competitor_count: int,
) -> dict[str, Any]:
    profile_updated_at = profile.get("updated_at") if profile else None
    fundamentals_updated_at = fundamentals.get("updated_at") if fundamentals else None
    technical_ready = int(stock.get("record_count", 0)) >= 60 and snapshot.get("latest_close") is not None
    industry_ready = bool(stock.get("industry") or (profile and (profile.get("industry") or profile.get("sector"))))
    business_ready = bool(profile and (profile.get("summary") or profile.get("products") or profile.get("business_lines")))
    profile_ready = bool(profile and (profile.get("summary") or profile.get("website") or profile.get("description")))
    fundamentals_ready = bool(
        fundamentals
        and any(
            fundamentals.get(key) is not None
            for key in (
                "revenue",
                "net_income",
                "gross_margin",
                "roe",
                "market_cap",
                "operating_cashflow",
            )
        )
    )
    competitor_ready = competitor_count > 0

    return {
        "technical": _availability(
            technical_ready,
            updated_at=stock.get("last_update"),
            source="local_csv",
            note=None if technical_ready else "需要更多日线数据",
        ),
        "fundamentals": _availability(
            fundamentals_ready,
            updated_at=fundamentals_updated_at,
            source=fundamentals.get("source") if fundamentals else None,
            note=None if fundamentals_ready else "尚未缓存标准化基本面快照",
        ),
        "industry": _availability(
            industry_ready,
            updated_at=profile_updated_at or stock.get("last_update"),
            source=profile.get("source") if profile else "local_db",
            note=None if industry_ready else "缺少行业标签",
        ),
        "competitors": _availability(
            competitor_ready,
            updated_at=stock.get("last_update"),
            source="local_similarity",
            note=None if competitor_ready else "尚未建立可比公司关系",
        ),
        "business": _availability(
            business_ready,
            updated_at=profile_updated_at,
            source=profile.get("source") if profile else None,
            note=None if business_ready else "缺少主营业务/产品描述",
        ),
        "profile": _availability(
            profile_ready,
            updated_at=profile_updated_at,
            source=profile.get("source") if profile else None,
            note=None if profile_ready else "缺少公司档案",
        ),
    }


def _apply_research_flags(
    stock: dict[str, Any],
    snapshot: dict[str, Any],
    profile: dict[str, Any] | None,
    fundamentals: dict[str, Any] | None,
    competitor_count: int,
    recent_symbols: set[str],
) -> dict[str, Any]:
    completeness = _build_completeness(stock, snapshot, profile, fundamentals, competitor_count)
    item = dict(stock)
    item["latest_close"] = snapshot.get("latest_close")
    item["change_pct"] = snapshot.get("change_pct")
    item["has_profile"] = bool(completeness["profile"]["ready"])
    item["has_fundamentals"] = bool(completeness["fundamentals"]["ready"])
    item["recently_analyzed"] = stock["ts_code"] in recent_symbols
    item["completeness"] = completeness
    return item


def _completeness_filter_match(item: dict[str, Any], completeness: Optional[str]) -> bool:
    if not completeness:
        return True
    status = item.get("completeness", {})
    if completeness == "complete":
        return all(section.get("ready") for section in status.values())
    if completeness == "needs_attention":
        return any(not section.get("ready") for section in status.values())
    if completeness == "missing_fundamentals":
        return not status.get("fundamentals", {}).get("ready")
    if completeness == "missing_profile":
        return not status.get("profile", {}).get("ready")
    if completeness == "missing_business":
        return not status.get("business", {}).get("ready")
    if completeness == "missing_competitors":
        return not status.get("competitors", {}).get("ready")
    return True


def _cached_research_maps() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    conn = _connect()
    profile_rows = conn.execute("SELECT * FROM stock_profiles").fetchall()
    snapshot_rows = conn.execute("SELECT * FROM fundamental_snapshots").fetchall()
    peer_rows = conn.execute(
        """
        SELECT ts_code, COUNT(*) AS peer_count
        FROM peer_relationships
        GROUP BY ts_code
        """
    ).fetchall()
    conn.close()
    profiles = {
        str(row["ts_code"]).upper(): _profile_from_row(row) or {}
        for row in profile_rows
    }
    snapshots = {
        str(row["ts_code"]).upper(): _snapshot_from_row(row) or {}
        for row in snapshot_rows
    }
    peer_counts = {
        str(row["ts_code"]).upper(): int(row["peer_count"] or 0)
        for row in peer_rows
    }
    return profiles, snapshots, peer_counts


def _build_market_pulse(sample_limit: int = 240) -> dict[str, Any]:
    conn = _connect()
    codes = [
        str(row["ts_code"]).upper()
        for row in conn.execute(
            """
            SELECT ts_code
            FROM stock_list
            WHERE EXISTS (
                SELECT 1
                FROM daily_data d
                WHERE d.ts_code = stock_list.ts_code
            )
            ORDER BY market ASC, ts_code ASC
            LIMIT ?
            """,
            (sample_limit,),
        ).fetchall()
    ]
    if not codes:
        conn.close()
        return {
            "sampled_stocks": 0,
            "rising_count_20d": 0,
            "positive_ratio_20d": 0.0,
            "avg_return_20d": 0.0,
            "avg_volatility_20d": 0.0,
            "risk_state": "neutral",
            "breadth_label": "观望",
            "last_trade_date": None,
        }

    placeholders = ",".join("?" for _ in codes)
    rows = conn.execute(
        f"""
        SELECT ts_code, trade_date, close
        FROM daily_data
        WHERE ts_code IN ({placeholders})
        ORDER BY trade_date ASC
        """,
        codes,
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "sampled_stocks": 0,
            "rising_count_20d": 0,
            "positive_ratio_20d": 0.0,
            "avg_return_20d": 0.0,
            "avg_volatility_20d": 0.0,
            "risk_state": "neutral",
            "breadth_label": "观望",
            "last_trade_date": None,
        }

    frame = pd.DataFrame(rows, columns=["ts_code", "trade_date", "close"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    pivot = frame.pivot_table(index="trade_date", columns="ts_code", values="close").sort_index()
    returns_20d: list[float] = []
    volatility_20d: list[float] = []

    for column in pivot.columns:
        series = pivot[column].dropna()
        if len(series) < 22:
            continue
        base = float(series.iloc[-21])
        if base == 0:
            continue
        returns_20d.append(float(series.iloc[-1] / base - 1))
        vol = series.pct_change().dropna().tail(20)
        if len(vol) >= 10:
            volatility_20d.append(float(vol.std() * (252 ** 0.5)))

    sampled = len(returns_20d)
    rising_count = sum(1 for item in returns_20d if item > 0)
    positive_ratio = (rising_count / sampled) if sampled else 0.0
    avg_return = sum(returns_20d) / sampled if sampled else 0.0
    avg_vol = sum(volatility_20d) / len(volatility_20d) if volatility_20d else 0.0

    if positive_ratio >= 0.6 and avg_return > 0:
        breadth_label = "偏强"
        risk_state = "constructive"
    elif positive_ratio <= 0.4 and avg_return < 0:
        breadth_label = "承压"
        risk_state = "risk_off"
    else:
        breadth_label = "分化"
        risk_state = "neutral"

    last_trade_date = str(pivot.index.max()) if len(pivot.index) else None
    return {
        "sampled_stocks": sampled,
        "rising_count_20d": rising_count,
        "positive_ratio_20d": positive_ratio,
        "avg_return_20d": avg_return,
        "avg_volatility_20d": avg_vol,
        "risk_state": risk_state,
        "breadth_label": breadth_label,
        "last_trade_date": last_trade_date,
    }


def get_statistics() -> dict:
    _ensure_local_data_indexed()
    _ensure_missing_stock_data_downloaded()
    conn = _connect()
    cursor = conn.cursor()
    stats: dict[str, Any] = {}

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM stock_list s
        WHERE EXISTS (
            SELECT 1
            FROM daily_data d
            WHERE d.ts_code = s.ts_code
        )
        """
    )
    stats["total_stocks"] = int(cursor.fetchone()[0])

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM stock_list s
        WHERE s.market='CN'
          AND EXISTS (
              SELECT 1
              FROM daily_data d
              WHERE d.ts_code = s.ts_code
          )
        """
    )
    stats["cn_count"] = int(cursor.fetchone()[0])

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM stock_list s
        WHERE s.market='US'
          AND EXISTS (
              SELECT 1
              FROM daily_data d
              WHERE d.ts_code = s.ts_code
          )
        """
    )
    stats["us_count"] = int(cursor.fetchone()[0])

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM stock_list s
        WHERE s.is_hs300=1
          AND EXISTS (
              SELECT 1
              FROM daily_data d
              WHERE d.ts_code = s.ts_code
          )
        """
    )
    stats["hs300_count"] = int(cursor.fetchone()[0])

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM stock_list s
        WHERE s.is_zz500=1
          AND EXISTS (
              SELECT 1
              FROM daily_data d
              WHERE d.ts_code = s.ts_code
          )
        """
    )
    stats["zz500_count"] = int(cursor.fetchone()[0])

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM stock_list s
        WHERE s.is_zz1000=1
          AND EXISTS (
              SELECT 1
              FROM daily_data d
              WHERE d.ts_code = s.ts_code
          )
        """
    )
    stats["zz1000_count"] = int(cursor.fetchone()[0])

    cursor.execute("SELECT COUNT(*) FROM daily_data")
    stats["total_records"] = int(cursor.fetchone()[0])

    cursor.execute("SELECT COUNT(DISTINCT ts_code) FROM daily_data")
    stats["stocks_with_data"] = int(cursor.fetchone()[0])

    cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_data")
    result = cursor.fetchone()
    stats["date_range"] = f"{result[0]} to {result[1]}" if result and result[0] else "N/A"
    stats["last_data_update"] = _format_trade_date(result[1]) if result and result[1] else None

    conn.close()
    return stats


def get_market_overview() -> dict[str, Any]:
    _ensure_local_data_indexed()
    summary = get_statistics()
    recent_symbols, watch_candidates = _load_recent_analysis_context(limit=120)
    items, _ = get_stocks(limit=5000, metadata_refresh=False)

    completeness_counts = {
        "technical_ready": sum(1 for item in items if item["completeness"]["technical"]["ready"]),
        "fundamentals_ready": sum(1 for item in items if item["completeness"]["fundamentals"]["ready"]),
        "industry_ready": sum(1 for item in items if item["completeness"]["industry"]["ready"]),
        "competitors_ready": sum(1 for item in items if item["completeness"]["competitors"]["ready"]),
        "business_ready": sum(1 for item in items if item["completeness"]["business"]["ready"]),
        "profile_ready": sum(1 for item in items if item["completeness"]["profile"]["ready"]),
    }

    sector_counter: dict[tuple[str, str], int] = {}
    for item in items:
        market_key = str(item.get("market") or "CN")
        industry_key = str(item.get("industry") or "未分类")
        sector_counter[(market_key, industry_key)] = sector_counter.get((market_key, industry_key), 0) + 1

    sector_distribution = [
        {"market": market, "industry": industry, "count": count}
        for (market, industry), count in sorted(
            sector_counter.items(),
            key=lambda entry: entry[1],
            reverse=True,
        )[:12]
    ]

    pulse = _build_market_pulse()
    return {
        "summary": summary,
        "completeness": completeness_counts,
        "market_pulse": pulse,
        "sector_distribution": sector_distribution,
        "candidate_symbols": [item["symbol"] for item in watch_candidates],
        "watch_candidates": watch_candidates,
    }


def get_stocks(
    market: Optional[str] = None,
    index_filter: Optional[str] = None,
    search: Optional[str] = None,
    industry: Optional[str] = None,
    completeness: Optional[str] = None,
    recently_analyzed: Optional[bool] = None,
    has_fundamentals: Optional[bool] = None,
    has_profile: Optional[bool] = None,
    offset: int = 0,
    limit: int = 50,
    metadata_refresh: bool = True,
) -> tuple[list[dict], int]:
    """Return a paginated list of stocks with research metadata."""
    _ensure_local_data_indexed()
    _ensure_missing_stock_data_downloaded()
    conn = _connect()

    where_clauses = [
        """
        EXISTS (
            SELECT 1
            FROM daily_data d0
            WHERE d0.ts_code = s.ts_code
        )
        """
    ]
    params: list[Any] = []

    if market:
        where_clauses.append("s.market = ?")
        params.append(market.upper())

    if index_filter == "hs300":
        where_clauses.append("s.is_hs300 = 1")
    elif index_filter == "zz500":
        where_clauses.append("s.is_zz500 = 1")
    elif index_filter == "zz1000":
        where_clauses.append("s.is_zz1000 = 1")

    if search:
        local_codes = _search_codes_from_local_metadata(search)
        if local_codes:
            placeholders = ",".join("?" for _ in local_codes)
            where_clauses.append(
                f"(s.ts_code LIKE ? OR s.name LIKE ? OR s.ts_code IN ({placeholders}))"
            )
            params.extend([f"%{search}%", f"%{search}%"])
            params.extend(local_codes)
        else:
            where_clauses.append("(s.ts_code LIKE ? OR s.name LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

    where_sql = " AND ".join(where_clauses)
    rows = conn.execute(
        f"""
        SELECT
            s.ts_code,
            s.name,
            COALESCE(NULLIF(s.industry, ''), NULLIF(p.industry, ''), NULLIF(p.sector, '')) AS industry,
            s.market,
            s.list_date,
            s.is_hs300,
            s.is_zz500,
            s.is_zz1000,
            s.last_update,
            COALESCE(d.cnt, 0) AS record_count,
            d.min_date,
            d.max_date,
            d.latest_close,
            d.previous_close
        FROM stock_list s
        LEFT JOIN stock_profiles p ON s.ts_code = p.ts_code
        LEFT JOIN (
            SELECT
                base.ts_code,
                COUNT(*) AS cnt,
                MIN(base.trade_date) AS min_date,
                MAX(base.trade_date) AS max_date,
                (
                    SELECT close
                    FROM daily_data d2
                    WHERE d2.ts_code = base.ts_code
                    ORDER BY d2.trade_date DESC
                    LIMIT 1
                ) AS latest_close,
                (
                    SELECT close
                    FROM daily_data d3
                    WHERE d3.ts_code = base.ts_code
                    ORDER BY d3.trade_date DESC
                    LIMIT 1 OFFSET 1
                ) AS previous_close
            FROM daily_data base
            GROUP BY base.ts_code
        ) d ON s.ts_code = d.ts_code
        WHERE {where_sql}
        ORDER BY COALESCE(d.cnt, 0) DESC, s.ts_code ASC
        """,
        params,
    ).fetchall()
    conn.close()

    profiles, fundamentals_map, peer_counts = _cached_research_maps()
    recent_symbols, _ = _load_recent_analysis_context(limit=120)

    items: list[dict[str, Any]] = []
    for row in rows:
        previous_close = float(row["previous_close"]) if row["previous_close"] is not None else None
        latest_close = float(row["latest_close"]) if row["latest_close"] is not None else None
        snapshot = {
            "latest_close": latest_close,
            "previous_close": previous_close,
            "change_pct": None
            if previous_close in {None, 0} or latest_close is None
            else latest_close / previous_close - 1,
        }
        code = str(row["ts_code"]).upper()
        profile = profiles.get(code)
        fundamentals = fundamentals_map.get(code)
        item = _enrich_stock_payload(
            {
                "ts_code": code,
                "name": row["name"],
                "industry": row["industry"],
                "market": row["market"],
                "list_date": row["list_date"],
                "is_hs300": bool(row["is_hs300"]),
                "is_zz500": bool(row["is_zz500"]),
                "is_zz1000": bool(row["is_zz1000"]),
                "last_update": row["last_update"],
                "record_count": int(row["record_count"] or 0),
                "date_start": row["min_date"],
                "date_end": row["max_date"],
            }
        )
        if profile and not item.get("industry"):
            item["industry"] = profile.get("industry") or profile.get("sector")
        item = _apply_research_flags(
            item,
            snapshot,
            profile,
            fundamentals,
            peer_counts.get(code, 0),
            recent_symbols,
        )
        items.append(item)

    filtered: list[dict[str, Any]] = []
    keyword = str(industry or "").strip().lower()
    for item in items:
        if keyword and keyword not in str(item.get("industry") or "").lower():
            continue
        if recently_analyzed is not None and bool(item.get("recently_analyzed")) != recently_analyzed:
            continue
        if has_fundamentals is not None and bool(item.get("has_fundamentals")) != has_fundamentals:
            continue
        if has_profile is not None and bool(item.get("has_profile")) != has_profile:
            continue
        if not _completeness_filter_match(item, completeness):
            continue
        filtered.append(item)

    filtered.sort(
        key=lambda item: (
            not bool(item.get("recently_analyzed")),
            not bool(item.get("has_profile")),
            not bool(item.get("has_fundamentals")),
            -int(item.get("record_count", 0)),
            str(item.get("ts_code")),
        )
    )

    total = len(filtered)
    page_items = filtered[offset : offset + limit]
    if metadata_refresh:
        metadata_updates = _ensure_metadata_for_codes(page_items)
        if metadata_updates:
            page_items = [
                _enrich_stock_payload({**item, **metadata_updates.get(item["ts_code"], {})})
                for item in page_items
            ]
    return page_items, total


def get_stock_detail(ts_code: str) -> dict | None:
    """Get single stock detail."""
    _ensure_local_data_indexed()
    _ensure_missing_stock_data_downloaded()
    conn = _connect()
    row = conn.execute(
        """
        SELECT
            s.ts_code,
            s.name,
            COALESCE(NULLIF(s.industry, ''), NULLIF(p.industry, ''), NULLIF(p.sector, '')) AS industry,
            s.market,
            s.list_date,
            s.is_hs300,
            s.is_zz500,
            s.is_zz1000,
            s.last_update,
            COALESCE(d.cnt, 0) AS record_count,
            d.min_date,
            d.max_date,
            d.latest_close,
            d.previous_close
        FROM stock_list s
        LEFT JOIN stock_profiles p ON s.ts_code = p.ts_code
        LEFT JOIN (
            SELECT
                base.ts_code,
                COUNT(*) AS cnt,
                MIN(base.trade_date) AS min_date,
                MAX(base.trade_date) AS max_date,
                (
                    SELECT close
                    FROM daily_data d2
                    WHERE d2.ts_code = base.ts_code
                    ORDER BY d2.trade_date DESC
                    LIMIT 1
                ) AS latest_close,
                (
                    SELECT close
                    FROM daily_data d3
                    WHERE d3.ts_code = base.ts_code
                    ORDER BY d3.trade_date DESC
                    LIMIT 1 OFFSET 1
                ) AS previous_close
            FROM daily_data base
            GROUP BY base.ts_code
        ) d ON s.ts_code = d.ts_code
        WHERE s.ts_code = ?
          AND EXISTS (
              SELECT 1
              FROM daily_data d0
              WHERE d0.ts_code = s.ts_code
          )
        """,
        (ts_code.upper(),),
    ).fetchone()
    conn.close()
    if not row:
        return None

    code = str(row["ts_code"]).upper()
    profiles, fundamentals_map, peer_counts = _cached_research_maps()
    recent_symbols, _ = _load_recent_analysis_context(limit=120)
    profile = profiles.get(code)
    fundamentals = fundamentals_map.get(code)
    latest_close = float(row["latest_close"]) if row["latest_close"] is not None else None
    previous_close = float(row["previous_close"]) if row["previous_close"] is not None else None
    stock = _enrich_stock_payload(
        {
            "ts_code": code,
            "name": row["name"],
            "industry": row["industry"],
            "market": row["market"],
            "list_date": row["list_date"],
            "is_hs300": bool(row["is_hs300"]),
            "is_zz500": bool(row["is_zz500"]),
            "is_zz1000": bool(row["is_zz1000"]),
            "last_update": row["last_update"],
            "record_count": int(row["record_count"] or 0),
            "date_start": row["min_date"],
            "date_end": row["max_date"],
        }
    )
    if profile and not stock.get("industry"):
        stock["industry"] = profile.get("industry") or profile.get("sector")
    metadata_updates = _ensure_metadata_for_codes([stock])
    if metadata_updates.get(code):
        stock = _enrich_stock_payload({**stock, **metadata_updates[code]})

    snapshot = {
        "latest_close": latest_close,
        "previous_close": previous_close,
        "change_pct": None
        if previous_close in {None, 0} or latest_close is None
        else latest_close / previous_close - 1,
    }
    return _apply_research_flags(
        stock,
        snapshot,
        profile,
        fundamentals,
        peer_counts.get(code, 0),
        recent_symbols,
    )


def get_stock_dossier(ts_code: str) -> dict | None:
    _ensure_local_data_indexed()
    stock = get_stock_detail(ts_code)
    if not stock:
        return None

    market = stock.get("market") or _guess_market(ts_code)
    profile, fundamentals, fundamental_series = _ensure_research_cache(ts_code, market)
    if profile and not stock.get("industry"):
        stock["industry"] = profile.get("industry") or profile.get("sector")

    records = get_ohlcv(ts_code)
    frame = _records_to_frame(records)
    snapshot = _price_snapshot(frame)
    if snapshot.get("latest_close") is None:
        snapshot["latest_close"] = stock.get("latest_close")
        snapshot["change_pct"] = stock.get("change_pct")

    competitors = get_competitors(ts_code, limit=8)

    from web.services import analysis_service

    analysis_history = analysis_service.get_stock_analysis_mentions(ts_code, limit=8)
    recent_symbols, _ = _load_recent_analysis_context(limit=120)

    stock = _apply_research_flags(
        stock,
        snapshot,
        profile,
        fundamentals,
        len(competitors),
        recent_symbols,
    )

    technical_notes: list[str] = []
    if snapshot.get("return_20d") is not None:
        technical_notes.append(f"近20日收益 {_pct_to_text(snapshot['return_20d'])}。")
    if snapshot.get("volatility_20d") is not None:
        technical_notes.append(f"20日年化波动率约 {_pct_to_text(snapshot['volatility_20d'])}。")
    if snapshot.get("support_level") is not None and snapshot.get("resistance_level") is not None:
        technical_notes.append(
            f"短线关注支撑 {_float_to_text(snapshot['support_level'])} / 阻力 {_float_to_text(snapshot['resistance_level'])}。"
        )

    display_name = stock.get("name") or stock["ts_code"]
    tags = _build_tags(stock)
    if profile and profile.get("sector") and profile["sector"] not in tags:
        tags.append(str(profile["sector"]))

    conn = _connect()
    industry_value = stock.get("industry") or (profile.get("industry") if profile else None) or (
        profile.get("sector") if profile else None
    )
    if industry_value:
        industry_count = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM stock_list s
                WHERE s.industry = ?
                  AND EXISTS (
                      SELECT 1
                      FROM daily_data d
                      WHERE d.ts_code = s.ts_code
                  )
                """,
                (industry_value,),
            ).fetchone()[0]
        )
    else:
        industry_count = 0
    conn.close()

    business_summary = ""
    if profile:
        business_summary = str(profile.get("summary") or profile.get("description") or "").strip()
    if not business_summary:
        business_summary = _build_profile_summary(stock, snapshot)

    return {
        "stock": stock,
        "display_name": display_name,
        "profile_summary": business_summary,
        "tags": tags[:8],
        "completeness": stock.get("completeness"),
        "quote": snapshot,
        "technical": {
            "key_metrics": _build_key_metrics(stock, snapshot),
            "company_metrics": _build_company_metrics(stock, snapshot),
            "factors": _build_factor_signals(frame),
            "notes": technical_notes,
        },
        "fundamentals": fundamentals
        or {
            "report_period": None,
            "currency": "CNY" if market == "CN" else "USD",
            "source": None,
            "fetched_at": None,
        },
        "fundamental_series": fundamental_series,
        "industry_context": {
            "market": market,
            "sector": profile.get("sector") if profile else None,
            "industry": industry_value,
            "industry_stock_count": industry_count,
            "peer_count": len(competitors),
            "summary": (
                f"当前归属于 {industry_value}，本地数据库中同业样本 {industry_count} 只，"
                f"已识别可比公司 {len(competitors)} 只。"
                if industry_value
                else "当前行业标签仍待补齐，暂以市场内相似走势标的做可比参考。"
            ),
            "notes": [
                "同行业可比优先，其次使用走势相关性补全竞对池。",
                "行业和业务信息严格来自本地缓存或外部源，缺失时不做伪造补写。",
            ],
        },
        "competitors": competitors,
        "business_profile": {
            "summary": business_summary,
            "products": profile.get("products", []) if profile else [],
            "business_lines": profile.get("business_lines", []) if profile else [],
            "website": profile.get("website") if profile else None,
            "city": profile.get("city") if profile else None,
            "region": profile.get("region") if profile else None,
            "country": profile.get("country") if profile else None,
            "employees": profile.get("employees") if profile else None,
            "source": profile.get("source") if profile else None,
            "fetched_at": profile.get("fetched_at") if profile else None,
        },
        "analysis_history": analysis_history,
    }


def get_stock_overview(ts_code: str) -> dict | None:
    dossier = get_stock_dossier(ts_code)
    if not dossier:
        return None

    technical = dossier.get("technical", {})
    return {
        "stock": dossier["stock"],
        "display_name": dossier["display_name"],
        "profile_summary": dossier["profile_summary"],
        "tags": dossier.get("tags", []),
        "key_metrics": technical.get("key_metrics", []),
        "company_metrics": technical.get("company_metrics", []),
        "factors": technical.get("factors", []),
        "recent_analysis": dossier.get("analysis_history", []),
    }


def get_ohlcv(
    ts_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    """Get OHLCV data for a stock."""
    _ensure_local_data_indexed()
    conn = _connect()

    query = "SELECT trade_date, open, high, low, close, volume, amount FROM daily_data WHERE ts_code = ?"
    params: list[Any] = [ts_code.upper()]

    if start_date:
        query += " AND trade_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND trade_date <= ?"
        params.append(end_date)

    query += " ORDER BY trade_date ASC"

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "trade_date": row["trade_date"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "amount": row["amount"] or 0.0,
        }
        for row in rows
    ]


def import_csv_data() -> dict:
    """Import all CSV data from local data/ directories into SQLite."""
    import csv

    conn = _connect()
    cursor = conn.cursor()
    data_dir = PROJECT_ROOT / "data"

    stats = {"cn_stocks": 0, "us_stocks": 0, "cn_records": 0, "us_records": 0, "errors": []}

    cn_universe_dir = data_dir / "cn_universe"
    hs300_symbols: set[str] = set()
    zz500_symbols: set[str] = set()
    zz1000_symbols: set[str] = set()

    for fname, target in [
        ("hs300_symbols.txt", hs300_symbols),
        ("zz500_symbols.txt", zz500_symbols),
        ("zz1000_symbols.txt", zz1000_symbols),
    ]:
        fpath = cn_universe_dir / fname
        if fpath.exists():
            target.update(line.strip() for line in fpath.read_text().splitlines() if line.strip())

    cn_dirs = [
        data_dir / "cn_market_full" / "hs300",
        data_dir / "cn_market_full" / "zz500",
        data_dir / "cn_market_full" / "zz1000",
    ]

    for market_dir in cn_dirs:
        if not market_dir.exists():
            continue
        for csv_file in sorted(market_dir.glob("*.csv")):
            ts_code = csv_file.stem.upper()
            meta = _local_metadata_cache().get(ts_code, {})
            try:
                with open(csv_file, "r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    rows_inserted = 0
                    for row in reader:
                        trade_date = row.get("trade_date", "").replace("-", "")
                        if not trade_date:
                            continue
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO daily_data
                            (ts_code, trade_date, open, high, low, close, volume, amount)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ts_code,
                                trade_date,
                                float(row.get("open", 0)),
                                float(row.get("high", 0)),
                                float(row.get("low", 0)),
                                float(row.get("close", 0)),
                                float(row.get("vol", 0)),
                                float(row.get("amount", 0)),
                            ),
                        )
                        rows_inserted += 1

                cursor.execute(
                    _STOCK_UPSERT_SQL,
                    (
                        ts_code,
                        meta.get("name"),
                        meta.get("industry"),
                        "CN",
                        meta.get("list_date"),
                        1 if ts_code in hs300_symbols else 0,
                        1 if ts_code in zz500_symbols else 0,
                        1 if ts_code in zz1000_symbols else 0,
                        datetime.now().isoformat(timespec="seconds"),
                    ),
                )
                stats["cn_stocks"] += 1
                stats["cn_records"] += rows_inserted
            except Exception as exc:
                stats["errors"].append(f"CN {ts_code}: {exc}")

    us_dirs = [
        data_dir / "us_market" / "large_cap",
        data_dir / "us_market" / "mid_cap",
        data_dir / "us_market" / "small_cap",
        data_dir / "us_market_full" / "large_cap",
        data_dir / "us_market_full" / "mid_cap",
        data_dir / "us_market_full" / "small_cap",
    ]

    for market_dir in us_dirs:
        if not market_dir.exists():
            continue
        for csv_file in sorted(market_dir.glob("*.csv")):
            ts_code = csv_file.stem.upper()
            meta = _local_metadata_cache().get(ts_code, {})
            try:
                with open(csv_file, "r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    rows_inserted = 0
                    for row in reader:
                        trade_date = row.get("Date", "").replace("-", "")
                        if not trade_date:
                            continue
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO daily_data
                            (ts_code, trade_date, open, high, low, close, volume, amount)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ts_code,
                                trade_date,
                                float(row.get("Open", 0)),
                                float(row.get("High", 0)),
                                float(row.get("Low", 0)),
                                float(row.get("Close", 0)),
                                float(row.get("Volume", 0)),
                                0.0,
                            ),
                        )
                        rows_inserted += 1

                cursor.execute(
                    _STOCK_UPSERT_SQL,
                    (
                        ts_code,
                        meta.get("name"),
                        meta.get("industry"),
                        "US",
                        meta.get("list_date"),
                        0,
                        0,
                        0,
                        datetime.now().isoformat(timespec="seconds"),
                    ),
                )
                stats["us_stocks"] += 1
                stats["us_records"] += rows_inserted
            except Exception as exc:
                stats["errors"].append(f"US {ts_code}: {exc}")

    conn.commit()
    conn.close()
    return stats


def get_competitors(ts_code: str, limit: int = 10) -> list[dict]:
    """Get same-industry peers, or correlation-based similar stocks as fallback."""
    _ensure_local_data_indexed()
    stock = get_stock_detail(ts_code)
    if not stock:
        return []

    conn = _connect()
    cursor = conn.cursor()

    cached_rows = cursor.execute(
        """
        SELECT
            pr.peer_ts_code AS ts_code,
            pr.relation_type,
            pr.similarity_score,
            pr.reason,
            pr.updated_at,
            s.name,
            COALESCE(NULLIF(s.industry, ''), NULLIF(p.industry, ''), NULLIF(p.sector, '')) AS industry,
            d.latest_close,
            COALESCE(d.cnt, 0) AS cnt
        FROM peer_relationships pr
        LEFT JOIN stock_list s ON pr.peer_ts_code = s.ts_code
        LEFT JOIN stock_profiles p ON pr.peer_ts_code = p.ts_code
        LEFT JOIN (
            SELECT
                base.ts_code,
                COUNT(*) AS cnt,
                (
                    SELECT close
                    FROM daily_data d2
                    WHERE d2.ts_code = base.ts_code
                    ORDER BY trade_date DESC
                    LIMIT 1
                ) AS latest_close
            FROM daily_data base
            GROUP BY base.ts_code
        ) d ON pr.peer_ts_code = d.ts_code
        WHERE pr.ts_code = ?
        ORDER BY CASE pr.relation_type WHEN 'industry' THEN 0 ELSE 1 END, pr.similarity_score DESC
        LIMIT ?
        """,
        (stock["ts_code"], limit),
    ).fetchall()
    if cached_rows and not _is_stale(str(cached_rows[0]["updated_at"])):
        conn.close()
        return [
            {
                "ts_code": row["ts_code"],
                "name": _enrich_stock_payload({"ts_code": row["ts_code"], "name": row["name"]}).get("name"),
                "industry": row["industry"],
                "latest_close": row["latest_close"],
                "record_count": int(row["cnt"] or 0),
                "reason": row["reason"],
                "similarity_score": row["similarity_score"],
            }
            for row in cached_rows
        ]

    if stock.get("industry"):
        cursor.execute(
            """
            SELECT s.ts_code, s.name, s.industry,
                   d.latest_close, d.cnt
            FROM stock_list s
            LEFT JOIN (
                SELECT ts_code,
                       COUNT(*) AS cnt,
                       (
                           SELECT close
                           FROM daily_data d2
                           WHERE d2.ts_code = daily_data.ts_code
                           ORDER BY d2.trade_date DESC
                           LIMIT 1
                       ) AS latest_close
                FROM daily_data
                GROUP BY ts_code
            ) d ON s.ts_code = d.ts_code
            WHERE s.industry = ? AND s.ts_code != ?
              AND EXISTS (
                  SELECT 1
                  FROM daily_data d0
                  WHERE d0.ts_code = s.ts_code
              )
            ORDER BY d.cnt DESC
            LIMIT ?
            """,
            (stock["industry"], stock["ts_code"], limit),
        )
        rows = cursor.fetchall()
        same_industry = [
            {
                "ts_code": row["ts_code"],
                "name": _enrich_stock_payload({"ts_code": row["ts_code"], "name": row["name"]}).get("name"),
                "industry": row["industry"],
                "latest_close": row["latest_close"],
                "record_count": row["cnt"] or 0,
                "reason": "同行业可比公司",
                "similarity_score": None,
            }
            for row in rows
        ]
        if same_industry:
            _store_peer_relationships(stock["ts_code"], "industry", same_industry, "local_db")
            conn.close()
            return same_industry[:limit]

    cursor.execute(
        """
        SELECT ts_code, name, industry, market, is_hs300, is_zz500, is_zz1000
        FROM stock_list
        WHERE market = ? AND ts_code != ?
          AND EXISTS (
              SELECT 1
              FROM daily_data d
              WHERE d.ts_code = stock_list.ts_code
          )
        LIMIT 240
        """,
        (stock["market"], stock["ts_code"]),
    )
    candidates = [dict(row) for row in cursor.fetchall()]

    preferred_codes = []
    for item in candidates:
        if stock.get("is_hs300") and item.get("is_hs300"):
            preferred_codes.append(item["ts_code"])
        elif stock.get("is_zz500") and item.get("is_zz500"):
            preferred_codes.append(item["ts_code"])
        elif stock.get("is_zz1000") and item.get("is_zz1000"):
            preferred_codes.append(item["ts_code"])

    ordered_codes = preferred_codes + [
        item["ts_code"]
        for item in candidates
        if item["ts_code"] not in preferred_codes
    ]
    peer_pool = ordered_codes[:120]

    if not peer_pool:
        conn.close()
        return []

    placeholders = ",".join("?" for _ in [stock["ts_code"], *peer_pool])
    rows = cursor.execute(
        f"""
        SELECT ts_code, trade_date, close
        FROM daily_data
        WHERE ts_code IN ({placeholders})
        ORDER BY trade_date ASC
        """,
        [stock["ts_code"], *peer_pool],
    ).fetchall()
    conn.close()

    price_df = pd.DataFrame(rows, columns=["ts_code", "trade_date", "close"])
    if price_df.empty:
        return []

    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    pivot = price_df.pivot_table(index="trade_date", columns="ts_code", values="close").tail(180)
    returns = pivot.pct_change().dropna(how="all")
    if stock["ts_code"] not in returns.columns:
        return []

    base_series = returns[stock["ts_code"]]
    correlations = returns.corrwith(base_series).drop(labels=[stock["ts_code"]], errors="ignore").dropna()
    correlations = correlations.sort_values(ascending=False).head(limit)

    latest_close_map = (
        pivot.ffill().iloc[-1].to_dict()
        if not pivot.empty
        else {}
    )

    results = []
    for peer_code, score in correlations.items():
        candidate = next((item for item in candidates if item["ts_code"] == peer_code), None)
        if candidate is None:
            continue
        enriched = _enrich_stock_payload(candidate)
        results.append(
            {
                "ts_code": peer_code,
                "name": enriched.get("name"),
                "industry": enriched.get("industry"),
                "latest_close": float(latest_close_map.get(peer_code)) if latest_close_map.get(peer_code) is not None else None,
                "record_count": int(price_df.loc[price_df["ts_code"] == peer_code, "trade_date"].nunique()),
                "reason": "走势相关性较高",
                "similarity_score": float(score),
            }
        )

    if results:
        _store_peer_relationships(stock["ts_code"], "correlation", results, "local_correlation")
    return results

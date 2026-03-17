#!/usr/bin/env python3
"""
Stock Database Manager - 股票数据库管理器

功能:
1. 支持 A 股 / 美股统一数据落盘
2. 本地 SQLite 数据库存储
3. 区间下载、向前回填、断点续传
4. 自动去重和边界一致性校验
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from credential_utils import create_tushare_pro
from logger import get_logger
from stock_universe import StockUniverse

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TUSHARE_URL = os.environ.get("TUSHARE_URL", "http://lianghua.nanyangqiankun.top")
CONSISTENCY_FIELDS = ("open", "high", "low", "close", "volume", "amount")
BACKFILL_GRACE_DAYS = 7
SUPPORTED_MARKETS = {"CN", "US"}
US_UNIVERSE_FILE = PROJECT_ROOT / "data" / "us_universe" / "complete_us_universe.json"
PRICE_MODE_QFQ = "qfq"
VOLUME_MODE_RAW = "raw"
STANDARDIZATION_NOTE = "回测主价格采用前复权/adjusted OHLC，volume/amount 保持原始成交口径"
NO_DATA_ERROR_PATTERNS = (
    "possibly delisted",
    "no timezone found",
    "no price data found",
    "no data found",
    "404",
)


def _default_db_path() -> Path:
    path = Path(config.DB_PATH or "data/stock_database.db")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _default_cache_dir() -> Path:
    return _default_db_path().parent / "cache"


@dataclass(frozen=True)
class DownloadTask:
    """单只股票单个区间的下载任务。"""

    ts_code: str
    start_date: str
    end_date: str
    reason: str
    market: str = "CN"
    list_date: Optional[str] = None
    existing_start: Optional[str] = None
    existing_end: Optional[str] = None


@dataclass(frozen=True)
class BackfillPlan:
    """历史回填计划。"""

    years: int
    anchor_start: str
    anchor_end: str
    target_start: str
    tasks: List[DownloadTask]

    @property
    def stock_count(self) -> int:
        return len({task.ts_code for task in self.tasks})


@dataclass
class DownloadProgress:
    """下载进度。"""

    total_stocks: int
    completed_stocks: int
    failed_stocks: List[str]
    last_update: datetime

    @property
    def progress_pct(self) -> float:
        if self.total_stocks == 0:
            return 0.0
        return (self.completed_stocks / self.total_stocks) * 100


class StockDatabase:
    """
    股票数据库管理器

    使用 SQLite 存储全市场股票数据，支持高效查询、增量更新和历史回填。
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
        token: Optional[str] = None,
        init_universe: bool = False,
    ):
        self.db_path = str(self._resolve_path(db_path, _default_db_path()))
        self.cache_dir = str(self._resolve_path(cache_dir, _default_cache_dir()))
        self.verbose = verbose
        self.token = (token or config.TUSHARE_TOKEN).strip()
        self.tushare_url = DEFAULT_TUSHARE_URL
        self.progress = DownloadProgress(0, 0, [], datetime.now())
        self._logger = get_logger("StockDatabase", verbose)
        self._thread_local = threading.local()
        self._universe: Optional[StockUniverse] = (
            StockUniverse(token=self.token or None) if init_universe else None
        )

        os.makedirs(Path(self.db_path).parent, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self._init_database()
        self._log(f"数据库已就绪: {self.db_path}")

    @staticmethod
    def _resolve_path(raw_path: Optional[str], default_path: Path) -> Path:
        if raw_path is None:
            return default_path
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @staticmethod
    def _normalize_date(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y%m%d")
            except ValueError:
                continue

        raise ValueError(f"无法识别日期格式: {value}")

    @staticmethod
    def _shift_years(date_str: str, years: int) -> str:
        dt = datetime.strptime(date_str, "%Y%m%d")
        try:
            shifted = dt.replace(year=dt.year + years)
        except ValueError:
            shifted = dt.replace(year=dt.year + years, month=2, day=28)
        return shifted.strftime("%Y%m%d")

    @staticmethod
    def _calendar_gap_days(start_date: str, end_date: str) -> int:
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        return (end_dt - start_dt).days

    @staticmethod
    def _next_day(date_str: str) -> str:
        dt = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=1)
        return dt.strftime("%Y%m%d")

    @staticmethod
    def _normalize_market_filter(market: Optional[str]) -> Optional[str]:
        if market is None:
            return None
        normalized_market = str(market).strip().upper()
        if not normalized_market or normalized_market == "ALL":
            return None
        if normalized_market not in SUPPORTED_MARKETS:
            raise ValueError(f"暂不支持市场过滤: {market}")
        return normalized_market

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=60)

    def _log(self, msg: str) -> None:
        self._logger.info(msg)

    def _ensure_universe(self) -> StockUniverse:
        if self._universe is None:
            self._universe = StockUniverse(token=self.token or None)
        return self._universe

    def _get_tushare_client(self):
        pro = getattr(self._thread_local, "pro", None)
        if pro is not None:
            return pro

        import tushare as ts

        pro = create_tushare_pro(ts, self.token, self.tushare_url)
        if pro is None:
            raise RuntimeError("TUSHARE_TOKEN 未设置，无法下载股票数据")

        self._thread_local.pro = pro
        return pro

    def _init_database(self) -> None:
        """初始化数据库表结构。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")

        cursor.execute(
            """
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
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_data (
                ts_code TEXT,
                trade_date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                PRIMARY KEY (ts_code, trade_date)
            )
            """
        )

        daily_columns = {
            row[1]
            for row in cursor.execute("PRAGMA table_info(daily_data)").fetchall()
        }
        if "adj_factor" not in daily_columns:
            cursor.execute("ALTER TABLE daily_data ADD COLUMN adj_factor REAL")
        if "price_mode" not in daily_columns:
            cursor.execute("ALTER TABLE daily_data ADD COLUMN price_mode TEXT")
        if "data_source" not in daily_columns:
            cursor.execute("ALTER TABLE daily_data ADD COLUMN data_source TEXT")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS factor_data (
                ts_code TEXT,
                trade_date TEXT,
                factor_name TEXT,
                factor_value REAL,
                PRIMARY KEY (ts_code, trade_date, factor_name)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS download_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_code TEXT,
                start_date TEXT,
                end_date TEXT,
                records_count INTEGER,
                status TEXT,
                message TEXT,
                created_at TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data_config (
                market TEXT PRIMARY KEY,
                price_mode TEXT,
                volume_mode TEXT,
                data_source TEXT,
                note TEXT,
                updated_at TEXT
            )
            """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_ts_code ON daily_data(ts_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_data(trade_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_code ON factor_data(ts_code)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_download_log_ts_code ON download_log(ts_code, created_at)"
        )

        conn.commit()
        conn.close()

    def get_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """获取数据库当前整体日期范围。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_data")
        min_date, max_date = cursor.fetchone()
        conn.close()
        return min_date, max_date

    def update_stock_list(self, include_all_stocks: bool = True) -> int:
        """
        更新股票列表。

        Args:
            include_all_stocks: True 时同步全市场在市 A 股，并保留指数标记。

        Returns:
            更新的股票数量。
        """
        self._log("更新股票列表...")

        universe = self._ensure_universe()
        hs300 = set(universe.get_hs300())
        zz500 = set(universe.get_zz500())
        zz1000 = set(universe.get_zz1000())

        metadata_by_code: Dict[str, Dict[str, Optional[str]]] = {}
        if include_all_stocks and getattr(universe, "pro", None) is not None:
            try:
                df = universe.pro.stock_basic(
                    exchange="",
                    list_status="L",
                    fields="ts_code,name,industry,market,list_date",
                )
                if df is not None and not df.empty:
                    df = df[~df["name"].str.contains("ST|退", na=False)]
                    if "market" in df.columns:
                        df = df[df["market"].isin(["主板", "创业板", "科创板"])]
                    for row in df.itertuples(index=False):
                        metadata_by_code[row.ts_code] = {
                            "name": row.name,
                            "industry": row.industry,
                            "market": "CN",
                            "list_date": self._normalize_date(row.list_date),
                        }
            except Exception as exc:
                self._log(f"获取全市场股票列表失败，降级为指数成分股: {exc}")

        us_metadata_by_code = self._load_us_metadata()
        metadata_by_code.update(
            {
                ts_code: {**meta, "market": "US"}
                for ts_code, meta in us_metadata_by_code.items()
            }
        )

        all_stocks = set(metadata_by_code) | hs300 | zz500 | zz1000
        if not all_stocks:
            raise RuntimeError("未获取到任何股票列表，无法更新 stock_list")

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT ts_code, name, industry, market, list_date FROM stock_list")
        existing = {
            row[0]: {
                "name": row[1],
                "industry": row[2],
                "market": row[3] or "CN",
                "list_date": self._normalize_date(row[4]),
            }
            for row in cursor.fetchall()
        }

        update_count = 0
        for ts_code in sorted(all_stocks):
            meta = metadata_by_code.get(ts_code, {})
            fallback = existing.get(ts_code, {})

            cursor.execute(
                """
                INSERT OR REPLACE INTO stock_list
                (ts_code, name, industry, market, list_date, is_hs300, is_zz500, is_zz1000, last_update)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_code,
                    meta.get("name") or fallback.get("name"),
                    meta.get("industry") or fallback.get("industry"),
                    meta.get("market") or fallback.get("market") or "CN",
                    meta.get("list_date") or fallback.get("list_date"),
                    1 if ts_code in hs300 else 0,
                    1 if ts_code in zz500 else 0,
                    1 if ts_code in zz1000 else 0,
                    datetime.now().isoformat(),
                ),
            )
            update_count += 1

        conn.commit()
        conn.close()

        self._log(f"股票列表更新完成: {update_count} 只")
        self._log(f"  - 沪深300: {len(hs300)} 只")
        self._log(f"  - 中证500: {len(zz500)} 只")
        self._log(f"  - 中证1000: {len(zz1000)} 只")
        return update_count

    def _load_stock_coverage(
        self,
    ) -> List[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], int]]:
        """加载每只股票的上市日期和现有覆盖区间。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                s.ts_code,
                s.market,
                s.list_date,
                MIN(d.trade_date) AS min_trade_date,
                MAX(d.trade_date) AS max_trade_date,
                COUNT(d.trade_date) AS row_count
            FROM stock_list s
            LEFT JOIN daily_data d ON d.ts_code = s.ts_code
            GROUP BY s.ts_code, s.market, s.list_date
            ORDER BY s.ts_code
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    def plan_download_tasks(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        batch_size: Optional[int] = None,
        market: Optional[str] = None,
    ) -> List[DownloadTask]:
        """
        为目标区间构建下载任务。

        当前策略会补齐区间两端缺口，并与现有边界保留 1 天重叠，用于一致性校验。
        """
        normalized_start = self._normalize_date(start_date)
        normalized_end = self._normalize_date(end_date) or datetime.now().strftime("%Y%m%d")
        if normalized_start is None:
            raise ValueError("start_date 不能为空")
        if normalized_start > normalized_end:
            raise ValueError("start_date 不能晚于 end_date")
        market_filter = self._normalize_market_filter(market)

        tasks: List[DownloadTask] = []
        for ts_code, market, list_date, min_trade_date, max_trade_date, row_count in self._load_stock_coverage():
            normalized_market = (market or "CN").upper()
            if normalized_market not in SUPPORTED_MARKETS:
                continue
            if market_filter is not None and normalized_market != market_filter:
                continue

            effective_start = normalized_start
            normalized_list_date = self._normalize_date(list_date)
            if normalized_list_date is not None and normalized_list_date > effective_start:
                effective_start = normalized_list_date

            if effective_start > normalized_end:
                continue

            normalized_min = self._normalize_date(min_trade_date)
            normalized_max = self._normalize_date(max_trade_date)

            if row_count == 0:
                tasks.append(
                    DownloadTask(
                        ts_code=ts_code,
                        start_date=effective_start,
                        end_date=normalized_end,
                        reason="empty_range",
                        market=normalized_market,
                        list_date=normalized_list_date,
                    )
                )
                continue

            if normalized_min is None or normalized_min > effective_start:
                prefix_end = min(normalized_min or normalized_end, normalized_end)
                gap_days = (
                    self._calendar_gap_days(effective_start, prefix_end)
                    if prefix_end is not None
                    else 0
                )
                if effective_start <= prefix_end and gap_days > BACKFILL_GRACE_DAYS:
                    tasks.append(
                        DownloadTask(
                            ts_code=ts_code,
                            start_date=effective_start,
                            end_date=prefix_end,
                            reason="prefix_fill",
                            market=normalized_market,
                            list_date=normalized_list_date,
                            existing_start=normalized_min,
                            existing_end=normalized_max,
                        )
                    )

            if normalized_max is None or normalized_max < normalized_end:
                suffix_start = max(normalized_max or effective_start, effective_start)
                if suffix_start <= normalized_end:
                    tasks.append(
                        DownloadTask(
                            ts_code=ts_code,
                            start_date=suffix_start,
                            end_date=normalized_end,
                            reason="suffix_fill",
                            market=normalized_market,
                            list_date=normalized_list_date,
                            existing_start=normalized_min,
                            existing_end=normalized_max,
                        )
                    )

        if batch_size is not None:
            tasks = tasks[:batch_size]

        return tasks

    def plan_historical_backfill(
        self,
        years: int = 7,
        anchor_start: Optional[str] = None,
        batch_size: Optional[int] = None,
        market: Optional[str] = None,
    ) -> BackfillPlan:
        """
        基于当前库内最早日期向前回填若干年历史。

        Args:
            years: 需要向前补齐的年数。
            anchor_start: 回填锚点，默认使用库内当前最早交易日。
            batch_size: 仅回填前 N 个任务，便于小批量验证。
        """
        if years <= 0:
            raise ValueError("years 必须为正整数")

        current_start, current_end = self.get_date_range()
        normalized_anchor_start = self._normalize_date(anchor_start) or current_start
        if normalized_anchor_start is None:
            raise ValueError("daily_data 为空，无法基于现有数据制定回填计划")

        normalized_anchor_end = current_end or datetime.now().strftime("%Y%m%d")
        target_start = self._shift_years(normalized_anchor_start, -years)
        market_filter = self._normalize_market_filter(market)
        tasks: List[DownloadTask] = []
        for ts_code, market, list_date, min_trade_date, max_trade_date, row_count in self._load_stock_coverage():
            normalized_market = (market or "CN").upper()
            if normalized_market not in SUPPORTED_MARKETS:
                continue
            if market_filter is not None and normalized_market != market_filter:
                continue

            effective_start = target_start
            normalized_list_date = self._normalize_date(list_date)
            if normalized_list_date is not None and normalized_list_date > effective_start:
                effective_start = normalized_list_date

            normalized_min = self._normalize_date(min_trade_date)
            normalized_max = self._normalize_date(max_trade_date)

            if row_count == 0:
                if effective_start > normalized_anchor_end:
                    continue
                fetch_end = normalized_anchor_end
            else:
                if effective_start > normalized_anchor_start:
                    continue
                if normalized_min is None or normalized_min <= effective_start:
                    continue
                if self._calendar_gap_days(effective_start, normalized_min) <= BACKFILL_GRACE_DAYS:
                    continue
                fetch_end = normalized_min

            task = DownloadTask(
                ts_code=ts_code,
                start_date=effective_start,
                end_date=fetch_end,
                reason="historical_backfill",
                market=normalized_market,
                list_date=normalized_list_date,
                existing_start=normalized_min,
                existing_end=normalized_max,
            )
            if self._is_backfill_exhausted(task):
                continue

            tasks.append(task)

        if batch_size is not None:
            tasks = tasks[:batch_size]

        return BackfillPlan(
            years=years,
            anchor_start=normalized_anchor_start,
            anchor_end=normalized_anchor_end,
            target_start=target_start,
            tasks=tasks,
        )

    def get_stocks_to_download(
        self,
        start_date: str,
        end_date: str,
        batch_size: Optional[int] = None,
        market: Optional[str] = None,
    ) -> List[str]:
        """
        获取需要下载的股票列表。

        兼容旧接口，内部改为基于区间缺口自动构建任务。
        """
        tasks = self.plan_download_tasks(
            start_date,
            end_date,
            batch_size=batch_size,
            market=market,
        )
        seen = set()
        stocks: List[str] = []
        for task in tasks:
            if task.ts_code not in seen:
                seen.add(task.ts_code)
                stocks.append(task.ts_code)
        return stocks

    def plan_missing_stock_downloads(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        batch_size: Optional[int] = None,
        market: Optional[str] = None,
    ) -> List[DownloadTask]:
        """
        为“股票清单里有，但本地尚未下载任何日线”的标的构建补齐任务。

        默认会沿用库内现有行情的最早日期作为起点；若库内还没有任何行情，
        则回退到 2020-01-01，避免把股票列表展示口径和实际行情覆盖脱钩。
        """
        normalized_end = self._normalize_date(end_date) or datetime.now().strftime("%Y%m%d")
        normalized_start = self._normalize_date(start_date)
        if normalized_start is None:
            current_start, _ = self.get_date_range()
            normalized_start = current_start or "20200101"

        tasks = self.plan_download_tasks(
            normalized_start,
            normalized_end,
            batch_size=None,
            market=market,
        )
        missing_tasks = [task for task in tasks if task.reason == "empty_range"]
        if batch_size is not None:
            missing_tasks = missing_tasks[:batch_size]
        return missing_tasks

    def download_missing_stocks(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_workers: int = 3,
        batch_size: Optional[int] = None,
        market: Optional[str] = None,
    ) -> DownloadProgress:
        """补齐股票清单中尚未落地任何日线数据的股票。"""
        tasks = self.plan_missing_stock_downloads(
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            market=market,
        )
        if not tasks:
            self._log("缺失股票补齐: 当前 stock_list 中所有股票都已有本地行情")
            return DownloadProgress(0, 0, [], datetime.now())

        return self._execute_tasks(tasks, max_workers=max_workers, log_label="缺失股票补齐")

    def _load_overlap_frame(self, task: DownloadTask) -> pd.DataFrame:
        conn = self._connect()
        overlap_df = pd.read_sql_query(
            """
            SELECT trade_date, open, high, low, close, volume, amount
            FROM daily_data
            WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
            """,
            conn,
            params=[task.ts_code, task.start_date, task.end_date],
        )
        conn.close()
        return overlap_df

    def _prepare_daily_frame(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "vol": "volume",
            "Date": "trade_date",
            "Datetime": "trade_date",
            "date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Amount": "amount",
        }
        df = raw_df.rename(columns=rename_map).copy()
        if "amount" not in df.columns:
            df["amount"] = 0.0
        expected_columns = {"trade_date", "open", "high", "low", "close", "volume", "amount"}
        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"下载结果缺少字段: {sorted(missing_columns)}")

        if "adj_factor" not in df.columns:
            df["adj_factor"] = np.nan
        if "price_mode" not in df.columns:
            df["price_mode"] = PRICE_MODE_QFQ
        if "data_source" not in df.columns:
            df["data_source"] = None

        ordered_columns = [
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "adj_factor",
            "price_mode",
            "data_source",
        ]
        df = df[ordered_columns]
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False)
        df["trade_date"] = df["trade_date"].str.slice(0, 8)

        for column in CONSISTENCY_FIELDS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")

        df["amount"] = df["amount"].fillna(0.0)
        df = df.dropna(subset=["trade_date", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset=["trade_date"], keep="last")
        df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    def _build_consistency_message(
        self,
        task: DownloadTask,
        overlap_df: pd.DataFrame,
        downloaded_df: pd.DataFrame,
    ) -> str:
        if overlap_df.empty:
            return task.reason

        merged = overlap_df.merge(downloaded_df, on="trade_date", suffixes=("_old", "_new"))
        if merged.empty:
            return f"{task.reason}; overlap=0"

        mismatch_mask = np.zeros(len(merged), dtype=bool)
        for field in CONSISTENCY_FIELDS:
            old_values = merged[f"{field}_old"].to_numpy(dtype=float)
            new_values = merged[f"{field}_new"].to_numpy(dtype=float)
            mismatch_mask |= ~np.isclose(old_values, new_values, equal_nan=True)

        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count > 0:
            return f"{task.reason}; overlap={len(merged)}; mismatched={mismatch_count}; action=replace"
        return f"{task.reason}; overlap={len(merged)}; boundary=verified"

    def _load_us_metadata(self) -> Dict[str, Dict[str, Optional[str]]]:
        """从缓存的美股 universe 中加载基础元数据。"""
        if not US_UNIVERSE_FILE.exists():
            self._log(f"未找到美股股票池缓存: {US_UNIVERSE_FILE}")
            return {}

        try:
            universe = json.loads(US_UNIVERSE_FILE.read_text())
        except Exception as exc:
            self._log(f"读取美股股票池缓存失败: {exc}")
            return {}

        if isinstance(universe, dict) and isinstance(universe.get("all"), list):
            symbols = universe["all"]
        elif isinstance(universe, list):
            symbols = universe
        else:
            self._log("美股股票池缓存格式异常，已跳过 US 补充")
            return {}

        metadata: Dict[str, Dict[str, Optional[str]]] = {}
        for symbol in symbols:
            normalized_symbol = str(symbol).strip().upper()
            if not normalized_symbol:
                continue
            metadata[normalized_symbol] = {
                "name": normalized_symbol,
                "industry": "US",
                "market": "US",
                "list_date": None,
            }
        return metadata

    def _download_cn_raw_data(self, task: DownloadTask) -> pd.DataFrame:
        pro = self._get_tushare_client()
        raw_df = pro.daily(ts_code=task.ts_code, start_date=task.start_date, end_date=task.end_date)
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        adjustment_end = max(task.end_date, task.existing_end or task.end_date)
        adj_df = pro.adj_factor(
            ts_code=task.ts_code,
            start_date=task.start_date,
            end_date=adjustment_end,
        )
        if adj_df is None or adj_df.empty:
            raw_df = raw_df.copy()
            raw_df["adj_factor"] = 1.0
            raw_df["price_mode"] = PRICE_MODE_QFQ
            raw_df["data_source"] = "tushare"
            return raw_df

        raw_df = raw_df.copy()
        raw_df["trade_date"] = raw_df["trade_date"].astype(str)

        adj_df = adj_df[["trade_date", "adj_factor"]].copy()
        adj_df["trade_date"] = adj_df["trade_date"].astype(str)
        adj_df = adj_df.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last")
        anchor_factor = float(adj_df["adj_factor"].iloc[-1])
        if np.isclose(anchor_factor, 0.0):
            raise RuntimeError(f"{task.ts_code} 复权锚点因子异常: {anchor_factor}")

        merged = raw_df.merge(adj_df, on="trade_date", how="left")
        merged["adj_factor"] = merged["adj_factor"].ffill().bfill()
        if merged["adj_factor"].isna().any():
            raise RuntimeError(f"{task.ts_code} 存在缺失复权因子，无法生成前复权价格")

        scale = merged["adj_factor"] / anchor_factor
        for column in ("open", "high", "low", "close"):
            merged[column] = pd.to_numeric(merged[column], errors="coerce") * scale

        merged["price_mode"] = PRICE_MODE_QFQ
        merged["data_source"] = "tushare"
        return merged

    def _download_us_raw_data(self, task: DownloadTask) -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance 未安装，无法下载美股数据")

        start_text = datetime.strptime(task.start_date, "%Y%m%d").strftime("%Y-%m-%d")
        end_text = datetime.strptime(self._next_day(task.end_date), "%Y%m%d").strftime("%Y-%m-%d")

        for attempt in range(3):
            try:
                raw_df = yf.Ticker(task.ts_code).history(
                    start=start_text,
                    end=end_text,
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                )
                if raw_df is None or raw_df.empty:
                    if self._probe_us_recent_data(task.ts_code):
                        if attempt == 2:
                            raise RuntimeError("美股历史区间返回空结果，但近期仍有可用行情")
                        time.sleep(1 + attempt)
                        continue
                    empty_df = pd.DataFrame()
                    empty_df.attrs["confirmed_empty"] = True
                    return empty_df
                raw_df = raw_df.reset_index()
                raw_df["adj_factor"] = np.nan
                raw_df["price_mode"] = PRICE_MODE_QFQ
                raw_df["data_source"] = "yfinance"
                return raw_df
            except Exception as exc:
                message = str(exc).lower()
                if any(pattern in message for pattern in NO_DATA_ERROR_PATTERNS):
                    if self._probe_us_recent_data(task.ts_code):
                        if attempt == 2:
                            raise RuntimeError(f"美股下载异常，但近期探测存在行情: {exc}") from exc
                        time.sleep(1 + attempt)
                        continue
                    empty_df = pd.DataFrame()
                    empty_df.attrs["confirmed_empty"] = True
                    return empty_df
                if attempt == 2:
                    raise RuntimeError(f"美股下载失败: {exc}") from exc
                time.sleep(1 + attempt)

        return pd.DataFrame()

    def _probe_us_recent_data(self, symbol: str) -> bool:
        """用近期行情探针区分“真没数据”和“历史请求偶发空结果”."""
        probe_df = yf.Ticker(symbol).history(
            period="1mo",
            interval="1d",
            auto_adjust=True,
            actions=False,
        )
        return probe_df is not None and not probe_df.empty

    @staticmethod
    def _source_for_market(market: str) -> str:
        normalized_market = (market or "CN").upper()
        if normalized_market == "US":
            return "yfinance"
        return "tushare"

    def _upsert_market_data_config(self, market: str, source: str) -> None:
        """记录当前市场的统一价格口径，方便回测前校验。"""
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO market_data_config
            (market, price_mode, volume_mode, data_source, note, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(market) DO UPDATE SET
                price_mode=excluded.price_mode,
                volume_mode=excluded.volume_mode,
                data_source=excluded.data_source,
                note=excluded.note,
                updated_at=excluded.updated_at
            """,
            (
                market.upper(),
                PRICE_MODE_QFQ,
                VOLUME_MODE_RAW,
                source,
                STANDARDIZATION_NOTE,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def download_task(self, task: DownloadTask) -> bool:
        """
        执行单个下载任务。

        任务区间与现有数据允许 1 天重叠；若重叠日不一致，将以本次下载结果覆盖。
        """
        try:
            market = (task.market or "CN").upper()
            source = self._source_for_market(market)
            if market == "CN":
                raw_df = self._download_cn_raw_data(task)
            elif market == "US":
                raw_df = self._download_us_raw_data(task)
            else:
                raise ValueError(f"暂不支持市场: {market}")

            confirmed_empty = bool(getattr(raw_df, "attrs", {}).get("confirmed_empty"))
            if raw_df is None or raw_df.empty:
                empty_marker = "; confirmed_empty=1" if confirmed_empty else ""
                self._log_download(
                    task.ts_code,
                    task.start_date,
                    task.end_date,
                    0,
                    "failed",
                    f"{task.reason}; market={market}; source={source}{empty_marker}; empty",
                )
                return False

            prepared_df = self._prepare_daily_frame(raw_df)
            if prepared_df.empty:
                self._log_download(
                    task.ts_code,
                    task.start_date,
                    task.end_date,
                    0,
                    "failed",
                    f"{task.reason}; market={market}; source={source}; cleaned_empty",
                )
                return False

            overlap_df = self._load_overlap_frame(task)
            message = self._build_consistency_message(task, overlap_df, prepared_df)
            actual_start = prepared_df["trade_date"].iloc[0]
            actual_end = prepared_df["trade_date"].iloc[-1]
            message = f"{message}; market={market}; source={source}; actual={actual_start}-{actual_end}"

            records = [
                (
                    task.ts_code,
                    row.trade_date,
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    float(row.volume),
                    float(row.amount),
                    None if pd.isna(row.adj_factor) else float(row.adj_factor),
                    row.price_mode,
                    row.data_source,
                )
                for row in prepared_df.itertuples(index=False)
            ]

            conn = self._connect()
            conn.executemany(
                """
                INSERT OR REPLACE INTO daily_data
                (ts_code, trade_date, open, high, low, close, volume, amount, adj_factor, price_mode, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.commit()
            conn.close()
            self._upsert_market_data_config(market, source)

            self._log_download(
                task.ts_code,
                task.start_date,
                task.end_date,
                len(prepared_df),
                "success",
                message,
            )
            return True

        except Exception as exc:
            self._log_download(
                task.ts_code,
                task.start_date,
                task.end_date,
                0,
                "failed",
                f"{task.reason}; {exc}",
            )
            return False

    def download_stock_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        reason: str = "manual",
    ) -> bool:
        """兼容旧接口，执行单只股票单区间下载。"""
        task = DownloadTask(
            ts_code=ts_code,
            start_date=self._normalize_date(start_date) or start_date,
            end_date=self._normalize_date(end_date) or end_date,
            reason=reason,
        )
        return self.download_task(task)

    def _log_download(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        records_count: int,
        status: str,
        message: str = "",
    ) -> None:
        """记录下载日志。"""
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO download_log
            (ts_code, start_date, end_date, records_count, status, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts_code, start_date, end_date, records_count, status, message, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def _is_backfill_exhausted(self, task: DownloadTask) -> bool:
        """判断同一回填任务是否已证明无法拿到更早数据。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT status, message
            FROM download_log
            WHERE ts_code = ? AND start_date = ? AND end_date = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (task.ts_code, task.start_date, task.end_date),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return False

        status, message = row
        expected_source = self._source_for_market(task.market)
        if status == "failed" and message and (
            f"market={task.market.upper()}" in message
            and f"source={expected_source}" in message
            and "confirmed_empty=1" in message
            and (
                message.endswith("; empty")
                or message.endswith("; cleaned_empty")
                or "; no_data" in message
            )
        ):
            return True

        if status != "success" or not message:
            return False

        match = re.search(r"actual=(\d{8})-(\d{8})", message)
        if not match:
            return False

        actual_start = match.group(1)
        return actual_start > task.start_date

    def plan_price_standardization(
        self,
        market: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[DownloadTask]:
        """为已有数据重建统一回测价格序列。"""
        market_filter = self._normalize_market_filter(market)
        tasks: List[DownloadTask] = []
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                s.ts_code,
                s.market,
                MIN(d.trade_date) AS min_trade_date,
                MAX(d.trade_date) AS max_trade_date,
                COUNT(d.trade_date) AS row_count,
                SUM(
                    CASE
                        WHEN d.price_mode = ? AND d.data_source = ?
                        THEN 1 ELSE 0
                    END
                ) AS standardized_rows
            FROM stock_list s
            JOIN daily_data d ON d.ts_code = s.ts_code
            GROUP BY s.ts_code, s.market
            ORDER BY s.ts_code
            """,
            (PRICE_MODE_QFQ, "tushare"),
        )
        rows = cursor.fetchall()
        conn.close()

        for ts_code, stock_market, min_trade_date, max_trade_date, row_count, standardized_rows in rows:
            normalized_market = (stock_market or "CN").upper()
            if normalized_market not in SUPPORTED_MARKETS:
                continue
            if market_filter is not None and normalized_market != market_filter:
                continue
            if row_count == 0 or min_trade_date is None or max_trade_date is None:
                continue
            if normalized_market == "US":
                continue
            if int(standardized_rows or 0) >= int(row_count):
                continue

            tasks.append(
                DownloadTask(
                    ts_code=ts_code,
                    start_date=min_trade_date,
                    end_date=max_trade_date,
                    reason="price_standardization",
                    market=normalized_market,
                    existing_start=min_trade_date,
                    existing_end=max_trade_date,
                )
            )

        if batch_size is not None:
            tasks = tasks[:batch_size]
        return tasks

    def standardize_price_series(
        self,
        market: Optional[str] = None,
        max_workers: int = 1,
        batch_size: Optional[int] = None,
    ) -> DownloadProgress:
        """
        把已有价格统一成可回测口径。

        当前策略:
        - CN: 重建为前复权 OHLC
        - US: 现有序列已是 adjusted OHLC，仅补充元数据
        """
        market_filter = self._normalize_market_filter(market)
        if market_filter in (None, "US"):
            conn = self._connect()
            conn.execute(
                """
                UPDATE daily_data
                SET price_mode = ?, data_source = COALESCE(data_source, ?)
                WHERE ts_code IN (SELECT ts_code FROM stock_list WHERE market = 'US')
                """,
                (PRICE_MODE_QFQ, "yfinance"),
            )
            conn.commit()
            conn.close()
            self._upsert_market_data_config("US", "yfinance")
            if market_filter == "US":
                return DownloadProgress(0, 0, [], datetime.now())

        tasks = self.plan_price_standardization(market="CN" if market_filter is None else market_filter, batch_size=batch_size)
        if not tasks:
            return DownloadProgress(0, 0, [], datetime.now())

        return self._execute_tasks(
            tasks,
            max_workers=max_workers,
            log_label="价格标准化",
        )

    def _execute_tasks(
        self,
        tasks: List[DownloadTask],
        max_workers: int = 5,
        log_label: str = "批量下载",
    ) -> DownloadProgress:
        if not tasks:
            self._log(f"{log_label}: 没有需要处理的任务")
            return DownloadProgress(0, 0, [], datetime.now())

        self.progress = DownloadProgress(
            total_stocks=len(tasks),
            completed_stocks=0,
            failed_stocks=[],
            last_update=datetime.now(),
        )

        self._log(f"{log_label}: 任务数 {len(tasks)}，并行线程 {max_workers}")

        def handle_result(task: DownloadTask, success: bool) -> None:
            if success:
                self.progress.completed_stocks += 1
            else:
                self.progress.failed_stocks.append(task.ts_code)

            processed = self.progress.completed_stocks + len(self.progress.failed_stocks)
            if processed % 20 == 0 or processed == self.progress.total_stocks:
                self._log(
                    f"{log_label} 进度: {self.progress.progress_pct:.1f}% "
                    f"({processed}/{self.progress.total_stocks})"
                )

        if max_workers <= 1:
            for task in tasks:
                handle_result(task, self.download_task(task))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(self.download_task, task): task for task in tasks}
                for future in as_completed(future_map):
                    task = future_map[future]
                    try:
                        handle_result(task, future.result())
                    except Exception as exc:
                        self._log(f"下载异常 {task.ts_code}: {exc}")
                        handle_result(task, False)

        self.progress.last_update = datetime.now()
        self._log(
            f"{log_label} 完成: 成功 {self.progress.completed_stocks}, "
            f"失败 {len(self.progress.failed_stocks)}"
        )
        return self.progress

    def batch_download(
        self,
        start_date: str = "20200101",
        end_date: Optional[str] = None,
        max_workers: int = 5,
        batch_size: int = 100,
        market: Optional[str] = None,
    ) -> DownloadProgress:
        """
        批量下载指定区间数据。

        Args:
            start_date: 开始日期。
            end_date: 结束日期，默认今天。
            max_workers: 并行线程数。
            batch_size: 最多执行多少个任务。
        """
        normalized_end = self._normalize_date(end_date) or datetime.now().strftime("%Y%m%d")
        tasks = self.plan_download_tasks(
            start_date,
            normalized_end,
            batch_size=batch_size,
            market=market,
        )
        if not tasks:
            self._log("所有股票数据已覆盖目标区间")
            return DownloadProgress(0, 0, [], datetime.now())

        return self._execute_tasks(tasks, max_workers=max_workers, log_label="区间下载")

    def backfill_history(
        self,
        years: int = 7,
        max_workers: int = 1,
        batch_size: Optional[int] = None,
        anchor_start: Optional[str] = None,
        market: Optional[str] = None,
    ) -> Tuple[BackfillPlan, DownloadProgress]:
        """
        基于当前已有数据向前回填历史。

        Returns:
            (回填计划, 执行进度)
        """
        plan = self.plan_historical_backfill(
            years=years,
            anchor_start=anchor_start,
            batch_size=batch_size,
            market=market,
        )
        if not plan.tasks:
            self._log(
                f"历史回填已完成: 当前最早日期 {plan.anchor_start}，目标起点 {plan.target_start}"
            )
            return plan, DownloadProgress(0, 0, [], datetime.now())

        progress = self._execute_tasks(
            plan.tasks,
            max_workers=max_workers,
            log_label=f"历史回填({plan.target_start} -> {plan.anchor_start})",
        )
        return plan, progress

    def get_data(
        self,
        ts_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        从数据库查询数据。

        Args:
            ts_codes: 股票代码列表，None 表示全部。
            start_date: 开始日期。
            end_date: 结束日期。

        Returns:
            查询结果 DataFrame。
        """
        conn = self._connect()

        query = "SELECT * FROM daily_data WHERE 1=1"
        params: List[str] = []

        if ts_codes:
            placeholders = ",".join(["?" for _ in ts_codes])
            query += f" AND ts_code IN ({placeholders})"
            params.extend(ts_codes)

        normalized_start = self._normalize_date(start_date)
        normalized_end = self._normalize_date(end_date)

        if normalized_start:
            query += " AND trade_date >= ?"
            params.append(normalized_start)

        if normalized_end:
            query += " AND trade_date <= ?"
            params.append(normalized_end)

        query += " ORDER BY ts_code, trade_date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_statistics(self) -> Dict[str, object]:
        """获取数据库统计信息。"""
        conn = self._connect()
        cursor = conn.cursor()

        stats: Dict[str, object] = {}
        cursor.execute("SELECT COUNT(*) FROM stock_list")
        stats["total_stocks"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM stock_list WHERE is_hs300=1")
        stats["hs300_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM stock_list WHERE is_zz500=1")
        stats["zz500_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM stock_list WHERE is_zz1000=1")
        stats["zz1000_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM daily_data")
        stats["total_records"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT ts_code) FROM daily_data")
        stats["stocks_with_data"] = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_data")
        min_date, max_date = cursor.fetchone()
        stats["date_range"] = f"{min_date} 至 {max_date}" if min_date else "N/A"

        cursor.execute(
            """
            SELECT market, price_mode, volume_mode, data_source
            FROM market_data_config
            ORDER BY market
            """
        )
        stats["price_config"] = {
            row[0]: {
                "price_mode": row[1],
                "volume_mode": row[2],
                "data_source": row[3],
            }
            for row in cursor.fetchall()
        }

        conn.close()
        return stats


def init_database(db_path: Optional[str] = None) -> StockDatabase:
    """初始化数据库。"""
    return StockDatabase(db_path=db_path)


def download_all_data(
    start_date: str = "20200101",
    end_date: Optional[str] = None,
    max_workers: int = 5,
    batch_size: int = 100,
) -> DownloadProgress:
    """下载指定区间的所有数据。"""
    db = StockDatabase()
    db.update_stock_list()
    return db.batch_download(
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Stock Database Manager - 测试")
    print("=" * 80)

    db = StockDatabase()
    db.update_stock_list()

    stats = db.get_statistics()
    print("\n数据库统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n测试回填计划...")
    plan = db.plan_historical_backfill(years=1, batch_size=5)
    print(f"  当前最早日期: {plan.anchor_start}")
    print(f"  目标起点: {plan.target_start}")
    print(f"  任务数: {len(plan.tasks)}")

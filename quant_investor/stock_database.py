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
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.data.universe.cn_universe import StockUniverse
from quant_investor.logger import get_logger
from quant_investor.stock_database_download import StockDatabaseDownloadMixin
from quant_investor.stock_database_types import (
    BACKFILL_GRACE_DAYS,
    CONSISTENCY_FIELDS,
    PROJECT_ROOT,
    SUPPORTED_MARKETS,
    BackfillPlan,
    DownloadProgress,
    DownloadTask,
    default_cache_dir,
    default_db_path,
)

_default_db_path = default_db_path
_default_cache_dir = default_cache_dir
DEFAULT_TUSHARE_URL = config.TUSHARE_URL

__all__ = [
    "BackfillPlan",
    "CONSISTENCY_FIELDS",
    "DEFAULT_TUSHARE_URL",
    "DownloadProgress",
    "DownloadTask",
    "PROJECT_ROOT",
    "SUPPORTED_MARKETS",
    "StockDatabase",
    "download_all_data",
    "init_database",
]


class StockDatabase(StockDatabaseDownloadMixin):
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

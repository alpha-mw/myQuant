"""Shared contracts and defaults for the stock database manager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from quant_investor.config import config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TUSHARE_URL = config.TUSHARE_URL
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


def default_db_path() -> Path:
    path = Path(config.DB_PATH or "data/stock_database.db")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def default_cache_dir() -> Path:
    return default_db_path().parent / "cache"


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
    tasks: list[DownloadTask]

    @property
    def stock_count(self) -> int:
        return len({task.ts_code for task in self.tasks})


@dataclass
class DownloadProgress:
    """下载进度。"""

    total_stocks: int
    completed_stocks: int
    failed_stocks: list[str]
    last_update: datetime

    @property
    def progress_pct(self) -> float:
        if self.total_stocks == 0:
            return 0.0
        return (self.completed_stocks / self.total_stocks) * 100


__all__ = [
    "BACKFILL_GRACE_DAYS",
    "CONSISTENCY_FIELDS",
    "DEFAULT_TUSHARE_URL",
    "NO_DATA_ERROR_PATTERNS",
    "PRICE_MODE_QFQ",
    "PROJECT_ROOT",
    "STANDARDIZATION_NOTE",
    "SUPPORTED_MARKETS",
    "US_UNIVERSE_FILE",
    "VOLUME_MODE_RAW",
    "BackfillPlan",
    "DownloadProgress",
    "DownloadTask",
    "default_cache_dir",
    "default_db_path",
]

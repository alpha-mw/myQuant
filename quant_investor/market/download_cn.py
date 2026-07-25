#!/usr/bin/env python3
"""
Download Full China A-Share Market Data - 下载完整A股市场数据

下载所有大中小盘股的3年历史数据
- 大盘股: 沪深300 (300只)
- 中盘股: 中证500 (500只)
- 小盘股: 中证1000 (1000只)
总计: 1800只股票
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Any, List, Dict, Optional, Set, Mapping
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.data_quality_contract import (
    DataQualityIssue,
    ISSUE_SEVERITY_WARNING,
    make_issue_id,
)
from quant_investor.fetch_cn_index_components import get_all_components, save_components
from quant_investor.logger import get_logger
from quant_investor.market.cn_resolver import CNUniverseResolver
from quant_investor.market.cn_nontrading_evidence import (
    canonical_json_sha256,
    dataframe_sha256,
    symbol_set_sha256,
)
from quant_investor.market.config import get_market_settings
from quant_investor.market.cn_symbol_status import evaluate_symbol_local_status, CNSymbolLocalStatusResult
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.market_data_store import MarketDataStore
from quant_investor.market.tushare_data_cleaning import (
    CLEANING_STATUS_FAIL,
    PARQUET_STATUS_SKIPPED,
    TushareStorageOptimizationConfig,
    clean_tushare_dataframe_to_file,
)

try:
    import tushare as ts
except ImportError:
    ts = None

# Tushare配置
TUSHARE_TOKEN = config.TUSHARE_TOKEN
TUSHARE_URL = config.TUSHARE_URL
EASTMONEY_SOURCE_SYSTEM = "eastmoney.push2his.kline"
_logger = get_logger("download_cn", verbose=False)


def _normalize_trade_date_key(value: Any) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _trade_date_key_to_iso(value: Any) -> str:
    key = _normalize_trade_date_key(value)
    if len(key) != 8:
        return ""
    return f"{key[:4]}-{key[4:6]}-{key[6:8]}"


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def _eastmoney_secid(symbol: str) -> str:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return ""
    code = normalized.split(".", 1)[0]
    suffix = normalized.rsplit(".", 1)[-1] if "." in normalized else ""
    if not code:
        return ""
    if suffix == "SH" or (not suffix and code.startswith(("5", "6", "9"))):
        return f"1.{code}"
    if suffix in {"SZ", "BJ"} or (not suffix and code.startswith(("0", "1", "2", "3", "4", "8"))):
        return f"0.{code}"
    return ""


def _fetch_eastmoney_json(url: str, retries: int = 3, timeout: int = 12) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Eastmoney kline request failed after {retries} attempts: {last_error}")


def fetch_eastmoney_daily_batch_frame(
    symbols: List[str] | tuple[str, ...],
    trade_date: str,
    *,
    fetch_json: Any = _fetch_eastmoney_json,
) -> pd.DataFrame:
    """Fetch target-date daily OHLCV rows from Eastmoney push2his.kline.

    Eastmoney amount is normalized from yuan to Tushare's thousand-yuan
    ``daily.amount`` convention before rows enter the maintenance pipeline.
    """
    target_key = _normalize_trade_date_key(trade_date)
    if not target_key:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for symbol in symbols or []:
        normalized_symbol = str(symbol or "").strip().upper()
        secid = _eastmoney_secid(normalized_symbol)
        if not secid:
            continue
        params = {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "0",
            "beg": target_key,
            "end": target_key,
        }
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode(params)
        payload = fetch_json(url)
        klines = ((payload.get("data") or {}).get("klines") or [])
        for item in klines:
            parts = str(item).split(",")
            if len(parts) < 7:
                continue
            row_key = _normalize_trade_date_key(parts[0])
            if row_key != target_key:
                continue
            open_price = _parse_float(parts[1])
            close = _parse_float(parts[2])
            high = _parse_float(parts[3])
            low = _parse_float(parts[4])
            volume = _parse_float(parts[5])
            amount_yuan = _parse_float(parts[6])
            pct_chg = _parse_float(parts[8]) if len(parts) > 8 else None
            change = _parse_float(parts[9]) if len(parts) > 9 else None
            if None in {open_price, close, high, low}:
                continue
            rows.append(
                {
                    "ts_code": normalized_symbol,
                    "trade_date": _trade_date_key_to_iso(row_key),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "pre_close": close - change if close is not None and change is not None else None,
                    "change": change,
                    "pct_chg": pct_chg,
                    "vol": volume,
                    "amount": amount_yuan / 1000.0 if amount_yuan is not None else None,
                    "data_source": EASTMONEY_SOURCE_SYSTEM,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


class CNFullMarketDownloader:
    """A股全市场数据下载器"""

    STAGED_NONCANONICAL_SAFE = False
    SUPPORTED_CATEGORIES = ("full_a", "hs300", "zz500", "zz1000")
    REQUESTS_PER_STOCK = 2
    REQUESTS_PER_MINUTE_BUDGET = config.TUSHARE_RATE_LIMIT_PER_MIN
    
    def __init__(self, 
                 data_dir: str | None = None,
                 years: int = 3,
                 max_workers: int = 5,
                 batch_size: int = 50):
        """
        初始化下载器
        
        Args:
            data_dir: 数据保存目录
            years: 下载年数
            max_workers: 并行下载线程数
            batch_size: 每批处理的股票数
        """
        resolved_data_dir = data_dir or get_market_settings("CN").data_dir
        self.data_dir = resolved_data_dir
        self.data_root = self._resolve_data_root(resolved_data_dir)
        self.components_file = str(self._default_components_file())
        self.years = years
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # 创建分层目录
        self.dirs = {
            'hs300': f"{self.data_dir}/hs300",
            'zz500': f"{self.data_dir}/zz500",
            'zz1000': f"{self.data_dir}/zz1000",
            'other': f"{self.data_dir}/other",
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        self.resolver = CNUniverseResolver(data_dir=self.data_dir, directories=self.dirs)
        self.market_reader = MarketDataReader(market="CN", data_root=self.data_root)
        self.market_store = MarketDataStore(market="CN", data_root=self.data_root)
        self.last_resolver_trace: dict[str, Any] = self.resolver.snapshot()
        self._full_a_write_category_by_symbol: dict[str, str] = {}
        self.freshness_mode = self._normalize_freshness_mode(config.CN_FRESHNESS_MODE)
        self.coverage_threshold = self._clamp_ratio(
            self._safe_float(config.CN_FRESHNESS_COVERAGE_THRESHOLD, 0.95)
        )
        self.strict_early_stop_sample_size = max(
            self._safe_int(config.CN_STRICT_EARLY_STOP_SAMPLE_SIZE, 10),
            1,
        )
        self.strict_early_stop_stale_ratio = self._clamp_ratio(
            self._safe_float(config.CN_STRICT_EARLY_STOP_STALE_RATIO, 0.80)
        )
        
        # 计算日期范围
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=years*365 + 30)
        self._download_data_quality_issues: list[DataQualityIssue] = []
        
        # 初始化 Tushare（内存模式，不落盘 token）
        self.pro = None
        if ts is not None:
            self.pro = create_tushare_pro(ts, TUSHARE_TOKEN, TUSHARE_URL)
        if self.pro is not None:
            self.strict_trade_date, self.stable_trade_date = self._resolve_trade_date_targets()
        else:
            local_trade_date = self._resolve_latest_trade_date_from_local_cache()
            if not local_trade_date:
                raise RuntimeError("tushare 未安装且本地没有可用的 A 股历史数据缓存")
            self.strict_trade_date = local_trade_date
            self.stable_trade_date = local_trade_date
        self.latest_trade_date = self._default_target_trade_date()
        self._latest_suspended_symbols_cache: dict[str, Set[str]] = {}
        self._suspend_query_run_id = (
            "suspend-d-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + f"-{os.getpid()}"
        )
        self._active_listing_dates_cache: dict[str, str] | None = None
        
        # 统计信息
        self.stats = {
            'total': 0,
            'updated': 0,
            'cached': 0,
            'stale_cached': 0,
            'failed': 0,
        }

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp_ratio(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _normalize_freshness_mode(value: Any) -> str:
        mode = str(value or "stable").strip().lower()
        if mode not in {"stable", "strict"}:
            return "stable"
        return mode

    def _default_target_trade_date(self) -> str:
        if self.freshness_mode == "strict":
            return self.strict_trade_date
        return self.stable_trade_date

    def _default_components_file(self) -> Path:
        data_root = Path(self.data_dir).expanduser()
        return data_root.parent / "cn_universe" / "cn_index_components.json"

    @staticmethod
    def _resolve_data_root(data_dir: str) -> Path:
        path = Path(data_dir).expanduser()
        if path.name == "cn_market_full":
            return path.parent
        if path.name in {"full_a", "hs300", "zz500", "zz1000", "other"}:
            return path.parent.parent
        return path

    @staticmethod
    def _normalize_allowed_symbols(symbols: Optional[List[str] | Set[str]]) -> Set[str]:
        """标准化允许跳过完整性检查的股票列表。"""
        if not symbols:
            return set()
        normalized = set()
        for symbol in symbols:
            if symbol and str(symbol).strip():
                normalized.add(str(symbol).strip().upper())
        return normalized

    def _record_data_quality_exception(
        self,
        *,
        func_name: str,
        exc: Exception,
        issue_type: str,
        symbol: str = "",
        as_of: str | None = None,
        field_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DataQualityIssue:
        """Record a swallowed download exception without changing control flow."""
        normalized_symbol = str(symbol or "").strip().upper()
        effective_as_of = str(
            as_of
            or getattr(self, "latest_trade_date", "")
            or getattr(self, "strict_trade_date", "")
            or ""
        ).strip()
        source = f"download_cn.{func_name}"
        issue = DataQualityIssue(
            issue_id=make_issue_id(
                symbol=normalized_symbol,
                market="CN",
                as_of=effective_as_of,
                issue_type=issue_type,
                field_name=field_name,
            ),
            symbol=normalized_symbol,
            market="CN",
            as_of=effective_as_of,
            field_name=field_name,
            issue_type=issue_type,
            severity=ISSUE_SEVERITY_WARNING,
            message=str(exc),
            source=source,
            metadata={
                **dict(metadata or {}),
                "exception_type": exc.__class__.__name__,
            },
        )
        if not hasattr(self, "_download_data_quality_issues"):
            self._download_data_quality_issues = []
        self._download_data_quality_issues.append(issue)
        _logger.debug("%s swallowed exception: %s", source, exc)
        return issue

    def _append_download_data_quality_issues(self, report: Dict[str, Any]) -> None:
        existing_ids = {
            str(issue.get("issue_id") or "")
            for issue in report.get("data_quality_issues", [])
            if isinstance(issue, Mapping)
        }
        for issue in getattr(self, "_download_data_quality_issues", []):
            payload = issue.to_dict()
            issue_id = str(payload.get("issue_id") or "")
            if issue_id and issue_id in existing_ids:
                continue
            report.setdefault("data_quality_issues", []).append(payload)
            if issue_id:
                existing_ids.add(issue_id)

    @classmethod
    def _normalize_categories(cls, categories: Optional[List[str]]) -> List[str]:
        """标准化待处理分类列表。"""
        if not categories:
            return ["full_a"]

        normalized: List[str] = []
        for category in categories:
            key = str(category).strip().lower()
            if key in {"all", "full", "all_a", "full_market"}:
                key = "full_a"
            if key not in cls.SUPPORTED_CATEGORIES:
                raise ValueError(f"不支持的分类: {category}")
            if key not in normalized:
                normalized.append(key)
        return normalized

    def _trade_date_cache_path(self) -> Path:
        """On-disk cache for (strict, stable) trade date pair, keyed by calendar date."""
        return Path(self.data_dir) / ".cache" / ".trade_date_cache.json"

    def _suspend_cache_path(self, trade_date: str) -> Path:
        """On-disk cache for suspend_d results for a given trade date."""
        return Path(self.data_dir) / ".cache" / f".suspend_{trade_date}.json"

    def _freshness_index_path(self) -> Path:
        """On-disk freshness index: symbol -> latest_trade_date (YYYYMMDD)."""
        return Path(self.data_dir) / ".cache" / "freshness_index.json"

    def _load_active_listing_dates(self) -> dict[str, str]:
        """Load current active-listing dates as {SYMBOL: YYYYMMDD}."""
        if self._active_listing_dates_cache is not None:
            return self._active_listing_dates_cache
        if bool(getattr(config, "PIT_UNIVERSE_ENABLED", False)):
            try:
                from quant_investor.market.pit_universe import PITUniverseStore

                records = PITUniverseStore.from_config().load_latest_records()
                listing_dates = {
                    record.symbol: record.list_date
                    for record in records
                    if record.symbol and record.list_date
                }
                if listing_dates:
                    self._active_listing_dates_cache = listing_dates
                    return self._active_listing_dates_cache
                if bool(getattr(config, "PIT_UNIVERSE_REQUIRED", False)):
                    self._active_listing_dates_cache = {}
                    return self._active_listing_dates_cache
            except Exception as exc:
                self._record_data_quality_exception(
                    func_name="_load_active_listing_dates",
                    exc=exc,
                    issue_type="pit_universe_listing_exception",
                    metadata={"source_root": getattr(config, "PIT_UNIVERSE_SOURCE_ROOT", "")},
                )
                if bool(getattr(config, "PIT_UNIVERSE_REQUIRED", False)):
                    self._active_listing_dates_cache = {}
                    return self._active_listing_dates_cache
        if self.pro is None:
            self._active_listing_dates_cache = {}
            return self._active_listing_dates_cache

        try:
            stock_basic = self.pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,list_date",
            )
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_load_active_listing_dates",
                exc=exc,
                issue_type="stock_basic_listing_exception",
                metadata={"list_status": "L"},
            )
            stock_basic = None

        if stock_basic is None or stock_basic.empty:
            self._active_listing_dates_cache = {}
            return self._active_listing_dates_cache

        listing_dates: dict[str, str] = {}
        for row in stock_basic.itertuples(index=False):
            symbol = str(getattr(row, "ts_code", "") or "").strip().upper()
            list_date = str(getattr(row, "list_date", "") or "").strip()
            if symbol and list_date:
                listing_dates[symbol] = list_date
        self._active_listing_dates_cache = listing_dates
        return self._active_listing_dates_cache

    def _pre_listing_symbols_for_target(
        self,
        *,
        components: Dict[str, Any],
        target_categories: List[str],
        target_trade_date: str,
    ) -> dict[str, str]:
        """Return symbols listed after *target_trade_date*.

        Those symbols should not be treated as missing/stale for the target day
        because they were not yet part of the tradable universe.
        """
        listing_dates = self._load_active_listing_dates()
        if not listing_dates:
            return {}

        relevant_symbols: Set[str] = set()
        for category in target_categories:
            for symbol in components.get(category, []) or []:
                normalized = str(symbol or "").strip().upper()
                if normalized:
                    relevant_symbols.add(normalized)

        pre_listing_symbols: dict[str, str] = {}
        for symbol in relevant_symbols:
            list_date = listing_dates.get(symbol, "")
            if list_date and list_date > target_trade_date:
                pre_listing_symbols[symbol] = list_date
        return pre_listing_symbols

    def _resolve_trade_date_targets(self) -> tuple[str, str]:
        """解析严格目标日与稳定目标日（带当日磁盘缓存）。"""
        today = datetime.now().strftime("%Y%m%d")
        cache_path = self._trade_date_cache_path()
        # ── disk cache hit ──
        try:
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload.get("cached_on") == today:
                    return str(payload["strict"]), str(payload["stable"])
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_resolve_trade_date_targets",
                exc=exc,
                issue_type="trade_date_cache_read_exception",
                as_of=today,
                metadata={"cache_path": str(cache_path)},
            )
            pass

        strict_fallback = self.end_date.strftime('%Y%m%d')
        stable_fallback = (self.end_date - timedelta(days=1)).strftime('%Y%m%d')
        try:
            start = (self.end_date - timedelta(days=30)).strftime('%Y%m%d')
            end = self.end_date.strftime('%Y%m%d')
            cal = self.pro.trade_cal(exchange='SSE', start_date=start, end_date=end, is_open='1')
            if cal is None or cal.empty or 'cal_date' not in cal.columns:
                return strict_fallback, stable_fallback

            open_days = sorted(str(value) for value in cal['cal_date'].dropna().astype(str).tolist())
            if not open_days:
                return strict_fallback, stable_fallback

            strict_trade_date = open_days[-1]
            stable_trade_date = open_days[-2] if len(open_days) >= 2 else strict_trade_date

            # ── persist to disk cache ──
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {"cached_on": today, "strict": strict_trade_date, "stable": stable_trade_date},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                self._record_data_quality_exception(
                    func_name="_resolve_trade_date_targets",
                    exc=exc,
                    issue_type="trade_date_cache_write_exception",
                    as_of=strict_trade_date,
                    metadata={"cache_path": str(cache_path)},
                )
                pass

            return strict_trade_date, stable_trade_date
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_resolve_trade_date_targets",
                exc=exc,
                issue_type="trade_calendar_exception",
                as_of=strict_fallback,
            )
            return strict_fallback, stable_fallback

    def _fetch_trade_date_close_probe(self, target_trade_date: str) -> tuple[pd.DataFrame, str, str]:
        """Fetch all-market close availability for a single trade date.

        The maintenance target needs full OHLCV from ``daily``.  For deciding
        whether the strict same-day close has been published at all, a
        date-scoped all-market probe is cheaper and more accurate than sampling
        per-symbol history requests.
        """
        if self.pro is None:
            return pd.DataFrame(), "", "provider_unavailable"

        daily_error = ""
        daily_empty = False
        try:
            df = self.pro.daily(
                trade_date=target_trade_date,
                fields="ts_code,trade_date,close",
            )
            if df is not None and not df.empty:
                return df, "daily", ""
            daily_empty = True
        except TypeError as exc:
            # Some test doubles and older SDK wrappers do not accept fields.
            daily_error = str(exc)
            try:
                df = self.pro.daily(trade_date=target_trade_date)
                if df is not None and not df.empty:
                    return df, "daily", ""
                daily_empty = True
            except Exception as retry_exc:
                daily_error = str(retry_exc)
        except Exception as exc:
            daily_error = str(exc)

        try:
            df = self.pro.daily_basic(
                trade_date=target_trade_date,
                fields="ts_code,trade_date,close",
            )
            if df is not None and not df.empty:
                return df, "daily_basic", ""
        except AttributeError:
            if daily_empty:
                return pd.DataFrame(), "daily", "empty"
            return pd.DataFrame(), "", daily_error or "daily_basic_unavailable"
        except TypeError as exc:
            basic_error = str(exc)
            try:
                df = self.pro.daily_basic(trade_date=target_trade_date)
                if df is not None and not df.empty:
                    return df, "daily_basic", ""
            except Exception as retry_exc:
                basic_error = str(retry_exc)
            if daily_empty:
                return pd.DataFrame(), "daily", "empty"
            return pd.DataFrame(), "", daily_error or basic_error
        except Exception as exc:
            if daily_empty:
                return pd.DataFrame(), "daily", "empty"
            return pd.DataFrame(), "", daily_error or str(exc)

        if daily_empty:
            return pd.DataFrame(), "daily", "empty"
        return pd.DataFrame(), "", daily_error or "empty"

    def _probe_strict_same_day_close_availability(
        self,
        *,
        components: Dict[str, Any],
        target_categories: List[str],
    ) -> Dict[str, Any]:
        if (
            self.freshness_mode != "strict"
            or self.strict_trade_date == self.stable_trade_date
        ):
            return {"applicable": False}

        expected_symbols = {
            str(symbol or "").strip().upper()
            for category in target_categories
            for symbol in components.get(category, []) or []
            if str(symbol or "").strip().upper()
        }
        expected_count = len(expected_symbols)
        if expected_count == 0:
            return {"applicable": False, "reason": "empty_universe"}

        df, source, error = self._fetch_trade_date_close_probe(self.strict_trade_date)
        if df is None or df.empty:
            if source != "daily" and error and error != "empty":
                return {
                    "applicable": True,
                    "source": source,
                    "trade_date": self.strict_trade_date,
                    "available": None,
                    "coverage_ratio": 0.0,
                    "available_count": 0,
                    "expected_count": expected_count,
                    "reason": error,
                }
            return {
                "applicable": True,
                "source": source,
                "trade_date": self.strict_trade_date,
                "available": False,
                "coverage_ratio": 0.0,
                "available_count": 0,
                "expected_count": expected_count,
                "reason": error or "empty",
            }

        probe = df.copy()
        if "trade_date" in probe.columns:
            probe["trade_date"] = probe["trade_date"].astype(str).str.replace("-", "", regex=False)
            probe = probe[probe["trade_date"] == self.strict_trade_date]
        if "ts_code" not in probe.columns:
            return {
                "applicable": True,
                "source": source,
                "trade_date": self.strict_trade_date,
                "available": None,
                "coverage_ratio": 0.0,
                "available_count": 0,
                "expected_count": expected_count,
                "reason": "missing_ts_code",
            }
        if "close" in probe.columns:
            probe = probe[pd.to_numeric(probe["close"], errors="coerce").notna()]

        available_symbols = {
            str(symbol or "").strip().upper()
            for symbol in probe["ts_code"].dropna().tolist()
        }
        matched_count = len(expected_symbols & available_symbols)
        coverage_ratio = matched_count / expected_count if expected_count else 1.0
        return {
            "applicable": True,
            "source": source,
            "trade_date": self.strict_trade_date,
            "available": coverage_ratio >= self.coverage_threshold,
            "coverage_ratio": coverage_ratio,
            "available_count": matched_count,
            "expected_count": expected_count,
            "reason": "",
        }

    def _resolve_latest_trade_date_from_local_cache(self) -> str:
        """Infer latest local trade date from the strict Parquet pointer."""
        try:
            return self.market_reader.latest_trade_date("full_a")
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_resolve_latest_trade_date_from_local_cache",
                exc=exc,
                issue_type="latest_trade_date_cache_exception",
            )
            return ""

    def _build_local_symbol_universe(self) -> Dict[str, List[str]]:
        """从本地 Parquet serving 文件构建不依赖 Tushare 的组件字典。"""
        try:
            full_a_symbols = self.market_reader.list_symbols("full_a")
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_build_local_symbol_universe",
                exc=exc,
                issue_type="local_symbol_universe_exception",
            )
            full_a_symbols = []
        category_lists: Dict[str, Set[str]] = {category: set() for category in self.SUPPORTED_CATEGORIES}
        category_lists["full_a"].update(full_a_symbols)

        result = {
            "full_a": full_a_symbols,
            "full_market": full_a_symbols,
            "all_a": full_a_symbols,
            "all": full_a_symbols,
            "hs300": sorted(category_lists.get("hs300", set())),
            "zz500": sorted(category_lists.get("zz500", set())),
            "zz1000": sorted(category_lists.get("zz1000", set())),
            "stats": {
                "full_a": len(full_a_symbols),
                "hs300": len(category_lists.get("hs300", set())),
                "zz500": len(category_lists.get("zz500", set())),
                "zz1000": len(category_lists.get("zz1000", set())),
                "total_unique": len(full_a_symbols),
            },
            "fetch_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "resolver": self.resolver.snapshot(),
        }
        return result

    def _collect_blocking_symbols(self, category_report: Dict[str, Any]) -> List[str]:
        symbols = [
            item['symbol']
            for item in category_report.get('blocking_stale_symbols', [])
        ]
        symbols.extend(category_report.get('blocking_missing_symbols', []))
        symbols.extend(
            item['symbol']
            for item in category_report.get('blocking_unreadable_symbols', [])
        )
        return list(dict.fromkeys(symbols))

    def _refresh_full_a_write_categories(self, components: Optional[Dict[str, Any]]) -> None:
        if not components:
            return
        category_by_symbol: dict[str, str] = {}
        for category in ("hs300", "zz500", "zz1000"):
            for symbol in components.get(category, []) or []:
                normalized = str(symbol or "").strip().upper()
                if normalized and normalized not in category_by_symbol:
                    category_by_symbol[normalized] = category
        for symbol in components.get("full_a", []) or []:
            normalized = str(symbol or "").strip().upper()
            if normalized and normalized not in category_by_symbol:
                category_by_symbol[normalized] = "other"
        self._full_a_write_category_by_symbol = category_by_symbol

    def _ensure_full_a_write_categories(self) -> None:
        if self._full_a_write_category_by_symbol:
            return
        try:
            components = self.load_components()
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_ensure_full_a_write_categories",
                exc=exc,
                issue_type="component_load_exception",
            )
            return
        self._refresh_full_a_write_categories(components)

    def _resolve_full_a_write_path(self, symbol: str, resolved_path: str = "") -> Path:
        if resolved_path:
            return Path(resolved_path)
        normalized = str(symbol or "").strip().upper()
        if normalized and not self._full_a_write_category_by_symbol:
            self._ensure_full_a_write_categories()
        category = self._full_a_write_category_by_symbol.get(normalized, "other")
        return Path(self.dirs.get(category, self.dirs["other"])) / f"{normalized}.csv"

    def _resolve_target_categories(self, components: Dict[str, Any], categories: Optional[List[str]]) -> List[str]:
        if categories is None:
            target_categories = [
                category
                for category in self.SUPPORTED_CATEGORIES
                if category in components and isinstance(components.get(category), list)
            ]
            if not target_categories:
                target_categories = ["full_a"]
            return target_categories
        return self._normalize_categories(categories)

    def _evaluate_symbol_local_status_for_target(
        self,
        symbol: str,
        *,
        category: str,
        target_trade_date: str,
        allowed_stale_symbols: Optional[List[str] | Set[str]] = None,
        suspended_symbols: Optional[Set[str]] = None,
        fast_date_peek: bool = False,
    ) -> CNSymbolLocalStatusResult:
        allowed = self._normalize_allowed_symbols(allowed_stale_symbols)
        local_state = evaluate_symbol_local_status(
            symbol,
            category=category,
            resolver=self.resolver,
            market_reader=self.market_reader,
            latest_trade_date=target_trade_date,
            allowed_stale_symbols=allowed,
            suspended_symbols=suspended_symbols or set(),
            fast_date_peek=fast_date_peek,
        )
        if (
            local_state.local_status == "stale"
            and local_state.latest_local_date
            and local_state.latest_local_date >= target_trade_date
        ):
            return local_state.with_local_status("up_to_date", allowed_stale_symbols=allowed)
        return local_state

    def _freshness_metadata(
        self,
        *,
        target_trade_date: str,
        coverage_ratio: float,
        coverage_complete_count: int,
        expected_scope_count: int,
        early_stop_reason: str = "",
    ) -> Dict[str, Any]:
        return {
            "latest_trade_date": target_trade_date,
            "strict_trade_date": self.strict_trade_date,
            "stable_trade_date": self.stable_trade_date,
            "effective_target_trade_date": target_trade_date,
            "freshness_mode": self.freshness_mode,
            "coverage_ratio": coverage_ratio,
            "coverage_complete_count": coverage_complete_count,
            "expected_scope_count": expected_scope_count,
            "coverage_threshold": self.coverage_threshold,
            "early_stop_reason": early_stop_reason,
        }

    def _build_completeness_report_for_target(
        self,
        *,
        components: Dict[str, Any],
        target_categories: List[str],
        target_trade_date: str,
        allowed_stale_symbols: Optional[List[str] | Set[str]] = None,
        early_stop_reason: str = "",
    ) -> Dict[str, Any]:
        allowed = self._normalize_allowed_symbols(allowed_stale_symbols)
        report: Dict[str, Any] = {
            'allowed_stale_symbols': sorted(allowed),
            'complete': True,
            'blocking_incomplete_count': 0,
            'categories_checked': list(target_categories),
            'categories': {},
            'data_quality_issues': [],
        }
        suspended_symbols = self._load_latest_suspended_symbols(target_trade_date)
        coverage_complete_count = 0
        expected_scope_count = 0
        pre_listing_symbols = self._pre_listing_symbols_for_target(
            components=components,
            target_categories=target_categories,
            target_trade_date=target_trade_date,
        )
        report['pre_listing_symbols'] = [
            {'symbol': symbol, 'list_date': list_date}
            for symbol, list_date in sorted(pre_listing_symbols.items())
        ]

        # Load freshness index once for all categories; collect new discoveries
        # to flush at the end so future checks are even faster.
        freshness_index = self._load_freshness_index()
        index_updates: dict[str, str] = {}

        for category in target_categories:
            date_counts: Counter[str] = Counter()
            status_counts: Counter[str] = Counter()
            missing_symbols: List[str] = []
            stale_symbols: List[Dict[str, str]] = []
            suspended_stale_symbols: List[Dict[str, str]] = []
            unreadable_symbols: List[Dict[str, str]] = []
            blocking_missing: List[str] = []
            blocking_stale: List[Dict[str, str]] = []
            blocking_unreadable: List[Dict[str, str]] = []
            category_coverage_complete_count = 0
            category_symbols = [
                str(symbol or "").strip().upper()
                for symbol in components.get(category, []) or []
                if str(symbol or "").strip().upper() not in pre_listing_symbols
            ]
            expected_count = len(category_symbols)
            expected_scope_count += expected_count
            category_pre_listing_symbols = [
                {'symbol': symbol, 'list_date': pre_listing_symbols[symbol]}
                for symbol in sorted(
                    {
                        str(symbol or "").strip().upper()
                        for symbol in components.get(category, []) or []
                        if str(symbol or "").strip().upper() in pre_listing_symbols
                    }
                )
            ]

            for normalized_sym in category_symbols:
                indexed_date = freshness_index.get(normalized_sym)

                if indexed_date:
                    # ── Fast path: derive status directly from the index ──
                    base = CNSymbolLocalStatusResult(
                        symbol=normalized_sym,
                        latest_local_date=indexed_date,
                        strict_trade_date=self.strict_trade_date,
                        stable_trade_date=self.stable_trade_date,
                        effective_target_trade_date=target_trade_date,
                        freshness_mode=self.freshness_mode,
                    )
                    if indexed_date >= target_trade_date:
                        local_state = base.with_local_status("up_to_date", allowed_stale_symbols=allowed)
                    elif normalized_sym in suspended_symbols:
                        local_state = base.with_local_status("suspended_stale", allowed_stale_symbols=allowed)
                    else:
                        local_state = base.with_local_status("stale", allowed_stale_symbols=allowed)
                else:
                    # ── Slow path: peek the CSV file ──
                    local_state = self._evaluate_symbol_local_status_for_target(
                        normalized_sym,
                        category=category,
                        target_trade_date=target_trade_date,
                        allowed_stale_symbols=allowed,
                        suspended_symbols=suspended_symbols,
                        fast_date_peek=True,
                    )
                    # Bootstrap the index with any date we discover here
                    if local_state.latest_local_date:
                        index_updates[normalized_sym] = local_state.latest_local_date
                status_counts[local_state.local_status] += 1
                if local_state.latest_local_date:
                    date_counts[local_state.latest_local_date] += 1
                if local_state.issues:
                    report['data_quality_issues'].extend(issue.to_dict() for issue in local_state.issues)
                if local_state.local_status in {'up_to_date', 'suspended_stale'}:
                    category_coverage_complete_count += 1

                if local_state.local_status == 'missing':
                    missing_symbols.append(local_state.symbol)
                    if local_state.is_blocking:
                        blocking_missing.append(local_state.symbol)
                elif local_state.local_status == 'stale':
                    item = {
                        'symbol': local_state.symbol,
                        'latest_local_date': local_state.latest_local_date,
                    }
                    stale_symbols.append(item)
                    if local_state.is_blocking:
                        blocking_stale.append(item)
                elif local_state.local_status == 'suspended_stale':
                    suspended_stale_symbols.append(
                        {
                            'symbol': local_state.symbol,
                            'latest_local_date': local_state.latest_local_date,
                        }
                    )
                elif local_state.local_status == 'unreadable':
                    item = {
                        'symbol': local_state.symbol,
                        'resolved_path': local_state.resolved_path,
                    }
                    unreadable_symbols.append(item)
                    if local_state.is_blocking:
                        blocking_unreadable.append(item)

            blocking_count = len(blocking_stale) + len(blocking_missing) + len(blocking_unreadable)
            coverage_complete_count += category_coverage_complete_count
            category_coverage_ratio = (
                category_coverage_complete_count / expected_count
                if expected_count
                else 1.0
            )
            report['categories'][category] = {
                'expected': expected_count,
                'latest_trade_date': target_trade_date,
                'pre_listing_symbols': category_pre_listing_symbols,
                'date_counts': dict(sorted(date_counts.items())),
                'status_counts': dict(sorted(status_counts.items())),
                'missing_symbols': sorted(missing_symbols),
                'stale_symbols': stale_symbols,
                'suspended_stale_symbols': suspended_stale_symbols,
                'unreadable_symbols': unreadable_symbols,
                'blocking_missing_symbols': sorted(blocking_missing),
                'blocking_stale_symbols': blocking_stale,
                'blocking_unreadable_symbols': blocking_unreadable,
                'blocking_incomplete_count': blocking_count,
                'coverage_complete_count': category_coverage_complete_count,
                'coverage_ratio': category_coverage_ratio,
            }

            if blocking_count > 0:
                report['complete'] = False
                report['blocking_incomplete_count'] += blocking_count

        if not self.resolver.trace.physical_directories_used_for_full_a:
            self.resolver.trace.physical_directories_used_for_full_a = [
                str(path) for path in self.resolver.physical_directories_for_full_a()
            ]
        coverage_ratio = (
            coverage_complete_count / expected_scope_count
            if expected_scope_count
            else 1.0
        )
        report.update(
            self._freshness_metadata(
                target_trade_date=target_trade_date,
                coverage_ratio=coverage_ratio,
                coverage_complete_count=coverage_complete_count,
                expected_scope_count=expected_scope_count,
                early_stop_reason=early_stop_reason,
            )
        )
        report['resolver'] = self.resolver.snapshot()
        self.last_resolver_trace = dict(report['resolver'])

        # Persist any date discoveries from the slow path so the next call
        # can use the fast (index) path for those symbols.
        self._flush_freshness_index(index_updates)
        self._append_download_data_quality_issues(report)
        report['data_quality_issue_count'] = len(report['data_quality_issues'])

        return report

    def build_completeness_report(
        self,
        components: Optional[Dict] = None,
        allowed_stale_symbols: Optional[List[str] | Set[str]] = None,
        categories: Optional[List[str]] = None,
        target_trade_date: Optional[str] = None,
        early_stop_reason: str = "",
    ) -> Dict[str, Any]:
        """
        构建本地数据完整性报告。

        完整性的定义是：所有目标股票都存在本地文件，且最新交易日达到
        `target_trade_date`。如明确声明了允许跳过的 symbol，则从阻塞项中排除。
        """
        if components is None:
            components = self.load_components()
        target_categories = self._resolve_target_categories(components, categories)
        if target_trade_date is not None:
            report = self._build_completeness_report_for_target(
                components=components,
                target_categories=target_categories,
                target_trade_date=target_trade_date,
                allowed_stale_symbols=allowed_stale_symbols,
                early_stop_reason=early_stop_reason,
            )
        else:
            # Pre-warm the suspend cache for both dates in parallel so
            # the second report (if needed) doesn't pay a second API call.
            if (
                self.freshness_mode == "stable"
                and self.strict_trade_date != self.stable_trade_date
            ):
                self._prefetch_suspended_symbols(
                    [self.strict_trade_date, self.stable_trade_date],
                )

            strict_report = self._build_completeness_report_for_target(
                components=components,
                target_categories=target_categories,
                target_trade_date=self.strict_trade_date,
                allowed_stale_symbols=allowed_stale_symbols,
            )
            if (
                self.freshness_mode == "stable"
                and self.strict_trade_date != self.stable_trade_date
                and strict_report["coverage_ratio"] < self.coverage_threshold
            ):
                report = self._build_completeness_report_for_target(
                    components=components,
                    target_categories=target_categories,
                    target_trade_date=self.stable_trade_date,
                    allowed_stale_symbols=allowed_stale_symbols,
                    early_stop_reason=early_stop_reason,
                )
            else:
                report = strict_report
        return report

    def _load_latest_suspended_symbols(
        self,
        target_trade_date: str,
        *,
        force_refresh: bool = False,
        query_run_id: str | None = None,
    ) -> Set[str]:
        """
        获取最新交易日停牌标的集合。

        对于当日停牌的股票，本地最后一个成交日早于最新交易日是合理状态，
        不应阻塞"完整性"判断。

        结果按 trade_date 写入磁盘(永久有效，历史停牌数据不会变化)，
        下次同日调用直接从磁盘加载，完全跳过 Tushare API 调用。
        """
        if (
            not force_refresh
            and target_trade_date in self._latest_suspended_symbols_cache
        ):
            return self._latest_suspended_symbols_cache[target_trade_date]

        # ── disk cache hit ──
        disk_path = self._suspend_cache_path(target_trade_date)
        try:
            if disk_path.exists() and not force_refresh:
                raw = json.loads(disk_path.read_text(encoding="utf-8"))
                canonical_payload = dict(raw) if isinstance(raw, dict) else {}
                declared_payload_sha256 = str(
                    canonical_payload.pop("payload_sha256", "") or ""
                ).strip().lower()
                computed_payload_sha256 = canonical_json_sha256(
                    canonical_payload
                )
                cached_symbols = [
                    str(symbol).upper()
                    for symbol in raw.get("symbols", [])
                    if str(symbol or "").strip()
                ] if isinstance(raw, dict) else []
                exact_event_records = (
                    raw.get("exact_event_records", [])
                    if isinstance(raw, dict)
                    else []
                )
                exact_event_records_valid = isinstance(
                    exact_event_records, list
                ) and all(
                    isinstance(record, Mapping)
                    and str(record.get("ts_code") or "").strip()
                    and str(record.get("trade_date") or "")
                    == target_trade_date
                    and str(record.get("suspend_type") or "").strip()
                    for record in exact_event_records
                )
                derived_suspend_symbols = sorted(
                    {
                        str(record.get("ts_code") or "").strip().upper()
                        for record in exact_event_records
                        if isinstance(record, Mapping)
                        and str(record.get("suspend_type") or "")
                        .strip()
                        .upper()
                        == "S"
                    }
                )
                derived_resume_symbols = sorted(
                    {
                        str(record.get("ts_code") or "").strip().upper()
                        for record in exact_event_records
                        if isinstance(record, Mapping)
                        and str(record.get("suspend_type") or "")
                        .strip()
                        .upper()
                        == "R"
                    }
                )
                derived_other_symbols = sorted(
                    {
                        str(record.get("ts_code") or "").strip().upper()
                        for record in exact_event_records
                        if isinstance(record, Mapping)
                        and str(record.get("suspend_type") or "")
                        .strip()
                        .upper()
                        not in {"S", "R"}
                    }
                )
                if (
                    isinstance(raw, dict)
                    and raw.get("version") == 5
                    and str(raw.get("trade_date") or "") == target_trade_date
                    and raw.get("query_succeeded") is True
                    and str(raw.get("source") or "") == "tushare.suspend_d"
                    and str(raw.get("query_variant") or "") == "trade_date"
                    and raw.get("query_params") == {
                        "trade_date": target_trade_date
                    }
                    and raw.get("exact_date_rows_validated") is True
                    and raw.get("continuation_state_complete") is False
                    and str(raw.get("query_run_id") or "").strip()
                    and exact_event_records_valid
                    and int(raw.get("exact_event_row_count") or 0)
                    == len(exact_event_records)
                    and str(raw.get("matched_symbols_sha256") or "").lower()
                    == symbol_set_sha256(cached_symbols)
                    and sorted(cached_symbols) == derived_suspend_symbols
                    and sorted(raw.get("resume_symbols", []) or [])
                    == derived_resume_symbols
                    and str(raw.get("resume_symbols_sha256") or "").lower()
                    == symbol_set_sha256(derived_resume_symbols)
                    and sorted(raw.get("other_event_symbols", []) or [])
                    == derived_other_symbols
                    and str(
                        raw.get("other_event_symbols_sha256") or ""
                    ).lower()
                    == symbol_set_sha256(derived_other_symbols)
                    and str(raw.get("exact_event_records_sha256") or "").lower()
                    == canonical_json_sha256(
                        raw.get("exact_event_records", []) or []
                    )
                    and declared_payload_sha256 == computed_payload_sha256
                ):
                    symbols: Set[str] = {
                        str(symbol).upper() for symbol in cached_symbols
                    }
                    self._latest_suspended_symbols_cache[target_trade_date] = symbols
                    return symbols
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_load_latest_suspended_symbols",
                exc=exc,
                issue_type="suspend_cache_read_exception",
                as_of=target_trade_date,
                metadata={"cache_path": str(disk_path)},
            )
            pass

        if self.pro is None:
            self._latest_suspended_symbols_cache[target_trade_date] = set()
            return self._latest_suspended_symbols_cache[target_trade_date]

        def _normalized_date(value: Any) -> str:
            digits = "".join(ch for ch in str(value or "") if ch.isdigit())
            return digits[:8] if len(digits) >= 8 else ""

        def _extract_target_events(
            suspend_df: pd.DataFrame,
        ) -> tuple[Set[str], list[dict[str, str]], bool]:
            if suspend_df is None or suspend_df.empty:
                return set(), [], True
            filtered = suspend_df.copy()
            date_columns = [column for column in ("trade_date", "suspend_date") if column in filtered.columns]
            if not date_columns:
                return set(), [], False
            date_mask = pd.Series(False, index=filtered.index)
            for column in date_columns:
                date_mask = date_mask | filtered[column].map(_normalized_date).eq(target_trade_date)
            filtered = filtered[date_mask]
            if not filtered.empty and (
                "ts_code" not in filtered.columns
                or "suspend_type" not in filtered.columns
            ):
                return set(), [], False
            exact_event_records: list[dict[str, str]] = []
            for row in filtered.to_dict(orient="records"):
                symbol = str(row.get("ts_code") or "").strip().upper()
                event_type = str(row.get("suspend_type") or "").strip().upper()
                if not symbol or not event_type:
                    return set(), [], False
                exact_event_records.append(
                    {
                        "ts_code": symbol,
                        "trade_date": target_trade_date,
                        "suspend_type": event_type,
                    }
                )
            exact_event_records.sort(
                key=lambda item: (
                    item["ts_code"],
                    item["suspend_type"],
                )
            )
            suspended_symbols = {
                item["ts_code"]
                for item in exact_event_records
                if item["suspend_type"] == "S"
            }
            return suspended_symbols, exact_event_records, True

        try:
            symbols: Set[str] = set()
            query_variant = "trade_date"
            query = {"trade_date": target_trade_date}
            suspend_df = self.pro.suspend_d(**query)
            if suspend_df is None or not isinstance(suspend_df, pd.DataFrame):
                suspend_df = pd.DataFrame()
            symbols, exact_event_records, exact_date_rows_validated = (
                _extract_target_events(
                suspend_df
                )
            )
            if not exact_date_rows_validated:
                raise RuntimeError(
                    "suspend_d response lacks an exact-date column"
                )

            # ── persist to disk cache (historic data is immutable) ──
            try:
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                resume_symbols = sorted(
                    {
                        item["ts_code"]
                        for item in exact_event_records
                        if item["suspend_type"] == "R"
                    }
                )
                other_event_symbols = sorted(
                    {
                        item["ts_code"]
                        for item in exact_event_records
                        if item["suspend_type"] not in {"S", "R"}
                    }
                )
                cache_payload = {
                    "version": 5,
                    "trade_date": target_trade_date,
                    "query_run_id": str(
                        query_run_id or self._suspend_query_run_id
                    ),
                    "query_succeeded": True,
                    "query_variant": query_variant,
                    "query_params": query,
                    "source": "tushare.suspend_d",
                    "semantic_scope": "exact_date_suspend_events_only",
                    "continuation_state_complete": False,
                    "exact_date_rows_validated": True,
                    "raw_row_count": int(len(suspend_df)),
                    "raw_rows_sha256": dataframe_sha256(suspend_df),
                    "matched_row_count": len(symbols),
                    "symbols": sorted(symbols),
                    "matched_symbols_sha256": symbol_set_sha256(symbols),
                    "resume_symbols": resume_symbols,
                    "resume_symbols_sha256": symbol_set_sha256(
                        resume_symbols
                    ),
                    "other_event_symbols": other_event_symbols,
                    "other_event_symbols_sha256": symbol_set_sha256(
                        other_event_symbols
                    ),
                    "exact_event_row_count": len(exact_event_records),
                    "exact_event_records": exact_event_records,
                    "exact_event_records_sha256": canonical_json_sha256(
                        exact_event_records
                    ),
                    "updated_at": datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
                cache_payload["payload_sha256"] = canonical_json_sha256(
                    cache_payload
                )
                temporary = disk_path.with_name(
                    f".{disk_path.name}.tmp-{os.getpid()}"
                )
                temporary.write_text(
                    json.dumps(cache_payload, ensure_ascii=False),
                    encoding="utf-8",
                )
                os.replace(temporary, disk_path)
            except Exception as exc:
                self._record_data_quality_exception(
                    func_name="_load_latest_suspended_symbols",
                    exc=exc,
                    issue_type="suspend_cache_write_exception",
                    as_of=target_trade_date,
                    metadata={"cache_path": str(disk_path)},
                )
                pass

            self._latest_suspended_symbols_cache[target_trade_date] = symbols
            return symbols
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_load_latest_suspended_symbols",
                exc=exc,
                issue_type="suspend_lookup_exception",
                as_of=target_trade_date,
            )
            self._latest_suspended_symbols_cache[target_trade_date] = set()
            return self._latest_suspended_symbols_cache[target_trade_date]

    def _prefetch_suspended_symbols(self, dates: List[str]) -> None:
        """Pre-warm the suspend cache for multiple dates using a thread pool.

        This turns two sequential ~10s Tushare API calls into a single
        ~10s parallel round-trip, cutting the suspend overhead in half.
        Dates already present in the in-memory or disk cache are skipped.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        uncached = [d for d in dates if d not in self._latest_suspended_symbols_cache]
        if not uncached:
            return

        def _fetch(date: str) -> None:
            self._load_latest_suspended_symbols(date)

        with ThreadPoolExecutor(max_workers=len(uncached)) as pool:
            futures = {pool.submit(_fetch, d): d for d in uncached}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    self._record_data_quality_exception(
                        func_name="_prefetch_suspended_symbols",
                        exc=exc,
                        issue_type="suspend_prefetch_exception",
                        as_of=futures[future],
                    )
                    pass

    # ── Freshness index ──────────────────────────────────────────────────────────

    def _load_freshness_index(self) -> dict[str, str]:
        """Load the on-disk freshness index as {SYMBOL: YYYYMMDD}.

        Returns an empty dict if the file is missing or unreadable.
        """
        path = self._freshness_index_path()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("symbols"), dict):
                    return {
                        str(k).strip().upper(): str(v)
                        for k, v in data["symbols"].items()
                        if k and v
                    }
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_load_freshness_index",
                exc=exc,
                issue_type="freshness_index_read_exception",
                metadata={"cache_path": str(path)},
            )
            pass
        return {}

    def _flush_freshness_index(self, updates: dict[str, str]) -> None:
        """Merge *updates* ({SYMBOL: YYYYMMDD}) into the on-disk freshness index.

        The write is atomic (tmp-file + rename) so partial writes never corrupt
        the index.  Silently ignores any I/O errors.
        """
        if not updates:
            return
        path = self._freshness_index_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            current = self._load_freshness_index()
            for sym, date in updates.items():
                normalized_sym = str(sym or "").strip().upper()
                if normalized_sym and date:
                    # Only advance the recorded date, never regress it
                    existing = current.get(normalized_sym, "")
                    if not existing or date > existing:
                        current[normalized_sym] = date
            payload = {
                "schema_version": 1,
                "written_at": datetime.now().strftime("%Y%m%dT%H%M%S"),
                "symbols": current,
            }
            tmp_path = path.with_name(path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(path)
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_flush_freshness_index",
                exc=exc,
                issue_type="freshness_index_write_exception",
                metadata={"cache_path": str(path)},
            )
            pass

    # ── Stock frame fetch ─────────────────────────────────────────────────────

    def _fetch_stock_frame(self, symbol: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
        """通过 Tushare 抓取指定时间窗口内的单只股票行情。"""
        if self.pro is None:
            return pd.DataFrame()
        df = self.pro.daily(ts_code=symbol, start_date=start_date_str, end_date=end_date_str)
        if df is None or df.empty:
            return pd.DataFrame()

        adj_df = self.pro.adj_factor(ts_code=symbol, start_date=start_date_str, end_date=end_date_str)
        if adj_df is not None and not adj_df.empty:
            df = df.merge(adj_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
            # 计算复权价格
            df['adj_close'] = df['close'] * df['adj_factor']
            df['adj_open'] = df['open'] * df['adj_factor']
            df['adj_high'] = df['high'] * df['adj_factor']
            df['adj_low'] = df['low'] * df['adj_factor']

        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        return (
            df.sort_values('trade_date')
            .drop_duplicates(subset=['trade_date'], keep='last')
            .reset_index(drop=True)
        )

    def _find_local_symbol_file(self, symbol: str, category: str) -> Optional[Path]:
        """查找 symbol 在本地缓存中的 CSV 文件。"""
        if category == "full_a":
            return self.resolver.resolve_symbol_file(symbol, universe_key="full_a")

        directory = Path(self.dirs.get(category, self.dirs["other"]))
        candidate = directory / f"{symbol}.csv"
        if candidate.exists():
            return candidate
        return None

    def _default_cleaning_result_fields(self, status: str = "skipped") -> dict[str, Any]:
        return {
            "cleaning_status": status,
            "cleaning_report_path": None,
            "factor_readiness_status": None,
            "storage_status": None,
            "parquet_status": PARQUET_STATUS_SKIPPED,
            "quarantine_path": None,
        }

    def _storage_optimization_config(self) -> TushareStorageOptimizationConfig:
        return TushareStorageOptimizationConfig(
            parquet_shadow_write=bool(config.TUSHARE_PARQUET_SHADOW_WRITE),
            parquet_canonical=bool(config.TUSHARE_PARQUET_CANONICAL),
            delete_redundant_csv=bool(config.TUSHARE_DELETE_REDUNDANT_CSV),
            parquet_dir=config.TUSHARE_PARQUET_DIR,
            parquet_compression=config.TUSHARE_PARQUET_COMPRESSION,
            metadata={"source": "CNFullMarketDownloader.download_stock"},
        )

    def _cleaning_result_fields(self, result: Mapping[str, Any] | None) -> dict[str, Any]:
        if not result:
            return self._default_cleaning_result_fields()
        return {
            "cleaning_status": result.get("cleaning_status"),
            "cleaning_report_path": result.get("cleaning_report_path"),
            "factor_readiness_status": result.get("factor_readiness_status"),
            "storage_status": result.get("storage_status"),
            "parquet_status": result.get("parquet_status", PARQUET_STATUS_SKIPPED),
            "quarantine_path": result.get("quarantine_path"),
        }

    def _write_parquet_bars(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        manifest = self.market_store.write_full_history_bars(
            frame,
            source=source,
            metadata=metadata,
        )
        self.market_reader = MarketDataReader(market="CN", data_root=self.data_root)
        self.market_store.reader = self.market_reader
        return manifest
    
    def load_components(self, components_file: str | None = None) -> Dict:
        """加载成分股"""
        components_path = Path(components_file) if components_file else Path(self.components_file)
        existing_components: Dict[str, Any] | None = None
        if not components_path.exists():
            if self.pro is not None:
                components = get_all_components(self.pro)
            else:
                components = self._build_local_symbol_universe()
            if components.get("full_a"):
                save_components(components, output_dir=str(components_path.parent))
        else:
            with open(components_path, 'r', encoding='utf-8') as f:
                existing_components = json.load(f)
            components = dict(existing_components)
            if "full_a" not in components or not components.get("full_a"):
                if self.pro is not None:
                    refreshed = get_all_components(self.pro)
                else:
                    refreshed = self._build_local_symbol_universe()
                if refreshed.get("full_a"):
                    components = refreshed
                    save_components(components, output_dir=str(components_path.parent))
                elif existing_components and existing_components.get("full_a"):
                    components = dict(existing_components)
                else:
                    components = refreshed
        if not self.resolver.trace.physical_directories_used_for_full_a:
            self.resolver.trace.physical_directories_used_for_full_a = [
                str(path) for path in self.resolver.physical_directories_for_full_a()
            ]
        self._refresh_full_a_write_categories(components)
        components["resolver"] = dict(components.get("resolver", self.resolver.snapshot()) or self.resolver.snapshot())
        self.last_resolver_trace = dict(components["resolver"])
        
        print("=" * 80)
        print("📊 加载A股成分股")
        print("=" * 80)
        print(f"全A股:    {len(components.get('full_a', []))} 只")
        print(f"沪深300:  {len(components['hs300'])} 只")
        print(f"中证500:  {len(components['zz500'])} 只")
        print(f"中证1000: {len(components['zz1000'])} 只")
        print(f"总计:     {components['stats']['total_unique']} 只")
        print("=" * 80)
        
        return components

    @staticmethod
    def _normalize_trade_date_column(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty or "trade_date" not in frame.columns:
            return pd.DataFrame() if frame is None else frame.copy()
        normalized = frame.copy()
        normalized["trade_date"] = normalized["trade_date"].map(
            lambda value: "".join(ch for ch in str(value or "") if ch.isdigit())[:8]
        )
        normalized = normalized[normalized["trade_date"].astype(str).str.len() == 8]
        return normalized.reset_index(drop=True)

    def _fetch_daily_batch_adj_factor_frame(self, target_trade_date: str) -> tuple[pd.DataFrame, int, str]:
        if self.pro is None:
            return pd.DataFrame(), 0, "provider_unavailable"

        api_calls = 0
        try:
            adj_df = self.pro.adj_factor(
                trade_date=target_trade_date,
                fields="ts_code,trade_date,adj_factor",
            )
            api_calls += 1
        except TypeError:
            try:
                adj_df = self.pro.adj_factor(trade_date=target_trade_date)
                api_calls += 1
            except Exception as exc:
                self._record_data_quality_exception(
                    func_name="_fetch_daily_batch_adj_factor_frame",
                    exc=exc,
                    issue_type="daily_adj_factor_exception",
                    as_of=target_trade_date,
                    metadata={"mode": "retry_without_fields"},
                )
                return pd.DataFrame(), api_calls, str(exc)[:100]
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_fetch_daily_batch_adj_factor_frame",
                exc=exc,
                issue_type="daily_adj_factor_exception",
                as_of=target_trade_date,
                metadata={"mode": "fields"},
            )
            return pd.DataFrame(), api_calls, str(exc)[:100]

        if adj_df is None or adj_df.empty:
            return pd.DataFrame(), api_calls, "adj_factor_empty"
        adj_df = self._normalize_trade_date_column(adj_df)
        if "ts_code" in adj_df.columns:
            adj_df["ts_code"] = adj_df["ts_code"].astype(str).str.strip().str.upper()
        return adj_df, api_calls, ""

    @staticmethod
    def _merge_adj_factor_and_adjusted_prices(frame: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame() if frame is None else frame.copy()
        merged = frame.copy()
        if adj_df is not None and not adj_df.empty:
            merge_keys = [
                key
                for key in ("ts_code", "trade_date")
                if key in merged.columns and key in adj_df.columns
            ]
            if "adj_factor" in adj_df.columns and merge_keys:
                if "adj_factor" in merged.columns:
                    merged = merged.drop(columns=["adj_factor"])
                merged = merged.merge(
                    adj_df[[*merge_keys, "adj_factor"]].drop_duplicates(subset=merge_keys),
                    on=merge_keys,
                    how="left",
                )
        if "adj_factor" in merged.columns:
            adj_factor = pd.to_numeric(merged["adj_factor"], errors="coerce")
            for source, target in (
                ("close", "adj_close"),
                ("open", "adj_open"),
                ("high", "adj_high"),
                ("low", "adj_low"),
            ):
                if source in merged.columns:
                    merged[target] = pd.to_numeric(merged[source], errors="coerce") * adj_factor
        return merged

    @staticmethod
    def _frame_data_source(frame: pd.DataFrame, default: str = "tushare.daily") -> str:
        if frame is None or frame.empty or "data_source" not in frame.columns:
            return default
        sources = sorted(
            {
                str(value or "").strip()
                for value in frame["data_source"].tolist()
                if str(value or "").strip()
            }
        )
        return "+".join(sources) if sources else default

    def _fetch_daily_batch_frame(self, target_trade_date: str) -> tuple[pd.DataFrame, int, str]:
        if self.pro is None:
            return pd.DataFrame(), 0, "provider_unavailable"

        api_calls = 0
        self._last_daily_batch_adj_factor_frame = pd.DataFrame()
        daily_fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
        try:
            df = self.pro.daily(trade_date=target_trade_date, fields=daily_fields)
            api_calls += 1
        except TypeError:
            try:
                df = self.pro.daily(trade_date=target_trade_date)
                api_calls += 1
            except Exception as exc:
                return pd.DataFrame(), api_calls, str(exc)[:100]
        except Exception as exc:
            return pd.DataFrame(), api_calls, str(exc)[:100]

        if df is None or df.empty:
            return pd.DataFrame(), api_calls, "daily_empty"

        df = self._normalize_trade_date_column(df)
        if df.empty:
            return pd.DataFrame(), api_calls, "daily_empty"

        adj_df, adj_api_calls, _adj_error = self._fetch_daily_batch_adj_factor_frame(target_trade_date)
        api_calls += adj_api_calls
        self._last_daily_batch_adj_factor_frame = adj_df
        df = self._merge_adj_factor_and_adjusted_prices(df, adj_df)

        try:
            basic_df = self.pro.daily_basic(
                trade_date=target_trade_date,
                fields="ts_code,trade_date,turnover_rate,pe,pb,total_mv,circ_mv",
            )
            api_calls += 1
        except AttributeError:
            basic_df = pd.DataFrame()
        except TypeError:
            try:
                basic_df = self.pro.daily_basic(trade_date=target_trade_date)
                api_calls += 1
            except Exception as exc:
                self._record_data_quality_exception(
                    func_name="_fetch_daily_batch_frame",
                    exc=exc,
                    issue_type="daily_basic_exception",
                    as_of=target_trade_date,
                    metadata={"mode": "retry_without_fields"},
                )
                basic_df = pd.DataFrame()
        except Exception as exc:
            self._record_data_quality_exception(
                func_name="_fetch_daily_batch_frame",
                exc=exc,
                issue_type="daily_basic_exception",
                as_of=target_trade_date,
                metadata={"mode": "fields"},
            )
            basic_df = pd.DataFrame()

        if basic_df is not None and not basic_df.empty:
            basic_df = self._normalize_trade_date_column(basic_df)
            merge_keys = [
                key
                for key in ("ts_code", "trade_date")
                if key in df.columns and key in basic_df.columns
            ]
            if merge_keys:
                basic_cols = [
                    column
                    for column in basic_df.columns
                    if column in merge_keys or column not in df.columns
                ]
                if len(basic_cols) > len(merge_keys):
                    df = df.merge(
                        basic_df[basic_cols].drop_duplicates(subset=merge_keys),
                        on=merge_keys,
                        how="left",
                    )

        df["data_source"] = "tushare.daily"
        df["trade_date"] = pd.to_datetime(
            df["trade_date"],
            format="%Y%m%d",
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["trade_date"])
        sort_keys = [key for key in ("ts_code", "trade_date") if key in df.columns]
        return df.sort_values(sort_keys).reset_index(drop=True), api_calls, ""

    def _existing_symbol_records(self, local_state: CNSymbolLocalStatusResult) -> tuple[pd.DataFrame, int]:
        existing_df = local_state.frame.copy()
        existing_records = len(existing_df) if not existing_df.empty else len(local_state.frame)
        return existing_df, existing_records

    def download_daily_batch(
        self,
        symbols: List[str],
        category: str,
        target_trade_date: Optional[str] = None,
        *,
        publish_canonical: bool = True,
    ) -> List[Dict]:
        """补齐一个交易日的一批股票；可显式禁止 canonical Parquet 发布。"""
        effective_target_trade_date = target_trade_date or self.latest_trade_date
        cleaning_skipped_fields = self._default_cleaning_result_fields()
        normalized_symbols: list[str] = []
        for symbol in symbols or []:
            normalized = str(symbol or "").strip().upper()
            if normalized and normalized not in normalized_symbols:
                normalized_symbols.append(normalized)

        print(f"\n📥 批量补齐 {category.upper()} {effective_target_trade_date}: {len(normalized_symbols)} 只")
        suspended_symbols = self._load_latest_suspended_symbols(effective_target_trade_date)
        local_states: dict[str, CNSymbolLocalStatusResult] = {}
        results: list[Dict[str, Any]] = []
        symbols_to_fetch: list[str] = []

        for symbol in normalized_symbols:
            try:
                local_state = self._evaluate_symbol_local_status_for_target(
                    symbol,
                    category=category,
                    target_trade_date=effective_target_trade_date,
                    allowed_stale_symbols=set(),
                    suspended_symbols=suspended_symbols,
                )
            except Exception as exc:
                results.append(
                    {
                        "symbol": symbol,
                        "category": category,
                        "status": "failed",
                        "local_status": "unreadable",
                        "records": 0,
                        "mode": "daily_batch",
                        "latest_local_date": "",
                        "latest_trade_date": effective_target_trade_date,
                        "resolved_path": "",
                        "api_calls": 0,
                        "batch_api_calls": 0,
                        "error": str(exc)[:100],
                        **cleaning_skipped_fields,
                    }
                )
                continue

            local_states[symbol] = local_state
            existing_df, existing_records = self._existing_symbol_records(local_state)
            del existing_df
            self.last_resolver_trace = self.resolver.snapshot()

            if local_state.local_status in {"up_to_date", "suspended_stale"}:
                results.append(
                    {
                        "symbol": local_state.symbol,
                        "category": category,
                        "status": "cached",
                        "local_status": local_state.local_status,
                        "records": existing_records,
                        "mode": local_state.local_status,
                        "latest_local_date": local_state.latest_local_date,
                        "latest_trade_date": effective_target_trade_date,
                        "resolved_path": local_state.resolved_path,
                        "api_calls": 0,
                        "batch_api_calls": 0,
                        "error": None,
                        **cleaning_skipped_fields,
                    }
                )
                continue

            if local_state.local_status in {"missing", "unreadable"}:
                results.append(
                    {
                        "symbol": local_state.symbol,
                        "category": category,
                        "status": "failed",
                        "local_status": local_state.local_status,
                        "records": existing_records,
                        "mode": "daily_batch",
                        "latest_local_date": local_state.latest_local_date,
                        "latest_trade_date": effective_target_trade_date,
                        "resolved_path": local_state.resolved_path,
                        "api_calls": 0,
                        "batch_api_calls": 0,
                        "error": "history_backfill_required",
                        **cleaning_skipped_fields,
                    }
                )
                continue

            symbols_to_fetch.append(symbol)

        batch_df = pd.DataFrame()
        batch_api_calls = 0
        batch_error = ""
        if symbols_to_fetch:
            batch_df, batch_api_calls, batch_error = self._fetch_daily_batch_frame(effective_target_trade_date)
            if not batch_df.empty and "ts_code" in batch_df.columns:
                batch_df["ts_code"] = batch_df["ts_code"].astype(str).str.strip().str.upper()
            fetched_symbols = (
                set(batch_df["ts_code"].astype(str).str.strip().str.upper().tolist())
                if not batch_df.empty and "ts_code" in batch_df.columns
                else set()
            )
            missing_symbols = [symbol for symbol in symbols_to_fetch if symbol not in fetched_symbols]
            if missing_symbols:
                try:
                    fallback_df = fetch_eastmoney_daily_batch_frame(
                        missing_symbols,
                        effective_target_trade_date,
                    )
                    if fallback_df is not None and not fallback_df.empty:
                        fallback_df = self._normalize_trade_date_column(fallback_df)
                        if "ts_code" in fallback_df.columns:
                            fallback_df["ts_code"] = fallback_df["ts_code"].astype(str).str.strip().str.upper()
                        adj_df = getattr(self, "_last_daily_batch_adj_factor_frame", pd.DataFrame())
                        if adj_df is None or adj_df.empty:
                            adj_df, adj_api_calls, adj_error = self._fetch_daily_batch_adj_factor_frame(
                                effective_target_trade_date
                            )
                            batch_api_calls += adj_api_calls
                            self._last_daily_batch_adj_factor_frame = adj_df
                            if adj_error and not batch_error:
                                batch_error = adj_error
                        fallback_df = self._merge_adj_factor_and_adjusted_prices(fallback_df, adj_df)
                        if "trade_date" in fallback_df.columns:
                            fallback_df["trade_date"] = pd.to_datetime(
                                fallback_df["trade_date"],
                                format="%Y%m%d",
                                errors="coerce",
                            ).dt.strftime("%Y-%m-%d")
                            fallback_df = fallback_df.dropna(subset=["trade_date"])
                        if not fallback_df.empty:
                            batch_df = pd.concat([batch_df, fallback_df], ignore_index=True, sort=False)
                            if {"ts_code", "trade_date"}.issubset(batch_df.columns):
                                batch_df = (
                                    batch_df.drop_duplicates(
                                        subset=["ts_code", "trade_date"],
                                        keep="first",
                                    )
                                    .sort_values(["ts_code", "trade_date"])
                                    .reset_index(drop=True)
                                )
                except Exception as exc:
                    self._record_data_quality_exception(
                        func_name="fetch_eastmoney_daily_batch_frame",
                        exc=exc,
                        issue_type="eastmoney_daily_exception",
                        as_of=effective_target_trade_date,
                        metadata={"symbols": missing_symbols},
                    )
                    if not batch_error:
                        batch_error = f"eastmoney_fallback_error:{type(exc).__name__}"

        for symbol in symbols_to_fetch:
            local_state = local_states[symbol]
            existing_df, existing_records = self._existing_symbol_records(local_state)
            filepath = (
                self._resolve_full_a_write_path(local_state.symbol, local_state.resolved_path)
                if category == "full_a"
                else Path(self.dirs.get(category, self.dirs["other"])) / f"{local_state.symbol}.csv"
            )
            symbol_df = pd.DataFrame()
            if not batch_df.empty and "ts_code" in batch_df.columns:
                symbol_df = batch_df[batch_df["ts_code"].astype(str).str.upper() == local_state.symbol].copy()
            data_source = self._frame_data_source(symbol_df, default="")

            if symbol_df.empty:
                if local_state.local_status == "stale" and existing_records > 0:
                    stale_cached_state = local_state.with_local_status("stale_cached")
                    results.append(
                        {
                            "symbol": stale_cached_state.symbol,
                            "category": category,
                            "status": "stale_cached",
                            "local_status": stale_cached_state.local_status,
                            "records": existing_records,
                            "mode": "stale_cached",
                            "latest_local_date": stale_cached_state.latest_local_date,
                            "latest_trade_date": effective_target_trade_date,
                            "resolved_path": stale_cached_state.resolved_path,
                            "api_calls": 0,
                            "batch_api_calls": batch_api_calls,
                            "data_source": "",
                            "error": None if batch_error in {"", "daily_empty"} else batch_error,
                            **cleaning_skipped_fields,
                        }
                    )
                    continue
                results.append(
                    {
                        "symbol": local_state.symbol,
                        "category": category,
                        "status": "failed",
                        "local_status": local_state.local_status,
                        "records": existing_records,
                        "mode": "daily_batch",
                        "latest_local_date": local_state.latest_local_date,
                        "latest_trade_date": effective_target_trade_date,
                        "resolved_path": local_state.resolved_path,
                        "api_calls": 0,
                        "batch_api_calls": batch_api_calls,
                        "data_source": "",
                        "error": batch_error or "Empty data",
                        **cleaning_skipped_fields,
                    }
                )
                continue

            try:
                final_df = symbol_df.copy()
                if not existing_df.empty and local_state.local_status != "unreadable":
                    existing_to_merge = existing_df.copy()
                    if "trade_date" not in existing_to_merge.columns and "date" in existing_to_merge.columns:
                        existing_to_merge = existing_to_merge.rename(columns={"date": "trade_date"})
                    if "trade_date" in existing_to_merge.columns:
                        existing_to_merge["trade_date"] = pd.to_datetime(
                            existing_to_merge["trade_date"],
                            errors="coerce",
                        ).dt.strftime("%Y-%m-%d")
                        existing_to_merge = existing_to_merge.dropna(subset=["trade_date"])
                    final_df = (
                        pd.concat([existing_to_merge, symbol_df], ignore_index=True)
                        .sort_values("trade_date")
                        .drop_duplicates(subset=["trade_date"], keep="last")
                        .reset_index(drop=True)
                    )
                final_df = final_df.drop(columns=["data_source"], errors="ignore")

                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                cleaning_result: Mapping[str, Any] | None = None
                if config.TUSHARE_AUTO_CLEAN:
                    cleaning_result = clean_tushare_dataframe_to_file(
                        final_df,
                        canonical_path=filepath,
                        table_name="daily",
                        promote=False,
                        raw_backup_dir=config.TUSHARE_RAW_BACKUP_DIR,
                        quarantine_dir=config.TUSHARE_QUARANTINE_DIR,
                        report_dir=config.TUSHARE_CLEANING_REPORT_DIR,
                        factor_readiness_dir=config.TUSHARE_FACTOR_READINESS_DIR,
                        enable_factor_readiness=bool(config.TUSHARE_FACTOR_READINESS),
                        enable_storage_audit=bool(config.TUSHARE_STORAGE_AUDIT),
                        storage_config=self._storage_optimization_config(),
                        metadata={
                            "symbol": local_state.symbol,
                            "category": category,
                            "target_trade_date": effective_target_trade_date,
                            "local_status": local_state.local_status,
                            "mode": "daily_batch",
                            "data_source": data_source,
                        },
                    )
                    cleaning_report = cleaning_result.get("cleaning_report")
                    if (
                        cleaning_result.get("cleaning_status") == CLEANING_STATUS_FAIL
                        or getattr(cleaning_report, "status", None) == CLEANING_STATUS_FAIL
                    ):
                        results.append(
                            {
                                "symbol": local_state.symbol,
                                "category": category,
                                "status": "failed",
                                "local_status": local_state.local_status,
                                "records": existing_records,
                                "mode": "daily_batch",
                                "latest_local_date": local_state.latest_local_date,
                                "latest_trade_date": effective_target_trade_date,
                                "resolved_path": str(filepath),
                                "api_calls": 0,
                                "batch_api_calls": batch_api_calls,
                                "data_source": data_source,
                                "error": "Tushare cleaning failed",
                                **self._cleaning_result_fields(cleaning_result),
                            }
                        )
                        continue
                    cleaned_df = cleaning_result.get("cleaned_df")
                    if isinstance(cleaned_df, pd.DataFrame):
                        final_df = cleaned_df
                else:
                    cleaning_result = cleaning_skipped_fields

                if publish_canonical:
                    self._write_parquet_bars(
                        final_df,
                        source="CNFullMarketDownloader.download_daily_batch",
                        metadata={
                            "symbol": local_state.symbol,
                            "category": category,
                            "target_trade_date": effective_target_trade_date,
                            "local_status": local_state.local_status,
                            "mode": "daily_batch",
                            "data_source": data_source,
                        },
                    )

                cleaning_fields = self._cleaning_result_fields(cleaning_result)
                latest_saved_ts = pd.NaT
                if "trade_date" in final_df.columns:
                    latest_saved_ts = pd.to_datetime(final_df["trade_date"], errors="coerce").max()
                elif "date" in final_df.columns:
                    latest_saved_ts = pd.to_datetime(final_df["date"], errors="coerce").max()
                latest_saved_date = (
                    latest_saved_ts.strftime("%Y%m%d")
                    if pd.notna(latest_saved_ts)
                    else local_state.latest_local_date
                )
                self.last_resolver_trace = self.resolver.snapshot()
                if latest_saved_date and latest_saved_date < effective_target_trade_date:
                    results.append(
                        {
                            "symbol": local_state.symbol,
                            "category": category,
                            "status": "stale_cached",
                            "local_status": "stale_cached",
                            "records": len(final_df),
                            "mode": "stale_cached",
                            "latest_local_date": latest_saved_date,
                            "latest_trade_date": effective_target_trade_date,
                            "resolved_path": str(filepath),
                            "api_calls": 0,
                            "batch_api_calls": batch_api_calls,
                            "data_source": data_source,
                            "error": None,
                            **cleaning_fields,
                        }
                    )
                    continue

                results.append(
                    {
                        "symbol": local_state.symbol,
                        "category": category,
                        "status": "updated",
                        "local_status": "up_to_date",
                        "records": len(final_df),
                        "mode": "daily_batch",
                        "latest_local_date": latest_saved_date,
                        "latest_trade_date": effective_target_trade_date,
                        "resolved_path": str(filepath),
                        "api_calls": 0,
                        "batch_api_calls": batch_api_calls,
                        "data_source": data_source,
                        "error": None,
                        **cleaning_fields,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "symbol": local_state.symbol,
                        "category": category,
                        "status": "failed",
                        "local_status": local_state.local_status,
                        "records": existing_records,
                        "mode": "daily_batch",
                        "latest_local_date": local_state.latest_local_date,
                        "latest_trade_date": effective_target_trade_date,
                        "resolved_path": local_state.resolved_path,
                        "api_calls": 0,
                        "batch_api_calls": batch_api_calls,
                        "data_source": data_source,
                        "error": str(exc)[:100],
                        **cleaning_skipped_fields,
                    }
                )

        index_updates: dict[str, str] = {}
        for result in results:
            self.stats["total"] += 1
            status = str(result.get("status") or "")
            if status == "updated":
                self.stats["updated"] += 1
            elif status == "cached":
                self.stats["cached"] += 1
            elif status == "stale_cached":
                self.stats["stale_cached"] += 1
            else:
                self.stats["failed"] += 1

            symbol = str(result.get("symbol") or "").strip().upper()
            latest_date = str(result.get("latest_local_date") or "")
            if symbol and latest_date and status in ("updated", "cached", "stale_cached"):
                index_updates[symbol] = latest_date
        if publish_canonical:
            self._flush_freshness_index(index_updates)
        print(
            f"✅ 批量补齐完成 | cached: {self.stats['cached']} | "
            f"stale_cached: {self.stats['stale_cached']} | "
            f"updated: {self.stats['updated']} | failed: {self.stats['failed']}"
        )
        return results
    
    def download_stock(
        self,
        symbol: str,
        category: str,
        target_trade_date: Optional[str] = None,
    ) -> Dict:
        """
        下载单只股票数据

        Returns:
            Dict with download result
        """
        effective_target_trade_date = target_trade_date or self.latest_trade_date
        cleaning_skipped_fields = self._default_cleaning_result_fields()
        suspended_symbols = self._load_latest_suspended_symbols(effective_target_trade_date)
        local_state = self._evaluate_symbol_local_status_for_target(
            symbol,
            category=category,
            target_trade_date=effective_target_trade_date,
            allowed_stale_symbols=set(),
            suspended_symbols=suspended_symbols,
        )
        normalized_existing_df = local_state.frame.copy()
        existing_df = normalized_existing_df.copy()
        existing_records = len(existing_df) if not existing_df.empty else len(normalized_existing_df)
        self.last_resolver_trace = self.resolver.snapshot()

        if local_state.local_status in {'up_to_date', 'suspended_stale'}:
            return {
                'symbol': local_state.symbol,
                'category': category,
                'status': 'cached',
                'local_status': local_state.local_status,
                'records': existing_records,
                'mode': local_state.local_status,
                'latest_local_date': local_state.latest_local_date,
                'latest_trade_date': effective_target_trade_date,
                'resolved_path': local_state.resolved_path,
                'api_calls': 0,
                'error': None,
                **cleaning_skipped_fields,
            }

        filepath = (
            self._resolve_full_a_write_path(local_state.symbol, local_state.resolved_path)
            if category == 'full_a'
            else Path(self.dirs.get(category, self.dirs['other'])) / f'{local_state.symbol}.csv'
        )

        try:
            end_date_str = effective_target_trade_date
            is_incremental = bool(local_state.latest_local_date and existing_records >= 200)
            start_date_str = self.start_date.strftime('%Y%m%d')

            if is_incremental and local_state.latest_local_date:
                overlap_start = pd.to_datetime(local_state.latest_local_date) - timedelta(days=1)
                start_date_str = max(overlap_start.strftime('%Y%m%d'), start_date_str)

            df = self._fetch_stock_frame(local_state.symbol, start_date_str, end_date_str)

            if df.empty:
                if local_state.local_status == 'stale' and existing_records > 0:
                    stale_cached_state = local_state.with_local_status('stale_cached')
                    return {
                        'symbol': stale_cached_state.symbol,
                        'category': category,
                        'status': 'stale_cached',
                        'local_status': stale_cached_state.local_status,
                        'records': existing_records,
                        'mode': 'stale_cached',
                        'latest_local_date': stale_cached_state.latest_local_date,
                        'latest_trade_date': effective_target_trade_date,
                        'resolved_path': stale_cached_state.resolved_path,
                        'api_calls': self.REQUESTS_PER_STOCK,
                        'error': None,
                        **cleaning_skipped_fields,
                    }
                return {
                    'symbol': local_state.symbol,
                    'category': category,
                    'status': 'failed',
                    'local_status': local_state.local_status,
                    'records': existing_records,
                    'mode': local_state.local_status,
                    'latest_local_date': local_state.latest_local_date,
                    'latest_trade_date': effective_target_trade_date,
                    'resolved_path': local_state.resolved_path,
                    'api_calls': self.REQUESTS_PER_STOCK,
                    'error': 'Empty data',
                    **cleaning_skipped_fields,
                }

            final_df = df.copy()
            if not existing_df.empty and local_state.local_status != 'unreadable':
                existing_to_merge = existing_df.copy()
                if 'trade_date' not in existing_to_merge.columns and 'date' in existing_to_merge.columns:
                    existing_to_merge = existing_to_merge.rename(columns={'date': 'trade_date'})
                if 'trade_date' in existing_to_merge.columns:
                    existing_to_merge['trade_date'] = pd.to_datetime(
                        existing_to_merge['trade_date'],
                        errors='coerce',
                    ).dt.strftime('%Y-%m-%d')
                    existing_to_merge = existing_to_merge.dropna(subset=['trade_date'])
                final_df = (
                    pd.concat([existing_to_merge, df], ignore_index=True)
                    .sort_values('trade_date')
                    .drop_duplicates(subset=['trade_date'], keep='last')
                    .reset_index(drop=True)
                )

            latest_candidate_ts = pd.NaT
            if 'trade_date' in final_df.columns:
                latest_candidate_ts = pd.to_datetime(final_df['trade_date'], errors='coerce').max()
            elif 'date' in final_df.columns:
                latest_candidate_ts = pd.to_datetime(final_df['date'], errors='coerce').max()
            latest_candidate_date = (
                latest_candidate_ts.strftime('%Y%m%d')
                if pd.notna(latest_candidate_ts)
                else local_state.latest_local_date
            )
            if (
                latest_candidate_date
                and latest_candidate_date < effective_target_trade_date
                and existing_records > 0
            ):
                return {
                    'symbol': local_state.symbol,
                    'category': category,
                    'status': 'stale_cached',
                    'local_status': 'stale_cached',
                    'records': len(final_df),
                    'mode': 'stale_cached',
                    'latest_local_date': latest_candidate_date,
                    'latest_trade_date': effective_target_trade_date,
                    'resolved_path': local_state.resolved_path,
                    'api_calls': self.REQUESTS_PER_STOCK,
                    'error': None,
                    **cleaning_skipped_fields,
                }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            cleaning_result: Mapping[str, Any] | None = None
            if config.TUSHARE_AUTO_CLEAN:
                cleaning_result = clean_tushare_dataframe_to_file(
                    final_df,
                    canonical_path=filepath,
                    table_name="daily",
                    promote=False,
                    raw_backup_dir=config.TUSHARE_RAW_BACKUP_DIR,
                    quarantine_dir=config.TUSHARE_QUARANTINE_DIR,
                    report_dir=config.TUSHARE_CLEANING_REPORT_DIR,
                    factor_readiness_dir=config.TUSHARE_FACTOR_READINESS_DIR,
                    enable_factor_readiness=bool(config.TUSHARE_FACTOR_READINESS),
                    enable_storage_audit=bool(config.TUSHARE_STORAGE_AUDIT),
                    storage_config=self._storage_optimization_config(),
                    metadata={
                        "symbol": local_state.symbol,
                        "category": category,
                        "target_trade_date": effective_target_trade_date,
                        "local_status": local_state.local_status,
                        "mode": "incremental" if is_incremental else "full",
                    },
                )
                cleaning_report = cleaning_result.get("cleaning_report")
                if (
                    cleaning_result.get("cleaning_status") == CLEANING_STATUS_FAIL
                    or getattr(cleaning_report, "status", None) == CLEANING_STATUS_FAIL
                ):
                    return {
                        'symbol': local_state.symbol,
                        'category': category,
                        'status': 'failed',
                        'local_status': local_state.local_status,
                        'records': existing_records,
                        'mode': local_state.local_status,
                        'latest_local_date': local_state.latest_local_date,
                        'latest_trade_date': effective_target_trade_date,
                        'resolved_path': str(filepath),
                        'api_calls': self.REQUESTS_PER_STOCK,
                        'error': 'Tushare cleaning failed',
                        **self._cleaning_result_fields(cleaning_result),
                    }
                cleaned_df = cleaning_result.get("cleaned_df")
                if isinstance(cleaned_df, pd.DataFrame):
                    final_df = cleaned_df
            else:
                cleaning_result = cleaning_skipped_fields
            self._write_parquet_bars(
                final_df,
                source="CNFullMarketDownloader.download_stock",
                metadata={
                    "symbol": local_state.symbol,
                    "category": category,
                    "target_trade_date": effective_target_trade_date,
                    "local_status": local_state.local_status,
                    "mode": "incremental" if is_incremental else "full",
                },
            )
            cleaning_fields = self._cleaning_result_fields(cleaning_result)
            latest_saved_ts = pd.NaT
            if 'trade_date' in final_df.columns:
                latest_saved_ts = pd.to_datetime(final_df['trade_date'], errors='coerce').max()
            elif 'date' in final_df.columns:
                latest_saved_ts = pd.to_datetime(final_df['date'], errors='coerce').max()
            latest_saved_date = (
                latest_saved_ts.strftime('%Y%m%d')
                if pd.notna(latest_saved_ts)
                else local_state.latest_local_date
            )
            self.last_resolver_trace = self.resolver.snapshot()
            if latest_saved_date and latest_saved_date < effective_target_trade_date:
                return {
                    'symbol': local_state.symbol,
                    'category': category,
                    'status': 'stale_cached',
                    'local_status': 'stale_cached',
                    'records': len(final_df),
                    'mode': 'stale_cached',
                    'latest_local_date': latest_saved_date,
                    'latest_trade_date': effective_target_trade_date,
                    'resolved_path': str(filepath),
                    'api_calls': self.REQUESTS_PER_STOCK,
                    'error': None,
                    **cleaning_fields,
                }

            return {
                'symbol': local_state.symbol,
                'category': category,
                'status': 'updated',
                'local_status': 'up_to_date',
                'records': len(final_df),
                'mode': 'incremental' if is_incremental else 'full',
                'latest_local_date': latest_saved_date,
                'latest_trade_date': effective_target_trade_date,
                'resolved_path': str(filepath),
                'api_calls': self.REQUESTS_PER_STOCK,
                'error': None,
                **cleaning_fields,
            }

        except Exception as e:
            return {
                'symbol': local_state.symbol,
                'category': category,
                'status': 'failed',
                'local_status': local_state.local_status,
                'records': existing_records,
                'mode': local_state.local_status,
                'latest_local_date': local_state.latest_local_date,
                'latest_trade_date': effective_target_trade_date,
                'resolved_path': local_state.resolved_path,
                'api_calls': self.REQUESTS_PER_STOCK,
                'error': str(e)[:100],
                **cleaning_skipped_fields,
            }
    
    def download_category(
        self,
        symbols: List[str],
        category: str,
        target_trade_date: Optional[str] = None,
        round_control: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        批量下载某一类别的股票数据

        Args:
            symbols: 股票代码列表
            category: 类别名称
        """
        effective_target_trade_date = target_trade_date or self.latest_trade_date
        print(f"\n{'='*80}")
        print(f"📥 下载 {category.upper()} ({len(symbols)} 只股票)")
        print(f"{'='*80}")
        print(f"时间范围: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"目标交易日: {effective_target_trade_date}")
        if category == 'full_a':
            print(
                '保存目录: '
                + ', '.join(str(path) for path in self.resolver.physical_directories_for_full_a())
            )
        else:
            print(f"保存目录: {self.dirs[category]}")
        print("Tushare API限速: 每分钟500次调用")
        estimated_minutes = (
            len(symbols) * self.REQUESTS_PER_STOCK / self.REQUESTS_PER_MINUTE_BUDGET
        )
        print(f"预计时间: {estimated_minutes:.1f} 分钟")
        print(f"{'='*80}\n")

        results = []
        start_time = time.time()

        for i, symbol in enumerate(symbols, 1):
            if round_control and round_control.get("stop"):
                break
            result = self.download_stock(symbol, category, target_trade_date=effective_target_trade_date)
            results.append(result)

            self.stats['total'] += 1
            status = result['status']
            if status == 'updated':
                self.stats['updated'] += 1
            elif status == 'cached':
                self.stats['cached'] += 1
            elif status == 'stale_cached':
                self.stats['stale_cached'] += 1
            else:
                self.stats['failed'] += 1

            if i % 10 == 0 or i == len(symbols):
                elapsed = time.time() - start_time
                progress = i / len(symbols)
                print(
                    f"  进度: {i}/{len(symbols)} ({progress*100:.1f}%) | "
                    f"cached: {self.stats['cached']} | "
                    f"stale_cached: {self.stats['stale_cached']} | "
                    f"updated: {self.stats['updated']} | "
                    f"failed: {self.stats['failed']} | "
                    f"耗时: {elapsed/60:.1f}分钟"
                )

            if round_control is not None:
                self._record_round_result(round_control, result)
                if round_control.get("stop"):
                    print("\n⏹️ 检测到严格目标日当日数据尚未广泛可用，提前结束本轮剩余下载。")
                    break

            api_calls = int(result.get('api_calls', self.REQUESTS_PER_STOCK) or 0)
            if api_calls > 0:
                time.sleep(api_calls * 60 / self.REQUESTS_PER_MINUTE_BUDGET)

        elapsed = time.time() - start_time
        print(f"\n✅ {category.upper()} 下载完成! 耗时: {elapsed/60:.1f} 分钟")

        # Update the freshness index with every symbol whose date is now known.
        # This covers both freshly downloaded symbols ('updated') and those that
        # were already up-to-date ('cached' / 'suspended_stale').
        index_updates: dict[str, str] = {}
        for r in results:
            sym = r.get("symbol", "")
            date = r.get("latest_local_date", "")
            if sym and date and r.get("status") in ("updated", "cached", "stale_cached"):
                index_updates[sym] = date
        self._flush_freshness_index(index_updates)

        return results

    def _create_round_control(self, target_trade_date: str) -> Dict[str, Any]:
        enabled = (
            target_trade_date == self.strict_trade_date
            and self.strict_trade_date != self.stable_trade_date
        )
        return {
            "enabled": enabled,
            "stop": False,
            "reason": "",
            "observed": 0,
            "stale_cached": 0,
        }

    def _record_round_result(self, round_control: Dict[str, Any], result: Dict[str, Any]) -> None:
        if not round_control.get("enabled") or round_control.get("stop"):
            return
        round_control["observed"] += 1
        if result.get("status") == "stale_cached":
            round_control["stale_cached"] += 1
        if round_control["observed"] < self.strict_early_stop_sample_size:
            return
        stale_ratio = round_control["stale_cached"] / max(round_control["observed"], 1)
        round_control["stale_ratio"] = stale_ratio
        round_control["enabled"] = False
        if stale_ratio >= self.strict_early_stop_stale_ratio:
            round_control["stop"] = True
            round_control["reason"] = "strict_same_day_unavailable"
    
    def _print_completeness_summary(self, completeness: Dict[str, Any]):
        """打印本地数据完整性摘要。"""
        print("\n" + "=" * 80)
        print("🧭 本地数据完整性检查")
        print("=" * 80)
        print(f"目标最新交易日: {completeness['latest_trade_date']}")
        print(
            f"freshness_mode: {completeness.get('freshness_mode', self.freshness_mode)} | "
            f"coverage: {completeness.get('coverage_complete_count', 0)}/"
            f"{completeness.get('expected_scope_count', 0)} "
            f"({completeness.get('coverage_ratio', 0.0):.1%})"
        )
        print(f"完整性状态: {'通过' if completeness['complete'] else '未通过'}")
        print(f"阻塞缺口总数: {completeness['blocking_incomplete_count']}")
        resolver = completeness.get("resolver", {})
        if resolver:
            print(
                "Resolver priority: "
                f"{resolver.get('directory_priority', [])} | "
                f"local-union-fallback={resolver.get('local_union_fallback_used', False)}"
            )
        for category, payload in completeness['categories'].items():
            date_counts = payload.get('date_counts', {})
            latest_count = int(date_counts.get(completeness['latest_trade_date'], 0))
            print(
                f"  - {category}: 最新 {latest_count}/{payload['expected']} | "
                f"阻塞缺口 {payload['blocking_incomplete_count']}"
            )
        print("=" * 80)

    def get_resolver_trace(self) -> dict[str, Any]:
        """返回最近一次 resolver 决策快照。"""
        return dict(self.resolver.snapshot())

    def download_all(
        self,
        components: Optional[Dict] = None,
        max_rounds: int = 1,
        fail_on_incomplete: bool = False,
        allowed_stale_symbols: Optional[List[str] | Set[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> Dict:
        """
        下载全市场数据

        Returns:
            Dict with download statistics
        """
        if components is None:
            components = self.load_components()
        self._refresh_full_a_write_categories(components)
        target_categories = self._resolve_target_categories(components, categories)
        same_day_probe = self._probe_strict_same_day_close_availability(
            components=components,
            target_categories=target_categories,
        )
        preflight_early_stop_reason = ""
        preflight_target_trade_date: str | None = None
        if same_day_probe.get("applicable") and same_day_probe.get("available") is False:
            preflight_early_stop_reason = "strict_same_day_unavailable"
            preflight_target_trade_date = self.stable_trade_date
            print(
                "\n⏹️ 严格目标日全市场收盘价尚未达到覆盖阈值，"
                "直接切换到稳定目标日。"
            )
            print(
                f"strict close probe: {same_day_probe.get('available_count', 0)}/"
                f"{same_day_probe.get('expected_count', 0)} "
                f"({same_day_probe.get('coverage_ratio', 0.0):.1%}) "
                f"via {same_day_probe.get('source') or 'unknown'}"
            )
        preflight_completeness = self.build_completeness_report(
            components=components,
            allowed_stale_symbols=allowed_stale_symbols,
            categories=target_categories,
            target_trade_date=preflight_target_trade_date,
            early_stop_reason=preflight_early_stop_reason,
        )
        effective_target_trade_date = preflight_completeness.get(
            'effective_target_trade_date',
            preflight_completeness.get('latest_trade_date', self.latest_trade_date),
        )

        all_results: Dict[str, Any] = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': {
                'years': self.years,
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'latest_trade_date': effective_target_trade_date,
                'strict_trade_date': self.strict_trade_date,
                'stable_trade_date': self.stable_trade_date,
                'effective_target_trade_date': effective_target_trade_date,
                'freshness_mode': self.freshness_mode,
                'coverage_ratio': preflight_completeness.get('coverage_ratio', 0.0),
                'coverage_threshold': self.coverage_threshold,
                'early_stop_reason': preflight_early_stop_reason,
                'same_day_close_probe': same_day_probe,
                'max_rounds': max_rounds,
                'fail_on_incomplete': fail_on_incomplete,
                'categories': list(target_categories),
                'allowed_stale_symbols': sorted(
                    self._normalize_allowed_symbols(allowed_stale_symbols)
                ),
            },
            'categories': {category: [] for category in target_categories},
            'rounds': [],
            'preflight_completeness': preflight_completeness,
            'completeness': preflight_completeness,
        }

        if preflight_completeness['complete']:
            print("\n✅ 首轮完整性已通过，跳过下载。")
            self._print_completeness_summary(preflight_completeness)
            self._save_report(all_results)
            self._print_final_summary(all_results, 0.0)
            return all_results

        round_symbols = {
            category: self._collect_blocking_symbols(preflight_completeness['categories'].get(category, {}))
            for category in target_categories
        }
        target_total = sum(len(symbols) for symbols in round_symbols.values())

        print("\n" + "=" * 80)
        print("🚀 开始下载完整A股市场数据")
        print("=" * 80)
        print(f"目标分类: {', '.join(target_categories)}")
        print(f"待补齐股票数: {target_total} 只")
        estimated_minutes = (
            target_total
            * self.REQUESTS_PER_STOCK
            / self.REQUESTS_PER_MINUTE_BUDGET
        )
        print(f"预计总时间: {estimated_minutes:.1f} 分钟")
        print("=" * 80)

        total_start = time.time()

        for round_no in range(1, max_rounds + 1):
            print("\n" + "=" * 80)
            print(f"🔁 下载轮次 {round_no}/{max_rounds}")
            print("=" * 80)

            round_payload: Dict[str, Any] = {
                'round': round_no,
                'categories': {},
                'effective_target_trade_date': effective_target_trade_date,
                'early_stop_reason': '',
            }
            round_control = self._create_round_control(effective_target_trade_date)

            for category in target_categories:
                symbols = round_symbols.get(category, [])
                if not symbols:
                    continue
                results = self.download_category(
                    symbols,
                    category,
                    target_trade_date=effective_target_trade_date,
                    round_control=round_control,
                )
                round_payload['categories'][category] = results
                all_results['categories'][category].extend(results)
                if round_control.get('stop'):
                    break

            if round_control.get('stop'):
                effective_target_trade_date = self.stable_trade_date
                round_payload['effective_target_trade_date'] = effective_target_trade_date
                round_payload['early_stop_reason'] = round_control.get('reason', '')
                all_results['config']['effective_target_trade_date'] = effective_target_trade_date
                all_results['config']['latest_trade_date'] = effective_target_trade_date
                all_results['config']['early_stop_reason'] = round_control.get('reason', '')
                completeness = self.build_completeness_report(
                    components=components,
                    allowed_stale_symbols=allowed_stale_symbols,
                    categories=target_categories,
                    target_trade_date=effective_target_trade_date,
                    early_stop_reason=round_control.get('reason', ''),
                )
            else:
                completeness = self.build_completeness_report(
                    components=components,
                    allowed_stale_symbols=allowed_stale_symbols,
                    categories=target_categories,
                    target_trade_date=effective_target_trade_date,
                )
            round_payload['completeness'] = completeness
            all_results['rounds'].append(round_payload)
            all_results['completeness'] = completeness
            self._print_completeness_summary(completeness)

            if completeness['complete']:
                break

            if round_no >= max_rounds:
                break

            round_symbols = {
                category: self._collect_blocking_symbols(payload)
                for category, payload in completeness['categories'].items()
            }

        total_elapsed = time.time() - total_start

        self._save_report(all_results)
        self._print_final_summary(all_results, total_elapsed)

        if fail_on_incomplete and not all_results.get('completeness', {}).get('complete', False):
            raise RuntimeError("A股全市场数据未完整更新到最新交易日，已按要求终止")
        
        return all_results
    
    def _save_report(self, results: Dict):
        """保存下载报告"""
        report_file = f"{self.data_dir}/download_report_{results['timestamp']}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n📊 详细报告已保存: {report_file}")
    
    def _print_final_summary(self, results: Dict, elapsed: float):
        """打印最终汇总"""
        print("\n" + "=" * 80)
        print("📊 下载完成汇总")
        print("=" * 80)
        print(f"总耗时: {elapsed/60:.1f} 分钟 ({elapsed/3600:.2f} 小时)")
        print()
        total = max(int(self.stats['total']), 1)
        print(f"总计处理: {self.stats['total']} 只股票")
        print(f"  💾 cached:        {self.stats['cached']} 只 ({self.stats['cached']/total*100:.1f}%)")
        print(f"  💤 stale_cached:  {self.stats['stale_cached']} 只 ({self.stats['stale_cached']/total*100:.1f}%)")
        print(f"  🔄 updated:       {self.stats['updated']} 只 ({self.stats['updated']/total*100:.1f}%)")
        print(f"  ❌ failed:        {self.stats['failed']} 只 ({self.stats['failed']/total*100:.1f}%)")
        print()
        completeness = results.get('completeness')
        if completeness:
            print(f"完整性检查: {'通过' if completeness['complete'] else '未通过'}")
            print(f"目标最新交易日: {completeness['latest_trade_date']}")
            print(
                f"freshness_mode: {completeness.get('freshness_mode', self.freshness_mode)} | "
                f"coverage: {completeness.get('coverage_complete_count', 0)}/"
                f"{completeness.get('expected_scope_count', 0)} "
                f"({completeness.get('coverage_ratio', 0.0):.1%})"
            )
            print(f"阻塞缺口总数: {completeness['blocking_incomplete_count']}")
            print()
        print("=" * 80)
        print(f"数据保存位置: {self.data_dir}/")
        print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='下载完整A股市场数据')
    parser.add_argument('--years', type=int, default=3, help='下载年数 (默认3)')
    parser.add_argument('--category', type=str, choices=['full_a', 'hs300', 'zz500', 'zz1000', 'all'],
                       default='all', help='下载类别 (默认all)')
    parser.add_argument('--max-rounds', type=int, default=1, help='最多补齐轮次 (默认1)')
    parser.add_argument(
        '--fail-on-incomplete',
        action='store_true',
        help='若最终仍未完整更新到最新交易日，则返回非零退出码',
    )
    parser.add_argument(
        '--check-complete',
        action='store_true',
        help='仅检查本地完整性，不发起下载',
    )
    parser.add_argument(
        '--allowed-stale-symbols',
        nargs='*',
        default=None,
        help='允许跳过完整性校验的 symbol 列表，如 002859.SZ',
    )
    
    args = parser.parse_args()
    
    downloader = CNFullMarketDownloader(years=args.years)
    components = downloader.load_components()
    target_categories = None if args.category == 'all' else [args.category]

    if args.check_complete:
        completeness = downloader.build_completeness_report(
            components=components,
            allowed_stale_symbols=args.allowed_stale_symbols,
            categories=target_categories,
        )
        downloader._print_completeness_summary(completeness)
        if args.fail_on_incomplete and not completeness['complete']:
            raise SystemExit(1)
        return
    
    downloader.download_all(
        components,
        max_rounds=args.max_rounds,
        fail_on_incomplete=args.fail_on_incomplete,
        allowed_stale_symbols=args.allowed_stale_symbols,
        categories=target_categories,
    )


if __name__ == '__main__':
    main()

"""Freshness and local-completeness helpers for the CN market downloader."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from quant_investor.market.cn_symbol_status import (
    CNSymbolLocalStatusResult,
    evaluate_symbol_local_status,
)


class CNDownloadFreshnessMixin:
    """Local freshness, completeness, and cache-index helpers."""

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
    def _normalize_allowed_symbols(symbols: Optional[List[str] | Set[str]]) -> Set[str]:
        """标准化允许跳过完整性检查的股票列表。"""
        if not symbols:
            return set()
        normalized = set()
        for symbol in symbols:
            if symbol and str(symbol).strip():
                normalized.add(str(symbol).strip().upper())
        return normalized

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
        if self.pro is None:
            self._active_listing_dates_cache = {}
            return self._active_listing_dates_cache

        try:
            stock_basic = self.pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,list_date",
            )
        except Exception:
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
        except Exception:
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
            except Exception:
                pass

            return strict_trade_date, stable_trade_date
        except Exception:
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
        """从本地 CSV 缓存中推断最新交易日。"""
        latest_dates: list[str] = []
        for directory in self.resolver.physical_directories_for_full_a():
            for csv_file in directory.glob("*.csv"):
                try:
                    result = self.csv_reader.read_path(csv_file, universe_key="full_a")
                except Exception:
                    continue
                df = result.frame
                if df is None or df.empty:
                    continue
                date_column = "trade_date" if "trade_date" in df.columns else "date" if "date" in df.columns else ""
                if not date_column:
                    continue
                try:
                    local_dates = pd.to_datetime(df[date_column], errors="coerce").dt.strftime("%Y%m%d")
                    values = [value for value in local_dates.dropna().astype(str).tolist() if value.strip()]
                    if values:
                        latest_dates.append(max(values))
                except Exception:
                    continue
        return max(latest_dates) if latest_dates else ""

    def _build_local_symbol_universe(self) -> Dict[str, List[str]]:
        """从本地 CSV 文件构建不依赖 Tushare 的组件字典。"""
        full_a_symbols, source_paths = self.resolver.collect_full_a_inventory(local_union_fallback_used=True)
        category_lists: Dict[str, Set[str]] = {category: set() for category in self.SUPPORTED_CATEGORIES}
        for symbol, path in source_paths.items():
            parent = Path(path).parent.name
            if parent in category_lists:
                category_lists[parent].add(symbol)

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
        except Exception:
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
            csv_reader=self.csv_reader,
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
        report['data_quality_issue_count'] = len(report['data_quality_issues'])

        # Persist any date discoveries from the slow path so the next call
        # can use the fast (index) path for those symbols.
        self._flush_freshness_index(index_updates)

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

    def _load_latest_suspended_symbols(self, target_trade_date: str) -> Set[str]:
        """
        获取最新交易日停牌标的集合。

        对于当日停牌的股票，本地最后一个成交日早于最新交易日是合理状态，
        不应阻塞"完整性"判断。

        结果按 trade_date 写入磁盘(永久有效，历史停牌数据不会变化)，
        下次同日调用直接从磁盘加载，完全跳过 Tushare API 调用。
        """
        if target_trade_date in self._latest_suspended_symbols_cache:
            return self._latest_suspended_symbols_cache[target_trade_date]

        # ── disk cache hit ──
        disk_path = self._suspend_cache_path(target_trade_date)
        try:
            if disk_path.exists():
                raw = json.loads(disk_path.read_text(encoding="utf-8"))
                symbols: Set[str] = set(raw) if isinstance(raw, list) else set()
                self._latest_suspended_symbols_cache[target_trade_date] = symbols
                return symbols
        except Exception:
            pass

        if self.pro is None:
            self._latest_suspended_symbols_cache[target_trade_date] = set()
            return self._latest_suspended_symbols_cache[target_trade_date]

        try:
            suspend_df = self.pro.suspend_d(trade_date=target_trade_date)
            if suspend_df is None or suspend_df.empty:
                symbols = set()
            else:
                filtered = suspend_df.copy()
                if 'suspend_type' in filtered.columns:
                    filtered = filtered[filtered['suspend_type'].astype(str).str.upper() == 'S']
                symbols = {
                    str(symbol).upper()
                    for symbol in filtered.get('ts_code', pd.Series(dtype=str)).dropna().astype(str)
                }

            # ── persist to disk cache (historic data is immutable) ──
            try:
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                disk_path.write_text(
                    json.dumps(sorted(symbols), ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass

            self._latest_suspended_symbols_cache[target_trade_date] = symbols
            return symbols
        except Exception:
            self._latest_suspended_symbols_cache[target_trade_date] = set()
            return self._latest_suspended_symbols_cache[target_trade_date]

    def _prefetch_suspended_symbols(self, dates: List[str]) -> None:
        """Pre-warm the suspend cache for multiple dates using a thread pool.

        This turns two sequential ~10s Tushare API calls into a single
        ~10s parallel round-trip, cutting the suspend overhead in half.
        Dates already present in the in-memory or disk cache are skipped.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        uncached = [
            d for d in dates
            if d not in self._latest_suspended_symbols_cache
            and not self._suspend_cache_path(d).exists()
        ]
        if not uncached:
            return

        def _fetch(date: str) -> None:
            self._load_latest_suspended_symbols(date)

        with ThreadPoolExecutor(max_workers=len(uncached)) as pool:
            futures = {pool.submit(_fetch, d): d for d in uncached}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
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
        except Exception:
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
        except Exception:
            pass


#!/usr/bin/env python3
"""
US 全市场下载器。

能力要求：
1. 先判定本轮要求的最新完整交易日；
2. 优先使用 Tushare 高积分接口，失败后回退 yfinance；
3. 支持 ticker alias 映射，例如 BRK.B/BF.B <-> BRK-B/BF-B；
4. 输出严格完整性报告，供统一 market pipeline 阻塞或放行正式分析。
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

try:
    import tushare as ts

    TUSHARE_AVAILABLE = True
except ImportError:  # pragma: no cover
    ts = None
    TUSHARE_AVAILABLE = False


CATEGORY_ORDER = ("large_cap", "mid_cap", "small_cap")
DEFAULT_UNIVERSE_FILE = "data/us_universe/complete_us_universe.json"
ALIAS_OVERRIDES = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}
NO_DATA_ERROR_PATTERNS = (
    "possibly delisted",
    "no timezone found",
    "no price data found",
    "no data found",
    "quote not found",
    "404",
)


def _normalize_storage_symbol(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    return ALIAS_OVERRIDES.get(text, text.replace(".", "-"))


def _dedupe_preserve_order(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        normalized = _normalize_storage_symbol(symbol)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _dedupe_raw_preserve_order(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for symbol in symbols:
        raw = str(symbol or "").strip().upper()
        if raw and raw not in seen:
            seen.add(raw)
            ordered.append(raw)
    return ordered


def _now_new_york() -> datetime:
    if ZoneInfo is None:  # pragma: no cover
        return datetime.utcnow()
    return datetime.now(ZoneInfo("America/New_York"))


class FullMarketDownloader:
    """全市场美股下载器。"""

    def __init__(
        self,
        data_dir: str = "data/us_market_full",
        years: int = 3,
        max_workers: int = 8,
        batch_size: int = 100,
    ) -> None:
        self.data_dir = data_dir
        self.years = years
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=years * 365 + 30)

        self.dirs = {
            category: f"{data_dir}/{category}"
            for category in CATEGORY_ORDER
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        self.stats: dict[str, int] = {}
        self._reset_stats()
        self.download_log: list[dict[str, Any]] = []
        self._tushare_quota_exhausted = False
        self.pro = None
        if TUSHARE_AVAILABLE:
            try:
                self.pro = create_tushare_pro(ts, config.TUSHARE_TOKEN, config.TUSHARE_URL)
            except Exception:
                self.pro = None

    def _reset_stats(self) -> None:
        self.stats = {
            "total": 0,
            "success": 0,
            "updated": 0,
            "cached": 0,
            "no_data": 0,
            "failed": 0,
        }

    @staticmethod
    def _candidate_provider_symbols(symbol: str) -> list[str]:
        normalized = _normalize_storage_symbol(symbol)
        candidates = [normalized]
        if "-" in normalized:
            candidates.append(normalized.replace("-", "."))
        if "." in normalized:
            candidates.append(normalized.replace(".", "-"))
        return _dedupe_raw_preserve_order(candidates)

    @staticmethod
    def _candidate_storage_symbols(symbol: str) -> list[str]:
        raw = str(symbol or "").strip().upper()
        candidates = [raw, _normalize_storage_symbol(raw)]
        if "-" in raw:
            candidates.append(raw.replace("-", "."))
        if "." in raw:
            candidates.append(raw.replace(".", "-"))
        return _dedupe_raw_preserve_order(candidates)

    def _normalize_categories(self, categories: Optional[list[str]]) -> list[str]:
        if not categories:
            return list(CATEGORY_ORDER)
        return [category for category in CATEGORY_ORDER if category in categories]

    def detect_latest_available_trade_date(self) -> str:
        """
        判定当前最新完整交易日。

        优先依赖 SPY 最近行情作为可用性探针，但会避开美股当日盘中尚未完整的 bar。
        """
        now_et = _now_new_york()
        today_text = now_et.strftime("%Y-%m-%d")

        if yf is not None:
            try:
                probe_df = yf.Ticker("SPY").history(
                    period="10d",
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                )
                if probe_df is not None and not probe_df.empty:
                    dates = sorted(str(index.date()) for index in probe_df.index)
                    latest = dates[-1]
                    if latest == today_text and now_et.hour < 18 and len(dates) >= 2:
                        return dates[-2]
                    return latest
            except Exception:
                pass

        candidate = now_et
        if candidate.hour < 18:
            candidate -= timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
        return candidate.strftime("%Y-%m-%d")

    def _load_existing_data(self, filepath: str) -> tuple[pd.DataFrame, Optional[str]]:
        if not os.path.exists(filepath):
            return pd.DataFrame(), None

        try:
            existing_df = pd.read_csv(filepath)
        except Exception:
            return pd.DataFrame(), None

        if existing_df.empty:
            return pd.DataFrame(), None

        date_col = None
        for candidate in ("Date", "date", "trade_date"):
            if candidate in existing_df.columns:
                date_col = candidate
                break
        if date_col is None:
            return pd.DataFrame(), None

        normalized = existing_df.copy()
        normalized[date_col] = pd.to_datetime(normalized[date_col], errors="coerce")
        normalized = normalized.dropna(subset=[date_col]).sort_values(date_col)
        normalized = normalized.drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
        normalized["Date"] = normalized[date_col].dt.strftime("%Y-%m-%d")

        preferred_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Amount"]
        available = [column for column in preferred_columns if column in normalized.columns]
        latest_date = normalized["Date"].iloc[-1] if not normalized.empty else None
        return normalized[available], latest_date

    def _locate_existing_file(self, symbol: str, category: str) -> Optional[Path]:
        category_dir = Path(self.dirs[category])
        for candidate in self._candidate_storage_symbols(symbol):
            path = category_dir / f"{candidate}.csv"
            if path.exists():
                return path
        return None

    @staticmethod
    def _format_tushare_us_frame(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.rename(
            columns={
                "trade_date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "vol": "Volume",
                "amount": "Amount",
            }
        ).copy()
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = normalized.dropna(subset=["Date"]).sort_values("Date")
        normalized["Date"] = normalized["Date"].dt.strftime("%Y-%m-%d")
        keep_cols = [column for column in ["Date", "Open", "High", "Low", "Close", "Volume", "Amount"] if column in normalized.columns]
        return normalized[keep_cols].reset_index(drop=True)

    @staticmethod
    def _format_yfinance_frame(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.reset_index().rename(columns={"Datetime": "Date"}).copy()
        if "Date" not in normalized.columns and "index" in normalized.columns:
            normalized = normalized.rename(columns={"index": "Date"})
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        normalized = normalized.dropna(subset=["Date"]).sort_values("Date")
        normalized["Date"] = normalized["Date"].dt.strftime("%Y-%m-%d")
        keep_cols = [column for column in ["Date", "Open", "High", "Low", "Close", "Volume", "Amount"] if column in normalized.columns]
        return normalized[keep_cols].reset_index(drop=True)

    def _download_from_tushare(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        if not self.pro or self._tushare_quota_exhausted:
            return None, None

        for provider_symbol in self._candidate_provider_symbols(symbol):
            try:
                df = self.pro.us_daily(
                    ts_code=provider_symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                if df is None or df.empty:
                    continue
                return self._format_tushare_us_frame(df), provider_symbol
            except Exception as exc:
                message = str(exc)
                if "每天最多访问该接口" in message or "每分钟最多访问该接口" in message:
                    self._tushare_quota_exhausted = True
                    return None, None
                continue

        return None, None

    def _download_from_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        if yf is None:
            return None, None

        start_text = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
        end_text = (datetime.strptime(end_date, "%Y%m%d") + timedelta(days=2)).strftime("%Y-%m-%d")

        for provider_symbol in self._candidate_provider_symbols(symbol):
            try:
                ticker = yf.Ticker(provider_symbol)
                df = ticker.history(
                    start=start_text,
                    end=end_text,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                )
                if df is None or df.empty:
                    continue
                return self._format_yfinance_frame(df), provider_symbol
            except Exception as exc:
                message = str(exc).lower()
                if any(pattern in message for pattern in NO_DATA_ERROR_PATTERNS):
                    continue
                continue

        return None, None

    @staticmethod
    def _merge_frames(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        if existing_df.empty:
            merged = new_df.copy()
        else:
            merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
        merged = merged.dropna(subset=["Date"]).sort_values("Date")
        merged = merged.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
        return merged

    def load_universe(self, universe_file: str = DEFAULT_UNIVERSE_FILE) -> dict[str, Any]:
        with open(universe_file, "r", encoding="utf-8") as file:
            raw_universe = json.load(file)

        normalized: dict[str, list[str]] = {}
        seen: set[str] = set()
        for category in CATEGORY_ORDER:
            deduped = []
            for symbol in raw_universe.get(category, []):
                normalized_symbol = _normalize_storage_symbol(symbol)
                if normalized_symbol and normalized_symbol not in seen:
                    seen.add(normalized_symbol)
                    deduped.append(normalized_symbol)
            normalized[category] = deduped

        all_symbols = [symbol for category in CATEGORY_ORDER for symbol in normalized[category]]
        universe: dict[str, Any] = {
            category: normalized[category]
            for category in CATEGORY_ORDER
        }
        universe["all"] = all_symbols
        universe["stats"] = {
            category: len(normalized[category])
            for category in CATEGORY_ORDER
        }
        universe["stats"]["total_unique"] = len(all_symbols)
        universe["storage_aliases"] = {
            key: _normalize_storage_symbol(value)
            for key, value in ALIAS_OVERRIDES.items()
        }
        universe["metadata"] = dict(raw_universe.get("metadata", {}))

        print("=" * 80)
        print("📊 加载 US investable universe")
        print("=" * 80)
        print(f"大盘股: {len(universe['large_cap'])} 只")
        print(f"中盘股: {len(universe['mid_cap'])} 只")
        print(f"小盘股: {len(universe['small_cap'])} 只")
        print(f"总计: {universe['stats']['total_unique']} 只")
        print("=" * 80)

        return universe

    def build_completeness_report(
        self,
        universe: Optional[dict[str, Any]] = None,
        categories: Optional[list[str]] = None,
        required_latest_trade_date: Optional[str] = None,
    ) -> dict[str, Any]:
        if universe is None:
            universe = self.load_universe()

        latest_trade_date = required_latest_trade_date or self.detect_latest_available_trade_date()
        target_categories = self._normalize_categories(categories)
        report: dict[str, Any] = {
            "latest_trade_date": latest_trade_date,
            "complete": True,
            "blocking_incomplete_count": 0,
            "categories_checked": list(target_categories),
            "categories": {},
        }

        for category in target_categories:
            date_counts: Counter[str] = Counter()
            missing_symbols: list[str] = []
            stale_symbols: list[dict[str, str]] = []
            expected_symbols = _dedupe_preserve_order(universe.get(category, []))

            for symbol in expected_symbols:
                filepath = self._locate_existing_file(symbol, category)
                if filepath is None:
                    missing_symbols.append(symbol)
                    continue

                _existing_df, latest_local_date = self._load_existing_data(str(filepath))
                if not latest_local_date:
                    missing_symbols.append(symbol)
                    continue

                date_counts[latest_local_date] += 1
                if latest_local_date != latest_trade_date:
                    stale_symbols.append(
                        {
                            "symbol": symbol,
                            "latest_local_date": latest_local_date,
                        }
                    )

            blocking_count = len(missing_symbols) + len(stale_symbols)
            report["categories"][category] = {
                "expected": len(expected_symbols),
                "latest_trade_date": latest_trade_date,
                "date_counts": dict(sorted(date_counts.items())),
                "missing_symbols": sorted(missing_symbols),
                "stale_symbols": stale_symbols,
                "blocking_missing_symbols": sorted(missing_symbols),
                "blocking_stale_symbols": stale_symbols,
                "blocking_incomplete_count": blocking_count,
            }
            if blocking_count > 0:
                report["complete"] = False
                report["blocking_incomplete_count"] += blocking_count

        return report

    def _print_completeness_summary(self, completeness: dict[str, Any]) -> None:
        print(f"目标最新交易日: {completeness.get('latest_trade_date')}")
        print(f"完整性状态: {'通过' if completeness.get('complete') else '未通过'}")
        print(f"阻塞缺口总数: {completeness.get('blocking_incomplete_count', 0)}")
        for category in completeness.get("categories_checked", []):
            payload = completeness.get("categories", {}).get(category, {})
            latest = payload.get("latest_trade_date")
            latest_count = int(payload.get("date_counts", {}).get(latest, 0)) if latest else 0
            print(
                f"  - {category}: 最新 {latest_count}/{payload.get('expected', 0)} | "
                f"阻塞缺口 {payload.get('blocking_incomplete_count', 0)}"
            )

    def download_stock(
        self,
        symbol: str,
        category: str,
        force_refresh: bool = False,
        required_latest_trade_date: Optional[str] = None,
    ) -> dict[str, Any]:
        storage_symbol = _normalize_storage_symbol(symbol)
        latest_trade_date = required_latest_trade_date or self.detect_latest_available_trade_date()
        filepath = Path(self.dirs[category]) / f"{storage_symbol}.csv"
        existing_path = self._locate_existing_file(storage_symbol, category) or filepath
        existing_df, latest_local_date = self._load_existing_data(str(existing_path))

        if latest_local_date == latest_trade_date and not force_refresh and not existing_df.empty:
            return {
                "symbol": storage_symbol,
                "category": category,
                "status": "cached",
                "records": len(existing_df),
                "latest_local_date": latest_local_date,
                "latest_trade_date": latest_trade_date,
                "source": None,
                "provider_symbol": None,
                "error": None,
            }

        start_text = self.start_date.strftime("%Y%m%d")
        if latest_local_date:
            overlap_start = pd.to_datetime(latest_local_date) - timedelta(days=5)
            start_text = max(start_text, overlap_start.strftime("%Y%m%d"))
        end_text = latest_trade_date.replace("-", "")

        df, provider_symbol = self._download_from_tushare(storage_symbol, start_text, end_text)
        source = "tushare" if df is not None and not df.empty else None
        if df is None or df.empty:
            df, provider_symbol = self._download_from_yfinance(storage_symbol, start_text, end_text)
            source = "yfinance" if df is not None and not df.empty else source

        if df is None or df.empty:
            return {
                "symbol": storage_symbol,
                "category": category,
                "status": "no_data" if existing_df.empty else "failed",
                "records": len(existing_df),
                "latest_local_date": latest_local_date,
                "latest_trade_date": latest_trade_date,
                "source": source,
                "provider_symbol": provider_symbol,
                "error": "No data from tushare/yfinance",
            }

        final_df = self._merge_frames(existing_df, df)
        final_df.to_csv(filepath, index=False)
        latest_saved_date = final_df["Date"].iloc[-1]

        return {
            "symbol": storage_symbol,
            "category": category,
            "status": "updated" if not existing_df.empty else "success",
            "records": len(final_df),
            "latest_local_date": latest_saved_date,
            "latest_trade_date": latest_trade_date,
            "source": source,
            "provider_symbol": provider_symbol,
            "error": None,
        }

    def download_category(
        self,
        symbols: list[str],
        category: str,
        force_refresh_symbols: Optional[Iterable[str]] = None,
        required_latest_trade_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        scoped_symbols = _dedupe_preserve_order(symbols)
        force_refresh_set = {
            _normalize_storage_symbol(symbol)
            for symbol in (force_refresh_symbols or [])
        }
        latest_trade_date = required_latest_trade_date or self.detect_latest_available_trade_date()
        print(f"\n{'=' * 80}")
        print(f"📥 下载 {category.upper()} ({len(scoped_symbols)} 只股票)")
        print(f"{'=' * 80}")
        print(f"目标最新交易日: {latest_trade_date}")
        print(f"时间范围: {self.start_date.strftime('%Y-%m-%d')} 至 {latest_trade_date}")
        print(f"保存目录: {self.dirs[category]}")
        print(f"并行线程: {self.max_workers}")
        print(f"{'=' * 80}\n")

        results: list[dict[str, Any]] = []
        started = time.time()
        total_batches = (len(scoped_symbols) + self.batch_size - 1) // self.batch_size

        for batch_index in range(total_batches):
            batch_start = batch_index * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(scoped_symbols))
            batch_symbols = scoped_symbols[batch_start:batch_end]
            print(
                f"  批次 {batch_index + 1}/{total_batches} "
                f"({batch_start + 1}-{batch_end}/{len(scoped_symbols)})..."
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(
                        self.download_stock,
                        symbol,
                        category,
                        _normalize_storage_symbol(symbol) in force_refresh_set,
                        latest_trade_date,
                    ): symbol
                    for symbol in batch_symbols
                }

                for future in as_completed(future_to_symbol):
                    result = future.result()
                    results.append(result)
                    self.download_log.append(result)
                    self.stats["total"] += 1
                    status = result["status"]
                    if status in self.stats:
                        self.stats[status] += 1
                    else:
                        self.stats["failed"] += 1

            elapsed = time.time() - started
            progress = len(results) / max(len(scoped_symbols), 1)
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0
            print(
                f"    进度: {len(results)}/{len(scoped_symbols)} ({progress * 100:.1f}%) | "
                f"成功: {self.stats['success'] + self.stats['updated']} | "
                f"缓存: {self.stats['cached']} | "
                f"失败: {self.stats['failed']} | "
                f"无数据: {self.stats['no_data']} | "
                f"ETA: {eta / 60:.1f} 分钟"
            )

        print(f"\n✅ {category.upper()} 下载完成! 耗时: {(time.time() - started) / 60:.1f} 分钟")
        return results

    def download_all(
        self,
        universe: Optional[dict[str, Any]] = None,
        categories: Optional[list[str]] = None,
        force_refresh_by_category: Optional[dict[str, list[str]]] = None,
    ) -> dict[str, Any]:
        if universe is None:
            universe = self.load_universe()

        self._reset_stats()
        selected_categories = self._normalize_categories(categories)
        latest_trade_date = self.detect_latest_available_trade_date()
        all_results: dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "latest_trade_date": latest_trade_date,
            "config": {
                "years": self.years,
                "max_workers": self.max_workers,
                "batch_size": self.batch_size,
            },
            "categories": {},
        }

        print("\n" + "=" * 80)
        print("🚀 开始下载完整美股市场数据")
        print("=" * 80)
        print(f"目标类别: {selected_categories}")
        print(f"目标最新交易日: {latest_trade_date}")
        print(f"总计股票数: {sum(len(universe.get(category, [])) for category in selected_categories)} 只")
        print("=" * 80)

        total_started = time.time()
        for category in selected_categories:
            all_results["categories"][category] = self.download_category(
                universe.get(category, []),
                category,
                force_refresh_symbols=(force_refresh_by_category or {}).get(category, []),
                required_latest_trade_date=latest_trade_date,
            )

        elapsed = time.time() - total_started
        self._save_report(all_results)
        self._print_final_summary(elapsed)
        return all_results

    def _save_report(self, results: dict[str, Any]) -> None:
        report_file = f"{self.data_dir}/download_report_{results['timestamp']}.json"
        with open(report_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False, default=str)
        print(f"\n📊 详细报告已保存: {report_file}")

    def _print_final_summary(self, elapsed: float) -> None:
        total = max(self.stats["total"], 1)
        print("\n" + "=" * 80)
        print("📊 下载完成汇总")
        print("=" * 80)
        print(f"总耗时: {elapsed / 60:.1f} 分钟 ({elapsed / 3600:.2f} 小时)")
        print(f"总计处理: {self.stats['total']} 只股票")
        print(f"  ✅ 新增成功: {self.stats['success']} 只 ({self.stats['success'] / total * 100:.1f}%)")
        print(f"  🔄 更新成功: {self.stats['updated']} 只 ({self.stats['updated'] / total * 100:.1f}%)")
        print(f"  💾 缓存跳过: {self.stats['cached']} 只 ({self.stats['cached'] / total * 100:.1f}%)")
        print(f"  ⚠️ 无数据:   {self.stats['no_data']} 只 ({self.stats['no_data'] / total * 100:.1f}%)")
        print(f"  ❌ 失败:     {self.stats['failed']} 只 ({self.stats['failed'] / total * 100:.1f}%)")
        print("=" * 80)
        print(f"数据保存位置: {self.data_dir}/")
        print("=" * 80)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="下载完整美股市场数据")
    parser.add_argument("--years", type=int, default=3, help="下载年数 (默认3)")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数 (默认8)")
    parser.add_argument("--batch", type=int, default=100, help="每批处理数量 (默认100)")
    parser.add_argument(
        "--category",
        type=str,
        choices=["large_cap", "mid_cap", "small_cap", "all"],
        default="all",
        help="下载类别 (默认 all)",
    )
    parser.add_argument("--check-complete", action="store_true", help="只检查完整性，不执行下载")

    args = parser.parse_args()
    downloader = FullMarketDownloader(
        years=args.years,
        max_workers=args.workers,
        batch_size=args.batch,
    )
    universe = downloader.load_universe()
    categories = list(CATEGORY_ORDER) if args.category == "all" else [args.category]

    if args.check_complete:
        report = downloader.build_completeness_report(
            universe=universe,
            categories=categories,
            required_latest_trade_date=downloader.detect_latest_available_trade_date(),
        )
        downloader._print_completeness_summary(report)
        return

    downloader.download_all(universe=universe, categories=categories)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download Full US Market Data - 下载完整美股市场数据

下载市值 100 亿美元以上美股的3年历史数据；低于门槛或未知市值的股票不进入下载/分析池。
美股批量价格默认使用 yfinance，Tushare us_daily 仅作为可配置后备源。
"""

import os
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List, Dict, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.market.us_market_cap_filter import (
    USMarketCapFilter,
)
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.market_data_store import MarketDataStore

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False


DEFAULT_US_PRICE_PROVIDER = "yfinance"
US_PRICE_PROVIDER_ENV = "MYQUANT_US_PRICE_PROVIDER"


class FullMarketDownloader:
    """全市场数据下载器"""
    
    def __init__(self, 
                 data_dir: str = 'data/us_market_full',
                 years: int = 3,
                 max_workers: int = 8,
                 batch_size: int = 100,
                 min_market_cap_usd: int | float | None = None,
                 market_cap_cache_file: str | None = None):
        """
        初始化下载器
        
        Args:
            data_dir: 数据保存目录
            years: 下载年数
            max_workers: 并行下载线程数
            batch_size: 每批处理的股票数
            min_market_cap_usd: 美股最小市值门槛，默认 100 亿美元
            market_cap_cache_file: 市值缓存文件
        """
        self.data_dir = data_dir
        self.data_root = self._resolve_data_root(data_dir)
        self.store = MarketDataStore(market="US", data_root=self.data_root)
        self.market_reader = MarketDataReader(market="US", data_root=self.data_root)
        self.years = years
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.market_cap_filter = USMarketCapFilter(
            threshold_usd=min_market_cap_usd,
            cache_file=market_cap_cache_file,
            max_workers=max_workers,
        )
        self.price_provider = self._normalize_price_provider(
            os.environ.get(US_PRICE_PROVIDER_ENV, DEFAULT_US_PRICE_PROVIDER)
        )
        
        # 创建分层目录
        self.dirs = {
            'full_us': f"{data_dir}/full_us",
            'large_cap': f"{data_dir}/large_cap",
            'mid_cap': f"{data_dir}/mid_cap",
            'small_cap': f"{data_dir}/small_cap"
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 计算日期范围
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=years*365 + 30)
        
        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'cached': 0,
            'no_data': 0
        }
        
        # 下载记录
        self.download_log = []
        self.pro = None
        self._tushare_quota_exhausted = False
        if TUSHARE_AVAILABLE:
            try:
                self.pro = create_tushare_pro(ts, config.TUSHARE_TOKEN, config.TUSHARE_URL)
            except Exception:
                self.pro = None

    @staticmethod
    def _normalize_price_provider(value: str | None) -> str:
        provider = str(value or DEFAULT_US_PRICE_PROVIDER).strip().lower()
        if provider in {"auto", "yfinance", "akshare", "tushare"}:
            return provider
        return DEFAULT_US_PRICE_PROVIDER

    @staticmethod
    def _resolve_data_root(data_dir: str) -> Path:
        path = Path(data_dir)
        if path.name == "us_market_full":
            return path.parent
        if path.name in {"full_us", "large_cap", "mid_cap", "small_cap"}:
            return path.parent.parent
        return path

    def _provider_order(self) -> list[str]:
        """Return US OHLCV provider order. yfinance is default for bulk US runs."""
        if self.price_provider == "tushare":
            return ["tushare", "yfinance", "akshare"]
        if self.price_provider == "akshare":
            return ["akshare", "yfinance", "tushare"]
        return ["yfinance", "akshare", "tushare"]

    @staticmethod
    def _is_tushare_quota_error(message: str) -> bool:
        text = str(message or "")
        return any(
            keyword in text
            for keyword in (
                "频率超限",
                "请求上限",
                "每天最多访问该接口",
                "每分钟最多访问该接口",
                "每小时最多访问该接口",
                "最多访问该接口",
            )
        )

    @staticmethod
    def _format_tushare_us_frame(df: pd.DataFrame) -> pd.DataFrame:
        """标准化 Tushare us_daily 输出为本地 Parquet 写入格式。"""
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
        keep_cols = [
            col for col in ["Date", "Open", "High", "Low", "Close", "Volume", "Amount"]
            if col in normalized.columns
        ]
        return normalized[keep_cols].reset_index(drop=True)

    def _latest_cached_date(self, symbol: str) -> Optional[date]:
        try:
            latest = self.market_reader.peek_symbol_latest_date(
                symbol,
                universe_key="full_us",
            )
        except Exception:
            return None
        if not latest:
            return None
        parsed = pd.to_datetime(latest, format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()

    def _expected_latest_date(self) -> date:
        expected = self.end_date.date() - timedelta(days=1)
        while expected.weekday() >= 5:
            expected -= timedelta(days=1)
        return expected

    def _normalize_downloaded_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        if "Date" in normalized.columns:
            normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        elif "date" in normalized.columns:
            normalized = normalized.rename(columns={"date": "Date"})
            normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
        else:
            return normalized
        normalized = normalized.dropna(subset=["Date"]).drop_duplicates(subset=["Date"])
        normalized = normalized.sort_values("Date")
        normalized["Date"] = normalized["Date"].dt.strftime("%Y-%m-%d")
        return normalized.reset_index(drop=True)

    def _download_from_tushare(self, symbol: str) -> Optional[pd.DataFrame]:
        """优先尝试从 Tushare 拉取美股数据。"""
        if not self.pro or self._tushare_quota_exhausted:
            return None

        try:
            df = self.pro.us_daily(
                ts_code=symbol,
                start_date=self.start_date.strftime("%Y%m%d"),
                end_date=self.end_date.strftime("%Y%m%d"),
            )
            if df is None or df.empty:
                return None
            return self._format_tushare_us_frame(df)
        except Exception as e:
            message = str(e)
            if self._is_tushare_quota_error(message):
                self._tushare_quota_exhausted = True
            return None

    def _download_from_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """从 yfinance 拉取美股数据。"""
        if yf is None:
            return None
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date.strftime('%Y-%m-%d'),
                end=self.end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            if df.empty:
                return None
            df.reset_index(inplace=True)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            return df
        except Exception:
            return None

    def _download_from_akshare(self, symbol: str) -> Optional[pd.DataFrame]:
        """Optional AKShare fallback when installed locally."""
        try:
            import akshare as ak  # type: ignore
        except Exception:
            return None

        try:
            try:
                df = ak.stock_us_daily(symbol=symbol, adjust="")
            except TypeError:
                df = ak.stock_us_daily(symbol=symbol)
            if df is None or df.empty:
                return None
            normalized = df.copy()
            rename_map = {
                "date": "Date",
                "日期": "Date",
                "open": "Open",
                "开盘": "Open",
                "high": "High",
                "最高": "High",
                "low": "Low",
                "最低": "Low",
                "close": "Close",
                "收盘": "Close",
                "volume": "Volume",
                "成交量": "Volume",
            }
            normalized = normalized.rename(
                columns={
                    key: value
                    for key, value in rename_map.items()
                    if key in normalized.columns
                }
            )
            if "Date" not in normalized.columns:
                return None
            normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
            normalized = normalized.dropna(subset=["Date"])
            normalized = normalized[
                (normalized["Date"] >= pd.Timestamp(self.start_date.date()))
                & (normalized["Date"] <= pd.Timestamp(self.end_date.date()))
            ]
            if normalized.empty:
                return None
            return normalized.reset_index(drop=True)
        except Exception:
            return None
        
    def load_universe(self, universe_file: str = 'data/us_universe/complete_us_universe.json') -> Dict:
        """加载股票池"""
        if os.path.exists(universe_file):
            with open(universe_file, 'r') as f:
                universe = json.load(f)
        else:
            universe = {}

        if not universe:
            universe = self._build_local_universe()
        else:
            universe = self._canonicalize_universe(universe)
        universe = self._filter_universe_by_market_cap(universe)

        print("=" * 80)
        print("📊 加载股票池")
        print("=" * 80)
        print(f"全美股: {len(universe['full_us'])} 只")
        market_cap_meta = dict(universe.get("metadata", {}).get("market_cap_filter", {}) or {})
        if market_cap_meta.get("enabled"):
            print(f"市值门槛: >= ${market_cap_meta['threshold_usd']:,}")
            print(
                "市值过滤: "
                f"{market_cap_meta['included_count']}/{market_cap_meta['input_count']} 只保留，"
                f"剔除 {market_cap_meta['excluded_count']} 只"
            )
        print(f"大盘股: {len(universe.get('large_cap', []))} 只")
        print(f"中盘股: {len(universe.get('mid_cap', []))} 只")
        print(f"小盘股: {len(universe.get('small_cap', []))} 只")
        print(f"总计: {universe['stats']['total_unique']} 只")
        print("=" * 80)
        
        return universe

    def _filter_universe_by_market_cap(self, universe: Dict[str, Any]) -> Dict[str, Any]:
        """只保留市值达到门槛的美股，未知市值不进入下载/分析池。"""
        raw_full = list(
            dict.fromkeys(
                universe.get("full_us", [])
                or universe.get("all_us", [])
                or universe.get("all", [])
                or (
                    universe.get("large_cap", [])
                    + universe.get("mid_cap", [])
                    + universe.get("small_cap", [])
                )
            )
        )
        filtered_full, metadata = self.market_cap_filter.filter_symbols(raw_full, fetch_missing=True)
        allowed = set(filtered_full)
        for key in ("large_cap", "mid_cap", "small_cap"):
            universe[key] = [symbol for symbol in list(universe.get(key, []) or []) if symbol in allowed]
        universe["full_us"] = filtered_full
        universe["full_market"] = filtered_full
        universe["all_us"] = filtered_full
        universe["all"] = filtered_full
        stats = dict(universe.get("stats", {}) or {})
        stats["full_us"] = len(filtered_full)
        stats["large_cap"] = len(universe.get("large_cap", []))
        stats["mid_cap"] = len(universe.get("mid_cap", []))
        stats["small_cap"] = len(universe.get("small_cap", []))
        stats["total_unique"] = len(filtered_full)
        universe["stats"] = stats
        metadata_parent = dict(universe.get("metadata", {}) or {})
        metadata_parent["market_cap_filter"] = metadata
        universe["metadata"] = metadata_parent
        return universe

    def _build_local_universe(self) -> Dict[str, List[str]]:
        """从本地 Parquet serving inventory 构建 full_us universe。"""
        try:
            full_us = self.market_reader.list_symbols("full_us")
        except Exception:
            full_us = []
        return {
            "full_us": full_us,
            "full_market": full_us,
            "all_us": full_us,
            "all": full_us,
            "large_cap": full_us,
            "mid_cap": full_us,
            "small_cap": full_us,
            "stats": {
                "full_us": len(full_us),
                "large_cap": len(full_us),
                "mid_cap": len(full_us),
                "small_cap": len(full_us),
                "total_unique": len(full_us),
            },
        }

    def _canonicalize_universe(self, universe: Dict[str, Any]) -> Dict[str, Any]:
        """补齐 full_us 及其别名，兼容旧 universe 文件。"""
        full_us = list(
            dict.fromkeys(
                universe.get("full_us", [])
                or universe.get("all_us", [])
                or universe.get("all", [])
                or (universe.get("large_cap", []) + universe.get("mid_cap", []) + universe.get("small_cap", []))
            )
        )
        large_cap = list(dict.fromkeys(universe.get("large_cap", []) or full_us))
        mid_cap = list(dict.fromkeys(universe.get("mid_cap", []) or full_us))
        small_cap = list(dict.fromkeys(universe.get("small_cap", []) or full_us))
        stats = dict(universe.get("stats", {}))
        stats.setdefault("full_us", len(full_us))
        stats.setdefault("large_cap", len(large_cap))
        stats.setdefault("mid_cap", len(mid_cap))
        stats.setdefault("small_cap", len(small_cap))
        stats.setdefault("total_unique", len(full_us))
        universe.update(
            {
                "full_us": full_us,
                "full_market": full_us,
                "all_us": full_us,
                "all": full_us,
                "large_cap": large_cap,
                "mid_cap": mid_cap,
                "small_cap": small_cap,
                "stats": stats,
            }
        )
        return universe
    
    def download_stock(self, symbol: str, category: str, force_refresh: bool = False) -> Dict:
        """
        下载单只股票数据
        
        Returns:
            Dict with download result
        """
        expected_latest = self._expected_latest_date()
        cached_latest = None
        
        # 检查是否已存在且数据足够新鲜
        if not force_refresh:
            try:
                existing_df = self.market_reader.read_symbol_frame(
                    symbol,
                    universe_key="full_us",
                ).frame
                cached_latest = self._latest_cached_date(symbol)
                if len(existing_df) > 200 and cached_latest and cached_latest >= expected_latest:
                    return {
                        'symbol': symbol,
                        'category': category,
                        'status': 'cached',
                        'records': len(existing_df),
                        'latest_date': cached_latest.isoformat(),
                        'error': None
                    }
            except:
                pass  # 重新下载
        
        downloaders = {
            "yfinance": self._download_from_yfinance,
            "akshare": self._download_from_akshare,
            "tushare": self._download_from_tushare,
        }
        df = None
        source = None
        attempted_sources: list[str] = []
        for provider in self._provider_order():
            attempted_sources.append(provider)
            provider_df = downloaders[provider](symbol)
            if provider_df is not None and not provider_df.empty:
                df = provider_df
                source = provider
                break

        if df is None or df.empty:
            return {
                'symbol': symbol,
                'category': category,
                'status': 'no_data',
                'records': 0,
                'error': f"No data from {'/'.join(attempted_sources)}",
                'source': None,
                'attempted_sources': attempted_sources,
            }

        try:
            df = self._normalize_downloaded_frame(df)
            downloaded_latest = None
            if "Date" in df.columns and not df.empty:
                downloaded_latest = pd.to_datetime(df["Date"], errors="coerce").dropna().max().date()
            if downloaded_latest is None:
                return {
                    'symbol': symbol,
                    'category': category,
                    'status': 'no_data',
                    'records': 0,
                    'error': 'Downloaded data missing valid Date column',
                    'source': source,
                    'attempted_sources': attempted_sources,
                }
            if cached_latest and downloaded_latest < cached_latest:
                return {
                    'symbol': symbol,
                    'category': category,
                    'status': 'source_regressed',
                    'records': len(df),
                    'latest_date': downloaded_latest.isoformat(),
                    'cached_latest_date': cached_latest.isoformat(),
                    'error': 'Provider returned older data than local cache',
                    'source': source,
                    'attempted_sources': attempted_sources,
                }
            if downloaded_latest < expected_latest:
                return {
                    'symbol': symbol,
                    'category': category,
                    'status': 'source_stale',
                    'records': len(df),
                    'latest_date': downloaded_latest.isoformat(),
                    'expected_latest_date': expected_latest.isoformat(),
                    'error': 'Provider data is older than expected latest trading date',
                    'source': source,
                    'attempted_sources': attempted_sources,
                }
            parquet_frame = df.copy()
            parquet_frame["symbol"] = str(symbol or "").strip().upper()
            manifest = self.store.write_full_history_bars(
                parquet_frame,
                source=str(source or ""),
                metadata={
                    "category": category,
                    "attempted_sources": attempted_sources,
                },
            )
            return {
                'symbol': symbol,
                'category': category,
                'status': 'success',
                'records': len(df),
                'latest_date': downloaded_latest.isoformat(),
                'error': None,
                'source': source,
                'attempted_sources': attempted_sources,
                'parquet_manifest_path': manifest.get("manifest_path", ""),
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'category': category,
                'status': 'failed',
                'records': 0,
                'error': str(e)[:100],
                'source': source,
                'attempted_sources': attempted_sources,
            }
    
    def download_category(
        self,
        symbols: List[str],
        category: str,
        force_refresh_symbols: Optional[set[str]] = None,
    ) -> List[Dict]:
        """
        批量下载某一类别的股票数据
        
        Args:
            symbols: 股票代码列表
            category: 类别名称
            force_refresh_symbols: 需要跳过本地缓存强制刷新检查的股票
        """
        print(f"\n{'='*80}")
        print(f"📥 下载 {category.upper()} ({len(symbols)} 只股票)")
        print(f"{'='*80}")
        print(f"时间范围: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"保存目录: {self.dirs[category]}")
        print(f"并行线程: {self.max_workers}")
        print(f"预计时间: {len(symbols) * 2 / self.max_workers:.0f} 秒 (~{len(symbols) * 2 / self.max_workers / 60:.1f} 分钟)")
        print(f"{'='*80}\n")
        
        results = []
        start_time = time.time()
        
        # 分批处理
        total_batches = (len(symbols) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(symbols))
            batch_symbols = symbols[batch_start:batch_end]
            
            print(f"  批次 {batch_idx + 1}/{total_batches} ({batch_start+1}-{batch_end}/{len(symbols)})...")
            
            # 并行下载
            force_refresh_set = {str(symbol).upper() for symbol in (force_refresh_symbols or set())}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(
                        self.download_stock,
                        symbol,
                        category,
                        symbol.upper() in force_refresh_set,
                    ): symbol
                    for symbol in batch_symbols
                }
                
                for future in as_completed(future_to_symbol):
                    result = future.result()
                    results.append(result)
                    
                    # 更新统计
                    self.stats['total'] += 1
                    if result['status'] == 'success':
                        self.stats['success'] += 1
                    elif result['status'] == 'cached':
                        self.stats['cached'] += 1
                    elif result['status'] == 'no_data':
                        self.stats['no_data'] += 1
                    else:
                        self.stats['failed'] += 1
            
            # 显示进度
            elapsed = time.time() - start_time
            progress = len(results) / len(symbols)
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0
            
            print(f"    进度: {len(results)}/{len(symbols)} ({progress*100:.1f}%) | "
                  f"成功: {self.stats['success']} | "
                  f"缓存: {self.stats['cached']} | "
                  f"失败: {self.stats['failed']} | "
                  f"ETA: {eta/60:.1f}分钟")
        
        elapsed = time.time() - start_time
        print(f"\n✅ {category.upper()} 下载完成! 耗时: {elapsed/60:.1f} 分钟")
        
        return results
    
    def download_all(self, universe: Optional[Dict] = None) -> Dict:
        """
        下载全市场数据
        
        Returns:
            Dict with download statistics
        """
        if universe is None:
            universe = self.load_universe()
        else:
            universe = self._canonicalize_universe(dict(universe))
            existing_filter = dict(universe.get("metadata", {}).get("market_cap_filter", {}) or {})
            if not (
                existing_filter.get("enabled")
                and int(existing_filter.get("included_count", -1)) == len(universe.get("full_us", []))
            ):
                universe = self._filter_universe_by_market_cap(universe)
        
        print("\n" + "=" * 80)
        print("🚀 开始下载完整美股市场数据")
        print("=" * 80)
        print(f"总计股票数: {universe['stats']['total_unique']} 只")
        print(f"预计总时间: {universe['stats']['total_unique'] * 2 / self.max_workers / 60:.1f} 分钟")
        print("=" * 80)
        
        all_results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': {
                'years': self.years,
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
                'price_provider': self.price_provider,
                'price_provider_order': self._provider_order(),
                'min_market_cap_usd': self.market_cap_filter.threshold_usd,
                'market_cap_cache_file': str(self.market_cap_filter.cache_file),
            },
            'market_cap_filter': dict(universe.get("metadata", {}).get("market_cap_filter", {}) or {}),
            'categories': {}
        }
        
        total_start = time.time()
        
        full_us_symbols = list(universe.get("full_us", []) or universe.get("all", []))
        all_results['categories']['full_us'] = self.download_category(
            full_us_symbols,
            'full_us'
        )

        # 兼容旧分层下载模式：如果 universe 里仍然有 legacy buckets，则保留输出
        for legacy_category in ['large_cap', 'mid_cap', 'small_cap']:
            legacy_symbols = list(universe.get(legacy_category, []) or [])
            if legacy_symbols and legacy_symbols != full_us_symbols:
                all_results['categories'][legacy_category] = self.download_category(
                    legacy_symbols,
                    legacy_category
                )
        
        total_elapsed = time.time() - total_start
        
        # 保存报告
        self._save_report(all_results)
        
        # 打印汇总
        self._print_final_summary(all_results, total_elapsed)
        
        return all_results
    
    def _save_report(self, results: Dict):
        """保存下载报告"""
        report_file = f"{self.data_dir}/download_report_{results['timestamp']}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📊 详细报告已保存: {report_file}")
    
    def _print_final_summary(self, results: Dict, elapsed: float):
        """打印最终汇总"""
        print("\n" + "=" * 80)
        print("📊 下载完成汇总")
        print("=" * 80)
        print(f"总耗时: {elapsed/60:.1f} 分钟 ({elapsed/3600:.2f} 小时)")
        print()
        print(f"总计处理: {self.stats['total']} 只股票")
        print(f"  ✅ 成功:     {self.stats['success']} 只 ({self.stats['success']/self.stats['total']*100:.1f}%)")
        print(f"  💾 缓存:     {self.stats['cached']} 只 ({self.stats['cached']/self.stats['total']*100:.1f}%)")
        print(f"  ⚠️  无数据:   {self.stats['no_data']} 只 ({self.stats['no_data']/self.stats['total']*100:.1f}%)")
        print(f"  ❌ 失败:     {self.stats['failed']} 只 ({self.stats['failed']/self.stats['total']*100:.1f}%)")
        print()
        print("=" * 80)
        print(f"数据保存位置: {self.data_dir}/")
        print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='下载完整美股市场数据')
    parser.add_argument('--years', type=int, default=3, help='下载年数 (默认3)')
    parser.add_argument('--workers', type=int, default=8, help='并行线程数 (默认8)')
    parser.add_argument('--batch', type=int, default=100, help='每批处理数量 (默认100)')
    parser.add_argument(
        '--category',
        type=str,
        choices=['full', 'full_us', 'large', 'mid', 'small', 'all'],
        default='all',
        help='下载类别 (默认all)',
    )
    
    args = parser.parse_args()
    
    downloader = FullMarketDownloader(
        years=args.years,
        max_workers=args.workers,
        batch_size=args.batch
    )
    
    universe = downloader.load_universe()
    
    if args.category == 'all':
        downloader.download_all(universe)
    else:
        category_map = {
            'full': 'full_us',
            'full_us': 'full_us',
            'large': 'large_cap',
            'mid': 'mid_cap',
            'small': 'small_cap'
        }
        cat = category_map[args.category]
        downloader.download_category(universe[cat], cat)


if __name__ == '__main__':
    main()

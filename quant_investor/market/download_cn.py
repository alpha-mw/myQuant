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
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional, Set, Mapping
import json
import time

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.fetch_cn_index_components import get_all_components, save_components
from quant_investor.market.cn_resolver import CNUniverseResolver
from quant_investor.market.config import get_market_settings
from quant_investor.market.download_cn_freshness import CNDownloadFreshnessMixin
from quant_investor.market.shared_csv_reader import SharedCSVReader
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


class CNFullMarketDownloader(CNDownloadFreshnessMixin):
    """A股全市场数据下载器"""

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
        self.csv_reader = SharedCSVReader(market="CN", data_dir=self.data_dir, resolver=self.resolver)
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
        self._active_listing_dates_cache: dict[str, str] | None = None
        
        # 统计信息
        self.stats = {
            'total': 0,
            'updated': 0,
            'cached': 0,
            'stale_cached': 0,
            'failed': 0,
        }

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
        existing_df = pd.DataFrame()
        if local_state.resolved_path:
            try:
                existing_df = pd.read_csv(local_state.resolved_path)
            except Exception:
                existing_df = pd.DataFrame()
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

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            cleaning_result: Mapping[str, Any] | None = None
            if config.TUSHARE_AUTO_CLEAN:
                cleaning_result = clean_tushare_dataframe_to_file(
                    final_df,
                    canonical_path=filepath,
                    table_name="daily",
                    promote=True,
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
                final_df.to_csv(filepath, index=False)
                cleaning_result = cleaning_skipped_fields
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

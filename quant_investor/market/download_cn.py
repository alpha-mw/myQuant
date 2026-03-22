#!/usr/bin/env python3
"""
Download Full China A-Share Market Data - 下载完整A股市场数据

下载所有大中小盘股的3年历史数据
- 大盘股: 沪深300 (300只)
- 中盘股: 中证500 (500只)
- 小盘股: 中证1000 (1000只)
总计: 1800只股票
"""

import os
from collections import Counter
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional, Set
import json
import time

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro

try:
    import tushare as ts
except ImportError:
    ts = None

# Tushare配置
TUSHARE_TOKEN = config.TUSHARE_TOKEN
TUSHARE_URL = os.environ.get('TUSHARE_URL', 'http://lianghua.nanyangqiankun.top')


class CNFullMarketDownloader:
    """A股全市场数据下载器"""

    SUPPORTED_CATEGORIES = ("hs300", "zz500", "zz1000")
    REQUESTS_PER_STOCK = 2
    REQUESTS_PER_MINUTE_BUDGET = 160
    
    def __init__(self, 
                 data_dir: str = 'data/cn_market_full',
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
        self.data_dir = data_dir
        self.years = years
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # 创建分层目录
        self.dirs = {
            'hs300': f"{data_dir}/hs300",
            'zz500': f"{data_dir}/zz500",
            'zz1000': f"{data_dir}/zz1000"
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 计算日期范围
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=years*365 + 30)
        
        # 初始化 Tushare（内存模式，不落盘 token）
        if ts is None:
            raise RuntimeError("tushare 未安装，无法下载 A 股全市场数据")
        self.pro = create_tushare_pro(ts, TUSHARE_TOKEN, TUSHARE_URL)
        if self.pro is None:
            raise RuntimeError("TUSHARE_TOKEN 未设置，无法下载 A 股全市场数据")
        self.latest_trade_date = self._resolve_latest_trade_date()
        self._latest_suspended_symbols_cache: Optional[Set[str]] = None
        
        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'updated': 0,
            'failed': 0,
            'cached': 0,
            'no_data': 0
        }

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
            return list(cls.SUPPORTED_CATEGORIES)

        normalized: List[str] = []
        for category in categories:
            if category not in cls.SUPPORTED_CATEGORIES:
                raise ValueError(f"不支持的分类: {category}")
            if category not in normalized:
                normalized.append(category)
        return normalized

    def _resolve_latest_trade_date(self) -> str:
        """
        确定当前可用的最新交易日。

        Tushare 日线通常在收盘后才完整，因此交易日白天默认回退到上一个开市日。
        """
        fallback_date = self.end_date
        if fallback_date.hour < 18:
            fallback_date -= timedelta(days=1)

        fallback = fallback_date.strftime('%Y%m%d')

        try:
            start = (self.end_date - timedelta(days=30)).strftime('%Y%m%d')
            end = self.end_date.strftime('%Y%m%d')
            cal = self.pro.trade_cal(exchange='SSE', start_date=start, end_date=end, is_open='1')
            if cal is None or cal.empty or 'cal_date' not in cal.columns:
                return fallback

            open_days = sorted(str(value) for value in cal['cal_date'].dropna().astype(str).tolist())
            if not open_days:
                return fallback

            today = self.end_date.strftime('%Y%m%d')
            if self.end_date.hour < 18 and open_days[-1] == today and len(open_days) >= 2:
                return open_days[-2]

            return open_days[-1]
        except Exception:
            return fallback

    def _load_existing_data(self, filepath: str) -> tuple[pd.DataFrame, Optional[str]]:
        """读取本地已有数据并返回最新交易日。"""
        if not os.path.exists(filepath):
            return pd.DataFrame(), None

        try:
            existing_df = pd.read_csv(filepath)
        except Exception:
            return pd.DataFrame(), None

        if existing_df.empty or 'trade_date' not in existing_df.columns:
            return pd.DataFrame(), None

        existing_df['trade_date'] = pd.to_datetime(existing_df['trade_date']).dt.strftime('%Y-%m-%d')
        existing_df = (
            existing_df.sort_values('trade_date')
            .drop_duplicates(subset=['trade_date'], keep='last')
            .reset_index(drop=True)
        )
        latest_trade_date = existing_df['trade_date'].iloc[-1].replace('-', '')
        return existing_df, latest_trade_date

    def build_completeness_report(
        self,
        components: Optional[Dict] = None,
        allowed_stale_symbols: Optional[List[str] | Set[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        构建本地数据完整性报告。

        完整性的定义是：所有目标股票都存在本地文件，且最新交易日达到
        `self.latest_trade_date`。如明确声明了允许跳过的 symbol，则从阻塞项中排除。
        """
        if components is None:
            components = self.load_components()

        target_categories = self._normalize_categories(categories)
        allowed = self._normalize_allowed_symbols(allowed_stale_symbols)
        report = {
            'latest_trade_date': self.latest_trade_date,
            'allowed_stale_symbols': sorted(allowed),
            'complete': True,
            'blocking_incomplete_count': 0,
            'categories_checked': list(target_categories),
            'categories': {},
        }
        suspended_symbols = self._load_latest_suspended_symbols()

        for category in target_categories:
            date_counts: Counter[str] = Counter()
            missing_symbols: List[str] = []
            stale_symbols: List[Dict[str, str]] = []
            suspended_stale_symbols: List[Dict[str, str]] = []

            for symbol in components.get(category, []):
                filepath = Path(self.dirs[category]) / f'{symbol}.csv'
                if not filepath.exists():
                    missing_symbols.append(symbol)
                    continue

                _existing_df, latest_local_date = self._load_existing_data(str(filepath))
                if not latest_local_date:
                    missing_symbols.append(symbol)
                    continue

                date_counts[latest_local_date] += 1
                if latest_local_date != self.latest_trade_date:
                    item = {
                        'symbol': symbol,
                        'latest_local_date': latest_local_date,
                    }
                    stale_symbols.append(item)
                    if symbol.upper() in suspended_symbols:
                        suspended_stale_symbols.append(item)

            blocking_stale = [
                item
                for item in stale_symbols
                if item['symbol'].upper() not in allowed
                and item['symbol'].upper() not in suspended_symbols
            ]
            blocking_missing = [
                symbol for symbol in missing_symbols if symbol.upper() not in allowed
            ]
            blocking_count = len(blocking_stale) + len(blocking_missing)

            report['categories'][category] = {
                'expected': len(components.get(category, [])),
                'latest_trade_date': self.latest_trade_date,
                'date_counts': dict(sorted(date_counts.items())),
                'missing_symbols': sorted(missing_symbols),
                'stale_symbols': stale_symbols,
                'suspended_stale_symbols': suspended_stale_symbols,
                'blocking_missing_symbols': sorted(blocking_missing),
                'blocking_stale_symbols': blocking_stale,
                'blocking_incomplete_count': blocking_count,
            }

            if blocking_count > 0:
                report['complete'] = False
                report['blocking_incomplete_count'] += blocking_count

        return report

    def _load_latest_suspended_symbols(self) -> Set[str]:
        """
        获取最新交易日停牌标的集合。

        对于当日停牌的股票，本地最后一个成交日早于最新交易日是合理状态，
        不应阻塞“完整性”判断。
        """
        if self._latest_suspended_symbols_cache is not None:
            return self._latest_suspended_symbols_cache

        try:
            suspend_df = self.pro.suspend_d(trade_date=self.latest_trade_date)
            if suspend_df is None or suspend_df.empty:
                self._latest_suspended_symbols_cache = set()
                return self._latest_suspended_symbols_cache

            filtered = suspend_df.copy()
            if 'suspend_type' in filtered.columns:
                filtered = filtered[filtered['suspend_type'].astype(str).str.upper() == 'S']
            symbols = {
                str(symbol).upper()
                for symbol in filtered.get('ts_code', pd.Series(dtype=str)).dropna().astype(str)
            }
            self._latest_suspended_symbols_cache = symbols
            return symbols
        except Exception:
            self._latest_suspended_symbols_cache = set()
            return self._latest_suspended_symbols_cache

    def _fetch_stock_frame(self, symbol: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
        """通过 Tushare 抓取指定时间窗口内的单只股票行情。"""
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
    
    def load_components(self, components_file: str = 'data/cn_universe/cn_index_components.json') -> Dict:
        """加载成分股"""
        with open(components_file, 'r', encoding='utf-8') as f:
            components = json.load(f)
        
        print("=" * 80)
        print("📊 加载A股成分股")
        print("=" * 80)
        print(f"沪深300:  {len(components['hs300'])} 只")
        print(f"中证500:  {len(components['zz500'])} 只")
        print(f"中证1000: {len(components['zz1000'])} 只")
        print(f"总计:     {components['stats']['total_unique']} 只")
        print("=" * 80)
        
        return components
    
    def download_stock(self, symbol: str, category: str) -> Dict:
        """
        下载单只股票数据
        
        Returns:
            Dict with download result
        """
        save_dir = self.dirs[category]
        filepath = f"{save_dir}/{symbol}.csv"
        existing_df, latest_local_date = self._load_existing_data(filepath)

        # 检查是否已更新到最新可得交易日
        if latest_local_date == self.latest_trade_date and len(existing_df) > 0:
            return {
                'symbol': symbol,
                'category': category,
                'status': 'cached',
                'records': len(existing_df),
                'mode': 'up_to_date',
                'latest_local_date': latest_local_date,
                'latest_trade_date': self.latest_trade_date,
                'api_calls': 0,
                'error': None
            }
        
        try:
            end_date_str = self.latest_trade_date
            is_incremental = bool(latest_local_date and len(existing_df) >= 200)
            start_date_str = self.start_date.strftime('%Y%m%d')

            # 增量更新时保留 1 个交易日重叠，便于覆盖最后一根 K 线的修正。
            if is_incremental and latest_local_date:
                overlap_start = pd.to_datetime(latest_local_date) - timedelta(days=1)
                start_date_str = max(overlap_start.strftime('%Y%m%d'), start_date_str)

            df = self._fetch_stock_frame(symbol, start_date_str, end_date_str)

            if df.empty:
                if not existing_df.empty:
                    return {
                        'symbol': symbol,
                        'category': category,
                        'status': 'cached',
                        'records': len(existing_df),
                        'mode': 'suspended_or_no_increment',
                        'latest_local_date': latest_local_date,
                        'latest_trade_date': self.latest_trade_date,
                        'api_calls': self.REQUESTS_PER_STOCK,
                        'error': None
                    }
                return {
                    'symbol': symbol,
                    'category': category,
                    'status': 'no_data',
                    'records': 0,
                    'api_calls': self.REQUESTS_PER_STOCK,
                    'error': 'Empty data'
                }

            final_df = df
            status = 'success'
            if not existing_df.empty:
                final_df = (
                    pd.concat([existing_df, df], ignore_index=True)
                    .sort_values('trade_date')
                    .drop_duplicates(subset=['trade_date'], keep='last')
                    .reset_index(drop=True)
                )
                status = 'updated'

            final_df.to_csv(filepath, index=False)
            latest_saved_date = final_df['trade_date'].iloc[-1].replace('-', '')

            return {
                'symbol': symbol,
                'category': category,
                'status': status,
                'records': len(final_df),
                'mode': 'incremental' if status == 'updated' else 'full',
                'latest_local_date': latest_saved_date,
                'latest_trade_date': self.latest_trade_date,
                'api_calls': self.REQUESTS_PER_STOCK,
                'error': None
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'category': category,
                'status': 'failed',
                'records': 0,
                'api_calls': self.REQUESTS_PER_STOCK,
                'error': str(e)[:100]
            }
    
    def download_category(self, symbols: List[str], category: str) -> List[Dict]:
        """
        批量下载某一类别的股票数据
        
        Args:
            symbols: 股票代码列表
            category: 类别名称
        """
        print(f"\n{'='*80}")
        print(f"📥 下载 {category.upper()} ({len(symbols)} 只股票)")
        print(f"{'='*80}")
        print(f"时间范围: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"保存目录: {self.dirs[category]}")
        print(f"Tushare API限速: 每分钟200次调用")
        estimated_minutes = (
            len(symbols) * self.REQUESTS_PER_STOCK / self.REQUESTS_PER_MINUTE_BUDGET
        )
        print(f"预计时间: {estimated_minutes:.1f} 分钟")
        print(f"{'='*80}\n")
        
        results = []
        start_time = time.time()
        
        # 串行下载（遵守Tushare限速）
        for i, symbol in enumerate(symbols, 1):
            result = self.download_stock(symbol, category)
            results.append(result)
            
            # 更新统计
            self.stats['total'] += 1
            if result['status'] == 'success':
                self.stats['success'] += 1
            elif result['status'] == 'updated':
                self.stats['updated'] += 1
            elif result['status'] == 'cached':
                self.stats['cached'] += 1
            elif result['status'] == 'no_data':
                self.stats['no_data'] += 1
            else:
                self.stats['failed'] += 1
            
            # 显示进度
            if i % 10 == 0 or i == len(symbols):
                elapsed = time.time() - start_time
                progress = i / len(symbols)
                eta = elapsed / progress * (1 - progress) if progress > 0 else 0
                
                print(f"  进度: {i}/{len(symbols)} ({progress*100:.1f}%) | "
                      f"成功: {self.stats['success']} | "
                      f"更新: {self.stats['updated']} | "
                      f"缓存: {self.stats['cached']} | "
                      f"失败: {self.stats['failed']} | "
                      f"ETA: {eta/60:.1f}分钟")
            
            # 仅在真正发起 Tushare 请求时限速；缓存命中不应拖慢补齐轮次。
            api_calls = int(result.get('api_calls', self.REQUESTS_PER_STOCK) or 0)
            if api_calls > 0:
                time.sleep(api_calls * 60 / self.REQUESTS_PER_MINUTE_BUDGET)
        
        elapsed = time.time() - start_time
        print(f"\n✅ {category.upper()} 下载完成! 耗时: {elapsed/60:.1f} 分钟")
        
        return results
    
    def _print_completeness_summary(self, completeness: Dict[str, Any]):
        """打印本地数据完整性摘要。"""
        print("\n" + "=" * 80)
        print("🧭 本地数据完整性检查")
        print("=" * 80)
        print(f"目标最新交易日: {completeness['latest_trade_date']}")
        print(f"完整性状态: {'通过' if completeness['complete'] else '未通过'}")
        print(f"阻塞缺口总数: {completeness['blocking_incomplete_count']}")
        for category, payload in completeness['categories'].items():
            date_counts = payload.get('date_counts', {})
            latest_count = int(date_counts.get(completeness['latest_trade_date'], 0))
            print(
                f"  - {category}: 最新 {latest_count}/{payload['expected']} | "
                f"阻塞缺口 {payload['blocking_incomplete_count']}"
            )
        print("=" * 80)

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
        target_categories = self._normalize_categories(categories)
        target_total = sum(len(components.get(category, [])) for category in target_categories)
        
        print("\n" + "=" * 80)
        print("🚀 开始下载完整A股市场数据")
        print("=" * 80)
        print(f"目标分类: {', '.join(target_categories)}")
        print(f"总计股票数: {target_total} 只")
        estimated_minutes = (
            target_total
            * self.REQUESTS_PER_STOCK
            / self.REQUESTS_PER_MINUTE_BUDGET
        )
        print(f"预计总时间: {estimated_minutes:.1f} 分钟")
        print("=" * 80)
        
        all_results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': {
                'years': self.years,
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'latest_trade_date': self.latest_trade_date,
                'max_rounds': max_rounds,
                'fail_on_incomplete': fail_on_incomplete,
                'categories': list(target_categories),
                'allowed_stale_symbols': sorted(
                    self._normalize_allowed_symbols(allowed_stale_symbols)
                ),
            },
            'categories': {category: [] for category in target_categories},
            'rounds': [],
        }
        
        total_start = time.time()

        round_symbols = {
            category: list(components.get(category, []))
            for category in target_categories
        }

        for round_no in range(1, max_rounds + 1):
            print("\n" + "=" * 80)
            print(f"🔁 下载轮次 {round_no}/{max_rounds}")
            print("=" * 80)

            round_payload = {
                'round': round_no,
                'categories': {},
            }

            for category in target_categories:
                symbols = round_symbols.get(category, [])
                if not symbols:
                    continue
                results = self.download_category(symbols, category)
                round_payload['categories'][category] = results
                all_results['categories'][category].extend(results)

            completeness = self.build_completeness_report(
                components=components,
                allowed_stale_symbols=allowed_stale_symbols,
                categories=target_categories,
            )
            round_payload['completeness'] = completeness
            all_results['rounds'].append(round_payload)
            all_results['completeness'] = completeness
            self._print_completeness_summary(completeness)

            if completeness['complete']:
                break

            if round_no >= max_rounds:
                break

            round_symbols = {}
            for category, payload in completeness['categories'].items():
                stale_symbols = [
                    item['symbol'] for item in payload.get('blocking_stale_symbols', [])
                ]
                missing_symbols = list(payload.get('blocking_missing_symbols', []))
                round_symbols[category] = stale_symbols + missing_symbols
        
        total_elapsed = time.time() - total_start
        
        # 保存报告
        self._save_report(all_results)
        
        # 打印汇总
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
        print(f"总计处理: {self.stats['total']} 只股票")
        print(f"  ✅ 成功:     {self.stats['success']} 只 ({self.stats['success']/self.stats['total']*100:.1f}%)")
        print(f"  🔄 更新:     {self.stats['updated']} 只 ({self.stats['updated']/self.stats['total']*100:.1f}%)")
        print(f"  💾 缓存:     {self.stats['cached']} 只 ({self.stats['cached']/self.stats['total']*100:.1f}%)")
        print(f"  ⚠️  无数据:   {self.stats['no_data']} 只 ({self.stats['no_data']/self.stats['total']*100:.1f}%)")
        print(f"  ❌ 失败:     {self.stats['failed']} 只 ({self.stats['failed']/self.stats['total']*100:.1f}%)")
        print()
        completeness = results.get('completeness')
        if completeness:
            print(f"完整性检查: {'通过' if completeness['complete'] else '未通过'}")
            print(f"目标最新交易日: {completeness['latest_trade_date']}")
            print(f"阻塞缺口总数: {completeness['blocking_incomplete_count']}")
            print()
        print("=" * 80)
        print(f"数据保存位置: {self.data_dir}/")
        print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='下载完整A股市场数据')
    parser.add_argument('--years', type=int, default=3, help='下载年数 (默认3)')
    parser.add_argument('--category', type=str, choices=['hs300', 'zz500', 'zz1000', 'all'],
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

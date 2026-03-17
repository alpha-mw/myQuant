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
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

sys.path.insert(0, str(os.path.dirname(__file__)))
from config import config
from credential_utils import create_tushare_pro

import tushare as ts

# Tushare配置
TUSHARE_TOKEN = config.TUSHARE_TOKEN
TUSHARE_URL = os.environ.get('TUSHARE_URL', 'http://lianghua.nanyangqiankun.top')


class CNFullMarketDownloader:
    """A股全市场数据下载器"""

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
        self.pro = create_tushare_pro(ts, TUSHARE_TOKEN, TUSHARE_URL)
        if self.pro is None:
            raise RuntimeError("TUSHARE_TOKEN 未设置，无法下载 A 股全市场数据")
        self.latest_trade_date = self._resolve_latest_trade_date()
        
        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'updated': 0,
            'failed': 0,
            'cached': 0,
            'no_data': 0
        }

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
                        'error': None
                    }
                return {
                    'symbol': symbol,
                    'category': category,
                    'status': 'no_data',
                    'records': 0,
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
                'error': None
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'category': category,
                'status': 'failed',
                'records': 0,
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
            
            # 每只股票通常包含 daily + adj_factor 两次调用，按 160 次/分钟预算限速。
            time.sleep(self.REQUESTS_PER_STOCK * 60 / self.REQUESTS_PER_MINUTE_BUDGET)
        
        elapsed = time.time() - start_time
        print(f"\n✅ {category.upper()} 下载完成! 耗时: {elapsed/60:.1f} 分钟")
        
        return results
    
    def download_all(self, components: Optional[Dict] = None) -> Dict:
        """
        下载全市场数据
        
        Returns:
            Dict with download statistics
        """
        if components is None:
            components = self.load_components()
        
        print("\n" + "=" * 80)
        print("🚀 开始下载完整A股市场数据")
        print("=" * 80)
        print(f"总计股票数: {components['stats']['total_unique']} 只")
        estimated_minutes = (
            components['stats']['total_unique']
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
                'end_date': self.end_date.strftime('%Y-%m-%d')
            },
            'categories': {}
        }
        
        total_start = time.time()
        
        # 下载沪深300
        all_results['categories']['hs300'] = self.download_category(
            components['hs300'], 
            'hs300'
        )
        
        # 下载中证500
        all_results['categories']['zz500'] = self.download_category(
            components['zz500'],
            'zz500'
        )
        
        # 下载中证1000
        all_results['categories']['zz1000'] = self.download_category(
            components['zz1000'],
            'zz1000'
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
        print("=" * 80)
        print(f"数据保存位置: {self.data_dir}/")
        print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='下载完整A股市场数据')
    parser.add_argument('--years', type=int, default=3, help='下载年数 (默认3)')
    parser.add_argument('--category', type=str, choices=['hs300', 'zz500', 'zz1000', 'all'],
                       default='all', help='下载类别 (默认all)')
    
    args = parser.parse_args()
    
    downloader = CNFullMarketDownloader(years=args.years)
    
    if args.category == 'all':
        components = downloader.load_components()
        downloader.download_all(components)
    else:
        components = downloader.load_components()
        category_map = {
            'hs300': 'hs300',
            'zz500': 'zz500',
            'zz1000': 'zz1000'
        }
        cat = category_map[args.category]
        downloader.download_category(components[cat], cat)


if __name__ == '__main__':
    main()

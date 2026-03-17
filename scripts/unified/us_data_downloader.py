#!/usr/bin/env python3
"""
US Market Data Downloader - 美股全市场数据下载器

下载大盘股/中盘股/小盘股的3年历史数据，保存到本地
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 添加路径
sys.path.insert(0, str(os.path.dirname(__file__)))
from us_stock_universe import USStockUniverse


class USDataDownloader:
    """美股数据下载器"""
    
    def __init__(self, 
                 data_dir: str = 'data/us_market',
                 years: int = 3,
                 max_workers: int = 5):
        """
        初始化下载器
        
        Args:
            data_dir: 数据保存目录
            years: 下载年数
            max_workers: 并行下载线程数
        """
        self.data_dir = data_dir
        self.years = years
        self.max_workers = max_workers
        
        # 创建分层目录
        self.large_cap_dir = f"{data_dir}/large_cap"
        self.mid_cap_dir = f"{data_dir}/mid_cap"
        self.small_cap_dir = f"{data_dir}/small_cap"
        
        for dir_path in [self.large_cap_dir, self.mid_cap_dir, self.small_cap_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 计算日期范围
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=years*365 + 30)  # 多下30天保险
        
        # 下载记录
        self.download_log = []
        
    def download_stock(self, symbol: str, save_dir: str) -> Dict:
        """
        下载单只股票数据
        
        Returns:
            Dict with 'symbol', 'status', 'records', 'error'
        """
        filepath = f"{save_dir}/{symbol}.csv"
        
        # 检查是否已存在
        if os.path.exists(filepath):
            try:
                existing_df = pd.read_csv(filepath)
                if len(existing_df) > 200:  # 数据足够
                    return {
                        'symbol': symbol,
                        'status': 'cached',
                        'records': len(existing_df),
                        'error': None
                    }
            except:
                pass  # 重新下载
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date.strftime('%Y-%m-%d'),
                end=self.end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'status': 'no_data',
                    'records': 0,
                    'error': 'Empty data returned'
                }
            
            # 重置索引并格式化
            df.reset_index(inplace=True)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # 保存
            df.to_csv(filepath, index=False)
            
            return {
                'symbol': symbol,
                'status': 'success',
                'records': len(df),
                'error': None
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'status': 'error',
                'records': 0,
                'error': str(e)
            }
    
    def download_category(self, symbols: List[str], save_dir: str, category_name: str) -> Dict:
        """
        批量下载某一类别的股票数据
        
        Args:
            symbols: 股票代码列表
            save_dir: 保存目录
            category_name: 类别名称 (用于显示)
        """
        print(f"\n{'='*80}")
        print(f"📥 下载 {category_name} 数据 ({len(symbols)} 只股票)")
        print(f"{'='*80}")
        print(f"时间范围: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}")
        print(f"保存目录: {save_dir}")
        print(f"并行线程: {self.max_workers}")
        print(f"{'='*80}\n")
        
        results = {
            'category': category_name,
            'total': len(symbols),
            'success': 0,
            'cached': 0,
            'failed': 0,
            'no_data': 0,
            'details': []
        }
        
        # 并行下载
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.download_stock, symbol, save_dir): symbol 
                for symbol in symbols
            }
            
            completed = 0
            for future in as_completed(future_to_symbol):
                result = future.result()
                results['details'].append(result)
                
                status = result['status']
                if status == 'success':
                    results['success'] += 1
                elif status == 'cached':
                    results['cached'] += 1
                elif status == 'no_data':
                    results['no_data'] += 1
                else:
                    results['failed'] += 1
                
                completed += 1
                if completed % 10 == 0 or completed == len(symbols):
                    print(f"  进度: {completed}/{len(symbols)} ({completed/len(symbols)*100:.1f}%)", end='\r')
        
        print(f"\n{'='*80}")
        print(f"✅ {category_name} 下载完成!")
        print(f"  成功: {results['success']}")
        print(f"  缓存: {results['cached']}")
        print(f"  无数据: {results['no_data']}")
        print(f"  失败: {results['failed']}")
        print(f"{'='*80}")
        
        return results
    
    def download_all(self, 
                     large_cap_limit: Optional[int] = 100,
                     mid_cap_limit: Optional[int] = 100,
                     small_cap_limit: Optional[int] = 100) -> Dict:
        """
        下载全市场数据
        
        Returns:
            Dict with download statistics
        """
        print("=" * 80)
        print("🚀 美股全市场数据下载")
        print("=" * 80)
        
        # 获取股票池
        universe = USStockUniverse().get_all_market(
            large_cap_limit=large_cap_limit,
            mid_cap_limit=mid_cap_limit,
            small_cap_limit=small_cap_limit
        )
        
        # 下载各层数据
        results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': {
                'years': self.years,
                'large_cap_limit': large_cap_limit,
                'mid_cap_limit': mid_cap_limit,
                'small_cap_limit': small_cap_limit
            },
            'categories': {}
        }
        
        # 大盘股
        results['categories']['large_cap'] = self.download_category(
            universe['large_cap'], 
            self.large_cap_dir,
            "大盘股 (Large-Cap)"
        )
        
        # 中盘股
        results['categories']['mid_cap'] = self.download_category(
            universe['mid_cap'],
            self.mid_cap_dir,
            "中盘股 (Mid-Cap)"
        )
        
        # 小盘股
        results['categories']['small_cap'] = self.download_category(
            universe['small_cap'],
            self.small_cap_dir,
            "小盘股 (Small-Cap)"
        )
        
        # 保存下载报告
        self._save_report(results)
        
        # 打印汇总
        self._print_summary(results)
        
        return results
    
    def _save_report(self, results: Dict):
        """保存下载报告"""
        report_file = f"{self.data_dir}/download_report_{results['timestamp']}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📊 下载报告已保存: {report_file}")
    
    def _print_summary(self, results: Dict):
        """打印下载汇总"""
        print("\n" + "=" * 80)
        print("📊 下载汇总")
        print("=" * 80)
        
        total_success = 0
        total_cached = 0
        total_failed = 0
        
        for category, data in results['categories'].items():
            print(f"\n{category.upper()}:")
            print(f"  总数: {data['total']}")
            print(f"  成功: {data['success']}")
            print(f"  缓存: {data['cached']}")
            print(f"  失败: {data['failed']}")
            total_success += data['success']
            total_cached += data['cached']
            total_failed += data['failed']
        
        print(f"\n{'='*80}")
        print(f"总计:")
        print(f"  成功下载: {total_success}")
        print(f"  使用缓存: {total_cached}")
        print(f"  失败: {total_failed}")
        print(f"{'='*80}")
    
    def load_stock_data(self, symbol: str, category: str) -> Optional[pd.DataFrame]:
        """
        加载已下载的股票数据
        
        Args:
            symbol: 股票代码
            category: 类别 (large_cap/mid_cap/small_cap)
        """
        dir_map = {
            'large_cap': self.large_cap_dir,
            'mid_cap': self.mid_cap_dir,
            'small_cap': self.small_cap_dir
        }
        
        if category not in dir_map:
            return None
        
        filepath = f"{dir_map[category]}/{symbol}.csv"
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            print(f"[USDataDownloader] 加载 {symbol} 失败: {e}")
            return None
    
    def get_all_downloaded_symbols(self, category: str) -> List[str]:
        """获取已下载的所有股票代码"""
        dir_map = {
            'large_cap': self.large_cap_dir,
            'mid_cap': self.mid_cap_dir,
            'small_cap': self.small_cap_dir
        }
        
        if category not in dir_map:
            return []
        
        csv_files = [f for f in os.listdir(dir_map[category]) if f.endswith('.csv')]
        return [f.replace('.csv', '') for f in csv_files]


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='美股数据下载器')
    parser.add_argument('--years', type=int, default=3, help='下载年数')
    parser.add_argument('--large', type=int, default=100, help='大盘股数量限制')
    parser.add_argument('--mid', type=int, default=100, help='中盘股数量限制')
    parser.add_argument('--small', type=int, default=100, help='小盘股数量限制')
    parser.add_argument('--workers', type=int, default=5, help='并行线程数')
    
    args = parser.parse_args()
    
    downloader = USDataDownloader(
        years=args.years,
        max_workers=args.workers
    )
    
    downloader.download_all(
        large_cap_limit=args.large,
        mid_cap_limit=args.mid,
        small_cap_limit=args.small
    )

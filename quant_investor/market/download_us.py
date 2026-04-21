#!/usr/bin/env python3
"""
Download Full US Market Data - 下载完整美股市场数据

下载所有大中小盘股的3年历史数据
- 大盘股: 525只 (S&P 500 + NASDAQ-100)
- 中盘股: 572只 (S&P MidCap 400)
- 小盘股: 560只 (Russell 2000)
总计: 1393只股票
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False


class FullMarketDownloader:
    """全市场数据下载器"""
    
    def __init__(self, 
                 data_dir: str = 'data/us_market_full',
                 years: int = 3,
                 max_workers: int = 8,
                 batch_size: int = 100):
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
    def _format_tushare_us_frame(df: pd.DataFrame) -> pd.DataFrame:
        """标准化 Tushare us_daily 输出为本地 CSV 格式。"""
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
            if "每天最多访问该接口" in message:
                self._tushare_quota_exhausted = True
            return None

    def _download_from_yfinance(self, symbol: str) -> Optional[pd.DataFrame]:
        """回退到 yfinance 拉取美股数据。"""
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
        
    def load_universe(self, universe_file: str = 'data/us_universe/complete_us_universe.json') -> Dict:
        """加载股票池"""
        if os.path.exists(universe_file):
            with open(universe_file, 'r') as f:
                universe = json.load(f)
        else:
            universe = {}

        if not universe:
            universe = self._build_local_universe()
        elif "full_us" not in universe or not universe.get("full_us"):
            universe = self._canonicalize_universe(universe)

        print("=" * 80)
        print("📊 加载股票池")
        print("=" * 80)
        print(f"全美股: {len(universe['full_us'])} 只")
        print(f"大盘股: {len(universe.get('large_cap', []))} 只")
        print(f"中盘股: {len(universe.get('mid_cap', []))} 只")
        print(f"小盘股: {len(universe.get('small_cap', []))} 只")
        print(f"总计: {universe['stats']['total_unique']} 只")
        print("=" * 80)
        
        return universe

    def _build_local_universe(self) -> Dict[str, List[str]]:
        """从本地文件目录构建 full_us universe。"""
        symbols = {
            path.stem.strip()
            for path in Path(self.data_dir).rglob("*.csv")
            if path.suffix.lower() == ".csv" and "_snapshots" not in path.parts and path.stem.strip()
        }
        full_us = sorted(symbols)
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
    
    def download_stock(self, symbol: str, category: str) -> Dict:
        """
        下载单只股票数据
        
        Returns:
            Dict with download result
        """
        save_dir = self.dirs.get(category, self.dirs["full_us"])
        filepath = f"{save_dir}/{symbol}.csv"
        
        # 检查是否已存在且数据完整
        if os.path.exists(filepath):
            try:
                existing_df = pd.read_csv(filepath)
                if len(existing_df) > 200:  # 至少200个交易日
                    return {
                        'symbol': symbol,
                        'category': category,
                        'status': 'cached',
                        'records': len(existing_df),
                        'error': None
                    }
            except:
                pass  # 重新下载
        
        df = self._download_from_tushare(symbol)
        source = "tushare"
        if df is None or df.empty:
            df = self._download_from_yfinance(symbol)
            source = "yfinance"

        if df is None or df.empty:
            return {
                'symbol': symbol,
                'category': category,
                'status': 'no_data',
                'records': 0,
                'error': 'No data from tushare/yfinance',
                'source': None,
            }

        try:
            df.to_csv(filepath, index=False)
            return {
                'symbol': symbol,
                'category': category,
                'status': 'success',
                'records': len(df),
                'error': None,
                'source': source,
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'category': category,
                'status': 'failed',
                'records': 0,
                'error': str(e)[:100],
                'source': source,
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
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self.download_stock, symbol, category): symbol 
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
        elif "full_us" not in universe or not universe.get("full_us"):
            universe = self._canonicalize_universe(dict(universe))
        
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
                'batch_size': self.batch_size
            },
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

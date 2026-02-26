#!/usr/bin/env python3
"""
Batch Data Fetcher - 批量数据获取器

支持：
- 多股票批量获取
- 大数据量分块处理
- 进度显示
- 断点续传
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from enhanced_data_layer import EnhancedDataLayer


class BatchDataFetcher:
    """批量数据获取器"""
    
    def __init__(
        self,
        market: str = "CN",
        max_workers: int = 5,
        batch_size: int = 50,
        retry_times: int = 3,
        verbose: bool = True
    ):
        self.market = market
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.retry_times = retry_times
        self.verbose = verbose
        
        self.data_layer = EnhancedDataLayer(market=market, verbose=False)
        self.results: Dict[str, pd.DataFrame] = {}
        self.failed_stocks: List[str] = []
    
    def _log(self, msg: str):
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [BatchFetcher] {msg}")
    
    def fetch_single_stock(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取单只股票数据（带重试）"""
        for attempt in range(self.retry_times):
            try:
                df = self.data_layer.fetch_and_process(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                if attempt < self.retry_times - 1:
                    time.sleep(1)
                    continue
                if self.verbose:
                    print(f"  获取 {symbol} 失败: {e}")
        
        return None
    
    def fetch_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        批量获取股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            progress_callback: 进度回调函数
        
        Returns:
            合并后的DataFrame
        """
        total = len(symbols)
        self._log(f"开始批量获取: {total} 只股票, 时间范围: {start_date} - {end_date}")
        
        all_data = []
        completed = 0
        
        # 使用线程池并行获取
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_symbol = {
                executor.submit(self.fetch_single_stock, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        all_data.append(df)
                        self.results[symbol] = df
                    else:
                        self.failed_stocks.append(symbol)
                except Exception as e:
                    self.failed_stocks.append(symbol)
                    if self.verbose:
                        print(f"  {symbol} 处理异常: {e}")
                
                completed += 1
                if self.verbose and completed % 10 == 0:
                    progress = completed / total * 100
                    self._log(f"进度: {completed}/{total} ({progress:.1f}%), 成功: {len(all_data)}, 失败: {len(self.failed_stocks)}")
                
                if progress_callback:
                    progress_callback(completed, total, symbol)
        
        # 合并所有数据
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self._log(f"批量获取完成: 成功 {len(all_data)} 只, 失败 {len(self.failed_stocks)} 只, 总记录 {len(combined)} 条")
            return combined
        else:
            self._log(f"批量获取失败: 所有 {total} 只股票都未能获取数据")
            return pd.DataFrame()
    
    def fetch_with_cache(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        cache_dir: str = "/tmp/quant_data_cache"
    ) -> pd.DataFrame:
        """带缓存的批量获取"""
        import pickle
        import hashlib
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存key
        cache_key = hashlib.md5(
            f"{self.market}_{start_date}_{end_date}_{len(symbols)}".encode()
        ).hexdigest()
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        # 检查缓存
        if os.path.exists(cache_file):
            self._log(f"从缓存加载数据: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # 获取数据
        df = self.fetch_batch(symbols, start_date, end_date)
        
        # 保存缓存
        if not df.empty:
            self._log(f"保存数据到缓存: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
        
        return df
    
    def get_failed_stocks(self) -> List[str]:
        """获取获取失败的股票列表"""
        return self.failed_stocks
    
    def retry_failed(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """重试获取失败的股票"""
        if not self.failed_stocks:
            return pd.DataFrame()
        
        self._log(f"重试 {len(self.failed_stocks)} 只失败的股票")
        
        retry_fetcher = BatchDataFetcher(
            market=self.market,
            max_workers=2,  # 减少并发
            verbose=self.verbose
        )
        
        df = retry_fetcher.fetch_batch(
            self.failed_stocks,
            start_date,
            end_date
        )
        
        # 更新结果
        self.results.update(retry_fetcher.results)
        self.failed_stocks = retry_fetcher.failed_stocks
        
        return df


# 便捷函数
def fetch_large_universe(
    symbols: List[str],
    lookback_years: float = 5.0,
    market: str = "CN",
    max_workers: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    获取大量股票数据
    
    Args:
        symbols: 股票代码列表
        lookback_years: 回溯年数
        market: 市场
        max_workers: 并行 workers
        verbose: 是否显示进度
    
    Returns:
        合并后的DataFrame
    """
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * lookback_years)
    
    fetcher = BatchDataFetcher(
        market=market,
        max_workers=max_workers,
        verbose=verbose
    )
    
    df = fetcher.fetch_batch(
        symbols=symbols,
        start_date=start_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d')
    )
    
    # 重试失败的股票
    if fetcher.get_failed_stocks():
        retry_df = fetcher.retry_failed(
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d')
        )
        if not retry_df.empty:
            df = pd.concat([df, retry_df], ignore_index=True)
    
    return df


# 测试
if __name__ == '__main__':
    print("=" * 80)
    print("Batch Data Fetcher - 测试")
    print("=" * 80)
    
    # 测试股票
    test_stocks = ['000001.SZ', '600000.SH', '000858.SZ']
    
    df = fetch_large_universe(
        symbols=test_stocks,
        lookback_years=0.5,
        max_workers=3,
        verbose=True
    )
    
    print(f"\n获取结果: {len(df)} 条记录")
    if not df.empty:
        print(f"列数: {len(df.columns)}")
        print(f"股票数: {df['symbol'].nunique()}")

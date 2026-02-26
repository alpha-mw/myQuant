#!/usr/bin/env python3
"""
Batch Data Pipeline - 批量数据获取流水线

支持：
- 5年历史数据
- 大批量股票并行获取
- 自动分批处理
- 进度保存和恢复
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pickle

# 导入现有模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enhanced_data_layer import EnhancedDataLayer, DataCleaner, FeatureEngineer
from stock_universe import StockUniverse, get_major_indices


class BatchDataPipeline:
    """
    批量数据获取流水线
    
    支持大规模股票数据的批量获取和处理
    """
    
    def __init__(
        self,
        market: str = "CN",
        max_workers: int = 5,
        batch_size: int = 50,
        cache_dir: str = "/tmp/quant_cache",
        verbose: bool = True
    ):
        self.market = market
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化组件
        self.data_layer = EnhancedDataLayer(market=market, verbose=False)
        self.stock_universe = StockUniverse()
        
        self.results: List[pd.DataFrame] = []
        self.failed_stocks: List[str] = []
    
    def _log(self, msg: str):
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [BatchPipeline] {msg}")
    
    def _get_cache_path(self, stock: str, start_date: str, end_date: str) -> str:
        """获取缓存文件路径"""
        cache_key = f"{stock}_{start_date}_{end_date}.pkl"
        return os.path.join(self.cache_dir, cache_key)
    
    def _load_from_cache(self, stock: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(stock, start_date, end_date)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, stock: str, start_date: str, end_date: str, df: pd.DataFrame):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(stock, start_date, end_date)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            self._log(f"缓存保存失败 {stock}: {e}")
    
    def fetch_single_stock(
        self,
        stock: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """获取单只股票数据"""
        # 尝试从缓存加载
        if use_cache:
            cached = self._load_from_cache(stock, start_date, end_date)
            if cached is not None:
                return cached
        
        try:
            df = self.data_layer.fetch_and_process(
                symbol=stock,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                # 保存到缓存
                if use_cache:
                    self._save_to_cache(stock, start_date, end_date, df)
                return df
            
        except Exception as e:
            self._log(f"获取失败 {stock}: {e}")
        
        return None
    
    def fetch_batch(
        self,
        stocks: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        批量获取股票数据
        
        Args:
            stocks: 股票代码列表
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            use_cache: 是否使用缓存
        
        Returns:
            合并后的DataFrame
        """
        self._log(f"开始批量获取: {len(stocks)} 只股票, {start_date} 至 {end_date}")
        
        all_data = []
        success_count = 0
        fail_count = 0
        
        # 分批处理
        for i in range(0, len(stocks), self.batch_size):
            batch = stocks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(stocks) + self.batch_size - 1) // self.batch_size
            
            self._log(f"处理第 {batch_num}/{total_batches} 批: {len(batch)} 只股票")
            
            # 并行获取
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.fetch_single_stock,
                        stock,
                        start_date,
                        end_date,
                        use_cache
                    ): stock for stock in batch
                }
                
                for future in as_completed(futures):
                    stock = futures[future]
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            all_data.append(df)
                            success_count += 1
                        else:
                            self.failed_stocks.append(stock)
                            fail_count += 1
                    except Exception as e:
                        self._log(f"处理异常 {stock}: {e}")
                        self.failed_stocks.append(stock)
                        fail_count += 1
            
            # 批次间暂停，避免rate limit
            if batch_num < total_batches:
                time.sleep(1)
        
        self._log(f"批量获取完成: 成功 {success_count}, 失败 {fail_count}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()
    
    def fetch_major_indices(
        self,
        lookback_years: float = 5.0,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取主要指数成分股数据 (沪深300+中证500+中证1000)
        
        Args:
            lookback_years: 回溯年数 (默认5年)
            use_cache: 是否使用缓存
        """
        # 获取股票池
        stocks = self.stock_universe.get_major_indices()
        
        if not stocks:
            self._log("未能获取股票池")
            return pd.DataFrame()
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * lookback_years)
        
        self._log(f"获取主要指数成分股: {len(stocks)} 只, {lookback_years} 年数据")
        
        return self.fetch_batch(
            stocks=stocks,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            use_cache=use_cache
        )
    
    def fetch_sample(
        self,
        n: int = 200,
        lookback_years: float = 5.0,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取抽样股票数据
        
        Args:
            n: 抽样数量
            lookback_years: 回溯年数
            use_cache: 是否使用缓存
        """
        # 获取抽样股票
        stocks = self.stock_universe.get_sample_stocks(n)
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * lookback_years)
        
        self._log(f"获取抽样股票: {len(stocks)} 只, {lookback_years} 年数据")
        
        return self.fetch_batch(
            stocks=stocks,
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            use_cache=use_cache
        )


# 便捷函数
def fetch_major_indices_data(
    lookback_years: float = 5.0,
    max_workers: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    便捷函数：获取主要指数成分股数据
    
    Args:
        lookback_years: 回溯年数 (默认5年)
        max_workers: 并行线程数
        verbose: 是否打印日志
    """
    pipeline = BatchDataPipeline(
        max_workers=max_workers,
        verbose=verbose
    )
    
    return pipeline.fetch_major_indices(lookback_years=lookback_years)


def fetch_sample_data(
    n: int = 200,
    lookback_years: float = 5.0,
    max_workers: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    便捷函数：获取抽样股票数据
    
    Args:
        n: 抽样数量
        lookback_years: 回溯年数
        max_workers: 并行线程数
        verbose: 是否打印日志
    """
    pipeline = BatchDataPipeline(
        max_workers=max_workers,
        verbose=verbose
    )
    
    return pipeline.fetch_sample(n=n, lookback_years=lookback_years)


if __name__ == '__main__':
    print("=" * 80)
    print("Batch Data Pipeline - 测试")
    print("=" * 80)
    
    # 测试获取抽样数据
    df = fetch_sample_data(n=10, lookback_years=1.0, verbose=True)
    
    print(f"\n获取数据: {len(df)} 行")
    if not df.empty:
        print(f"股票数: {df['symbol'].nunique()}")
        print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"列数: {len(df.columns)}")

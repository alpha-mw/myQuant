"""
PersistentUSDataManager - 集成持久化存储的美股数据管理器

基于V2.6的USMacroDataManager，集成V2.7的PersistentDataManager，
实现美股数据的持久化存储和增量更新。

V2.7 核心组件
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd

# 添加路径
sys.path.insert(0, '/home/ubuntu/skills/quant-investor/scripts/v2.7')
sys.path.insert(0, '/home/ubuntu/skills/quant-investor/scripts/v2.6/us_macro_data')

from persistent_data_manager import PersistentDataManager

# 尝试导入美股数据客户端
try:
    from fred_client import FREDClient
    from yfinance_client import YFinanceClient
    from finnhub_client import FinnhubClient
    CLIENTS_AVAILABLE = True
except ImportError:
    CLIENTS_AVAILABLE = False


class PersistentUSDataManager:
    """集成持久化存储的美股数据管理器"""
    
    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        finnhub_api_key: Optional[str] = None,
        data_dir: Optional[str] = None
    ):
        """
        初始化数据管理器
        
        Args:
            fred_api_key: FRED API密钥
            finnhub_api_key: Finnhub API密钥
            data_dir: 数据存储目录
        """
        # 初始化持久化存储管理器
        self.storage = PersistentDataManager(data_dir=data_dir)
        
        # 初始化各数据源客户端
        self.fred_client = None
        self.yfinance_client = None
        self.finnhub_client = None
        
        if CLIENTS_AVAILABLE:
            # FRED客户端
            fred_key = fred_api_key or os.environ.get('FRED_API_KEY')
            if fred_key:
                try:
                    self.fred_client = FREDClient(api_key=fred_key)
                    print("✓ FRED客户端初始化成功")
                except Exception as e:
                    print(f"✗ FRED客户端初始化失败: {e}")
            
            # YFinance客户端
            try:
                self.yfinance_client = YFinanceClient()
                print("✓ YFinance客户端初始化成功")
            except Exception as e:
                print(f"✗ YFinance客户端初始化失败: {e}")
            
            # Finnhub客户端
            fh_key = finnhub_api_key or os.environ.get('FINNHUB_API_KEY')
            if fh_key:
                try:
                    self.finnhub_client = FinnhubClient(api_key=fh_key)
                    print("✓ Finnhub客户端初始化成功")
                except Exception as e:
                    print(f"✗ Finnhub客户端初始化失败: {e}")
    
    # ========== 宏观经济数据 ==========
    
    def get_gdp(self, start_date: Optional[str] = None, real: bool = True, 
                force_refresh: bool = False) -> pd.DataFrame:
        """
        获取美国GDP数据（持久化存储）
        
        Args:
            start_date: 开始日期
            real: 是否获取实际GDP
            force_refresh: 是否强制刷新
        """
        def fetch_func():
            if self.fred_client:
                return self.fred_client.get_gdp(start_date=start_date, real=real)
            return None
        
        return self.storage.query(
            data_type='us_macro_gdp',
            start_date=start_date,
            fetch_func=fetch_func,
            source='fred',
            date_column='date',
            force_refresh=force_refresh
        )
    
    def get_cpi(self, start_date: Optional[str] = None, core: bool = False,
                force_refresh: bool = False) -> pd.DataFrame:
        """获取美国CPI数据（持久化存储）"""
        def fetch_func():
            if self.fred_client:
                return self.fred_client.get_cpi(start_date=start_date, core=core)
            return None
        
        data_type = 'us_macro_cpi_core' if core else 'us_macro_cpi'
        return self.storage.query(
            data_type=data_type,
            start_date=start_date,
            fetch_func=fetch_func,
            source='fred',
            date_column='date',
            force_refresh=force_refresh
        )
    
    def get_pce(self, start_date: Optional[str] = None, core: bool = True,
                force_refresh: bool = False) -> pd.DataFrame:
        """获取美国PCE数据（持久化存储）"""
        def fetch_func():
            if self.fred_client:
                return self.fred_client.get_pce(start_date=start_date, core=core)
            return None
        
        data_type = 'us_macro_pce_core' if core else 'us_macro_pce'
        return self.storage.query(
            data_type=data_type,
            start_date=start_date,
            fetch_func=fetch_func,
            source='fred',
            date_column='date',
            force_refresh=force_refresh
        )
    
    def get_unemployment(self, start_date: Optional[str] = None,
                         force_refresh: bool = False) -> pd.DataFrame:
        """获取美国失业率数据（持久化存储）"""
        def fetch_func():
            if self.fred_client:
                return self.fred_client.get_unemployment(start_date=start_date)
            return None
        
        return self.storage.query(
            data_type='us_macro_unemployment',
            start_date=start_date,
            fetch_func=fetch_func,
            source='fred',
            date_column='date',
            force_refresh=force_refresh
        )
    
    def get_fed_rate(self, start_date: Optional[str] = None,
                     force_refresh: bool = False) -> pd.DataFrame:
        """获取联邦基金利率（持久化存储）"""
        def fetch_func():
            if self.fred_client:
                return self.fred_client.get_fed_rate(start_date=start_date)
            return None
        
        return self.storage.query(
            data_type='us_macro_fed_rate',
            start_date=start_date,
            fetch_func=fetch_func,
            source='fred',
            date_column='date',
            force_refresh=force_refresh
        )
    
    def get_treasury_yield(self, maturity: str = '10y', start_date: Optional[str] = None,
                           force_refresh: bool = False) -> pd.DataFrame:
        """获取美国国债收益率（持久化存储）"""
        def fetch_func():
            if self.fred_client:
                return self.fred_client.get_treasury_yield(maturity=maturity, start_date=start_date)
            return None
        
        return self.storage.query(
            data_type=f'us_macro_treasury_{maturity}',
            start_date=start_date,
            fetch_func=fetch_func,
            source='fred',
            date_column='date',
            force_refresh=force_refresh
        )
    
    # ========== 美股行情数据 ==========
    
    def get_stock_history(
        self,
        ticker: str,
        period: str = '1y',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取美股历史行情数据（持久化存储）
        
        Args:
            ticker: 股票代码
            period: 时间周期
            start_date: 开始日期
            end_date: 结束日期
            force_refresh: 是否强制刷新
        """
        def fetch_func():
            if self.yfinance_client:
                return self.yfinance_client.get_stock_history(
                    ticker, period=period, start_date=start_date, end_date=end_date
                )
            return None
        
        return self.storage.query(
            data_type='us_stock_daily',
            symbol=ticker.upper(),
            start_date=start_date,
            end_date=end_date,
            fetch_func=fetch_func,
            source='yfinance',
            date_column='date',
            force_refresh=force_refresh
        )
    
    def get_stock_info(self, ticker: str, force_refresh: bool = False) -> Dict:
        """获取股票基本信息（持久化存储）"""
        # 股票信息使用JSON存储，这里简化处理
        if self.yfinance_client:
            return self.yfinance_client.get_stock_info(ticker)
        return {}
    
    # ========== 市场指数数据 ==========
    
    def get_vix(self, start_date: Optional[str] = None, period: str = '1y',
                force_refresh: bool = False) -> pd.DataFrame:
        """获取VIX恐慌指数（持久化存储）"""
        def fetch_func():
            if self.yfinance_client:
                df = self.yfinance_client.get_index_history('vix', period=period)
                if not df.empty:
                    return df[['date', 'close']].rename(columns={'close': 'value'})
            return None
        
        return self.storage.query(
            data_type='us_index_vix',
            start_date=start_date,
            fetch_func=fetch_func,
            source='yfinance',
            date_column='date',
            force_refresh=force_refresh
        )
    
    def get_sp500(self, start_date: Optional[str] = None, period: str = '1y',
                  force_refresh: bool = False) -> pd.DataFrame:
        """获取标普500指数（持久化存储）"""
        def fetch_func():
            if self.yfinance_client:
                df = self.yfinance_client.get_index_history('sp500', period=period)
                if not df.empty:
                    return df[['date', 'close']].rename(columns={'close': 'value'})
            return None
        
        return self.storage.query(
            data_type='us_index_sp500',
            start_date=start_date,
            fetch_func=fetch_func,
            source='yfinance',
            date_column='date',
            force_refresh=force_refresh
        )
    
    # ========== 工具方法 ==========
    
    def get_storage_summary(self) -> Dict:
        """获取存储摘要"""
        return self.storage.get_storage_summary()
    
    def list_cached_data(self, data_type: Optional[str] = None) -> List[Dict]:
        """列出已缓存的数据"""
        return self.storage.list_data(data_type)
    
    def clear_expired_data(self, days: int = 30) -> int:
        """清理过期数据"""
        return self.storage.clear_expired(days)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.storage.get_stats()


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("PersistentUSDataManager 测试")
    print("=" * 60)
    
    # 初始化管理器
    manager = PersistentUSDataManager()
    
    # 测试获取美股数据
    print("\n1. 测试获取AAPL历史数据:")
    try:
        aapl = manager.get_stock_history('AAPL', period='1mo')
        if aapl is not None and not aapl.empty:
            print(f"   获取到 {len(aapl)} 条数据")
            print(f"   日期范围: {aapl['date'].min()} - {aapl['date'].max()}")
        else:
            print("   未获取到数据")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 测试再次获取（应该命中缓存）
    print("\n2. 再次获取AAPL数据（应命中缓存）:")
    try:
        aapl2 = manager.get_stock_history('AAPL', period='1mo')
        if aapl2 is not None:
            print(f"   获取到 {len(aapl2)} 条数据")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 显示存储摘要
    print("\n3. 存储摘要:")
    summary = manager.get_storage_summary()
    print(f"   总数据集: {summary['total_datasets']}")
    print(f"   总行数: {summary['total_rows']}")
    print(f"   磁盘使用: {summary['disk_usage_mb']} MB")
    
    # 显示统计信息
    print("\n4. 缓存统计:")
    stats = manager.get_stats()
    print(f"   缓存命中: {stats['cache_hits']}")
    print(f"   部分命中: {stats['partial_hits']}")
    print(f"   缓存未命中: {stats['cache_misses']}")

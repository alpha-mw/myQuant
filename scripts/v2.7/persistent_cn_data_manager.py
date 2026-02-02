"""
PersistentCNDataManager - 集成持久化存储的A股数据管理器

基于V2.5的CNMacroDataManager和TushareClientExtended，
集成V2.7的PersistentDataManager，实现A股数据的持久化存储和增量更新。

V2.7 核心组件
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd

# 添加路径
sys.path.insert(0, '/home/ubuntu/skills/quant-investor/scripts/v2.7')
sys.path.insert(0, '/home/ubuntu/skills/quant-investor/scripts/v2.5')

from persistent_data_manager import PersistentDataManager

# 尝试导入Tushare
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

# 尝试导入AKShare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


class PersistentCNDataManager:
    """集成持久化存储的A股数据管理器"""
    
    # 主要市场指数代码
    INDEX_CODES = {
        'sse': '000001.SH',      # 上证指数
        'szse': '399001.SZ',     # 深证成指
        'hs300': '000300.SH',    # 沪深300
        'csi500': '000905.SH',   # 中证500
        'gem': '399006.SZ',      # 创业板指
        'star50': '000688.SH',   # 科创50
    }
    
    def __init__(
        self,
        tushare_token: Optional[str] = None,
        data_dir: Optional[str] = None
    ):
        """
        初始化数据管理器
        
        Args:
            tushare_token: Tushare Pro API token
            data_dir: 数据存储目录
        """
        # 初始化持久化存储管理器
        self.storage = PersistentDataManager(data_dir=data_dir)
        
        # 初始化Tushare客户端
        self.pro = None
        self.token = tushare_token or os.environ.get('TUSHARE_TOKEN')
        
        if TUSHARE_AVAILABLE and self.token:
            try:
                ts.set_token(self.token)
                self.pro = ts.pro_api()
                print("✓ Tushare客户端初始化成功")
            except Exception as e:
                print(f"✗ Tushare客户端初始化失败: {e}")
    
    # ========== 宏观经济数据 ==========
    
    def get_gdp(self, start_quarter: Optional[str] = None, end_quarter: Optional[str] = None,
                force_refresh: bool = False) -> pd.DataFrame:
        """
        获取中国GDP数据（持久化存储）
        
        Args:
            start_quarter: 开始季度，格式如 '2020Q1'
            end_quarter: 结束季度
            force_refresh: 是否强制刷新
        """
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.cn_gdp()
                except Exception as e:
                    print(f"Tushare获取GDP失败: {e}")
            
            if AKSHARE_AVAILABLE:
                try:
                    return ak.macro_china_gdp()
                except Exception as e:
                    print(f"AKShare获取GDP失败: {e}")
            
            return None
        
        return self.storage.query(
            data_type='cn_macro_gdp',
            fetch_func=fetch_func,
            source='tushare',
            date_column='quarter',
            force_refresh=force_refresh
        )
    
    def get_cpi(self, start_month: Optional[str] = None, end_month: Optional[str] = None,
                force_refresh: bool = False) -> pd.DataFrame:
        """获取中国CPI数据（持久化存储）"""
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.cn_cpi(start_m=start_month, end_m=end_month)
                except Exception as e:
                    print(f"Tushare获取CPI失败: {e}")
            
            if AKSHARE_AVAILABLE:
                try:
                    return ak.macro_china_cpi_monthly()
                except Exception as e:
                    print(f"AKShare获取CPI失败: {e}")
            
            return None
        
        return self.storage.query(
            data_type='cn_macro_cpi',
            fetch_func=fetch_func,
            source='tushare',
            date_column='month',
            force_refresh=force_refresh
        )
    
    def get_ppi(self, start_month: Optional[str] = None, end_month: Optional[str] = None,
                force_refresh: bool = False) -> pd.DataFrame:
        """获取中国PPI数据（持久化存储）"""
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.cn_ppi(start_m=start_month, end_m=end_month)
                except Exception as e:
                    print(f"Tushare获取PPI失败: {e}")
            
            if AKSHARE_AVAILABLE:
                try:
                    return ak.macro_china_ppi_yearly()
                except Exception as e:
                    print(f"AKShare获取PPI失败: {e}")
            
            return None
        
        return self.storage.query(
            data_type='cn_macro_ppi',
            fetch_func=fetch_func,
            source='tushare',
            date_column='month',
            force_refresh=force_refresh
        )
    
    def get_pmi(self, start_month: Optional[str] = None, end_month: Optional[str] = None,
                force_refresh: bool = False) -> pd.DataFrame:
        """获取中国PMI数据（持久化存储）"""
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.cn_pmi(start_m=start_month, end_m=end_month)
                except Exception as e:
                    print(f"Tushare获取PMI失败: {e}")
            
            if AKSHARE_AVAILABLE:
                try:
                    return ak.macro_china_pmi()
                except Exception as e:
                    print(f"AKShare获取PMI失败: {e}")
            
            return None
        
        return self.storage.query(
            data_type='cn_macro_pmi',
            fetch_func=fetch_func,
            source='tushare',
            date_column='month',
            force_refresh=force_refresh
        )
    
    # ========== 货币政策数据 ==========
    
    def get_lpr(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                force_refresh: bool = False) -> pd.DataFrame:
        """获取LPR贷款基础利率（持久化存储）"""
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.shibor_lpr(start_date=start_date, end_date=end_date)
                except Exception as e:
                    print(f"Tushare获取LPR失败: {e}")
            
            if AKSHARE_AVAILABLE:
                try:
                    return ak.macro_china_lpr()
                except Exception as e:
                    print(f"AKShare获取LPR失败: {e}")
            
            return None
        
        return self.storage.query(
            data_type='cn_macro_lpr',
            start_date=start_date,
            end_date=end_date,
            fetch_func=fetch_func,
            source='tushare',
            date_column='date',
            force_refresh=force_refresh
        )
    
    def get_shibor(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                   force_refresh: bool = False) -> pd.DataFrame:
        """获取Shibor上海银行间同业拆放利率（持久化存储）"""
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.shibor(start_date=start_date, end_date=end_date)
                except Exception as e:
                    print(f"Tushare获取Shibor失败: {e}")
            return None
        
        return self.storage.query(
            data_type='cn_macro_shibor',
            start_date=start_date,
            end_date=end_date,
            fetch_func=fetch_func,
            source='tushare',
            date_column='date',
            force_refresh=force_refresh
        )
    
    # ========== A股行情数据 ==========
    
    def get_stock_daily(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取A股日线行情数据（持久化存储）
        
        Args:
            ts_code: 股票代码，如 '000001.SZ'
            start_date: 开始日期，格式 'YYYYMMDD'
            end_date: 结束日期
            force_refresh: 是否强制刷新
        """
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                except Exception as e:
                    print(f"Tushare获取{ts_code}日线失败: {e}")
            return None
        
        return self.storage.query(
            data_type='cn_stock_daily',
            symbol=ts_code,
            start_date=start_date,
            end_date=end_date,
            fetch_func=fetch_func,
            source='tushare',
            date_column='trade_date',
            force_refresh=force_refresh
        )
    
    def get_stock_basic(self, ts_code: str, force_refresh: bool = False) -> Dict:
        """获取股票基本信息"""
        if self.pro:
            try:
                df = self.pro.stock_basic(ts_code=ts_code)
                if not df.empty:
                    return df.iloc[0].to_dict()
            except Exception as e:
                print(f"获取{ts_code}基本信息失败: {e}")
        return {}
    
    # ========== 指数数据 ==========
    
    def get_index_daily(
        self,
        ts_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取指数日线行情（持久化存储）
        
        Args:
            ts_code: 指数代码，如 '000300.SH'
            start_date: 开始日期
            end_date: 结束日期
            force_refresh: 是否强制刷新
        """
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                except Exception as e:
                    print(f"Tushare获取{ts_code}指数日线失败: {e}")
            return None
        
        return self.storage.query(
            data_type='cn_index_daily',
            symbol=ts_code,
            start_date=start_date,
            end_date=end_date,
            fetch_func=fetch_func,
            source='tushare',
            date_column='trade_date',
            force_refresh=force_refresh
        )
    
    # ========== 高级数据（一万分权限） ==========
    
    def get_margin(
        self,
        trade_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取融资融券数据（持久化存储）
        
        Args:
            trade_date: 交易日期
            start_date: 开始日期
            end_date: 结束日期
            force_refresh: 是否强制刷新
        """
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.margin(trade_date=trade_date, start_date=start_date, end_date=end_date)
                except Exception as e:
                    print(f"Tushare获取融资融券数据失败: {e}")
            return None
        
        return self.storage.query(
            data_type='cn_margin',
            start_date=start_date,
            end_date=end_date,
            fetch_func=fetch_func,
            source='tushare',
            date_column='trade_date',
            force_refresh=force_refresh
        )
    
    def get_top_list(
        self,
        trade_date: str,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取龙虎榜数据（持久化存储）
        
        Args:
            trade_date: 交易日期
            force_refresh: 是否强制刷新
        """
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.top_list(trade_date=trade_date)
                except Exception as e:
                    print(f"Tushare获取龙虎榜数据失败: {e}")
            return None
        
        return self.storage.query(
            data_type='cn_top_list',
            symbol=trade_date,  # 使用日期作为symbol区分不同日期的数据
            fetch_func=fetch_func,
            source='tushare',
            date_column='trade_date',
            force_refresh=force_refresh
        )
    
    def get_hk_hold(
        self,
        trade_date: Optional[str] = None,
        ts_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取沪深股通持股数据（持久化存储）
        
        Args:
            trade_date: 交易日期
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            force_refresh: 是否强制刷新
        """
        def fetch_func():
            if self.pro:
                try:
                    return self.pro.hk_hold(
                        trade_date=trade_date, ts_code=ts_code,
                        start_date=start_date, end_date=end_date
                    )
                except Exception as e:
                    print(f"Tushare获取沪深股通数据失败: {e}")
            return None
        
        symbol = ts_code if ts_code else trade_date
        return self.storage.query(
            data_type='cn_hk_hold',
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            fetch_func=fetch_func,
            source='tushare',
            date_column='trade_date',
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
    print("PersistentCNDataManager 测试")
    print("=" * 60)
    
    # 初始化管理器
    manager = PersistentCNDataManager()
    
    # 测试获取A股数据
    print("\n1. 测试获取平安银行日线数据:")
    try:
        stock = manager.get_stock_daily('000001.SZ', start_date='20240101')
        if stock is not None and not stock.empty:
            print(f"   获取到 {len(stock)} 条数据")
            print(f"   日期范围: {stock['trade_date'].min()} - {stock['trade_date'].max()}")
        else:
            print("   未获取到数据（可能需要配置Tushare Token）")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 显示存储摘要
    print("\n2. 存储摘要:")
    summary = manager.get_storage_summary()
    print(f"   总数据集: {summary['total_datasets']}")
    print(f"   总行数: {summary['total_rows']}")
    print(f"   磁盘使用: {summary['disk_usage_mb']} MB")
    
    # 显示统计信息
    print("\n3. 缓存统计:")
    stats = manager.get_stats()
    print(f"   缓存命中: {stats['cache_hits']}")
    print(f"   部分命中: {stats['partial_hits']}")
    print(f"   缓存未命中: {stats['cache_misses']}")

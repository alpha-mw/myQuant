"""
数据源管理器
负责智能调度Tushare和AKShare两个数据源，实现自动切换和容错
"""

import pandas as pd
from typing import Optional, Dict, Any, Callable
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

from tushare_client import TushareClient
from akshare_client import AKShareClient


class DataSourceManager:
    """数据源管理器"""
    
    def __init__(self, 
                 tushare_token: Optional[str] = None,
                 cache_dir: str = '/home/ubuntu/.quant_cache'):
        """
        初始化数据源管理器
        
        Args:
            tushare_token: Tushare Pro的token
            cache_dir: 缓存目录
        """
        # 初始化数据源客户端
        try:
            self.tushare = TushareClient(token=tushare_token)
            self.tushare_available = True
        except Exception as e:
            print(f"⚠️  Tushare初始化失败: {e}")
            self.tushare = None
            self.tushare_available = False
        
        try:
            self.akshare = AKShareClient()
            self.akshare_available = True
        except Exception as e:
            print(f"⚠️  AKShare初始化失败: {e}")
            self.akshare = None
            self.akshare_available = False
        
        # 缓存配置
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = True
        
        # 数据源优先级策略
        self.source_priority = {
            'stock_basic': ['tushare', 'akshare'],
            'daily': ['tushare', 'akshare'],
            'financial': ['tushare', 'akshare'],
            'macro': ['tushare', 'akshare'],
            'fund_flow': ['akshare', 'tushare'],
            'concept': ['akshare'],
            'hot_rank': ['akshare']
        }
        
        # 统计信息
        self.stats = {
            'tushare_calls': 0,
            'akshare_calls': 0,
            'cache_hits': 0,
            'errors': []
        }
    
    def _get_cache_key(self, data_type: str, params: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            data_type: 数据类型
            params: 参数字典
            
        Returns:
            str: 缓存键
        """
        # 将参数排序后转换为JSON字符串
        params_str = json.dumps(params, sort_keys=True)
        # 生成MD5哈希
        hash_obj = hashlib.md5(f"{data_type}:{params_str}".encode())
        return hash_obj.hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.parquet"
    
    def _load_from_cache(self, cache_key: str, ttl_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        Args:
            cache_key: 缓存键
            ttl_hours: 缓存有效期（小时）
            
        Returns:
            DataFrame或None: 缓存的数据，如果缓存不存在或已过期则返回None
        """
        if not self.enable_cache:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # 检查缓存是否过期
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - cache_time > timedelta(hours=ttl_hours):
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            self.stats['cache_hits'] += 1
            return df
        except Exception as e:
            print(f"⚠️  缓存加载失败: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, df: pd.DataFrame):
        """
        保存数据到缓存
        
        Args:
            cache_key: 缓存键
            df: 要缓存的数据
        """
        if not self.enable_cache or df is None or df.empty:
            return
        
        try:
            cache_path = self._get_cache_path(cache_key)
            df.to_parquet(cache_path)
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")
    
    def _fetch_with_fallback(self,
                            data_type: str,
                            tushare_func: Optional[Callable] = None,
                            akshare_func: Optional[Callable] = None,
                            params: Optional[Dict[str, Any]] = None,
                            ttl_hours: int = 24) -> pd.DataFrame:
        """
        从数据源获取数据，支持自动降级
        
        Args:
            data_type: 数据类型
            tushare_func: Tushare获取函数
            akshare_func: AKShare获取函数
            params: 参数字典
            ttl_hours: 缓存有效期（小时）
            
        Returns:
            DataFrame: 获取的数据
        """
        params = params or {}
        
        # 生成缓存键
        cache_key = self._get_cache_key(data_type, params)
        
        # 尝试从缓存加载
        cached_data = self._load_from_cache(cache_key, ttl_hours)
        if cached_data is not None:
            return cached_data
        
        # 获取数据源优先级
        priority = self.source_priority.get(data_type, ['tushare', 'akshare'])
        
        # 按优先级尝试获取数据
        for source in priority:
            if source == 'tushare' and self.tushare_available and tushare_func:
                try:
                    df = tushare_func(**params)
                    self.stats['tushare_calls'] += 1
                    self._save_to_cache(cache_key, df)
                    return df
                except PermissionError as e:
                    error_msg = f"Tushare权限不足: {e}"
                    self.stats['errors'].append(error_msg)
                    print(f"❌ {error_msg}")
                    print("⚠️  请联系用户解决Tushare权限问题")
                    # 权限不足时，尝试下一个数据源
                    continue
                except Exception as e:
                    error_msg = f"Tushare获取失败 ({data_type}): {e}"
                    self.stats['errors'].append(error_msg)
                    print(f"⚠️  {error_msg}，尝试使用备用数据源...")
                    continue
            
            elif source == 'akshare' and self.akshare_available and akshare_func:
                try:
                    df = akshare_func(**params)
                    self.stats['akshare_calls'] += 1
                    self._save_to_cache(cache_key, df)
                    return df
                except Exception as e:
                    error_msg = f"AKShare获取失败 ({data_type}): {e}"
                    self.stats['errors'].append(error_msg)
                    print(f"⚠️  {error_msg}")
                    continue
        
        # 所有数据源都失败
        raise RuntimeError(f"所有数据源都无法获取数据 ({data_type})")
    
    # ==================== 统一数据接口 ====================
    
    def get_stock_list(self, exchange: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            exchange: 交易所 SSE上交所 SZSE深交所
            
        Returns:
            DataFrame: 股票列表
        """
        return self._fetch_with_fallback(
            data_type='stock_basic',
            tushare_func=lambda **kw: self.tushare.get_stock_basic(exchange=exchange),
            akshare_func=lambda **kw: self.akshare.get_stock_info_a_code_name(),
            params={'exchange': exchange},
            ttl_hours=24 * 7  # 股票列表缓存7天
        )
    
    def get_daily_price(self,
                       stock_code: str,
                       start_date: str,
                       end_date: str,
                       adjust: str = 'qfq') -> pd.DataFrame:
        """
        获取股票日线行情
        
        Args:
            stock_code: 股票代码（Tushare格式，如 600519.SH）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            adjust: 复权类型 '' 不复权 'qfq' 前复权 'hfq' 后复权
            
        Returns:
            DataFrame: 日线行情数据
        """
        # 转换股票代码格式（Tushare: 600519.SH, AKShare: 600519）
        akshare_code = stock_code.split('.')[0]
        
        return self._fetch_with_fallback(
            data_type='daily',
            tushare_func=lambda **kw: self.tushare.get_daily(
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date
            ),
            akshare_func=lambda **kw: self.akshare.get_stock_zh_a_hist(
                symbol=akshare_code,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            ),
            params={'stock_code': stock_code, 'start_date': start_date, 'end_date': end_date, 'adjust': adjust},
            ttl_hours=24  # 日线数据缓存1天
        )
    
    def get_financial_indicator(self,
                                stock_code: str,
                                period: str) -> pd.DataFrame:
        """
        获取财务指标
        
        Args:
            stock_code: 股票代码（Tushare格式）
            period: 报告期 YYYYMMDD
            
        Returns:
            DataFrame: 财务指标数据
        """
        return self._fetch_with_fallback(
            data_type='financial',
            tushare_func=lambda **kw: self.tushare.get_fina_indicator(
                ts_code=stock_code,
                period=period
            ),
            params={'stock_code': stock_code, 'period': period},
            ttl_hours=24 * 30  # 财务数据缓存30天
        )
    
    def get_macro_gdp(self) -> pd.DataFrame:
        """
        获取GDP数据
        
        Returns:
            DataFrame: GDP数据
        """
        return self._fetch_with_fallback(
            data_type='macro',
            tushare_func=lambda **kw: self.tushare.get_cn_gdp(),
            akshare_func=lambda **kw: self.akshare.get_macro_china_gdp(),
            params={},
            ttl_hours=24 * 30  # 宏观数据缓存30天
        )
    
    def get_macro_cpi(self) -> pd.DataFrame:
        """
        获取CPI数据
        
        Returns:
            DataFrame: CPI数据
        """
        return self._fetch_with_fallback(
            data_type='macro',
            tushare_func=lambda **kw: self.tushare.get_cn_cpi(),
            akshare_func=lambda **kw: self.akshare.get_macro_china_cpi(),
            params={},
            ttl_hours=24 * 30
        )
    
    def get_fund_flow(self,
                     stock_code: str,
                     start_date: str,
                     end_date: str) -> pd.DataFrame:
        """
        获取个股资金流向
        
        Args:
            stock_code: 股票代码（Tushare格式）
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            
        Returns:
            DataFrame: 资金流向数据
        """
        # 转换股票代码格式
        akshare_code = stock_code.split('.')[0]
        market = 'sh' if stock_code.endswith('.SH') else 'sz'
        
        return self._fetch_with_fallback(
            data_type='fund_flow',
            tushare_func=lambda **kw: self.tushare.get_moneyflow(
                ts_code=stock_code,
                start_date=start_date,
                end_date=end_date
            ),
            akshare_func=lambda **kw: self.akshare.get_stock_individual_fund_flow(
                stock=akshare_code,
                market=market
            ),
            params={'stock_code': stock_code, 'start_date': start_date, 'end_date': end_date},
            ttl_hours=24
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据源管理器统计信息"""
        return {
            **self.stats,
            'tushare_available': self.tushare_available,
            'akshare_available': self.akshare_available,
            'cache_enabled': self.enable_cache,
            'cache_dir': str(self.cache_dir)
        }


if __name__ == '__main__':
    # 测试代码
    import os
    
    token = os.getenv('TUSHARE_TOKEN')
    manager = DataSourceManager(tushare_token=token)
    
    print("=" * 60)
    print("数据源管理器测试")
    print("=" * 60)
    
    # 测试获取股票列表
    print("\n1. 测试获取股票列表...")
    try:
        stock_list = manager.get_stock_list(exchange='SSE')
        print(f"✅ 成功获取上交所股票 {len(stock_list)} 只")
        print(stock_list.head())
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # 测试获取日线行情
    print("\n2. 测试获取日线行情...")
    try:
        daily_data = manager.get_daily_price('600519.SH', '20260101', '20260131', 'qfq')
        print(f"✅ 成功获取贵州茅台1月行情 {len(daily_data)} 条")
        print(daily_data.head())
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # 测试缓存（再次获取相同数据）
    print("\n3. 测试缓存机制（再次获取相同数据）...")
    try:
        daily_data2 = manager.get_daily_price('600519.SH', '20260101', '20260131', 'qfq')
        print(f"✅ 成功从缓存获取数据 {len(daily_data2)} 条")
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # 测试获取财务指标
    print("\n4. 测试获取财务指标...")
    try:
        fina_data = manager.get_financial_indicator('600519.SH', '20231231')
        print(f"✅ 成功获取贵州茅台2023年财务指标")
        print(fina_data.head())
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # 测试获取宏观数据
    print("\n5. 测试获取GDP数据...")
    try:
        gdp_data = manager.get_macro_gdp()
        print(f"✅ 成功获取GDP数据 {len(gdp_data)} 条")
        print(gdp_data.tail())
    except Exception as e:
        print(f"❌ 失败: {e}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据源管理器统计信息:")
    print("=" * 60)
    stats = manager.get_stats()
    for key, value in stats.items():
        if key != 'errors':
            print(f"{key}: {value}")
    
    if stats['errors']:
        print("\n错误记录:")
        for error in stats['errors'][:5]:  # 只显示前5个错误
            print(f"  - {error}")

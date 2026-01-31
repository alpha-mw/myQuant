"""
统一数据API
为上层分析应用提供统一、简洁、语义化的数据访问接口
"""

import sys
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.5/data_acquisition')

import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta
from data_source_manager import DataSourceManager


class DataAPI:
    """统一数据API"""
    
    def __init__(self, tushare_token: Optional[str] = None):
        """
        初始化数据API
        
        Args:
            tushare_token: Tushare Pro的token
        """
        self.manager = DataSourceManager(tushare_token=tushare_token)
    
    # ==================== 股票数据 ====================
    
    def get_stock_list(self, market: str = 'all') -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            market: 市场 'all' 全部 'sh' 上交所 'sz' 深交所
            
        Returns:
            DataFrame: 股票列表
        """
        if market == 'all':
            return self.manager.get_stock_list()
        elif market == 'sh':
            return self.manager.get_stock_list(exchange='SSE')
        elif market == 'sz':
            return self.manager.get_stock_list(exchange='SZSE')
        else:
            raise ValueError(f"不支持的市场类型: {market}")
    
    def get_stock_price(self,
                       stock_code: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       adjust: str = 'qfq',
                       days: int = 250) -> pd.DataFrame:
        """
        获取股票价格数据
        
        Args:
            stock_code: 股票代码（如 600519.SH）
            start_date: 开始日期 YYYYMMDD，如果不提供则自动计算
            end_date: 结束日期 YYYYMMDD，如果不提供则使用今天
            adjust: 复权类型 'qfq' 前复权 'hfq' 后复权 '' 不复权
            days: 如果不提供start_date，则获取最近N天的数据
            
        Returns:
            DataFrame: 价格数据
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if not start_date:
            start_dt = datetime.now() - timedelta(days=days)
            start_date = start_dt.strftime('%Y%m%d')
        
        return self.manager.get_daily_price(stock_code, start_date, end_date, adjust)
    
    def get_stock_financial(self,
                           stock_code: str,
                           period: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票财务指标
        
        Args:
            stock_code: 股票代码（如 600519.SH）
            period: 报告期 YYYYMMDD，如果不提供则获取最新的
            
        Returns:
            DataFrame: 财务指标数据
        """
        if not period:
            # 获取最近的年报期（去年12月31日）
            last_year = datetime.now().year - 1
            period = f"{last_year}1231"
        
        return self.manager.get_financial_indicator(stock_code, period)
    
    def get_stock_fund_flow(self,
                           stock_code: str,
                           days: int = 30) -> pd.DataFrame:
        """
        获取股票资金流向
        
        Args:
            stock_code: 股票代码（如 600519.SH）
            days: 获取最近N天的数据
            
        Returns:
            DataFrame: 资金流向数据
        """
        end_date = datetime.now().strftime('%Y%m%d')
        start_dt = datetime.now() - timedelta(days=days)
        start_date = start_dt.strftime('%Y%m%d')
        
        return self.manager.get_fund_flow(stock_code, start_date, end_date)
    
    # ==================== 宏观数据 ====================
    
    def get_gdp(self) -> pd.DataFrame:
        """
        获取GDP数据
        
        Returns:
            DataFrame: GDP数据
        """
        return self.manager.get_macro_gdp()
    
    def get_cpi(self) -> pd.DataFrame:
        """
        获取CPI数据
        
        Returns:
            DataFrame: CPI数据
        """
        return self.manager.get_macro_cpi()
    
    # ==================== 分析辅助方法 ====================
    
    def get_latest_trading_day(self) -> str:
        """
        获取最新交易日
        
        Returns:
            str: 最新交易日 YYYYMMDD
        """
        # 简单实现：返回今天或昨天（如果今天是周末）
        today = datetime.now()
        if today.weekday() == 5:  # 周六
            today = today - timedelta(days=1)
        elif today.weekday() == 6:  # 周日
            today = today - timedelta(days=2)
        
        return today.strftime('%Y%m%d')
    
    def normalize_stock_code(self, code: str) -> str:
        """
        标准化股票代码为Tushare格式
        
        Args:
            code: 股票代码（可能是 600519 或 600519.SH）
            
        Returns:
            str: 标准化后的代码（600519.SH）
        """
        if '.' in code:
            return code
        
        # 根据代码前缀判断市场
        if code.startswith('6'):
            return f"{code}.SH"
        elif code.startswith(('0', '3')):
            return f"{code}.SZ"
        elif code.startswith('8') or code.startswith('4'):
            return f"{code}.BJ"
        else:
            raise ValueError(f"无法识别的股票代码: {code}")
    
    def get_stats(self):
        """获取数据API统计信息"""
        return self.manager.get_stats()


if __name__ == '__main__':
    # 测试代码
    import os
    
    token = os.getenv('TUSHARE_TOKEN')
    api = DataAPI(tushare_token=token)
    
    print("=" * 60)
    print("统一数据API测试")
    print("=" * 60)
    
    # 测试获取股票列表
    print("\n1. 测试获取上交所股票列表...")
    stock_list = api.get_stock_list(market='sh')
    print(f"✅ 获取 {len(stock_list)} 只股票")
    print(stock_list.head())
    
    # 测试获取股票价格
    print("\n2. 测试获取股票价格（最近30天）...")
    price_data = api.get_stock_price('600519.SH', days=30, adjust='qfq')
    print(f"✅ 获取 {len(price_data)} 条价格数据")
    print(price_data.head())
    
    # 测试获取财务指标
    print("\n3. 测试获取财务指标...")
    financial_data = api.get_stock_financial('600519.SH')
    print(f"✅ 获取财务指标")
    print(financial_data.head())
    
    # 测试获取GDP数据
    print("\n4. 测试获取GDP数据...")
    gdp_data = api.get_gdp()
    print(f"✅ 获取 {len(gdp_data)} 条GDP数据")
    print(gdp_data.tail())
    
    # 测试代码标准化
    print("\n5. 测试股票代码标准化...")
    print(f"600519 -> {api.normalize_stock_code('600519')}")
    print(f"000001 -> {api.normalize_stock_code('000001')}")
    print(f"300750 -> {api.normalize_stock_code('300750')}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据API统计信息:")
    print("=" * 60)
    stats = api.get_stats()
    for key, value in stats.items():
        if key != 'errors':
            print(f"{key}: {value}")

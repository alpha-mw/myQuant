"""
Tushare Pro数据客户端
负责与Tushare Pro API进行交互，获取股票、宏观、行业等金融数据
"""

import tushare as ts
import pandas as pd
from typing import Optional, Dict, Any, List
import time
import os
from datetime import datetime


class TushareClient:
    """Tushare Pro数据客户端"""
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化Tushare客户端
        
        Args:
            token: Tushare Pro的token，如果不提供则从环境变量TUSHARE_TOKEN读取
        """
        self.token = token or os.getenv('TUSHARE_TOKEN')
        if not self.token:
            raise ValueError("Tushare token未提供，请设置TUSHARE_TOKEN环境变量或传入token参数")
        
        # 设置token
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        
        # API调用统计
        self.call_count = 0
        self.last_call_time = None
        
    def _rate_limit(self, min_interval: float = 0.2):
        """
        API调用频率限制
        
        Args:
            min_interval: 最小调用间隔（秒）
        """
        if self.last_call_time:
            elapsed = time.time() - self.last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        
        self.last_call_time = time.time()
        self.call_count += 1
    
    def _call_api(self, api_name: str, **kwargs) -> pd.DataFrame:
        """
        统一的API调用接口
        
        Args:
            api_name: API名称
            **kwargs: API参数
            
        Returns:
            DataFrame: 返回的数据
        """
        self._rate_limit()
        
        try:
            api_func = getattr(self.pro, api_name)
            df = api_func(**kwargs)
            return df
        except Exception as e:
            error_msg = str(e)
            # 检查是否是权限不足的错误
            if '权限' in error_msg or '积分' in error_msg or 'permission' in error_msg.lower():
                raise PermissionError(f"Tushare权限不足: {error_msg}")
            else:
                raise RuntimeError(f"Tushare API调用失败 ({api_name}): {error_msg}")
    
    # ==================== 股票基础数据 ====================
    
    def get_stock_basic(self, 
                       exchange: Optional[str] = None,
                       list_status: str = 'L') -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            exchange: 交易所 SSE上交所 SZSE深交所 BSE北交所
            list_status: 上市状态 L上市 D退市 P暂停上市
            
        Returns:
            DataFrame: 股票列表
        """
        params = {'list_status': list_status}
        if exchange:
            params['exchange'] = exchange
        
        return self._call_api('stock_basic', **params)
    
    def get_trade_cal(self, 
                     exchange: str = 'SSE',
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取交易日历
        
        Args:
            exchange: 交易所 SSE上交所 SZSE深交所
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            
        Returns:
            DataFrame: 交易日历
        """
        params = {'exchange': exchange}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('trade_cal', **params)
    
    # ==================== 股票行情数据 ====================
    
    def get_daily(self,
                 ts_code: Optional[str] = None,
                 trade_date: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票日线行情
        
        Args:
            ts_code: 股票代码 (如 600000.SH)
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            
        Returns:
            DataFrame: 日线行情数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('daily', **params)
    
    def get_adj_factor(self,
                      ts_code: Optional[str] = None,
                      trade_date: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取复权因子
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            
        Returns:
            DataFrame: 复权因子数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('adj_factor', **params)
    
    def get_daily_basic(self,
                       ts_code: Optional[str] = None,
                       trade_date: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取每日指标（PE、PB、PS、市值、换手率等）
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            
        Returns:
            DataFrame: 每日指标数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('daily_basic', **params)
    
    # ==================== 财务数据 ====================
    
    def get_income(self,
                  ts_code: Optional[str] = None,
                  period: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取利润表
        
        Args:
            ts_code: 股票代码
            period: 报告期 YYYYMMDD
            start_date: 报告期开始日期
            end_date: 报告期结束日期
            
        Returns:
            DataFrame: 利润表数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if period:
            params['period'] = period
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('income', **params)
    
    def get_balancesheet(self,
                        ts_code: Optional[str] = None,
                        period: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取资产负债表
        
        Args:
            ts_code: 股票代码
            period: 报告期 YYYYMMDD
            start_date: 报告期开始日期
            end_date: 报告期结束日期
            
        Returns:
            DataFrame: 资产负债表数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if period:
            params['period'] = period
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('balancesheet', **params)
    
    def get_cashflow(self,
                    ts_code: Optional[str] = None,
                    period: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取现金流量表
        
        Args:
            ts_code: 股票代码
            period: 报告期 YYYYMMDD
            start_date: 报告期开始日期
            end_date: 报告期结束日期
            
        Returns:
            DataFrame: 现金流量表数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if period:
            params['period'] = period
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('cashflow', **params)
    
    def get_fina_indicator(self,
                          ts_code: Optional[str] = None,
                          period: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取财务指标数据（ROE、ROA、负债率等）
        
        Args:
            ts_code: 股票代码
            period: 报告期 YYYYMMDD
            start_date: 报告期开始日期
            end_date: 报告期结束日期
            
        Returns:
            DataFrame: 财务指标数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if period:
            params['period'] = period
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('fina_indicator', **params)
    
    def get_forecast(self,
                    ts_code: Optional[str] = None,
                    ann_date: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    period: Optional[str] = None) -> pd.DataFrame:
        """
        获取业绩预告
        
        Args:
            ts_code: 股票代码
            ann_date: 公告日期 YYYYMMDD
            start_date: 公告开始日期
            end_date: 公告结束日期
            period: 报告期 YYYYMMDD
            
        Returns:
            DataFrame: 业绩预告数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if period:
            params['period'] = period
        
        return self._call_api('forecast', **params)
    
    # ==================== 宏观经济数据 ====================
    
    def get_cn_gdp(self,
                  start_q: Optional[str] = None,
                  end_q: Optional[str] = None) -> pd.DataFrame:
        """
        获取中国GDP数据
        
        Args:
            start_q: 开始季度 YYYYQ1/Q2/Q3/Q4
            end_q: 结束季度
            
        Returns:
            DataFrame: GDP数据
        """
        params = {}
        if start_q:
            params['start_q'] = start_q
        if end_q:
            params['end_q'] = end_q
        
        return self._call_api('cn_gdp', **params)
    
    def get_cn_cpi(self,
                  start_m: Optional[str] = None,
                  end_m: Optional[str] = None) -> pd.DataFrame:
        """
        获取中国CPI数据
        
        Args:
            start_m: 开始月份 YYYYMM
            end_m: 结束月份
            
        Returns:
            DataFrame: CPI数据
        """
        params = {}
        if start_m:
            params['start_m'] = start_m
        if end_m:
            params['end_m'] = end_m
        
        return self._call_api('cn_cpi', **params)
    
    def get_cn_ppi(self,
                  start_m: Optional[str] = None,
                  end_m: Optional[str] = None) -> pd.DataFrame:
        """
        获取中国PPI数据
        
        Args:
            start_m: 开始月份 YYYYMM
            end_m: 结束月份
            
        Returns:
            DataFrame: PPI数据
        """
        params = {}
        if start_m:
            params['start_m'] = start_m
        if end_m:
            params['end_m'] = end_m
        
        return self._call_api('cn_ppi', **params)
    
    def get_cn_m(self,
                start_m: Optional[str] = None,
                end_m: Optional[str] = None) -> pd.DataFrame:
        """
        获取中国货币供应量（M0、M1、M2）
        
        Args:
            start_m: 开始月份 YYYYMM
            end_m: 结束月份
            
        Returns:
            DataFrame: 货币供应量数据
        """
        params = {}
        if start_m:
            params['start_m'] = start_m
        if end_m:
            params['end_m'] = end_m
        
        return self._call_api('cn_m', **params)
    
    # ==================== 资金流向数据 ====================
    
    def get_moneyflow(self,
                     ts_code: Optional[str] = None,
                     trade_date: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取个股资金流向
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 资金流向数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('moneyflow', **params)
    
    def get_stk_factor(self,
                      ts_code: Optional[str] = None,
                      trade_date: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取技术因子（MACD、KDJ、RSI等）
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 技术因子数据
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('stk_factor', **params)
    
    def get_call_stats(self) -> Dict[str, Any]:
        """获取API调用统计信息"""
        return {
            'total_calls': self.call_count,
            'last_call_time': datetime.fromtimestamp(self.last_call_time) if self.last_call_time else None
        }


if __name__ == '__main__':
    # 测试代码
    try:
        client = TushareClient()
        
        # 测试获取股票列表
        print("测试获取股票列表...")
        stock_list = client.get_stock_basic(exchange='SSE', list_status='L')
        print(f"上交所上市股票数量: {len(stock_list)}")
        print(stock_list.head())
        
        # 测试获取日线行情
        print("\n测试获取日线行情...")
        daily_data = client.get_daily(ts_code='600519.SH', start_date='20260101', end_date='20260131')
        print(f"贵州茅台1月行情数据: {len(daily_data)}条")
        print(daily_data.head())
        
        # 测试获取财务指标
        print("\n测试获取财务指标...")
        fina_data = client.get_fina_indicator(ts_code='600519.SH', period='20231231')
        print("贵州茅台2023年财务指标:")
        print(fina_data.head())
        
        # 打印调用统计
        print("\nAPI调用统计:")
        print(client.get_call_stats())
        
    except PermissionError as e:
        print(f"❌ Tushare权限不足，请联系用户解决: {e}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

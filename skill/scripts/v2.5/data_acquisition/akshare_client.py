"""
AKShare数据客户端
负责与AKShare库进行交互，获取股票、宏观、行业等金融数据
作为Tushare的补充数据源
"""

import akshare as ak
import pandas as pd
from typing import Optional, Dict, Any
import time


class AKShareClient:
    """AKShare数据客户端"""
    
    def __init__(self):
        """初始化AKShare客户端"""
        self.call_count = 0
        self.last_call_time = None
        
    def _rate_limit(self, min_interval: float = 0.5):
        """
        API调用频率限制（AKShare爬虫需要更长的间隔）
        
        Args:
            min_interval: 最小调用间隔（秒）
        """
        if self.last_call_time:
            elapsed = time.time() - self.last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        
        self.last_call_time = time.time()
        self.call_count += 1
    
    def _call_api(self, func_name: str, *args, **kwargs) -> pd.DataFrame:
        """
        统一的API调用接口
        
        Args:
            func_name: AKShare函数名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            DataFrame: 返回的数据
        """
        self._rate_limit()
        
        try:
            func = getattr(ak, func_name)
            df = func(*args, **kwargs)
            return df
        except Exception as e:
            raise RuntimeError(f"AKShare API调用失败 ({func_name}): {str(e)}")
    
    # ==================== 股票基础数据 ====================
    
    def get_stock_info_a_code_name(self) -> pd.DataFrame:
        """
        获取A股股票代码和名称
        
        Returns:
            DataFrame: 股票代码和名称
        """
        return self._call_api('stock_info_a_code_name')
    
    def get_stock_zh_a_spot_em(self) -> pd.DataFrame:
        """
        获取A股实时行情（东方财富）
        
        Returns:
            DataFrame: A股实时行情
        """
        return self._call_api('stock_zh_a_spot_em')
    
    # ==================== 股票行情数据 ====================
    
    def get_stock_zh_a_hist(self,
                           symbol: str,
                           period: str = 'daily',
                           start_date: str = '20200101',
                           end_date: str = '20261231',
                           adjust: str = '') -> pd.DataFrame:
        """
        获取A股历史行情数据
        
        Args:
            symbol: 股票代码（不带后缀，如 600519）
            period: 周期 daily日线 weekly周线 monthly月线
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            adjust: 复权类型 '' 不复权 'qfq' 前复权 'hfq' 后复权
            
        Returns:
            DataFrame: 历史行情数据
        """
        return self._call_api('stock_zh_a_hist',
                             symbol=symbol,
                             period=period,
                             start_date=start_date,
                             end_date=end_date,
                             adjust=adjust)
    
    def get_stock_zh_a_hist_min_em(self,
                                   symbol: str,
                                   period: str = '5',
                                   adjust: str = '') -> pd.DataFrame:
        """
        获取A股分钟级行情数据
        
        Args:
            symbol: 股票代码（不带后缀，如 600519）
            period: 周期 1 5 15 30 60
            adjust: 复权类型 '' 不复权 'qfq' 前复权 'hfq' 后复权
            
        Returns:
            DataFrame: 分钟级行情数据
        """
        return self._call_api('stock_zh_a_hist_min_em',
                             symbol=symbol,
                             period=period,
                             adjust=adjust)
    
    # ==================== 财务数据 ====================
    
    def get_stock_financial_report_sina(self,
                                       stock: str,
                                       symbol: str) -> pd.DataFrame:
        """
        获取新浪财经财务报表
        
        Args:
            stock: 股票代码（如 sh600519）
            symbol: 报表类型 资产负债表 利润表 现金流量表
            
        Returns:
            DataFrame: 财务报表数据
        """
        return self._call_api('stock_financial_report_sina',
                             stock=stock,
                             symbol=symbol)
    
    def get_stock_yjbb_em(self, date: str) -> pd.DataFrame:
        """
        获取东方财富业绩报表
        
        Args:
            date: 报告期 YYYY-MM-DD
            
        Returns:
            DataFrame: 业绩报表数据
        """
        return self._call_api('stock_yjbb_em', date=date)
    
    def get_stock_yjyg_em(self, date: str) -> pd.DataFrame:
        """
        获取东方财富业绩预告
        
        Args:
            date: 报告期 YYYY-MM-DD
            
        Returns:
            DataFrame: 业绩预告数据
        """
        return self._call_api('stock_yjyg_em', date=date)
    
    # ==================== 宏观经济数据 ====================
    
    def get_macro_china_gdp(self) -> pd.DataFrame:
        """
        获取中国GDP数据
        
        Returns:
            DataFrame: GDP数据
        """
        return self._call_api('macro_china_gdp')
    
    def get_macro_china_cpi(self) -> pd.DataFrame:
        """
        获取中国CPI数据
        
        Returns:
            DataFrame: CPI数据
        """
        return self._call_api('macro_china_cpi')
    
    def get_macro_china_ppi(self) -> pd.DataFrame:
        """
        获取中国PPI数据
        
        Returns:
            DataFrame: PPI数据
        """
        return self._call_api('macro_china_ppi')
    
    def get_macro_china_pmi(self) -> pd.DataFrame:
        """
        获取中国PMI数据
        
        Returns:
            DataFrame: PMI数据
        """
        return self._call_api('macro_china_pmi')
    
    def get_macro_china_money_supply(self) -> pd.DataFrame:
        """
        获取中国货币供应量（M0、M1、M2）
        
        Returns:
            DataFrame: 货币供应量数据
        """
        return self._call_api('macro_china_money_supply')
    
    def get_macro_china_shrzgm(self) -> pd.DataFrame:
        """
        获取中国社会融资规模
        
        Returns:
            DataFrame: 社会融资规模数据
        """
        return self._call_api('macro_china_shrzgm')
    
    # ==================== 资金流向数据 ====================
    
    def get_stock_individual_fund_flow(self,
                                      stock: str,
                                      market: str = 'sh') -> pd.DataFrame:
        """
        获取个股资金流向
        
        Args:
            stock: 股票代码（如 600519）
            market: 市场 sh上交所 sz深交所
            
        Returns:
            DataFrame: 个股资金流向数据
        """
        return self._call_api('stock_individual_fund_flow',
                             stock=stock,
                             market=market)
    
    def get_stock_individual_fund_flow_rank(self, indicator: str = '今日') -> pd.DataFrame:
        """
        获取个股资金流向排名
        
        Args:
            indicator: 指标 今日 3日 5日 10日
            
        Returns:
            DataFrame: 资金流向排名
        """
        return self._call_api('stock_individual_fund_flow_rank',
                             indicator=indicator)
    
    def get_stock_sector_fund_flow_rank(self,
                                       indicator: str = '今日',
                                       sector_type: str = '行业资金流') -> pd.DataFrame:
        """
        获取板块资金流向排名
        
        Args:
            indicator: 指标 今日 3日 5日 10日
            sector_type: 板块类型 行业资金流 概念资金流 地域资金流
            
        Returns:
            DataFrame: 板块资金流向排名
        """
        return self._call_api('stock_sector_fund_flow_rank',
                             indicator=indicator,
                             sector_type=sector_type)
    
    # ==================== 概念板块数据 ====================
    
    def get_stock_board_concept_name_em(self) -> pd.DataFrame:
        """
        获取东方财富概念板块列表
        
        Returns:
            DataFrame: 概念板块列表
        """
        return self._call_api('stock_board_concept_name_em')
    
    def get_stock_board_concept_cons_em(self, symbol: str) -> pd.DataFrame:
        """
        获取东方财富概念板块成分股
        
        Args:
            symbol: 概念板块代码
            
        Returns:
            DataFrame: 概念板块成分股
        """
        return self._call_api('stock_board_concept_cons_em', symbol=symbol)
    
    # ==================== 行业板块数据 ====================
    
    def get_stock_board_industry_name_em(self) -> pd.DataFrame:
        """
        获取东方财富行业板块列表
        
        Returns:
            DataFrame: 行业板块列表
        """
        return self._call_api('stock_board_industry_name_em')
    
    def get_stock_board_industry_cons_em(self, symbol: str) -> pd.DataFrame:
        """
        获取东方财富行业板块成分股
        
        Args:
            symbol: 行业板块代码
            
        Returns:
            DataFrame: 行业板块成分股
        """
        return self._call_api('stock_board_industry_cons_em', symbol=symbol)
    
    # ==================== 龙虎榜数据 ====================
    
    def get_stock_lhb_detail_em(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取龙虎榜详情
        
        Args:
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            
        Returns:
            DataFrame: 龙虎榜详情
        """
        return self._call_api('stock_lhb_detail_em',
                             start_date=start_date,
                             end_date=end_date)
    
    # ==================== 股票热度数据 ====================
    
    def get_stock_hot_rank_em(self) -> pd.DataFrame:
        """
        获取东方财富人气榜
        
        Returns:
            DataFrame: 人气榜数据
        """
        return self._call_api('stock_hot_rank_em')
    
    def get_stock_hot_rank_latest_em(self) -> pd.DataFrame:
        """
        获取东方财富最新人气榜
        
        Returns:
            DataFrame: 最新人气榜数据
        """
        return self._call_api('stock_hot_rank_latest_em')
    
    def get_call_stats(self) -> Dict[str, Any]:
        """获取API调用统计信息"""
        from datetime import datetime
        return {
            'total_calls': self.call_count,
            'last_call_time': datetime.fromtimestamp(self.last_call_time) if self.last_call_time else None
        }


if __name__ == '__main__':
    # 测试代码
    try:
        client = AKShareClient()
        
        # 测试获取股票列表
        print("测试获取A股股票列表...")
        stock_list = client.get_stock_info_a_code_name()
        print(f"A股股票数量: {len(stock_list)}")
        print(stock_list.head())
        
        # 测试获取历史行情
        print("\n测试获取历史行情...")
        hist_data = client.get_stock_zh_a_hist(symbol='600519', 
                                              start_date='20260101',
                                              end_date='20260131',
                                              adjust='qfq')
        print(f"贵州茅台1月行情数据: {len(hist_data)}条")
        print(hist_data.head())
        
        # 测试获取宏观数据
        print("\n测试获取GDP数据...")
        gdp_data = client.get_macro_china_gdp()
        print(f"GDP数据: {len(gdp_data)}条")
        print(gdp_data.tail())
        
        # 打印调用统计
        print("\nAPI调用统计:")
        print(client.get_call_stats())
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

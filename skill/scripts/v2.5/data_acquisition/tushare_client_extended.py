"""
Tushare Pro数据客户端扩展模块
利用一万分权限，新增情绪、筹码、衍生品等高级数据接口
"""

import tushare as ts
import pandas as pd
from typing import Optional, Dict, Any, List
import time
import os
from datetime import datetime


class TushareClientExtended:
    """Tushare Pro数据客户端扩展版（一万分权限）"""
    
    def __init__(self, token: Optional[str] = None, http_url: Optional[str] = None):
        """
        初始化Tushare客户端
        
        Args:
            token: Tushare Pro的token
            http_url: 自定义API服务地址
        """
        self.token = token or os.getenv('TUSHARE_TOKEN')
        self.http_url = http_url or os.getenv('TUSHARE_HTTP_URL')
        
        if not self.token:
            raise ValueError("Tushare token未提供，请设置TUSHARE_TOKEN环境变量或传入token参数")
        
        # 设置token
        ts.set_token(self.token)
        self.pro = ts.pro_api(self.token)
        
        # 配置自定义API服务（如果提供）
        if self.http_url:
            self.pro._DataApi__token = self.token
            self.pro._DataApi__http_url = self.http_url
        
        # API调用统计
        self.call_count = 0
        self.last_call_time = None
        
    def _rate_limit(self, min_interval: float = 0.2):
        """API调用频率限制"""
        if self.last_call_time:
            elapsed = time.time() - self.last_call_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        
        self.last_call_time = time.time()
        self.call_count += 1
    
    def _call_api(self, api_name: str, **kwargs) -> pd.DataFrame:
        """统一的API调用接口"""
        self._rate_limit()
        
        try:
            api_func = getattr(self.pro, api_name)
            df = api_func(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            error_msg = str(e)
            if '权限' in error_msg or '积分' in error_msg or 'permission' in error_msg.lower():
                raise PermissionError(f"Tushare权限不足: {error_msg}")
            else:
                raise RuntimeError(f"Tushare API调用失败 ({api_name}): {error_msg}")
    
    # ==================== 龙虎榜数据 ====================
    
    def get_top_list(self,
                    trade_date: Optional[str] = None,
                    ts_code: Optional[str] = None) -> pd.DataFrame:
        """
        获取龙虎榜每日明细
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            ts_code: 股票代码
            
        Returns:
            DataFrame: 龙虎榜数据（上榜原因、买入额、卖出额、净买入等）
        """
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
        if ts_code:
            params['ts_code'] = ts_code
        
        return self._call_api('top_list', **params)
    
    def get_top_inst(self,
                    trade_date: Optional[str] = None,
                    ts_code: Optional[str] = None) -> pd.DataFrame:
        """
        获取龙虎榜机构交易明细
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            ts_code: 股票代码
            
        Returns:
            DataFrame: 机构席位买卖明细
        """
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
        if ts_code:
            params['ts_code'] = ts_code
        
        return self._call_api('top_inst', **params)
    
    # ==================== 融资融券数据 ====================
    
    def get_margin(self,
                  trade_date: Optional[str] = None,
                  exchange_id: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取融资融券交易汇总
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            exchange_id: 交易所代码 SSE/SZSE
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 融资融券汇总数据（融资余额、融券余额等）
        """
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
        if exchange_id:
            params['exchange_id'] = exchange_id
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('margin', **params)
    
    def get_margin_detail(self,
                         trade_date: Optional[str] = None,
                         ts_code: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取融资融券交易明细
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 个股融资融券明细
        """
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
        if ts_code:
            params['ts_code'] = ts_code
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('margin_detail', **params)
    
    # ==================== 股东数据 ====================
    
    def get_stk_holdernumber(self,
                            ts_code: Optional[str] = None,
                            enddate: Optional[str] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取股东人数
        
        Args:
            ts_code: 股票代码
            enddate: 截止日期 YYYYMMDD
            start_date: 公告开始日期
            end_date: 公告结束日期
            
        Returns:
            DataFrame: 股东人数数据（股东户数、户均持股等）
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if enddate:
            params['enddate'] = enddate
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('stk_holdernumber', **params)
    
    def get_stk_holdertrade(self,
                           ts_code: Optional[str] = None,
                           ann_date: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           trade_type: Optional[str] = None) -> pd.DataFrame:
        """
        获取股东增减持
        
        Args:
            ts_code: 股票代码
            ann_date: 公告日期 YYYYMMDD
            start_date: 公告开始日期
            end_date: 公告结束日期
            trade_type: 交易类型 IN增持 DE减持
            
        Returns:
            DataFrame: 股东增减持数据
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
        if trade_type:
            params['trade_type'] = trade_type
        
        return self._call_api('stk_holdertrade', **params)
    
    # ==================== 大宗交易数据 ====================
    
    def get_block_trade(self,
                       ts_code: Optional[str] = None,
                       trade_date: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取大宗交易
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 大宗交易数据（成交价、成交量、折溢价等）
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
        
        return self._call_api('block_trade', **params)
    
    # ==================== 沪深股通数据 ====================
    
    def get_hk_hold(self,
                   ts_code: Optional[str] = None,
                   trade_date: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   exchange: Optional[str] = None) -> pd.DataFrame:
        """
        获取沪深股通持股明细
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所 SH/SZ
            
        Returns:
            DataFrame: 北向资金持股数据（持股数量、持股市值、持股比例等）
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
        if exchange:
            params['exchange'] = exchange
        
        return self._call_api('hk_hold', **params)
    
    # ==================== 限售股解禁数据 ====================
    
    def get_share_float(self,
                       ts_code: Optional[str] = None,
                       ann_date: Optional[str] = None,
                       float_date: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取限售股解禁
        
        Args:
            ts_code: 股票代码
            ann_date: 公告日期 YYYYMMDD
            float_date: 解禁日期 YYYYMMDD
            start_date: 解禁开始日期
            end_date: 解禁结束日期
            
        Returns:
            DataFrame: 限售股解禁数据（解禁数量、解禁比例等）
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if ann_date:
            params['ann_date'] = ann_date
        if float_date:
            params['float_date'] = float_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('share_float', **params)
    
    # ==================== 期权数据 ====================
    
    def get_opt_basic(self,
                     exchange: Optional[str] = None,
                     call_put: Optional[str] = None) -> pd.DataFrame:
        """
        获取期权合约列表
        
        Args:
            exchange: 交易所 SSE/SZSE/CFFEX/DCE/CZCE/SHFE
            call_put: 期权类型 C认购 P认沽
            
        Returns:
            DataFrame: 期权合约基础信息
        """
        params = {}
        if exchange:
            params['exchange'] = exchange
        if call_put:
            params['call_put'] = call_put
        
        return self._call_api('opt_basic', **params)
    
    def get_opt_daily(self,
                     ts_code: Optional[str] = None,
                     trade_date: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     exchange: Optional[str] = None) -> pd.DataFrame:
        """
        获取期权日线行情
        
        Args:
            ts_code: 期权代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所
            
        Returns:
            DataFrame: 期权日线数据（收盘价、结算价、成交量、持仓量等）
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
        if exchange:
            params['exchange'] = exchange
        
        return self._call_api('opt_daily', **params)
    
    # ==================== 期货数据 ====================
    
    def get_fut_basic(self,
                     exchange: Optional[str] = None,
                     fut_type: Optional[str] = None) -> pd.DataFrame:
        """
        获取期货合约列表
        
        Args:
            exchange: 交易所 CFFEX/DCE/CZCE/SHFE/INE
            fut_type: 合约类型 1主力 2连续
            
        Returns:
            DataFrame: 期货合约基础信息
        """
        params = {}
        if exchange:
            params['exchange'] = exchange
        if fut_type:
            params['fut_type'] = fut_type
        
        return self._call_api('fut_basic', **params)
    
    def get_fut_daily(self,
                     ts_code: Optional[str] = None,
                     trade_date: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     exchange: Optional[str] = None) -> pd.DataFrame:
        """
        获取期货日线行情
        
        Args:
            ts_code: 期货代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所
            
        Returns:
            DataFrame: 期货日线数据
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
        if exchange:
            params['exchange'] = exchange
        
        return self._call_api('fut_daily', **params)
    
    def get_fut_holding(self,
                       trade_date: Optional[str] = None,
                       symbol: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       exchange: Optional[str] = None) -> pd.DataFrame:
        """
        获取期货每日成交持仓排名
        
        Args:
            trade_date: 交易日期 YYYYMMDD
            symbol: 合约品种
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所
            
        Returns:
            DataFrame: 期货持仓排名数据
        """
        params = {}
        if trade_date:
            params['trade_date'] = trade_date
        if symbol:
            params['symbol'] = symbol
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if exchange:
            params['exchange'] = exchange
        
        return self._call_api('fut_holding', **params)
    
    # ==================== 基金数据 ====================
    
    def get_fund_basic(self,
                      market: Optional[str] = None,
                      status: Optional[str] = None) -> pd.DataFrame:
        """
        获取公募基金列表
        
        Args:
            market: 交易市场 E场内 O场外
            status: 存续状态 D摘牌 I发行 L上市中
            
        Returns:
            DataFrame: 基金列表
        """
        params = {}
        if market:
            params['market'] = market
        if status:
            params['status'] = status
        
        return self._call_api('fund_basic', **params)
    
    def get_fund_nav(self,
                    ts_code: Optional[str] = None,
                    nav_date: Optional[str] = None,
                    market: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取公募基金净值
        
        Args:
            ts_code: 基金代码
            nav_date: 净值日期 YYYYMMDD
            market: 交易市场 E场内 O场外
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 基金净值数据（单位净值、累计净值等）
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if nav_date:
            params['nav_date'] = nav_date
        if market:
            params['market'] = market
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('fund_nav', **params)
    
    def get_fund_portfolio(self,
                          ts_code: Optional[str] = None,
                          ann_date: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取公募基金持仓数据
        
        Args:
            ts_code: 基金代码
            ann_date: 公告日期 YYYYMMDD
            start_date: 报告期开始日期
            end_date: 报告期结束日期
            
        Returns:
            DataFrame: 基金持仓明细（持仓股票、持仓数量、持仓市值、占比等）
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
        
        return self._call_api('fund_portfolio', **params)
    
    # ==================== 指数数据 ====================
    
    def get_index_basic(self,
                       market: Optional[str] = None,
                       publisher: Optional[str] = None,
                       category: Optional[str] = None) -> pd.DataFrame:
        """
        获取指数基本信息
        
        Args:
            market: 交易所 SSE/SZSE/CSI/CICC/SW/OTH
            publisher: 发布商
            category: 指数类别
            
        Returns:
            DataFrame: 指数基本信息
        """
        params = {}
        if market:
            params['market'] = market
        if publisher:
            params['publisher'] = publisher
        if category:
            params['category'] = category
        
        return self._call_api('index_basic', **params)
    
    def get_index_daily(self,
                       ts_code: Optional[str] = None,
                       trade_date: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取指数日线行情
        
        Args:
            ts_code: 指数代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 指数日线数据
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
        
        return self._call_api('index_daily', **params)
    
    def get_index_weight(self,
                        index_code: Optional[str] = None,
                        trade_date: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取指数成分和权重
        
        Args:
            index_code: 指数代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 指数成分股及权重
        """
        params = {}
        if index_code:
            params['index_code'] = index_code
        if trade_date:
            params['trade_date'] = trade_date
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        return self._call_api('index_weight', **params)
    
    def get_index_dailybasic(self,
                            ts_code: Optional[str] = None,
                            trade_date: Optional[str] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取大盘指数每日指标
        
        Args:
            ts_code: 指数代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 指数每日指标（PE、PB、换手率等）
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
        
        return self._call_api('index_dailybasic', **params)
    
    # ==================== 申万行业分类 ====================
    
    def get_index_classify(self,
                          level: Optional[str] = None,
                          src: str = 'SW2021') -> pd.DataFrame:
        """
        获取申万行业分类
        
        Args:
            level: 行业级别 L1一级 L2二级 L3三级
            src: 指数来源 SW2021申万2021版
            
        Returns:
            DataFrame: 行业分类列表
        """
        params = {'src': src}
        if level:
            params['level'] = level
        
        return self._call_api('index_classify', **params)
    
    def get_index_member(self,
                        index_code: Optional[str] = None,
                        ts_code: Optional[str] = None,
                        is_new: Optional[str] = None) -> pd.DataFrame:
        """
        获取申万行业成分
        
        Args:
            index_code: 行业代码
            ts_code: 股票代码
            is_new: 是否最新 Y是 N否
            
        Returns:
            DataFrame: 行业成分股列表
        """
        params = {}
        if index_code:
            params['index_code'] = index_code
        if ts_code:
            params['ts_code'] = ts_code
        if is_new:
            params['is_new'] = is_new
        
        return self._call_api('index_member', **params)
    
    # ==================== 可转债数据 ====================
    
    def get_cb_basic(self,
                    ts_code: Optional[str] = None,
                    list_date: Optional[str] = None,
                    exchange: Optional[str] = None) -> pd.DataFrame:
        """
        获取可转债基础信息
        
        Args:
            ts_code: 转债代码
            list_date: 上市日期 YYYYMMDD
            exchange: 交易所 SSE/SZSE
            
        Returns:
            DataFrame: 可转债基础信息
        """
        params = {}
        if ts_code:
            params['ts_code'] = ts_code
        if list_date:
            params['list_date'] = list_date
        if exchange:
            params['exchange'] = exchange
        
        return self._call_api('cb_basic', **params)
    
    def get_cb_daily(self,
                    ts_code: Optional[str] = None,
                    trade_date: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取可转债日线数据
        
        Args:
            ts_code: 转债代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 可转债日线数据
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
        
        return self._call_api('cb_daily', **params)
    
    # ==================== 外汇数据 ====================
    
    def get_fx_obasic(self,
                     exchange: Optional[str] = None,
                     classify: Optional[str] = None) -> pd.DataFrame:
        """
        获取外汇基础信息
        
        Args:
            exchange: 交易所
            classify: 分类
            
        Returns:
            DataFrame: 外汇基础信息
        """
        params = {}
        if exchange:
            params['exchange'] = exchange
        if classify:
            params['classify'] = classify
        
        return self._call_api('fx_obasic', **params)
    
    def get_fx_daily(self,
                    ts_code: Optional[str] = None,
                    trade_date: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    exchange: Optional[str] = None) -> pd.DataFrame:
        """
        获取外汇日线行情
        
        Args:
            ts_code: 外汇代码
            trade_date: 交易日期 YYYYMMDD
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所
            
        Returns:
            DataFrame: 外汇日线数据
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
        if exchange:
            params['exchange'] = exchange
        
        return self._call_api('fx_daily', **params)
    
    def get_call_stats(self) -> Dict[str, Any]:
        """获取API调用统计信息"""
        return {
            'total_calls': self.call_count,
            'last_call_time': datetime.fromtimestamp(self.last_call_time) if self.last_call_time else None
        }


if __name__ == '__main__':
    # 测试代码
    import os
    
    # 设置环境变量
    token = os.getenv('TUSHARE_TOKEN')  # 从环境变量获取token
    http_url = os.getenv('TUSHARE_HTTP_URL', 'http://lianghua.nanyangqiankun.top')
    
    try:
        client = TushareClientExtended(token=token, http_url=http_url)
        
        print("=" * 60)
        print("Tushare一万分权限数据接口测试")
        print("=" * 60)
        
        # 测试龙虎榜数据
        print("\n1. 测试龙虎榜数据...")
        top_list = client.get_top_list(trade_date='20260128')
        print(f"   龙虎榜数据: {len(top_list)}条")
        if len(top_list) > 0:
            print(top_list.head(3))
        
        # 测试融资融券数据
        print("\n2. 测试融资融券数据...")
        margin = client.get_margin(trade_date='20260128')
        print(f"   融资融券汇总: {len(margin)}条")
        if len(margin) > 0:
            print(margin.head())
        
        # 测试股东人数
        print("\n3. 测试股东人数...")
        holder_num = client.get_stk_holdernumber(ts_code='600519.SH')
        print(f"   贵州茅台股东人数: {len(holder_num)}条")
        if len(holder_num) > 0:
            print(holder_num.head())
        
        # 测试大宗交易
        print("\n4. 测试大宗交易...")
        block = client.get_block_trade(trade_date='20260128')
        print(f"   大宗交易: {len(block)}条")
        if len(block) > 0:
            print(block.head(3))
        
        # 测试北向资金持股
        print("\n5. 测试北向资金持股...")
        hk_hold = client.get_hk_hold(ts_code='600519.SH')
        print(f"   贵州茅台北向持股: {len(hk_hold)}条")
        if len(hk_hold) > 0:
            print(hk_hold.head())
        
        # 测试期权数据
        print("\n6. 测试期权数据...")
        opt_basic = client.get_opt_basic(exchange='SSE')
        print(f"   上交所期权合约: {len(opt_basic)}条")
        if len(opt_basic) > 0:
            print(opt_basic.head(3))
        
        # 测试申万行业分类
        print("\n7. 测试申万行业分类...")
        sw_class = client.get_index_classify(level='L1')
        print(f"   申万一级行业: {len(sw_class)}条")
        if len(sw_class) > 0:
            print(sw_class.head())
        
        # 测试指数成分权重
        print("\n8. 测试指数成分权重...")
        index_weight = client.get_index_weight(index_code='000300.SH')
        print(f"   沪深300成分股: {len(index_weight)}条")
        if len(index_weight) > 0:
            print(index_weight.head())
        
        # 打印调用统计
        print("\n" + "=" * 60)
        print("API调用统计:")
        print(client.get_call_stats())
        print("=" * 60)
        
    except PermissionError as e:
        print(f"❌ Tushare权限不足: {e}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

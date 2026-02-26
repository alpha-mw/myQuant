#!/usr/bin/env python3
"""
Stock Universe - 全市场股票池获取

支持获取：
- 沪深300成分股
- 中证500成分股  
- 中证1000成分股
- 全市场股票
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime

# 读取credentials
sys.path.insert(0, '/root/.openclaw/workspace/myQuant')
try:
    from credentials import TUSHARE_TOKEN, TUSHARE_URL
except ImportError:
    TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '33d6ebd3bad7812192d768a191e29ebe653a1839b3f63ec8a0dd7da94172')
    TUSHARE_URL = os.environ.get('TUSHARE_URL', 'http://lianghua.nanyangqiankun.top')

import tushare as ts


class StockUniverse:
    """全市场股票池管理"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or TUSHARE_TOKEN
        self.pro = ts.pro_api(self.token)
        self.pro._DataApi__token = self.token
        self.pro._DataApi__http_url = TUSHARE_URL
        
        self._hs300_cache: Optional[List[str]] = None
        self._zz500_cache: Optional[List[str]] = None
        self._zz1000_cache: Optional[List[str]] = None
        self._all_stocks_cache: Optional[List[str]] = None
    
    def get_hs300(self, refresh: bool = False) -> List[str]:
        """获取沪深300成分股"""
        if self._hs300_cache is not None and not refresh:
            return self._hs300_cache
        
        try:
            df = self.pro.index_weight(index_code='000300.SH')
            if df is not None and not df.empty:
                self._hs300_cache = df['con_code'].tolist()
                print(f"[StockUniverse] 沪深300: {len(self._hs300_cache)} 只")
                return self._hs300_cache
        except Exception as e:
            print(f"[StockUniverse] 获取沪深300失败: {e}")
        
        return []
    
    def get_zz500(self, refresh: bool = False) -> List[str]:
        """获取中证500成分股"""
        if self._zz500_cache is not None and not refresh:
            return self._zz500_cache
        
        try:
            df = self.pro.index_weight(index_code='000905.SH')
            if df is not None and not df.empty:
                self._zz500_cache = df['con_code'].tolist()
                print(f"[StockUniverse] 中证500: {len(self._zz500_cache)} 只")
                return self._zz500_cache
        except Exception as e:
            print(f"[StockUniverse] 获取中证500失败: {e}")
        
        return []
    
    def get_zz1000(self, refresh: bool = False) -> List[str]:
        """获取中证1000成分股"""
        if self._zz1000_cache is not None and not refresh:
            return self._zz1000_cache
        
        try:
            df = self.pro.index_weight(index_code='000852.SH')
            if df is not None and not df.empty:
                self._zz1000_cache = df['con_code'].tolist()
                print(f"[StockUniverse] 中证1000: {len(self._zz1000_cache)} 只")
                return self._zz1000_cache
        except Exception as e:
            print(f"[StockUniverse] 获取中证1000失败: {e}")
        
        return []
    
    def get_major_indices(self, refresh: bool = False) -> List[str]:
        """获取主要指数成分股 (沪深300+中证500+中证1000)"""
        hs300 = self.get_hs300(refresh)
        zz500 = self.get_zz500(refresh)
        zz1000 = self.get_zz1000(refresh)
        
        # 合并去重
        all_stocks = list(set(hs300 + zz500 + zz1000))
        print(f"[StockUniverse] 主要指数合计: {len(all_stocks)} 只 (HS300:{len(hs300)}, ZZ500:{len(zz500)}, ZZ1000:{len(zz1000)})")
        
        return all_stocks
    
    def get_all_stocks(self, refresh: bool = False, limit: Optional[int] = None) -> List[str]:
        """获取全市场股票"""
        if self._all_stocks_cache is not None and not refresh:
            stocks = self._all_stocks_cache
            if limit:
                stocks = stocks[:limit]
            return stocks
        
        try:
            # 获取所有上市股票
            df = self.pro.stock_basic(exchange='', list_status='L')
            if df is not None and not df.empty:
                # 过滤掉ST、*ST、退市股票
                df = df[~df['name'].str.contains('ST|退|*', na=False)]
                # 只保留主板、创业板、科创板
                df = df[df['market'].isin(['主板', '创业板', '科创板'])]
                
                self._all_stocks_cache = df['ts_code'].tolist()
                print(f"[StockUniverse] 全市场股票: {len(self._all_stocks_cache)} 只")
                
                if limit:
                    return self._all_stocks_cache[:limit]
                return self._all_stocks_cache
        except Exception as e:
            print(f"[StockUniverse] 获取全市场股票失败: {e}")
        
        return []
    
    def get_sample_stocks(self, n: int = 100, seed: int = 42) -> List[str]:
        """随机抽样获取股票"""
        import random
        
        all_stocks = self.get_all_stocks()
        if len(all_stocks) <= n:
            return all_stocks
        
        random.seed(seed)
        sample = random.sample(all_stocks, n)
        print(f"[StockUniverse] 随机抽样: {n} 只")
        return sample


# 便捷函数
def get_hs300(token: Optional[str] = None) -> List[str]:
    """获取沪深300成分股"""
    return StockUniverse(token).get_hs300()

def get_zz500(token: Optional[str] = None) -> List[str]:
    """获取中证500成分股"""
    return StockUniverse(token).get_zz500()

def get_zz1000(token: Optional[str] = None) -> List[str]:
    """获取中证1000成分股"""
    return StockUniverse(token).get_zz1000()

def get_major_indices(token: Optional[str] = None) -> List[str]:
    """获取主要指数成分股 (沪深300+中证500+中证1000)"""
    return StockUniverse(token).get_major_indices()


if __name__ == '__main__':
    print("=" * 80)
    print("Stock Universe - 测试")
    print("=" * 80)
    
    universe = StockUniverse()
    
    # 测试获取主要指数
    stocks = universe.get_major_indices()
    print(f"\n前10只股票: {stocks[:10]}")

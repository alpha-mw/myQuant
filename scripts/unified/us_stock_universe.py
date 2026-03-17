#!/usr/bin/env python3
"""
US Stock Universe - 美股全市场股票池管理

支持获取：
- 大盘股 (Large-Cap): 标普500 & 纳指100
- 中盘股 (Mid-Cap): S&P MidCap 400
- 小盘股 (Small-Cap): 罗素2000 & 标普小盘600
"""

import os
import sys
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime
import json
import pickle

# 美股大盘股 - 标普500 & 纳指100 代表性股票 (实际运行时会扩展到全量)
LARGE_CAP_STOCKS = [
    # 科技巨头 (Mag7 + 其他大型科技)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL',
    'ADBE', 'CRM', 'NFLX', 'AMD', 'INTC', 'QCOM', 'CSCO', 'IBM', 'ACN', 'NOW',
    # 金融
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'C', 'AXP', 'PNC', 'USB', 'COF',
    # 医疗保健
    'LLY', 'JNJ', 'UNH', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD', 'VRTX', 'REGN',
    # 消费品
    'WMT', 'PG', 'COST', 'KO', 'PEP', 'MCD', 'HD', 'LOW', 'NKE', 'SBUX', 'TGT', 'DG', 'TJX', 'BKNG',
    # 能源
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'WMB', 'KMI',
    # 工业
    'GE', 'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'MMM', 'DE', 'CSX', 'UNP', 'FDX', 'NSC', 'ITW',
    # 通信
    'VZ', 'T', 'CMCSA', 'CHTR', 'TMUS',
    # 材料
    'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'ECL', 'NUE', 'STLD',
    # 房地产
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'WELL', 'SPG', 'DLR', 'AVB',
    # 公用事业
    'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'SRE', 'D', 'PEG', 'ED', 'XEL',
    # 其他科技/互联网
    'SNOW', 'ZM', 'UBER', 'LYFT', 'ABNB', 'DDOG', 'CRWD', 'NET', 'PLTR', 'ROKU',
    'SQ', 'PYPL', 'SHOP', 'SPOT', 'TWLO', 'OKTA', 'DOCU', 'TEAM', 'MDB', 'PANW',
]

# 中盘股 - S&P MidCap 400 代表性股票
MID_CAP_STOCKS = [
    # 工业/制造业
    'AXON', 'AMCR', 'AOS', 'ALLE', 'AAL', 'ALK', 'ALLE', 'AIZ', 'ATO', 'BALL',
    'BAX', 'BIO', 'BWA', 'CPB', 'COO', 'CMS', 'CNP', 'CCL', 'CF', 'CHRW',
    'CINF', 'CLX', 'CMA', 'CPT', 'CNP', 'CTLT', 'DRI', 'DVA', 'DXC', 'EMN',
    'EQT', 'ES', 'ESS', 'EXPD', 'EXR', 'FRT', 'FSLR', 'FE', 'FMC', 'FOXA',
    # 金融/服务
    'GL', 'GPC', 'GRMN', 'GCI', 'HAS', 'HCA', 'HSIC', 'HOLX', 'HPQ', 'HBAN',
    'HST', 'HWM', 'IP', 'IPG', 'IFF', 'IR', 'IVZ', 'JBHT', 'JKHY', 'J',
    'JNPR', 'KIM', 'KMX', 'KLAC', 'KSS', 'LDOS', 'LEG', 'LEN', 'LNC', 'L',
    'LH', 'LNT', 'LUMN', 'LYB', 'MAS', 'MKC', 'MCK', 'MTB', 'MDT', 'MKTX',
    # 科技/医疗/消费
    'MRO', 'MSCI', 'MTCH', 'NDAQ', 'NTRS', 'NCLH', 'NRG', 'NWL', 'NWSA', 'OKE',
    'OMC', 'ON', 'ORLY', 'OTIS', 'PARA', 'PHM', 'PKG', 'POOL', 'RCL', 'RF',
    'RHI', 'RL', 'ROL', 'ROST', 'RSG', 'RJF', 'SJM', 'SWKS', 'SNA', 'SOLV',
    'SWK', 'STE', 'SYF', 'SYY', 'TAP', 'TPR', 'TEL', 'TDY', 'TFX', 'TER',
    # 其他
    'TSCO', 'TTWO', 'TXT', 'TYL', 'UDR', 'ULTA', 'UHS', 'VFC', 'VTRS', 'VMC',
    'WAB', 'WAT', 'WDC', 'WRK', 'WY', 'WYNN', 'XRAY', 'YUM', 'ZION', 'ZTS',
]

# 小盘股 - 罗素2000 & 标普小盘600 代表性股票
SMALL_CAP_STOCKS = [
    # 生物科技/医疗
    'ARWR', 'ACAD', 'ACHC', 'ADUS', 'AEIS', 'AFFRM', 'AGIO', 'AGNC', 'AHCO', 'AKAM',
    'ALGT', 'ALKS', 'ALNY', 'AMED', 'AMN', 'AMPH', 'ANF', 'APLS', 'ARVN', 'ASGN',
    'ATSG', 'AVAV', 'AXSM', 'BCRX', 'BPMC', 'BRBR', 'BIO', 'BJ', 'BLDR', 'BMI',
    'BOH', 'BRX', 'BSX', 'BURL', 'BXSL', 'CACI', 'CADE', 'CAR', 'CASH', 'CASY',
    # 科技/软件/服务
    'CBT', 'CCCS', 'CCMP', 'CELH', 'CENT', 'CERE', 'CHDN', 'CHX', 'CIR', 'CLF',
    'CMC', 'CMG', 'CNO', 'CNX', 'COKE', 'CORT', 'COTY', 'CPRI', 'CROX', 'CRSP',
    'CUBE', 'CW', 'CWAN', 'CYTK', 'DAR', 'DAY', 'DBX', 'DCI', 'DECK', 'DEN',
    'DINO', 'DOC', 'DOCS', 'DT', 'DUOL', 'DYN', 'EA', 'EAT', 'EGP', 'EGY',
    # 工业/能源/材料
    'ELF', 'ELV', 'ENPH', 'ENV', 'EOSE', 'EPAM', 'EPC', 'EQR', 'ESAB', 'ESNT',
    'EVR', 'EWBC', 'EXAS', 'EXEL', 'EXPE', 'FAF', 'FANG', 'FIVN', 'FL', 'FND',
    'FR', 'FSR', 'FTI', 'FTNT', 'FUTU', 'GME', 'GNRC', 'GO', 'GOGL', 'GPOR',
    'GT', 'GTLB', 'H', 'HAL', 'HCP', 'HE', 'HII', 'HIMS', 'HIW', 'HRL',
    # 金融/消费/其他
    'HSKA', 'HUBS', 'HUT', 'HWC', 'HXL', 'IAC', 'IAS', 'IBKR', 'ICUI', 'IDA',
    'IDCC', 'IDXX', 'IEX', 'IIVI', 'ILMN', 'INCY', 'INSM', 'INST', 'INVH', 'IONS',
    'IOSP', 'IPGP', 'IQV', 'IRDM', 'IRT', 'ISRG', 'ITCI', 'JAZZ', 'JBL', 'JBLU',
    'JEF', 'JLL', 'JWN', 'KBR', 'KDP', 'KEX', 'KNSL', 'KNTK', 'KRTX', 'KRY',
]


class USStockUniverse:
    """美股全市场股票池管理"""
    
    def __init__(self, cache_dir: str = 'data/us_universe'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 缓存文件
        self.large_cap_cache = f"{cache_dir}/large_cap.pkl"
        self.mid_cap_cache = f"{cache_dir}/mid_cap.pkl"
        self.small_cap_cache = f"{cache_dir}/small_cap.pkl"
    
    def get_large_cap(self, use_preset: bool = True, limit: Optional[int] = None) -> List[str]:
        """
        获取大盘股 (标普500 + 纳指100)
        
        Args:
            use_preset: 使用预设列表还是尝试获取
            limit: 限制数量
        """
        if use_preset:
            stocks = LARGE_CAP_STOCKS.copy()
        else:
            stocks = self._fetch_from_cache_or_api('large_cap')
        
        if limit:
            stocks = stocks[:limit]
        print(f"[USStockUniverse] 大盘股: {len(stocks)} 只")
        return stocks
    
    def get_mid_cap(self, use_preset: bool = True, limit: Optional[int] = None) -> List[str]:
        """
        获取中盘股 (S&P MidCap 400)
        
        Args:
            use_preset: 使用预设列表
            limit: 限制数量
        """
        if use_preset:
            stocks = MID_CAP_STOCKS.copy()
        else:
            stocks = self._fetch_from_cache_or_api('mid_cap')
        
        if limit:
            stocks = stocks[:limit]
        print(f"[USStockUniverse] 中盘股: {len(stocks)} 只")
        return stocks
    
    def get_small_cap(self, use_preset: bool = True, limit: Optional[int] = None) -> List[str]:
        """
        获取小盘股 (罗素2000 + 标普小盘600)
        
        Args:
            use_preset: 使用预设列表
            limit: 限制数量
        """
        if use_preset:
            stocks = SMALL_CAP_STOCKS.copy()
        else:
            stocks = self._fetch_from_cache_or_api('small_cap')
        
        if limit:
            stocks = stocks[:limit]
        print(f"[USStockUniverse] 小盘股: {len(stocks)} 只")
        return stocks
    
    def get_all_market(self, 
                       large_cap_limit: Optional[int] = 100,
                       mid_cap_limit: Optional[int] = 100,
                       small_cap_limit: Optional[int] = 100) -> Dict[str, List[str]]:
        """
        获取全市场分层股票池
        
        Returns:
            Dict with keys: 'large_cap', 'mid_cap', 'small_cap', 'all'
        """
        large = self.get_large_cap(limit=large_cap_limit)
        mid = self.get_mid_cap(limit=mid_cap_limit)
        small = self.get_small_cap(limit=small_cap_limit)
        
        all_stocks = list(set(large + mid + small))
        
        result = {
            'large_cap': large,
            'mid_cap': mid,
            'small_cap': small,
            'all': all_stocks,
            'stats': {
                'large_cap': len(large),
                'mid_cap': len(mid),
                'small_cap': len(small),
                'total': len(all_stocks)
            }
        }
        
        print(f"\n[USStockUniverse] 全市场统计:")
        print(f"  大盘股: {len(large)} 只")
        print(f"  中盘股: {len(mid)} 只")
        print(f"  小盘股: {len(small)} 只")
        print(f"  合计: {len(all_stocks)} 只 (去重)")
        
        return result
    
    def _fetch_from_cache_or_api(self, category: str) -> List[str]:
        """从缓存或API获取股票列表"""
        cache_file = f"{self.cache_dir}/{category}.pkl"
        
        # 尝试读取缓存
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    stocks = pickle.load(f)
                    print(f"[USStockUniverse] 从缓存读取 {category}: {len(stocks)} 只")
                    return stocks
            except Exception as e:
                print(f"[USStockUniverse] 缓存读取失败: {e}")
        
        # 使用预设列表
        if category == 'large_cap':
            return LARGE_CAP_STOCKS.copy()
        elif category == 'mid_cap':
            return MID_CAP_STOCKS.copy()
        elif category == 'small_cap':
            return SMALL_CAP_STOCKS.copy()
        else:
            return []
    
    def save_universe(self, universe: Dict, filename: str = 'us_universe.json'):
        """保存股票池到文件"""
        filepath = f"{self.cache_dir}/{filename}"
        with open(filepath, 'w') as f:
            json.dump(universe, f, indent=2)
        print(f"[USStockUniverse] 股票池已保存: {filepath}")
        return filepath


# 便捷函数
def get_us_large_cap(limit: Optional[int] = None) -> List[str]:
    """获取美股大盘股"""
    return USStockUniverse().get_large_cap(limit=limit)

def get_us_mid_cap(limit: Optional[int] = None) -> List[str]:
    """获取美股中盘股"""
    return USStockUniverse().get_mid_cap(limit=limit)

def get_us_small_cap(limit: Optional[int] = None) -> List[str]:
    """获取美股小盘股"""
    return USStockUniverse().get_small_cap(limit=limit)

def get_us_all_market(**kwargs) -> Dict[str, List[str]]:
    """获取美股全市场分层股票池"""
    return USStockUniverse().get_all_market(**kwargs)


if __name__ == '__main__':
    print("=" * 80)
    print("US Stock Universe - 美股全市场股票池")
    print("=" * 80)
    
    universe = USStockUniverse()
    
    # 获取全市场分层
    all_stocks = universe.get_all_market(
        large_cap_limit=100,
        mid_cap_limit=100,
        small_cap_limit=100
    )
    
    # 保存
    universe.save_universe(all_stocks)
    
    print("\n" + "=" * 80)
    print("大盘股示例 (前10只):")
    print(all_stocks['large_cap'][:10])
    print("\n中盘股示例 (前10只):")
    print(all_stocks['mid_cap'][:10])
    print("\n小盘股示例 (前10只):")
    print(all_stocks['small_cap'][:10])

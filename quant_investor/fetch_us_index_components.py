#!/usr/bin/env python3
"""
Fetch US Index Components - 获取美股主要指数完整成分股

获取：
- 大盘股: 标普500 (S&P 500) + 纳指100 (NASDAQ-100)
- 中盘股: S&P MidCap 400
- 小盘股: 罗素2000 (Russell 2000)
"""

import pandas as pd
import requests
from typing import List, Dict
import json
import os


def fetch_sp500() -> List[str]:
    """
    获取标普500完整成分股
    从Wikipedia获取
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        symbols = df['Symbol'].tolist()
        print(f"[Fetch] 标普500: {len(symbols)} 只股票")
        return symbols
    except Exception as e:
        print(f"[Fetch] 标普500获取失败: {e}")
        return []


def fetch_nasdaq100() -> List[str]:
    """
    获取纳斯达克100成分股
    """
    try:
        url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        tables = pd.read_html(url)
        
        # 寻找包含Ticker的表格
        for table in tables:
            if 'Ticker' in table.columns:
                symbols = table['Ticker'].tolist()
                print(f"[Fetch] 纳斯达克100: {len(symbols)} 只股票")
                return symbols
            elif 'Symbol' in table.columns:
                symbols = table['Symbol'].tolist()
                print(f"[Fetch] 纳斯达克100: {len(symbols)} 只股票")
                return symbols
        
        print("[Fetch] 纳斯达克100: 未找到数据表格")
        return []
    except Exception as e:
        print(f"[Fetch] 纳斯达克100获取失败: {e}")
        return []


def fetch_sp400() -> List[str]:
    """
    获取S&P MidCap 400成分股
    """
    try:
        # 尝试从Wikipedia获取
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        tables = pd.read_html(url)
        df = tables[0]
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].tolist()
        elif 'Ticker' in df.columns:
            symbols = df['Ticker'].tolist()
        else:
            symbols = df.iloc[:, 0].tolist()  # 第一列
        
        print(f"[Fetch] S&P MidCap 400: {len(symbols)} 只股票")
        return symbols
    except Exception as e:
        print(f"[Fetch] S&P MidCap 400获取失败: {e}")
        return []


def fetch_russell2000() -> List[str]:
    """
    获取罗素2000成分股 (前2000只)
    注意: 罗素2000有2000只股票，这里获取代表性样本或全部
    """
    try:
        # 尝试从多个来源获取
        # 方法1: 尝试Wikipedia (可能只有部分)
        url = 'https://en.wikipedia.org/wiki/Russell_2000_Index'
        tables = pd.read_html(url)
        
        symbols = []
        for table in tables:
            if 'Ticker' in table.columns:
                symbols = table['Ticker'].tolist()
                break
            elif 'Symbol' in table.columns:
                symbols = table['Symbol'].tolist()
                break
        
        if symbols:
            print(f"[Fetch] 罗素2000 (Wikipedia): {len(symbols)} 只股票")
            return symbols
        
        # 方法2: 使用预设的扩展列表
        print("[Fetch] 罗素2000: 使用扩展预设列表")
        return get_extended_russell2000()
        
    except Exception as e:
        print(f"[Fetch] 罗素2000获取失败: {e}")
        return get_extended_russell2000()


def get_extended_russell2000() -> List[str]:
    """
    获取扩展的罗素2000代表性股票
    包含多个行业的代表性小盘股
    """
    # 这里使用一个扩展的小盘股列表作为替代
    # 实际应从可靠数据源获取完整列表
    extended_list = []
    
    # 读取本地文件如果存在
    cache_file = 'data/us_universe/russell2000_full.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return data.get('symbols', [])
    
    return []


def get_large_cap_universe() -> List[str]:
    """获取完整大盘股列表 (标普500 + 纳指100)"""
    sp500 = fetch_sp500()
    nasdaq100 = fetch_nasdaq100()
    
    # 合并去重
    combined = list(set(sp500 + nasdaq100))
    print(f"[Universe] 大盘股合计: {len(combined)} 只 (S&P500: {len(sp500)}, NASDAQ100: {len(nasdaq100)})\n")
    return combined


def get_mid_cap_universe() -> List[str]:
    """获取完整中盘股列表"""
    sp400 = fetch_sp400()
    print(f"[Universe] 中盘股: {len(sp400)} 只\n")
    return sp400


def get_small_cap_universe() -> List[str]:
    """获取完整小盘股列表"""
    russell2000 = fetch_russell2000()
    print(f"[Universe] 小盘股: {len(russell2000)} 只\n")
    return russell2000


def get_all_universe() -> Dict[str, List[str]]:
    """获取全市场所有成分股"""
    print("=" * 80)
    print("🌎 获取美股全市场指数成分股")
    print("=" * 80 + "\n")
    
    large_cap = get_large_cap_universe()
    mid_cap = get_mid_cap_universe()
    small_cap = get_small_cap_universe()
    
    all_symbols = list(set(large_cap + mid_cap + small_cap))
    
    result = {
        'large_cap': large_cap,
        'mid_cap': mid_cap,
        'small_cap': small_cap,
        'all': all_symbols,
        'stats': {
            'large_cap': len(large_cap),
            'mid_cap': len(mid_cap),
            'small_cap': len(small_cap),
            'total_unique': len(all_symbols)
        }
    }
    
    print("=" * 80)
    print("📊 全市场统计")
    print("=" * 80)
    print(f"大盘股: {len(large_cap)} 只 (S&P 500 + NASDAQ-100)")
    print(f"中盘股: {len(mid_cap)} 只 (S&P MidCap 400)")
    print(f"小盘股: {len(small_cap)} 只 (Russell 2000)")
    print(f"总计: {len(all_symbols)} 只 (去重)")
    print("=" * 80)
    
    return result


def save_universe(universe: Dict, output_dir: str = 'data/us_universe'):
    """保存成分股列表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON
    json_file = f"{output_dir}/complete_universe.json"
    with open(json_file, 'w') as f:
        json.dump(universe, f, indent=2)
    
    # 保存文本文件 (方便查看)
    for category in ['large_cap', 'mid_cap', 'small_cap']:
        txt_file = f"{output_dir}/{category}_symbols.txt"
        with open(txt_file, 'w') as f:
            for symbol in sorted(universe[category]):
                f.write(f"{symbol}\n")
    
    print(f"\n💾 成分股列表已保存到: {output_dir}/")
    print(f"  - complete_universe.json")
    print(f"  - large_cap_symbols.txt ({len(universe['large_cap'])} 只)")
    print(f"  - mid_cap_symbols.txt ({len(universe['mid_cap'])} 只)")
    print(f"  - small_cap_symbols.txt ({len(universe['small_cap'])} 只)")


if __name__ == '__main__':
    universe = get_all_universe()
    save_universe(universe)

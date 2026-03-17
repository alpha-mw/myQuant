#!/usr/bin/env python3
"""
Fetch China A-Share Index Components - 获取A股指数成分股

获取：
- 大盘股: 沪深300 (CSI 300) - 300只
- 中盘股: 中证500 (CSI 500) - 500只
- 小盘股: 中证1000 (CSI 1000) - 1000只
"""

import os
import sys
import json
from typing import List, Dict
from datetime import datetime

# 添加路径
sys.path.insert(0, str(os.path.dirname(__file__)))
from config import config
from credential_utils import create_tushare_pro

import tushare as ts

# Tushare配置
TUSHARE_TOKEN = config.TUSHARE_TOKEN
TUSHARE_URL = os.environ.get('TUSHARE_URL', 'http://lianghua.nanyangqiankun.top')


def init_tushare():
    """初始化Tushare API"""
    pro = create_tushare_pro(ts, TUSHARE_TOKEN, TUSHARE_URL)
    if pro is None:
        raise RuntimeError("TUSHARE_TOKEN 未设置，无法初始化 Tushare API")
    return pro


def fetch_hs300(pro) -> List[str]:
    """
    获取沪深300成分股
    代码: 000300.SH
    """
    try:
        df = pro.index_weight(index_code='000300.SH')
        if df is not None and not df.empty:
            # 获取最新的成分股权重
            df = df.sort_values('trade_date', ascending=False)
            latest_date = df['trade_date'].iloc[0]
            latest_df = df[df['trade_date'] == latest_date]
            symbols = latest_df['con_code'].tolist()
            print(f"✅ 沪深300: {len(symbols)} 只股票 (日期: {latest_date})")
            return symbols
    except Exception as e:
        print(f"❌ 沪深300获取失败: {e}")
    return []


def fetch_zz500(pro) -> List[str]:
    """
    获取中证500成分股
    代码: 000905.SH
    """
    try:
        df = pro.index_weight(index_code='000905.SH')
        if df is not None and not df.empty:
            df = df.sort_values('trade_date', ascending=False)
            latest_date = df['trade_date'].iloc[0]
            latest_df = df[df['trade_date'] == latest_date]
            symbols = latest_df['con_code'].tolist()
            print(f"✅ 中证500: {len(symbols)} 只股票 (日期: {latest_date})")
            return symbols
    except Exception as e:
        print(f"❌ 中证500获取失败: {e}")
    return []


def fetch_zz1000(pro) -> List[str]:
    """
    获取中证1000成分股
    代码: 000852.SH
    """
    try:
        df = pro.index_weight(index_code='000852.SH')
        if df is not None and not df.empty:
            df = df.sort_values('trade_date', ascending=False)
            latest_date = df['trade_date'].iloc[0]
            latest_df = df[df['trade_date'] == latest_date]
            symbols = latest_df['con_code'].tolist()
            print(f"✅ 中证1000: {len(symbols)} 只股票 (日期: {latest_date})")
            return symbols
    except Exception as e:
        print(f"❌ 中证1000获取失败: {e}")
    return []


def get_all_components(pro=None) -> Dict[str, List[str]]:
    """获取全部A股指数成分股"""
    print("=" * 80)
    print("🇨🇳 获取A股指数成分股")
    print("=" * 80)
    print()
    
    if pro is None:
        pro = init_tushare()
        print("✅ Tushare API初始化成功\n")
    
    hs300 = fetch_hs300(pro)
    zz500 = fetch_zz500(pro)
    zz1000 = fetch_zz1000(pro)
    
    # 合并去重
    all_symbols = list(set(hs300 + zz500 + zz1000))
    
    result = {
        'hs300': hs300,
        'zz500': zz500,
        'zz1000': zz1000,
        'all': all_symbols,
        'stats': {
            'hs300': len(hs300),
            'zz500': len(zz500),
            'zz1000': len(zz1000),
            'total_unique': len(all_symbols)
        },
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print()
    print("=" * 80)
    print("📊 A股指数成分股统计")
    print("=" * 80)
    print(f"沪深300:   {len(hs300):4d} 只 (大盘股)")
    print(f"中证500:   {len(zz500):4d} 只 (中盘股)")
    print(f"中证1000:  {len(zz1000):4d} 只 (小盘股)")
    print("-" * 80)
    print(f"总计:      {len(all_symbols):4d} 只 (去重)")
    print("=" * 80)
    
    return result


def save_components(components: Dict, output_dir: str = 'data/cn_universe'):
    """保存成分股列表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON
    json_file = f"{output_dir}/cn_index_components.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(components, f, indent=2, ensure_ascii=False)
    
    # 保存文本文件
    for category in ['hs300', 'zz500', 'zz1000']:
        txt_file = f"{output_dir}/{category}_symbols.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for symbol in sorted(components[category]):
                f.write(f"{symbol}\n")
    
    print()
    print("💾 成分股列表已保存:")
    print(f"  {output_dir}/cn_index_components.json")
    print(f"  {output_dir}/hs300_symbols.txt    ({len(components['hs300'])} 只)")
    print(f"  {output_dir}/zz500_symbols.txt    ({len(components['zz500'])} 只)")
    print(f"  {output_dir}/zz1000_symbols.txt   ({len(components['zz1000'])} 只)")


def fetch_stock_basic_info(pro, symbols: List[str]) -> Dict:
    """
    获取股票基本信息
    行业、市值等
    """
    try:
        # 分批获取，避免请求过大
        batch_size = 100
        all_info = []
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            codes = ','.join(batch)
            df = pro.stock_basic(ts_code=codes, fields='ts_code,name,industry,market,list_date')
            if df is not None and not df.empty:
                all_info.append(df)
        
        if all_info:
            result_df = pd.concat(all_info, ignore_index=True)
            return result_df.to_dict('records')
    except Exception as e:
        print(f"获取基本信息失败: {e}")
    return {}


if __name__ == '__main__':
    import pandas as pd
    
    print("=" * 80)
    print("中国A股指数成分股获取工具")
    print("=" * 80)
    print()
    
    # 获取成分股
    pro = init_tushare()
    print("✅ Tushare API初始化成功\n")
    
    components = get_all_components(pro)
    save_components(components)
    
    print("\n✅ 完成! 现在可以运行A股数据下载脚本了。")

#!/usr/bin/env python3
"""
Fetch China A-Share Index Components - 获取A股指数成分股

获取：
- 大盘股: 沪深300 (CSI 300) - 300只
- 中盘股: 中证500 (CSI 500) - 500只
- 小盘股: 中证1000 (CSI 1000) - 1000只
"""

import os
import json
from typing import List, Dict
from datetime import datetime

from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro
from quant_investor.market.cn_resolver import CNUniverseResolver
from quant_investor.market.config import get_market_settings

try:  # pragma: no cover - optional dependency
    import tushare as ts  # type: ignore
except Exception:  # pragma: no cover - fallback when tushare is unavailable
    ts = None  # type: ignore[assignment]

# Tushare配置
TUSHARE_TOKEN = config.TUSHARE_TOKEN
TUSHARE_URL = os.environ.get('TUSHARE_URL', 'http://lianghua.nanyangqiankun.top')


def init_tushare():
    """初始化Tushare API"""
    if ts is None:
        raise RuntimeError("tushare 未安装，无法初始化 Tushare API")
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


def fetch_full_a(pro) -> List[str]:
    """
    获取全 A 股可交易股票池。

    以当前可上市股票为准，作为 CN 默认 canonical universe。
    """
    try:
        df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name")
        if df is not None and not df.empty:
            symbols = sorted(
                str(value).strip()
                for value in df["ts_code"].dropna().astype(str).tolist()
                if str(value).strip()
            )
            print(f"✅ 全A股: {len(symbols)} 只股票")
            return symbols
    except Exception as e:
        print(f"❌ 全A股获取失败: {e}")
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
    
    full_a = fetch_full_a(pro)
    hs300 = fetch_hs300(pro)
    zz500 = fetch_zz500(pro)
    zz1000 = fetch_zz1000(pro)

    resolver = CNUniverseResolver(data_dir=get_market_settings("CN").data_dir)
    resolver.trace.physical_directories_used_for_full_a = [
        str(path) for path in resolver.physical_directories_for_full_a()
    ]

    if not full_a:
        local_full_a, _ = resolver.collect_full_a_inventory(local_union_fallback_used=True)
        if local_full_a:
            print(f"⚠️ Tushare 全A股结果为空，回退到本地 CSV union: {len(local_full_a)} 只")
            full_a = local_full_a

    # 合并去重
    all_symbols = list(dict.fromkeys(full_a or (hs300 + zz500 + zz1000)))
    if not all_symbols:
        all_symbols, _ = resolver.collect_full_a_inventory(local_union_fallback_used=True)
    if not full_a and all_symbols:
        print(f"⚠️ 使用指数篮子 union 作为 full_a canonical universe: {len(all_symbols)} 只")
        full_a = all_symbols
        resolver.trace.local_union_fallback_used = True
        resolver.trace.resolution_strategy = "index_bucket_union"
    if full_a and not resolver.trace.resolution_strategy:
        resolver.trace.resolution_strategy = "upstream_fetch"
    
    result = {
        'full_a': full_a,
        'full_market': full_a,
        'all_a': full_a,
        'all': all_symbols,
        'hs300': hs300,
        'zz500': zz500,
        'zz1000': zz1000,
        'stats': {
            'full_a': len(full_a),
            'hs300': len(hs300),
            'zz500': len(zz500),
            'zz1000': len(zz1000),
            'total_unique': len(all_symbols)
        },
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'resolver': resolver.snapshot(),
    }
    
    print()
    print("=" * 80)
    print("📊 A股指数成分股统计")
    print("=" * 80)
    print(f"全A股:     {len(full_a):4d} 只")
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
    for category in ['full_a', 'hs300', 'zz500', 'zz1000']:
        txt_file = f"{output_dir}/{category}_symbols.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for symbol in sorted(components[category]):
                f.write(f"{symbol}\n")
    
    print()
    print("💾 成分股列表已保存:")
    print(f"  {output_dir}/cn_index_components.json")
    print(f"  {output_dir}/full_a_symbols.txt    ({len(components['full_a'])} 只)")
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

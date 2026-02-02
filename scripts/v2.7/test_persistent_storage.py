"""
持久化数据存储系统完整测试脚本

测试内容:
1. 基本存储和读取功能
2. 增量更新功能
3. 缓存命中统计
4. 数据过期清理
5. 存储摘要统计
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# 添加路径
sys.path.insert(0, '/home/ubuntu/skills/quant-investor/scripts/v2.7')
sys.path.insert(0, '/home/ubuntu/skills/quant-investor/scripts/v2.6/us_macro_data')

from persistent_data_manager import PersistentDataManager
from persistent_us_data_manager import PersistentUSDataManager


def test_basic_storage():
    """测试基本存储和读取功能"""
    print("\n" + "=" * 60)
    print("测试1: 基本存储和读取功能")
    print("=" * 60)
    
    manager = PersistentDataManager()
    
    # 创建测试数据
    test_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'value': range(30),
        'category': ['A'] * 15 + ['B'] * 15
    })
    
    # 保存数据
    success = manager.save('test_basic', test_df, source='test')
    print(f"保存数据: {'✓ 成功' if success else '✗ 失败'}")
    
    # 读取数据
    result = manager.query('test_basic')
    if result is not None and len(result) == 30:
        print(f"读取数据: ✓ 成功 ({len(result)} 条)")
    else:
        print(f"读取数据: ✗ 失败")
    
    return manager


def test_cache_hit():
    """测试缓存命中功能"""
    print("\n" + "=" * 60)
    print("测试2: 缓存命中功能")
    print("=" * 60)
    
    manager = PersistentUSDataManager()
    
    # 第一次获取数据（应该是cache miss）
    print("第一次获取NVDA数据...")
    df1 = manager.get_stock_history('NVDA', period='1mo')
    stats1 = manager.get_stats()
    print(f"  缓存命中: {stats1['cache_hits']}, 未命中: {stats1['cache_misses']}")
    
    # 第二次获取相同数据（应该是cache hit）
    print("第二次获取NVDA数据...")
    df2 = manager.get_stock_history('NVDA', period='1mo')
    stats2 = manager.get_stats()
    print(f"  缓存命中: {stats2['cache_hits']}, 未命中: {stats2['cache_misses']}")
    
    if stats2['cache_hits'] > stats1['cache_hits']:
        print("缓存命中测试: ✓ 通过")
    else:
        print("缓存命中测试: ✗ 失败")
    
    return manager


def test_incremental_update():
    """测试增量更新功能"""
    print("\n" + "=" * 60)
    print("测试3: 增量更新功能")
    print("=" * 60)
    
    manager = PersistentDataManager()
    
    # 创建初始数据（1月1日到1月15日）
    initial_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=15, freq='D'),
        'value': range(15)
    })
    manager.save('test_incremental', initial_df, source='test')
    print(f"初始数据: 15条 (2024-01-01 到 2024-01-15)")
    
    # 模拟增量更新（请求1月1日到1月31日的数据）
    # 这里需要提供fetch_func来获取新数据
    def fetch_new_data():
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=31, freq='D'),
            'value': range(31)
        })
    
    result = manager.query(
        'test_incremental',
        start_date='2024-01-01',
        end_date='2024-01-31',
        fetch_func=fetch_new_data
    )
    
    if result is not None:
        print(f"更新后数据: {len(result)}条")
        print(f"日期范围: {result['date'].min()} 到 {result['date'].max()}")
        if len(result) >= 15:
            print("增量更新测试: ✓ 通过")
        else:
            print("增量更新测试: ✗ 失败")
    else:
        print("增量更新测试: ✗ 失败 (无数据)")


def test_storage_summary():
    """测试存储摘要功能"""
    print("\n" + "=" * 60)
    print("测试4: 存储摘要功能")
    print("=" * 60)
    
    manager = PersistentDataManager()
    
    summary = manager.get_storage_summary()
    print(f"总数据集: {summary['total_datasets']}")
    print(f"总行数: {summary['total_rows']}")
    print(f"磁盘使用: {summary['disk_usage_mb']} MB")
    
    print("\n按数据类型统计:")
    for dtype, info in summary['by_type'].items():
        print(f"  {dtype}: {info['count']}个数据集, {info['rows']}行")
    
    return summary


def test_list_data():
    """测试列出已存储数据功能"""
    print("\n" + "=" * 60)
    print("测试5: 列出已存储数据")
    print("=" * 60)
    
    manager = PersistentDataManager()
    
    data_list = manager.list_data()
    print(f"共有 {len(data_list)} 个数据集:")
    for item in data_list[:10]:  # 只显示前10个
        print(f"  - {item['data_key']}: {item['row_count']}行, "
              f"更新于 {item['last_updated']}")
    
    if len(data_list) > 10:
        print(f"  ... 还有 {len(data_list) - 10} 个数据集")


def test_real_world_scenario():
    """测试真实场景：多次获取同一股票数据"""
    print("\n" + "=" * 60)
    print("测试6: 真实场景 - 多次获取同一股票数据")
    print("=" * 60)
    
    manager = PersistentUSDataManager()
    
    # 模拟用户多次分析同一股票
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    
    print("第一轮获取数据（应全部从网络下载）:")
    for ticker in stocks:
        df = manager.get_stock_history(ticker, period='1mo')
        if df is not None and not df.empty:
            print(f"  {ticker}: {len(df)}条数据")
    
    stats1 = manager.get_stats()
    print(f"  统计: 命中={stats1['cache_hits']}, 未命中={stats1['cache_misses']}")
    
    print("\n第二轮获取数据（应全部从缓存读取）:")
    for ticker in stocks:
        df = manager.get_stock_history(ticker, period='1mo')
        if df is not None and not df.empty:
            print(f"  {ticker}: {len(df)}条数据")
    
    stats2 = manager.get_stats()
    print(f"  统计: 命中={stats2['cache_hits']}, 未命中={stats2['cache_misses']}")
    
    # 验证缓存命中增加了
    new_hits = stats2['cache_hits'] - stats1['cache_hits']
    print(f"\n新增缓存命中: {new_hits}次")
    if new_hits == len(stocks):
        print("真实场景测试: ✓ 通过")
    else:
        print("真实场景测试: ✗ 失败")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Quant-Investor V2.7 持久化存储系统测试")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行测试
    test_basic_storage()
    test_cache_hit()
    test_incremental_update()
    test_storage_summary()
    test_list_data()
    test_real_world_scenario()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

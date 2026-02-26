#!/usr/bin/env python3
"""
后台数据下载脚本

持续下载所有股票数据，支持断点续传
用法: nohup python3 download_all.py > download.log 2>&1 &
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_database import StockDatabase


def download_all_with_resume():
    """
    持续下载所有数据，支持断点续传
    """
    db = StockDatabase()
    
    # 配置
    start_date = '20200101'
    end_date = datetime.now().strftime('%Y%m%d')
    batch_size = 100  # 每批100只
    max_workers = 5
    
    print(f"[{datetime.now()}] 开始下载所有股票数据")
    print(f"  日期范围: {start_date} 至 {end_date}")
    print(f"  批次大小: {batch_size}")
    print(f"  并行线程: {max_workers}")
    print()
    
    total_batches = 0
    total_success = 0
    total_failed = 0
    
    while True:
        # 获取需要下载的股票
        stocks = db.get_stocks_to_download(start_date, end_date)
        
        if not stocks:
            print(f"[{datetime.now()}] ✅ 所有股票数据下载完成！")
            break
        
        print(f"[{datetime.now()}] 还有 {len(stocks)} 只股票需要下载")
        
        # 下载一批
        progress = db.batch_download(
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        total_batches += 1
        total_success += progress.completed_stocks
        total_failed += len(progress.failed_stocks)
        
        print(f"[{datetime.now()}] 批次 {total_batches} 完成: "
              f"成功 {progress.completed_stocks}, 失败 {len(progress.failed_stocks)}")
        print(f"  累计: 成功 {total_success}, 失败 {total_failed}")
        print()
        
        # 每10批暂停一下，避免rate limit
        if total_batches % 10 == 0:
            print(f"[{datetime.now()}] 暂停60秒...")
            time.sleep(60)
        else:
            # 批次间短暂暂停
            time.sleep(5)
    
    # 最终统计
    stats = db.get_statistics()
    print()
    print("=" * 60)
    print("下载完成统计")
    print("=" * 60)
    print(f"总批次: {total_batches}")
    print(f"成功: {total_success}")
    print(f"失败: {total_failed}")
    print(f"数据库记录: {stats.get('total_records', 0):,} 条")
    print(f"有数据股票: {stats.get('stocks_with_data', 0)} 只")
    print("=" * 60)


if __name__ == '__main__':
    try:
        download_all_with_resume()
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] 用户中断，已保存进度")
        print("可以重新运行脚本继续下载")
    except Exception as e:
        print(f"\n[{datetime.now()}] 错误: {e}")
        print("可以重新运行脚本继续下载")

#!/usr/bin/env python3
"""
数据下载管理脚本

用法:
    python data_manager.py init              # 初始化数据库
    python data_manager.py update_list       # 更新股票列表
    python data_manager.py download          # 下载数据 (默认100只)
    python data_manager.py download --all    # 下载所有股票
    python data_manager.py stats             # 查看统计
    python data_manager.py query --code 000001.SZ  # 查询数据
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_database import StockDatabase, download_all_data


def cmd_init(args):
    """初始化数据库"""
    print("初始化数据库...")
    db = StockDatabase()
    db.update_stock_list()
    print("✅ 数据库初始化完成")
    
    # 显示统计
    cmd_stats(args)


def cmd_update_list(args):
    """更新股票列表"""
    print("更新股票列表...")
    db = StockDatabase()
    count = db.update_stock_list()
    print(f"✅ 更新了 {count} 只股票")


def cmd_download(args):
    """下载数据"""
    db = StockDatabase()
    
    # 确定批次大小
    batch_size = args.batch_size if args.batch_size else (1800 if args.all else 100)
    
    print(f"开始下载数据...")
    print(f"  市场范围: {args.market}")
    print(f"  日期范围: {args.start_date} 至 {args.end_date or '今天'}")
    print(f"  批次大小: {batch_size} 只")
    print(f"  并行线程: {args.workers}")
    print()
    
    progress = db.batch_download(
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.workers,
        batch_size=batch_size,
        market=args.market,
    )
    
    print()
    print(f"下载完成:")
    print(f"  成功: {progress.completed_stocks} 只")
    print(f"  失败: {len(progress.failed_stocks)} 只")
    
    if progress.failed_stocks:
        print(f"  失败列表: {progress.failed_stocks[:10]}...")  # 只显示前10个


def cmd_backfill(args):
    """向前回填历史数据"""
    db = StockDatabase()
    batch_size = None if args.all else args.batch_size

    plan = db.plan_historical_backfill(
        years=args.years,
        anchor_start=args.anchor_start,
        batch_size=batch_size,
        market=args.market,
    )

    print("开始历史回填...")
    print(f"  市场范围: {args.market}")
    print(f"  当前最早日期: {plan.anchor_start}")
    print(f"  当前最新日期: {plan.anchor_end}")
    print(f"  目标起点: {plan.target_start}")
    print(f"  回填年数: {plan.years}")
    print(f"  任务数量: {len(plan.tasks)}")
    print(f"  覆盖股票: {plan.stock_count} 只")
    print(f"  并行线程: {args.workers}")
    print()

    if not plan.tasks:
        print("✅ 当前数据已经覆盖目标历史区间，无需回填")
        return

    _, progress = db.backfill_history(
        years=args.years,
        max_workers=args.workers,
        batch_size=batch_size,
        anchor_start=args.anchor_start,
        market=args.market,
    )

    print()
    print("回填完成:")
    print(f"  成功: {progress.completed_stocks} 个任务")
    print(f"  失败: {len(progress.failed_stocks)} 个任务")

    if progress.failed_stocks:
        print(f"  失败列表: {progress.failed_stocks[:10]}...")


def cmd_stats(args):
    """查看统计"""
    db = StockDatabase()
    stats = db.get_statistics()
    
    print("=" * 60)
    print("数据库统计")
    print("=" * 60)
    print(f"股票总数: {stats.get('total_stocks', 0)}")
    print(f"  - 沪深300: {stats.get('hs300_count', 0)}")
    print(f"  - 中证500: {stats.get('zz500_count', 0)}")
    print(f"  - 中证1000: {stats.get('zz1000_count', 0)}")
    print()
    print(f"数据记录: {stats.get('total_records', 0):,} 条")
    print(f"有数据股票: {stats.get('stocks_with_data', 0)} 只")
    print(f"日期范围: {stats.get('date_range', 'N/A')}")
    if stats.get('price_config'):
        print()
        print("价格口径:")
        for market, config in stats["price_config"].items():
            print(
                f"  - {market}: {config.get('price_mode')} / "
                f"volume={config.get('volume_mode')} / source={config.get('data_source')}"
            )
    print("=" * 60)


def cmd_standardize_prices(args):
    """统一价格口径为可回测序列"""
    db = StockDatabase()
    batch_size = None if args.all else args.batch_size

    tasks = db.plan_price_standardization(
        market=args.market,
        batch_size=batch_size,
    )

    print("开始价格标准化...")
    print(f"  市场范围: {args.market}")
    print("  目标口径: 前复权/adjusted OHLC")
    print("  volume/amount: 保持原始成交口径")
    print(f"  需要重建任务: {len(tasks)}")
    print(f"  并行线程: {args.workers}")
    print()

    progress = db.standardize_price_series(
        market=args.market,
        max_workers=args.workers,
        batch_size=batch_size,
    )

    print()
    print("价格标准化完成:")
    print(f"  成功: {progress.completed_stocks} 个任务")
    print(f"  失败: {len(progress.failed_stocks)} 个任务")

    if progress.failed_stocks:
        print(f"  失败列表: {progress.failed_stocks[:10]}...")


def cmd_query(args):
    """查询数据"""
    db = StockDatabase()
    
    ts_codes = args.code.split(',') if args.code else None
    
    df = db.get_data(
        ts_codes=ts_codes,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    print(f"查询结果: {len(df)} 条记录")
    
    if not df.empty:
        print(df.head(20))
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n已保存到: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='股票数据管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s init                           # 初始化数据库
  %(prog)s download                       # 下载100只股票数据
  %(prog)s download --all --market US     # 下载全部美股区间数据
  %(prog)s backfill --years 7 --all       # 基于现有数据向前补7年
  %(prog)s backfill --market US --all     # 仅回填美股
  %(prog)s standardize_prices --market CN --all   # 把A股统一成回测价
  %(prog)s stats                          # 查看统计
  %(prog)s query --code 000001.SZ         # 查询单只股票
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # init
    init_parser = subparsers.add_parser('init', help='初始化数据库')
    
    # update_list
    subparsers.add_parser('update_list', help='更新股票列表')
    
    # download
    download_parser = subparsers.add_parser('download', help='下载数据')
    download_parser.add_argument('--start', dest='start_date', default='20200101',
                                help='开始日期 (默认: 20200101)')
    download_parser.add_argument('--end', dest='end_date', default=None,
                                help='结束日期 (默认: 今天)')
    download_parser.add_argument('--batch-size', type=int, default=None,
                                help='每批下载数量 (默认: 100)')
    download_parser.add_argument('--workers', type=int, default=5,
                                help='并行线程数 (默认: 5)')
    download_parser.add_argument('--all', action='store_true',
                                help='下载当前市场范围内的全部股票')
    download_parser.add_argument('--market', default='ALL', choices=['ALL', 'CN', 'US'],
                                help='市场范围 (默认: ALL)')

    # backfill
    backfill_parser = subparsers.add_parser('backfill', help='向前回填历史数据')
    backfill_parser.add_argument('--years', type=int, default=7,
                                 help='向前回填的年数 (默认: 7)')
    backfill_parser.add_argument('--anchor-start', dest='anchor_start', default=None,
                                 help='回填锚点起始日，默认使用库内当前最早交易日')
    backfill_parser.add_argument('--batch-size', type=int, default=None,
                                 help='仅处理前 N 个回填任务，便于小批验证')
    backfill_parser.add_argument('--workers', type=int, default=1,
                                 help='并行线程数 (默认: 1，优先保证一致性)')
    backfill_parser.add_argument('--all', action='store_true',
                                 help='回填全部需要补齐的股票')
    backfill_parser.add_argument('--market', default='ALL', choices=['ALL', 'CN', 'US'],
                                 help='市场范围 (默认: ALL)')

    # standardize_prices
    standardize_parser = subparsers.add_parser('standardize_prices', help='统一价格口径为可回测序列')
    standardize_parser.add_argument('--batch-size', type=int, default=None,
                                    help='仅处理前 N 个标准化任务，便于小批验证')
    standardize_parser.add_argument('--workers', type=int, default=1,
                                    help='并行线程数 (默认: 1)')
    standardize_parser.add_argument('--all', action='store_true',
                                    help='处理当前市场范围内全部需要重建的股票')
    standardize_parser.add_argument('--market', default='ALL', choices=['ALL', 'CN', 'US'],
                                    help='市场范围 (默认: ALL)')
    
    # stats
    subparsers.add_parser('stats', help='查看统计')
    
    # query
    query_parser = subparsers.add_parser('query', help='查询数据')
    query_parser.add_argument('--code', required=True,
                             help='股票代码，多个用逗号分隔')
    query_parser.add_argument('--start', dest='start_date', default=None,
                             help='开始日期')
    query_parser.add_argument('--end', dest='end_date', default=None,
                             help='结束日期')
    query_parser.add_argument('--output', default=None,
                             help='输出CSV文件')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行命令
    commands = {
        'init': cmd_init,
        'update_list': cmd_update_list,
        'download': cmd_download,
        'backfill': cmd_backfill,
        'standardize_prices': cmd_standardize_prices,
        'stats': cmd_stats,
        'query': cmd_query,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        print(f"未知命令: {args.command}")
        parser.print_help()


if __name__ == '__main__':
    main()

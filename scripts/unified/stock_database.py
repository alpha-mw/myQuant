#!/usr/bin/env python3
"""
Stock Database Manager - 股票数据库管理器

功能:
1. 支持1800+只股票 (HS300+ZZ500+ZZ1000)
2. 本地SQLite数据库存储
3. 分批下载，断点续传
4. 增量更新，只下载新数据
5. 自动去重和数据校验
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import pickle
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stock_universe import StockUniverse


@dataclass
class DownloadProgress:
    """下载进度"""
    total_stocks: int
    completed_stocks: int
    failed_stocks: List[str]
    last_update: datetime

    @property
    def progress_pct(self) -> float:
        if self.total_stocks == 0:
            return 0.0
        return (self.completed_stocks / self.total_stocks) * 100


class StockDatabase:
    """
    股票数据库管理器

    使用SQLite存储所有股票数据，支持高效查询和增量更新
    """

    def __init__(
        self,
        db_path: str = "/root/.openclaw/workspace/myQuant/data/stock_database.db",
        cache_dir: str = "/root/.openclaw/workspace/myQuant/data/cache",
        verbose: bool = True
    ):
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.verbose = verbose

        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        # 初始化数据库
        self._init_database()

        # 股票池管理
        self.universe = StockUniverse()
        self.progress = DownloadProgress(0, 0, [], datetime.now())

    def _log(self, msg: str):
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] [StockDB] {msg}")

    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 股票列表表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_list (
                ts_code TEXT PRIMARY KEY,
                name TEXT,
                industry TEXT,
                market TEXT,
                list_date TEXT,
                is_hs300 INTEGER DEFAULT 0,
                is_zz500 INTEGER DEFAULT 0,
                is_zz1000 INTEGER DEFAULT 0,
                last_update TEXT
            )
        ''')

        # 日线数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_data (
                ts_code TEXT,
                trade_date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                PRIMARY KEY (ts_code, trade_date)
            )
        ''')

        # 因子数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_data (
                ts_code TEXT,
                trade_date TEXT,
                factor_name TEXT,
                factor_value REAL,
                PRIMARY KEY (ts_code, trade_date, factor_name)
            )
        ''')

        # 下载日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS download_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_code TEXT,
                start_date TEXT,
                end_date TEXT,
                records_count INTEGER,
                status TEXT,
                message TEXT,
                created_at TEXT
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_ts_code ON daily_data(ts_code)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_data(trade_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_factor_code ON factor_data(ts_code)')

        conn.commit()
        conn.close()

        self._log("数据库初始化完成")

    def update_stock_list(self) -> int:
        """
        更新股票列表

        获取HS300+ZZ500+ZZ1000成分股并更新到数据库

        Returns:
            更新的股票数量
        """
        self._log("更新股票列表...")

        # 获取各指数成分股
        hs300 = set(self.universe.get_hs300())
        zz500 = set(self.universe.get_zz500())
        zz1000 = set(self.universe.get_zz1000())

        # 合并所有股票
        all_stocks = hs300 | zz500 | zz1000

        self._log(f"获取到 {len(all_stocks)} 只唯一股票")
        self._log(f"  - 沪深300: {len(hs300)} 只")
        self._log(f"  - 中证500: {len(zz500)} 只")
        self._log(f"  - 中证1000: {len(zz1000)} 只")

        # 更新到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        update_count = 0
        for ts_code in all_stocks:
            is_hs300 = 1 if ts_code in hs300 else 0
            is_zz500 = 1 if ts_code in zz500 else 0
            is_zz1000 = 1 if ts_code in zz1000 else 0

            cursor.execute('''
                INSERT OR REPLACE INTO stock_list
                (ts_code, is_hs300, is_zz500, is_zz1000, last_update)
                VALUES (?, ?, ?, ?, ?)
            ''', (ts_code, is_hs300, is_zz500, is_zz1000, datetime.now().isoformat()))

            update_count += 1

        conn.commit()
        conn.close()

        self._log(f"股票列表更新完成: {update_count} 只")
        return update_count

    def get_stocks_to_download(
        self,
        start_date: str,
        end_date: str,
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        获取需要下载的股票列表

        根据数据库中已有数据，计算需要补充下载的股票和日期
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取所有股票
        cursor.execute('SELECT ts_code FROM stock_list')
        all_stocks = [row[0] for row in cursor.fetchall()]
        
        # 检查每只股票的数据完整性
        stocks_to_download = []
        
        for ts_code in all_stocks:
            # 检查该股票在日期范围内是否有数据
            cursor.execute('''
                SELECT COUNT(*) FROM daily_data 
                WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
            ''', (ts_code, start_date, end_date))
            
            count = cursor.fetchone()[0]
            
            # 如果没有数据或数据很少，加入下载列表
            # 修正：只要有数据就认为已下载（因为停牌期间确实没有数据）
            if count < 10:  # 少于10条认为需要下载
                stocks_to_download.append(ts_code)
        
        conn.close()
        
        if batch_size:
            stocks_to_download = stocks_to_download[:batch_size]
        
        return stocks_to_download

    def download_stock_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str
    ) -> bool:
        """
        下载单只股票数据

        Returns:
            是否成功
        """
        try:
            # 使用tushare获取数据
            import tushare as ts

            pro = ts.pro_api()
            pro._DataApi__token = self.universe.token
            pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'

            # 获取日线数据
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )

            if df is None or df.empty:
                return False

            # 数据清洗
            df = df.rename(columns={
                'trade_date': 'trade_date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount'
            })

            # 保存到数据库
            conn = sqlite3.connect(self.db_path)

            for _, row in df.iterrows():
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO daily_data
                        (ts_code, trade_date, open, high, low, close, volume, amount)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ts_code,
                        row['trade_date'],
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume']),
                        float(row['amount']) if pd.notna(row['amount']) else 0
                    ))
                except Exception as e:
                    pass  # 跳过错误记录

            conn.commit()
            conn.close()

            # 记录下载日志
            self._log_download(ts_code, start_date, end_date, len(df), 'success')

            return True

        except Exception as e:
            self._log_download(ts_code, start_date, end_date, 0, 'failed', str(e))
            return False

    def _log_download(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        records_count: int,
        status: str,
        message: str = ''
    ):
        """记录下载日志"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO download_log
            (ts_code, start_date, end_date, records_count, status, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (ts_code, start_date, end_date, records_count, status, message, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def batch_download(
        self,
        start_date: str = '20200101',
        end_date: Optional[str] = None,
        max_workers: int = 5,
        batch_size: int = 100
    ) -> DownloadProgress:
        """
        批量下载数据

        Args:
            start_date: 开始日期
            end_date: 结束日期 (默认今天)
            max_workers: 并行线程数
            batch_size: 每批处理的股票数

        Returns:
            下载进度
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        # 获取需要下载的股票
        stocks = self.get_stocks_to_download(start_date, end_date)

        if not stocks:
            self._log("所有股票数据已是最新")
            return DownloadProgress(0, 0, [], datetime.now())

        self._log(f"需要下载 {len(stocks)} 只股票，日期范围: {start_date} 至 {end_date}")

        # 限制批次大小
        if len(stocks) > batch_size:
            self._log(f"分批处理，本次处理前 {batch_size} 只")
            stocks = stocks[:batch_size]

        # 初始化进度
        self.progress = DownloadProgress(
            total_stocks=len(stocks),
            completed_stocks=0,
            failed_stocks=[],
            last_update=datetime.now()
        )

        # 并行下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.download_stock_data,
                    ts_code,
                    start_date,
                    end_date
                ): ts_code for ts_code in stocks
            }

            for future in as_completed(futures):
                ts_code = futures[future]
                try:
                    success = future.result()
                    if success:
                        self.progress.completed_stocks += 1
                    else:
                        self.progress.failed_stocks.append(ts_code)
                except Exception as e:
                    self._log(f"下载异常 {ts_code}: {e}")
                    self.progress.failed_stocks.append(ts_code)

                # 显示进度
                if (self.progress.completed_stocks + len(self.progress.failed_stocks)) % 10 == 0:
                    self._log(f"进度: {self.progress.progress_pct:.1f}% "
                             f"({self.progress.completed_stocks}/{self.progress.total_stocks})")

        self.progress.last_update = datetime.now()

        self._log(f"批量下载完成: 成功 {self.progress.completed_stocks}, "
                 f"失败 {len(self.progress.failed_stocks)}")

        return self.progress

    def get_data(
        self,
        ts_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        从数据库查询数据

        Args:
            ts_codes: 股票代码列表 (None表示全部)
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM daily_data WHERE 1=1"
        params = []

        if ts_codes:
            placeholders = ','.join(['?' for _ in ts_codes])
            query += f" AND ts_code IN ({placeholders})"
            params.extend(ts_codes)

        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)

        query += " ORDER BY ts_code, trade_date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df

    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # 股票数量
        cursor.execute('SELECT COUNT(*) FROM stock_list')
        stats['total_stocks'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM stock_list WHERE is_hs300=1')
        stats['hs300_count'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM stock_list WHERE is_zz500=1')
        stats['zz500_count'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM stock_list WHERE is_zz1000=1')
        stats['zz1000_count'] = cursor.fetchone()[0]

        # 数据记录数
        cursor.execute('SELECT COUNT(*) FROM daily_data')
        stats['total_records'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT ts_code) FROM daily_data')
        stats['stocks_with_data'] = cursor.fetchone()[0]

        # 日期范围
        cursor.execute('SELECT MIN(trade_date), MAX(trade_date) FROM daily_data')
        result = cursor.fetchone()
        stats['date_range'] = f"{result[0]} 至 {result[1]}" if result[0] else "N/A"

        conn.close()

        return stats


# 便捷函数
def init_database(db_path: Optional[str] = None) -> StockDatabase:
    """初始化数据库"""
    return StockDatabase(db_path=db_path)


def download_all_data(
    start_date: str = '20200101',
    end_date: Optional[str] = None,
    max_workers: int = 5,
    batch_size: int = 100
) -> DownloadProgress:
    """下载所有数据"""
    db = StockDatabase()

    # 先更新股票列表
    db.update_stock_list()

    # 批量下载
    return db.batch_download(
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
        batch_size=batch_size
    )


if __name__ == '__main__':
    print("=" * 80)
    print("Stock Database Manager - 测试")
    print("=" * 80)

    # 初始化数据库
    db = StockDatabase()

    # 更新股票列表
    db.update_stock_list()

    # 查看统计
    stats = db.get_statistics()
    print(f"\n数据库统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试下载一小批
    print("\n测试下载10只股票...")
    progress = db.batch_download(
        start_date='20240101',
        end_date='20241231',
        max_workers=3,
        batch_size=10
    )

    print(f"\n下载结果: {progress.progress_pct:.1f}% 完成")

#!/usr/bin/env python3
"""
财报PIT数据处理 (Financial Report Point-in-Time Data Processing) - quant-investor V2.3

借鉴Qlib的PIT数据库设计，处理财务报表数据的多次修订问题，消除Look-Ahead Bias。

作者: Manus AI
日期: 2026-01-31
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import sqlite3
from datetime import datetime


class FinancialPIT:
    """
    财报Point-in-Time数据管理器
    """
    
    def __init__(self, db_path: str = None):
        """
        初始化PIT数据管理器
        
        Args:
            db_path: SQLite数据库路径，默认为~/.quant-investor/financial_pit.db
        """
        if db_path is None:
            db_path = Path.home() / '.quant-investor' / 'financial_pit.db'
        else:
            db_path = Path(db_path)
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        
        # 创建表
        self._create_tables()
        
        print(f"[FinancialPIT] 数据库路径: {self.db_path}")
    
    def _create_tables(self):
        """创建数据库表"""
        cursor = self.conn.cursor()
        
        # 创建财报数据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT NOT NULL,
            field_name TEXT NOT NULL,
            report_date TEXT NOT NULL,
            publish_date TEXT NOT NULL,
            value REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(stock_code, field_name, report_date, publish_date)
        )
        ''')
        
        # 创建索引
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_stock_field_report 
        ON financial_data(stock_code, field_name, report_date)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_publish_date 
        ON financial_data(publish_date)
        ''')
        
        self.conn.commit()
    
    def insert_data(
        self,
        stock_code: str,
        field_name: str,
        report_date: str,
        publish_date: str,
        value: float
    ):
        """
        插入财报数据
        
        Args:
            stock_code: 股票代码
            field_name: 字段名称（如revenue, net_profit）
            report_date: 报告期（如2020-12-31）
            publish_date: 发布日期（如2021-04-30）
            value: 数据值
        """
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO financial_data 
            (stock_code, field_name, report_date, publish_date, value)
            VALUES (?, ?, ?, ?, ?)
            ''', (stock_code, field_name, report_date, publish_date, value))
            
            self.conn.commit()
        except Exception as e:
            print(f"[FinancialPIT] 插入数据失败: {e}")
            self.conn.rollback()
    
    def query_at_time(
        self,
        stock_code: str,
        field_name: str,
        query_date: str,
        report_date: Optional[str] = None
    ) -> Optional[float]:
        """
        查询某个时间点的财报数据（Point-in-Time查询）
        
        Args:
            stock_code: 股票代码
            field_name: 字段名称
            query_date: 查询日期（回测时间点）
            report_date: 报告期（可选，如果不指定则返回最新的报告期数据）
            
        Returns:
            该时间点可用的数据值，如果不存在则返回None
        """
        cursor = self.conn.cursor()
        
        if report_date is not None:
            # 查询特定报告期的数据
            cursor.execute('''
            SELECT value FROM financial_data
            WHERE stock_code = ? 
              AND field_name = ?
              AND report_date = ?
              AND publish_date <= ?
            ORDER BY publish_date DESC
            LIMIT 1
            ''', (stock_code, field_name, report_date, query_date))
        else:
            # 查询最新的报告期数据
            cursor.execute('''
            SELECT value FROM financial_data
            WHERE stock_code = ? 
              AND field_name = ?
              AND publish_date <= ?
            ORDER BY report_date DESC, publish_date DESC
            LIMIT 1
            ''', (stock_code, field_name, query_date))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def batch_query_at_time(
        self,
        stock_codes: List[str],
        field_name: str,
        query_date: str
    ) -> Dict[str, float]:
        """
        批量查询多只股票在某个时间点的财报数据
        
        Args:
            stock_codes: 股票代码列表
            field_name: 字段名称
            query_date: 查询日期
            
        Returns:
            股票代码到数据值的字典
        """
        result = {}
        for stock_code in stock_codes:
            value = self.query_at_time(stock_code, field_name, query_date)
            if value is not None:
                result[stock_code] = value
        
        return result
    
    def get_history(
        self,
        stock_code: str,
        field_name: str,
        report_date: str
    ) -> pd.DataFrame:
        """
        获取某个报告期的所有修订历史
        
        Args:
            stock_code: 股票代码
            field_name: 字段名称
            report_date: 报告期
            
        Returns:
            包含所有修订记录的DataFrame
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
        SELECT publish_date, value FROM financial_data
        WHERE stock_code = ? 
          AND field_name = ?
          AND report_date = ?
        ORDER BY publish_date
        ''', (stock_code, field_name, report_date))
        
        rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame(columns=['publish_date', 'value'])
        
        return pd.DataFrame(rows, columns=['publish_date', 'value'])
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()


def demo():
    """演示财报PIT数据处理的使用"""
    print("=" * 60)
    print("财报PIT数据处理演示")
    print("=" * 60)
    
    # 创建PIT管理器
    pit = FinancialPIT()
    
    # 模拟插入财报数据（包含修订）
    print("\n【插入数据】")
    
    # 某公司2020年报，首次发布
    pit.insert_data(
        stock_code='600000.SH',
        field_name='net_profit',
        report_date='2020-12-31',
        publish_date='2021-04-30',
        value=1000000000.0
    )
    print("插入: 600000.SH, 2020年报, 净利润=10亿, 发布日期=2021-04-30")
    
    # 修订版本
    pit.insert_data(
        stock_code='600000.SH',
        field_name='net_profit',
        report_date='2020-12-31',
        publish_date='2021-06-30',
        value=1050000000.0
    )
    print("插入: 600000.SH, 2020年报, 净利润=10.5亿, 发布日期=2021-06-30 (修订)")
    
    # 2021年报
    pit.insert_data(
        stock_code='600000.SH',
        field_name='net_profit',
        report_date='2021-12-31',
        publish_date='2022-04-30',
        value=1200000000.0
    )
    print("插入: 600000.SH, 2021年报, 净利润=12亿, 发布日期=2022-04-30")
    
    # Point-in-Time查询
    print("\n【Point-in-Time查询】")
    
    # 2021年5月1日查询（应该返回首次发布的10亿）
    value_20210501 = pit.query_at_time(
        stock_code='600000.SH',
        field_name='net_profit',
        query_date='2021-05-01',
        report_date='2020-12-31'
    )
    print(f"2021-05-01查询2020年报净利润: {value_20210501 / 1e8:.2f}亿")
    
    # 2021年7月1日查询（应该返回修订后的10.5亿）
    value_20210701 = pit.query_at_time(
        stock_code='600000.SH',
        field_name='net_profit',
        query_date='2021-07-01',
        report_date='2020-12-31'
    )
    print(f"2021-07-01查询2020年报净利润: {value_20210701 / 1e8:.2f}亿")
    
    # 查询修订历史
    print("\n【修订历史】")
    history = pit.get_history(
        stock_code='600000.SH',
        field_name='net_profit',
        report_date='2020-12-31'
    )
    print(history)
    
    # 关闭连接
    pit.close()
    print("\n[FinancialPIT] 演示完成")


if __name__ == "__main__":
    demo()

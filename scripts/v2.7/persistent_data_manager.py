"""
PersistentDataManager - 持久化数据存储管理器

核心功能:
1. 使用SQLite管理数据元信息（索引）
2. 使用Parquet存储实际数据（高效列式存储）
3. 智能增量更新：只下载缺失的数据
4. 统一数据访问接口

V2.7 核心组件
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Tuple, Union
import pandas as pd


class PersistentDataManager:
    """持久化数据存储管理器"""
    
    # 数据类型配置：定义每种数据类型的默认缓存策略
    DATA_TYPE_CONFIG = {
        # 美股数据
        'us_macro_gdp': {'ttl_days': 30, 'frequency': 'quarterly'},
        'us_macro_cpi': {'ttl_days': 7, 'frequency': 'monthly'},
        'us_macro_pce': {'ttl_days': 7, 'frequency': 'monthly'},
        'us_macro_unemployment': {'ttl_days': 7, 'frequency': 'monthly'},
        'us_macro_fed_rate': {'ttl_days': 1, 'frequency': 'daily'},
        'us_macro_treasury': {'ttl_days': 1, 'frequency': 'daily'},
        'us_stock_daily': {'ttl_days': 1, 'frequency': 'daily'},
        'us_stock_info': {'ttl_days': 7, 'frequency': 'static'},
        'us_index_daily': {'ttl_days': 1, 'frequency': 'daily'},
        
        # A股数据
        'cn_macro_gdp': {'ttl_days': 30, 'frequency': 'quarterly'},
        'cn_macro_cpi': {'ttl_days': 7, 'frequency': 'monthly'},
        'cn_macro_ppi': {'ttl_days': 7, 'frequency': 'monthly'},
        'cn_macro_pmi': {'ttl_days': 7, 'frequency': 'monthly'},
        'cn_macro_lpr': {'ttl_days': 1, 'frequency': 'daily'},
        'cn_macro_shibor': {'ttl_days': 1, 'frequency': 'daily'},
        'cn_macro_m2': {'ttl_days': 7, 'frequency': 'monthly'},
        'cn_stock_daily': {'ttl_days': 1, 'frequency': 'daily'},
        'cn_stock_info': {'ttl_days': 7, 'frequency': 'static'},
        'cn_index_daily': {'ttl_days': 1, 'frequency': 'daily'},
        
        # 高级数据（Tushare一万分权限）
        'cn_margin': {'ttl_days': 1, 'frequency': 'daily'},
        'cn_top_list': {'ttl_days': 1, 'frequency': 'daily'},
        'cn_block_trade': {'ttl_days': 1, 'frequency': 'daily'},
        'cn_hk_hold': {'ttl_days': 1, 'frequency': 'daily'},
        'cn_holder_number': {'ttl_days': 7, 'frequency': 'periodic'},
    }
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化持久化数据管理器
        
        Args:
            data_dir: 数据存储根目录，默认为 ~/.quant_investor/data
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path.home() / '.quant_investor' / 'data'
        
        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化SQLite数据库
        self.db_path = self.data_dir / 'metadata.db'
        self._init_database()
        
        # 统计信息
        self.stats = {
            'cache_hits': 0,
            'partial_hits': 0,
            'cache_misses': 0,
            'data_saved': 0,
        }
    
    def _init_database(self):
        """初始化SQLite元数据数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建元数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_metadata (
                data_key TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                symbol TEXT,
                source TEXT,
                start_date TEXT,
                end_date TEXT,
                last_updated TEXT NOT NULL,
                file_path TEXT NOT NULL,
                row_count INTEGER DEFAULT 0,
                extra_info TEXT
            )
        ''')
        
        # 创建索引以加速查询
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON data_metadata(data_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON data_metadata(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_updated ON data_metadata(last_updated)')
        
        conn.commit()
        conn.close()
    
    def _get_data_key(self, data_type: str, symbol: Optional[str] = None) -> str:
        """生成数据唯一键"""
        if symbol:
            return f"{data_type}_{symbol}"
        return data_type
    
    def _get_data_path(self, data_type: str, symbol: Optional[str] = None) -> Path:
        """获取数据文件存储路径"""
        # 提取数据类别（如 us_macro, cn_stock 等）
        parts = data_type.split('_')
        if len(parts) >= 2:
            category = f"{parts[0]}_{parts[1]}"
        else:
            category = data_type
        
        # 创建分类目录
        category_dir = self.data_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        if symbol:
            filename = f"{symbol}.parquet"
        else:
            filename = f"{data_type}.parquet"
        
        return category_dir / filename
    
    def _get_metadata(self, data_key: str) -> Optional[Dict]:
        """获取数据元信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data_key, data_type, symbol, source, start_date, end_date, 
                   last_updated, file_path, row_count, extra_info
            FROM data_metadata WHERE data_key = ?
        ''', (data_key,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'data_key': row[0],
                'data_type': row[1],
                'symbol': row[2],
                'source': row[3],
                'start_date': row[4],
                'end_date': row[5],
                'last_updated': row[6],
                'file_path': row[7],
                'row_count': row[8],
                'extra_info': json.loads(row[9]) if row[9] else {}
            }
        return None
    
    def _save_metadata(self, data_key: str, data_type: str, symbol: Optional[str],
                       source: str, start_date: str, end_date: str, 
                       file_path: str, row_count: int, extra_info: Optional[Dict] = None):
        """保存数据元信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO data_metadata 
            (data_key, data_type, symbol, source, start_date, end_date, 
             last_updated, file_path, row_count, extra_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_key, data_type, symbol, source, start_date, end_date,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            str(file_path), row_count,
            json.dumps(extra_info) if extra_info else None
        ))
        
        conn.commit()
        conn.close()
    
    def _is_data_fresh(self, metadata: Dict, data_type: str) -> bool:
        """检查数据是否仍然新鲜（未过期）"""
        config = self.DATA_TYPE_CONFIG.get(data_type, {'ttl_days': 1})
        ttl_days = config['ttl_days']
        
        last_updated = datetime.strptime(metadata['last_updated'], '%Y-%m-%d %H:%M:%S')
        return (datetime.now() - last_updated) < timedelta(days=ttl_days)
    
    def _load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """从Parquet文件加载数据"""
        try:
            path = Path(file_path)
            if path.exists():
                return pd.read_parquet(path)
        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
        return None
    
    def _save_data(self, df: pd.DataFrame, file_path: Path):
        """保存数据到Parquet文件"""
        try:
            df.to_parquet(file_path, index=False)
            self.stats['data_saved'] += 1
        except Exception as e:
            print(f"保存数据失败 {file_path}: {e}")
            raise
    
    def query(
        self,
        data_type: str,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fetch_func: Optional[Callable] = None,
        source: str = 'unknown',
        date_column: str = 'date',
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        查询数据，智能判断从本地读取还是从网络下载
        
        Args:
            data_type: 数据类型，如 'us_stock_daily', 'cn_macro_gdp'
            symbol: 股票/标的代码（可选）
            start_date: 开始日期
            end_date: 结束日期
            fetch_func: 数据获取函数，当本地数据不足时调用
            source: 数据来源标识
            date_column: 日期列名
            force_refresh: 是否强制刷新数据
            
        Returns:
            DataFrame或None
        """
        data_key = self._get_data_key(data_type, symbol)
        file_path = self._get_data_path(data_type, symbol)
        
        # 获取元数据
        metadata = self._get_metadata(data_key)
        
        # 情况1: 本地有数据且未过期
        if metadata and not force_refresh:
            local_df = self._load_data(metadata['file_path'])
            
            if local_df is not None and not local_df.empty:
                # 检查数据是否新鲜
                if self._is_data_fresh(metadata, data_type):
                    # 检查日期范围是否满足需求
                    if self._check_date_coverage(local_df, start_date, end_date, date_column):
                        self.stats['cache_hits'] += 1
                        return self._filter_by_date(local_df, start_date, end_date, date_column)
                    else:
                        # 部分覆盖，需要增量更新
                        self.stats['partial_hits'] += 1
                        return self._incremental_update(
                            data_key, data_type, symbol, local_df,
                            start_date, end_date, fetch_func, source, 
                            date_column, file_path
                        )
        
        # 情况2: 本地无数据或数据过期，需要完全下载
        self.stats['cache_misses'] += 1
        
        if fetch_func is None:
            return None
        
        try:
            new_df = fetch_func()
            if new_df is not None and not new_df.empty:
                self._save_data(new_df, file_path)
                
                # 计算日期范围
                df_start, df_end = self._get_date_range(new_df, date_column)
                
                self._save_metadata(
                    data_key, data_type, symbol, source,
                    df_start, df_end, str(file_path), len(new_df)
                )
                
                return self._filter_by_date(new_df, start_date, end_date, date_column)
        except Exception as e:
            print(f"获取数据失败 {data_key}: {e}")
        
        return None
    
    def _check_date_coverage(self, df: pd.DataFrame, start_date: Optional[str], 
                             end_date: Optional[str], date_column: str) -> bool:
        """检查本地数据是否完全覆盖请求的日期范围"""
        if df.empty or date_column not in df.columns:
            return False
        
        # 获取本地数据的日期范围
        df_dates = pd.to_datetime(df[date_column])
        local_start = df_dates.min()
        local_end = df_dates.max()
        
        # 如果没有指定日期范围，认为已覆盖
        if start_date is None and end_date is None:
            return True
        
        # 检查是否覆盖
        req_start = pd.to_datetime(start_date) if start_date else local_start
        req_end = pd.to_datetime(end_date) if end_date else datetime.now()
        
        # 允许1天的误差（因为最新数据可能还未发布）
        return local_start <= req_start and local_end >= (req_end - timedelta(days=1))
    
    def _filter_by_date(self, df: pd.DataFrame, start_date: Optional[str],
                        end_date: Optional[str], date_column: str) -> pd.DataFrame:
        """按日期范围过滤数据"""
        if df.empty or date_column not in df.columns:
            return df
        
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        if start_date:
            df = df[df[date_column] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[date_column] <= pd.to_datetime(end_date)]
        
        return df
    
    def _get_date_range(self, df: pd.DataFrame, date_column: str) -> Tuple[str, str]:
        """获取DataFrame的日期范围"""
        if df.empty or date_column not in df.columns:
            return '', ''
        
        dates = pd.to_datetime(df[date_column])
        return dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d')
    
    def _incremental_update(
        self,
        data_key: str,
        data_type: str,
        symbol: Optional[str],
        local_df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str],
        fetch_func: Optional[Callable],
        source: str,
        date_column: str,
        file_path: Path
    ) -> Optional[pd.DataFrame]:
        """增量更新数据"""
        if fetch_func is None:
            return self._filter_by_date(local_df, start_date, end_date, date_column)
        
        # 获取本地数据的日期范围
        local_dates = pd.to_datetime(local_df[date_column])
        local_end = local_dates.max()
        
        # 计算需要下载的日期范围（从本地数据结束日期的下一天开始）
        fetch_start = (local_end + timedelta(days=1)).strftime('%Y-%m-%d')
        fetch_end = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # 如果需要下载的范围有效
        if pd.to_datetime(fetch_start) <= pd.to_datetime(fetch_end):
            try:
                new_df = fetch_func()
                if new_df is not None and not new_df.empty:
                    # 合并数据
                    new_df[date_column] = pd.to_datetime(new_df[date_column])
                    local_df[date_column] = pd.to_datetime(local_df[date_column])
                    
                    # 只保留新数据中比本地数据更新的部分
                    new_df = new_df[new_df[date_column] > local_end]
                    
                    if not new_df.empty:
                        merged_df = pd.concat([local_df, new_df], ignore_index=True)
                        merged_df = merged_df.drop_duplicates(subset=[date_column]).sort_values(date_column)
                        
                        # 保存合并后的数据
                        self._save_data(merged_df, file_path)
                        
                        # 更新元数据
                        df_start, df_end = self._get_date_range(merged_df, date_column)
                        self._save_metadata(
                            data_key, data_type, symbol, source,
                            df_start, df_end, str(file_path), len(merged_df)
                        )
                        
                        return self._filter_by_date(merged_df, start_date, end_date, date_column)
            except Exception as e:
                print(f"增量更新失败 {data_key}: {e}")
        
        # 如果增量更新失败，返回本地数据
        return self._filter_by_date(local_df, start_date, end_date, date_column)
    
    def save(
        self,
        data_type: str,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        source: str = 'unknown',
        date_column: str = 'date',
        extra_info: Optional[Dict] = None
    ) -> bool:
        """
        直接保存数据到持久化存储
        
        Args:
            data_type: 数据类型
            df: 要保存的DataFrame
            symbol: 股票/标的代码
            source: 数据来源
            date_column: 日期列名
            extra_info: 额外信息
            
        Returns:
            是否保存成功
        """
        if df is None or df.empty:
            return False
        
        data_key = self._get_data_key(data_type, symbol)
        file_path = self._get_data_path(data_type, symbol)
        
        try:
            self._save_data(df, file_path)
            
            df_start, df_end = self._get_date_range(df, date_column)
            
            self._save_metadata(
                data_key, data_type, symbol, source,
                df_start, df_end, str(file_path), len(df), extra_info
            )
            
            return True
        except Exception as e:
            print(f"保存数据失败 {data_key}: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def list_data(self, data_type: Optional[str] = None) -> List[Dict]:
        """
        列出已存储的数据
        
        Args:
            data_type: 可选，按数据类型过滤
            
        Returns:
            数据元信息列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if data_type:
            cursor.execute('''
                SELECT data_key, data_type, symbol, source, start_date, end_date, 
                       last_updated, row_count
                FROM data_metadata WHERE data_type = ?
                ORDER BY last_updated DESC
            ''', (data_type,))
        else:
            cursor.execute('''
                SELECT data_key, data_type, symbol, source, start_date, end_date, 
                       last_updated, row_count
                FROM data_metadata
                ORDER BY last_updated DESC
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'data_key': row[0],
                'data_type': row[1],
                'symbol': row[2],
                'source': row[3],
                'start_date': row[4],
                'end_date': row[5],
                'last_updated': row[6],
                'row_count': row[7]
            }
            for row in rows
        ]
    
    def clear_expired(self, days: int = 30) -> int:
        """
        清理过期数据
        
        Args:
            days: 超过多少天未更新的数据将被清理
            
        Returns:
            清理的数据条数
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        
        # 获取要删除的文件路径
        cursor.execute('''
            SELECT file_path FROM data_metadata WHERE last_updated < ?
        ''', (cutoff_date,))
        
        rows = cursor.fetchall()
        deleted_count = 0
        
        for row in rows:
            file_path = Path(row[0])
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"删除文件失败 {file_path}: {e}")
        
        # 删除元数据记录
        cursor.execute('''
            DELETE FROM data_metadata WHERE last_updated < ?
        ''', (cutoff_date,))
        
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def get_storage_summary(self) -> Dict:
        """获取存储摘要信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 按数据类型统计
        cursor.execute('''
            SELECT data_type, COUNT(*), SUM(row_count)
            FROM data_metadata
            GROUP BY data_type
        ''')
        
        by_type = {row[0]: {'count': row[1], 'rows': row[2] or 0} for row in cursor.fetchall()}
        
        # 总计
        cursor.execute('SELECT COUNT(*), SUM(row_count) FROM data_metadata')
        total = cursor.fetchone()
        
        conn.close()
        
        # 计算磁盘使用量
        total_size = 0
        for path in self.data_dir.rglob('*.parquet'):
            total_size += path.stat().st_size
        
        return {
            'total_datasets': total[0] or 0,
            'total_rows': total[1] or 0,
            'disk_usage_mb': round(total_size / (1024 * 1024), 2),
            'by_type': by_type
        }


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("PersistentDataManager 测试")
    print("=" * 60)
    
    # 初始化管理器
    manager = PersistentDataManager()
    
    # 测试保存数据
    print("\n1. 测试保存数据:")
    test_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'value': range(10)
    })
    
    success = manager.save('test_data', test_df, source='test')
    print(f"   保存结果: {'成功' if success else '失败'}")
    
    # 测试查询数据
    print("\n2. 测试查询数据:")
    result = manager.query('test_data')
    if result is not None:
        print(f"   查询到 {len(result)} 条数据")
    
    # 测试存储摘要
    print("\n3. 存储摘要:")
    summary = manager.get_storage_summary()
    print(f"   总数据集: {summary['total_datasets']}")
    print(f"   总行数: {summary['total_rows']}")
    print(f"   磁盘使用: {summary['disk_usage_mb']} MB")
    
    # 测试统计信息
    print("\n4. 统计信息:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

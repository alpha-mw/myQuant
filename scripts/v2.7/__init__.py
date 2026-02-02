"""
Quant-Investor V2.7 - 持久化数据存储层

核心组件:
- PersistentDataManager: 持久化数据存储管理器
- PersistentUSDataManager: 集成持久化存储的美股数据管理器
- PersistentCNDataManager: 集成持久化存储的A股数据管理器

V2.7 核心特性:
1. 数据持久化：所有下载的数据永久保存在本地
2. 增量更新：智能判断本地数据覆盖范围，只下载缺失数据
3. 统一管理：通过SQLite元数据数据库统一管理所有数据
4. 高性能：采用Parquet列式存储，优化读写性能
"""

from .persistent_data_manager import PersistentDataManager
from .persistent_us_data_manager import PersistentUSDataManager
from .persistent_cn_data_manager import PersistentCNDataManager

__all__ = [
    'PersistentDataManager',
    'PersistentUSDataManager',
    'PersistentCNDataManager',
]

__version__ = '2.7.0'

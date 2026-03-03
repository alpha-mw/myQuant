"""
Quant-Investor V7.0 配置管理模块
安全地加载环境变量和配置
"""

import os
from pathlib import Path
from typing import Optional

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


class Config:
    """配置类"""
    
    # Tushare配置
    TUSHARE_TOKEN: str = os.environ.get('TUSHARE_TOKEN', '')
    TUSHARE_URL: Optional[str] = os.environ.get('TUSHARE_URL')
    
    # 数据库配置
    DB_PATH: str = os.environ.get('DB_PATH', 'data/stock_database.db')
    
    # 日志配置
    LOG_LEVEL: str = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Redis配置
    REDIS_HOST: str = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.environ.get('REDIS_PORT', '6379'))
    REDIS_DB: int = int(os.environ.get('REDIS_DB', '0'))
    
    # 回测配置
    INITIAL_CASH: float = float(os.environ.get('INITIAL_CASH', '1000000'))
    COMMISSION_RATE: float = float(os.environ.get('COMMISSION_RATE', '0.0003'))
    STAMP_DUTY_RATE: float = float(os.environ.get('STAMP_DUTY_RATE', '0.001'))
    SLIPPAGE: float = float(os.environ.get('SLIPPAGE', '0.001'))
    
    @classmethod
    def validate(cls) -> list:
        """验证配置是否完整"""
        errors = []
        
        if not cls.TUSHARE_TOKEN:
            errors.append("TUSHARE_TOKEN 未设置")
        
        return errors


# 导出配置
config = Config()

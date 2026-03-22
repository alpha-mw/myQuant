"""
Quant-Investor V7.0 配置管理模块
安全地加载环境变量和配置
"""

import os
from pathlib import Path
from typing import Optional

from quant_investor.credential_utils import get_secret

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


def _default_kronos_model_path() -> str:
    candidates = [
        os.environ.get('KRONOS_MODEL_PATH'),
        str(Path(__file__).resolve().parents[1] / 'data' / 'models' / 'kronos'),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return candidates[-1]


def _default_chronos_model_name() -> str:
    candidates = [
        os.environ.get('CHRONOS_MODEL_NAME'),
        str(Path(__file__).resolve().parents[1] / 'data' / 'models' / 'chronos-2'),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return candidates[-1]


class Config:
    """配置类"""
    
    # Tushare配置
    TUSHARE_TOKEN: str = get_secret('TUSHARE_TOKEN')
    TUSHARE_URL: Optional[str] = os.environ.get('TUSHARE_URL')

    # LLM / 外部 API 凭据
    OPENAI_API_KEY: str = get_secret('OPENAI_API_KEY')
    ANTHROPIC_API_KEY: str = get_secret('ANTHROPIC_API_KEY')
    DEEPSEEK_API_KEY: str = get_secret('DEEPSEEK_API_KEY')
    GOOGLE_API_KEY: str = get_secret('GOOGLE_API_KEY')
    DASHSCOPE_API_KEY: str = get_secret('DASHSCOPE_API_KEY')
    FRED_API_KEY: str = get_secret('FRED_API_KEY')
    FINNHUB_API_KEY: str = get_secret('FINNHUB_API_KEY')
    
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

    # K线分析后端配置
    KRONOS_MODEL_PATH: str = _default_kronos_model_path()
    KRONOS_MODEL_SIZE: str = os.environ.get('KRONOS_MODEL_SIZE', 'base')
    CHRONOS_MODEL_NAME: str = _default_chronos_model_name()
    KLINE_BACKEND: str = os.environ.get('KLINE_BACKEND', 'hybrid')
    KLINE_EVALUATOR: str = os.environ.get('KLINE_EVALUATOR', 'placeholder')
    KLINE_ALLOW_REMOTE_MODEL_DOWNLOAD: bool = os.environ.get(
        'KLINE_ALLOW_REMOTE_MODEL_DOWNLOAD', 'false'
    ).lower() in {'1', 'true', 'yes', 'on'}

    @classmethod
    def validate(cls) -> list:
        """验证配置是否完整"""
        errors = []
        
        if not cls.TUSHARE_TOKEN:
            errors.append("TUSHARE_TOKEN 未设置")
        
        return errors


# 导出配置
config = Config()

"""
Quant-Investor V7.0 日志配置
使用loguru进行结构化日志记录
"""

import sys
from pathlib import Path
from loguru import logger

# 日志目录
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 移除默认处理器
logger.remove()

# 添加控制台处理器（带颜色）
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8>}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# 添加文件处理器（详细日志）
logger.add(
    LOG_DIR / "quant_investor_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # 每天轮转
    retention="30 days",  # 保留30天
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8>} | {name}:{function}:{line} - {message}",
    encoding="utf-8",
)

# 添加错误日志处理器（仅ERROR及以上）
logger.add(
    LOG_DIR / "error_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8>} | {name}:{function}:{line} - {message}",
    encoding="utf-8",
)


def get_logger(name: str):
    """获取带名称的logger实例"""
    return logger.bind(name=name)


# 导出logger
__all__ = ["logger", "get_logger"]

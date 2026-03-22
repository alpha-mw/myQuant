"""
Quant-Investor V7.0 集中日志模块
替代各类中重复的 _log() 方法，提供统一的日志控制
"""

import logging
import sys

_LOG_FORMAT = "[%(name)s] %(message)s"


def get_logger(name: str, verbose: bool = True) -> logging.Logger:
    """
    获取具名日志器

    Args:
        name: 日志器名称，通常为类名
        verbose: True=INFO级别输出，False=静默(WARNING以上才输出)

    Returns:
        配置好的 logging.Logger 实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    # Prevent messages propagating to root logger (avoids duplicate output)
    logger.propagate = False
    return logger

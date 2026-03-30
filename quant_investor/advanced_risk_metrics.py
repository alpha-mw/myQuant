"""
Quant-Investor V7.0 高级风险指标模块
实现Omega、Sortino、Calmar等比率
"""

import numpy as np
import pandas as pd
from typing import Union


def calculate_omega_ratio(returns: Union[pd.Series, np.ndarray],
                          threshold: float = 0) -> float:
    """
    计算Omega比率
    
    Omega = 正收益总和 / 负收益绝对值总和
    
    Args:
        returns: 收益率序列
        threshold: 阈值（默认0）
        
    Returns:
        Omega比率
    """
    returns = np.array(returns).flatten()
    
    # 正收益
    positive_returns = returns[returns > threshold] - threshold
    # 负收益
    negative_returns = returns[returns <= threshold] - threshold
    
    if len(negative_returns) == 0 or np.sum(np.abs(negative_returns)) == 0:
        return np.inf
    
    omega = np.sum(positive_returns) / np.sum(np.abs(negative_returns))
    return omega


def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray],
                            target_return: float = 0,
                            annualized: bool = True) -> float:
    """
    计算Sortino比率
    
    Sortino = (平均收益 - 目标收益) / 下行标准差
    
    Args:
        returns: 收益率序列
        target_return: 目标收益（默认0）
        annualized: 是否年化（默认True）
        
    Returns:
        Sortino比率
    """
    returns = np.array(returns).flatten()
    
    # 平均收益
    mean_return = np.mean(returns)
    
    # 下行标准差（只考虑低于目标的收益）
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return np.inf
    
    # Sortino比率
    sortino = (mean_return - target_return) / downside_std
    
    if annualized:
        sortino *= np.sqrt(252)

    return float(sortino)


def calculate_calmar_ratio(returns: Union[pd.Series, np.ndarray],
                           annualized: bool = True) -> float:
    """
    计算Calmar比率
    
    Calmar = 年化收益 / 最大回撤
    
    Args:
        returns: 收益率序列
        annualized: 是否年化（默认True）
        
    Returns:
        Calmar比率
    """
    returns = np.array(returns).flatten()
    
    # 年化收益
    if annualized:
        annual_return = np.mean(returns) * 252
    else:
        annual_return = np.mean(returns) * len(returns)
    
    # 计算最大回撤
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    if max_drawdown == 0:
        return np.inf
    
    calmar = annual_return / abs(max_drawdown)
    return calmar


def calculate_all_advanced_metrics(returns: Union[pd.Series, np.ndarray]) -> dict:
    """
    计算所有高级风险指标
    
    Args:
        returns: 收益率序列
        
    Returns:
        包含所有指标的字典
    """
    returns = np.array(returns).flatten()
    
    return {
        'omega_ratio': calculate_omega_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
    }

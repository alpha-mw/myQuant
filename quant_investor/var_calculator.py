"""
Quant-Investor V7.0 VaR和CVaR计算模块
实现多种VaR计算方法
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List


def calculate_historical_var(returns: Union[pd.Series, np.ndarray], 
                             confidence_level: float = 0.95) -> float:
    """
    历史模拟法VaR
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平 (默认0.95)
        
    Returns:
        VaR值（负数表示损失）
    """
    returns = np.array(returns).flatten()
    alpha = 1 - confidence_level
    var = np.percentile(returns, alpha * 100)
    return var


def calculate_parametric_var(returns: Union[pd.Series, np.ndarray],
                             confidence_level: float = 0.95) -> float:
    """
    参数法VaR（方差-协方差法）
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平 (默认0.95)
        
    Returns:
        VaR值
    """
    returns = np.array(returns).flatten()
    mean = np.mean(returns)
    std = np.std(returns)
    
    # Z值
    z = stats.norm.ppf(1 - confidence_level)
    var = mean + z * std
    
    return var


def calculate_cornish_fisher_var(returns: Union[pd.Series, np.ndarray],
                                  confidence_level: float = 0.95) -> float:
    """
    Cornish-Fisher调整VaR
    考虑收益率的偏度和峰度
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平 (默认0.95)
        
    Returns:
        VaR值
    """
    returns = np.array(returns).flatten()
    mean = np.mean(returns)
    std = np.std(returns)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # 标准正态Z值
    z = stats.norm.ppf(1 - confidence_level)
    
    # Cornish-Fisher调整
    z_cf = (z + 
            (z**2 - 1) * skewness / 6 + 
            (z**3 - 3*z) * kurtosis / 24 - 
            (2*z**3 - 5*z) * skewness**2 / 36)
    
    var = mean + z_cf * std
    return var


def calculate_monte_carlo_var(returns: Union[pd.Series, np.ndarray],
                               confidence_level: float = 0.95,
                               n_simulations: int = 10000) -> float:
    """
    蒙特卡洛模拟VaR
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平 (默认0.95)
        n_simulations: 模拟次数 (默认10000)
        
    Returns:
        VaR值
    """
    returns = np.array(returns).flatten()
    mean = np.mean(returns)
    std = np.std(returns)
    
    # 生成模拟收益
    simulated_returns = np.random.normal(mean, std, n_simulations)
    
    # 计算VaR
    alpha = 1 - confidence_level
    var = np.percentile(simulated_returns, alpha * 100)
    
    return var


def calculate_cvar(returns: Union[pd.Series, np.ndarray],
                   confidence_level: float = 0.95,
                   method: str = 'historical') -> float:
    """
    计算CVaR（条件VaR/期望损失）
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平 (默认0.95)
        method: 计算方法 ('historical' 或 'parametric')
        
    Returns:
        CVaR值
    """
    returns = np.array(returns).flatten()
    
    if method == 'historical':
        # 历史模拟法
        var = calculate_historical_var(returns, confidence_level)
        # CVaR = 超过VaR的平均损失
        cvar = np.mean(returns[returns <= var])
        
    elif method == 'parametric':
        # 参数法（正态分布假设）
        mean = np.mean(returns)
        std = np.std(returns)
        confidence = confidence_level
        
        # CVaR = mean - std * phi(z) / (1-confidence)
        z = stats.norm.ppf(1 - confidence)
        phi_z = stats.norm.pdf(z)
        
        cvar = mean - std * phi_z / (1 - confidence)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(cvar)


def calculate_all_var_metrics(returns: Union[pd.Series, np.ndarray],
                               confidence_level: float = 0.95) -> dict:
    """
    计算所有VaR指标
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平 (默认0.95)
        
    Returns:
        包含各种VaR方法结果的字典
    """
    returns = np.array(returns).flatten()
    
    return {
        'historical_var': calculate_historical_var(returns, confidence_level),
        'parametric_var': calculate_parametric_var(returns, confidence_level),
        'cornish_fisher_var': calculate_cornish_fisher_var(returns, confidence_level),
        'monte_carlo_var': calculate_monte_carlo_var(returns, confidence_level),
        'historical_cvar': calculate_cvar(returns, confidence_level, 'historical'),
        'parametric_cvar': calculate_cvar(returns, confidence_level, 'parametric'),
        'confidence_level': confidence_level,
    }

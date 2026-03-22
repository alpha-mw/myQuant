"""
Quant-Investor V7.0 因子中性化处理
实现行业和市值中性化
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class FactorNeutralizer:
    """
    因子中性化器
    
    使用方法:
        neutralizer = FactorNeutralizer()
        
        # 市值中性化
        factor_neutral = neutralizer.market_cap_neutralize(
            df['factor_value'], 
            df['market_cap']
        )
        
        # 行业中性化
        factor_neutral = neutralizer.industry_neutralize(
            df['factor_value'],
            df['industry_code']
        )
        
        # 双重中性化
        factor_neutral = neutralizer.double_neutralize(
            df['factor_value'],
            df['market_cap'],
            df['industry_code']
        )
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def market_cap_neutralize(self,
                              factor_values: pd.Series,
                              market_cap: pd.Series) -> pd.Series:
        """
        市值中性化
        
        去除因子中的市值暴露
        
        Args:
            factor_values: 因子值
            market_cap: 市值 (对数化)
            
        Returns:
            中性化后的因子
        """
        # 对数市值
        log_cap = np.log(market_cap.replace(0, np.nan))
        
        # 去除NaN
        valid_idx = factor_values.notna() & log_cap.notna()
        
        if valid_idx.sum() < 10:
            return factor_values
        
        # 回归
        X = log_cap[valid_idx].values.reshape(-1, 1)
        y = factor_values[valid_idx].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 残差 = 中性化后的因子
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # 放回Series
        result = factor_values.copy()
        result[valid_idx] = residuals
        
        return result
    
    def industry_neutralize(self,
                           factor_values: pd.Series,
                           industry_codes: pd.Series) -> pd.Series:
        """
        行业中性化
        
        在每个行业内标准化因子
        
        Args:
            factor_values: 因子值
            industry_codes: 行业代码
            
        Returns:
            中性化后的因子
        """
        result = pd.Series(index=factor_values.index, dtype=float)
        
        for industry in industry_codes.unique():
            mask = industry_codes == industry
            if mask.sum() < 5:  # 行业内样本太少，跳过
                continue
            
            industry_factors = factor_values[mask]
            
            # 行业内标准化
            mean = industry_factors.mean()
            std = industry_factors.std()
            
            if std > 0:
                result[mask] = (industry_factors - mean) / std
            else:
                result[mask] = 0
        
        return result
    
    def double_neutralize(self,
                         factor_values: pd.Series,
                         market_cap: pd.Series,
                         industry_codes: pd.Series) -> pd.Series:
        """
        双重中性化（市值+行业）
        
        Args:
            factor_values: 因子值
            market_cap: 市值
            industry_codes: 行业代码
            
        Returns:
            中性化后的因子
        """
        # 先进行市值中性化
        factor_cap_neutral = self.market_cap_neutralize(factor_values, market_cap)
        
        # 再进行行业中性化
        factor_double_neutral = self.industry_neutralize(factor_cap_neutral, industry_codes)
        
        return factor_double_neutral
    
    def risk_neutralize(self,
                       factor_values: pd.Series,
                       risk_factors: pd.DataFrame) -> pd.Series:
        """
        多因子风险中性化
        
        去除多个风险因子的暴露
        
        Args:
            factor_values: 因子值
            risk_factors: 风险因子矩阵 (市值、行业dummy等)
            
        Returns:
            中性化后的因子
        """
        # 去除NaN
        valid_idx = factor_values.notna() & risk_factors.notna().all(axis=1)
        
        if valid_idx.sum() < 10:
            return factor_values
        
        X = risk_factors[valid_idx].values
        y = factor_values[valid_idx].values
        
        # 标准化风险因子
        X_scaled = self.scaler.fit_transform(X)
        
        # 回归
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # 残差
        y_pred = model.predict(X_scaled)
        residuals = y - y_pred
        
        # 放回Series
        result = factor_values.copy()
        result[valid_idx] = residuals
        
        return result


class PortfolioNeutralizer:
    """
    组合中性化器
    确保投资组合在行业/市值上保持中性
    """
    
    def __init__(self):
        self.neutralizer = FactorNeutralizer()
    
    def neutralize_portfolio_weights(self,
                                     weights: pd.Series,
                                     market_cap: pd.Series,
                                     industry_codes: pd.Series) -> pd.Series:
        """
        中性化组合权重
        
        调整权重使其在市值和行业上保持中性
        
        Args:
            weights: 原始权重
            market_cap: 市值
            industry_codes: 行业代码
            
        Returns:
            中性化后的权重
        """
        # 1. 计算行业目标权重（等权）
        industry_weights = pd.Series(index=weights.index, dtype=float)
        
        for industry in industry_codes.unique():
            mask = industry_codes == industry
            n_stocks = mask.sum()
            if n_stocks > 0:
                # 该行业在组合中的目标权重 = 等权
                target_industry_weight = 1.0 / industry_codes.nunique()
                # 行业内等权
                industry_weights[mask] = target_industry_weight / n_stocks
        
        # 2. 结合原始权重和行业权重
        # 使用原始权重的排名，但调整幅度以符合行业中性
        neutral_weights = weights.copy()
        
        for industry in industry_codes.unique():
            mask = industry_codes == industry
            if mask.sum() > 0:
                # 获取该行业的原始权重
                industry_original = weights[mask]
                
                # 标准化为行业内排名权重
                ranks = industry_original.rank()
                industry_sum = weights[mask].sum()
                
                if industry_sum > 0:
                    # 保持原始权重比例，但归一化到行业目标权重
                    neutral_weights[mask] = (weights[mask] / industry_sum) * industry_weights[mask]
        
        # 3. 归一化到总和为1
        neutral_weights = neutral_weights / neutral_weights.sum()
        
        return neutral_weights
    
    def check_neutrality(self,
                        weights: pd.Series,
                        market_cap: pd.Series,
                        industry_codes: pd.Series) -> dict:
        """
        检查组合的中性化程度
        
        Args:
            weights: 组合权重
            market_cap: 市值
            industry_codes: 行业代码
            
        Returns:
            中性化检查结果
        """
        # 市值暴露
        weighted_cap = (weights * np.log(market_cap)).sum()
        avg_log_cap = np.log(market_cap).mean()
        cap_exposure = weighted_cap - avg_log_cap
        
        # 行业暴露
        industry_exposures = {}
        for industry in industry_codes.unique():
            mask = industry_codes == industry
            exposure = weights[mask].sum() - (1.0 / industry_codes.nunique())
            industry_exposures[industry] = exposure
        
        max_industry_deviation = max(abs(v) for v in industry_exposures.values())
        
        return {
            'market_cap_exposure': cap_exposure,
            'max_industry_deviation': max_industry_deviation,
            'industry_exposures': industry_exposures,
            'is_market_cap_neutral': abs(cap_exposure) < 0.1,
            'is_industry_neutral': max_industry_deviation < 0.05,
        }

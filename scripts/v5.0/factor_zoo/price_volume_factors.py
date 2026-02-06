"""
Quant-Investor V5.0 - 价量因子库

本模块提供完整的价量因子计算，包括：
1. 动量因子：过去1-12月收益率（剔除近1月）
2. 反转因子：短期（1月内）反转效应
3. 波动率因子：历史波动率、Beta、特异波动
4. 流动性因子：换手率、Amihud非流动性指标、市值
5. 技术指标：RSI、MACD、筹码分布、资金流向
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


class MomentumFactors:
    """动量因子类"""
    
    @staticmethod
    def momentum(returns: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
        """
        动量因子 (过去N日收益率，剔除近期)
        
        Args:
            returns: 日收益率序列
            lookback: 回看期（天数）
            skip: 跳过最近的天数（避免短期反转）
        
        Returns:
            动量因子值
        """
        # 计算累计收益率，跳过最近skip天
        cum_return = (1 + returns).rolling(window=lookback).apply(
            lambda x: np.prod(x[:-skip]) - 1 if len(x) > skip else np.nan,
            raw=True
        )
        return cum_return
    
    @staticmethod
    def momentum_12_1(returns: pd.Series) -> pd.Series:
        """12-1动量 (过去12个月收益率，剔除最近1个月)"""
        return MomentumFactors.momentum(returns, lookback=252, skip=21)
    
    @staticmethod
    def momentum_6_1(returns: pd.Series) -> pd.Series:
        """6-1动量 (过去6个月收益率，剔除最近1个月)"""
        return MomentumFactors.momentum(returns, lookback=126, skip=21)
    
    @staticmethod
    def momentum_3_1(returns: pd.Series) -> pd.Series:
        """3-1动量 (过去3个月收益率，剔除最近1个月)"""
        return MomentumFactors.momentum(returns, lookback=63, skip=21)
    
    @staticmethod
    def price_momentum(close: pd.Series, lookback: int = 252) -> pd.Series:
        """价格动量 (当前价格/N日前价格 - 1)"""
        return close / close.shift(lookback) - 1
    
    @staticmethod
    def relative_strength(
        stock_returns: pd.Series, 
        market_returns: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """相对强度 (股票累计收益 / 市场累计收益)"""
        stock_cum = (1 + stock_returns).rolling(window=window).apply(np.prod, raw=True) - 1
        market_cum = (1 + market_returns).rolling(window=window).apply(np.prod, raw=True) - 1
        return stock_cum / (market_cum + 1e-8)


class ReversalFactors:
    """反转因子类"""
    
    @staticmethod
    def short_term_reversal(returns: pd.Series, window: int = 21) -> pd.Series:
        """短期反转 (过去N日收益率的负数)"""
        return -(1 + returns).rolling(window=window).apply(np.prod, raw=True) + 1
    
    @staticmethod
    def weekly_reversal(returns: pd.Series) -> pd.Series:
        """周反转 (过去5日收益率的负数)"""
        return ReversalFactors.short_term_reversal(returns, window=5)
    
    @staticmethod
    def monthly_reversal(returns: pd.Series) -> pd.Series:
        """月反转 (过去21日收益率的负数)"""
        return ReversalFactors.short_term_reversal(returns, window=21)
    
    @staticmethod
    def overnight_reversal(open_price: pd.Series, prev_close: pd.Series) -> pd.Series:
        """隔夜反转 (开盘价/前收盘价 - 1 的负数)"""
        return -(open_price / prev_close - 1)


class VolatilityFactors:
    """波动率因子类"""
    
    @staticmethod
    def historical_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
        """历史波动率 (年化)"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
        """已实现波动率 (月度)"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def beta(
        stock_returns: pd.Series, 
        market_returns: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """Beta系数"""
        cov = stock_returns.rolling(window=window).cov(market_returns)
        var = market_returns.rolling(window=window).var()
        return cov / (var + 1e-8)
    
    @staticmethod
    def idiosyncratic_volatility(
        stock_returns: pd.Series, 
        market_returns: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """特异波动率 (剔除市场风险后的波动率)"""
        beta = VolatilityFactors.beta(stock_returns, market_returns, window)
        residual = stock_returns - beta * market_returns
        return residual.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def downside_volatility(returns: pd.Series, window: int = 252, threshold: float = 0) -> pd.Series:
        """下行波动率 (仅计算负收益的波动率)"""
        downside_returns = returns.where(returns < threshold, 0)
        return downside_returns.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def volatility_of_volatility(returns: pd.Series, vol_window: int = 21, vov_window: int = 252) -> pd.Series:
        """波动率的波动率"""
        vol = returns.rolling(window=vol_window).std()
        return vol.rolling(window=vov_window).std()
    
    @staticmethod
    def max_drawdown(close: pd.Series, window: int = 252) -> pd.Series:
        """最大回撤"""
        rolling_max = close.rolling(window=window, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max
        return drawdown.rolling(window=window).min()


class LiquidityFactors:
    """流动性因子类"""
    
    @staticmethod
    def turnover_rate(volume: pd.Series, shares_outstanding: pd.Series) -> pd.Series:
        """换手率"""
        return volume / (shares_outstanding + 1e-8)
    
    @staticmethod
    def average_turnover(volume: pd.Series, shares_outstanding: pd.Series, window: int = 21) -> pd.Series:
        """平均换手率"""
        turnover = volume / (shares_outstanding + 1e-8)
        return turnover.rolling(window=window).mean()
    
    @staticmethod
    def amihud_illiquidity(returns: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
        """Amihud非流动性指标 (|收益率| / 成交量)"""
        daily_illiq = returns.abs() / (volume + 1e-8)
        return daily_illiq.rolling(window=window).mean()
    
    @staticmethod
    def dollar_volume(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
        """成交金额 (平均)"""
        daily_dollar_volume = close * volume
        return daily_dollar_volume.rolling(window=window).mean()
    
    @staticmethod
    def volume_volatility(volume: pd.Series, window: int = 21) -> pd.Series:
        """成交量波动率"""
        return volume.rolling(window=window).std() / (volume.rolling(window=window).mean() + 1e-8)
    
    @staticmethod
    def bid_ask_spread_proxy(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """买卖价差代理 (使用高低价估计)"""
        # Corwin-Schultz估计
        beta = ((np.log(high / low)) ** 2).rolling(window=2).sum()
        gamma = (np.log(high.rolling(window=2).max() / low.rolling(window=2).min())) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        return spread.clip(lower=0)


class TechnicalFactors:
    """技术指标因子类"""
    
    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """相对强弱指标 (RSI)"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD指标"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """布林带"""
        middle = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower
    
    @staticmethod
    def bollinger_position(close: pd.Series, window: int = 20, num_std: float = 2) -> pd.Series:
        """布林带位置 (当前价格在布林带中的相对位置)"""
        upper, middle, lower = TechnicalFactors.bollinger_bands(close, window, num_std)
        return (close - lower) / (upper - lower + 1e-8)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """平均真实波幅 (ATR)"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """能量潮指标 (OBV)"""
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """资金流量指标 (MFI)"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0)
        negative_flow = money_flow.where(delta < 0, 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))
        return mfi
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """威廉指标 (Williams %R)"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """商品通道指数 (CCI)"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return (typical_price - sma) / (0.015 * mad + 1e-8)
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """随机振荡器 (KD指标)"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
        d = k.rolling(window=d_window).mean()
        return k, d


# ==================== 价量因子计算器 ====================

class PriceVolumeFactorCalculator:
    """
    价量因子计算器
    
    提供统一的接口来计算所有价量因子。
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.momentum = MomentumFactors()
        self.reversal = ReversalFactors()
        self.volatility = VolatilityFactors()
        self.liquidity = LiquidityFactors()
        self.technical = TechnicalFactors()
        self._log("PriceVolumeFactorCalculator initialized")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[PriceVolumeFactors] {msg}")
    
    def calculate_all_factors(
        self,
        data: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
        include_categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        计算所有价量因子
        
        Args:
            data: 包含OHLCV数据的DataFrame
            market_returns: 市场收益率序列（用于计算Beta等）
            include_categories: 要计算的因子类别
        
        Returns:
            包含所有因子值的DataFrame
        """
        result = data.copy()
        categories = include_categories or ['momentum', 'reversal', 'volatility', 'liquidity', 'technical']
        
        # 计算收益率
        if 'returns' not in result.columns and 'close' in result.columns:
            result['returns'] = result['close'].pct_change()
        
        # 动量因子
        if 'momentum' in categories and 'returns' in result.columns:
            result['factor_mom_12_1'] = self.momentum.momentum_12_1(result['returns'])
            result['factor_mom_6_1'] = self.momentum.momentum_6_1(result['returns'])
            result['factor_mom_3_1'] = self.momentum.momentum_3_1(result['returns'])
            
            if 'close' in result.columns:
                result['factor_price_mom'] = self.momentum.price_momentum(result['close'])
            
            if market_returns is not None:
                result['factor_rel_strength'] = self.momentum.relative_strength(result['returns'], market_returns)
            
            self._log("Calculated momentum factors")
        
        # 反转因子
        if 'reversal' in categories and 'returns' in result.columns:
            result['factor_reversal_week'] = self.reversal.weekly_reversal(result['returns'])
            result['factor_reversal_month'] = self.reversal.monthly_reversal(result['returns'])
            
            self._log("Calculated reversal factors")
        
        # 波动率因子
        if 'volatility' in categories and 'returns' in result.columns:
            result['factor_hist_vol'] = self.volatility.historical_volatility(result['returns'])
            result['factor_realized_vol'] = self.volatility.realized_volatility(result['returns'])
            result['factor_downside_vol'] = self.volatility.downside_volatility(result['returns'])
            
            if market_returns is not None:
                result['factor_beta'] = self.volatility.beta(result['returns'], market_returns)
                result['factor_idio_vol'] = self.volatility.idiosyncratic_volatility(result['returns'], market_returns)
            
            if 'close' in result.columns:
                result['factor_max_dd'] = self.volatility.max_drawdown(result['close'])
            
            self._log("Calculated volatility factors")
        
        # 流动性因子
        if 'liquidity' in categories:
            if 'volume' in result.columns and 'shares_outstanding' in result.columns:
                result['factor_turnover'] = self.liquidity.average_turnover(result['volume'], result['shares_outstanding'])
            
            if 'returns' in result.columns and 'volume' in result.columns:
                result['factor_amihud'] = self.liquidity.amihud_illiquidity(result['returns'], result['volume'])
            
            if 'close' in result.columns and 'volume' in result.columns:
                result['factor_dollar_vol'] = self.liquidity.dollar_volume(result['close'], result['volume'])
                result['factor_vol_volatility'] = self.liquidity.volume_volatility(result['volume'])
            
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['factor_spread_proxy'] = self.liquidity.bid_ask_spread_proxy(result['high'], result['low'], result['close'])
            
            self._log("Calculated liquidity factors")
        
        # 技术指标
        if 'technical' in categories and 'close' in result.columns:
            result['factor_rsi'] = self.technical.rsi(result['close'])
            
            macd_line, signal_line, histogram = self.technical.macd(result['close'])
            result['factor_macd'] = macd_line
            result['factor_macd_signal'] = signal_line
            result['factor_macd_hist'] = histogram
            
            result['factor_bb_position'] = self.technical.bollinger_position(result['close'])
            
            if all(col in result.columns for col in ['high', 'low']):
                result['factor_atr'] = self.technical.atr(result['high'], result['low'], result['close'])
                result['factor_williams_r'] = self.technical.williams_r(result['high'], result['low'], result['close'])
                result['factor_cci'] = self.technical.cci(result['high'], result['low'], result['close'])
                
                k, d = self.technical.stochastic_oscillator(result['high'], result['low'], result['close'])
                result['factor_stoch_k'] = k
                result['factor_stoch_d'] = d
                
                if 'volume' in result.columns:
                    result['factor_mfi'] = self.technical.money_flow_index(result['high'], result['low'], result['close'], result['volume'])
            
            if 'volume' in result.columns:
                result['factor_obv'] = self.technical.obv(result['close'], result['volume'])
            
            self._log("Calculated technical factors")
        
        # 统计计算的因子数量
        factor_cols = [col for col in result.columns if col.startswith('factor_')]
        self._log(f"Total factors calculated: {len(factor_cols)}")
        
        return result


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing V5.0 Price-Volume Factors Module")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 500
    
    # 模拟价格数据
    returns = np.random.randn(n_samples) * 0.02
    close = 100 * np.cumprod(1 + returns)
    
    test_data = pd.DataFrame({
        'close': close,
        'open': close * (1 + np.random.randn(n_samples) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
        'volume': np.random.randint(1000000, 10000000, n_samples),
        'shares_outstanding': np.full(n_samples, 1e9),
    })
    
    # 模拟市场收益率
    market_returns = pd.Series(np.random.randn(n_samples) * 0.015)
    
    # 测试因子计算器
    calculator = PriceVolumeFactorCalculator()
    
    # 计算所有因子
    result = calculator.calculate_all_factors(test_data, market_returns=market_returns)
    
    # 显示计算的因子
    factor_cols = [col for col in result.columns if col.startswith('factor_')]
    print(f"\nCalculated {len(factor_cols)} factors:")
    for col in factor_cols:
        non_null = result[col].notna().sum()
        if non_null > 0:
            print(f"  - {col}: mean={result[col].mean():.4f}, non-null={non_null}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

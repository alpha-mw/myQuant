"""
Quant-Investor V5.0 - 宏观因子与另类因子库

本模块提供完整的宏观因子和另类因子计算，包括：
1. 宏观因子：利率敏感度、通胀暴露、汇率暴露、经济周期
2. 另类因子：情绪因子、ESG因子、事件因子
3. 行业因子：行业动量、行业集中度
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple


# ==================== 宏观因子 ====================

class MacroFactors:
    """宏观因子类"""
    
    @staticmethod
    def interest_rate_sensitivity(
        stock_returns: pd.Series, 
        rate_changes: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """
        利率敏感度 (股票收益与利率变化的相关性)
        
        Args:
            stock_returns: 股票收益率序列
            rate_changes: 利率变化序列
            window: 滚动窗口
        
        Returns:
            利率敏感度因子
        """
        return stock_returns.rolling(window=window).corr(rate_changes)
    
    @staticmethod
    def duration_factor(
        stock_returns: pd.Series, 
        bond_returns: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """
        久期因子 (股票收益与债券收益的Beta)
        
        衡量股票对利率风险的暴露程度
        """
        cov = stock_returns.rolling(window=window).cov(bond_returns)
        var = bond_returns.rolling(window=window).var()
        return cov / (var + 1e-8)
    
    @staticmethod
    def inflation_beta(
        stock_returns: pd.Series, 
        inflation_changes: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """
        通胀Beta (股票收益与通胀变化的敏感度)
        """
        cov = stock_returns.rolling(window=window).cov(inflation_changes)
        var = inflation_changes.rolling(window=window).var()
        return cov / (var + 1e-8)
    
    @staticmethod
    def real_rate_sensitivity(
        stock_returns: pd.Series, 
        real_rate_changes: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """
        实际利率敏感度
        """
        return stock_returns.rolling(window=window).corr(real_rate_changes)
    
    @staticmethod
    def fx_sensitivity(
        stock_returns: pd.Series, 
        fx_returns: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """
        汇率敏感度 (股票收益与汇率变化的Beta)
        
        用于识别进出口企业的汇率风险暴露
        """
        cov = stock_returns.rolling(window=window).cov(fx_returns)
        var = fx_returns.rolling(window=window).var()
        return cov / (var + 1e-8)
    
    @staticmethod
    def commodity_beta(
        stock_returns: pd.Series, 
        commodity_returns: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """
        大宗商品Beta (股票收益与商品价格的敏感度)
        """
        cov = stock_returns.rolling(window=window).cov(commodity_returns)
        var = commodity_returns.rolling(window=window).var()
        return cov / (var + 1e-8)
    
    @staticmethod
    def economic_cycle_beta(
        stock_returns: pd.Series, 
        gdp_growth: pd.Series, 
        window: int = 12  # 季度数据
    ) -> pd.Series:
        """
        经济周期Beta (股票收益与GDP增长的敏感度)
        """
        cov = stock_returns.rolling(window=window).cov(gdp_growth)
        var = gdp_growth.rolling(window=window).var()
        return cov / (var + 1e-8)
    
    @staticmethod
    def credit_spread_sensitivity(
        stock_returns: pd.Series, 
        credit_spread_changes: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """
        信用利差敏感度 (股票收益与信用利差变化的相关性)
        """
        return stock_returns.rolling(window=window).corr(credit_spread_changes)
    
    @staticmethod
    def vix_sensitivity(
        stock_returns: pd.Series, 
        vix_changes: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """
        VIX敏感度 (股票收益与VIX变化的相关性)
        
        衡量股票对市场恐慌情绪的敏感度
        """
        return stock_returns.rolling(window=window).corr(vix_changes)


# ==================== 另类因子 ====================

class SentimentFactors:
    """情绪因子类"""
    
    @staticmethod
    def analyst_revision(
        current_estimate: pd.Series, 
        previous_estimate: pd.Series
    ) -> pd.Series:
        """
        分析师预期修正 (当前预期 / 前期预期 - 1)
        """
        return current_estimate / (previous_estimate + 1e-8) - 1
    
    @staticmethod
    def earnings_surprise(
        actual_eps: pd.Series, 
        estimated_eps: pd.Series
    ) -> pd.Series:
        """
        盈利惊喜 (实际EPS - 预期EPS) / |预期EPS|
        """
        return (actual_eps - estimated_eps) / (estimated_eps.abs() + 1e-8)
    
    @staticmethod
    def standardized_unexpected_earnings(
        actual_eps: pd.Series, 
        estimated_eps: pd.Series, 
        eps_std: pd.Series
    ) -> pd.Series:
        """
        标准化意外盈利 (SUE)
        """
        return (actual_eps - estimated_eps) / (eps_std + 1e-8)
    
    @staticmethod
    def analyst_coverage(num_analysts: pd.Series) -> pd.Series:
        """
        分析师覆盖度 (跟踪分析师数量)
        """
        return num_analysts
    
    @staticmethod
    def analyst_dispersion(
        high_estimate: pd.Series, 
        low_estimate: pd.Series, 
        mean_estimate: pd.Series
    ) -> pd.Series:
        """
        分析师分歧度 (预期范围 / 平均预期)
        """
        return (high_estimate - low_estimate) / (mean_estimate.abs() + 1e-8)
    
    @staticmethod
    def short_interest_ratio(
        short_interest: pd.Series, 
        shares_outstanding: pd.Series
    ) -> pd.Series:
        """
        做空比率 (做空股数 / 流通股数)
        """
        return short_interest / (shares_outstanding + 1e-8)
    
    @staticmethod
    def days_to_cover(
        short_interest: pd.Series, 
        avg_daily_volume: pd.Series
    ) -> pd.Series:
        """
        空头回补天数 (做空股数 / 日均成交量)
        """
        return short_interest / (avg_daily_volume + 1e-8)
    
    @staticmethod
    def put_call_ratio(put_volume: pd.Series, call_volume: pd.Series) -> pd.Series:
        """
        看跌/看涨比率
        """
        return put_volume / (call_volume + 1e-8)
    
    @staticmethod
    def news_sentiment_score(
        positive_mentions: pd.Series, 
        negative_mentions: pd.Series, 
        total_mentions: pd.Series
    ) -> pd.Series:
        """
        新闻情绪得分 (正面提及 - 负面提及) / 总提及
        """
        return (positive_mentions - negative_mentions) / (total_mentions + 1e-8)
    
    @staticmethod
    def social_media_sentiment(
        bullish_posts: pd.Series, 
        bearish_posts: pd.Series
    ) -> pd.Series:
        """
        社交媒体情绪 (看多帖子 - 看空帖子) / 总帖子
        """
        total = bullish_posts + bearish_posts
        return (bullish_posts - bearish_posts) / (total + 1e-8)


class ESGFactors:
    """ESG因子类"""
    
    @staticmethod
    def esg_score(
        environmental: pd.Series, 
        social: pd.Series, 
        governance: pd.Series,
        weights: Tuple[float, float, float] = (0.33, 0.33, 0.34)
    ) -> pd.Series:
        """
        综合ESG得分 (加权平均)
        """
        return (environmental * weights[0] + 
                social * weights[1] + 
                governance * weights[2])
    
    @staticmethod
    def environmental_score(carbon_intensity: pd.Series, energy_efficiency: pd.Series) -> pd.Series:
        """
        环境得分 (碳强度和能源效率的综合)
        """
        # 碳强度越低越好，取负数
        normalized_carbon = -carbon_intensity / (carbon_intensity.max() + 1e-8)
        normalized_energy = energy_efficiency / (energy_efficiency.max() + 1e-8)
        return (normalized_carbon + normalized_energy) / 2
    
    @staticmethod
    def governance_score(
        board_independence: pd.Series, 
        audit_quality: pd.Series,
        executive_compensation_alignment: pd.Series
    ) -> pd.Series:
        """
        治理得分
        """
        return (board_independence + audit_quality + executive_compensation_alignment) / 3
    
    @staticmethod
    def controversy_score(num_controversies: pd.Series, severity: pd.Series) -> pd.Series:
        """
        争议得分 (负面，越低越好)
        """
        return -(num_controversies * severity)
    
    @staticmethod
    def carbon_intensity(carbon_emissions: pd.Series, revenue: pd.Series) -> pd.Series:
        """
        碳强度 (碳排放 / 营收)
        """
        return carbon_emissions / (revenue + 1e-8)
    
    @staticmethod
    def green_revenue_ratio(green_revenue: pd.Series, total_revenue: pd.Series) -> pd.Series:
        """
        绿色收入比例
        """
        return green_revenue / (total_revenue + 1e-8)


class EventFactors:
    """事件因子类"""
    
    @staticmethod
    def earnings_announcement_return(
        price_after: pd.Series, 
        price_before: pd.Series
    ) -> pd.Series:
        """
        财报公告收益率
        """
        return price_after / price_before - 1
    
    @staticmethod
    def post_earnings_drift(
        earnings_surprise: pd.Series, 
        cumulative_return: pd.Series
    ) -> pd.Series:
        """
        盈余公告后漂移 (PEAD)
        """
        # 正向盈利惊喜后的累计收益
        return earnings_surprise * cumulative_return
    
    @staticmethod
    def lockup_expiration_effect(days_to_lockup: pd.Series) -> pd.Series:
        """
        解禁效应 (距离解禁日的天数)
        
        解禁前可能有卖压
        """
        # 解禁前30天内为负，表示潜在卖压
        return np.where(
            (days_to_lockup > 0) & (days_to_lockup <= 30),
            -1 / (days_to_lockup + 1),
            0
        )
    
    @staticmethod
    def buyback_signal(
        buyback_amount: pd.Series, 
        market_cap: pd.Series
    ) -> pd.Series:
        """
        回购信号 (回购金额 / 市值)
        """
        return buyback_amount / (market_cap + 1e-8)
    
    @staticmethod
    def insider_trading_signal(
        insider_buys: pd.Series, 
        insider_sells: pd.Series
    ) -> pd.Series:
        """
        内部人交易信号 (买入 - 卖出) / (买入 + 卖出)
        """
        total = insider_buys + insider_sells
        return (insider_buys - insider_sells) / (total + 1e-8)
    
    @staticmethod
    def dividend_announcement_effect(
        dividend_change: pd.Series, 
        previous_dividend: pd.Series
    ) -> pd.Series:
        """
        股息公告效应 (股息变化率)
        """
        return dividend_change / (previous_dividend + 1e-8)
    
    @staticmethod
    def stock_split_effect(split_ratio: pd.Series) -> pd.Series:
        """
        股票拆分效应
        """
        return np.log(split_ratio + 1)
    
    @staticmethod
    def index_inclusion_effect(
        days_to_inclusion: pd.Series, 
        is_addition: pd.Series
    ) -> pd.Series:
        """
        指数纳入效应
        
        即将被纳入指数的股票可能有买入压力
        """
        effect = np.where(
            (days_to_inclusion > 0) & (days_to_inclusion <= 30),
            1 / (days_to_inclusion + 1),
            0
        )
        return effect * is_addition


# ==================== 行业因子 ====================

class IndustryFactors:
    """行业因子类"""
    
    @staticmethod
    def industry_momentum(
        industry_returns: pd.DataFrame, 
        window: int = 252
    ) -> pd.DataFrame:
        """
        行业动量 (行业过去N日累计收益)
        """
        return (1 + industry_returns).rolling(window=window).apply(np.prod, raw=True) - 1
    
    @staticmethod
    def industry_relative_strength(
        stock_returns: pd.Series, 
        industry_returns: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """
        行业相对强度 (股票收益 / 行业收益)
        """
        stock_cum = (1 + stock_returns).rolling(window=window).apply(np.prod, raw=True) - 1
        industry_cum = (1 + industry_returns).rolling(window=window).apply(np.prod, raw=True) - 1
        return stock_cum / (industry_cum + 1e-8)
    
    @staticmethod
    def industry_concentration(market_caps: pd.Series) -> float:
        """
        行业集中度 (HHI指数)
        """
        total_market_cap = market_caps.sum()
        market_shares = market_caps / total_market_cap
        hhi = (market_shares ** 2).sum()
        return hhi
    
    @staticmethod
    def industry_rotation_signal(
        industry_returns: pd.DataFrame, 
        economic_indicator: pd.Series,
        window: int = 63
    ) -> pd.DataFrame:
        """
        行业轮动信号 (基于经济指标的行业敏感度)
        """
        sensitivities = {}
        for col in industry_returns.columns:
            cov = industry_returns[col].rolling(window=window).cov(economic_indicator)
            var = economic_indicator.rolling(window=window).var()
            sensitivities[col] = cov / (var + 1e-8)
        return pd.DataFrame(sensitivities)
    
    @staticmethod
    def industry_dispersion(industry_returns: pd.DataFrame) -> pd.Series:
        """
        行业离散度 (行业收益的标准差)
        """
        return industry_returns.std(axis=1)


# ==================== 宏观另类因子计算器 ====================

class MacroAlternativeFactorCalculator:
    """
    宏观和另类因子计算器
    
    提供统一的接口来计算所有宏观和另类因子。
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.macro = MacroFactors()
        self.sentiment = SentimentFactors()
        self.esg = ESGFactors()
        self.event = EventFactors()
        self.industry = IndustryFactors()
        self._log("MacroAlternativeFactorCalculator initialized")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[MacroAltFactors] {msg}")
    
    def calculate_macro_factors(
        self,
        stock_returns: pd.Series,
        macro_data: Dict[str, pd.Series],
        window: int = 252
    ) -> pd.DataFrame:
        """
        计算宏观因子
        
        Args:
            stock_returns: 股票收益率序列
            macro_data: 宏观数据字典，可包含：
                - 'rate_changes': 利率变化
                - 'inflation': 通胀变化
                - 'fx_returns': 汇率收益
                - 'vix_changes': VIX变化
            window: 滚动窗口
        
        Returns:
            宏观因子DataFrame
        """
        result = pd.DataFrame(index=stock_returns.index)
        
        if 'rate_changes' in macro_data:
            result['factor_rate_sensitivity'] = self.macro.interest_rate_sensitivity(
                stock_returns, macro_data['rate_changes'], window
            )
        
        if 'inflation' in macro_data:
            result['factor_inflation_beta'] = self.macro.inflation_beta(
                stock_returns, macro_data['inflation'], window
            )
        
        if 'fx_returns' in macro_data:
            result['factor_fx_sensitivity'] = self.macro.fx_sensitivity(
                stock_returns, macro_data['fx_returns'], window
            )
        
        if 'vix_changes' in macro_data:
            result['factor_vix_sensitivity'] = self.macro.vix_sensitivity(
                stock_returns, macro_data['vix_changes'], window
            )
        
        if 'credit_spread' in macro_data:
            result['factor_credit_sensitivity'] = self.macro.credit_spread_sensitivity(
                stock_returns, macro_data['credit_spread'], window
            )
        
        self._log(f"Calculated {len(result.columns)} macro factors")
        return result
    
    def calculate_sentiment_factors(
        self,
        sentiment_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        计算情绪因子
        
        Args:
            sentiment_data: 情绪数据字典
        
        Returns:
            情绪因子DataFrame
        """
        result = pd.DataFrame()
        
        if 'current_estimate' in sentiment_data and 'previous_estimate' in sentiment_data:
            result['factor_analyst_revision'] = self.sentiment.analyst_revision(
                sentiment_data['current_estimate'], 
                sentiment_data['previous_estimate']
            )
        
        if 'actual_eps' in sentiment_data and 'estimated_eps' in sentiment_data:
            result['factor_earnings_surprise'] = self.sentiment.earnings_surprise(
                sentiment_data['actual_eps'], 
                sentiment_data['estimated_eps']
            )
        
        if 'short_interest' in sentiment_data and 'shares_outstanding' in sentiment_data:
            result['factor_short_ratio'] = self.sentiment.short_interest_ratio(
                sentiment_data['short_interest'], 
                sentiment_data['shares_outstanding']
            )
        
        if 'num_analysts' in sentiment_data:
            result['factor_analyst_coverage'] = self.sentiment.analyst_coverage(
                sentiment_data['num_analysts']
            )
        
        self._log(f"Calculated {len(result.columns)} sentiment factors")
        return result
    
    def calculate_esg_factors(
        self,
        esg_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        计算ESG因子
        
        Args:
            esg_data: ESG数据字典
        
        Returns:
            ESG因子DataFrame
        """
        result = pd.DataFrame()
        
        if all(k in esg_data for k in ['environmental', 'social', 'governance']):
            result['factor_esg_score'] = self.esg.esg_score(
                esg_data['environmental'],
                esg_data['social'],
                esg_data['governance']
            )
        
        if 'carbon_emissions' in esg_data and 'revenue' in esg_data:
            result['factor_carbon_intensity'] = self.esg.carbon_intensity(
                esg_data['carbon_emissions'],
                esg_data['revenue']
            )
        
        if 'green_revenue' in esg_data and 'total_revenue' in esg_data:
            result['factor_green_ratio'] = self.esg.green_revenue_ratio(
                esg_data['green_revenue'],
                esg_data['total_revenue']
            )
        
        self._log(f"Calculated {len(result.columns)} ESG factors")
        return result
    
    def get_factor_list(self) -> Dict[str, List[str]]:
        """获取所有可用因子的列表"""
        return {
            'macro': [
                'interest_rate_sensitivity', 'duration_factor', 'inflation_beta',
                'real_rate_sensitivity', 'fx_sensitivity', 'commodity_beta',
                'economic_cycle_beta', 'credit_spread_sensitivity', 'vix_sensitivity'
            ],
            'sentiment': [
                'analyst_revision', 'earnings_surprise', 'standardized_unexpected_earnings',
                'analyst_coverage', 'analyst_dispersion', 'short_interest_ratio',
                'days_to_cover', 'put_call_ratio', 'news_sentiment_score', 'social_media_sentiment'
            ],
            'esg': [
                'esg_score', 'environmental_score', 'governance_score',
                'controversy_score', 'carbon_intensity', 'green_revenue_ratio'
            ],
            'event': [
                'earnings_announcement_return', 'post_earnings_drift', 'lockup_expiration_effect',
                'buyback_signal', 'insider_trading_signal', 'dividend_announcement_effect',
                'stock_split_effect', 'index_inclusion_effect'
            ],
            'industry': [
                'industry_momentum', 'industry_relative_strength', 'industry_concentration',
                'industry_rotation_signal', 'industry_dispersion'
            ]
        }


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing V5.0 Macro & Alternative Factors Module")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 252
    
    # 模拟股票收益率
    stock_returns = pd.Series(np.random.randn(n_samples) * 0.02)
    
    # 模拟宏观数据
    macro_data = {
        'rate_changes': pd.Series(np.random.randn(n_samples) * 0.001),
        'inflation': pd.Series(np.random.randn(n_samples) * 0.002),
        'fx_returns': pd.Series(np.random.randn(n_samples) * 0.005),
        'vix_changes': pd.Series(np.random.randn(n_samples) * 0.1),
    }
    
    # 模拟情绪数据
    sentiment_data = {
        'current_estimate': pd.Series(np.random.uniform(1, 5, n_samples)),
        'previous_estimate': pd.Series(np.random.uniform(1, 5, n_samples)),
        'actual_eps': pd.Series(np.random.uniform(1, 5, n_samples)),
        'estimated_eps': pd.Series(np.random.uniform(1, 5, n_samples)),
        'num_analysts': pd.Series(np.random.randint(1, 30, n_samples)),
    }
    
    # 模拟ESG数据
    esg_data = {
        'environmental': pd.Series(np.random.uniform(0, 100, n_samples)),
        'social': pd.Series(np.random.uniform(0, 100, n_samples)),
        'governance': pd.Series(np.random.uniform(0, 100, n_samples)),
    }
    
    # 测试因子计算器
    calculator = MacroAlternativeFactorCalculator()
    
    # 计算宏观因子
    print("\n1. Testing Macro Factors...")
    macro_factors = calculator.calculate_macro_factors(stock_returns, macro_data)
    print(f"   Calculated factors: {list(macro_factors.columns)}")
    
    # 计算情绪因子
    print("\n2. Testing Sentiment Factors...")
    sentiment_factors = calculator.calculate_sentiment_factors(sentiment_data)
    print(f"   Calculated factors: {list(sentiment_factors.columns)}")
    
    # 计算ESG因子
    print("\n3. Testing ESG Factors...")
    esg_factors = calculator.calculate_esg_factors(esg_data)
    print(f"   Calculated factors: {list(esg_factors.columns)}")
    
    # 显示因子列表
    print("\n" + "=" * 60)
    print("Available Factor List:")
    print("=" * 60)
    factor_list = calculator.get_factor_list()
    for category, factors in factor_list.items():
        print(f"\n{category.upper()}:")
        for f in factors:
            print(f"  - {f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

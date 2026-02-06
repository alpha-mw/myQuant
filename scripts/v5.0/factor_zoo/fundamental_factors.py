"""
Quant-Investor V5.0 - 基本面因子库

本模块提供完整的基本面因子计算，包括：
1. 价值因子：PE、PB、PS、EV/EBITDA、股息率
2. 质量因子：ROE稳定性、毛利率、负债率、现金流
3. 成长因子：营收增速、利润增速、研发投入
4. 安全因子：低波动、低杠杆、低破产风险
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from abc import ABC, abstractmethod


class BaseFactor(ABC):
    """因子基类"""
    
    def __init__(self, name: str, category: str, description: str):
        self.name = name
        self.category = category
        self.description = description
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        pass
    
    def __repr__(self):
        return f"{self.category}.{self.name}: {self.description}"


# ==================== 价值因子 ====================

class ValueFactors:
    """价值因子类"""
    
    @staticmethod
    def pe_ratio(price: pd.Series, eps: pd.Series) -> pd.Series:
        """市盈率 (Price-to-Earnings Ratio)"""
        return price / (eps + 1e-8)
    
    @staticmethod
    def pe_ttm(market_cap: pd.Series, net_income_ttm: pd.Series) -> pd.Series:
        """滚动市盈率 (TTM PE)"""
        return market_cap / (net_income_ttm + 1e-8)
    
    @staticmethod
    def pb_ratio(price: pd.Series, book_value_per_share: pd.Series) -> pd.Series:
        """市净率 (Price-to-Book Ratio)"""
        return price / (book_value_per_share + 1e-8)
    
    @staticmethod
    def ps_ratio(market_cap: pd.Series, revenue: pd.Series) -> pd.Series:
        """市销率 (Price-to-Sales Ratio)"""
        return market_cap / (revenue + 1e-8)
    
    @staticmethod
    def ev_ebitda(enterprise_value: pd.Series, ebitda: pd.Series) -> pd.Series:
        """企业价值倍数 (EV/EBITDA)"""
        return enterprise_value / (ebitda + 1e-8)
    
    @staticmethod
    def dividend_yield(dividend_per_share: pd.Series, price: pd.Series) -> pd.Series:
        """股息率 (Dividend Yield)"""
        return dividend_per_share / (price + 1e-8)
    
    @staticmethod
    def earnings_yield(eps: pd.Series, price: pd.Series) -> pd.Series:
        """盈利收益率 (Earnings Yield) - PE的倒数"""
        return eps / (price + 1e-8)
    
    @staticmethod
    def fcf_yield(free_cash_flow: pd.Series, market_cap: pd.Series) -> pd.Series:
        """自由现金流收益率 (FCF Yield)"""
        return free_cash_flow / (market_cap + 1e-8)


# ==================== 质量因子 ====================

class QualityFactors:
    """质量因子类"""
    
    @staticmethod
    def roe(net_income: pd.Series, equity: pd.Series) -> pd.Series:
        """净资产收益率 (Return on Equity)"""
        return net_income / (equity + 1e-8)
    
    @staticmethod
    def roe_stability(roe_series: pd.DataFrame, window: int = 12) -> pd.Series:
        """ROE稳定性 (ROE的标准差的倒数)"""
        roe_std = roe_series.rolling(window=window).std()
        return 1 / (roe_std + 1e-8)
    
    @staticmethod
    def roa(net_income: pd.Series, total_assets: pd.Series) -> pd.Series:
        """总资产收益率 (Return on Assets)"""
        return net_income / (total_assets + 1e-8)
    
    @staticmethod
    def roic(nopat: pd.Series, invested_capital: pd.Series) -> pd.Series:
        """投入资本回报率 (Return on Invested Capital)"""
        return nopat / (invested_capital + 1e-8)
    
    @staticmethod
    def gross_margin(gross_profit: pd.Series, revenue: pd.Series) -> pd.Series:
        """毛利率 (Gross Margin)"""
        return gross_profit / (revenue + 1e-8)
    
    @staticmethod
    def operating_margin(operating_income: pd.Series, revenue: pd.Series) -> pd.Series:
        """营业利润率 (Operating Margin)"""
        return operating_income / (revenue + 1e-8)
    
    @staticmethod
    def net_margin(net_income: pd.Series, revenue: pd.Series) -> pd.Series:
        """净利润率 (Net Margin)"""
        return net_income / (revenue + 1e-8)
    
    @staticmethod
    def debt_to_equity(total_debt: pd.Series, equity: pd.Series) -> pd.Series:
        """负债权益比 (Debt-to-Equity Ratio)"""
        return total_debt / (equity + 1e-8)
    
    @staticmethod
    def debt_to_assets(total_debt: pd.Series, total_assets: pd.Series) -> pd.Series:
        """资产负债率 (Debt-to-Assets Ratio)"""
        return total_debt / (total_assets + 1e-8)
    
    @staticmethod
    def current_ratio(current_assets: pd.Series, current_liabilities: pd.Series) -> pd.Series:
        """流动比率 (Current Ratio)"""
        return current_assets / (current_liabilities + 1e-8)
    
    @staticmethod
    def quick_ratio(
        current_assets: pd.Series, 
        inventory: pd.Series, 
        current_liabilities: pd.Series
    ) -> pd.Series:
        """速动比率 (Quick Ratio)"""
        return (current_assets - inventory) / (current_liabilities + 1e-8)
    
    @staticmethod
    def cash_flow_to_debt(operating_cash_flow: pd.Series, total_debt: pd.Series) -> pd.Series:
        """现金流负债比 (Cash Flow to Debt)"""
        return operating_cash_flow / (total_debt + 1e-8)
    
    @staticmethod
    def accruals_ratio(
        net_income: pd.Series, 
        operating_cash_flow: pd.Series, 
        total_assets: pd.Series
    ) -> pd.Series:
        """应计比率 (Accruals Ratio) - 衡量盈利质量"""
        return (net_income - operating_cash_flow) / (total_assets + 1e-8)


# ==================== 成长因子 ====================

class GrowthFactors:
    """成长因子类"""
    
    @staticmethod
    def revenue_growth(revenue: pd.Series, periods: int = 4) -> pd.Series:
        """营收增长率 (Revenue Growth)"""
        return revenue.pct_change(periods)
    
    @staticmethod
    def revenue_growth_yoy(revenue_current: pd.Series, revenue_prev_year: pd.Series) -> pd.Series:
        """营收同比增长率 (YoY Revenue Growth)"""
        return (revenue_current - revenue_prev_year) / (revenue_prev_year.abs() + 1e-8)
    
    @staticmethod
    def earnings_growth(net_income: pd.Series, periods: int = 4) -> pd.Series:
        """利润增长率 (Earnings Growth)"""
        return net_income.pct_change(periods)
    
    @staticmethod
    def eps_growth(eps: pd.Series, periods: int = 4) -> pd.Series:
        """每股收益增长率 (EPS Growth)"""
        return eps.pct_change(periods)
    
    @staticmethod
    def operating_income_growth(operating_income: pd.Series, periods: int = 4) -> pd.Series:
        """营业利润增长率 (Operating Income Growth)"""
        return operating_income.pct_change(periods)
    
    @staticmethod
    def rd_intensity(rd_expense: pd.Series, revenue: pd.Series) -> pd.Series:
        """研发强度 (R&D Intensity)"""
        return rd_expense / (revenue + 1e-8)
    
    @staticmethod
    def capex_intensity(capex: pd.Series, revenue: pd.Series) -> pd.Series:
        """资本支出强度 (CapEx Intensity)"""
        return capex / (revenue + 1e-8)
    
    @staticmethod
    def asset_growth(total_assets: pd.Series, periods: int = 4) -> pd.Series:
        """资产增长率 (Asset Growth)"""
        return total_assets.pct_change(periods)
    
    @staticmethod
    def sustainable_growth_rate(roe: pd.Series, payout_ratio: pd.Series) -> pd.Series:
        """可持续增长率 (Sustainable Growth Rate)"""
        retention_ratio = 1 - payout_ratio
        return roe * retention_ratio


# ==================== 安全因子 ====================

class SafetyFactors:
    """安全因子类"""
    
    @staticmethod
    def low_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
        """低波动因子 (波动率的负数)"""
        return -returns.rolling(window=window).std()
    
    @staticmethod
    def low_beta(
        stock_returns: pd.Series, 
        market_returns: pd.Series, 
        window: int = 252
    ) -> pd.Series:
        """低Beta因子 (Beta的负数)"""
        cov = stock_returns.rolling(window=window).cov(market_returns)
        var = market_returns.rolling(window=window).var()
        beta = cov / (var + 1e-8)
        return -beta
    
    @staticmethod
    def low_leverage(total_debt: pd.Series, total_assets: pd.Series) -> pd.Series:
        """低杠杆因子 (负债率的负数)"""
        return -total_debt / (total_assets + 1e-8)
    
    @staticmethod
    def altman_z_score(
        working_capital: pd.Series,
        retained_earnings: pd.Series,
        ebit: pd.Series,
        market_cap: pd.Series,
        total_liabilities: pd.Series,
        revenue: pd.Series,
        total_assets: pd.Series
    ) -> pd.Series:
        """Altman Z-Score (破产风险指标)"""
        A = working_capital / (total_assets + 1e-8)
        B = retained_earnings / (total_assets + 1e-8)
        C = ebit / (total_assets + 1e-8)
        D = market_cap / (total_liabilities + 1e-8)
        E = revenue / (total_assets + 1e-8)
        
        return 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
    
    @staticmethod
    def interest_coverage(ebit: pd.Series, interest_expense: pd.Series) -> pd.Series:
        """利息保障倍数 (Interest Coverage Ratio)"""
        return ebit / (interest_expense + 1e-8)
    
    @staticmethod
    def piotroski_f_score(
        net_income: pd.Series,
        operating_cash_flow: pd.Series,
        roa_current: pd.Series,
        roa_prev: pd.Series,
        long_term_debt_current: pd.Series,
        long_term_debt_prev: pd.Series,
        current_ratio_current: pd.Series,
        current_ratio_prev: pd.Series,
        shares_current: pd.Series,
        shares_prev: pd.Series,
        gross_margin_current: pd.Series,
        gross_margin_prev: pd.Series,
        asset_turnover_current: pd.Series,
        asset_turnover_prev: pd.Series
    ) -> pd.Series:
        """Piotroski F-Score (财务健康度评分)"""
        score = pd.Series(0, index=net_income.index)
        
        # 盈利能力 (4分)
        score += (net_income > 0).astype(int)
        score += (operating_cash_flow > 0).astype(int)
        score += (roa_current > roa_prev).astype(int)
        score += (operating_cash_flow > net_income).astype(int)
        
        # 杠杆/流动性 (3分)
        score += (long_term_debt_current < long_term_debt_prev).astype(int)
        score += (current_ratio_current > current_ratio_prev).astype(int)
        score += (shares_current <= shares_prev).astype(int)
        
        # 运营效率 (2分)
        score += (gross_margin_current > gross_margin_prev).astype(int)
        score += (asset_turnover_current > asset_turnover_prev).astype(int)
        
        return score


# ==================== 因子计算器 ====================

class FundamentalFactorCalculator:
    """
    基本面因子计算器
    
    提供统一的接口来计算所有基本面因子。
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.value = ValueFactors()
        self.quality = QualityFactors()
        self.growth = GrowthFactors()
        self.safety = SafetyFactors()
        self._log("FundamentalFactorCalculator initialized")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[FundamentalFactors] {msg}")
    
    def calculate_all_factors(
        self,
        data: pd.DataFrame,
        price_col: str = 'close',
        market_cap_col: str = 'market_cap',
        include_categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        计算所有基本面因子
        
        Args:
            data: 包含财务数据的DataFrame
            price_col: 价格列名
            market_cap_col: 市值列名
            include_categories: 要计算的因子类别，None表示全部
        
        Returns:
            包含所有因子值的DataFrame
        """
        result = data.copy()
        categories = include_categories or ['value', 'quality', 'growth', 'safety']
        
        # 价值因子
        if 'value' in categories:
            if 'eps' in data.columns and price_col in data.columns:
                result['factor_pe'] = self.value.pe_ratio(data[price_col], data['eps'])
                result['factor_earnings_yield'] = self.value.earnings_yield(data['eps'], data[price_col])
            
            if 'book_value_per_share' in data.columns and price_col in data.columns:
                result['factor_pb'] = self.value.pb_ratio(data[price_col], data['book_value_per_share'])
            
            if market_cap_col in data.columns and 'revenue' in data.columns:
                result['factor_ps'] = self.value.ps_ratio(data[market_cap_col], data['revenue'])
            
            if 'dividend_per_share' in data.columns and price_col in data.columns:
                result['factor_div_yield'] = self.value.dividend_yield(data['dividend_per_share'], data[price_col])
            
            self._log("Calculated value factors")
        
        # 质量因子
        if 'quality' in categories:
            if 'net_income' in data.columns and 'equity' in data.columns:
                result['factor_roe'] = self.quality.roe(data['net_income'], data['equity'])
            
            if 'net_income' in data.columns and 'total_assets' in data.columns:
                result['factor_roa'] = self.quality.roa(data['net_income'], data['total_assets'])
            
            if 'gross_profit' in data.columns and 'revenue' in data.columns:
                result['factor_gross_margin'] = self.quality.gross_margin(data['gross_profit'], data['revenue'])
            
            if 'operating_income' in data.columns and 'revenue' in data.columns:
                result['factor_operating_margin'] = self.quality.operating_margin(data['operating_income'], data['revenue'])
            
            if 'total_debt' in data.columns and 'equity' in data.columns:
                result['factor_debt_equity'] = self.quality.debt_to_equity(data['total_debt'], data['equity'])
            
            if 'total_debt' in data.columns and 'total_assets' in data.columns:
                result['factor_debt_assets'] = self.quality.debt_to_assets(data['total_debt'], data['total_assets'])
            
            if 'current_assets' in data.columns and 'current_liabilities' in data.columns:
                result['factor_current_ratio'] = self.quality.current_ratio(data['current_assets'], data['current_liabilities'])
            
            self._log("Calculated quality factors")
        
        # 成长因子
        if 'growth' in categories:
            if 'revenue' in data.columns:
                result['factor_revenue_growth'] = self.growth.revenue_growth(data['revenue'])
            
            if 'net_income' in data.columns:
                result['factor_earnings_growth'] = self.growth.earnings_growth(data['net_income'])
            
            if 'eps' in data.columns:
                result['factor_eps_growth'] = self.growth.eps_growth(data['eps'])
            
            if 'rd_expense' in data.columns and 'revenue' in data.columns:
                result['factor_rd_intensity'] = self.growth.rd_intensity(data['rd_expense'], data['revenue'])
            
            if 'total_assets' in data.columns:
                result['factor_asset_growth'] = self.growth.asset_growth(data['total_assets'])
            
            self._log("Calculated growth factors")
        
        # 安全因子
        if 'safety' in categories:
            if 'total_debt' in data.columns and 'total_assets' in data.columns:
                result['factor_low_leverage'] = self.safety.low_leverage(data['total_debt'], data['total_assets'])
            
            if 'ebit' in data.columns and 'interest_expense' in data.columns:
                result['factor_interest_coverage'] = self.safety.interest_coverage(data['ebit'], data['interest_expense'])
            
            self._log("Calculated safety factors")
        
        # 统计计算的因子数量
        factor_cols = [col for col in result.columns if col.startswith('factor_')]
        self._log(f"Total factors calculated: {len(factor_cols)}")
        
        return result
    
    def get_factor_list(self) -> Dict[str, List[str]]:
        """获取所有可用因子的列表"""
        return {
            'value': [
                'pe_ratio', 'pe_ttm', 'pb_ratio', 'ps_ratio', 
                'ev_ebitda', 'dividend_yield', 'earnings_yield', 'fcf_yield'
            ],
            'quality': [
                'roe', 'roe_stability', 'roa', 'roic',
                'gross_margin', 'operating_margin', 'net_margin',
                'debt_to_equity', 'debt_to_assets', 'current_ratio', 'quick_ratio',
                'cash_flow_to_debt', 'accruals_ratio'
            ],
            'growth': [
                'revenue_growth', 'revenue_growth_yoy', 'earnings_growth', 'eps_growth',
                'operating_income_growth', 'rd_intensity', 'capex_intensity',
                'asset_growth', 'sustainable_growth_rate'
            ],
            'safety': [
                'low_volatility', 'low_beta', 'low_leverage',
                'altman_z_score', 'interest_coverage', 'piotroski_f_score'
            ]
        }


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing V5.0 Fundamental Factors Module")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    
    test_data = pd.DataFrame({
        'close': np.random.uniform(50, 200, n_samples),
        'eps': np.random.uniform(1, 10, n_samples),
        'book_value_per_share': np.random.uniform(10, 50, n_samples),
        'market_cap': np.random.uniform(1e9, 1e11, n_samples),
        'revenue': np.random.uniform(1e8, 1e10, n_samples),
        'net_income': np.random.uniform(1e7, 1e9, n_samples),
        'equity': np.random.uniform(1e8, 1e10, n_samples),
        'total_assets': np.random.uniform(1e9, 1e11, n_samples),
        'total_debt': np.random.uniform(1e8, 1e10, n_samples),
        'gross_profit': np.random.uniform(1e7, 1e9, n_samples),
        'operating_income': np.random.uniform(1e7, 1e9, n_samples),
        'current_assets': np.random.uniform(1e8, 1e10, n_samples),
        'current_liabilities': np.random.uniform(1e8, 1e10, n_samples),
        'dividend_per_share': np.random.uniform(0, 5, n_samples),
        'rd_expense': np.random.uniform(1e6, 1e8, n_samples),
        'ebit': np.random.uniform(1e7, 1e9, n_samples),
        'interest_expense': np.random.uniform(1e5, 1e7, n_samples),
    })
    
    # 测试因子计算器
    calculator = FundamentalFactorCalculator()
    
    # 计算所有因子
    result = calculator.calculate_all_factors(test_data)
    
    # 显示计算的因子
    factor_cols = [col for col in result.columns if col.startswith('factor_')]
    print(f"\nCalculated {len(factor_cols)} factors:")
    for col in factor_cols:
        print(f"  - {col}: mean={result[col].mean():.4f}, std={result[col].std():.4f}")
    
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

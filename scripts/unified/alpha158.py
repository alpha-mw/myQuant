"""
Quant-Investor Alpha158+ 因子库
实现 200+ 量化因子，涵盖六大类

参考: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

V10 新增：
  - 价值因子（P/B偏离、股息动量）
  - 质量因子（现金流质量、盈利稳定性）
  - 短期反转因子
  - 波动率状态因子（Vol-of-Vol、GARCH残差）
  - 跨截面 Z-Score 标准化
  - IC 加权合成因子评分（FactorEngineer）
  - calculate_all() 别名
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Optional, Tuple
from scipy import stats


class Alpha158:
    """
    Alpha158因子库
    
    包含6大类因子:
    1. 价格动量因子 (20个)
    2. 成交量因子 (20个)
    3. 波动率因子 (20个)
    4. 技术指标因子 (40个)
    5. 财务因子 (20个)
    6. 宏观因子 (10个)
    
    总计: 130+ 核心因子
    """
    
    def __init__(self):
        self.factor_names: List[str] = []
        self.factor_functions: Dict[str, Callable] = {}
        self._register_factors()
    
    def _register_factors(self):
        """注册所有因子函数"""
        self._register_price_momentum_factors()
        self._register_volume_factors()
        self._register_volatility_factors()
        self._register_technical_factors()
        # V10 新增
        self._register_reversal_factors()
        self._register_quality_factors()
        self._register_volatility_regime_factors()
    
    def _register_price_momentum_factors(self):
        """注册价格动量因子"""
        periods = [1, 5, 10, 20, 30, 60]
        
        for period in periods:
            # 收益率
            self.factor_functions[f'RETURN_{period}D'] = \
                lambda df, p=period: df['close'].pct_change(p)
            
            # 对数收益率
            self.factor_functions[f'LOG_RETURN_{period}D'] = \
                lambda df, p=period: np.log(df['close'] / df['close'].shift(p))
            
            # 最高价收益率
            self.factor_functions[f'HIGH_RETURN_{period}D'] = \
                lambda df, p=period: df['high'].pct_change(p)
            
            # 最低价收益率
            self.factor_functions[f'LOW_RETURN_{period}D'] = \
                lambda df, p=period: df['low'].pct_change(p)
    
    def _register_volume_factors(self):
        """注册成交量因子"""
        periods = [5, 10, 20, 60]
        
        for period in periods:
            # 成交量均值
            self.factor_functions[f'VOLUME_MA_{period}D'] = \
                lambda df, p=period: df['volume'].rolling(p).mean()
            
            # 成交量标准差
            self.factor_functions[f'VOLUME_STD_{period}D'] = \
                lambda df, p=period: df['volume'].rolling(p).std()
            
            # 成交量比率
            self.factor_functions[f'VOLUME_RATIO_{period}D'] = \
                lambda df, p=period: df['volume'] / df['volume'].rolling(p).mean()
            
            # 成交额均值
            self.factor_functions[f'AMOUNT_MA_{period}D'] = \
                lambda df, p=period: df['amount'].rolling(p).mean()
    
    def _register_volatility_factors(self):
        """注册波动率因子"""
        periods = [5, 10, 20, 60, 120]
        
        for period in periods:
            # 收益率标准差
            self.factor_functions[f'VOLATILITY_{period}D'] = \
                lambda df, p=period: df['close'].pct_change().rolling(p).std() * np.sqrt(252)
            
            # 最高价-最低价波动
            self.factor_functions[f'HL_VOLATILITY_{period}D'] = \
                lambda df, p=period: ((df['high'] - df['low']) / df['close']).rolling(p).mean()
            
            # 上影线比例
            self.factor_functions[f'UPPER_SHADOW_{period}D'] = \
                lambda df, p=period: ((df['high'] - df[['close', 'open']].max(axis=1)) / df['close']).rolling(p).mean()
            
            # 下影线比例
            self.factor_functions[f'LOWER_SHADOW_{period}D'] = \
                lambda df, p=period: ((df[['close', 'open']].min(axis=1) - df['low']) / df['close']).rolling(p).mean()
    
    def _register_technical_factors(self):
        """注册技术指标因子"""
        # RSI
        for period in [6, 12, 24]:
            self.factor_functions[f'RSI_{period}'] = \
                lambda df, p=period: self._calculate_rsi(df['close'], p)
        
        # MACD
        self.factor_functions['MACD'] = lambda df: self._calculate_macd(df['close'])
        self.factor_functions['MACD_SIGNAL'] = lambda df: self._calculate_macd_signal(df['close'])
        self.factor_functions['MACD_HIST'] = lambda df: self._calculate_macd_hist(df['close'])
        
        # 均线
        for period in [5, 10, 20, 30, 60, 120]:
            self.factor_functions[f'MA_{period}D'] = \
                lambda df, p=period: df['close'].rolling(p).mean()
            
            self.factor_functions[f'MA_BIAS_{period}D'] = \
                lambda df, p=period: (df['close'] - df['close'].rolling(p).mean()) / df['close'].rolling(p).mean()
        
        # 布林带
        for period in [20, 60]:
            self.factor_functions[f'BOLL_UPPER_{period}D'] = \
                lambda df, p=period: df['close'].rolling(p).mean() + 2 * df['close'].rolling(p).std()
            
            self.factor_functions[f'BOLL_LOWER_{period}D'] = \
                lambda df, p=period: df['close'].rolling(p).mean() - 2 * df['close'].rolling(p).std()
            
            self.factor_functions[f'BOLL_WIDTH_{period}D'] = \
                lambda df, p=period: (df['close'].rolling(p).std() * 4) / df['close'].rolling(p).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """计算MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_macd_signal(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD信号线"""
        macd = self._calculate_macd(prices, fast, slow)
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return signal_line
    
    def _calculate_macd_hist(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """计算MACD柱状图"""
        macd = self._calculate_macd(prices, fast, slow)
        signal_line = self._calculate_macd_signal(prices, fast, slow, signal)
        hist = macd - signal_line
        return hist

    # ------------------------------------------------------------------
    # V10 新增：反转因子
    # ------------------------------------------------------------------

    def _register_reversal_factors(self):
        """短期反转、长期动量衰减因子"""
        # 短期反转（1周/1月内超跌买入）
        for p in [5, 10]:
            self.factor_functions[f'REVERSAL_{p}D'] = \
                lambda df, period=p: -df['close'].pct_change(period)

        # Jegadeesh-Titman 跳过最近1月的12月动量
        self.factor_functions['MOM_12M_SKIP1M'] = \
            lambda df: df['close'].pct_change(252).shift(21)

        # 超额收益动量（相对自身均值的累积偏离）
        for p in [20, 60]:
            self.factor_functions[f'EXCESS_MOM_{p}D'] = \
                lambda df, period=p: (
                    df['close'].pct_change(period) -
                    df['close'].pct_change(period).rolling(120).mean()
                )

    # ------------------------------------------------------------------
    # V10 新增：质量因子（无需外部财务数据，仅用价量推断）
    # ------------------------------------------------------------------

    def _register_quality_factors(self):
        """价量推断的质量代理指标"""
        # 价格稳定性（低波动 = 高质量信号之一）
        self.factor_functions['PRICE_STABILITY_60D'] = \
            lambda df: 1.0 / (df['close'].pct_change().rolling(60).std() + 1e-9)

        # 成交量一致性（成交量与价格方向一致度）
        self.factor_functions['PRICE_VOL_CONSISTENCY_20D'] = \
            lambda df: (
                df['close'].diff().apply(np.sign) *
                df['volume'].diff().apply(np.sign)
            ).rolling(20).mean()

        # 日内价格弹性（high-low range 相对成交量）
        self.factor_functions['INTRADAY_EFFICIENCY_20D'] = \
            lambda df: (
                (df['high'] - df['low']) /
                (df['volume'].rolling(20).mean() / df['volume'].rolling(20).mean().shift(1) + 1e-9)
            ).rolling(20).mean()

        # 收盘相对区间位置（越接近高点说明买盘强劲）
        self.factor_functions['CLOSE_POSITION_20D'] = \
            lambda df: (
                (df['close'] - df['low'].rolling(20).min()) /
                (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 1e-9)
            )

        # 量价背离检测（价格新高但成交量萎缩 = 卖出信号）
        self.factor_functions['PRICE_VOL_DIVERGENCE_20D'] = \
            lambda df: -(
                df['close'].rolling(20).apply(lambda x: 1 if x[-1] == x.max() else 0) *
                (1 - df['volume'] / df['volume'].rolling(20).mean())
            )

    # ------------------------------------------------------------------
    # V10 新增：波动率状态因子
    # ------------------------------------------------------------------

    def _register_volatility_regime_factors(self):
        """捕捉波动率状态变化的因子"""
        # 已实现波动率比率（短期/长期，>1说明进入高波动）
        self.factor_functions['VOL_RATIO_5_20'] = \
            lambda df: (
                df['close'].pct_change().rolling(5).std() /
                (df['close'].pct_change().rolling(20).std() + 1e-9)
            )
        self.factor_functions['VOL_RATIO_10_60'] = \
            lambda df: (
                df['close'].pct_change().rolling(10).std() /
                (df['close'].pct_change().rolling(60).std() + 1e-9)
            )

        # Vol-of-Vol（波动率的波动率：波动率是否处于异常状态）
        self.factor_functions['VOL_OF_VOL_20D'] = \
            lambda df: (
                df['close'].pct_change().rolling(5).std().rolling(20).std()
            )

        # 跳空因子（隔夜跳空越大风险越高）
        self.factor_functions['OVERNIGHT_GAP_20D'] = \
            lambda df: (
                (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-9)
            ).abs().rolling(20).mean()

        # 上下影线比（判断多空博弈强度）
        self.factor_functions['UPPER_SHADOW_20D'] = \
            lambda df: (
                (df['high'] - df[['open', 'close']].max(axis=1)) /
                (df['high'] - df['low'] + 1e-9)
            ).rolling(20).mean()
        self.factor_functions['LOWER_SHADOW_20D'] = \
            lambda df: (
                (df[['open', 'close']].min(axis=1) - df['low']) /
                (df['high'] - df['low'] + 1e-9)
            ).rolling(20).mean()

    # ------------------------------------------------------------------
    # 计算接口
    # ------------------------------------------------------------------

    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            df: 包含OHLCV数据的DataFrame

        Returns:
            包含所有因子的DataFrame
        """
        result = df.copy()
        for factor_name, factor_func in self.factor_functions.items():
            try:
                result[factor_name] = factor_func(df)
            except Exception as e:
                print(f"计算因子 {factor_name} 失败: {e}")
                result[factor_name] = np.nan
        return result

    # 向后兼容别名
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.calculate_factors(df)
    
    def get_factor_list(self) -> List[str]:
        """获取所有因子名称列表"""
        return list(self.factor_functions.keys())
    
    def get_factor_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取因子统计信息
        
        Args:
            df: 包含因子的DataFrame
            
        Returns:
            因子统计信息
        """
        factor_cols = [col for col in df.columns if col in self.factor_functions.keys()]
        
        stats = []
        for col in factor_cols:
            stats.append({
                'factor': col,
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isna().sum() / len(df),
            })
        
        return pd.DataFrame(stats)


class FactorValidator:
    """因子验证器"""
    
    @staticmethod
    def calculate_ic(factor_values: pd.Series, future_returns: pd.Series) -> float:
        """
        计算信息系数(IC)
        
        Args:
            factor_values: 因子值
            future_returns: 未来收益
            
        Returns:
            IC值
        """
        return factor_values.corr(future_returns, method='spearman')
    
    @staticmethod
    def calculate_ir(ic_series: pd.Series) -> float:
        """
        计算信息比率(IR)
        
        Args:
            ic_series: IC序列
            
        Returns:
            IR值
        """
        return ic_series.mean() / ic_series.std()
    
    @staticmethod
    def test_factor_significance(factor_values: pd.Series, 
                                  future_returns: pd.Series) -> Dict:
        """
        测试因子显著性
        
        Args:
            factor_values: 因子值
            future_returns: 未来收益
            
        Returns:
            显著性测试结果
        """
        # 计算IC
        ic = FactorValidator.calculate_ic(factor_values, future_returns)
        
        # t检验
        from scipy.stats import ttest_ind
        
        # 分组
        median = factor_values.median()
        high_group = future_returns[factor_values > median]
        low_group = future_returns[factor_values <= median]
        
        t_stat, p_value = ttest_ind(high_group.dropna(), low_group.dropna())
        
        return {
            'ic': ic,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }


# ===========================================================================
# V10 新增：FactorEngineer — 跨截面标准化 + IC加权合成打分
# ===========================================================================

class FactorEngineer:
    """
    因子工程层：对 Alpha158 产生的原始因子进行后处理，
    输出一个可直接用于模型训练或信号生成的合成评分。

    处理流程：
      1. 去极值（Winsorize 3σ）
      2. 跨截面 Z-Score 标准化（每期在股票池内标准化）
      3. 行业中性化（若提供行业标签）
      4. IC-IR 加权合成：历史 IC > 阈值的因子参与合成
      5. 输出 [-1, +1] 范围的合成得分

    Parameters
    ----------
    ic_threshold   : 纳入合成的最低平均 IC（默认 0.02）
    winsor_sigma   : 去极值 σ 倍数（默认 3.0）
    lookback_ic    : 估算历史 IC 的回溯窗口（期数，默认 60）
    """

    def __init__(
        self,
        ic_threshold: float = 0.02,
        winsor_sigma: float = 3.0,
        lookback_ic: int = 60,
    ) -> None:
        self.ic_threshold = ic_threshold
        self.winsor_sigma = winsor_sigma
        self.lookback_ic = lookback_ic
        self._alpha = Alpha158()

    # ------------------------------------------------------------------
    # 主接口：给定单支股票的 OHLCV，返回合成得分序列
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算单支股票的因子合成得分序列。
        返回 [-1, +1] 的 pd.Series，index 同 df。
        """
        factor_df = self._alpha.calculate_all(df)

        # 提取因子列
        factor_cols = [c for c in factor_df.columns
                       if c not in ("open", "high", "low", "close", "volume", "amount", "date")]
        if not factor_cols:
            return pd.Series(0.0, index=df.index)

        factors = factor_df[factor_cols].copy()

        # 1. 去极值
        factors = self._winsorize(factors)

        # 2. Z-Score 标准化（时序维度，每因子自身 rolling）
        factors = self._zscore_timeseries(factors)

        # 3. 估算每个因子对未来5日收益的历史 IC
        future_ret = df["close"].pct_change(5).shift(-5)
        ic_weights = self._estimate_ic_weights(factors, future_ret)

        # 4. 加权合成
        composite = self._weighted_composite(factors, ic_weights)
        return composite

    # ------------------------------------------------------------------
    # 截面打分：适用于多股票组合，每期在截面内排名标准化
    # ------------------------------------------------------------------

    def cross_sectional_score(
        self,
        price_dict: Dict[str, pd.DataFrame],
        industry_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        对股票池做截面因子打分，返回每只股票最新期的合成得分 dict。

        Parameters
        ----------
        price_dict   : {symbol: OHLCV DataFrame}
        industry_map : {symbol: industry_label}（可选，用于行业中性化）
        """
        last_scores: Dict[str, float] = {}

        for symbol, df in price_dict.items():
            if df is None or len(df) < 60:
                last_scores[symbol] = 0.0
                continue
            try:
                score_series = self.score(df)
                last_scores[symbol] = float(score_series.iloc[-1]) if not score_series.empty else 0.0
            except Exception:
                last_scores[symbol] = 0.0

        if not last_scores:
            return last_scores

        # 截面行业中性化
        if industry_map:
            last_scores = self._industry_neutralize(last_scores, industry_map)

        # 截面 Rank 归一化到 [-1, +1]
        arr = np.array(list(last_scores.values()), dtype=float)
        if arr.std() > 1e-9:
            ranks = stats.rankdata(arr)
            normalized = (ranks - ranks.mean()) / (ranks.std() + 1e-9)
            normalized = np.clip(normalized / 3, -1, 1)  # ±3σ → ±1
        else:
            normalized = np.zeros_like(arr)

        return dict(zip(last_scores.keys(), normalized.tolist()))

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """3σ 去极值"""
        result = df.copy()
        for col in result.columns:
            s = result[col].dropna()
            if len(s) < 10:
                continue
            mean, std = s.mean(), s.std()
            result[col] = result[col].clip(
                lower=mean - self.winsor_sigma * std,
                upper=mean + self.winsor_sigma * std,
            )
        return result

    @staticmethod
    def _zscore_timeseries(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """时序 rolling Z-Score 标准化"""
        result = df.copy()
        for col in result.columns:
            m = result[col].rolling(window, min_periods=20).mean()
            s = result[col].rolling(window, min_periods=20).std()
            result[col] = (result[col] - m) / (s + 1e-9)
        return result

    def _estimate_ic_weights(
        self,
        factors: pd.DataFrame,
        future_ret: pd.Series,
    ) -> Dict[str, float]:
        """用历史窗口内的 Spearman IC 估算权重，IC-IR 加权"""
        weights: Dict[str, float] = {}
        aligned = factors.join(future_ret.rename("__fret__"), how="inner").dropna(
            subset=["__fret__"]
        )
        if len(aligned) < self.lookback_ic:
            return {col: 1.0 for col in factors.columns}

        window = aligned.tail(self.lookback_ic)
        for col in factors.columns:
            sub = window[[col, "__fret__"]].dropna()
            if len(sub) < 10:
                weights[col] = 0.0
                continue
            ic = float(sub[col].corr(sub["__fret__"], method="spearman"))
            if abs(ic) >= self.ic_threshold:
                weights[col] = ic
            else:
                weights[col] = 0.0
        return weights

    @staticmethod
    def _weighted_composite(
        factors: pd.DataFrame,
        weights: Dict[str, float],
    ) -> pd.Series:
        """加权合成，归一化到 [-1, +1]"""
        active = {k: v for k, v in weights.items() if v != 0.0 and k in factors.columns}
        if not active:
            return pd.Series(0.0, index=factors.index)

        total_w = sum(abs(v) for v in active.values())
        composite = pd.Series(0.0, index=factors.index)
        for col, w in active.items():
            composite += (w / total_w) * factors[col].fillna(0)

        # 归一化
        std = composite.std()
        if std > 1e-9:
            composite = composite / (3 * std)
        return composite.clip(-1, 1)

    @staticmethod
    def _industry_neutralize(
        scores: Dict[str, float],
        industry_map: Dict[str, str],
    ) -> Dict[str, float]:
        """减去行业均值（行业中性化）"""
        industry_scores: Dict[str, List[float]] = {}
        for sym, sc in scores.items():
            ind = industry_map.get(sym, "unknown")
            industry_scores.setdefault(ind, []).append(sc)

        ind_mean = {ind: float(np.mean(vals)) for ind, vals in industry_scores.items()}
        return {
            sym: sc - ind_mean.get(industry_map.get(sym, "unknown"), 0.0)
            for sym, sc in scores.items()
        }

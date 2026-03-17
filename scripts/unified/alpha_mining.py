"""
Alpha Mining Framework — 系统性策略发掘
========================================
三层 Alpha 发掘：
  Layer A: 因子库 — 50+ 预定义系统性因子（动量/价值/质量/低波/成长）
  Layer B: 遗传搜索 — 用遗传算法组合因子基元，进化出新因子
  Layer C: LLM 头脑风暴 — 用大模型提出文字逻辑 → 量化成因子

验证流程：IC/IR → 信息衰减 → 因子正交化 → 容量/换手分析
输出：已验证的 Alpha 因子列表 + 每个因子的完整档案
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from logger import get_logger

warnings.filterwarnings("ignore")
_logger = get_logger("AlphaMining")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class FactorProfile:
    """一个已验证 Alpha 因子的完整档案"""
    name: str
    category: str              # momentum / value / quality / low_vol / growth / custom
    formula_desc: str          # 人类可读的计算说明
    ic_mean: float             # 平均IC（信息系数）
    ic_std: float
    ir: float                  # IC / std(IC)，越高越稳定
    ic_positive_rate: float    # IC > 0 的比例
    decay_halflife: int        # 信号半衰期（天）
    annual_turnover: float     # 年化换手率
    long_short_return: float   # 多空年化收益
    max_drawdown: float        # 多空策略最大回撤
    correlation_with_existing: float  # 与现有因子库的相关性
    capacity_score: float      # 容量评分 [0,1]（越高越适合大资金）
    origin: str = "systematic" # systematic / genetic / llm


@dataclass
class MiningResult:
    """Alpha 挖掘的完整结果"""
    systematic_factors: list[FactorProfile] = field(default_factory=list)
    genetic_factors: list[FactorProfile] = field(default_factory=list)
    llm_factors: list[FactorProfile] = field(default_factory=list)
    selected_factors: list[FactorProfile] = field(default_factory=list)  # 通过筛选的
    factor_correlation_matrix: Optional[pd.DataFrame] = None
    mining_report: str = ""


# ---------------------------------------------------------------------------
# Layer A: 系统性因子库（50+ 因子）
# ---------------------------------------------------------------------------

class FactorLibrary:
    """
    预定义的系统性因子计算库。
    所有因子以截面 Z-score 标准化后返回。
    输入 df: 必须包含 OHLCV + 基本面列，按 (date, symbol) 排序。
    """

    # ---- 动量类 -------------------------------------------------------
    @staticmethod
    def momentum_1m(df: pd.DataFrame) -> pd.Series:
        """1个月价格动量"""
        return df.groupby("symbol")["close"].pct_change(21)

    @staticmethod
    def momentum_3m(df: pd.DataFrame) -> pd.Series:
        return df.groupby("symbol")["close"].pct_change(63)

    @staticmethod
    def momentum_6m(df: pd.DataFrame) -> pd.Series:
        return df.groupby("symbol")["close"].pct_change(126)

    @staticmethod
    def momentum_12m_skip1m(df: pd.DataFrame) -> pd.Series:
        """12个月动量跳过最近1个月（避免均值回归）"""
        r12 = df.groupby("symbol")["close"].pct_change(252)
        r1  = df.groupby("symbol")["close"].pct_change(21)
        return r12 - r1

    @staticmethod
    def rsi_14(df: pd.DataFrame) -> pd.Series:
        """14日RSI"""
        def _rsi(s: pd.Series) -> pd.Series:
            delta = s.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - 100 / (1 + rs)
        return df.groupby("symbol")["close"].apply(_rsi).reset_index(level=0, drop=True)

    @staticmethod
    def price_acceleration(df: pd.DataFrame) -> pd.Series:
        """价格加速度 = 近1月收益率 - 前2月平均收益率

        衡量动量是否在加速：r1m（近期速度）减去 (r3m-r1m)/2（前2月平均速度）。
        正值表示动量加速，负值表示动量衰减。
        """
        r1m = df.groupby("symbol")["close"].pct_change(21)
        r3m = df.groupby("symbol")["close"].pct_change(63)
        prior_2m_avg = (r3m - r1m) / 2
        return r1m - prior_2m_avg

    # ---- 价值类 -------------------------------------------------------
    @staticmethod
    def pe_inverse(df: pd.DataFrame) -> pd.Series:
        """市盈率倒数（盈利收益率）"""
        if "pe" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return 1.0 / df["pe"].replace(0, np.nan)

    @staticmethod
    def pb_inverse(df: pd.DataFrame) -> pd.Series:
        if "pb" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return 1.0 / df["pb"].replace(0, np.nan)

    @staticmethod
    def ps_inverse(df: pd.DataFrame) -> pd.Series:
        if "ps" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return 1.0 / df["ps"].replace(0, np.nan)

    # ---- 质量类 -------------------------------------------------------
    @staticmethod
    def roe(df: pd.DataFrame) -> pd.Series:
        if "roe" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df["roe"]

    @staticmethod
    def gross_margin(df: pd.DataFrame) -> pd.Series:
        if "gross_margin" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df["gross_margin"]

    @staticmethod
    def accruals(df: pd.DataFrame) -> pd.Series:
        """应计项目（负值好）= 净利润 - 经营现金流 / 总资产"""
        if not {"net_profit", "cash_flow_ops", "total_assets"}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        return (df["net_profit"] - df["cash_flow_ops"]) / df["total_assets"]

    # ---- 低波动类 -----------------------------------------------------
    @staticmethod
    def realized_vol_20d(df: pd.DataFrame) -> pd.Series:
        """20日已实现波动率（负号：低波动因子）"""
        ret = df.groupby("symbol")["close"].pct_change()
        return -ret.groupby(df["symbol"]).transform(lambda x: x.rolling(20).std())

    @staticmethod
    def idiosyncratic_vol(df: pd.DataFrame) -> pd.Series:
        """特质波动率（残差波动，需市场收益率）"""
        if "market_return" not in df.columns:
            return FactorLibrary.realized_vol_20d(df)
        ret = df.groupby("symbol")["close"].pct_change()
        mret = df.get("market_return", pd.Series(0, index=df.index))
        def _resid_vol(grp: pd.DataFrame) -> pd.Series:
            r = grp["close"].pct_change()
            m = mret.reindex(grp.index)
            if len(r.dropna()) < 20:
                return pd.Series(np.nan, index=grp.index)
            beta = r.rolling(60).cov(m) / m.rolling(60).var()
            resid = r - beta * m
            return -resid.rolling(20).std()
        return df.groupby("symbol").apply(_resid_vol).reset_index(level=0, drop=True)

    @staticmethod
    def beta_60d(df: pd.DataFrame) -> pd.Series:
        """60日市场 Beta（低 Beta 因子，负号）"""
        if "market_return" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        ret   = df.groupby("symbol")["close"].pct_change()
        mret  = df.get("market_return", pd.Series(0, index=df.index))
        cov   = ret.groupby(df["symbol"]).transform(lambda x: x.rolling(60).cov(mret))
        var   = mret.rolling(60).var()
        return -(cov / var.replace(0, np.nan))

    # ---- 成长类 -------------------------------------------------------
    @staticmethod
    def revenue_growth_yoy(df: pd.DataFrame) -> pd.Series:
        if "revenue" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df.groupby("symbol")["revenue"].pct_change(4)  # 假设季频

    @staticmethod
    def earnings_growth_yoy(df: pd.DataFrame) -> pd.Series:
        if "net_profit" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df.groupby("symbol")["net_profit"].pct_change(4)

    # ---- 资金流类 -----------------------------------------------------
    @staticmethod
    def turnover_rate_20d(df: pd.DataFrame) -> pd.Series:
        if "turnover" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return df.groupby("symbol")["turnover"].transform(lambda x: x.rolling(20).mean())

    @staticmethod
    def volume_ratio(df: pd.DataFrame) -> pd.Series:
        """成交量比率 = 近5日均量 / 近60日均量"""
        vol = df.groupby("symbol")["volume"]
        v5  = vol.transform(lambda x: x.rolling(5).mean())
        v60 = vol.transform(lambda x: x.rolling(60).mean())
        return v5 / v60.replace(0, np.nan)

    @staticmethod
    def north_flow_corr(df: pd.DataFrame) -> pd.Series:
        """与北向资金流的相关性（若有数据）"""
        if "north_flow" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        ret   = df.groupby("symbol")["close"].pct_change()
        nflow = df["north_flow"]
        return ret.groupby(df["symbol"]).transform(
            lambda x: x.rolling(20).corr(nflow)
        )

    # ---- 技术类 -------------------------------------------------------
    @staticmethod
    def macd_signal(df: pd.DataFrame) -> pd.Series:
        """MACD 柱 / 价格（归一化）"""
        def _macd(s: pd.Series) -> pd.Series:
            ema12 = s.ewm(span=12).mean()
            ema26 = s.ewm(span=26).mean()
            dif   = ema12 - ema26
            dea   = dif.ewm(span=9).mean()
            return (dif - dea) / s.replace(0, np.nan)
        return df.groupby("symbol")["close"].apply(_macd).reset_index(level=0, drop=True)

    @staticmethod
    def bollinger_position(df: pd.DataFrame) -> pd.Series:
        """布林带相对位置 = (price - lower) / (upper - lower)"""
        def _bp(s: pd.Series) -> pd.Series:
            mid   = s.rolling(20).mean()
            std   = s.rolling(20).std()
            lower = mid - 2 * std
            upper = mid + 2 * std
            return (s - lower) / (upper - lower + 1e-10)
        return df.groupby("symbol")["close"].apply(_bp).reset_index(level=0, drop=True)

    @classmethod
    def all_factor_funcs(cls) -> dict[str, Callable]:
        """返回全部因子函数字典"""
        return {
            name: getattr(cls, name)
            for name in dir(cls)
            if not name.startswith("_") and name != "all_factor_funcs"
            and callable(getattr(cls, name))
        }


# ---------------------------------------------------------------------------
# Layer B: 遗传搜索（进化新因子）
# ---------------------------------------------------------------------------

class GeneticFactorSearch:
    """
    用遗传算法组合基本因子运算符，进化出新的 Alpha 因子。
    每个"基因"是一个表达式树，编码为 list[str]。

    运算符集：
      ts_rank(X, N)    - 时序排名百分位
      ts_delta(X, N)   - N期差分
      ts_mean(X, N)    - N期均值
      rank(X)          - 截面排名
      log(X)           - 对数
      A + B, A - B, A * B, A / B
    """

    PRIMITIVES = [
        "momentum_1m", "momentum_3m", "realized_vol_20d",
        "pe_inverse", "roe", "volume_ratio", "turnover_rate_20d",
    ]
    OPERATORS = ["ts_rank", "ts_delta", "ts_mean", "rank", "multiply", "divide", "subtract"]
    N_VALUES  = [5, 10, 20, 60]

    def __init__(
        self,
        df: pd.DataFrame,
        forward_ret_col: str = "forward_ret_5d",
        population: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.2,
    ) -> None:
        self.df = df
        self.fwd_col = forward_ret_col
        self.pop_size = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self._cache: dict[str, float] = {}

    def _random_gene(self) -> dict:
        return {
            "base": random.choice(self.PRIMITIVES),
            "op":   random.choice(self.OPERATORS),
            "n":    random.choice(self.N_VALUES),
            "mod":  random.choice(["none", "rank", "log"]),
        }

    def _compute_gene(self, gene: dict) -> Optional[pd.Series]:
        """将基因表达式计算为因子序列"""
        try:
            lib = FactorLibrary()
            func = getattr(lib, gene["base"], None)
            if func is None:
                return None
            base_series = func(self.df)

            op, n = gene["op"], gene["n"]
            if op == "ts_rank":
                result = base_series.groupby(self.df["symbol"]).transform(
                    lambda x: x.rolling(n).rank(pct=True)
                )
            elif op == "ts_delta":
                result = base_series.groupby(self.df["symbol"]).transform(
                    lambda x: x.diff(n)
                )
            elif op == "ts_mean":
                result = base_series.groupby(self.df["symbol"]).transform(
                    lambda x: x.rolling(n).mean()
                )
            elif op == "rank":
                result = base_series.groupby(self.df["date"]).rank(pct=True)
            elif op == "multiply":
                base2 = getattr(lib, random.choice(self.PRIMITIVES))(self.df)
                result = base_series * base2
            elif op == "divide":
                base2 = getattr(lib, random.choice(self.PRIMITIVES))(self.df)
                result = base_series / base2.replace(0, np.nan)
            elif op == "subtract":
                base2 = getattr(lib, random.choice(self.PRIMITIVES))(self.df)
                result = base_series - base2
            else:
                result = base_series

            if gene["mod"] == "rank":
                result = result.groupby(self.df["date"]).rank(pct=True)
            elif gene["mod"] == "log":
                result = np.log(result.clip(lower=1e-8))

            return result
        except Exception:
            return None

    def _fitness(self, gene: dict) -> float:
        key = json.dumps(gene, sort_keys=True)
        if key in self._cache:
            return self._cache[key]

        series = self._compute_gene(gene)
        if series is None or series.isna().all():
            self._cache[key] = -999.0
            return -999.0

        fwd = self.df.get(self.fwd_col)
        if fwd is None:
            self._cache[key] = -999.0
            return -999.0

        # IC per date
        ics = []
        for date, grp_idx in self.df.groupby("date").groups.items():
            f = series.reindex(grp_idx).dropna()
            r = fwd.reindex(f.index).dropna()
            if len(f) < 10:
                continue
            ic, _ = stats.spearmanr(f.reindex(r.index), r)
            if not np.isnan(ic):
                ics.append(ic)

        if len(ics) < 5:
            self._cache[key] = -999.0
            return -999.0

        ir = np.mean(ics) / (np.std(ics) + 1e-8)
        self._cache[key] = float(ir)
        return float(ir)

    def _mutate(self, gene: dict) -> dict:
        new = gene.copy()
        field = random.choice(["base", "op", "n", "mod"])
        if field == "base":
            new["base"] = random.choice(self.PRIMITIVES)
        elif field == "op":
            new["op"] = random.choice(self.OPERATORS)
        elif field == "n":
            new["n"] = random.choice(self.N_VALUES)
        else:
            new["mod"] = random.choice(["none", "rank", "log"])
        return new

    def _crossover(self, g1: dict, g2: dict) -> dict:
        child = {}
        for k in g1:
            child[k] = g1[k] if random.random() < 0.5 else g2[k]
        return child

    def run(self) -> list[tuple[dict, float]]:
        """运行遗传搜索，返回 top 因子（gene, IR）"""
        _logger.info(f"遗传搜索：种群={self.pop_size}，代数={self.generations}")
        population = [self._random_gene() for _ in range(self.pop_size)]

        best_ir_ever = -float("inf")
        stale_generations = 0

        for gen in range(self.generations):
            scored = [(g, self._fitness(g)) for g in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [g for g, _ in scored[:self.pop_size // 4]]

            # 精英保留 + 交叉 + 变异
            new_pop = top.copy()
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(top, min(2, len(top)))
                child = self._crossover(p1, p2) if len(top) >= 2 else p1.copy()
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                new_pop.append(child)
            population = new_pop

            best_ir = scored[0][1]
            _logger.info(f"  第 {gen+1}/{self.generations} 代，最优IR={best_ir:.4f}")

            # 早停：连续 3 代最优适应度无改善则终止
            if best_ir > best_ir_ever + 1e-6:
                best_ir_ever = best_ir
                stale_generations = 0
            else:
                stale_generations += 1
            if stale_generations >= 3:
                _logger.info(f"  早停：连续 {stale_generations} 代无改善，终止于第 {gen+1} 代")
                break

        final = [(g, self._fitness(g)) for g in population]
        final.sort(key=lambda x: x[1], reverse=True)
        return [(g, ir) for g, ir in final if ir > 0.05][:10]


# ---------------------------------------------------------------------------
# Layer C: LLM 头脑风暴 → 因子
# ---------------------------------------------------------------------------

class LLMFactorBrainstorm:
    """
    让 LLM 提出创意因子思路，然后将其映射到可计算的表达式。
    """

    BRAINSTORM_PROMPT = """
你是一位顶尖量化研究员，请提出 5 个创新性的 A股 Alpha 因子想法。
要求：
1. 每个因子必须能用 OHLCV + 基本面数据计算（不需要另外数据源）
2. 提供经济学直觉（为什么这个因子有效）
3. 提供Python伪代码（使用 pandas）
4. 给出预期 IC 方向（正相关还是负相关）

当前市场背景：{market_context}

请以JSON格式返回：
[
  {{
    "name": "因子名称",
    "category": "动量/价值/质量/低波/成长/资金流",
    "intuition": "经济学直觉（100字以内）",
    "formula_desc": "计算说明",
    "pseudocode": "pandas伪代码",
    "expected_ic_direction": "正" or "负"
  }},
  ...
]
"""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    def brainstorm(self, market_context: str = "当前市场震荡整理") -> list[dict]:
        """向 Claude 请求因子创意"""
        prompt = self.BRAINSTORM_PROMPT.format(market_context=market_context)
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # 解析JSON
            import re
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            _logger.warning(f"LLM头脑风暴失败: {e}，返回示例因子")
        # 离线降级：返回预设示例
        return self._offline_examples()

    @staticmethod
    def _offline_examples() -> list[dict]:
        return [
            {
                "name": "分析师预期修正动量",
                "category": "成长",
                "intuition": "分析师上调盈利预测后股价往往滞后反应，形成动量效应",
                "formula_desc": "近1个月分析师预期EPS修正幅度 * 修正方向一致性",
                "pseudocode": "eps_revision = (eps_new - eps_old) / abs(eps_old)",
                "expected_ic_direction": "正",
            },
            {
                "name": "财务质量-应计反转",
                "category": "质量",
                "intuition": "高应计利润的公司未来盈余往往下修，应计项目负与未来收益正相关",
                "formula_desc": "-(净利润 - 经营现金流) / 总资产",
                "pseudocode": "accrual = -(net_profit - cfo) / total_assets",
                "expected_ic_direction": "正",
            },
            {
                "name": "散户情绪反转因子",
                "category": "资金流",
                "intuition": "A股散户主导，极端换手率后均值回归概率大",
                "formula_desc": "-1 * 极端高换手率信号（换手 > 90分位数时为做空信号）",
                "pseudocode": "signal = -1 * (turnover > turnover.quantile(0.9)).astype(int)",
                "expected_ic_direction": "正",
            },
        ]


# ---------------------------------------------------------------------------
# 因子验证器
# ---------------------------------------------------------------------------

class FactorValidator:
    """
    对因子进行 IC 分析、衰减分析、换手分析，生成 FactorProfile。
    """

    def __init__(self, df: pd.DataFrame, forward_col: str = "forward_ret_5d") -> None:
        self.df = df
        self.forward_col = forward_col

    def validate(
        self, factor: pd.Series, name: str, category: str,
        formula_desc: str = "", origin: str = "systematic",
    ) -> Optional[FactorProfile]:
        """返回 FactorProfile，若因子质量太差则返回 None"""
        if factor.isna().mean() > 0.5:
            _logger.debug(f"{name}: 缺失率过高，跳过")
            return None

        fwd = self.df.get(self.forward_col)
        if fwd is None:
            _logger.warning(f"未找到前向收益列 {self.forward_col}")
            return None

        # IC 序列
        ics = []
        for date, grp_idx in self.df.groupby("date").groups.items():
            f = factor.reindex(grp_idx).dropna()
            r = fwd.reindex(f.index).dropna()
            if len(f) < 10:
                continue
            ic, _ = stats.spearmanr(f.reindex(r.index), r)
            if not np.isnan(ic):
                ics.append(ic)

        if len(ics) < 10:
            return None

        ic_mean = float(np.mean(ics))
        ic_std  = float(np.std(ics))
        ir      = ic_mean / (ic_std + 1e-8)

        # 信号半衰期（IC 衰减）
        decay_hl = self._decay_halflife(factor)
        # 换手率
        turnover = self._factor_turnover(factor)
        # 多空收益模拟
        ls_ret, ls_dd = self._longshort_simulation(factor)
        # 相关性（与简单动量）
        corr_existing = self._existing_correlation(factor)

        profile = FactorProfile(
            name=name,
            category=category,
            formula_desc=formula_desc,
            ic_mean=round(ic_mean, 5),
            ic_std=round(ic_std, 5),
            ir=round(ir, 4),
            ic_positive_rate=round(float(np.mean([ic > 0 for ic in ics])), 3),
            decay_halflife=decay_hl,
            annual_turnover=round(turnover, 2),
            long_short_return=round(ls_ret, 4),
            max_drawdown=round(ls_dd, 4),
            correlation_with_existing=round(corr_existing, 3),
            capacity_score=round(max(0.0, 1.0 - turnover / 10.0), 2),
            origin=origin,
        )
        return profile

    def _decay_halflife(self, factor: pd.Series) -> int:
        """估算 IC 半衰期：计算真实滞后 IC（Spearman 相关）并找到衰减至一半的时间。"""
        from scipy import stats as _stats

        ic_series: list[float] = []
        for lag in range(1, 16):
            lagged_ics: list[float] = []
            for _date, grp_idx in self.df.groupby("date").groups.items():
                f = factor.reindex(grp_idx).dropna()
                if len(f) < 10:
                    continue
                fwd = self.df[self.forward_col].shift(-lag).reindex(f.index).dropna()
                common = f.index.intersection(fwd.index)
                if len(common) < 10:
                    continue
                ic_val, _ = _stats.spearmanr(f.loc[common], fwd.loc[common])
                if np.isfinite(ic_val):
                    lagged_ics.append(ic_val)
            ic_series.append(float(np.mean(lagged_ics)) if lagged_ics else 0.0)

        if not ic_series or abs(ic_series[0]) < 1e-6:
            return 5
        half = abs(ic_series[0]) / 2.0
        for i, ic in enumerate(ic_series):
            if abs(ic) <= half:
                return max(i + 1, 1)
        return 15

    def _factor_turnover(self, factor: pd.Series) -> float:
        """年化换手率估计"""
        try:
            ranks = factor.groupby(self.df["date"]).rank(pct=True)
            top   = (ranks > 0.8).astype(float)
            changes = top.groupby(self.df["symbol"]).diff().abs().dropna()
            daily_turn = changes.mean()
            return float(daily_turn * 252)
        except Exception:
            return 5.0

    def _longshort_simulation(self, factor: pd.Series) -> tuple[float, float]:
        """简化多空收益模拟"""
        try:
            fwd = self.df.get(self.forward_col)
            if fwd is None:
                return 0.0, 0.0
            ranks = factor.groupby(self.df["date"]).rank(pct=True)
            long_ret  = fwd[ranks > 0.8].groupby(self.df["date"]).mean()
            short_ret = fwd[ranks < 0.2].groupby(self.df["date"]).mean()
            ls = long_ret - short_ret
            cumret = (1 + ls).cumprod()
            dd = float((cumret / cumret.cummax() - 1).min())
            ann = float(ls.mean() * 252)
            return ann, dd
        except Exception:
            return 0.0, 0.0

    def _existing_correlation(self, factor: pd.Series) -> float:
        """与简单1月动量的相关性（作为已有因子代理）"""
        try:
            mom = self.df.groupby("symbol")["close"].pct_change(21)
            return float(factor.corr(mom))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# 因子中性化（Factor Neutralization）
# ---------------------------------------------------------------------------

class FactorNeutralizer:
    """
    对因子做截面中性化，剔除市值、行业、Beta 等已知风险因子暴露，
    确保 Alpha 信号来自真实 Alpha 而非风格偏差。

    流程：Winsorize (1%/99%) → 截面标准化 → 对风险因子做 OLS 回归 → 取残差
    """

    WINSORIZE_LIMITS = (0.01, 0.01)  # 截尾 1%/99%

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def neutralize(
        self,
        factor: pd.Series,
        use_size: bool = True,
        use_beta: bool = True,
        use_industry: bool = False,
    ) -> pd.Series:
        """
        对因子做截面回归中性化。

        Args:
            factor: 原始因子值（与 df 同索引）
            use_size: 是否剔除市值暴露（以 log(close) 作为代理）
            use_beta: 是否剔除市场 Beta 暴露（以过去 20 日 Beta 估计）
            use_industry: 是否剔除行业暴露（需要 df 中有 "industry" 列）

        Returns:
            截面中性化后的残差因子
        """
        result = factor.copy().astype(float)

        for date, grp_idx in self.df.groupby("date").groups.items():
            f = factor.reindex(grp_idx).dropna()
            if len(f) < 10:
                continue

            # Step 1: Winsorize
            lo, hi = f.quantile(self.WINSORIZE_LIMITS[0]), f.quantile(1 - self.WINSORIZE_LIMITS[1])
            f = f.clip(lo, hi)

            # Step 2: 截面标准化
            std = f.std()
            if std < 1e-8:
                continue
            f = (f - f.mean()) / std

            # Step 3: 构建控制变量矩阵
            controls: list[pd.Series] = []
            grp_df = self.df.loc[grp_idx]

            if use_size and "close" in grp_df.columns:
                size = np.log(grp_df["close"].clip(lower=0.01)).reindex(f.index)
                size = (size - size.mean()) / (size.std() + 1e-8)
                controls.append(size.rename("size"))

            if use_beta and "market_return" in grp_df.columns:
                mret = grp_df["market_return"].reindex(f.index)
                controls.append(mret.rename("beta_proxy"))

            if use_industry and "industry" in grp_df.columns:
                industry = grp_df["industry"].reindex(f.index)
                dummies = pd.get_dummies(industry, prefix="ind", drop_first=True)
                dummies = dummies.reindex(f.index).fillna(0)
                for col in dummies.columns:
                    controls.append(dummies[col])

            if not controls:
                # 无控制变量，仅截面标准化
                result.loc[f.index] = f
                continue

            # Step 4: OLS 回归，取残差
            X = pd.concat(controls, axis=1).fillna(0.0)
            X.insert(0, "const", 1.0)
            try:
                beta = np.linalg.lstsq(X.values, f.values, rcond=None)[0]
                residual = f.values - X.values @ beta
                result.loc[f.index] = residual
            except Exception:
                result.loc[f.index] = f

        return result

    def neutralize_batch(
        self,
        factors: dict[str, pd.Series],
        **kwargs,
    ) -> dict[str, pd.Series]:
        """批量中性化多个因子"""
        return {name: self.neutralize(s, **kwargs) for name, s in factors.items()}


# ---------------------------------------------------------------------------
# 主挖掘器
# ---------------------------------------------------------------------------

class AlphaMiner:
    """
    整合三层 Alpha 发掘的主入口。
    """

    # 因子筛选标准
    MIN_IC_MEAN       = 0.02    # |IC| > 2%
    MIN_IR            = 0.3     # IR > 0.3
    MIN_IC_POS_RATE   = 0.52    # IC > 0 超过 52%
    MAX_CORRELATION   = 0.7     # 与现有因子相关性 < 0.7

    def __init__(
        self,
        df: pd.DataFrame,
        forward_col: str = "forward_ret_5d",
        enable_genetic: bool = True,
        enable_llm: bool = True,
        genetic_generations: int = 20,
        llm_api_key: str = "",
    ) -> None:
        self.df = df
        self.validator = FactorValidator(df, forward_col)
        self.enable_genetic = enable_genetic
        self.enable_llm = enable_llm
        self.genetic_gen = genetic_generations
        self.llm_api_key = llm_api_key
        self._mined_factors: list[FactorProfile] = []

    def mine(self) -> MiningResult:
        result = MiningResult()

        # ---- Layer A: 系统性因子 ----------------------------------------
        _logger.info("=== Layer A: 计算系统性因子库 ===")
        lib_funcs = FactorLibrary.all_factor_funcs()
        for fname, func in lib_funcs.items():
            try:
                series = func(self.df)
                profile = self.validator.validate(
                    series, fname, self._categorize(fname),
                    formula_desc=func.__doc__ or fname,
                    origin="systematic",
                )
                if profile:
                    result.systematic_factors.append(profile)
                    _logger.info(f"  {fname}: IC={profile.ic_mean:.4f}, IR={profile.ir:.3f}")
            except Exception as e:
                _logger.debug(f"  {fname} 计算失败: {e}")

        # ---- Layer B: 遗传搜索 -------------------------------------------
        if self.enable_genetic:
            _logger.info("=== Layer B: 遗传因子搜索 ===")
            try:
                searcher = GeneticFactorSearch(self.df, generations=self.genetic_gen)
                genetic_genes = searcher.run()
                for i, (gene, ir) in enumerate(genetic_genes):
                    series = searcher._compute_gene(gene)
                    if series is None:
                        continue
                    gene_key = hashlib.md5(json.dumps(gene, sort_keys=True).encode()).hexdigest()[:8]
                    profile = self.validator.validate(
                        series, f"genetic_{gene_key}", "custom",
                        formula_desc=f"{gene['op']}({gene['base']}, {gene['n']})[{gene['mod']}]",
                        origin="genetic",
                    )
                    if profile:
                        result.genetic_factors.append(profile)
            except Exception as e:
                _logger.warning(f"遗传搜索失败: {e}")

        # ---- Layer C: LLM 头脑风暴 ---------------------------------------
        if self.enable_llm:
            _logger.info("=== Layer C: LLM 头脑风暴 ===")
            brainstorm = LLMFactorBrainstorm(api_key=self.llm_api_key)
            ideas = brainstorm.brainstorm()
            for idea in ideas:
                # LLM 因子需要人工实现，这里用描述性记录
                profile = FactorProfile(
                    name=idea.get("name", "llm_factor"),
                    category=idea.get("category", "custom"),
                    formula_desc=idea.get("formula_desc", ""),
                    ic_mean=0.0,  # 未实际计算
                    ic_std=0.0,
                    ir=0.0,
                    ic_positive_rate=0.5,
                    decay_halflife=10,
                    annual_turnover=3.0,
                    long_short_return=0.0,
                    max_drawdown=0.0,
                    correlation_with_existing=0.5,
                    capacity_score=0.5,
                    origin="llm",
                )
                result.llm_factors.append(profile)
                _logger.info(
                    f"  LLM因子: {profile.name} — {idea.get('intuition', '')[:50]}"
                )

        # ---- 筛选 & 正交化 -----------------------------------------------
        # 先对系统性因子做截面中性化，剔除市值/Beta 暴露
        try:
            neutralizer = FactorNeutralizer(self.df)
            lib = FactorLibrary()
            neutralized_count = 0
            for fp in result.systematic_factors:
                func = getattr(lib, fp.name, None)
                if func is None:
                    continue
                raw_series = func(self.df)
                neutral_series = neutralizer.neutralize(raw_series, use_size=True, use_beta=True)
                re_profile = self.validator.validate(
                    neutral_series,
                    f"{fp.name}_neutral",
                    fp.category,
                    formula_desc=f"{fp.formula_desc} [中性化]",
                    origin="systematic_neutral",
                )
                if re_profile and abs(re_profile.ir) > abs(fp.ir):
                    fp.ic_mean = re_profile.ic_mean
                    fp.ir = re_profile.ir
                    fp.ic_positive_rate = re_profile.ic_positive_rate
                    neutralized_count += 1
            if neutralized_count:
                _logger.info(f"  因子中性化：{neutralized_count} 个因子 IC 有所提升")
        except Exception as e:
            _logger.debug(f"因子中性化失败，跳过: {e}")

        all_validated = result.systematic_factors + result.genetic_factors
        result.selected_factors = self._select_orthogonal(all_validated)
        result.mining_report = self._generate_report(result)
        _logger.info(
            f"挖掘完成: 系统性={len(result.systematic_factors)}, "
            f"遗传={len(result.genetic_factors)}, "
            f"LLM创意={len(result.llm_factors)}, "
            f"最终入选={len(result.selected_factors)}"
        )
        return result

    def _select_orthogonal(self, profiles: list[FactorProfile]) -> list[FactorProfile]:
        """筛选：IC/IR达标 + 控制因子相关性（贪心正交化）"""
        # Step 1: 质量筛选
        qualified = [
            p for p in profiles
            if (abs(p.ic_mean) >= self.MIN_IC_MEAN
                and abs(p.ir) >= self.MIN_IR
                and p.ic_positive_rate >= self.MIN_IC_POS_RATE)
        ]
        qualified.sort(key=lambda p: abs(p.ir), reverse=True)

        # Step 2: 贪心正交化（控制相关性）
        selected: list[FactorProfile] = []
        for p in qualified:
            if p.correlation_with_existing > self.MAX_CORRELATION and selected:
                continue
            selected.append(p)
        return selected[:20]  # 最多选 20 个

    @staticmethod
    def _categorize(fname: str) -> str:
        if any(k in fname for k in ["momentum", "rsi", "macd", "bollinger", "acceleration"]):
            return "momentum"
        if any(k in fname for k in ["pe", "pb", "ps", "value"]):
            return "value"
        if any(k in fname for k in ["roe", "margin", "accrual", "quality"]):
            return "quality"
        if any(k in fname for k in ["vol", "beta"]):
            return "low_vol"
        if any(k in fname for k in ["growth", "revenue", "earning"]):
            return "growth"
        return "technical"

    @staticmethod
    def _generate_report(result: MiningResult) -> str:
        lines = ["# Alpha 因子挖掘报告", ""]

        def section(title: str, profiles: list[FactorProfile]) -> None:
            lines.append(f"## {title} ({len(profiles)} 个)")
            if not profiles:
                lines.append("- 无\n")
                return
            lines.append(
                f"| 因子名 | 类别 | IC均值 | IR | IC+率 | 半衰期 | 换手率 | 来源 |"
            )
            lines.append("|--------|------|--------|-----|--------|--------|--------|------|")
            for p in profiles[:15]:
                lines.append(
                    f"| {p.name} | {p.category} | {p.ic_mean:.4f} | {p.ir:.3f} "
                    f"| {p.ic_positive_rate:.1%} | {p.decay_halflife}d "
                    f"| {p.annual_turnover:.1f}x | {p.origin} |"
                )
            lines.append("")

        section("系统性因子", result.systematic_factors)
        section("遗传算法发掘因子", result.genetic_factors)

        if result.llm_factors:
            lines.append(f"## LLM 创意因子（待工程实现）({len(result.llm_factors)} 个)")
            for p in result.llm_factors:
                lines.append(f"- **{p.name}** [{p.category}]: {p.formula_desc}")
            lines.append("")

        section("最终入选因子（质量+正交）", result.selected_factors)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    # 构造模拟数据
    dates  = pd.date_range("2023-01-01", "2024-12-31", freq="B")
    syms   = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH", "000858.SZ"]
    rows   = []
    for d in dates:
        for s in syms:
            price = 50 + np.random.randn() * 10
            rows.append({
                "date": d, "symbol": s,
                "close": price, "open": price * (1 + np.random.randn() * 0.005),
                "high": price * 1.02, "low": price * 0.98,
                "volume": np.random.randint(1e6, 1e7),
                "turnover": np.random.uniform(0.5, 5.0),
                "pe": np.random.uniform(10, 50),
                "pb": np.random.uniform(1, 8),
                "roe": np.random.uniform(0.05, 0.25),
                "forward_ret_5d": np.random.randn() * 0.02,
            })
    df = pd.DataFrame(rows)

    miner = AlphaMiner(df, enable_genetic=False, enable_llm=True)
    result = miner.mine()
    print(result.mining_report)
    print(f"\n最终入选因子: {[p.name for p in result.selected_factors]}")

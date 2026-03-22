"""
Financial Analysis Engine
==========================
专业财务分析模块 - 多维度基本面评估框架

核心功能：
  1. Piotroski F-Score    - 财务健康度评分（0-9分）
  2. Beneish M-Score      - 盈利操纵检测
  3. Altman Z-Score       - 破产风险评估
  4. DCF 估值模型          - 内在价值计算
  5. DuPont 分析          - ROE 拆解
  6. 股息分析              - 分红稳定性和成长性
  7. 行业比较分析          - 相对估值
  8. 成长质量评估          - 营收/利润成长质量
  9. 资本配置效率          - ROIC/WACC 分析
 10. 自由现金流分析        - FCF 质量和一致性
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quant_investor.logger import get_logger

_logger = get_logger("FinancialAnalysis")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class PiotroskiScore:
    """Piotroski F-Score (0–9)"""
    symbol: str
    total_score: int           # 0–9
    profitability_score: int   # 0–4: ROA > 0, CFO > 0, ΔROA > 0, 应计比 < 0
    leverage_score: int        # 0–3: Δ负债率<0, Δ流动比率>0, 无新股发行
    efficiency_score: int      # 0–2: Δ毛利率>0, Δ资产周转率>0
    grade: str                 # "强" | "中" | "弱"
    detail: dict = field(default_factory=dict)


@dataclass
class BeneishScore:
    """Beneish M-Score 盈利操纵检测"""
    symbol: str
    m_score: float
    manipulation_risk: str    # "高风险" | "中风险" | "低风险"
    dsri: float = 0.0         # Days Sales Receivable Index
    gmi: float = 0.0          # Gross Margin Index
    aqi: float = 0.0          # Asset Quality Index
    sgi: float = 0.0          # Sales Growth Index
    depi: float = 0.0         # Depreciation Index
    sgai: float = 0.0         # SG&A Expenses Index
    lvgi: float = 0.0         # Leverage Index
    tata: float = 0.0         # Total Accruals to Total Assets


@dataclass
class AltmanZScore:
    """Altman Z-Score 破产风险"""
    symbol: str
    z_score: float
    risk_zone: str            # "安全区" | "灰色区" | "危险区"
    x1: float = 0.0           # 营运资本/总资产
    x2: float = 0.0           # 留存收益/总资产
    x3: float = 0.0           # EBIT/总资产
    x4: float = 0.0           # 市值/总负债
    x5: float = 0.0           # 营业收入/总资产


@dataclass
class DCFValuation:
    """DCF 内在价值评估"""
    symbol: str
    intrinsic_value: float     # 每股内在价值（人民币）
    current_price: float
    margin_of_safety: float    # 安全边际（%）
    upside_downside: float     # 上行/下行空间（%）
    valuation_grade: str       # "严重低估" | "低估" | "合理" | "高估" | "严重高估"
    fcf_growth_rate: float     # 假设FCF增长率
    terminal_growth_rate: float
    discount_rate: float


@dataclass
class DuPontAnalysis:
    """DuPont 分析（ROE拆解）"""
    symbol: str
    roe: float                 # Return on Equity
    net_profit_margin: float   # 净利润率
    asset_turnover: float      # 资产周转率
    financial_leverage: float  # 财务杠杆
    roa: float                 # Return on Assets
    quality_score: float       # ROE质量评分（0–1）


@dataclass
class FinancialHealthReport:
    """综合财务健康报告"""
    symbol: str
    stock_name: str
    piotroski: Optional[PiotroskiScore] = None
    beneish: Optional[BeneishScore] = None
    altman: Optional[AltmanZScore] = None
    dcf: Optional[DCFValuation] = None
    dupont: Optional[DuPontAnalysis] = None
    overall_score: float = 0.0    # 综合评分 0–100
    overall_grade: str = "N/A"    # "A+" | "A" | "B" | "C" | "D"
    key_strengths: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# 核心分析引擎
# ---------------------------------------------------------------------------

class FinancialAnalyzer:
    """
    多维度财务分析引擎。

    使用方式：
    ----------
    analyzer = FinancialAnalyzer()
    report = analyzer.full_analysis(
        symbol="600519.SH",
        stock_name="贵州茅台",
        financial_data=df_financials,   # 财务指标 DataFrame
        price_data=df_price,            # 价格 DataFrame
        balance_sheet=df_balance,       # 资产负债表
        income_stmt=df_income,          # 利润表
        cash_flow=df_cashflow,          # 现金流量表
    )
    """

    def __init__(self, market: str = "CN") -> None:
        self.market = market

    # ------------------------------------------------------------------
    # 1. Piotroski F-Score
    # ------------------------------------------------------------------

    def calc_piotroski(
        self,
        symbol: str,
        financial_df: pd.DataFrame,
    ) -> PiotroskiScore:
        """
        计算 Piotroski F-Score。

        financial_df 需要包含以下列（过去2年数据，最新在上）：
            roa, cfo（经营现金流/总资产）, gross_margin,
            asset_turnover, current_ratio, debt_ratio,
            shares_outstanding, net_income, total_assets, revenue
        """
        detail: dict = {}
        prof = lev = eff = 0

        try:
            curr = financial_df.iloc[0]
            prev = financial_df.iloc[1] if len(financial_df) > 1 else curr

            # ---------- 盈利能力 (4分) ----------
            # F1: ROA > 0
            roa_curr = _safe_get(curr, ["roa", "ROA", "return_on_assets"], 0.0)
            f1 = 1 if roa_curr > 0 else 0
            detail["F1_ROA正"] = f"ROA={roa_curr:.3f} → {'✅' if f1 else '❌'}"

            # F2: 经营现金流 > 0
            cfo = _safe_get(curr, ["cfo", "CFO", "oper_cash_flow", "n_cashflow_act"], 0.0)
            total_assets = _safe_get(curr, ["total_assets", "total_hldr_eqy_exc_min_int"], 1.0)
            cfo_ratio = cfo / (total_assets + 1e-10)
            f2 = 1 if cfo_ratio > 0 else 0
            detail["F2_CFO正"] = f"CFO/总资产={cfo_ratio:.3f} → {'✅' if f2 else '❌'}"

            # F3: ΔROA > 0
            roa_prev = _safe_get(prev, ["roa", "ROA", "return_on_assets"], roa_curr)
            f3 = 1 if roa_curr > roa_prev else 0
            detail["F3_ROA改善"] = f"ΔROA={roa_curr - roa_prev:+.3f} → {'✅' if f3 else '❌'}"

            # F4: 应计比 < 0 (CFO > 净利润/总资产)
            net_income = _safe_get(curr, ["net_income", "n_income", "profit_dedt"], 0.0)
            ni_ratio = net_income / (total_assets + 1e-10)
            f4 = 1 if cfo_ratio > ni_ratio else 0
            detail["F4_应计比低"] = f"CFO/TA={cfo_ratio:.3f} vs NI/TA={ni_ratio:.3f} → {'✅' if f4 else '❌'}"

            prof = f1 + f2 + f3 + f4

            # ---------- 杠杆与偿债 (3分) ----------
            # F5: 长期负债率下降
            dr_curr = _safe_get(curr, ["debt_ratio", "debt_to_assets", "debt_asset_ratio"], 0.5)
            dr_prev = _safe_get(prev, ["debt_ratio", "debt_to_assets", "debt_asset_ratio"], dr_curr)
            f5 = 1 if dr_curr < dr_prev else 0
            detail["F5_负债率降"] = f"负债率: {dr_prev:.3f}→{dr_curr:.3f} → {'✅' if f5 else '❌'}"

            # F6: 流动比率提升
            cr_curr = _safe_get(curr, ["current_ratio", "curr_ratio"], 1.5)
            cr_prev = _safe_get(prev, ["current_ratio", "curr_ratio"], cr_curr)
            f6 = 1 if cr_curr > cr_prev else 0
            detail["F6_流动比率升"] = f"流动比率: {cr_prev:.2f}→{cr_curr:.2f} → {'✅' if f6 else '❌'}"

            # F7: 无稀释性增发
            shares_curr = _safe_get(curr, ["shares_outstanding", "total_share", "float_share"], 1.0)
            shares_prev = _safe_get(prev, ["shares_outstanding", "total_share", "float_share"], shares_curr)
            f7 = 1 if shares_curr <= shares_prev * 1.01 else 0
            detail["F7_无稀释增发"] = f"股本变化: {(shares_curr/shares_prev - 1)*100:.1f}% → {'✅' if f7 else '❌'}"

            lev = f5 + f6 + f7

            # ---------- 运营效率 (2分) ----------
            # F8: 毛利率提升
            gm_curr = _safe_get(curr, ["gross_profit_margin", "grossprofit_margin", "gross_margin"], 0.3)
            gm_prev = _safe_get(prev, ["gross_profit_margin", "grossprofit_margin", "gross_margin"], gm_curr)
            f8 = 1 if gm_curr > gm_prev else 0
            detail["F8_毛利率升"] = f"毛利率: {gm_prev:.3f}→{gm_curr:.3f} → {'✅' if f8 else '❌'}"

            # F9: 资产周转率提升
            at_curr = _safe_get(curr, ["asset_turnover", "assets_turn"], 0.5)
            at_prev = _safe_get(prev, ["asset_turnover", "assets_turn"], at_curr)
            f9 = 1 if at_curr > at_prev else 0
            detail["F9_周转率升"] = f"资产周转: {at_prev:.3f}→{at_curr:.3f} → {'✅' if f9 else '❌'}"

            eff = f8 + f9

        except Exception as e:
            _logger.warning(f"Piotroski计算异常 [{symbol}]: {e}")

        total = prof + lev + eff
        if total >= 7:
            grade = "强"
        elif total >= 4:
            grade = "中"
        else:
            grade = "弱"

        return PiotroskiScore(
            symbol=symbol,
            total_score=total,
            profitability_score=prof,
            leverage_score=lev,
            efficiency_score=eff,
            grade=grade,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # 2. Beneish M-Score（盈利操纵）
    # ------------------------------------------------------------------

    def calc_beneish(
        self,
        symbol: str,
        financial_df: pd.DataFrame,
    ) -> BeneishScore:
        """计算 Beneish M-Score（M < -2.22 认为操纵风险低）"""
        try:
            curr = financial_df.iloc[0]
            prev = financial_df.iloc[1] if len(financial_df) > 1 else curr

            # 应收账款日数指数
            rec_curr = _safe_get(curr, ["accounts_receivable", "ar", "notes_receiv"], 0.1)
            rec_prev = _safe_get(prev, ["accounts_receivable", "ar", "notes_receiv"], rec_curr)
            rev_curr = _safe_get(curr, ["revenue", "total_revenue", "total_operate_income"], 1.0)
            rev_prev = _safe_get(prev, ["revenue", "total_revenue", "total_operate_income"], rev_curr)
            dsri = (rec_curr / rev_curr) / (rec_prev / rev_prev + 1e-10) if rev_prev > 0 else 1.0

            # 毛利率指数
            gm_curr = _safe_get(curr, ["gross_profit_margin", "grossprofit_margin"], 0.3)
            gm_prev = _safe_get(prev, ["gross_profit_margin", "grossprofit_margin"], gm_curr)
            gmi = (1 - gm_prev) / (1 - gm_curr + 1e-10)

            # 资产质量指数
            ca_curr = _safe_get(curr, ["total_cur_assets", "current_assets"], 0.4)
            ta_curr = _safe_get(curr, ["total_assets", "total_assets_sum"], 1.0)
            ppe_curr = _safe_get(curr, ["fixed_assets", "ppe_net"], 0.3)
            ca_prev = _safe_get(prev, ["total_cur_assets", "current_assets"], ca_curr)
            ta_prev = _safe_get(prev, ["total_assets", "total_assets_sum"], ta_curr)
            ppe_prev = _safe_get(prev, ["fixed_assets", "ppe_net"], ppe_curr)
            nca_ratio_curr = 1 - (ca_curr + ppe_curr) / (ta_curr + 1e-10)
            nca_ratio_prev = 1 - (ca_prev + ppe_prev) / (ta_prev + 1e-10)
            aqi = nca_ratio_curr / (nca_ratio_prev + 1e-10)

            # 销售增长指数
            sgi = rev_curr / (rev_prev + 1e-10)

            # 折旧指数（折旧/固定资产）
            dep_curr = _safe_get(curr, ["depreciation", "depr_fa_coga_and_lp"], 0.05)
            dep_prev = _safe_get(prev, ["depreciation", "depr_fa_coga_and_lp"], dep_curr)
            depi = (dep_prev / (dep_prev + ppe_prev + 1e-10)) / (dep_curr / (dep_curr + ppe_curr + 1e-10) + 1e-10)

            # 销管费用指数
            sga_curr = _safe_get(curr, ["selling_general_admin", "sell_exp", "admin_exp"], 0.1)
            sga_prev = _safe_get(prev, ["selling_general_admin", "sell_exp", "admin_exp"], sga_curr)
            sgai = (sga_curr / rev_curr) / (sga_prev / rev_prev + 1e-10) if rev_prev > 0 else 1.0

            # 杠杆指数
            tl_curr = _safe_get(curr, ["total_liab", "total_liabilities"], 0.5 * ta_curr)
            tl_prev = _safe_get(prev, ["total_liab", "total_liabilities"], 0.5 * ta_prev)
            lvgi = (tl_curr / ta_curr) / (tl_prev / ta_prev + 1e-10)

            # 总应计比（TATA）
            ni_curr = _safe_get(curr, ["net_income", "n_income"], 0.0)
            cfo = _safe_get(curr, ["cfo", "n_cashflow_act"], ni_curr * 0.8)
            tata = (ni_curr - cfo) / (ta_curr + 1e-10)

            # M-Score 计算
            m = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi +
                 0.892 * sgi + 0.115 * depi - 0.172 * sgai +
                 4.679 * tata - 0.327 * lvgi)

            risk = "高风险" if m > -1.78 else ("中风险" if m > -2.22 else "低风险")

            return BeneishScore(
                symbol=symbol, m_score=round(m, 4), manipulation_risk=risk,
                dsri=round(dsri, 4), gmi=round(gmi, 4), aqi=round(aqi, 4),
                sgi=round(sgi, 4), depi=round(depi, 4), sgai=round(sgai, 4),
                lvgi=round(lvgi, 4), tata=round(tata, 4),
            )

        except Exception as e:
            _logger.warning(f"Beneish计算异常 [{symbol}]: {e}")
            return BeneishScore(symbol=symbol, m_score=-3.0, manipulation_risk="低风险")

    # ------------------------------------------------------------------
    # 3. Altman Z-Score
    # ------------------------------------------------------------------

    def calc_altman_z(
        self,
        symbol: str,
        financial_df: pd.DataFrame,
        market_cap: Optional[float] = None,
    ) -> AltmanZScore:
        """计算 Altman Z-Score（Z>2.99 安全, 1.81–2.99 灰色区, <1.81 危险）"""
        try:
            curr = financial_df.iloc[0]
            ta = _safe_get(curr, ["total_assets"], 1.0)

            # X1: 营运资本/总资产
            ca = _safe_get(curr, ["total_cur_assets", "current_assets"], 0.3 * ta)
            cl = _safe_get(curr, ["total_cur_liab", "current_liabilities"], 0.2 * ta)
            x1 = (ca - cl) / (ta + 1e-10)

            # X2: 留存收益/总资产
            re = _safe_get(curr, ["retained_earnings", "undistr_porfit", "surplus_rsrv"], 0.2 * ta)
            x2 = re / (ta + 1e-10)

            # X3: EBIT/总资产
            ebit = _safe_get(curr, ["ebit", "operate_income", "ebt"], 0.05 * ta)
            x3 = ebit / (ta + 1e-10)

            # X4: 市值/总负债（非上市公司用账面权益）
            tl = _safe_get(curr, ["total_liab", "total_liabilities"], 0.5 * ta)
            if market_cap and market_cap > 0:
                x4 = market_cap / (tl + 1e-10)
            else:
                equity = _safe_get(curr, ["total_hldr_eqy_exc_min_int", "shareholders_equity"], 0.5 * ta)
                x4 = equity / (tl + 1e-10)

            # X5: 营收/总资产
            rev = _safe_get(curr, ["revenue", "total_revenue", "total_operate_income"], 0.5 * ta)
            x5 = rev / (ta + 1e-10)

            z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

            if z > 2.99:
                zone = "安全区"
            elif z > 1.81:
                zone = "灰色区"
            else:
                zone = "危险区"

            return AltmanZScore(
                symbol=symbol, z_score=round(z, 4), risk_zone=zone,
                x1=round(x1, 4), x2=round(x2, 4), x3=round(x3, 4),
                x4=round(x4, 4), x5=round(x5, 4),
            )

        except Exception as e:
            _logger.warning(f"Altman Z-Score计算异常 [{symbol}]: {e}")
            return AltmanZScore(symbol=symbol, z_score=0.0, risk_zone="灰色区")

    # ------------------------------------------------------------------
    # 4. DCF 估值
    # ------------------------------------------------------------------

    def calc_dcf(
        self,
        symbol: str,
        fcf_history: list[float],           # 历史FCF（亿元），最新在后
        current_price: float,
        shares_outstanding: float,          # 总股本（亿股）
        growth_rates: Optional[list[float]] = None,
        terminal_growth: float = 0.03,
        discount_rate: float = 0.10,
        forecast_years: int = 10,
    ) -> DCFValuation:
        """
        两阶段 DCF 估值。
        Stage 1: 前5年使用预测成长率
        Stage 2: 后5年向终值成长率过渡
        """
        try:
            # 基准 FCF（近3年平均）
            if len(fcf_history) >= 3:
                base_fcf = float(np.mean(fcf_history[-3:]))
            elif fcf_history:
                base_fcf = float(fcf_history[-1])
            else:
                return DCFValuation(
                    symbol=symbol, intrinsic_value=0.0, current_price=current_price,
                    margin_of_safety=0.0, upside_downside=0.0,
                    valuation_grade="数据不足", fcf_growth_rate=0.0,
                    terminal_growth_rate=terminal_growth, discount_rate=discount_rate,
                )

            if base_fcf <= 0:
                return DCFValuation(
                    symbol=symbol, intrinsic_value=0.0, current_price=current_price,
                    margin_of_safety=0.0, upside_downside=0.0,
                    valuation_grade="FCF为负", fcf_growth_rate=0.0,
                    terminal_growth_rate=terminal_growth, discount_rate=discount_rate,
                )

            # 推断成长率（历史CAGR）
            if growth_rates is None:
                if len(fcf_history) >= 2 and fcf_history[0] > 0:
                    n = len(fcf_history) - 1
                    hist_cagr = (fcf_history[-1] / fcf_history[0]) ** (1 / n) - 1
                    hist_cagr = np.clip(hist_cagr, -0.1, 0.25)  # 限制在合理范围
                else:
                    hist_cagr = 0.05
                # 两阶段成长率：前5年保守用历史CAGR的70%
                g1 = hist_cagr * 0.7
                g2 = max(terminal_growth, hist_cagr * 0.4)
                growth_rates = [g1] * 5 + [g2] * 5
            else:
                while len(growth_rates) < forecast_years:
                    growth_rates.append(terminal_growth)

            # 计算各年FCF现值
            pv_sum = 0.0
            fcf = base_fcf
            for i, g in enumerate(growth_rates[:forecast_years]):
                fcf = fcf * (1 + g)
                pv = fcf / (1 + discount_rate) ** (i + 1)
                pv_sum += pv

            # 终值（Gordon Growth Model）
            terminal_fcf = fcf * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            terminal_pv = terminal_value / (1 + discount_rate) ** forecast_years

            intrinsic_total = (pv_sum + terminal_pv) * 1e8  # 亿元→元
            if shares_outstanding > 0:
                intrinsic_per_share = intrinsic_total / (shares_outstanding * 1e8)
            else:
                intrinsic_per_share = 0.0

            if current_price > 0 and intrinsic_per_share > 0:
                upside = (intrinsic_per_share / current_price - 1) * 100
                mos = upside  # 安全边际即上行空间
            else:
                upside = 0.0
                mos = 0.0

            if upside > 50:
                grade = "严重低估"
            elif upside > 20:
                grade = "低估"
            elif upside > -20:
                grade = "合理"
            elif upside > -40:
                grade = "高估"
            else:
                grade = "严重高估"

            return DCFValuation(
                symbol=symbol,
                intrinsic_value=round(intrinsic_per_share, 2),
                current_price=current_price,
                margin_of_safety=round(mos, 1),
                upside_downside=round(upside, 1),
                valuation_grade=grade,
                fcf_growth_rate=round(float(growth_rates[0]), 4),
                terminal_growth_rate=terminal_growth,
                discount_rate=discount_rate,
            )

        except Exception as e:
            _logger.warning(f"DCF计算异常 [{symbol}]: {e}")
            return DCFValuation(
                symbol=symbol, intrinsic_value=0.0, current_price=current_price,
                margin_of_safety=0.0, upside_downside=0.0, valuation_grade="计算错误",
                fcf_growth_rate=0.0, terminal_growth_rate=terminal_growth,
                discount_rate=discount_rate,
            )

    # ------------------------------------------------------------------
    # 5. DuPont 分析
    # ------------------------------------------------------------------

    def calc_dupont(
        self,
        symbol: str,
        financial_df: pd.DataFrame,
    ) -> DuPontAnalysis:
        """三因素 DuPont ROE 分解：ROE = 净利润率 × 资产周转率 × 财务杠杆"""
        try:
            curr = financial_df.iloc[0]
            roe = _safe_get(curr, ["roe", "ROE", "return_on_equity", "roe_yearly"], 0.0)
            npm = _safe_get(curr, ["net_profit_margin", "netprofit_margin", "net_margin"], 0.0)
            at  = _safe_get(curr, ["asset_turnover", "assets_turn"], 0.0)
            fl  = _safe_get(curr, ["financial_leverage", "equity_multiplier"], 1.0)
            roa = _safe_get(curr, ["roa", "ROA", "return_on_assets"], 0.0)

            if roe == 0 and npm > 0 and at > 0:
                roe = npm * at * fl

            # ROE质量评分：高ROE + 低杠杆 + 高利润率 = 高质量
            quality = 0.0
            if roe > 0.15:
                quality += 0.4
            elif roe > 0.10:
                quality += 0.2
            if fl < 2.0:
                quality += 0.3
            elif fl < 3.0:
                quality += 0.15
            if npm > 0.15:
                quality += 0.3
            elif npm > 0.08:
                quality += 0.15

            return DuPontAnalysis(
                symbol=symbol,
                roe=round(roe, 4),
                net_profit_margin=round(npm, 4),
                asset_turnover=round(at, 4),
                financial_leverage=round(fl, 4),
                roa=round(roa, 4),
                quality_score=round(quality, 3),
            )

        except Exception as e:
            _logger.warning(f"DuPont计算异常 [{symbol}]: {e}")
            return DuPontAnalysis(
                symbol=symbol, roe=0.0, net_profit_margin=0.0,
                asset_turnover=0.0, financial_leverage=1.0, roa=0.0, quality_score=0.0,
            )

    # ------------------------------------------------------------------
    # 综合分析
    # ------------------------------------------------------------------

    def full_analysis(
        self,
        symbol: str,
        stock_name: str,
        financial_df: pd.DataFrame,
        current_price: float = 0.0,
        market_cap: float = 0.0,
        fcf_history: Optional[list[float]] = None,
        shares_outstanding: float = 0.0,
    ) -> FinancialHealthReport:
        """
        运行所有财务分析模块并生成综合健康报告。
        """
        _logger.info(f"开始财务分析: {symbol} ({stock_name})")

        report = FinancialHealthReport(symbol=symbol, stock_name=stock_name)
        score_components: list[float] = []

        # 1. Piotroski F-Score
        if not financial_df.empty:
            report.piotroski = self.calc_piotroski(symbol, financial_df)
            score_components.append(report.piotroski.total_score / 9 * 100)
            _logger.debug(f"  Piotroski: {report.piotroski.total_score}/9 ({report.piotroski.grade})")

        # 2. Beneish M-Score
        if not financial_df.empty:
            report.beneish = self.calc_beneish(symbol, financial_df)
            beneish_score = 100 if report.beneish.manipulation_risk == "低风险" else (50 if report.beneish.manipulation_risk == "中风险" else 10)
            score_components.append(float(beneish_score))
            _logger.debug(f"  Beneish M: {report.beneish.m_score:.2f} ({report.beneish.manipulation_risk})")

        # 3. Altman Z-Score
        if not financial_df.empty:
            report.altman = self.calc_altman_z(symbol, financial_df, market_cap)
            z_score_component = 100 if report.altman.risk_zone == "安全区" else (50 if report.altman.risk_zone == "灰色区" else 10)
            score_components.append(float(z_score_component))
            _logger.debug(f"  Altman Z: {report.altman.z_score:.2f} ({report.altman.risk_zone})")

        # 4. DCF 估值
        if fcf_history and current_price > 0 and shares_outstanding > 0:
            report.dcf = self.calc_dcf(symbol, fcf_history, current_price, shares_outstanding)
            dcf_component = max(0, min(100, 50 + report.dcf.upside_downside))
            score_components.append(dcf_component)
            _logger.debug(f"  DCF: 内在价值={report.dcf.intrinsic_value:.2f} ({report.dcf.valuation_grade})")

        # 5. DuPont 分析
        if not financial_df.empty:
            report.dupont = self.calc_dupont(symbol, financial_df)
            score_components.append(report.dupont.quality_score * 100)
            _logger.debug(f"  DuPont: ROE={report.dupont.roe:.2%} 质量={report.dupont.quality_score:.2f}")

        # 综合评分
        if score_components:
            report.overall_score = round(float(np.mean(score_components)), 1)

        if report.overall_score >= 80:
            report.overall_grade = "A+"
        elif report.overall_score >= 70:
            report.overall_grade = "A"
        elif report.overall_score >= 55:
            report.overall_grade = "B"
        elif report.overall_score >= 40:
            report.overall_grade = "C"
        else:
            report.overall_grade = "D"

        # 提炼优势和风险
        report.key_strengths = self._extract_strengths(report)
        report.key_risks = self._extract_risks(report)
        report.summary = self._generate_summary(report)

        return report

    def _extract_strengths(self, r: FinancialHealthReport) -> list[str]:
        strengths = []
        if r.piotroski and r.piotroski.total_score >= 7:
            strengths.append(f"财务健康度优秀（Piotroski F={r.piotroski.total_score}/9）")
        if r.beneish and r.beneish.manipulation_risk == "低风险":
            strengths.append("盈利质量高，无操纵嫌疑")
        if r.altman and r.altman.risk_zone == "安全区":
            strengths.append(f"财务安全性强（Altman Z={r.altman.z_score:.2f}）")
        if r.dcf and r.dcf.upside_downside > 20:
            strengths.append(f"DCF估值显著低估（上行空间{r.dcf.upside_downside:.0f}%）")
        if r.dupont and r.dupont.roe > 0.15 and r.dupont.financial_leverage < 2:
            strengths.append(f"高质量ROE（{r.dupont.roe:.1%}，低杠杆）")
        return strengths[:5]

    def _extract_risks(self, r: FinancialHealthReport) -> list[str]:
        risks = []
        if r.piotroski and r.piotroski.total_score <= 3:
            risks.append(f"财务健康度较差（Piotroski F={r.piotroski.total_score}/9）")
        if r.beneish and r.beneish.manipulation_risk == "高风险":
            risks.append(f"盈利操纵风险高（M-Score={r.beneish.m_score:.2f}）")
        if r.altman and r.altman.risk_zone == "危险区":
            risks.append(f"破产风险较高（Altman Z={r.altman.z_score:.2f}）")
        if r.dcf and r.dcf.upside_downside < -30:
            risks.append(f"DCF估值显著高估（下行风险{abs(r.dcf.upside_downside):.0f}%）")
        if r.dupont and r.dupont.financial_leverage > 4:
            risks.append(f"财务杠杆过高（{r.dupont.financial_leverage:.1f}x）")
        return risks[:5]

    def _generate_summary(self, r: FinancialHealthReport) -> str:
        lines = [
            f"## {r.stock_name} ({r.symbol}) 财务健康报告\n\n",
            f"**综合评级**: {r.overall_grade}  **综合得分**: {r.overall_score:.0f}/100\n\n",
        ]

        if r.piotroski:
            lines.append(
                f"### Piotroski F-Score: {r.piotroski.total_score}/9 ({r.piotroski.grade})\n"
                f"- 盈利能力: {r.piotroski.profitability_score}/4 | "
                f"杠杆偿债: {r.piotroski.leverage_score}/3 | "
                f"运营效率: {r.piotroski.efficiency_score}/2\n\n"
            )
        if r.beneish:
            lines.append(
                f"### Beneish M-Score: {r.beneish.m_score:.3f} → {r.beneish.manipulation_risk}\n"
                f"- 临界值：-2.22（低于此值操纵风险低）\n\n"
            )
        if r.altman:
            lines.append(
                f"### Altman Z-Score: {r.altman.z_score:.3f} → {r.altman.risk_zone}\n"
                f"- X1(营运资本比)={r.altman.x1:.3f} | X2(留存比)={r.altman.x2:.3f} | "
                f"X3(EBIT比)={r.altman.x3:.3f}\n\n"
            )
        if r.dcf:
            lines.append(
                f"### DCF 内在价值: ¥{r.dcf.intrinsic_value:.2f} vs 当前¥{r.dcf.current_price:.2f}\n"
                f"- 空间: {r.dcf.upside_downside:+.1f}% → **{r.dcf.valuation_grade}**\n\n"
            )
        if r.dupont:
            lines.append(
                f"### DuPont 分析\n"
                f"- ROE={r.dupont.roe:.2%} = 净利润率{r.dupont.net_profit_margin:.2%} × "
                f"资产周转{r.dupont.asset_turnover:.2f} × 财务杠杆{r.dupont.financial_leverage:.2f}\n\n"
            )

        if r.key_strengths:
            lines.append("### 核心优势\n" + "\n".join(f"- ✅ {s}" for s in r.key_strengths) + "\n\n")
        if r.key_risks:
            lines.append("### 主要风险\n" + "\n".join(f"- ⚠️ {r2}" for r2 in r.key_risks) + "\n\n")

        return "".join(lines)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_get(row: pd.Series, keys: list[str], default: float = 0.0) -> float:
    """从 Series 中安全获取数值，尝试多个列名"""
    for k in keys:
        if k in row.index:
            val = row[k]
            if pd.notna(val) and val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return default

"""US macro risk terminal implementation."""

from __future__ import annotations

from typing import List, Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from quant_investor.config import config
from quant_investor.macro_terminal_tushare import MacroRiskTerminalBase
from quant_investor.macro_terminal_types import IndicatorResult, ModuleResult


class USMacroRiskTerminal(MacroRiskTerminalBase):
    """美股宏观风控终端 - FRED + yfinance"""

    MARKET = "US"
    MARKET_NAME = "美股"

    HISTORICAL_REFS = {
        'buffett_2000_peak': {'ratio': 183},
        'buffett_2021_peak': {'ratio': 205},
        'buffett_fair_value': {'low': 80, 'high': 120}
    }

    def __init__(self, fred_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.fred_api_key = fred_api_key or config.FRED_API_KEY if hasattr(config, 'FRED_API_KEY') else None
        self._fred = None

    @property
    def fred(self):
        """延迟加载 FRED API 客户端"""
        if self._fred is None and self.fred_api_key:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.fred_api_key)
                self._log("FRED API 初始化成功")
            except ImportError:
                self._log("fredapi 未安装，将使用 yfinance/AKShare 降级获取数据", "warning")
            except Exception as e:
                self._log(f"FRED API 初始化失败: {e}", "warning")
        return self._fred

    def get_modules(self) -> List[ModuleResult]:
        modules = []
        modules.append(self._analyze_monetary_policy())
        modules.append(self._analyze_growth())
        modules.append(self._analyze_valuation())
        modules.append(self._analyze_inflation())
        modules.append(self._analyze_sentiment())
        return modules

    def _analyze_monetary_policy(self) -> ModuleResult:
        module = ModuleResult("货币政策", "Monetary Policy")

        ffr = self._fetch_fred_series('FEDFUNDS')
        if ffr is not None:
            ind = IndicatorResult(name="联邦基金利率", value=round(ffr, 2), unit="%", data_source="FRED")
            if ffr >= 5.0:
                ind.status, ind.signal = "紧缩", "🔴"
            elif ffr >= 3.0:
                ind.status, ind.signal = "偏紧", "🟡"
            elif ffr >= 1.0:
                ind.status, ind.signal = "中性", "🟢"
            else:
                ind.status, ind.signal = "宽松", "🟢"
            ind.threshold_rules = ">=5%紧缩, 3-5%偏紧, 1-3%中性, <1%宽松"
            ind.analysis_detail = f"联邦基金利率{ffr:.2f}%"
            module.indicators.append(ind)

        bs = self._fetch_fred_series('WALCL')
        if bs is not None:
            bs_tn = bs / 1e6
            ind = IndicatorResult(name="美联储总资产", value=round(bs_tn, 2), unit="万亿美元", data_source="FRED")
            if bs_tn > 8.0:
                ind.status, ind.signal = "流动性充裕", "🟢"
            elif bs_tn > 6.0:
                ind.status, ind.signal = "缩表进行中", "🟡"
            else:
                ind.status, ind.signal = "正常水平", "🟢"
            ind.historical_ref = "疫情后峰值约9万亿，2020年前约4万亿"
            ind.analysis_detail = f"美联储总资产{bs_tn:.2f}万亿美元"
            module.indicators.append(ind)

        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals) if signals else "🟡"
        return module

    def _analyze_growth(self) -> ModuleResult:
        module = ModuleResult("经济增长", "Growth")

        gdp = self._fetch_fred_series('A191RL1Q225SBEA')
        if gdp is not None:
            ind = IndicatorResult(name="GDP年化季环比", value=round(gdp, 1), unit="%", data_source="FRED")
            if gdp > 3.0:
                ind.status, ind.signal = "强劲增长", "🟢"
            elif gdp > 1.5:
                ind.status, ind.signal = "温和增长", "🟢"
            elif gdp > 0:
                ind.status, ind.signal = "增长放缓", "🟡"
            elif gdp > -1.0:
                ind.status, ind.signal = "接近衰退", "🟡"
            else:
                ind.status, ind.signal = "衰退", "🔴"
            ind.threshold_rules = ">3%强劲, 1.5-3%温和, 0-1.5%放缓, -1~0%接近衰退, <-1%衰退"
            ind.analysis_detail = f"GDP年化季环比{gdp:.1f}%"
            module.indicators.append(ind)

        unemp = self._fetch_fred_series('UNRATE')
        if unemp is not None:
            ind = IndicatorResult(name="失业率", value=round(unemp, 1), unit="%", data_source="FRED")
            if unemp > 7.0:
                ind.status, ind.signal = "高失业", "🔴"
            elif unemp > 5.0:
                ind.status, ind.signal = "偏高", "🟡"
            elif unemp > 4.0:
                ind.status, ind.signal = "正常", "🟢"
            else:
                ind.status, ind.signal = "充分就业", "🟢"
            ind.threshold_rules = ">7%高失业, 5-7%偏高, 4-5%正常, <4%充分就业"
            ind.analysis_detail = f"失业率{unemp:.1f}%"
            module.indicators.append(ind)

        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals) if signals else "🟡"
        return module

    def _analyze_valuation(self) -> ModuleResult:
        module = ModuleResult("整体估值", "Valuation")

        buffett = self._fetch_fred_series('DDDM01USA156NWDB')
        if buffett is None:
            buffett = self._estimate_buffett_yfinance()
        if buffett is not None:
            ind = IndicatorResult(
                name="巴菲特指标(市值/GDP)", value=round(buffett, 1), unit="%",
                data_source="FRED" if self.fred else "yfinance(估算)",
                historical_ref="2000年泡沫~183%, 2021年泡沫~205%, 合理区间80-120%"
            )
            if buffett > 200:
                ind.status, ind.signal = "极度高估", "🔴"
            elif buffett > 150:
                ind.status, ind.signal = "显著高估", "🟡"
            elif buffett > 120:
                ind.status, ind.signal = "偏高", "🟡"
            elif buffett > 80:
                ind.status, ind.signal = "合理区间", "🟢"
            elif buffett > 60:
                ind.status, ind.signal = "低估", "🟢"
            else:
                ind.status, ind.signal = "极度低估", "🔵"
            ind.threshold_rules = ">200%极度高估, 150-200%显著高估, 120-150%偏高, 80-120%合理, 60-80%低估, <60%极度低估"
            ind.analysis_detail = f"巴菲特指标{buffett:.1f}%"
            module.indicators.append(ind)

        cape = self._fetch_sp500_pe()
        if cape is not None:
            ind = IndicatorResult(
                name="S&P 500 PE", value=round(cape, 1), unit="x",
                data_source="yfinance", historical_ref="Shiller PE历史均值约17x"
            )
            if cape > 35:
                ind.status, ind.signal = "显著高估", "🔴"
            elif cape > 25:
                ind.status, ind.signal = "偏高", "🟡"
            elif cape > 15:
                ind.status, ind.signal = "合理", "🟢"
            else:
                ind.status, ind.signal = "低估", "🔵"
            ind.threshold_rules = ">35x显著高估, 25-35x偏高, 15-25x合理, <15x低估"
            ind.analysis_detail = f"S&P 500 PE {cape:.1f}x"
            module.indicators.append(ind)

        signals = [ind.signal for ind in module.indicators if ind.value > 0]
        module.overall_signal = self._aggregate_signals(signals) if signals else "🟡"
        return module

    def _analyze_inflation(self) -> ModuleResult:
        module = ModuleResult("通胀", "Inflation")

        cpi = self._fetch_fred_yoy('CPIAUCSL')
        if cpi is not None:
            ind = IndicatorResult(name="CPI同比", value=round(cpi, 1), unit="%", data_source="FRED")
            if cpi > 5.0:
                ind.status, ind.signal = "高通胀", "🔴"
            elif cpi > 3.0:
                ind.status, ind.signal = "通胀偏高", "🟡"
            elif cpi >= 1.5:
                ind.status, ind.signal = "温和通胀", "🟢"
            elif cpi >= 0:
                ind.status, ind.signal = "低通胀", "🟡"
            else:
                ind.status, ind.signal = "通缩", "🔴"
            ind.threshold_rules = ">5%高通胀, 3-5%偏高, 1.5-3%温和, 0-1.5%低通胀, <0%通缩"
            ind.analysis_detail = f"CPI同比{cpi:.1f}%，美联储目标2%"
            module.indicators.append(ind)

        ppi = self._fetch_fred_yoy('PPIACO')
        if ppi is not None:
            ind = IndicatorResult(name="PPI同比", value=round(ppi, 1), unit="%", data_source="FRED")
            if ppi > 5.0:
                ind.status, ind.signal = "生产成本过热", "🔴"
            elif ppi > 2.0:
                ind.status, ind.signal = "偏高", "🟡"
            elif ppi >= 0:
                ind.status, ind.signal = "正常", "🟢"
            else:
                ind.status, ind.signal = "生产通缩", "🟡"
            ind.threshold_rules = ">5%过热, 2-5%偏高, 0-2%正常, <0%生产通缩"
            ind.analysis_detail = f"PPI同比{ppi:.1f}%"
            module.indicators.append(ind)

        pce = self._fetch_fred_yoy('PCEPILFE')
        if pce is not None:
            ind = IndicatorResult(name="核心PCE同比", value=round(pce, 1), unit="%", data_source="FRED")
            if pce > 4.0:
                ind.status, ind.signal = "核心通胀过高", "🔴"
            elif pce > 2.5:
                ind.status, ind.signal = "高于目标", "🟡"
            elif pce >= 1.5:
                ind.status, ind.signal = "接近目标", "🟢"
            else:
                ind.status, ind.signal = "低于目标", "🟡"
            ind.threshold_rules = ">4%过高, 2.5-4%高于目标, 1.5-2.5%接近目标, <1.5%低于目标"
            ind.analysis_detail = f"核心PCE同比{pce:.1f}%，美联储首选通胀指标"
            module.indicators.append(ind)

        signals = [ind.signal for ind in module.indicators if ind.value != 0]
        module.overall_signal = self._aggregate_signals(signals) if signals else "🟡"
        return module

    def _analyze_sentiment(self) -> ModuleResult:
        module = ModuleResult("情绪与收益率曲线", "Sentiment & Yield Curve")

        spread = self._fetch_yield_spread()
        if spread is not None:
            ind = IndicatorResult(name="10Y-2Y国债利差", value=round(spread, 0), unit="bp", data_source="FRED")
            if spread < -50:
                ind.status, ind.signal = "深度倒挂", "🔴"
                ind.historical_ref = "强烈衰退预警，历史上倒挂后12-18个月常出现衰退"
            elif spread < 0:
                ind.status, ind.signal = "倒挂", "🔴"
                ind.historical_ref = "衰退预警信号"
            elif spread < 50:
                ind.status, ind.signal = "平坦", "🟡"
                ind.historical_ref = "经济周期后期"
            else:
                ind.status, ind.signal = "正常", "🟢"
                ind.historical_ref = "经济扩张期"
            ind.threshold_rules = "<-50bp深度倒挂, <0倒挂, 0-50bp平坦, >50bp正常"
            ind.analysis_detail = f"10Y-2Y利差{spread:.0f}bp"
            module.indicators.append(ind)

        sentiment = self._fetch_fred_series('UMCSENT')
        if sentiment is not None:
            ind = IndicatorResult(
                name="消费者信心指数", value=round(sentiment, 1), unit="",
                data_source="FRED (UMich)", historical_ref="历史均值约85, 2022年低点约50"
            )
            if sentiment > 90:
                ind.status, ind.signal = "乐观", "🟢"
            elif sentiment > 70:
                ind.status, ind.signal = "中性", "🟢"
            elif sentiment > 55:
                ind.status, ind.signal = "悲观", "🟡"
            else:
                ind.status, ind.signal = "极度悲观", "🔴"
            ind.threshold_rules = ">90乐观, 70-90中性, 55-70悲观, <55极度悲观"
            ind.analysis_detail = f"密歇根消费者信心指数{sentiment:.1f}"
            module.indicators.append(ind)

        vix = self._fetch_vix()
        if vix is not None:
            ind = IndicatorResult(name="VIX恐慌指数", value=round(vix, 1), unit="", data_source="yfinance")
            if vix > 30:
                ind.status, ind.signal = "恐慌", "🔴"
                ind.historical_ref = "市场极度恐慌，可能是逆向买入机会"
            elif vix > 20:
                ind.status, ind.signal = "偏高", "🟡"
                ind.historical_ref = "市场不确定性增加"
            elif vix > 12:
                ind.status, ind.signal = "正常", "🟢"
                ind.historical_ref = "市场情绪稳定"
            else:
                ind.status, ind.signal = "极度平静", "🟡"
                ind.historical_ref = "可能过度自满，警惕黑天鹅"
            ind.threshold_rules = ">30恐慌, 20-30偏高, 12-20正常, <12极度平静"
            ind.analysis_detail = f"VIX {vix:.1f}"
            module.indicators.append(ind)

        signals = [ind.signal for ind in module.indicators if ind.value != 0]
        module.overall_signal = self._aggregate_signals(signals) if signals else "🟡"
        return module

    def _fetch_fred_series(self, series_id: str) -> Optional[float]:
        """获取FRED序列最新值"""
        if not self.fred:
            return None
        try:
            data = self.fred.get_series(series_id)
            if data is not None and len(data) > 0:
                val = float(data.dropna().iloc[-1])
                self._log(f"FRED获取{series_id}成功: {val}")
                return val
        except Exception as e:
            self._log(f"FRED获取{series_id}失败: {e}", "warning")
        return None

    def _fetch_fred_yoy(self, series_id: str) -> Optional[float]:
        """获取FRED月度序列的同比增速"""
        if not self.fred:
            return None
        try:
            data = self.fred.get_series(series_id)
            if data is not None and len(data) > 12:
                latest = float(data.dropna().iloc[-1])
                year_ago = float(data.dropna().iloc[-13])
                yoy = (latest / year_ago - 1) * 100
                self._log(f"FRED获取{series_id}同比成功: {yoy:.1f}%")
                return yoy
        except Exception as e:
            self._log(f"FRED获取{series_id}同比失败: {e}", "warning")
        return None

    def _fetch_yield_spread(self) -> Optional[float]:
        """获取10Y-2Y国债利差（bp）"""
        if self.fred:
            try:
                t10y = self.fred.get_series('DGS10')
                t2y = self.fred.get_series('DGS2')
                if t10y is not None and t2y is not None:
                    t10 = float(t10y.dropna().iloc[-1])
                    t2 = float(t2y.dropna().iloc[-1])
                    spread = (t10 - t2) * 100
                    self._log(f"FRED获取利差成功: 10Y={t10:.2f}%, 2Y={t2:.2f}%, 利差={spread:.0f}bp")
                    return spread
            except Exception as e:
                self._log(f"FRED获取利差失败: {e}", "warning")

        if YFINANCE_AVAILABLE:
            try:
                t10 = yf.Ticker("^TNX").history(period="5d")
                t2 = yf.Ticker("^IRX").history(period="5d")
                if not t10.empty and not t2.empty:
                    spread = (float(t10['Close'].iloc[-1]) - float(t2['Close'].iloc[-1])) * 100
                    self._log(f"yfinance获取利差: ~{spread:.0f}bp")
                    return spread
            except Exception as e:
                self._log(f"yfinance获取利差失败: {e}", "warning")
        return None

    def _estimate_buffett_yfinance(self) -> Optional[float]:
        """通过yfinance估算巴菲特指标"""
        if not YFINANCE_AVAILABLE:
            return None
        try:
            w5000 = yf.Ticker("^W5000")
            hist = w5000.history(period="5d")
            if hist is not None and not hist.empty:
                latest_price = float(hist['Close'].iloc[-1])
                est_market_cap = latest_price * 1.1
                est_gdp = 29000
                ratio = est_market_cap / est_gdp * 100 * 1000
                self._log(f"yfinance估算巴菲特指标: ~{ratio:.1f}%")
                return ratio
        except Exception as e:
            self._log(f"yfinance估算巴菲特指标失败: {e}", "warning")
        return None

    def _fetch_sp500_pe(self) -> Optional[float]:
        """获取S&P 500 PE"""
        if not YFINANCE_AVAILABLE:
            return None
        try:
            sp500 = yf.Ticker("^GSPC")
            info = sp500.info
            pe = info.get('trailingPE', None) or info.get('forwardPE', None)
            if pe:
                self._log(f"yfinance获取S&P 500 PE: {pe:.1f}x")
                return float(pe)
        except Exception as e:
            self._log(f"yfinance获取S&P 500 PE失败: {e}", "warning")
        return None

    def _fetch_vix(self) -> Optional[float]:
        """获取VIX"""
        if not YFINANCE_AVAILABLE:
            return None
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
            if hist is not None and not hist.empty:
                val = float(hist['Close'].iloc[-1])
                self._log(f"yfinance获取VIX: {val:.1f}")
                return val
        except Exception as e:
            self._log(f"yfinance获取VIX失败: {e}", "warning")
        return None


__all__ = ["USMacroRiskTerminal"]

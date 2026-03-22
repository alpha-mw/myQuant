#!/usr/bin/env python3
"""
MacroRiskTerminal V6.3 Enhanced - Tushare优先版本
多市场宏观风控终端 - 第0层风控

数据源优先级:
1. Tushare (首选) - 使用提供的token和自定义URL
2. AKShare (降级)
3. yfinance (美股)
4. 模拟数据 (最后备选)
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field, asdict

import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MacroRiskTerminal')

# ==================== Tushare配置 ====================
from quant_investor.config import config
from quant_investor.credential_utils import create_tushare_pro

TUSHARE_TOKEN = config.TUSHARE_TOKEN
TUSHARE_URL = config.TUSHARE_URL

# 初始化Tushare
try:
    import tushare as ts
    if TUSHARE_TOKEN:
        try:
            _probe = create_tushare_pro(ts, TUSHARE_TOKEN, TUSHARE_URL)
            if _probe is None:
                raise RuntimeError("missing_token")
            TUSHARE_AVAILABLE = True
            logger.info("Tushare 初始化成功（内存模式，无落盘持久化）")
        except Exception:
            TUSHARE_AVAILABLE = False
            logger.warning("Tushare 初始化失败，将使用降级数据源")
    else:
        TUSHARE_AVAILABLE = False
        logger.warning("TUSHARE_TOKEN未设置，将使用降级数据源")
except ImportError:
    TUSHARE_AVAILABLE = False
    logger.warning("Tushare未安装，将使用降级数据源")

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from fredapi import Fred as _FredProbe
    FREDAPI_AVAILABLE = True
except ImportError:
    FREDAPI_AVAILABLE = False


# ==================== 数据结构 ====================

@dataclass
class DataAcquisitionStep:
    """数据获取步骤记录"""
    timestamp: str
    data_source: str
    data_type: str
    attempt_method: str
    params: Dict[str, Any]
    result_status: str
    result_summary: str
    error_message: str = ""
    fallback_plan: str = ""


@dataclass
class AnalysisStep:
    """分析步骤记录"""
    timestamp: str
    step_name: str
    input_data: str
    analysis_method: str
    reasoning_process: str
    conclusion: str
    confidence: str = ""


@dataclass
class IndicatorResult:
    """单个指标的分析结果"""
    name: str
    value: float = 0.0
    unit: str = ""
    status: str = ""
    signal: str = "🟡"
    data_source: str = ""
    data_date: str = ""
    acquisition_steps: List[DataAcquisitionStep] = field(default_factory=list)
    historical_ref: str = ""
    analysis_steps: List[AnalysisStep] = field(default_factory=list)
    analysis_detail: str = ""
    threshold_rules: str = ""


@dataclass
class ModuleResult:
    """单个模块的分析结果"""
    module_name: str
    module_name_en: str
    indicators: List[IndicatorResult] = field(default_factory=list)
    overall_signal: str = "🟡"
    module_analysis_log: List[AnalysisStep] = field(default_factory=list)


@dataclass
class RiskTerminalReport:
    """宏观风控终端完整报告"""
    timestamp: str = ""
    version: str = "V6.4"
    market: str = ""
    market_name: str = ""
    modules: List[ModuleResult] = field(default_factory=list)
    overall_signal: str = "🟡"
    overall_risk_level: str = ""
    recommendation: str = ""
    execution_log: List[str] = field(default_factory=list)


# ==================== 基类 ====================

class MacroRiskTerminalBase(ABC):
    """宏观风控终端基类"""
    
    MARKET: str = ""
    MARKET_NAME: str = ""
    
    def __init__(self, cache_dir: str = '/tmp/macro_risk_cache', verbose: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.execution_log: List[str] = []
        self._logger = logging.getLogger(f"MacroRiskTerminal.{self.__class__.__name__}")
        if not self._logger.handlers:
            _handler = logging.StreamHandler()
            _handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self._logger.addHandler(_handler)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.propagate = False

        # 初始化Tushare Pro API
        self.pro = None
        if TUSHARE_AVAILABLE:
            try:
                self.pro = create_tushare_pro(ts, TUSHARE_TOKEN, TUSHARE_URL)
                self._log("Tushare Pro API初始化成功")
            except Exception:
                self._log("Tushare 初始化失败，已切换降级数据源", "warning")

    def _log(self, msg: str, level: str = "info") -> None:
        self.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_fn = getattr(self._logger, level if level in ("info", "warning", "error", "debug") else "info")
        log_fn(msg)
    
    @abstractmethod
    def get_modules(self) -> List[ModuleResult]:
        pass

    def _aggregate_signals(self, signals: List[str]) -> str:
        """聚合多个信号为模块综合信号"""
        if "🔴" in signals:
            return "🔴"
        elif "🟡" in signals:
            return "🟡"
        elif "🔵" in signals:
            return "🔵"
        return "🟢"
    
    def generate_risk_report(self) -> RiskTerminalReport:
        """生成完整报告"""
        self.execution_log = []
        self._log("=" * 80)
        self._log(f"{self.MARKET_NAME}宏观风控终端 V6.3 (Tushare优先) 开始运行")
        self._log("=" * 80)
        
        report = RiskTerminalReport(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            version="V6.4",
            market=self.MARKET,
            market_name=self.MARKET_NAME
        )
        
        report.modules = self.get_modules()
        
        # 计算综合信号
        all_signals = [m.overall_signal for m in report.modules]
        red_count = all_signals.count("🔴")
        yellow_count = all_signals.count("🟡")
        
        if red_count >= 2:
            report.overall_signal = "🔴"
            report.overall_risk_level = "高风险"
            report.recommendation = "降低仓位，防御为主"
        elif red_count >= 1 or yellow_count >= 2:
            report.overall_signal = "🟡"
            report.overall_risk_level = "中风险"
            report.recommendation = "控制仓位，精选个股"
        else:
            report.overall_signal = "🟢"
            report.overall_risk_level = "低风险"
            report.recommendation = "正常配置，积极布局"
        
        self._log(f"综合信号: {report.overall_signal} {report.overall_risk_level}")
        
        report.execution_log = self.execution_log.copy()
        return report
    
    def format_report_markdown(self, report: RiskTerminalReport) -> str:
        """格式化报告"""
        lines = []
        lines.append(f"# {report.market_name}宏观风控终端 ({report.version})")
        lines.append(f"**报告时间**: {report.timestamp}")
        lines.append("")
        lines.append(f"**综合信号**: {report.overall_signal} {report.overall_risk_level}")
        lines.append(f"**投资建议**: {report.recommendation}")
        lines.append("")
        
        for module in report.modules:
            lines.append(f"## {module.module_name} ({module.module_name_en}) {module.overall_signal}")
            lines.append("")
            
            for ind in module.indicators:
                lines.append(f"### {ind.name} {ind.signal}")
                lines.append(f"- **数值**: {ind.value} {ind.unit}")
                lines.append(f"- **状态**: {ind.status}")
                lines.append(f"- **数据源**: {ind.data_source}")
                if ind.historical_ref:
                    lines.append(f"- **历史对标**: {ind.historical_ref}")
                if ind.threshold_rules:
                    lines.append(f"- **判断依据**: {ind.threshold_rules}")
                if ind.analysis_detail:
                    lines.append(f"- **详细说明**: {ind.analysis_detail}")
                lines.append("")
        
        return "\n".join(lines)


# ==================== A股实现 ====================

class CNMacroRiskTerminal(MacroRiskTerminalBase):
    """A股宏观风控终端 - Tushare优先"""
    
    MARKET = "CN"
    MARKET_NAME = "A股"
    
    HISTORICAL_REFS = {
        'margin_2015_peak': {'balance': 2.27, 'ratio': 4.5},
        'buffett_2007_peak': 125.0,
        'buffett_2015_peak': 110.0,
        'buffett_bottom_range': (40.0, 60.0)
    }
    
    def get_modules(self) -> List[ModuleResult]:
        modules = []
        modules.append(self._analyze_leverage())
        modules.append(self._analyze_growth())
        modules.append(self._analyze_valuation())
        modules.append(self._analyze_inflation_money())
        modules.append(self._analyze_trade_fx())
        modules.append(self._analyze_fiscal_policy())
        return modules
    
    def _analyze_leverage(self) -> ModuleResult:
        """资金杠杆与情绪"""
        module = ModuleResult("资金杠杆与情绪", "Leverage")
        
        # 获取两融余额 - Tushare优先
        margin_balance = self._fetch_margin_balance_tushare()
        
        if margin_balance:
            margin_tn = margin_balance / 1e4
            margin_ind = IndicatorResult(
                name="两融余额",
                value=round(margin_tn, 2),
                unit="万亿",
                data_source="Tushare",
                historical_ref=f"2015牛市顶: {self.HISTORICAL_REFS['margin_2015_peak']['balance']}万亿"
            )
            
            if margin_tn > 2.0:
                margin_ind.status = "偏热"
                margin_ind.signal = "🟡"
                margin_ind.threshold_rules = ">2.0万亿为偏热"
            elif margin_tn > 1.5:
                margin_ind.status = "结构健康"
                margin_ind.signal = "🟢"
                margin_ind.threshold_rules = "1.5-2.0万亿为健康"
            else:
                margin_ind.status = "偏冷"
                margin_ind.signal = "🟡"
                margin_ind.threshold_rules = "<1.5万亿为偏冷"
            
            margin_ind.analysis_detail = f"两融余额{margin_tn:.2f}万亿"
            module.indicators.append(margin_ind)
        
        # 计算两融/流通市值比
        float_mv = self._fetch_float_market_value_tushare()
        if margin_balance and float_mv:
            ratio = margin_balance / float_mv * 100
            ratio_ind = IndicatorResult(
                name="两融/流通市值比",
                value=round(ratio, 2),
                unit="%",
                data_source="Tushare计算",
                historical_ref=f"2015牛市顶: {self.HISTORICAL_REFS['margin_2015_peak']['ratio']}%"
            )
            
            if ratio > 4.0:
                ratio_ind.status = "极度疯狂"
                ratio_ind.signal = "🔴"
                ratio_ind.threshold_rules = ">4.0%为极度疯狂"
            elif ratio > 3.0:
                ratio_ind.status = "偏热"
                ratio_ind.signal = "🟡"
                ratio_ind.threshold_rules = "3.0-4.0%为偏热"
            elif ratio > 2.0:
                ratio_ind.status = "结构健康"
                ratio_ind.signal = "🟢"
                ratio_ind.threshold_rules = "2.0-3.0%为健康"
            else:
                ratio_ind.status = "偏冷"
                ratio_ind.signal = "🟡"
                ratio_ind.threshold_rules = "<2.0%为偏冷"
            
            ratio_ind.analysis_detail = f"两融占比{ratio:.2f}%"
            module.indicators.append(ratio_ind)
        
        # 模块综合信号
        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals)
        
        return module
    
    def _fetch_margin_balance_tushare(self) -> Optional[float]:
        """从Tushare获取两融余额"""
        if not self.pro:
            return None
        
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            
            self._log(f"从Tushare获取两融余额: {start_date} 至 {end_date}")
            
            df = self.pro.margin(start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                balance = float(latest['rzye']) / 1e8  # 转为亿元
                self._log(f"Tushare获取两融余额成功: {balance:.0f}亿元")
                return balance
        except Exception as e:
            self._log(f"Tushare获取两融余额失败: {e}", "warning")
        
        # 降级到AKShare
        if AKSHARE_AVAILABLE:
            try:
                self._log("尝试AKShare获取两融余额...")
                df = ak.stock_margin_sse(start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'))
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    balance = float(latest.get('融资融券余额', 0)) / 1e8
                    self._log(f"AKShare获取两融余额成功: {balance:.0f}亿元")
                    return balance
            except Exception as e:
                self._log(f"AKShare获取两融余额失败: {e}", "warning")
        
        return None
    
    def _fetch_float_market_value_tushare(self) -> Optional[float]:
        """从Tushare获取流通市值"""
        if not self.pro:
            return None
        
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            self._log(f"从Tushare获取流通市值: {yesterday}")
            
            df = self.pro.daily_basic(trade_date=yesterday, fields='ts_code,circ_mv')
            if df is not None and not df.empty:
                total = df['circ_mv'].sum() / 1e4  # 转为亿元
                self._log(f"Tushare获取流通市值成功: {total:.0f}亿元")
                return total
        except Exception as e:
            self._log(f"Tushare获取流通市值失败: {e}", "warning")
        
        return None
    
    def _analyze_growth(self) -> ModuleResult:
        """经济景气度"""
        module = ModuleResult("经济景气度", "Growth")
        
        gdp = self._fetch_gdp_tushare()
        
        if gdp:
            gdp_ind = IndicatorResult(
                name="GDP同比增速",
                value=round(gdp, 1),
                unit="%",
                data_source="Tushare"
            )
            
            if gdp > 6.0:
                gdp_ind.status = "高速增长"
                gdp_ind.signal = "🟢"
            elif gdp > 5.0:
                gdp_ind.status = "稳健增长"
                gdp_ind.signal = "🟢"
            elif gdp > 4.0:
                gdp_ind.status = "中速增长"
                gdp_ind.signal = "🟡"
            else:
                gdp_ind.status = "增长放缓"
                gdp_ind.signal = "🟡"
            
            gdp_ind.analysis_detail = f"GDP增速{gdp:.1f}%"
            module.indicators.append(gdp_ind)
        
        # PMI
        pmi = self._fetch_pmi_tushare()
        if pmi is not None:
            pmi_ind = IndicatorResult(
                name="制造业PMI",
                value=round(pmi, 1),
                unit="",
                data_source="Tushare"
            )
            if pmi > 52:
                pmi_ind.status = "景气扩张"
                pmi_ind.signal = "🟢"
                pmi_ind.threshold_rules = ">52为景气扩张"
            elif pmi > 50:
                pmi_ind.status = "温和扩张"
                pmi_ind.signal = "🟡"
                pmi_ind.threshold_rules = "50-52为温和扩张"
            elif pmi > 48:
                pmi_ind.status = "收缩预警"
                pmi_ind.signal = "🟡"
                pmi_ind.threshold_rules = "48-50为收缩预警"
            else:
                pmi_ind.status = "显著收缩"
                pmi_ind.signal = "🔴"
                pmi_ind.threshold_rules = "<48为显著收缩"
            pmi_ind.analysis_detail = f"制造业PMI {pmi:.1f}，荣枯线50"
            pmi_ind.historical_ref = "荣枯线50，高于50为扩张，低于50为收缩"
            module.indicators.append(pmi_ind)

        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals) if signals else "🟡"
        return module

    def _fetch_pmi_tushare(self) -> Optional[float]:
        """从Tushare获取制造业PMI"""
        if not self.pro:
            return None
        try:
            self._log("从Tushare获取PMI数据...")
            end_m = datetime.now().strftime('%Y%m')
            start_m = (datetime.now() - timedelta(days=90)).strftime('%Y%m')
            df = self.pro.cn_pmi(start_m=start_m, end_m=end_m)
            if df is not None and not df.empty:
                pmi = float(df.iloc[0].get('pmi_mfg', 0))
                self._log(f"Tushare获取PMI成功: {pmi}")
                return pmi
        except Exception as e:
            self._log(f"Tushare获取PMI失败: {e}", "warning")

        if AKSHARE_AVAILABLE:
            try:
                self._log("尝试AKShare获取PMI...")
                df = ak.macro_china_pmi()
                if df is not None and not df.empty:
                    pmi = float(df.iloc[-1].get('制造业', df.iloc[-1].get('pmi', 0)))
                    self._log(f"AKShare获取PMI成功: {pmi}")
                    return pmi
            except Exception as e:
                self._log(f"AKShare获取PMI失败: {e}", "warning")

        return None

    def _fetch_gdp_tushare(self) -> Optional[float]:
        """从Tushare获取GDP"""
        if not self.pro:
            return None
        
        try:
            self._log("从Tushare获取GDP数据...")
            df = self.pro.cn_gdp()
            if df is not None and not df.empty:
                latest = df.iloc[0]
                gdp = float(latest.get('gdp_yoy', 0))
                self._log(f"Tushare获取GDP成功: {gdp}%")
                return gdp
        except Exception as e:
            self._log(f"Tushare获取GDP失败: {e}", "warning")
        
        return None
    
    def _analyze_valuation(self) -> ModuleResult:
        """整体估值锚"""
        module = ModuleResult("整体估值锚", "Valuation")
        
        total_mv = self._fetch_total_market_value_tushare()
        gdp = self._fetch_annual_gdp_tushare()
        
        if total_mv and gdp:
            buffett_ratio = (total_mv / 1e4 / gdp) * 100
            
            buffett_ind = IndicatorResult(
                name="巴菲特指标(市值/GDP)",
                value=round(buffett_ratio, 1),
                unit="%",
                data_source="Tushare计算",
                historical_ref=f"2007顶{self.HISTORICAL_REFS['buffett_2007_peak']}%, 2015顶{self.HISTORICAL_REFS['buffett_2015_peak']}%, 底部{self.HISTORICAL_REFS['buffett_bottom_range'][0]}-{self.HISTORICAL_REFS['buffett_bottom_range'][1]}%"
            )
            
            if buffett_ratio > 120:
                buffett_ind.status = "极度高估"
                buffett_ind.signal = "🔴"
                buffett_ind.threshold_rules = ">120%为极度高估"
            elif buffett_ratio > 100:
                buffett_ind.status = "估值偏高"
                buffett_ind.signal = "🟡"
                buffett_ind.threshold_rules = "100-120%为偏高"
            elif buffett_ratio > 80:
                buffett_ind.status = "合理区间"
                buffett_ind.signal = "🟢"
                buffett_ind.threshold_rules = "80-100%为合理"
            elif buffett_ratio > 60:
                buffett_ind.status = "低估区间"
                buffett_ind.signal = "🟢"
                buffett_ind.threshold_rules = "60-80%为低估"
            else:
                buffett_ind.status = "极度低估"
                buffett_ind.signal = "🔵"
                buffett_ind.threshold_rules = "<60%为极度低估"
            
            buffett_ind.analysis_detail = f"巴菲特指标{buffett_ratio:.1f}%"
            module.indicators.append(buffett_ind)
        
        module.overall_signal = module.indicators[0].signal if module.indicators else "🟡"
        return module
    
    def _fetch_total_market_value_tushare(self) -> Optional[float]:
        """从Tushare获取总市值"""
        if not self.pro:
            return None
        
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            self._log(f"从Tushare获取总市值: {yesterday}")
            
            df = self.pro.daily_basic(trade_date=yesterday, fields='ts_code,total_mv')
            if df is not None and not df.empty:
                total = df['total_mv'].sum() / 1e4
                self._log(f"Tushare获取总市值成功: {total:.0f}亿元")
                return total
        except Exception as e:
            self._log(f"Tushare获取总市值失败: {e}", "warning")
        
        return None
    
    def _fetch_annual_gdp_tushare(self) -> Optional[float]:
        """从Tushare获取年度GDP"""
        if not self.pro:
            return None
        
        try:
            self._log("从Tushare获取GDP...")
            df = self.pro.cn_gdp()
            if df is not None and not df.empty:
                latest = df.iloc[0]
                gdp = float(latest.get('gdp', 0)) / 1e4
                self._log(f"Tushare获取GDP成功: {gdp:.2f}万亿")
                return gdp
        except Exception as e:
            self._log(f"Tushare获取GDP失败: {e}", "warning")
        
        return None
    
    def _analyze_inflation_money(self) -> ModuleResult:
        """通胀与货币"""
        module = ModuleResult("通胀与货币", "Inflation & Money")
        
        # CPI
        cpi = self._fetch_cpi_tushare()
        if cpi:
            cpi_ind = IndicatorResult(
                name="CPI同比",
                value=round(cpi, 1),
                unit="%",
                data_source="Tushare"
            )
            if cpi > 3:
                cpi_ind.status = "通胀偏高"
                cpi_ind.signal = "🟡"
            elif cpi > 1:
                cpi_ind.status = "温和通胀"
                cpi_ind.signal = "🟢"
            else:
                cpi_ind.status = "低通胀"
                cpi_ind.signal = "🟡"
            module.indicators.append(cpi_ind)
        
        # PPI
        ppi = self._fetch_ppi_tushare()
        if ppi:
            ppi_ind = IndicatorResult(
                name="PPI同比",
                value=round(ppi, 1),
                unit="%",
                data_source="Tushare"
            )
            if ppi > 5:
                ppi_ind.status = "过热"
                ppi_ind.signal = "🔴"
            elif ppi > 0:
                ppi_ind.status = "正常"
                ppi_ind.signal = "🟢"
            else:
                ppi_ind.status = "下行"
                ppi_ind.signal = "🟡"
            module.indicators.append(ppi_ind)
        
        # M1-M2
        m1, m2 = self._fetch_m1_m2_tushare()
        if m1 and m2:
            scissors = m1 - m2
            scissors_ind = IndicatorResult(
                name="M1-M2剪刀差",
                value=round(scissors, 1),
                unit="%",
                data_source="Tushare"
            )
            if scissors > 0:
                scissors_ind.status = "资金活化"
                scissors_ind.signal = "🟢"
            elif scissors > -3:
                scissors_ind.status = "轻度定期化"
                scissors_ind.signal = "🟡"
            else:
                scissors_ind.status = "严重定期化"
                scissors_ind.signal = "🔴"
            module.indicators.append(scissors_ind)
            
            # M2增速
            m2_ind = IndicatorResult(
                name="M2增速",
                value=round(m2, 1),
                unit="%",
                data_source="Tushare"
            )
            if m2 > 10:
                m2_ind.status = "宽松"
                m2_ind.signal = "🟢"
            elif m2 > 8:
                m2_ind.status = "适度"
                m2_ind.signal = "🟡"
            else:
                m2_ind.status = "偏紧"
                m2_ind.signal = "🔴"
            module.indicators.append(m2_ind)
        
        # 社融当月增量
        sf_value = self._fetch_social_financing_tushare()
        if sf_value is not None:
            sf_ind = IndicatorResult(
                name="社融当月增量",
                value=round(sf_value, 0),
                unit="亿",
                data_source="Tushare"
            )
            if sf_value > 30000:
                sf_ind.status = "信用扩张"
                sf_ind.signal = "🟢"
                sf_ind.threshold_rules = ">30000亿为信用扩张"
            elif sf_value > 15000:
                sf_ind.status = "信用平稳"
                sf_ind.signal = "🟡"
                sf_ind.threshold_rules = "15000-30000亿为平稳"
            else:
                sf_ind.status = "信用收缩"
                sf_ind.signal = "🔴"
                sf_ind.threshold_rules = "<15000亿为信用收缩"
            sf_ind.analysis_detail = f"社融当月增量{sf_value:.0f}亿元，需结合历史同期对比"
            module.indicators.append(sf_ind)

        # 模块综合信号
        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals)

        return module

    def _fetch_social_financing_tushare(self) -> Optional[float]:
        """从Tushare获取社融当月增量"""
        if not self.pro:
            return None
        try:
            self._log("从Tushare获取社融数据...")
            df = self.pro.sf_month()
            if df is not None and not df.empty:
                latest = df.iloc[0]
                sf_value = float(latest.get('sf', latest.get('当月值', 0)))
                self._log(f"Tushare获取社融成功: {sf_value:.0f}亿")
                return sf_value
        except Exception as e:
            self._log(f"Tushare获取社融失败: {e}", "warning")

        if AKSHARE_AVAILABLE:
            try:
                self._log("尝试AKShare获取社融...")
                df = ak.macro_china_shrzgm()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    sf_value = float(latest.get('社会融资规模当月值', 0))
                    self._log(f"AKShare获取社融成功: {sf_value:.0f}亿")
                    return sf_value
            except Exception as e:
                self._log(f"AKShare获取社融失败: {e}", "warning")

        return None

    def _fetch_cpi_tushare(self) -> Optional[float]:
        if not self.pro:
            return None
        try:
            df = self.pro.cn_cpi()
            if df is not None and not df.empty:
                return float(df.iloc[0].get('cpi_yoy', 0))
        except Exception as e:
            self._log(f"Tushare获取CPI失败: {e}", "warning")
        return None
    
    def _fetch_ppi_tushare(self) -> Optional[float]:
        if not self.pro:
            return None
        try:
            df = self.pro.cn_ppi()
            if df is not None and not df.empty:
                return float(df.iloc[0].get('ppi_yoy', 0))
        except Exception as e:
            self._log(f"Tushare获取PPI失败: {e}", "warning")
        return None
    
    def _fetch_m1_m2_tushare(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.pro:
            return None, None
        try:
            df = self.pro.cn_m()
            if df is not None and not df.empty:
                latest = df.iloc[0]
                return float(latest.get('m1_yoy', 0)), float(latest.get('m2_yoy', 0))
        except Exception as e:
            self._log(f"Tushare获取M1/M2失败: {e}", "warning")
        return None, None
    

    def _analyze_trade_fx(self) -> ModuleResult:
        """外贸与外汇"""
        module = ModuleResult("外贸与外汇", "Trade & FX")

        # 贸易差额
        trade_balance = self._fetch_trade_balance()
        if trade_balance is not None:
            tb_ind = IndicatorResult(
                name="贸易差额",
                value=round(trade_balance, 1),
                unit="亿美元",
                data_source="AKShare"
            )
            if trade_balance > 500:
                tb_ind.status = "大幅顺差"
                tb_ind.signal = "🟢"
                tb_ind.threshold_rules = ">500亿美元为大幅顺差"
            elif trade_balance > 0:
                tb_ind.status = "顺差"
                tb_ind.signal = "🟡"
                tb_ind.threshold_rules = "0-500亿美元为温和顺差"
            else:
                tb_ind.status = "逆差"
                tb_ind.signal = "🔴"
                tb_ind.threshold_rules = "<0为逆差"
            tb_ind.analysis_detail = f"贸易差额{trade_balance:.1f}亿美元"
            module.indicators.append(tb_ind)

        # 外汇储备
        fx_reserve = self._fetch_fx_reserves()
        if fx_reserve is not None:
            fx_ind = IndicatorResult(
                name="外汇储备",
                value=round(fx_reserve, 2),
                unit="万亿美元",
                data_source="AKShare",
                historical_ref="2014峰值~3.99万亿, 2016低点~2.99万亿"
            )
            if fx_reserve > 3.1:
                fx_ind.status = "储备充裕"
                fx_ind.signal = "🟢"
                fx_ind.threshold_rules = ">3.1万亿为充裕"
            elif fx_reserve > 3.0:
                fx_ind.status = "储备关注"
                fx_ind.signal = "🟡"
                fx_ind.threshold_rules = "3.0-3.1万亿需关注"
            else:
                fx_ind.status = "储备预警"
                fx_ind.signal = "🔴"
                fx_ind.threshold_rules = "<3.0万亿为预警"
            fx_ind.analysis_detail = f"外汇储备{fx_reserve:.2f}万亿美元"
            module.indicators.append(fx_ind)

        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals) if signals else "🟡"
        return module

    def _fetch_trade_balance(self) -> Optional[float]:
        """获取贸易差额（亿美元）"""
        if not AKSHARE_AVAILABLE:
            return None
        try:
            self._log("从AKShare获取贸易差额...")
            df = ak.macro_china_trade_balance()
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                balance = float(latest.get('贸易差额', latest.get('trade_balance', 0)))
                self._log(f"AKShare获取贸易差额成功: {balance:.1f}亿美元")
                return balance
        except Exception as e:
            self._log(f"AKShare获取贸易差额失败: {e}", "warning")
        return None

    def _fetch_fx_reserves(self) -> Optional[float]:
        """获取外汇储备（万亿美元）"""
        if not AKSHARE_AVAILABLE:
            return None
        try:
            self._log("从AKShare获取外汇储备...")
            df = ak.macro_china_foreign_exchange_gold()
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                reserves = float(latest.get('外汇储备', latest.get('foreign_exchange', 0)))
                # AKShare返回值单位可能是亿美元，转换为万亿
                if reserves > 100:
                    reserves = reserves / 10000
                self._log(f"AKShare获取外汇储备成功: {reserves:.2f}万亿美元")
                return reserves
        except Exception as e:
            self._log(f"AKShare获取外汇储备失败: {e}", "warning")
        return None

    def _analyze_fiscal_policy(self) -> ModuleResult:
        """财政与政策"""
        module = ModuleResult("财政与政策", "Fiscal & Policy")

        fiscal_rev = self._fetch_fiscal_revenue()
        if fiscal_rev is not None:
            rev_ind = IndicatorResult(
                name="财政收入同比",
                value=round(fiscal_rev, 1),
                unit="%",
                data_source="AKShare"
            )
            if fiscal_rev > 8:
                rev_ind.status = "财政强劲"
                rev_ind.signal = "🟢"
                rev_ind.threshold_rules = ">8%为强劲"
            elif fiscal_rev > 3:
                rev_ind.status = "财政平稳"
                rev_ind.signal = "🟡"
                rev_ind.threshold_rules = "3-8%为平稳"
            else:
                rev_ind.status = "财政疲弱"
                rev_ind.signal = "🔴"
                rev_ind.threshold_rules = "<3%为疲弱"
            rev_ind.analysis_detail = f"财政收入同比{fiscal_rev:.1f}%"
            module.indicators.append(rev_ind)

        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals) if signals else "🟡"
        return module

    def _fetch_fiscal_revenue(self) -> Optional[float]:
        """获取财政收入同比增速"""
        if not AKSHARE_AVAILABLE:
            return None
        try:
            self._log("从AKShare获取财政收入...")
            df = ak.macro_china_national_tax_receipts()
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                yoy = float(latest.get('同比增长', latest.get('yoy', 0)))
                self._log(f"AKShare获取财政收入同比成功: {yoy:.1f}%")
                return yoy
        except Exception as e:
            self._log(f"AKShare获取财政收入失败: {e}", "warning")
        return None


# ==================== 美股实现 ====================

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

    # ---------- 模块1: 货币政策 ----------

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
            bs_tn = bs / 1e6  # 百万 -> 万亿美元
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

    # ---------- 模块2: 经济增长 ----------

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

    # ---------- 模块3: 整体估值 ----------

    def _analyze_valuation(self) -> ModuleResult:
        module = ModuleResult("整体估值", "Valuation")

        # 巴菲特指标: 尝试 FRED DDDM01USA156NWDB (World Bank 市值/GDP)
        buffett = self._fetch_fred_series('DDDM01USA156NWDB')
        if buffett is None:
            # 降级: Wilshire 5000 / GDP 估算
            buffett = self._estimate_buffett_yfinance()
        if buffett is not None:
            ind = IndicatorResult(
                name="巴菲特指标(市值/GDP)", value=round(buffett, 1), unit="%",
                data_source="FRED" if self.fred else "yfinance(估算)",
                historical_ref=f"2000年泡沫~183%, 2021年泡沫~205%, 合理区间80-120%"
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

        # Shiller PE (CAPE) - yfinance S&P 500 trailing PE 作为近似
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

    # ---------- 模块4: 通胀 ----------

    def _analyze_inflation(self) -> ModuleResult:
        module = ModuleResult("通胀", "Inflation")

        # CPI 同比
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

        # PPI 同比
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

        # 核心PCE 同比
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

    # ---------- 模块5: 情绪与收益率曲线 ----------

    def _analyze_sentiment(self) -> ModuleResult:
        module = ModuleResult("情绪与收益率曲线", "Sentiment & Yield Curve")

        # 10Y-2Y 国债利差
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

        # 消费者信心指数
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

        # VIX
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

    # ---------- 数据获取工具方法 ----------

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
                est_market_cap = latest_price * 1.1  # 近似总市值(十亿美元)
                est_gdp = 29000  # 美国GDP估值(十亿美元)
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


# ==================== 工厂函数 ====================

def create_terminal(market: str = "CN", **kwargs) -> MacroRiskTerminalBase:
    """创建对应市场的宏观风控终端"""
    market = market.upper()

    if market == "CN":
        return CNMacroRiskTerminal(**kwargs)
    elif market == "US":
        fred_api_key = kwargs.pop('fred_api_key', None)
        return USMacroRiskTerminal(fred_api_key=fred_api_key, **kwargs)
    else:
        raise ValueError(f"暂不支持市场 '{market}'。当前支持: CN, US")


def detect_market(tickers: Optional[List[str]] = None) -> str:
    """自动检测市场"""
    if tickers:
        for t in tickers:
            t_upper = t.upper()
            if any(t_upper.endswith(suffix) for suffix in ['.SZ', '.SH', '.BJ']):
                return "CN"
            elif t_upper.endswith('.HK'):
                return "HK"
            elif t_upper.isalpha():
                return "US"
    return "CN"


# 向后兼容
MacroRiskTerminal = CNMacroRiskTerminal


# ==================== 测试 ====================

if __name__ == '__main__':
    import sys
    
    market = sys.argv[1].upper() if len(sys.argv) > 1 else "CN"
    
    print(f"正在运行 {market} 市场宏观风控终端 (Tushare优先)...")
    print("=" * 80)
    
    terminal = create_terminal(market, verbose=True)
    report = terminal.generate_risk_report()
    
    print("\n" + terminal.format_report_markdown(report))
    
    # 保存报告
    output_file = f'/tmp/macro_risk_report_{market.lower()}.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(terminal.format_report_markdown(report))
    
    print(f"\n报告已保存: {output_file}")

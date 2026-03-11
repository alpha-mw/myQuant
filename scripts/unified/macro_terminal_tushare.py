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
import sys
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MacroRiskTerminal')

# ==================== Tushare配置 ====================
from config import config

TUSHARE_TOKEN = config.TUSHARE_TOKEN
TUSHARE_URL = config.TUSHARE_URL

# 初始化Tushare
try:
    import tushare as ts
    if TUSHARE_TOKEN:
        ts.set_token(TUSHARE_TOKEN)
        TUSHARE_AVAILABLE = True
        logger.info("Tushare初始化成功")
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
    version: str = "V6.3-Tushare"
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
                self.pro = ts.pro_api(TUSHARE_TOKEN)
                # 必须设置token和URL
                self.pro._DataApi__token = TUSHARE_TOKEN
                self.pro._DataApi__http_url = TUSHARE_URL
                self._log("Tushare Pro API初始化成功")
            except Exception as e:
                self._log(f"Tushare初始化失败: {e}", "warning")

    def _log(self, msg: str, level: str = "info") -> None:
        self.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        log_fn = getattr(self._logger, level if level in ("info", "warning", "error", "debug") else "info")
        log_fn(msg)
    
    @abstractmethod
    def get_modules(self) -> List[ModuleResult]:
        pass
    
    def generate_risk_report(self) -> RiskTerminalReport:
        """生成完整报告"""
        self.execution_log = []
        self._log("=" * 80)
        self._log(f"{self.MARKET_NAME}宏观风控终端 V6.3 (Tushare优先) 开始运行")
        self._log("=" * 80)
        
        report = RiskTerminalReport(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            version="V6.3-Tushare",
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
        
        module.overall_signal = module.indicators[0].signal if module.indicators else "🟡"
        return module
    
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
        
        # 模块综合信号
        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals)
        
        return module
    
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
    
    def _aggregate_signals(self, signals: List[str]) -> str:
        if "🔴" in signals:
            return "🔴"
        elif "🟡" in signals:
            return "🟡"
        elif "🔵" in signals:
            return "🔵"
        return "🟢"


# ==================== 美股实现 ====================

class USMacroRiskTerminal(MacroRiskTerminalBase):
    """美股宏观风控终端 - yfinance为主"""
    
    MARKET = "US"
    MARKET_NAME = "美股"
    
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
        
        ffr_ind = IndicatorResult(
            name="联邦基金利率",
            value=4.5,
            unit="%",
            data_source="模拟数据",
            status="偏紧",
            signal="🟡"
        )
        module.indicators.append(ffr_ind)
        
        bs_ind = IndicatorResult(
            name="美联储总资产",
            value=7.2,
            unit="万亿美元",
            data_source="模拟数据",
            status="缩表进行中",
            signal="🟡"
        )
        module.indicators.append(bs_ind)
        
        module.overall_signal = "🟡"
        return module
    
    def _analyze_growth(self) -> ModuleResult:
        module = ModuleResult("经济增长", "Growth")
        
        gdp_ind = IndicatorResult(
            name="GDP年化季环比",
            value=2.3,
            unit="%",
            data_source="模拟数据",
            status="温和增长",
            signal="🟢"
        )
        module.indicators.append(gdp_ind)
        
        unemp_ind = IndicatorResult(
            name="失业率",
            value=4.1,
            unit="%",
            data_source="模拟数据",
            status="正常",
            signal="🟢"
        )
        module.indicators.append(unemp_ind)
        
        module.overall_signal = "🟢"
        return module
    
    def _analyze_valuation(self) -> ModuleResult:
        module = ModuleResult("整体估值", "Valuation")
        
        buffett_ind = IndicatorResult(
            name="巴菲特指标",
            value=180.0,
            unit="%",
            data_source="模拟数据",
            status="显著高估",
            signal="🟡"
        )
        module.indicators.append(buffett_ind)
        
        cape_ind = IndicatorResult(
            name="Shiller PE",
            value=32.0,
            unit="x",
            data_source="模拟数据",
            status="偏高",
            signal="🟡"
        )
        module.indicators.append(cape_ind)
        
        module.overall_signal = "🟡"
        return module
    
    def _analyze_inflation(self) -> ModuleResult:
        module = ModuleResult("通胀", "Inflation")
        
        cpi_ind = IndicatorResult(
            name="CPI同比",
            value=3.2,
            unit="%",
            data_source="模拟数据",
            status="通胀偏高",
            signal="🟡"
        )
        module.indicators.append(cpi_ind)
        
        module.overall_signal = "🟡"
        return module
    
    def _analyze_sentiment(self) -> ModuleResult:
        module = ModuleResult("情绪与收益率曲线", "Sentiment")
        
        spread_ind = IndicatorResult(
            name="10Y-2Y利差",
            value=46.0,
            unit="bp",
            data_source="模拟数据",
            status="平坦",
            signal="🟡"
        )
        module.indicators.append(spread_ind)
        
        vix_ind = IndicatorResult(
            name="VIX",
            value=18.7,
            unit="",
            data_source="模拟数据",
            status="正常",
            signal="🟢"
        )
        module.indicators.append(vix_ind)
        
        module.overall_signal = "🟡"
        return module


# ==================== 工厂函数 ====================

def create_terminal(market: str = "CN", **kwargs) -> MacroRiskTerminalBase:
    """创建对应市场的宏观风控终端"""
    market = market.upper()
    
    if market == "CN":
        return CNMacroRiskTerminal(**kwargs)
    elif market == "US":
        return USMacroRiskTerminal(**kwargs)
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

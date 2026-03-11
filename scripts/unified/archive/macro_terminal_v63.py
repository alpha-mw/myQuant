#!/usr/bin/env python3
"""
MacroRiskTerminal V6.3 Enhanced
基于完整指标体系文档的增强版宏观风控终端

支持市场:
- CN (A股): 四大模块，完整指标体系
- US (美股): 五大模块，完整指标体系
- 可扩展: HK, EU, JP等
"""

import os
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# ==================== 数据结构 ====================

@dataclass
class IndicatorResult:
    """单个指标的分析结果"""
    name: str
    value: float = 0.0
    unit: str = ""
    status: str = ""
    signal: str = "🟡"
    historical_ref: str = ""
    data_date: str = ""
    data_source: str = ""
    analysis_detail: str = ""


@dataclass
class ModuleResult:
    """单个模块的分析结果"""
    module_name: str
    module_name_en: str
    indicators: List[IndicatorResult] = field(default_factory=list)
    overall_signal: str = "🟡"
    analysis_log: List[str] = field(default_factory=list)


@dataclass
class RiskTerminalReport:
    """宏观风控终端完整报告"""
    timestamp: str = ""
    version: str = "V6.3"
    market: str = ""
    market_name: str = ""
    modules: List[ModuleResult] = field(default_factory=list)
    overall_signal: str = "🟡"
    overall_risk_level: str = ""
    recommendation: str = ""
    data_acquisition_log: List[str] = field(default_factory=list)
    analysis_process_log: List[str] = field(default_factory=list)


# ==================== 基类 ====================

class MacroRiskTerminalBase(ABC):
    """宏观风控终端基类"""
    
    MARKET: str = ""
    MARKET_NAME: str = ""
    
    # 信号阈值配置
    SIGNAL_THRESHOLDS = {
        'high_risk': {'modules_red': 2},
        'medium_risk': {'modules_red': 1, 'modules_yellow': 2},
        'low_risk': {'default': 'green'},
        'extreme_low': {'modules_blue': 2}
    }
    
    def __init__(self, cache_dir: str = '/tmp/macro_risk_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_log: List[str] = []
        self.analysis_log: List[str] = []
    
    def _log_data(self, msg: str):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.data_log.append(entry)
    
    def _log_analysis(self, msg: str):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.analysis_log.append(entry)
    
    @abstractmethod
    def get_modules(self) -> List[ModuleResult]:
        """返回该市场的所有宏观风控模块"""
        pass
    
    def generate_risk_report(self) -> RiskTerminalReport:
        """生成完整的宏观风控终端报告"""
        self.data_log = []
        self.analysis_log = []
        
        report = RiskTerminalReport(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            version="V6.3",
            market=self.MARKET,
            market_name=self.MARKET_NAME
        )
        
        self._log_data("=" * 60)
        self._log_data(f"{self.MARKET_NAME}宏观风控终端 V6.3 开始运行")
        self._log_data("=" * 60)
        
        self._log_analysis("开始执行各模块分析...")
        
        # 获取各模块分析
        report.modules = self.get_modules()
        
        for i, m in enumerate(report.modules, 1):
            self._log_analysis(f"模块{i}[{m.module_name}] 完成, 信号: {m.overall_signal}")
        
        # 综合风控信号计算
        report.overall_signal, report.overall_risk_level, report.recommendation = \
            self._calculate_overall_signal(report.modules)
        
        self._log_analysis(
            f"综合风控信号: {report.overall_signal} {report.overall_risk_level} - {report.recommendation}"
        )
        
        report.data_acquisition_log = self.data_log.copy()
        report.analysis_process_log = self.analysis_log.copy()
        
        return report
    
    def _calculate_overall_signal(self, modules: List[ModuleResult]) -> Tuple[str, str, str]:
        """
        计算综合风控信号
        
        规则:
        - 🔴 高风险: 任意2个模块红色
        - 🟡 中风险: 任意1个红色或2个黄色
        - 🟢 低风险: 多数绿色
        - 🔵 极低风险: 多数蓝色(底部区域)
        """
        all_signals = [m.overall_signal for m in modules]
        red_count = all_signals.count("🔴")
        yellow_count = all_signals.count("🟡")
        blue_count = all_signals.count("🔵")
        
        if red_count >= 2:
            return "🔴", "高风险", "降低仓位，防御为主"
        elif red_count >= 1 or yellow_count >= 2:
            return "🟡", "中风险", "控制仓位，精选个股"
        elif blue_count >= 2:
            return "🔵", "极低风险", "加大配置，逆向布局"
        else:
            return "🟢", "低风险", "正常配置，积极布局"
    
    def format_report_markdown(self, report: RiskTerminalReport) -> str:
        """格式化报告为Markdown"""
        lines = []
        lines.append(f"## {report.market_name}宏观风控终端 ({report.version}) | {report.timestamp}")
        lines.append("")
        lines.append(f"**综合风控信号: {report.overall_signal} {report.overall_risk_level}** — {report.recommendation}")
        lines.append("")
        
        # 数据获取过程
        lines.append("### 📋 数据获取过程")
        lines.append("")
        for log in report.data_acquisition_log:
            lines.append(f"- {log}")
        lines.append("")
        
        # 各模块详情
        for module in report.modules:
            lines.append(f"### {module.module_name} ({module.module_name_en}) {module.overall_signal}")
            lines.append("")
            lines.append("| 核心指标 | 数值 | 状态/历史对标 |")
            lines.append("|:---|:---|:---|")
            for ind in module.indicators:
                value_str = f"{ind.value} {ind.unit}" if ind.value else "---"
                status_str = f"{ind.signal} {ind.status}" if ind.status else "---"
                if ind.historical_ref:
                    status_str += f" | {ind.historical_ref}"
                lines.append(f"| {ind.name} | {value_str} | {status_str} |")
            lines.append("")
            
            for ind in module.indicators:
                if ind.analysis_detail:
                    lines.append(f"> **{ind.name}分析**: {ind.analysis_detail}")
                    lines.append("")
        
        # 分析过程日志
        lines.append("### 🔍 分析过程日志")
        lines.append("")
        for log in report.analysis_process_log:
            lines.append(f"- {log}")
        lines.append("")
        
        return "\n".join(lines)


# ==================== A股 (CN) 宏观风控终端 ====================

class CNMacroRiskTerminal(MacroRiskTerminalBase):
    """
    A股宏观风控终端 - 四大模块
    1. 资金杠杆与情绪 (Leverage)
    2. 经济景气度 (Growth)  
    3. 整体估值锚 (Valuation)
    4. 通胀与货币 (Inflation & Money)
    """
    
    MARKET = "CN"
    MARKET_NAME = "A股"
    
    # 历史参考值
    HISTORICAL_REFS = {
        'margin_2015_peak': {'balance': 2.27, 'ratio': 4.5},
        'buffett_2007_peak': 125.0,
        'buffett_2015_peak': 110.0,
        'buffett_bottom_range': (40.0, 60.0)
    }
    
    def __init__(self, tushare_token: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.token = tushare_token or os.environ.get('TUSHARE_TOKEN')
        self.pro = None
        if self.token:
            try:
                import tushare as ts
                ts.set_token(self.token)
                self.pro = ts.pro_api()
            except:
                pass
    
    def get_modules(self) -> List[ModuleResult]:
        modules = []
        modules.append(self._analyze_leverage())
        modules.append(self._analyze_growth())
        modules.append(self._analyze_valuation())
        modules.append(self._analyze_inflation_money())
        return modules
    
    def _analyze_leverage(self) -> ModuleResult:
        """模块1: 资金杠杆与情绪"""
        module = ModuleResult("资金杠杆与情绪", "Leverage")
        
        # 模拟数据 (实际应接入Tushare/AKShare)
        margin_indicator = IndicatorResult(
            name="两融余额",
            value=1.85,
            unit="万亿",
            status="偏热",
            signal="🟡",
            historical_ref="2015牛市顶2.27万亿",
            analysis_detail="当前两融余额1.85万亿，为2015年顶部的81%，处于偏热区间"
        )
        module.indicators.append(margin_indicator)
        
        ratio_indicator = IndicatorResult(
            name="两融/流通市值比",
            value=2.8,
            unit="%",
            status="结构健康",
            signal="🟢",
            historical_ref="2015牛市顶4.5%",
            analysis_detail="两融占比2.8%，处于2-3%的健康区间"
        )
        module.indicators.append(ratio_indicator)
        
        # 模块综合信号
        signals = [ind.signal for ind in module.indicators]
        if "🔴" in signals:
            module.overall_signal = "🔴"
        elif "🟡" in signals:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"
        
        return module
    
    def _analyze_growth(self) -> ModuleResult:
        """模块2: 经济景气度"""
        module = ModuleResult("经济景气度", "Growth")
        
        gdp_indicator = IndicatorResult(
            name="GDP同比增速",
            value=5.2,
            unit="%",
            status="稳健增长",
            signal="🟢",
            analysis_detail="GDP增速5.2%，处于5-6%的稳健增长区间"
        )
        module.indicators.append(gdp_indicator)
        
        module.overall_signal = "🟢"
        return module
    
    def _analyze_valuation(self) -> ModuleResult:
        """模块3: 整体估值锚"""
        module = ModuleResult("整体估值锚", "Valuation")
        
        buffett_indicator = IndicatorResult(
            name="巴菲特指标(市值/GDP)",
            value=85.0,
            unit="%",
            status="合理偏高",
            signal="🟡",
            historical_ref="2007顶125%, 2015顶110%, 底部40-60%",
            analysis_detail="巴菲特指标85%，处于80-100%的合理偏高区间"
        )
        module.indicators.append(buffett_indicator)
        
        module.overall_signal = "🟡"
        return module
    
    def _analyze_inflation_money(self) -> ModuleResult:
        """模块4: 通胀与货币"""
        module = ModuleResult("通胀与货币", "Inflation & Money")
        
        cpi = IndicatorResult(
            name="CPI同比",
            value=2.1,
            unit="%",
            status="温和通胀",
            signal="🟢",
            analysis_detail="CPI 2.1%，处于1.5-3%的温和通胀区间"
        )
        module.indicators.append(cpi)
        
        ppi = IndicatorResult(
            name="PPI同比",
            value=-0.8,
            unit="%",
            status="工业价格下行",
            signal="🟡",
            analysis_detail="PPI -0.8%，处于-3~0%的下行区间"
        )
        module.indicators.append(ppi)
        
        m1m2 = IndicatorResult(
            name="M1-M2剪刀差",
            value=-1.5,
            unit="%",
            status="轻度存款定期化",
            signal="🟡",
            analysis_detail="剪刀差-1.5%，资金活化程度一般"
        )
        module.indicators.append(m1m2)
        
        m2 = IndicatorResult(
            name="M2增速",
            value=10.5,
            unit="%",
            status="宽松",
            signal="🟢",
            historical_ref=">10%宽松利好股市",
            analysis_detail="M2增速10.5%，流动性环境宽松"
        )
        module.indicators.append(m2)
        
        # 综合信号
        signals = [ind.signal for ind in module.indicators]
        red_count = signals.count("🔴")
        yellow_count = signals.count("🟡")
        
        if red_count >= 2:
            module.overall_signal = "🔴"
        elif red_count >= 1 or yellow_count >= 2:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"
        
        return module


# ==================== 美股 (US) 宏观风控终端 ====================

class USMacroRiskTerminal(MacroRiskTerminalBase):
    """
    美股宏观风控终端 - 五大模块
    1. 货币政策 (Monetary Policy)
    2. 经济增长 (Growth)
    3. 整体估值 (Valuation)
    4. 通胀 (Inflation)
    5. 情绪与收益率曲线 (Sentiment & Yield Curve)
    """
    
    MARKET = "US"
    MARKET_NAME = "美股"
    
    HISTORICAL_REFS = {
        'buffett_2000_peak': 183.0,
        'buffett_2021_peak': 205.0,
        'buffett_fair_value': (80.0, 120.0),
        'shiller_mean': 17.0
    }
    
    def __init__(self, fred_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.fred_key = fred_api_key or os.environ.get('FRED_API_KEY')
    
    def get_modules(self) -> List[ModuleResult]:
        modules = []
        modules.append(self._analyze_monetary_policy())
        modules.append(self._analyze_growth())
        modules.append(self._analyze_valuation())
        modules.append(self._analyze_inflation())
        modules.append(self._analyze_sentiment_yield())
        return modules
    
    def _analyze_monetary_policy(self) -> ModuleResult:
        """模块1: 货币政策"""
        module = ModuleResult("货币政策", "Monetary Policy")
        
        ffr = IndicatorResult(
            name="联邦基金利率",
            value=4.5,
            unit="%",
            status="偏紧",
            signal="🟡",
            historical_ref="关注转向信号",
            analysis_detail="利率4.5%，处于3-5%的偏紧区间，需关注美联储转向信号"
        )
        module.indicators.append(ffr)
        
        bs = IndicatorResult(
            name="美联储总资产",
            value=7.2,
            unit="万亿美元",
            status="缩表进行中",
            signal="🟡",
            historical_ref="峰值9万亿，疫情前4万亿",
            analysis_detail="资产负债表7.2万亿，处于6-8万亿的缩表区间"
        )
        module.indicators.append(bs)
        
        module.overall_signal = "🟡"
        return module
    
    def _analyze_growth(self) -> ModuleResult:
        """模块2: 经济增长"""
        module = ModuleResult("经济增长", "Growth")
        
        gdp = IndicatorResult(
            name="GDP年化季环比",
            value=2.3,
            unit="%",
            status="温和增长",
            signal="🟢",
            analysis_detail="GDP增速2.3%，处于1.5-3%的温和增长区间"
        )
        module.indicators.append(gdp)
        
        unemp = IndicatorResult(
            name="失业率",
            value=4.1,
            unit="%",
            status="正常",
            signal="🟢",
            historical_ref="充分就业区间",
            analysis_detail="失业率4.1%，就业市场健康"
        )
        module.indicators.append(unemp)
        
        module.overall_signal = "🟢"
        return module
    
    def _analyze_valuation(self) -> ModuleResult:
        """模块3: 整体估值"""
        module = ModuleResult("整体估值", "Valuation")
        
        # 使用yfinance获取Wilshire 5000估算
        buffett_ratio = self._get_buffett_ratio()
        
        buffett = IndicatorResult(
            name="巴菲特指标(Wilshire 5000/GDP)",
            value=round(buffett_ratio, 1) if buffett_ratio else 0,
            unit="%",
            status="偏高" if buffett_ratio and buffett_ratio < 200 else "极度高估",
            signal="🟡" if buffett_ratio and buffett_ratio < 200 else "🔴",
            historical_ref="2000泡沫183%, 2021泡沫205%",
            analysis_detail=f"巴菲特指标{buffett_ratio:.1f}%，处于150-200%的偏高区间" if buffett_ratio else "数据获取失败"
        )
        module.indicators.append(buffett)
        
        cape = IndicatorResult(
            name="Shiller PE (CAPE)",
            value=32.0,
            unit="x",
            status="偏高",
            signal="🟡",
            historical_ref="历史均值~17x",
            analysis_detail="CAPE 32x，高于历史均值，估值偏高"
        )
        module.indicators.append(cape)
        
        signals = [ind.signal for ind in module.indicators]
        if "🔴" in signals:
            module.overall_signal = "🔴"
        elif "🟡" in signals:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"
        
        return module
    
    def _get_buffett_ratio(self) -> Optional[float]:
        """获取巴菲特指标估算值"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            # 使用S&P 500作为Wilshire 5000的近似
            sp500 = yf.Ticker("^GSPC")
            info = sp500.info
            
            # 获取市值估算 (使用指数点位 * 成分股平均市值的简化估算)
            # 实际应使用Wilshire 5000总市值
            # 这里使用简化估算: S&P 500市值约占美股总市值的80%
            sp500_market_cap = info.get('marketCap', 0)
            if sp500_market_cap:
                total_market_cap = sp500_market_cap / 0.8  # 估算全市场市值
                
                # 美国GDP约27万亿美元 (2024年估算)
                us_gdp = 27.0 * 1e12
                
                buffett_ratio = (total_market_cap / us_gdp) * 100
                return buffett_ratio
        except:
            pass
        
        return None
    
    def _analyze_inflation(self) -> ModuleResult:
        """模块4: 通胀"""
        module = ModuleResult("通胀", "Inflation")
        
        cpi = IndicatorResult(
            name="CPI同比",
            value=3.2,
            unit="%",
            status="通胀偏高",
            signal="🟡",
            historical_ref="美联储目标2%",
            analysis_detail="CPI 3.2%，高于美联储2%目标，但已从高点回落"
        )
        module.indicators.append(cpi)
        
        ppi = IndicatorResult(
            name="PPI同比",
            value=1.8,
            unit="%",
            status="正常",
            signal="🟢",
            analysis_detail="PPI 1.8%，处于0-2%的正常区间"
        )
        module.indicators.append(ppi)
        
        pce = IndicatorResult(
            name="核心PCE同比",
            value=2.8,
            unit="%",
            status="高于目标",
            signal="🟡",
            historical_ref="美联储首选指标",
            analysis_detail="核心PCE 2.8%，略高于2.5%的舒适区间"
        )
        module.indicators.append(pce)
        
        signals = [ind.signal for ind in module.indicators]
        red_count = signals.count("🔴")
        yellow_count = signals.count("🟡")
        
        if red_count >= 2:
            module.overall_signal = "🔴"
        elif red_count >= 1 or yellow_count >= 2:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"
        
        return module
    
    def _analyze_sentiment_yield(self) -> ModuleResult:
        """模块5: 情绪与收益率曲线"""
        module = ModuleResult("情绪与收益率曲线", "Sentiment & Yield Curve")
        
        # 收益率曲线
        spread = self._get_yield_spread()
        yield_curve = IndicatorResult(
            name="10Y-2Y国债利差",
            value=round(spread, 0) if spread else 0,
            unit="bp",
            status="平坦" if spread and spread < 50 else "正常",
            signal="🟡" if spread and spread < 50 else "🟢",
            historical_ref="<0倒挂预警衰退",
            analysis_detail=f"利差{spread:.0f}bp，处于0-50bp的平坦区间，经济周期后期" if spread else "数据获取失败"
        )
        module.indicators.append(yield_curve)
        
        # VIX
        vix = self._get_vix()
        vix_indicator = IndicatorResult(
            name="VIX恐慌指数",
            value=round(vix, 1) if vix else 0,
            unit="",
            status="正常",
            signal="🟢",
            historical_ref="12-20正常区间",
            analysis_detail=f"VIX {vix:.1f}，市场情绪稳定" if vix else "数据获取失败"
        )
        module.indicators.append(vix_indicator)
        
        # 消费者信心
        sentiment = IndicatorResult(
            name="消费者信心指数",
            value=78.0,
            unit="",
            status="中性",
            signal="🟢",
            historical_ref="历史均值~85",
            analysis_detail="消费者信心78，接近历史均值"
        )
        module.indicators.append(sentiment)
        
        signals = [ind.signal for ind in module.indicators]
        if "🔴" in signals:
            module.overall_signal = "🔴"
        elif "🟡" in signals:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"
        
        return module
    
    def _get_yield_spread(self) -> Optional[float]:
        """获取10Y-2Y国债利差"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            t10 = yf.Ticker("^TNX")  # 10年期
            t2 = yf.Ticker("^IRX")   # 13周国库券近似2年期
            
            h10 = t10.history(period="5d")
            h2 = t2.history(period="5d")
            
            if not h10.empty and not h2.empty:
                rate_10y = float(h10['Close'].iloc[-1])
                rate_2y_approx = float(h2['Close'].iloc[-1])
                spread = (rate_10y - rate_2y_approx) * 100  # 转为bp
                return spread
        except:
            pass
        
        return None
    
    def _get_vix(self) -> Optional[float]:
        """获取VIX指数"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
            if hist is not None and not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        
        return None


# ==================== 工厂函数 ====================

def detect_market(tickers: Optional[List[str]] = None, market: Optional[str] = None) -> str:
    """自动检测市场"""
    if market:
        return market.upper()
    
    if tickers:
        for t in tickers:
            t_upper = t.upper()
            if t_upper.endswith(('.SZ', '.SH', '.BJ')):
                return "CN"
            elif t_upper.endswith('.HK'):
                return "HK"
            elif t_upper.isalpha():
                return "US"
    
    return "CN"


def create_terminal(market: str = "CN", **kwargs) -> MacroRiskTerminalBase:
    """工厂函数"""
    market = market.upper()
    
    if market == "CN":
        return CNMacroRiskTerminal(**kwargs)
    elif market == "US":
        return USMacroRiskTerminal(**kwargs)
    else:
        raise ValueError(f"暂不支持市场 '{market}'。当前支持: CN, US")


# 向后兼容
MacroRiskTerminal = CNMacroRiskTerminal


# ==================== 测试 ====================

if __name__ == '__main__':
    import sys
    
    market = sys.argv[1].upper() if len(sys.argv) > 1 else "US"
    
    print(f"正在运行 {market} 市场宏观风控终端 V6.3...")
    print("=" * 70)
    
    terminal = create_terminal(market)
    report = terminal.generate_risk_report()
    
    print(terminal.format_report_markdown(report))

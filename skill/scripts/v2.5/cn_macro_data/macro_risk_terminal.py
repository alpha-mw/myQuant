"""
MacroRiskTerminal - 多市场宏观风控终端 (Multi-Market Macro Risk Terminal)

V6.3 升级: 从纯A股终端扩展为多市场适配架构。
根据分析标的所在市场，自动适配对应国家/地区的宏观指标体系。

支持市场:
- CN (A股): 两融余额、GDP、巴菲特指标、CPI/PPI、M1-M2剪刀差、M2增速、社融
- US (美股): 联邦基金利率、CPI/PPI、GDP增速、国债收益率曲线、失业率、消费者信心指数、美联储资产负债表
- 可扩展: 港股(HK)、欧洲(EU)、日本(JP)等

架构: 基类(MacroRiskTerminalBase) + 市场子类(CNMacroRiskTerminal / USMacroRiskTerminal) + 工厂函数(create_terminal)
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
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

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
class IndicatorResult:
    """单个指标的分析结果"""
    name: str                    # 指标名称
    value: float = 0.0           # 当前值
    unit: str = ""               # 单位
    status: str = ""             # 状态判断
    signal: str = "🟡"           # 信号灯（🔴🟡🟢🔵）
    historical_ref: str = ""     # 历史对标说明
    data_date: str = ""          # 数据日期
    data_source: str = ""        # 数据来源
    analysis_detail: str = ""    # 详细分析过程说明


@dataclass
class ModuleResult:
    """单个模块的分析结果"""
    module_name: str             # 模块名称
    module_name_en: str          # 模块英文名
    indicators: List[IndicatorResult] = field(default_factory=list)
    overall_signal: str = "🟡"   # 模块综合信号
    analysis_log: List[str] = field(default_factory=list)


@dataclass
class RiskTerminalReport:
    """宏观风控终端完整报告"""
    timestamp: str = ""
    version: str = "V2.3"
    market: str = ""             # 市场标识: CN / US / HK / ...
    market_name: str = ""        # 市场名称: A股 / 美股 / ...
    modules: List[ModuleResult] = field(default_factory=list)
    overall_signal: str = "🟡"
    overall_risk_level: str = ""
    recommendation: str = ""
    data_acquisition_log: List[str] = field(default_factory=list)
    analysis_process_log: List[str] = field(default_factory=list)


# ==================== 基类: 宏观风控终端 ====================

class MacroRiskTerminalBase(ABC):
    """宏观风控终端基类 — 定义通用接口和综合信号计算逻辑"""

    MARKET: str = ""
    MARKET_NAME: str = ""

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
        """返回该市场的所有宏观风控模块分析结果（由子类实现）"""
        ...

    def generate_risk_report(self) -> RiskTerminalReport:
        """生成完整的宏观风控终端报告（通用逻辑）"""
        self.data_log = []
        self.analysis_log = []

        report = RiskTerminalReport(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            version="V2.3",
            market=self.MARKET,
            market_name=self.MARKET_NAME
        )

        self._log_data("=" * 60)
        self._log_data(f"{self.MARKET_NAME}宏观风控终端 开始运行")
        self._log_data("=" * 60)

        self._log_analysis("开始执行各模块分析...")

        report.modules = self.get_modules()

        for i, m in enumerate(report.modules, 1):
            self._log_analysis(f"模块{i}[{m.module_name}] 完成, 信号: {m.overall_signal}")

        # 综合风控信号（通用逻辑）
        all_signals = [m.overall_signal for m in report.modules]
        red_count = all_signals.count("🔴")
        yellow_count = all_signals.count("🟡")
        blue_count = all_signals.count("🔵")

        if red_count >= 2:
            report.overall_signal = "🔴"
            report.overall_risk_level = "高风险"
            report.recommendation = "降低仓位，防御为主"
        elif red_count >= 1 or yellow_count >= 2:
            report.overall_signal = "🟡"
            report.overall_risk_level = "中风险"
            report.recommendation = "控制仓位，精选个股"
        elif blue_count >= 2:
            report.overall_signal = "🔵"
            report.overall_risk_level = "极低风险"
            report.recommendation = "加大配置，逆向布局"
        else:
            report.overall_signal = "🟢"
            report.overall_risk_level = "低风险"
            report.recommendation = "正常配置，积极布局"

        self._log_analysis(
            f"综合风控信号: {report.overall_signal} {report.overall_risk_level} - {report.recommendation}"
        )

        report.data_acquisition_log = self.data_log.copy()
        report.analysis_process_log = self.analysis_log.copy()

        return report

    def format_report_markdown(self, report: RiskTerminalReport) -> str:
        """将报告格式化为Markdown（通用逻辑）"""
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


# ==================== CN (A股) 宏观风控终端 ====================

class CNMacroRiskTerminal(MacroRiskTerminalBase):
    """A股宏观风控终端 — 四大模块: Leverage / Growth / Valuation / Inflation & Money"""

    MARKET = "CN"
    MARKET_NAME = "A股"

    HISTORICAL_REFS = {
        'margin_2015_peak': {
            'margin_balance': 2.27,
            'margin_ratio': 4.5,
            'note': '2015年疯牛顶'
        },
        'buffett_2007_peak': {'ratio': 125.0, 'note': '2007年疯牛顶'},
        'buffett_2015_peak': {'ratio': 110.0, 'note': '2015年疯牛顶'},
        'buffett_bottom_range': {'low': 40.0, 'high': 60.0, 'note': '底部安全区间'}
    }

    def __init__(self, tushare_token: Optional[str] = None,
                 cache_dir: str = '/tmp/macro_risk_cache'):
        super().__init__(cache_dir=cache_dir)
        self.token = tushare_token or os.environ.get('TUSHARE_TOKEN')
        self.pro = None
        if self.token and TUSHARE_AVAILABLE:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            custom_url = os.environ.get('TUSHARE_HTTP_URL',
                                        'http://lianghua.nanyangqiankun.top')
            if custom_url:
                self.pro._DataApi__http_url = custom_url

    def get_modules(self) -> List[ModuleResult]:
        modules = []
        modules.append(self.analyze_leverage())
        modules.append(self.analyze_growth())
        modules.append(self.analyze_valuation())
        modules.append(self.analyze_inflation_money())
        return modules

    # ---------- 模块1: 资金杠杆与情绪 ----------

    def analyze_leverage(self) -> ModuleResult:
        module = ModuleResult(module_name="资金杠杆与情绪", module_name_en="Leverage")

        margin_balance = None
        margin_date = ""
        data_source = ""

        self._log_data("开始获取两融余额数据...")

        if self.pro:
            try:
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                df = self.pro.margin(start_date=start_date, end_date=end_date)
                if df is not None and not df.empty:
                    latest_date = df['trade_date'].max()
                    daily = df[df['trade_date'] == latest_date]
                    margin_balance = daily['rzye'].sum() / 1e8
                    margin_date = latest_date
                    data_source = "Tushare"
                    self._log_data(f"Tushare获取两融余额成功: {margin_balance:.0f}亿元, 日期={latest_date}")
            except Exception as e:
                self._log_data(f"Tushare获取两融余额失败: {e}")

        if margin_balance is None and AKSHARE_AVAILABLE:
            try:
                df = ak.stock_margin_sse(start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'))
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    margin_balance = float(latest.get('融资融券余额', 0)) / 1e8
                    margin_date = str(latest.get('信用交易日期', ''))
                    data_source = "AKShare"
                    self._log_data(f"AKShare获取两融余额成功: {margin_balance:.0f}亿元")
            except Exception as e:
                self._log_data(f"AKShare获取两融余额失败: {e}")

        # 获取A股流通市值
        float_mv = None
        self._log_data("开始获取A股流通市值数据...")

        if self.pro:
            try:
                today = datetime.now().strftime('%Y%m%d')
                df = self.pro.daily_basic(trade_date=today, fields='ts_code,float_share,close,circ_mv')
                if df is None or df.empty:
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                    df = self.pro.daily_basic(trade_date=yesterday, fields='ts_code,circ_mv')
                if df is not None and not df.empty:
                    float_mv = df['circ_mv'].sum() / 1e4
                    self._log_data(f"Tushare获取流通市值成功: {float_mv:.0f}亿元")
            except Exception as e:
                self._log_data(f"Tushare获取流通市值失败: {e}")

        margin_balance_tn = margin_balance / 1e4 if margin_balance else 0

        margin_indicator = IndicatorResult(
            name="两融余额", value=round(margin_balance_tn, 2) if margin_balance else 0,
            unit="万亿", data_date=margin_date, data_source=data_source
        )

        if margin_balance_tn > 0:
            ref = self.HISTORICAL_REFS['margin_2015_peak']
            pct_of_2015 = margin_balance_tn / ref['margin_balance'] * 100
            margin_indicator.historical_ref = (
                f"2015牛市顶参考: 两融余额 {ref['margin_balance']}万亿 / 占比 {ref['margin_ratio']}%"
            )
            self._log_analysis(
                f"两融余额 {margin_balance_tn:.2f}万亿, "
                f"为2015年顶部({ref['margin_balance']}万亿)的{pct_of_2015:.1f}%"
            )
            if margin_balance_tn > ref['margin_balance']:
                margin_indicator.status = "极度疯狂"
                margin_indicator.signal = "🔴"
            elif margin_balance_tn > ref['margin_balance'] * 0.8:
                margin_indicator.status = "偏热"
                margin_indicator.signal = "🟡"
            else:
                margin_indicator.status = "正常"
                margin_indicator.signal = "🟢"

        margin_indicator.analysis_detail = (
            f"获取了最近30天的两融余额数据(来源:{data_source})。"
            f"当前两融余额为{margin_balance_tn:.2f}万亿元。"
            f"对标2015年牛市顶部两融余额2.27万亿，判断当前杠杆水平。"
        )
        module.indicators.append(margin_indicator)

        # 两融/流通市值比
        ratio_indicator = IndicatorResult(
            name="两融/流通市值比", unit="%", data_date=margin_date, data_source=data_source
        )

        if margin_balance and float_mv and float_mv > 0:
            ratio = margin_balance / float_mv * 100
            ratio_indicator.value = round(ratio, 2)
            self._log_analysis(
                f"两融/流通市值比 = {margin_balance:.0f}亿 / {float_mv:.0f}亿 × 100% = {ratio:.2f}%"
            )
            if ratio > 4.0:
                ratio_indicator.status = "极度疯狂"
                ratio_indicator.signal = "🔴"
            elif ratio > 3.0:
                ratio_indicator.status = "偏热"
                ratio_indicator.signal = "🟡"
            elif ratio > 2.0:
                ratio_indicator.status = "结构健康"
                ratio_indicator.signal = "🟢"
            elif ratio > 1.5:
                ratio_indicator.status = "偏冷"
                ratio_indicator.signal = "🟡"
            else:
                ratio_indicator.status = "极度冷清"
                ratio_indicator.signal = "🔵"
            ratio_indicator.historical_ref = f"2015牛市顶占比4.5%, 当前{ratio:.2f}%"
            ratio_indicator.analysis_detail = (
                f"计算公式: 两融余额({margin_balance:.0f}亿) / 流通市值({float_mv:.0f}亿) × 100%。"
                f"结果为{ratio:.2f}%。"
                f"判断标准: >4%极度疯狂, 3-4%偏热, 2-3%结构健康, 1.5-2%偏冷, <1.5%极度冷清。"
            )
        module.indicators.append(ratio_indicator)

        signals = [ind.signal for ind in module.indicators if ind.signal]
        if "🔴" in signals:
            module.overall_signal = "🔴"
        elif "🟡" in signals:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"

        return module

    # ---------- 模块2: 经济景气度 ----------

    def analyze_growth(self) -> ModuleResult:
        module = ModuleResult(module_name="经济景气度", module_name_en="Growth")

        gdp_yoy = None
        gdp_quarter = ""
        data_source = ""

        self._log_data("开始获取GDP同比增速数据...")

        if self.pro:
            try:
                df = self.pro.cn_gdp()
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    gdp_yoy = float(latest.get('gdp_yoy', 0))
                    gdp_quarter = str(latest.get('quarter', ''))
                    data_source = "Tushare"
                    self._log_data(f"Tushare获取GDP成功: {gdp_yoy}%, 季度={gdp_quarter}")
            except Exception as e:
                self._log_data(f"Tushare获取GDP失败: {e}")

        if gdp_yoy is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_china_gdp()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    gdp_yoy = float(latest.get('同比增长', latest.get('累计同比', 0)))
                    gdp_quarter = str(latest.get('季度', ''))
                    data_source = "AKShare"
                    self._log_data(f"AKShare获取GDP成功: {gdp_yoy}%")
            except Exception as e:
                self._log_data(f"AKShare获取GDP失败: {e}")

        indicator = IndicatorResult(
            name="GDP同比增速",
            value=round(gdp_yoy, 1) if gdp_yoy else 0,
            unit="%", data_date=gdp_quarter, data_source=data_source
        )

        if gdp_yoy is not None:
            self._log_analysis(f"GDP同比增速为{gdp_yoy:.1f}%, 数据季度: {gdp_quarter}")
            if gdp_yoy > 6.0:
                indicator.status = "高速增长"
                indicator.signal = "🟢"
            elif gdp_yoy > 5.0:
                indicator.status = "稳健增长"
                indicator.signal = "🟢"
            elif gdp_yoy > 4.0:
                indicator.status = "中速增长"
                indicator.signal = "🟡"
            elif gdp_yoy > 3.0:
                indicator.status = "低速增长"
                indicator.signal = "🟡"
            else:
                indicator.status = "增长乏力"
                indicator.signal = "🔴"
            indicator.historical_ref = f"{gdp_quarter} 增速"
            indicator.analysis_detail = (
                f"获取了最新GDP季度数据(来源:{data_source})。"
                f"GDP同比增速为{gdp_yoy:.1f}%({gdp_quarter})。"
                f"判断标准: >6%高速增长, 5-6%稳健, 4-5%中速, 3-4%低速, <3%增长乏力。"
                f"当前判断: {indicator.status}。"
            )

        module.indicators.append(indicator)
        module.overall_signal = indicator.signal
        return module

    # ---------- 模块3: 整体估值锚 ----------

    def analyze_valuation(self) -> ModuleResult:
        module = ModuleResult(module_name="整体估值锚", module_name_en="Valuation")

        total_mv = None
        gdp_estimate = None
        data_source_mv = ""

        self._log_data("开始获取A股总市值数据...")

        if self.pro:
            try:
                today = datetime.now().strftime('%Y%m%d')
                df = self.pro.daily_basic(trade_date=today, fields='ts_code,total_mv')
                if df is None or df.empty:
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                    df = self.pro.daily_basic(trade_date=yesterday, fields='ts_code,total_mv')
                if df is not None and not df.empty:
                    total_mv = df['total_mv'].sum() / 1e4
                    data_source_mv = "Tushare"
                    self._log_data(f"Tushare获取A股总市值成功: {total_mv:.0f}亿元 ({total_mv/1e4:.2f}万亿元)")
            except Exception as e:
                self._log_data(f"Tushare获取总市值失败: {e}")

        self._log_data("开始获取/估算年度GDP数据...")

        if self.pro:
            try:
                df = self.pro.cn_gdp()
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    gdp_val = float(latest.get('gdp', 0))
                    quarter = str(latest.get('quarter', ''))
                    if 'Q4' in quarter or '四' in quarter:
                        gdp_estimate = gdp_val / 1e4
                    else:
                        q_num = 4
                        if 'Q1' in quarter or '一' in quarter:
                            q_num = 1
                        elif 'Q2' in quarter or '二' in quarter:
                            q_num = 2
                        elif 'Q3' in quarter or '三' in quarter:
                            q_num = 3
                        gdp_estimate = (gdp_val / q_num * 4) / 1e4
                    self._log_data(f"GDP估算: 基于{quarter}数据推算年度GDP约{gdp_estimate:.2f}万亿元")
            except Exception as e:
                self._log_data(f"获取GDP用于估值计算失败: {e}")

        mv_tn = total_mv / 1e4 if total_mv else 0
        mv_indicator = IndicatorResult(
            name="A股总市值", value=round(mv_tn, 2), unit="万亿",
            data_source=data_source_mv, data_date=datetime.now().strftime('%Y-%m-%d')
        )
        mv_indicator.analysis_detail = f"A股全市场总市值为{mv_tn:.2f}万亿元(来源:{data_source_mv})。"
        module.indicators.append(mv_indicator)

        gdp_indicator = IndicatorResult(
            name="年度GDP（预）",
            value=round(gdp_estimate, 2) if gdp_estimate else 0,
            unit="万亿", data_source="Tushare/推算"
        )
        gdp_indicator.analysis_detail = f"年度GDP预估值为{gdp_estimate:.2f}万亿元。" if gdp_estimate else "GDP数据获取失败。"
        module.indicators.append(gdp_indicator)

        buffett_indicator = IndicatorResult(name="市值/GDP（巴菲特）", unit="%")

        if total_mv and gdp_estimate and gdp_estimate > 0:
            buffett_ratio = mv_tn / gdp_estimate * 100
            buffett_indicator.value = round(buffett_ratio, 1)
            self._log_analysis(
                f"巴菲特指标 = A股总市值({mv_tn:.2f}万亿) / 年度GDP({gdp_estimate:.2f}万亿) × 100% = {buffett_ratio:.1f}%"
            )
            ref_2007 = self.HISTORICAL_REFS['buffett_2007_peak']
            ref_2015 = self.HISTORICAL_REFS['buffett_2015_peak']
            ref_bottom = self.HISTORICAL_REFS['buffett_bottom_range']
            buffett_indicator.historical_ref = (
                f"2007年疯牛顶~{ref_2007['ratio']:.0f}%, "
                f"2015年疯牛顶~{ref_2015['ratio']:.0f}%, "
                f"底部安全区间{ref_bottom['low']:.0f}%-{ref_bottom['high']:.0f}%"
            )
            if buffett_ratio > 120:
                buffett_indicator.status = "极度高估"
                buffett_indicator.signal = "🔴"
            elif buffett_ratio > 100:
                buffett_indicator.status = "估值偏高"
                buffett_indicator.signal = "🟡"
            elif buffett_ratio > 80:
                buffett_indicator.status = "合理偏高"
                buffett_indicator.signal = "🟡"
            elif buffett_ratio > 60:
                buffett_indicator.status = "合理区间"
                buffett_indicator.signal = "🟢"
            elif buffett_ratio > 40:
                buffett_indicator.status = "低估区间"
                buffett_indicator.signal = "🟢"
            else:
                buffett_indicator.status = "极度低估"
                buffett_indicator.signal = "🔵"
            buffett_indicator.analysis_detail = (
                f"计算公式: A股总市值({mv_tn:.2f}万亿) / 年度GDP({gdp_estimate:.2f}万亿) × 100%。"
                f"结果为{buffett_ratio:.1f}%。"
                f"历史对标: 2007年疯牛顶~125%, 2015年疯牛顶~110%, 底部安全区间40%-60%。"
                f"当前判断: {buffett_indicator.status}。"
            )

        module.indicators.append(buffett_indicator)
        signals = [ind.signal for ind in module.indicators if ind.signal and ind.name == "市值/GDP（巴菲特）"]
        module.overall_signal = signals[0] if signals else "🟡"
        return module

    # ---------- 模块4: 通胀与货币 ----------

    def analyze_inflation_money(self) -> ModuleResult:
        module = ModuleResult(module_name="通胀与货币", module_name_en="Inflation & Money")

        self._log_data("开始获取CPI同比数据...")
        module.indicators.append(self._get_cpi_indicator())

        self._log_data("开始获取PPI同比数据...")
        module.indicators.append(self._get_ppi_indicator())

        self._log_data("开始获取货币供应(M1/M2)数据...")
        m1m2, m2 = self._get_money_indicators()
        module.indicators.append(m1m2)
        module.indicators.append(m2)

        self._log_data("开始获取社融当月增量数据...")
        module.indicators.append(self._get_social_financing_indicator())

        signals = [ind.signal for ind in module.indicators if ind.signal]
        red_count = signals.count("🔴")
        yellow_count = signals.count("🟡")
        if red_count >= 2:
            module.overall_signal = "🔴"
        elif red_count >= 1 or yellow_count >= 2:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"

        return module

    def _get_cpi_indicator(self) -> IndicatorResult:
        indicator = IndicatorResult(name="CPI同比", unit="%")
        cpi_yoy = None

        if self.pro:
            try:
                df = self.pro.cn_cpi()
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    cpi_yoy = float(latest.get('nt_yoy', 0))
                    indicator.data_date = str(latest.get('month', ''))
                    indicator.data_source = "Tushare"
                    self._log_data(f"Tushare获取CPI成功: {cpi_yoy}%")
            except Exception as e:
                self._log_data(f"Tushare获取CPI失败: {e}")

        if cpi_yoy is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_china_cpi_monthly()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    cpi_yoy = float(latest.get('同比增长', 0))
                    indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取CPI成功: {cpi_yoy}%")
            except Exception as e:
                self._log_data(f"AKShare获取CPI失败: {e}")

        if cpi_yoy is not None:
            indicator.value = round(cpi_yoy, 1)
            if cpi_yoy > 3:
                indicator.status = "通胀偏高"
                indicator.signal = "🟡"
            elif cpi_yoy >= 1:
                indicator.status = "温和"
                indicator.signal = "🟢"
            elif cpi_yoy >= 0:
                indicator.status = "低通胀"
                indicator.signal = "🟡"
            else:
                indicator.status = "通缩"
                indicator.signal = "🔴"
            indicator.analysis_detail = (
                f"CPI同比为{cpi_yoy:.1f}%(来源:{indicator.data_source})。"
                f"判断标准: >3%通胀偏高, 1-3%温和, 0-1%低通胀, <0%通缩。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    def _get_ppi_indicator(self) -> IndicatorResult:
        indicator = IndicatorResult(name="PPI同比", unit="%")
        ppi_yoy = None

        if self.pro:
            try:
                df = self.pro.cn_ppi()
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    ppi_yoy = float(latest.get('ppi_yoy', 0))
                    indicator.data_date = str(latest.get('month', ''))
                    indicator.data_source = "Tushare"
                    self._log_data(f"Tushare获取PPI成功: {ppi_yoy}%")
            except Exception as e:
                self._log_data(f"Tushare获取PPI失败: {e}")

        if ppi_yoy is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_china_ppi_yearly()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    ppi_yoy = float(latest.get('同比增长', 0))
                    indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取PPI成功: {ppi_yoy}%")
            except Exception as e:
                self._log_data(f"AKShare获取PPI失败: {e}")

        if ppi_yoy is not None:
            indicator.value = round(ppi_yoy, 1)
            if ppi_yoy > 5:
                indicator.status = "工业品价格过热"
                indicator.signal = "🔴"
            elif ppi_yoy >= 0:
                indicator.status = "工业价格"
                indicator.signal = "🟢"
            elif ppi_yoy >= -3:
                indicator.status = "工业价格下行"
                indicator.signal = "🟡"
            else:
                indicator.status = "工业通缩"
                indicator.signal = "🔴"
            indicator.analysis_detail = (
                f"PPI同比为{ppi_yoy:.1f}%(来源:{indicator.data_source})。"
                f"判断标准: >5%过热, 0-5%正常, -3-0%下行, <-3%工业通缩。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    def _get_money_indicators(self) -> Tuple[IndicatorResult, IndicatorResult]:
        m1m2_indicator = IndicatorResult(name="M1-M2 剪刀差", unit="%")
        m2_indicator = IndicatorResult(name="M2增速", unit="%")
        m1_yoy = None
        m2_yoy = None

        if self.pro:
            try:
                df = self.pro.cn_m()
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    m1_yoy = float(latest.get('m1_yoy', 0))
                    m2_yoy = float(latest.get('m2_yoy', 0))
                    m1m2_indicator.data_date = str(latest.get('month', ''))
                    m1m2_indicator.data_source = "Tushare"
                    m2_indicator.data_date = m1m2_indicator.data_date
                    m2_indicator.data_source = "Tushare"
                    self._log_data(f"Tushare获取M1/M2成功: M1增速={m1_yoy}%, M2增速={m2_yoy}%")
            except Exception as e:
                self._log_data(f"Tushare获取M1/M2失败: {e}")

        if m2_yoy is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_china_money_supply()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    m1_yoy = float(latest.get('M1同比', 0))
                    m2_yoy = float(latest.get('M2同比', 0))
                    m1m2_indicator.data_source = "AKShare"
                    m2_indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取M1/M2成功: M1={m1_yoy}%, M2={m2_yoy}%")
            except Exception as e:
                self._log_data(f"AKShare获取M1/M2失败: {e}")

        if m1_yoy is not None and m2_yoy is not None:
            scissors = m1_yoy - m2_yoy
            m1m2_indicator.value = round(scissors, 1)
            self._log_analysis(f"M1-M2剪刀差 = M1增速({m1_yoy:.1f}%) - M2增速({m2_yoy:.1f}%) = {scissors:.1f}%")
            if scissors > 0:
                m1m2_indicator.status = "资金活化"
                m1m2_indicator.signal = "🟢"
            elif scissors >= -3:
                m1m2_indicator.status = "轻度存款定期化"
                m1m2_indicator.signal = "🟡"
            else:
                m1m2_indicator.status = "存款定期化"
                m1m2_indicator.signal = "🔴"
            m1m2_indicator.analysis_detail = (
                f"M1增速{m1_yoy:.1f}% - M2增速{m2_yoy:.1f}% = 剪刀差{scissors:.1f}%。"
                f"判断标准: >0资金活化, -3~0轻度定期化, <-3存款定期化严重。"
                f"当前判断: {m1m2_indicator.status}。"
            )

        if m2_yoy is not None:
            m2_indicator.value = round(m2_yoy, 1)
            if m2_yoy > 10:
                m2_indicator.status = "印钞速度"
                m2_indicator.signal = "🟢"
                m2_indicator.historical_ref = "宽松，利好股市"
            elif m2_yoy >= 8:
                m2_indicator.status = "印钞速度"
                m2_indicator.signal = "🟡"
                m2_indicator.historical_ref = "适度"
            else:
                m2_indicator.status = "印钞速度"
                m2_indicator.signal = "🔴"
                m2_indicator.historical_ref = "偏紧"
            m2_indicator.analysis_detail = (
                f"M2增速为{m2_yoy:.1f}%(来源:{m2_indicator.data_source})。"
                f"判断标准: >10%宽松(利好股市), 8-10%适度, <8%偏紧。"
            )

        return m1m2_indicator, m2_indicator

    def _get_social_financing_indicator(self) -> IndicatorResult:
        indicator = IndicatorResult(name="社融当月增量", unit="亿")
        sf_value = None

        if self.pro:
            try:
                df = self.pro.sf_month()
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    sf_value = float(latest.get('sf', latest.get('当月值', 0)))
                    indicator.data_date = str(latest.get('month', ''))
                    indicator.data_source = "Tushare"
                    self._log_data(f"Tushare获取社融成功: {sf_value:.0f}亿")
            except Exception as e:
                self._log_data(f"Tushare获取社融失败: {e}")

        if sf_value is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_china_shrzgm()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    sf_value = float(latest.get('社会融资规模当月值', 0))
                    indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取社融成功: {sf_value:.0f}亿")
            except Exception as e:
                self._log_data(f"AKShare获取社融失败: {e}")

        if sf_value is not None:
            indicator.value = round(sf_value, 0)
            if sf_value > 30000:
                indicator.status = "信用扩张"
                indicator.signal = "🟢"
            elif sf_value > 15000:
                indicator.status = "信用平稳"
                indicator.signal = "🟡"
            else:
                indicator.status = "信用收缩"
                indicator.signal = "🔴"
            indicator.analysis_detail = (
                f"社融当月增量为{sf_value:.0f}亿元(来源:{indicator.data_source})。"
                f"需结合历史同期数据对比判断信用扩张/收缩状态。"
                f"当前初步判断: {indicator.status}。"
            )

        return indicator


# ==================== US (美股) 宏观风控终端 ====================

class USMacroRiskTerminal(MacroRiskTerminalBase):
    """美股宏观风控终端 — 五大模块:
    1. Monetary Policy (联邦基金利率 / 美联储资产负债表)
    2. Growth (GDP增速 / 失业率)
    3. Valuation (巴菲特指标: Wilshire 5000 / GDP)
    4. Inflation (CPI / PPI / PCE)
    5. Sentiment & Yield Curve (消费者信心指数 / 国债收益率曲线)
    """

    MARKET = "US"
    MARKET_NAME = "美股"

    HISTORICAL_REFS = {
        'buffett_2000_peak': {'ratio': 183.0, 'note': '2000年互联网泡沫顶'},
        'buffett_2021_peak': {'ratio': 205.0, 'note': '2021年流动性泡沫顶'},
        'buffett_fair_value': {'low': 80.0, 'high': 120.0, 'note': '合理估值区间'},
        'buffett_undervalued': {'ratio': 70.0, 'note': '低估区间'},
    }

    def __init__(self, fred_api_key: Optional[str] = None,
                 cache_dir: str = '/tmp/macro_risk_cache'):
        super().__init__(cache_dir=cache_dir)
        self.fred_api_key = fred_api_key or os.environ.get('FRED_API_KEY')
        self._fred = None

    @property
    def fred(self):
        """延迟加载 FRED API 客户端"""
        if self._fred is None and self.fred_api_key:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.fred_api_key)
            except ImportError:
                self._log_data("fredapi 未安装, 将使用 yfinance/AKShare 降级获取数据")
        return self._fred

    def get_modules(self) -> List[ModuleResult]:
        modules = []
        modules.append(self.analyze_monetary_policy())
        modules.append(self.analyze_growth())
        modules.append(self.analyze_valuation())
        modules.append(self.analyze_inflation())
        modules.append(self.analyze_sentiment_yield_curve())
        return modules

    # ---------- 模块1: 货币政策 (Monetary Policy) ----------

    def analyze_monetary_policy(self) -> ModuleResult:
        module = ModuleResult(module_name="货币政策", module_name_en="Monetary Policy")

        # 1. 联邦基金利率 (Federal Funds Rate)
        self._log_data("开始获取联邦基金利率数据...")
        ffr_indicator = self._get_fed_funds_rate()
        module.indicators.append(ffr_indicator)

        # 2. 美联储资产负债表 (Fed Balance Sheet)
        self._log_data("开始获取美联储资产负债表数据...")
        bs_indicator = self._get_fed_balance_sheet()
        module.indicators.append(bs_indicator)

        signals = [ind.signal for ind in module.indicators if ind.signal]
        if "🔴" in signals:
            module.overall_signal = "🔴"
        elif signals.count("🟡") >= 1:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"

        return module

    def _get_fed_funds_rate(self) -> IndicatorResult:
        indicator = IndicatorResult(name="联邦基金利率", unit="%")
        ffr = None

        # 尝试 FRED
        if self.fred:
            try:
                data = self.fred.get_series('FEDFUNDS')
                if data is not None and len(data) > 0:
                    ffr = float(data.iloc[-1])
                    indicator.data_date = str(data.index[-1].strftime('%Y-%m'))
                    indicator.data_source = "FRED"
                    self._log_data(f"FRED获取联邦基金利率成功: {ffr}%")
            except Exception as e:
                self._log_data(f"FRED获取联邦基金利率失败: {e}")

        # 降级: AKShare
        if ffr is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_usa_interest_rate()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    ffr = float(latest.get('利率决议', latest.get('今值', 0)))
                    indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取联邦基金利率成功: {ffr}%")
            except Exception as e:
                self._log_data(f"AKShare获取联邦基金利率失败: {e}")

        if ffr is not None:
            indicator.value = round(ffr, 2)
            if ffr >= 5.0:
                indicator.status = "紧缩"
                indicator.signal = "🔴"
                indicator.historical_ref = "高利率环境，压制估值"
            elif ffr >= 3.0:
                indicator.status = "偏紧"
                indicator.signal = "🟡"
                indicator.historical_ref = "利率偏高，关注转向信号"
            elif ffr >= 1.0:
                indicator.status = "中性"
                indicator.signal = "🟢"
                indicator.historical_ref = "利率适中"
            else:
                indicator.status = "宽松"
                indicator.signal = "🟢"
                indicator.historical_ref = "低利率环境，利好风险资产"
            indicator.analysis_detail = (
                f"联邦基金利率为{ffr:.2f}%(来源:{indicator.data_source})。"
                f"判断标准: >=5%紧缩(压制估值), 3-5%偏紧, 1-3%中性, <1%宽松(利好风险资产)。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    def _get_fed_balance_sheet(self) -> IndicatorResult:
        indicator = IndicatorResult(name="美联储总资产", unit="万亿美元")
        total_assets = None

        if self.fred:
            try:
                data = self.fred.get_series('WALCL')  # 周频数据，百万美元
                if data is not None and len(data) > 0:
                    total_assets = float(data.iloc[-1]) / 1e6  # 百万 -> 万亿
                    indicator.data_date = str(data.index[-1].strftime('%Y-%m-%d'))
                    indicator.data_source = "FRED"
                    self._log_data(f"FRED获取美联储资产负债表成功: {total_assets:.2f}万亿美元")
            except Exception as e:
                self._log_data(f"FRED获取美联储资产负债表失败: {e}")

        if total_assets is not None:
            indicator.value = round(total_assets, 2)
            # 2020年前约4万亿，疫情后峰值约9万亿，当前缩表中
            if total_assets > 8.0:
                indicator.status = "资产负债表膨胀"
                indicator.signal = "🟢"
                indicator.historical_ref = "流动性充裕"
            elif total_assets > 6.0:
                indicator.status = "缩表进行中"
                indicator.signal = "🟡"
                indicator.historical_ref = "流动性收紧"
            else:
                indicator.status = "资产负债表正常"
                indicator.signal = "🟢"
                indicator.historical_ref = "疫情前水平"
            indicator.analysis_detail = (
                f"美联储总资产为{total_assets:.2f}万亿美元(来源:{indicator.data_source})。"
                f"疫情后峰值约9万亿，2020年前约4万亿。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    # ---------- 模块2: 经济增长 (Growth) ----------

    def analyze_growth(self) -> ModuleResult:
        module = ModuleResult(module_name="经济增长", module_name_en="Growth")

        # 1. GDP增速
        self._log_data("开始获取美国GDP增速数据...")
        gdp_indicator = self._get_us_gdp()
        module.indicators.append(gdp_indicator)

        # 2. 失业率
        self._log_data("开始获取美国失业率数据...")
        unemp_indicator = self._get_unemployment_rate()
        module.indicators.append(unemp_indicator)

        signals = [ind.signal for ind in module.indicators if ind.signal]
        if "🔴" in signals:
            module.overall_signal = "🔴"
        elif "🟡" in signals:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"

        return module

    def _get_us_gdp(self) -> IndicatorResult:
        indicator = IndicatorResult(name="GDP年化季环比", unit="%")
        gdp_growth = None

        if self.fred:
            try:
                data = self.fred.get_series('A191RL1Q225SBEA')  # 实际GDP年化季环比
                if data is not None and len(data) > 0:
                    gdp_growth = float(data.iloc[-1])
                    indicator.data_date = str(data.index[-1].strftime('%Y-Q'))
                    # 推算季度
                    q = (data.index[-1].month - 1) // 3 + 1
                    indicator.data_date = f"{data.index[-1].year}Q{q}"
                    indicator.data_source = "FRED"
                    self._log_data(f"FRED获取美国GDP增速成功: {gdp_growth}%")
            except Exception as e:
                self._log_data(f"FRED获取美国GDP增速失败: {e}")

        if gdp_growth is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_usa_gdp_monthly()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    gdp_growth = float(latest.get('今值', 0))
                    indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取美国GDP增速成功: {gdp_growth}%")
            except Exception as e:
                self._log_data(f"AKShare获取美国GDP增速失败: {e}")

        if gdp_growth is not None:
            indicator.value = round(gdp_growth, 1)
            if gdp_growth > 3.0:
                indicator.status = "强劲增长"
                indicator.signal = "🟢"
            elif gdp_growth > 1.5:
                indicator.status = "温和增长"
                indicator.signal = "🟢"
            elif gdp_growth > 0:
                indicator.status = "增长放缓"
                indicator.signal = "🟡"
            elif gdp_growth > -1.0:
                indicator.status = "接近衰退"
                indicator.signal = "🟡"
            else:
                indicator.status = "衰退"
                indicator.signal = "🔴"
            indicator.analysis_detail = (
                f"美国GDP年化季环比为{gdp_growth:.1f}%(来源:{indicator.data_source})。"
                f"判断标准: >3%强劲, 1.5-3%温和, 0-1.5%放缓, -1~0%接近衰退, <-1%衰退。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    def _get_unemployment_rate(self) -> IndicatorResult:
        indicator = IndicatorResult(name="失业率", unit="%")
        unemp = None

        if self.fred:
            try:
                data = self.fred.get_series('UNRATE')
                if data is not None and len(data) > 0:
                    unemp = float(data.iloc[-1])
                    indicator.data_date = str(data.index[-1].strftime('%Y-%m'))
                    indicator.data_source = "FRED"
                    self._log_data(f"FRED获取失业率成功: {unemp}%")
            except Exception as e:
                self._log_data(f"FRED获取失业率失败: {e}")

        if unemp is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_usa_unemployment_rate()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    unemp = float(latest.get('今值', 0))
                    indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取失业率成功: {unemp}%")
            except Exception as e:
                self._log_data(f"AKShare获取失业率失败: {e}")

        if unemp is not None:
            indicator.value = round(unemp, 1)
            if unemp > 7.0:
                indicator.status = "高失业"
                indicator.signal = "🔴"
                indicator.historical_ref = "经济衰退信号"
            elif unemp > 5.0:
                indicator.status = "偏高"
                indicator.signal = "🟡"
                indicator.historical_ref = "就业市场走弱"
            elif unemp > 4.0:
                indicator.status = "正常"
                indicator.signal = "🟢"
                indicator.historical_ref = "就业市场健康"
            else:
                indicator.status = "充分就业"
                indicator.signal = "🟢"
                indicator.historical_ref = "就业市场强劲，可能推升通胀"
            indicator.analysis_detail = (
                f"美国失业率为{unemp:.1f}%(来源:{indicator.data_source})。"
                f"判断标准: >7%高失业(衰退信号), 5-7%偏高, 4-5%正常, <4%充分就业。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    # ---------- 模块3: 整体估值 (Valuation) ----------

    def analyze_valuation(self) -> ModuleResult:
        module = ModuleResult(module_name="整体估值", module_name_en="Valuation")

        # 巴菲特指标: Wilshire 5000 Total Market Index / GDP
        self._log_data("开始获取美股巴菲特指标数据...")
        buffett_indicator = self._get_us_buffett_indicator()
        module.indicators.append(buffett_indicator)

        # S&P 500 Shiller PE (CAPE)
        self._log_data("开始获取Shiller PE (CAPE)数据...")
        cape_indicator = self._get_shiller_pe()
        module.indicators.append(cape_indicator)

        signals = [ind.signal for ind in module.indicators if ind.signal and ind.value > 0]
        if "🔴" in signals:
            module.overall_signal = "🔴"
        elif "🟡" in signals:
            module.overall_signal = "🟡"
        elif "🔵" in signals:
            module.overall_signal = "🔵"
        else:
            module.overall_signal = "🟢"

        return module

    def _get_us_buffett_indicator(self) -> IndicatorResult:
        indicator = IndicatorResult(name="Wilshire 5000/GDP（巴菲特指标）", unit="%")
        buffett_ratio = None

        if self.fred:
            try:
                # Wilshire 5000 Full Cap Price Index
                wilshire = self.fred.get_series('WILL5000PRFC')
                # 名义GDP (季频, 十亿美元)
                gdp = self.fred.get_series('GDP')
                if wilshire is not None and gdp is not None and len(wilshire) > 0 and len(gdp) > 0:
                    latest_wilshire = float(wilshire.iloc[-1])
                    latest_gdp = float(gdp.iloc[-1])
                    # Wilshire 5000 指数值近似等于总市值(十亿美元), GDP也是十亿美元
                    # 实际上 Wilshire 5000 Full Cap Price Index 需要乘以系数
                    # 使用 FRED 的 WILL5000IND (市值/GDP 比率) 更准确
                    indicator.data_source = "FRED"
                    self._log_data(f"FRED获取Wilshire 5000和GDP数据成功")
            except Exception as e:
                self._log_data(f"FRED获取巴菲特指标数据失败: {e}")

        # 尝试直接获取市值/GDP比率
        if self.fred and buffett_ratio is None:
            try:
                data = self.fred.get_series('DDDM01USA156NWDB')  # 市值/GDP
                if data is not None and len(data) > 0:
                    buffett_ratio = float(data.iloc[-1])
                    indicator.data_date = str(data.index[-1].strftime('%Y'))
                    indicator.data_source = "FRED (World Bank)"
                    self._log_data(f"FRED获取巴菲特指标成功: {buffett_ratio:.1f}%")
            except Exception as e:
                self._log_data(f"FRED获取巴菲特指标(World Bank)失败: {e}")

        # 降级: 使用 yfinance 获取 Wilshire 5000 + 手动计算
        if buffett_ratio is None and YFINANCE_AVAILABLE:
            try:
                w5000 = yf.Ticker("^W5000")
                hist = w5000.history(period="5d")
                if hist is not None and not hist.empty:
                    latest_price = float(hist['Close'].iloc[-1])
                    # Wilshire 5000 指数值 × 1.1 ≈ 总市值(十亿美元), 近似
                    est_market_cap = latest_price * 1.1  # 十亿美元
                    # 美国GDP约28万亿 = 28000十亿美元 (2024年)
                    est_gdp = 29000  # 十亿美元, 2025年预估
                    buffett_ratio = est_market_cap / est_gdp * 100 * 1000  # 修正比例
                    indicator.data_source = "yfinance (估算)"
                    self._log_data(f"yfinance估算巴菲特指标: ~{buffett_ratio:.1f}%")
            except Exception as e:
                self._log_data(f"yfinance获取巴菲特指标失败: {e}")

        if buffett_ratio is not None:
            indicator.value = round(buffett_ratio, 1)
            ref_2000 = self.HISTORICAL_REFS['buffett_2000_peak']
            ref_2021 = self.HISTORICAL_REFS['buffett_2021_peak']
            ref_fair = self.HISTORICAL_REFS['buffett_fair_value']
            indicator.historical_ref = (
                f"2000年互联网泡沫顶~{ref_2000['ratio']:.0f}%, "
                f"2021年流动性泡沫顶~{ref_2021['ratio']:.0f}%, "
                f"合理区间{ref_fair['low']:.0f}%-{ref_fair['high']:.0f}%"
            )
            if buffett_ratio > 200:
                indicator.status = "极度高估"
                indicator.signal = "🔴"
            elif buffett_ratio > 150:
                indicator.status = "显著高估"
                indicator.signal = "🟡"
            elif buffett_ratio > 120:
                indicator.status = "偏高"
                indicator.signal = "🟡"
            elif buffett_ratio > 80:
                indicator.status = "合理区间"
                indicator.signal = "🟢"
            elif buffett_ratio > 60:
                indicator.status = "低估"
                indicator.signal = "🟢"
            else:
                indicator.status = "极度低估"
                indicator.signal = "🔵"
            indicator.analysis_detail = (
                f"美股巴菲特指标(总市值/GDP)为{buffett_ratio:.1f}%(来源:{indicator.data_source})。"
                f"历史对标: 2000年互联网泡沫~183%, 2021年流动性泡沫~205%, 合理区间80-120%。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    def _get_shiller_pe(self) -> IndicatorResult:
        indicator = IndicatorResult(name="Shiller PE (CAPE)", unit="x")
        cape = None

        if self.fred:
            try:
                # Multpl.com 的 CAPE 数据在 FRED 上没有直接序列
                # 尝试通过 yfinance 或 web 获取
                pass
            except Exception:
                pass

        # 使用 AKShare 获取
        if AKSHARE_AVAILABLE:
            try:
                # AKShare 可能没有直接的 CAPE 数据
                pass
            except Exception:
                pass

        # 使用 yfinance 获取 S&P 500 PE
        if cape is None and YFINANCE_AVAILABLE:
            try:
                sp500 = yf.Ticker("^GSPC")
                info = sp500.info
                pe = info.get('trailingPE', None) or info.get('forwardPE', None)
                if pe:
                    # 注意: 这是trailing PE，不是CAPE，但可作参考
                    cape = float(pe)
                    indicator.name = "S&P 500 PE"
                    indicator.data_source = "yfinance"
                    indicator.data_date = datetime.now().strftime('%Y-%m-%d')
                    self._log_data(f"yfinance获取S&P 500 PE成功: {cape:.1f}x")
            except Exception as e:
                self._log_data(f"yfinance获取S&P 500 PE失败: {e}")

        if cape is not None:
            indicator.value = round(cape, 1)
            # Shiller PE / CAPE 历史均值约 17, 当前常在 30+
            if cape > 35:
                indicator.status = "显著高估"
                indicator.signal = "🔴"
                indicator.historical_ref = "远高于历史均值(~17x)"
            elif cape > 25:
                indicator.status = "偏高"
                indicator.signal = "🟡"
                indicator.historical_ref = "高于历史均值"
            elif cape > 15:
                indicator.status = "合理"
                indicator.signal = "🟢"
                indicator.historical_ref = "接近历史均值"
            else:
                indicator.status = "低估"
                indicator.signal = "🔵"
                indicator.historical_ref = "低于历史均值"
            indicator.analysis_detail = (
                f"S&P 500 PE为{cape:.1f}x(来源:{indicator.data_source})。"
                f"Shiller PE(CAPE)历史均值约17x。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    # ---------- 模块4: 通胀 (Inflation) ----------

    def analyze_inflation(self) -> ModuleResult:
        module = ModuleResult(module_name="通胀", module_name_en="Inflation")

        self._log_data("开始获取美国CPI数据...")
        module.indicators.append(self._get_us_cpi())

        self._log_data("开始获取美国PPI数据...")
        module.indicators.append(self._get_us_ppi())

        self._log_data("开始获取美国核心PCE数据...")
        module.indicators.append(self._get_core_pce())

        signals = [ind.signal for ind in module.indicators if ind.signal and ind.value != 0]
        red_count = signals.count("🔴")
        yellow_count = signals.count("🟡")
        if red_count >= 2:
            module.overall_signal = "🔴"
        elif red_count >= 1 or yellow_count >= 2:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"

        return module

    def _get_us_cpi(self) -> IndicatorResult:
        indicator = IndicatorResult(name="CPI同比", unit="%")
        cpi = None

        if self.fred:
            try:
                data = self.fred.get_series('CPIAUCSL')  # CPI-U 月度
                if data is not None and len(data) > 12:
                    latest = float(data.iloc[-1])
                    year_ago = float(data.iloc[-13])
                    cpi = (latest / year_ago - 1) * 100
                    indicator.data_date = str(data.index[-1].strftime('%Y-%m'))
                    indicator.data_source = "FRED"
                    self._log_data(f"FRED获取美国CPI成功: {cpi:.1f}%")
            except Exception as e:
                self._log_data(f"FRED获取美国CPI失败: {e}")

        if cpi is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_usa_cpi_monthly()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    cpi = float(latest.get('今值', 0))
                    indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取美国CPI成功: {cpi}%")
            except Exception as e:
                self._log_data(f"AKShare获取美国CPI失败: {e}")

        if cpi is not None:
            indicator.value = round(cpi, 1)
            if cpi > 5.0:
                indicator.status = "高通胀"
                indicator.signal = "🔴"
            elif cpi > 3.0:
                indicator.status = "通胀偏高"
                indicator.signal = "🟡"
            elif cpi >= 1.5:
                indicator.status = "温和通胀"
                indicator.signal = "🟢"
            elif cpi >= 0:
                indicator.status = "低通胀"
                indicator.signal = "🟡"
            else:
                indicator.status = "通缩"
                indicator.signal = "🔴"
            indicator.analysis_detail = (
                f"美国CPI同比为{cpi:.1f}%(来源:{indicator.data_source})。"
                f"美联储目标通胀率2%。判断标准: >5%高通胀, 3-5%偏高, 1.5-3%温和, 0-1.5%低通胀, <0%通缩。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    def _get_us_ppi(self) -> IndicatorResult:
        indicator = IndicatorResult(name="PPI同比", unit="%")
        ppi = None

        if self.fred:
            try:
                data = self.fred.get_series('PPIACO')  # PPI All Commodities
                if data is not None and len(data) > 12:
                    latest = float(data.iloc[-1])
                    year_ago = float(data.iloc[-13])
                    ppi = (latest / year_ago - 1) * 100
                    indicator.data_date = str(data.index[-1].strftime('%Y-%m'))
                    indicator.data_source = "FRED"
                    self._log_data(f"FRED获取美国PPI成功: {ppi:.1f}%")
            except Exception as e:
                self._log_data(f"FRED获取美国PPI失败: {e}")

        if ppi is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_usa_ppi()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    ppi = float(latest.get('今值', 0))
                    indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取美国PPI成功: {ppi}%")
            except Exception as e:
                self._log_data(f"AKShare获取美国PPI失败: {e}")

        if ppi is not None:
            indicator.value = round(ppi, 1)
            if ppi > 5.0:
                indicator.status = "生产成本过热"
                indicator.signal = "🔴"
            elif ppi > 2.0:
                indicator.status = "偏高"
                indicator.signal = "🟡"
            elif ppi >= 0:
                indicator.status = "正常"
                indicator.signal = "🟢"
            else:
                indicator.status = "生产通缩"
                indicator.signal = "🟡"
            indicator.analysis_detail = (
                f"美国PPI同比为{ppi:.1f}%(来源:{indicator.data_source})。"
                f"判断标准: >5%过热, 2-5%偏高, 0-2%正常, <0%生产通缩。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    def _get_core_pce(self) -> IndicatorResult:
        indicator = IndicatorResult(name="核心PCE同比", unit="%")
        pce = None

        if self.fred:
            try:
                data = self.fred.get_series('PCEPILFE')  # 核心PCE价格指数
                if data is not None and len(data) > 12:
                    latest = float(data.iloc[-1])
                    year_ago = float(data.iloc[-13])
                    pce = (latest / year_ago - 1) * 100
                    indicator.data_date = str(data.index[-1].strftime('%Y-%m'))
                    indicator.data_source = "FRED"
                    self._log_data(f"FRED获取核心PCE成功: {pce:.1f}%")
            except Exception as e:
                self._log_data(f"FRED获取核心PCE失败: {e}")

        if pce is not None:
            indicator.value = round(pce, 1)
            if pce > 4.0:
                indicator.status = "核心通胀过高"
                indicator.signal = "🔴"
            elif pce > 2.5:
                indicator.status = "高于目标"
                indicator.signal = "🟡"
            elif pce >= 1.5:
                indicator.status = "接近目标"
                indicator.signal = "🟢"
            else:
                indicator.status = "低于目标"
                indicator.signal = "🟡"
            indicator.analysis_detail = (
                f"核心PCE同比为{pce:.1f}%(来源:{indicator.data_source})。"
                f"美联储首选通胀指标，目标2%。判断标准: >4%过高, 2.5-4%高于目标, 1.5-2.5%接近目标, <1.5%低于目标。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    # ---------- 模块5: 情绪与收益率曲线 (Sentiment & Yield Curve) ----------

    def analyze_sentiment_yield_curve(self) -> ModuleResult:
        module = ModuleResult(module_name="情绪与收益率曲线", module_name_en="Sentiment & Yield Curve")

        # 1. 国债收益率曲线 (10Y-2Y利差)
        self._log_data("开始获取国债收益率曲线数据...")
        yield_indicator = self._get_yield_curve()
        module.indicators.append(yield_indicator)

        # 2. 消费者信心指数
        self._log_data("开始获取消费者信心指数数据...")
        sentiment_indicator = self._get_consumer_sentiment()
        module.indicators.append(sentiment_indicator)

        # 3. VIX 恐慌指数
        self._log_data("开始获取VIX恐慌指数数据...")
        vix_indicator = self._get_vix()
        module.indicators.append(vix_indicator)

        signals = [ind.signal for ind in module.indicators if ind.signal and ind.value != 0]
        if signals.count("🔴") >= 2:
            module.overall_signal = "🔴"
        elif "🔴" in signals or signals.count("🟡") >= 2:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"

        return module

    def _get_yield_curve(self) -> IndicatorResult:
        indicator = IndicatorResult(name="10Y-2Y国债利差", unit="bp")
        spread = None

        if self.fred:
            try:
                t10y = self.fred.get_series('DGS10')  # 10年期国债收益率
                t2y = self.fred.get_series('DGS2')    # 2年期国债收益率
                if t10y is not None and t2y is not None:
                    # 取最近非NaN值
                    t10y_clean = t10y.dropna()
                    t2y_clean = t2y.dropna()
                    if len(t10y_clean) > 0 and len(t2y_clean) > 0:
                        spread_pct = float(t10y_clean.iloc[-1]) - float(t2y_clean.iloc[-1])
                        spread = spread_pct * 100  # 转为bp
                        indicator.data_date = str(t10y_clean.index[-1].strftime('%Y-%m-%d'))
                        indicator.data_source = "FRED"
                        self._log_data(
                            f"FRED获取收益率曲线成功: 10Y={t10y_clean.iloc[-1]:.2f}%, "
                            f"2Y={t2y_clean.iloc[-1]:.2f}%, 利差={spread:.0f}bp"
                        )
            except Exception as e:
                self._log_data(f"FRED获取收益率曲线失败: {e}")

        if spread is None and YFINANCE_AVAILABLE:
            try:
                t10 = yf.Ticker("^TNX")
                t2 = yf.Ticker("^IRX")  # 13周国库券利率作为近似
                h10 = t10.history(period="5d")
                h2 = t2.history(period="5d")
                if not h10.empty and not h2.empty:
                    spread = (float(h10['Close'].iloc[-1]) - float(h2['Close'].iloc[-1])) * 100
                    indicator.data_source = "yfinance"
                    self._log_data(f"yfinance获取收益率曲线: 利差≈{spread:.0f}bp")
            except Exception as e:
                self._log_data(f"yfinance获取收益率曲线失败: {e}")

        if spread is not None:
            indicator.value = round(spread, 0)
            if spread < -50:
                indicator.status = "深度倒挂"
                indicator.signal = "🔴"
                indicator.historical_ref = "强烈衰退预警，历史上倒挂后12-18个月常出现衰退"
            elif spread < 0:
                indicator.status = "倒挂"
                indicator.signal = "🔴"
                indicator.historical_ref = "衰退预警信号"
            elif spread < 50:
                indicator.status = "平坦"
                indicator.signal = "🟡"
                indicator.historical_ref = "经济周期后期"
            else:
                indicator.status = "正常"
                indicator.signal = "🟢"
                indicator.historical_ref = "经济扩张期"
            indicator.analysis_detail = (
                f"10Y-2Y国债利差为{spread:.0f}bp(来源:{indicator.data_source})。"
                f"判断标准: <-50bp深度倒挂(强烈衰退预警), <0倒挂(衰退预警), 0-50bp平坦(周期后期), >50bp正常(扩张期)。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    def _get_consumer_sentiment(self) -> IndicatorResult:
        indicator = IndicatorResult(name="消费者信心指数", unit="")
        sentiment = None

        if self.fred:
            try:
                data = self.fred.get_series('UMCSENT')  # 密歇根大学消费者信心指数
                if data is not None and len(data) > 0:
                    sentiment = float(data.iloc[-1])
                    indicator.data_date = str(data.index[-1].strftime('%Y-%m'))
                    indicator.data_source = "FRED (UMich)"
                    self._log_data(f"FRED获取消费者信心指数成功: {sentiment}")
            except Exception as e:
                self._log_data(f"FRED获取消费者信心指数失败: {e}")

        if sentiment is None and AKSHARE_AVAILABLE:
            try:
                df = ak.macro_usa_michigan_consumer_sentiment()
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    sentiment = float(latest.get('今值', 0))
                    indicator.data_source = "AKShare"
                    self._log_data(f"AKShare获取消费者信心指数成功: {sentiment}")
            except Exception as e:
                self._log_data(f"AKShare获取消费者信心指数失败: {e}")

        if sentiment is not None:
            indicator.value = round(sentiment, 1)
            # 历史均值约85, 2020年低点约71, 2022年低点约50
            if sentiment > 90:
                indicator.status = "乐观"
                indicator.signal = "🟢"
                indicator.historical_ref = "高于历史均值(~85)"
            elif sentiment > 70:
                indicator.status = "中性"
                indicator.signal = "🟢"
                indicator.historical_ref = "接近历史均值"
            elif sentiment > 55:
                indicator.status = "悲观"
                indicator.signal = "🟡"
                indicator.historical_ref = "低于历史均值"
            else:
                indicator.status = "极度悲观"
                indicator.signal = "🔴"
                indicator.historical_ref = "历史低位区域"
            indicator.analysis_detail = (
                f"密歇根大学消费者信心指数为{sentiment:.1f}(来源:{indicator.data_source})。"
                f"历史均值约85。判断标准: >90乐观, 70-90中性, 55-70悲观, <55极度悲观。"
                f"当前判断: {indicator.status}。"
            )
        return indicator

    def _get_vix(self) -> IndicatorResult:
        indicator = IndicatorResult(name="VIX恐慌指数", unit="")
        vix = None

        if YFINANCE_AVAILABLE:
            try:
                vix_ticker = yf.Ticker("^VIX")
                hist = vix_ticker.history(period="5d")
                if hist is not None and not hist.empty:
                    vix = float(hist['Close'].iloc[-1])
                    indicator.data_date = str(hist.index[-1].strftime('%Y-%m-%d'))
                    indicator.data_source = "yfinance"
                    self._log_data(f"yfinance获取VIX成功: {vix:.1f}")
            except Exception as e:
                self._log_data(f"yfinance获取VIX失败: {e}")

        if vix is not None:
            indicator.value = round(vix, 1)
            if vix > 30:
                indicator.status = "恐慌"
                indicator.signal = "🔴"
                indicator.historical_ref = "市场极度恐慌，可能是逆向买入机会"
            elif vix > 20:
                indicator.status = "偏高"
                indicator.signal = "🟡"
                indicator.historical_ref = "市场不确定性增加"
            elif vix > 12:
                indicator.status = "正常"
                indicator.signal = "🟢"
                indicator.historical_ref = "市场情绪稳定"
            else:
                indicator.status = "极度平静"
                indicator.signal = "🟡"
                indicator.historical_ref = "可能过度自满，警惕黑天鹅"
            indicator.analysis_detail = (
                f"VIX恐慌指数为{vix:.1f}(来源:{indicator.data_source})。"
                f"判断标准: >30恐慌, 20-30偏高, 12-20正常, <12极度平静(可能过度自满)。"
                f"当前判断: {indicator.status}。"
            )
        return indicator


# ==================== 工厂函数 & 兼容层 ====================

def detect_market(tickers: Optional[List[str]] = None, market: Optional[str] = None) -> str:
    """根据标的代码或显式指定自动检测市场

    规则:
    - 显式指定 market 参数优先
    - 股票代码含 .SZ / .SH / .BJ → CN
    - 纯字母代码(如 AAPL, MSFT) → US
    - 含 .HK → HK
    - 默认 CN
    """
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
    """工厂函数: 根据市场创建对应的宏观风控终端实例

    Args:
        market: 市场标识 ("CN" / "US" / ...)
        **kwargs: 传递给具体终端的参数 (如 tushare_token, fred_api_key, cache_dir)

    Returns:
        MacroRiskTerminalBase 的具体子类实例
    """
    market = market.upper()
    if market == "CN":
        return CNMacroRiskTerminal(
            tushare_token=kwargs.get('tushare_token'),
            cache_dir=kwargs.get('cache_dir', '/tmp/macro_risk_cache')
        )
    elif market == "US":
        return USMacroRiskTerminal(
            fred_api_key=kwargs.get('fred_api_key'),
            cache_dir=kwargs.get('cache_dir', '/tmp/macro_risk_cache')
        )
    else:
        raise ValueError(
            f"暂不支持市场 '{market}'。当前支持: CN (A股), US (美股)。"
            f"可通过继承 MacroRiskTerminalBase 扩展新市场。"
        )


# 向后兼容: 保留 MacroRiskTerminal 作为 CNMacroRiskTerminal 的别名
MacroRiskTerminal = CNMacroRiskTerminal


# ==================== 测试 ====================

if __name__ == '__main__':
    import sys

    # 支持命令行参数: python macro_risk_terminal.py [CN|US]
    market = sys.argv[1].upper() if len(sys.argv) > 1 else "CN"

    print(f"正在运行 {market} 市场宏观风控终端...")

    if market == "CN":
        token = os.environ.get('TUSHARE_TOKEN')
        terminal = create_terminal("CN", tushare_token=token)
    elif market == "US":
        fred_key = os.environ.get('FRED_API_KEY')
        terminal = create_terminal("US", fred_api_key=fred_key)
    else:
        print(f"不支持的市场: {market}")
        sys.exit(1)

    report = terminal.generate_risk_report()
    md = terminal.format_report_markdown(report)
    print(md)

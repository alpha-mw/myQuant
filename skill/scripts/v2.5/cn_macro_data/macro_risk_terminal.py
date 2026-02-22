"""
MacroRiskTerminal - A股宏观风控终端
对标"A股宏观风控终端 V2.3"，提供四大模块的宏观风控指标分析：

1. 资金杠杆与情绪 (Leverage): 两融余额、两融/流通市值比
2. 经济景气度 (Growth): GDP同比增速
3. 整体估值锚 (Valuation): A股总市值、年度GDP预估、巴菲特指标
4. 通胀与货币 (Inflation & Money): CPI、PPI、M1-M2剪刀差、M2增速、社融当月增量

每个指标包含：数据获取、状态判断、历史对标
"""

import os
import json
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


# ==================== 数据结构 ====================

@dataclass
class IndicatorResult:
    """单个指标的分析结果"""
    name: str                    # 指标名称
    value: float = 0.0           # 当前值
    unit: str = ""               # 单位
    status: str = ""             # 状态判断（如"极度疯狂"、"结构健康"等）
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
    analysis_log: List[str] = field(default_factory=list)  # 分析过程日志


@dataclass
class RiskTerminalReport:
    """宏观风控终端完整报告"""
    timestamp: str = ""
    version: str = "V2.3"
    modules: List[ModuleResult] = field(default_factory=list)
    overall_signal: str = "🟡"
    overall_risk_level: str = ""
    recommendation: str = ""
    data_acquisition_log: List[str] = field(default_factory=list)  # 数据获取日志
    analysis_process_log: List[str] = field(default_factory=list)  # 分析过程日志


# ==================== 宏观风控终端 ====================

class MacroRiskTerminal:
    """A股宏观风控终端"""

    # 历史大顶参考数据
    HISTORICAL_REFS = {
        'margin_2015_peak': {
            'margin_balance': 2.27,        # 万亿
            'margin_ratio': 4.5,           # %
            'note': '2015年疯牛顶'
        },
        'buffett_2007_peak': {
            'ratio': 125.0,               # %
            'note': '2007年疯牛顶'
        },
        'buffett_2015_peak': {
            'ratio': 110.0,               # %
            'note': '2015年疯牛顶'
        },
        'buffett_bottom_range': {
            'low': 40.0,                  # %
            'high': 60.0,                 # %
            'note': '底部安全区间'
        }
    }

    def __init__(self, tushare_token: Optional[str] = None,
                 cache_dir: str = '/tmp/macro_risk_cache'):
        self.token = tushare_token or os.environ.get('TUSHARE_TOKEN')
        self.pro = None
        if self.token and TUSHARE_AVAILABLE:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            # 设置自定义URL（如果有）
            custom_url = os.environ.get('TUSHARE_HTTP_URL',
                                        'http://lianghua.nanyangqiankun.top')
            if custom_url:
                self.pro._DataApi__http_url = custom_url

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_log = []  # 记录所有数据获取操作
        self.analysis_log = []  # 记录所有分析步骤

    def _log_data(self, msg: str):
        """记录数据获取日志"""
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.data_log.append(entry)

    def _log_analysis(self, msg: str):
        """记录分析过程日志"""
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.analysis_log.append(entry)

    # ==================== 模块1: 资金杠杆与情绪 ====================

    def analyze_leverage(self) -> ModuleResult:
        """分析资金杠杆与情绪模块"""
        module = ModuleResult(
            module_name="资金杠杆与情绪",
            module_name_en="Leverage"
        )

        # 1. 获取两融余额数据
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
                    # 汇总全市场两融余额
                    latest_date = df['trade_date'].max()
                    daily = df[df['trade_date'] == latest_date]
                    margin_balance = daily['rzye'].sum() / 1e8  # 转为亿元
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

        # 2. 获取A股流通市值
        float_mv = None
        self._log_data("开始获取A股流通市值数据...")

        if self.pro:
            try:
                today = datetime.now().strftime('%Y%m%d')
                df = self.pro.daily_basic(trade_date=today, fields='ts_code,float_share,close,circ_mv')
                if df is None or df.empty:
                    # 尝试前一个交易日
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                    df = self.pro.daily_basic(trade_date=yesterday, fields='ts_code,circ_mv')
                if df is not None and not df.empty:
                    float_mv = df['circ_mv'].sum() / 1e4  # 万元转亿元
                    self._log_data(f"Tushare获取流通市值成功: {float_mv:.0f}亿元")
            except Exception as e:
                self._log_data(f"Tushare获取流通市值失败: {e}")

        # 3. 计算指标并判断状态
        margin_balance_tn = margin_balance / 1e4 if margin_balance else 0  # 亿转万亿

        # 两融余额指标
        margin_indicator = IndicatorResult(
            name="两融余额",
            value=round(margin_balance_tn, 2) if margin_balance else 0,
            unit="万亿",
            data_date=margin_date,
            data_source=data_source
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

        # 两融/流通市值比指标
        ratio_indicator = IndicatorResult(
            name="两融/流通市值比",
            unit="%",
            data_date=margin_date,
            data_source=data_source
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

            ratio_indicator.historical_ref = (
                f"2015牛市顶占比4.5%, 当前{ratio:.2f}%"
            )
            ratio_indicator.analysis_detail = (
                f"计算公式: 两融余额({margin_balance:.0f}亿) / 流通市值({float_mv:.0f}亿) × 100%。"
                f"结果为{ratio:.2f}%。"
                f"判断标准: >4%极度疯狂, 3-4%偏热, 2-3%结构健康, 1.5-2%偏冷, <1.5%极度冷清。"
            )
        module.indicators.append(ratio_indicator)

        # 模块综合信号
        signals = [ind.signal for ind in module.indicators if ind.signal]
        if "🔴" in signals:
            module.overall_signal = "🔴"
        elif "🟡" in signals:
            module.overall_signal = "🟡"
        else:
            module.overall_signal = "🟢"

        return module

    # ==================== 模块2: 经济景气度 ====================

    def analyze_growth(self) -> ModuleResult:
        """分析经济景气度模块"""
        module = ModuleResult(
            module_name="经济景气度",
            module_name_en="Growth"
        )

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
            unit="%",
            data_date=gdp_quarter,
            data_source=data_source
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

    # ==================== 模块3: 整体估值锚 ====================

    def analyze_valuation(self) -> ModuleResult:
        """分析整体估值锚模块"""
        module = ModuleResult(
            module_name="整体估值锚",
            module_name_en="Valuation"
        )

        total_mv = None
        gdp_estimate = None
        data_source_mv = ""

        # 1. 获取A股总市值
        self._log_data("开始获取A股总市值数据...")

        if self.pro:
            try:
                today = datetime.now().strftime('%Y%m%d')
                df = self.pro.daily_basic(trade_date=today, fields='ts_code,total_mv')
                if df is None or df.empty:
                    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                    df = self.pro.daily_basic(trade_date=yesterday, fields='ts_code,total_mv')
                if df is not None and not df.empty:
                    total_mv = df['total_mv'].sum() / 1e4  # 万元转亿元
                    data_source_mv = "Tushare"
                    self._log_data(f"Tushare获取A股总市值成功: {total_mv:.0f}亿元 ({total_mv/1e4:.2f}万亿元)")
            except Exception as e:
                self._log_data(f"Tushare获取总市值失败: {e}")

        # 2. 获取/估算年度GDP
        self._log_data("开始获取/估算年度GDP数据...")

        if self.pro:
            try:
                df = self.pro.cn_gdp()
                if df is not None and not df.empty:
                    # 取最近4个季度的GDP累计
                    latest = df.iloc[0]
                    gdp_val = float(latest.get('gdp', 0))
                    quarter = str(latest.get('quarter', ''))
                    # 如果是全年数据直接用，否则按季度推算
                    if 'Q4' in quarter or '四' in quarter:
                        gdp_estimate = gdp_val / 1e4  # 亿转万亿
                    else:
                        # 简单年化推算
                        q_num = 4  # 默认
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

        # A股总市值指标
        mv_tn = total_mv / 1e4 if total_mv else 0  # 亿转万亿
        mv_indicator = IndicatorResult(
            name="A股总市值",
            value=round(mv_tn, 2),
            unit="万亿",
            data_source=data_source_mv,
            data_date=datetime.now().strftime('%Y-%m-%d')
        )
        mv_indicator.analysis_detail = f"A股全市场总市值为{mv_tn:.2f}万亿元(来源:{data_source_mv})。"
        module.indicators.append(mv_indicator)

        # 年度GDP预估指标
        gdp_indicator = IndicatorResult(
            name="年度GDP（预）",
            value=round(gdp_estimate, 2) if gdp_estimate else 0,
            unit="万亿",
            data_source="Tushare/推算"
        )
        gdp_indicator.analysis_detail = f"年度GDP预估值为{gdp_estimate:.2f}万亿元。" if gdp_estimate else "GDP数据获取失败。"
        module.indicators.append(gdp_indicator)

        # 巴菲特指标
        buffett_indicator = IndicatorResult(
            name="市值/GDP（巴菲特）",
            unit="%"
        )

        if total_mv and gdp_estimate and gdp_estimate > 0:
            buffett_ratio = mv_tn / gdp_estimate * 100
            buffett_indicator.value = round(buffett_ratio, 1)

            self._log_analysis(
                f"巴菲特指标 = A股总市值({mv_tn:.2f}万亿) / 年度GDP({gdp_estimate:.2f}万亿) × 100% = {buffett_ratio:.1f}%"
            )

            # 历史对标
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

        # 模块综合信号
        signals = [ind.signal for ind in module.indicators if ind.signal and ind.name == "市值/GDP（巴菲特）"]
        module.overall_signal = signals[0] if signals else "🟡"
        return module

    # ==================== 模块4: 通胀与货币 ====================

    def analyze_inflation_money(self) -> ModuleResult:
        """分析通胀与货币模块"""
        module = ModuleResult(
            module_name="通胀与货币",
            module_name_en="Inflation & Money"
        )

        # 1. CPI同比
        self._log_data("开始获取CPI同比数据...")
        cpi_indicator = self._get_cpi_indicator()
        module.indicators.append(cpi_indicator)

        # 2. PPI同比
        self._log_data("开始获取PPI同比数据...")
        ppi_indicator = self._get_ppi_indicator()
        module.indicators.append(ppi_indicator)

        # 3. M1-M2剪刀差 & M2增速
        self._log_data("开始获取货币供应(M1/M2)数据...")
        m1m2_indicator, m2_indicator = self._get_money_indicators()
        module.indicators.append(m1m2_indicator)
        module.indicators.append(m2_indicator)

        # 4. 社融当月增量
        self._log_data("开始获取社融当月增量数据...")
        sf_indicator = self._get_social_financing_indicator()
        module.indicators.append(sf_indicator)

        # 模块综合信号
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
        """获取CPI同比指标"""
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
        """获取PPI同比指标"""
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
        """获取M1-M2剪刀差和M2增速指标"""
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

        # M1-M2剪刀差
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

        # M2增速
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
        """获取社融当月增量指标"""
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
            # 社融需要与历史同期对比，这里给出基本判断
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

    # ==================== 综合报告 ====================

    def generate_risk_report(self) -> RiskTerminalReport:
        """生成完整的宏观风控终端报告"""
        self.data_log = []
        self.analysis_log = []

        report = RiskTerminalReport(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            version="V2.3"
        )

        self._log_data("=" * 60)
        self._log_data("A股宏观风控终端 开始运行")
        self._log_data("=" * 60)

        # 运行四大模块
        self._log_analysis("开始执行四大模块分析...")

        module_leverage = self.analyze_leverage()
        report.modules.append(module_leverage)
        self._log_analysis(f"模块1[资金杠杆与情绪] 完成, 信号: {module_leverage.overall_signal}")

        module_growth = self.analyze_growth()
        report.modules.append(module_growth)
        self._log_analysis(f"模块2[经济景气度] 完成, 信号: {module_growth.overall_signal}")

        module_valuation = self.analyze_valuation()
        report.modules.append(module_valuation)
        self._log_analysis(f"模块3[整体估值锚] 完成, 信号: {module_valuation.overall_signal}")

        module_inflation = self.analyze_inflation_money()
        report.modules.append(module_inflation)
        self._log_analysis(f"模块4[通胀与货币] 完成, 信号: {module_inflation.overall_signal}")

        # 综合风控信号
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
        """将报告格式化为Markdown"""
        lines = []
        lines.append(f"## A股宏观风控终端 ({report.version}) | {report.timestamp}")
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

            # 分析过程详情
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


# ==================== 测试 ====================

if __name__ == '__main__':
    token = os.environ.get('TUSHARE_TOKEN')
    terminal = MacroRiskTerminal(tushare_token=token)
    report = terminal.generate_risk_report()
    md = terminal.format_report_markdown(report)
    print(md)

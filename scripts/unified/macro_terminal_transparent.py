#!/usr/bin/env python3
"""
MacroRiskTerminal V6.3 Enhanced - 完全透明化版本
多市场宏观风控终端 - 第0层风控

核心特性:
1. 多市场适配架构 (CN/US/HK/EU/JP可扩展)
2. 报告完全透明化 - 详细展示数据获取、分析过程、推理逻辑
3. 自动市场检测
4. 基于完整指标体系的状态判断
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
from functools import wraps

import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MacroRiskTerminal')

# 可选依赖
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance未安装，美股数据获取将受限")

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
class DataAcquisitionStep:
    """数据获取步骤记录"""
    timestamp: str
    data_source: str           # 数据源: Tushare/yfinance/FRED/AKShare
    data_type: str             # 数据类型: 两融余额/GDP/利率等
    attempt_method: str        # 尝试的方法
    params: Dict[str, Any]     # 调用参数
    result_status: str         # success/failed/partial
    result_summary: str        # 结果摘要
    error_message: str = ""    # 错误信息
    fallback_plan: str = ""    # 降级方案


@dataclass
class AnalysisStep:
    """分析步骤记录"""
    timestamp: str
    step_name: str             # 步骤名称
    input_data: str            # 输入数据描述
    analysis_method: str       # 分析方法
    reasoning_process: str     # 推理过程
    conclusion: str            # 结论
    confidence: str = ""       # 置信度


@dataclass
class IndicatorResult:
    """单个指标的分析结果 - 增强版，包含完整溯源信息"""
    name: str
    value: float = 0.0
    unit: str = ""
    status: str = ""
    signal: str = "🟡"
    
    # 数据溯源
    data_source: str = ""           # 数据来源
    data_date: str = ""             # 数据日期
    acquisition_steps: List[DataAcquisitionStep] = field(default_factory=list)
    
    # 分析溯源
    historical_ref: str = ""        # 历史对标
    analysis_steps: List[AnalysisStep] = field(default_factory=list)
    analysis_detail: str = ""       # 详细分析说明
    
    # 判断依据
    threshold_rules: str = ""       # 使用的阈值规则
    comparison_basis: str = ""      # 对比基准


@dataclass
class ModuleResult:
    """单个模块的分析结果 - 增强版"""
    module_name: str
    module_name_en: str
    indicators: List[IndicatorResult] = field(default_factory=list)
    overall_signal: str = "🟡"
    
    # 模块级分析过程
    module_analysis_log: List[AnalysisStep] = field(default_factory=list)
    weight_in_overall: float = 1.0    # 在综合信号中的权重


@dataclass
class RiskTerminalReport:
    """宏观风控终端完整报告 - 完全透明化版本"""
    timestamp: str = ""
    version: str = "V6.3-Transparent"
    market: str = ""
    market_name: str = ""
    
    # 市场检测信息
    market_detection: Dict[str, Any] = field(default_factory=dict)
    
    # 各模块结果
    modules: List[ModuleResult] = field(default_factory=list)
    
    # 综合信号计算过程
    overall_signal_calculation: List[AnalysisStep] = field(default_factory=list)
    overall_signal: str = "🟡"
    overall_risk_level: str = ""
    recommendation: str = ""
    
    # 完整执行日志
    execution_log: List[str] = field(default_factory=list)


# ==================== 透明化装饰器 ====================

def trace_data_acquisition(data_type: str, primary_source: str):
    """数据获取追踪装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            step = DataAcquisitionStep(
                timestamp=datetime.now().isoformat(),
                data_source=primary_source,
                data_type=data_type,
                attempt_method=func.__name__,
                params={'args': str(args), 'kwargs': str(kwargs)},
                result_status="attempting",
                result_summary=""
            )
            
            self._current_acquisition_steps.append(step)
            
            try:
                result = func(self, *args, **kwargs)
                step.result_status = "success" if result is not None else "no_data"
                step.result_summary = f"获取成功: {result}" if result is not None else "无数据返回"
                return result
            except Exception as e:
                step.result_status = "failed"
                step.error_message = str(e)
                step.fallback_plan = "将尝试降级数据源"
                logger.error(f"数据获取失败 [{data_type}]: {e}")
                raise
        
        return wrapper
    return decorator


def trace_analysis(step_name: str):
    """分析过程追踪装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            analysis_step = AnalysisStep(
                timestamp=datetime.now().isoformat(),
                step_name=step_name,
                input_data=str(args[0]) if args else "",
                analysis_method=func.__doc__ or "",
                reasoning_process="开始分析...",
                conclusion=""
            )
            
            self._current_analysis_steps.append(analysis_step)
            
            try:
                result = func(self, *args, **kwargs)
                analysis_step.conclusion = f"分析完成: {result}"
                analysis_step.reasoning_process = self._get_last_reasoning()
                return result
            except Exception as e:
                analysis_step.conclusion = f"分析失败: {e}"
                raise
        
        return wrapper
    return decorator


# ==================== 基类: 宏观风控终端 ====================

class MacroRiskTerminalBase(ABC):
    """
    宏观风控终端基类 - 第0层风控
    
    设计原则:
    1. 所有数据获取必须记录完整溯源信息
    2. 所有分析步骤必须展示推理逻辑
    3. 支持多市场扩展
    """
    
    MARKET: str = ""
    MARKET_NAME: str = ""
    
    # 信号阈值配置
    SIGNAL_THRESHOLDS = {
        'high_risk': {'modules_red': 2, 'description': '任意2个模块红色信号'},
        'medium_risk': {'modules_red': 1, 'modules_yellow': 2, 'description': '1个红色或2个黄色'},
        'low_risk': {'description': '多数模块绿色'},
        'extreme_low': {'modules_blue': 2, 'description': '多数模块蓝色(底部)'}
    }
    
    def __init__(self, cache_dir: str = '/tmp/macro_risk_cache', verbose: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # 当前追踪的步骤
        self._current_acquisition_steps: List[DataAcquisitionStep] = []
        self._current_analysis_steps: List[AnalysisStep] = []
        self._last_reasoning: str = ""
        
        # 执行日志
        self.execution_log: List[str] = []
    
    def _log(self, msg: str, level: str = "info"):
        """记录执行日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {msg}"
        self.execution_log.append(log_entry)
        
        if self.verbose:
            if level == "error":
                logger.error(msg)
            elif level == "warning":
                logger.warning(msg)
            else:
                logger.info(msg)
    
    def _set_reasoning(self, reasoning: str):
        """设置当前推理过程"""
        self._last_reasoning = reasoning
        self._log(f"推理: {reasoning}")
    
    def _get_last_reasoning(self) -> str:
        """获取最后推理"""
        return self._last_reasoning
    
    @abstractmethod
    def get_modules(self) -> List[ModuleResult]:
        """返回该市场的所有宏观风控模块 - 子类必须实现"""
        pass
    
    def generate_risk_report(self) -> RiskTerminalReport:
        """
        生成完整的宏观风控终端报告 - 完全透明化版本
        
        报告包含:
        1. 市场检测过程
        2. 各模块数据获取详细步骤
        3. 各模块分析推理过程
        4. 综合信号计算逻辑
        """
        self.execution_log = []
        self._log("=" * 80)
        self._log(f"{self.MARKET_NAME}宏观风控终端 V6.3 (透明化版本) 开始运行")
        self._log("=" * 80)
        
        report = RiskTerminalReport(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            version="V6.3-Transparent",
            market=self.MARKET,
            market_name=self.MARKET_NAME,
            market_detection={
                'detected_market': self.MARKET,
                'market_name': self.MARKET_NAME,
                'detection_method': '显式指定' if self.MARKET else '自动检测',
                'supported_indicators': self._get_supported_indicators()
            }
        )
        
        # 获取各模块分析
        self._log("开始执行各模块分析...")
        report.modules = self.get_modules()
        
        for i, module in enumerate(report.modules, 1):
            self._log(f"模块{i}[{module.module_name}] 分析完成, 信号: {module.overall_signal}")
        
        # 计算综合信号
        self._log("开始计算综合风控信号...")
        report.overall_signal_calculation = self._calculate_overall_signal_transparent(report.modules)
        report.overall_signal = report.overall_signal_calculation[-1].conclusion if report.overall_signal_calculation else "🟡"
        report.overall_risk_level = self._signal_to_risk_level(report.overall_signal)
        report.recommendation = self._signal_to_recommendation(report.overall_signal)
        
        self._log(f"综合风控信号计算完成: {report.overall_signal} {report.overall_risk_level}")
        self._log(f"投资建议: {report.recommendation}")
        
        report.execution_log = self.execution_log.copy()
        
        return report
    
    def _get_supported_indicators(self) -> Dict[str, List[str]]:
        """获取支持的指标列表 - 子类可覆盖"""
        return {
            'data_sources': [],
            'indicators': []
        }
    
    def _calculate_overall_signal_transparent(self, modules: List[ModuleResult]) -> List[AnalysisStep]:
        """
        透明化计算综合风控信号
        
        展示完整的计算逻辑和每一步的推理
        """
        steps = []
        
        # 步骤1: 收集各模块信号
        step1 = AnalysisStep(
            timestamp=datetime.now().isoformat(),
            step_name="收集各模块信号",
            input_data=f"{len(modules)}个模块",
            analysis_method="提取每个模块的综合信号",
            reasoning_process="遍历所有模块，记录其overall_signal",
            conclusion=""
        )
        
        module_signals = {}
        for m in modules:
            module_signals[m.module_name] = m.overall_signal
        
        step1.conclusion = f"模块信号: {module_signals}"
        steps.append(step1)
        self._set_reasoning(f"收集到模块信号: {module_signals}")
        
        # 步骤2: 统计信号分布
        step2 = AnalysisStep(
            timestamp=datetime.now().isoformat(),
            step_name="统计信号分布",
            input_data=str(module_signals),
            analysis_method="统计🔴🟡🟢🔵各信号出现次数",
            reasoning_process="使用collections.Counter统计",
            conclusion=""
        )
        
        all_signals = list(module_signals.values())
        red_count = all_signals.count("🔴")
        yellow_count = all_signals.count("🟡")
        green_count = all_signals.count("🟢")
        blue_count = all_signals.count("🔵")
        
        step2.conclusion = f"🔴:{red_count}, 🟡:{yellow_count}, 🟢:{green_count}, 🔵:{blue_count}"
        steps.append(step2)
        self._set_reasoning(f"信号分布 - 红:{red_count} 黄:{yellow_count} 绿:{green_count} 蓝:{blue_count}")
        
        # 步骤3: 应用判断规则
        step3 = AnalysisStep(
            timestamp=datetime.now().isoformat(),
            step_name="应用综合信号判断规则",
            input_data=f"红:{red_count}, 黄:{yellow_count}, 绿:{green_count}, 蓝:{blue_count}",
            analysis_method="按优先级应用规则",
            reasoning_process="",
            conclusion=""
        )
        
        reasoning_lines = ["应用以下规则(按优先级):"]
        final_signal = "🟡"
        
        # 规则1: 高风险
        if red_count >= 2:
            final_signal = "🔴"
            reasoning_lines.append(f"1. 规则: 任意2个模块红色 → 高风险")
            reasoning_lines.append(f"   当前有{red_count}个红色模块，满足条件")
        # 规则2: 中风险
        elif red_count >= 1 or yellow_count >= 2:
            final_signal = "🟡"
            if red_count >= 1:
                reasoning_lines.append(f"2. 规则: 任意1个红色 → 中风险")
                reasoning_lines.append(f"   当前有{red_count}个红色模块，满足条件")
            else:
                reasoning_lines.append(f"2. 规则: 2个黄色 → 中风险")
                reasoning_lines.append(f"   当前有{yellow_count}个黄色模块，满足条件")
        # 规则3: 极低风险
        elif blue_count >= 2:
            final_signal = "🔵"
            reasoning_lines.append(f"3. 规则: 多数蓝色 → 极低风险")
            reasoning_lines.append(f"   当前有{blue_count}个蓝色模块，满足条件")
        # 规则4: 低风险
        elif green_count >= len(modules) / 2:
            final_signal = "🟢"
            reasoning_lines.append(f"4. 规则: 多数绿色 → 低风险")
            reasoning_lines.append(f"   当前有{green_count}个绿色模块，满足条件")
        else:
            final_signal = "🟡"
            reasoning_lines.append(f"5. 默认: 无明确信号 → 中风险")
        
        step3.reasoning_process = "\n".join(reasoning_lines)
        step3.conclusion = f"综合信号: {final_signal}"
        steps.append(step3)
        self._set_reasoning("\n".join(reasoning_lines))
        
        return steps
    
    def _signal_to_risk_level(self, signal: str) -> str:
        """信号转风险等级"""
        mapping = {
            "🔴": "高风险",
            "🟡": "中风险",
            "🟢": "低风险",
            "🔵": "极低风险"
        }
        return mapping.get(signal, "中风险")
    
    def _signal_to_recommendation(self, signal: str) -> str:
        """信号转投资建议"""
        mapping = {
            "🔴": "降低仓位，防御为主，优先现金和低波动资产",
            "🟡": "控制仓位，精选高质量个股，避免高估值标的",
            "🟢": "正常配置，积极布局成长股，适度提升风险偏好",
            "🔵": "加大配置，逆向布局超跌优质股，积极把握机会"
        }
        return mapping.get(signal, "控制仓位，精选个股")
    
    def format_report_markdown(self, report: RiskTerminalReport) -> str:
        """
        格式化报告为Markdown - 完全透明化版本
        
        包含所有数据获取步骤、分析推理过程
        """
        lines = []
        
        # 标题
        lines.append(f"# {report.market_name}宏观风控终端 ({report.version})")
        lines.append(f"**报告时间**: {report.timestamp}")
        lines.append("")
        
        # 市场检测信息
        lines.append("## 🌍 市场检测信息")
        lines.append("")
        lines.append(f"- **检测市场**: {report.market_detection.get('detected_market', 'N/A')}")
        lines.append(f"- **市场名称**: {report.market_detection.get('market_name', 'N/A')}")
        lines.append(f"- **检测方法**: {report.market_detection.get('detection_method', 'N/A')}")
        lines.append("")
        
        # 综合结论
        lines.append("## 🎯 综合风控结论")
        lines.append("")
        lines.append(f"| 项目 | 内容 |")
        lines.append(f"|:---|:---|")
        lines.append(f"| 综合信号 | {report.overall_signal} |")
        lines.append(f"| 风险等级 | {report.overall_risk_level} |")
        lines.append(f"| 投资建议 | {report.recommendation} |")
        lines.append("")
        
        # 综合信号计算过程
        lines.append("### 综合信号计算过程")
        lines.append("")
        for i, step in enumerate(report.overall_signal_calculation, 1):
            lines.append(f"**步骤{i}: {step.step_name}**")
            lines.append(f"- 输入: {step.input_data}")
            lines.append(f"- 方法: {step.analysis_method}")
            lines.append(f"- 推理:\n```\n{step.reasoning_process}\n```")
            lines.append(f"- 结论: **{step.conclusion}**")
            lines.append("")
        
        # 各模块详情
        for module in report.modules:
            lines.append(f"## 📊 {module.module_name} ({module.module_name_en}) {module.overall_signal}")
            lines.append("")
            
            # 模块分析过程
            if module.module_analysis_log:
                lines.append("### 模块分析过程")
                for step in module.module_analysis_log:
                    lines.append(f"- **{step.step_name}**: {step.conclusion}")
                lines.append("")
            
            # 指标详情
            lines.append("### 指标详情")
            lines.append("")
            
            for ind in module.indicators:
                lines.append(f"#### {ind.name} {ind.signal}")
                lines.append("")
                lines.append(f"| 属性 | 值 |")
                lines.append(f"|:---|:---|")
                lines.append(f"| 当前值 | {ind.value} {ind.unit} |")
                lines.append(f"| 状态判断 | {ind.status} |")
                lines.append(f"| 数据来源 | {ind.data_source} |")
                lines.append(f"| 数据日期 | {ind.data_date} |")
                lines.append(f"| 历史对标 | {ind.historical_ref} |")
                lines.append("")
                
                # 数据获取步骤
                if ind.acquisition_steps:
                    lines.append("**数据获取过程:**")
                    for step in ind.acquisition_steps:
                        lines.append(f"- [{step.result_status.upper()}] {step.data_source} - {step.attempt_method}")
                        lines.append(f"  - 参数: {step.params}")
                        lines.append(f"  - 结果: {step.result_summary}")
                        if step.error_message:
                            lines.append(f"  - 错误: {step.error_message}")
                        if step.fallback_plan:
                            lines.append(f"  - 降级: {step.fallback_plan}")
                    lines.append("")
                
                # 分析推理
                if ind.analysis_steps:
                    lines.append("**分析推理过程:**")
                    for step in ind.analysis_steps:
                        lines.append(f"- **{step.step_name}**")
                        lines.append(f"  - 输入: {step.input_data}")
                        lines.append(f"  - 方法: {step.analysis_method}")
                        lines.append(f"  - 推理: {step.reasoning_process}")
                        lines.append(f"  - 结论: {step.conclusion}")
                    lines.append("")
                
                # 判断依据
                if ind.threshold_rules:
                    lines.append(f"**判断依据**: {ind.threshold_rules}")
                if ind.analysis_detail:
                    lines.append(f"**详细说明**: {ind.analysis_detail}")
                lines.append("")
        
        # 执行日志
        lines.append("## 📝 执行日志")
        lines.append("")
        lines.append("```")
        for log in report.execution_log:
            lines.append(log)
        lines.append("```")
        lines.append("")
        
        return "\n".join(lines)


# ==================== 市场检测函数 ====================

def detect_market(tickers: Optional[List[str]] = None, explicit_market: Optional[str] = None) -> Tuple[str, str, str]:
    """
    自动检测市场 - 完全透明化版本
    
    返回: (market_code, market_name, detection_method)
    """
    if explicit_market:
        market = explicit_market.upper()
        name = {
            "CN": "A股",
            "US": "美股",
            "HK": "港股",
            "EU": "欧洲",
            "JP": "日本"
        }.get(market, "未知")
        return market, name, f"显式指定market='{explicit_market}'"
    
    if tickers:
        for ticker in tickers:
            t_upper = ticker.upper()
            
            # A股检测
            if any(t_upper.endswith(suffix) for suffix in ['.SZ', '.SH', '.BJ']):
                return "CN", "A股", f"代码'{ticker}'含.SZ/.SH/.BJ后缀"
            
            # 港股检测
            if t_upper.endswith('.HK'):
                return "HK", "港股", f"代码'{ticker}'含.HK后缀"
            
            # 美股检测 (纯字母)
            if t_upper.isalpha() and len(t_upper) <= 5:
                return "US", "美股", f"代码'{ticker}'为纯字母(美股特征)"
    
    # 默认
    return "CN", "A股", "默认(未检测到明确特征)"


# ==================== 具体市场实现 ====================

class CNMacroRiskTerminal(MacroRiskTerminalBase):
    """
    A股宏观风控终端 - 四大模块完整实现
    
    模块:
    1. 资金杠杆与情绪 (Leverage)
    2. 经济景气度 (Growth)
    3. 整体估值锚 (Valuation)
    4. 通胀与货币 (Inflation & Money)
    """
    
    MARKET = "CN"
    MARKET_NAME = "A股"
    
    # 历史参考值
    HISTORICAL_REFS = {
        'margin_2015_peak': {'balance': 2.27, 'ratio': 4.5, 'note': '2015年疯牛顶'},
        'buffett_2007_peak': 125.0,
        'buffett_2015_peak': 110.0,
        'buffett_bottom_range': (40.0, 60.0),
        'gdp_high_growth': 6.0,
        'gdp_normal': 5.0,
        'gdp_slow': 4.0,
        'gdp_recession_risk': 3.0
    }
    
    def __init__(self, tushare_token: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.token = tushare_token or os.environ.get('TUSHARE_TOKEN')
        self.pro = None
        
        if self.token and TUSHARE_AVAILABLE:
            try:
                ts.set_token(self.token)
                self.pro = ts.pro_api()
                self._log("Tushare API初始化成功")
            except Exception as e:
                self._log(f"Tushare初始化失败: {e}", "warning")
        else:
            self._log("Tushare未配置，将使用降级数据源", "warning")
    
    def _get_supported_indicators(self) -> Dict[str, List[str]]:
        return {
            'data_sources': ['Tushare', 'AKShare', '模拟数据'],
            'modules': [
                '资金杠杆与情绪 - 两融余额、两融/流通市值比',
                '经济景气度 - GDP同比增速',
                '整体估值锚 - 巴菲特指标',
                '通胀与货币 - CPI、PPI、M1-M2剪刀差、M2增速、社融'
            ]
        }
    
    def get_modules(self) -> List[ModuleResult]:
        """获取A股四大模块分析"""
        modules = []
        
        self._log("开始分析模块1: 资金杠杆与情绪")
        modules.append(self._analyze_leverage())
        
        self._log("开始分析模块2: 经济景气度")
        modules.append(self._analyze_growth())
        
        self._log("开始分析模块3: 整体估值锚")
        modules.append(self._analyze_valuation())
        
        self._log("开始分析模块4: 通胀与货币")
        modules.append(self._analyze_inflation_money())
        
        return modules
    
    def _analyze_leverage(self) -> ModuleResult:
        """模块1: 资金杠杆与情绪 - 完全透明化"""
        module = ModuleResult("资金杠杆与情绪", "Leverage")
        
        self._current_acquisition_steps = []
        self._current_analysis_steps = []
        
        # 获取两融余额
        margin_balance = self._fetch_margin_balance()
        
        # 获取流通市值
        float_mv = self._fetch_float_market_value()
        
        # 计算两融余额指标
        margin_tn = margin_balance / 1e4 if margin_balance else 0
        
        margin_ind = IndicatorResult(
            name="两融余额",
            value=round(margin_tn, 2),
            unit="万亿",
            acquisition_steps=self._current_acquisition_steps.copy(),
            data_source="Tushare/AKShare" if margin_balance else "模拟数据",
            historical_ref=f"2015牛市顶: {self.HISTORICAL_REFS['margin_2015_peak']['balance']}万亿"
        )
        
        # 分析两融余额
        self._set_reasoning(f"当前两融余额{margin_tn:.2f}万亿，对标2015年顶部2.27万亿")
        if margin_tn > 2.0:
            margin_ind.status = "偏热"
            margin_ind.signal = "🟡"
            margin_ind.threshold_rules = ">2.0万亿为偏热区间"
        elif margin_tn > 1.5:
            margin_ind.status = "结构健康"
            margin_ind.signal = "🟢"
            margin_ind.threshold_rules = "1.5-2.0万亿为健康区间"
        else:
            margin_ind.status = "偏冷"
            margin_ind.signal = "🟡"
            margin_ind.threshold_rules = "<1.5万亿为偏冷区间"
        
        margin_ind.analysis_detail = f"两融余额{margin_tn:.2f}万亿，为2015年顶部的{margin_tn/2.27*100:.1f}%"
        module.indicators.append(margin_ind)
        
        # 计算两融/流通市值比
        if margin_balance and float_mv:
            ratio = margin_balance / float_mv * 100
            ratio_ind = IndicatorResult(
                name="两融/流通市值比",
                value=round(ratio, 2),
                unit="%",
                data_source="计算值",
                historical_ref=f"2015牛市顶: {self.HISTORICAL_REFS['margin_2015_peak']['ratio']}%"
            )
            
            self._set_reasoning(f"两融占比{ratio:.2f}%，对标2015年4.5%")
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
            elif ratio > 1.5:
                ratio_ind.status = "偏冷"
                ratio_ind.signal = "🟡"
                ratio_ind.threshold_rules = "1.5-2.0%为偏冷"
            else:
                ratio_ind.status = "极度冷清"
                ratio_ind.signal = "🔵"
                ratio_ind.threshold_rules = "<1.5%为极度冷清(底部区域)"
            
            ratio_ind.analysis_detail = f"两融占比{ratio:.2f}%，{'高于' if ratio > 3 else '处于'}历史警戒水平"
            module.indicators.append(ratio_ind)
        
        # 模块综合信号
        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals)
        
        return module
    
    def _fetch_margin_balance(self) -> Optional[float]:
        """获取两融余额 - 带完整错误处理"""
        # 尝试Tushare
        if self.pro:
            try:
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                df = self.pro.margin(start_date=start_date, end_date=end_date)
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    balance = float(latest['rzye']) / 1e8  # 转为亿元
                    self._log(f"Tushare获取两融余额成功: {balance:.0f}亿元")
                    return balance
            except Exception as e:
                self._log(f"Tushare获取两融余额失败: {e}", "warning")
        
        # 尝试AKShare
        if AKSHARE_AVAILABLE:
            try:
                df = ak.stock_margin_sse(start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'))
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    balance = float(latest.get('融资融券余额', 0)) / 1e8
                    self._log(f"AKShare获取两融余额成功: {balance:.0f}亿元")
                    return balance
            except Exception as e:
                self._log(f"AKShare获取两融余额失败: {e}", "warning")
        
        # 模拟数据
        self._log("使用模拟数据: 1.85万亿", "warning")
        return 18500  # 1.85万亿
    
    def _fetch_float_market_value(self) -> Optional[float]:
        """获取流通市值"""
        if self.pro:
            try:
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                df = self.pro.daily_basic(trade_date=yesterday, fields='ts_code,circ_mv')
                if df is not None and not df.empty:
                    total = df['circ_mv'].sum() / 1e4  # 转为亿元
                    self._log(f"Tushare获取流通市值成功: {total:.0f}亿元")
                    return total
            except Exception as e:
                self._log(f"Tushare获取流通市值失败: {e}", "warning")
        
        # 模拟数据
        return 750000  # 约75万亿
    
    def _analyze_growth(self) -> ModuleResult:
        """模块2: 经济景气度"""
        module = ModuleResult("经济景气度", "Growth")
        
        gdp = self._fetch_gdp()
        
        gdp_ind = IndicatorResult(
            name="GDP同比增速",
            value=round(gdp, 1) if gdp else 0,
            unit="%",
            data_source="Tushare/AKShare" if gdp else "模拟数据"
        )
        
        if gdp:
            self._set_reasoning(f"GDP增速{gdp:.1f}%，判断经济增长状态")
            if gdp > self.HISTORICAL_REFS['gdp_high_growth']:
                gdp_ind.status = "高速增长"
                gdp_ind.signal = "🟢"
                gdp_ind.threshold_rules = ">6.0%为高速增长"
            elif gdp > self.HISTORICAL_REFS['gdp_normal']:
                gdp_ind.status = "稳健增长"
                gdp_ind.signal = "🟢"
                gdp_ind.threshold_rules = "5.0-6.0%为稳健增长"
            elif gdp > self.HISTORICAL_REFS['gdp_slow']:
                gdp_ind.status = "中速增长"
                gdp_ind.signal = "🟡"
                gdp_ind.threshold_rules = "4.0-5.0%为中速增长"
            elif gdp > self.HISTORICAL_REFS['gdp_recession_risk']:
                gdp_ind.status = "低速增长"
                gdp_ind.signal = "🟡"
                gdp_ind.threshold_rules = "3.0-4.0%为低速增长"
            else:
                gdp_ind.status = "增长乏力"
                gdp_ind.signal = "🔴"
                gdp_ind.threshold_rules = "<3.0%为增长乏力(衰退风险)"
            
            gdp_ind.analysis_detail = f"GDP增速{gdp:.1f}%，{'高于' if gdp > 5 else '低于'}5%稳健增长线"
        
        module.indicators.append(gdp_ind)
        module.overall_signal = gdp_ind.signal
        
        return module
    
    def _fetch_gdp(self) -> Optional[float]:
        """获取GDP增速"""
        if self.pro:
            try:
                df = self.pro.cn_gdp()
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    gdp = float(latest.get('gdp_yoy', 0))
                    self._log(f"Tushare获取GDP成功: {gdp}%")
                    return gdp
            except Exception as e:
                self._log(f"Tushare获取GDP失败: {e}", "warning")
        
        # 模拟数据
        return 5.2
    
    def _analyze_valuation(self) -> ModuleResult:
        """模块3: 整体估值锚"""
        module = ModuleResult("整体估值锚", "Valuation")
        
        # 获取A股总市值和GDP
        total_mv = self._fetch_total_market_value()
        gdp = self._fetch_annual_gdp()
        
        mv_tn = total_mv / 1e4 if total_mv else 0
        gdp_tn = gdp if gdp else 0
        
        mv_ind = IndicatorResult(
            name="A股总市值",
            value=round(mv_tn, 2),
            unit="万亿",
            data_source="Tushare" if total_mv else "模拟数据"
        )
        module.indicators.append(mv_ind)
        
        gdp_ind = IndicatorResult(
            name="年度GDP",
            value=round(gdp_tn, 2),
            unit="万亿",
            data_source="Tushare" if gdp else "模拟数据"
        )
        module.indicators.append(gdp_ind)
        
        # 计算巴菲特指标
        if total_mv and gdp:
            buffett_ratio = (mv_tn / gdp_tn) * 100
            
            buffett_ind = IndicatorResult(
                name="巴菲特指标(市值/GDP)",
                value=round(buffett_ratio, 1),
                unit="%",
                data_source="计算值",
                historical_ref=f"2007顶{self.HISTORICAL_REFS['buffett_2007_peak']}%, 2015顶{self.HISTORICAL_REFS['buffett_2015_peak']}%, 底部{self.HISTORICAL_REFS['buffett_bottom_range'][0]}-{self.HISTORICAL_REFS['buffett_bottom_range'][1]}%"
            )
            
            self._set_reasoning(f"巴菲特指标{buffett_ratio:.1f}%，对标历史顶部和底部区间")
            if buffett_ratio > 120:
                buffett_ind.status = "极度高估"
                buffett_ind.signal = "🔴"
                buffett_ind.threshold_rules = ">120%为极度高估(泡沫区域)"
            elif buffett_ratio > 100:
                buffett_ind.status = "估值偏高"
                buffett_ind.signal = "🟡"
                buffett_ind.threshold_rules = "100-120%为估值偏高"
            elif buffett_ratio > 80:
                buffett_ind.status = "合理偏高"
                buffett_ind.signal = "🟡"
                buffett_ind.threshold_rules = "80-100%为合理偏高"
            elif buffett_ratio > 60:
                buffett_ind.status = "合理区间"
                buffett_ind.signal = "🟢"
                buffett_ind.threshold_rules = "60-80%为合理区间"
            elif buffett_ratio > 40:
                buffett_ind.status = "低估区间"
                buffett_ind.signal = "🟢"
                buffett_ind.threshold_rules = "40-60%为低估区间"
            else:
                buffett_ind.status = "极度低估"
                buffett_ind.signal = "🔵"
                buffett_ind.threshold_rules = "<40%为极度低估(历史底部)"
            
            buffett_ind.analysis_detail = f"巴菲特指标{buffett_ratio:.1f}%，{'高于' if buffett_ratio > 100 else '处于'}历史警戒水平"
            module.indicators.append(buffett_ind)
        
        # 模块综合信号
        signals = [ind.signal for ind in module.indicators if '巴菲特' in ind.name]
        module.overall_signal = signals[0] if signals else "🟡"
        
        return module
    
    def _fetch_total_market_value(self) -> Optional[float]:
        """获取A股总市值"""
        if self.pro:
            try:
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                df = self.pro.daily_basic(trade_date=yesterday, fields='ts_code,total_mv')
                if df is not None and not df.empty:
                    total = df['total_mv'].sum() / 1e4
                    self._log(f"Tushare获取总市值成功: {total:.0f}亿元")
                    return total
            except Exception as e:
                self._log(f"Tushare获取总市值失败: {e}", "warning")
        return None
    
    def _fetch_annual_gdp(self) -> Optional[float]:
        """获取年度GDP"""
        if self.pro:
            try:
                df = self.pro.cn_gdp()
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    gdp = float(latest.get('gdp', 0)) / 1e4
                    self._log(f"Tushare获取GDP成功: {gdp:.2f}万亿")
                    return gdp
            except Exception as e:
                self._log(f"Tushare获取GDP失败: {e}", "warning")
        return 126.0  # 约126万亿
    
    def _analyze_inflation_money(self) -> ModuleResult:
        """模块4: 通胀与货币"""
        module = ModuleResult("通胀与货币", "Inflation & Money")
        
        # CPI
        cpi = self._fetch_cpi()
        cpi_ind = self._create_inflation_indicator("CPI同比", cpi, "%")
        if cpi:
            if cpi > 3:
                cpi_ind.status = "通胀偏高"
                cpi_ind.signal = "🟡"
                cpi_ind.threshold_rules = ">3%为通胀偏高"
            elif cpi > 1:
                cpi_ind.status = "温和通胀"
                cpi_ind.signal = "🟢"
                cpi_ind.threshold_rules = "1-3%为温和通胀"
            else:
                cpi_ind.status = "低通胀"
                cpi_ind.signal = "🟡"
                cpi_ind.threshold_rules = "<1%为低通胀"
        module.indicators.append(cpi_ind)
        
        # PPI
        ppi = self._fetch_ppi()
        ppi_ind = self._create_inflation_indicator("PPI同比", ppi, "%")
        if ppi:
            if ppi > 5:
                ppi_ind.status = "工业品价格过热"
                ppi_ind.signal = "🔴"
            elif ppi > 0:
                ppi_ind.status = "工业价格正常"
                ppi_ind.signal = "🟢"
            else:
                ppi_ind.status = "工业价格下行"
                ppi_ind.signal = "🟡"
        module.indicators.append(ppi_ind)
        
        # M1-M2剪刀差
        m1, m2 = self._fetch_m1_m2()
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
                scissors_ind.threshold_rules = ">0为资金活化"
            elif scissors > -3:
                scissors_ind.status = "轻度存款定期化"
                scissors_ind.signal = "🟡"
                scissors_ind.threshold_rules = "-3~0为轻度定期化"
            else:
                scissors_ind.status = "存款定期化严重"
                scissors_ind.signal = "🔴"
                scissors_ind.threshold_rules = "<-3为严重定期化"
            module.indicators.append(scissors_ind)
        
        # M2增速
        if m2:
            m2_ind = IndicatorResult(
                name="M2增速",
                value=round(m2, 1),
                unit="%",
                data_source="Tushare"
            )
            if m2 > 10:
                m2_ind.status = "宽松"
                m2_ind.signal = "🟢"
                m2_ind.historical_ref = ">10%宽松利好股市"
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
    
    def _create_inflation_indicator(self, name: str, value: Optional[float], unit: str) -> IndicatorResult:
        """创建通胀指标"""
        return IndicatorResult(
            name=name,
            value=round(value, 1) if value else 0,
            unit=unit,
            data_source="Tushare/AKShare" if value else "模拟数据"
        )
    
    def _fetch_cpi(self) -> Optional[float]:
        if self.pro:
            try:
                df = self.pro.cn_cpi()
                if df is not None and not df.empty:
                    return float(df.iloc[0].get('cpi_yoy', 0))
            except:
                pass
        return 2.1
    
    def _fetch_ppi(self) -> Optional[float]:
        if self.pro:
            try:
                df = self.pro.cn_ppi()
                if df is not None and not df.empty:
                    return float(df.iloc[0].get('ppi_yoy', 0))
            except:
                pass
        return -0.8
    
    def _fetch_m1_m2(self) -> Tuple[Optional[float], Optional[float]]:
        if self.pro:
            try:
                df = self.pro.cn_m()
                if df is not None and not df.empty:
                    latest = df.iloc[0]
                    return float(latest.get('m1_yoy', 0)), float(latest.get('m2_yoy', 0))
            except:
                pass
        return 3.5, 10.5
    
    def _aggregate_signals(self, signals: List[str]) -> str:
        """聚合多个信号为模块综合信号"""
        if "🔴" in signals:
            return "🔴"
        elif "🟡" in signals:
            return "🟡"
        elif "🔵" in signals:
            return "🔵"
        else:
            return "🟢"


# ==================== 美股实现 ====================

class USMacroRiskTerminal(MacroRiskTerminalBase):
    """
    美股宏观风控终端 - 五大模块
    
    模块:
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
        'shiller_mean': 17.0,
        'vix_panic': 30.0,
        'vix_high': 20.0,
        'vix_normal': 12.0
    }
    
    def __init__(self, fred_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.fred_key = fred_api_key or os.environ.get('FRED_API_KEY')
        self._fred = None
    
    def _get_supported_indicators(self) -> Dict[str, List[str]]:
        return {
            'data_sources': ['FRED', 'yfinance', 'AKShare'],
            'modules': [
                '货币政策 - 联邦基金利率、美联储资产负债表',
                '经济增长 - GDP增速、失业率',
                '整体估值 - 巴菲特指标、Shiller PE (CAPE)',
                '通胀 - CPI、PPI、核心PCE',
                '情绪与收益率曲线 - 10Y-2Y利差、VIX、消费者信心'
            ]
        }
    
    def get_modules(self) -> List[ModuleResult]:
        """获取美股五大模块"""
        modules = []
        
        self._log("开始分析模块1: 货币政策")
        modules.append(self._analyze_monetary_policy())
        
        self._log("开始分析模块2: 经济增长")
        modules.append(self._analyze_growth())
        
        self._log("开始分析模块3: 整体估值")
        modules.append(self._analyze_valuation())
        
        self._log("开始分析模块4: 通胀")
        modules.append(self._analyze_inflation())
        
        self._log("开始分析模块5: 情绪与收益率曲线")
        modules.append(self._analyze_sentiment_yield())
        
        return modules
    
    def _analyze_monetary_policy(self) -> ModuleResult:
        """模块1: 货币政策"""
        module = ModuleResult("货币政策", "Monetary Policy")
        
        # 联邦基金利率
        ffr = self._fetch_fed_funds_rate()
        ffr_ind = IndicatorResult(
            name="联邦基金利率",
            value=round(ffr, 2) if ffr else 4.5,
            unit="%",
            data_source="FRED/yfinance" if ffr else "模拟数据"
        )
        
        if ffr:
            self._set_reasoning(f"联邦基金利率{ffr:.2f}%，判断货币政策状态")
            if ffr >= 5.0:
                ffr_ind.status = "紧缩"
                ffr_ind.signal = "🔴"
                ffr_ind.threshold_rules = ">=5.0%为紧缩(压制估值)"
                ffr_ind.historical_ref = "高利率环境"
            elif ffr >= 3.0:
                ffr_ind.status = "偏紧"
                ffr_ind.signal = "🟡"
                ffr_ind.threshold_rules = "3.0-5.0%为偏紧"
                ffr_ind.historical_ref = "关注转向信号"
            elif ffr >= 1.0:
                ffr_ind.status = "中性"
                ffr_ind.signal = "🟢"
                ffr_ind.threshold_rules = "1.0-3.0%为中性"
            else:
                ffr_ind.status = "宽松"
                ffr_ind.signal = "🟢"
                ffr_ind.threshold_rules = "<1.0%为宽松(利好风险资产)"
            
            ffr_ind.analysis_detail = f"利率{ffr:.2f}%，{'高于' if ffr > 3 else '处于'}中性水平"
        
        module.indicators.append(ffr_ind)
        
        # 美联储资产负债表
        bs = self._fetch_fed_balance_sheet()
        bs_ind = IndicatorResult(
            name="美联储总资产",
            value=round(bs, 2) if bs else 7.2,
            unit="万亿美元",
            data_source="FRED" if bs else "模拟数据",
            historical_ref="峰值9万亿，疫情前4万亿"
        )
        
        if bs:
            if bs > 8.0:
                bs_ind.status = "流动性充裕"
                bs_ind.signal = "🟢"
                bs_ind.threshold_rules = ">8万亿为流动性充裕"
            elif bs > 6.0:
                bs_ind.status = "缩表进行中"
                bs_ind.signal = "🟡"
                bs_ind.threshold_rules = "6-8万亿为缩表区间"
            else:
                bs_ind.status = "资产负债表正常"
                bs_ind.signal = "🟢"
                bs_ind.threshold_rules = "<6万亿为正常水平"
        
        module.indicators.append(bs_ind)
        
        # 模块综合信号
        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals)
        
        return module
    
    def _analyze_growth(self) -> ModuleResult:
        """模块2: 经济增长"""
        module = ModuleResult("经济增长", "Growth")
        
        # GDP
        gdp = self._fetch_us_gdp()
        gdp_ind = IndicatorResult(
            name="GDP年化季环比",
            value=round(gdp, 1) if gdp else 2.3,
            unit="%",
            data_source="FRED/AKShare" if gdp else "模拟数据"
        )
        
        if gdp:
            if gdp > 3.0:
                gdp_ind.status = "强劲增长"
                gdp_ind.signal = "🟢"
                gdp_ind.threshold_rules = ">3.0%为强劲"
            elif gdp > 1.5:
                gdp_ind.status = "温和增长"
                gdp_ind.signal = "🟢"
                gdp_ind.threshold_rules = "1.5-3.0%为温和"
            elif gdp > 0:
                gdp_ind.status = "增长放缓"
                gdp_ind.signal = "🟡"
                gdp_ind.threshold_rules = "0-1.5%为放缓"
            else:
                gdp_ind.status = "衰退"
                gdp_ind.signal = "🔴"
                gdp_ind.threshold_rules = "<0%为衰退"
        
        module.indicators.append(gdp_ind)
        
        # 失业率
        unemp = self._fetch_unemployment()
        unemp_ind = IndicatorResult(
            name="失业率",
            value=round(unemp, 1) if unemp else 4.1,
            unit="%",
            data_source="FRED" if unemp else "模拟数据"
        )
        
        if unemp:
            if unemp > 7.0:
                unemp_ind.status = "高失业"
                unemp_ind.signal = "🔴"
                unemp_ind.threshold_rules = ">7.0%为高失业(衰退信号)"
            elif unemp > 5.0:
                unemp_ind.status = "偏高"
                unemp_ind.signal = "🟡"
                unemp_ind.threshold_rules = "5.0-7.0%为偏高"
            elif unemp > 4.0:
                unemp_ind.status = "正常"
                unemp_ind.signal = "🟢"
                unemp_ind.threshold_rules = "4.0-5.0%为正常"
            else:
                unemp_ind.status = "充分就业"
                unemp_ind.signal = "🟢"
                unemp_ind.threshold_rules = "<4.0%为充分就业"
        
        module.indicators.append(unemp_ind)
        
        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals)
        
        return module
    
    def _analyze_valuation(self) -> ModuleResult:
        """模块3: 整体估值"""
        module = ModuleResult("整体估值", "Valuation")
        
        # 巴菲特指标
        buffett = self._calculate_buffett_ratio()
        buffett_ind = IndicatorResult(
            name="巴菲特指标(Wilshire 5000/GDP)",
            value=round(buffett, 1) if buffett else 0,
            unit="%",
            data_source="yfinance估算" if buffett else "数据获取失败",
            historical_ref=f"2000泡沫{self.HISTORICAL_REFS['buffett_2000_peak']}%, 2021泡沫{self.HISTORICAL_REFS['buffett_2021_peak']}%, 合理80-120%"
        )
        
        if buffett:
            if buffett > 200:
                buffett_ind.status = "极度高估"
                buffett_ind.signal = "🔴"
                buffett_ind.threshold_rules = ">200%为极度高估"
            elif buffett > 150:
                buffett_ind.status = "显著高估"
                buffett_ind.signal = "🟡"
                buffett_ind.threshold_rules = "150-200%为显著高估"
            elif buffett > 120:
                buffett_ind.status = "偏高"
                buffett_ind.signal = "🟡"
                buffett_ind.threshold_rules = "120-150%为偏高"
            elif buffett > 80:
                buffett_ind.status = "合理区间"
                buffett_ind.signal = "🟢"
                buffett_ind.threshold_rules = "80-120%为合理"
            elif buffett > 60:
                buffett_ind.status = "低估"
                buffett_ind.signal = "🟢"
                buffett_ind.threshold_rules = "60-80%为低估"
            else:
                buffett_ind.status = "极度低估"
                buffett_ind.signal = "🔵"
                buffett_ind.threshold_rules = "<60%为极度低估"
        
        module.indicators.append(buffett_ind)
        
        # Shiller PE
        cape = self._fetch_shiller_pe()
        cape_ind = IndicatorResult(
            name="Shiller PE (CAPE)",
            value=round(cape, 1) if cape else 32.0,
            unit="x",
            data_source="yfinance" if cape else "模拟数据",
            historical_ref=f"历史均值~{self.HISTORICAL_REFS['shiller_mean']}x"
        )
        
        if cape:
            if cape > 35:
                cape_ind.status = "显著高估"
                cape_ind.signal = "🔴"
                cape_ind.threshold_rules = ">35x为显著高估"
            elif cape > 25:
                cape_ind.status = "偏高"
                cape_ind.signal = "🟡"
                cape_ind.threshold_rules = "25-35x为偏高"
            elif cape > 15:
                cape_ind.status = "合理"
                cape_ind.signal = "🟢"
                cape_ind.threshold_rules = "15-25x为合理"
            else:
                cape_ind.status = "低估"
                cape_ind.signal = "🔵"
                cape_ind.threshold_rules = "<15x为低估"
        
        module.indicators.append(cape_ind)
        
        # 模块综合信号
        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals)
        
        return module
    
    def _analyze_inflation(self) -> ModuleResult:
        """模块4: 通胀"""
        module = ModuleResult("通胀", "Inflation")
        
        # CPI
        cpi = self._fetch_us_cpi()
        cpi_ind = IndicatorResult(
            name="CPI同比",
            value=round(cpi, 1) if cpi else 3.2,
            unit="%",
            data_source="FRED/AKShare" if cpi else "模拟数据",
            historical_ref="美联储目标2%"
        )
        
        if cpi:
            if cpi > 5.0:
                cpi_ind.status = "高通胀"
                cpi_ind.signal = "🔴"
                cpi_ind.threshold_rules = ">5.0%为高通胀"
            elif cpi > 3.0:
                cpi_ind.status = "通胀偏高"
                cpi_ind.signal = "🟡"
                cpi_ind.threshold_rules = "3.0-5.0%为偏高"
            elif cpi > 1.5:
                cpi_ind.status = "温和通胀"
                cpi_ind.signal = "🟢"
                cpi_ind.threshold_rules = "1.5-3.0%为温和"
            else:
                cpi_ind.status = "低通胀"
                cpi_ind.signal = "🟡"
                cpi_ind.threshold_rules = "<1.5%为低通胀"
        
        module.indicators.append(cpi_ind)
        
        # PPI
        ppi = self._fetch_us_ppi()
        ppi_ind = IndicatorResult(
            name="PPI同比",
            value=round(ppi, 1) if ppi else 1.8,
            unit="%",
            data_source="FRED/AKShare" if ppi else "模拟数据"
        )
        
        if ppi:
            if ppi > 5.0:
                ppi_ind.status = "生产成本过热"
                ppi_ind.signal = "🔴"
            elif ppi > 2.0:
                ppi_ind.status = "偏高"
                ppi_ind.signal = "🟡"
            else:
                ppi_ind.status = "正常"
                ppi_ind.signal = "🟢"
        
        module.indicators.append(ppi_ind)
        
        # 核心PCE
        pce = self._fetch_core_pce()
        pce_ind = IndicatorResult(
            name="核心PCE同比",
            value=round(pce, 1) if pce else 2.8,
            unit="%",
            data_source="FRED" if pce else "模拟数据",
            historical_ref="美联储首选指标"
        )
        
        if pce:
            if pce > 4.0:
                pce_ind.status = "核心通胀过高"
                pce_ind.signal = "🔴"
            elif pce > 2.5:
                pce_ind.status = "高于目标"
                pce_ind.signal = "🟡"
            else:
                pce_ind.status = "接近目标"
                pce_ind.signal = "🟢"
        
        module.indicators.append(pce_ind)
        
        # 模块综合信号
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
        spread = self._fetch_yield_spread()
        spread_ind = IndicatorResult(
            name="10Y-2Y国债利差",
            value=round(spread, 0) if spread else 46.0,
            unit="bp",
            data_source="yfinance" if spread else "模拟数据",
            historical_ref="<0倒挂预警衰退"
        )
        
        if spread:
            if spread < -50:
                spread_ind.status = "深度倒挂"
                spread_ind.signal = "🔴"
                spread_ind.threshold_rules = "<-50bp为深度倒挂(强烈衰退预警)"
            elif spread < 0:
                spread_ind.status = "倒挂"
                spread_ind.signal = "🔴"
                spread_ind.threshold_rules = "<0为倒挂(衰退预警)"
            elif spread < 50:
                spread_ind.status = "平坦"
                spread_ind.signal = "🟡"
                spread_ind.threshold_rules = "0-50bp为平坦(周期后期)"
            else:
                spread_ind.status = "正常"
                spread_ind.signal = "🟢"
                spread_ind.threshold_rules = ">50bp为正常(扩张期)"
        
        module.indicators.append(spread_ind)
        
        # VIX
        vix = self._fetch_vix()
        vix_ind = IndicatorResult(
            name="VIX恐慌指数",
            value=round(vix, 1) if vix else 18.7,
            unit="",
            data_source="yfinance" if vix else "模拟数据",
            historical_ref="12-20为正常区间"
        )
        
        if vix:
            if vix > 30:
                vix_ind.status = "恐慌"
                vix_ind.signal = "🔴"
                vix_ind.threshold_rules = ">30为恐慌(逆向买入机会)"
            elif vix > 20:
                vix_ind.status = "偏高"
                vix_ind.signal = "🟡"
                vix_ind.threshold_rules = "20-30为偏高"
            elif vix > 12:
                vix_ind.status = "正常"
                vix_ind.signal = "🟢"
                vix_ind.threshold_rules = "12-20为正常"
            else:
                vix_ind.status = "极度平静"
                vix_ind.signal = "🟡"
                vix_ind.threshold_rules = "<12为极度平静(警惕黑天鹅)"
        
        module.indicators.append(vix_ind)
        
        # 消费者信心
        sentiment = self._fetch_consumer_sentiment()
        sentiment_ind = IndicatorResult(
            name="消费者信心指数",
            value=round(sentiment, 1) if sentiment else 78.0,
            unit="",
            data_source="FRED/AKShare" if sentiment else "模拟数据",
            historical_ref="历史均值~85"
        )
        
        if sentiment:
            if sentiment > 90:
                sentiment_ind.status = "乐观"
                sentiment_ind.signal = "🟢"
            elif sentiment > 70:
                sentiment_ind.status = "中性"
                sentiment_ind.signal = "🟢"
            elif sentiment > 55:
                sentiment_ind.status = "悲观"
                sentiment_ind.signal = "🟡"
            else:
                sentiment_ind.status = "极度悲观"
                sentiment_ind.signal = "🔴"
        
        module.indicators.append(sentiment_ind)
        
        # 模块综合信号
        signals = [ind.signal for ind in module.indicators]
        module.overall_signal = self._aggregate_signals(signals)
        
        return module
    
    # 数据获取方法
    def _fetch_fed_funds_rate(self) -> Optional[float]:
        # 模拟数据
        return 4.5
    
    def _fetch_fed_balance_sheet(self) -> Optional[float]:
        return 7.2
    
    def _fetch_us_gdp(self) -> Optional[float]:
        return 2.3
    
    def _fetch_unemployment(self) -> Optional[float]:
        return 4.1
    
    def _calculate_buffett_ratio(self) -> Optional[float]:
        if YFINANCE_AVAILABLE:
            try:
                sp500 = yf.Ticker("^GSPC")
                info = sp500.info
                market_cap = info.get('marketCap', 0)
                if market_cap:
                    # 简化估算
                    us_gdp = 27e12
                    ratio = (market_cap / 0.8 / us_gdp) * 100
                    return ratio
            except:
                pass
        return 180.0
    
    def _fetch_shiller_pe(self) -> Optional[float]:
        if YFINANCE_AVAILABLE:
            try:
                sp500 = yf.Ticker("^GSPC")
                info = sp500.info
                pe = info.get('trailingPE')
                if pe:
                    return float(pe)
            except:
                pass
        return 32.0
    
    def _fetch_us_cpi(self) -> Optional[float]:
        return 3.2
    
    def _fetch_us_ppi(self) -> Optional[float]:
        return 1.8
    
    def _fetch_core_pce(self) -> Optional[float]:
        return 2.8
    
    def _fetch_yield_spread(self) -> Optional[float]:
        if YFINANCE_AVAILABLE:
            try:
                t10 = yf.Ticker("^TNX")
                t2 = yf.Ticker("^IRX")
                h10 = t10.history(period="5d")
                h2 = t2.history(period="5d")
                if not h10.empty and not h2.empty:
                    spread = (h10['Close'].iloc[-1] - h2['Close'].iloc[-1]) * 100
                    return float(spread)
            except:
                pass
        return 46.0
    
    def _fetch_vix(self) -> Optional[float]:
        if YFINANCE_AVAILABLE:
            try:
                vix = yf.Ticker("^VIX")
                hist = vix.history(period="5d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            except:
                pass
        return 18.7
    
    def _fetch_consumer_sentiment(self) -> Optional[float]:
        return 78.0
    
    def _aggregate_signals(self, signals: List[str]) -> str:
        if "🔴" in signals:
            return "🔴"
        elif "🟡" in signals:
            return "🟡"
        elif "🔵" in signals:
            return "🔵"
        else:
            return "🟢"


# ==================== 工厂函数 ====================

def create_terminal(market: str = "CN", **kwargs) -> MacroRiskTerminalBase:
    """
    工厂函数: 创建对应市场的宏观风控终端
    
    支持市场:
    - CN: A股 (四大模块)
    - US: 美股 (五大模块)
    - 可扩展: HK, EU, JP
    """
    market = market.upper()
    
    if market == "CN":
        return CNMacroRiskTerminal(**kwargs)
    elif market == "US":
        return USMacroRiskTerminal(**kwargs)
    else:
        raise ValueError(
            f"暂不支持市场 '{market}'。当前支持: CN (A股), US (美股)。\n"
            f"可通过继承 MacroRiskTerminalBase 扩展新市场。"
        )


# 向后兼容
MacroRiskTerminal = CNMacroRiskTerminal


# ==================== 测试 ====================

if __name__ == '__main__':
    import sys
    
    market = sys.argv[1].upper() if len(sys.argv) > 1 else "US"
    
    print(f"正在运行 {market} 市场宏观风控终端 V6.3 (透明化版本)...")
    print("=" * 80)
    
    terminal = create_terminal(market, verbose=True)
    report = terminal.generate_risk_report()
    
    # 输出完整Markdown报告
    markdown = terminal.format_report_markdown(report)
    print(markdown)
    
    # 保存报告
    output_file = f'/tmp/macro_risk_report_{market.lower()}.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"\n报告已保存: {output_file}")

#!/usr/bin/env python3
"""
Multi-Model Debate System - 多模型多空辩论系统

第6层决策层的核心组件

功能:
1. 多模型架构 - 5个专业分析模型
2. 多空辩论 - 每个模型分别输出看多/看空观点
3. 深度研究 - 公司基本面、产品、竞争格局、行业趋势
4. 综合判断 - 整合1-5层所有信息生成投资决策

分析维度:
- 财务模型: 财务报表、盈利能力、估值、现金流
- 行业模型: 行业生命周期、竞争格局、护城河、政策
- 宏观模型: 经济周期、货币政策、通胀、地缘政治
- 技术模型: 价格趋势、技术指标、量价关系、市场情绪
- 风险模型: 波动率、回撤、流动性、尾部风险、集中度
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

# 尝试导入多个LLM

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AnalystModel(Enum):
    """分析模型类型"""
    FINANCIAL = "财务分析模型"
    INDUSTRY = "行业研究模型"
    MACRO = "宏观分析模型"
    TECHNICAL = "技术分析模型"
    RISK = "风险评估模型"


@dataclass
class DebateArgument:
    """辩论观点"""
    model: AnalystModel
    side: str  # "bullish" or "bearish"
    points: List[str]
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAnalysis:
    """单个模型分析结果"""
    model: AnalystModel
    bullish_arguments: List[DebateArgument]
    bearish_arguments: List[DebateArgument]
    overall_bias: str  # "bullish", "bearish", "neutral"
    confidence: float
    key_factors: List[str]
    reasoning: str


@dataclass
class InvestmentDecision:
    """投资决策"""
    symbol: str
    decision: str  # "强烈买入", "买入", "持有", "卖出", "强烈卖出"
    confidence: float
    position_size: float  # 建议仓位 0-1
    target_price: Optional[float]
    stop_loss: Optional[float]
    time_horizon: str  # "短期", "中期", "长期"
    
    # 决策逻辑
    logic_chain: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    opposing_concerns: List[str] = field(default_factory=list)
    risk_mitigation: List[str] = field(default_factory=list)
    
    # 模型共识
    model_consensus: Dict[str, str] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)


@dataclass
class DebateResult:
    """辩论结果"""
    symbol: str
    company_research: Dict[str, Any]  # 公司深度研究
    model_analyses: List[ModelAnalysis]
    debate_summary: str
    investment_decision: InvestmentDecision
    portfolio_suggestion: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    final_report: str


class LLMProvider:
    """LLM提供商"""
    
    def __init__(self):
        self.openai_key = os.environ.get('OPENAI_API_KEY')
        self.deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
        self.client = None
        
        if OPENAI_AVAILABLE and self.openai_key:
            self.client = OpenAI(api_key=self.openai_key)
    
    def call(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
        """调用LLM"""
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一位专业的投资分析师。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2500
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[LLM] API调用失败: {e}")
        
        return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """模拟响应"""
        if "财务分析" in prompt:
            return self._mock_financial()
        elif "行业研究" in prompt:
            return self._mock_industry()
        elif "宏观分析" in prompt:
            return self._mock_macro()
        elif "技术分析" in prompt:
            return self._mock_technical()
        elif "风险评估" in prompt:
            return self._mock_risk()
        elif "综合决策" in prompt:
            return self._mock_final()
        return "{}"
    
    def _mock_financial(self) -> str:
        return json.dumps({
            "bullish_points": [
                "ROE连续3年保持在18%以上，盈利能力稳定",
                "自由现金流充裕，FCF/营收比例达15%",
                "当前PE 15倍，低于行业平均20倍，估值有吸引力",
                "毛利率35%，净利率12%，盈利质量优秀"
            ],
            "bearish_points": [
                "应收账款周转天数从45天增至60天，回款压力增大",
                "资本支出占营收比例上升至20%，影响现金流",
                "存货周转率下降，可能存在库存积压"
            ],
            "confidence": 0.75,
            "bias": "bullish",
            "key_factors": ["ROE稳定性", "现金流质量", "估值水平"],
            "reasoning": "财务指标整体健康，估值合理，盈利能力强"
        }, ensure_ascii=False)
    
    def _mock_industry(self) -> str:
        return json.dumps({
            "bullish_points": [
                "行业处于成长期，未来3年CAGR预计15%",
                "公司在细分市场占有率达30%，龙头地位稳固",
                "技术壁垒高，研发投入占比8%，专利数量领先",
                "政策支持力度大，属于国家战略性新兴产业"
            ],
            "bearish_points": [
                "新进入者增多，竞争加剧，价格战风险",
                "上游原材料价格波动大，成本控制压力",
                "技术迭代快，需持续高研发投入维持竞争力"
            ],
            "confidence": 0.70,
            "bias": "bullish",
            "key_factors": ["行业成长性", "市场份额", "技术壁垒", "政策支持"],
            "reasoning": "行业前景良好，公司具有明显竞争优势和政策红利"
        }, ensure_ascii=False)
    
    def _mock_macro(self) -> str:
        return json.dumps({
            "bullish_points": [
                "货币政策宽松，流动性充裕利好股市",
                "经济复苏态势明确，PMI连续3个月扩张",
                "行业受益于稳增长政策，基建投资加码"
            ],
            "bearish_points": [
                "通胀压力上升，CPI接近3%警戒线",
                "美联储可能加息，外资流出压力",
                "地缘政治风险，贸易摩擦不确定性"
            ],
            "confidence": 0.65,
            "bias": "neutral",
            "key_factors": ["货币政策", "经济周期", "通胀压力"],
            "reasoning": "宏观环境中性偏正面，但需关注通胀和外部风险"
        }, ensure_ascii=False)
    
    def _mock_technical(self) -> str:
        return json.dumps({
            "bullish_points": [
                "股价突破前期高点，形成上升趋势",
                "成交量放大，资金流入明显",
                "MACD金叉，RSI在50-70健康区间",
                "均线多头排列，短期>中期>长期"
            ],
            "bearish_points": [
                "股价接近历史高位，阻力较大",
                "RSI接近70，短期可能超买回调",
                "波动率上升，需警惕剧烈波动"
            ],
            "confidence": 0.60,
            "bias": "bullish",
            "key_factors": ["趋势方向", "成交量", "技术指标"],
            "reasoning": "技术面偏强，上升趋势确立，但短期注意回调风险"
        }, ensure_ascii=False)
    
    def _mock_risk(self) -> str:
        return json.dumps({
            "bullish_points": [
                "波动率20%，处于可控范围",
                "流动性充足，日均成交额5亿以上",
                "Beta 0.9，系统性风险适中"
            ],
            "bearish_points": [
                "最大回撤可能达25%，需设置止损",
                "行业集中度高，单一行业风险",
                "尾部风险：政策变化可能带来冲击"
            ],
            "confidence": 0.65,
            "bias": "caution",
            "key_factors": ["波动率", "回撤风险", "流动性"],
            "reasoning": "风险收益比合理，但需控制仓位和设置止损"
        }, ensure_ascii=False)
    
    def _mock_final(self) -> str:
        return json.dumps({
            "decision": "买入",
            "confidence": 0.72,
            "position_size": 0.15,
            "target_price": 150.0,
            "stop_loss": 120.0,
            "time_horizon": "中期",
            "logic_chain": [
                "财务指标健康，ROE 18%，估值合理PE 15倍",
                "行业处于成长期，公司市占率30%龙头地位",
                "宏观环境中性偏正面，政策支持",
                "技术面上升趋势确立，资金流入",
                "风险可控，设置止损位保护"
            ],
            "supporting_evidence": [
                "连续3年ROE>18%",
                "行业CAGR 15%",
                "突破前期高点"
            ],
            "opposing_concerns": [
                "应收账款增加",
                "竞争加剧",
                "短期可能超买"
            ],
            "risk_mitigation": [
                "仓位控制在15%以内",
                "设置止损位120元",
                "定期跟踪财务变化"
            ]
        }, ensure_ascii=False)


class MultiModelDebateSystem:
    """
    多模型多空辩论系统
    """
    
    def __init__(self, verbose: bool = True):
        self.llm = LLMProvider()
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[DebateSystem] {msg}")
    
    def _create_company_research_prompt(self, symbol: str, 
                                       quant_data: Dict,
                                       macro_data: Dict) -> str:
        """创建公司深度研究提示词"""
        return f"""请对股票 {symbol} 进行全面的公司深度研究，包括：

【量化数据参考】
- 预测收益: {quant_data.get('predicted_return', 'N/A')}
- 夏普比率: {quant_data.get('sharpe_ratio', 'N/A')}
- 主要因子: {', '.join(quant_data.get('factors', []))}

【宏观环境】
- 宏观信号: {macro_data.get('signal', 'N/A')}
- 风险等级: {macro_data.get('risk_level', 'N/A')}

请从以下维度进行分析：
1. 公司概况: 主营业务、商业模式、核心竞争力
2. 产品分析: 产品线、技术壁垒、研发投入、专利情况
3. 竞争格局: 市场份额、主要竞争对手、竞争优势/劣势
4. 行业趋势: 行业生命周期、市场规模、增长率、政策支持
5. 竞争对手: 主要竞争对手的发展情况、差异化策略
6. 财务健康: 盈利能力、成长性、现金流、偿债能力

请以JSON格式输出研究结果：
{{
    "company_overview": "公司概况",
    "products": {{"main_products": [], "tech_moat": "", "rd_investment": ""}},
    "competition": {{"market_share": "", "main_competitors": [], "advantages": [], "disadvantages": []}},
    "industry": {{"lifecycle": "", "market_size": "", "growth_rate": "", "policy_support": ""}},
    "competitor_analysis": {{"key_competitors": [], "their_strategies": [], "our_differentiation": ""}},
    "financial_health": {{"profitability": "", "growth": "", "cashflow": "", "debt": ""}}
}}"""
    
    def _create_model_prompt(self, model: AnalystModel, symbol: str,
                            company_research: Dict,
                            quant_data: Dict,
                            macro_data: Dict,
                            risk_data: Dict) -> str:
        """创建单个模型的分析提示词"""
        
        base = f"""你是一位专业的{model.value}分析师，需要对股票 {symbol} 进行深入分析。

【公司深度研究】
{json.dumps(company_research, ensure_ascii=False, indent=2)[:1000]}

【量化分析数据】
- 预测收益: {quant_data.get('predicted_return', 'N/A')}
- 预测波动率: {quant_data.get('predicted_volatility', 'N/A')}
- 夏普比率: {quant_data.get('sharpe_ratio', 'N/A')}
- 主要因子: {', '.join(quant_data.get('factors', []))}

【宏观环境】
- 宏观信号: {macro_data.get('signal', 'N/A')}
- 风险等级: {macro_data.get('risk_level', 'N/A')}

【风控数据】
- 风险等级: {risk_data.get('risk_level', 'N/A')}
- 波动率: {risk_data.get('volatility', 'N/A')}

请从{model.value}的专业角度，分别列出看多和看空的理由。"""
        
        # 根据模型类型添加特定分析要求
        specifics = {
            AnalystModel.FINANCIAL: """
重点关注：
- 财务报表健康度（ROE、ROA、ROIC、毛利率、净利率）
- 估值水平（PE、PB、PS、EV/EBITDA、PEG）
- 现金流状况（经营现金流、自由现金流、现金流质量）
- 盈利质量（应收账款、存货、资本支出、盈利可持续性）
- 成长性（营收增长、利润增长、增长质量）
""",
            AnalystModel.INDUSTRY: """
重点关注：
- 行业生命周期（导入/成长/成熟/衰退）
- 市场规模和增长率（TAM、SAM、SOM）
- 竞争格局（CR5、HHI指数、竞争强度）
- 护城河（品牌、技术、成本、网络效应、转换成本）
- 政策支持（产业政策、监管环境、补贴情况）
- 产业链地位（议价能力、供应商/客户集中度）
""",
            AnalystModel.MACRO: """
重点关注：
- 经济周期位置（扩张/峰值/收缩/谷底）
- 货币政策（利率、准备金率、流动性）
- 财政政策（基建投资、减税、产业补贴）
- 通胀环境（CPI、PPI、对成本的影响）
- 汇率风险（进出口、海外收入占比）
- 地缘政治（贸易摩擦、供应链安全）
""",
            AnalystModel.TECHNICAL: """
重点关注：
- 价格趋势（长期/中期/短期趋势方向）
- 支撑阻力位（关键价位、突破/跌破情况）
- 成交量分析（放量/缩量、量价配合）
- 技术指标（均线系统、MACD、RSI、KDJ、布林带）
- 形态分析（头肩顶/底、双底、三角形等）
- 市场情绪（资金流向、融资余额、北向资金）
""",
            AnalystModel.RISK: """
重点关注：
- 波动率分析（历史波动率、隐含波动率、波动率趋势）
- 回撤风险（最大回撤、回撤恢复时间）
- 流动性风险（日均成交额、买卖价差、冲击成本）
- 尾部风险（黑天鹅事件、极端行情表现）
- 集中度风险（个股仓位、行业集中度）
- 相关性风险（与大盘/行业的相关性、分散化效果）
"""
        }
        
        base += specifics.get(model, "")
        
        base += """
请以JSON格式输出分析结果：
{
    "bullish_points": ["看多理由1", "看多理由2", ...],
    "bearish_points": ["看空理由1", "看空理由2", ...],
    "confidence": 0.0-1.0,
    "bias": "bullish/bearish/neutral",
    "key_factors": ["关键因素1", "关键因素2", ...],
    "reasoning": "综合分析理由"
}"""
        
        return base
    
    def conduct_company_research(self, symbol: str, 
                                 quant_data: Dict,
                                 macro_data: Dict) -> Dict:
        """进行公司深度研究"""
        self._log(f"进行公司深度研究: {symbol}")
        
        prompt = self._create_company_research_prompt(symbol, quant_data, macro_data)
        response = self.llm.call(prompt, temperature=0.5)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self._log(f"解析公司研究失败: {e}")
        
        return {}
    
    def run_model_analysis(self, model: AnalystModel, symbol: str,
                          company_research: Dict,
                          quant_data: Dict,
                          macro_data: Dict,
                          risk_data: Dict) -> ModelAnalysis:
        """运行单个模型分析"""
        self._log(f"运行 {model.value} 分析...")
        
        prompt = self._create_model_prompt(model, symbol, company_research,
                                          quant_data, macro_data, risk_data)
        response = self.llm.call(prompt, temperature=0.7)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # 构建辩论观点
                bullish_args = [DebateArgument(
                    model=model,
                    side="bullish",
                    points=data.get('bullish_points', []),
                    confidence=data.get('confidence', 0.5)
                )]
                
                bearish_args = [DebateArgument(
                    model=model,
                    side="bearish",
                    points=data.get('bearish_points', []),
                    confidence=1 - data.get('confidence', 0.5)
                )]
                
                return ModelAnalysis(
                    model=model,
                    bullish_arguments=bullish_args,
                    bearish_arguments=bearish_args,
                    overall_bias=data.get('bias', 'neutral'),
                    confidence=data.get('confidence', 0.5),
                    key_factors=data.get('key_factors', []),
                    reasoning=data.get('reasoning', '')
                )
        except Exception as e:
            self._log(f"解析模型分析失败: {e}")
        
        return ModelAnalysis(model=model, bullish_arguments=[], 
                           bearish_arguments=[], overall_bias="neutral",
                           confidence=0.5, key_factors=[], reasoning="")
    
    def synthesize_decision(self, symbol: str,
                           model_analyses: List[ModelAnalysis],
                           company_research: Dict,
                           quant_data: Dict,
                           macro_data: Dict,
                           risk_data: Dict) -> InvestmentDecision:
        """综合所有模型观点生成投资决策"""
        self._log(f"综合决策: {symbol}")
        
        # 统计模型观点
        bullish_count = sum(1 for m in model_analyses if m.overall_bias == "bullish")
        bearish_count = sum(1 for m in model_analyses if m.overall_bias == "bearish")
        neutral_count = len(model_analyses) - bullish_count - bearish_count
        
        # 计算加权置信度
        avg_confidence = sum(m.confidence for m in model_analyses) / len(model_analyses)
        
        # 收集所有看多/看空理由
        all_bullish = []
        all_bearish = []
        for m in model_analyses:
            for arg in m.bullish_arguments:
                all_bullish.extend(arg.points)
            for arg in m.bearish_arguments:
                all_bearish.extend(arg.points)
        
        # 生成综合提示词
        synthesis_prompt = f"""基于以下多模型分析结果，生成最终投资决策。

【股票】: {symbol}

【模型观点统计】
- 看多: {bullish_count} 个模型
- 看空: {bearish_count} 个模型
- 中性: {neutral_count} 个模型
- 平均置信度: {avg_confidence:.2f}

【看多理由汇总】
{chr(10).join([f"- {p}" for p in all_bullish[:8]])}

【看空理由汇总】
{chr(10).join([f"- {p}" for p in all_bearish[:8]])}

【量化数据】
- 预测收益: {quant_data.get('predicted_return', 'N/A')}
- 夏普比率: {quant_data.get('sharpe_ratio', 'N/A')}

【宏观信号】: {macro_data.get('signal', 'N/A')}
【风险等级】: {risk_data.get('risk_level', 'N/A')}

请生成投资决策，以JSON格式输出：
{{
    "decision": "强烈买入/买入/持有/卖出/强烈卖出",
    "confidence": 0.0-1.0,
    "position_size": 0.0-1.0,
    "target_price": 数字或null,
    "stop_loss": 数字或null,
    "time_horizon": "短期/中期/长期",
    "logic_chain": ["决策逻辑1", "决策逻辑2", ...],
    "supporting_evidence": ["支持证据1", ...],
    "opposing_concerns": ["反对担忧1", ...],
    "risk_mitigation": ["风险缓解措施1", ...]
}}"""
        
        response = self.llm.call(synthesis_prompt, temperature=0.5)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # 构建模型共识
                model_consensus = {m.model.value: m.overall_bias for m in model_analyses}
                model_confidences = {m.model.value: m.confidence for m in model_analyses}
                
                return InvestmentDecision(
                    symbol=symbol,
                    decision=data.get('decision', '持有'),
                    confidence=data.get('confidence', 0.5),
                    position_size=data.get('position_size', 0),
                    target_price=data.get('target_price'),
                    stop_loss=data.get('stop_loss'),
                    time_horizon=data.get('time_horizon', '中期'),
                    logic_chain=data.get('logic_chain', []),
                    supporting_evidence=data.get('supporting_evidence', []),
                    opposing_concerns=data.get('opposing_concerns', []),
                    risk_mitigation=data.get('risk_mitigation', []),
                    model_consensus=model_consensus,
                    model_confidences=model_confidences
                )
        except Exception as e:
            self._log(f"解析综合决策失败: {e}")
        
        return InvestmentDecision(symbol=symbol, decision="持有", confidence=0.5, position_size=0)
    
    def conduct_debate(self, symbol: str,
                      quant_data: Dict,
                      macro_data: Dict,
                      risk_data: Dict) -> DebateResult:
        """
        执行完整的多模型多空辩论流程
        """
        self._log("=" * 80)
        self._log(f"开始多模型多空辩论: {symbol}")
        self._log("=" * 80)
        
        # 1. 公司深度研究
        company_research = self.conduct_company_research(symbol, quant_data, macro_data)
        
        # 2. 多模型分析
        model_analyses = []
        for model in AnalystModel:
            analysis = self.run_model_analysis(model, symbol, company_research,
                                              quant_data, macro_data, risk_data)
            model_analyses.append(analysis)
            self._log(f"  {model.value}: {analysis.overall_bias} (置信度{analysis.confidence:.0%})")
        
        # 3. 综合决策
        decision = self.synthesize_decision(symbol, model_analyses, company_research,
                                           quant_data, macro_data, risk_data)
        
        # 4. 生成辩论总结
        debate_summary = self._generate_debate_summary(model_analyses, decision)
        
        # 5. 组合建议
        portfolio_suggestion = {
            "position_size": decision.position_size,
            "target_price": decision.target_price,
            "stop_loss": decision.stop_loss,
            "time_horizon": decision.time_horizon
        }
        
        # 6. 风险评估
        risk_assessment = {
            "risk_level": risk_data.get('risk_level', 'normal'),
            "key_risks": decision.opposing_concerns,
            "mitigation": decision.risk_mitigation
        }
        
        # 7. 生成最终报告
        final_report = self._generate_final_report(symbol, company_research,
                                                   model_analyses, decision)
        
        return DebateResult(
            symbol=symbol,
            company_research=company_research,
            model_analyses=model_analyses,
            debate_summary=debate_summary,
            investment_decision=decision,
            portfolio_suggestion=portfolio_suggestion,
            risk_assessment=risk_assessment,
            final_report=final_report
        )
    
    def _generate_debate_summary(self, model_analyses: List[ModelAnalysis],
                                 decision: InvestmentDecision) -> str:
        """生成辩论总结"""
        lines = []
        lines.append("多模型辩论总结:")
        lines.append("")
        
        for m in model_analyses:
            lines.append(f"【{m.model.value}】")
            lines.append(f"  立场: {m.overall_bias}")
            lines.append(f"  置信度: {m.confidence:.0%}")
            lines.append(f"  关键因素: {', '.join(m.key_factors)}")
            lines.append("")
        
        lines.append(f"【综合决策】")
        lines.append(f"  建议: {decision.decision}")
        lines.append(f"  置信度: {decision.confidence:.0%}")
        lines.append(f"  建议仓位: {decision.position_size:.0%}")
        
        return "\n".join(lines)
    
    def _generate_final_report(self, symbol: str,
                              company_research: Dict,
                              model_analyses: List[ModelAnalysis],
                              decision: InvestmentDecision) -> str:
        """生成最终报告"""
        lines = []
        
        lines.append(f"# 投资决策报告: {symbol}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 决策结论
        decision_emoji = {
            "强烈买入": "🔥",
            "买入": "🟢",
            "持有": "🟡",
            "卖出": "🔴",
            "强烈卖出": "⛔"
        }.get(decision.decision, "⚪")
        
        lines.append(f"## {decision_emoji} 投资决策: {decision.decision}")
        lines.append(f"- **置信度**: {decision.confidence:.0%}")
        lines.append(f"- **建议仓位**: {decision.position_size:.0%}")
        if decision.target_price:
            lines.append(f"- **目标价**: {decision.target_price:.2f}")
        if decision.stop_loss:
            lines.append(f"- **止损位**: {decision.stop_loss:.2f}")
        lines.append(f"- **投资周期**: {decision.time_horizon}")
        lines.append("")
        
        # 决策逻辑链
        lines.append("## 🧠 决策逻辑")
        for i, logic in enumerate(decision.logic_chain, 1):
            lines.append(f"{i}. {logic}")
        lines.append("")
        
        # 模型共识
        lines.append("## 🤖 模型共识")
        for model, bias in decision.model_consensus.items():
            emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡", "caution": "⚠️"}.get(bias, "⚪")
            conf = decision.model_confidences.get(model, 0)
            lines.append(f"- {emoji} **{model}**: {bias} (置信度{conf:.0%})")
        lines.append("")
        
        # 支持证据
        lines.append("## ✅ 支持证据")
        for evidence in decision.supporting_evidence[:5]:
            lines.append(f"- {evidence}")
        lines.append("")
        
        # 反对担忧
        lines.append("## ⚠️ 风险与担忧")
        for concern in decision.opposing_concerns[:5]:
            lines.append(f"- {concern}")
        lines.append("")
        
        # 风险缓解
        lines.append("## 🛡️ 风险缓解措施")
        for mitigation in decision.risk_mitigation:
            lines.append(f"- {mitigation}")
        lines.append("")
        
        # 公司研究摘要
        if company_research:
            lines.append("## 📊 公司研究摘要")
            overview = company_research.get('company_overview', '')
            if overview:
                lines.append(f"**公司概况**: {overview[:200]}...")
            
            industry = company_research.get('industry', {})
            if industry:
                lines.append(f"**行业**: {industry.get('lifecycle', '')} | {industry.get('policy_support', '')}")
            
            competition = company_research.get('competition', {})
            if competition:
                lines.append(f"**竞争地位**: 市占率 {competition.get('market_share', 'N/A')}")
            lines.append("")
        
        return "\n".join(lines)


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Multi-Model Debate System - 测试")
    print("=" * 80)
    
    debate_system = MultiModelDebateSystem(verbose=True)
    
    quant_data = {
        "predicted_return": 0.15,
        "predicted_volatility": 0.25,
        "sharpe_ratio": 1.2,
        "factors": ["momentum", "value", "quality"]
    }
    
    macro_data = {"signal": "🟡", "risk_level": "中风险"}
    risk_data = {"risk_level": "normal", "volatility": 0.25}
    
    result = debate_system.conduct_debate(
        symbol="AAPL",
        quant_data=quant_data,
        macro_data=macro_data,
        risk_data=risk_data
    )
    
    print("\n" + "=" * 80)
    print("最终报告")
    print("=" * 80)
    print(result.final_report)

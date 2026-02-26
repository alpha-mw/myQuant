#!/usr/bin/env python3
"""
Decision Layer - 决策层 (第6层)

功能:
1. 多Agent架构 - 5个专业分析师Agent
2. LLM多空辩论 - 对公司、市场、经济、产品多维度分析
3. 综合量化分析(1-3层) + 宏观分析(4层) + LLM深度分析
4. 生成具体投资建议 - 组合配置、个股买卖建议

Agents:
- 财务分析师: 分析财务报表、盈利能力、估值
- 行业专家: 分析行业趋势、竞争格局、护城河
- 宏观经济学家: 分析经济周期、政策影响
- 技术分析师: 分析价格走势、技术指标
- 风险管理师: 评估风险、提出风控建议
"""

import os
import sys
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

# 尝试导入OpenAI

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AgentRole(Enum):
    """Agent角色"""
    FINANCIAL_ANALYST = "财务分析师"
    INDUSTRY_EXPERT = "行业专家"
    MACRO_ECONOMIST = "宏观经济学家"
    TECHNICAL_ANALYST = "技术分析师"
    RISK_MANAGER = "风险管理师"


@dataclass
class AgentOpinion:
    """Agent观点"""
    role: AgentRole
    bullish_points: List[str] = field(default_factory=list)
    bearish_points: List[str] = field(default_factory=list)
    confidence: float = 0.5
    recommendation: str = ""
    reasoning: str = ""


@dataclass
class StockRecommendation:
    """个股推荐"""
    symbol: str
    name: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    agent_consensus: Dict[str, str] = field(default_factory=dict)


@dataclass
class PortfolioRecommendation:
    """组合推荐"""
    total_positions: Dict[str, float]
    cash_ratio: float
    sector_allocation: Dict[str, float]
    risk_level: str
    expected_return: float
    expected_volatility: float
    rebalancing_suggestions: List[str] = field(default_factory=list)


@dataclass
class DecisionLayerResult:
    """决策层结果"""
    agent_opinions: List[AgentOpinion] = field(default_factory=list)
    debate_summary: str = ""
    stock_recommendations: List[StockRecommendation] = field(default_factory=list)
    portfolio_recommendation: Optional[PortfolioRecommendation] = None
    market_outlook: str = ""
    risk_warnings: List[str] = field(default_factory=list)
    final_report: str = ""


class LLMClient:
    """LLM客户端"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def chat(self, messages: List[Dict[str, str]], 
             model: str = "gpt-4",
             temperature: float = 0.7) -> str:
        """调用LLM"""
        if not self.client:
            return self._mock_response(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLMClient] LLM调用失败: {e}")
            return self._mock_response(messages)
    
    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        """模拟LLM响应（当API不可用时）"""
        # 提取用户消息
        user_msg = ""
        for msg in messages:
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '')
                break
        
        # 根据角色返回模拟响应
        if "财务分析师" in user_msg:
            return self._mock_financial_response()
        elif "行业专家" in user_msg:
            return self._mock_industry_response()
        elif "宏观经济学家" in user_msg:
            return self._mock_macro_response()
        elif "技术分析师" in user_msg:
            return self._mock_technical_response()
        elif "风险管理师" in user_msg:
            return self._mock_risk_response()
        elif "综合投资建议" in user_msg:
            return self._mock_final_response()
        
        return "基于当前分析，建议谨慎投资。"
    
    def _mock_financial_response(self) -> str:
        return """{
            "bullish_points": ["ROE稳定在15%以上", "现金流充裕", "估值合理PE<20"],
            "bearish_points": ["毛利率略有下滑", "应收账款增加"],
            "confidence": 0.75,
            "recommendation": "BUY",
            "reasoning": "财务指标整体健康，盈利能力稳定，估值有吸引力"
        }"""
    
    def _mock_industry_response(self) -> str:
        return """{
            "bullish_points": ["行业处于成长期", "市场份额领先", "技术壁垒高"],
            "bearish_points": ["竞争加剧", "新进入者威胁"],
            "confidence": 0.70,
            "recommendation": "BUY",
            "reasoning": "行业前景良好，公司具有护城河优势"
        }"""
    
    def _mock_macro_response(self) -> str:
        return """{
            "bullish_points": ["货币政策宽松", "经济复苏", "行业政策支持"],
            "bearish_points": ["通胀压力", "地缘政治风险"],
            "confidence": 0.65,
            "recommendation": "HOLD",
            "reasoning": "宏观环境中性偏正面，但需关注通胀风险"
        }"""
    
    def _mock_technical_response(self) -> str:
        return """{
            "bullish_points": ["突破关键阻力位", "成交量放大", "MACD金叉"],
            "bearish_points": ["RSI超买", "接近前期高点"],
            "confidence": 0.60,
            "recommendation": "HOLD",
            "reasoning": "技术面偏强，但短期可能回调"
        }"""
    
    def _mock_risk_response(self) -> str:
        return """{
            "bullish_points": ["波动率可控", "流动性充足"],
            "bearish_points": ["Beta较高", "集中度风险"],
            "confidence": 0.55,
            "recommendation": "CAUTION",
            "reasoning": "风险收益比合理，但需控制仓位"
        }"""
    
    def _mock_final_response(self) -> str:
        return """{
            "market_outlook": "谨慎乐观",
            "portfolio_allocation": {
                "AAPL": 0.25,
                "MSFT": 0.25,
                "GOOGL": 0.20,
                "NVDA": 0.15,
                "CASH": 0.15
            },
            "stock_recommendations": [
                {"symbol": "AAPL", "action": "BUY", "confidence": 0.80, "reasoning": "财务健康+行业龙头"},
                {"symbol": "MSFT", "action": "BUY", "confidence": 0.75, "reasoning": "云计算增长+AI布局"},
                {"symbol": "GOOGL", "action": "HOLD", "confidence": 0.65, "reasoning": "估值合理但增长放缓"}
            ],
            "risk_warnings": ["关注美联储政策", "控制单一股票仓位<20%"],
            "expected_return": 0.15,
            "expected_volatility": 0.20
        }"""


class DecisionLayer:
    """
    决策层 - LLM多Agent多空辩论
    """
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        self.llm = LLMClient(api_key)
        self.verbose = verbose
        self.result = DecisionLayerResult()
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[DecisionLayer] {msg}")
    
    def _create_agent_prompt(self, role: AgentRole, symbol: str, 
                            quant_data: Dict, macro_data: Dict) -> str:
        """创建Agent提示词"""
        
        base_prompt = f"""你是一位专业的{role.value}，需要对股票 {symbol} 进行深入分析。

【量化分析数据】
- 预测收益: {quant_data.get('predicted_return', 'N/A')}
- 预测波动率: {quant_data.get('predicted_volatility', 'N/A')}
- 夏普比率: {quant_data.get('sharpe_ratio', 'N/A')}
- 主要因子: {', '.join(quant_data.get('factors', []))}

【宏观环境】
- 宏观信号: {macro_data.get('signal', 'N/A')}
- 风险等级: {macro_data.get('risk_level', 'N/A')}

请从{role.value}的专业角度，分析该股票的多空因素。

请以JSON格式输出：
{{
    "bullish_points": ["利多因素1", "利多因素2", ...],
    "bearish_points": ["利空因素1", "利空因素2", ...],
    "confidence": 0.0-1.0,
    "recommendation": "BUY/SELL/HOLD/CAUTION",
    "reasoning": "详细分析理由"
}}"""
        
        # 根据角色添加特定提示
        if role == AgentRole.FINANCIAL_ANALYST:
            base_prompt += """

重点关注：
- 财务报表健康度（ROE、ROA、毛利率）
- 估值水平（PE、PB、PS）
- 现金流状况
- 盈利质量
"""
        elif role == AgentRole.INDUSTRY_EXPERT:
            base_prompt += """

重点关注：
- 行业生命周期（成长/成熟/衰退）
- 竞争格局和市场份额
- 护城河（品牌、技术、成本、网络效应）
- 行业政策影响
"""
        elif role == AgentRole.MACRO_ECONOMIST:
            base_prompt += """

重点关注：
- 经济周期位置
- 货币政策和利率环境
- 通胀影响
- 汇率风险（对跨国公司）
"""
        elif role == AgentRole.TECHNICAL_ANALYST:
            base_prompt += """

重点关注：
- 价格趋势（动量、反转）
- 支撑阻力位
- 成交量变化
- 技术指标信号（RSI、MACD、布林带）
"""
        elif role == AgentRole.RISK_MANAGER:
            base_prompt += """

重点关注：
- 波动率和最大回撤
- Beta和系统性风险
- 流动性风险
- 集中度风险
- 尾部风险
"""
        
        return base_prompt
    
    def _parse_agent_response(self, response: str, role: AgentRole) -> AgentOpinion:
        """解析Agent响应"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                return AgentOpinion(
                    role=role,
                    bullish_points=data.get('bullish_points', []),
                    bearish_points=data.get('bearish_points', []),
                    confidence=data.get('confidence', 0.5),
                    recommendation=data.get('recommendation', 'HOLD'),
                    reasoning=data.get('reasoning', '')
                )
        except Exception as e:
            self._log(f"解析Agent响应失败: {e}")
        
        # 返回默认观点
        return AgentOpinion(role=role)
    
    def run_agent_analysis(self, symbol: str, quant_data: Dict, 
                          macro_data: Dict) -> List[AgentOpinion]:
        """
        运行多Agent分析
        """
        self._log(f"开始多Agent分析: {symbol}")
        
        opinions = []
        
        for role in AgentRole:
            self._log(f"  运行 {role.value}...")
            
            prompt = self._create_agent_prompt(role, symbol, quant_data, macro_data)
            
            messages = [
                {"role": "system", "content": f"你是一位专业的{role.value}，擅长投资分析。请基于提供的数据给出客观分析。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.chat(messages)
            opinion = self._parse_agent_response(response, role)
            opinions.append(opinion)
            
            self._log(f"    {role.value}: {opinion.recommendation} (置信度{opinion.confidence:.0%})")
        
        return opinions
    
    def debate_and_consensus(self, opinions: List[AgentOpinion], 
                            symbol: str) -> StockRecommendation:
        """
        Agent辩论并达成共识
        """
        self._log(f"Agent辩论: {symbol}")
        
        # 统计观点
        recommendations = [op.recommendation for op in opinions]
        buy_count = recommendations.count('BUY')
        sell_count = recommendations.count('SELL')
        hold_count = recommendations.count('HOLD')
        
        # 计算加权置信度
        total_confidence = sum(op.confidence for op in opinions)
        avg_confidence = total_confidence / len(opinions) if opinions else 0
        
        # 确定最终建议
        if buy_count >= 3:
            final_action = "BUY"
        elif sell_count >= 3:
            final_action = "SELL"
        else:
            final_action = "HOLD"
        
        # 生成理由
        all_bullish = []
        all_bearish = []
        for op in opinions:
            all_bullish.extend(op.bullish_points)
            all_bearish.extend(op.bearish_points)
        
        reasoning = f"看多因素({len(all_bullish)}条): " + "; ".join(all_bullish[:3])
        reasoning += f" | 看空因素({len(all_bearish)}条): " + "; ".join(all_bearish[:3])
        
        # Agent共识记录
        consensus = {op.role.value: op.recommendation for op in opinions}
        
        return StockRecommendation(
            symbol=symbol,
            name=symbol,
            action=final_action,
            confidence=avg_confidence,
            reasoning=reasoning,
            agent_consensus=consensus
        )
    
    def generate_portfolio_recommendation(
        self,
        stock_recommendations: List[StockRecommendation],
        quant_results: Dict,
        macro_signal: str,
        risk_result: Dict
    ) -> PortfolioRecommendation:
        """
        生成组合推荐
        """
        self._log("生成组合推荐...")
        
        # 基于宏观信号确定基础仓位
        base_allocation = {
            "🔴": 0.3,
            "🟡": 0.5,
            "🟢": 0.8,
            "🔵": 1.0
        }.get(macro_signal, 0.5)
        
        # 筛选BUY推荐
        buy_stocks = [s for s in stock_recommendations if s.action == "BUY"]
        
        # 等权重分配
        positions = {}
        if buy_stocks:
            weight_per_stock = base_allocation / len(buy_stocks)
            for stock in buy_stocks:
                positions[stock.symbol] = min(weight_per_stock, 0.2)  # 单票不超过20%
        
        # 重新归一化
        total = sum(positions.values())
        if total > 0:
            positions = {k: v/total * base_allocation for k, v in positions.items()}
        
        cash_ratio = 1 - sum(positions.values())
        
        return PortfolioRecommendation(
            total_positions=positions,
            cash_ratio=cash_ratio,
            sector_allocation={},  # 简化
            risk_level=risk_result.get('risk_level', 'normal'),
            expected_return=quant_results.get('expected_return', 0.1),
            expected_volatility=quant_results.get('expected_volatility', 0.2),
            rebalancing_suggestions=["定期再平衡", "关注风险信号变化"]
        )
    
    def run_decision_process(
        self,
        symbols: List[str],
        quant_data: Dict[str, Dict],
        macro_data: Dict,
        risk_data: Dict
    ) -> DecisionLayerResult:
        """
        运行完整决策流程
        """
        self._log("=" * 80)
        self._log("【第6层】决策层 - LLM多Agent多空辩论")
        self._log("=" * 80)
        
        result = DecisionLayerResult()
        
        # 1. 多Agent分析每只股票
        stock_recommendations = []
        
        for symbol in symbols:
            self._log(f"\n分析股票: {symbol}")
            
            # 获取该股票的量化数据
            symbol_quant = quant_data.get(symbol, {})
            
            # 运行Agent分析
            opinions = self.run_agent_analysis(symbol, symbol_quant, macro_data)
            result.agent_opinions.extend(opinions)
            
            # Agent辩论达成共识
            stock_rec = self.debate_and_consensus(opinions, symbol)
            stock_recommendations.append(stock_rec)
            
            self._log(f"最终建议: {stock_rec.action} (置信度{stock_rec.confidence:.0%})")
        
        result.stock_recommendations = stock_recommendations
        
        # 2. 生成组合推荐
        result.portfolio_recommendation = self.generate_portfolio_recommendation(
            stock_recommendations,
            quant_data,
            macro_data.get('signal', '🟡'),
            risk_data
        )
        
        # 3. 生成市场展望
        buy_count = sum(1 for s in stock_recommendations if s.action == "BUY")
        sell_count = sum(1 for s in stock_recommendations if s.action == "SELL")
        
        if buy_count > sell_count:
            result.market_outlook = "结构性机会，精选个股"
        elif sell_count > buy_count:
            result.market_outlook = "防御为主，降低仓位"
        else:
            result.market_outlook = "震荡市，均衡配置"
        
        # 4. 生成最终报告
        result.final_report = self._generate_final_report(result)
        
        self._log("\n决策层完成")
        
        return result
    
    def _generate_final_report(self, result: DecisionLayerResult) -> str:
        """生成最终报告"""
        lines = []
        
        lines.append("# 投资决策报告")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 市场展望
        lines.append("## 📊 市场展望")
        lines.append(result.market_outlook)
        lines.append("")
        
        # 个股推荐
        lines.append("## 📈 个股推荐")
        lines.append("")
        lines.append("| 股票 | 建议 | 置信度 | 理由 |")
        lines.append("|:---|:---:|:---:|:---|")
        for rec in result.stock_recommendations:
            action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(rec.action, "⚪")
            lines.append(f"| {rec.symbol} | {action_emoji} {rec.action} | {rec.confidence:.0%} | {rec.reasoning[:50]}... |")
        lines.append("")
        
        # 组合配置
        if result.portfolio_recommendation:
            lines.append("## 💼 组合配置")
            lines.append("")
            lines.append("**持仓建议**:")
            for symbol, weight in sorted(
                result.portfolio_recommendation.total_positions.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                lines.append(f"- {symbol}: {weight:.1%}")
            lines.append(f"- 现金: {result.portfolio_recommendation.cash_ratio:.1%}")
            lines.append("")
            lines.append(f"**预期收益**: {result.portfolio_recommendation.expected_return:.1%}")
            lines.append(f"**预期波动**: {result.portfolio_recommendation.expected_volatility:.1%}")
            lines.append("")
        
        # Agent共识
        lines.append("## 🤖 Agent共识")
        lines.append("")
        for rec in result.stock_recommendations[:3]:
            lines.append(f"**{rec.symbol}**:")
            for agent, action in rec.agent_consensus.items():
                lines.append(f"  - {agent}: {action}")
            lines.append("")
        
        return "\n".join(lines)


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 80)
    print("Decision Layer - 测试")
    print("=" * 80)
    
    # 创建决策层
    decision_layer = DecisionLayer(verbose=True)
    
    # 测试数据
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    quant_data = {
        "AAPL": {"predicted_return": 0.15, "predicted_volatility": 0.25, "sharpe_ratio": 1.2, "factors": ["momentum", "value"]},
        "MSFT": {"predicted_return": 0.12, "predicted_volatility": 0.22, "sharpe_ratio": 1.1, "factors": ["quality", "growth"]},
        "GOOGL": {"predicted_return": 0.08, "predicted_volatility": 0.28, "sharpe_ratio": 0.8, "factors": ["value"]}
    }
    
    macro_data = {"signal": "🟡", "risk_level": "中风险"}
    risk_data = {"risk_level": "normal"}
    
    # 运行决策流程
    result = decision_layer.run_decision_process(
        symbols=symbols,
        quant_data=quant_data,
        macro_data=macro_data,
        risk_data=risk_data
    )
    
    print("\n" + "=" * 80)
    print("最终报告")
    print("=" * 80)
    print(result.final_report)

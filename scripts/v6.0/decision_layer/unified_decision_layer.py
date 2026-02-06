#!/usr/bin/env python3
"""
Quant-Investor V6.0 - 统一决策层 (Unified Decision Layer)

整合所有历史版本的定性分析能力：
- V2.9: 多Agent辩论系统 (财务/行业/护城河/估值/风险 5大专家)
- V3.6: 多LLM适配器 (OpenAI/Gemini/DeepSeek/Qwen/Kimi)
- V4.0: 定性分析与估值 (DCF/反向DCF/可比公司)

设计原则：
1. 自动检测可用LLM并选择最优
2. 多Agent独立分析 → 交叉质询 → 综合结论
3. 结构化输出，便于与量化信号融合
4. 安全的API密钥管理
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum


# ==================== LLM适配器 (源自V3.6) ====================

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    KIMI = "kimi"


PROVIDER_CONFIGS = {
    LLMProvider.OPENAI: {
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "models": ["gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini"]
    },
    LLMProvider.GEMINI: {
        "base_url": None,
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    },
    LLMProvider.DEEPSEEK: {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"]
    },
    LLMProvider.QWEN: {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen3-max"]
    },
    LLMProvider.KIMI: {
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-8k",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]
    }
}


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    latency_ms: float = 0.0
    tokens_used: int = 0


class LLMAdapter:
    """
    统一LLM适配器 (源自V3.6，增强版)
    
    自动检测可用的LLM提供商，提供统一的对话接口。
    """
    
    def __init__(self, preferred_providers: List[str] = None, verbose: bool = True):
        self.verbose = verbose
        self.preferred_providers = preferred_providers or ["gemini", "openai", "deepseek", "qwen"]
        self._load_credentials()
        self._available_adapters = {}
        self._init_adapters()
    
    def _load_credentials(self):
        """从安全存储加载API密钥"""
        cred_path = os.path.expanduser("~/.quant_investor/credentials.env")
        if os.path.exists(cred_path):
            with open(cred_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key, value = key.strip(), value.strip()
                        if value and not value.startswith('your_') and '...' not in value:
                            os.environ[key] = value
    
    def _init_adapters(self):
        """初始化所有可用的LLM适配器"""
        for provider_name in self.preferred_providers:
            try:
                provider = LLMProvider(provider_name.lower())
                config = PROVIDER_CONFIGS[provider]
                api_key = os.getenv(config["env_key"], "")
                
                if not api_key or len(api_key) < 10:
                    continue
                
                if provider == LLMProvider.GEMINI:
                    self._init_gemini(provider, config, api_key)
                else:
                    self._init_openai_compatible(provider, config, api_key)
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ {provider_name} 初始化失败: {e}")
        
        if self.verbose:
            print(f"  ✅ 可用LLM: {list(self._available_adapters.keys())}")
    
    def _init_gemini(self, provider, config, api_key):
        """初始化Gemini适配器"""
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            self._available_adapters[provider.value] = {
                'type': 'gemini', 'client': client,
                'model': config['default_model']
            }
            if self.verbose:
                print(f"  ✅ Gemini 初始化成功")
        except ImportError:
            pass
    
    def _init_openai_compatible(self, provider, config, api_key):
        """初始化OpenAI兼容适配器"""
        try:
            from openai import OpenAI
            
            base_url = config.get("base_url")
            if provider == LLMProvider.OPENAI:
                base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            self._available_adapters[provider.value] = {
                'type': 'openai', 'client': client,
                'model': config['default_model']
            }
            if self.verbose:
                print(f"  ✅ {provider.value} 初始化成功")
        except ImportError:
            pass
    
    def chat(self, messages: List[Dict[str, str]], provider: str = None,
             temperature: float = 0.3, max_tokens: int = 4000) -> LLMResponse:
        """
        发送聊天请求
        
        自动选择可用的LLM，支持故障转移。
        """
        providers_to_try = []
        if provider and provider in self._available_adapters:
            providers_to_try.append(provider)
        providers_to_try.extend([p for p in self._available_adapters if p not in providers_to_try])
        
        for prov in providers_to_try:
            try:
                adapter = self._available_adapters[prov]
                start = time.time()
                
                if adapter['type'] == 'gemini':
                    response = self._call_gemini(adapter, messages, temperature, max_tokens)
                else:
                    response = self._call_openai(adapter, messages, temperature, max_tokens)
                
                latency = (time.time() - start) * 1000
                return LLMResponse(
                    content=response, model=adapter['model'],
                    provider=prov, latency_ms=latency
                )
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️ {prov} 调用失败: {e}")
                continue
        
        # 所有LLM都失败，返回模拟响应
        return LLMResponse(
            content=self._get_mock_response(messages),
            model="mock", provider="mock", latency_ms=0
        )
    
    def _call_gemini(self, adapter, messages, temperature, max_tokens):
        """调用Gemini"""
        from google.genai import types
        
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(
                    role="user", parts=[types.Part(text=msg["content"])]
                ))
            elif msg["role"] == "assistant":
                contents.append(types.Content(
                    role="model", parts=[types.Part(text=msg["content"])]
                ))
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction
        )
        
        response = adapter['client'].models.generate_content(
            model=adapter['model'], contents=contents, config=config
        )
        return response.text
    
    def _call_openai(self, adapter, messages, temperature, max_tokens):
        """调用OpenAI兼容API"""
        response = adapter['client'].chat.completions.create(
            model=adapter['model'],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _get_mock_response(self, messages):
        """模拟响应（所有LLM不可用时）"""
        return json.dumps({
            "summary": "LLM服务暂时不可用，返回模拟分析结果",
            "score": 6.0, "confidence": 0.5,
            "conclusion": "建议在LLM服务恢复后重新运行分析",
            "key_findings": [{"finding": "需要LLM服务支持深度分析"}],
            "risks": ["LLM服务不可用"], "opportunities": ["待分析"]
        }, ensure_ascii=False)
    
    def get_available_providers(self) -> List[str]:
        return list(self._available_adapters.keys())
    
    def is_available(self) -> bool:
        return len(self._available_adapters) > 0


# ==================== 分析Agent (源自V2.9) ====================

@dataclass
class AgentAnalysis:
    """Agent分析结果"""
    agent_name: str
    agent_role: str
    score: float = 5.0
    confidence: float = 0.5
    summary: str = ""
    conclusion: str = ""
    key_findings: List[Dict] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    detailed_analysis: Dict = field(default_factory=dict)


class AnalysisAgent:
    """通用分析Agent"""
    
    def __init__(self, name: str, role: str, system_prompt: str, llm: LLMAdapter):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm
    
    def analyze(self, stock_code: str, company_name: str, 
                data_summary: str, quant_summary: str = "") -> AgentAnalysis:
        """执行分析"""
        user_prompt = f"""请对以下股票进行{self.role}分析：

## 股票信息
- 代码: {stock_code}
- 名称: {company_name}

## 数据摘要
{data_summary}

## 量化分析结果
{quant_summary}

请以JSON格式返回分析结果，包含以下字段：
- summary: 一句话总结
- score: 评分 (1-10, 10为最佳)
- confidence: 置信度 (0-1)
- conclusion: 详细结论 (2-3段)
- key_findings: 关键发现列表 [{{"finding": "...", "evidence": "...", "impact": "高/中/低"}}]
- risks: 风险列表
- opportunities: 机会列表
"""
        
        response = self.llm.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        return self._parse_response(response.content)
    
    def rebut(self, questions: List[str], original_analysis: AgentAnalysis) -> Dict:
        """回应质询"""
        prompt = f"""你之前对该股票的分析结论是：
{original_analysis.conclusion}
评分: {original_analysis.score}/10

现在主持人提出了以下质疑：
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(questions))}

请认真回应每个问题，如有必要可以调整你的观点。
以JSON格式返回：
- responses: 对每个问题的回应列表
- stance_changed: 是否调整观点 (true/false)
- new_score: 调整后的评分 (如果调整了)
"""
        
        response = self.llm.chat([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ])
        
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(response.content)
        except:
            return {"responses": [response.content], "stance_changed": False}
    
    def _parse_response(self, content: str) -> AgentAnalysis:
        """解析LLM响应"""
        import re
        
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(content)
        except:
            parsed = {
                "summary": content[:200],
                "score": 6.0,
                "confidence": 0.5,
                "conclusion": content[:1000],
                "key_findings": [],
                "risks": [],
                "opportunities": []
            }
        
        return AgentAnalysis(
            agent_name=self.name,
            agent_role=self.role,
            score=float(parsed.get("score", 6.0)),
            confidence=float(parsed.get("confidence", 0.5)),
            summary=parsed.get("summary", ""),
            conclusion=parsed.get("conclusion", ""),
            key_findings=parsed.get("key_findings", []),
            risks=parsed.get("risks", []),
            opportunities=parsed.get("opportunities", []),
            detailed_analysis=parsed
        )


# ==================== 辩论引擎 (源自V2.9) ====================

@dataclass
class DebateResult:
    """辩论结果"""
    stock_code: str
    company_name: str
    
    # 各Agent分析
    agent_analyses: Dict[str, AgentAnalysis] = field(default_factory=dict)
    
    # 辩论轮次记录
    debate_rounds: List[Dict] = field(default_factory=list)
    
    # 最终结论
    final_score: float = 5.0
    final_confidence: float = 0.5
    investment_rating: str = "持有"
    consensus: str = ""
    bull_case: str = ""
    bear_case: str = ""
    
    # 估值
    valuation_summary: str = ""
    
    # 元信息
    duration_seconds: float = 0.0
    llm_providers_used: List[str] = field(default_factory=list)


class DebateEngine:
    """
    多Agent辩论引擎 (源自V2.9，增强版)
    
    流程：
    1. 各专家Agent独立分析
    2. 主持人组织交叉质询
    3. 综合形成最终结论
    """
    
    def __init__(self, llm: LLMAdapter, max_rounds: int = 1, verbose: bool = True):
        self.llm = llm
        self.max_rounds = max_rounds
        self.verbose = verbose
        
        # 创建5大专家Agent
        self.agents = self._create_agents()
    
    def _create_agents(self) -> List[AnalysisAgent]:
        """创建专家Agent团队"""
        agents = [
            AnalysisAgent("财务分析师", "财务分析", 
                """你是一位资深的财务分析师，专注于：
                - 财务报表分析（三表联动）
                - 盈利能力（ROE、毛利率、净利率）
                - 成长性（营收增速、利润增速）
                - 现金流质量
                - 资产负债结构
                请基于数据给出客观、量化的分析。""", self.llm),
            
            AnalysisAgent("行业分析师", "行业分析",
                """你是一位资深的行业分析师，专注于：
                - 行业发展趋势和市场空间
                - 竞争格局和市场份额
                - 行业政策和监管环境
                - 技术变革和创新驱动
                - 上下游产业链分析
                请结合行业数据给出深入分析。""", self.llm),
            
            AnalysisAgent("护城河分析师", "护城河分析",
                """你是一位专注于企业竞争优势的分析师，专注于：
                - 品牌价值和客户粘性
                - 网络效应和规模经济
                - 技术壁垒和专利保护
                - 转换成本和锁定效应
                - 成本优势和资源垄断
                请评估企业护城河的宽度和持久性。""", self.llm),
            
            AnalysisAgent("估值分析师", "估值分析",
                """你是一位专业的估值分析师，专注于：
                - DCF估值（自由现金流折现）
                - 反向DCF（市场隐含的增长预期）
                - 可比公司估值（PE/PB/PS/EV-EBITDA）
                - 历史估值区间分析
                - 安全边际评估
                请给出具体的估值区间和安全边际。""", self.llm),
            
            AnalysisAgent("风险分析师", "风险分析",
                """你是一位专注于风险管理的分析师，专注于：
                - 系统性风险（宏观经济、政策）
                - 经营风险（管理层、战略）
                - 财务风险（杠杆、流动性）
                - 市场风险（估值泡沫、流动性）
                - 黑天鹅事件和尾部风险
                请全面评估各类风险及其概率和影响。""", self.llm),
        ]
        return agents
    
    def run_debate(self, stock_code: str, company_name: str,
                    data_summary: str, quant_summary: str = "") -> DebateResult:
        """
        运行完整的辩论流程
        
        Args:
            stock_code: 股票代码
            company_name: 公司名称
            data_summary: 数据摘要
            quant_summary: 量化分析摘要
        
        Returns:
            DebateResult: 辩论结果
        """
        start_time = time.time()
        result = DebateResult(stock_code=stock_code, company_name=company_name)
        
        if self.verbose:
            print(f"\n  🎯 开始辩论: {company_name} ({stock_code})")
        
        # Round 0: 各Agent独立分析
        if self.verbose:
            print(f"    Round 0: 各专家独立分析")
        
        for agent in self.agents:
            if self.verbose:
                print(f"      [{agent.name}] 分析中...")
            
            try:
                analysis = agent.analyze(stock_code, company_name, data_summary, quant_summary)
                result.agent_analyses[agent.name] = analysis
                
                if self.verbose:
                    print(f"      [{agent.name}] 评分: {analysis.score}/10, 置信度: {analysis.confidence:.0%}")
            except Exception as e:
                if self.verbose:
                    print(f"      [{agent.name}] 分析失败: {e}")
                result.agent_analyses[agent.name] = AgentAnalysis(
                    agent_name=agent.name, agent_role=agent.role,
                    summary=f"分析失败: {e}"
                )
        
        # Round 1-N: 交叉质询
        for round_num in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"    Round {round_num}: 交叉质询")
            
            questions = self._generate_questions(result.agent_analyses)
            rebuttals = {}
            
            for agent in self.agents:
                if agent.name in questions and questions[agent.name]:
                    try:
                        original = result.agent_analyses.get(agent.name)
                        if original:
                            rebuttal = agent.rebut(questions[agent.name], original)
                            rebuttals[agent.name] = rebuttal
                            
                            # 更新评分
                            if rebuttal.get('stance_changed') and rebuttal.get('new_score'):
                                result.agent_analyses[agent.name].score = float(rebuttal['new_score'])
                    except Exception:
                        pass
            
            result.debate_rounds.append({
                'round': round_num,
                'questions': questions,
                'rebuttals': rebuttals
            })
        
        # Final: 综合结论
        if self.verbose:
            print(f"    Final: 综合形成结论")
        
        self._synthesize_conclusion(result, data_summary)
        
        result.duration_seconds = time.time() - start_time
        result.llm_providers_used = self.llm.get_available_providers()
        
        if self.verbose:
            print(f"  ✅ 辩论完成: {result.investment_rating}, "
                  f"评分: {result.final_score:.1f}/10, "
                  f"耗时: {result.duration_seconds:.1f}s")
        
        return result
    
    def _generate_questions(self, analyses: Dict[str, AgentAnalysis]) -> Dict[str, List[str]]:
        """生成交叉质询问题"""
        questions = {}
        
        # 找出分歧最大的观点
        scores = {name: a.score for name, a in analyses.items()}
        if not scores:
            return questions
        
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        for name, analysis in analyses.items():
            qs = []
            if analysis.score == max_score and max_score - min_score > 2:
                qs.append(f"你的评分({analysis.score})是最高的，是否过于乐观？请提供更多证据支持。")
            elif analysis.score == min_score and max_score - min_score > 2:
                qs.append(f"你的评分({analysis.score})是最低的，是否过于悲观？其他分析师看到了哪些你忽略的积极因素？")
            
            if analysis.risks:
                qs.append(f"你提到的风险'{analysis.risks[0]}'，有多大概率发生？影响程度如何？")
            
            if qs:
                questions[name] = qs
        
        return questions
    
    def _synthesize_conclusion(self, result: DebateResult, data_summary: str):
        """综合形成最终结论"""
        # 计算加权平均分
        scores = []
        for analysis in result.agent_analyses.values():
            scores.append(analysis.score * analysis.confidence)
        
        if scores:
            result.final_score = sum(scores) / sum(a.confidence for a in result.agent_analyses.values())
            result.final_confidence = sum(a.confidence for a in result.agent_analyses.values()) / len(result.agent_analyses)
        
        # 确定投资评级
        if result.final_score >= 8:
            result.investment_rating = "强烈买入"
        elif result.final_score >= 7:
            result.investment_rating = "买入"
        elif result.final_score >= 5:
            result.investment_rating = "持有"
        elif result.final_score >= 3:
            result.investment_rating = "减仓"
        else:
            result.investment_rating = "卖出"
        
        # 使用LLM生成综合结论
        analyses_text = ""
        for name, analysis in result.agent_analyses.items():
            analyses_text += f"\n### {name} (评分: {analysis.score}/10)\n{analysis.summary}\n"
            if analysis.risks:
                analyses_text += f"风险: {', '.join(analysis.risks[:3])}\n"
            if analysis.opportunities:
                analyses_text += f"机会: {', '.join(analysis.opportunities[:3])}\n"
        
        synthesis_prompt = f"""作为投资委员会主持人，请综合以下各专家的分析，形成最终结论：

{analyses_text}

请以JSON格式返回：
- consensus: 综合结论 (3-5句话)
- bull_case: 多方核心观点 (2-3句话)
- bear_case: 空方核心观点 (2-3句话)
- valuation_summary: 估值总结 (2-3句话)
"""
        
        try:
            response = self.llm.chat([
                {"role": "system", "content": "你是一位经验丰富的投资委员会主持人，擅长综合多方观点形成客观结论。"},
                {"role": "user", "content": synthesis_prompt}
            ])
            
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(response.content)
            
            result.consensus = parsed.get("consensus", "")
            result.bull_case = parsed.get("bull_case", "")
            result.bear_case = parsed.get("bear_case", "")
            result.valuation_summary = parsed.get("valuation_summary", "")
        except Exception:
            # 简单汇总
            result.consensus = f"综合{len(result.agent_analyses)}位专家分析，综合评分{result.final_score:.1f}/10，评级：{result.investment_rating}"
            result.bull_case = "; ".join(
                a.opportunities[0] if a.opportunities else "" 
                for a in result.agent_analyses.values()
            )[:500]
            result.bear_case = "; ".join(
                a.risks[0] if a.risks else "" 
                for a in result.agent_analyses.values()
            )[:500]


# ==================== 统一决策层 ====================

@dataclass
class DecisionLayerOutput:
    """决策层完整输出"""
    # 各股票的辩论结果
    debate_results: Dict[str, DebateResult] = field(default_factory=dict)
    
    # 最终推荐
    final_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # 统计
    stats: Dict[str, Any] = field(default_factory=dict)


class UnifiedDecisionLayer:
    """
    V6.0 统一决策层
    
    对候选股票进行深入的定性分析和多Agent辩论。
    """
    
    def __init__(self, llm_preference: List[str] = None, verbose: bool = True,
                  max_debate_rounds: int = 1):
        self.verbose = verbose
        self.max_debate_rounds = max_debate_rounds
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🧠 V6.0 统一决策层初始化")
            print(f"{'='*60}")
        
        # 初始化LLM适配器
        self.llm = LLMAdapter(preferred_providers=llm_preference, verbose=verbose)
        
        # 初始化辩论引擎
        self.debate_engine = DebateEngine(
            llm=self.llm, max_rounds=max_debate_rounds, verbose=verbose
        )
    
    def analyze(self, ranked_stocks: List[Dict], data_bundle=None,
                quant_summary: str = "", max_stocks: int = 5) -> DecisionLayerOutput:
        """
        对排名靠前的候选股票进行深度定性分析
        
        Args:
            ranked_stocks: 排序后的候选股票 (来自模型层)
            data_bundle: 数据包 (来自数据层)
            quant_summary: 量化分析摘要
            max_stocks: 最多分析的股票数量
        
        Returns:
            DecisionLayerOutput: 决策层完整输出
        """
        output = DecisionLayerOutput()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🧠 V6.0 统一决策层")
            print(f"{'='*60}")
        
        if not self.llm.is_available():
            if self.verbose:
                print(f"  ⚠️ 无可用LLM，跳过定性分析")
            output.final_recommendations = ranked_stocks[:max_stocks]
            return output
        
        # 对Top N股票进行辩论
        stocks_to_analyze = ranked_stocks[:max_stocks]
        
        for i, stock in enumerate(stocks_to_analyze, 1):
            code = stock.get('code', '')
            name = stock.get('name', code)
            
            if self.verbose:
                print(f"\n  {'─'*40}")
                print(f"  [{i}/{len(stocks_to_analyze)}] 分析: {name} ({code})")
                print(f"  {'─'*40}")
            
            # 构建数据摘要
            data_summary = self._build_data_summary(stock, data_bundle)
            
            # 运行辩论
            debate_result = self.debate_engine.run_debate(
                stock_code=code,
                company_name=name,
                data_summary=data_summary,
                quant_summary=quant_summary
            )
            
            output.debate_results[code] = debate_result
        
        # 生成最终推荐
        output.final_recommendations = self._generate_recommendations(
            ranked_stocks[:max_stocks], output.debate_results
        )
        
        # 统计
        output.stats = {
            "stocks_analyzed": len(output.debate_results),
            "llm_providers": self.llm.get_available_providers(),
            "debate_rounds": self.max_debate_rounds,
            "recommendations": len(output.final_recommendations),
        }
        
        if self.verbose:
            print(f"\n  ✅ 决策层处理完成")
            print(f"     分析股票: {output.stats['stocks_analyzed']} 只")
            print(f"     使用LLM: {output.stats['llm_providers']}")
        
        return output
    
    def _build_data_summary(self, stock: Dict, data_bundle) -> str:
        """构建股票数据摘要"""
        lines = [f"## {stock.get('name', '')} ({stock.get('code', '')})"]
        
        lines.append(f"\n### 量化信号")
        lines.append(f"- ML预测信号: {stock.get('ml_signal', 'N/A')}")
        lines.append(f"- 因子综合得分: {stock.get('factor_score', 'N/A')}")
        lines.append(f"- 综合排名得分: {stock.get('combined_score', 'N/A')}")
        lines.append(f"- 行业: {stock.get('industry', 'N/A')}")
        lines.append(f"- 最新价格: {stock.get('latest_price', 'N/A')}")
        
        # 从data_bundle获取更多信息
        if data_bundle and hasattr(data_bundle, 'stock_universe'):
            stock_record = data_bundle.stock_universe.get(stock.get('code', ''))
            if stock_record and stock_record.financial_data:
                fd = stock_record.financial_data
                lines.append(f"\n### 财务指标")
                lines.append(f"- 年化收益率: {fd.get('annual_return', 0):.2%}")
                lines.append(f"- 年化波动率: {fd.get('annual_volatility', 0):.2%}")
                lines.append(f"- 夏普比率: {fd.get('sharpe_ratio', 0):.2f}")
                lines.append(f"- 最大回撤: {fd.get('max_drawdown', 0):.2%}")
                lines.append(f"- 近1月收益: {fd.get('return_1m', 0):.2%}")
                lines.append(f"- 近3月收益: {fd.get('return_3m', 0):.2%}")
                lines.append(f"- 近1年收益: {fd.get('return_1y', 0):.2%}")
                lines.append(f"- 52周最高: {fd.get('price_52w_high', 0):.2f}")
                lines.append(f"- 52周最低: {fd.get('price_52w_low', 0):.2f}")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, ranked_stocks: List[Dict],
                                    debate_results: Dict[str, DebateResult]) -> List[Dict]:
        """综合量化信号和定性分析生成最终推荐"""
        recommendations = []
        
        for stock in ranked_stocks:
            code = stock.get('code', '')
            debate = debate_results.get(code)
            
            rec = {
                'code': code,
                'name': stock.get('name', code),
                'quant_score': stock.get('combined_score', 0),
                'qualitative_score': debate.final_score if debate else 5.0,
                'investment_rating': debate.investment_rating if debate else "待分析",
                'confidence': debate.final_confidence if debate else 0.5,
                'consensus': debate.consensus if debate else "",
                'bull_case': debate.bull_case if debate else "",
                'bear_case': debate.bear_case if debate else "",
                'valuation': debate.valuation_summary if debate else "",
            }
            
            # 综合得分 (量化60% + 定性40%)
            rec['final_score'] = rec['quant_score'] * 0.6 + (rec['qualitative_score'] / 10) * 0.4
            
            recommendations.append(rec)
        
        # 按综合得分排序
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        return recommendations


# ==================== 便捷函数 ====================

def run_decision_analysis(ranked_stocks: List[Dict], data_bundle=None,
                           quant_summary: str = "", verbose: bool = True,
                           max_stocks: int = 5) -> DecisionLayerOutput:
    """便捷函数：运行决策分析"""
    layer = UnifiedDecisionLayer(verbose=verbose)
    return layer.analyze(ranked_stocks, data_bundle, quant_summary, max_stocks)


if __name__ == "__main__":
    print("=" * 60)
    print("V6.0 统一决策层测试")
    print("=" * 60)
    
    # 测试LLM适配器
    llm = LLMAdapter(verbose=True)
    print(f"\n可用LLM: {llm.get_available_providers()}")
    
    if llm.is_available():
        response = llm.chat([
            {"role": "user", "content": "请用一句话介绍你自己"}
        ])
        print(f"\nLLM响应: {response.content[:200]}")
        print(f"提供商: {response.provider}, 模型: {response.model}")

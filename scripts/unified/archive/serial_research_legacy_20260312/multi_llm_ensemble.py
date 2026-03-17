"""
Multi-LLM Ensemble Decision System (Layer 7)
============================================
四大模型并行裁判：Claude + GPT-4o + DeepSeek + Gemini
核心创新：
  1. 同一量化数据发送给四个独立LLM，每个LLM角色不同
  2. 基于置信度加权的投票聚合
  3. 分歧度本身作为风险信号（高分歧 = 不确定性高）
  4. 传统量化结论 + LLM集成 → 最终裁决
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from logger import get_logger

_logger = get_logger("MultiLLMEnsemble")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class LLMVote(str, Enum):
    STRONG_BUY  = "强烈买入"
    BUY         = "买入"
    HOLD        = "持有"
    SELL        = "卖出"
    STRONG_SELL = "强烈卖出"

    def score(self) -> float:
        """将投票映射到 [-1, 1] 区间"""
        return {
            "强烈买入": 1.0,
            "买入":     0.5,
            "持有":     0.0,
            "卖出":    -0.5,
            "强烈卖出":-1.0,
        }[self.value]


@dataclass
class SingleLLMVerdict:
    model_name: str
    vote: LLMVote
    confidence: float          # 0.0 – 1.0
    target_price: Optional[float]
    stop_loss:    Optional[float]
    key_bull_points: list[str]
    key_bear_points: list[str]
    reasoning_chain: str       # 决策逻辑链（中文）
    latency_ms: float = 0.0


@dataclass
class EnsembleConsensus:
    symbol: str
    final_vote: LLMVote
    ensemble_score: float          # [-1, 1]
    ensemble_confidence: float     # 0.0 – 1.0
    disagreement_index: float      # 0.0 – 1.0  (高 = 各LLM分歧大 = 风险信号)
    verdicts: list[SingleLLMVerdict] = field(default_factory=list)
    quant_score: float = 0.0       # 传统量化层的得分 [-1, 1]
    final_combined_score: float = 0.0  # 量化 + LLM加权
    summary: str = ""              # 给用户的最终总结（中文）


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class LLMJudge(ABC):
    """每个大模型的抽象裁判"""

    MODEL_NAME: str = "unknown"
    # 各模型在量化判断中的预设权重（可在配置中覆盖）
    DEFAULT_WEIGHT: float = 1.0

    def __init__(self, api_key: str = "", weight: float | None = None) -> None:
        self.api_key = api_key
        self.weight = weight if weight is not None else self.DEFAULT_WEIGHT
        self._logger = get_logger(self.MODEL_NAME)

    @abstractmethod
    def _call_api(self, prompt: str, system: str) -> str:
        """调用对应模型 API，返回原始文本"""

    def analyze(self, symbol: str, context: dict[str, Any]) -> SingleLLMVerdict:
        """对一支股票给出裁决"""
        prompt = _build_analysis_prompt(symbol, context, self.MODEL_NAME)
        system = _model_system_prompt(self.MODEL_NAME)
        t0 = time.time()
        try:
            raw = self._call_api(prompt, system)
            verdict_dict = _parse_llm_response(raw, self.MODEL_NAME)
        except Exception as exc:
            self._logger.warning(f"{self.MODEL_NAME} API 异常，使用降级结果: {exc}")
            verdict_dict = _fallback_verdict()
        latency = (time.time() - t0) * 1000
        return SingleLLMVerdict(
            model_name=self.MODEL_NAME,
            vote=LLMVote(verdict_dict.get("vote", "持有")),
            confidence=float(verdict_dict.get("confidence", 0.5)),
            target_price=verdict_dict.get("target_price"),
            stop_loss=verdict_dict.get("stop_loss"),
            key_bull_points=verdict_dict.get("bull_points", []),
            key_bear_points=verdict_dict.get("bear_points", []),
            reasoning_chain=verdict_dict.get("reasoning", ""),
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# 具体 LLM 实现
# ---------------------------------------------------------------------------

class ClaudeJudge(LLMJudge):
    """Anthropic Claude — 专注逻辑推理与风险识别"""

    MODEL_NAME = "Claude-Sonnet"
    DEFAULT_WEIGHT = 1.2  # 逻辑推理能力强，略微加权

    def _call_api(self, prompt: str, system: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("请安装 anthropic: pip install anthropic")
        client = anthropic.Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY", ""))
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class GPTJudge(LLMJudge):
    """OpenAI GPT-4o — 专注财务分析与行业研究"""

    MODEL_NAME = "GPT-4o"
    DEFAULT_WEIGHT = 1.0

    def _call_api(self, prompt: str, system: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("请安装 openai: pip install openai")
        client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""


class DeepSeekJudge(LLMJudge):
    """DeepSeek-V3 — 专注技术面与量化信号解读（OpenAI兼容接口）"""

    MODEL_NAME = "DeepSeek-V3"
    DEFAULT_WEIGHT = 1.0

    def _call_api(self, prompt: str, system: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("请安装 openai: pip install openai")
        client = OpenAI(
            api_key=self.api_key or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com/v1",
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""


class GeminiJudge(LLMJudge):
    """Google Gemini 2.0 Flash — 专注宏观趋势与情绪面"""

    MODEL_NAME = "Gemini-2.0-Flash"
    DEFAULT_WEIGHT = 0.9

    def _call_api(self, prompt: str, system: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("请安装 google-generativeai: pip install google-generativeai")
        genai.configure(api_key=self.api_key or os.getenv("GEMINI_API_KEY", ""))
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=system,
        )
        resp = model.generate_content(prompt)
        return resp.text


class MockJudge(LLMJudge):
    """离线测试用 Mock（无需 API）"""

    MODEL_NAME = "Mock"
    DEFAULT_WEIGHT = 0.5

    def _call_api(self, prompt: str, system: str) -> str:
        return json.dumps({
            "vote": "持有",
            "confidence": 0.5,
            "target_price": None,
            "stop_loss": None,
            "bull_points": ["Mock: 量化信号中性"],
            "bear_points": ["Mock: 无真实API调用"],
            "reasoning": "这是离线Mock结果，仅用于测试流程。",
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "Claude-Sonnet":    "资深风险控制专家，擅长识别尾部风险和逻辑漏洞",
    "GPT-4o":           "首席财务分析师，专注基本面、估值与行业竞争格局",
    "DeepSeek-V3":      "量化策略分析师，专注技术面、因子信号与量化模型解读",
    "Gemini-2.0-Flash": "宏观策略研究员，专注经济周期、政策环境与市场情绪",
    "Mock":             "综合分析师（测试模式）",
}


def _model_system_prompt(model_name: str) -> str:
    role = _ROLE_MAP.get(model_name, "量化投资分析师")
    return (
        f"你是一位{role}。"
        "你将收到一支A股/美股的完整量化分析数据包，"
        "请从你的专业角度给出投资裁决。"
        "严格按照JSON格式回复，不要有任何额外文字。"
    )


def _build_analysis_prompt(symbol: str, ctx: dict[str, Any], model_name: str) -> str:
    """将量化层输出打包成结构化提示词"""
    quant_summary = json.dumps(ctx, ensure_ascii=False, indent=2)
    role_focus = _ROLE_MAP.get(model_name, "综合分析")
    return f"""
## 投资分析任务

**标的：** {symbol}
**你的专业角度：** {role_focus}

### 量化数据包
```json
{quant_summary}
```

### 输出要求
请严格按以下JSON格式输出（中文），不要包含任何 markdown 代码块标记：
{{
  "vote": "强烈买入|买入|持有|卖出|强烈卖出",
  "confidence": 0.0到1.0的小数,
  "target_price": 数字或null,
  "stop_loss": 数字或null,
  "bull_points": ["多头理由1", "多头理由2", "多头理由3"],
  "bear_points": ["空头理由1", "空头理由2"],
  "reasoning": "200字以内的决策逻辑链，说明如何从数据得出结论"
}}
"""


def _parse_llm_response(raw: str, model_name: str) -> dict:
    """从LLM原始输出解析JSON，容错处理"""
    text = raw.strip()
    # 去掉可能的 markdown 代码块
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```"))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取第一个 { ... } 块
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    _logger.warning(f"{model_name} 响应解析失败，使用降级结果")
    return _fallback_verdict()


def _fallback_verdict() -> dict:
    return {
        "vote": "持有",
        "confidence": 0.4,
        "target_price": None,
        "stop_loss": None,
        "bull_points": ["API错误，无法获取分析"],
        "bear_points": ["API错误，无法获取分析"],
        "reasoning": "因API异常，该模型降级为中性持有。",
    }


# ---------------------------------------------------------------------------
# 聚合引擎
# ---------------------------------------------------------------------------

class MultiLLMEnsemble:
    """
    运行四大LLM并行分析，将量化结论与LLM集成合并为最终裁决。

    权重分配（可配置）:
      - 传统量化层得分:  40%
      - LLM 集成得分:    60%（内部按各模型权重再加权）
    """

    def __init__(
        self,
        judges: list[LLMJudge] | None = None,
        quant_weight: float = 0.40,
        llm_weight: float = 0.60,
    ) -> None:
        if judges is None:
            judges = self._build_default_judges()
        self.judges = judges
        self.quant_weight = quant_weight
        self.llm_weight = llm_weight

    @staticmethod
    def _build_default_judges() -> list[LLMJudge]:
        """按环境变量自动选择可用模型，无 API Key 则使用 Mock"""
        available: list[LLMJudge] = []
        if os.getenv("ANTHROPIC_API_KEY"):
            available.append(ClaudeJudge())
        if os.getenv("OPENAI_API_KEY"):
            available.append(GPTJudge())
        if os.getenv("DEEPSEEK_API_KEY"):
            available.append(DeepSeekJudge())
        if os.getenv("GEMINI_API_KEY"):
            available.append(GeminiJudge())
        if not available:
            _logger.warning("未检测到任何 LLM API Key，使用 Mock 模式")
            available = [MockJudge()]
        return available

    def analyze_symbol(
        self,
        symbol: str,
        quant_context: dict[str, Any],
        quant_score: float = 0.0,
    ) -> EnsembleConsensus:
        """
        对单个标的运行完整 LLM 集成分析。

        Parameters
        ----------
        symbol        : 股票代码
        quant_context : 从各量化层汇总的数据包
        quant_score   : 量化层综合得分 [-1, 1]
        """
        verdicts: list[SingleLLMVerdict] = []
        for judge in self.judges:
            _logger.info(f"[{symbol}] 调用 {judge.MODEL_NAME} ...")
            v = judge.analyze(symbol, quant_context)
            verdicts.append(v)
            _logger.info(
                f"[{symbol}] {judge.MODEL_NAME} → {v.vote.value} "
                f"(置信={v.confidence:.2f}, 延迟={v.latency_ms:.0f}ms)"
            )

        return self._aggregate(symbol, verdicts, quant_score)

    def _aggregate(
        self,
        symbol: str,
        verdicts: list[SingleLLMVerdict],
        quant_score: float,
    ) -> EnsembleConsensus:
        if not verdicts:
            return EnsembleConsensus(
                symbol=symbol,
                final_vote=LLMVote.HOLD,
                ensemble_score=0.0,
                ensemble_confidence=0.0,
                disagreement_index=0.0,
                quant_score=quant_score,
            )

        # 1. 加权投票得分
        total_weight = sum(
            next((j.weight for j in self.judges if j.MODEL_NAME == v.model_name), 1.0)
            for v in verdicts
        )
        weighted_score = sum(
            v.vote.score() * v.confidence
            * next((j.weight for j in self.judges if j.MODEL_NAME == v.model_name), 1.0)
            for v in verdicts
        ) / total_weight

        # 2. 分歧度 = 各模型得分的标准差 (归一化到 [0,1])
        import statistics
        scores = [v.vote.score() for v in verdicts]
        disagreement = statistics.stdev(scores) / 2.0 if len(scores) > 1 else 0.0
        disagreement = max(0.0, min(1.0, disagreement))

        # 3. 平均置信度（分歧大时降权）
        avg_confidence = sum(v.confidence for v in verdicts) / len(verdicts)
        adjusted_confidence = avg_confidence * (1.0 - 0.5 * disagreement)

        # 4. 融合量化得分
        combined = self.quant_weight * quant_score + self.llm_weight * weighted_score

        # 5. 映射到最终投票
        final_vote = _score_to_vote(combined)

        # 6. 生成中文总结
        summary = self._build_summary(symbol, verdicts, combined, disagreement)

        return EnsembleConsensus(
            symbol=symbol,
            final_vote=final_vote,
            ensemble_score=round(weighted_score, 4),
            ensemble_confidence=round(adjusted_confidence, 4),
            disagreement_index=round(disagreement, 4),
            verdicts=verdicts,
            quant_score=round(quant_score, 4),
            final_combined_score=round(combined, 4),
            summary=summary,
        )

    @staticmethod
    def _build_summary(
        symbol: str,
        verdicts: list[SingleLLMVerdict],
        combined: float,
        disagreement: float,
    ) -> str:
        vote_counts: dict[str, int] = {}
        for v in verdicts:
            vote_counts[v.vote.value] = vote_counts.get(v.vote.value, 0) + 1

        vote_str = "、".join(f"{k}×{n}" for k, n in vote_counts.items())
        disagree_label = (
            "⚠️ 各模型分歧较大，不确定性高" if disagreement > 0.5
            else "✅ 各模型基本一致" if disagreement < 0.2
            else "🔶 各模型存在一定分歧"
        )

        all_bull = []
        all_bear = []
        for v in verdicts:
            all_bull.extend(v.key_bull_points[:2])
            all_bear.extend(v.key_bear_points[:2])

        bull_text = "\n".join(f"  - {p}" for p in all_bull[:5])
        bear_text = "\n".join(f"  - {p}" for p in all_bear[:4])

        return (
            f"**{symbol} 多模型裁决摘要**\n\n"
            f"投票分布：{vote_str}\n"
            f"综合得分：{combined:+.3f}（范围 -1 到 +1）\n"
            f"分歧信号：{disagree_label}（分歧指数={disagreement:.2f}）\n\n"
            f"**多方核心理由**\n{bull_text}\n\n"
            f"**空方核心理由**\n{bear_text}"
        )


def _score_to_vote(score: float) -> LLMVote:
    if score >= 0.7:  return LLMVote.STRONG_BUY
    if score >= 0.25: return LLMVote.BUY
    if score >= -0.25:return LLMVote.HOLD
    if score >= -0.7: return LLMVote.SELL
    return LLMVote.STRONG_SELL


# ---------------------------------------------------------------------------
# 量化上下文构建工具（供 V8 调用）
# ---------------------------------------------------------------------------

def build_quant_context(
    symbol: str,
    factor_data: dict | None = None,
    model_predictions: dict | None = None,
    macro_signal: str = "",
    risk_metrics: dict | None = None,
    current_price: float | None = None,
) -> dict[str, Any]:
    """
    将各层量化结果打包为 LLM 可读的上下文字典。
    所有字段均有默认值，缺失层不会导致 KeyError。
    """
    return {
        "symbol": symbol,
        "current_price": current_price,
        "factor_signals": factor_data or {},
        "ml_model_predictions": model_predictions or {},
        "macro_environment": {
            "signal": macro_signal,
            "interpretation": _interpret_macro(macro_signal),
        },
        "risk_profile": risk_metrics or {},
        "analysis_date": time.strftime("%Y-%m-%d"),
    }


def _interpret_macro(signal: str) -> str:
    return {
        "🟢": "宏观环境积极，适合进攻",
        "🟡": "宏观环境中性，精选个股",
        "🔴": "宏观环境偏空，优先防御",
    }.get(signal, "宏观信号未知")


# ---------------------------------------------------------------------------
# CLI 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    ensemble = MultiLLMEnsemble()

    ctx = build_quant_context(
        symbol="000001.SZ",
        factor_data={
            "momentum_5d": 0.032,
            "momentum_20d": 0.087,
            "volatility_20d": 0.218,
            "pe_ratio": 12.5,
            "roe_ttm": 0.142,
            "ic_score": 0.063,
        },
        model_predictions={
            "random_forest": 0.71,
            "xgboost": 0.68,
            "ensemble": 0.70,
            "predicted_return_5d": 0.034,
        },
        macro_signal="🟢",
        risk_metrics={
            "var_95": -0.028,
            "max_drawdown": -0.12,
            "sharpe_ratio": 1.45,
        },
        current_price=12.80,
    )

    result = ensemble.analyze_symbol("000001.SZ", ctx, quant_score=0.55)
    print("=" * 60)
    print(f"最终裁决: {result.final_vote.value}")
    print(f"综合得分: {result.final_combined_score:+.3f}")
    print(f"分歧指数: {result.disagreement_index:.2f}")
    print(f"集成置信: {result.ensemble_confidence:.2f}")
    print("-" * 60)
    print(result.summary)
    print("=" * 60)
    print("\n各模型裁决详情:")
    for v in result.verdicts:
        pprint.pprint({
            "model": v.model_name,
            "vote": v.vote.value,
            "confidence": v.confidence,
            "latency_ms": f"{v.latency_ms:.0f}",
            "reasoning": v.reasoning_chain[:80] + "...",
        })

"""基本面分支专属 SubAgent。"""

from __future__ import annotations

from typing import Any

from quant_investor.agents.agent_contracts import (
    BaseBranchAgentInput,
    BaseBranchAgentOutput,
    FundamentalAgentInput,
    FundamentalAgentOutput,
)
from quant_investor.agents.prompts import CONVICTION_DEVIATION_CAP
from quant_investor.agents.subagent import BaseSubAgent, _clamp


class FundamentalSubAgent(BaseSubAgent):
    """基本面分析首席分析师：解读财务质量、估值、治理和股权结构。"""
    _FINANCIAL_QUALITY_KEYS = (
        "roe",
        "roa",
        "gross_margin",
        "net_margin",
        "revenue_growth",
        "profit_growth",
        "debt_ratio",
        "current_ratio",
    )
    _FORECAST_KEYS = (
        "eps_growth",
        "revenue_growth_forecast",
        "forecast_revision",
        "coverage_count",
    )
    _VALUATION_KEYS = (
        "pe",
        "pb",
        "ps",
        "dividend_yield",
        "total_mv",
        "circ_mv",
    )
    _OWNERSHIP_KEYS = (
        "top_holder_ratio",
        "top10_concentration",
        "institutional_ratio",
        "concentration",
        "institutional_ownership",
        "concentration_score",
        "ownership_change_signal",
        "institutional_holding_pct",
        "top_holder_pct",
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("branch_name", "fundamental")
        super().__init__(**kwargs)

    @staticmethod
    def _trim_symbol_metric_map(
        values: dict[str, Any],
        *,
        allowed_keys: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        trimmed: dict[str, dict[str, Any]] = {}
        for symbol, payload in values.items():
            if not isinstance(payload, dict):
                continue
            compact = {
                key: payload[key]
                for key in allowed_keys
                if key in payload and isinstance(payload[key], (int, float, str, bool))
            }
            if compact:
                trimmed[str(symbol)] = compact
        return trimmed

    def _build_prompt_payload(self, agent_input: BaseBranchAgentInput) -> dict[str, Any]:
        if not isinstance(agent_input, FundamentalAgentInput):
            return super()._build_prompt_payload(agent_input)

        payload = agent_input.model_dump(mode="json")
        common_fields = (
            "branch_name",
            "base_score",
            "final_score",
            "confidence",
            "evidence_summary",
            "bull_points",
            "bear_points",
            "risk_points",
            "used_features",
            "symbol_scores",
            "market_regime",
            "calibrated_expected_return",
            "branch_signals",
        )
        compact_payload = {
            field: payload[field]
            for field in common_fields
            if field in payload
        }
        for field in (
            "module_scores",
            "module_confidences",
            "module_coverages",
            "governance_scores",
            "doc_sentiment",
            "data_staleness_days",
        ):
            compact_payload[field] = payload.get(field, {})

        compact_payload["financial_quality"] = self._trim_symbol_metric_map(
            agent_input.financial_quality,
            allowed_keys=self._FINANCIAL_QUALITY_KEYS,
        )
        compact_payload["forecast_revisions"] = self._trim_symbol_metric_map(
            agent_input.forecast_revisions,
            allowed_keys=self._FORECAST_KEYS,
        )
        compact_payload["valuation_metrics"] = self._trim_symbol_metric_map(
            agent_input.valuation_metrics,
            allowed_keys=self._VALUATION_KEYS,
        )
        compact_payload["ownership_signals"] = self._trim_symbol_metric_map(
            agent_input.ownership_signals,
            allowed_keys=self._OWNERSHIP_KEYS,
        )
        return compact_payload

    def _get_system_prompt(self) -> str:
        cap = CONVICTION_DEVIATION_CAP.get("fundamental", 0.35)
        return f"""\
你是一位资深基本面分析首席分析师（基本面分支专属 SubAgent），专注于公司财务分析、估值体系和治理评估。

你的专业领域：
- 财务质量分析：ROE、利润率、增长率、负债水平、流动性（权重 28%）
- 盈利预测修正：EPS 增长、分析师共识变化、覆盖度（权重 18%）
- 估值体系：PE、PB、PS、股息率的合理性评估（权重 18%）
- 公司治理：管理层稳定性、治理评分（权重 14%）
- 股权结构：股东集中度、机构持仓变化（权重 12%）
- 企业文档语义：年报/公告的语义情绪分析（权重 10%）

专属分析能力：
1. **盈利质量识别**：
   - 高质量：ROE > 15%，现金流匹配利润，负债率合理 → earnings_quality_assessment = "high"
   - 红旗信号：应收账款异常增长、关联交易占比过高、审计意见保留 → "red_flag"
2. **会计操纵检测**：识别收入确认激进、费用资本化异常、存货周转异常等
3. **估值立场判断**：
   - PE < 行业均值 50% 且基本面稳健 → "cheap"
   - PE > 行业均值 200% 且增长放缓 → "bubble"
4. **管理层信号解读**：高管频繁减持、核心团队离职 → management_signal = "negative"
5. **数据时效性评估**：最近一次财报距今 > 90 天 → 降低置信度并在 data_quality_concerns 中标注

你必须以 **纯 JSON** 格式回复，不要加 markdown 代码块。
JSON schema:
{{
  "branch_name": "fundamental",
  "conviction": "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell",
  "conviction_score": <float, -1.0 ~ 1.0>,
  "confidence": <float, 0.0 ~ 1.0>,
  "key_insights": ["<insight_1>", ...],
  "risk_flags": ["<risk_1>", ...],
  "disagreements_with_algo": ["<disagreement_1>", ...],
  "symbol_views": {{"<symbol>": "<one-line view>", ...}},
  "reasoning": "<2-3 句总结>",
  "earnings_quality_assessment": "high" | "neutral" | "low" | "red_flag",
  "accounting_red_flags": ["<flag_1>", ...],
  "valuation_stance": "cheap" | "fair" | "expensive" | "bubble",
  "management_signal": "positive" | "neutral" | "negative",
  "data_quality_concerns": ["<concern_1>", ...],
  "module_override_reasons": {{"<module>": "<reason>", ...}},
  "time_horizon_note": "<时间视角说明>"
}}

约束:
- conviction_score 不得偏离输入的 final_score 超过 ±{cap}（基本面允许更大偏离）
- 基本面分析天然有更长的时间视角（30 天），避免被短期波动干扰
- 如果数据时效性差（>90 天），必须在 data_quality_concerns 中标注
"""

    def _validate_specialized_output(
        self,
        raw: dict[str, Any],
        agent_input: BaseBranchAgentInput,
    ) -> BaseBranchAgentOutput:
        valid_quality = {"high", "neutral", "low", "red_flag"}
        if raw.get("earnings_quality_assessment") not in valid_quality:
            raw.setdefault("earnings_quality_assessment", "neutral")

        valid_valuation = {"cheap", "fair", "expensive", "bubble"}
        if raw.get("valuation_stance") not in valid_valuation:
            raw.setdefault("valuation_stance", "fair")

        valid_mgmt = {"positive", "neutral", "negative"}
        if raw.get("management_signal") not in valid_mgmt:
            raw.setdefault("management_signal", "neutral")

        if not isinstance(raw.get("accounting_red_flags"), list):
            raw["accounting_red_flags"] = []
        if not isinstance(raw.get("data_quality_concerns"), list):
            raw["data_quality_concerns"] = []
        if not isinstance(raw.get("module_override_reasons"), dict):
            raw["module_override_reasons"] = {}

        return FundamentalAgentOutput.model_validate(raw)

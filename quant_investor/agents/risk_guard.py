"""
规则化 RiskGuard。
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    CoverageScope,
    EventNote,
    RiskDecision,
    RiskAdvisory,
    RiskLevel,
)
from quant_investor.agents.base import BaseAgent
from quant_investor.v16.candidate_pipeline import Stage2Decision
from quant_investor.v16.protocol_matrix import PROTOCOL_VERSIONS


RISK_GUARD_SINGLE_NAME_WEIGHT_CAP_ENV = "RISK_GUARD_SINGLE_NAME_WEIGHT_CAP"
RISK_GUARD_SINGLE_NAME_WEIGHT_CAP_DEFAULT = 0.15
RISK_GUARD_SINGLE_NAME_WEIGHT_CAP = RISK_GUARD_SINGLE_NAME_WEIGHT_CAP_DEFAULT

# Default hard-veto keywords, matched as case-insensitive substrings against
# branch/macro ``investment_risks`` texts and ``constraints["risk_flags"]``.
# CN-market branches emit risk texts in Chinese, so the default set must be
# bilingual or the one-vote veto silently fails for A-share runs.
# ``constraints["veto_keywords"]`` still replaces this set when provided.
DEFAULT_VETO_KEYWORDS: tuple[str, ...] = (
    # English
    "fraud",
    "halt",
    "delist",
    "hard veto",
    "veto",
    "liquidity freeze",
    # Chinese (A-share regulatory / delisting triggers)
    "退市",        # delisting (covers 退市风险 / 强制退市 / 退市整理 / 退市风险警示)
    "停牌",        # trading halt / suspension
    "造假",        # fabrication (covers 财务造假 / 数据造假)
    "欺诈",        # fraud (covers 欺诈发行)
    "立案调查",    # regulatory investigation formally filed
    "风险警示",    # ST / *ST risk-warning designation (退市风险警示 / 其他风险警示)
    "*st",         # explicit *ST marker; bare "st" would over-match English text
    "流动性冻结",  # liquidity freeze
    "一票否决",    # explicit one-vote veto phrase
)


def risk_guard_single_name_weight_cap() -> float:
    """Return the fail-safe RiskGuard single-name cap, with env rollback support."""

    raw_value = os.environ.get(RISK_GUARD_SINGLE_NAME_WEIGHT_CAP_ENV)
    try:
        value = float(
            raw_value
            if raw_value is not None
            else RISK_GUARD_SINGLE_NAME_WEIGHT_CAP_DEFAULT
        )
    except (TypeError, ValueError):
        value = RISK_GUARD_SINGLE_NAME_WEIGHT_CAP_DEFAULT
    if not math.isfinite(value):
        value = RISK_GUARD_SINGLE_NAME_WEIGHT_CAP_DEFAULT
    return max(0.0, min(1.0, value))


class RiskGuard(BaseAgent):
    """读取结构化分支结论并施加硬约束。"""

    agent_name = "RiskGuard"

    def run(self, payload: Mapping[str, Any]) -> RiskDecision:
        envelope = self.ensure_payload(payload)
        self.require_keys(envelope, "branch_verdicts", "portfolio_state", "constraints")

        branch_verdicts = self._normalize_branch_verdicts(envelope["branch_verdicts"])
        macro_verdict = envelope.get("macro_verdict")
        portfolio_state = self.ensure_payload(envelope.get("portfolio_state", {}))
        constraints = self.ensure_payload(envelope.get("constraints", {}))

        candidate_symbols = self._collect_candidate_symbols(branch_verdicts, portfolio_state)
        risk_texts = self._collect_risk_texts(branch_verdicts, macro_verdict, constraints)
        single_name_cap = risk_guard_single_name_weight_cap()

        explicit_action_cap = constraints.get("action_cap", ActionLabel.BUY)
        action_cap = (
            explicit_action_cap if isinstance(explicit_action_cap, ActionLabel)
            else ActionLabel(str(explicit_action_cap).strip().lower())
        )
        gross_cap = self._get_float(constraints, "gross_exposure_cap", default=1.0)
        max_weight = self._get_float(constraints, "max_weight", default=1.0)
        if macro_verdict and isinstance(macro_verdict, BranchVerdict):
            gross_cap = min(
                gross_cap,
                float(macro_verdict.metadata.get("target_gross_exposure", 1.0)),
            )
            if macro_verdict.final_score <= -0.2:
                action_cap = self.more_restrictive_action(action_cap, ActionLabel.HOLD)
                gross_cap = min(gross_cap, 0.5)
                max_weight = min(max_weight, single_name_cap)

        veto = bool(constraints.get("force_veto")) or self._has_veto_keyword(risk_texts, constraints)
        blocked_symbols = {str(symbol) for symbol in constraints.get("blocked_symbols", [])}
        if veto:
            veto_cap_raw = constraints.get("veto_action_cap", ActionLabel.HOLD)
            veto_cap = veto_cap_raw if isinstance(veto_cap_raw, ActionLabel) else ActionLabel(str(veto_cap_raw).strip().lower())
            action_cap = self.more_restrictive_action(action_cap, veto_cap)
            gross_cap = min(gross_cap, self._get_float(constraints, "veto_gross_exposure_cap", default=0.0))
            max_weight = min(max_weight, self._get_float(constraints, "veto_max_weight", default=0.0))
            blocked_symbols.update(candidate_symbols)

        if not veto and len(risk_texts) >= 3:
            action_cap = self.more_restrictive_action(action_cap, ActionLabel.HOLD)
            gross_cap = min(gross_cap, 0.6)
            max_weight = min(max_weight, single_name_cap)


        gross_cap = self.clamp(gross_cap, 0.0, 1.0)
        max_weight = self.clamp(max_weight, 0.0, 1.0)
        unblocked_symbols = [symbol for symbol in candidate_symbols if symbol not in blocked_symbols]
        position_limits = {symbol: max_weight for symbol in unblocked_symbols}

        risk_level = self._infer_risk_level(veto=veto, risk_count=len(risk_texts), gross_cap=gross_cap)
        status = AgentStatus.VETOED if veto else (
            AgentStatus.DEGRADED if action_cap != ActionLabel.BUY or gross_cap < 1.0 else AgentStatus.SUCCESS
        )

        reasons = []
        if veto:
            reasons.append("RiskGuard 触发硬否决，仅允许保留更保守的动作上限。")
        if macro_verdict and isinstance(macro_verdict, BranchVerdict):
            reasons.append(
                f"宏观约束要求总暴露不高于 {float(macro_verdict.metadata.get('target_gross_exposure', gross_cap)):.0%}。"
            )
        reasons.extend(risk_texts[:5])
        if not reasons:
            reasons.append("未触发额外风险约束，维持基础上限。")

        events = [
            EventNote(
                title="risk_guard_applied",
                message=(
                    f"action_cap={action_cap.value}, gross_exposure_cap={gross_cap:.2f}, "
                    f"max_weight={max_weight:.2f}, veto={veto}"
                ),
                scope=CoverageScope.PORTFOLIO,
                risk_level=risk_level,
            )
        ]

        metadata = {
            "candidate_symbols": list(candidate_symbols),
            "unblocked_symbols": list(unblocked_symbols),
            "rule_based": True,
            "single_name_weight_cap": single_name_cap,
        }

        return RiskDecision(
            status=status,
            risk_level=risk_level,
            hard_veto=veto,
            veto=veto,
            action_cap=action_cap,
            max_weight=max_weight,
            gross_exposure_cap=gross_cap,
            target_exposure_cap=gross_cap,
            blocked_symbols=sorted(blocked_symbols),
            position_limits=position_limits,
            reasons=reasons,
            events=events,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_branch_verdicts(payload: Any) -> dict[str, BranchVerdict]:
        if isinstance(payload, Mapping):
            return {
                str(name): verdict
                for name, verdict in payload.items()
                if isinstance(verdict, BranchVerdict)
            }
        raise TypeError("branch_verdicts 必须是 Mapping[str, BranchVerdict]")

    @staticmethod
    def _get_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
        value = mapping.get(key, default)
        return float(default if value is None else value)


    @staticmethod
    def _collect_candidate_symbols(
        branch_verdicts: Mapping[str, BranchVerdict],
        portfolio_state: Mapping[str, Any],
    ) -> list[str]:
        symbols: list[str] = []
        seen: set[str] = set()

        for symbol in portfolio_state.get("candidate_symbols", []):
            text = str(symbol)
            if text and text not in seen:
                seen.add(text)
                symbols.append(text)

        current_weights = portfolio_state.get("current_weights", {})
        if isinstance(current_weights, Mapping):
            for symbol in current_weights:
                text = str(symbol)
                if text and text not in seen:
                    seen.add(text)
                    symbols.append(text)

        for verdict in branch_verdicts.values():
            if verdict.symbol and verdict.symbol not in seen:
                seen.add(verdict.symbol)
                symbols.append(verdict.symbol)
            for item in verdict.evidence:
                for symbol in item.symbols:
                    text = str(symbol)
                    if text and text not in seen:
                        seen.add(text)
                        symbols.append(text)

        return symbols

    @staticmethod
    def _collect_risk_texts(
        branch_verdicts: Mapping[str, BranchVerdict],
        macro_verdict: Any,
        constraints: Mapping[str, Any],
    ) -> list[str]:
        texts: list[str] = []
        for verdict in branch_verdicts.values():
            texts.extend(str(item) for item in verdict.investment_risks if str(item).strip())
        if isinstance(macro_verdict, BranchVerdict):
            texts.extend(str(item) for item in macro_verdict.investment_risks if str(item).strip())
        texts.extend(str(item) for item in constraints.get("risk_flags", []) if str(item).strip())
        return texts

    @staticmethod
    def _has_veto_keyword(risk_texts: list[str], constraints: Mapping[str, Any]) -> bool:
        keywords = {
            str(keyword).strip().lower()
            for keyword in constraints.get("veto_keywords", DEFAULT_VETO_KEYWORDS)
            # An empty keyword would substring-match every text and force a
            # permanent veto, so blank entries are dropped defensively.
            if str(keyword).strip()
        }
        if not keywords:
            return False
        for text in risk_texts:
            lowered = text.lower()
            if any(keyword in lowered for keyword in keywords):
                return True
        return False

    @staticmethod
    def _infer_risk_level(veto: bool, risk_count: int, gross_cap: float) -> RiskLevel:
        if veto or gross_cap <= 0.1:
            return RiskLevel.EXTREME
        if gross_cap <= 0.4 or risk_count >= 3:
            return RiskLevel.HIGH
        if gross_cap <= 0.75 or risk_count >= 1:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class RiskAdvisor(BaseAgent):
    """V16 risk narrative that has no authority over actions or weights."""

    agent_name = "RiskAdvisor"
    protocol_version = PROTOCOL_VERSIONS["risk_advisor_version"]
    _SEVERE_KEYWORDS = frozenset(DEFAULT_VETO_KEYWORDS)

    def run(self, payload: Mapping[str, Any]) -> RiskAdvisory:
        envelope = self.ensure_payload(payload)
        self.require_keys(envelope, "branch_verdicts")
        branch_verdicts = RiskGuard._normalize_branch_verdicts(
            envelope["branch_verdicts"]
        )
        macro_verdict = envelope.get("macro_verdict")
        advisory_context = self.ensure_payload(envelope.get("advisory_context", {}))

        flags = self._dedupe(
            RiskGuard._collect_risk_texts(
                branch_verdicts,
                macro_verdict,
                {"risk_flags": advisory_context.get("flags", [])},
            )
        )
        severe_flag_count = sum(
            1
            for flag in flags
            if any(keyword in flag.lower() for keyword in self._SEVERE_KEYWORDS)
        )
        if severe_flag_count:
            severity = RiskLevel.EXTREME
        elif len(flags) >= 3:
            severity = RiskLevel.HIGH
        elif flags:
            severity = RiskLevel.MEDIUM
        else:
            severity = RiskLevel.LOW

        scenarios = self._dedupe(advisory_context.get("scenarios", []))
        suggestions = self._dedupe(advisory_context.get("suggestions", []))
        if flags and not scenarios:
            scenarios = ["监测已识别风险事件是否恶化及其二阶影响。"]
        if flags and not suggestions:
            suggestions = ["在下一次研究复核中逐项回应风险标记并保留证据。"]
        rationale = str(advisory_context.get("rationale") or "").strip()
        if not rationale:
            rationale = (
                f"识别到 {len(flags)} 个风险标记；该结论仅供研究解释，"
                "不构成否决、动作上限或仓位约束。"
            )

        return RiskAdvisory(
            severity=severity,
            flags=flags,
            scenarios=scenarios,
            suggestions=suggestions,
            rationale=rationale,
        )

    @staticmethod
    def validate_severe_buy_rationale(
        advisory: RiskAdvisory,
        decisions: list[Stage2Decision],
    ) -> None:
        """Reject an unexplained severe-risk BUY after Stage2 schema validation.

        This check never rewrites action, rank, or target weight.  A caller may
        either accept the already-validated response or fail closed and request
        a new response containing the missing rationale.
        """

        if advisory.severity not in {RiskLevel.HIGH, RiskLevel.EXTREME}:
            return
        missing = sorted(
            decision.symbol
            for decision in decisions
            if decision.action == "BUY"
            and not str(decision.risk_acceptance_rationale or "").strip()
        )
        if missing:
            raise ValueError(
                "severe-risk BUY requires risk_acceptance_rationale: "
                + ", ".join(missing)
            )

    @staticmethod
    def _dedupe(values: Any) -> list[str]:
        if not isinstance(values, (list, tuple, set)):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

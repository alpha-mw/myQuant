"""
规则化 RiskGuard。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    CoverageScope,
    EventNote,
    RiskDecision,
    RiskLevel,
)
from quant_investor.agents.base import BaseAgent


RISK_GUARD_SINGLE_NAME_WEIGHT_CAP = 0.50


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
                max_weight = min(max_weight, RISK_GUARD_SINGLE_NAME_WEIGHT_CAP)

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
            max_weight = min(max_weight, RISK_GUARD_SINGLE_NAME_WEIGHT_CAP)

        theme_enabled = self._as_bool(constraints.get("theme_risk_guard_enabled", False))
        theme_risk_flags: list[str] = []
        theme_risk_by_symbol: dict[str, Any] = {}
        theme_position_limits = {}
        theme_gross_cap: float | None = None
        theme_triggered = False
        if theme_enabled:
            theme_risk_flags = self._sanitize_text_list(constraints.get("theme_risk_flags", []))
            theme_risk_by_symbol = self._compact_theme_risk_by_symbol(
                constraints.get("theme_risk_by_symbol", {})
            )
            theme_position_limits = self._sanitize_position_limits(
                constraints.get("theme_position_limits", {})
            )
            theme_gross_cap = self._optional_clamped_float(
                constraints.get("theme_gross_exposure_cap")
            )
            if str(constraints.get("theme_action_cap") or "").strip().lower() == ActionLabel.HOLD.value:
                action_cap = self.more_restrictive_action(action_cap, ActionLabel.HOLD)
                theme_triggered = True
            if theme_gross_cap is not None:
                gross_cap = min(gross_cap, theme_gross_cap)
                theme_triggered = True

        gross_cap = self.clamp(gross_cap, 0.0, 1.0)
        max_weight = self.clamp(max_weight, 0.0, 1.0)
        unblocked_symbols = [symbol for symbol in candidate_symbols if symbol not in blocked_symbols]
        position_limits = {symbol: max_weight for symbol in unblocked_symbols}
        applied_theme_position_limits: dict[str, float] = {}
        if theme_enabled:
            for symbol, cap in theme_position_limits.items():
                if symbol not in position_limits:
                    continue
                applied_cap = min(float(position_limits[symbol]), cap)
                position_limits[symbol] = applied_cap
                applied_theme_position_limits[symbol] = applied_cap
            if applied_theme_position_limits or theme_risk_flags:
                theme_triggered = True

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
        if theme_enabled and theme_triggered:
            flag_text = ", ".join(theme_risk_flags[:5]) if theme_risk_flags else "symbol_position_cap"
            reasons.append(f"Theme risk overlay applied: {flag_text}")
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
        if theme_enabled and theme_triggered:
            theme_event_level = (
                RiskLevel.HIGH
                if risk_level in {RiskLevel.HIGH, RiskLevel.EXTREME}
                else RiskLevel.MEDIUM
            )
            events.append(
                EventNote(
                    title="theme_risk_guard_applied",
                    message=(
                        f"action_cap={action_cap.value}, gross_exposure_cap={gross_cap:.2f}, "
                        f"symbol_count={len(applied_theme_position_limits)}"
                    ),
                    scope=CoverageScope.PORTFOLIO,
                    risk_level=theme_event_level,
                )
            )

        metadata = {
            "candidate_symbols": list(candidate_symbols),
            "unblocked_symbols": list(unblocked_symbols),
            "rule_based": True,
        }
        if theme_enabled:
            metadata.update(
                {
                    "theme_risk_guard_enabled": True,
                    "theme_risk_flags": list(theme_risk_flags),
                    "theme_position_limits": dict(applied_theme_position_limits),
                    "theme_risk_by_symbol": theme_risk_by_symbol,
                }
            )

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
    def _optional_clamped_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        text = str(value).strip().lower()
        if text in {"", "0", "false", "no", "off"}:
            return False
        return True

    @staticmethod
    def _sanitize_text_list(value: Any) -> list[str]:
        if isinstance(value, (str, bytes)):
            items = [value]
        else:
            try:
                items = list(value or [])
            except TypeError:
                items = []
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @classmethod
    def _sanitize_position_limits(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, Mapping):
            return {}
        limits: dict[str, float] = {}
        for symbol, cap_value in value.items():
            symbol_text = str(symbol or "").strip()
            cap = cls._optional_clamped_float(cap_value)
            if not symbol_text or cap is None:
                continue
            limits[symbol_text] = cap
        return limits

    @classmethod
    def _compact_theme_risk_by_symbol(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        compact: dict[str, Any] = {}
        for symbol, metadata in list(value.items())[:50]:
            symbol_text = str(symbol or "").strip()
            if not symbol_text:
                continue
            if not isinstance(metadata, Mapping):
                compact[symbol_text] = {"available": False}
                continue
            compact[symbol_text] = {
                "available": bool(metadata.get("available", False)),
                "phase": str(metadata.get("phase") or ""),
                "primary_theme_id": str(metadata.get("primary_theme_id") or ""),
                "primary_theme_name": str(metadata.get("primary_theme_name") or ""),
                "symbol_score": cls._optional_clamped_float(metadata.get("symbol_score")) or 0.0,
                "risk_flags": cls._sanitize_text_list(metadata.get("risk_flags", [])),
            }
        return compact

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
            str(keyword).lower()
            for keyword in constraints.get(
                "veto_keywords",
                ["fraud", "halt", "delist", "hard veto", "veto", "liquidity freeze"],
            )
        }
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

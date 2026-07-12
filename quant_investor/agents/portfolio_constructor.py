"""
确定性的 PortfolioConstructor。

只消费结构化 ICDecision、宏观约束、风险上限与可交易快照，
不读取 Narrator 或自由文本来驱动目标权重。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_investor.agent_protocol import (
    ActionLabel,
    AgentStatus,
    BranchVerdict,
    ICDecision,
    PortfolioPlan,
    RiskDecision,
)
from quant_investor.agents.base import BaseAgent
from quant_investor.portfolio_backtest import PortfolioConstructor as LegacyPortfolioConstructor


@dataclass(frozen=True)
class _SymbolIntent:
    """单个标的的结构化投资意图。"""

    symbol: str
    score: float
    confidence: float
    calibrated_confidence: float
    momentum_strength: float
    fake_breakout_penalty: float
    action: ActionLabel
    position_mode: str
    sector: str


class PortfolioConstructor(BaseAgent):
    """基于规则和约束构建目标组合。"""

    agent_name = "PortfolioConstructor"
    _NON_POSITION_MODES = {"watch", "reject", "research_only", "avoid", "sell"}
    _ACTION_MULTIPLIER = {
        ActionLabel.BUY: 1.0,
        ActionLabel.HOLD: 0.6,
    }
    _ACTION_ORDER = {
        ActionLabel.AVOID: 0,
        ActionLabel.SELL: 1,
        ActionLabel.WATCH: 2,
        ActionLabel.HOLD: 3,
        ActionLabel.BUY: 4,
    }

    def run(self, payload: Mapping[str, Any]) -> PortfolioPlan:
        envelope = self.ensure_payload(payload)
        self.require_keys(
            envelope,
            "ic_decisions",
            "macro_verdict",
            "risk_limits",
            "existing_portfolio",
            "tradability_snapshot",
        )

        ic_decisions = self._normalize_ic_decisions(envelope["ic_decisions"])
        macro_verdict = envelope["macro_verdict"]
        if not isinstance(macro_verdict, BranchVerdict):
            raise TypeError("macro_verdict 必须是 BranchVerdict")

        risk_limits = self._normalize_risk_limits(envelope["risk_limits"])
        tradability = self._normalize_tradability(envelope["tradability_snapshot"])
        intents, explicit_rejects = self._collect_symbol_intents(ic_decisions, tradability)

        gross_cap = min(
            risk_limits["gross_exposure_cap"],
            float(macro_verdict.metadata.get("target_gross_exposure", 1.0)),
        )
        gross_cap = self.clamp(gross_cap, 0.0, 1.0)
        blocked_symbols = set(risk_limits["blocked_symbols"]) | explicit_rejects
        sector_caps = self._build_sector_caps(risk_limits)

        disallowed_symbols: set[str] = set(blocked_symbols)
        reject_reasons: dict[str, str] = {
            symbol: "risk_blocked_or_explicit_reject"
            for symbol in sorted(disallowed_symbols)
        }
        eligible_intents: dict[str, _SymbolIntent] = {}

        for symbol in sorted(intents):
            intent = intents[symbol]
            tradable_info = tradability.get(symbol, {})
            if intent.position_mode in self._NON_POSITION_MODES:
                disallowed_symbols.add(symbol)
                reject_reasons[symbol] = f"position_mode={intent.position_mode}"
                continue
            if not self._is_tradable(tradable_info):
                disallowed_symbols.add(symbol)
                reject_reasons[symbol] = "not_tradable"
                continue
            if intent.score <= 0.0:
                disallowed_symbols.add(symbol)
                reject_reasons[symbol] = "non_positive_score"
                continue
            if self._resolve_symbol_cap(
                symbol=symbol,
                intent=intent,
                tradability_info=tradable_info,
                risk_limits=risk_limits,
                gross_cap=gross_cap,
            ) <= 0.0:
                disallowed_symbols.add(symbol)
                reject_reasons[symbol] = "zero_symbol_cap"
                continue
            eligible_intents[symbol] = intent

        # 先把现有仓位投影到同一组约束内，再做 turnover 平滑，保证结果可复现。
        baseline_weights = self._feasible_existing_weights(
            existing_portfolio=envelope["existing_portfolio"],
            tradability=tradability,
            risk_limits=risk_limits,
            sector_caps=sector_caps,
            gross_cap=gross_cap,
            disallowed_symbols=disallowed_symbols,
        )

        target_weights = self._allocate_target_weights(
            intents=eligible_intents,
            tradability=tradability,
            risk_limits=risk_limits,
            sector_caps=sector_caps,
            gross_cap=gross_cap,
        )

        turnover_cap = risk_limits.get("turnover_cap")
        turnover_applied = False
        if turnover_cap is not None:
            turnover_cap = max(0.0, float(turnover_cap))
            target_weights, turnover_applied = self._apply_turnover_cap(
                baseline=baseline_weights,
                target=target_weights,
                turnover_cap=turnover_cap,
            )

        theme_cap_metadata: dict[str, Any] = {}
        theme_cap_notes: list[str] = []
        if risk_limits.get("theme_portfolio_cap_enabled"):
            target_weights, theme_cap_metadata, theme_cap_notes = self._apply_theme_caps(
                target_weights,
                risk_limits,
            )

        target_weights = self._cleanup_weights(target_weights)
        target_gross = round(sum(target_weights.values()), 6)
        target_net = target_gross
        turnover_estimate = self._estimate_turnover(baseline_weights, target_weights)
        concentration_metrics = self._build_concentration_metrics(target_weights, tradability)
        if theme_cap_metadata:
            concentration_metrics.update(theme_cap_metadata)

        construction_notes = [
            (
                f"target_weight 仅由 final_score、final_confidence、action multiplier 与约束规则决定，"
                f"gross_cap={gross_cap:.2f}。"
            ),
            "watch/reject/research_only、不可交易标的与显式 blocked symbols 不进入目标仓位。",
        ]
        if turnover_cap is not None:
            if turnover_applied:
                construction_notes.append(
                    f"已按 turnover_cap={turnover_cap:.2f} 对理想组合进行确定性平滑。"
                )
            else:
                construction_notes.append(
                    f"当前理想组合 turnover={turnover_estimate:.2f}，未触发额外 turnover 平滑。"
                )
        if risk_limits["sector_caps"]:
            construction_notes.append("行业权重按 sector_caps 做线性约束。")
        if theme_cap_notes:
            construction_notes.extend(theme_cap_notes)
        construction_notes.append("NarratorAgent 与 IC thesis 不可直接改写 target_weight。")

        metadata = {
            "risk_gross_cap": risk_limits["gross_exposure_cap"],
            "macro_gross_cap": float(macro_verdict.metadata.get("target_gross_exposure", 1.0)),
            "applied_gross_cap": gross_cap,
            "reject_reasons": reject_reasons,
            "baseline_weights": baseline_weights,
            "rule_based": True,
            "deterministic": True,
        }
        if theme_cap_metadata:
            metadata.update(theme_cap_metadata)

        return PortfolioPlan(
            status=AgentStatus.SUCCESS if target_weights else AgentStatus.DEGRADED,
            target_exposure=target_gross,
            target_gross_exposure=target_gross,
            target_net_exposure=target_net,
            cash_ratio=self.clamp(1.0 - target_gross, 0.0, 1.0),
            target_weights=target_weights,
            target_positions=target_weights,
            position_limits=risk_limits["position_limits"],
            blocked_symbols=sorted(disallowed_symbols),
            rejected_symbols=sorted(disallowed_symbols),
            concentration_metrics=concentration_metrics,
            turnover_estimate=turnover_estimate,
            execution_notes=construction_notes,
            construction_notes=construction_notes,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_ic_decisions(payload: Any) -> list[ICDecision]:
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise TypeError("ic_decisions 必须是 ICDecision 列表")
        decisions = [item for item in payload if isinstance(item, ICDecision)]
        if len(decisions) != len(payload):
            raise TypeError("ic_decisions 中存在非 ICDecision 项")
        return decisions

    @staticmethod
    def _normalize_tradability(payload: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise TypeError("tradability_snapshot 必须是 Mapping")
        source = payload.get("symbols") if isinstance(payload.get("symbols"), Mapping) else payload
        result: dict[str, dict[str, Any]] = {}
        for symbol, info in source.items():
            if not isinstance(info, Mapping):
                continue
            result[str(symbol)] = dict(info)
        return result

    def _normalize_risk_limits(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, RiskDecision):
            return {
                "gross_exposure_cap": float(payload.gross_exposure_cap),
                "max_weight": float(payload.max_weight),
                "position_limits": dict(payload.position_limits),
                "blocked_symbols": list(payload.blocked_symbols),
                "sector_caps": {},
                "turnover_cap": None,
                "theme_portfolio_cap_enabled": False,
                "theme_exposure_map": {},
                "theme_caps": {},
                "theme_names": {},
                "theme_phases": {},
                "theme_tactical_lane": {
                    "enabled": False,
                    "status": "disabled",
                    "regime": "",
                    "non_tech_symbols": [],
                    "nav_cap": 0.0,
                    "max_positions": 0,
                    "protocol_hash": "",
                    "formal_kill_switch": True,
                },
                "theme_portfolio_diagnostic_notes": [],
            }
        if not isinstance(payload, Mapping):
            raise TypeError("risk_limits 必须是 Mapping 或 RiskDecision")
        sector_caps_raw = payload.get("sector_caps", {})
        sector_caps = (
            {
                str(sector): self.clamp(float(limit), 0.0, 1.0)
                for sector, limit in sector_caps_raw.items()
            }
            if isinstance(sector_caps_raw, Mapping)
            else {}
        )
        position_limits_raw = payload.get("position_limits", {})
        position_limits = (
            {
                str(symbol): self.clamp(float(limit), 0.0, 1.0)
                for symbol, limit in position_limits_raw.items()
            }
            if isinstance(position_limits_raw, Mapping)
            else {}
        )
        blocked_symbols = sorted(
            {
                str(symbol).strip()
                for symbol in payload.get("blocked_symbols", [])
                if str(symbol).strip()
            }
        )
        turnover_cap = payload.get("turnover_cap")
        theme_enabled = self._truthy(payload.get("theme_portfolio_cap_enabled", False))
        theme_exposure_map: dict[str, dict[str, Any]] = {}
        theme_caps: dict[str, float] = {}
        theme_names: dict[str, str] = {}
        theme_phases: dict[str, str] = {}
        theme_tactical_lane: dict[str, Any] = {
            "enabled": False,
            "status": "disabled",
            "regime": "",
            "non_tech_symbols": [],
            "nav_cap": 0.0,
            "max_positions": 0,
            "protocol_hash": "",
            "formal_kill_switch": True,
        }
        theme_notes: list[str] = []
        if theme_enabled:
            theme_exposure_map, exposure_notes = self._normalize_theme_exposure_map(
                payload.get("theme_exposure_map", {})
            )
            theme_caps, cap_notes = self._normalize_theme_caps(payload.get("theme_caps", {}))
            theme_names = self._normalize_text_mapping(payload.get("theme_names", {}))
            theme_phases = self._normalize_text_mapping(payload.get("theme_phases", {}))
            theme_tactical_lane, tactical_notes = (
                self._normalize_theme_tactical_lane(
                    payload.get("theme_tactical_lane", {})
                )
            )
            theme_notes.extend(exposure_notes)
            theme_notes.extend(cap_notes)
            theme_notes.extend(tactical_notes)
        return {
            "gross_exposure_cap": self.clamp(float(payload.get("gross_exposure_cap", 1.0)), 0.0, 1.0),
            "max_weight": self.clamp(float(payload.get("max_weight", 1.0)), 0.0, 1.0),
            "position_limits": position_limits,
            "blocked_symbols": blocked_symbols,
            "sector_caps": sector_caps,
            "turnover_cap": None if turnover_cap is None else float(turnover_cap),
            "theme_portfolio_cap_enabled": theme_enabled,
            "theme_exposure_map": theme_exposure_map,
            "theme_caps": theme_caps,
            "theme_names": theme_names,
            "theme_phases": theme_phases,
            "theme_tactical_lane": theme_tactical_lane,
            "theme_portfolio_diagnostic_notes": theme_notes,
        }

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _normalize_theme_exposure_map(cls, payload: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
        if not isinstance(payload, Mapping):
            return {}, ["theme_portfolio_cap_malformed_exposure_map"]

        normalized: dict[str, dict[str, Any]] = {}
        for symbol, metadata in payload.items():
            symbol_text = str(symbol or "").strip()
            if not symbol_text or not isinstance(metadata, Mapping):
                continue
            theme_id = str(metadata.get("primary_theme_id") or "").strip()
            if not theme_id:
                continue
            normalized[symbol_text] = {
                "primary_theme_id": theme_id,
                "primary_theme_name": str(metadata.get("primary_theme_name") or ""),
                "phase": str(metadata.get("phase") or ""),
                "symbol_score": cls.clamp(
                    cls._finite_float(metadata.get("symbol_score", 0.0)),
                    0.0,
                    1.0,
                ),
                "risk_flags": cls._normalize_text_list(metadata.get("risk_flags", [])),
            }
        return normalized, []

    @classmethod
    def _normalize_theme_caps(cls, payload: Any) -> tuple[dict[str, float], list[str]]:
        if not isinstance(payload, Mapping):
            return {}, ["theme_portfolio_cap_malformed_caps"]

        normalized: dict[str, float] = {}
        invalid_count = 0
        for theme_id, raw_cap in payload.items():
            theme_text = str(theme_id or "").strip()
            cap = cls._optional_finite_float(raw_cap)
            if not theme_text or cap is None:
                invalid_count += 1
                continue
            normalized[theme_text] = cls.clamp(cap, 0.0, 1.0)

        notes = ["theme_portfolio_cap_malformed_caps"] if invalid_count and not normalized else []
        return normalized, notes

    @classmethod
    def _normalize_theme_tactical_lane(
        cls,
        payload: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        disabled = {
            "enabled": False,
            "status": "disabled",
            "regime": "",
            "non_tech_symbols": [],
            "nav_cap": 0.0,
            "max_positions": 0,
            "protocol_hash": "",
            "formal_kill_switch": True,
        }
        if not payload:
            return disabled, []
        if not isinstance(payload, Mapping):
            return disabled, ["theme_tactical_lane_malformed"]
        status = str(payload.get("status") or "disabled").strip()
        symbols = cls._normalize_text_list(
            payload.get("non_tech_symbols", [])
        )
        nav_cap = cls._optional_finite_float(payload.get("nav_cap"))
        try:
            max_positions = int(payload.get("max_positions", 0) or 0)
        except (TypeError, ValueError):
            max_positions = -1
        protocol_hash = str(payload.get("protocol_hash") or "").strip()
        formal_kill_switch = cls._truthy(
            payload.get("formal_kill_switch", True)
        )
        enforceable = status in {"active", "closed_by_markov"}
        invalid = (
            nav_cap is None
            or not 0.0 <= nav_cap <= 1.0
            or max_positions < 0
            or (enforceable and not protocol_hash)
            or (status == "active" and payload.get("enabled") is not True)
            or (status == "closed_by_markov" and (nav_cap != 0.0 or max_positions != 0))
        )
        if invalid:
            return disabled, ["theme_tactical_lane_malformed"]
        return {
            "enabled": payload.get("enabled") is True,
            "status": status,
            "regime": str(payload.get("regime") or ""),
            "non_tech_symbols": symbols,
            "nav_cap": cls.clamp(float(nav_cap), 0.0, 1.0),
            "max_positions": max_positions,
            "protocol_hash": protocol_hash,
            "formal_kill_switch": formal_kill_switch,
        }, []

    @staticmethod
    def _normalize_text_mapping(payload: Any) -> dict[str, str]:
        if not isinstance(payload, Mapping):
            return {}
        return {
            str(key): str(value or "")
            for key, value in payload.items()
            if str(key or "").strip()
        }

    @staticmethod
    def _normalize_text_list(payload: Any) -> list[str]:
        if isinstance(payload, (str, bytes)):
            return []
        try:
            items = list(payload or [])
        except TypeError:
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    @staticmethod
    def _finite_float(value: Any, default: float = 0.0) -> float:
        numeric = PortfolioConstructor._optional_finite_float(value)
        return default if numeric is None else numeric

    @staticmethod
    def _optional_finite_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _collect_symbol_intents(
        self,
        ic_decisions: Sequence[ICDecision],
        tradability: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, _SymbolIntent], set[str]]:
        aggregated: dict[str, dict[str, Any]] = {}
        explicit_rejects: set[str] = set()

        for decision in ic_decisions:
            for symbol in decision.rejected_symbols:
                text = str(symbol).strip()
                if text:
                    explicit_rejects.add(text)

            for item in self._expand_decision_items(decision):
                symbol = item["symbol"]
                bucket = aggregated.setdefault(
                    symbol,
                    {
                        "scores": [],
                        "confidences": [],
                        "calibrated_confidences": [],
                        "momentum_strengths": [],
                        "fake_breakout_penalties": [],
                        "actions": [],
                        "position_modes": [],
                        "sectors": [],
                    },
                )
                bucket["scores"].append(self.clamp(float(item["score"]), -1.0, 1.0))
                bucket["confidences"].append(self.clamp(float(item["confidence"]), 0.0, 1.0))
                bucket["calibrated_confidences"].append(
                    self.clamp(float(item.get("calibrated_confidence", item["confidence"])), 0.0, 1.0)
                )
                bucket["momentum_strengths"].append(
                    self.clamp(float(item.get("momentum_strength", max(float(item["score"]), 0.0))), 0.0, 1.0)
                )
                bucket["fake_breakout_penalties"].append(
                    self.clamp(float(item.get("fake_breakout_penalty", 0.0)), 0.0, 1.0)
                )
                bucket["actions"].append(self._coerce_action(item["action"]))
                bucket["position_modes"].append(self._normalize_position_mode(item["position_mode"]))
                sector = str(item.get("sector") or tradability.get(symbol, {}).get("sector") or "unknown")
                bucket["sectors"].append(sector)

        intents: dict[str, _SymbolIntent] = {}
        for symbol in sorted(aggregated):
            payload = aggregated[symbol]
            score = sum(payload["scores"]) / len(payload["scores"])
            confidence = sum(payload["confidences"]) / len(payload["confidences"])
            calibrated_confidence = sum(payload["calibrated_confidences"]) / len(payload["calibrated_confidences"])
            momentum_strength = sum(payload["momentum_strengths"]) / len(payload["momentum_strengths"])
            fake_breakout_penalty = sum(payload["fake_breakout_penalties"]) / len(payload["fake_breakout_penalties"])
            action = self._merge_action(payload["actions"])
            position_mode = self._merge_position_mode(payload["position_modes"], action, symbol, explicit_rejects)
            sector = sorted(str(item) for item in payload["sectors"] if str(item).strip())[0]
            intents[symbol] = _SymbolIntent(
                symbol=symbol,
                score=self.clamp(score, -1.0, 1.0),
                confidence=self.clamp(confidence, 0.0, 1.0),
                calibrated_confidence=self.clamp(calibrated_confidence, 0.0, 1.0),
                momentum_strength=self.clamp(momentum_strength, 0.0, 1.0),
                fake_breakout_penalty=self.clamp(fake_breakout_penalty, 0.0, 1.0),
                action=action,
                position_mode=position_mode,
                sector=sector,
            )
        return intents, explicit_rejects

    def _expand_decision_items(self, decision: ICDecision) -> list[dict[str, Any]]:
        metadata = decision.metadata if isinstance(decision.metadata, Mapping) else {}
        items: list[dict[str, Any]] = []

        symbol_candidates = metadata.get("symbol_candidates")
        if isinstance(symbol_candidates, Sequence) and not isinstance(symbol_candidates, (str, bytes)):
            for candidate in symbol_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                symbol = str(candidate.get("symbol", "")).strip()
                if not symbol:
                    continue
                items.append(
                    {
                        "symbol": symbol,
                        "score": candidate.get("score", decision.final_score),
                        "confidence": candidate.get("confidence", decision.final_confidence),
                        "calibrated_confidence": candidate.get("calibrated_confidence", candidate.get("confidence", decision.final_confidence)),
                        "momentum_strength": candidate.get("momentum_strength", max(float(candidate.get("score", decision.final_score)), 0.0)),
                        "fake_breakout_penalty": candidate.get("fake_breakout_penalty", 0.0),
                        "action": candidate.get("action", decision.action),
                        "position_mode": candidate.get("position_mode", self._default_position_mode(decision.action)),
                        "sector": candidate.get("sector", ""),
                    }
                )

        symbol_keys: set[str] = set()
        meta_symbol = str(metadata.get("symbol", "")).strip()
        if meta_symbol:
            symbol_keys.add(meta_symbol)
        symbol_keys.update(str(symbol).strip() for symbol in decision.selected_symbols if str(symbol).strip())

        for field_name in (
            "symbol_scores",
            "symbol_confidences",
            "symbol_actions",
            "symbol_modes",
            "symbol_sectors",
        ):
            field_value = metadata.get(field_name)
            if isinstance(field_value, Mapping):
                symbol_keys.update(str(symbol).strip() for symbol in field_value if str(symbol).strip())

        symbol_scores = metadata.get("symbol_scores", {})
        symbol_confidences = metadata.get("symbol_confidences", {})
        symbol_calibrated_confidences = metadata.get("symbol_calibrated_confidences", {})
        symbol_momentum_strengths = metadata.get("symbol_momentum_strengths", {})
        symbol_fake_breakout_penalties = metadata.get("symbol_fake_breakout_penalties", {})
        symbol_actions = metadata.get("symbol_actions", {})
        symbol_modes = metadata.get("symbol_modes", {})
        symbol_sectors = metadata.get("symbol_sectors", {})

        for symbol in sorted(symbol_keys):
            items.append(
                {
                    "symbol": symbol,
                    "score": symbol_scores.get(symbol, decision.final_score)
                    if isinstance(symbol_scores, Mapping)
                    else decision.final_score,
                    "confidence": symbol_confidences.get(symbol, decision.final_confidence)
                    if isinstance(symbol_confidences, Mapping)
                    else decision.final_confidence,
                    "calibrated_confidence": symbol_calibrated_confidences.get(symbol, decision.final_confidence)
                    if isinstance(symbol_calibrated_confidences, Mapping)
                    else decision.final_confidence,
                    "momentum_strength": symbol_momentum_strengths.get(symbol, max(float(decision.final_score), 0.0))
                    if isinstance(symbol_momentum_strengths, Mapping)
                    else max(float(decision.final_score), 0.0),
                    "fake_breakout_penalty": symbol_fake_breakout_penalties.get(symbol, 0.0)
                    if isinstance(symbol_fake_breakout_penalties, Mapping)
                    else 0.0,
                    "action": symbol_actions.get(symbol, decision.action)
                    if isinstance(symbol_actions, Mapping)
                    else decision.action,
                    "position_mode": symbol_modes.get(
                        symbol,
                        self._default_position_mode(
                            symbol_actions.get(symbol, decision.action)
                            if isinstance(symbol_actions, Mapping)
                            else decision.action
                        ),
                    )
                    if isinstance(symbol_modes, Mapping)
                    else self._default_position_mode(decision.action),
                    "sector": symbol_sectors.get(symbol, "")
                    if isinstance(symbol_sectors, Mapping)
                    else "",
                }
            )

        # 去重并保持稳定排序。
        deduped: dict[str, dict[str, Any]] = {}
        for item in items:
            deduped[item["symbol"]] = item
        return [deduped[symbol] for symbol in sorted(deduped)]

    @classmethod
    def _coerce_action(cls, value: ActionLabel | str) -> ActionLabel:
        return value if isinstance(value, ActionLabel) else ActionLabel(str(value).strip().lower())

    @staticmethod
    def _normalize_position_mode(value: Any) -> str:
        text = str(value or "").strip().lower()
        return text or "target"

    @classmethod
    def _default_position_mode(cls, action: ActionLabel | str) -> str:
        label = cls._coerce_action(action)
        if label is ActionLabel.WATCH:
            return "watch"
        if label in {ActionLabel.SELL, ActionLabel.AVOID}:
            return "reject"
        return "target"

    @classmethod
    def _merge_action(cls, actions: Sequence[ActionLabel]) -> ActionLabel:
        ranked = sorted(actions, key=lambda action: (cls._ACTION_ORDER[action], action.value))
        return ranked[0]

    @classmethod
    def _merge_position_mode(
        cls,
        position_modes: Sequence[str],
        action: ActionLabel,
        symbol: str,
        explicit_rejects: set[str],
    ) -> str:
        modes = {cls._normalize_position_mode(item) for item in position_modes}
        if symbol in explicit_rejects or "reject" in modes:
            return "reject"
        if "research_only" in modes:
            return "research_only"
        if "watch" in modes or action is ActionLabel.WATCH:
            return "watch"
        if action in {ActionLabel.SELL, ActionLabel.AVOID}:
            return "reject"
        return "target"

    @staticmethod
    def _is_tradable(info: Mapping[str, Any]) -> bool:
        if not info:
            return True
        if info.get("is_tradable") is False or info.get("tradable") is False:
            return False
        if info.get("halted") or info.get("suspended"):
            return False
        return True

    @staticmethod
    def _build_sector_caps(risk_limits: Mapping[str, Any]) -> dict[str, float]:
        return {
            str(sector): float(limit)
            for sector, limit in risk_limits.get("sector_caps", {}).items()
        }

    def _resolve_symbol_cap(
        self,
        symbol: str,
        intent: _SymbolIntent | None,
        tradability_info: Mapping[str, Any],
        risk_limits: Mapping[str, Any],
        gross_cap: float,
    ) -> float:
        cap = float(risk_limits["max_weight"])
        cap = min(cap, float(risk_limits["position_limits"].get(symbol, cap)))

        for key in (
            "max_weight",
            "position_cap",
            "max_liquidity_weight",
            "liquidity_cap",
            "tradable_weight_cap",
        ):
            if tradability_info.get(key) is not None:
                cap = min(cap, float(tradability_info[key]))

        liquidity_score = tradability_info.get("liquidity_score")
        if liquidity_score is not None:
            cap = min(cap, self.clamp(float(liquidity_score), 0.0, 1.0) * gross_cap)

        if intent and intent.action not in self._ACTION_MULTIPLIER:
            return 0.0
        if not self._is_tradable(tradability_info):
            return 0.0
        return self.clamp(cap, 0.0, 1.0)

    def _feasible_existing_weights(
        self,
        existing_portfolio: Any,
        tradability: Mapping[str, Mapping[str, Any]],
        risk_limits: Mapping[str, Any],
        sector_caps: Mapping[str, float],
        gross_cap: float,
        disallowed_symbols: set[str],
    ) -> dict[str, float]:
        if not isinstance(existing_portfolio, Mapping):
            raise TypeError("existing_portfolio 必须是 Mapping")

        weights_source = (
            existing_portfolio.get("current_weights")
            or existing_portfolio.get("positions")
            or existing_portfolio.get("target_positions")
            or existing_portfolio.get("target_weights")
            or {}
        )
        if not isinstance(weights_source, Mapping):
            return {}

        feasible: dict[str, float] = {}
        for symbol in sorted(str(item) for item in weights_source):
            if symbol in disallowed_symbols:
                continue
            raw_weight = float(weights_source[symbol])
            if raw_weight <= 0.0:
                continue
            tradability_info = tradability.get(symbol, {})
            cap = self._resolve_symbol_cap(
                symbol=symbol,
                intent=None,
                tradability_info=tradability_info,
                risk_limits=risk_limits,
                gross_cap=gross_cap,
            )
            if cap <= 0.0:
                continue
            feasible[symbol] = min(raw_weight, cap)

        feasible = self._enforce_sector_caps(feasible, tradability, sector_caps)
        total = sum(feasible.values())
        if total > gross_cap and total > 0.0:
            scale = gross_cap / total
            feasible = {symbol: round(weight * scale, 6) for symbol, weight in feasible.items()}
        return self._cleanup_weights(feasible)

    def _allocate_target_weights(
        self,
        intents: Mapping[str, _SymbolIntent],
        tradability: Mapping[str, Mapping[str, Any]],
        risk_limits: Mapping[str, Any],
        sector_caps: Mapping[str, float],
        gross_cap: float,
    ) -> dict[str, float]:
        if gross_cap <= 0.0 or not intents:
            return {}

        strengths: dict[str, float] = {}
        symbol_caps: dict[str, float] = {}
        sectors: dict[str, str] = {}
        for symbol in sorted(intents):
            intent = intents[symbol]
            confidence_term = max(intent.calibrated_confidence, intent.confidence)
            strength = max(intent.momentum_strength, max(intent.score, 0.0))
            strength *= 0.35 + 0.65 * confidence_term
            strength *= 1.0 - min(intent.fake_breakout_penalty, 0.80) * 0.45
            strength *= self._ACTION_MULTIPLIER.get(intent.action, 0.0)
            cap = self._resolve_symbol_cap(
                symbol=symbol,
                intent=intent,
                tradability_info=tradability.get(symbol, {}),
                risk_limits=risk_limits,
                gross_cap=gross_cap,
            )
            if strength <= 0.0 or cap <= 0.0:
                continue
            strengths[symbol] = strength
            symbol_caps[symbol] = cap
            sectors[symbol] = intent.sector

        if not strengths:
            return {}

        base_weights = LegacyPortfolioConstructor.score_weight(
            pd.Series(strengths, dtype=float),
            n_top=len(strengths),
        ).to_dict()
        weights: dict[str, float] = {symbol: 0.0 for symbol in sorted(base_weights)}
        remaining_gross = gross_cap
        remaining_symbols = set(weights)
        sector_allocations = {sector: 0.0 for sector in sorted(sector_caps)}

        while remaining_symbols and remaining_gross > 1e-8:
            active = sorted(remaining_symbols)
            weight_sum = sum(base_weights[symbol] for symbol in active)
            if weight_sum <= 0.0:
                break

            progress = 0.0
            exhausted: set[str] = set()
            for symbol in active:
                sector = sectors[symbol]
                sector_cap = float(sector_caps.get(sector, gross_cap))
                sector_room = max(0.0, sector_cap - sector_allocations.get(sector, 0.0))
                symbol_room = max(0.0, symbol_caps[symbol] - weights[symbol])
                if sector_room <= 1e-8 or symbol_room <= 1e-8:
                    exhausted.add(symbol)
                    continue

                proposed = remaining_gross * (base_weights[symbol] / weight_sum)
                allocation = min(proposed, sector_room, symbol_room)
                if allocation <= 1e-8:
                    exhausted.add(symbol)
                    continue

                weights[symbol] += allocation
                sector_allocations[sector] = sector_allocations.get(sector, 0.0) + allocation
                progress += allocation
                if symbol_caps[symbol] - weights[symbol] <= 1e-8:
                    exhausted.add(symbol)
                if sector_cap - sector_allocations[sector] <= 1e-8:
                    exhausted.update(
                        other
                        for other in active
                        if sectors[other] == sector
                    )

            if progress <= 1e-8:
                break

            remaining_gross = max(0.0, gross_cap - sum(weights.values()))
            remaining_symbols -= exhausted

        weights = self._enforce_sector_caps(weights, tradability, sector_caps)
        return self._cleanup_weights(weights)

    @staticmethod
    def _enforce_sector_caps(
        weights: Mapping[str, float],
        tradability: Mapping[str, Mapping[str, Any]],
        sector_caps: Mapping[str, float],
    ) -> dict[str, float]:
        adjusted = {str(symbol): float(weight) for symbol, weight in weights.items() if float(weight) > 0.0}
        if not sector_caps:
            return adjusted

        grouped: dict[str, list[str]] = {}
        for symbol in sorted(adjusted):
            sector = str(tradability.get(symbol, {}).get("sector") or "unknown")
            grouped.setdefault(sector, []).append(symbol)

        for sector in sorted(grouped):
            cap = sector_caps.get(sector)
            if cap is None:
                continue
            total = sum(adjusted[symbol] for symbol in grouped[sector])
            if total <= cap or total <= 0.0:
                continue
            scale = cap / total
            for symbol in grouped[sector]:
                adjusted[symbol] = round(adjusted[symbol] * scale, 6)
        return adjusted

    def _apply_theme_caps(
        self,
        weights: Mapping[str, float],
        risk_limits: Mapping[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any], list[str]]:
        adjusted = {
            str(symbol): float(weight)
            for symbol, weight in weights.items()
            if float(weight) > 0.0
        }
        exposure_map = risk_limits.get("theme_exposure_map", {})
        theme_caps = risk_limits.get("theme_caps", {})
        theme_names = risk_limits.get("theme_names", {})
        theme_phases = risk_limits.get("theme_phases", {})
        tactical_lane = risk_limits.get("theme_tactical_lane", {})
        diagnostic_notes = [
            str(note)
            for note in risk_limits.get("theme_portfolio_diagnostic_notes", [])
            if str(note).strip()
        ]
        notes = list(diagnostic_notes)

        if "theme_tactical_lane_malformed" in notes:
            dropped = sorted(adjusted)
            notes.append(
                "theme_tactical_lane_fail_closed: malformed active contract cleared "
                "all Theme-governed target weights"
            )
            metadata = {
                "theme_portfolio_cap_enabled": True,
                "theme_exposures_before": self._theme_exposures(
                    adjusted,
                    exposure_map,
                ),
                "theme_exposures_after": {},
                "theme_caps": dict(theme_caps)
                if isinstance(theme_caps, Mapping)
                else {},
                "theme_names": dict(theme_names)
                if isinstance(theme_names, Mapping)
                else {},
                "theme_phases": dict(theme_phases)
                if isinstance(theme_phases, Mapping)
                else {},
                "theme_exposure_map": dict(exposure_map)
                if isinstance(exposure_map, Mapping)
                else {},
                "theme_cap_applied_count": 0,
                "theme_portfolio_diagnostic_notes": diagnostic_notes,
                "theme_tactical_lane": {
                    "status": "blocked_malformed",
                    "applied": True,
                    "dropped_symbols": dropped,
                    "non_tech_symbols": [],
                    "exposure_before": round(sum(adjusted.values()), 6),
                    "exposure_after": 0.0,
                    "position_count_before": len(dropped),
                    "position_count_after": 0,
                },
            }
            return {}, metadata, notes

        exposures_before = self._theme_exposures(adjusted, exposure_map)
        applied_count = 0

        grouped: dict[str, list[str]] = {}
        if isinstance(exposure_map, Mapping) and isinstance(theme_caps, Mapping):
            for symbol in sorted(adjusted):
                metadata = exposure_map.get(symbol)
                if not isinstance(metadata, Mapping):
                    continue
                theme_id = str(metadata.get("primary_theme_id") or "").strip()
                if theme_id and theme_id in theme_caps:
                    grouped.setdefault(theme_id, []).append(symbol)

        for theme_id in sorted(grouped):
            cap = self._optional_finite_float(theme_caps.get(theme_id))
            if cap is None:
                continue
            cap = self.clamp(cap, 0.0, 1.0)
            total = sum(adjusted[symbol] for symbol in grouped[theme_id])
            if total <= cap + 1e-8 or total <= 0.0:
                continue
            scale = cap / total
            for symbol in grouped[theme_id]:
                adjusted[symbol] = round(adjusted[symbol] * scale, 6)
            applied_count += 1
            after = sum(adjusted[symbol] for symbol in grouped[theme_id])
            if after > cap + 1e-8:
                overflow = after - cap
                trim_symbol = sorted(
                    grouped[theme_id],
                    key=lambda symbol: (-adjusted[symbol], symbol),
                )[0]
                adjusted[trim_symbol] = round(max(0.0, adjusted[trim_symbol] - overflow), 6)
                after = sum(adjusted[symbol] for symbol in grouped[theme_id])
            notes.append(
                f"theme_portfolio_cap_applied: {theme_id} "
                f"cap={cap:.2f} before={total:.2f} after={after:.2f}"
            )

        tactical_metadata, tactical_notes = self._apply_theme_tactical_lane(
            adjusted,
            tactical_lane,
        )
        adjusted = tactical_metadata.pop("adjusted_weights")
        notes.extend(tactical_notes)

        exposures_after = self._theme_exposures(adjusted, exposure_map)
        metadata = {
            "theme_portfolio_cap_enabled": True,
            "theme_exposures_before": exposures_before,
            "theme_exposures_after": exposures_after,
            "theme_caps": dict(theme_caps) if isinstance(theme_caps, Mapping) else {},
            "theme_names": dict(theme_names) if isinstance(theme_names, Mapping) else {},
            "theme_phases": dict(theme_phases) if isinstance(theme_phases, Mapping) else {},
            "theme_exposure_map": dict(exposure_map) if isinstance(exposure_map, Mapping) else {},
            "theme_cap_applied_count": applied_count,
            "theme_portfolio_diagnostic_notes": diagnostic_notes,
            "theme_tactical_lane": tactical_metadata,
        }
        return adjusted, metadata, notes

    def _apply_theme_tactical_lane(
        self,
        weights: Mapping[str, float],
        payload: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        adjusted = {
            str(symbol): float(weight)
            for symbol, weight in weights.items()
            if float(weight) > 0.0
        }
        lane = dict(payload or {}) if isinstance(payload, Mapping) else {}
        status = str(lane.get("status") or "disabled")
        formal_kill_switch = bool(lane.get("formal_kill_switch", True))
        enforce = (
            status in {"active", "closed_by_markov"}
            and not formal_kill_switch
            and bool(str(lane.get("protocol_hash") or ""))
        )
        tactical_symbols = sorted(
            {
                str(symbol)
                for symbol in lane.get("non_tech_symbols", []) or []
                if str(symbol) in adjusted
            }
        )
        before_weight = sum(adjusted[symbol] for symbol in tactical_symbols)
        before_count = sum(
            1 for symbol in tactical_symbols if adjusted[symbol] > 1e-8
        )
        notes: list[str] = []
        dropped_symbols: list[str] = []
        if enforce:
            nav_cap = self.clamp(float(lane.get("nav_cap", 0.0)), 0.0, 1.0)
            max_positions = max(int(lane.get("max_positions", 0) or 0), 0)
            ranked = sorted(
                tactical_symbols,
                key=lambda symbol: (-adjusted[symbol], symbol),
            )
            keep = set(ranked[:max_positions])
            dropped_symbols = [symbol for symbol in ranked if symbol not in keep]
            for symbol in dropped_symbols:
                adjusted[symbol] = 0.0
            kept_total = sum(adjusted[symbol] for symbol in keep)
            if kept_total > nav_cap + 1e-8 and kept_total > 0.0:
                scale = nav_cap / kept_total
                for symbol in keep:
                    adjusted[symbol] = round(adjusted[symbol] * scale, 6)
            notes.append(
                "theme_tactical_lane_applied: "
                f"regime={lane.get('regime') or 'unknown'} "
                f"nav_cap={nav_cap:.2f} max_positions={max_positions}"
            )
        after_weight = sum(
            adjusted.get(symbol, 0.0) for symbol in tactical_symbols
        )
        after_count = sum(
            1
            for symbol in tactical_symbols
            if adjusted.get(symbol, 0.0) > 1e-8
        )
        return {
            "adjusted_weights": adjusted,
            "status": status,
            "applied": enforce,
            "regime": str(lane.get("regime") or ""),
            "nav_cap": float(lane.get("nav_cap", 0.0) or 0.0),
            "max_positions": int(lane.get("max_positions", 0) or 0),
            "protocol_hash": str(lane.get("protocol_hash") or ""),
            "non_tech_symbols": tactical_symbols,
            "dropped_symbols": dropped_symbols,
            "exposure_before": round(before_weight, 6),
            "exposure_after": round(after_weight, 6),
            "position_count_before": before_count,
            "position_count_after": after_count,
        }, notes

    @staticmethod
    def _theme_exposures(
        weights: Mapping[str, float],
        exposure_map: Any,
    ) -> dict[str, float]:
        if not isinstance(exposure_map, Mapping):
            return {}

        totals: dict[str, float] = {}
        for symbol, weight in weights.items():
            metadata = exposure_map.get(symbol)
            if not isinstance(metadata, Mapping):
                continue
            theme_id = str(metadata.get("primary_theme_id") or "").strip()
            if not theme_id:
                continue
            totals[theme_id] = totals.get(theme_id, 0.0) + float(weight)
        return {
            theme_id: round(total, 6)
            for theme_id, total in sorted(totals.items())
            if total > 1e-8
        }

    def _apply_turnover_cap(
        self,
        baseline: Mapping[str, float],
        target: Mapping[str, float],
        turnover_cap: float,
    ) -> tuple[dict[str, float], bool]:
        turnover = self._estimate_turnover(baseline, target)
        if turnover <= turnover_cap + 1e-8:
            return dict(target), False
        if turnover <= 0.0:
            return dict(target), False

        blend = turnover_cap / turnover
        symbols = sorted(set(baseline) | set(target))
        adjusted = {
            symbol: round(
                float(baseline.get(symbol, 0.0))
                + blend * (float(target.get(symbol, 0.0)) - float(baseline.get(symbol, 0.0))),
                6,
            )
            for symbol in symbols
        }
        return self._cleanup_weights(adjusted), True

    @staticmethod
    def _estimate_turnover(before: Mapping[str, float], after: Mapping[str, float]) -> float:
        symbols = set(before) | set(after)
        gross_change = sum(abs(float(after.get(symbol, 0.0)) - float(before.get(symbol, 0.0))) for symbol in symbols)
        return round(gross_change / 2.0, 6)

    @staticmethod
    def _cleanup_weights(weights: Mapping[str, float]) -> dict[str, float]:
        return {
            str(symbol): round(float(weight), 6)
            for symbol, weight in sorted(weights.items())
            if float(weight) > 1e-8
        }

    @staticmethod
    def _build_concentration_metrics(
        weights: Mapping[str, float],
        tradability: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, float]:
        ordered = sorted((float(weight) for weight in weights.values()), reverse=True)
        if not ordered:
            return {
                "top1_weight": 0.0,
                "top3_weight": 0.0,
                "top5_weight": 0.0,
                "hhi": 0.0,
                "effective_n": 0.0,
                "max_sector_weight": 0.0,
            }

        sector_totals: dict[str, float] = {}
        for symbol, weight in weights.items():
            sector = str(tradability.get(symbol, {}).get("sector") or "unknown")
            sector_totals[sector] = sector_totals.get(sector, 0.0) + float(weight)

        hhi = sum(weight * weight for weight in ordered)
        effective_n = 0.0 if hhi <= 0.0 else 1.0 / hhi
        return {
            "top1_weight": round(sum(ordered[:1]), 6),
            "top3_weight": round(sum(ordered[:3]), 6),
            "top5_weight": round(sum(ordered[:5]), 6),
            "hhi": round(hhi, 6),
            "effective_n": round(effective_n, 6),
            "max_sector_weight": round(max(sector_totals.values()), 6),
        }

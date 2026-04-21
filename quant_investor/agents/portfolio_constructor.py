"""
确定性的 PortfolioConstructor。

只消费结构化 ICDecision、宏观约束、风险上限与可交易快照，
不读取 Narrator 或自由文本来驱动目标权重。
"""

from __future__ import annotations

from dataclasses import dataclass
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

        target_weights = self._cleanup_weights(target_weights)
        target_gross = round(sum(target_weights.values()), 6)
        target_net = target_gross
        turnover_estimate = self._estimate_turnover(baseline_weights, target_weights)
        concentration_metrics = self._build_concentration_metrics(target_weights, tradability)

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
        construction_notes.append("NarratorAgent 与 IC thesis 不可直接改写 target_weight。")

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
            metadata={
                "risk_gross_cap": risk_limits["gross_exposure_cap"],
                "macro_gross_cap": float(macro_verdict.metadata.get("target_gross_exposure", 1.0)),
                "applied_gross_cap": gross_cap,
                "reject_reasons": reject_reasons,
                "baseline_weights": baseline_weights,
                "rule_based": True,
                "deterministic": True,
            },
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
        return {
            "gross_exposure_cap": self.clamp(float(payload.get("gross_exposure_cap", 1.0)), 0.0, 1.0),
            "max_weight": self.clamp(float(payload.get("max_weight", 1.0)), 0.0, 1.0),
            "position_limits": position_limits,
            "blocked_symbols": blocked_symbols,
            "sector_caps": sector_caps,
            "turnover_cap": None if turnover_cap is None else float(turnover_cap),
        }

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

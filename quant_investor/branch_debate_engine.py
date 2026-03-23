#!/usr/bin/env python3
"""
V9 branch-local debate engine。

当前默认实现遵循两条硬约束：
1. 没有 provider / key 时自动降级为 neutral verdict
2. 即使有 provider，也只允许对 base result 做 bounded adjustment
"""

from __future__ import annotations

import os
from typing import Any, Callable

from quant_investor.branch_contracts import BranchResult, DebateVerdict, EvidencePacket
from quant_investor.debate_templates import BRANCH_DEBATE_ADJUSTMENT_CAPS, build_debate_prompt
from quant_investor.versioning import DEBATE_TEMPLATE_VERSION

ALLOWED_DEBATE_STATUSES = {"llm", "deterministic_stub", "skipped", "timed_out", "error"}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class BranchDebateEngine:
    """各分支共用的 branch-local debate engine。"""

    def __init__(
        self,
        enabled: bool = True,
        model: str = "gpt-5.4-mini",
        timeout_sec: float = 8.0,
        min_abs_score: float = 0.08,
        responder: Callable[..., dict[str, Any] | DebateVerdict] | None = None,
    ) -> None:
        self.enabled = enabled
        self.model = model
        self.timeout_sec = max(float(timeout_sec), 0.1)
        self.min_abs_score = max(float(min_abs_score), 0.0)
        self.responder = responder

    @staticmethod
    def _has_provider() -> bool:
        provider_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "GOOGLE_API_KEY",
        ]
        return any(bool(os.getenv(key)) for key in provider_keys)

    @staticmethod
    def _neutral_verdict(reason: str, used_features: list[str] | None = None) -> DebateVerdict:
        status = BranchDebateEngine._normalize_status("", reason, default="skipped")
        return DebateVerdict(
            direction="neutral",
            confidence=0.0,
            score_adjustment=0.0,
            risk_flags=[],
            unknowns=[reason],
            used_features=list(used_features or []),
            hard_veto=False,
            metadata={
                "status": status,
                "reason": reason,
                "template_version": DEBATE_TEMPLATE_VERSION,
            },
        )

    @staticmethod
    def _normalize_status(status: str, reason: str, default: str = "llm") -> str:
        normalized = str(status or "").strip().lower()
        if normalized in ALLOWED_DEBATE_STATUSES:
            return normalized
        reason_text = str(reason or "").strip().lower()
        if "timeout" in normalized or "timeout" in reason_text:
            return "timed_out"
        if normalized in {"degraded", "scheduled"}:
            return default if default in ALLOWED_DEBATE_STATUSES else "skipped"
        if reason_text in {
            "branch_debate_disabled",
            "llm_provider_missing",
            "scheduler_gate_not_met",
            "scheduler_budget_exhausted",
            "branch_unsuccessful",
        }:
            return "skipped"
        if default in ALLOWED_DEBATE_STATUSES:
            return default
        return "skipped"

    @staticmethod
    def _narrative_fields(
        branch_name: str,
        evidence: EvidencePacket,
        base_result: BranchResult,
        direction: str,
    ) -> dict[str, str]:
        summary = str(evidence.summary or base_result.explanation or f"{branch_name} 分支完成补充复核。")
        bull_case = "；".join(str(item) for item in (evidence.bull_points or [])[:2]) or "暂无新增看多补充。"
        bear_case = "；".join(str(item) for item in (evidence.bear_points or evidence.risk_points or [])[:2]) or "暂无新增看空补充。"
        net_conclusion = {
            "bullish": f"{branch_name} 分支复核后维持偏多结论。",
            "bearish": f"{branch_name} 分支复核后维持偏谨慎结论。",
            "neutral": f"{branch_name} 分支复核后维持中性结论。",
        }.get(direction, f"{branch_name} 分支复核后维持中性结论。")
        return {
            "summary": summary,
            "bull_case": bull_case,
            "bear_case": bear_case,
            "net_conclusion": net_conclusion,
        }

    def _finalize_verdict(
        self,
        branch_name: str,
        evidence: EvidencePacket,
        base_result: BranchResult,
        verdict: DebateVerdict,
        default_status: str,
    ) -> DebateVerdict:
        verdict.metadata = dict(verdict.metadata)
        verdict.metadata["status"] = self._normalize_status(
            str(verdict.metadata.get("status", "")),
            str(verdict.metadata.get("reason", "")),
            default=default_status,
        )
        for key, value in self._narrative_fields(branch_name, evidence, base_result, verdict.direction).items():
            verdict.metadata.setdefault(key, value)
        verdict.metadata.setdefault("template_version", DEBATE_TEMPLATE_VERSION)
        return verdict

    def evaluate(
        self,
        branch_name: str,
        evidence: EvidencePacket,
        base_result: BranchResult,
    ) -> DebateVerdict:
        """生成分支 debate verdict。"""
        if not self.enabled:
            return self._neutral_verdict("branch_debate_disabled", evidence.used_features)

        if self.responder is None and not self._has_provider():
            return self._neutral_verdict("llm_provider_missing", evidence.used_features)

        payload: dict[str, Any] | DebateVerdict
        if self.responder is not None:
            try:
                payload = self.responder(
                    branch_name=branch_name,
                    evidence=evidence,
                    base_result=base_result,
                    model=self.model,
                    timeout_sec=self.timeout_sec,
                    prompt=build_debate_prompt(branch_name, evidence),
                )
            except TimeoutError:
                payload = self._neutral_verdict("timeout", evidence.used_features)
                payload.metadata["status"] = "timed_out"
            except Exception as exc:
                payload = self._neutral_verdict("responder_error", evidence.used_features)
                payload.metadata["status"] = self._normalize_status("", str(exc), default="error")
                payload.metadata["reason"] = str(exc) if "timeout" in str(exc).lower() else "responder_error"
        else:
            payload = self._deterministic_stub(branch_name, evidence, base_result)

        if isinstance(payload, DebateVerdict):
            verdict = payload
            default_status = "deterministic_stub" if self.responder is None else "llm"
        else:
            verdict = self._from_dict(payload)
            default_status = "llm" if self.responder is not None else "deterministic_stub"
        verdict = self._finalize_verdict(branch_name, evidence, base_result, verdict, default_status)
        return self._bound_verdict(branch_name, verdict, base_result)

    def _deterministic_stub(
        self,
        branch_name: str,
        evidence: EvidencePacket,
        base_result: BranchResult,
    ) -> DebateVerdict:
        """没有真实 provider wiring 时的保底 bounded reviewer。"""
        cap = BRANCH_DEBATE_ADJUSTMENT_CAPS.get(branch_name, 0.10)
        bull_strength = min(len(evidence.bull_points) / 4, 1.0)
        bear_strength = min(len(evidence.bear_points) / 4, 1.0)
        risk_drag = min(len(evidence.risk_points) / 5, 1.0)
        net_support = bull_strength - bear_strength - 0.4 * risk_drag
        direction = "neutral"
        if net_support > 0.15:
            direction = "bullish"
        elif net_support < -0.15:
            direction = "bearish"

        base_score = float(base_result.base_score if base_result.base_score is not None else base_result.score)
        disagreement = base_score * net_support < 0
        adjustment_scale = 0.45 if abs(base_score) >= self.min_abs_score else 0.20
        score_adjustment = _clamp(net_support * cap * adjustment_scale, -cap, cap)

        if disagreement:
            score_adjustment *= 0.65

        hard_veto = bool(
            base_score > 0.35
            and bear_strength > 0.65
            and len(evidence.risk_points) >= 2
        )

        risk_flags = list(evidence.risk_points[:3])
        if disagreement:
            risk_flags.append("branch_local_debate_detected_disagreement")

        confidence = _clamp(
            0.35 + 0.25 * max(bull_strength, bear_strength) - 0.10 * len(evidence.unknowns),
            0.0,
            0.75,
        )
        return DebateVerdict(
            direction=direction,
            confidence=confidence,
            score_adjustment=score_adjustment,
            bull_points=list(evidence.bull_points[:3]),
            bear_points=list(evidence.bear_points[:3]),
            risk_flags=risk_flags,
            unknowns=list(evidence.unknowns[:3]),
            used_features=list(evidence.used_features[:8]),
            hard_veto=hard_veto,
            metadata={
                "status": "deterministic_stub",
                "model": self.model,
                "template_version": DEBATE_TEMPLATE_VERSION,
                **self._narrative_fields(branch_name, evidence, base_result, direction),
            },
        )

    @staticmethod
    def _from_dict(payload: dict[str, Any]) -> DebateVerdict:
        metadata = dict(payload.get("metadata", {}))
        for key in ("summary", "bull_case", "bear_case", "net_conclusion"):
            if key in payload and key not in metadata:
                metadata[key] = str(payload.get(key, ""))
        return DebateVerdict(
            direction=str(payload.get("direction", "neutral")),
            confidence=float(payload.get("confidence", 0.0)),
            score_adjustment=float(payload.get("score_adjustment", 0.0)),
            bull_points=[str(item) for item in payload.get("bull_points", [])],
            bear_points=[str(item) for item in payload.get("bear_points", [])],
            risk_flags=[str(item) for item in payload.get("risk_flags", [])],
            unknowns=[str(item) for item in payload.get("unknowns", [])],
            used_features=[str(item) for item in payload.get("used_features", [])],
            hard_veto=bool(payload.get("hard_veto", False)),
            metadata=metadata,
        )

    def _bound_verdict(
        self,
        branch_name: str,
        verdict: DebateVerdict,
        base_result: BranchResult,
    ) -> DebateVerdict:
        cap = BRANCH_DEBATE_ADJUSTMENT_CAPS.get(branch_name, 0.10)
        base_score = float(base_result.base_score if base_result.base_score is not None else base_result.score)
        bounded_adjustment = _clamp(float(verdict.score_adjustment), -cap, cap)

        if base_score > 0 and base_score + bounded_adjustment < -0.05:
            bounded_adjustment = max(-base_score, -cap)
        if base_score < 0 and base_score + bounded_adjustment > 0.05:
            bounded_adjustment = min(-base_score, cap)

        verdict.score_adjustment = bounded_adjustment
        verdict.confidence = _clamp(float(verdict.confidence), 0.0, 1.0)
        verdict.direction = verdict.direction if verdict.direction in {"bullish", "bearish", "neutral"} else "neutral"
        verdict.metadata["status"] = self._normalize_status(
            str(verdict.metadata.get("status", "")),
            str(verdict.metadata.get("reason", "")),
            default="llm",
        )
        verdict.metadata.setdefault("template_version", DEBATE_TEMPLATE_VERSION)
        return verdict

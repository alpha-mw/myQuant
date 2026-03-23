"""
branch-local debate 中心调度器。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from quant_investor.branch_contracts import DebateVerdict, EvidencePacket
from quant_investor.branch_debate_engine import ALLOWED_DEBATE_STATUSES
from quant_investor.debate_templates import BRANCH_DEBATE_ADJUSTMENT_CAPS
from quant_investor.versioning import DEBATE_TEMPLATE_VERSION


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _normalize_status(status: str, reason: str, default: str = "skipped") -> str:
    normalized = str(status or "").strip().lower()
    if normalized in ALLOWED_DEBATE_STATUSES:
        return normalized
    reason_text = str(reason or "").strip().lower()
    if "timeout" in normalized or "timeout" in reason_text:
        return "timed_out"
    if default in ALLOWED_DEBATE_STATUSES:
        return default
    return "skipped"


@dataclass
class DebateRetryPolicy:
    max_attempts: int = 1
    retryable_reasons: tuple[str, ...] = ("timeout", "transient_error")


@dataclass
class DebateScheduler:
    """控制 branch-local debate 的预算、缓存与触发门槛。"""

    engine: Any
    per_batch_max_llm_calls: int = 12
    per_branch_max_calls: int = 3
    single_symbol_max_calls: int = 1
    timeout_sec: float = 8.0
    retry_policy: DebateRetryPolicy = field(default_factory=DebateRetryPolicy)
    cache: dict[tuple[str, str, str, str, str, str], DebateVerdict] = field(default_factory=dict)
    batch_call_count: int = 0
    branch_call_count: dict[str, int] = field(default_factory=dict)
    symbol_call_count: dict[tuple[str, str], int] = field(default_factory=dict)

    def evaluate_branch(
        self,
        branch_name: str,
        evidence: EvidencePacket,
        base_result: Any,
        debate_top_k: int,
        debate_min_abs_score: float,
        data_bundle: Any | None = None,
    ) -> DebateVerdict:
        if not bool(getattr(base_result, "success", True)):
            return self._neutral("branch_unsuccessful")
        if branch_name == "macro":
            return self._evaluate_market_level(branch_name, evidence, base_result)

        candidate_symbols = self._candidate_symbols(
            branch_name=branch_name,
            evidence=evidence,
            base_result=base_result,
            debate_top_k=debate_top_k,
            debate_min_abs_score=debate_min_abs_score,
            data_bundle=data_bundle,
        )
        if not candidate_symbols:
            return self._neutral("scheduler_gate_not_met")

        verdicts: list[DebateVerdict] = []
        reviewed_symbols: list[str] = []
        cache_hits = 0

        for symbol in candidate_symbols:
            if not self._has_budget(branch_name, symbol):
                continue
            symbol_evidence = self._build_symbol_evidence(evidence, symbol)
            cache_key = self._cache_key(symbol, branch_name, symbol_evidence)
            cached = self.cache.get(cache_key)
            if cached is not None:
                verdicts.append(cached)
                reviewed_symbols.append(symbol)
                cache_hits += 1
                continue

            verdict = self._evaluate_once(branch_name, symbol_evidence, base_result)
            self.cache[cache_key] = verdict
            verdicts.append(verdict)
            reviewed_symbols.append(symbol)
            self.batch_call_count += 1
            self.branch_call_count[branch_name] = self.branch_call_count.get(branch_name, 0) + 1
            symbol_key = (branch_name, symbol)
            self.symbol_call_count[symbol_key] = self.symbol_call_count.get(symbol_key, 0) + 1

        if not verdicts:
            return self._neutral("scheduler_budget_exhausted")
        return self._aggregate(branch_name, verdicts, reviewed_symbols, cache_hits)

    def _evaluate_market_level(
        self,
        branch_name: str,
        evidence: EvidencePacket,
        base_result: Any,
    ) -> DebateVerdict:
        symbol = "__market__"
        if not self._has_budget(branch_name, symbol):
            return self._neutral("scheduler_budget_exhausted")
        cache_key = self._cache_key(symbol, branch_name, evidence)
        cached = self.cache.get(cache_key)
        if cached is not None:
            verdict = self._aggregate(branch_name, [cached], [symbol], cache_hits=1)
            verdict.metadata["scope"] = "market"
            return verdict

        verdict = self._evaluate_once(branch_name, evidence, base_result)
        self.cache[cache_key] = verdict
        self.batch_call_count += 1
        self.branch_call_count[branch_name] = self.branch_call_count.get(branch_name, 0) + 1
        self.symbol_call_count[(branch_name, symbol)] = self.symbol_call_count.get((branch_name, symbol), 0) + 1
        aggregated = self._aggregate(branch_name, [verdict], [symbol], cache_hits=0)
        aggregated.metadata["scope"] = "market"
        return aggregated

    def _candidate_symbols(
        self,
        branch_name: str,
        evidence: EvidencePacket,
        base_result: Any,
        debate_top_k: int,
        debate_min_abs_score: float,
        data_bundle: Any | None,
    ) -> list[str]:
        ranked_symbols = [
            symbol
            for symbol, _ in sorted(
                base_result.symbol_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        top_k_symbols = ranked_symbols[: max(int(debate_top_k), 0)]
        severe_symbols = {
            symbol
            for symbol in base_result.symbol_scores
            if self._requires_strong_review(symbol, branch_name, base_result, data_bundle)
        }
        should_review_score = abs(float(base_result.base_score or 0.0)) >= float(debate_min_abs_score)
        selected = set(top_k_symbols)
        if should_review_score:
            selected.update(top_k_symbols or ranked_symbols[:1])
        selected.update(severe_symbols)
        return [symbol for symbol in ranked_symbols if symbol in selected]

    @staticmethod
    def _requires_strong_review(
        symbol: str,
        branch_name: str,
        base_result: Any,
        data_bundle: Any | None,
    ) -> bool:
        quality = getattr(base_result, "data_quality", {}) or {}
        if bool(quality.get("provider_missing")):
            return True
        if float(quality.get("coverage_ratio", 1.0)) < 0.45:
            return True
        missing_modules = quality.get("missing_modules", {})
        if isinstance(missing_modules, dict) and missing_modules.get(symbol):
            return True
        if branch_name == "intelligence" and data_bundle is not None:
            events = (data_bundle.event_data or {}).get(symbol, [])
            if any(abs(float(item.get("impact", 0.0))) >= 0.35 for item in events):
                return True
        return False

    def _has_budget(self, branch_name: str, symbol: str) -> bool:
        if self.batch_call_count >= self.per_batch_max_llm_calls:
            return False
        if self.branch_call_count.get(branch_name, 0) >= self.per_branch_max_calls:
            return False
        if self.symbol_call_count.get((branch_name, symbol), 0) >= self.single_symbol_max_calls:
            return False
        return True

    def _evaluate_once(
        self,
        branch_name: str,
        evidence: EvidencePacket,
        base_result: Any,
    ) -> DebateVerdict:
        verdict = self._neutral("scheduler_default")
        max_attempts = max(int(self.retry_policy.max_attempts), 1)
        for attempt in range(max_attempts):
            verdict = self.engine.evaluate(branch_name, evidence, base_result)
            reason = str(verdict.metadata.get("reason", ""))
            if reason not in self.retry_policy.retryable_reasons or attempt + 1 >= max_attempts:
                return verdict
        return verdict

    def _aggregate(
        self,
        branch_name: str,
        verdicts: list[DebateVerdict],
        reviewed_symbols: list[str],
        cache_hits: int,
    ) -> DebateVerdict:
        cap = BRANCH_DEBATE_ADJUSTMENT_CAPS.get(branch_name, 0.10)
        reasons = _dedupe([str(item.metadata.get("reason", "")) for item in verdicts if item.metadata.get("reason")])
        statuses = [
            _normalize_status(
                str(item.metadata.get("status", "")),
                str(item.metadata.get("reason", "")),
                default="llm",
            )
            for item in verdicts
        ]
        adjustment = _clamp(
            sum(float(item.score_adjustment) for item in verdicts) / max(len(verdicts), 1),
            -cap,
            cap,
        )
        confidence = _clamp(
            sum(float(item.confidence) for item in verdicts) / max(len(verdicts), 1),
            0.0,
            1.0,
        )
        direction = "neutral"
        if adjustment > 1e-6:
            direction = "bullish"
        elif adjustment < -1e-6:
            direction = "bearish"

        unique_statuses = _dedupe(statuses)
        if "error" in unique_statuses:
            aggregate_status = "error"
        elif "timed_out" in unique_statuses:
            aggregate_status = "timed_out"
        elif "llm" in unique_statuses:
            aggregate_status = "llm"
        elif "deterministic_stub" in unique_statuses:
            aggregate_status = "deterministic_stub"
        else:
            aggregate_status = "skipped"

        return DebateVerdict(
            direction=direction,
            confidence=confidence,
            score_adjustment=adjustment,
            bull_points=_dedupe([point for item in verdicts for point in item.bull_points])[:6],
            bear_points=_dedupe([point for item in verdicts for point in item.bear_points])[:6],
            risk_flags=_dedupe([point for item in verdicts for point in item.risk_flags])[:6],
            unknowns=_dedupe([point for item in verdicts for point in item.unknowns])[:6],
            used_features=_dedupe([point for item in verdicts for point in item.used_features])[:10],
            hard_veto=any(item.hard_veto for item in verdicts),
            metadata={
                "status": aggregate_status,
                "reason": reasons[0] if len(reasons) == 1 else "",
                "reviewed_symbols": list(reviewed_symbols),
                "reviewed_symbol_count": len(reviewed_symbols),
                "cache_hits": cache_hits,
                "timeout_sec": self.timeout_sec,
                "retry_policy": {
                    "max_attempts": self.retry_policy.max_attempts,
                    "retryable_reasons": list(self.retry_policy.retryable_reasons),
                },
                "template_version": DEBATE_TEMPLATE_VERSION,
            },
        )

    @staticmethod
    def _build_symbol_evidence(evidence: EvidencePacket, symbol: str) -> EvidencePacket:
        context = evidence.symbol_context.get(symbol, {})
        summary = str(context.get("summary", evidence.summary))
        bull_points = [str(item) for item in context.get("bull_points", evidence.bull_points)]
        bear_points = [str(item) for item in context.get("bear_points", evidence.bear_points)]
        risk_points = [str(item) for item in context.get("risk_points", evidence.risk_points)]
        unknowns = [str(item) for item in context.get("unknowns", evidence.unknowns)]
        used_features = [str(item) for item in context.get("used_features", evidence.used_features)]
        return EvidencePacket(
            branch_name=evidence.branch_name,
            as_of=evidence.as_of,
            scope="symbol",
            summary=summary,
            symbols=[symbol],
            top_symbols=[symbol],
            bull_points=bull_points[:6],
            bear_points=bear_points[:6],
            risk_points=risk_points[:6],
            unknowns=unknowns[:4],
            used_features=used_features[:8],
            feature_values=dict(evidence.feature_values),
            symbol_context={symbol: dict(context)},
            metadata=dict(evidence.metadata),
        )

    def _cache_key(
        self,
        symbol: str,
        branch_name: str,
        evidence: EvidencePacket,
    ) -> tuple[str, str, str, str, str, str]:
        evidence_payload = {
            "summary": evidence.summary,
            "bull_points": evidence.bull_points,
            "bear_points": evidence.bear_points,
            "risk_points": evidence.risk_points,
            "unknowns": evidence.unknowns,
            "used_features": evidence.used_features,
            "feature_values": evidence.feature_values,
            "symbol_context": evidence.symbol_context,
        }
        evidence_hash = hashlib.sha256(
            json.dumps(evidence_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        model = str(getattr(self.engine, "model", ""))
        return (
            symbol,
            branch_name,
            str(evidence.as_of or ""),
            model,
            evidence_hash,
            DEBATE_TEMPLATE_VERSION,
        )

    @staticmethod
    def _neutral(reason: str) -> DebateVerdict:
        return DebateVerdict(
            direction="neutral",
            confidence=0.0,
            score_adjustment=0.0,
            metadata={"status": "skipped", "reason": reason, "template_version": DEBATE_TEMPLATE_VERSION},
        )

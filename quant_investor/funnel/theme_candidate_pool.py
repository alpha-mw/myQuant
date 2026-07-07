"""Production theme candidate-pool construction for the deterministic funnel."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from quant_investor.agent_protocol import GlobalContext


DEFAULT_ALLOWED_PHASES: tuple[str, ...] = (
    "accumulation",
    "early_acceleration",
    "confirmed_rotation",
)
DEFAULT_BLOCKED_PHASES: tuple[str, ...] = ("distribution",)
DEFAULT_BLOCKED_FLAGS: tuple[str, ...] = (
    "theme_distribution_risk",
    "theme_fake_breakout_risk",
)
STRICT_BLOCKED_FLAGS: tuple[str, ...] = (
    "theme_distribution_risk",
    "theme_fake_breakout_risk",
    "theme_low_breadth",
    "theme_overextended",
    "theme_overextended_no_chase",
)
RISK_WATCH_BUCKETS: tuple[str, ...] = (
    "risk_watch_distribution",
    "risk_watch_overextended",
    "risk_watch_fake_breakout",
)
BUCKET_PRIORITIES: dict[str, int] = {
    "core": 0,
    "extended": 1,
    "extended_low_score": 1,
    "extended_low_breadth": 1,
    "forced_theme": 1,
    "residual_theme_alpha": 2,
    "risk_watch_distribution": 3,
    "risk_watch_overextended": 3,
    "risk_watch_fake_breakout": 3,
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return lower
    if not math.isfinite(numeric):
        return lower
    return max(lower, min(upper, numeric))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_score(value: Any, default: float = 0.0) -> float:
    numeric = _safe_float(value, default)
    if numeric > 1.0:
        numeric /= 100.0
    return _clamp(numeric)


def _normalized_phase(value: Any) -> str:
    return str(value or "").strip().lower()


def _dedupe_texts(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        return ()
    try:
        items = list(value or [])
    except TypeError:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        text = str(symbol or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


@dataclass(frozen=True)
class ThemePoolConfig:
    enabled: bool = True
    required: bool = True
    use_markov_policy: bool = True
    score_source: str = "smoothed"
    fallback_to_raw_score: bool = True
    base_min_theme_score: float = 0.58
    base_min_symbol_score: float = 0.55
    base_top_themes: int = 8
    max_symbols_per_theme: int = 30
    residual_ratio: float = 0.25
    min_residual_symbols: int = 20
    min_admitted_themes: int = 2
    allow_unthemed_residual: bool = False
    include_risk_watch: bool = True
    risk_watch_max_ratio: float = 0.20
    symbol_gate_mode: str = "classify"
    allowed_phases: tuple[str, ...] = DEFAULT_ALLOWED_PHASES
    blocked_phases: tuple[str, ...] = DEFAULT_BLOCKED_PHASES
    blocked_flags: tuple[str, ...] = DEFAULT_BLOCKED_FLAGS
    min_member_count: int = 0


@dataclass(frozen=True)
class ThemeGatePolicy:
    min_theme_score: float
    min_symbol_score: float
    top_themes: int
    allowed_phases: tuple[str, ...]
    blocked_phases: tuple[str, ...]
    blocked_flags: tuple[str, ...]
    residual_ratio: float
    max_symbols_per_theme: int
    risk_watch_max_ratio: float
    candidate_pressure: float = 1.0
    score_penalty_multiplier: float = 1.0
    regime: str = ""
    transition_risk: float = 0.0
    confidence: float = 0.0
    production_eligible: bool = False

    @classmethod
    def from_config(cls, config: ThemePoolConfig) -> "ThemeGatePolicy":
        return cls(
            min_theme_score=_clamp(config.base_min_theme_score),
            min_symbol_score=_clamp(config.base_min_symbol_score),
            top_themes=max(int(config.base_top_themes), 0),
            allowed_phases=tuple(_normalized_phase(item) for item in config.allowed_phases),
            blocked_phases=tuple(_normalized_phase(item) for item in config.blocked_phases),
            blocked_flags=tuple(str(item) for item in config.blocked_flags),
            residual_ratio=_clamp(config.residual_ratio),
            max_symbols_per_theme=max(int(config.max_symbols_per_theme), 1),
            risk_watch_max_ratio=_clamp(config.risk_watch_max_ratio),
            candidate_pressure=1.0,
            score_penalty_multiplier=1.0,
            regime="baseline",
            production_eligible=True,
        )

    @classmethod
    def from_markov(
        cls,
        markov_payload: Mapping[str, Any] | None,
        config: ThemePoolConfig,
    ) -> "ThemeGatePolicy":
        payload = dict(markov_payload or {})
        regime = str(payload.get("dominant_regime") or "").strip()
        transition_risk = _clamp(_safe_float(payload.get("transition_risk"), 0.0))
        confidence = _clamp(_safe_float(payload.get("confidence"), 0.0))
        production_eligible = payload.get("production_eligible") is True
        configured_risk_watch_ratio = _clamp(config.risk_watch_max_ratio)

        if not production_eligible:
            policy = cls(
                min_theme_score=0.65,
                min_symbol_score=0.60,
                top_themes=5,
                allowed_phases=("confirmed_rotation",),
                blocked_phases=tuple(_normalized_phase(item) for item in config.blocked_phases),
                blocked_flags=tuple(str(item) for item in config.blocked_flags),
                residual_ratio=min(_clamp(config.residual_ratio), 0.15),
                max_symbols_per_theme=max(int(config.max_symbols_per_theme), 1),
                risk_watch_max_ratio=min(configured_risk_watch_ratio, 0.10),
                candidate_pressure=0.70,
                score_penalty_multiplier=1.25,
                regime=regime or "conservative",
                transition_risk=transition_risk,
                confidence=confidence,
                production_eligible=False,
            )
            return _tighten_for_transition(policy)

        if regime == "趋势上涨":
            policy = cls(
                min_theme_score=0.55,
                min_symbol_score=0.52,
                top_themes=10,
                allowed_phases=(
                    "accumulation",
                    "early_acceleration",
                    "confirmed_rotation",
                ),
                blocked_phases=("distribution",),
                blocked_flags=(
                    "theme_distribution_risk",
                    "theme_fake_breakout_risk",
                ),
                residual_ratio=max(_clamp(config.residual_ratio), 0.30),
                max_symbols_per_theme=35,
                risk_watch_max_ratio=max(configured_risk_watch_ratio, 0.25),
                candidate_pressure=1.0,
                score_penalty_multiplier=0.85,
                regime=regime,
                transition_risk=transition_risk,
                confidence=confidence,
                production_eligible=True,
            )
        elif regime == "震荡低波":
            policy = cls(
                min_theme_score=0.60,
                min_symbol_score=0.55,
                top_themes=8,
                allowed_phases=(
                    "accumulation",
                    "early_acceleration",
                    "confirmed_rotation",
                ),
                blocked_phases=("distribution",),
                blocked_flags=(
                    "theme_distribution_risk",
                    "theme_fake_breakout_risk",
                ),
                residual_ratio=_clamp(config.residual_ratio),
                max_symbols_per_theme=30,
                risk_watch_max_ratio=configured_risk_watch_ratio,
                candidate_pressure=1.0,
                score_penalty_multiplier=1.0,
                regime=regime,
                transition_risk=transition_risk,
                confidence=confidence,
                production_eligible=True,
            )
        elif regime == "震荡高波":
            policy = cls(
                min_theme_score=0.65,
                min_symbol_score=0.60,
                top_themes=6,
                allowed_phases=("confirmed_rotation",),
                blocked_phases=("distribution", "overextended"),
                blocked_flags=STRICT_BLOCKED_FLAGS,
                residual_ratio=min(_clamp(config.residual_ratio), 0.15),
                max_symbols_per_theme=24,
                risk_watch_max_ratio=min(configured_risk_watch_ratio, 0.12),
                candidate_pressure=0.75,
                score_penalty_multiplier=1.30,
                regime=regime,
                transition_risk=transition_risk,
                confidence=confidence,
                production_eligible=True,
            )
        elif regime == "趋势下跌":
            policy = cls(
                min_theme_score=0.70,
                min_symbol_score=0.65,
                top_themes=4,
                allowed_phases=("confirmed_rotation",),
                blocked_phases=("distribution", "overextended"),
                blocked_flags=STRICT_BLOCKED_FLAGS,
                residual_ratio=min(_clamp(config.residual_ratio), 0.10),
                max_symbols_per_theme=18,
                risk_watch_max_ratio=min(configured_risk_watch_ratio, 0.08),
                candidate_pressure=0.55,
                score_penalty_multiplier=1.60,
                regime=regime,
                transition_risk=transition_risk,
                confidence=confidence,
                production_eligible=True,
            )
        else:
            policy = cls(
                min_theme_score=0.65,
                min_symbol_score=0.60,
                top_themes=5,
                allowed_phases=("confirmed_rotation",),
                blocked_phases=tuple(_normalized_phase(item) for item in config.blocked_phases),
                blocked_flags=tuple(str(item) for item in config.blocked_flags),
                residual_ratio=min(_clamp(config.residual_ratio), 0.15),
                max_symbols_per_theme=max(int(config.max_symbols_per_theme), 1),
                risk_watch_max_ratio=min(configured_risk_watch_ratio, 0.10),
                candidate_pressure=0.70,
                score_penalty_multiplier=1.25,
                regime=regime or "conservative",
                transition_risk=transition_risk,
                confidence=confidence,
                production_eligible=True,
            )
        return _tighten_for_transition(policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_theme_score": float(self.min_theme_score),
            "min_symbol_score": float(self.min_symbol_score),
            "top_themes": int(self.top_themes),
            "allowed_phases": list(self.allowed_phases),
            "blocked_phases": list(self.blocked_phases),
            "blocked_flags": list(self.blocked_flags),
            "residual_ratio": float(self.residual_ratio),
            "residual_enabled": False,
            "residual_concept": "disabled_by_theme_pool_hard_filter",
            "hard_theme_constraint": True,
            "max_symbols_per_theme": int(self.max_symbols_per_theme),
            "risk_watch_max_ratio": float(self.risk_watch_max_ratio),
            "candidate_pressure": float(self.candidate_pressure),
            "score_penalty_multiplier": float(self.score_penalty_multiplier),
            "regime": str(self.regime),
            "transition_risk": float(self.transition_risk),
            "confidence": float(self.confidence),
            "production_eligible": bool(self.production_eligible),
        }


def _tighten_for_transition(policy: ThemeGatePolicy) -> ThemeGatePolicy:
    min_theme_score = float(policy.min_theme_score)
    min_symbol_score = float(policy.min_symbol_score)
    top_themes = int(policy.top_themes)
    residual_ratio = float(policy.residual_ratio)
    risk_watch_max_ratio = float(policy.risk_watch_max_ratio)
    candidate_pressure = float(policy.candidate_pressure)
    score_penalty_multiplier = float(policy.score_penalty_multiplier)
    if policy.transition_risk >= 0.60:
        min_theme_score += 0.05
        min_symbol_score += 0.03
        top_themes = max(2, int(top_themes * 0.65))
        residual_ratio = min(residual_ratio, 0.15)
        risk_watch_max_ratio = min(risk_watch_max_ratio, 0.10)
        candidate_pressure *= 0.85
        score_penalty_multiplier *= 1.15
    if policy.confidence < 0.45:
        min_theme_score += 0.03
        top_themes = max(3, int(top_themes * 0.80))
        candidate_pressure *= 0.90
        score_penalty_multiplier *= 1.10
    return ThemeGatePolicy(
        min_theme_score=_clamp(min_theme_score),
        min_symbol_score=_clamp(min_symbol_score),
        top_themes=max(top_themes, 0),
        allowed_phases=policy.allowed_phases,
        blocked_phases=policy.blocked_phases,
        blocked_flags=policy.blocked_flags,
        residual_ratio=_clamp(residual_ratio),
        max_symbols_per_theme=max(int(policy.max_symbols_per_theme), 1),
        risk_watch_max_ratio=_clamp(risk_watch_max_ratio),
        candidate_pressure=_clamp(candidate_pressure, 0.10, 1.0),
        score_penalty_multiplier=max(score_penalty_multiplier, 0.0),
        regime=policy.regime,
        transition_risk=policy.transition_risk,
        confidence=policy.confidence,
        production_eligible=policy.production_eligible,
    )


@dataclass
class ThemeCandidatePoolOutput:
    symbols: list[str] = field(default_factory=list)
    excluded_symbols: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    symbol_sources: dict[str, str] = field(default_factory=dict)
    symbol_pool_scores: dict[str, float] = field(default_factory=dict)
    symbol_reasons: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _ThemeCandidate:
    theme_id: str
    score: float
    rank_score: float
    phase: str
    risk_flags: tuple[str, ...]
    payload: Mapping[str, Any]
    quality_flags: tuple[str, ...] = ()
    forced: bool = False
    force_reason: str = ""
    original_rejection_reason: str = ""


@dataclass(frozen=True)
class _SymbolCandidate:
    symbol: str
    score: float
    raw_score: float
    score_penalty: float
    bucket: str
    source: str
    primary_theme_id: str
    primary_theme_name: str
    theme_score: float
    symbol_theme_score: float
    phase: str
    risk_flags: tuple[str, ...]
    theme_forced_admission: bool = False
    candidate_intent: str = ""

    @property
    def is_risk_watch(self) -> bool:
        return self.bucket in RISK_WATCH_BUCKETS


class ThemeCandidatePoolBuilder:
    """Build a production candidate pool from scanner theme membership."""

    def __init__(self, config: ThemePoolConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        symbols: list[str],
        global_context: GlobalContext,
        quant_scores: Mapping[str, float],
        max_candidates: int,
    ) -> ThemeCandidatePoolOutput:
        input_symbols = _dedupe_symbols(list(symbols or []))
        if not self.config.enabled:
            return ThemeCandidatePoolOutput(
                symbols=list(input_symbols),
                metadata={
                    "enabled": False,
                    "required": bool(self.config.required),
                    "status": "disabled",
                },
            )

        rotation = _mapping_or_empty(
            _mapping_or_empty(getattr(global_context, "metadata", {})).get("theme_rotation")
        )
        if str(rotation.get("status") or "").strip().lower() != "success":
            reason = "theme_pool_required_but_theme_rotation_not_success"
            if self.config.required:
                raise RuntimeError(reason)
            return ThemeCandidatePoolOutput(
                symbols=list(input_symbols),
                metadata={
                    "enabled": True,
                    "required": False,
                    "status": "fallback",
                    "reason": reason,
                },
            )

        policy = self._policy(global_context)
        effective_max_candidates = self._effective_max_candidates(max_candidates, policy)
        theme_scores = _mapping_or_empty(rotation.get("theme_scores"))
        symbol_scores = _mapping_or_empty(rotation.get("symbol_scores"))
        symbol_smoothed_scores = _mapping_or_empty(rotation.get("symbol_smoothed_scores"))
        symbol_primary_theme = _mapping_or_empty(rotation.get("symbol_primary_theme"))
        symbol_phase = _mapping_or_empty(rotation.get("symbol_phase"))
        symbol_risk_flags = _mapping_or_empty(rotation.get("symbol_risk_flags"))
        symbol_market_state = _mapping_or_empty(
            _mapping_or_empty(getattr(global_context, "metadata", {})).get("symbol_market_state")
        )
        liquidity_scores = _mapping_or_empty(
            _mapping_or_empty(getattr(global_context, "liquidity_filter", {})).get("liquidity_scores")
        )

        admitted_themes, rejected_themes, rejected_theme_candidates, all_theme_candidates = self._admit_themes(
            theme_scores=theme_scores,
            policy=policy,
        )
        natural_admitted_theme_count = len(admitted_themes)
        admitted_themes, rejected_themes, forced_themes = self._force_minimum_themes(
            admitted_themes=admitted_themes,
            rejected_themes=rejected_themes,
            rejected_theme_candidates=rejected_theme_candidates,
            input_symbols=input_symbols,
            symbol_primary_theme=symbol_primary_theme,
        )
        admitted_theme_ids = {theme.theme_id for theme in admitted_themes}
        theme_by_id = {theme.theme_id: theme for theme in all_theme_candidates}
        for theme in admitted_themes:
            theme_by_id[theme.theme_id] = theme

        symbol_candidates, initial_exclusions, residual_candidates = self._classify_symbols(
            input_symbols=input_symbols,
            admitted_themes=admitted_themes,
            admitted_theme_ids=admitted_theme_ids,
            theme_by_id=theme_by_id,
            symbol_scores=symbol_scores,
            symbol_smoothed_scores=symbol_smoothed_scores,
            symbol_primary_theme=symbol_primary_theme,
            symbol_phase=symbol_phase,
            symbol_risk_flags=symbol_risk_flags,
            symbol_market_state=symbol_market_state,
            liquidity_scores=liquidity_scores,
            quant_scores=quant_scores,
            policy=policy,
            max_candidates=effective_max_candidates,
        )
        selected, cap_exclusions, risk_watch_limit = self._apply_caps(
            candidates=symbol_candidates,
            max_candidates=effective_max_candidates,
            policy=policy,
        )
        final_symbols = [candidate.symbol for candidate in selected]
        final_set = set(final_symbols)

        excluded_symbols = self._build_exclusions(
            input_symbols=input_symbols,
            final_set=final_set,
            initial_exclusions=initial_exclusions,
            cap_exclusions=cap_exclusions,
        )
        symbol_reasons = {
            symbol: "admitted" if symbol in final_set else excluded_symbols.get(symbol, "theme_pool_rank_cutoff")
            for symbol in input_symbols
        }
        selected_by_symbol = {candidate.symbol: candidate for candidate in selected}
        candidates_by_symbol = {candidate.symbol: candidate for candidate in symbol_candidates}
        symbol_sources = {
            symbol: candidate.source
            for symbol, candidate in selected_by_symbol.items()
        }
        symbol_pool_scores = {
            symbol: float(candidate.score)
            for symbol, candidate in selected_by_symbol.items()
        }
        symbol_metadata = {
            symbol: self._symbol_metadata(
                symbol=symbol,
                candidate=selected_by_symbol.get(symbol) or candidates_by_symbol.get(symbol),
                symbol_primary_theme=symbol_primary_theme,
                theme_by_id=theme_by_id,
                symbol_scores=symbol_scores,
                symbol_smoothed_scores=symbol_smoothed_scores,
                symbol_phase=symbol_phase,
                symbol_risk_flags=symbol_risk_flags,
                symbol_reason=symbol_reasons.get(symbol, ""),
                policy=policy,
                admitted=bool(symbol in final_set),
            )
            for symbol in input_symbols
        }
        bucket_counts = _count_by(selected, "bucket")
        source_counts = _count_by(selected, "source")
        metadata = {
            "enabled": True,
            "required": bool(self.config.required),
            "status": "applied",
            "policy": policy.to_dict(),
            "admitted_theme_count": len(admitted_themes),
            "natural_admitted_theme_count": natural_admitted_theme_count,
            "forced_theme_count": len(forced_themes),
            "min_admitted_themes": max(int(self.config.min_admitted_themes or 0), 0),
            "rejected_theme_count": len(rejected_themes),
            "core_symbol_count": source_counts.get("core", 0),
            "extended_symbol_count": source_counts.get("extended", 0),
            "risk_watch_symbol_count": source_counts.get("risk_watch", 0),
            "residual_theme_symbol_count": source_counts.get("residual_theme", 0),
            "residual_symbol_count": source_counts.get("residual_theme", 0),
            "excluded_symbol_count": len(excluded_symbols),
            "unthemed_exclusion_count": sum(
                1 for reason in excluded_symbols.values()
                if reason == "theme_pool_missing_theme_membership"
            ),
            "missing_theme_metadata_exclusion_count": sum(
                1 for reason in excluded_symbols.values()
                if reason == "theme_pool_missing_theme_metadata"
            ),
            "bucket_counts": bucket_counts,
            "source_counts": source_counts,
            "risk_watch_limit": risk_watch_limit,
            "risk_watch_max_ratio": float(policy.risk_watch_max_ratio),
            "effective_max_candidates": int(effective_max_candidates),
            "requested_max_candidates": max(int(max_candidates), 0),
            "candidate_pressure": float(policy.candidate_pressure),
            "allow_unthemed_residual": bool(self.config.allow_unthemed_residual),
            "include_risk_watch": bool(self.config.include_risk_watch),
            "symbol_gate_mode": str(self.config.symbol_gate_mode or "classify"),
            "admitted_themes": [
                _theme_metadata(theme)
                for theme in admitted_themes
            ],
            "rejected_themes": rejected_themes,
            "score_source": self._score_source(),
            "fallback_to_raw_score": bool(self.config.fallback_to_raw_score),
            "symbols": symbol_metadata,
            "residual_theme_alpha_candidates": [
                candidate.symbol for candidate in residual_candidates
            ],
        }
        return ThemeCandidatePoolOutput(
            symbols=final_symbols,
            excluded_symbols=excluded_symbols,
            metadata=metadata,
            symbol_sources=symbol_sources,
            symbol_pool_scores=symbol_pool_scores,
            symbol_reasons=symbol_reasons,
        )

    def _policy(self, global_context: GlobalContext) -> ThemeGatePolicy:
        if not self.config.use_markov_policy:
            return ThemeGatePolicy.from_config(self.config)
        markov_payload = _extract_markov_payload(global_context)
        return ThemeGatePolicy.from_markov(markov_payload, self.config)

    def _score_source(self) -> str:
        source = str(self.config.score_source or "smoothed").strip().lower()
        return source if source in {"raw", "smoothed"} else "smoothed"

    def _effective_max_candidates(self, max_candidates: int, policy: ThemeGatePolicy) -> int:
        requested = max(int(max_candidates), 0)
        if requested <= 0:
            return 0
        pressure = _clamp(policy.candidate_pressure, 0.10, 1.0)
        return max(1, min(requested, int(math.ceil(requested * pressure))))

    def _symbol_theme_score(
        self,
        *,
        symbol: str,
        symbol_scores: Mapping[str, Any],
        symbol_smoothed_scores: Mapping[str, Any],
    ) -> float:
        source = self._score_source()
        if source == "smoothed":
            if symbol in symbol_smoothed_scores:
                return _normalize_score(symbol_smoothed_scores.get(symbol))
            if self.config.fallback_to_raw_score and symbol in symbol_scores:
                return _normalize_score(symbol_scores.get(symbol))
            return 0.0
        return _normalize_score(symbol_scores.get(symbol, 0.0))

    def _admit_themes(
        self,
        *,
        theme_scores: Mapping[str, Any],
        policy: ThemeGatePolicy,
    ) -> tuple[list[_ThemeCandidate], list[dict[str, Any]], list[_ThemeCandidate], list[_ThemeCandidate]]:
        candidates: list[_ThemeCandidate] = []
        for theme_id, raw_payload in sorted(theme_scores.items(), key=lambda item: str(item[0])):
            candidate = self._theme_candidate(theme_id=theme_id, raw_payload=raw_payload)
            quality_flags = self._theme_quality_flags(
                payload=candidate.payload,
                score=candidate.score,
                phase=candidate.phase,
                flags=candidate.risk_flags,
                policy=policy,
            )
            candidates.append(replace(candidate, quality_flags=quality_flags))
        ranked = sorted(candidates, key=lambda item: (-item.rank_score, item.theme_id))
        eligible = [item for item in ranked if not item.quality_flags]
        admitted = eligible[: max(int(policy.top_themes), 0)]
        admitted_ids = {item.theme_id for item in admitted}
        rejected: list[dict[str, Any]] = []
        rejected_candidates: list[_ThemeCandidate] = []
        for item in ranked:
            if item.theme_id in admitted_ids:
                continue
            reason = _theme_rejection_reason(item)
            rejected.append(_theme_rejection_entry(item, reason))
            rejected_candidates.append(replace(item, original_rejection_reason=reason))
        return admitted, rejected, rejected_candidates, ranked

    def _theme_candidate(self, *, theme_id: str, raw_payload: Any) -> _ThemeCandidate:
        payload = _mapping_or_empty(raw_payload)
        theme_key = str(payload.get("theme_id") or theme_id)
        score = _normalize_score(payload.get("score", payload.get("raw_score", 0.0)))
        phase = _normalized_phase(payload.get("phase"))
        flags = _dedupe_texts(payload.get("risk_flags", [])) + _dedupe_texts(payload.get("policy_risk_flags", []))
        smoothed_or_raw_score = _normalize_score(
            payload.get(
                "smoothed_score",
                payload.get("raw_score", payload.get("score", 0.0)),
            )
        )
        rank_score = _theme_rank_score(
            payload=payload,
            score=score,
            smoothed_or_raw_score=smoothed_or_raw_score,
        )
        return _ThemeCandidate(
            theme_id=theme_key,
            score=score,
            rank_score=rank_score,
            phase=phase,
            risk_flags=flags,
            payload=payload,
        )

    def _force_minimum_themes(
        self,
        *,
        admitted_themes: list[_ThemeCandidate],
        rejected_themes: list[dict[str, Any]],
        rejected_theme_candidates: list[_ThemeCandidate],
        input_symbols: list[str],
        symbol_primary_theme: Mapping[str, Any],
    ) -> tuple[list[_ThemeCandidate], list[dict[str, Any]], list[_ThemeCandidate]]:
        target_count = max(int(self.config.min_admitted_themes or 0), 0)
        if target_count <= 0 or len(admitted_themes) >= target_count:
            return admitted_themes, rejected_themes, []

        present_theme_ids = {
            str(symbol_primary_theme.get(symbol, "") or "")
            for symbol in input_symbols
            if str(symbol_primary_theme.get(symbol, "") or "")
        }
        admitted_ids = {theme.theme_id for theme in admitted_themes}
        available = [
            candidate
            for candidate in rejected_theme_candidates
            if candidate.theme_id in present_theme_ids and candidate.theme_id not in admitted_ids
        ]
        slots = max(target_count - len(admitted_themes), 0)
        ranked = sorted(available, key=lambda item: (-item.rank_score, -item.score, item.theme_id))
        forced = [
            replace(
                candidate,
                forced=True,
                force_reason="forced_top_theme_min_admitted_themes",
                original_rejection_reason=(
                    candidate.original_rejection_reason or "natural_theme_gate_not_passed"
                ),
            )
            for candidate in ranked[:slots]
        ]
        if not forced:
            return admitted_themes, rejected_themes, []

        forced_ids = {theme.theme_id for theme in forced}
        remaining_rejected = [
            item for item in rejected_themes if str(item.get("theme_id") or "") not in forced_ids
        ]
        combined = sorted(
            list(admitted_themes) + forced,
            key=lambda item: (-item.rank_score, item.theme_id),
        )
        return combined, remaining_rejected, forced

    def _theme_quality_flags(
        self,
        *,
        payload: Mapping[str, Any],
        score: float,
        phase: str,
        flags: tuple[str, ...],
        policy: ThemeGatePolicy,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if score < policy.min_theme_score:
            reasons.append("theme_score_below_threshold")
        if policy.allowed_phases and phase not in policy.allowed_phases:
            reasons.append("phase_not_allowed")
        if phase in policy.blocked_phases:
            reasons.append("phase_blocked")
        if set(flags).intersection(policy.blocked_flags):
            reasons.append("risk_flag_present")
        if "member_count" in payload and int(self.config.min_member_count or 0) > 0:
            if _safe_int(payload.get("member_count"), 0) < int(self.config.min_member_count):
                reasons.append("member_count_below_min")
        if "confidence" in payload and _safe_float(payload.get("confidence"), 0.0) < 0.35:
            reasons.append("confidence_below_min")
        breadth_min = 0.45 if policy.regime in {"震荡高波", "趋势下跌"} else 0.35
        if _safe_float(payload.get("breadth"), 0.0) < breadth_min:
            reasons.append("breadth_below_min")
        return tuple(reasons)

    def _classify_symbols(
        self,
        *,
        input_symbols: list[str],
        admitted_themes: list[_ThemeCandidate],
        admitted_theme_ids: set[str],
        theme_by_id: Mapping[str, _ThemeCandidate],
        symbol_scores: Mapping[str, Any],
        symbol_smoothed_scores: Mapping[str, Any],
        symbol_primary_theme: Mapping[str, Any],
        symbol_phase: Mapping[str, Any],
        symbol_risk_flags: Mapping[str, Any],
        symbol_market_state: Mapping[str, Any],
        liquidity_scores: Mapping[str, Any],
        quant_scores: Mapping[str, float],
        policy: ThemeGatePolicy,
        max_candidates: int,
    ) -> tuple[list[_SymbolCandidate], dict[str, str], list[_SymbolCandidate]]:
        del admitted_themes
        candidates: list[_SymbolCandidate] = []
        initial_exclusions: dict[str, str] = {}
        residual_pool: list[_SymbolCandidate] = []
        for symbol in input_symbols:
            theme_id = str(symbol_primary_theme.get(symbol, "") or "")
            if not theme_id:
                initial_exclusions[symbol] = "theme_pool_missing_theme_membership"
                continue
            theme = theme_by_id.get(theme_id)
            if theme is None:
                initial_exclusions[symbol] = "theme_pool_missing_theme_metadata"
                continue
            bucket = self._bucket_for_symbol(
                symbol=symbol,
                theme=theme,
                is_residual_theme=theme_id not in admitted_theme_ids,
                symbol_scores=symbol_scores,
                symbol_smoothed_scores=symbol_smoothed_scores,
                symbol_phase=symbol_phase,
                symbol_risk_flags=symbol_risk_flags,
                policy=policy,
            )
            candidate = self._symbol_candidate(
                symbol=symbol,
                theme=theme,
                theme_id=theme_id,
                bucket=bucket,
                symbol_scores=symbol_scores,
                symbol_smoothed_scores=symbol_smoothed_scores,
                symbol_phase=symbol_phase,
                symbol_risk_flags=symbol_risk_flags,
                symbol_market_state=symbol_market_state,
                liquidity_scores=liquidity_scores,
                quant_scores=quant_scores,
                policy=policy,
            )
            if theme_id in admitted_theme_ids:
                candidates.append(candidate)
            else:
                initial_exclusions[symbol] = "theme_pool_theme_not_admitted"

        return candidates, initial_exclusions, residual_pool

    def _residual_theme_target(self, max_candidates: int, policy: ThemeGatePolicy) -> int:
        target = max(
            int(self.config.min_residual_symbols or 0),
            int(max(int(max_candidates), 0) * _clamp(policy.residual_ratio)),
        )
        return min(target, max(int(max_candidates), 0))

    def _bucket_for_symbol(
        self,
        *,
        symbol: str,
        theme: _ThemeCandidate,
        is_residual_theme: bool,
        symbol_scores: Mapping[str, Any],
        symbol_smoothed_scores: Mapping[str, Any],
        symbol_phase: Mapping[str, Any],
        symbol_risk_flags: Mapping[str, Any],
        policy: ThemeGatePolicy,
    ) -> str:
        phase = _normalized_phase(symbol_phase.get(symbol)) or theme.phase
        flags = _symbol_risk_flags(symbol=symbol, theme=theme, symbol_risk_flags=symbol_risk_flags)
        if self.config.include_risk_watch:
            if "theme_fake_breakout_risk" in flags:
                return "risk_watch_fake_breakout"
            if phase == "distribution" or "theme_distribution_risk" in flags:
                return "risk_watch_distribution"
            if phase == "overextended" or set(flags).intersection(
                {"theme_overextended", "theme_overextended_no_chase"}
            ):
                return "risk_watch_overextended"
        if is_residual_theme:
            return "residual_theme_alpha"
        if theme.forced:
            return "forced_theme"
        symbol_score = self._symbol_theme_score(
            symbol=symbol,
            symbol_scores=symbol_scores,
            symbol_smoothed_scores=symbol_smoothed_scores,
        )
        if symbol_score < policy.min_symbol_score:
            return "extended_low_score"
        breadth_min = 0.45 if policy.regime in {"震荡高波", "趋势下跌"} else 0.35
        if "theme_low_breadth" in flags or _safe_float(theme.payload.get("breadth"), 0.0) < breadth_min:
            return "extended_low_breadth"
        if theme.score < policy.min_theme_score:
            return "extended"
        if phase != "confirmed_rotation":
            return "extended"
        return "core"

    def _symbol_candidate(
        self,
        *,
        symbol: str,
        theme: _ThemeCandidate | None,
        theme_id: str,
        bucket: str,
        symbol_scores: Mapping[str, Any],
        symbol_smoothed_scores: Mapping[str, Any],
        symbol_phase: Mapping[str, Any],
        symbol_risk_flags: Mapping[str, Any],
        symbol_market_state: Mapping[str, Any],
        liquidity_scores: Mapping[str, Any],
        quant_scores: Mapping[str, float],
        policy: ThemeGatePolicy,
    ) -> _SymbolCandidate:
        symbol_theme_score = self._symbol_theme_score(
            symbol=symbol,
            symbol_scores=symbol_scores,
            symbol_smoothed_scores=symbol_smoothed_scores,
        )
        phase = _normalized_phase(symbol_phase.get(symbol)) or (theme.phase if theme else "")
        risk_flags = _symbol_risk_flags(symbol=symbol, theme=theme, symbol_risk_flags=symbol_risk_flags)
        raw_score = _symbol_pool_score(
            symbol=symbol,
            symbol_theme_score=symbol_theme_score,
            quant_score=_safe_float(quant_scores.get(symbol, 0.0)),
            symbol_market_state=symbol_market_state,
            liquidity_scores=liquidity_scores,
        )
        penalty = _score_penalty(
            bucket=bucket,
            symbol_theme_score=symbol_theme_score,
            theme_score=theme.score if theme else 0.0,
            policy=policy,
        )
        return _SymbolCandidate(
            symbol=symbol,
            score=_clamp(raw_score - penalty),
            raw_score=float(raw_score),
            score_penalty=float(penalty),
            bucket=bucket,
            source=_source_for_bucket(bucket),
            primary_theme_id=theme_id,
            primary_theme_name=str(_mapping_or_empty(theme.payload if theme else {}).get("theme_name") or theme_id),
            theme_score=float(theme.score if theme else 0.0),
            symbol_theme_score=float(symbol_theme_score),
            phase=phase,
            risk_flags=risk_flags,
            theme_forced_admission=bool(theme.forced) if theme else False,
            candidate_intent=(
                "research_candidate_not_buy_signal"
                if bucket in RISK_WATCH_BUCKETS
                else ""
            ),
        )

    def _apply_caps(
        self,
        *,
        candidates: list[_SymbolCandidate],
        max_candidates: int,
        policy: ThemeGatePolicy,
    ) -> tuple[list[_SymbolCandidate], dict[str, str], int]:
        sorted_candidates = sorted(candidates, key=_symbol_sort_key)
        risk_watch = [candidate for candidate in sorted_candidates if candidate.is_risk_watch]
        non_risk = [candidate for candidate in sorted_candidates if not candidate.is_risk_watch]
        if not self.config.include_risk_watch:
            risk_watch_limit = 0
        elif max_candidates <= 0:
            risk_watch_limit = 0
        elif policy.risk_watch_max_ratio <= 0:
            risk_watch_limit = 0
        else:
            risk_watch_limit = max(1, int(math.floor(max_candidates * policy.risk_watch_max_ratio)))
        admitted_risk = risk_watch[:risk_watch_limit]
        cap_exclusions = {
            candidate.symbol: "theme_pool_risk_watch_ratio_cutoff"
            for candidate in risk_watch[risk_watch_limit:]
        }
        combined = sorted(non_risk + admitted_risk, key=_symbol_sort_key)
        selected = combined[: max(int(max_candidates), 0)]
        selected_symbols = {candidate.symbol for candidate in selected}
        for candidate in combined[max(int(max_candidates), 0):]:
            cap_exclusions.setdefault(candidate.symbol, "theme_pool_rank_cutoff")
        for candidate in candidates:
            if candidate.symbol not in selected_symbols and candidate.symbol not in cap_exclusions:
                cap_exclusions.setdefault(candidate.symbol, "theme_pool_rank_cutoff")
        return selected, cap_exclusions, risk_watch_limit

    def _build_exclusions(
        self,
        *,
        input_symbols: list[str],
        final_set: set[str],
        initial_exclusions: Mapping[str, str],
        cap_exclusions: Mapping[str, str],
    ) -> dict[str, str]:
        excluded: dict[str, str] = {}
        for symbol in input_symbols:
            if symbol in final_set:
                continue
            if symbol in initial_exclusions:
                excluded[symbol] = initial_exclusions[symbol]
            elif symbol in cap_exclusions:
                excluded[symbol] = cap_exclusions[symbol]
            else:
                excluded[symbol] = "theme_pool_rank_cutoff"
        return excluded

    def _symbol_metadata(
        self,
        *,
        symbol: str,
        candidate: _SymbolCandidate | None,
        symbol_primary_theme: Mapping[str, Any],
        theme_by_id: Mapping[str, _ThemeCandidate],
        symbol_scores: Mapping[str, Any],
        symbol_smoothed_scores: Mapping[str, Any],
        symbol_phase: Mapping[str, Any],
        symbol_risk_flags: Mapping[str, Any],
        symbol_reason: str,
        policy: ThemeGatePolicy,
        admitted: bool,
    ) -> dict[str, Any]:
        if candidate is not None:
            return {
                "admitted": admitted,
                "source": candidate.source if admitted else "none",
                "primary_theme_id": candidate.primary_theme_id,
                "primary_theme_name": candidate.primary_theme_name,
                "theme_score": float(candidate.theme_score),
                "symbol_theme_score": float(candidate.symbol_theme_score),
                "phase": candidate.phase,
                "risk_flags": list(candidate.risk_flags),
                "bucket": candidate.bucket,
                "candidate_intent": candidate.candidate_intent,
                "score_penalty": float(candidate.score_penalty),
                "raw_theme_pool_score": float(candidate.raw_score),
                "theme_pool_score": float(candidate.score if admitted else 0.0),
                "theme_forced_admission": bool(candidate.theme_forced_admission),
                "theme_policy_regime": str(policy.regime),
                "theme_pool_reason": symbol_reason,
            }
        theme_id = str(symbol_primary_theme.get(symbol, "") or "")
        theme = theme_by_id.get(theme_id)
        symbol_theme_score = self._symbol_theme_score(
            symbol=symbol,
            symbol_scores=symbol_scores,
            symbol_smoothed_scores=symbol_smoothed_scores,
        )
        return {
            "admitted": False,
            "source": "none",
            "primary_theme_id": theme_id,
            "primary_theme_name": str(_mapping_or_empty(theme.payload if theme else {}).get("theme_name") or theme_id),
            "theme_score": float(theme.score if theme else 0.0),
            "symbol_theme_score": float(symbol_theme_score),
            "phase": _normalized_phase(symbol_phase.get(symbol)) or (theme.phase if theme else ""),
            "risk_flags": list(_symbol_risk_flags(symbol=symbol, theme=theme, symbol_risk_flags=symbol_risk_flags)),
            "bucket": "none",
            "candidate_intent": "",
            "score_penalty": 0.0,
            "raw_theme_pool_score": 0.0,
            "theme_pool_score": 0.0,
            "theme_forced_admission": False,
            "theme_policy_regime": str(policy.regime),
            "theme_pool_reason": symbol_reason,
        }


def _theme_rank_score(
    *,
    payload: Mapping[str, Any],
    score: float,
    smoothed_or_raw_score: float,
) -> float:
    return _clamp(
        0.40 * score
        + 0.20 * smoothed_or_raw_score
        + 0.15 * _clamp(_safe_float(payload.get("breadth"), 0.0))
        + 0.10 * _clamp(_safe_float(payload.get("confidence"), 0.0))
        + 0.10 * _clamp(_safe_float(payload.get("acceleration"), 0.0))
        + 0.05 * _clamp(_safe_float(payload.get("volume_confirmation"), 0.0))
        - 0.15 * _clamp(_safe_float(payload.get("overextension_risk"), 0.0))
        - 0.20 * _clamp(_safe_float(payload.get("fake_breakout_risk"), 0.0))
    )


def _theme_rejection_entry(candidate: _ThemeCandidate, reason: str) -> dict[str, Any]:
    return {
        "theme_id": candidate.theme_id,
        "theme_name": str(candidate.payload.get("theme_name") or candidate.theme_id),
        "reason": reason,
        "quality_flags": list(candidate.quality_flags),
        "score": float(candidate.score),
        "phase": candidate.phase,
        "risk_flags": list(candidate.risk_flags),
        "theme_rank_score": float(candidate.rank_score),
    }


def _theme_rejection_reason(candidate: _ThemeCandidate) -> str:
    if candidate.quality_flags:
        return "theme_pool_theme_quality_gate_failed"
    return "theme_pool_theme_rank_cutoff"


def _theme_metadata(theme: _ThemeCandidate) -> dict[str, Any]:
    return {
        "theme_id": theme.theme_id,
        "theme_name": str(theme.payload.get("theme_name") or theme.theme_id),
        "theme_rank_score": float(theme.rank_score),
        "theme_score": float(theme.score),
        "phase": theme.phase,
        "risk_flags": list(theme.risk_flags),
        "quality_flags": list(theme.quality_flags),
        "forced": bool(theme.forced),
        "force_reason": str(theme.force_reason),
        "original_rejection_reason": str(theme.original_rejection_reason),
    }


def _extract_markov_payload(global_context: GlobalContext) -> Mapping[str, Any]:
    metadata = getattr(global_context, "metadata", {}) or {}
    if isinstance(metadata, Mapping) and isinstance(metadata.get("markov_regime"), Mapping):
        return metadata.get("markov_regime", {})  # type: ignore[return-value]
    regime_params = getattr(global_context, "regime_params", {}) or {}
    if isinstance(regime_params, Mapping) and isinstance(regime_params.get("markov"), Mapping):
        return regime_params.get("markov", {})  # type: ignore[return-value]
    return {}


def _symbol_pool_score(
    *,
    symbol: str,
    symbol_theme_score: float,
    quant_score: float,
    symbol_market_state: Mapping[str, Any],
    liquidity_scores: Mapping[str, Any],
) -> float:
    state = _mapping_or_empty(symbol_market_state.get(symbol))
    quant_score_norm = _clamp((quant_score + 1.0) / 2.0)
    drawdown_penalty = _clamp(_safe_float(state.get("max_drawdown_pct"), 0.0) / 0.18)
    return _clamp(
        0.25 * _clamp(symbol_theme_score)
        + 0.20 * quant_score_norm
        + 0.20 * _clamp(_safe_float(state.get("momentum_strength"), 0.0))
        + 0.15 * _clamp(_safe_float(state.get("breakout_readiness"), 0.0))
        + 0.10 * _clamp(_safe_float(state.get("volume_confirmation"), 0.0))
        + 0.05 * _clamp(_safe_float(state.get("trend_stability"), 0.0))
        + 0.05 * _clamp(_safe_float(liquidity_scores.get(symbol, 1.0), 1.0))
        - 0.20 * _clamp(_safe_float(state.get("fake_breakout_risk"), 0.0))
        - 0.15 * drawdown_penalty
    )


def _symbol_risk_flags(
    *,
    symbol: str,
    theme: _ThemeCandidate | None,
    symbol_risk_flags: Mapping[str, Any],
) -> tuple[str, ...]:
    return _dedupe_texts(symbol_risk_flags.get(symbol, [])) + (theme.risk_flags if theme else ())


def _source_for_bucket(bucket: str) -> str:
    if bucket == "core":
        return "core"
    if bucket == "forced_theme":
        return "core"
    if bucket == "residual_theme_alpha":
        return "residual_theme"
    if bucket in RISK_WATCH_BUCKETS:
        return "risk_watch"
    return "extended"


def _symbol_sort_key(candidate: _SymbolCandidate) -> tuple[int, float, str, str]:
    return (
        BUCKET_PRIORITIES.get(candidate.bucket, 9),
        -float(candidate.score),
        candidate.primary_theme_id,
        candidate.symbol,
    )


def _score_penalty(
    *,
    bucket: str,
    symbol_theme_score: float,
    theme_score: float,
    policy: ThemeGatePolicy,
) -> float:
    penalty = 0.0
    if bucket == "extended_low_score":
        penalty = 0.04 + max(policy.min_symbol_score - symbol_theme_score, 0.0) * 0.25
    elif bucket == "extended_low_breadth":
        penalty = 0.08
    elif bucket == "forced_theme":
        penalty = 0.06 + max(policy.min_theme_score - theme_score, 0.0) * 0.15
    elif bucket == "residual_theme_alpha":
        penalty = 0.07 + max(policy.min_theme_score - theme_score, 0.0) * 0.12
    elif bucket == "risk_watch_distribution":
        penalty = 0.25
    elif bucket == "risk_watch_overextended":
        penalty = 0.18
    elif bucket == "risk_watch_fake_breakout":
        penalty = 0.22
    elif bucket == "extended":
        penalty = max(policy.min_theme_score - theme_score, 0.0) * 0.10
    return _clamp(penalty * max(policy.score_penalty_multiplier, 0.0), 0.0, 0.80)


def _count_by(candidates: list[_SymbolCandidate], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = str(getattr(candidate, attr))
        counts[key] = counts.get(key, 0) + 1
    return counts

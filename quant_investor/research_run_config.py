from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from quant_investor.config import config as runtime_config
from quant_investor.llm_provider_priority import (
    RoleModelConfig,
    normalize_model_name,
    resolve_runtime_role_models,
)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _dedupe_models(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        model = normalize_model_name(str(value or "").strip())
        if not model or model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    return ordered


def _mapping_str(mapping: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _mapping_float(mapping: Mapping[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        value = mapping.get(key)
        if value is None or value == "":
            continue
        return float(value)
    return float(default)


def _mapping_int(mapping: Mapping[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        value = mapping.get(key)
        if value is None or value == "":
            continue
        return int(value)
    return int(default)


def _mapping_int_list(mapping: Mapping[str, Any], *keys: str, default: tuple[int, ...]) -> tuple[int, ...]:
    for key in keys:
        value = mapping.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple, set)):
            result = [int(item) for item in value if str(item).strip()]
            return tuple(result) if result else tuple(default)
        text = str(value).strip()
        if not text:
            continue
        result: list[int] = []
        for item in text.split(","):
            part = str(item).strip()
            if not part:
                continue
            result.append(int(part))
        return tuple(result) if result else tuple(default)
    return tuple(default)


def _mapping_bool(
    mapping: Mapping[str, Any],
    *,
    enabled_key: str,
    disabled_key: str | None = None,
    default: bool,
) -> bool:
    if enabled_key in mapping and mapping.get(enabled_key) is not None:
        return bool(mapping.get(enabled_key))
    if disabled_key and disabled_key in mapping and mapping.get(disabled_key) is not None:
        return not bool(mapping.get(disabled_key))
    return bool(default)


def _requested_review_priority(mapping: Mapping[str, Any]) -> list[str]:
    explicit_priority = _dedupe_models(_as_str_list(mapping.get("review_model_priority")))
    if explicit_priority:
        return explicit_priority
    legacy_models = _dedupe_models(
        [
            _mapping_str(mapping, "agent_model"),
            _mapping_str(mapping, "agent_fallback_model"),
            _mapping_str(mapping, "master_model"),
            _mapping_str(mapping, "master_fallback_model"),
        ]
    )
    return legacy_models


@dataclass(frozen=True)
class ResolvedReviewModels:
    review_model_priority: list[str] = field(default_factory=list)
    branch_primary_model: str = ""
    branch_fallback_model: str = ""
    branch_candidate_models: list[str] = field(default_factory=list)
    branch_source: str = "default"
    master_primary_model: str = ""
    master_fallback_model: str = ""
    master_candidate_models: list[str] = field(default_factory=list)
    master_source: str = "default"
    master_reasoning_effort: str = "high"
    agent_timeout: float = runtime_config.DEFAULT_AGENT_TIMEOUT_SECONDS
    master_timeout: float = runtime_config.DEFAULT_MASTER_TIMEOUT_SECONDS

    @property
    def branch(self) -> RoleModelConfig:
        return RoleModelConfig(
            role="branch",
            primary_model=self.branch_primary_model,
            fallback_model=self.branch_fallback_model,
            candidate_models=list(self.branch_candidate_models),
            source=self.branch_source,
        )

    @property
    def master(self) -> RoleModelConfig:
        return RoleModelConfig(
            role="master",
            primary_model=self.master_primary_model,
            fallback_model=self.master_fallback_model,
            candidate_models=list(self.master_candidate_models),
            source=self.master_source,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ResolvedReviewModels":
        requested_priority = _requested_review_priority(mapping)
        branch_config, master_config = resolve_runtime_role_models(
            review_model_priority=requested_priority,
            agent_model=_mapping_str(mapping, "agent_model"),
            agent_fallback_model=_mapping_str(mapping, "agent_fallback_model"),
            master_model=_mapping_str(mapping, "master_model"),
            master_fallback_model=_mapping_str(mapping, "master_fallback_model"),
        )
        return cls(
            review_model_priority=list(requested_priority),
            branch_primary_model=branch_config.primary_model,
            branch_fallback_model=branch_config.fallback_model,
            branch_candidate_models=list(branch_config.candidate_models),
            branch_source=branch_config.source,
            master_primary_model=master_config.primary_model,
            master_fallback_model=master_config.fallback_model,
            master_candidate_models=list(master_config.candidate_models),
            master_source=master_config.source,
            master_reasoning_effort=_mapping_str(
                mapping,
                "master_reasoning_effort",
                default="high",
            )
            or "high",
            agent_timeout=_mapping_float(
                mapping,
                "agent_timeout",
                default=runtime_config.DEFAULT_AGENT_TIMEOUT_SECONDS,
            ),
            master_timeout=_mapping_float(
                mapping,
                "master_timeout",
                default=runtime_config.DEFAULT_MASTER_TIMEOUT_SECONDS,
            ),
        )

    def to_runtime_kwargs(self) -> dict[str, Any]:
        return {
            "review_model_priority": list(self.review_model_priority),
            "agent_model": self.branch_primary_model,
            "agent_fallback_model": self.branch_fallback_model,
            "master_model": self.master_primary_model,
            "master_fallback_model": self.master_fallback_model,
            "master_reasoning_effort": self.master_reasoning_effort,
            "agent_timeout": float(self.agent_timeout),
            "master_timeout": float(self.master_timeout),
        }

    def apply_to_mapping(self, mapping: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(mapping)
        payload.update(self.to_runtime_kwargs())
        return payload


@dataclass(frozen=True)
class ResearchRunConfig:
    stock_pool: list[str]
    market: str
    lookback_years: float = 1.0
    total_capital: float = 1_000_000.0
    risk_level: str = "中等"
    enable_macro: bool = True
    enable_quant: bool = True
    enable_kline: bool = True
    enable_fundamental: bool = True
    enable_intelligence: bool = True
    enable_agent_layer: bool = True
    kline_backend: str = "hybrid"
    allow_synthetic_for_research: bool = False
    enable_document_semantics: bool = True
    review_models: ResolvedReviewModels = field(default_factory=ResolvedReviewModels)
    universe_key: str = "full_a"
    funnel_profile: str = runtime_config.FUNNEL_PROFILE
    max_candidates: int = runtime_config.FUNNEL_MAX_CANDIDATES
    trend_windows: tuple[int, ...] = field(default_factory=lambda: tuple(runtime_config.FUNNEL_TREND_WINDOWS))
    volume_spike_threshold: float = runtime_config.FUNNEL_VOLUME_SPIKE_THRESHOLD
    breakout_distance_pct: float = runtime_config.FUNNEL_BREAKOUT_DISTANCE_PCT
    sector_bucket_limit: int = runtime_config.FUNNEL_SECTOR_BUCKET_LIMIT
    recall_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        recall_context: Mapping[str, Any] | None = None,
    ) -> "ResearchRunConfig":
        review_models = ResolvedReviewModels.from_mapping(mapping)
        stock_pool = _as_str_list(mapping.get("stock_pool") or mapping.get("stocks"))
        return cls(
            stock_pool=stock_pool,
            market=_mapping_str(mapping, "market", default="CN") or "CN",
            lookback_years=_mapping_float(mapping, "lookback_years", "lookback", default=1.0),
            total_capital=_mapping_float(mapping, "total_capital", "capital", default=1_000_000.0),
            risk_level=_mapping_str(mapping, "risk_level", "risk", default="中等") or "中等",
            enable_macro=_mapping_bool(
                mapping,
                enabled_key="enable_macro",
                disabled_key="no_macro",
                default=True,
            ),
            enable_quant=_mapping_bool(
                mapping,
                enabled_key="enable_quant",
                disabled_key="no_quant",
                default=True,
            ),
            enable_kline=_mapping_bool(
                mapping,
                enabled_key="enable_kline",
                disabled_key="no_kline",
                default=True,
            ),
            enable_fundamental=_mapping_bool(
                mapping,
                enabled_key="enable_fundamental",
                disabled_key="no_fundamental",
                default=True,
            ),
            enable_intelligence=_mapping_bool(
                mapping,
                enabled_key="enable_intelligence",
                disabled_key="no_intelligence",
                default=True,
            ),
            enable_agent_layer=_mapping_bool(
                mapping,
                enabled_key="enable_agent_layer",
                disabled_key="no_agent_layer",
                default=True,
            ),
            kline_backend=_mapping_str(mapping, "kline_backend", default="hybrid") or "hybrid",
            allow_synthetic_for_research=_mapping_bool(
                mapping,
                enabled_key="allow_synthetic_for_research",
                default=False,
            ),
            enable_document_semantics=_mapping_bool(
                mapping,
                enabled_key="enable_document_semantics",
                disabled_key="disable_document_semantics",
                default=True,
            ),
            review_models=review_models,
            universe_key=_mapping_str(mapping, "universe_key", default="full_a") or "full_a",
            funnel_profile=_mapping_str(
                mapping,
                "funnel_profile",
                default=runtime_config.FUNNEL_PROFILE,
            )
            or runtime_config.FUNNEL_PROFILE,
            max_candidates=_mapping_int(
                mapping,
                "max_candidates",
                "funnel_max_candidates",
                default=runtime_config.FUNNEL_MAX_CANDIDATES,
            ),
            trend_windows=_mapping_int_list(
                mapping,
                "trend_windows",
                default=tuple(runtime_config.FUNNEL_TREND_WINDOWS),
            ),
            volume_spike_threshold=_mapping_float(
                mapping,
                "volume_spike_threshold",
                default=runtime_config.FUNNEL_VOLUME_SPIKE_THRESHOLD,
            ),
            breakout_distance_pct=_mapping_float(
                mapping,
                "breakout_distance_pct",
                default=runtime_config.FUNNEL_BREAKOUT_DISTANCE_PCT,
            ),
            sector_bucket_limit=_mapping_int(
                mapping,
                "sector_bucket_limit",
                default=runtime_config.FUNNEL_SECTOR_BUCKET_LIMIT,
            ),
            recall_context=dict(recall_context or {}),
        )

    def to_quant_investor_kwargs(self, *, verbose: bool) -> dict[str, Any]:
        payload = {
            "stock_pool": list(self.stock_pool),
            "market": self.market,
            "lookback_years": float(self.lookback_years),
            "total_capital": float(self.total_capital),
            "risk_level": self.risk_level,
            "enable_macro": bool(self.enable_macro),
            "enable_quant": bool(self.enable_quant),
            "enable_kline": bool(self.enable_kline),
            "enable_fundamental": bool(self.enable_fundamental),
            "enable_intelligence": bool(self.enable_intelligence),
            "kline_backend": self.kline_backend,
            "allow_synthetic_for_research": bool(self.allow_synthetic_for_research),
            "enable_document_semantics": bool(self.enable_document_semantics),
            "enable_agent_layer": bool(self.enable_agent_layer),
            "universe_key": self.universe_key,
            "funnel_profile": str(self.funnel_profile or runtime_config.FUNNEL_PROFILE),
            "max_candidates": int(self.max_candidates),
            "trend_windows": list(self.trend_windows),
            "volume_spike_threshold": float(self.volume_spike_threshold),
            "breakout_distance_pct": float(self.breakout_distance_pct),
            "sector_bucket_limit": int(self.sector_bucket_limit),
            "recall_context": dict(self.recall_context),
            "verbose": bool(verbose),
        }
        payload.update(self.review_models.to_runtime_kwargs())
        return payload

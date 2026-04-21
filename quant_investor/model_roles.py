from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quant_investor.llm_gateway import has_provider_for_model


@dataclass
class ModelRoleResolution:
    role: str = ""
    primary_model: str = ""
    fallback_model: str = ""
    resolved_model: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    provider_available: bool = False
    fallback_provider_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "resolved_model": self.resolved_model,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "provider_available": self.provider_available,
            "fallback_provider_available": self.fallback_provider_available,
            "metadata": dict(self.metadata),
        }


def resolve_model_role(
    *,
    role: str,
    primary_model: str,
    fallback_model: str = "",
) -> ModelRoleResolution:
    primary = str(primary_model or "").strip()
    fallback = str(fallback_model or "").strip()
    primary_available = bool(primary and has_provider_for_model(primary))
    fallback_available = bool(fallback and has_provider_for_model(fallback))

    if primary_available:
        return ModelRoleResolution(
            role=role,
            primary_model=primary,
            fallback_model=fallback,
            resolved_model=primary,
            fallback_used=False,
            fallback_reason="",
            provider_available=True,
            fallback_provider_available=fallback_available,
        )

    if fallback_available:
        reason = f"primary_model_unavailable:{primary or 'unset'}"
        return ModelRoleResolution(
            role=role,
            primary_model=primary,
            fallback_model=fallback,
            resolved_model=fallback,
            fallback_used=True,
            fallback_reason=reason,
            provider_available=False,
            fallback_provider_available=True,
            metadata={"primary_available": False, "fallback_used": True},
        )

    raise RuntimeError(
        f"{role} 模型不可用：primary={primary!r}, fallback={fallback!r}，"
        "请先配置对应 API Key 或调整模型配置。"
    )

"""Injected, opt-in provider boundary for macro observations.

This module never discovers credentials or opens a network connection by itself.
Transports must return already normalized PIT rows including authoritative
``release_at`` and conservative ``available_at`` timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from quant_investor.macro.contracts import (
    MacroObservation,
    is_official_source,
    is_tushare_source,
)
from quant_investor.macro.store import DEFAULT_OBSERVATIONS_ROOT, publish_observations


@dataclass(frozen=True)
class MacroFetchRequest:
    market: str
    as_of: str
    indicator_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProviderFetchResult:
    observations: tuple[MacroObservation, ...] = ()
    provider_manifest: Mapping[str, Any] = field(default_factory=dict)
    status: str = "OK"
    blockers: tuple[str, ...] = ()


class MacroProvider(Protocol):
    provider_id: str

    def fetch(self, request: MacroFetchRequest) -> ProviderFetchResult:
        ...


NormalizedTransport = Callable[[str, MacroFetchRequest], Iterable[Mapping[str, Any]]]


class _InjectedProvider:
    provider_id = "injected"
    required_source_token = ""

    def __init__(self, transport: NormalizedTransport | None) -> None:
        self._transport = transport

    def _source_matches(self, source: str) -> bool:
        return self.required_source_token in source

    def fetch(self, request: MacroFetchRequest) -> ProviderFetchResult:
        if self._transport is None:
            return ProviderFetchResult(
                provider_manifest={"provider_id": self.provider_id, "transport_injected": False},
                status="blocked_provider_unavailable",
                blockers=(f"provider_unavailable:{self.provider_id}",),
            )
        observations: list[MacroObservation] = []
        blockers: list[str] = []
        requested = tuple(sorted(set(request.indicator_ids)))
        for indicator_id in requested:
            try:
                rows = list(self._transport(indicator_id, request))
                for row in rows:
                    candidate = dict(row)
                    if str(candidate.get("indicator_id") or "") != indicator_id:
                        raise ValueError("provider_indicator_mismatch")
                    source = str(candidate.get("source_system") or "").lower()
                    if not self._source_matches(source):
                        raise ValueError("provider_source_provenance_mismatch")
                    observations.append(MacroObservation.from_mapping(candidate))
            except Exception as exc:
                blockers.append(f"provider_fetch_failed:{self.provider_id}:{indicator_id}:{type(exc).__name__}:{exc}")
        status = "OK" if not blockers else ("degraded" if observations else "blocked")
        return ProviderFetchResult(
            observations=tuple(observations),
            provider_manifest={
                "provider_id": self.provider_id,
                "transport_injected": True,
                "requested_indicator_ids": list(requested),
                "observation_count": len(observations),
            },
            status=status,
            blockers=tuple(blockers),
        )


class OfficialMacroProvider(_InjectedProvider):
    provider_id = "official"
    required_source_token = "official"

    def _source_matches(self, source: str) -> bool:
        return is_official_source(source)


class TushareMacroProvider(_InjectedProvider):
    provider_id = "tushare_fallback"
    required_source_token = "tushare"

    def _source_matches(self, source: str) -> bool:
        return is_tushare_source(source)


def fetch_official_first(
    request: MacroFetchRequest,
    *,
    official_provider: MacroProvider | None,
    tushare_provider: MacroProvider | None = None,
    allow_tushare_fallback: bool = False,
) -> ProviderFetchResult:
    """Fetch official rows first and use Tushare only for missing indicators."""

    official = (official_provider or OfficialMacroProvider(None)).fetch(request)
    observations = list(official.observations)
    manifests: list[Mapping[str, Any]] = [official.provider_manifest]
    blockers = list(official.blockers)
    covered = {item.indicator_id for item in observations}
    missing = tuple(item for item in request.indicator_ids if item not in covered)
    if missing and allow_tushare_fallback:
        fallback_request = MacroFetchRequest(request.market, request.as_of, missing)
        fallback = (tushare_provider or TushareMacroProvider(None)).fetch(fallback_request)
        observations.extend(fallback.observations)
        manifests.append(fallback.provider_manifest)
        blockers.extend(fallback.blockers)
    elif missing:
        blockers.extend(f"official_observation_missing:{item}" for item in missing)
    final_covered = {item.indicator_id for item in observations}
    unresolved = sorted(set(request.indicator_ids) - final_covered)
    blockers.extend(f"observation_unresolved:{item}" for item in unresolved)
    status = "OK" if not blockers else ("degraded" if observations else "blocked")
    return ProviderFetchResult(
        observations=tuple(observations),
        provider_manifest={"strategy": "official_first", "providers": manifests},
        status=status,
        blockers=tuple(sorted(set(blockers))),
    )


def maintain_macro_observations(
    *,
    local_observations: Iterable[Mapping[str, Any] | MacroObservation] = (),
    market: str = "CN",
    as_of: str,
    indicator_ids: Sequence[str] = (),
    root: str = str(DEFAULT_OBSERVATIONS_ROOT),
    run_id: str,
    allow_live: bool = False,
    allow_tushare_fallback: bool = False,
    official_provider: MacroProvider | None = None,
    tushare_provider: MacroProvider | None = None,
) -> dict[str, Any]:
    """Validate local rows and optionally call explicitly injected live providers."""

    candidates: list[MacroObservation | Mapping[str, Any]] = list(local_observations)
    provider_result = ProviderFetchResult(
        provider_manifest={"strategy": "local_only", "live_requested": False},
        status="not_requested",
    )
    if allow_live:
        request = MacroFetchRequest(str(market).upper(), as_of, tuple(sorted(set(indicator_ids))))
        provider_result = fetch_official_first(
            request,
            official_provider=official_provider,
            tushare_provider=tushare_provider,
            allow_tushare_fallback=allow_tushare_fallback,
        )
        candidates.extend(provider_result.observations)
        fatal_provider_blockers = [
            item
            for item in provider_result.blockers
            if item.startswith(("provider_fetch_failed:", "observation_unresolved:"))
        ]
        if fatal_provider_blockers:
            return {
                "status": "blocked",
                "promoted": False,
                "reason": "provider_result_not_publishable",
                "blockers": list(provider_result.blockers),
                "provider_manifest": dict(provider_result.provider_manifest),
            }
    if not candidates:
        return {
            "status": provider_result.status if allow_live else "no_update",
            "promoted": False,
            "reason": "no_valid_observations",
            "blockers": list(provider_result.blockers),
            "provider_manifest": dict(provider_result.provider_manifest),
        }
    promoted = publish_observations(
        candidates,
        root=root,
        run_id=run_id,
        metadata={
            "market": str(market).upper(),
            "as_of": as_of,
            "provider_manifest": dict(provider_result.provider_manifest),
            "provider_status": provider_result.status,
            "provider_blockers": list(provider_result.blockers),
        },
    )
    return {**promoted, "provider_status": provider_result.status, "blockers": list(provider_result.blockers)}


__all__ = [
    "MacroFetchRequest",
    "MacroProvider",
    "OfficialMacroProvider",
    "ProviderFetchResult",
    "TushareMacroProvider",
    "fetch_official_first",
    "maintain_macro_observations",
]

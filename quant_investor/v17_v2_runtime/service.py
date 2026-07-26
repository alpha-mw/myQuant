"""Application service for the isolated v17 protocol-v2 research runtime.

The service intentionally has no broker, network, LLM, or production-routing
surface.  It verifies the frozen package and executes only the pure in-memory
pipeline.  Durable source and shadow-ledger writes remain explicit lower-level
operations guarded by :mod:`quant_investor.v17_v2_runtime.gate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from quant_investor.v17_v2_contract.resources import (
    expected_ledger_implementation_bindings,
    load_package_manifest,
    load_packaged_json,
    verify_packaged_assets,
)
from quant_investor.v17_v2_contract.validators import validate_source_role_matrix

from .pipeline import PipelineInput, PipelineResult, run_deterministic_pipeline

PROTOCOL_VERSION = "myquant.v17.v2"
RUNTIME_AUTHORITY = False

_PIPELINE_INPUT_KEYS = frozenset(
    {
        "cutoff",
        "strategy_id",
        "fundamental_rows",
        "fundamental_history",
        "forward_observations",
        "price_history",
        "timing_observations",
        "deep_response",
        "sealed_evidence_ids",
        "holdings",
        "cash",
        "nav",
        "risk_policy",
        "cost_policy",
        "tradability",
        "risk_model",
        "clusters",
        "macro",
        "markov",
        "portfolio_candidates",
    }
)


class RuntimeServiceError(ValueError):
    """A fail-closed service boundary rejection."""


@dataclass(frozen=True)
class RuntimeReadiness:
    protocol_version: str
    distribution_version: str
    matrix_status: str
    runtime_usable: bool
    pending_registry: tuple[str, ...]
    authority: bool
    packaged_asset_count: int
    implementation_binding_count: int

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "distribution_version": self.distribution_version,
            "matrix_status": self.matrix_status,
            "runtime_usable": self.runtime_usable,
            "pending_registry": list(self.pending_registry),
            "authority": False,
            "packaged_asset_count": self.packaged_asset_count,
            "implementation_binding_count": self.implementation_binding_count,
        }


def verify_runtime() -> RuntimeReadiness:
    """Verify exact packaged bytes and the Phase-1 admission contract."""

    verified_assets = verify_packaged_assets()
    package = load_package_manifest()
    matrix = validate_source_role_matrix(load_packaged_json("resources/source_role_matrix.v1.json"))
    distribution = package.get("distribution")
    if type(distribution) is not dict or type(distribution.get("version")) is not str:
        raise RuntimeServiceError("package distribution version is missing")
    pending = matrix.get("pending_registry")
    if type(pending) is not list or any(type(item) is not str for item in pending):
        raise RuntimeServiceError("source role matrix pending_registry is invalid")
    readiness = RuntimeReadiness(
        protocol_version=PROTOCOL_VERSION,
        distribution_version=distribution["version"],
        matrix_status=str(matrix.get("completeness")),
        runtime_usable=matrix.get("runtime_usable") is True,
        pending_registry=tuple(pending),
        authority=False,
        packaged_asset_count=len(verified_assets),
        implementation_binding_count=len(expected_ledger_implementation_bindings()),
    )
    if (
        readiness.matrix_status != "COMPLETE"
        or not readiness.runtime_usable
        or readiness.pending_registry
        or package.get("runtime_usable") is not True
        or package.get("authority") is not False
        or matrix.get("authority") is not False
    ):
        raise RuntimeServiceError("Phase-1 runtime admission contract is not complete")
    return readiness


def _tuple_of_mappings(value: Any, *, label: str) -> tuple[Mapping[str, Any], ...]:
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise RuntimeServiceError(f"{label} must be an array of objects")
    return tuple(value)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise RuntimeServiceError(f"{label} must be an object")
    return value


def pipeline_input_from_mapping(payload: Mapping[str, Any]) -> PipelineInput:
    """Create a typed pipeline input without fallback or implicit defaults."""

    if type(payload) is not dict:
        raise RuntimeServiceError("analysis input root must be an object")
    if frozenset(payload) != _PIPELINE_INPUT_KEYS:
        missing = sorted(_PIPELINE_INPUT_KEYS.difference(payload))
        extra = sorted(set(payload).difference(_PIPELINE_INPUT_KEYS))
        raise RuntimeServiceError(f"analysis input keys mismatch; missing={missing}, extra={extra}")
    price_history = _mapping(payload["price_history"], label="price_history")
    normalized_prices: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for symbol, rows in price_history.items():
        if type(symbol) is not str:
            raise RuntimeServiceError("price_history keys must be strings")
        normalized_prices[symbol] = _tuple_of_mappings(
            rows,
            label=f"price_history.{symbol}",
        )
    evidence = _mapping(payload["sealed_evidence_ids"], label="sealed_evidence_ids")
    normalized_evidence: dict[str, tuple[str, ...]] = {}
    for symbol, ids in evidence.items():
        if (
            type(symbol) is not str
            or type(ids) is not list
            or any(type(item) is not str for item in ids)
        ):
            raise RuntimeServiceError("sealed_evidence_ids must map symbols to string arrays")
        normalized_evidence[symbol] = tuple(ids)
    return PipelineInput(
        cutoff=payload["cutoff"],
        strategy_id=payload["strategy_id"],
        fundamental_rows=_tuple_of_mappings(payload["fundamental_rows"], label="fundamental_rows"),
        fundamental_history=_tuple_of_mappings(
            payload["fundamental_history"], label="fundamental_history"
        ),
        forward_observations=_tuple_of_mappings(
            payload["forward_observations"], label="forward_observations"
        ),
        price_history=normalized_prices,
        timing_observations=_tuple_of_mappings(
            payload["timing_observations"], label="timing_observations"
        ),
        deep_response=_mapping(payload["deep_response"], label="deep_response"),
        sealed_evidence_ids=normalized_evidence,
        holdings=_mapping(payload["holdings"], label="holdings"),
        cash=payload["cash"],
        nav=payload["nav"],
        risk_policy=_mapping(payload["risk_policy"], label="risk_policy"),
        cost_policy=_mapping(payload["cost_policy"], label="cost_policy"),
        tradability=_mapping(payload["tradability"], label="tradability"),
        risk_model=_mapping(payload["risk_model"], label="risk_model"),
        clusters=_mapping(payload["clusters"], label="clusters"),
        macro=_mapping(payload["macro"], label="macro"),
        markov=_mapping(payload["markov"], label="markov"),
        portfolio_candidates=_tuple_of_mappings(
            payload["portfolio_candidates"], label="portfolio_candidates"
        ),
    )


def analyze_mapping(payload: Mapping[str, Any]) -> PipelineResult:
    """Verify runtime admission and execute one deterministic shadow analysis."""

    verify_runtime()
    return run_deterministic_pipeline(pipeline_input_from_mapping(payload))


__all__ = [
    "RUNTIME_AUTHORITY",
    "RuntimeReadiness",
    "RuntimeServiceError",
    "analyze_mapping",
    "pipeline_input_from_mapping",
    "verify_runtime",
]

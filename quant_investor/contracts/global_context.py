from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GlobalContext:
    """Global market context shared across symbol research."""

    market: str = ""
    as_of_date: str = ""
    latest_trade_date: str = ""
    universe: list[str] = field(default_factory=list)
    universe_hash: str = ""
    completeness_passed: bool = False
    data_gate: dict[str, Any] = field(default_factory=dict)
    macro_regime: dict[str, Any] = field(default_factory=dict)
    quant_factor_scores: dict[str, float] = field(default_factory=dict)
    style_exposures: dict[str, dict[str, float]] = field(default_factory=dict)
    liquidity_snapshot: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_budget: dict[str, Any] = field(default_factory=dict)
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    phase1_context: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    cache_key: str = ""
    cache_path: str = ""
    schema_version: str = "global-context.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalContext":
        return cls(
            market=str(payload.get("market", "")),
            as_of_date=str(payload.get("as_of_date", "")),
            latest_trade_date=str(payload.get("latest_trade_date", "")),
            universe=[str(item) for item in payload.get("universe", [])],
            universe_hash=str(payload.get("universe_hash", "")),
            completeness_passed=bool(payload.get("completeness_passed", False)),
            data_gate=dict(payload.get("data_gate", {})),
            macro_regime=dict(payload.get("macro_regime", {})),
            quant_factor_scores={
                str(key): float(value)
                for key, value in dict(payload.get("quant_factor_scores", {})).items()
            },
            style_exposures=dict(payload.get("style_exposures", {})),
            liquidity_snapshot=dict(payload.get("liquidity_snapshot", {})),
            risk_budget=dict(payload.get("risk_budget", {})),
            correlation_matrix=dict(payload.get("correlation_matrix", {})),
            phase1_context=dict(payload.get("phase1_context", {})),
            source_metadata=dict(payload.get("source_metadata", {})),
            cache_key=str(payload.get("cache_key", "")),
            cache_path=str(payload.get("cache_path", "")),
            schema_version=str(payload.get("schema_version", "global-context.v1")),
        )

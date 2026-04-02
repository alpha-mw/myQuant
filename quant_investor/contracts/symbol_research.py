from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SymbolResearchPacket:
    """Standardized per-symbol research artifact."""

    symbol: str
    market: str = ""
    as_of_date: str = ""
    global_context_ref: str = ""
    kline_view: dict[str, Any] = field(default_factory=dict)
    fundamental_view: dict[str, Any] = field(default_factory=dict)
    intelligence_view: dict[str, Any] = field(default_factory=dict)
    merged_score: float = 0.0
    confidence: float = 0.0
    risk_flags: list[str] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SymbolResearchPacket":
        return cls(
            symbol=str(payload.get("symbol", "")),
            market=str(payload.get("market", "")),
            as_of_date=str(payload.get("as_of_date", "")),
            global_context_ref=str(payload.get("global_context_ref", "")),
            kline_view=dict(payload.get("kline_view", {})),
            fundamental_view=dict(payload.get("fundamental_view", {})),
            intelligence_view=dict(payload.get("intelligence_view", {})),
            merged_score=float(payload.get("merged_score", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            risk_flags=[str(item) for item in payload.get("risk_flags", [])],
            evidence={str(key): [str(item) for item in value] for key, value in dict(payload.get("evidence", {})).items()},
            metadata=dict(payload.get("metadata", {})),
        )

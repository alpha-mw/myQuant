from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ShortlistItem:
    """Deterministic shortlist artifact before portfolio construction."""

    symbol: str
    rank_score: float = 0.0
    expected_return: float = 0.0
    downside_risk: float = 0.0
    liquidity_score: float = 0.0
    conviction: str = "neutral"
    packet_ref: str = ""
    sector: str = ""
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShortlistItem":
        return cls(
            symbol=str(payload.get("symbol", "")),
            rank_score=float(payload.get("rank_score", 0.0)),
            expected_return=float(payload.get("expected_return", 0.0)),
            downside_risk=float(payload.get("downside_risk", 0.0)),
            liquidity_score=float(payload.get("liquidity_score", 0.0)),
            conviction=str(payload.get("conviction", "neutral")),
            packet_ref=str(payload.get("packet_ref", "")),
            sector=str(payload.get("sector", "")),
            rationale=str(payload.get("rationale", "")),
            metadata=dict(payload.get("metadata", {})),
        )

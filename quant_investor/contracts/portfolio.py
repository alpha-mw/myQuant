from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PortfolioDecision:
    """Structured portfolio decision for a single symbol."""

    symbol: str
    action: str = "watch"
    target_weight: float = 0.0
    entry_price: float | None = None
    target_price: float | None = None
    stop_loss: float | None = None
    thesis: str = ""
    what_if_scenarios: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    packet_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortfolioDecision":
        return cls(
            symbol=str(payload.get("symbol", "")),
            action=str(payload.get("action", "watch")),
            target_weight=float(payload.get("target_weight", 0.0)),
            entry_price=(
                float(payload["entry_price"])
                if payload.get("entry_price") is not None
                else None
            ),
            target_price=(
                float(payload["target_price"])
                if payload.get("target_price") is not None
                else None
            ),
            stop_loss=(
                float(payload["stop_loss"])
                if payload.get("stop_loss") is not None
                else None
            ),
            thesis=str(payload.get("thesis", "")),
            what_if_scenarios=[dict(item) for item in payload.get("what_if_scenarios", [])],
            confidence=float(payload.get("confidence", 0.0)),
            packet_ref=str(payload.get("packet_ref", "")),
            metadata=dict(payload.get("metadata", {})),
        )

"""Neutral market data read result for Parquet-backed runtime paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataQualityIssue:
    """Version-neutral diagnostic emitted by strict market-data readers."""

    path: str = ""
    symbol: str = ""
    category: str = ""
    universe_key: str = ""
    issue_type: str = ""
    severity: str = "warning"
    message: str = ""
    resolver_strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketDataReadResult:
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    path: str = ""
    symbol: str = ""
    category: str = ""
    universe_key: str = ""
    resolver_trace: dict[str, Any] = field(default_factory=dict)
    issues: list[DataQualityIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame"] = (
            self.frame.to_dict(orient="records")
            if isinstance(self.frame, pd.DataFrame)
            else []
        )
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload


__all__ = ["DataQualityIssue", "MarketDataReadResult"]

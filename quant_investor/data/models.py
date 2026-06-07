"""Source-backed data model definitions for the public data layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OHLCVData:
    symbol: str = ""
    date: str = ""
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    amount: float | None = None
    adj_close: float | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TickData:
    symbol: str = ""
    timestamp: str = ""
    price: float | None = None
    volume: float | None = None
    amount: float | None = None
    bid: float | None = None
    ask: float | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FundamentalData:
    symbol: str = ""
    report_date: str = ""
    roe: float | None = None
    roa: float | None = None
    gross_margin: float | None = None
    net_margin: float | None = None
    revenue_growth: float | None = None
    profit_growth: float | None = None
    debt_ratio: float | None = None
    current_ratio: float | None = None
    cash_flow: float | None = None
    pe: float | None = None
    pb: float | None = None
    ps: float | None = None
    dividend_yield: float | None = None
    eps: float | None = None
    revenue: float | None = None
    net_profit: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroData:
    market: str = ""
    as_of: str = ""
    macro_score: float | None = None
    liquidity_score: float | None = None
    volatility_percentile: float | None = None
    policy_signal: str = ""
    indicators: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

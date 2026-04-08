"""Candidate filter gates for the deterministic funnel.

Each gate implements a simple ``filter(symbols, context) -> (passed, excluded)``
interface.  Gates are composable and run sequentially.
"""

from __future__ import annotations

from typing import Any

from quant_investor.agent_protocol import GlobalContext


class DataQualityGate:
    """Exclude symbols that are quarantined due to data quality issues."""

    def filter(
        self,
        symbols: list[str],
        context: GlobalContext,
    ) -> tuple[list[str], dict[str, str]]:
        quarantine = set(context.data_quality_quarantine)
        passed: list[str] = []
        excluded: dict[str, str] = {}
        for symbol in symbols:
            if symbol in quarantine:
                excluded[symbol] = "data_quality_quarantine"
            else:
                passed.append(symbol)
        return passed, excluded


class TradabilityGate:
    """Exclude symbols that fail tradability checks.

    Uses ``context.liquidity_filter`` with a ``suspended`` key listing
    currently suspended symbols.
    """

    def filter(
        self,
        symbols: list[str],
        context: GlobalContext,
    ) -> tuple[list[str], dict[str, str]]:
        suspended = set(context.liquidity_filter.get("suspended", []))
        passed: list[str] = []
        excluded: dict[str, str] = {}
        for symbol in symbols:
            if symbol in suspended:
                excluded[symbol] = "suspended"
            else:
                passed.append(symbol)
        return passed, excluded


class LiquidityGate:
    """Exclude symbols below a liquidity percentile threshold.

    Uses ``context.liquidity_filter`` with an ``illiquid`` key or a
    ``liquidity_scores`` dict mapping symbol -> percentile rank.
    """

    def __init__(self, percentile_min: float = 0.10) -> None:
        self.percentile_min = percentile_min

    def filter(
        self,
        symbols: list[str],
        context: GlobalContext,
    ) -> tuple[list[str], dict[str, str]]:
        illiquid = set(context.liquidity_filter.get("illiquid", []))
        scores = context.liquidity_filter.get("liquidity_scores", {})
        passed: list[str] = []
        excluded: dict[str, str] = {}
        for symbol in symbols:
            if symbol in illiquid:
                excluded[symbol] = "illiquid"
            elif scores and scores.get(symbol, 1.0) < self.percentile_min:
                excluded[symbol] = f"liquidity_below_{self.percentile_min:.0%}"
            else:
                passed.append(symbol)
        return passed, excluded

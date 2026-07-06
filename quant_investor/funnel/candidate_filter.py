"""Candidate filter gates for the deterministic funnel.

Each gate implements a simple ``filter(symbols, context) -> (passed, excluded)``
interface.  Gates are composable and run sequentially.
"""

from __future__ import annotations

from typing import Any

from quant_investor.agent_protocol import GlobalContext


def _pit_payload_from_context(context: GlobalContext) -> dict[str, Any]:
    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    pit_payload = metadata.get("pit_universe", {})
    if isinstance(pit_payload, dict) and pit_payload:
        return dict(pit_payload)
    data_snapshot = metadata.get("data_snapshot", {})
    if isinstance(data_snapshot, dict):
        nested_payload = data_snapshot.get("pit_universe", {})
        if isinstance(nested_payload, dict):
            return dict(nested_payload)
    return {}


class DataQualityGate:
    """Exclude symbols that are quarantined due to data quality issues."""

    def filter(
        self,
        symbols: list[str],
        context: GlobalContext,
    ) -> tuple[list[str], dict[str, str]]:
        quarantine = set(context.data_quality_quarantine)
        pit_payload = _pit_payload_from_context(context)
        pit_reasons = dict(pit_payload.get("reasons", {}) or {})
        pit_required = bool(pit_payload.get("required", False))
        pit_always_block_reasons = {"conflicting_status_rows", "missing_list_date", "missing_delist_date"}
        passed: list[str] = []
        excluded: dict[str, str] = {}
        for symbol in symbols:
            if symbol in quarantine:
                excluded[symbol] = "data_quality_quarantine"
            elif pit_reasons.get(symbol) == "missing_pit_record" and pit_required:
                excluded[symbol] = str(pit_reasons[symbol])
            elif pit_reasons.get(symbol) in pit_always_block_reasons:
                excluded[symbol] = str(pit_reasons[symbol])
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
        pit_untradable = set(context.liquidity_filter.get("pit_untradable", []))
        pit_reasons = dict(context.liquidity_filter.get("pit_reasons", {}) or {})
        pit_payload = _pit_payload_from_context(context)
        pit_untradable.update(str(symbol) for symbol in pit_payload.get("untradable_symbols", []) or [])
        pit_reasons.update(dict(pit_payload.get("reasons", {}) or {}))
        passed: list[str] = []
        excluded: dict[str, str] = {}
        for symbol in symbols:
            if symbol in suspended:
                excluded[symbol] = "suspended"
            elif symbol in pit_untradable:
                excluded[symbol] = str(pit_reasons.get(symbol) or "pit_untradable")
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

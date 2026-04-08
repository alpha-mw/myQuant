from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

import pandas as pd

from quant_investor.agent_protocol import DataQualityIssue
from quant_investor.market.cn_resolver import CNUniverseResolver
from quant_investor.market.shared_csv_reader import SharedCSVReader

CNSymbolLocalStatus = Literal[
    "up_to_date",
    "stale",
    "missing",
    "unreadable",
    "suspended_stale",
    "stale_cached",
]

CN_LOCAL_STATUS_VALUES: tuple[CNSymbolLocalStatus, ...] = (
    "up_to_date",
    "stale",
    "missing",
    "unreadable",
    "suspended_stale",
    "stale_cached",
)
CN_COMPLETE_LOCAL_STATUSES = frozenset({"up_to_date", "suspended_stale"})
CN_ALLOWED_BLOCKING_OVERRIDE_STATUSES = frozenset({"stale", "missing", "stale_cached"})
CN_ALWAYS_BLOCKING_STATUSES = frozenset({"unreadable"})


def _normalize_symbols(symbols: Iterable[str] | None) -> set[str]:
    normalized: set[str] = set()
    for symbol in symbols or []:
        text = str(symbol or "").strip().upper()
        if text:
            normalized.add(text)
    return normalized


def _status_flags(
    *,
    symbol: str,
    local_status: CNSymbolLocalStatus,
    allowed_stale_symbols: set[str],
) -> tuple[bool, bool]:
    if local_status in CN_COMPLETE_LOCAL_STATUSES:
        return True, False
    if local_status in CN_ALWAYS_BLOCKING_STATUSES:
        return False, True
    if local_status in CN_ALLOWED_BLOCKING_OVERRIDE_STATUSES:
        return False, symbol.upper() not in allowed_stale_symbols
    raise ValueError(f"Unsupported CN local status: {local_status}")


def _extract_latest_local_date(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    if "trade_date" in frame.columns:
        series = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    elif "date" in frame.columns:
        series = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y%m%d")
    else:
        return ""
    values = [value for value in series.dropna().astype(str).tolist() if value.strip()]
    return max(values) if values else ""


@dataclass(frozen=True)
class CNSymbolLocalStatusResult:
    symbol: str
    resolved_path: str = ""
    latest_local_date: str = ""
    local_status: CNSymbolLocalStatus = "missing"
    is_complete: bool = False
    is_blocking: bool = True
    strict_trade_date: str = ""
    stable_trade_date: str = ""
    effective_target_trade_date: str = ""
    freshness_mode: str = "stable"
    issues: list[DataQualityIssue] = field(default_factory=list)
    frame: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def with_local_status(
        self,
        local_status: CNSymbolLocalStatus,
        *,
        allowed_stale_symbols: Iterable[str] | None = None,
    ) -> "CNSymbolLocalStatusResult":
        normalized_allowed = _normalize_symbols(allowed_stale_symbols)
        is_complete, is_blocking = _status_flags(
            symbol=self.symbol,
            local_status=local_status,
            allowed_stale_symbols=normalized_allowed,
        )
        return replace(
            self,
            local_status=local_status,
            is_complete=is_complete,
            is_blocking=is_blocking,
        )


def evaluate_symbol_local_status(
    symbol: str,
    *,
    category: str,
    resolver: CNUniverseResolver,
    csv_reader: SharedCSVReader,
    latest_trade_date: str,
    allowed_stale_symbols: Iterable[str] | None,
    suspended_symbols: Iterable[str] | None,
    freshness_mode: str = "stable",
    strict_trade_date: str = "",
    stable_trade_date: str = "",
    fast_date_peek: bool = False,
) -> CNSymbolLocalStatusResult:
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol 不能为空")

    normalized_category = str(category or "").strip().lower() or "full_a"
    allowed = _normalize_symbols(allowed_stale_symbols)
    suspended = _normalize_symbols(suspended_symbols)
    universe_key = "full_a" if normalized_category == "full_a" else normalized_category

    # `resolver` stays in the signature so every CN caller is wired to the same resolver chain.
    _ = resolver

    resolved_path = csv_reader.resolve_symbol_path(
        normalized_symbol,
        universe_key=universe_key,
        category=normalized_category,
    )
    if resolved_path is None:
        base = CNSymbolLocalStatusResult(
            symbol=normalized_symbol,
            strict_trade_date=strict_trade_date,
            stable_trade_date=stable_trade_date,
            effective_target_trade_date=latest_trade_date,
            freshness_mode=freshness_mode,
        )
        return base.with_local_status("missing", allowed_stale_symbols=allowed)

    # ── fast path: only peek the latest date from the tail of the CSV ──
    if fast_date_peek:
        latest_local_date = csv_reader.peek_symbol_latest_date(
            normalized_symbol,
            universe_key=universe_key,
            category=normalized_category,
        )
        base = CNSymbolLocalStatusResult(
            symbol=normalized_symbol,
            resolved_path=str(resolved_path),
            latest_local_date=latest_local_date,
            strict_trade_date=strict_trade_date,
            stable_trade_date=stable_trade_date,
            effective_target_trade_date=latest_trade_date,
            freshness_mode=freshness_mode,
        )
        if not latest_local_date:
            return base.with_local_status("unreadable", allowed_stale_symbols=allowed)
        if latest_local_date == latest_trade_date:
            return base.with_local_status("up_to_date", allowed_stale_symbols=allowed)
        if normalized_symbol in suspended:
            return base.with_local_status("suspended_stale", allowed_stale_symbols=allowed)
        return base.with_local_status("stale", allowed_stale_symbols=allowed)

    # ── full path: load complete DataFrame (needed for analysis, not just freshness) ──
    read_result = csv_reader.read_symbol_frame(
        normalized_symbol,
        universe_key=universe_key,
        category=normalized_category,
    )
    latest_local_date = _extract_latest_local_date(read_result.frame)
    base = CNSymbolLocalStatusResult(
        symbol=normalized_symbol,
        resolved_path=str(resolved_path),
        latest_local_date=latest_local_date,
        strict_trade_date=strict_trade_date,
        stable_trade_date=stable_trade_date,
        effective_target_trade_date=latest_trade_date,
        freshness_mode=freshness_mode,
        issues=list(read_result.issues),
        frame=read_result.frame.copy(),
    )

    if read_result.frame.empty or not latest_local_date:
        return base.with_local_status("unreadable", allowed_stale_symbols=allowed)
    if latest_local_date == latest_trade_date:
        return base.with_local_status("up_to_date", allowed_stale_symbols=allowed)
    if normalized_symbol in suspended:
        return base.with_local_status("suspended_stale", allowed_stale_symbols=allowed)
    return base.with_local_status("stale", allowed_stale_symbols=allowed)


def evaluate_batch_completeness(
    symbols: Iterable[str],
    *,
    category: str,
    resolver: CNUniverseResolver,
    csv_reader: SharedCSVReader,
    latest_trade_date: str,
    allowed_stale_symbols: Iterable[str] | None = None,
    suspended_symbols: Iterable[str] | None = None,
    freshness_mode: str = "stable",
    strict_trade_date: str = "",
    stable_trade_date: str = "",
) -> dict[str, CNSymbolLocalStatusResult]:
    """Evaluate local status for a batch of symbols.

    Returns a mapping of symbol -> CNSymbolLocalStatusResult.
    """
    results: dict[str, CNSymbolLocalStatusResult] = {}
    for symbol in symbols:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            continue
        results[normalized] = evaluate_symbol_local_status(
            normalized,
            category=category,
            resolver=resolver,
            csv_reader=csv_reader,
            latest_trade_date=latest_trade_date,
            allowed_stale_symbols=allowed_stale_symbols,
            suspended_symbols=suspended_symbols,
            freshness_mode=freshness_mode,
            strict_trade_date=strict_trade_date,
            stable_trade_date=stable_trade_date,
        )
    return results

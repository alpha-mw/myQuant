"""Pure complete-date coverage for the existing official-close writer."""

from collections import Counter
from typing import Any, Iterable, Sequence

BENCHMARK_SYMBOLS = ("000300.SH", "000688.SH", "399006.SZ")


def analyze_close_coverage(
    *,
    required_dates: Sequence[str],
    event_dates: Iterable[str],
    benchmark_keys: Iterable[tuple[str, str]],
    held_close_keys: Iterable[tuple[str, str]],
    symbols: Sequence[str],
    registered_event_dates: Iterable[str] = (),
) -> dict[str, Any]:
    """Duplicate or absent exact keys are incomplete; no inferred empty events."""
    events = Counter(event_dates)
    registered = Counter(registered_event_dates)
    benchmarks = Counter(benchmark_keys)
    holdings = Counter(held_close_keys)
    rows = []
    blockers = []
    for day in required_dates:
        missing_indices = [s for s in BENCHMARK_SYMBOLS if benchmarks[(day, s)] != 1]
        missing_stocks = [s for s in symbols if holdings[(day, s)] != 1]
        event_ok = events[day] == 1 and registered[day] == 0
        registered_ok = registered[day] == 1 and events[day] == 0
        evidence_ok = event_ok or registered_ok
        if events[day] and registered[day]:
            blockers.append("EVENT_EVIDENCE_CONFLICT:" + day)
        elif not evidence_ok:
            blockers.append("EVENT_STATE_CLOSURE_MISSING:" + day)
        if missing_indices:
            blockers.append("BENCHMARK_EXACT_CLOSE_MISSING:" + day)
        if missing_stocks:
            blockers.append("HELD_SECURITY_EXACT_CLOSE_MISSING:" + day)
        rows.append(
            {
                "trade_date": day,
                "event_closed": event_ok,
                "missing_benchmark_symbols": missing_indices,
                "missing_held_symbols": missing_stocks,
                "complete": evidence_ok and not missing_indices and not missing_stocks,
                **({"registered_event_proven": registered_ok} if registered else {}),
            }
        )
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "dates": rows,
        "required_dates": list(required_dates),
        "blockers": blockers,
    }

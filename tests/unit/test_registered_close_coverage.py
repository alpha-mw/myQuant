"""Registered owner evidence is distinct from a formal empty Event closure."""

import pytest

from quant_investor.strategy_records.close_coverage import analyze_close_coverage, BENCHMARK_SYMBOLS


@pytest.mark.parametrize(
    "empty,registered,ready",
    [(True, False, True), (False, True, True), (False, False, False), (True, True, False)],
)
def test_exactly_one_event_evidence_kind_is_required(empty, registered, ready):
    day = "2026-08-25"
    result = analyze_close_coverage(
        required_dates=[day],
        event_dates=[day] if empty else [],
        registered_event_dates=[day] if registered else [],
        benchmark_keys=[(day, s) for s in BENCHMARK_SYMBOLS],
        held_close_keys=[(day, "002463.SZ")],
        symbols=["002463.SZ"],
    )
    assert (result["status"] == "READY") is ready
    row = result["dates"][0]
    if registered:
        assert row["event_closed"] is False
        assert row["registered_event_proven"] is (not empty)
    else:
        assert "registered_event_proven" not in row


def test_duplicate_registered_evidence_cannot_pass_coverage():
    day = "2026-08-25"
    result = analyze_close_coverage(
        required_dates=[day],
        event_dates=[],
        registered_event_dates=[day, day],
        benchmark_keys=[(day, s) for s in BENCHMARK_SYMBOLS],
        held_close_keys=[(day, "002463.SZ")],
        symbols=["002463.SZ"],
    )
    assert result["status"] == "BLOCKED"
    assert result["dates"][0]["registered_event_proven"] is False

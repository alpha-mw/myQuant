"""Sell-signal rules against the sealed policies and the 2026-09-30 risk monitor."""

from __future__ import annotations

from quant_investor.paper.rules import (
    EXIT_100,
    HOLD,
    REDUCE_25,
    REDUCE_50,
    REVIEW_ONLY,
    evaluate_portfolio,
    evaluate_position,
)


def _position(**overrides):
    row = {
        "symbol": "002008.SZ",
        "shares": 1000,
        "settled_shares": 1000,
        "avg_cost": "65.300000",
        "close": "80.03",
        "hard_stop": None,
        "hard_stop_source": "",
        "giveback_ratio": None,
        "peak_price": None,
        "review_price": None,
        "reduce_price": None,
        "deterioration_evidence": [],
    }
    row.update(overrides)
    return row


def test_owner_stop_breach_clears_the_position() -> None:
    signal = evaluate_position(
        _position(
            symbol="601899.SH",
            shares=5000,
            settled_shares=5000,
            avg_cost="33.453680",
            close="29.79",
            hard_stop="31.60",
            hard_stop_source="owner-stop-policy-20260828-v1:601899.SH",
        )
    )
    assert signal["action"] == EXIT_100
    assert signal["policy_row"] == "owner_stop_strict_close_breach"
    assert signal["needs_review"] is False


def test_owner_stop_equality_is_a_breach() -> None:
    signal = evaluate_position(_position(close="31.60", hard_stop="31.60", hard_stop_source="test"))
    assert signal["action"] == EXIT_100


def test_giveback_at_or_above_35_percent_reduces_half() -> None:
    for giveback in ("0.35", "0.5705539358600583090379008746", "1.000000"):
        signal = evaluate_position(_position(giveback_ratio=giveback, peak_price="99.60"))
        assert signal["action"] == REDUCE_50, giveback
        assert signal["policy_row"] == "profit_giveback_at_least_35_percent"


def test_giveback_band_without_deterioration_is_review_only() -> None:
    signal = evaluate_position(
        _position(
            symbol="002384.SZ",
            avg_cost="110.320000",
            close="167.40",
            peak_price="198.12",
            giveback_ratio="0.3498861047835990888382687927",
        )
    )
    assert signal["action"] == REVIEW_ONLY
    assert signal["needs_review"] is True
    assert signal["policy_row"] == "profit_giveback_20_to_35_percent_without_deterioration"


def test_giveback_band_with_deterioration_reduces_a_quarter() -> None:
    signal = evaluate_position(
        _position(
            giveback_ratio="0.250000",
            peak_price="99.60",
            deterioration_evidence=["FALLING_SCORE"],
        )
    )
    assert signal["action"] == REDUCE_25
    assert signal["reasons"] == ["GIVEBACK:0.250000", "FALLING_SCORE"]


def test_materiality_floor_keeps_a_thin_peak_in_review() -> None:
    """沪电股份: peak profit +1.2% of cost; a 100% giveback there is noise."""

    signal = evaluate_position(
        _position(
            symbol="002463.SZ",
            shares=2000,
            settled_shares=2000,
            avg_cost="126.754605",
            close="115.93",
            peak_price="128.25",
            giveback_ratio="1",
        )
    )
    assert signal["action"] == REVIEW_ONLY
    assert signal["policy_row"] == "trailing_peak_profit_below_materiality_floor"
    assert "TRAILING_PEAK_PROFIT_BELOW_MATERIALITY_FLOOR" in signal["reasons"]


def test_materiality_floor_boundary_acts_from_ten_percent() -> None:
    below = evaluate_position(
        _position(avg_cost="100.00", close="90.00", peak_price="109.99", giveback_ratio="0.60")
    )
    at = evaluate_position(
        _position(avg_cost="100.00", close="90.00", peak_price="110.00", giveback_ratio="0.60")
    )
    assert below["action"] == REVIEW_ONLY
    assert at["action"] == REDUCE_50


def test_missing_peak_evidence_never_acts() -> None:
    signal = evaluate_position(_position(giveback_ratio="0.900000", peak_price=None))
    assert signal["action"] == REVIEW_ONLY


def test_owner_stop_outranks_the_trailing_ladder() -> None:
    signal = evaluate_position(
        _position(
            close="20.00",
            hard_stop="31.60",
            hard_stop_source="owner-stop-policy-20260828-v1:601899.SH",
            giveback_ratio="0.900000",
            peak_price="99.60",
        )
    )
    assert signal["action"] == EXIT_100
    assert signal["policy_row"] == "owner_stop_strict_close_breach"


def test_missing_risk_inputs_hold_and_flag_for_review() -> None:
    signal = evaluate_position(_position(symbol="002916.SZ"))
    assert signal["action"] == HOLD
    assert signal["needs_review"] is True
    assert signal["reasons"] == ["NO_HARD_STOP_AND_NO_TRAILING_ANCHOR"]


def test_hard_stop_without_breach_and_without_anchor_holds_quietly() -> None:
    signal = evaluate_position(
        _position(symbol="002463.SZ", close="115.93", hard_stop="100.09", hard_stop_source="ledger")
    )
    assert signal["action"] == HOLD
    assert signal["needs_review"] is False


def test_portfolio_matches_the_20260930_risk_monitor() -> None:
    """Rows mirror scripts/export_cn_research_risk.py for as_of 2026-09-30."""

    rows = [
        _position(
            symbol="002008.SZ",
            close="80.03",
            avg_cost="65.300000",
            peak_price="99.6",
            giveback_ratio="0.5705539358600583090379008746",
        ),
        _position(
            symbol="002384.SZ",
            shares=400,
            settled_shares=400,
            close="167.4",
            avg_cost="110.320000",
            peak_price="198.12",
            giveback_ratio="0.3498861047835990888382687927",
        ),
        _position(
            symbol="002463.SZ",
            shares=2000,
            settled_shares=2000,
            close="115.93",
            avg_cost="126.754605",
            peak_price="128.25",
            giveback_ratio="1",
        ),
        _position(
            symbol="002916.SZ",
            shares=200,
            settled_shares=200,
            close="373.01",
            avg_cost="362.239800",
            peak_price="414.77",
            giveback_ratio="0.7949712736673380265066571230",
        ),
        _position(
            symbol="601899.SH",
            shares=5000,
            settled_shares=5000,
            close="29.79",
            avg_cost="33.453680",
            hard_stop="31.60",
            hard_stop_source="owner-stop-policy-20260828-v1:601899.SH",
        ),
        _position(
            symbol="605358.SH",
            shares=500,
            settled_shares=500,
            close="45.36",
            avg_cost="65.260000",
            peak_price="48.34",
            hard_stop="35.32",
            hard_stop_source="ledger",
        ),
        _position(
            symbol="688183.SH",
            shares=500,
            settled_shares=500,
            close="108.00",
            avg_cost="109.742080",
            peak_price="132.93",
            giveback_ratio="1",
        ),
    ]
    portfolio = evaluate_portfolio(rows)
    assert [(s["symbol"], s["action"]) for s in portfolio["signals"]] == [
        ("002008.SZ", REDUCE_50),
        ("002384.SZ", REVIEW_ONLY),
        ("002463.SZ", REVIEW_ONLY),
        ("002916.SZ", REDUCE_50),
        ("601899.SH", EXIT_100),
        ("605358.SH", HOLD),
        ("688183.SH", REDUCE_50),
    ]
    assert portfolio["actionable_count"] == 4
    assert portfolio["review_count"] == 2
    assert portfolio["hold_count"] == 1


def test_lot_rounding_matches_the_writer() -> None:
    from quant_investor.paper.execution import calculate_sell_shares

    assert calculate_sell_shares(action=EXIT_100, settled_shares=5000) == 5000
    assert calculate_sell_shares(action=REDUCE_50, settled_shares=500) == 200
    assert calculate_sell_shares(action=REDUCE_25, settled_shares=1000) == 200

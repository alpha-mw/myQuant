"""Sell-signal rules against the sealed policies and the 2026-09-30 ledger rows."""

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
    signal = evaluate_position(
        _position(close="31.60", hard_stop="31.60", hard_stop_source="test")
    )
    assert signal["action"] == EXIT_100


def test_giveback_at_or_above_35_percent_reduces_half() -> None:
    for giveback in ("0.35", "0.664818", "1.000000"):
        signal = evaluate_position(
            _position(hard_stop="79.99", hard_stop_source="ledger", giveback_ratio=giveback)
        )
        assert signal["action"] == REDUCE_50, giveback
        assert signal["policy_row"] == "profit_giveback_at_least_35_percent"


def test_giveback_band_without_deterioration_is_review_only() -> None:
    signal = evaluate_position(
        _position(hard_stop="79.99", hard_stop_source="ledger", giveback_ratio="0.250000")
    )
    assert signal["action"] == REVIEW_ONLY
    assert signal["needs_review"] is True
    assert signal["policy_row"] == "profit_giveback_20_to_35_percent_without_deterioration"


def test_giveback_band_with_deterioration_reduces_a_quarter() -> None:
    signal = evaluate_position(
        _position(
            hard_stop="79.99",
            hard_stop_source="ledger",
            giveback_ratio="0.250000",
            deterioration_evidence=["FALLING_SCORE"],
        )
    )
    assert signal["action"] == REDUCE_25
    assert signal["reasons"] == ["GIVEBACK:0.250000", "FALLING_SCORE"]


def test_owner_stop_outranks_the_trailing_ladder() -> None:
    signal = evaluate_position(
        _position(
            close="20.00",
            hard_stop="31.60",
            hard_stop_source="owner-stop-policy-20260828-v1:601899.SH",
            giveback_ratio="0.900000",
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


def test_portfolio_matches_the_20260930_ledger_rows() -> None:
    rows = [
        _position(
            symbol="002008.SZ",
            close="80.03",
            avg_cost="65.300000",
            hard_stop="79.99",
            hard_stop_source="ledger",
            giveback_ratio="0.664818",
        ),
        _position(
            symbol="002384.SZ",
            shares=400,
            settled_shares=400,
            close="167.40",
            avg_cost="110.320000",
            hard_stop="160.37",
            hard_stop_source="ledger",
            giveback_ratio="0.492169",
        ),
        _position(
            symbol="002463.SZ",
            shares=2000,
            settled_shares=2000,
            close="115.93",
            avg_cost="126.754605",
            hard_stop="100.09",
            hard_stop_source="ledger",
        ),
        _position(symbol="002916.SZ", shares=200, settled_shares=200, close="373.01"),
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
            hard_stop="35.32",
            hard_stop_source="ledger",
            giveback_ratio="1.000000",
        ),
        _position(symbol="688183.SH", shares=500, settled_shares=500, close="108.00"),
    ]
    portfolio = evaluate_portfolio(rows)
    assert [signal["symbol"] for signal in portfolio["signals"]] == [row["symbol"] for row in rows]
    assert [
        (signal["symbol"], signal["action"]) for signal in portfolio["signals"]
    ] == [
        ("002008.SZ", REDUCE_50),
        ("002384.SZ", REDUCE_50),
        ("002463.SZ", HOLD),
        ("002916.SZ", HOLD),
        ("601899.SH", EXIT_100),
        ("605358.SH", REDUCE_50),
        ("688183.SH", HOLD),
    ]
    assert portfolio["actionable_count"] == 4
    assert portfolio["review_count"] == 2
    assert portfolio["hold_count"] == 3


def test_lot_rounding_matches_the_writer() -> None:
    from quant_investor.paper.execution import calculate_sell_shares

    assert calculate_sell_shares(action=EXIT_100, settled_shares=5000) == 5000
    assert calculate_sell_shares(action=REDUCE_50, settled_shares=500) == 200
    assert calculate_sell_shares(action=REDUCE_25, settled_shares=1000) == 200

from __future__ import annotations

from quant_investor.v17.optimizer import (
    FeasiblePortfolio,
    ProposedTrade,
    optimize_lexicographic,
)


def _candidate(
    candidate_id: str,
    *,
    symbol: str,
    expected: float,
    cost: float,
    turnover: float,
    action: str = "BUY",
) -> FeasiblePortfolio:
    return FeasiblePortfolio(
        candidate_id=candidate_id,
        target_weights={symbol: turnover},
        trades=(ProposedTrade(symbol=symbol, action=action, notional_fraction=turnover),),
        expected_adjusted_q25=expected,
        transaction_cost=cost,
        turnover=turnover,
    )


def test_selects_net_q25_then_turnover_then_security_code() -> None:
    candidates = (
        _candidate("b", symbol="000002.SZ", expected=0.11, cost=0.01, turnover=0.20),
        _candidate("a", symbol="000001.SZ", expected=0.105, cost=0.005, turnover=0.10),
        _candidate("c", symbol="000003.SZ", expected=0.10, cost=0.0, turnover=0.10),
    )
    permissions = {symbol: {"BUY"} for symbol in ("000001.SZ", "000002.SZ", "000003.SZ")}
    result = optimize_lexicographic(
        candidates,
        permission_mask=permissions,
        current_weights={},
        effective_gross=0.80,
    )
    # All three net to 0.10. Lower turnover wins; security code breaks that tie.
    assert result.selected is not None
    assert result.selected.candidate_id == "a"


def test_permission_mask_filters_but_never_creates_trade() -> None:
    unauthorized = FeasiblePortfolio(
        candidate_id="unauthorized",
        target_weights={},
        trades=(ProposedTrade(symbol="000001.SZ", action="SELL", notional_fraction=0.20),),
        expected_adjusted_q25=1.0,
        transaction_cost=0.0,
        turnover=0.20,
    )
    result = optimize_lexicographic(
        (unauthorized,),
        permission_mask={"000001.SZ": {"BUY"}},
        current_weights={"000001.SZ": 0.20},
        effective_gross=0.8,
    )
    assert result.status == "SHADOW_PORTFOLIO_INFEASIBLE"
    assert result.selected is None
    assert result.rejected["unauthorized"] == ("action_not_permitted:000001.SZ:SELL",)


def test_recomputes_turnover_and_rejects_normalized_symbol_collisions() -> None:
    candidate = FeasiblePortfolio(
        candidate_id="bad-turnover",
        target_weights={"000001.SZ": 0.20},
        trades=(ProposedTrade(symbol="000001.SZ", action="BUY", notional_fraction=0.20),),
        expected_adjusted_q25=0.10,
        transaction_cost=0.0,
        turnover=0.10,
    )
    result = optimize_lexicographic(
        (candidate,),
        permission_mask={"000001.SZ": {"BUY"}},
        current_weights={},
        effective_gross=0.8,
    )
    assert "reported_turnover_mismatch" in result.rejected["bad-turnover"]

    try:
        optimize_lexicographic(
            (),
            permission_mask={"000001.SZ": {"BUY"}, " 000001.SZ": {"SELL"}},
            current_weights={},
            effective_gross=0.8,
        )
    except ValueError as exc:
        assert "duplicate permission symbol" in str(exc)
    else:
        raise AssertionError("normalized permission collision was accepted")


def test_sub_epsilon_target_change_still_requires_permission_and_trade() -> None:
    candidate = FeasiblePortfolio(
        candidate_id="tiny-unauthorized",
        target_weights={"000001.SZ": 1e-13},
        trades=(),
        expected_adjusted_q25=0.10,
        transaction_cost=0.0,
        turnover=1e-13,
    )
    result = optimize_lexicographic(
        (candidate,),
        permission_mask={"000001.SZ": {"LOCK"}},
        current_weights={},
        effective_gross=0.8,
    )
    assert result.status == "SHADOW_PORTFOLIO_INFEASIBLE"
    assert "target_change_without_trade:000001.SZ:BUY" in result.rejected["tiny-unauthorized"]


def test_nonzero_target_change_requires_exact_trade_and_permission() -> None:
    candidate = FeasiblePortfolio(
        candidate_id="tiny-unauthorized",
        target_weights={"000001.SZ": 1e-13},
        trades=(),
        expected_adjusted_q25=0.10,
        transaction_cost=0.0,
        turnover=1e-13,
    )
    result = optimize_lexicographic(
        (candidate,),
        permission_mask={},
        current_weights={},
        effective_gross=0.8,
    )
    assert result.status == "SHADOW_PORTFOLIO_INFEASIBLE"
    assert result.rejected["tiny-unauthorized"] == ("target_change_without_trade:000001.SZ:BUY",)


def test_selected_candidate_returns_canonical_symbols_and_actions() -> None:
    candidate = FeasiblePortfolio(
        candidate_id=" canonical ",
        target_weights={" 000001.SZ ": 0.2},
        trades=(ProposedTrade(symbol=" 000001.SZ ", action=" buy ", notional_fraction=0.2),),
        expected_adjusted_q25=0.10,
        transaction_cost=0.0,
        turnover=0.2,
    )
    result = optimize_lexicographic(
        (candidate,),
        permission_mask={"000001.SZ": {"BUY"}},
        current_weights={},
        effective_gross=0.8,
    )
    assert result.selected is not None
    assert result.selected.candidate_id == "canonical"
    assert result.selected.target_weights == {"000001.SZ": 0.2}
    assert result.selected.trades == (
        ProposedTrade(symbol="000001.SZ", action="BUY", notional_fraction=0.2),
    )

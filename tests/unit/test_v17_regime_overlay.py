from __future__ import annotations

import pytest

from quant_investor.v17.contracts import V17ContractError
from quant_investor.v17.regime_overlay import (
    build_available_overlay_input,
    build_disabled_overlay_input,
    build_unavailable_overlay_input,
    compute_regime_portfolio_overlay,
    validate_regime_portfolio_overlay,
)


def test_overlay_uses_min_cap_max_floor_and_effective_gross() -> None:
    payload = compute_regime_portfolio_overlay(
        base=build_available_overlay_input(name="base", gross_cap=0.9, cash_floor=0.1),
        macro=build_available_overlay_input(name="macro", gross_cap=0.7, cash_floor=0.2),
        markov=build_available_overlay_input(name="markov", gross_cap=0.8, cash_floor=0.4),
    )
    assert payload["gross_cap"] == 0.7
    assert payload["cash_floor"] == 0.4
    assert payload["effective_gross"] == 0.6
    assert payload["authority"] is False
    assert validate_regime_portfolio_overlay(payload) == payload


def test_disabled_component_is_explicitly_excluded_without_default() -> None:
    payload = compute_regime_portfolio_overlay(
        base=build_available_overlay_input(name="base", gross_cap=0.8, cash_floor=0.2),
        macro=build_disabled_overlay_input(name="macro"),
        markov=build_disabled_overlay_input(name="markov"),
    )
    assert payload["effective_gross"] == 0.8


def test_enabled_missing_input_yields_unavailable_no_caps() -> None:
    payload = compute_regime_portfolio_overlay(
        base=build_available_overlay_input(name="base", gross_cap=0.8, cash_floor=0.2),
        macro=build_unavailable_overlay_input(name="macro", reason="macro_missing"),
        markov=build_disabled_overlay_input(name="markov"),
    )
    assert payload["availability"] == "UNAVAILABLE"
    assert payload["missing_inputs"] == ["macro"]
    assert "gross_cap" not in payload
    assert validate_regime_portfolio_overlay(payload) == payload


def test_base_cannot_be_disabled_or_unavailable() -> None:
    with pytest.raises(V17ContractError, match="base overlay"):
        compute_regime_portfolio_overlay(
            base=build_disabled_overlay_input(name="base"),
            macro=build_disabled_overlay_input(name="macro"),
            markov=build_disabled_overlay_input(name="markov"),
        )

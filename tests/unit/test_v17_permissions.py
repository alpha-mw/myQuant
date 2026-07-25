from __future__ import annotations

import pytest

from quant_investor.v17.permissions import (
    apply_permission_restrictions,
    build_permission_restriction,
    determine_trade_permission,
    validate_trade_permission,
)


@pytest.mark.parametrize(
    ("held", "tradable", "fundamental", "red", "quant", "expected"),
    [
        (False, False, "F_ELIGIBLE", False, "BUY_NOW", (False, False, False)),
        (False, True, "F_ELIGIBLE", False, "BUY_NOW", (True, False, False)),
        (False, True, "F_INELIGIBLE", False, "BUY_NOW", (False, False, False)),
        (False, True, "F_ELIGIBLE", True, "BUY_NOW", (False, False, False)),
        (False, True, "F_ELIGIBLE", False, "WATCH", (False, False, False)),
        (False, True, "F_ELIGIBLE", False, "TRIM_TIMING", (False, False, False)),
        (True, False, "F_ELIGIBLE", False, "TRIM_TIMING", (False, False, True)),
        (True, True, "UNAVAILABLE", True, "TRIM_TIMING", (False, True, False)),
        (True, True, "F_ELIGIBLE", False, "WATCH", (False, False, True)),
        (True, True, "F_ELIGIBLE", False, "BUY_NOW", (True, False, False)),
        (True, True, "F_INELIGIBLE", False, "BUY_NOW", (False, False, True)),
        (True, True, "F_ELIGIBLE", True, "BUY_NOW", (False, False, True)),
    ],
)
def test_trade_permission_truth_table(
    held: bool,
    tradable: bool,
    fundamental: str,
    red: bool,
    quant: str,
    expected: tuple[bool, bool, bool],
) -> None:
    payload = determine_trade_permission(
        symbol="600000.SH",
        held=held,
        tradable=tradable,
        fundamental_eligibility=fundamental,
        severe_red_flag=red,
        quant_timing=quant,
    )
    assert (payload["can_buy"], payload["can_sell"], payload["position_locked"]) == expected
    assert payload["authority"] is False
    assert validate_trade_permission(payload) == payload


def test_risk_and_optimizer_can_only_shrink_base_permission() -> None:
    base = determine_trade_permission(
        symbol="600000.SH",
        held=False,
        tradable=True,
        fundamental_eligibility="F_ELIGIBLE",
        severe_red_flag=False,
        quant_timing="BUY_NOW",
    )
    final = apply_permission_restrictions(
        base,
        restrictions=[
            build_permission_restriction(
                gate="risk", allow_buy=False, allow_sell=True, reason="risk_veto"
            ),
            build_permission_restriction(
                gate="optimizer", allow_buy=True, allow_sell=True, reason="optimizer_pass"
            ),
        ],
    )
    assert final["can_buy"] is False
    assert final["can_sell"] is False
    assert validate_trade_permission(final) == final

    no_base_sell = determine_trade_permission(
        symbol="600000.SH",
        held=False,
        tradable=True,
        fundamental_eligibility="F_INELIGIBLE",
        severe_red_flag=False,
        quant_timing="WATCH",
    )
    cannot_create = apply_permission_restrictions(
        no_base_sell,
        restrictions=[
            build_permission_restriction(
                gate="risk", allow_buy=True, allow_sell=True, reason="risk_pass"
            ),
            build_permission_restriction(
                gate="optimizer", allow_buy=True, allow_sell=True, reason="optimizer_pass"
            ),
        ],
    )
    assert not cannot_create["can_buy"] and not cannot_create["can_sell"]

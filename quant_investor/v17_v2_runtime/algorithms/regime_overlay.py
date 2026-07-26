"""Pure min-cap/max-floor Macro and Markov overlay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._semantic import seal_semantic

VERSION = "myquant.v17.v2.regime-portfolio-overlay.v1"


def _ratio(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a ratio")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a ratio") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return result


def _name(value: object, *, expected: str) -> str:
    if value != expected:
        raise ValueError(f"overlay input name must be {expected}")
    return expected


def build_available_overlay_input(
    *, name: str, gross_cap: float, cash_floor: float
) -> dict[str, Any]:
    return {
        "name": _name(name, expected=name),
        "enabled": True,
        "availability": "AVAILABLE",
        "gross_cap": _ratio(gross_cap, label=f"{name}.gross_cap"),
        "cash_floor": _ratio(cash_floor, label=f"{name}.cash_floor"),
    }


def build_unavailable_overlay_input(*, name: str, reason: str) -> dict[str, Any]:
    if not isinstance(reason, str) or not reason or reason.strip() != reason:
        raise ValueError(f"{name}.reason must be a canonical non-empty string")
    return {
        "name": _name(name, expected=name),
        "enabled": True,
        "availability": "UNAVAILABLE",
        "reason": reason,
    }


def build_disabled_overlay_input(*, name: str) -> dict[str, Any]:
    return {"name": _name(name, expected=name), "enabled": False}


def _validate(
    payload: Mapping[str, Any],
    *,
    expected_name: str,
) -> tuple[bool, str | None, float | None, float | None]:
    if not isinstance(payload, Mapping) or payload.get("name") != expected_name:
        raise ValueError(f"{expected_name} overlay input invalid")
    if type(payload.get("enabled")) is not bool:
        raise ValueError(f"{expected_name}.enabled must be boolean")
    if not payload["enabled"]:
        if set(payload) != {"name", "enabled"}:
            raise ValueError(f"disabled {expected_name} overlay keys mismatch")
        return False, None, None, None
    availability = payload.get("availability")
    if availability == "UNAVAILABLE":
        if set(payload) != {"name", "enabled", "availability", "reason"}:
            raise ValueError(f"unavailable {expected_name} overlay keys mismatch")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason or reason.strip() != reason:
            raise ValueError(f"{expected_name}.reason invalid")
        return True, "UNAVAILABLE", None, None
    if availability != "AVAILABLE" or set(payload) != {
        "name",
        "enabled",
        "availability",
        "gross_cap",
        "cash_floor",
    }:
        raise ValueError(f"available {expected_name} overlay keys mismatch")
    return (
        True,
        "AVAILABLE",
        _ratio(payload["gross_cap"], label=f"{expected_name}.gross_cap"),
        _ratio(payload["cash_floor"], label=f"{expected_name}.cash_floor"),
    )


def compute_regime_portfolio_overlay(
    *,
    base: Mapping[str, Any],
    macro: Mapping[str, Any],
    markov: Mapping[str, Any],
) -> dict[str, Any]:
    base_state = _validate(base, expected_name="base")
    macro_state = _validate(macro, expected_name="macro")
    markov_state = _validate(markov, expected_name="markov")
    if not base_state[0] or base_state[1] != "AVAILABLE":
        raise ValueError("base overlay input must be enabled and AVAILABLE")
    named = (("macro", macro_state), ("markov", markov_state))
    missing = [name for name, state in named if state[0] and state[1] == "UNAVAILABLE"]
    if missing:
        return seal_semantic(
            {
                "version": VERSION,
                "availability": "UNAVAILABLE",
                "reason": "enabled_regime_input_unavailable",
                "missing_inputs": missing,
                "authority": False,
            }
        )
    enabled = [base_state] + [state for _, state in named if state[0] and state[1] == "AVAILABLE"]
    gross_cap = min(float(state[2]) for state in enabled)
    cash_floor = max(float(state[3]) for state in enabled)
    return seal_semantic(
        {
            "version": VERSION,
            "availability": "AVAILABLE",
            "gross_cap": gross_cap,
            "cash_floor": cash_floor,
            "effective_gross": min(gross_cap, 1.0 - cash_floor),
            "components": {
                "base": dict(base),
                "macro": dict(macro),
                "markov": dict(markov),
            },
            "authority": False,
        }
    )

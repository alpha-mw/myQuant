"""Macro/Markov portfolio overlay with no stock-selection authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import (
    Availability,
    V17ContractError,
    coerce_enum,
    require_bool,
    require_exact_keys,
    require_nonempty_string,
    require_ratio,
)
from .semantic import seal_semantic, validate_semantic_seal

REGIME_OVERLAY_VERSION = "myquant.v17.regime-portfolio-overlay.v1"

DISABLED_INPUT_KEYS = frozenset({"name", "enabled"})
AVAILABLE_INPUT_KEYS = frozenset({"name", "enabled", "availability", "gross_cap", "cash_floor"})
UNAVAILABLE_INPUT_KEYS = frozenset({"name", "enabled", "availability", "reason"})

OVERLAY_AVAILABLE_KEYS = frozenset(
    {
        "version",
        "availability",
        "gross_cap",
        "cash_floor",
        "effective_gross",
        "components",
        "authority",
        "semantic_sha256",
    }
)
OVERLAY_UNAVAILABLE_KEYS = frozenset(
    {
        "version",
        "availability",
        "reason",
        "missing_inputs",
        "authority",
        "semantic_sha256",
    }
)


def build_available_overlay_input(
    *, name: str, gross_cap: float, cash_floor: float
) -> dict[str, Any]:
    payload = {
        "name": name,
        "enabled": True,
        "availability": Availability.AVAILABLE.value,
        "gross_cap": gross_cap,
        "cash_floor": cash_floor,
    }
    _validate_overlay_input(payload, expected_name=name)
    return payload


def build_unavailable_overlay_input(*, name: str, reason: str) -> dict[str, Any]:
    payload = {
        "name": name,
        "enabled": True,
        "availability": Availability.UNAVAILABLE.value,
        "reason": reason,
    }
    _validate_overlay_input(payload, expected_name=name)
    return payload


def build_disabled_overlay_input(*, name: str) -> dict[str, Any]:
    payload = {"name": name, "enabled": False}
    _validate_overlay_input(payload, expected_name=name)
    return payload


def _validate_overlay_input(
    payload: Mapping[str, Any], *, expected_name: str
) -> tuple[bool, Availability | None, float | None, float | None]:
    if not isinstance(payload, Mapping):
        raise V17ContractError(f"{expected_name} overlay input must be an object")
    if payload.get("name") != expected_name:
        raise V17ContractError(f"overlay input name must be {expected_name}")
    enabled = require_bool(payload.get("enabled"), label=f"{expected_name}.enabled")
    if not enabled:
        require_exact_keys(
            payload, DISABLED_INPUT_KEYS, label=f"disabled {expected_name} overlay input"
        )
        return False, None, None, None

    availability = coerce_enum(
        payload.get("availability"),
        Availability,
        label=f"{expected_name}.availability",
    )
    if availability is Availability.UNAVAILABLE:
        require_exact_keys(
            payload,
            UNAVAILABLE_INPUT_KEYS,
            label=f"UNAVAILABLE {expected_name} overlay input",
        )
        require_nonempty_string(
            payload.get("reason"), label=f"{expected_name}.reason", max_chars=512
        )
        return True, Availability.UNAVAILABLE, None, None

    require_exact_keys(
        payload,
        AVAILABLE_INPUT_KEYS,
        label=f"AVAILABLE {expected_name} overlay input",
    )
    gross_cap = require_ratio(payload.get("gross_cap"), label=f"{expected_name}.gross_cap")
    cash_floor = require_ratio(payload.get("cash_floor"), label=f"{expected_name}.cash_floor")
    return True, Availability.AVAILABLE, gross_cap, cash_floor


def compute_regime_portfolio_overlay(
    *,
    base: Mapping[str, Any],
    macro: Mapping[str, Any],
    markov: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply explicit portfolio-level caps after selection/timing.

    Enabled-but-unavailable Macro or Markov state yields an UNAVAILABLE overlay
    and therefore a no-portfolio terminal upstream.  No neutral/default value is
    substituted.
    """

    base_state = _validate_overlay_input(base, expected_name="base")
    macro_state = _validate_overlay_input(macro, expected_name="macro")
    markov_state = _validate_overlay_input(markov, expected_name="markov")
    if not base_state[0] or base_state[1] is not Availability.AVAILABLE:
        raise V17ContractError("base overlay input must be enabled and AVAILABLE")

    named_states = (("macro", macro_state), ("markov", markov_state))
    missing_inputs = [
        name for name, state in named_states if state[0] and state[1] is Availability.UNAVAILABLE
    ]
    if missing_inputs:
        return seal_semantic(
            {
                "version": REGIME_OVERLAY_VERSION,
                "availability": Availability.UNAVAILABLE.value,
                "reason": "enabled_regime_input_unavailable",
                "missing_inputs": missing_inputs,
                "authority": False,
            }
        )

    enabled_states = [base_state] + [
        state for _, state in named_states if state[0] and state[1] is Availability.AVAILABLE
    ]
    gross_cap = min(float(state[2]) for state in enabled_states)
    cash_floor = max(float(state[3]) for state in enabled_states)
    effective_gross = min(gross_cap, 1.0 - cash_floor)
    return seal_semantic(
        {
            "version": REGIME_OVERLAY_VERSION,
            "availability": Availability.AVAILABLE.value,
            "gross_cap": gross_cap,
            "cash_floor": cash_floor,
            "effective_gross": effective_gross,
            "components": {
                "base": dict(base),
                "macro": dict(macro),
                "markov": dict(markov),
            },
            "authority": False,
        }
    )


def validate_regime_portfolio_overlay(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = validate_semantic_seal(payload)
    if sealed.get("version") != REGIME_OVERLAY_VERSION:
        raise V17ContractError("regime overlay version mismatch")
    if sealed.get("authority") is not False:
        raise V17ContractError("regime overlay authority must be false")
    availability = coerce_enum(sealed.get("availability"), Availability, label="availability")
    if availability is Availability.UNAVAILABLE:
        require_exact_keys(sealed, OVERLAY_UNAVAILABLE_KEYS, label="UNAVAILABLE regime overlay")
        require_nonempty_string(sealed.get("reason"), label="reason", max_chars=512)
        missing = sealed.get("missing_inputs")
        if not isinstance(missing, list) or not missing:
            raise V17ContractError("missing_inputs must be a nonempty array")
        if missing != [name for name in ("macro", "markov") if name in missing]:
            raise V17ContractError("missing_inputs must be unique and canonically ordered")
        return sealed

    require_exact_keys(sealed, OVERLAY_AVAILABLE_KEYS, label="AVAILABLE regime overlay")
    components = sealed.get("components")
    if not isinstance(components, Mapping):
        raise V17ContractError("regime overlay components must be an object")
    require_exact_keys(components, frozenset({"base", "macro", "markov"}), label="components")
    recomputed = compute_regime_portfolio_overlay(
        base=components["base"], macro=components["macro"], markov=components["markov"]
    )
    if recomputed != sealed:
        raise V17ContractError("regime overlay arithmetic mismatch")
    return sealed


@dataclass(frozen=True)
class RegimePortfolioOverlay:
    _payload: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RegimePortfolioOverlay":
        return cls(validate_regime_portfolio_overlay(payload))

    @property
    def availability(self) -> Availability:
        return Availability(self._payload["availability"])

    def to_dict(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(dict(self._payload))


__all__ = [
    "REGIME_OVERLAY_VERSION",
    "RegimePortfolioOverlay",
    "build_available_overlay_input",
    "build_disabled_overlay_input",
    "build_unavailable_overlay_input",
    "compute_regime_portfolio_overlay",
    "validate_regime_portfolio_overlay",
]

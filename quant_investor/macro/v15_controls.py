"""Fail-closed v15 projection from authoritative Macro observations.

The projection is deliberately pure.  It does not fetch, persist, promote, or
change the strict catalog. Callers may use this projection only after the
observation snapshot and its source lineage have passed their own readback
gates; publication remains a separate ``macro-promote`` CAS operation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from quant_investor.macro.registry import NATIONAL_DOMAIN_WEIGHTS


V15_MACRO_CONTROL_SCHEMA_VERSION = "cn-macro-controls.v15.v1"
MIN_NATIONAL_COVERAGE = 0.80


class V15MacroControlError(ValueError):
    """Raised when a Macro observation snapshot cannot authorize v15 use."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_v15_macro_controls(
    snapshot: Mapping[str, Any],
    *,
    volatility_percentile: float,
) -> dict[str, Any]:
    """Build the v15 control projection without mutating canonical data."""

    readiness = str(snapshot.get("readiness_status") or "").strip()
    if readiness != "pass":
        raise V15MacroControlError("macro_v15_snapshot_not_ready")
    states = snapshot.get("national_states")
    if not isinstance(states, Mapping):
        raise V15MacroControlError("macro_v15_national_states_missing")
    expected_domains = set(NATIONAL_DOMAIN_WEIGHTS)
    actual_domains = {str(key) for key in states}
    missing_domains = sorted(expected_domains - actual_domains)
    if missing_domains:
        raise V15MacroControlError(
            "macro_v15_domain_missing:" + ",".join(missing_domains)
        )
    normalized: dict[str, float] = {}
    for domain in sorted(expected_domains):
        try:
            value = float(states[domain])
        except (TypeError, ValueError) as exc:
            raise V15MacroControlError(
                f"macro_v15_domain_invalid:{domain}"
            ) from exc
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise V15MacroControlError(f"macro_v15_domain_invalid:{domain}")
        normalized[domain] = value
    coverage_payload = snapshot.get("coverage")
    try:
        coverage = float(
            coverage_payload.get("national")
            if isinstance(coverage_payload, Mapping)
            else None
        )
    except (TypeError, ValueError) as exc:
        raise V15MacroControlError("macro_v15_national_coverage_invalid") from exc
    if not math.isfinite(coverage) or coverage < MIN_NATIONAL_COVERAGE:
        raise V15MacroControlError("macro_v15_national_coverage_below_80pct")
    try:
        volatility = float(volatility_percentile)
    except (TypeError, ValueError) as exc:
        raise V15MacroControlError("macro_v15_volatility_invalid") from exc
    if not math.isfinite(volatility) or not 0.0 <= volatility <= 100.0:
        raise V15MacroControlError("macro_v15_volatility_invalid")

    macro_score = sum(
        normalized[domain] * weight
        for domain, weight in NATIONAL_DOMAIN_WEIGHTS.items()
    )
    fiscal = normalized["policy_fiscal"]
    policy_signal = (
        "supportive"
        if fiscal >= 0.25
        else "restrictive" if fiscal <= -0.25 else "neutral"
    )
    payload: dict[str, Any] = {
        "schema_version": V15_MACRO_CONTROL_SCHEMA_VERSION,
        "macro_score": round(macro_score, 8),
        "macro_score_100": round(50.0 * (macro_score + 1.0), 8),
        "liquidity_score": round(normalized["credit_liquidity"], 8),
        "volatility_percentile": round(volatility, 8),
        "policy_signal": policy_signal,
        "national_coverage": round(coverage, 8),
        "national_states": normalized,
        "snapshot_hash": str(snapshot.get("snapshot_hash") or ""),
        "read_only_projection": True,
    }
    payload["semantic_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "MIN_NATIONAL_COVERAGE",
    "V15_MACRO_CONTROL_SCHEMA_VERSION",
    "V15MacroControlError",
    "build_v15_macro_controls",
]

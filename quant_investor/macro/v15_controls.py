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

from quant_investor.macro.contracts import canonical_hash
from quant_investor.macro.registry import NATIONAL_DOMAIN_WEIGHTS


V15_MACRO_CONTROL_SCHEMA_VERSION = "cn-macro-controls.v15.v1"
MACRO_SNAPSHOT_SCHEMA_VERSION = "macro-snapshot.v2"
MIN_NATIONAL_COVERAGE = 0.80
_SHA256_LENGTH = 64


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


def _required_sha256(value: Any, *, blocker: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise V15MacroControlError(blocker)
    return text


def _validated_observation_generation(
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    generation_id = str(generation.get("generation_id") or "").strip()
    if not generation_id:
        raise V15MacroControlError(
            "macro_v15_observation_generation_id_missing"
        )
    try:
        row_count = int(generation.get("row_count", -1))
    except (TypeError, ValueError) as exc:
        raise V15MacroControlError(
            "macro_v15_observation_row_count_invalid"
        ) from exc
    if row_count <= 0:
        raise V15MacroControlError(
            "macro_v15_observation_row_count_invalid"
        )
    return {
        "generation_id": generation_id,
        "pointer_sha256": _required_sha256(
            generation.get("pointer_sha256"),
            blocker="macro_v15_observation_pointer_hash_invalid",
        ),
        "parquet_sha256": _required_sha256(
            generation.get("parquet_sha256"),
            blocker="macro_v15_observation_parquet_hash_invalid",
        ),
        "manifest_sha256": _required_sha256(
            generation.get("manifest_sha256"),
            blocker="macro_v15_observation_manifest_hash_invalid",
        ),
        "content_set_hash": _required_sha256(
            generation.get("content_set_hash"),
            blocker="macro_v15_observation_content_hash_invalid",
        ),
        "row_count": row_count,
    }


def _validated_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(snapshot)
    if payload.get("schema_version") != MACRO_SNAPSHOT_SCHEMA_VERSION:
        raise V15MacroControlError("macro_v15_snapshot_schema_invalid")
    if str(payload.get("market") or "").strip().upper() != "CN":
        raise V15MacroControlError("macro_v15_snapshot_market_invalid")
    if not str(payload.get("as_of") or "").strip():
        raise V15MacroControlError("macro_v15_snapshot_as_of_missing")
    snapshot_hash = _required_sha256(
        payload.get("snapshot_hash"),
        blocker="macro_v15_snapshot_hash_invalid",
    )
    semantic = dict(payload)
    semantic.pop("snapshot_hash", None)
    if canonical_hash(semantic) != snapshot_hash:
        raise V15MacroControlError("macro_v15_snapshot_hash_mismatch")
    return payload


def build_v15_macro_controls(
    snapshot: Mapping[str, Any],
    *,
    volatility_percentile: float,
    observation_generation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the v15 control projection without mutating canonical data."""

    validated_snapshot = _validated_snapshot(snapshot)
    observation_binding = _validated_observation_generation(
        observation_generation
    )
    readiness = str(
        validated_snapshot.get("readiness_status") or ""
    ).strip()
    if readiness != "pass":
        raise V15MacroControlError("macro_v15_snapshot_not_ready")
    states = validated_snapshot.get("national_states")
    if not isinstance(states, Mapping):
        raise V15MacroControlError("macro_v15_national_states_missing")
    expected_domains = set(NATIONAL_DOMAIN_WEIGHTS)
    actual_domains = {str(key) for key in states}
    missing_domains = sorted(expected_domains - actual_domains)
    if missing_domains:
        raise V15MacroControlError(
            "macro_v15_domain_missing:" + ",".join(missing_domains)
        )
    extra_domains = sorted(actual_domains - expected_domains)
    if extra_domains:
        raise V15MacroControlError(
            "macro_v15_domain_unexpected:" + ",".join(extra_domains)
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
    coverage_payload = validated_snapshot.get("coverage")
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
        "snapshot_hash": str(validated_snapshot["snapshot_hash"]),
        "snapshot_as_of": str(validated_snapshot.get("as_of") or ""),
        "observation_generation": observation_binding,
        "read_only_projection": True,
        "production_control_projection": True,
    }
    payload["semantic_sha256"] = _canonical_sha256(payload)
    return payload


def validate_v15_macro_controls(
    controls: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
    observation_generation: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and exactly validate one persisted v15 projection."""

    if controls.get("schema_version") != V15_MACRO_CONTROL_SCHEMA_VERSION:
        raise V15MacroControlError("macro_v15_controls_schema_invalid")
    expected = build_v15_macro_controls(
        snapshot,
        volatility_percentile=controls.get("volatility_percentile"),
        observation_generation=observation_generation,
    )
    if dict(controls) != expected:
        raise V15MacroControlError("macro_v15_controls_projection_mismatch")
    return expected


__all__ = [
    "MIN_NATIONAL_COVERAGE",
    "MACRO_SNAPSHOT_SCHEMA_VERSION",
    "V15_MACRO_CONTROL_SCHEMA_VERSION",
    "V15MacroControlError",
    "build_v15_macro_controls",
    "validate_v15_macro_controls",
]

"""Closed, content-addressed receipts for the R2.2 research evaluator.

This module deliberately contains no filesystem, provider, selector, portfolio,
governance, or execution integration.  Builders return immutable-value JSON
documents; persistence remains the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any, Final

from .._core import (
    NO_AUTHORITY,
    IntelligenceContractError,
    content_ref,
    seal_content_addressed,
    sha256,
    timestamp,
    validate_content_addressed,
)
from ..memory.chain import memory_tip, validate_memory_chain

REQUEST_VERSION: Final = "myquant.v17.research-intelligence.forward-evaluation-request.v1"
POLICY_VERSION: Final = "myquant.v17.research-intelligence.forward-evaluation-policy.v1"
UNIVERSE_INVENTORY_VERSION: Final = (
    "myquant.v17.research-intelligence.evaluation-universe-inventory.v1"
)
FACTOR_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence.factor-evaluation-receipt.v1"
VARIANT_FACTOR_RECEIPT_VERSION: Final = (
    "myquant.v17.research-intelligence.variant-factor-evaluation-receipt.v1"
)
VARIANT_COMPARISON_VERSION: Final = (
    "myquant.v17.research-intelligence.variant-comparison-receipt.v1"
)
HYPOTHESIS_RECEIPT_VERSION: Final = (
    "myquant.v17.research-intelligence.hypothesis-evaluation-receipt.v1"
)
CALIBRATION_RECEIPT_VERSION: Final = (
    "myquant.v17.research-intelligence.calibration-evidence-receipt.v1"
)
REGIME_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence.regime-evaluation-receipt.v1"
MEMORY_INVENTORY_VERSION: Final = "myquant.v17.research-intelligence.memory-chain-inventory.v1"
MEMORY_PROPOSAL_VERSION: Final = "myquant.v17.research-intelligence.memory-append-proposal.v1"
MAIN_RECEIPT_VERSION: Final = "myquant.v17.research-intelligence.forward-evaluation-receipt.v1"
ENVELOPE_VERSION: Final = "myquant.v17.research-intelligence.forward-evaluation-envelope.v1"
ERROR_ENVELOPE_VERSION: Final = "myquant.v17.research-intelligence.forward-evaluation-error.v1"

MAX_SUBRECEIPT_BYTES: Final = 8 * 1024 * 1024
MAX_ENVELOPE_BYTES: Final = 16 * 1024 * 1024
MAX_METRIC_ROWS: Final = 32
MAX_LIMITATIONS: Final = 64
MAX_BLOCKERS: Final = 32
CODE_RE: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")

_COMMON_FIELDS: Final = {
    "authority",
    "broker",
    "decision_protocol",
    "execution",
    "mainline_authority",
    "operational_activation_unchanged",
    "order",
    "production",
    "research_only",
    "timestamp",
    "trade",
    "version",
}


def authority_fields(timestamp_value: str) -> dict[str, Any]:
    """Return the exact research-only authority closure for one artifact."""

    return {
        "authority": dict(NO_AUTHORITY),
        "broker": False,
        "decision_protocol": "myquant.v17.v4",
        "execution": False,
        "mainline_authority": False,
        "operational_activation_unchanged": True,
        "order": False,
        "production": False,
        "research_only": True,
        "timestamp": timestamp(timestamp_value, label="timestamp"),
        "trade": False,
    }


def _codes(values: Sequence[Any], *, label: str, maximum: int) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceContractError(f"{label} must be a sequence")
    result: list[str] = []
    for index, value in enumerate(values):
        if type(value) is not str or CODE_RE.fullmatch(value) is None:
            raise IntelligenceContractError(f"{label}[{index}] is not a canonical code")
        result.append(value)
    if len(result) > maximum or len(result) != len(set(result)):
        raise IntelligenceContractError(f"{label} cardinality is invalid")
    return sorted(result, key=lambda value: value.encode("ascii"))


def limitation_codes(values: Sequence[Any]) -> list[str]:
    return _codes(values, label="limitations", maximum=MAX_LIMITATIONS)


def blocker_codes(values: Sequence[Any]) -> list[str]:
    return _codes(values, label="blocker_codes", maximum=MAX_BLOCKERS)


def research_input_ref(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    """Validate the five-field exact reference used for sealed I0 inputs."""

    fields = {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "relative_path",
        "semantic_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise IntelligenceContractError(f"{label} must be a research input reference")
    artifact_id = value["artifact_id"]
    artifact_version = value["artifact_version"]
    relative_path = value["relative_path"]
    artifact_id = sha256(artifact_id, label=f"{label}.artifact_id")
    if type(artifact_version) is not str or not artifact_version:
        raise IntelligenceContractError(f"{label}.artifact_version is required")
    if type(relative_path) is not str or not relative_path:
        raise IntelligenceContractError(f"{label}.relative_path is required")
    expected = f"data/private/research_intelligence/evaluation_inputs/{artifact_id}.json"
    if relative_path != expected:
        raise IntelligenceContractError(f"{label} path is not content-bound")
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": sha256(value["byte_sha256"], label=f"{label}.byte_sha256"),
        "relative_path": relative_path,
        "semantic_sha256": sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256"),
    }


def _seal(
    *,
    version: str,
    identity_field: str,
    timestamp_value: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if identity_field in payload or "semantic_sha256" in payload:
        raise IntelligenceContractError("receipt payload must be unsealed")
    return seal_content_addressed(
        {
            **authority_fields(timestamp_value),
            **dict(payload),
            "version": version,
        },
        identity_field=identity_field,
    )


def validate_closed_receipt(
    document: Mapping[str, Any],
    *,
    version: str,
    identity_field: str,
    payload_fields: set[str],
) -> dict[str, Any]:
    """Validate identity, exact shape, timestamp, and no-authority fields."""

    row = validate_content_addressed(document, identity_field=identity_field)
    expected_fields = _COMMON_FIELDS | payload_fields | {identity_field, "semantic_sha256"}
    if set(row) != expected_fields or row.get("version") != version:
        raise IntelligenceContractError(f"{version} shape/version mismatch")
    if row.get("authority") != NO_AUTHORITY or any(
        row.get(field) is not False for field in ("broker", "execution", "order", "trade")
    ):
        raise IntelligenceContractError("receipt authority is open")
    if (
        row.get("research_only") is not True
        or row.get("production") is not False
        or row.get("decision_protocol") != "myquant.v17.v4"
        or row.get("mainline_authority") is not False
        or row.get("operational_activation_unchanged") is not True
    ):
        raise IntelligenceContractError("receipt protocol state is open")
    timestamp(row.get("timestamp"), label="receipt.timestamp")
    return row


def build_universe_inventory(
    *, rows: Sequence[Mapping[str, Any]], evaluated_at: str
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for value in rows:
        if type(value) is not dict or set(value) != {
            "origin_id",
            "universe_factor_id",
            "universe_observation_ref",
        }:
            raise IntelligenceContractError("universe inventory row shape is invalid")
        normalized.append(dict(value))
    normalized.sort(key=lambda row: str(row["origin_id"]).encode("ascii"))
    if not 1 <= len(normalized) <= 64 or len({row["origin_id"] for row in normalized}) != len(
        normalized
    ):
        raise IntelligenceContractError("universe inventory cardinality is invalid")
    return _seal(
        version=UNIVERSE_INVENTORY_VERSION,
        identity_field="inventory_id",
        timestamp_value=evaluated_at,
        payload={"rows": normalized},
    )


def build_subject_receipt(
    *,
    subject_type: str,
    subject_id: str,
    subject_ref: Mapping[str, Any],
    evaluation_window: Mapping[str, Any],
    universe_ref: Mapping[str, Any],
    observation_refs: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    origin_metrics: Sequence[Mapping[str, Any]],
    limitations: Sequence[str],
    evaluated_at: str,
) -> dict[str, Any]:
    if subject_type not in {"FACTOR", "VARIANT"}:
        raise IntelligenceContractError("subject_type is invalid")
    id_field = "factor_id" if subject_type == "FACTOR" else "variant_id"
    ref_field = "factor_ref" if subject_type == "FACTOR" else "variant_ref"
    version = FACTOR_RECEIPT_VERSION if subject_type == "FACTOR" else VARIANT_FACTOR_RECEIPT_VERSION
    metric_rows = sorted((dict(row) for row in metrics), key=lambda row: str(row["metric_id"]))
    if not 1 <= len(metric_rows) <= MAX_METRIC_ROWS:
        raise IntelligenceContractError("subject metric cardinality is invalid")
    origins = sorted((dict(row) for row in origin_metrics), key=lambda row: str(row["origin_id"]))
    payload = {
        "evaluation_window": dict(evaluation_window),
        id_field: subject_id,
        ref_field: dict(subject_ref),
        "limitations": limitation_codes(limitations),
        "metrics": metric_rows,
        "observation_refs": list(observation_refs),
        "origin_metrics": origins,
        "universe_ref": dict(universe_ref),
    }
    return _seal(
        version=version,
        identity_field="receipt_id",
        timestamp_value=evaluated_at,
        payload=payload,
    )


def build_variant_comparison_receipt(
    *,
    baseline_factor_receipt_ref: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    limitations: Sequence[str],
    evaluated_at: str,
) -> dict[str, Any]:
    rows = sorted((dict(row) for row in candidate_rows), key=lambda row: str(row["variant_id"]))
    if len(rows) != 2:
        raise IntelligenceContractError("variant comparison requires two candidates")
    return _seal(
        version=VARIANT_COMPARISON_VERSION,
        identity_field="receipt_id",
        timestamp_value=evaluated_at,
        payload={
            "baseline_factor_receipt_ref": dict(baseline_factor_receipt_ref),
            "candidate_rows": rows,
            "limitations": limitation_codes(limitations),
        },
    )


def build_hypothesis_receipt(*, evaluated_at: str, **payload: Any) -> dict[str, Any]:
    return _seal(
        version=HYPOTHESIS_RECEIPT_VERSION,
        identity_field="receipt_id",
        timestamp_value=evaluated_at,
        payload=payload,
    )


def build_calibration_receipt(
    *, group_rows: Sequence[Mapping[str, Any]], limitations: Sequence[str], evaluated_at: str
) -> dict[str, Any]:
    rows = sorted(
        (dict(row) for row in group_rows),
        key=lambda row: (
            str(row["source_type"]),
            str(row["direction"]),
            str(row["strength"]),
        ),
    )
    return _seal(
        version=CALIBRATION_RECEIPT_VERSION,
        identity_field="receipt_id",
        timestamp_value=evaluated_at,
        payload={"group_rows": rows, "limitations": limitation_codes(limitations)},
    )


def build_regime_receipt(
    *,
    layer_rows: Sequence[Mapping[str, Any]],
    unconditional_factor_refs: Sequence[Mapping[str, Any]],
    limitations: Sequence[str],
    evaluated_at: str,
) -> dict[str, Any]:
    return _seal(
        version=REGIME_RECEIPT_VERSION,
        identity_field="receipt_id",
        timestamp_value=evaluated_at,
        payload={
            "layer_rows": [dict(row) for row in layer_rows],
            "limitations": limitation_codes(limitations),
            "unconditional_factor_refs": list(unconditional_factor_refs),
        },
    )


def build_memory_inventory(
    *, entries: Sequence[Mapping[str, Any]], timestamp_value: str
) -> dict[str, Any]:
    chain = validate_memory_chain(entries)
    expected_timestamp = "1970-01-01T00:00:00Z" if not chain else str(chain[-1]["timestamp"])
    if timestamp(timestamp_value, label="memory_inventory.timestamp") != expected_timestamp:
        raise IntelligenceContractError("memory inventory timestamp mismatch")
    return _seal(
        version=MEMORY_INVENTORY_VERSION,
        identity_field="inventory_id",
        timestamp_value=timestamp_value,
        payload={"entries": list(chain), "tip": memory_tip(chain)},
    )


def validate_memory_inventory(document: Mapping[str, Any]) -> dict[str, Any]:
    row = validate_closed_receipt(
        document,
        version=MEMORY_INVENTORY_VERSION,
        identity_field="inventory_id",
        payload_fields={"entries", "tip"},
    )
    chain = validate_memory_chain(row["entries"], expected_tip=row["tip"])
    expected = "1970-01-01T00:00:00Z" if not chain else str(chain[-1]["timestamp"])
    if row["timestamp"] != expected:
        raise IntelligenceContractError("memory inventory time is not tip-bound")
    return row


def build_memory_proposal(
    *,
    expected_before_tip: str,
    observed_after_tip: str,
    proposed_entries: Sequence[Mapping[str, Any]],
    source_inventory_ref: Mapping[str, Any],
    evaluated_at: str,
) -> dict[str, Any]:
    return _seal(
        version=MEMORY_PROPOSAL_VERSION,
        identity_field="receipt_id",
        timestamp_value=evaluated_at,
        payload={
            "expected_before_tip": sha256(expected_before_tip, label="expected_before_tip"),
            "observed_after_tip": sha256(observed_after_tip, label="observed_after_tip"),
            "proposed_entries": [dict(row) for row in proposed_entries],
            "source_inventory_ref": dict(source_inventory_ref),
        },
    )


def build_main_receipt(*, evaluated_at: str, **payload: Any) -> dict[str, Any]:
    return _seal(
        version=MAIN_RECEIPT_VERSION,
        identity_field="evaluation_id",
        timestamp_value=evaluated_at,
        payload=payload,
    )


def build_envelope(*, evaluated_at: str, **payload: Any) -> dict[str, Any]:
    return _seal(
        version=ENVELOPE_VERSION,
        identity_field="envelope_id",
        timestamp_value=evaluated_at,
        payload=payload,
    )


def blocked_envelope(
    *, status: str, blocker_code: str, preserved_artifact_refs: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Build the non-content-addressed envelope usable before request parsing."""

    if status not in {"BLOCKED", "INTERNAL_ERROR"}:
        raise IntelligenceContractError("error status is invalid")
    return {
        "authority": dict(NO_AUTHORITY),
        "blocker_code": blocker_code,
        "broker": False,
        "decision_protocol": "myquant.v17.v4",
        "execution": False,
        "mainline_authority": False,
        "operational_activation_unchanged": True,
        "order": False,
        "preserved_artifact_refs": list(preserved_artifact_refs),
        "production": False,
        "research_only": True,
        "status": status,
        "trade": False,
        "version": ERROR_ENVELOPE_VERSION,
    }


def receipt_ref(
    document: Mapping[str, Any], *, identity_field: str = "receipt_id"
) -> dict[str, str]:
    return content_ref(document, identity_field=identity_field)


__all__ = [
    "CALIBRATION_RECEIPT_VERSION",
    "ENVELOPE_VERSION",
    "ERROR_ENVELOPE_VERSION",
    "FACTOR_RECEIPT_VERSION",
    "HYPOTHESIS_RECEIPT_VERSION",
    "MAIN_RECEIPT_VERSION",
    "MAX_ENVELOPE_BYTES",
    "MAX_SUBRECEIPT_BYTES",
    "MEMORY_INVENTORY_VERSION",
    "MEMORY_PROPOSAL_VERSION",
    "POLICY_VERSION",
    "REGIME_RECEIPT_VERSION",
    "REQUEST_VERSION",
    "UNIVERSE_INVENTORY_VERSION",
    "VARIANT_COMPARISON_VERSION",
    "VARIANT_FACTOR_RECEIPT_VERSION",
    "authority_fields",
    "blocked_envelope",
    "blocker_codes",
    "build_calibration_receipt",
    "build_envelope",
    "build_hypothesis_receipt",
    "build_main_receipt",
    "build_memory_inventory",
    "build_memory_proposal",
    "build_regime_receipt",
    "build_subject_receipt",
    "build_universe_inventory",
    "build_variant_comparison_receipt",
    "limitation_codes",
    "receipt_ref",
    "research_input_ref",
    "validate_closed_receipt",
    "validate_memory_inventory",
]

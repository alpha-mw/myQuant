"""Strict, authority-free inventories and V17 v4 forward evaluation receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
from typing import Any, Final, Literal, NoReturn

from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import validate_semantic_sha
from quant_investor.v17_v4_contract.identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)

from .factor_observation import LABEL_HORIZONS, NO_AUTHORITY

FORWARD_EVIDENCE_ORIGIN_INVENTORY_VERSION: Final = (
    "myquant.v17.v4.forward-evidence-origin-inventory.v1"
)
EXISTING_FACTOR_INVENTORY_VERSION: Final = "myquant.v17.v4.existing-factor-inventory.v1"
FORWARD_EVALUATION_RECEIPT_VERSION: Final = "myquant.v17.v4.forward-evaluation-receipt.v1"
FACTOR_EVALUATION_RECEIPT_VERSION: Final = FORWARD_EVALUATION_RECEIPT_VERSION
INDUSTRY_EVALUATION_RECEIPT_VERSION: Final = FORWARD_EVALUATION_RECEIPT_VERSION
STRATEGY_EVALUATION_RECEIPT_VERSION: Final = FORWARD_EVALUATION_RECEIPT_VERSION
RECEIPT_TYPES: Final = (
    "factor_evaluation_receipt",
    "industry_evaluation_receipt",
    "strategy_evaluation_receipt",
)
METRIC_STATES: Final = ("COMPLETE", "UNAVAILABLE")
_DISABLED: Final = {
    "promotion_eligible": False,
    "provider_authority": False,
    "provider_invoked": False,
    "shadow_only": True,
}
_REF_FIELDS: Final = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "cutoff",
    "relative_path",
    "semantic_sha256",
    "strategy_id",
}
_LINEAGE_FIELDS: Final = {
    "factor_definition_sha256",
    "factor_name",
    "factor_set_sha256",
    "horizon_sessions",
    "quant_policy_sha256",
    "source_lineage_sha256",
}


class ForwardEvaluationReceiptError(ValueError):
    """Raised when accumulated forward evidence is ambiguous or authoritative."""

    exit_code = 2


def _blocked(reason: str) -> NoReturn:
    raise ForwardEvaluationReceiptError(f"V17_V4_FORWARD_EVALUATION_BLOCKED:{reason}")


def _identity(value: Any, *, label: str) -> str:
    try:
        return require_opaque_id(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")


def _sha256(value: Any, *, label: str) -> str:
    try:
        return require_sha256(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")


def _timestamp(value: Any, *, label: str) -> str:
    try:
        return require_utc_timestamp(value, label=label)
    except IdentityContractError:
        _blocked(f"{label}_invalid")


def _instant(value: Any, *, label: str) -> datetime:
    text = _timestamp(value, label=label)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        _blocked(f"{label}_invalid")


def _session(value: Any, *, label: str) -> str:
    if type(value) is not str:
        _blocked(f"{label}_invalid")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _blocked(f"{label}_invalid")
    if parsed.isoformat() != value:
        _blocked(f"{label}_noncanonical")
    return value


def _relative_path(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        _blocked(f"{label}_invalid")
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        _blocked(f"{label}_non_ascii")
    return value


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is bool or type(value) not in {Decimal, float, int, str}:
        _blocked(f"{label}_invalid")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        _blocked(f"{label}_invalid")
    if not result.is_finite():
        _blocked(f"{label}_nonfinite")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _artifact_ref(
    value: Mapping[str, Any],
    *,
    strategy_id: str,
    cutoff: str,
    label: str,
    expected_version: str | None = None,
) -> dict[str, str]:
    if type(value) is not dict or set(value) != _REF_FIELDS:
        _blocked(f"{label}_shape")
    artifact_strategy = _identity(
        value["strategy_id"],
        label=f"{label}.strategy_id",
    )
    if artifact_strategy != strategy_id:
        _blocked(f"{label}_strategy")
    artifact_cutoff = _timestamp(value["cutoff"], label=f"{label}.cutoff")
    if artifact_cutoff > cutoff:
        _blocked(f"{label}_after_cutoff")
    version = _identity(
        value["artifact_version"],
        label=f"{label}.artifact_version",
    )
    if expected_version is not None and version != expected_version:
        _blocked(f"{label}_version")
    return {
        "artifact_id": _identity(
            value["artifact_id"],
            label=f"{label}.artifact_id",
        ),
        "artifact_version": version,
        "byte_sha256": _sha256(
            value["byte_sha256"],
            label=f"{label}.byte_sha256",
        ),
        "cutoff": artifact_cutoff,
        "relative_path": _relative_path(
            value["relative_path"],
            label=f"{label}.relative_path",
        ),
        "semantic_sha256": _sha256(
            value["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        ),
        "strategy_id": artifact_strategy,
    }


def _artifact_refs(
    values: Sequence[Mapping[str, Any]],
    *,
    strategy_id: str,
    cutoff: str,
    label: str,
    require_nonempty: bool = True,
) -> list[dict[str, str]]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or (require_nonempty and not values)
    ):
        _blocked(f"{label}_invalid")
    normalized = [
        _artifact_ref(
            value,
            strategy_id=strategy_id,
            cutoff=cutoff,
            label=f"{label}[{index}]",
        )
        for index, value in enumerate(values)
    ]
    normalized.sort(key=_ref_sort_key)
    deduplicated: list[dict[str, str]] = []
    for row in normalized:
        if not deduplicated or row != deduplicated[-1]:
            deduplicated.append(row)
    return deduplicated


def _ref_sort_key(reference: Mapping[str, str]) -> tuple[bytes, ...]:
    return (
        reference["relative_path"].encode("ascii"),
        reference["artifact_id"].encode("utf-8"),
        reference["artifact_version"].encode("ascii"),
        reference["byte_sha256"].encode("ascii"),
        reference["semantic_sha256"].encode("ascii"),
    )


def _registered(document: dict[str, Any], *, label: str) -> dict[str, Any]:
    try:
        validate_artifact(document)
    except Exception as exc:
        _blocked(f"{label}_registered_schema:{exc}")
    return document


def _lineage(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _LINEAGE_FIELDS:
        _blocked(f"{label}_shape")
    horizon = value["horizon_sessions"]
    if type(horizon) is not int or horizon not in LABEL_HORIZONS:
        _blocked(f"{label}_horizon")
    return {
        "factor_definition_sha256": _sha256(
            value["factor_definition_sha256"],
            label=f"{label}.factor_definition_sha256",
        ),
        "factor_name": _identity(
            value["factor_name"],
            label=f"{label}.factor_name",
        ),
        "factor_set_sha256": _sha256(
            value["factor_set_sha256"],
            label=f"{label}.factor_set_sha256",
        ),
        "horizon_sessions": horizon,
        "quant_policy_sha256": _sha256(
            value["quant_policy_sha256"],
            label=f"{label}.quant_policy_sha256",
        ),
        "source_lineage_sha256": _sha256(
            value["source_lineage_sha256"],
            label=f"{label}.source_lineage_sha256",
        ),
    }


def _lineage_key(
    *,
    factor_name: str,
    factor_definition_sha256: str,
    factor_set_sha256: str,
    quant_policy_sha256: str,
    horizon_sessions: int,
    source_lineage_sha256: str,
) -> dict[str, Any]:
    return _lineage(
        {
            "factor_definition_sha256": factor_definition_sha256,
            "factor_name": factor_name,
            "factor_set_sha256": factor_set_sha256,
            "horizon_sessions": horizon_sessions,
            "quant_policy_sha256": quant_policy_sha256,
            "source_lineage_sha256": source_lineage_sha256,
        },
        label="lineage_key",
    )


def _inventory_artifact_ref(
    artifact: Mapping[str, Any],
    *,
    relative_path: str,
    identity_field: str,
) -> dict[str, str]:
    try:
        normalized = validate_semantic_sha(artifact)
    except Exception:
        _blocked("inventory_semantic_sha")
    return {
        "artifact_id": _identity(
            normalized.get(identity_field),
            label=f"inventory.{identity_field}",
        ),
        "artifact_version": _identity(
            normalized.get("version"),
            label="inventory.version",
        ),
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(normalized)).hexdigest(),
        "cutoff": _timestamp(
            normalized.get("cutoff"),
            label="inventory.cutoff",
        ),
        "relative_path": _relative_path(
            relative_path,
            label="inventory.relative_path",
        ),
        "semantic_sha256": _sha256(
            normalized.get("semantic_sha256"),
            label="inventory.semantic_sha256",
        ),
        "strategy_id": _identity(
            normalized.get("strategy_id"),
            label="inventory.strategy_id",
        ),
    }


def build_forward_evidence_origin_inventory(
    *,
    inventory_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    request_ref: Mapping[str, Any],
    origins: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an origin inventory and fail closed on differing duplicate bytes."""

    strategy = _identity(strategy_id, label="strategy_id")
    session = _session(decision_session, label="decision_session")
    cutoff_text = _timestamp(cutoff, label="cutoff")
    if session > cutoff_text[:10]:
        _blocked("decision_session_after_cutoff")
    if isinstance(origins, (str, bytes)) or not isinstance(origins, Sequence) or not origins:
        _blocked("origins_invalid")
    grouped: dict[tuple[str, bytes], dict[str, Any]] = {}
    for index, value in enumerate(origins):
        label = f"origins[{index}]"
        if type(value) is not dict or set(value) != {
            "evidence_ref",
            "lineage_key",
            "origin",
        }:
            _blocked(f"{label}_shape")
        origin = _session(value["origin"], label=f"{label}.origin")
        if origin > session:
            _blocked(f"{label}_after_decision_session")
        lineage = _lineage(value["lineage_key"], label=f"{label}.lineage_key")
        evidence_ref = _artifact_ref(
            value["evidence_ref"],
            strategy_id=strategy,
            cutoff=cutoff_text,
            label=f"{label}.evidence_ref",
            expected_version="myquant.v17.v4.forward-label.v1",
        )
        key = (origin, canonical_bytes(lineage))
        bucket = grouped.setdefault(
            key,
            {
                "lineage_key": lineage,
                "origin": origin,
                "refs": [],
            },
        )
        refs = bucket["refs"]
        assert isinstance(refs, list)
        refs.append(evidence_ref)
    rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        refs = sorted(bucket["refs"], key=_ref_sort_key)
        unique_refs: list[dict[str, str]] = []
        for reference in refs:
            if not unique_refs or reference != unique_refs[-1]:
                unique_refs.append(reference)
        byte_semantic_pairs = {
            (reference["byte_sha256"], reference["semantic_sha256"]) for reference in unique_refs
        }
        if len(byte_semantic_pairs) != 1:
            _blocked("DUPLICATE_ORIGIN_CONFLICT")
        lineage = bucket["lineage_key"]
        rows.append(
            {
                "canonical_evidence_ref": unique_refs[0],
                "duplicate_origin_status": (
                    "UNIQUE" if len(unique_refs) == 1 else "DUPLICATE_IDENTICAL"
                ),
                "evidence_refs": unique_refs,
                "lineage_key": lineage,
                "lineage_key_sha256": hashlib.sha256(canonical_bytes(lineage)).hexdigest(),
                "origin": bucket["origin"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["origin"].encode("ascii"),
            row["lineage_key_sha256"].encode("ascii"),
        )
    )
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            **dict(_DISABLED),
            "cutoff": cutoff_text,
            "decision_session": session,
            "inventory_id": _identity(
                inventory_id,
                label="inventory_id",
            ),
            "origins": rows,
            "protocol_version": PROTOCOL_VERSION,
            "request_ref": _artifact_ref(
                request_ref,
                strategy_id=strategy,
                cutoff=cutoff_text,
                label="request_ref",
                expected_version="myquant.v17.v4.forward-run-request.v1",
            ),
            "strategy_id": strategy,
            "version": FORWARD_EVIDENCE_ORIGIN_INVENTORY_VERSION,
        }
    )
    return _registered(document, label="forward_evidence_origin_inventory")


def validate_forward_evidence_origin_inventory(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay schema and duplicate-origin rules against an inventory."""

    try:
        normalized = validate_semantic_sha(document)
    except Exception:
        _blocked("origin_inventory_semantic_sha")
    _registered(normalized, label="forward_evidence_origin_inventory")
    previous: tuple[bytes, bytes] | None = None
    for index, row in enumerate(normalized.get("origins", ())):
        label = f"origins[{index}]"
        lineage = _lineage(row.get("lineage_key"), label=f"{label}.lineage_key")
        if row.get("lineage_key_sha256") != hashlib.sha256(canonical_bytes(lineage)).hexdigest():
            _blocked(f"{label}_lineage_sha")
        refs = row.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            _blocked(f"{label}_evidence_refs")
        if (
            len({(reference["byte_sha256"], reference["semantic_sha256"]) for reference in refs})
            != 1
        ):
            _blocked("DUPLICATE_ORIGIN_CONFLICT")
        if refs != sorted(refs, key=_ref_sort_key):
            _blocked(f"{label}_evidence_ref_order")
        if row.get("canonical_evidence_ref") != refs[0]:
            _blocked(f"{label}_canonical_ref")
        expected_status = "UNIQUE" if len(refs) == 1 else "DUPLICATE_IDENTICAL"
        if row.get("duplicate_origin_status") != expected_status:
            _blocked(f"{label}_duplicate_status")
        key = (
            _session(row.get("origin"), label=f"{label}.origin").encode("ascii"),
            row["lineage_key_sha256"].encode("ascii"),
        )
        if previous is not None and key <= previous:
            _blocked("origins_order_or_duplicate")
        previous = key
    return normalized


def build_existing_factor_inventory(
    *,
    inventory_id: str,
    strategy_id: str,
    decision_session: str,
    cutoff: str,
    request_ref: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    factors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind unique factor names and definitions to exact exposure observations."""

    strategy = _identity(strategy_id, label="strategy_id")
    session = _session(decision_session, label="decision_session")
    cutoff_text = _timestamp(cutoff, label="cutoff")
    if session > cutoff_text[:10]:
        _blocked("decision_session_after_cutoff")
    if isinstance(factors, (str, bytes)) or not isinstance(factors, Sequence):
        _blocked("factors_invalid")
    by_name: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(factors):
        label = f"factors[{index}]"
        if type(value) is not dict or set(value) != {
            "definition_sha256",
            "exposure_observation_refs",
            "factor_name",
            "factor_ref",
            "lifecycle",
        }:
            _blocked(f"{label}_shape")
        name = _identity(value["factor_name"], label=f"{label}.factor_name")
        lifecycle = value["lifecycle"]
        if lifecycle not in {"ACTIVE", "INACTIVE", "RETIRED"}:
            _blocked(f"{label}_lifecycle")
        row = {
            "definition_sha256": _sha256(
                value["definition_sha256"],
                label=f"{label}.definition_sha256",
            ),
            "exposure_observation_refs": _artifact_refs(
                value["exposure_observation_refs"],
                strategy_id=strategy,
                cutoff=cutoff_text,
                label=f"{label}.exposure_observation_refs",
            ),
            "factor_name": name,
            "factor_ref": _artifact_ref(
                value["factor_ref"],
                strategy_id=strategy,
                cutoff=cutoff_text,
                label=f"{label}.factor_ref",
            ),
            "lifecycle": lifecycle,
        }
        previous = by_name.get(name)
        if previous is not None and previous != row:
            _blocked("existing_factor_duplicate_conflict")
        by_name[name] = row
    rows = sorted(
        by_name.values(),
        key=lambda row: row["factor_name"].encode("utf-8"),
    )
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            **dict(_DISABLED),
            "cutoff": cutoff_text,
            "decision_session": session,
            "factors": rows,
            "inventory_id": _identity(
                inventory_id,
                label="inventory_id",
            ),
            "protocol_version": PROTOCOL_VERSION,
            "request_ref": _artifact_ref(
                request_ref,
                strategy_id=strategy,
                cutoff=cutoff_text,
                label="request_ref",
                expected_version="myquant.v17.v4.forward-run-request.v1",
            ),
            "source_refs": _artifact_refs(
                source_refs,
                strategy_id=strategy,
                cutoff=cutoff_text,
                label="source_refs",
            ),
            "strategy_id": strategy,
            "version": EXISTING_FACTOR_INVENTORY_VERSION,
        }
    )
    return _registered(document, label="existing_factor_inventory")


def validate_existing_factor_inventory(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate schema and deterministic unique-factor ordering."""

    try:
        normalized = validate_semantic_sha(document)
    except Exception:
        _blocked("existing_factor_inventory_semantic_sha")
    _registered(normalized, label="existing_factor_inventory")
    names = [row["factor_name"] for row in normalized.get("factors", ())]
    if names != sorted(names, key=lambda value: value.encode("utf-8")) or len(names) != len(
        set(names)
    ):
        _blocked("existing_factor_inventory_order")
    return normalized


def _metric_rows(
    *,
    metric_state: str,
    metrics: Mapping[str, Any] | None,
    unavailable_reason: str | None,
) -> tuple[list[dict[str, Any]], list[str], str, str]:
    if metric_state not in METRIC_STATES:
        _blocked("metric_state")
    if metric_state == "UNAVAILABLE":
        if metrics:
            _blocked("unavailable_metrics_must_be_empty")
        reason = _identity(
            unavailable_reason,
            label="unavailable_reason",
        )
        return [], [reason], "UNAVAILABLE", "BLOCKED"
    if unavailable_reason is not None or not isinstance(metrics, Mapping) or not metrics:
        _blocked("complete_metric_contract")
    rows: list[dict[str, Any]] = []
    for metric_id, value in metrics.items():
        name = _identity(metric_id, label="metric_id")
        rows.append(
            {
                "metric_id": name,
                "status": "AVAILABLE",
                "value": _decimal_text(_decimal(value, label=f"metrics.{name}")),
            }
        )
    rows.sort(key=lambda row: row["metric_id"].encode("utf-8"))
    if len({row["metric_id"] for row in rows}) != len(rows):
        _blocked("metric_duplicate")
    return rows, [], "COMPLETE", "SUCCEEDED"


def _matching_origin_rows(
    inventory: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [row for row in inventory["origins"] if row["lineage_key"] == lineage]


def _build_evaluation_receipt(
    *,
    receipt_type: str,
    receipt_id: str,
    subject_id: str,
    factor_name: str,
    factor_definition_sha256: str,
    factor_set_sha256: str,
    quant_policy_sha256: str,
    horizon_sessions: int,
    source_lineage_sha256: str,
    cutoff: str,
    created_at: str,
    metric_state: str,
    metrics: Mapping[str, Any] | None,
    unavailable_reason: str | None,
    observation_run_ref: Mapping[str, Any],
    forward_evidence_origin_inventory: Mapping[str, Any],
    forward_evidence_origin_inventory_path: str,
    existing_factor_inventory: Mapping[str, Any],
    existing_factor_inventory_path: str,
) -> dict[str, Any]:
    if receipt_type not in RECEIPT_TYPES:
        _blocked("receipt_type")
    origin_inventory = validate_forward_evidence_origin_inventory(forward_evidence_origin_inventory)
    factor_inventory = validate_existing_factor_inventory(existing_factor_inventory)
    strategy = _identity(origin_inventory["strategy_id"], label="strategy_id")
    if factor_inventory["strategy_id"] != strategy:
        _blocked("inventory_strategy_mismatch")
    cutoff_text = _timestamp(cutoff, label="cutoff")
    recorded_at = _timestamp(created_at, label="created_at")
    if _instant(recorded_at, label="created_at") < _instant(
        cutoff_text,
        label="cutoff",
    ):
        _blocked("receipt_recorded_before_cutoff")
    if origin_inventory["cutoff"] > cutoff_text or factor_inventory["cutoff"] > cutoff_text:
        _blocked("inventory_after_cutoff")
    lineage = _lineage_key(
        factor_name=factor_name,
        factor_definition_sha256=factor_definition_sha256,
        factor_set_sha256=factor_set_sha256,
        quant_policy_sha256=quant_policy_sha256,
        horizon_sessions=horizon_sessions,
        source_lineage_sha256=source_lineage_sha256,
    )
    matching = _matching_origin_rows(origin_inventory, lineage)
    metric_rows, blockers, completeness, outcome = _metric_rows(
        metric_state=metric_state,
        metrics=metrics,
        unavailable_reason=unavailable_reason,
    )
    if metric_state == "COMPLETE" and not matching:
        _blocked("complete_without_matching_origins")
    label_refs = sorted(
        {
            canonical_bytes(row["canonical_evidence_ref"]): row["canonical_evidence_ref"]
            for row in matching
        }.values(),
        key=_ref_sort_key,
    )
    run_ref = _artifact_ref(
        observation_run_ref,
        strategy_id=strategy,
        cutoff=cutoff_text,
        label="observation_run_ref",
        expected_version="myquant.v17.v4.forward-observation-run.v1",
    )
    origin_ref = _inventory_artifact_ref(
        origin_inventory,
        relative_path=forward_evidence_origin_inventory_path,
        identity_field="inventory_id",
    )
    factor_ref = _inventory_artifact_ref(
        factor_inventory,
        relative_path=existing_factor_inventory_path,
        identity_field="inventory_id",
    )
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            **dict(_DISABLED),
            "blockers": sorted(blockers, key=lambda value: value.encode("utf-8")),
            "completeness": completeness,
            "cutoff": cutoff_text,
            "decision_session": str(origin_inventory["decision_session"]),
            "evidence_origin_inventory_ref": origin_ref,
            "execution_outcome": outcome,
            "existing_factor_inventory_ref": factor_ref,
            "label_refs": label_refs,
            "lineage_key": lineage,
            "lineage_key_sha256": hashlib.sha256(canonical_bytes(lineage)).hexdigest(),
            "metric_rows": metric_rows,
            "observation_run_ref": run_ref,
            "origin_count": len(matching),
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": _identity(receipt_id, label="receipt_id"),
            "receipt_type": receipt_type,
            "recorded_at": recorded_at,
            "strategy_id": strategy,
            "subject_id": _identity(subject_id, label="subject_id"),
            "version": FORWARD_EVALUATION_RECEIPT_VERSION,
        }
    )
    return _registered(document, label="forward_evaluation_receipt")


def build_factor_evaluation_receipt(
    *,
    receipt_id: str,
    factor_name: str,
    factor_definition_sha256: str,
    factor_set_sha256: str,
    quant_policy_sha256: str,
    horizon_sessions: int,
    source_lineage_sha256: str,
    cutoff: str,
    created_at: str,
    metric_state: Literal["COMPLETE", "UNAVAILABLE"],
    metrics: Mapping[str, Any] | None,
    observation_run_ref: Mapping[str, Any],
    forward_evidence_origin_inventory: Mapping[str, Any],
    forward_evidence_origin_inventory_path: str,
    existing_factor_inventory: Mapping[str, Any],
    existing_factor_inventory_path: str,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    return _build_evaluation_receipt(
        receipt_type="factor_evaluation_receipt",
        receipt_id=receipt_id,
        subject_id=factor_name,
        factor_name=factor_name,
        factor_definition_sha256=factor_definition_sha256,
        factor_set_sha256=factor_set_sha256,
        quant_policy_sha256=quant_policy_sha256,
        horizon_sessions=horizon_sessions,
        source_lineage_sha256=source_lineage_sha256,
        cutoff=cutoff,
        created_at=created_at,
        metric_state=metric_state,
        metrics=metrics,
        unavailable_reason=unavailable_reason,
        observation_run_ref=observation_run_ref,
        forward_evidence_origin_inventory=forward_evidence_origin_inventory,
        forward_evidence_origin_inventory_path=(forward_evidence_origin_inventory_path),
        existing_factor_inventory=existing_factor_inventory,
        existing_factor_inventory_path=existing_factor_inventory_path,
    )


def build_industry_evaluation_receipt(
    *,
    receipt_id: str,
    industry_id: str,
    factor_name: str,
    factor_definition_sha256: str,
    factor_set_sha256: str,
    quant_policy_sha256: str,
    horizon_sessions: int,
    source_lineage_sha256: str,
    cutoff: str,
    created_at: str,
    metric_state: Literal["COMPLETE", "UNAVAILABLE"],
    metrics: Mapping[str, Any] | None,
    observation_run_ref: Mapping[str, Any],
    forward_evidence_origin_inventory: Mapping[str, Any],
    forward_evidence_origin_inventory_path: str,
    existing_factor_inventory: Mapping[str, Any],
    existing_factor_inventory_path: str,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    return _build_evaluation_receipt(
        receipt_type="industry_evaluation_receipt",
        receipt_id=receipt_id,
        subject_id=industry_id,
        factor_name=factor_name,
        factor_definition_sha256=factor_definition_sha256,
        factor_set_sha256=factor_set_sha256,
        quant_policy_sha256=quant_policy_sha256,
        horizon_sessions=horizon_sessions,
        source_lineage_sha256=source_lineage_sha256,
        cutoff=cutoff,
        created_at=created_at,
        metric_state=metric_state,
        metrics=metrics,
        unavailable_reason=unavailable_reason,
        observation_run_ref=observation_run_ref,
        forward_evidence_origin_inventory=forward_evidence_origin_inventory,
        forward_evidence_origin_inventory_path=(forward_evidence_origin_inventory_path),
        existing_factor_inventory=existing_factor_inventory,
        existing_factor_inventory_path=existing_factor_inventory_path,
    )


def build_strategy_evaluation_receipt(
    *,
    receipt_id: str,
    strategy_id: str,
    factor_name: str,
    factor_definition_sha256: str,
    factor_set_sha256: str,
    quant_policy_sha256: str,
    horizon_sessions: int,
    source_lineage_sha256: str,
    cutoff: str,
    created_at: str,
    metric_state: Literal["COMPLETE", "UNAVAILABLE"],
    metrics: Mapping[str, Any] | None,
    observation_run_ref: Mapping[str, Any],
    forward_evidence_origin_inventory: Mapping[str, Any],
    forward_evidence_origin_inventory_path: str,
    existing_factor_inventory: Mapping[str, Any],
    existing_factor_inventory_path: str,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    return _build_evaluation_receipt(
        receipt_type="strategy_evaluation_receipt",
        receipt_id=receipt_id,
        subject_id=strategy_id,
        factor_name=factor_name,
        factor_definition_sha256=factor_definition_sha256,
        factor_set_sha256=factor_set_sha256,
        quant_policy_sha256=quant_policy_sha256,
        horizon_sessions=horizon_sessions,
        source_lineage_sha256=source_lineage_sha256,
        cutoff=cutoff,
        created_at=created_at,
        metric_state=metric_state,
        metrics=metrics,
        unavailable_reason=unavailable_reason,
        observation_run_ref=observation_run_ref,
        forward_evidence_origin_inventory=forward_evidence_origin_inventory,
        forward_evidence_origin_inventory_path=(forward_evidence_origin_inventory_path),
        existing_factor_inventory=existing_factor_inventory,
        existing_factor_inventory_path=existing_factor_inventory_path,
    )


def validate_evaluation_receipt(
    document: Mapping[str, Any],
    *,
    forward_evidence_origin_inventory: Mapping[str, Any] | None = None,
    existing_factor_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the single-version receipt, lineage, and optional readback."""

    try:
        normalized = validate_semantic_sha(document)
    except Exception:
        _blocked("receipt_semantic_sha")
    _registered(normalized, label="forward_evaluation_receipt")
    if normalized.get("receipt_type") not in RECEIPT_TYPES:
        _blocked("receipt_type")
    lineage = _lineage(normalized.get("lineage_key"), label="lineage_key")
    if normalized.get("lineage_key_sha256") != hashlib.sha256(canonical_bytes(lineage)).hexdigest():
        _blocked("lineage_key_sha")
    if normalized["recorded_at"] < normalized["cutoff"]:
        _blocked("receipt_recorded_before_cutoff")
    if forward_evidence_origin_inventory is not None:
        inventory = validate_forward_evidence_origin_inventory(forward_evidence_origin_inventory)
        expected_ref = _inventory_artifact_ref(
            inventory,
            relative_path=normalized["evidence_origin_inventory_ref"]["relative_path"],
            identity_field="inventory_id",
        )
        if expected_ref != normalized["evidence_origin_inventory_ref"]:
            _blocked("origin_inventory_readback")
        matching = _matching_origin_rows(inventory, lineage)
        if normalized["origin_count"] != len(matching):
            _blocked("origin_inventory_lineage")
        expected_labels = sorted(
            {
                canonical_bytes(row["canonical_evidence_ref"]): row["canonical_evidence_ref"]
                for row in matching
            }.values(),
            key=_ref_sort_key,
        )
        if normalized["label_refs"] != expected_labels:
            _blocked("origin_inventory_label_refs")
    if existing_factor_inventory is not None:
        inventory = validate_existing_factor_inventory(existing_factor_inventory)
        expected_ref = _inventory_artifact_ref(
            inventory,
            relative_path=normalized["existing_factor_inventory_ref"]["relative_path"],
            identity_field="inventory_id",
        )
        if expected_ref != normalized["existing_factor_inventory_ref"]:
            _blocked("existing_factor_inventory_readback")
    return normalized


def validate_factor_evaluation_receipt(
    document: Mapping[str, Any],
    **inventories: Any,
) -> dict[str, Any]:
    normalized = validate_evaluation_receipt(document, **inventories)
    if normalized["receipt_type"] != "factor_evaluation_receipt":
        _blocked("not_factor_evaluation_receipt")
    return normalized


def validate_industry_evaluation_receipt(
    document: Mapping[str, Any],
    **inventories: Any,
) -> dict[str, Any]:
    normalized = validate_evaluation_receipt(document, **inventories)
    if normalized["receipt_type"] != "industry_evaluation_receipt":
        _blocked("not_industry_evaluation_receipt")
    return normalized


def validate_strategy_evaluation_receipt(
    document: Mapping[str, Any],
    **inventories: Any,
) -> dict[str, Any]:
    normalized = validate_evaluation_receipt(document, **inventories)
    if normalized["receipt_type"] != "strategy_evaluation_receipt":
        _blocked("not_strategy_evaluation_receipt")
    return normalized


__all__ = [
    "EXISTING_FACTOR_INVENTORY_VERSION",
    "FACTOR_EVALUATION_RECEIPT_VERSION",
    "FORWARD_EVALUATION_RECEIPT_VERSION",
    "FORWARD_EVIDENCE_ORIGIN_INVENTORY_VERSION",
    "ForwardEvaluationReceiptError",
    "INDUSTRY_EVALUATION_RECEIPT_VERSION",
    "METRIC_STATES",
    "RECEIPT_TYPES",
    "STRATEGY_EVALUATION_RECEIPT_VERSION",
    "build_existing_factor_inventory",
    "build_factor_evaluation_receipt",
    "build_forward_evidence_origin_inventory",
    "build_industry_evaluation_receipt",
    "build_strategy_evaluation_receipt",
    "validate_evaluation_receipt",
    "validate_existing_factor_inventory",
    "validate_factor_evaluation_receipt",
    "validate_forward_evidence_origin_inventory",
    "validate_industry_evaluation_receipt",
    "validate_strategy_evaluation_receipt",
]

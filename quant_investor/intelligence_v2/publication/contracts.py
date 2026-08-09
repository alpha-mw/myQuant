"""Pure publication profiles, sidecar DAGs and CAS protocol contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any, Final, Protocol

from .._core import (
    IntelligenceV2ContractError,
    canonical_bytes,
    content_ref,
    exact_ref,
    identifier,
    require_exact_keys,
    seal,
    sha256,
    timestamp,
    validate_content_ref,
    validate_seal,
)

PUBLICATION_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "mainline_write_performed": False,
    "order": False,
    "paper_only": True,
    "production": False,
    "provider": False,
    "research_only": True,
    "trade": False,
}
PUBLICATION_PROFILE: Final = "INTELLIGENCE_V2_PUBLICATION_V1"

LEGACY_MARKER_PROFILE_VERSION: Final = "myquant.v17.intelligence-v2.legacy-marker-profile.v1"
LEGACY_MARKER_VERSION: Final = "myquant.v17.intelligence-v2.legacy-marker.v1"
PUBLICATION_SIDECAR_VERSION: Final = "myquant.v17.intelligence-v2.activation-sidecar.v1"
PUBLICATION_CLOSURE_VERSION: Final = "myquant.v17.intelligence-v2.publication-closure.v1"
PREACTIVATION_VERSION: Final = "myquant.v17.intelligence-v2.preactivation-receipt.v1"
CAS_REQUEST_VERSION: Final = "myquant.v17.intelligence-v2.pointer-cas-request.v2"
CAS_RECEIPT_VERSION: Final = "myquant.v17.intelligence-v2.pointer-cas-receipt.v2"
QUARANTINE_VERSION: Final = "myquant.v17.intelligence-v2.quarantine-receipt.v2"
ROLLBACK_VERSION: Final = "myquant.v17.intelligence-v2.rollback-receipt.v2"

LEGACY_PROTOCOL: Final = "myquant.v17.v4"
LEGACY_RUN_SCHEMA: Final = "myquant.v17.v4.mainline-run.v1"
LEGACY_POINTER_SCHEMA: Final = "myquant.v17.v4.mainline-active-pointer.v1"
LEGACY_PUBLIC_SCHEMA: Final = "myquant.v17.v4.mainline-public-run.v1"
PUBLICATION_CLOSURE_VERSIONS: Final = {
    "DECISION_V2": "myquant.v17.research-intelligence-v2.decision-receipt.v1",
    "EVIDENCE_GRAPH_V2": "myquant.v17.research-intelligence-v2.evidence-graph.v1",
    "GRADUATION": "myquant.v17.intelligence-v2.graduation-receipt.v2",
    "GRADUATION_POLICY": "myquant.v17.intelligence-v2.graduation-policy.v2",
    "I5_ADVISORY_RANK": "myquant.v17.research-intelligence-v2.i5-advisory-rank.v1",
    "I5_PRIVATE_CAPABILITY": "myquant.v17.research-intelligence-v2.i5-private-capability.v1",
    "LEGACY_MARKER_PROFILE": LEGACY_MARKER_PROFILE_VERSION,
    "PAPER_EXECUTION_POLICY": "myquant.v17.intelligence-v2.paper-execution-policy.v2",
    "PAPER_LEDGER": "myquant.v17.intelligence-v2.paper-ledger.v2",
    "PAPER_CAPITAL_GATE": "myquant.v17.intelligence-v2.paper-capital-gate-receipt.v1",
    "PORTFOLIO": "myquant.v17.intelligence-v2.portfolio-construction-receipt.v2",
    "PORTFOLIO_POLICY": "myquant.v17.intelligence-v2.portfolio-risk-policy.v2",
    "PREACTIVATION": PREACTIVATION_VERSION,
    "PUBLICATION_OWNER_POLICY": "myquant.v17.intelligence-v2.publication-owner-policy.v1",
}

_COMMON_FIELDS: Final = {
    "authority",
    "decision_protocol",
    "production",
    "publication_profile",
    "research_only",
    "semantic_sha256",
    "timestamp",
    "version",
}
PROFILE_FIELDS: Final = _COMMON_FIELDS | {
    "canonical_strategy_id",
    "legacy_active_pointer_schema",
    "legacy_mainline_root",
    "legacy_public_schema",
    "legacy_run_schema",
    "marker_required",
    "profile_id",
    "sidecar_root",
    "single_legacy_pointer",
}
MARKER_FIELDS: Final = _COMMON_FIELDS | {
    "canonical_strategy_id",
    "graduation_ref",
    "legacy_run_ref",
    "marker_id",
    "marker_path",
    "portfolio_ref",
    "profile_ref",
    "risk_ref",
    "target_pointer_ref",
    "transaction_id",
}
SIDECAR_FIELDS: Final = _COMMON_FIELDS | {
    "canonical_strategy_id",
    "legacy_run_ref",
    "marker_ref",
    "publication_closure_ref",
    "sidecar_id",
    "sidecar_path",
    "target_pointer_ref",
    "transaction_id",
}
PUBLICATION_CLOSURE_FIELDS: Final = _COMMON_FIELDS | {
    "canonical_strategy_id",
    "closure_id",
    "closure_path",
    "edges",
    "nodes",
    "outcome_refs",
    "transaction_id",
}
PREACTIVATION_FIELDS: Final = _COMMON_FIELDS | {
    "blocker_codes",
    "candidate_refs",
    "expected_pointer_sha256",
    "preactivation_id",
    "readiness",
    "rollback_target_ref",
    "status",
    "write_performed",
}
CAS_REQUEST_FIELDS: Final = _COMMON_FIELDS | {
    "canonical_strategy_id",
    "expected_pointer_sha256",
    "permit_ref",
    "request_id",
    "request_path",
    "sidecar_ref",
    "target_pointer_ref",
    "transaction_id",
    "write_performed",
}
CAS_RECEIPT_FIELDS: Final = _COMMON_FIELDS | {
    "expected_pointer_sha256",
    "observed_pointer_sha256",
    "receipt_id",
    "request_ref",
    "status",
    "target_pointer_sha256",
    "write_performed",
}
QUARANTINE_FIELDS: Final = _COMMON_FIELDS | {
    "canonical_strategy_id",
    "permit_ref",
    "quarantine_id",
    "reason_codes",
    "sidecar_ref",
    "status",
    "transaction_id",
    "quarantine_sidecar_path",
    "write_performed",
}
ROLLBACK_FIELDS: Final = _COMMON_FIELDS | {
    "expected_current_pointer_sha256",
    "permit_ref",
    "rollback_id",
    "rollback_target_ref",
    "sidecar_ref",
    "status",
    "write_performed",
}


class PublicationContractError(IntelligenceV2ContractError):
    """Stable fail-closed publication contract error."""


class ExpectedPointerCAS(Protocol):
    """External capability boundary; I6 supplies no implementation."""

    def compare_and_swap(self, request: Mapping[str, Any], /) -> Mapping[str, Any]: ...


def publication_common(*, at: str) -> dict[str, Any]:
    return {
        "authority": dict(PUBLICATION_AUTHORITY),
        "decision_protocol": LEGACY_PROTOCOL,
        "production": False,
        "publication_profile": PUBLICATION_PROFILE,
        "research_only": True,
        "timestamp": timestamp(at, label="timestamp"),
    }


def _exact_at(value: Mapping[str, Any], *, label: str, at: str) -> dict[str, str]:
    row = exact_ref(value, label=label)
    if row["available_at"] > at or row["cutoff"] > at:
        raise PublicationContractError(f"{label} contains future input")
    return row


def _replay(
    document: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    identity_field: str,
    fields: set[str],
    version: str,
    label: str,
) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field=identity_field)
    require_exact_keys(normalized, fields, label=label)
    if normalized != expected or normalized["version"] != version:
        raise PublicationContractError(f"{label} replay mismatch")
    return normalized


def _path_id(value: Any, *, label: str) -> str:
    result = identifier(value, label=label)
    if "/" in result or ".." in result:
        raise PublicationContractError(f"{label} is unsafe")
    return result


def _v2_store_ref(value: Mapping[str, Any], *, label: str, at: str) -> dict[str, str]:
    row = _exact_at(value, label=label, at=at)
    if not row["relative_path"].startswith("results/v17_intelligence_v2/"):
        raise PublicationContractError(f"{label} is outside the immutable v2 store")
    return row


def _document_exact_ref(
    document: Mapping[str, Any],
    *,
    identity_field: str,
    relative_path: str,
    available_at: str,
) -> dict[str, str]:
    normalized = validate_seal(document, identity_field=identity_field)
    return exact_ref(
        {
            "artifact_id": str(normalized[identity_field]),
            "artifact_version": str(normalized["version"]),
            "available_at": available_at,
            "byte_sha256": hashlib.sha256(canonical_bytes(normalized)).hexdigest(),
            "cutoff": str(normalized["timestamp"]),
            "relative_path": relative_path,
            "semantic_sha256": str(normalized["semantic_sha256"]),
        },
        label="document exact ref",
    )


def derive_publication_paths(
    *,
    strategy_id: str,
    transaction_id: str,
    target_pointer_sha256: str,
    run_id: str,
) -> dict[str, str]:
    strategy = _path_id(strategy_id, label="strategy_id")
    transaction = _path_id(transaction_id, label="transaction_id")
    target_sha = sha256(target_pointer_sha256, label="target_pointer_sha256")
    run = _path_id(run_id, label="run_id")
    immutable_root = (
        f"results/v17_intelligence_v2/{PUBLICATION_PROFILE}/strategies/"
        f"{strategy}/transactions/{transaction}"
    )
    legacy_root = f"results/v17_mainline/strategies/{strategy}"
    return {
        "activation_permit": f"{legacy_root}/activation_permits/{target_sha}.json",
        "activation_sidecar": f"{immutable_root}/activation-sidecar.json",
        "cas_receipt": f"{immutable_root}/pointer-cas-receipt.json",
        "cas_request": f"{immutable_root}/pointer-cas-request.json",
        "legacy_marker": f"{immutable_root}/legacy-marker.json",
        "publication_closure": f"{immutable_root}/publication-closure.json",
        "quarantine_sidecar": f"{legacy_root}/quarantine/{run}.json",
        "rollback_receipt": f"{immutable_root}/rollback-receipt.json",
    }


def build_legacy_marker_profile(*, created_at: str, canonical_strategy_id: str) -> dict[str, Any]:
    issued_at = timestamp(created_at, label="created_at")
    strategy = _path_id(canonical_strategy_id, label="canonical_strategy_id")
    return seal(
        {
            **publication_common(at=issued_at),
            "canonical_strategy_id": strategy,
            "legacy_active_pointer_schema": LEGACY_POINTER_SCHEMA,
            "legacy_mainline_root": "results/v17_mainline",
            "legacy_public_schema": LEGACY_PUBLIC_SCHEMA,
            "legacy_run_schema": LEGACY_RUN_SCHEMA,
            "marker_required": True,
            "sidecar_root": "results/v17_intelligence_v2",
            "single_legacy_pointer": True,
            "version": LEGACY_MARKER_PROFILE_VERSION,
        },
        identity_field="profile_id",
    )


def validate_legacy_marker_profile(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="profile_id")
    require_exact_keys(normalized, PROFILE_FIELDS, label="legacy marker profile")
    expected = build_legacy_marker_profile(
        created_at=normalized.get("timestamp"),
        canonical_strategy_id=normalized.get("canonical_strategy_id"),
    )
    if normalized != expected or normalized["version"] != LEGACY_MARKER_PROFILE_VERSION:
        raise PublicationContractError("legacy marker profile replay mismatch")
    return normalized


def build_legacy_marker(
    *,
    profile: Mapping[str, Any],
    transaction_id: str,
    legacy_run_ref: Mapping[str, Any],
    target_pointer_ref: Mapping[str, Any],
    portfolio_ref: Mapping[str, Any],
    risk_ref: Mapping[str, Any],
    graduation_ref: Mapping[str, Any],
    built_at: str,
) -> dict[str, Any]:
    validated_profile = validate_legacy_marker_profile(profile)
    issued_at = timestamp(built_at, label="built_at")
    transaction = _path_id(transaction_id, label="transaction_id")
    run_ref = _exact_at(legacy_run_ref, label="legacy_run_ref", at=issued_at)
    pointer_ref = _exact_at(target_pointer_ref, label="target_pointer_ref", at=issued_at)
    if run_ref["artifact_version"] != LEGACY_RUN_SCHEMA:
        raise PublicationContractError("legacy run schema mismatch")
    if pointer_ref["artifact_version"] != LEGACY_POINTER_SCHEMA:
        raise PublicationContractError("legacy pointer schema mismatch")
    normalized_risk_ref = validate_content_ref(risk_ref, label="risk_ref")
    if normalized_risk_ref["artifact_version"] != PUBLICATION_CLOSURE_VERSIONS["EVIDENCE_GRAPH_V2"]:
        raise PublicationContractError("risk_ref must bind the v2 evidence graph")
    return seal(
        {
            **publication_common(at=issued_at),
            "canonical_strategy_id": validated_profile["canonical_strategy_id"],
            "graduation_ref": validate_content_ref(graduation_ref, label="graduation_ref"),
            "legacy_run_ref": run_ref,
            "marker_path": derive_publication_paths(
                strategy_id=validated_profile["canonical_strategy_id"],
                transaction_id=transaction,
                target_pointer_sha256=pointer_ref["byte_sha256"],
                run_id=run_ref["artifact_id"],
            )["legacy_marker"],
            "portfolio_ref": validate_content_ref(portfolio_ref, label="portfolio_ref"),
            "profile_ref": content_ref(validated_profile, identity_field="profile_id"),
            "risk_ref": normalized_risk_ref,
            "target_pointer_ref": pointer_ref,
            "transaction_id": transaction,
            "version": LEGACY_MARKER_VERSION,
        },
        identity_field="marker_id",
    )


def validate_legacy_marker(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    return _replay(
        document,
        expected=build_legacy_marker(**closure),
        identity_field="marker_id",
        fields=MARKER_FIELDS,
        version=LEGACY_MARKER_VERSION,
        label="legacy marker",
    )


def build_publication_closure(
    *,
    canonical_strategy_id: str,
    transaction_id: str,
    closure_refs: Mapping[str, Mapping[str, Any]],
    outcome_refs: Sequence[Mapping[str, Any]],
    built_at: str,
) -> dict[str, Any]:
    """Build the upstream research-only closure; no legacy bytes are reachable."""

    issued_at = timestamp(built_at, label="built_at")
    strategy = _path_id(canonical_strategy_id, label="canonical_strategy_id")
    transaction = _path_id(transaction_id, label="transaction_id")
    closure = require_exact_keys(
        closure_refs,
        set(PUBLICATION_CLOSURE_VERSIONS),
        label="publication closure refs",
    )
    nodes = {
        key: _v2_store_ref(value, label=f"closure_refs.{key}", at=issued_at)
        for key, value in closure.items()
    }
    for key, row in nodes.items():
        if row["artifact_version"] != PUBLICATION_CLOSURE_VERSIONS[key]:
            raise PublicationContractError(f"publication closure {key} artifact version is invalid")
    node_identities = [(row["byte_sha256"], row["semantic_sha256"]) for row in nodes.values()]
    if len(node_identities) != len(set(node_identities)):
        raise PublicationContractError("publication closure contains duplicate refs")
    refs = [_v2_store_ref(value, label="outcome_ref", at=issued_at) for value in outcome_refs]
    keys = [
        (
            row["artifact_id"],
            row["artifact_version"],
            row["byte_sha256"],
            row["semantic_sha256"],
        )
        for row in refs
    ]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise PublicationContractError("outcome refs must be sorted and unique")
    edge_sources = sorted(key for key in PUBLICATION_CLOSURE_VERSIONS if key != "PREACTIVATION")
    return seal(
        {
            **publication_common(at=issued_at),
            "canonical_strategy_id": strategy,
            "closure_path": derive_publication_paths(
                strategy_id=strategy,
                transaction_id=transaction,
                target_pointer_sha256="0" * 64,
                run_id="path-only",
            )["publication_closure"],
            "edges": [{"from": source, "to": "PREACTIVATION"} for source in edge_sources],
            "nodes": nodes,
            "outcome_refs": refs,
            "transaction_id": transaction,
            "version": PUBLICATION_CLOSURE_VERSION,
        },
        identity_field="closure_id",
    )


def validate_publication_closure(document: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_seal(document, identity_field="closure_id")
    require_exact_keys(normalized, PUBLICATION_CLOSURE_FIELDS, label="publication closure")
    expected = build_publication_closure(
        canonical_strategy_id=normalized.get("canonical_strategy_id"),
        transaction_id=normalized.get("transaction_id"),
        closure_refs=normalized.get("nodes"),
        outcome_refs=normalized.get("outcome_refs"),
        built_at=normalized.get("timestamp"),
    )
    if normalized != expected or normalized["version"] != PUBLICATION_CLOSURE_VERSION:
        raise PublicationContractError("publication closure replay mismatch")
    return normalized


def build_activation_sidecar(
    *,
    marker: Mapping[str, Any],
    marker_validation_closure: Mapping[str, Any],
    publication_closure: Mapping[str, Any],
    built_at: str,
) -> dict[str, Any]:
    if type(marker_validation_closure) is not dict:
        raise PublicationContractError("marker_validation_closure must be exact")
    expected_marker = build_legacy_marker(**dict(marker_validation_closure))
    normalized_marker = validate_seal(marker, identity_field="marker_id")
    if normalized_marker != expected_marker:
        raise PublicationContractError("legacy marker replay mismatch")
    normalized_closure = validate_publication_closure(publication_closure)
    if (
        normalized_closure["canonical_strategy_id"] != normalized_marker["canonical_strategy_id"]
        or normalized_closure["transaction_id"] != normalized_marker["transaction_id"]
    ):
        raise PublicationContractError("publication closure scope mismatch")
    profile_node = normalized_closure["nodes"]["LEGACY_MARKER_PROFILE"]
    profile_content_ref = {
        key: profile_node[key]
        for key in (
            "artifact_id",
            "artifact_version",
            "byte_sha256",
            "semantic_sha256",
        )
    }
    if normalized_marker["profile_ref"] != profile_content_ref:
        raise PublicationContractError("legacy marker profile is outside publication closure")
    issued_at = timestamp(built_at, label="built_at")
    if normalized_closure["timestamp"] > issued_at or normalized_marker["timestamp"] > issued_at:
        raise PublicationContractError("activation sidecar contains future input")
    marker_ref = _document_exact_ref(
        normalized_marker,
        identity_field="marker_id",
        relative_path=normalized_marker["marker_path"],
        available_at=issued_at,
    )
    closure_ref = _document_exact_ref(
        normalized_closure,
        identity_field="closure_id",
        relative_path=normalized_closure["closure_path"],
        available_at=issued_at,
    )
    return seal(
        {
            **publication_common(at=issued_at),
            "canonical_strategy_id": normalized_marker["canonical_strategy_id"],
            "legacy_run_ref": normalized_marker["legacy_run_ref"],
            "marker_ref": marker_ref,
            "publication_closure_ref": closure_ref,
            "sidecar_path": derive_publication_paths(
                strategy_id=normalized_marker["canonical_strategy_id"],
                transaction_id=normalized_marker["transaction_id"],
                target_pointer_sha256=normalized_marker["target_pointer_ref"]["byte_sha256"],
                run_id=normalized_marker["legacy_run_ref"]["artifact_id"],
            )["activation_sidecar"],
            "target_pointer_ref": normalized_marker["target_pointer_ref"],
            "transaction_id": normalized_marker["transaction_id"],
            "version": PUBLICATION_SIDECAR_VERSION,
        },
        identity_field="sidecar_id",
    )


def validate_activation_sidecar(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    return _replay(
        document,
        expected=build_activation_sidecar(**closure),
        identity_field="sidecar_id",
        fields=SIDECAR_FIELDS,
        version=PUBLICATION_SIDECAR_VERSION,
        label="activation sidecar",
    )


def build_preactivation_receipt(
    *,
    candidate_refs: Sequence[Mapping[str, Any]],
    expected_pointer_sha256: str,
    rollback_target_ref: Mapping[str, Any] | None,
    blocker_codes: Sequence[str],
    evaluated_at: str,
) -> dict[str, Any]:
    """Seal owner-review readiness without performing or authorizing a write."""

    issued_at = timestamp(evaluated_at, label="evaluated_at")
    expected_pointer = (
        "EMPTY"
        if expected_pointer_sha256 == "EMPTY"
        else sha256(expected_pointer_sha256, label="expected_pointer_sha256")
    )
    if expected_pointer == "EMPTY":
        if rollback_target_ref is not None:
            raise PublicationContractError("first activation cannot claim a rollback target")
        rollback_target = None
    else:
        if rollback_target_ref is None:
            raise PublicationContractError("existing pointer requires an exact rollback target")
        rollback_target = _exact_at(
            rollback_target_ref,
            label="rollback_target_ref",
            at=issued_at,
        )
        if rollback_target["artifact_version"] != LEGACY_POINTER_SCHEMA:
            raise PublicationContractError("rollback target is not a legacy pointer")
        if rollback_target["byte_sha256"] != expected_pointer:
            raise PublicationContractError("rollback target does not match expected pointer")
    candidates = [
        _v2_store_ref(value, label="candidate_ref", at=issued_at) for value in candidate_refs
    ]
    candidate_keys = [
        (
            row["artifact_id"],
            row["artifact_version"],
            row["byte_sha256"],
            row["semantic_sha256"],
        )
        for row in candidates
    ]
    if not candidates or candidate_keys != sorted(candidate_keys):
        raise PublicationContractError("candidate refs must be nonempty and sorted")
    if len(candidate_keys) != len(set(candidate_keys)):
        raise PublicationContractError("candidate refs must be unique")
    blockers = [identifier(value, label="blocker_code") for value in blocker_codes]
    if blockers != sorted(blockers) or len(blockers) != len(set(blockers)):
        raise PublicationContractError("blocker codes must be sorted and unique")
    ready = not blockers
    return seal(
        {
            **publication_common(at=issued_at),
            "blocker_codes": blockers,
            "candidate_refs": candidates,
            "expected_pointer_sha256": expected_pointer,
            "readiness": ready,
            "rollback_target_ref": rollback_target,
            "status": "READY" if ready else "NOT_READY",
            "version": PREACTIVATION_VERSION,
            "write_performed": False,
        },
        identity_field="preactivation_id",
    )


def validate_preactivation_receipt(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    return _replay(
        document,
        expected=build_preactivation_receipt(**closure),
        identity_field="preactivation_id",
        fields=PREACTIVATION_FIELDS,
        version=PREACTIVATION_VERSION,
        label="preactivation receipt",
    )


def build_expected_pointer_cas_request(
    *,
    sidecar_ref: Mapping[str, Any],
    permit_ref: Mapping[str, Any],
    canonical_strategy_id: str,
    transaction_id: str,
    run_id: str,
    expected_pointer_sha256: str,
    target_pointer_ref: Mapping[str, Any],
    requested_at: str,
) -> dict[str, Any]:
    issued_at = timestamp(requested_at, label="requested_at")
    expected = (
        "EMPTY"
        if expected_pointer_sha256 == "EMPTY"
        else sha256(expected_pointer_sha256, label="expected_pointer_sha256")
    )
    strategy = _path_id(canonical_strategy_id, label="canonical_strategy_id")
    transaction = _path_id(transaction_id, label="transaction_id")
    target = _exact_at(target_pointer_ref, label="target_pointer_ref", at=issued_at)
    if target["artifact_version"] != LEGACY_POINTER_SCHEMA:
        raise PublicationContractError("CAS target is not the legacy active pointer")
    return seal(
        {
            **publication_common(at=issued_at),
            "canonical_strategy_id": strategy,
            "expected_pointer_sha256": expected,
            "permit_ref": validate_content_ref(permit_ref, label="permit_ref"),
            "request_path": derive_publication_paths(
                strategy_id=strategy,
                transaction_id=transaction,
                target_pointer_sha256=target["byte_sha256"],
                run_id=run_id,
            )["cas_request"],
            "sidecar_ref": validate_content_ref(sidecar_ref, label="sidecar_ref"),
            "target_pointer_ref": target,
            "transaction_id": transaction,
            "version": CAS_REQUEST_VERSION,
            "write_performed": False,
        },
        identity_field="request_id",
    )


def validate_expected_pointer_cas_request(
    document: Mapping[str, Any], **closure: Any
) -> dict[str, Any]:
    return _replay(
        document,
        expected=build_expected_pointer_cas_request(**closure),
        identity_field="request_id",
        fields=CAS_REQUEST_FIELDS,
        version=CAS_REQUEST_VERSION,
        label="pointer CAS request",
    )


def build_expected_pointer_cas_receipt(
    *,
    request_ref: Mapping[str, Any],
    expected_pointer_sha256: str,
    observed_pointer_sha256: str,
    target_pointer_sha256: str,
    status: str,
    received_at: str,
) -> dict[str, Any]:
    issued_at = timestamp(received_at, label="received_at")
    if status not in {"APPLIED", "CONFLICT", "NOT_ATTEMPTED"}:
        raise PublicationContractError("CAS receipt status is invalid")
    expected = (
        "EMPTY"
        if expected_pointer_sha256 == "EMPTY"
        else sha256(expected_pointer_sha256, label="expected_pointer_sha256")
    )
    observed = (
        "EMPTY"
        if observed_pointer_sha256 == "EMPTY"
        else sha256(observed_pointer_sha256, label="observed_pointer_sha256")
    )
    target = sha256(target_pointer_sha256, label="target_pointer_sha256")
    if status == "APPLIED" and observed != expected:
        raise PublicationContractError("applied CAS receipt has predecessor mismatch")
    if status == "CONFLICT" and observed == expected:
        raise PublicationContractError("conflict CAS receipt has no conflict")
    return seal(
        {
            **publication_common(at=issued_at),
            "expected_pointer_sha256": expected,
            "observed_pointer_sha256": observed,
            "request_ref": validate_content_ref(request_ref, label="request_ref"),
            "status": status,
            "target_pointer_sha256": target,
            "version": CAS_RECEIPT_VERSION,
            "write_performed": status == "APPLIED",
        },
        identity_field="receipt_id",
    )


def validate_expected_pointer_cas_receipt(
    document: Mapping[str, Any], **closure: Any
) -> dict[str, Any]:
    return _replay(
        document,
        expected=build_expected_pointer_cas_receipt(**closure),
        identity_field="receipt_id",
        fields=CAS_RECEIPT_FIELDS,
        version=CAS_RECEIPT_VERSION,
        label="pointer CAS receipt",
    )


def build_quarantine_receipt(
    *,
    sidecar_ref: Mapping[str, Any],
    permit_ref: Mapping[str, Any],
    canonical_strategy_id: str,
    transaction_id: str,
    target_pointer_sha256: str,
    run_id: str,
    reason_codes: Sequence[str],
    quarantined_at: str,
) -> dict[str, Any]:
    issued_at = timestamp(quarantined_at, label="quarantined_at")
    reasons = [identifier(value, label="reason_code") for value in reason_codes]
    if not reasons or reasons != sorted(reasons) or len(reasons) != len(set(reasons)):
        raise PublicationContractError("quarantine reasons must be sorted and unique")
    strategy = _path_id(canonical_strategy_id, label="canonical_strategy_id")
    transaction = _path_id(transaction_id, label="transaction_id")
    return seal(
        {
            **publication_common(at=issued_at),
            "canonical_strategy_id": strategy,
            "permit_ref": validate_content_ref(permit_ref, label="permit_ref"),
            "quarantine_sidecar_path": derive_publication_paths(
                strategy_id=strategy,
                transaction_id=transaction,
                target_pointer_sha256=target_pointer_sha256,
                run_id=run_id,
            )["quarantine_sidecar"],
            "reason_codes": reasons,
            "sidecar_ref": validate_content_ref(sidecar_ref, label="sidecar_ref"),
            "status": "QUARANTINED",
            "transaction_id": transaction,
            "version": QUARANTINE_VERSION,
            "write_performed": False,
        },
        identity_field="quarantine_id",
    )


def validate_quarantine_receipt(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    return _replay(
        document,
        expected=build_quarantine_receipt(**closure),
        identity_field="quarantine_id",
        fields=QUARANTINE_FIELDS,
        version=QUARANTINE_VERSION,
        label="quarantine receipt",
    )


def build_rollback_receipt(
    *,
    sidecar_ref: Mapping[str, Any],
    permit_ref: Mapping[str, Any],
    expected_current_pointer_sha256: str,
    rollback_target_ref: Mapping[str, Any],
    status: str,
    rolled_back_at: str,
) -> dict[str, Any]:
    issued_at = timestamp(rolled_back_at, label="rolled_back_at")
    if status not in {"REQUESTED", "APPLIED", "CONFLICT"}:
        raise PublicationContractError("rollback status is invalid")
    target = _exact_at(rollback_target_ref, label="rollback_target_ref", at=issued_at)
    if target["artifact_version"] != LEGACY_POINTER_SCHEMA:
        raise PublicationContractError("rollback target is not a legacy pointer")
    return seal(
        {
            **publication_common(at=issued_at),
            "expected_current_pointer_sha256": sha256(
                expected_current_pointer_sha256,
                label="expected_current_pointer_sha256",
            ),
            "permit_ref": validate_content_ref(permit_ref, label="permit_ref"),
            "rollback_target_ref": target,
            "sidecar_ref": validate_content_ref(sidecar_ref, label="sidecar_ref"),
            "status": status,
            "version": ROLLBACK_VERSION,
            "write_performed": status == "APPLIED",
        },
        identity_field="rollback_id",
    )


def validate_rollback_receipt(document: Mapping[str, Any], **closure: Any) -> dict[str, Any]:
    return _replay(
        document,
        expected=build_rollback_receipt(**closure),
        identity_field="rollback_id",
        fields=ROLLBACK_FIELDS,
        version=ROLLBACK_VERSION,
        label="rollback receipt",
    )


__all__ = [
    "CAS_RECEIPT_VERSION",
    "CAS_REQUEST_VERSION",
    "ExpectedPointerCAS",
    "LEGACY_MARKER_PROFILE_VERSION",
    "LEGACY_MARKER_VERSION",
    "PREACTIVATION_VERSION",
    "PUBLICATION_CLOSURE_VERSION",
    "PUBLICATION_SIDECAR_VERSION",
    "PUBLICATION_PROFILE",
    "PUBLICATION_CLOSURE_VERSIONS",
    "PublicationContractError",
    "build_activation_sidecar",
    "build_expected_pointer_cas_receipt",
    "build_expected_pointer_cas_request",
    "build_legacy_marker",
    "build_legacy_marker_profile",
    "build_preactivation_receipt",
    "build_publication_closure",
    "build_quarantine_receipt",
    "build_rollback_receipt",
    "derive_publication_paths",
    "validate_legacy_marker_profile",
    "validate_activation_sidecar",
    "validate_expected_pointer_cas_receipt",
    "validate_expected_pointer_cas_request",
    "validate_legacy_marker",
    "validate_preactivation_receipt",
    "validate_publication_closure",
    "validate_quarantine_receipt",
    "validate_rollback_receipt",
]

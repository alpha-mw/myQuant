"""Fail-closed validation for marked Investment Intelligence v2 publications."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal
import json
from pathlib import PurePosixPath
from typing import Any, Final

from ..intelligence_v2._core import (
    FROZEN_V1_MANIFEST_SHA256,
    canonical_bytes as v2_canonical_bytes,
    content_ref,
    validate_content_ref,
    validate_seal,
)
from ..intelligence_v2.llm_research import validate_private_capability
from ..intelligence_v2.portfolio import (
    validate_graduation_policy,
    validate_market_risk_projection,
    validate_paper_execution_policy,
    validate_portfolio_risk_policy,
)
from ..intelligence_v2.publication import (
    PUBLICATION_CLOSURE_VERSION,
    PUBLICATION_PROFILE,
    build_preactivation_receipt,
    derive_publication_paths,
    validate_action_permit,
    validate_activation_sidecar,
    validate_legacy_marker,
    validate_legacy_marker_profile,
    validate_preactivation_receipt,
    validate_publication_closure,
    validate_publication_owner_policy,
    validate_quarantine_receipt,
)
from ..intelligence_v2.publication.contracts import (
    PUBLICATION_CLOSURE_VERSIONS,
    PUBLICATION_SIDECAR_VERSION,
    QUARANTINE_VERSION,
)
from .constants import (
    ACTIVE_POINTER_SCHEMA_ID,
    INTELLIGENCE_V2_ROOT,
    MAINLINE_RUN_SCHEMA_ID,
    PROTOCOL,
)
from .contracts import byte_sha256, validate_ref
from .storage import MainlineNotFound, MainlineStore, StoredBytes

_IDENTITY_FIELDS: Final = {
    "DECISION_V2": "decision_id",
    "EVIDENCE_GRAPH_V2": "graph_id",
    "GRADUATION": "graduation_id",
    "GRADUATION_POLICY": "policy_id",
    "I5_ADVISORY_RANK": "advisory_rank_id",
    "I5_PRIVATE_CAPABILITY": "private_capability_id",
    "LEGACY_MARKER_PROFILE": "profile_id",
    "MARKET_RISK_PROJECTION": "projection_id",
    "PAPER_CAPITAL_GATE": "receipt_id",
    "PAPER_EXECUTION_POLICY": "policy_id",
    "PAPER_LEDGER": "ledger_id",
    "PORTFOLIO": "receipt_id",
    "PORTFOLIO_POLICY": "policy_id",
    "PREACTIVATION": "preactivation_id",
    "PUBLICATION_OWNER_POLICY": "policy_id",
}
_SELF_VALIDATORS: Final[dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]] = {
    "GRADUATION_POLICY": validate_graduation_policy,
    "I5_PRIVATE_CAPABILITY": validate_private_capability,
    "LEGACY_MARKER_PROFILE": validate_legacy_marker_profile,
    "PAPER_EXECUTION_POLICY": validate_paper_execution_policy,
    "PORTFOLIO_POLICY": validate_portfolio_risk_policy,
    "PUBLICATION_OWNER_POLICY": validate_publication_owner_policy,
}
_FORBIDDEN_TRUE_AUTHORITY_KEYS: Final = frozenset(
    {
        "broker",
        "execution",
        "mainline_authority",
        "mainline_write_performed",
        "order",
        "portfolio_mutation",
        "production",
        "provider",
        "trade",
    }
)


class IntelligenceV2PublicationError(ValueError):
    """A marked active run lacks a complete, valid publication closure."""


class ActiveRunQuarantined(IntelligenceV2PublicationError):
    """The exact active run has a valid fixed-path quarantine marker."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntelligenceV2PublicationError("v2 artifact contains duplicate JSON key")
        result[key] = value
    return result


def _parse_v2(stored: StoredBytes) -> dict[str, Any]:
    try:
        value = json.loads(
            stored.data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except IntelligenceV2PublicationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IntelligenceV2PublicationError("v2 artifact is not canonical JSON") from exc
    if type(value) is not dict or v2_canonical_bytes(value) != stored.data:
        raise IntelligenceV2PublicationError("v2 artifact bytes are not canonical")
    return value


def _authority_closed(document: Mapping[str, Any]) -> None:
    if (
        document.get("research_only") is not True
        or document.get("production") is not False
        or document.get("decision_protocol") != PROTOCOL
    ):
        raise IntelligenceV2PublicationError("v2 artifact authority is open")
    frozen = document.get("frozen_v1_manifest_sha256")
    if frozen is not None and frozen != FROZEN_V1_MANIFEST_SHA256:
        raise IntelligenceV2PublicationError("v2 artifact frozen-v1 binding mismatch")
    authority = document.get("authority")
    if type(authority) is not dict:
        raise IntelligenceV2PublicationError("v2 artifact authority is missing")
    if any(authority.get(key) is True for key in _FORBIDDEN_TRUE_AUTHORITY_KEYS):
        raise IntelligenceV2PublicationError("v2 artifact grants forbidden authority")


def _read_exact_v2(
    store: MainlineStore,
    reference: Mapping[str, Any],
    *,
    expected_version: str,
    identity_field: str,
) -> tuple[dict[str, Any], StoredBytes]:
    path = reference.get("relative_path")
    if type(path) is not str or not path.startswith(INTELLIGENCE_V2_ROOT + "/"):
        raise IntelligenceV2PublicationError("v2 artifact path is outside immutable root")
    stored = store.read(path, reference.get("byte_sha256"))
    document = validate_seal(_parse_v2(stored), identity_field=identity_field)
    if document.get("version") != expected_version:
        raise IntelligenceV2PublicationError("v2 artifact version substitution")
    _authority_closed(document)
    expected_content = content_ref(document, identity_field=identity_field)
    observed_content = {
        key: reference.get(key)
        for key in ("artifact_id", "artifact_version", "byte_sha256", "semantic_sha256")
    }
    if validate_content_ref(observed_content, label="v2 exact ref") != expected_content:
        raise IntelligenceV2PublicationError("v2 exact ref does not match artifact bytes")
    if document.get("timestamp") != reference.get("cutoff"):
        raise IntelligenceV2PublicationError("v2 exact ref cutoff mismatch")
    return document, stored


def _validate_self_closed_nodes(nodes: Mapping[str, Mapping[str, Any]]) -> None:
    for key, validator in _SELF_VALIDATORS.items():
        try:
            if dict(validator(nodes[key])) != nodes[key]:
                raise IntelligenceV2PublicationError("v2 self-validation projection mismatch")
        except IntelligenceV2PublicationError:
            raise
        except Exception as exc:
            raise IntelligenceV2PublicationError(f"{key} replay validation failed") from exc


def _validate_market_risk_node(nodes: Mapping[str, Mapping[str, Any]]) -> None:
    try:
        projection = validate_market_risk_projection(
            nodes["MARKET_RISK_PROJECTION"],
            portfolio_policy=nodes["PORTFOLIO_POLICY"],
        )
    except Exception as exc:
        raise IntelligenceV2PublicationError("market risk projection replay failed") from exc
    if projection.get("status") != "AVAILABLE" or projection.get("blocker_codes") != []:
        raise IntelligenceV2PublicationError("market risk projection is not AVAILABLE")


def _validate_preactivation(document: Mapping[str, Any]) -> dict[str, Any]:
    closure = {
        "candidate_refs": document.get("candidate_refs"),
        "expected_pointer_sha256": document.get("expected_pointer_sha256"),
        "rollback_target_ref": document.get("rollback_target_ref"),
        "blocker_codes": document.get("blocker_codes"),
        "evaluated_at": document.get("timestamp"),
    }
    expected = build_preactivation_receipt(**closure)
    validated = validate_preactivation_receipt(document, **closure)
    if (
        validated != expected
        or validated.get("readiness") is not True
        or validated.get("status") != "READY"
        or validated.get("blocker_codes") != []
        or validated.get("write_performed") is not False
    ):
        raise IntelligenceV2PublicationError("preactivation receipt is not READY")
    return validated


def _read_publication_nodes(
    store: MainlineStore,
    closure: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for key in sorted(PUBLICATION_CLOSURE_VERSIONS):
        document, _ = _read_exact_v2(
            store,
            closure["nodes"][key],
            expected_version=PUBLICATION_CLOSURE_VERSIONS[key],
            identity_field=_IDENTITY_FIELDS[key],
        )
        nodes[key] = document
    _validate_self_closed_nodes(nodes)
    _validate_market_risk_node(nodes)
    nodes["PREACTIVATION"] = _validate_preactivation(nodes["PREACTIVATION"])
    return nodes


def _read_candidate_refs(
    store: MainlineStore,
    preactivation: Mapping[str, Any],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    documents: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    version_to_identity = {
        version: _IDENTITY_FIELDS[key] for key, version in PUBLICATION_CLOSURE_VERSIONS.items()
    }
    for reference in preactivation["candidate_refs"]:
        version = reference["artifact_version"]
        identity_field = version_to_identity.get(version)
        if identity_field is None:
            raise IntelligenceV2PublicationError("preactivation candidate version is unknown")
        document, _ = _read_exact_v2(
            store,
            reference,
            expected_version=version,
            identity_field=identity_field,
        )
        key = tuple(
            reference[field]
            for field in ("artifact_id", "artifact_version", "byte_sha256", "semantic_sha256")
        )
        if key in documents:
            raise IntelligenceV2PublicationError("preactivation candidate ref is duplicated")
        documents[key] = document
    return documents


def _content_key(reference: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(reference["artifact_id"]),
        str(reference["artifact_version"]),
        str(reference["byte_sha256"]),
        str(reference["semantic_sha256"]),
    )


def _validate_candidate_topology(
    nodes: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> None:
    portfolio = nodes["PORTFOLIO"]
    admitted = portfolio.get("admitted_decision_refs")
    if type(admitted) is not list or not admitted:
        raise IntelligenceV2PublicationError("portfolio has no admitted Decision v2 closure")
    for decision_ref in admitted:
        decision = candidates.get(_content_key(decision_ref))
        if decision is None or decision.get("state") != "PAPER_CANDIDATE":
            raise IntelligenceV2PublicationError("portfolio decision is not replay-closed")
        graph = candidates.get(_content_key(decision.get("graph_ref", {})))
        if graph is None or graph.get("company_code") != decision.get("company_code"):
            raise IntelligenceV2PublicationError("Decision v2 graph closure is missing")
    primary_decision = content_ref(nodes["DECISION_V2"], identity_field="decision_id")
    primary_graph = content_ref(nodes["EVIDENCE_GRAPH_V2"], identity_field="graph_id")
    if _content_key(primary_decision) not in candidates:
        raise IntelligenceV2PublicationError("primary Decision v2 is not preactivated")
    if _content_key(primary_graph) not in candidates:
        raise IntelligenceV2PublicationError("primary Evidence Graph is not preactivated")


def _legacy_exact_match(
    reference: Mapping[str, Any],
    *,
    stored: StoredBytes,
    document: Mapping[str, Any],
    version: str,
) -> None:
    if (
        reference.get("artifact_version") != version
        or reference.get("relative_path") != stored.relative_path
        or reference.get("byte_sha256") != stored.byte_sha256
        or reference.get("semantic_sha256") != document.get("semantic_sha256")
    ):
        raise IntelligenceV2PublicationError("legacy exact ref mismatch")


def _marker_validation_closure(
    marker: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "profile": profile,
        "transaction_id": marker.get("transaction_id"),
        "legacy_run_ref": marker.get("legacy_run_ref"),
        "target_pointer_ref": marker.get("target_pointer_ref"),
        "portfolio_ref": marker.get("portfolio_ref"),
        "risk_ref": marker.get("risk_ref"),
        "graduation_ref": marker.get("graduation_ref"),
        "built_at": marker.get("timestamp"),
    }


def _validate_legacy_portfolio_projection(
    legacy: Mapping[str, Any],
    v2_portfolio: Mapping[str, Any],
) -> None:
    final = v2_portfolio.get("final_portfolio")
    if type(final) is not dict or final.get("status") != "COMPLETE":
        raise IntelligenceV2PublicationError("v2 final portfolio is not complete")
    v2_targets = final.get("targets")
    legacy_targets = legacy.get("targets")
    if type(v2_targets) is not list or type(legacy_targets) is not list:
        raise IntelligenceV2PublicationError("portfolio target projection is invalid")
    projected = {
        row.get("company_code"): Decimal(str(row.get("final_weight"))) for row in v2_targets
    }
    published = {row.get("symbol"): Decimal(str(row.get("final_target"))) for row in legacy_targets}
    if (
        projected != published
        or Decimal(str(final.get("cash_weight"))) != Decimal(str(legacy.get("cash_weight")))
        or Decimal(str(final.get("gross_weight"))) != Decimal(str(legacy.get("gross_weight")))
    ):
        raise IntelligenceV2PublicationError("legacy portfolio differs from v2 final portfolio")


def _read_downstream_sidecar(
    store: MainlineStore,
    *,
    closure: Mapping[str, Any],
    run: Mapping[str, Any],
    run_bytes: StoredBytes,
    pointer: Mapping[str, Any],
    pointer_bytes: StoredBytes,
    portfolio: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    closure_path = PurePosixPath(str(closure["closure_path"]))
    sidecar_path = str(closure_path.with_name("activation-sidecar.json"))
    sidecar_stored = store.read(sidecar_path)
    sidecar = validate_seal(_parse_v2(sidecar_stored), identity_field="sidecar_id")
    if sidecar.get("version") != PUBLICATION_SIDECAR_VERSION:
        raise IntelligenceV2PublicationError("activation sidecar version mismatch")
    _authority_closed(sidecar)
    marker_ref = sidecar.get("marker_ref")
    if type(marker_ref) is not dict:
        raise IntelligenceV2PublicationError("activation sidecar marker ref is missing")
    marker_document, _ = _read_exact_v2(
        store,
        marker_ref,
        expected_version="myquant.v17.intelligence-v2.legacy-marker.v1",
        identity_field="marker_id",
    )
    marker_closure = _marker_validation_closure(
        marker_document,
        nodes["LEGACY_MARKER_PROFILE"],
    )
    validate_legacy_marker(marker_document, **marker_closure)
    validate_activation_sidecar(
        sidecar,
        marker=marker_document,
        marker_validation_closure=marker_closure,
        publication_closure=closure,
        built_at=sidecar.get("timestamp"),
    )
    _legacy_exact_match(
        sidecar["legacy_run_ref"],
        stored=run_bytes,
        document=run,
        version=MAINLINE_RUN_SCHEMA_ID,
    )
    _legacy_exact_match(
        sidecar["target_pointer_ref"],
        stored=pointer_bytes,
        document=pointer,
        version=ACTIVE_POINTER_SCHEMA_ID,
    )
    portfolio_ref = content_ref(nodes["PORTFOLIO"], identity_field="receipt_id")
    graph_ref = content_ref(nodes["EVIDENCE_GRAPH_V2"], identity_field="graph_id")
    graduation_ref = content_ref(nodes["GRADUATION"], identity_field="graduation_id")
    if (
        marker_document["portfolio_ref"] != portfolio_ref
        or marker_document["risk_ref"] != graph_ref
        or marker_document["graduation_ref"] != graduation_ref
        or marker_document["canonical_strategy_id"] != run["canonical_strategy_id"]
    ):
        raise IntelligenceV2PublicationError("legacy marker topology mismatch")
    _validate_legacy_portfolio_projection(portfolio, nodes["PORTFOLIO"])
    return sidecar, marker_document


def _validate_activation_permit(
    store: MainlineStore,
    *,
    sidecar: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    pointer: Mapping[str, Any],
    pointer_bytes: StoredBytes,
    run: Mapping[str, Any],
) -> None:
    preactivation = nodes["PREACTIVATION"]
    paths = derive_publication_paths(
        strategy_id=run["canonical_strategy_id"],
        transaction_id=sidecar["transaction_id"],
        target_pointer_sha256=pointer_bytes.byte_sha256,
        run_id=run["run_id"],
    )
    permit_stored = store.read(paths["activation_permit"])
    permit = _parse_v2(permit_stored)
    sidecar_ref = content_ref(sidecar, identity_field="sidecar_id")
    validate_action_permit(
        permit,
        owner_policy=nodes["PUBLICATION_OWNER_POLICY"],
        expected_action="ACTIVATE",
        expected_subject_ref=sidecar_ref,
        expected_strategy_id=run["canonical_strategy_id"],
        expected_pointer_sha256=preactivation["expected_pointer_sha256"],
        target_pointer_sha256=pointer_bytes.byte_sha256,
        verified_at=pointer["updated_at"],
    )


def _check_quarantine(
    store: MainlineStore,
    *,
    sidecar: Mapping[str, Any],
    pointer_bytes: StoredBytes,
    run: Mapping[str, Any],
) -> None:
    paths = derive_publication_paths(
        strategy_id=run["canonical_strategy_id"],
        transaction_id=sidecar["transaction_id"],
        target_pointer_sha256=pointer_bytes.byte_sha256,
        run_id=run["run_id"],
    )
    try:
        stored = store.read(paths["quarantine_sidecar"])
    except MainlineNotFound:
        return
    document = validate_seal(_parse_v2(stored), identity_field="quarantine_id")
    if document.get("version") != QUARANTINE_VERSION:
        raise IntelligenceV2PublicationError("quarantine marker version mismatch")
    validate_quarantine_receipt(
        document,
        sidecar_ref=content_ref(sidecar, identity_field="sidecar_id"),
        permit_ref=document.get("permit_ref"),
        canonical_strategy_id=run["canonical_strategy_id"],
        transaction_id=sidecar["transaction_id"],
        target_pointer_sha256=pointer_bytes.byte_sha256,
        run_id=run["run_id"],
        reason_codes=document.get("reason_codes"),
        quarantined_at=document.get("timestamp"),
    )
    raise ActiveRunQuarantined("active v2 run is quarantined")


def validate_marked_publication(
    store: MainlineStore,
    *,
    formal: Mapping[str, Any],
    run: Mapping[str, Any],
    run_bytes: StoredBytes,
    pointer: Mapping[str, Any],
    pointer_bytes: StoredBytes,
    portfolio: Mapping[str, Any],
) -> None:
    """Validate the complete reader-visible marked-run publication boundary."""

    if formal.get("publication_profile") != PUBLICATION_PROFILE:
        raise IntelligenceV2PublicationError("publication profile is invalid")
    evidence_refs = formal.get("evidence_refs")
    if type(evidence_refs) is not list:
        raise IntelligenceV2PublicationError("formal evidence refs are invalid")
    matching = [
        value
        for value in evidence_refs
        if type(value) is dict and value.get("schema_id") == PUBLICATION_CLOSURE_VERSION
    ]
    if len(matching) != 1:
        raise IntelligenceV2PublicationError("formal requires exactly one v2 closure ref")
    closure_ref = validate_ref(
        matching[0],
        label="publication_closure_ref",
        expected_schema_id=PUBLICATION_CLOSURE_VERSION,
        required_prefix=INTELLIGENCE_V2_ROOT,
    )
    closure_stored = store.read(closure_ref["relative_path"], closure_ref["byte_sha256"])
    closure = validate_publication_closure(_parse_v2(closure_stored))
    if (
        closure["closure_path"] != closure_ref["relative_path"]
        or closure["canonical_strategy_id"] != run["canonical_strategy_id"]
        or byte_sha256(closure_stored.data) != closure_ref["byte_sha256"]
    ):
        raise IntelligenceV2PublicationError("formal v2 closure ref mismatch")
    nodes = _read_publication_nodes(store, closure)
    candidates = _read_candidate_refs(store, nodes["PREACTIVATION"])
    _validate_candidate_topology(nodes, candidates)
    sidecar, _ = _read_downstream_sidecar(
        store,
        closure=closure,
        run=run,
        run_bytes=run_bytes,
        pointer=pointer,
        pointer_bytes=pointer_bytes,
        portfolio=portfolio,
        nodes=nodes,
    )
    _validate_activation_permit(
        store,
        sidecar=sidecar,
        nodes=nodes,
        pointer=pointer,
        pointer_bytes=pointer_bytes,
        run=run,
    )
    _check_quarantine(
        store,
        sidecar=sidecar,
        pointer_bytes=pointer_bytes,
        run=run,
    )


__all__ = [
    "ActiveRunQuarantined",
    "IntelligenceV2PublicationError",
    "validate_marked_publication",
]

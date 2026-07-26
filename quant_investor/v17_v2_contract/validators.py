"""Pure cross-document validators for the isolated v17 protocol-v2 contracts.

These helpers validate in-memory values only.  They never discover files,
create directories, acquire locks, import :mod:`quant_investor.v17`, or repair
state.  Callers must read bytes before entering this layer and must not write
unless validation has completed successfully.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import re
from typing import Any, Final, NoReturn

from .canonical import (
    CanonicalContractError,
    canonical_json_bytes,
    canonical_resource_bytes,
    load_canonical_resource,
    seal_semantic as _seal_semantic,
    semantic_sha256 as _semantic_sha256,
    stored_byte_sha256,
    typed_scalar_total_order_key,
    validate_semantic_seal as _validate_semantic_seal,
    validate_json_limits,
)
from .identities import (
    IdentityContractError,
    require_opaque_id,
    require_security_code,
    require_sha256,
)
from .limits import (
    LIMITS,
    ContractLimitError,
    checked_add,
    require_nonnegative_int,
)
from .resources import (
    PACKAGE_ASSET_SHA256S,
    PackageResourceError,
    expected_ledger_contract_bindings,
    expected_ledger_implementation_bindings,
    load_packaged_json,
)
from .schema_validation import (
    SchemaValidationError,
    schema_path_for_version,
    validate_mapping_against_packaged_schema,
)

PROTOCOL_VERSION: Final = "myquant.v17.v2"
SEMANTIC_SHA_FIELD: Final = "semantic_sha256"
SOURCE_ROLE_MATRIX_VERSION: Final = "myquant.v17.v2.source-role-matrix.v1"
DATASET_RECORD_SCHEMA_REGISTRY_VERSION: Final = (
    "myquant.v17.v2.dataset-record-schema-registry.v1"
)
DATASET_MANIFEST_VERSION: Final = "myquant.v17.v2.dataset-manifest.v1"
SOURCE_MANIFEST_VERSION: Final = "myquant.v17.v2.source-manifest.v1"
GENERATION_CATALOG_VERSION: Final = "myquant.v17.v2.generation-catalog.v1"
OBSERVATION_DISPOSITION_VERSION: Final = "myquant.v17.v2.observation-disposition.v1"
DATASET_SUMMARY_VERSION: Final = "myquant.v17.v2.dataset-summary.v1"
DATASET_SCHEMA_DIGEST_VERSION: Final = "myquant.v17.v2.dataset-schema-digest.v1"
SOURCE_BINDING_SET_VERSION: Final = "myquant.v17.v2.source-binding-set.v1"
SOURCE_LOCATOR_VERSION: Final = "myquant.v17.v2.source-locator.v1"
DEEP_RESEARCH_REQUEST_VERSION: Final = "myquant.v17.v2.deep-research-request.v1"
DEEP_RESEARCH_RESPONSE_VERSION: Final = "myquant.v17.v2.deep-research-response.v1"
DEEP_RESEARCH_REPORT_VERSION: Final = "myquant.v17.v2.deep-research-report.v1"
SHADOW_LEDGER_VERSION: Final = "myquant.v17.v2.shadow-ledger.v1"
SHADOW_OUTPUT_VERSION: Final = "myquant.v17.v2.shadow-output.v1"
SHADOW_LATEST_POINTER_VERSION: Final = "myquant.v17.v2.shadow-latest-pointer.v1"
ACTION_FAILURE_RECEIPT_VERSION: Final = "myquant.v17.v2.action-failure-receipt.v1"
MARKET_POINTER_VERSION: Final = "myquant.v17.v2.market-pointer.v1"
MARKET_SNAPSHOT_MANIFEST_VERSION: Final = "myquant.v17.v2.market-snapshot-manifest.v1"
RISK_POLICY_SNAPSHOT_VERSION: Final = "myquant.v17.v2.risk-policy-snapshot.v1"
PORTFOLIO_REQUIRED_INPUTS_VERSION: Final = "myquant.v17.v2.portfolio-required-inputs.v1"
MACRO_OVERLAY_VERSION: Final = "myquant.v17.v2.macro-overlay.v1"
MARKOV_OVERLAY_VERSION: Final = "myquant.v17.v2.markov-overlay.v1"
RANK_OUTPUT_VERSION: Final = "myquant.v17.v2.rank-output.v1"
PORTFOLIO_OUTPUT_VERSION: Final = "myquant.v17.v2.portfolio-output.v1"


class SourceAdmissionDisposition(str, Enum):
    """Closed runtime outcomes after exact source-registry admission."""

    ADMITTED = "ADMITTED"
    SHADOW_RANK_COMPLETE_NO_PORTFOLIO = "SHADOW_RANK_COMPLETE_NO_PORTFOLIO"


@dataclass(frozen=True)
class SourceAdmissionOutcome:
    """Typed, immutable result of runtime source admission."""

    disposition: SourceAdmissionDisposition
    locator: Mapping[str, Any]
    locator_byte_sha256: str
    input_bindings: tuple[tuple[str, str, str, str, str, str], ...]
    unavailable_required_roles: tuple[str, ...]


SUPPORTED_DOCUMENT_VERSIONS: Final = frozenset(
    {
        DATASET_MANIFEST_VERSION,
        SOURCE_MANIFEST_VERSION,
        GENERATION_CATALOG_VERSION,
        OBSERVATION_DISPOSITION_VERSION,
        DATASET_SUMMARY_VERSION,
        SOURCE_BINDING_SET_VERSION,
        SOURCE_LOCATOR_VERSION,
        DEEP_RESEARCH_REQUEST_VERSION,
        DEEP_RESEARCH_RESPONSE_VERSION,
        DEEP_RESEARCH_REPORT_VERSION,
        SHADOW_LEDGER_VERSION,
        SHADOW_OUTPUT_VERSION,
        SHADOW_LATEST_POINTER_VERSION,
        ACTION_FAILURE_RECEIPT_VERSION,
        MARKET_POINTER_VERSION,
        MARKET_SNAPSHOT_MANIFEST_VERSION,
        RISK_POLICY_SNAPSHOT_VERSION,
        PORTFOLIO_REQUIRED_INPUTS_VERSION,
        MACRO_OVERLAY_VERSION,
        MARKOV_OVERLAY_VERSION,
        RANK_OUTPUT_VERSION,
        PORTFOLIO_OUTPUT_VERSION,
    }
)

_DOCUMENT_ID_FIELDS: Final = {
    DATASET_MANIFEST_VERSION: "dataset_id",
    SOURCE_MANIFEST_VERSION: "manifest_id",
    GENERATION_CATALOG_VERSION: "catalog_id",
    OBSERVATION_DISPOSITION_VERSION: "disposition_id",
    DATASET_SUMMARY_VERSION: "summary_id",
    SOURCE_BINDING_SET_VERSION: "binding_set_id",
    SOURCE_LOCATOR_VERSION: "locator_id",
    DEEP_RESEARCH_REQUEST_VERSION: "request_id",
    DEEP_RESEARCH_RESPONSE_VERSION: "response_id",
    DEEP_RESEARCH_REPORT_VERSION: "report_id",
    SHADOW_LEDGER_VERSION: "run_id",
    SHADOW_OUTPUT_VERSION: "run_id",
    SHADOW_LATEST_POINTER_VERSION: "run_id",
    ACTION_FAILURE_RECEIPT_VERSION: "receipt_id",
    MARKET_POINTER_VERSION: "pointer_id",
    MARKET_SNAPSHOT_MANIFEST_VERSION: "manifest_id",
    RISK_POLICY_SNAPSHOT_VERSION: "policy_id",
    PORTFOLIO_REQUIRED_INPUTS_VERSION: "input_id",
    MACRO_OVERLAY_VERSION: "overlay_id",
    MARKOV_OVERLAY_VERSION: "overlay_id",
    RANK_OUTPUT_VERSION: "output_id",
    PORTFOLIO_OUTPUT_VERSION: "output_id",
}

_DATASET_CATALOG_ROLE: Final = {
    "H00300_total_return_dataset": "pit_generation_catalog",
    "cn_open_day_calendar_dataset": "pit_generation_catalog",
    "corporate_actions_dataset": "pit_generation_catalog",
    "deep_evidence_dataset": "fundamental_generation_catalog",
    "fundamental_raw_tables_dataset": "fundamental_generation_catalog",
    "market_bars_dataset": "pit_generation_catalog",
    "official_delisting_cash_dataset": "pit_generation_catalog",
}
_PHASE1_ROLE_DECLARATIONS: Final = {
    "H00300_total_return_dataset": (
        "RANK",
        "DATASET",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.dataset-manifest.schema.v1",
    ),
    "cn_open_day_calendar_dataset": (
        "RANK",
        "DATASET",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.dataset-manifest.schema.v1",
    ),
    "corporate_actions_dataset": (
        "RANK",
        "DATASET",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.dataset-manifest.schema.v1",
    ),
    "deep_evidence_dataset": (
        "RANK",
        "DATASET",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.dataset-manifest.schema.v1",
    ),
    "fundamental_generation_catalog": (
        "RANK",
        "OBJECT",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.generation-catalog.schema.v1",
    ),
    "fundamental_raw_tables_dataset": (
        "RANK",
        "DATASET",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.dataset-manifest.schema.v1",
    ),
    "macro_overlay": (
        "PORTFOLIO",
        "OBJECT",
        False,
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "myquant.v17.v2.macro-overlay.schema.v1",
    ),
    "market_bars_dataset": (
        "RANK",
        "DATASET",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.dataset-manifest.schema.v1",
    ),
    "market_pointer": (
        "RANK",
        "OBJECT",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.market-pointer.schema.v1",
    ),
    "market_snapshot_manifest": (
        "RANK",
        "OBJECT",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.market-snapshot-manifest.schema.v1",
    ),
    "markov_overlay": (
        "PORTFOLIO",
        "OBJECT",
        False,
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "myquant.v17.v2.markov-overlay.schema.v1",
    ),
    "official_delisting_cash_dataset": (
        "RANK",
        "DATASET",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.dataset-manifest.schema.v1",
    ),
    "pit_generation_catalog": (
        "RANK",
        "OBJECT",
        True,
        "REJECT_BEFORE_INITIALIZED_ZERO_WRITE",
        "myquant.v17.v2.generation-catalog.schema.v1",
    ),
    "portfolio_required_inputs": (
        "PORTFOLIO",
        "OBJECT",
        True,
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "myquant.v17.v2.portfolio-required-inputs.schema.v1",
    ),
    "risk_policy_snapshot": (
        "PORTFOLIO",
        "OBJECT",
        True,
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "myquant.v17.v2.risk-policy-snapshot.schema.v1",
    ),
}

_ARTIFACT_REF_KEYS: Final = frozenset(
    {
        "artifact_id",
        "artifact_version",
        "relative_path",
        "byte_sha256",
        "semantic_sha256",
    }
)
_ROLE_MATRIX_REF_KEYS: Final = frozenset({"resource_name", "resource_version", "byte_sha256"})
_GENERATION_CATALOG_KEYS: Final = frozenset(
    {
        "protocol_version",
        "version",
        "catalog_id",
        "generation_id",
        "role",
        "phase",
        "market",
        "cutoff",
        "created_at",
        "source_manifest_ref",
        "table_ordering",
        "tables",
        "authority",
        "semantic_sha256",
    }
)
_GENERATION_CATALOG_TABLE_KEYS: Final = frozenset(
    {
        "stage",
        "role",
        "table_id",
        "dataset_manifest_ref",
        "summary_ref",
        "record_schema_id",
        "primary_key",
        "valid_time_field",
        "available_time_field",
        "selection_policy",
        "conflict_policy",
    }
)
_SOURCE_BINDING_SET_KEYS: Final = frozenset(
    {
        "protocol_version",
        "version",
        "binding_set_id",
        "market",
        "cutoff",
        "source_manifest_ref",
        "binding_ordering",
        "bindings",
        "authority",
        "semantic_sha256",
    }
)
_SOURCE_BINDING_KEYS: Final = frozenset(
    {
        "stage",
        "role",
        "catalog_ref",
        "summary_ref",
        "dataset_manifest_ref",
        "disposition_id",
        "observation_disposition_ref",
    }
)

_SIGNAL_KEYS: Final = frozenset(
    {
        "financial",
        "business_model",
        "industry",
        "competitiveness",
        "management",
        "valuation",
    }
)
_SIGNAL_VALUES: Final = frozenset({-1.0, -0.5, 0.0, 0.5, 1.0})
_COVERAGE_ORDER: Final = (
    "financial_reports_and_three_statement_reconciliation",
    "normalization_and_reversible_adjustments",
    "segments",
    "management_and_governance",
    "ownership",
    "industry_and_competition",
    "products_and_technology",
    "dcf",
    "reverse_dcf",
    "comparable_companies",
    "sotp_if_applicable",
    "bull_base_bear_scenarios",
    "catalysts",
    "counterevidence",
    "falsification_conditions",
    "continuous_monitoring_items",
)
_LAYER_ORDER: Final = (
    "raw_facts",
    "derived_metrics",
    "research_inferences",
    "investment_judgments",
    "risk_alerts",
)
_RED_FLAG_ORDER: Final = (
    "audit_or_going_concern",
    "restatement_or_three_statement_failure",
    "fraud_or_material_penalty",
    "controller_appropriation_or_pledge_crisis",
    "material_related_party_or_governance_conflict",
    "liquidity_or_refinancing_break",
    "customer_or_supplier_concentration_break",
    "product_or_technology_obsolescence",
    "listing_or_delisting_risk",
    "core_thesis_falsified",
)
_TERMINAL_STATES: Final = frozenset(
    {
        "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        "SHADOW_PORTFOLIO_INFEASIBLE",
        "HARD_STOP_SNAPSHOT_DRIFT",
        "HARD_STOP_INVALID_EVIDENCE",
    }
)
_ALL_STATES: Final = frozenset(
    {
        "PREPARED",
        "DETERMINISTIC_COMPLETE",
        "DEEP_REQUEST_READY",
        "DEEP_RESPONSE_RECEIVED",
        "PORTFOLIO_COMPLETE",
        *_TERMINAL_STATES,
    }
)
_TRANSITIONS: Final = {
    "PREPARED": frozenset(
        {
            "DETERMINISTIC_COMPLETE",
            "HARD_STOP_SNAPSHOT_DRIFT",
            "HARD_STOP_INVALID_EVIDENCE",
        }
    ),
    "DETERMINISTIC_COMPLETE": frozenset(
        {
            "DEEP_REQUEST_READY",
            "HARD_STOP_SNAPSHOT_DRIFT",
            "HARD_STOP_INVALID_EVIDENCE",
        }
    ),
    "DEEP_REQUEST_READY": frozenset(
        {
            "DEEP_RESPONSE_RECEIVED",
            "HARD_STOP_SNAPSHOT_DRIFT",
            "HARD_STOP_INVALID_EVIDENCE",
        }
    ),
    "DEEP_RESPONSE_RECEIVED": frozenset(
        {
            "PORTFOLIO_COMPLETE",
            "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
            "SHADOW_PORTFOLIO_INFEASIBLE",
            "HARD_STOP_SNAPSHOT_DRIFT",
            "HARD_STOP_INVALID_EVIDENCE",
        }
    ),
    "PORTFOLIO_COMPLETE": frozenset(
        {
            "SHADOW_COMPLETE_AWAITING_HUMAN_DECISION",
            "HARD_STOP_SNAPSHOT_DRIFT",
            "HARD_STOP_INVALID_EVIDENCE",
        }
    ),
}
_ACTIONS: Final = frozenset(
    {
        "SOURCE_MAINTAIN",
        "RISK_POLICY_SEAL",
        "SHADOW_PREPARE",
        "SHADOW_RECEIVE",
        "SHADOW_FINALIZE",
        "READ_STATUS",
        "READ_ARTIFACT",
        "REPAIR_LATEST",
    }
)
_CHECKPOINTS: Final = frozenset({"PRE_IMPORT", "ACCEPTED", "INITIALIZED"})
_PATH_COMPONENT: Final = r"[a-z0-9][a-z0-9_.-]{0,127}"
_NESTED_JSON_PATH: Final = rf"(?:{_PATH_COMPONENT}/)*{_PATH_COMPONENT}\.json"
_PROTOCOL_PATH_RE: Final = re.compile(
    rf"^(?:"
    rf"data/private/v17_sources/protocol-v2/"
    rf"(?:objects/[0-9a-f]{{2}}/[0-9a-f]{{64}}\.(?:json|parquet|blob)"
    rf"|manifests/{_PATH_COMPONENT}\.json"
    rf"|locators/{_PATH_COMPONENT}\.json)"
    rf"|results/v17_shadow/protocol-v2/"
    rf"(?:runs/{_PATH_COMPONENT}/{_NESTED_JSON_PATH}"
    rf"|models/objects/[0-9a-f]{{2}}/[0-9a-f]{{64}}\.json"
    rf"|outcomes/{_PATH_COMPONENT}\.json"
    rf"|_latest/shadow\.json)"
    rf")$",
    re.ASCII,
)
_ARTIFACT_VERSION_RE: Final = re.compile(
    r"^myquant\.v17\.v2\.[a-z0-9-]+\.v1$",
    re.ASCII,
)
_PACKAGE_RESOURCE_PATH_RE: Final = re.compile(
    rf"^resources/{_PATH_COMPONENT}\.json$",
    re.ASCII,
)
_PACKAGE_SCHEMA_PATH_RE: Final = re.compile(
    rf"^schemas/{_PATH_COMPONENT}\.schema\.json$",
    re.ASCII,
)
_PACKAGE_MODULE_PATH_RE: Final = re.compile(
    r"^(?:[a-z0-9_]+/)*[a-z0-9_]+\.py$",
    re.ASCII,
)


class V17V2ValidationError(ValueError):
    """Raised when an active protocol-v2 document set is not self-consistent."""

    exit_code = 2


def _fail(message: str) -> NoReturn:
    raise V17V2ValidationError(message)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an object")
    return value


def _array(value: Any, *, label: str, maximum: int | None = None) -> list[Any]:
    if type(value) is not list:
        _fail(f"{label} must be an array")
    if maximum is not None and len(value) > maximum:
        _fail(f"{label} exceeds the inclusive maximum {maximum}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        _fail(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be a nonempty string")
    return value


def _rfc3339_instant(value: Any, *, label: str) -> datetime:
    timestamp = _string(value, label=label)
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise V17V2ValidationError(f"{label} is not a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        _fail(f"{label} must include an RFC3339 offset")
    return parsed.astimezone(timezone.utc)


def _opaque_id(value: Any, *, label: str) -> str:
    try:
        return require_opaque_id(value, label=label)
    except IdentityContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc


def _sha256_or_empty(value: Any, *, label: str) -> str:
    if value == "EMPTY":
        return "EMPTY"
    try:
        return require_sha256(value, label=label)
    except IdentityContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc


def _reject_forbidden_role(
    role: Any,
    *,
    suffixes: Sequence[str] = ("_verification_receipt",),
    label: str,
) -> str:
    value = _string(role, label=label)
    folded = value.lower()
    for suffix in suffixes:
        if folded.endswith(suffix.lower()):
            _fail(f"{label} uses a forbidden role suffix: {suffix}")
    return value


def _require_protocol_path(value: Any, *, label: str) -> str:
    path = _string(value, label=label)
    if (
        path.startswith("/")
        or "\\" in path
        or "//" in path
        or path.endswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or _PROTOCOL_PATH_RE.fullmatch(path) is None
    ):
        _fail(f"{label} is not a canonical protocol-v2 relative path")
    return path


def _document_id(document: Mapping[str, Any]) -> str:
    version = document.get("version")
    field = _DOCUMENT_ID_FIELDS.get(version)
    if field is None:
        _fail(f"unsupported document version: {version!r}")
    try:
        return require_opaque_id(document.get(field), label=field)
    except IdentityContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc


def semantic_sha256(document: Mapping[str, Any]) -> str:
    """Hash a document after deleting only its root semantic seal.

    Nested ``semantic_sha256`` values remain part of the digest by design.
    """

    try:
        return _semantic_sha256(dict(_mapping(document, label="semantic document")))
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc


def seal_semantic(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive sealed copy and reject an already-present root seal."""

    try:
        return _seal_semantic(dict(_mapping(document, label="semantic document")))
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc


def validate_semantic_seal(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the root semantic seal without removing nested digest fields."""

    try:
        return _validate_semantic_seal(dict(_mapping(document, label="semantic document")))
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc


def document_byte_sha256(document: Mapping[str, Any]) -> str:
    """Return the SHA-256 of canonical compact JSON plus exactly one newline."""

    try:
        return stored_byte_sha256(dict(document))
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc


def validate_document_identity(
    document: Mapping[str, Any],
    *,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Validate packaged schema, semantic seal, and the identity envelope."""

    candidate = dict(_mapping(document, label="contract document"))
    version = candidate.get("version")
    if version not in SUPPORTED_DOCUMENT_VERSIONS:
        _fail(f"unsupported document version: {version!r}")
    if expected_version is not None and version != expected_version:
        _fail(f"document version mismatch: expected {expected_version}")
    try:
        validate_mapping_against_packaged_schema(
            candidate,
            expected_version=str(version),
        )
    except (PackageResourceError, SchemaValidationError) as exc:
        raise V17V2ValidationError(str(exc)) from exc
    payload = validate_semantic_seal(candidate)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        _fail("protocol_version mismatch")
    if payload.get("authority") is not False:
        _fail("authority must be false")
    _document_id(payload)
    try:
        validate_json_limits(payload)
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    return payload


def _raw_bytes(value: bytes | Mapping[str, Any], *, label: str) -> bytes:
    if type(value) is bytes:
        return value
    if type(value) is dict:
        try:
            return canonical_resource_bytes(value)
        except CanonicalContractError as exc:
            raise V17V2ValidationError(f"{label}: {exc}") from exc
    _fail(f"{label} must be bytes or an object")


def _require_content_addressed_path(
    path: str,
    *,
    byte_sha256: str,
    label: str,
) -> None:
    private_prefix = "data/private/v17_sources/protocol-v2/objects/"
    model_prefix = "results/v17_shadow/protocol-v2/models/objects/"
    if path.startswith(private_prefix):
        extension = path.rsplit(".", 1)[-1]
        expected = f"{private_prefix}{byte_sha256[:2]}/{byte_sha256}.{extension}"
        if path != expected:
            _fail(f"{label}.relative_path/content digest mismatch")
    elif path.startswith(model_prefix):
        expected = f"{model_prefix}{byte_sha256[:2]}/{byte_sha256}.json"
        if path != expected:
            _fail(f"{label}.relative_path/content digest mismatch")


def _validate_artifact_ref(
    reference: Mapping[str, Any],
    *,
    document: Mapping[str, Any] | None = None,
    raw: bytes | Mapping[str, Any] | None = None,
    expected_path: str | None = None,
    expected_version: str | None = None,
    label: str,
) -> dict[str, Any]:
    ref = dict(_mapping(reference, label=label))
    _exact_keys(ref, _ARTIFACT_REF_KEYS, label=label)
    try:
        require_opaque_id(ref.get("artifact_id"), label=f"{label}.artifact_id")
        require_sha256(ref.get("byte_sha256"), label=f"{label}.byte_sha256")
        require_sha256(
            ref.get("semantic_sha256"),
            label=f"{label}.semantic_sha256",
        )
    except IdentityContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    path = _string(ref.get("relative_path"), label=f"{label}.relative_path")
    path = _require_protocol_path(path, label=f"{label}.relative_path")
    artifact_version = ref.get("artifact_version")
    if (
        type(artifact_version) is not str
        or _ARTIFACT_VERSION_RE.fullmatch(artifact_version) is None
    ):
        _fail(f"{label}.artifact_version is not canonical")
    if expected_path is not None and path != expected_path:
        _fail(f"{label}.relative_path mismatch")
    if expected_version is not None and ref.get("artifact_version") != expected_version:
        _fail(f"{label}.artifact_version mismatch")
    if document is None and raw is None:
        _fail(f"{label} has no bound artifact")
    if document is not None and raw is not None:
        _fail(f"{label} has ambiguous bound artifact")
    if document is not None:
        bound = validate_document_identity(document, expected_version=expected_version)
        if ref["artifact_id"] != _document_id(bound):
            _fail(f"{label}.artifact_id mismatch")
        observed_bytes = canonical_resource_bytes(bound)
        if ref["semantic_sha256"] != bound["semantic_sha256"]:
            _fail(f"{label}.semantic_sha256 mismatch")
    else:
        observed_bytes = _raw_bytes(raw, label=label)  # type: ignore[arg-type]
    if hashlib.sha256(observed_bytes).hexdigest() != ref["byte_sha256"]:
        _fail(f"{label}.byte_sha256 mismatch")
    _require_content_addressed_path(
        path,
        byte_sha256=str(ref["byte_sha256"]),
        label=label,
    )
    return ref


def _validate_unresolved_artifact_ref(
    reference: Mapping[str, Any],
    *,
    expected_version: str | None = None,
    label: str,
) -> dict[str, Any]:
    ref = dict(_mapping(reference, label=label))
    _exact_keys(ref, _ARTIFACT_REF_KEYS, label=label)
    try:
        require_opaque_id(ref.get("artifact_id"), label=f"{label}.artifact_id")
        require_sha256(ref.get("byte_sha256"), label=f"{label}.byte_sha256")
        require_sha256(
            ref.get("semantic_sha256"),
            label=f"{label}.semantic_sha256",
        )
    except IdentityContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    if expected_version is not None and ref.get("artifact_version") != expected_version:
        _fail(f"{label}.artifact_version mismatch")
    artifact_version = ref.get("artifact_version")
    if (
        type(artifact_version) is not str
        or _ARTIFACT_VERSION_RE.fullmatch(artifact_version) is None
    ):
        _fail(f"{label}.artifact_version is not canonical")
    path = _require_protocol_path(
        ref.get("relative_path"),
        label=f"{label}.relative_path",
    )
    _require_content_addressed_path(
        path,
        byte_sha256=str(ref["byte_sha256"]),
        label=label,
    )
    return ref


def _reference_path(reference: Mapping[str, Any], *, label: str) -> str:
    return _require_protocol_path(
        reference.get("relative_path"),
        label=f"{label}.relative_path",
    )


def _resolve_document(
    reference: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    *,
    expected_version: str | None,
    label: str,
) -> Mapping[str, Any]:
    path = _reference_path(reference, label=label)
    document = documents.get(path)
    if document is None:
        _fail(f"{label} does not resolve: {path}")
    _validate_artifact_ref(
        reference,
        document=document,
        expected_path=path,
        expected_version=expected_version,
        label=label,
    )
    return document


def _binding_order_key(binding: Mapping[str, Any]) -> tuple[str, ...]:
    catalog_ref = _mapping(binding.get("catalog_ref"), label="binding.catalog_ref")
    summary_ref = _mapping(binding.get("summary_ref"), label="binding.summary_ref")
    dataset_ref = _mapping(
        binding.get("dataset_manifest_ref"),
        label="binding.dataset_manifest_ref",
    )
    return (
        _string(binding.get("stage"), label="binding.stage"),
        _string(binding.get("role"), label="binding.role"),
        _reference_path(catalog_ref, label="binding.catalog_ref"),
        _string(catalog_ref.get("byte_sha256"), label="binding.catalog_ref.byte_sha256"),
        _reference_path(summary_ref, label="binding.summary_ref"),
        _string(summary_ref.get("byte_sha256"), label="binding.summary_ref.byte_sha256"),
        _reference_path(dataset_ref, label="binding.dataset_manifest_ref"),
        _string(
            dataset_ref.get("byte_sha256"),
            label="binding.dataset_manifest_ref.byte_sha256",
        ),
        _string(binding.get("disposition_id"), label="binding.disposition_id"),
    )


def _dataset_content_set_sha256(shards: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "domain": "myquant.v17.v2.dataset-content-set.v1",
        "shards": list(shards),
    }
    try:
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc


def _dataset_schema_sha256(schema: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "version": DATASET_SCHEMA_DIGEST_VERSION,
        "schema": [dict(entry) for entry in schema],
    }
    try:
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc


def _validate_dataset_manifest_objects(
    document: Mapping[str, Any],
    *,
    source_objects: Mapping[str, bytes | Mapping[str, Any]],
    referenced_object_paths: set[str],
    label: str,
) -> None:
    _exact_keys(
        document,
        frozenset(
            {
                "protocol_version",
                "version",
                "dataset_id",
                "role",
                "format",
                "media_type",
                "schema",
                "primary_key",
                "partition_keys",
                "sort_keys",
                "shards",
                "total_row_count",
                "total_size_bytes",
                "content_set_sha256",
                "authority",
                "semantic_sha256",
            }
        ),
        label=label,
    )
    format_name = document.get("format")
    if format_name not in {"PARQUET", "BLOB"}:
        _fail(f"{label}.format invalid")
    schema_rows = _array(
        document.get("schema"),
        label=f"{label}.schema",
        maximum=4096,
    )
    primary_keys = _array(
        document.get("primary_key"),
        label=f"{label}.primary_key",
        maximum=64,
    )
    partition_keys = _array(
        document.get("partition_keys"),
        label=f"{label}.partition_keys",
        maximum=64,
    )
    sort_keys = _array(
        document.get("sort_keys"),
        label=f"{label}.sort_keys",
        maximum=64,
    )
    schema_nullable: dict[str, bool] = {}
    normalized_schema: list[Mapping[str, Any]] = []
    for index, raw_schema in enumerate(schema_rows):
        schema_entry = _mapping(raw_schema, label=f"{label}.schema[{index}]")
        _exact_keys(
            schema_entry,
            frozenset({"name", "logical_type", "nullable"}),
            label=f"{label}.schema[{index}]",
        )
        name = _opaque_id(
            schema_entry.get("name"),
            label=f"{label}.schema[{index}].name",
        )
        if name in schema_nullable:
            _fail(f"{label}.schema has duplicate field: {name}")
        if type(schema_entry.get("nullable")) is not bool:
            _fail(f"{label}.schema[{index}].nullable must be boolean")
        _string(
            schema_entry.get("logical_type"),
            label=f"{label}.schema[{index}].logical_type",
        )
        schema_nullable[name] = bool(schema_entry["nullable"])
        normalized_schema.append(dict(schema_entry))
    expected_schema_sha256 = _dataset_schema_sha256(normalized_schema)
    for key_set_name, key_values in (
        ("primary_key", primary_keys),
        ("partition_keys", partition_keys),
        ("sort_keys", sort_keys),
    ):
        normalized_keys = [
            _opaque_id(item, label=f"{label}.{key_set_name}[{index}]")
            for index, item in enumerate(key_values)
        ]
        if len(set(normalized_keys)) != len(normalized_keys):
            _fail(f"{label}.{key_set_name} contains duplicates")
        if any(item not in schema_nullable for item in normalized_keys):
            _fail(f"{label}.{key_set_name} references an unknown schema field")
    if format_name == "PARQUET":
        if (
            document.get("media_type") != "application/vnd.apache.parquet"
            or not schema_rows
            or not primary_keys
        ):
            _fail(f"{label} PARQUET shape invalid")
    elif (
        document.get("media_type") != "application/octet-stream"
        or schema_rows
        or primary_keys
        or partition_keys
        or sort_keys
        or document.get("total_row_count") != 0
    ):
        _fail(f"{label} BLOB shape invalid")

    shards = _array(
        document.get("shards"),
        label=f"{label}.shards",
        maximum=LIMITS["max_dataset_shards"],
    )
    if not shards:
        _fail(f"{label}.shards must be nonempty")
    total_bytes = 0
    total_rows = 0
    order: list[tuple[Any, ...]] = []
    ranges: list[
        tuple[
            tuple[tuple[int, Any, bytes], ...],
            tuple[tuple[int, Any, bytes], ...],
            tuple[tuple[int, Any, bytes], ...],
        ]
    ] = []
    normalized: list[Mapping[str, Any]] = []
    for index, value in enumerate(shards):
        shard = _mapping(value, label=f"{label}.shards[{index}]")
        _exact_keys(
            shard,
            frozenset(
                {
                    "logical_name",
                    "partition_values",
                    "object_path",
                    "byte_sha256",
                    "size_bytes",
                    "row_count",
                    "min_key",
                    "max_key",
                    "schema_sha256",
                }
            ),
            label=f"{label}.shards[{index}]",
        )
        logical_name = _string(
            shard.get("logical_name"),
            label=f"{label}.shards[{index}].logical_name",
        )
        if (
            logical_name.startswith("/")
            or logical_name.endswith("/")
            or "//" in logical_name
            or any(part in {"", ".", ".."} for part in logical_name.split("/"))
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/=-]*", logical_name) is None
        ):
            _fail(f"{label}.shards[{index}].logical_name invalid")
        object_path = _require_protocol_path(
            shard.get("object_path"),
            label=f"{label}.shards[{index}].object_path",
        )
        try:
            byte_sha = require_sha256(
                shard.get("byte_sha256"),
                label=f"{label}.shards[{index}].byte_sha256",
            )
            shard_schema_sha256 = require_sha256(
                shard.get("schema_sha256"),
                label=f"{label}.shards[{index}].schema_sha256",
            )
            size_bytes = require_nonnegative_int(
                shard.get("size_bytes"),
                label=f"{label}.shards[{index}].size_bytes",
                maximum=LIMITS["max_shard_bytes"],
            )
            row_count = require_nonnegative_int(
                shard.get("row_count"),
                label=f"{label}.shards[{index}].row_count",
                maximum=LIMITS["max_dataset_rows"],
            )
            total_bytes = checked_add(
                total_bytes,
                size_bytes,
                label=f"{label} aggregate bytes",
                maximum=LIMITS["max_dataset_bytes"],
            )
            total_rows = checked_add(
                total_rows,
                row_count,
                label=f"{label} aggregate rows",
                maximum=LIMITS["max_dataset_rows"],
            )
        except (IdentityContractError, ContractLimitError) as exc:
            raise V17V2ValidationError(str(exc)) from exc
        if size_bytes <= 0:
            _fail(f"{label}.shards[{index}].size_bytes must be positive")
        if shard_schema_sha256 != expected_schema_sha256:
            _fail(f"{label}.shards[{index}].schema_sha256 mismatch")
        if format_name == "PARQUET" and row_count <= 0:
            _fail(f"{label}.shards[{index}].row_count must be positive for PARQUET")
        if format_name == "BLOB" and row_count != 0:
            _fail(f"{label}.shards[{index}].row_count must be zero for BLOB")
        extension = "parquet" if format_name == "PARQUET" else "blob"
        expected_object_path = (
            "data/private/v17_sources/protocol-v2/objects/" f"{byte_sha[:2]}/{byte_sha}.{extension}"
        )
        if object_path != expected_object_path:
            _fail(f"{label}.shards[{index}].object_path/content mismatch")
        raw = source_objects.get(object_path)
        if raw is None:
            _fail(f"{label}.shards[{index}] object does not resolve")
        observed = _raw_bytes(raw, label=f"{label}.shards[{index}]")
        if len(observed) != size_bytes:
            _fail(f"{label}.shards[{index}].size_bytes mismatch")
        if hashlib.sha256(observed).hexdigest() != byte_sha:
            _fail(f"{label}.shards[{index}].byte_sha256 mismatch")
        referenced_object_paths.add(object_path)

        partition_values = _mapping(
            shard.get("partition_values"),
            label=f"{label}.shards[{index}].partition_values",
        )
        if set(partition_values) != set(partition_keys):
            _fail(f"{label}.shards[{index}].partition_values key mismatch")
        try:
            partition_order = tuple(
                typed_scalar_total_order_key(
                    partition_values[key],
                    allow_null=schema_nullable.get(str(key), False),
                    label=f"{label}.shards[{index}].partition_values.{key}",
                )
                for key in partition_keys
            )
        except CanonicalContractError as exc:
            raise V17V2ValidationError(str(exc)) from exc

        minimum_raw = shard.get("min_key")
        maximum_raw = shard.get("max_key")
        if format_name == "BLOB":
            if partition_values or minimum_raw is not None or maximum_raw is not None:
                _fail(f"{label}.shards[{index}] BLOB key metadata must be empty")
            minimum_order: tuple[tuple[int, Any, bytes], ...] = ()
            maximum_order: tuple[tuple[int, Any, bytes], ...] = ()
        else:
            minimum_values = _array(
                minimum_raw,
                label=f"{label}.shards[{index}].min_key",
                maximum=64,
            )
            maximum_values = _array(
                maximum_raw,
                label=f"{label}.shards[{index}].max_key",
                maximum=64,
            )
            if len(minimum_values) != len(primary_keys) or len(maximum_values) != len(primary_keys):
                _fail(f"{label}.shards[{index}] key width mismatch")
            try:
                minimum_order = tuple(
                    typed_scalar_total_order_key(
                        item,
                        allow_null=schema_nullable.get(str(primary_keys[position]), False),
                        label=f"{label}.shards[{index}].min_key[{position}]",
                    )
                    for position, item in enumerate(minimum_values)
                )
                maximum_order = tuple(
                    typed_scalar_total_order_key(
                        item,
                        allow_null=schema_nullable.get(str(primary_keys[position]), False),
                        label=f"{label}.shards[{index}].max_key[{position}]",
                    )
                    for position, item in enumerate(maximum_values)
                )
            except CanonicalContractError as exc:
                raise V17V2ValidationError(str(exc)) from exc
            if minimum_order > maximum_order:
                _fail(f"{label}.shards[{index}] min_key exceeds max_key")
        ranges.append((partition_order, minimum_order, maximum_order))
        order.append(
            (
                partition_order,
                (0,) if minimum_raw is None else (1, minimum_order),
                logical_name,
                byte_sha,
                object_path,
                canonical_json_bytes(shard),
            )
        )
        normalized.append(shard)
    if order != sorted(order) or len(set(order)) != len(order):
        _fail(f"{label}.shards are not in canonical complete order")
    previous_partition: tuple[tuple[int, Any, bytes], ...] | None = None
    previous_maximum: tuple[tuple[int, Any, bytes], ...] | None = None
    for partition_order, minimum_order, maximum_order in ranges:
        if (
            format_name == "PARQUET"
            and partition_order == previous_partition
            and previous_maximum is not None
            and previous_maximum >= minimum_order
        ):
            _fail(f"{label}.shards contain overlapping or duplicate key ranges")
        previous_partition = partition_order
        previous_maximum = maximum_order
    try:
        declared_total_bytes = require_nonnegative_int(
            document.get("total_size_bytes"),
            label=f"{label}.total_size_bytes",
            maximum=LIMITS["max_dataset_bytes"],
        )
        declared_total_rows = require_nonnegative_int(
            document.get("total_row_count"),
            label=f"{label}.total_row_count",
            maximum=LIMITS["max_dataset_rows"],
        )
    except ContractLimitError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    if declared_total_bytes != total_bytes:
        _fail(f"{label}.total_size_bytes mismatch")
    if declared_total_rows != total_rows:
        _fail(f"{label}.total_row_count mismatch")
    try:
        declared_content_set = require_sha256(
            document.get("content_set_sha256"),
            label=f"{label}.content_set_sha256",
        )
    except IdentityContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    if declared_content_set != _dataset_content_set_sha256(normalized):
        _fail(f"{label}.content_set_sha256 mismatch")


def validate_dataset_manifest(
    document: Mapping[str, Any],
    *,
    source_objects: Mapping[str, bytes | Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one dataset manifest and the exact closure of its shard objects."""

    payload = validate_document_identity(
        document,
        expected_version=DATASET_MANIFEST_VERSION,
    )
    referenced: set[str] = set()
    _validate_dataset_manifest_objects(
        payload,
        source_objects=source_objects,
        referenced_object_paths=referenced,
        label="dataset manifest",
    )
    if referenced != set(source_objects):
        _fail("dataset manifest object closure mismatch")
    return dict(payload)


def validate_dataset_record_schema_registry(
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact Phase 1 dataset-record registry relationships."""

    payload = dict(_mapping(resource, label="dataset record schema registry"))
    try:
        validate_mapping_against_packaged_schema(
            payload,
            expected_version=DATASET_RECORD_SCHEMA_REGISTRY_VERSION,
        )
    except (PackageResourceError, SchemaValidationError) as exc:
        raise V17V2ValidationError(str(exc)) from exc
    records = _array(
        payload.get("records"),
        label="dataset record schema registry records",
        maximum=LIMITS["max_sources"],
    )
    by_role: dict[str, Mapping[str, Any]] = {}
    observed_order: list[str] = []
    observed_ids: set[str] = set()
    for index, item in enumerate(records):
        record = _mapping(item, label=f"dataset record schema registry records[{index}]")
        role = _string(record.get("role"), label=f"dataset record schema registry role[{index}]")
        record_id = _string(
            record.get("record_schema_id"),
            label=f"dataset record schema registry record_schema_id[{index}]",
        )
        expected_id = role.lower().replace("_", "-") + ".v1"
        if record_id != expected_id:
            _fail(f"dataset record schema id does not derive from role: {role}")
        if role in by_role or record_id in observed_ids:
            _fail("dataset record schema registry contains duplicate role or record id")
        fields = _array(
            record.get("logical_fields"),
            label=f"dataset record schema registry fields[{index}]",
            maximum=64,
        )
        field_names = [
            _string(
                _mapping(field, label=f"dataset record field[{field_index}]").get("name"),
                label=f"dataset record field name[{field_index}]",
            )
            for field_index, field in enumerate(fields)
        ]
        if len(field_names) != len(set(field_names)):
            _fail(f"dataset record schema contains duplicate fields: {role}")
        key_fields = [
            *record["primary_key"],
            *record["partition_keys"],
            *record["sort_keys"],
            record["effective_time_field"],
            record["available_time_field"],
        ]
        if not set(key_fields).issubset(field_names):
            _fail(f"dataset record schema key/time field is undeclared: {role}")
        by_role[role] = record
        observed_order.append(role)
        observed_ids.add(record_id)
    if observed_order != sorted(observed_order, key=lambda value: (value.lower(), value)):
        _fail("dataset record schema registry is not canonically ordered")
    if set(by_role) != set(_DATASET_CATALOG_ROLE):
        _fail("dataset record schema registry role inventory mismatch")
    validate_json_limits(payload)
    return payload


def require_runtime_usable_dataset_record_schema_registry() -> dict[str, Any]:
    """Load and require the exact packaged Phase 1 dataset registry."""

    resource_path = "resources/dataset_record_schema_registry.v1.json"
    try:
        payload = load_packaged_json(resource_path)
    except PackageResourceError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    validated = validate_dataset_record_schema_registry(payload)
    if (
        hashlib.sha256(canonical_resource_bytes(validated)).hexdigest()
        != PACKAGE_ASSET_SHA256S.get(resource_path)
    ):
        _fail("dataset record schema registry is not the exact approved frozen resource")
    return validated


def _validate_phase1_document(
    document: Mapping[str, Any],
    *,
    expected_version: str,
) -> dict[str, Any]:
    return validate_document_identity(document, expected_version=expected_version)


def validate_market_pointer(document: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_phase1_document(document, expected_version=MARKET_POINTER_VERSION)


def validate_market_snapshot_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_phase1_document(
        document,
        expected_version=MARKET_SNAPSHOT_MANIFEST_VERSION,
    )


def validate_risk_policy_snapshot(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_phase1_document(
        document,
        expected_version=RISK_POLICY_SNAPSHOT_VERSION,
    )
    if _rfc3339_instant(payload["expires_at"], label="risk expires_at") <= _rfc3339_instant(
        payload["cutoff"], label="risk cutoff"
    ):
        _fail("risk policy snapshot is expired at cutoff")
    return payload


def validate_portfolio_required_inputs(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_phase1_document(
        document,
        expected_version=PORTFOLIO_REQUIRED_INPUTS_VERSION,
    )
    attestation = _mapping(payload["owner_attestation"], label="owner attestation")
    if attestation["nav"] <= 0:
        _fail("owner-attested NAV must be positive")
    holdings = _array(attestation["holdings"], label="owner-attested holdings", maximum=10000)
    codes = [str(_mapping(item, label="holding")["security_code"]) for item in holdings]
    if codes != sorted(codes) or len(codes) != len(set(codes)):
        _fail("owner-attested holdings are not ordered and unique")
    if payload["cutoff"] != attestation["attested_at"]:
        _fail("owner attestation cutoff mismatch")
    return payload


def validate_macro_overlay(document: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_phase1_document(document, expected_version=MACRO_OVERLAY_VERSION)


def validate_markov_overlay(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_phase1_document(document, expected_version=MARKOV_OVERLAY_VERSION)
    probabilities = _array(payload["probabilities"], label="markov probabilities", maximum=32)
    states = [str(_mapping(item, label="markov probability")["state"]) for item in probabilities]
    if states != sorted(states) or len(states) != len(set(states)):
        _fail("markov probabilities are not ordered and unique")
    if abs(sum(float(item["probability"]) for item in probabilities) - 1.0) > 1e-12:
        _fail("markov probabilities do not sum to one")
    if payload["selected_state"] not in states:
        _fail("markov selected_state is absent from probabilities")
    return payload


def validate_rank_output(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_phase1_document(document, expected_version=RANK_OUTPUT_VERSION)
    candidates = _array(payload["candidates"], label="rank candidates", maximum=1024)
    order = [
        (int(_mapping(item, label="rank candidate")["rank"]), str(item["security_code"]))
        for item in candidates
    ]
    if order != sorted(order) or [rank for rank, _ in order] != list(
        range(1, len(order) + 1)
    ):
        _fail("rank candidates are not in contiguous total order")
    return payload


def validate_portfolio_output(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_phase1_document(document, expected_version=PORTFOLIO_OUTPUT_VERSION)
    roles = [
        str(_mapping(item, label="portfolio input binding")["role"])
        for item in _array(payload["input_bindings"], label="portfolio input bindings", maximum=4)
    ]
    if roles != sorted(roles) or not {
        "portfolio_required_inputs",
        "risk_policy_snapshot",
    }.issubset(roles):
        _fail("portfolio output input binding inventory mismatch")
    return payload


def validate_source_role_matrix(resource: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the honest partial registry without treating it as runtime authority."""

    payload = dict(_mapping(resource, label="source role matrix"))
    try:
        validate_mapping_against_packaged_schema(
            payload,
            expected_version=SOURCE_ROLE_MATRIX_VERSION,
        )
    except (PackageResourceError, SchemaValidationError) as exc:
        raise V17V2ValidationError(str(exc)) from exc
    required = frozenset(
        {
            "protocol_version",
            "version",
            "authority",
            "completeness",
            "runtime_usable",
            "forbidden_role_suffixes",
            "ordering",
            "pending_registry",
            "conditional_semantics",
            "roles",
        }
    )
    _exact_keys(payload, required, label="source role matrix")
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("version") != SOURCE_ROLE_MATRIX_VERSION
        or payload.get("authority") is not False
    ):
        _fail("source role matrix envelope mismatch")
    completeness = payload.get("completeness")
    if completeness not in {"PARTIAL", "COMPLETE"}:
        _fail("source role matrix completeness invalid")
    if type(payload.get("runtime_usable")) is not bool:
        _fail("source role matrix runtime_usable must be boolean")
    if completeness == "PARTIAL" and payload["runtime_usable"] is not False:
        _fail("PARTIAL source role matrix cannot be runtime usable")
    suffixes = _array(
        payload.get("forbidden_role_suffixes"),
        label="source role matrix forbidden_role_suffixes",
        maximum=64,
    )
    normalized_suffixes = [
        _string(item, label=f"source role matrix forbidden_role_suffixes[{index}]")
        for index, item in enumerate(suffixes)
    ]
    if (
        normalized_suffixes != sorted(normalized_suffixes, key=lambda item: (item.lower(), item))
        or len({item.lower() for item in normalized_suffixes}) != len(normalized_suffixes)
        or any(re.fullmatch(r"_[a-z0-9_]+", item) is None for item in normalized_suffixes)
    ):
        _fail("source role matrix forbidden_role_suffixes invalid")
    pending = _array(
        payload.get("pending_registry"),
        label="source role matrix pending_registry",
        maximum=LIMITS["max_sources"],
    )
    normalized_pending = [
        _string(item, label=f"source role matrix pending_registry[{index}]")
        for index, item in enumerate(pending)
    ]
    if normalized_pending != sorted(
        normalized_pending,
        key=lambda item: (item.lower(), item),
    ) or len({item.lower() for item in normalized_pending}) != len(normalized_pending):
        _fail("source role matrix pending_registry is not canonically ordered")
    roles = _array(
        payload.get("roles"),
        label="source role matrix roles",
        maximum=LIMITS["max_sources"],
    )
    observed_roles: list[str] = []
    for index, item in enumerate(roles):
        row = _mapping(item, label=f"source role matrix roles[{index}]")
        _exact_keys(
            row,
            frozenset(
                {
                    "role",
                    "phase",
                    "kind",
                    "required",
                    "availability_disposition",
                    "schema_status",
                    "schema_version",
                }
            ),
            label=f"source role matrix roles[{index}]",
        )
        role = _reject_forbidden_role(
            row.get("role"),
            suffixes=normalized_suffixes,
            label=f"roles[{index}].role",
        )
        observed_roles.append(role)
        status = row.get("schema_status")
        version = row.get("schema_version")
        if status == "FROZEN":
            if (
                type(version) is not str
                or re.fullmatch(
                    r"myquant\.v17\.v2\.[a-z0-9-]+\.schema\.v1",
                    version,
                )
                is None
            ):
                _fail(f"roles[{index}] frozen schema_version invalid")
        elif status == "PENDING":
            if version is not None:
                _fail(f"roles[{index}] pending schema_version must be null")
        else:
            _fail(f"roles[{index}] schema_status invalid")
    if observed_roles != sorted(
        observed_roles,
        key=lambda item: (item.lower(), item),
    ):
        _fail("source role matrix roles are not canonically ordered")
    if len({role.lower() for role in observed_roles}) != len(observed_roles):
        _fail("source role matrix role collision")
    validate_json_limits(payload)
    return payload


def require_runtime_usable_source_role_matrix(
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact hash-bound, complete registry before runtime use."""

    payload = validate_source_role_matrix(resource)
    if payload.get("completeness") != "COMPLETE":
        _fail("source role matrix is not COMPLETE")
    if payload.get("runtime_usable") is not True:
        _fail("source role matrix is not runtime usable")
    if payload.get("pending_registry") != []:
        _fail("runtime source role matrix has pending registry entries")
    if any(
        _mapping(item, label="source role matrix role").get("schema_status") != "FROZEN"
        for item in _array(
            payload.get("roles"),
            label="source role matrix roles",
            maximum=LIMITS["max_sources"],
        )
    ):
        _fail("runtime source role matrix contains PENDING roles")
    resource_path = "resources/source_role_matrix.v1.json"
    try:
        approved = load_packaged_json(resource_path)
    except PackageResourceError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    observed_sha = hashlib.sha256(canonical_resource_bytes(payload)).hexdigest()
    if observed_sha != PACKAGE_ASSET_SHA256S[resource_path] or payload != approved:
        _fail("runtime source role matrix is not the exact approved frozen resource")
    roles = _array(
        payload.get("roles"),
        label="runtime source role matrix roles",
        maximum=LIMITS["max_sources"],
    )
    if not any(_mapping(row, label="runtime source role").get("required") is True for row in roles):
        _fail("runtime source role matrix required-role inventory is incomplete")
    observed = {
        str(row["role"]): (
            row["phase"],
            row["kind"],
            row["required"],
            row["availability_disposition"],
            row["schema_version"],
        )
        for row in roles
    }
    if observed != _PHASE1_ROLE_DECLARATIONS:
        _fail("runtime source role matrix Phase 1 inventory mismatch")
    expected_conditionals = [
        {
            "controller": controller,
            "disabled_mapping_may_be_missing": True,
            "enabled_input_unavailable_terminal": (
                "SHADOW_RANK_COMPLETE_NO_PORTFOLIO"
            ),
            "enabled_mapping_required": True,
        }
        for controller in ("macro", "markov")
    ]
    if payload.get("conditional_semantics") != expected_conditionals:
        _fail("runtime source role matrix conditional semantics mismatch")
    require_runtime_usable_dataset_record_schema_registry()
    return payload


def validate_source_hash_dag(
    *,
    source_role_matrix: Mapping[str, Any],
    source_objects: Mapping[str, bytes | Mapping[str, Any]],
    dataset_manifests: Mapping[str, Mapping[str, Any]],
    observation_dispositions: Mapping[str, Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_manifest_path: str,
    generation_catalogs: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    source_binding_set: Mapping[str, Any],
    source_binding_set_path: str,
    source_locator: Mapping[str, Any],
    source_locator_path: str,
) -> dict[str, Any]:
    """Structurally validate the source hash DAG and return its terminal locator.

    The accepted edge order is:

    ``source/dataset/disposition -> source manifest -> catalogs+summaries
    -> binding set -> source locator``.  This function grants no runtime
    admission; callers must use :func:`admit_runtime_source_hash_dag`.
    """

    role_matrix = validate_source_role_matrix(source_role_matrix)
    forbidden_suffixes = tuple(
        _array(
            role_matrix.get("forbidden_role_suffixes"),
            label="source role matrix forbidden_role_suffixes",
            maximum=64,
        )
    )
    role_rows = {
        str(row["role"]): row
        for item in _array(
            role_matrix.get("roles"),
            label="source role matrix roles",
            maximum=LIMITS["max_sources"],
        )
        for row in [_mapping(item, label="source role matrix role")]
    }
    registered_roles = set(role_rows)
    dataset_registry = require_runtime_usable_dataset_record_schema_registry()
    dataset_record_by_role = {
        str(record["role"]): _mapping(record, label="dataset registry record")
        for record in dataset_registry["records"]
    }
    _require_protocol_path(source_manifest_path, label="source manifest path")
    _require_protocol_path(source_binding_set_path, label="source binding set path")
    _require_protocol_path(source_locator_path, label="source locator path")
    if (
        re.fullmatch(
            rf"data/private/v17_sources/protocol-v2/manifests/"
            rf"{_PATH_COMPONENT}\.bindings\.json",
            source_binding_set_path,
        )
        is None
    ):
        _fail("source binding set path is not canonical")
    manifest = validate_document_identity(
        source_manifest,
        expected_version=SOURCE_MANIFEST_VERSION,
    )
    cutoff_instant = _rfc3339_instant(
        manifest.get("cutoff"),
        label="source manifest cutoff",
    )
    manifest_created_at = _rfc3339_instant(
        manifest.get("created_at"),
        label="source manifest created_at",
    )
    if manifest_created_at < cutoff_instant:
        _fail("source manifest created_at precedes cutoff")
    binding_set = validate_document_identity(
        source_binding_set,
        expected_version=SOURCE_BINDING_SET_VERSION,
    )
    _exact_keys(
        binding_set,
        _SOURCE_BINDING_SET_KEYS,
        label="source binding set",
    )
    locator = validate_document_identity(
        source_locator,
        expected_version=SOURCE_LOCATOR_VERSION,
    )

    matrix_ref = _mapping(manifest.get("role_matrix_ref"), label="role_matrix_ref")
    _exact_keys(matrix_ref, _ROLE_MATRIX_REF_KEYS, label="role_matrix_ref")
    if (
        matrix_ref.get("resource_name") != "source_role_matrix.v1.json"
        or matrix_ref.get("resource_version") != SOURCE_ROLE_MATRIX_VERSION
    ):
        _fail("role_matrix_ref identity mismatch")
    try:
        declared_matrix_sha = require_sha256(
            matrix_ref.get("byte_sha256"),
            label="role_matrix_ref.byte_sha256",
        )
    except IdentityContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    if hashlib.sha256(canonical_resource_bytes(role_matrix)).hexdigest() != declared_matrix_sha:
        _fail("role_matrix_ref.byte_sha256 mismatch")

    referenced_object_paths: set[str] = set()
    dataset_docs: dict[str, Mapping[str, Any]] = {}
    for path, document in dataset_manifests.items():
        _require_protocol_path(path, label="dataset manifest path")
        dataset_document = validate_document_identity(
            document,
            expected_version=DATASET_MANIFEST_VERSION,
        )
        dataset_role = _reject_forbidden_role(
            dataset_document.get("role"),
            suffixes=forbidden_suffixes,
            label=f"dataset manifest {path}.role",
        )
        if dataset_role not in registered_roles:
            _fail(f"dataset manifest {path}.role is not registered")
        registry_record = dataset_record_by_role.get(dataset_role)
        if registry_record is None:
            _fail(f"dataset manifest {path}.role lacks a record schema")
        if (
            dataset_document.get("schema") != registry_record.get("logical_fields")
            or dataset_document.get("primary_key") != registry_record.get("primary_key")
            or dataset_document.get("partition_keys") != registry_record.get("partition_keys")
            or dataset_document.get("sort_keys") != registry_record.get("sort_keys")
        ):
            _fail(f"dataset manifest {path} does not match its record schema")
        _validate_dataset_manifest_objects(
            dataset_document,
            source_objects=source_objects,
            referenced_object_paths=referenced_object_paths,
            label=f"dataset manifest {path}",
        )
        dataset_docs[path] = dataset_document
    disposition_docs: dict[str, Mapping[str, Any]] = {}
    for path, document in observation_dispositions.items():
        _require_protocol_path(path, label="observation disposition path")
        disposition = validate_document_identity(
            document,
            expected_version=OBSERVATION_DISPOSITION_VERSION,
        )
        dataset_ref = _mapping(
            disposition.get("dataset_manifest_ref"),
            label=f"disposition {path}.dataset_manifest_ref",
        )
        _resolve_document(
            dataset_ref,
            dataset_docs,
            expected_version=DATASET_MANIFEST_VERSION,
            label=f"disposition {path}.dataset_manifest_ref",
        )
        disposition_docs[path] = disposition

    source_rows = _array(
        manifest.get("sources"),
        label="source manifest sources",
        maximum=LIMITS["max_sources"],
    )
    source_order: list[tuple[str, str]] = []
    seen_source_ids: set[str] = set()
    seen_source_roles: set[str] = set()
    source_rows_by_role: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(source_rows):
        source = _mapping(item, label=f"sources[{index}]")
        source_id = _opaque_id(
            source.get("source_id"),
            label=f"sources[{index}].source_id",
        )
        role = _reject_forbidden_role(
            source.get("role"),
            suffixes=forbidden_suffixes,
            label=f"sources[{index}].role",
        )
        if role not in registered_roles:
            _fail(f"sources[{index}].role is not registered")
        if source_id in seen_source_ids or role in seen_source_roles:
            _fail("source manifest contains duplicate source_id or role")
        seen_source_ids.add(source_id)
        seen_source_roles.add(role)
        source_order.append((role, source_id))
        availability = source.get("availability")
        if availability == "AVAILABLE":
            source_ref = _mapping(source.get("source_ref"), label=f"sources[{index}].source_ref")
            path = _reference_path(source_ref, label=f"sources[{index}].source_ref")
            raw = source_objects.get(path)
            if raw is None:
                _fail(f"sources[{index}].source_ref does not resolve")
            _validate_artifact_ref(
                source_ref,
                raw=raw,
                expected_path=path,
                label=f"sources[{index}].source_ref",
            )
            referenced_object_paths.add(path)
        elif availability != "UNAVAILABLE":
            _fail(f"sources[{index}].availability invalid")
        source_rows_by_role[role] = source
    if source_order != sorted(source_order):
        _fail("source manifest sources are not canonically ordered")
    if referenced_object_paths != set(source_objects):
        _fail("source object closure mismatch")

    manifest_dataset_refs = _array(
        manifest.get("dataset_manifest_refs"),
        label="source manifest dataset_manifest_refs",
        maximum=LIMITS["max_sources"],
    )
    if len(manifest_dataset_refs) != len(dataset_docs):
        _fail("source manifest dataset reference closure mismatch")
    dataset_ref_order: list[tuple[str, str, str]] = []
    for index, ref in enumerate(manifest_dataset_refs):
        dataset_ref = _mapping(ref, label=f"dataset_manifest_refs[{index}]")
        _resolve_document(
            dataset_ref,
            dataset_docs,
            expected_version=DATASET_MANIFEST_VERSION,
            label=f"dataset_manifest_refs[{index}]",
        )
        dataset_ref_order.append(
            (
                str(dataset_ref["artifact_id"]),
                str(dataset_ref["relative_path"]),
                str(dataset_ref["byte_sha256"]),
            )
        )
    if dataset_ref_order != sorted(dataset_ref_order) or len(set(dataset_ref_order)) != len(
        dataset_ref_order
    ):
        _fail("source manifest dataset refs are not canonical and unique")
    manifest_disposition_refs = _array(
        manifest.get("observation_disposition_refs"),
        label="source manifest observation_disposition_refs",
        maximum=LIMITS["max_sources"],
    )
    if len(manifest_disposition_refs) != len(disposition_docs):
        _fail("source manifest disposition reference closure mismatch")
    disposition_ref_order: list[tuple[str, str, str]] = []
    for index, ref in enumerate(manifest_disposition_refs):
        disposition_ref = _mapping(
            ref,
            label=f"observation_disposition_refs[{index}]",
        )
        _resolve_document(
            disposition_ref,
            disposition_docs,
            expected_version=OBSERVATION_DISPOSITION_VERSION,
            label=f"observation_disposition_refs[{index}]",
        )
        disposition_ref_order.append(
            (
                str(disposition_ref["artifact_id"]),
                str(disposition_ref["relative_path"]),
                str(disposition_ref["byte_sha256"]),
            )
        )
    if disposition_ref_order != sorted(disposition_ref_order) or len(
        set(disposition_ref_order)
    ) != len(disposition_ref_order):
        _fail("source manifest disposition refs are not canonical and unique")

    manifest_ref = {
        "artifact_id": _document_id(manifest),
        "artifact_version": SOURCE_MANIFEST_VERSION,
        "relative_path": source_manifest_path,
        "byte_sha256": document_byte_sha256(manifest),
        "semantic_sha256": manifest["semantic_sha256"],
    }
    catalog_docs: dict[str, Mapping[str, Any]] = {}
    catalog_paths_by_role: dict[str, str] = {}
    catalog_created_instants: list[datetime] = []
    table_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for path, document in generation_catalogs.items():
        _require_protocol_path(path, label="generation catalog path")
        catalog = validate_document_identity(
            document,
            expected_version=GENERATION_CATALOG_VERSION,
        )
        _exact_keys(
            catalog,
            _GENERATION_CATALOG_KEYS,
            label=f"generation catalog {path}",
        )
        catalog_role = _reject_forbidden_role(
            catalog.get("role"),
            suffixes=forbidden_suffixes,
            label=f"generation catalog {path}.role",
        )
        role_row = role_rows.get(catalog_role)
        if role_row is None:
            _fail(f"generation catalog role is not registered: {catalog_role}")
        if (
            role_row.get("kind") != "OBJECT"
            or role_row.get("schema_status") != "FROZEN"
            or role_row.get("schema_version") != "myquant.v17.v2.generation-catalog.schema.v1"
        ):
            _fail(f"generation catalog role declaration mismatch: {catalog_role}")
        catalog_phase = _string(
            catalog.get("phase"),
            label=f"generation catalog {path}.phase",
        )
        if catalog_phase != role_row.get("phase"):
            _fail(f"generation catalog role phase mismatch: {catalog_role}")
        catalog_source = source_rows_by_role.get(catalog_role)
        if catalog_source is None or catalog_source.get("availability") != "AVAILABLE":
            _fail(f"generation catalog lacks AVAILABLE source evidence: {catalog_role}")
        if catalog_role in catalog_paths_by_role:
            _fail(f"duplicate generation catalog role carrier: {catalog_role}")
        catalog_paths_by_role[catalog_role] = path
        if catalog.get("market") != manifest.get("market") or catalog.get("cutoff") != manifest.get(
            "cutoff"
        ):
            _fail(f"generation catalog identity mismatch: {path}")
        catalog_created_at = _rfc3339_instant(
            catalog.get("created_at"),
            label=f"generation catalog {path}.created_at",
        )
        if catalog_created_at < manifest_created_at:
            _fail(f"generation catalog predates source manifest: {path}")
        catalog_created_instants.append(catalog_created_at)
        if (
            _mapping(catalog.get("source_manifest_ref"), label=f"catalog {path} source ref")
            != manifest_ref
        ):
            _fail(f"generation catalog source_manifest_ref mismatch: {path}")
        tables = _array(
            catalog.get("tables"),
            label=f"catalog {path} tables",
            maximum=LIMITS["max_sources"],
        )
        table_order: list[tuple[str, ...]] = []
        for index, item in enumerate(tables):
            table = _mapping(item, label=f"catalog {path} tables[{index}]")
            _exact_keys(
                table,
                _GENERATION_CATALOG_TABLE_KEYS,
                label=f"catalog {path} tables[{index}]",
            )
            dataset_ref = _mapping(
                table.get("dataset_manifest_ref"),
                label=f"catalog {path} tables[{index}].dataset_manifest_ref",
            )
            resolved_dataset = _resolve_document(
                dataset_ref,
                dataset_docs,
                expected_version=DATASET_MANIFEST_VERSION,
                label=f"catalog {path} tables[{index}].dataset_manifest_ref",
            )
            summary_ref_full = _mapping(
                table.get("summary_ref"),
                label=f"catalog {path} tables[{index}].summary_ref",
            )
            embedded = _mapping(
                summary_ref_full.get("dataset_manifest_ref"),
                label=f"catalog {path} tables[{index}].summary_ref.dataset_manifest_ref",
            )
            if embedded != dataset_ref:
                _fail("catalog summary embedded dataset ref mismatch")
            summary_ref = {
                key: summary_ref_full[key] for key in _ARTIFACT_REF_KEYS if key in summary_ref_full
            }
            summary_path = _reference_path(
                summary_ref,
                label=f"catalog {path} tables[{index}].summary_ref",
            )
            summary = summaries.get(summary_path)
            if summary is None:
                _fail(f"catalog summary does not resolve: {summary_path}")
            summary = validate_document_identity(
                summary,
                expected_version=DATASET_SUMMARY_VERSION,
            )
            if (
                summary.get("protocol_version") != PROTOCOL_VERSION
                or summary.get("version") != summary_ref.get("artifact_version")
                or summary.get("summary_id") != summary_ref.get("artifact_id")
                or summary.get("authority") is not False
                or summary.get("semantic_sha256") != summary_ref.get("semantic_sha256")
            ):
                _fail(f"catalog summary identity mismatch: {summary_path}")
            if summary.get("source_manifest_ref") != manifest_ref:
                _fail(f"catalog summary source_manifest_ref mismatch: {summary_path}")
            _validate_artifact_ref(
                summary_ref,
                raw=summary,
                expected_path=summary_path,
                label=f"catalog {path} tables[{index}].summary_ref",
            )
            summary_dataset_ref = _mapping(
                summary.get("dataset_manifest_ref"),
                label=f"summary {summary_path}.dataset_manifest_ref",
            )
            if summary_dataset_ref != dataset_ref:
                _fail("summary document embedded dataset ref mismatch")
            if summary.get("row_count") != resolved_dataset.get("total_row_count"):
                _fail("summary row_count does not match dataset manifest")
            stage = _string(table.get("stage"), label="catalog table stage")
            role = _reject_forbidden_role(
                table.get("role"),
                suffixes=forbidden_suffixes,
                label="catalog table role",
            )
            if role not in registered_roles:
                _fail(f"catalog table role is not registered: {role}")
            registry_record = dataset_record_by_role.get(role)
            if registry_record is None:
                _fail(f"catalog table role lacks a record schema: {role}")
            if (
                catalog_role != _DATASET_CATALOG_ROLE[role]
                or table.get("record_schema_id") != registry_record.get("record_schema_id")
                or table.get("primary_key") != registry_record.get("primary_key")
                or table.get("valid_time_field") != registry_record.get("effective_time_field")
                or table.get("available_time_field")
                != registry_record.get("available_time_field")
            ):
                _fail(f"catalog table record schema binding mismatch: {role}")
            table_id = _string(table.get("table_id"), label="catalog table_id")
            identity = (path, stage, role)
            if identity in table_index:
                _fail(f"duplicate catalog table identity: {identity}")
            table_index[identity] = table
            table_order.append(
                (
                    stage,
                    role,
                    table_id,
                    summary_path,
                    str(summary_ref["byte_sha256"]),
                    _reference_path(dataset_ref, label="catalog dataset ref"),
                    str(dataset_ref["byte_sha256"]),
                )
            )
        if table_order != sorted(table_order):
            _fail(f"generation catalog tables are not in total order: {path}")
        catalog_docs[path] = catalog

    if (
        _mapping(binding_set.get("source_manifest_ref"), label="binding set source ref")
        != manifest_ref
    ):
        _fail("source binding set source_manifest_ref mismatch")
    if binding_set.get("market") != manifest.get("market") or binding_set.get(
        "cutoff"
    ) != manifest.get("cutoff"):
        _fail("source binding set identity mismatch")
    bindings = _array(
        binding_set.get("bindings"),
        label="source binding set bindings",
        maximum=LIMITS["max_sources"],
    )
    binding_order: list[tuple[str, ...]] = []
    disposition_pairs: set[tuple[str, str]] = set()
    referenced_catalog_paths: set[str] = set()
    referenced_summary_paths: set[str] = set()
    referenced_dataset_paths: set[str] = set()
    referenced_disposition_paths: set[str] = set()
    referenced_table_keys: set[tuple[str, str, str]] = set()
    for index, item in enumerate(bindings):
        binding = _mapping(item, label=f"bindings[{index}]")
        _exact_keys(
            binding,
            _SOURCE_BINDING_KEYS,
            label=f"bindings[{index}]",
        )
        stage = _string(binding.get("stage"), label=f"bindings[{index}].stage")
        role = _reject_forbidden_role(
            binding.get("role"),
            suffixes=forbidden_suffixes,
            label=f"bindings[{index}].role",
        )
        if role not in registered_roles:
            _fail(f"bindings[{index}].role is not registered")
        disposition_id = _string(
            binding.get("disposition_id"),
            label=f"bindings[{index}].disposition_id",
        )
        disposition_pair = (stage, disposition_id)
        if disposition_pair in disposition_pairs:
            _fail(f"duplicate (stage, disposition_id): {disposition_pair}")
        disposition_pairs.add(disposition_pair)

        catalog_ref = _mapping(binding.get("catalog_ref"), label=f"bindings[{index}].catalog_ref")
        catalog_path = _reference_path(catalog_ref, label=f"bindings[{index}].catalog_ref")
        referenced_catalog_paths.add(catalog_path)
        _resolve_document(
            catalog_ref,
            catalog_docs,
            expected_version=GENERATION_CATALOG_VERSION,
            label=f"bindings[{index}].catalog_ref",
        )
        table = table_index.get((catalog_path, stage, role))
        if table is None:
            _fail(f"bindings[{index}] has no matching catalog table")
        referenced_table_keys.add((catalog_path, stage, role))
        if binding.get("summary_ref") != {
            key: _mapping(table.get("summary_ref"), label="catalog summary ref")[key]
            for key in _ARTIFACT_REF_KEYS
        }:
            _fail(f"bindings[{index}].summary_ref mismatch")
        if binding.get("dataset_manifest_ref") != table.get("dataset_manifest_ref"):
            _fail(f"bindings[{index}].dataset_manifest_ref mismatch")
        referenced_summary_paths.add(
            _reference_path(
                _mapping(binding.get("summary_ref"), label=f"bindings[{index}].summary_ref"),
                label=f"bindings[{index}].summary_ref",
            )
        )
        referenced_dataset_paths.add(
            _reference_path(
                _mapping(
                    binding.get("dataset_manifest_ref"),
                    label=f"bindings[{index}].dataset_manifest_ref",
                ),
                label=f"bindings[{index}].dataset_manifest_ref",
            )
        )
        disposition_ref = _mapping(
            binding.get("observation_disposition_ref"),
            label=f"bindings[{index}].observation_disposition_ref",
        )
        resolved_disposition = _resolve_document(
            disposition_ref,
            disposition_docs,
            expected_version=OBSERVATION_DISPOSITION_VERSION,
            label=f"bindings[{index}].observation_disposition_ref",
        )
        referenced_disposition_paths.add(
            _reference_path(
                disposition_ref,
                label=f"bindings[{index}].observation_disposition_ref",
            )
        )
        if resolved_disposition.get("disposition_id") != disposition_id:
            _fail(f"bindings[{index}].disposition_id mismatch")
        if resolved_disposition.get("stage") != stage:
            _fail(f"bindings[{index}] stage/disposition mismatch")
        if resolved_disposition.get("dataset_manifest_ref") != binding.get("dataset_manifest_ref"):
            _fail(f"bindings[{index}] dataset/disposition mismatch")
        binding_order.append(_binding_order_key(binding))
    if binding_order != sorted(binding_order):
        _fail("source binding set is not in the required complete total order")
    if referenced_catalog_paths != set(catalog_docs):
        _fail("source binding set catalog closure mismatch")
    if referenced_summary_paths != set(summaries):
        _fail("source binding set summary closure mismatch")
    if referenced_dataset_paths != set(dataset_docs):
        _fail("source binding set dataset closure mismatch")
    if referenced_disposition_paths != set(disposition_docs):
        _fail("source binding set disposition closure mismatch")
    if referenced_table_keys != set(table_index):
        _fail("source binding set table closure mismatch")

    binding_set_ref = _mapping(
        locator.get("binding_set_ref"),
        label="source locator binding set ref",
    )
    _exact_keys(
        locator,
        frozenset(
            {
                "protocol_version",
                "version",
                "locator_id",
                "market",
                "cutoff",
                "created_at",
                "binding_set_ref",
                "authority",
                "semantic_sha256",
            }
        ),
        label="source locator",
    )
    _validate_artifact_ref(
        binding_set_ref,
        document=binding_set,
        expected_path=source_binding_set_path,
        expected_version=SOURCE_BINDING_SET_VERSION,
        label="source locator binding set ref",
    )
    locator_id = _opaque_id(locator.get("locator_id"), label="source locator locator_id")
    expected_locator_path = "data/private/v17_sources/protocol-v2/locators/" f"{locator_id}.json"
    if source_locator_path != expected_locator_path:
        _fail("source locator path/locator_id mismatch")
    locator_created_at = _rfc3339_instant(
        locator.get("created_at"),
        label="source locator created_at",
    )
    if locator_created_at < max([manifest_created_at, *catalog_created_instants]):
        _fail("source locator predates its source DAG")
    if locator.get("market") != manifest.get("market") or locator.get("cutoff") != manifest.get(
        "cutoff"
    ):
        _fail("source locator identity mismatch")
    return dict(locator)


def _artifact_version_for_declared_schema(schema_version: Any, *, label: str) -> str:
    schema_id = _string(schema_version, label=label)
    if not schema_id.endswith(".schema.v1"):
        _fail(f"{label} is not a protocol-v2 schema identity")
    artifact_version = schema_id.removesuffix(".schema.v1") + ".v1"
    try:
        schema_path = schema_path_for_version(artifact_version)
        schema = load_packaged_json(schema_path)
    except (PackageResourceError, SchemaValidationError) as exc:
        raise V17V2ValidationError(f"{label} is not hash-bound to a packaged schema") from exc
    if schema.get("$id") != schema_id:
        _fail(f"{label} does not match the packaged schema identity")
    return artifact_version


def _load_runtime_object_document(
    *,
    role: str,
    role_row: Mapping[str, Any],
    source: Mapping[str, Any],
    source_objects: Mapping[str, bytes | Mapping[str, Any]],
) -> dict[str, Any]:
    source_ref = _mapping(source.get("source_ref"), label=f"runtime source {role}.source_ref")
    source_path = _reference_path(
        source_ref,
        label=f"runtime source {role}.source_ref",
    )
    raw = source_objects.get(source_path)
    if type(raw) is not bytes:
        _fail(f"runtime OBJECT role requires canonical stored bytes: {role}")
    try:
        document = load_canonical_resource(raw, label=f"runtime source object {role}")
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    if type(document) is not dict:
        _fail(f"runtime OBJECT role root must be an object: {role}")
    expected_version = _artifact_version_for_declared_schema(
        role_row.get("schema_version"),
        label=f"runtime role {role}.schema_version",
    )
    if expected_version not in SUPPORTED_DOCUMENT_VERSIONS:
        _fail(f"runtime OBJECT role schema lacks an identity validator: {role}")
    validators = {
        MARKET_POINTER_VERSION: validate_market_pointer,
        MARKET_SNAPSHOT_MANIFEST_VERSION: validate_market_snapshot_manifest,
        RISK_POLICY_SNAPSHOT_VERSION: validate_risk_policy_snapshot,
        PORTFOLIO_REQUIRED_INPUTS_VERSION: validate_portfolio_required_inputs,
        MACRO_OVERLAY_VERSION: validate_macro_overlay,
        MARKOV_OVERLAY_VERSION: validate_markov_overlay,
    }
    validator = validators.get(expected_version)
    if validator is None:
        _fail(f"runtime OBJECT role lacks a Phase 1 cross-validator: {role}")
    validated = validator(document)
    _validate_artifact_ref(
        source_ref,
        document=validated,
        expected_path=source_path,
        expected_version=expected_version,
        label=f"runtime source {role}.source_ref",
    )
    if validated.get("role") != role or validated.get("phase") != role_row.get("phase"):
        _fail(f"runtime OBJECT carrier does not bind exact role and phase: {role}")
    return validated


def _admit_runtime_source_hash_dag_core(
    *,
    source_role_matrix: Mapping[str, Any],
    source_objects: Mapping[str, bytes | Mapping[str, Any]],
    dataset_manifests: Mapping[str, Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    generation_catalogs: Mapping[str, Mapping[str, Any]],
    source_binding_set: Mapping[str, Any],
    source_locator: Mapping[str, Any],
) -> SourceAdmissionOutcome:
    """Apply the complete registry to an already structurally valid source DAG.

    The public admission function supplies the exact approved registry.  This
    core is separate only so its exhaustive role semantics can be tested with
    synthetic COMPLETE registries without weakening the package-byte trust
    root.
    """

    for path, raw in source_objects.items():
        if type(raw) is not bytes:
            _fail(f"runtime source object must be exact stored bytes: {path}")

    role_matrix = validate_source_role_matrix(source_role_matrix)
    if role_matrix.get("completeness") != "COMPLETE":
        _fail("runtime source role matrix is not COMPLETE")
    if role_matrix.get("runtime_usable") is not True:
        _fail("runtime source role matrix is not runtime usable")
    if role_matrix.get("pending_registry") != []:
        _fail("runtime source role matrix has pending registry entries")
    role_rows = {
        str(row["role"]): row
        for item in _array(
            role_matrix.get("roles"),
            label="runtime source role matrix roles",
            maximum=LIMITS["max_sources"],
        )
        for row in [_mapping(item, label="runtime source role matrix role")]
    }
    expected_versions: dict[str, str] = {}
    for role, row in role_rows.items():
        if row.get("schema_status") != "FROZEN":
            _fail(f"runtime source role schema is not frozen: {role}")
        expected_versions[role] = _artifact_version_for_declared_schema(
            row.get("schema_version"),
            label=f"runtime role {role}.schema_version",
        )
        kind = row.get("kind")
        if kind == "DATASET" and expected_versions[role] != DATASET_MANIFEST_VERSION:
            _fail(f"runtime DATASET role has the wrong declared schema: {role}")
        if kind == "DISPOSITION":
            _fail(f"runtime DISPOSITION role lacks an unambiguous role carrier: {role}")

    manifest = validate_document_identity(
        source_manifest,
        expected_version=SOURCE_MANIFEST_VERSION,
    )
    sources_by_role: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(
        _array(
            manifest.get("sources"),
            label="runtime source manifest sources",
            maximum=LIMITS["max_sources"],
        )
    ):
        source = _mapping(item, label=f"runtime sources[{index}]")
        role = _string(source.get("role"), label=f"runtime sources[{index}].role")
        if role in sources_by_role:
            _fail(f"runtime source role is duplicated: {role}")
        sources_by_role[role] = source

    dataset_docs_by_role: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    dataset_docs_by_path: dict[str, Mapping[str, Any]] = {}
    for path, document in dataset_manifests.items():
        dataset = validate_document_identity(
            document,
            expected_version=DATASET_MANIFEST_VERSION,
        )
        role = _string(dataset.get("role"), label=f"runtime dataset {path}.role")
        dataset_docs_by_role.setdefault(role, []).append((path, dataset))
        dataset_docs_by_path[path] = dataset

    catalog_docs_by_role: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    table_links: dict[str, list[tuple[str, str]]] = {}
    for catalog_path, document in generation_catalogs.items():
        catalog = validate_document_identity(
            document,
            expected_version=GENERATION_CATALOG_VERSION,
        )
        catalog_role = _string(
            catalog.get("role"),
            label=f"runtime catalog {catalog_path}.role",
        )
        catalog_docs_by_role.setdefault(catalog_role, []).append((catalog_path, catalog))
        for index, item in enumerate(
            _array(
                catalog.get("tables"),
                label=f"runtime catalog {catalog_path} tables",
                maximum=LIMITS["max_sources"],
            )
        ):
            table = _mapping(item, label=f"runtime catalog {catalog_path} tables[{index}]")
            role = _string(table.get("role"), label="runtime catalog table role")
            stage = _string(table.get("stage"), label="runtime catalog table stage")
            dataset_ref = _mapping(
                table.get("dataset_manifest_ref"),
                label="runtime catalog table dataset_manifest_ref",
            )
            dataset_path = _reference_path(
                dataset_ref,
                label="runtime catalog table dataset_manifest_ref",
            )
            resolved = dataset_docs_by_path.get(dataset_path)
            if resolved is None:
                _fail(f"runtime catalog table dataset does not resolve: {dataset_path}")
            if resolved.get("role") != role:
                _fail(f"runtime catalog table substituted dataset role: {role}")
            table_links.setdefault(role, []).append((stage, dataset_path))

    binding_set = validate_document_identity(
        source_binding_set,
        expected_version=SOURCE_BINDING_SET_VERSION,
    )
    binding_links: dict[str, list[tuple[str, str]]] = {}
    for index, item in enumerate(
        _array(
            binding_set.get("bindings"),
            label="runtime source binding set bindings",
            maximum=LIMITS["max_sources"],
        )
    ):
        binding = _mapping(item, label=f"runtime bindings[{index}]")
        role = _string(binding.get("role"), label=f"runtime bindings[{index}].role")
        stage = _string(binding.get("stage"), label=f"runtime bindings[{index}].stage")
        dataset_ref = _mapping(
            binding.get("dataset_manifest_ref"),
            label=f"runtime bindings[{index}].dataset_manifest_ref",
        )
        dataset_path = _reference_path(
            dataset_ref,
            label=f"runtime bindings[{index}].dataset_manifest_ref",
        )
        resolved = dataset_docs_by_path.get(dataset_path)
        if resolved is None:
            _fail(f"runtime binding dataset does not resolve: {dataset_path}")
        if resolved.get("role") != role:
            _fail(f"runtime binding substituted dataset role: {role}")
        binding_links.setdefault(role, []).append((stage, dataset_path))

    unexpected_roles = sorted(
        (
            set(sources_by_role)
            | set(dataset_docs_by_role)
            | set(catalog_docs_by_role)
            | set(table_links)
            | set(binding_links)
        )
        - set(role_rows)
    )
    if unexpected_roles:
        _fail(f"runtime source DAG contains unregistered roles: {unexpected_roles}")

    unavailable_required_roles: list[str] = []
    reject_required_roles: list[str] = []
    admitted_input_bindings: list[tuple[str, str, str, str, str, str]] = []
    object_carrier_identities: dict[tuple[str, str], str] = {}
    object_documents: dict[str, Mapping[str, Any]] = {}
    for role, row in role_rows.items():
        source = sources_by_role.get(role)
        if source is None:
            if row.get("required") is True:
                _fail(f"runtime required role lacks an availability row: {role}")
            continue
        availability = source.get("availability")
        kind = row.get("kind")
        datasets = dataset_docs_by_role.get(role, [])
        catalogs = catalog_docs_by_role.get(role, [])
        tables = table_links.get(role, [])
        bindings = binding_links.get(role, [])
        if availability == "UNAVAILABLE":
            if datasets or catalogs or tables or bindings:
                _fail(f"runtime unavailable role has bound artifacts: {role}")
            if row.get("required") is True:
                unavailable_required_roles.append(role)
                if row.get("availability_disposition") == "REJECT_BEFORE_INITIALIZED_ZERO_WRITE":
                    reject_required_roles.append(role)
            continue
        if availability != "AVAILABLE":
            _fail(f"runtime source availability is invalid: {role}")
        if kind == "DATASET":
            if len(datasets) != 1 or len(tables) != 1 or len(bindings) != 1:
                _fail(f"runtime DATASET role closure is not one-to-one: {role}")
            declared_phase = row.get("phase")
            if tables[0][0] != declared_phase or bindings[0][0] != declared_phase:
                _fail(f"runtime role phase mismatch: {role}")
            if tables[0][1] != bindings[0][1]:
                _fail(f"runtime role table/binding dataset mismatch: {role}")
            dataset_path, dataset_document = datasets[0]
            admitted_input_bindings.append(
                (
                    role,
                    _document_id(dataset_document),
                    DATASET_MANIFEST_VERSION,
                    dataset_path,
                    document_byte_sha256(dataset_document),
                    str(dataset_document["semantic_sha256"]),
                )
            )
        elif kind == "OBJECT":
            if datasets or tables or bindings:
                _fail(f"runtime OBJECT role appears in a DATASET carrier: {role}")
            object_ref: Mapping[str, Any]
            if expected_versions[role] == GENERATION_CATALOG_VERSION:
                if len(catalogs) != 1:
                    _fail(f"runtime generation-catalog role closure is not one-to-one: {role}")
                object_path, object_document = catalogs[0]
                if object_document.get("phase") != row.get("phase"):
                    _fail(f"runtime OBJECT carrier phase mismatch: {role}")
                object_ref = {
                    "relative_path": object_path,
                    "byte_sha256": document_byte_sha256(object_document),
                }
                source_ref = _mapping(
                    source.get("source_ref"),
                    label=f"runtime source {role}.source_ref",
                )
                source_path = _reference_path(
                    source_ref,
                    label=f"runtime source {role}.source_ref",
                )
                _validate_artifact_ref(
                    source_ref,
                    raw=source_objects.get(source_path),
                    expected_path=source_path,
                    label=f"runtime source {role}.source_ref",
                )
            else:
                if catalogs:
                    _fail(f"runtime non-catalog OBJECT role has a catalog carrier: {role}")
                object_document = _load_runtime_object_document(
                    role=role,
                    role_row=row,
                    source=source,
                    source_objects=source_objects,
                )
                object_ref = _mapping(
                    source.get("source_ref"),
                    label=f"runtime source {role}.source_ref",
                )
            object_identity = (
                str(object_ref["relative_path"]),
                str(object_ref["byte_sha256"]),
            )
            previous_role = object_carrier_identities.get(object_identity)
            if previous_role is not None:
                _fail("runtime OBJECT carrier is shared across roles: " f"{previous_role}, {role}")
            object_carrier_identities[object_identity] = role
            object_documents[role] = object_document
            admitted_input_bindings.append(
                (
                    role,
                    _document_id(object_document),
                    str(object_document["version"]),
                    str(object_ref["relative_path"]),
                    str(object_ref["byte_sha256"]),
                    str(object_document["semantic_sha256"]),
                )
            )
        else:
            _fail(f"runtime source role kind is unsupported: {role}")

    pointer = object_documents.get("market_pointer")
    snapshot = object_documents.get("market_snapshot_manifest")
    if pointer is not None and snapshot is not None:
        snapshot_ref = _mapping(
            pointer.get("snapshot_manifest_ref"),
            label="market pointer snapshot_manifest_ref",
        )
        snapshot_source = _mapping(
            sources_by_role["market_snapshot_manifest"].get("source_ref"),
            label="market snapshot source_ref",
        )
        if (
            snapshot_ref != snapshot_source
            or pointer.get("snapshot_id") != snapshot.get("snapshot_id")
            or pointer.get("cutoff") != snapshot.get("cutoff")
        ):
            _fail("market pointer/snapshot manifest binding mismatch")
        expected_dataset_roles = {
            "cn_open_day_calendar_dataset",
            "market_bars_dataset",
        }
        bound_roles: set[str] = set()
        for item in _array(
            snapshot.get("dataset_bindings"),
            label="market snapshot dataset_bindings",
            maximum=2,
        ):
            binding = _mapping(item, label="market snapshot dataset binding")
            role = str(binding["role"])
            datasets = dataset_docs_by_role.get(role, [])
            if len(datasets) != 1:
                _fail(f"market snapshot dataset binding does not resolve: {role}")
            dataset_path, dataset = datasets[0]
            expected_ref = {
                "artifact_id": _document_id(dataset),
                "artifact_version": DATASET_MANIFEST_VERSION,
                "relative_path": dataset_path,
                "byte_sha256": document_byte_sha256(dataset),
                "semantic_sha256": dataset["semantic_sha256"],
            }
            if binding.get("dataset_manifest_ref") != expected_ref:
                _fail(f"market snapshot dataset binding mismatch: {role}")
            bound_roles.add(role)
        if bound_roles != expected_dataset_roles:
            _fail("market snapshot dataset role inventory mismatch")

    portfolio_inputs = object_documents.get("portfolio_required_inputs")
    if portfolio_inputs is not None:
        for role in ("risk_policy_snapshot", "macro_overlay", "markov_overlay"):
            document = object_documents.get(role)
            if document is not None and (
                document.get("strategy_id") != portfolio_inputs.get("strategy_id")
                or document.get("cutoff") != portfolio_inputs.get("cutoff")
            ):
                _fail(f"portfolio source identity mismatch: {role}")
        controllers = _mapping(
            portfolio_inputs.get("controllers"),
            label="portfolio controllers",
        )
        for controller in ("macro", "markov"):
            enabled = _mapping(
                controllers.get(controller),
                label=f"portfolio controller {controller}",
            ).get("enabled")
            overlay_role = f"{controller}_overlay"
            source = sources_by_role.get(overlay_role)
            if enabled is True and (
                source is None or source.get("availability") != "AVAILABLE"
            ):
                unavailable_required_roles.append(overlay_role)

    for role, datasets in dataset_docs_by_role.items():
        if len(datasets) != 1:
            _fail(f"runtime dataset role is duplicated: {role}")
        if sources_by_role.get(role, {}).get("availability") != "AVAILABLE":
            _fail(f"runtime dataset role lacks AVAILABLE source evidence: {role}")
        if role_rows[role].get("kind") != "DATASET":
            _fail(f"runtime role kind mismatch for dataset carrier: {role}")
    for role, catalogs in catalog_docs_by_role.items():
        if len(catalogs) != 1:
            _fail(f"runtime generation-catalog role is duplicated: {role}")
        if sources_by_role.get(role, {}).get("availability") != "AVAILABLE":
            _fail(f"runtime generation-catalog role lacks AVAILABLE source evidence: {role}")
        if (
            role_rows[role].get("kind") != "OBJECT"
            or expected_versions[role] != GENERATION_CATALOG_VERSION
        ):
            _fail(f"runtime role kind/schema mismatch for generation-catalog carrier: {role}")
    for role in set(table_links) | set(binding_links):
        if role_rows[role].get("kind") != "DATASET":
            _fail(f"runtime non-DATASET role has stage-bearing links: {role}")
        if len(table_links.get(role, [])) != 1 or len(binding_links.get(role, [])) != 1:
            _fail(f"runtime role has duplicate or missing stage-bearing links: {role}")

    unavailable = tuple(sorted(set(unavailable_required_roles)))
    if reject_required_roles:
        _fail(
            "runtime required source is unavailable and requires zero-write rejection: "
            f"{sorted(reject_required_roles)}"
        )
    disposition = (
        SourceAdmissionDisposition.SHADOW_RANK_COMPLETE_NO_PORTFOLIO
        if unavailable
        else SourceAdmissionDisposition.ADMITTED
    )
    return SourceAdmissionOutcome(
        disposition=disposition,
        locator=dict(source_locator),
        locator_byte_sha256=document_byte_sha256(source_locator),
        input_bindings=tuple(sorted(admitted_input_bindings)),
        unavailable_required_roles=unavailable,
    )


def _validate_stored_source_document_closure(
    *,
    dataset_manifests: Mapping[str, Mapping[str, Any]],
    observation_dispositions: Mapping[str, Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_manifest_path: str,
    generation_catalogs: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    source_binding_set: Mapping[str, Any],
    source_binding_set_path: str,
    source_locator: Mapping[str, Any],
    source_locator_path: str,
    stored_document_bytes: Mapping[str, bytes],
) -> None:
    document_items: list[tuple[str, Mapping[str, Any]]] = [
        *dataset_manifests.items(),
        *observation_dispositions.items(),
        (source_manifest_path, source_manifest),
        *generation_catalogs.items(),
        *summaries.items(),
        (source_binding_set_path, source_binding_set),
        (source_locator_path, source_locator),
    ]
    expected_documents: dict[str, Mapping[str, Any]] = {}
    for path, document in document_items:
        if path in expected_documents:
            _fail(f"runtime source document path is duplicated: {path}")
        expected_documents[path] = document
    if set(stored_document_bytes) != set(expected_documents):
        missing = sorted(set(expected_documents) - set(stored_document_bytes))
        extra = sorted(set(stored_document_bytes) - set(expected_documents))
        _fail(
            "runtime stored source document closure mismatch; " f"missing={missing}, extra={extra}"
        )
    for path, document in expected_documents.items():
        raw = stored_document_bytes[path]
        if type(raw) is not bytes:
            _fail(f"runtime stored source document must be bytes: {path}")
        try:
            parsed = load_canonical_resource(
                raw,
                label=f"runtime stored source document {path}",
            )
        except CanonicalContractError as exc:
            raise V17V2ValidationError(str(exc)) from exc
        if parsed != document:
            _fail(f"runtime stored source document bytes do not match mapping: {path}")


def admit_runtime_source_hash_dag(
    *,
    source_role_matrix: Mapping[str, Any],
    source_objects: Mapping[str, bytes | Mapping[str, Any]],
    dataset_manifests: Mapping[str, Mapping[str, Any]],
    observation_dispositions: Mapping[str, Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_manifest_path: str,
    generation_catalogs: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    source_binding_set: Mapping[str, Any],
    source_binding_set_path: str,
    source_locator: Mapping[str, Any],
    source_locator_path: str,
    stored_document_bytes: Mapping[str, bytes],
) -> SourceAdmissionOutcome:
    """Admit an exact COMPLETE source DAG or return a typed business outcome."""

    approved_role_matrix = require_runtime_usable_source_role_matrix(source_role_matrix)
    for path, raw in source_objects.items():
        if type(raw) is not bytes:
            _fail(f"runtime source object must be exact stored bytes: {path}")
    _validate_stored_source_document_closure(
        dataset_manifests=dataset_manifests,
        observation_dispositions=observation_dispositions,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        generation_catalogs=generation_catalogs,
        summaries=summaries,
        source_binding_set=source_binding_set,
        source_binding_set_path=source_binding_set_path,
        source_locator=source_locator,
        source_locator_path=source_locator_path,
        stored_document_bytes=stored_document_bytes,
    )
    locator = validate_source_hash_dag(
        source_role_matrix=approved_role_matrix,
        source_objects=source_objects,
        dataset_manifests=dataset_manifests,
        observation_dispositions=observation_dispositions,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        generation_catalogs=generation_catalogs,
        summaries=summaries,
        source_binding_set=source_binding_set,
        source_binding_set_path=source_binding_set_path,
        source_locator=source_locator,
        source_locator_path=source_locator_path,
    )
    return _admit_runtime_source_hash_dag_core(
        source_role_matrix=approved_role_matrix,
        source_objects=source_objects,
        dataset_manifests=dataset_manifests,
        source_manifest=source_manifest,
        generation_catalogs=generation_catalogs,
        source_binding_set=source_binding_set,
        source_locator=locator,
    )


def _evidence_id_array(
    value: Any,
    *,
    allowed: frozenset[str],
    label: str,
    require_nonempty: bool,
) -> list[str]:
    values = _array(value, label=label, maximum=LIMITS["max_evidence_refs"])
    normalized = [_opaque_id(item, label=f"{label}[{index}]") for index, item in enumerate(values)]
    if require_nonempty and not normalized:
        _fail(f"{label} must be nonempty")
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        _fail(f"{label} must be sorted and unique")
    if not set(normalized).issubset(allowed):
        _fail(f"{label} cites evidence outside the sealed request")
    return normalized


def validate_deep_research_chain(
    *,
    request: Mapping[str, Any],
    request_path: str,
    response: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate report structure and evidence closure to one sealed request."""

    _require_protocol_path(request_path, label="deep research request path")
    request_doc = validate_document_identity(
        request,
        expected_version=DEEP_RESEARCH_REQUEST_VERSION,
    )
    response_doc = validate_document_identity(
        response,
        expected_version=DEEP_RESEARCH_RESPONSE_VERSION,
    )
    _exact_keys(
        request_doc,
        frozenset(
            {
                "protocol_version",
                "version",
                "request_id",
                "run_id",
                "market",
                "cutoff",
                "source_locator_ref",
                "deterministic_result_ref",
                "template_resource_sha256",
                "symbol_ordering",
                "symbols",
                "evidence_by_symbol",
                "authority",
                "semantic_sha256",
            }
        ),
        label="deep research request",
    )
    _exact_keys(
        response_doc,
        frozenset(
            {
                "protocol_version",
                "version",
                "response_id",
                "run_id",
                "market",
                "cutoff",
                "request_ref",
                "review_ordering",
                "review_results",
                "generated_at",
                "received_at",
                "authority",
                "semantic_sha256",
            }
        ),
        label="deep research response",
    )
    source_locator_ref = _validate_unresolved_artifact_ref(
        _mapping(
            request_doc.get("source_locator_ref"),
            label="request.source_locator_ref",
        ),
        expected_version=SOURCE_LOCATOR_VERSION,
        label="request.source_locator_ref",
    )
    locator_id = _opaque_id(
        source_locator_ref.get("artifact_id"),
        label="request.source_locator_ref.artifact_id",
    )
    if source_locator_ref.get("relative_path") != (
        f"data/private/v17_sources/protocol-v2/locators/{locator_id}.json"
    ):
        _fail("request.source_locator_ref locator path mismatch")
    _validate_unresolved_artifact_ref(
        _mapping(
            request_doc.get("deterministic_result_ref"),
            label="request.deterministic_result_ref",
        ),
        label="request.deterministic_result_ref",
    )
    request_ref = _mapping(response_doc.get("request_ref"), label="response.request_ref")
    _validate_artifact_ref(
        request_ref,
        document=request_doc,
        expected_path=request_path,
        expected_version=DEEP_RESEARCH_REQUEST_VERSION,
        label="response.request_ref",
    )
    for field in ("run_id", "market", "cutoff"):
        if response_doc.get(field) != request_doc.get(field):
            _fail(f"response.{field} does not match request")

    symbols_raw = _array(
        request_doc.get("symbols"),
        label="request.symbols",
        maximum=LIMITS["max_candidates"],
    )
    symbols: list[str] = []
    for index, value in enumerate(symbols_raw):
        try:
            symbols.append(require_security_code(value, label=f"request.symbols[{index}]"))
        except IdentityContractError as exc:
            raise V17V2ValidationError(str(exc)) from exc
    if not symbols or len(set(symbols)) != len(symbols):
        _fail("request.symbols must be nonempty and unique")
    evidence_rows = _array(
        request_doc.get("evidence_by_symbol"),
        label="request.evidence_by_symbol",
        maximum=LIMITS["max_candidates"],
    )
    if [row.get("symbol") for row in evidence_rows if type(row) is dict] != symbols:
        _fail("request evidence_by_symbol does not follow symbol order")

    evidence_by_symbol: dict[str, dict[str, Mapping[str, Any]]] = {}
    globally_seen_evidence_ids: set[str] = set()
    evidence_count = 0
    for row_index, value in enumerate(evidence_rows):
        row = _mapping(value, label=f"request.evidence_by_symbol[{row_index}]")
        _exact_keys(
            row,
            frozenset({"symbol", "evidence_ready", "evidence"}),
            label=f"request.evidence_by_symbol[{row_index}]",
        )
        symbol = symbols[row_index]
        if type(row.get("evidence_ready")) is not bool:
            _fail(f"request.evidence_by_symbol[{row_index}].evidence_ready must be boolean")
        evidence_items = _array(
            row.get("evidence"),
            label=f"request evidence {symbol}",
            maximum=LIMITS["max_evidence_refs"],
        )
        if bool(evidence_items) != row["evidence_ready"]:
            _fail(f"request evidence readiness mismatch: {symbol}")
        try:
            evidence_count = checked_add(
                evidence_count,
                len(evidence_items),
                label="request global evidence count",
                maximum=LIMITS["max_evidence_refs"],
            )
        except ContractLimitError as exc:
            raise V17V2ValidationError(str(exc)) from exc
        symbol_evidence: dict[str, Mapping[str, Any]] = {}
        evidence_order: list[str] = []
        for evidence_index, evidence_value in enumerate(evidence_items):
            evidence = _mapping(
                evidence_value,
                label=f"request evidence {symbol}[{evidence_index}]",
            )
            _exact_keys(
                evidence,
                frozenset({"evidence_id", "kind", "object_ref", "layers", "coverage"}),
                label=f"request evidence {symbol}[{evidence_index}]",
            )
            evidence_id = _opaque_id(
                evidence.get("evidence_id"),
                label=f"request evidence {symbol}[{evidence_index}].evidence_id",
            )
            if evidence_id in globally_seen_evidence_ids:
                _fail(f"duplicate request evidence_id: {evidence_id}")
            globally_seen_evidence_ids.add(evidence_id)
            evidence_order.append(evidence_id)
            object_ref = _validate_unresolved_artifact_ref(
                _mapping(
                    evidence.get("object_ref"),
                    label=f"request evidence {symbol}[{evidence_index}].object_ref",
                ),
                label=f"request evidence {symbol}[{evidence_index}].object_ref",
            )
            if object_ref.get("artifact_id") != evidence_id:
                _fail(f"request evidence object identity mismatch: {evidence_id}")
            layers = _array(
                evidence.get("layers"),
                label=f"request evidence {symbol}[{evidence_index}].layers",
                maximum=len(_LAYER_ORDER),
            )
            expected_layers = sorted(
                layers,
                key=lambda item: (
                    _LAYER_ORDER.index(str(item)) if item in _LAYER_ORDER else len(_LAYER_ORDER)
                ),
            )
            if (
                not layers
                or layers != expected_layers
                or len(set(layers)) != len(layers)
                or any(item not in _LAYER_ORDER for item in layers)
            ):
                _fail(f"request evidence layer order invalid: {evidence_id}")
            coverage = _array(
                evidence.get("coverage"),
                label=f"request evidence {symbol}[{evidence_index}].coverage",
                maximum=len(_COVERAGE_ORDER),
            )
            expected_coverage = sorted(
                coverage,
                key=lambda item: (
                    _COVERAGE_ORDER.index(str(item))
                    if item in _COVERAGE_ORDER
                    else len(_COVERAGE_ORDER)
                ),
            )
            if (
                coverage != expected_coverage
                or len(set(coverage)) != len(coverage)
                or any(item not in _COVERAGE_ORDER for item in coverage)
            ):
                _fail(f"request evidence coverage order invalid: {evidence_id}")
            symbol_evidence[evidence_id] = evidence
        if evidence_order != sorted(evidence_order):
            _fail(f"request evidence order invalid: {symbol}")
        evidence_by_symbol[symbol] = symbol_evidence

    reviews = _array(
        response_doc.get("review_results"),
        label="response.review_results",
        maximum=LIMITS["max_deep_reviews"],
    )
    if [row.get("symbol") for row in reviews if type(row) is dict] != symbols:
        _fail("response reviews do not follow request symbol order")
    referenced_report_paths: set[str] = set()
    for index, value in enumerate(reviews):
        review = _mapping(value, label=f"review_results[{index}]")
        symbol = _string(review.get("symbol"), label=f"review_results[{index}].symbol")
        try:
            require_security_code(symbol, label=f"review_results[{index}].symbol")
        except IdentityContractError as exc:
            raise V17V2ValidationError(str(exc)) from exc
        if review.get("status") == "UNAVAILABLE":
            _exact_keys(
                review,
                frozenset({"symbol", "status", "reason"}),
                label=f"review_results[{index}]",
            )
            _string(review.get("reason"), label=f"review_results[{index}].reason")
            continue
        if review.get("status") != "COMPLETE":
            _fail(f"review_results[{index}].status invalid")
        _exact_keys(
            review,
            frozenset({"symbol", "status", "research_report_ref"}),
            label=f"review_results[{index}]",
        )
        if not evidence_by_symbol[symbol]:
            _fail(f"complete review lacks sealed evidence: {symbol}")
        report_ref = _mapping(
            review.get("research_report_ref"),
            label=f"review_results[{index}].research_report_ref",
        )
        report_path = _reference_path(
            report_ref,
            label=f"review_results[{index}].research_report_ref",
        )
        if report_path in referenced_report_paths:
            _fail(f"research report reused: {report_path}")
        referenced_report_paths.add(report_path)
        report = reports.get(report_path)
        if report is None:
            _fail(f"research report does not resolve: {report_path}")
        report_doc = validate_document_identity(
            report,
            expected_version=DEEP_RESEARCH_REPORT_VERSION,
        )
        _exact_keys(
            report_doc,
            frozenset(
                {
                    "protocol_version",
                    "version",
                    "report_id",
                    "request_ref",
                    "run_id",
                    "market",
                    "cutoff",
                    "symbol",
                    "template_resource_sha256",
                    "evidence_refs",
                    "coverage",
                    "layers",
                    "signals",
                    "severe_red_flags",
                    "generated_at",
                    "authority",
                    "semantic_sha256",
                }
            ),
            label="deep research report",
        )
        _validate_artifact_ref(
            report_ref,
            document=report_doc,
            expected_path=report_path,
            expected_version=DEEP_RESEARCH_REPORT_VERSION,
            label=f"review_results[{index}].research_report_ref",
        )
        if _mapping(report_doc.get("request_ref"), label="report.request_ref") != request_ref:
            _fail("research report request_ref mismatch")
        for field in ("run_id", "market", "cutoff"):
            if report_doc.get(field) != request_doc.get(field):
                _fail(f"research report {field} mismatch")
        if report_doc.get("symbol") != symbol:
            _fail("research report symbol mismatch")
        if report_doc.get("template_resource_sha256") != request_doc.get(
            "template_resource_sha256"
        ):
            _fail("research report template binding mismatch")

        allowed_evidence = frozenset(evidence_by_symbol[symbol])
        cited_evidence: set[str] = set()
        coverage_items = _array(
            report_doc.get("coverage"),
            label="research report coverage",
            maximum=len(_COVERAGE_ORDER),
        )
        if (
            tuple(
                _mapping(item, label="research report coverage item").get("area")
                for item in coverage_items
            )
            != _COVERAGE_ORDER
        ):
            _fail("research report coverage order or completeness mismatch")
        for coverage_index, value in enumerate(coverage_items):
            item = _mapping(value, label=f"research report coverage[{coverage_index}]")
            _exact_keys(
                item,
                frozenset({"area", "conclusion", "evidence_ids"}),
                label=f"research report coverage[{coverage_index}]",
            )
            _string(
                item.get("conclusion"),
                label=f"research report coverage[{coverage_index}].conclusion",
            )
            cited_evidence.update(
                _evidence_id_array(
                    item.get("evidence_ids"),
                    allowed=allowed_evidence,
                    label=f"research report coverage[{coverage_index}].evidence_ids",
                    require_nonempty=True,
                )
            )

        layers = _array(
            report_doc.get("layers"),
            label="research report layers",
            maximum=len(_LAYER_ORDER),
        )
        if (
            tuple(_mapping(layer, label="research report layer").get("layer") for layer in layers)
            != _LAYER_ORDER
        ):
            _fail("research report layer order or completeness mismatch")
        for layer_index, value in enumerate(layers):
            item = _mapping(value, label=f"research report layers[{layer_index}]")
            _exact_keys(
                item,
                frozenset({"layer", "content", "evidence_ids"}),
                label=f"research report layers[{layer_index}]",
            )
            _string(
                item.get("content"),
                label=f"research report layers[{layer_index}].content",
            )
            cited_evidence.update(
                _evidence_id_array(
                    item.get("evidence_ids"),
                    allowed=allowed_evidence,
                    label=f"research report layers[{layer_index}].evidence_ids",
                    require_nonempty=True,
                )
            )

        signals = _mapping(report_doc.get("signals"), label="research report signals")
        if set(signals) != _SIGNAL_KEYS:
            _fail("research report six-signal set incomplete")
        for name in sorted(_SIGNAL_KEYS):
            item = _mapping(signals[name], label=f"research report signals.{name}")
            _exact_keys(
                item,
                frozenset({"signal", "evidence_ids"}),
                label=f"research report signals.{name}",
            )
            signal_value = item.get("signal")
            if type(signal_value) not in {int, float} or float(signal_value) not in _SIGNAL_VALUES:
                _fail(f"research report signal invalid: {name}")
            cited_evidence.update(
                _evidence_id_array(
                    item.get("evidence_ids"),
                    allowed=allowed_evidence,
                    label=f"research report signals.{name}.evidence_ids",
                    require_nonempty=True,
                )
            )

        red_flags = _array(
            report_doc.get("severe_red_flags"),
            label="research report severe_red_flags",
            maximum=len(_RED_FLAG_ORDER),
        )
        if (
            tuple(
                _mapping(item, label="research report red flag").get("flag") for item in red_flags
            )
            != _RED_FLAG_ORDER
        ):
            _fail("research report red flag order or completeness mismatch")
        for flag_index, value in enumerate(red_flags):
            item = _mapping(value, label=f"research report severe_red_flags[{flag_index}]")
            _exact_keys(
                item,
                frozenset({"flag", "triggered", "evidence_ids"}),
                label=f"research report severe_red_flags[{flag_index}]",
            )
            if type(item.get("triggered")) is not bool:
                _fail(f"research report severe_red_flags[{flag_index}].triggered invalid")
            evidence_ids = _evidence_id_array(
                item.get("evidence_ids"),
                allowed=allowed_evidence,
                label=f"research report severe_red_flags[{flag_index}].evidence_ids",
                require_nonempty=bool(item["triggered"]),
            )
            cited_evidence.update(evidence_ids)

        expected_evidence_refs = [
            _mapping(
                evidence_by_symbol[symbol][evidence_id].get("object_ref"),
                label=f"request evidence for {symbol}:{evidence_id}",
            )
            for evidence_id in sorted(cited_evidence)
        ]
        actual_evidence_refs = _array(
            report_doc.get("evidence_refs"),
            label="research report evidence_refs",
            maximum=LIMITS["max_evidence_refs"],
        )
        for evidence_index, evidence_ref in enumerate(actual_evidence_refs):
            _validate_unresolved_artifact_ref(
                _mapping(
                    evidence_ref,
                    label=f"research report evidence_refs[{evidence_index}]",
                ),
                label=f"research report evidence_refs[{evidence_index}]",
            )
        if actual_evidence_refs != expected_evidence_refs:
            _fail("research report evidence binding mismatch")
    if referenced_report_paths != set(reports):
        _fail("research report closure mismatch")
    return dict(response_doc)


def _validate_package_binding_rows(
    values: Any,
    *,
    id_field: str,
    path_pattern: re.Pattern[str],
    label: str,
) -> list[tuple[str, str, str]]:
    rows = _array(values, label=label, maximum=LIMITS["max_sources"])
    if not rows:
        _fail(f"{label} must be nonempty")
    order: list[tuple[str, str, str]] = []
    for index, value in enumerate(rows):
        row = _mapping(value, label=f"{label}[{index}]")
        _exact_keys(
            row,
            frozenset({id_field, "relative_path", "byte_sha256"}),
            label=f"{label}[{index}]",
        )
        identifier = _opaque_id(
            row.get(id_field),
            label=f"{label}[{index}].{id_field}",
        )
        path = _string(
            row.get("relative_path"),
            label=f"{label}[{index}].relative_path",
        )
        if (
            path.startswith("/")
            or "\\" in path
            or "//" in path
            or path.endswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or path_pattern.fullmatch(path) is None
        ):
            _fail(f"{label}[{index}].relative_path invalid")
        try:
            byte_sha = require_sha256(
                row.get("byte_sha256"),
                label=f"{label}[{index}].byte_sha256",
            )
        except IdentityContractError as exc:
            raise V17V2ValidationError(str(exc)) from exc
        order.append((identifier, path, byte_sha))
    if order != sorted(order):
        _fail(f"{label} is not in complete total order")
    for position, name in enumerate(("id", "path", "SHA-256")):
        values_at_position = [item[position] for item in order]
        if len(values_at_position) != len(set(values_at_position)):
            _fail(f"{label} has duplicate {name}")
    return order


def _transition_action(from_state: str | None, to_state: str) -> str:
    if from_state is None:
        if to_state != "PREPARED":
            _fail("ledger history must initialize at PREPARED")
        return "SHADOW_PREPARE"
    if from_state in {"PREPARED", "DETERMINISTIC_COMPLETE"}:
        return "SHADOW_PREPARE"
    if from_state == "DEEP_REQUEST_READY":
        return "SHADOW_RECEIVE"
    if from_state in {"DEEP_RESPONSE_RECEIVED", "PORTFOLIO_COMPLETE"}:
        return "SHADOW_FINALIZE"
    _fail(f"ledger cannot transition from immutable state: {from_state}")


def validate_shadow_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one ledger snapshot and its internal history.

    This function validates the frozen package/module bindings and the
    snapshot's internal predecessor declarations.  Historical integrity
    requires :func:`validate_shadow_ledger_chain`, which receives and hashes
    every stored predecessor byte string.
    """

    ledger_doc = validate_document_identity(
        ledger,
        expected_version=SHADOW_LEDGER_VERSION,
    )
    _exact_keys(
        ledger_doc,
        frozenset(
            {
                "protocol_version",
                "version",
                "run_id",
                "strategy_id",
                "market",
                "cutoff",
                "state",
                "sequence",
                "action",
                "checkpoint",
                "created_at",
                "updated_at",
                "previous_ledger_sha256",
                "locator_binding",
                "contract_bindings",
                "implementation_bindings",
                "input_bindings",
                "artifacts",
                "history",
                "authority",
                "semantic_sha256",
            }
        ),
        label="shadow ledger",
    )
    if ledger_doc.get("state") not in _ALL_STATES:
        _fail("ledger state invalid")
    sequence = ledger_doc.get("sequence")
    if type(sequence) is not int or sequence < 0 or sequence >= LIMITS["max_ledger_history"]:
        _fail("ledger sequence invalid")
    if ledger_doc.get("checkpoint") != "INITIALIZED":
        _fail("ledger checkpoint must be INITIALIZED")

    locator_binding = _mapping(
        ledger_doc.get("locator_binding"),
        label="ledger.locator_binding",
    )
    _exact_keys(
        locator_binding,
        frozenset({"locator_id", "locator_ref"}),
        label="ledger.locator_binding",
    )
    locator_id = _opaque_id(
        locator_binding.get("locator_id"),
        label="ledger.locator_binding.locator_id",
    )
    locator_ref = _validate_unresolved_artifact_ref(
        _mapping(
            locator_binding.get("locator_ref"),
            label="ledger.locator_binding.locator_ref",
        ),
        expected_version=SOURCE_LOCATOR_VERSION,
        label="ledger.locator_binding.locator_ref",
    )
    if (
        locator_ref.get("artifact_id") != locator_id
        or locator_ref.get("relative_path")
        != f"data/private/v17_sources/protocol-v2/locators/{locator_id}.json"
    ):
        _fail("ledger locator binding identity mismatch")

    contract_bindings = _mapping(
        ledger_doc.get("contract_bindings"),
        label="ledger.contract_bindings",
    )
    _exact_keys(
        contract_bindings,
        frozenset({"package_manifest_sha256", "resource_bindings", "schema_bindings"}),
        label="ledger.contract_bindings",
    )
    try:
        expected_contract_bindings = expected_ledger_contract_bindings()
    except PackageResourceError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    if (
        contract_bindings.get("package_manifest_sha256")
        != expected_contract_bindings["package_manifest_sha256"]
    ):
        _fail("ledger package manifest binding mismatch")
    _validate_package_binding_rows(
        contract_bindings.get("resource_bindings"),
        id_field="binding_id",
        path_pattern=_PACKAGE_RESOURCE_PATH_RE,
        label="ledger.contract_bindings.resource_bindings",
    )
    _validate_package_binding_rows(
        contract_bindings.get("schema_bindings"),
        id_field="binding_id",
        path_pattern=_PACKAGE_SCHEMA_PATH_RE,
        label="ledger.contract_bindings.schema_bindings",
    )
    if (
        contract_bindings.get("resource_bindings")
        != expected_contract_bindings["resource_bindings"]
    ):
        _fail("ledger frozen resource binding inventory mismatch")
    if contract_bindings.get("schema_bindings") != expected_contract_bindings["schema_bindings"]:
        _fail("ledger frozen schema binding inventory mismatch")
    _validate_package_binding_rows(
        ledger_doc.get("implementation_bindings"),
        id_field="module_id",
        path_pattern=_PACKAGE_MODULE_PATH_RE,
        label="ledger.implementation_bindings",
    )
    try:
        expected_implementation_bindings = expected_ledger_implementation_bindings()
    except PackageResourceError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    if ledger_doc.get("implementation_bindings") != expected_implementation_bindings:
        _fail("ledger implementation binding inventory mismatch")

    input_bindings = _array(
        ledger_doc.get("input_bindings"),
        label="ledger.input_bindings",
        maximum=LIMITS["max_sources"],
    )
    input_order: list[tuple[str, str, str]] = []
    for index, value in enumerate(input_bindings):
        binding = _mapping(value, label=f"ledger.input_bindings[{index}]")
        _exact_keys(
            binding,
            frozenset({"role", "artifact_ref"}),
            label=f"ledger.input_bindings[{index}]",
        )
        role = _reject_forbidden_role(
            binding.get("role"),
            label=f"ledger.input_bindings[{index}].role",
        )
        artifact_ref = _validate_unresolved_artifact_ref(
            _mapping(
                binding.get("artifact_ref"),
                label=f"ledger.input_bindings[{index}].artifact_ref",
            ),
            label=f"ledger.input_bindings[{index}].artifact_ref",
        )
        input_order.append(
            (
                role,
                str(artifact_ref["relative_path"]),
                str(artifact_ref["byte_sha256"]),
            )
        )
    if input_order != sorted(input_order):
        _fail("ledger input_bindings are not canonically ordered")
    if len({item[0] for item in input_order}) != len(input_order):
        _fail("ledger input binding roles must be unique")
    input_binding_sha256s = sorted(item[2] for item in input_order)

    artifacts = _array(
        ledger_doc.get("artifacts"),
        label="ledger.artifacts",
        maximum=LIMITS["max_ledger_artifacts"],
    )
    artifact_order: list[tuple[str, int, str, str]] = []
    artifact_roles_by_sequence: dict[int, list[str]] = {}
    artifact_states_by_sequence: dict[int, set[str]] = {}
    for index, value in enumerate(artifacts):
        binding = _mapping(value, label=f"ledger.artifacts[{index}]")
        _exact_keys(
            binding,
            frozenset({"role", "artifact_ref", "sequence", "state"}),
            label=f"ledger.artifacts[{index}]",
        )
        role = _reject_forbidden_role(
            binding.get("role"),
            label=f"ledger.artifacts[{index}].role",
        )
        artifact_sequence = binding.get("sequence")
        if (
            type(artifact_sequence) is not int
            or artifact_sequence < 0
            or artifact_sequence > sequence
        ):
            _fail(f"ledger.artifacts[{index}].sequence invalid")
        artifact_state = binding.get("state")
        if artifact_state not in _ALL_STATES:
            _fail(f"ledger.artifacts[{index}].state invalid")
        artifact_ref = _validate_unresolved_artifact_ref(
            _mapping(
                binding.get("artifact_ref"),
                label=f"ledger.artifacts[{index}].artifact_ref",
            ),
            label=f"ledger.artifacts[{index}].artifact_ref",
        )
        artifact_order.append(
            (
                role,
                artifact_sequence,
                str(artifact_ref["relative_path"]),
                str(artifact_ref["byte_sha256"]),
            )
        )
        artifact_roles_by_sequence.setdefault(artifact_sequence, []).append(role)
        artifact_states_by_sequence.setdefault(artifact_sequence, set()).add(str(artifact_state))
    if artifact_order != sorted(artifact_order):
        _fail("ledger artifacts are not canonically ordered")
    if len({item[0] for item in artifact_order}) != len(artifact_order):
        _fail("ledger artifact roles must be unique")

    history = _array(
        ledger_doc.get("history"),
        label="ledger.history",
        maximum=LIMITS["max_ledger_history"],
    )
    if len(history) != sequence + 1:
        _fail("ledger history length does not match sequence")
    previous_state: str | None = None
    previous_at: datetime | None = None
    attempt_ids: set[str] = set()
    for index, value in enumerate(history):
        entry = _mapping(value, label=f"ledger.history[{index}]")
        _exact_keys(
            entry,
            frozenset(
                {
                    "sequence",
                    "attempt_id",
                    "action",
                    "acceptance_checkpoint",
                    "from_state",
                    "to_state",
                    "at",
                    "expected_ledger_sha256",
                    "input_binding_sha256s",
                    "artifact_roles",
                }
            ),
            label=f"ledger.history[{index}]",
        )
        if entry.get("sequence") != index:
            _fail("ledger history sequence mismatch")
        attempt_id = _opaque_id(
            entry.get("attempt_id"),
            label=f"ledger.history[{index}].attempt_id",
        )
        if attempt_id in attempt_ids:
            _fail("ledger history attempt_id must be unique")
        attempt_ids.add(attempt_id)
        if entry.get("acceptance_checkpoint") != "INITIALIZED":
            _fail("ledger history acceptance checkpoint must be INITIALIZED")
        from_state = entry.get("from_state")
        to_state = entry.get("to_state")
        if from_state != previous_state:
            _fail("ledger history from_state chain mismatch")
        if type(to_state) is not str or to_state not in _ALL_STATES:
            _fail("ledger history to_state invalid")
        if index == 0:
            if from_state is not None or to_state != "PREPARED":
                _fail("ledger history initial state mismatch")
        else:
            if type(from_state) is not str or to_state not in _TRANSITIONS.get(
                from_state, frozenset()
            ):
                _fail("ledger history transition invalid")
        expected_action = _transition_action(
            str(from_state) if from_state is not None else None,
            to_state,
        )
        if entry.get("action") != expected_action:
            _fail("ledger history action/transition mismatch")
        expected_ledger_sha = _sha256_or_empty(
            entry.get("expected_ledger_sha256"),
            label=f"ledger.history[{index}].expected_ledger_sha256",
        )
        if index == 0 and expected_ledger_sha != "EMPTY":
            _fail("ledger history sequence zero must bind EMPTY")
        if index > 0 and expected_ledger_sha == "EMPTY":
            _fail("ledger history successor must bind predecessor SHA-256")
        observed_input_sha256s = _array(
            entry.get("input_binding_sha256s"),
            label=f"ledger.history[{index}].input_binding_sha256s",
            maximum=LIMITS["max_sources"],
        )
        for sha_index, sha in enumerate(observed_input_sha256s):
            try:
                require_sha256(
                    sha,
                    label=(f"ledger.history[{index}]." f"input_binding_sha256s[{sha_index}]"),
                )
            except IdentityContractError as exc:
                raise V17V2ValidationError(str(exc)) from exc
        if observed_input_sha256s != input_binding_sha256s:
            _fail("ledger history input binding SHA-256 set mismatch")
        artifact_roles = _array(
            entry.get("artifact_roles"),
            label=f"ledger.history[{index}].artifact_roles",
            maximum=LIMITS["max_ledger_artifacts"],
        )
        expected_roles = sorted(artifact_roles_by_sequence.get(index, []))
        if artifact_roles != expected_roles:
            _fail("ledger history artifact role binding mismatch")
        if artifact_states_by_sequence.get(index, {to_state}) != {to_state}:
            _fail("ledger artifact state/history mismatch")
        at = _rfc3339_instant(
            entry.get("at"),
            label=f"ledger.history[{index}].at",
        )
        if previous_at is not None and at < previous_at:
            _fail("ledger history timestamp regressed")
        previous_at = at
        previous_state = to_state

    if previous_state != ledger_doc.get("state"):
        _fail("ledger history does not end at current state")
    if ledger_doc.get("action") != history[-1].get("action"):
        _fail("ledger root action/history mismatch")
    if ledger_doc.get("created_at") != history[0].get("at"):
        _fail("ledger created_at/history mismatch")
    if ledger_doc.get("updated_at") != history[-1].get("at"):
        _fail("ledger updated_at/history mismatch")
    previous_ledger_sha = _sha256_or_empty(
        ledger_doc.get("previous_ledger_sha256"),
        label="ledger.previous_ledger_sha256",
    )
    if sequence == 0:
        if previous_ledger_sha != "EMPTY":
            _fail("ledger sequence zero previous SHA-256 must be EMPTY")
    elif previous_ledger_sha != history[-1].get("expected_ledger_sha256"):
        _fail("ledger predecessor/history mismatch")
    return dict(ledger_doc)


def _load_stored_shadow_ledger(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes:
        _fail(f"{label} must be bytes")
    try:
        document = load_canonical_resource(raw, label=label)
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    if type(document) is not dict:
        _fail(f"{label} root must be an object")
    return validate_shadow_ledger(document)


def _validate_shadow_ledger_successor_documents(
    predecessor: Mapping[str, Any],
    predecessor_raw: bytes,
    successor: Mapping[str, Any],
) -> None:
    predecessor_sequence = predecessor.get("sequence")
    if type(predecessor_sequence) is not int:
        _fail("predecessor ledger sequence invalid")
    if successor.get("sequence") != predecessor_sequence + 1:
        _fail("ledger successor sequence mismatch")
    if predecessor.get("state") in _TERMINAL_STATES:
        _fail("immutable terminal ledger cannot have a successor")
    predecessor_sha = hashlib.sha256(predecessor_raw).hexdigest()
    if successor.get("previous_ledger_sha256") != predecessor_sha:
        _fail("ledger successor predecessor byte SHA-256 mismatch")
    successor_history = _array(
        successor.get("history"),
        label="successor ledger history",
        maximum=LIMITS["max_ledger_history"],
    )
    predecessor_history = _array(
        predecessor.get("history"),
        label="predecessor ledger history",
        maximum=LIMITS["max_ledger_history"],
    )
    if successor_history[:-1] != predecessor_history:
        _fail("ledger successor history prefix mismatch")
    if successor_history[-1].get("expected_ledger_sha256") != predecessor_sha:
        _fail("ledger successor history does not bind predecessor bytes")
    if successor_history[-1].get("from_state") != predecessor.get("state"):
        _fail("ledger successor state boundary mismatch")
    for field in (
        "run_id",
        "strategy_id",
        "market",
        "cutoff",
        "created_at",
        "locator_binding",
        "contract_bindings",
        "implementation_bindings",
        "input_bindings",
    ):
        if successor.get(field) != predecessor.get(field):
            _fail(f"ledger successor changed immutable field: {field}")
    predecessor_artifacts = _array(
        predecessor.get("artifacts"),
        label="predecessor ledger artifacts",
        maximum=LIMITS["max_ledger_artifacts"],
    )
    successor_artifacts = _array(
        successor.get("artifacts"),
        label="successor ledger artifacts",
        maximum=LIMITS["max_ledger_artifacts"],
    )
    retained = [
        artifact
        for artifact in successor_artifacts
        if _mapping(artifact, label="successor artifact").get("sequence") <= predecessor_sequence
    ]
    if retained != predecessor_artifacts:
        _fail("ledger successor artifact prefix mismatch")


def validate_shadow_ledger_successor(
    *,
    predecessor_ledger_bytes: bytes,
    successor_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash one stored predecessor and validate one CAS successor snapshot."""

    predecessor = _load_stored_shadow_ledger(
        predecessor_ledger_bytes,
        label="predecessor shadow ledger",
    )
    successor = validate_shadow_ledger(successor_ledger)
    _validate_shadow_ledger_successor_documents(
        predecessor,
        predecessor_ledger_bytes,
        successor,
    )
    return successor


def validate_shadow_ledger_chain(
    ledger_bytes: Sequence[bytes],
) -> dict[str, Any]:
    """Validate the complete byte-exact ledger chain from sequence zero."""

    if isinstance(ledger_bytes, (str, bytes, bytearray)) or not isinstance(
        ledger_bytes,
        Sequence,
    ):
        _fail("shadow ledger chain must be a sequence of stored byte strings")
    chain = list(ledger_bytes)
    if not chain or len(chain) > LIMITS["max_ledger_history"]:
        _fail("shadow ledger chain length invalid")
    documents: list[dict[str, Any]] = []
    for index, raw in enumerate(chain):
        document = _load_stored_shadow_ledger(
            raw,
            label=f"shadow ledger chain[{index}]",
        )
        if document.get("sequence") != index:
            _fail("shadow ledger chain must start at sequence zero and be contiguous")
        if index:
            _validate_shadow_ledger_successor_documents(
                documents[-1],
                chain[index - 1],
                document,
            )
        documents.append(document)
    if len(documents) != int(documents[-1]["sequence"]) + 1:
        _fail("shadow ledger chain does not cover complete history")
    return documents[-1]


def validate_action_failure_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate receipts that may exist only after INITIALIZED acceptance."""

    receipt_doc = validate_document_identity(
        receipt,
        expected_version=ACTION_FAILURE_RECEIPT_VERSION,
    )
    _exact_keys(
        receipt_doc,
        frozenset(
            {
                "protocol_version",
                "version",
                "receipt_id",
                "receipt_path",
                "run_id",
                "action",
                "acceptance_checkpoint",
                "status",
                "reason_code",
                "detail",
                "expected_ledger_sha256",
                "observed_ledger_sha256",
                "durably_committed",
                "write_effect",
                "created_at",
                "authority",
                "semantic_sha256",
            }
        ),
        label="action failure receipt",
    )
    receipt_id = _opaque_id(receipt_doc.get("receipt_id"), label="receipt_id")
    run_id = _opaque_id(receipt_doc.get("run_id"), label="run_id")
    if receipt_doc.get("receipt_path") != (
        "results/v17_shadow/protocol-v2/runs/" f"{run_id}/receipts/{receipt_id}.json"
    ):
        _fail("action failure receipt path mismatch")
    if receipt_doc.get("action") not in _ACTIONS:
        _fail("action failure receipt action invalid")
    if receipt_doc.get("acceptance_checkpoint") != "INITIALIZED":
        _fail("action failure receipt acceptance checkpoint must be INITIALIZED")
    _opaque_id(receipt_doc.get("reason_code"), label="reason_code")
    _string(receipt_doc.get("detail"), label="detail")
    _string(receipt_doc.get("created_at"), label="created_at")
    _sha256_or_empty(
        receipt_doc.get("expected_ledger_sha256"),
        label="expected_ledger_sha256",
    )
    observed = receipt_doc.get("observed_ledger_sha256")
    if observed is not None:
        _sha256_or_empty(observed, label="observed_ledger_sha256")
    semantics = {
        "UNPUBLISHED_NOT_COMMITTED": (False, "RECEIPT_ONLY"),
        "POST_COMMIT_UNCERTAIN": (True, "LEDGER_COMMITTED"),
        "TERMINAL_UNPUBLISHED": (True, "TERMINAL_COMMITTED"),
    }
    expected = semantics.get(str(receipt_doc.get("status")))
    if expected is None:
        _fail("action failure receipt status invalid")
    if (
        receipt_doc.get("durably_committed"),
        receipt_doc.get("write_effect"),
    ) != expected:
        _fail("action failure receipt status/write semantics mismatch")
    return dict(receipt_doc)


def _validate_shadow_terminal_chain_admitted(
    *,
    ledger_bytes: bytes,
    predecessor_ledger_bytes: Sequence[bytes],
    ledger_path: str,
    output_bytes: bytes,
    output_path: str,
    latest_pointer: Mapping[str, Any],
    previous_pointer_bytes: bytes | None,
    source_admission: SourceAdmissionOutcome,
) -> dict[str, Any]:
    """Validate a terminal chain after exact source admission has succeeded."""

    ledger_doc = _load_stored_shadow_ledger(
        ledger_bytes,
        label="terminal shadow ledger",
    )
    chain_doc = validate_shadow_ledger_chain([*predecessor_ledger_bytes, ledger_bytes])
    if chain_doc != ledger_doc:
        _fail("terminal ledger does not match validated byte chain")
    try:
        output = load_canonical_resource(
            output_bytes,
            label="terminal shadow output",
        )
    except CanonicalContractError as exc:
        raise V17V2ValidationError(str(exc)) from exc
    if type(output) is not dict:
        _fail("terminal shadow output root must be an object")
    output_doc = validate_document_identity(output, expected_version=SHADOW_OUTPUT_VERSION)
    rank_output = validate_rank_output(
        _mapping(output_doc.get("rank_output"), label="output.rank_output")
    )
    for field in ("run_id", "strategy_id", "market", "cutoff"):
        if rank_output.get(field) != output_doc.get(field):
            _fail(f"rank output identity mismatch: {field}")
    portfolio_value = output_doc.get("portfolio_output")
    if portfolio_value is not None:
        portfolio_output = validate_portfolio_output(
            _mapping(portfolio_value, label="output.portfolio_output")
        )
        for field in ("run_id", "strategy_id", "market", "cutoff"):
            if portfolio_output.get(field) != output_doc.get(field):
                _fail(f"portfolio output identity mismatch: {field}")
    latest_doc = validate_document_identity(
        latest_pointer,
        expected_version=SHADOW_LATEST_POINTER_VERSION,
    )
    _exact_keys(
        latest_doc,
        frozenset(
            {
                "protocol_version",
                "version",
                "pointer_path",
                "run_id",
                "terminal_state",
                "ledger_ref",
                "terminal_output_ref",
                "previous_pointer_byte_sha256",
                "publication_mode",
                "published_at",
                "authority",
                "semantic_sha256",
            }
        ),
        label="shadow latest pointer",
    )
    run_id = _opaque_id(ledger_doc.get("run_id"), label="ledger.run_id")
    expected_ledger_path = f"results/v17_shadow/protocol-v2/runs/{run_id}/ledger.json"
    expected_output_path = f"results/v17_shadow/protocol-v2/outcomes/{run_id}.json"
    if ledger_path != expected_ledger_path:
        _fail("terminal ledger path/run_id mismatch")
    if output_path != expected_output_path:
        _fail("terminal output path/run_id mismatch")
    if type(source_admission) is not SourceAdmissionOutcome:
        _fail("terminal source admission outcome is invalid")
    locator = validate_document_identity(
        source_admission.locator,
        expected_version=SOURCE_LOCATOR_VERSION,
    )
    if document_byte_sha256(locator) != source_admission.locator_byte_sha256:
        _fail("terminal source admission locator byte SHA-256 mismatch")
    if locator.get("market") != ledger_doc.get("market") or locator.get("cutoff") != ledger_doc.get(
        "cutoff"
    ):
        _fail("terminal source locator identity does not match ledger")
    locator_created_at = _rfc3339_instant(
        locator.get("created_at"),
        label="terminal source locator created_at",
    )
    ledger_created_at = _rfc3339_instant(
        ledger_doc.get("created_at"),
        label="terminal ledger created_at",
    )
    if ledger_created_at < locator_created_at:
        _fail("terminal ledger predates admitted source locator")
    locator_id = _document_id(locator)
    locator_ref = {
        "artifact_id": locator_id,
        "artifact_version": SOURCE_LOCATOR_VERSION,
        "relative_path": ("data/private/v17_sources/protocol-v2/locators/" f"{locator_id}.json"),
        "byte_sha256": source_admission.locator_byte_sha256,
        "semantic_sha256": locator["semantic_sha256"],
    }
    if any(type(row) is not tuple or len(row) != 6 for row in source_admission.input_bindings):
        _fail("terminal source admission input binding row is invalid")
    if len({row[0] for row in source_admission.input_bindings}) != len(
        source_admission.input_bindings
    ):
        _fail("terminal source admission input binding roles are not unique")
    if source_admission.input_bindings != tuple(sorted(source_admission.input_bindings)):
        _fail("terminal source admission input bindings are not canonically ordered")
    expected_input_bindings: list[dict[str, Any]] = [
        {
            "role": role,
            "artifact_ref": {
                "artifact_id": artifact_id,
                "artifact_version": artifact_version,
                "relative_path": relative_path,
                "byte_sha256": byte_sha256,
                "semantic_sha256": semantic_sha256_value,
            },
        }
        for (
            role,
            artifact_id,
            artifact_version,
            relative_path,
            byte_sha256,
            semantic_sha256_value,
        ) in source_admission.input_bindings
    ]
    if not expected_input_bindings:
        _fail("terminal source admission input binding inventory is empty")
    for index, binding in enumerate(expected_input_bindings):
        _reject_forbidden_role(
            binding["role"],
            label=f"terminal source admission input_bindings[{index}].role",
        )
        _validate_unresolved_artifact_ref(
            binding["artifact_ref"],
            label=f"terminal source admission input_bindings[{index}].artifact_ref",
        )
    if ledger_doc.get("input_bindings") != expected_input_bindings:
        _fail("terminal ledger input bindings do not match admitted source DAG")
    ledger_locator_binding = _mapping(
        ledger_doc.get("locator_binding"),
        label="ledger.locator_binding",
    )
    if ledger_locator_binding != {
        "locator_id": locator_id,
        "locator_ref": locator_ref,
    }:
        _fail("terminal ledger locator does not match admitted source DAG")

    forbidden_state_paths = {
        ledger_path,
        output_path,
        "results/v17_shadow/protocol-v2/_latest/shadow.json",
    }
    forbidden_state_versions = {
        SHADOW_LEDGER_VERSION,
        SHADOW_OUTPUT_VERSION,
        SHADOW_LATEST_POINTER_VERSION,
    }
    for collection_name in ("input_bindings", "artifacts"):
        for index, item in enumerate(
            _array(
                ledger_doc.get(collection_name),
                label=f"ledger.{collection_name}",
                maximum=(
                    LIMITS["max_sources"]
                    if collection_name == "input_bindings"
                    else LIMITS["max_ledger_artifacts"]
                ),
            )
        ):
            artifact_ref = _mapping(
                _mapping(
                    item,
                    label=f"ledger.{collection_name}[{index}]",
                ).get("artifact_ref"),
                label=f"ledger.{collection_name}[{index}].artifact_ref",
            )
            if (
                artifact_ref.get("relative_path") in forbidden_state_paths
                or artifact_ref.get("artifact_version") in forbidden_state_versions
            ):
                _fail("terminal ledger contains a state-carrier artifact cycle")
    if ledger_doc.get("state") not in _TERMINAL_STATES:
        _fail("latest ledger is not terminal")
    if (
        source_admission.disposition is SourceAdmissionDisposition.SHADOW_RANK_COMPLETE_NO_PORTFOLIO
        and ledger_doc.get("state") != "SHADOW_RANK_COMPLETE_NO_PORTFOLIO"
    ):
        _fail("no-portfolio source admission has the wrong terminal state")
    if output_doc.get("terminal_state") != ledger_doc.get("state"):
        _fail("shadow output terminal state mismatch")
    if latest_doc.get("terminal_state") != ledger_doc.get("state"):
        _fail("latest pointer terminal state mismatch")
    if latest_doc.get("run_id") != run_id or output_doc.get("run_id") != run_id:
        _fail("terminal run_id mismatch")
    for field in ("strategy_id", "market", "cutoff"):
        if output_doc.get(field) != ledger_doc.get(field):
            _fail(f"terminal output {field} mismatch")
    ledger_updated_at = _rfc3339_instant(
        ledger_doc.get("updated_at"),
        label="terminal ledger updated_at",
    )
    output_generated_at = _rfc3339_instant(
        output_doc.get("generated_at"),
        label="terminal output generated_at",
    )
    if output_generated_at < ledger_updated_at:
        _fail("terminal output predates terminal ledger")
    if latest_doc.get("pointer_path") != ("results/v17_shadow/protocol-v2/_latest/shadow.json"):
        _fail("latest pointer path mismatch")
    if latest_doc.get("publication_mode") not in {"NORMAL", "REPAIR"}:
        _fail("latest pointer publication_mode invalid")
    previous_pointer_sha = _sha256_or_empty(
        latest_doc.get("previous_pointer_byte_sha256"),
        label="latest.previous_pointer_byte_sha256",
    )
    if latest_doc.get("publication_mode") == "REPAIR" and previous_pointer_sha == "EMPTY":
        _fail("REPAIR latest pointer requires a predecessor pointer")
    previous_published_at: datetime | None = None
    if previous_pointer_sha == "EMPTY":
        if previous_pointer_bytes is not None:
            _fail("latest pointer declares EMPTY but predecessor bytes were supplied")
    else:
        if type(previous_pointer_bytes) is not bytes:
            _fail("latest pointer predecessor bytes are required")
        try:
            previous_pointer = load_canonical_resource(
                previous_pointer_bytes,
                label="previous shadow latest pointer",
            )
        except CanonicalContractError as exc:
            raise V17V2ValidationError(str(exc)) from exc
        if type(previous_pointer) is not dict:
            _fail("previous latest pointer root must be an object")
        validate_document_identity(
            previous_pointer,
            expected_version=SHADOW_LATEST_POINTER_VERSION,
        )
        if hashlib.sha256(previous_pointer_bytes).hexdigest() != previous_pointer_sha:
            _fail("latest pointer predecessor byte SHA-256 mismatch")
        if previous_pointer_bytes == canonical_resource_bytes(latest_doc):
            _fail("latest pointer cannot bind itself as predecessor")
        previous_published_at = _rfc3339_instant(
            previous_pointer.get("published_at"),
            label="previous latest pointer published_at",
        )
    published_at = _rfc3339_instant(
        latest_doc.get("published_at"),
        label="latest.published_at",
    )
    if published_at < output_generated_at:
        _fail("latest pointer predates terminal output")
    if previous_published_at is not None and published_at < previous_published_at:
        _fail("latest pointer publication timestamp regressed")
    _validate_artifact_ref(
        _mapping(output_doc.get("ledger_ref"), label="output.ledger_ref"),
        document=ledger_doc,
        expected_path=ledger_path,
        expected_version=SHADOW_LEDGER_VERSION,
        label="output.ledger_ref",
    )
    _validate_artifact_ref(
        _mapping(latest_doc.get("ledger_ref"), label="latest.ledger_ref"),
        document=ledger_doc,
        expected_path=ledger_path,
        expected_version=SHADOW_LEDGER_VERSION,
        label="latest.ledger_ref",
    )
    _validate_artifact_ref(
        _mapping(
            latest_doc.get("terminal_output_ref"),
            label="latest.terminal_output_ref",
        ),
        document=output_doc,
        expected_path=output_path,
        expected_version=SHADOW_OUTPUT_VERSION,
        label="latest.terminal_output_ref",
    )
    output_locator_ref = _mapping(
        output_doc.get("source_locator_ref"),
        label="output.source_locator_ref",
    )
    if output_locator_ref != locator_ref:
        _fail("terminal output source locator does not match admitted source DAG")
    return dict(latest_doc)


def validate_shadow_terminal_chain(
    *,
    ledger_bytes: bytes,
    predecessor_ledger_bytes: Sequence[bytes],
    ledger_path: str,
    output_bytes: bytes,
    output_path: str,
    latest_pointer: Mapping[str, Any],
    previous_pointer_bytes: bytes | None,
    source_role_matrix: Mapping[str, Any],
    source_objects: Mapping[str, bytes | Mapping[str, Any]],
    dataset_manifests: Mapping[str, Mapping[str, Any]],
    observation_dispositions: Mapping[str, Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_manifest_path: str,
    generation_catalogs: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    source_binding_set: Mapping[str, Any],
    source_binding_set_path: str,
    source_locator: Mapping[str, Any],
    source_locator_path: str,
    stored_source_document_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    """Validate a terminal publication only after exact runtime source admission."""

    source_admission = admit_runtime_source_hash_dag(
        source_role_matrix=source_role_matrix,
        source_objects=source_objects,
        dataset_manifests=dataset_manifests,
        observation_dispositions=observation_dispositions,
        source_manifest=source_manifest,
        source_manifest_path=source_manifest_path,
        generation_catalogs=generation_catalogs,
        summaries=summaries,
        source_binding_set=source_binding_set,
        source_binding_set_path=source_binding_set_path,
        source_locator=source_locator,
        source_locator_path=source_locator_path,
        stored_document_bytes=stored_source_document_bytes,
    )
    return _validate_shadow_terminal_chain_admitted(
        ledger_bytes=ledger_bytes,
        predecessor_ledger_bytes=predecessor_ledger_bytes,
        ledger_path=ledger_path,
        output_bytes=output_bytes,
        output_path=output_path,
        latest_pointer=latest_pointer,
        previous_pointer_bytes=previous_pointer_bytes,
        source_admission=source_admission,
    )


__all__ = [
    "ACTION_FAILURE_RECEIPT_VERSION",
    "DATASET_MANIFEST_VERSION",
    "DATASET_RECORD_SCHEMA_REGISTRY_VERSION",
    "DATASET_SCHEMA_DIGEST_VERSION",
    "DATASET_SUMMARY_VERSION",
    "DEEP_RESEARCH_REPORT_VERSION",
    "DEEP_RESEARCH_REQUEST_VERSION",
    "DEEP_RESEARCH_RESPONSE_VERSION",
    "GENERATION_CATALOG_VERSION",
    "MACRO_OVERLAY_VERSION",
    "MARKET_POINTER_VERSION",
    "MARKET_SNAPSHOT_MANIFEST_VERSION",
    "MARKOV_OVERLAY_VERSION",
    "OBSERVATION_DISPOSITION_VERSION",
    "PROTOCOL_VERSION",
    "PORTFOLIO_OUTPUT_VERSION",
    "PORTFOLIO_REQUIRED_INPUTS_VERSION",
    "RANK_OUTPUT_VERSION",
    "RISK_POLICY_SNAPSHOT_VERSION",
    "SEMANTIC_SHA_FIELD",
    "SHADOW_LATEST_POINTER_VERSION",
    "SHADOW_LEDGER_VERSION",
    "SHADOW_OUTPUT_VERSION",
    "SOURCE_BINDING_SET_VERSION",
    "SOURCE_LOCATOR_VERSION",
    "SOURCE_MANIFEST_VERSION",
    "SOURCE_ROLE_MATRIX_VERSION",
    "SourceAdmissionDisposition",
    "SourceAdmissionOutcome",
    "SUPPORTED_DOCUMENT_VERSIONS",
    "V17V2ValidationError",
    "admit_runtime_source_hash_dag",
    "document_byte_sha256",
    "seal_semantic",
    "semantic_sha256",
    "require_runtime_usable_source_role_matrix",
    "require_runtime_usable_dataset_record_schema_registry",
    "validate_dataset_record_schema_registry",
    "validate_action_failure_receipt",
    "validate_dataset_manifest",
    "validate_deep_research_chain",
    "validate_document_identity",
    "validate_macro_overlay",
    "validate_market_pointer",
    "validate_market_snapshot_manifest",
    "validate_markov_overlay",
    "validate_portfolio_output",
    "validate_portfolio_required_inputs",
    "validate_rank_output",
    "validate_risk_policy_snapshot",
    "validate_semantic_seal",
    "validate_shadow_ledger",
    "validate_shadow_ledger_chain",
    "validate_shadow_ledger_successor",
    "validate_shadow_terminal_chain",
    "validate_source_hash_dag",
    "validate_source_role_matrix",
]

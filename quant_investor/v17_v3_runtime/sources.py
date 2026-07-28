"""Exact-byte admission and materialization of protocol-v3 source locators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
from io import BytesIO
import math
from typing import Any

from quant_investor.v17_v3_contract.canonical import (
    CanonicalContractError,
    canonical_bytes,
    load_canonical_resource,
)
from quant_investor.v17_v3_contract.identities import (
    IdentityContractError,
    require_casefold_unique,
    require_opaque_id,
    require_sha256,
    require_utc_cutoff,
)
from quant_investor.v17_v3_contract.resources import package_resource_session

from .authority import PROTOCOL_VERSION, authority_envelope
from .artifacts import artifact_reference, load_typed_artifact
from .redaction import assert_public_envelope_safe
from .storage import (
    PRIVATE_SOURCES_ROOT,
    SecureStore,
    StorageError,
    StorageSecurityError,
    canonical_relative_path,
)

_PARQUET_ROLES = frozenset(
    {
        "benchmark_total_return",
        "cn_open_day_calendar",
        "corporate_actions",
        "market_bars",
        "official_delisting_cash",
        "pit_fundamentals",
        "universe_membership",
    }
)


class SourceAdmissionError(ValueError):
    """A locator or its exact transitive inputs failed closed."""

    exit_code = 2


@dataclass(frozen=True)
class SourceReference:
    role: str
    relative_path: str
    byte_sha256: str
    required: bool
    artifact_ref: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AdmittedSources:
    locator_id: str
    strategy_id: str
    cutoff: str
    locator_path: str
    locator_byte_sha256: str
    closure_sha256: str
    references: tuple[SourceReference, ...]
    documents: Mapping[str, Any]
    raw_objects: Mapping[str, bytes]

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(reference.role for reference in self.references)

    def materialize(self, role: str) -> Any:
        """Return one admitted role; callers cannot supply replacement arrays."""

        if type(role) is not str:
            raise SourceAdmissionError("source role must be text")
        if role not in self.documents:
            raise SourceAdmissionError("required admitted role is unavailable")
        return self.documents[role]

    def reference_for_role(self, role: str) -> Mapping[str, Any]:
        for reference in self.references:
            if reference.role == role and reference.artifact_ref is not None:
                return reference.artifact_ref
        raise SourceAdmissionError("required admitted role reference is unavailable")

    def to_public_wire(self) -> dict[str, Any]:
        payload = {
            "version": f"{PROTOCOL_VERSION}.source-admission-result.v1",
            "status": "ADMITTED",
            "locator_id": self.locator_id,
            "locator_byte_sha256": self.locator_byte_sha256,
            "closure_sha256": self.closure_sha256,
            "role_count": len(self.references),
            "required_role_count": sum(reference.required for reference in self.references),
            **authority_envelope(),
        }
        assert_public_envelope_safe(payload)
        return payload


def _instant(value: Any, *, label: str) -> datetime:
    if type(value) is not str:
        raise SourceAdmissionError(f"{label} must be a timezone-aware instant")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceAdmissionError(f"{label} must be a timezone-aware instant") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SourceAdmissionError(f"{label} must be timezone-aware")
    return parsed


def _protocol_path(
    value: Any,
    *,
    label: str,
    require_source_root: bool = True,
) -> str:
    if type(value) is not str:
        raise SourceAdmissionError(f"{label} must be a path")
    try:
        path = canonical_relative_path(value)
    except StorageSecurityError as exc:
        raise SourceAdmissionError(f"{label} is not a governed V3 path") from exc
    if (
        require_source_root
        and path != PRIVATE_SOURCES_ROOT
        and PRIVATE_SOURCES_ROOT not in path.parents
    ):
        raise SourceAdmissionError(f"{label} is outside the V3 source root")
    return str(path)


def _reference(value: Any, *, index: int) -> SourceReference:
    if type(value) is not dict:
        raise SourceAdmissionError(f"roles[{index}] must be an object")
    role = value.get("role")
    relative_path = value.get("relative_path")
    byte_sha256 = value.get("byte_sha256", value.get("expected_sha256"))
    required = value.get("required", True)
    try:
        role = require_opaque_id(role, label=f"roles[{index}].role")
        byte_sha256 = require_sha256(
            byte_sha256,
            label=f"roles[{index}].byte_sha256",
        )
    except IdentityContractError as exc:
        raise SourceAdmissionError(str(exc)) from exc
    if type(required) is not bool:
        raise SourceAdmissionError(f"roles[{index}].required must be boolean")
    return SourceReference(
        role,
        _protocol_path(relative_path, label=f"roles[{index}].relative_path"),
        byte_sha256,
        required,
    )


def _references(locator: Mapping[str, Any]) -> tuple[SourceReference, ...]:
    raw = locator.get("roles", locator.get("bindings"))
    if type(raw) is dict:
        normalized: list[dict[str, Any]] = []
        for role, reference in raw.items():
            if type(reference) is not dict:
                raise SourceAdmissionError("locator roles must map to reference objects")
            normalized.append({"role": role, **reference})
        raw = normalized
    if type(raw) is not list or not raw:
        raise SourceAdmissionError("locator must contain a nonempty roles array")
    result = tuple(_reference(value, index=index) for index, value in enumerate(raw))
    roles = tuple(reference.role for reference in result)
    paths = tuple(reference.relative_path for reference in result)
    try:
        require_casefold_unique(roles, label="source roles")
    except IdentityContractError as exc:
        raise SourceAdmissionError(str(exc)) from exc
    if len(paths) != len(set(paths)) or len(paths) != len({path.casefold() for path in paths}):
        raise SourceAdmissionError("source paths contain a duplicate or casefold collision")
    if roles != tuple(sorted(roles, key=str.casefold)):
        raise SourceAdmissionError("source roles must be in ASCII-casefold order")
    return result


def _validate_artifact_if_registered(document: Mapping[str, Any]) -> None:
    version = document.get("version")
    if type(version) is not str or not version.startswith(f"{PROTOCOL_VERSION}."):
        return
    try:
        from quant_investor.v17_v3_contract import validate_artifact

        validate_artifact(document)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise SourceAdmissionError("source artifact failed its registered schema") from exc


def _validate_cutoff(document: Any, *, locator_cutoff: datetime, label: str) -> None:
    if not isinstance(document, Mapping):
        return
    for key in ("cutoff", "decision_cutoff", "available_at", "effective_at"):
        if key not in document:
            continue
        observed = _instant(document[key], label=f"{label}.{key}")
        if observed > locator_cutoff:
            raise SourceAdmissionError("source artifact exceeds locator cutoff")


def _canonical_document(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = load_canonical_resource(raw, label=label)
    except CanonicalContractError as exc:
        raise SourceAdmissionError("source closure contains noncanonical JSON") from exc
    if type(value) is not dict:
        raise SourceAdmissionError("source closure artifacts must be JSON objects")
    _validate_artifact_if_registered(value)
    return value


def _validate_official_delisting_cash(
    raw: bytes,
    *,
    locator_cutoff: datetime,
) -> None:
    """Validate the exact official delisting-cash Parquet truth table."""

    try:
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        table = parquet.read_table(source=BytesIO(raw))
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise SourceAdmissionError("official_delisting_cash parquet is unreadable") from exc
    expected_columns = (
        "symbol",
        "event_date",
        "cash_per_share",
        "announced_at",
        "available_at",
    )
    if tuple(table.column_names) != expected_columns:
        raise SourceAdmissionError("official_delisting_cash columns are not the closed schema")
    schema = table.schema
    cash_type = schema.field("cash_per_share").type
    if (
        schema.field("symbol").type != arrow.string()
        or schema.field("event_date").type != arrow.date32()
        or not (arrow.types.is_floating(cash_type) or arrow.types.is_decimal(cash_type))
    ):
        raise SourceAdmissionError("official_delisting_cash column types are invalid")
    for field_name in ("announced_at", "available_at"):
        observed_type = schema.field(field_name).type
        if not arrow.types.is_timestamp(observed_type) or observed_type.tz != "UTC":
            raise SourceAdmissionError("official_delisting_cash timestamps must be UTC")
    columns = {name: table[name].to_pylist() for name in expected_columns}
    if any(any(value is None for value in columns[name]) for name in expected_columns):
        raise SourceAdmissionError("official_delisting_cash contains null cells")
    seen: set[tuple[str, date]] = set()
    for index in range(table.num_rows):
        symbol = columns["symbol"][index]
        event_date = columns["event_date"][index]
        cash = columns["cash_per_share"][index]
        announced_at = columns["announced_at"][index]
        available_at = columns["available_at"][index]
        if (
            type(symbol) is not str
            or not symbol
            or symbol.strip() != symbol
            or type(event_date) is not date
            or not isinstance(announced_at, datetime)
            or not isinstance(available_at, datetime)
        ):
            raise SourceAdmissionError("official_delisting_cash row values are invalid")
        if hasattr(cash, "is_finite"):
            cash_is_finite = bool(cash.is_finite())
        else:
            try:
                cash_is_finite = math.isfinite(float(cash))
            except (TypeError, ValueError, OverflowError):
                cash_is_finite = False
        if not cash_is_finite or cash < 0:
            raise SourceAdmissionError("official_delisting_cash cash_per_share is invalid")
        primary_key = (symbol, event_date)
        if primary_key in seen:
            raise SourceAdmissionError("official_delisting_cash primary key is duplicated")
        seen.add(primary_key)
        if announced_at > available_at or available_at > locator_cutoff:
            raise SourceAdmissionError("official_delisting_cash cutoff ordering is invalid")


def _factor_field(document: Mapping[str, Any], name: str) -> Any:
    if name in document:
        return document[name]
    payload = document.get("payload")
    if isinstance(payload, Mapping):
        return payload.get(name)
    return None


def _validate_factor_baseline_bindings(
    *,
    documents: Mapping[str, Any],
    references: Sequence[SourceReference],
    locator_cutoff: datetime,
) -> None:
    inputs = documents.get("quant_preselection_inputs")
    readiness = documents.get("factor_governance_readiness")
    provisional = documents.get("provisional_factor_baseline")
    if not any(isinstance(value, Mapping) for value in (readiness, provisional)) and not (
        isinstance(inputs, Mapping) and _factor_field(inputs, "factor_baseline_mode") is not None
    ):
        return
    if not isinstance(inputs, Mapping) or not isinstance(readiness, Mapping):
        raise SourceAdmissionError("typed factor readiness/baseline closure is incomplete")
    reference_by_role = {
        reference.role: reference.artifact_ref
        for reference in references
        if reference.artifact_ref is not None
    }
    readiness_ref = reference_by_role.get("factor_governance_readiness")
    if not isinstance(readiness_ref, Mapping):
        raise SourceAdmissionError("factor readiness exact reference is unavailable")
    if (
        readiness.get("version") != "myquant.v17.v3.factor-governance-readiness.v1"
        or readiness.get("role") != "factor_governance_readiness"
    ):
        raise SourceAdmissionError("factor readiness artifact identity is invalid")
    source_as_of = readiness.get("source_as_of")
    if type(source_as_of) is not str:
        raise SourceAdmissionError("factor readiness source_as_of is invalid")
    try:
        source_date = date.fromisoformat(source_as_of)
    except ValueError as exc:
        raise SourceAdmissionError("factor readiness source_as_of is invalid") from exc
    readiness_age_days = (locator_cutoff.date() - source_date).days
    if readiness_age_days < 0 or readiness_age_days > 8:
        raise SourceAdmissionError("factor readiness exceeds the 8-day freshness limit")

    baseline_mode = _factor_field(inputs, "factor_baseline_mode")
    baseline_ref = _factor_field(inputs, "factor_baseline_ref")
    if baseline_mode not in {
        "PROVISIONAL_RESEARCH",
        "FACTOR_V4_PRODUCTION",
    } or not isinstance(baseline_ref, Mapping):
        raise SourceAdmissionError("factor baseline mode/reference is invalid")
    declared_readiness_ref = _factor_field(
        inputs,
        "factor_governance_readiness_ref",
    )
    if declared_readiness_ref is not None and (
        not isinstance(declared_readiness_ref, Mapping)
        or dict(declared_readiness_ref) != dict(readiness_ref)
    ):
        raise SourceAdmissionError("factor readiness exact binding mismatch")
    if baseline_mode == "PROVISIONAL_RESEARCH":
        provisional_ref = reference_by_role.get("provisional_factor_baseline")
        if (
            not isinstance(provisional, Mapping)
            or not isinstance(provisional_ref, Mapping)
            or provisional.get("version") != "myquant.v17.v3.provisional-factor-baseline.v1"
            or provisional.get("role") != "provisional_factor_baseline"
            or dict(baseline_ref) != dict(provisional_ref)
            or provisional.get("factor_governance_readiness_ref") != dict(readiness_ref)
        ):
            raise SourceAdmissionError("provisional factor baseline exact binding mismatch")
        return
    if dict(baseline_ref) != dict(readiness_ref):
        raise SourceAdmissionError("production factor baseline must bind exact readiness")
    if (
        readiness.get("readiness_status") != "FACTOR_V4_READY"
        or readiness.get("factor_governance_ready") is not True
        or readiness.get("activation_receipt_valid") is not True
        or not isinstance(readiness.get("production_factor_count"), int)
        or readiness["production_factor_count"] < 5
        or not isinstance(readiness.get("production_family_count"), int)
        or readiness["production_family_count"] < 3
    ):
        raise SourceAdmissionError("production factor readiness gate is not satisfied")


def _artifact_reference(
    value: Any,
    *,
    label: str,
    role: str,
) -> SourceReference:
    if type(value) is not dict:
        raise SourceAdmissionError(f"{label} must be an artifact reference")
    try:
        byte_sha256 = require_sha256(
            value.get("byte_sha256"),
            label=f"{label}.byte_sha256",
        )
    except IdentityContractError as exc:
        raise SourceAdmissionError(str(exc)) from exc
    return SourceReference(
        role=role,
        relative_path=_protocol_path(
            value.get("relative_path"),
            label=f"{label}.relative_path",
            require_source_root=False,
        ),
        byte_sha256=byte_sha256,
        required=True,
        artifact_ref=dict(value),
    )


def _manifest_references(
    store: SecureStore,
    *,
    locator: Mapping[str, Any],
    locator_cutoff: str,
    strategy_id: str,
) -> tuple[
    tuple[SourceReference, ...],
    dict[str, Any],
    bytes,
    SourceReference,
    dict[str, Any] | None,
    bytes | None,
    SourceReference | None,
]:
    manifest_ref_value = locator.get("source_manifest_ref")
    manifest_ref = _artifact_reference(
        manifest_ref_value,
        label="source_manifest_ref",
        role="source_manifest",
    )
    if not isinstance(manifest_ref_value, Mapping):
        raise SourceAdmissionError("source_manifest_ref must be an object")
    if (
        manifest_ref_value.get("cutoff") != locator_cutoff
        or manifest_ref_value.get("strategy_id") != strategy_id
    ):
        raise SourceAdmissionError("source manifest reference scope mismatch")
    try:
        manifest_raw = store.read(
            manifest_ref.relative_path,
            manifest_ref.byte_sha256,
        )
    except StorageError as exc:
        raise SourceAdmissionError("source manifest exact-byte read failed") from exc
    manifest = dict(_canonical_document(manifest_raw, label="source manifest"))
    if manifest.get("cutoff") != locator_cutoff or manifest.get("strategy_id") != strategy_id:
        raise SourceAdmissionError("source manifest scope mismatch")
    if artifact_reference(
        relative_path=manifest_ref.relative_path,
        document=manifest,
        raw=manifest_raw,
    ) != dict(manifest_ref_value):
        raise SourceAdmissionError("source manifest exact reference mismatch")
    rows = manifest.get("sources")
    if type(rows) is not list or not rows:
        raise SourceAdmissionError("source manifest has no role bindings")
    raw_manifest: dict[str, Any] | None = None
    raw_manifest_raw: bytes | None = None
    raw_manifest_ref: SourceReference | None = None
    if manifest.get("closure_kind") == "DERIVED_CLOSURE":
        parent_value = manifest.get("parent_raw_manifest_ref")
        raw_manifest_ref = _artifact_reference(
            parent_value,
            label="parent_raw_manifest_ref",
            role="raw_source_manifest",
        )
        if not isinstance(parent_value, Mapping):
            raise SourceAdmissionError("parent raw manifest reference must be an object")
        try:
            raw_manifest_raw = store.read(
                raw_manifest_ref.relative_path,
                raw_manifest_ref.byte_sha256,
            )
        except StorageError as exc:
            raise SourceAdmissionError("parent raw manifest exact-byte read failed") from exc
        raw_manifest = dict(
            _canonical_document(raw_manifest_raw, label="parent raw source manifest")
        )
        if (
            raw_manifest.get("version") != "myquant.v17.v3.source-manifest.v1"
            or raw_manifest.get("closure_kind") != "RAW"
            or raw_manifest.get("strategy_id") != strategy_id
            or raw_manifest.get("cutoff") != locator_cutoff
            or artifact_reference(
                relative_path=raw_manifest_ref.relative_path,
                document=raw_manifest,
                raw=raw_manifest_raw,
            )
            != dict(parent_value)
        ):
            raise SourceAdmissionError("parent raw manifest exact binding mismatch")
        raw_rows = raw_manifest.get("sources")
        if type(raw_rows) is not list or not raw_rows:
            raise SourceAdmissionError("parent raw manifest has no role bindings")
        phase = manifest.get("phase")
        raw_profile = raw_manifest.get("raw_profile")
        if (
            type(phase) is str
            and phase.startswith("SHADOW_CURRENT_")
            and raw_profile != "SHADOW_CURRENT"
        ):
            raise SourceAdmissionError("shadow-current locator requires SHADOW_CURRENT raw profile")
        if (
            type(phase) is not str or not phase.startswith("SHADOW_CURRENT_")
        ) and raw_profile == "SHADOW_CURRENT":
            raise SourceAdmissionError("historical locator cannot bind SHADOW_CURRENT raw profile")
        rows = [*raw_rows, *rows]
    references: list[SourceReference] = []
    for index, row in enumerate(rows):
        if type(row) is not dict:
            raise SourceAdmissionError("source binding must be an object")
        try:
            role = require_opaque_id(
                row.get("role"),
                label=f"source binding {index} role",
            )
        except IdentityContractError as exc:
            raise SourceAdmissionError(str(exc)) from exc
        reference_value = row.get("artifact_ref")
        reference = _artifact_reference(
            reference_value,
            label=f"source binding {index}.artifact_ref",
            role=role,
        )
        if not isinstance(reference_value, Mapping):
            raise SourceAdmissionError("source artifact reference must be an object")
        if (
            reference_value.get("strategy_id") != strategy_id
            or reference_value.get("cutoff") > locator_cutoff
        ):
            raise SourceAdmissionError("source artifact reference scope mismatch")
        references.append(reference)
    # RAW and DERIVED manifests are each independently ordered, but their
    # concatenation is not necessarily globally ordered.  Normalize only after
    # both registered manifests have been validated and combined.
    result = tuple(sorted(references, key=lambda reference: reference.role.casefold()))
    roles = tuple(reference.role for reference in result)
    paths = tuple(reference.relative_path for reference in result)
    try:
        require_casefold_unique(roles, label="source roles")
        from quant_investor.v17_v3_contract.policy import validate_source_roles

        validate_source_roles(manifest.get("phase"), roles)
    except IdentityContractError as exc:
        raise SourceAdmissionError("source role closure has duplicate roles") from exc
    except ValueError as exc:
        raise SourceAdmissionError("source role closure violates the phase matrix") from exc
    if roles != tuple(sorted(roles, key=str.casefold)):
        raise SourceAdmissionError("source roles must be in ASCII-casefold order")
    if len(paths) != len(set(paths)) or len(paths) != len({path.casefold() for path in paths}):
        raise SourceAdmissionError("source paths contain a duplicate or casefold collision")
    return (
        result,
        manifest,
        manifest_raw,
        manifest_ref,
        raw_manifest,
        raw_manifest_raw,
        raw_manifest_ref,
    )


@package_resource_session()
def admit_source_locator(
    store: SecureStore,
    *,
    locator_path: str,
    expected_locator_sha256: str,
    required_roles: Sequence[str] = (),
) -> AdmittedSources:
    """Load and validate an exact locator and its complete role closure."""

    if not isinstance(store, SecureStore):
        raise TypeError("store must be SecureStore")
    try:
        expected = require_sha256(
            expected_locator_sha256,
            label="expected source locator SHA-256",
        )
    except IdentityContractError as exc:
        raise SourceAdmissionError(str(exc)) from exc
    relative_locator = store.relative_from_path(locator_path)
    if (
        relative_locator != PRIVATE_SOURCES_ROOT
        and PRIVATE_SOURCES_ROOT not in relative_locator.parents
    ):
        raise SourceAdmissionError("source locator is outside the fixed V3 source root")
    try:
        locator_raw = store.read(relative_locator, expected)
    except StorageError as exc:
        raise SourceAdmissionError("source locator exact-byte read failed") from exc
    locator = _canonical_document(locator_raw, label="source locator")
    try:
        locator_id = require_opaque_id(locator.get("locator_id"), label="locator_id")
        strategy_id = require_opaque_id(locator.get("strategy_id"), label="strategy_id")
        cutoff = require_utc_cutoff(locator.get("cutoff"), label="locator cutoff")
    except IdentityContractError as exc:
        raise SourceAdmissionError(str(exc)) from exc
    declared_root = locator.get("source_root", str(PRIVATE_SOURCES_ROOT))
    if declared_root != str(PRIVATE_SOURCES_ROOT):
        raise SourceAdmissionError("locator source_root mismatch")
    declared_locator_path = locator.get("relative_path")
    if declared_locator_path is not None and declared_locator_path != str(relative_locator):
        raise SourceAdmissionError("locator path binding mismatch")

    if locator.get("version") != "myquant.v17.v3.source-locator.v1":
        raise SourceAdmissionError("source locator must be a registered V3 artifact")
    (
        references,
        manifest,
        manifest_raw,
        manifest_ref,
        raw_manifest,
        raw_manifest_raw,
        raw_manifest_ref,
    ) = _manifest_references(
        store,
        locator=locator,
        locator_cutoff=cutoff,
        strategy_id=strategy_id,
    )
    requested = tuple(required_roles)
    try:
        require_casefold_unique(requested, label="required source roles")
    except IdentityContractError as exc:
        raise SourceAdmissionError(str(exc)) from exc
    available = frozenset(reference.role for reference in references if reference.required)
    missing = sorted(set(requested).difference(available))
    if missing:
        raise SourceAdmissionError("locator is missing required admitted roles")

    locator_cutoff = _instant(cutoff, label="locator cutoff")
    documents: dict[str, Any] = {}
    raw_objects: dict[str, bytes] = {}
    closure_records: list[dict[str, Any]] = []
    for reference in references:
        try:
            raw = store.read(reference.relative_path, reference.byte_sha256)
        except StorageError as exc:
            raise SourceAdmissionError("source closure exact-byte read failed") from exc
        if reference.role == "official_delisting_cash":
            _validate_official_delisting_cash(
                raw,
                locator_cutoff=locator_cutoff,
            )
        if reference.role in _PARQUET_ROLES and raw.startswith(b"PAR1") and raw.endswith(b"PAR1"):
            document: Any = raw
        else:
            document = _canonical_document(
                raw,
                label=f"source role {reference.role}",
            )
        if isinstance(document, Mapping):
            _validate_cutoff(
                document,
                locator_cutoff=locator_cutoff,
                label=f"source role {reference.role}",
            )
            declared_role = document.get("role")
            if declared_role is not None and declared_role != reference.role:
                raise SourceAdmissionError("source artifact role binding mismatch")
            if (
                reference.artifact_ref is not None
                and document.get("version", "").startswith(f"{PROTOCOL_VERSION}.")
                and artifact_reference(
                    relative_path=reference.relative_path,
                    document=document,
                    raw=raw,
                )
                != dict(reference.artifact_ref)
            ):
                raise SourceAdmissionError("source artifact exact reference mismatch")
        documents[reference.role] = document
        raw_objects[reference.role] = raw
        closure_records.append(
            {
                "role": reference.role,
                "relative_path": reference.relative_path,
                "byte_sha256": reference.byte_sha256,
                "required": reference.required,
            }
        )
    documents["source_locator"] = locator
    raw_objects["source_locator"] = locator_raw
    documents["source_manifest"] = manifest
    raw_objects["source_manifest"] = manifest_raw
    closure_records.insert(
        0,
        {
            "role": "source_manifest",
            "relative_path": manifest_ref.relative_path,
            "byte_sha256": manifest_ref.byte_sha256,
            "required": True,
        },
    )
    if raw_manifest is not None and raw_manifest_raw is not None and raw_manifest_ref is not None:
        documents["raw_source_manifest"] = raw_manifest
        raw_objects["raw_source_manifest"] = raw_manifest_raw
        closure_records.insert(
            0,
            {
                "role": "raw_source_manifest",
                "relative_path": raw_manifest_ref.relative_path,
                "byte_sha256": raw_manifest_ref.byte_sha256,
                "required": True,
            },
        )
    _validate_factor_baseline_bindings(
        documents=documents,
        references=references,
        locator_cutoff=locator_cutoff,
    )
    branch_locator_refs = [
        document.get("source_locator_ref")
        for role in ("quant_branch_output", "fundamental_branch_output")
        if isinstance((document := documents.get(role)), Mapping)
    ]
    declared_lineage_ref = locator.get("preselection_locator_ref")
    if branch_locator_refs or declared_lineage_ref is not None:
        if not isinstance(declared_lineage_ref, Mapping):
            raise SourceAdmissionError("derived locator has no preselection lineage")
        if any(
            not isinstance(reference, Mapping) or dict(reference) != dict(declared_lineage_ref)
            for reference in branch_locator_refs
        ):
            raise SourceAdmissionError("branch source-locator lineage mismatch")
        lineage_ref = declared_lineage_ref
        admitted_pool = documents.get("initial_pool_output")
        if isinstance(admitted_pool, Mapping) and admitted_pool.get("source_locator_ref") != dict(
            lineage_ref
        ):
            raise SourceAdmissionError("initial-pool source-locator lineage mismatch")
        lineage_path = _protocol_path(
            lineage_ref.get("relative_path"),
            label="branch source_locator_ref.relative_path",
        )
        try:
            lineage_raw = store.read(
                lineage_path,
                require_sha256(
                    lineage_ref.get("byte_sha256"),
                    label="branch source locator byte SHA-256",
                ),
            )
            lineage_locator = load_typed_artifact(
                lineage_raw,
                label="branch source locator",
                expected_version="myquant.v17.v3.source-locator.v1",
            )
        except (IdentityContractError, StorageError, ValueError) as exc:
            raise SourceAdmissionError("branch source-locator lineage exact read failed") from exc
        if (
            lineage_locator.get("strategy_id") != strategy_id
            or lineage_locator.get("cutoff") != cutoff
            or artifact_reference(
                relative_path=lineage_path,
                document=lineage_locator,
                raw=lineage_raw,
            )
            != dict(lineage_ref)
        ):
            raise SourceAdmissionError("branch source-locator lineage binding mismatch")
        lineage_admission = admit_source_locator(
            store,
            locator_path=lineage_path,
            expected_locator_sha256=str(lineage_ref["byte_sha256"]),
        )
        if (
            lineage_admission.strategy_id != strategy_id
            or lineage_admission.cutoff != cutoff
            or lineage_admission.reference_for_role("quant_preselection_inputs")
            != next(
                (
                    reference.artifact_ref
                    for reference in references
                    if reference.role == "quant_preselection_inputs"
                ),
                None,
            )
        ):
            raise SourceAdmissionError("preselection source closure does not match analyze inputs")
        documents["preselection_source_locator"] = lineage_locator
        raw_objects["preselection_source_locator"] = lineage_raw
        documents["preselection_source_manifest"] = lineage_admission.documents["source_manifest"]
        raw_objects["preselection_source_manifest"] = lineage_admission.raw_objects[
            "source_manifest"
        ]
        if "raw_source_manifest" in lineage_admission.documents:
            documents["preselection_raw_source_manifest"] = lineage_admission.documents[
                "raw_source_manifest"
            ]
            raw_objects["preselection_raw_source_manifest"] = lineage_admission.raw_objects[
                "raw_source_manifest"
            ]
        closure_records.insert(
            0,
            {
                "role": "preselection_source_locator",
                "relative_path": lineage_path,
                "byte_sha256": str(lineage_ref["byte_sha256"]),
                "required": True,
            },
        )
        closure_records.extend(
            {
                "role": f"preselection:{reference.role}",
                "relative_path": reference.relative_path,
                "byte_sha256": reference.byte_sha256,
                "required": reference.required,
            }
            for reference in lineage_admission.references
        )
    closure_sha = hashlib.sha256(canonical_bytes(closure_records)).hexdigest()
    declared_closure = locator.get("closure_sha256")
    if declared_closure is not None and declared_closure != closure_sha:
        raise SourceAdmissionError("source locator closure SHA-256 mismatch")
    return AdmittedSources(
        locator_id=locator_id,
        strategy_id=strategy_id,
        cutoff=cutoff,
        locator_path=str(relative_locator),
        locator_byte_sha256=expected,
        closure_sha256=closure_sha,
        references=references,
        documents=documents,
        raw_objects=raw_objects,
    )


__all__ = [
    "AdmittedSources",
    "SourceAdmissionError",
    "SourceReference",
    "admit_source_locator",
]

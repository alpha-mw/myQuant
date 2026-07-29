"""Pure cross-document validators for the V17 v4 contract scaffold."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import re
from types import MappingProxyType
from typing import Any, Final

from .canonical import (
    CanonicalContractError,
    canonical_bytes,
    load_canonical_resource,
    validate_semantic_sha,
)
from .identities import (
    IdentityContractError,
    require_opaque_id,
    require_sha256,
    require_utc_timestamp,
)

PROTOCOL_VERSION: Final = "myquant.v17.v4"
_REF_KEYS: Final = {
    "artifact_id",
    "artifact_version",
    "byte_sha256",
    "cutoff",
    "relative_path",
    "semantic_sha256",
    "strategy_id",
}
_AUTHORITY_KEYS: Final = {
    "broker",
    "execution",
    "formal_research_publication",
    "order",
    "research_runtime_default",
    "trade",
}
_FACTOR_CONTROL_REF_KEYS: Final = {
    "artifact_schema",
    "byte_sha256",
    "relative_path",
    "schema_version",
    "semantic_sha256",
}
_FACTOR_ACTIVE_SET_SCHEMA: Final = (
    "factor-governance-production-control.active-set-pointer.schema.v1"
)
_FACTOR_CONTROL_RECEIPT_SCHEMA: Final = (
    "factor-governance-production-control.activation-receipt.schema.v1"
)
_FACTOR_ARTIFACT_REF_SCHEMA: Final = "factor-governance-production-control.artifact-ref.v1"
SHADOW_RUN_RESEARCH_VERSION: Final = "myquant.v17.v4.shadow-run.v2"
_CN_SYMBOL_RE: Final = re.compile(
    r"^[0-9]{6}\.(?:BJ|SH|SZ)$",
    re.ASCII,
)


class ArtifactContractError(ValueError):
    """Raised when a v4 artifact fails semantic validation."""

    exit_code = 2


@dataclass(frozen=True)
class ValidatedArtifact:
    version: str
    strategy_id: str
    semantic_sha256: str
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class FormalActivationReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FormalActivationIntentArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FormalActivationRejectionArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FormalActivePointerArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FormalOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DefaultEligibilityReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DefaultEligibilityIntentArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DefaultEligiblePointerArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class CanaryReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class CanaryTransitionIntentArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class CanaryPointerArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DualRunComparisonArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class HistoricalCanaryPolicyArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class PitGenerationCatalogArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class PitCatalogPointerArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class PreselectLocatorArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class InitialPoolOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class BranchOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ResearchQuantBranchOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ResearchFactorShadowAssertionArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class TotalReturnLabelsArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FusionTop24Artifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class OfficialEvidenceArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class IssuerDossierArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class EventScanArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DeepEvidenceBundleArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DeepAssessmentManifestArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DeepEvidenceBundleV2Artifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class OfficialEvidenceV2Artifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class IssuerDossierV2Artifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class EventScanV2Artifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ShadowReadinessArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ShadowRunArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ShadowSessionRefArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class HoldingsSnapshotArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class PortfolioRiskPolicyArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class PretradePermissionsArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class PortfolioOverlayArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class PortfolioOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class RegimeEvidenceArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class CalibrationOriginInventoryArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class CalibrationReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FusionPromotionReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class PublicSurfaceCompatibilityReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class PublicRunDTOArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class CanaryPublicSnapshotArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ValidationReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class RollbackDrillReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ResearchShadowArtifact(ValidatedArtifact):
    """Schema-closed additive artifact with no formal or execution authority."""


def _authority(
    value: Any,
    *,
    formal_research_publication: bool,
    research_runtime_default: bool = False,
) -> None:
    if type(value) is not dict or set(value) != _AUTHORITY_KEYS:
        raise ArtifactContractError("v4 authority envelope shape mismatch")
    expected = {
        "broker": False,
        "execution": False,
        "formal_research_publication": formal_research_publication,
        "order": False,
        "research_runtime_default": research_runtime_default,
        "trade": False,
    }
    if value != expected:
        raise ArtifactContractError("v4 authority exceeds the artifact state authority ceiling")


def _common(
    payload: Mapping[str, Any],
    artifact_class: type[ValidatedArtifact],
    *,
    formal_research_publication: bool,
    schema_checked: bool,
) -> ValidatedArtifact:
    if type(payload) is not dict:
        raise ArtifactContractError("v4 artifact must be an object")
    if not schema_checked:
        from .schema_validation import validate_schema_version

        validate_schema_version(payload, payload.get("version"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ArtifactContractError("v4 artifact protocol mismatch")
    version_value = payload.get("version")
    if type(version_value) is str and "v17.v3" in version_value:
        raise ArtifactContractError("v3 artifact identity cannot be relabelled as v4")
    try:
        sealed = validate_semantic_sha(payload)
        version = require_opaque_id(version_value, label="artifact version")
        strategy_id = require_opaque_id(payload.get("strategy_id"), label="strategy_id")
        digest = require_sha256(payload.get("semantic_sha256"), label="semantic_sha256")
    except (CanonicalContractError, IdentityContractError) as exc:
        raise ArtifactContractError(str(exc)) from exc
    _authority(
        payload.get("authority"),
        formal_research_publication=formal_research_publication,
    )
    return artifact_class(
        version,
        strategy_id,
        digest,
        MappingProxyType(sealed),
    )


def validate_research_shadow_artifact(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ResearchShadowArtifact:
    """Validate an additive Shadow-only artifact without widening formal lanes."""

    result = _common(
        payload,
        ResearchShadowArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    required_flags = {
        "canary_evidence_eligible": False,
        "formal_activation_eligible": False,
        "shadow_only": True,
    }
    for key, expected in required_flags.items():
        if key in payload and payload[key] is not expected:
            raise ArtifactContractError(f"research Shadow artifact {key} mismatch")
    if (
        "performance_evidence_eligible" in payload
        and payload["performance_evidence_eligible"] is not False
    ):
        raise ArtifactContractError(
            "research Shadow artifact performance evidence authority mismatch"
        )
    return result


def _ref(
    value: Any,
    *,
    strategy_id: str,
    cutoff: str | None = None,
    expected_version: str | None = None,
    expected_versions: Sequence[str] | None = None,
    version_prefix: str | None = None,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _REF_KEYS:
        raise ArtifactContractError(f"{label} shape mismatch")
    try:
        require_opaque_id(value["artifact_id"], label=f"{label}.artifact_id")
        version = require_opaque_id(
            value["artifact_version"],
            label=f"{label}.artifact_version",
        )
        require_sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
        require_sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256")
        require_utc_timestamp(value["cutoff"], label=f"{label}.cutoff")
    except IdentityContractError as exc:
        raise ArtifactContractError(str(exc)) from exc
    if value["strategy_id"] != strategy_id:
        raise ArtifactContractError(f"{label} strategy mismatch")
    if cutoff is not None and value["cutoff"] > cutoff:
        raise ArtifactContractError(f"{label} is after the enclosing cutoff")
    if (
        expected_version is not None
        and expected_versions is not None
        and expected_version not in expected_versions
    ):
        raise ArtifactContractError(f"{label} expected version mismatch")
    if expected_versions is not None and version not in expected_versions:
        raise ArtifactContractError(f"{label} artifact version mismatch")
    if expected_version is not None and version != expected_version:
        raise ArtifactContractError(f"{label} artifact version mismatch")
    if version_prefix is not None and not version.startswith(version_prefix):
        raise ArtifactContractError(f"{label} protocol identity mismatch")
    if version.startswith("myquant.v17.v3."):
        raise ArtifactContractError(f"{label} cannot relabel a v3 artifact as v4")
    path = value["relative_path"]
    if type(path) is not str or path.startswith("/") or ".." in path.split("/") or "\\" in path:
        raise ArtifactContractError(f"{label} path is unsafe")
    return dict(value)


def _factor_control_ref(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _FACTOR_CONTROL_REF_KEYS:
        raise ArtifactContractError(f"{label} shape mismatch")
    if (
        value["artifact_schema"] != _FACTOR_ACTIVE_SET_SCHEMA
        or value["schema_version"] != _FACTOR_ARTIFACT_REF_SCHEMA
    ):
        raise ArtifactContractError(f"{label} schema mismatch")
    try:
        require_sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
        require_sha256(
            value["semantic_sha256"],
            label=f"{label}.semantic_sha256",
        )
    except IdentityContractError as exc:
        raise ArtifactContractError(str(exc)) from exc
    path = value["relative_path"]
    if (
        type(path) is not str
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ArtifactContractError(f"{label} path is unsafe")
    return dict(value)


def _read_exact_ref(
    reference: Mapping[str, Any],
    *,
    artifact_loader: Callable[[Mapping[str, str]], bytes] | None,
    expected_version: str,
    strategy_id: str,
    cutoff: str,
    label: str,
) -> dict[str, Any]:
    normalized_ref = _ref(
        reference,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=expected_version,
        label=label,
    )
    if artifact_loader is None:
        raise ArtifactContractError(f"{label} requires canonical artifact readback")
    string_ref = {field: str(value) for field, value in normalized_ref.items()}
    try:
        raw = artifact_loader(string_ref)
    except Exception as exc:
        raise ArtifactContractError(f"{label} readback failed") from exc
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != normalized_ref["byte_sha256"]:
        raise ArtifactContractError(f"{label} byte SHA mismatch")
    try:
        document = load_canonical_resource(raw, label=label)
        sealed = validate_semantic_sha(document)
    except CanonicalContractError as exc:
        raise ArtifactContractError(f"{label} canonical readback failed") from exc
    identity = next(
        (
            sealed.get(field)
            for field in (
                "inventory_id",
                "receipt_id",
            )
            if field in sealed
        ),
        None,
    )
    if (
        sealed.get("version") != expected_version
        or sealed.get("strategy_id") != strategy_id
        or sealed.get("cutoff") != normalized_ref["cutoff"]
        or sealed.get("semantic_sha256") != normalized_ref["semantic_sha256"]
        or identity != normalized_ref["artifact_id"]
    ):
        raise ArtifactContractError(f"{label} document binding mismatch")
    return sealed


def _refs(
    values: Any,
    *,
    strategy_id: str,
    cutoff: str | None,
    label: str,
    expected_version: str | None = None,
    path_ordered: bool = True,
) -> list[dict[str, Any]]:
    if type(values) is not list:
        raise ArtifactContractError(f"{label} must be an array")
    result = [
        _ref(
            value,
            strategy_id=strategy_id,
            cutoff=cutoff,
            expected_version=expected_version,
            label=f"{label}[{index}]",
        )
        for index, value in enumerate(values)
    ]
    identities = [(row["relative_path"], row["byte_sha256"], row["artifact_id"]) for row in result]
    if len(identities) != len(set(identities)):
        raise ArtifactContractError(f"{label} contains duplicate references")
    if path_ordered and identities != sorted(identities):
        raise ArtifactContractError(f"{label} must be path ordered")
    return result


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is not str:
        raise ArtifactContractError(f"{label} must be a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ArtifactContractError(f"{label} is not a decimal") from exc
    if not result.is_finite():
        raise ArtifactContractError(f"{label} must be finite")
    return result


def _cas(
    payload: Mapping[str, Any],
    *,
    success: bool,
    label: str,
) -> None:
    expected = payload["expected_pointer_sha256"]
    observed = payload["observed_pointer_sha256"]
    proposed = payload["proposed_pointer_sha256"]
    post = payload["post_readback_sha256"]
    if observed != expected:
        raise ArtifactContractError(f"{label} observed pointer differs from expected CAS")
    if success:
        if post != proposed:
            raise ArtifactContractError(f"{label} post-readback differs from proposed bytes")
    elif post != observed:
        raise ArtifactContractError(f"{label} rejection cannot claim a pointer write")


def _require_evidence_contains(
    evidence: Sequence[Mapping[str, Any]],
    required: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    evidence_keys = {
        (row["artifact_version"], row["relative_path"], row["byte_sha256"]) for row in evidence
    }
    required_keys = {
        (row["artifact_version"], row["relative_path"], row["byte_sha256"]) for row in required
    }
    if not required_keys.issubset(evidence_keys):
        raise ArtifactContractError(f"{label} omits an explicit transition reference")


def validate_formal_activation_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FormalActivationReceiptArtifact:
    result = _common(
        payload,
        FormalActivationReceiptArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FormalActivationReceiptArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    intent_ref = _ref(
        payload["intent_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.formal-activation-intent.v1",
        label="intent_ref",
    )
    pointer_ref = _ref(
        payload["pointer_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.formal-active-pointer.v1",
        label="pointer_ref",
    )
    evidence = _refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="evidence_refs",
    )
    _require_evidence_contains(
        evidence,
        [intent_ref, pointer_ref],
        label="formal activation completion evidence",
    )
    if pointer_ref["byte_sha256"] != payload["proposed_pointer_sha256"]:
        raise ArtifactContractError(
            "formal activation pointer reference differs from proposed bytes"
        )
    _cas(payload, success=True, label="formal activation completion")
    return result


def validate_formal_activation_intent(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FormalActivationIntentArtifact:
    result = _common(
        payload,
        FormalActivationIntentArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FormalActivationIntentArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    require_utc_timestamp(payload["created_at"], label="created_at")
    if (payload["from_state"] == "V15_DEFAULT") != (payload["expected_pointer_sha256"] == "EMPTY"):
        raise ArtifactContractError("formal intent state does not match pointer prevalue")
    expected_versions = {
        "formal_output_ref": "myquant.v17.v4.formal-output.v1",
        "source_locator_ref": "myquant.v17.v4.preselect-locator.v1",
        "quant_calibration_receipt_ref": ("myquant.v17.v4.calibration-receipt.v1"),
        "fundamental_calibration_receipt_ref": ("myquant.v17.v4.calibration-receipt.v1"),
        "fusion_promotion_receipt_ref": ("myquant.v17.v4.fusion-promotion-receipt.v1"),
        "deep_bundle_ref": "myquant.v17.v4.deep-evidence-bundle.v1",
        "portfolio_output_ref": "myquant.v17.v4.portfolio-output.v1",
        "holdings_snapshot_ref": "myquant.v17.v4.holdings-snapshot.v1",
        "risk_policy_ref": "myquant.v17.v4.portfolio-risk-policy.v1",
        "macro_overlay_ref": "myquant.v17.v4.portfolio-overlay.v1",
        "markov_overlay_ref": "myquant.v17.v4.portfolio-overlay.v1",
        "factor_control_active_set_ref": (
            "factor-governance-production-control." "active-set-pointer.schema.v1"
        ),
        "factor_control_activation_receipt_ref": (
            "factor-governance-production-control." "activation-receipt.schema.v1"
        ),
        "package_manifest_ref": "myquant.v17.v4.package-manifest.v1",
        "runtime_manifest_ref": ("myquant.v17.v4.runtime-build-manifest.v1"),
    }
    explicit = [
        _ref(
            payload[field],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version=version,
            label=field,
        )
        for field, version in expected_versions.items()
    ]
    evidence = _refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="evidence_refs",
    )
    _require_evidence_contains(
        evidence,
        explicit,
        label="formal activation intent evidence",
    )
    return result


def validate_formal_activation_rejection(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FormalActivationRejectionArtifact:
    result = _common(
        payload,
        FormalActivationRejectionArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FormalActivationRejectionArtifact)
    require_utc_timestamp(payload["recorded_at"], label="recorded_at")
    _refs(
        payload["attempted_evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=None,
        label="attempted_evidence_refs",
    )
    if payload["to_state"] != payload["from_state"]:
        raise ArtifactContractError("formal rejection cannot change state")
    return result


def validate_formal_active_pointer(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FormalActivePointerArtifact:
    result = _common(
        payload,
        FormalActivePointerArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FormalActivePointerArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["intent_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.formal-activation-intent.v1",
        label="intent_ref",
    )
    if payload["updated_at"] < cutoff:
        raise ArtifactContractError("formal pointer updated_at precedes cutoff")
    return result


def validate_formal_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FormalOutputArtifact:
    result = _common(
        payload,
        FormalOutputArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FormalOutputArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    evidence = _refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="evidence_refs",
    )
    if not evidence:
        raise ArtifactContractError("formal output requires evidence")
    return result


def _source_file_refs(
    values: Any,
    *,
    label: str,
) -> tuple[dict[str, str], ...]:
    if type(values) is not list or not values:
        raise ArtifactContractError(f"{label} must be a nonempty array")
    normalized: list[dict[str, str]] = []
    for index, value in enumerate(values):
        if type(value) is not dict or set(value) != {
            "byte_sha256",
            "relative_path",
        }:
            raise ArtifactContractError(f"{label}[{index}] shape mismatch")
        relative_path = value["relative_path"]
        if (
            type(relative_path) is not str
            or not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise ArtifactContractError(f"{label}[{index}] path is noncanonical")
        try:
            relative_path.encode("ascii")
            byte_sha256 = require_sha256(
                value["byte_sha256"],
                label=f"{label}[{index}].byte_sha256",
            )
        except (UnicodeEncodeError, IdentityContractError) as exc:
            raise ArtifactContractError(f"{label}[{index}] is invalid") from exc
        normalized.append(
            {
                "byte_sha256": byte_sha256,
                "relative_path": relative_path,
            }
        )
    expected = sorted(
        normalized,
        key=lambda row: row["relative_path"],
    )
    if normalized != expected or len(
        {row["relative_path"].casefold() for row in normalized}
    ) != len(normalized):
        raise ArtifactContractError(f"{label} must be unique and ASCII path ordered")
    return tuple(normalized)


def validate_public_surface_compatibility_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PublicSurfaceCompatibilityReceiptArtifact:
    result = _common(
        payload,
        PublicSurfaceCompatibilityReceiptArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(
        result,
        PublicSurfaceCompatibilityReceiptArtifact,
    )
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created_at = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    if created_at < cutoff:
        raise ArtifactContractError("public surface receipt predates its run cutoff")
    _ref(
        payload["formal_active_pointer_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.formal-active-pointer.v1",
        label="formal_active_pointer_ref",
    )
    _source_file_refs(
        payload["surface_file_refs"],
        label="surface_file_refs",
    )
    _source_file_refs(
        payload["v15_compatibility_refs"],
        label="v15_compatibility_refs",
    )
    return result


def validate_validation_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ValidationReceiptArtifact:
    result = _common(
        payload,
        ValidationReceiptArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, ValidationReceiptArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    recorded_at = require_utc_timestamp(
        payload["recorded_at"],
        label="recorded_at",
    )
    if recorded_at < cutoff:
        raise ArtifactContractError("validation receipt predates its run cutoff")
    return result


def validate_rollback_drill_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> RollbackDrillReceiptArtifact:
    result = _common(
        payload,
        RollbackDrillReceiptArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, RollbackDrillReceiptArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    recorded_at = require_utc_timestamp(
        payload["recorded_at"],
        label="recorded_at",
    )
    if recorded_at < cutoff:
        raise ArtifactContractError("rollback drill receipt predates its run cutoff")
    if payload["scenarios"] != [
        "BOOTSTRAP",
        "CUTOVER",
        "CRASH_AFTER_CAS_RECOVERY",
        "ROLLBACK",
    ]:
        raise ArtifactContractError("rollback drill scenarios differ from the fixed order")
    return result


def validate_public_run_dto(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PublicRunDTOArtifact:
    result = _common(
        payload,
        PublicRunDTOArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, PublicRunDTOArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    expected_refs = (
        (
            "formal_active_pointer_ref",
            "myquant.v17.v4.formal-active-pointer.v1",
        ),
        (
            "formal_activation_receipt_ref",
            "myquant.v17.v4.formal-activation-receipt.v1",
        ),
        ("formal_output_ref", "myquant.v17.v4.formal-output.v1"),
        (
            "portfolio_output_ref",
            "myquant.v17.v4.portfolio-output.v1",
        ),
    )
    for field, version in expected_refs:
        reference = _ref(
            payload[field],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version=version,
            label=field,
        )
        if reference["cutoff"] != cutoff:
            raise ArtifactContractError(f"{field} cutoff must equal the public run cutoff")
    if payload["side_effects"] != {
        "broker_calls": False,
        "execution_calls": False,
        "llm_control_calls": False,
        "order_calls": False,
        "provider_calls": False,
        "selector_writes": False,
        "trade_calls": False,
    }:
        raise ArtifactContractError("public run side effects must all be false")
    targets = payload["targets"]
    symbols = [row["symbol"] for row in targets]
    if (
        symbols != sorted(symbols)
        or len(symbols) != len(set(symbols))
        or any(_CN_SYMBOL_RE.fullmatch(symbol) is None for symbol in symbols)
    ):
        raise ArtifactContractError("public run targets must be unique and symbol ordered")
    gross = _decimal(payload["gross_weight"], label="gross_weight")
    cash = _decimal(payload["cash_weight"], label="cash_weight")
    final_total = sum(
        (
            _decimal(
                row["final_target"],
                label=f"targets[{index}].final_target",
            )
            for index, row in enumerate(targets)
        ),
        Decimal("0"),
    )
    if gross + cash != Decimal("1") or final_total != gross:
        raise ArtifactContractError("public run portfolio weights do not close")
    return result


def validate_canary_public_snapshot(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> CanaryPublicSnapshotArtifact:
    result = _common(
        payload,
        CanaryPublicSnapshotArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, CanaryPublicSnapshotArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created_at = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    if created_at < cutoff:
        raise ArtifactContractError("canary public snapshot predates its run cutoff")
    public = validate_public_run_dto(
        payload["public_run"],
        schema_checked=True,
    )
    if (
        public.strategy_id != result.strategy_id
        or payload["public_run"]["surface"] != "SCHEDULE"
        or payload["public_run"]["cutoff"] != cutoff
        or payload["formal_active_pointer_ref"]
        != payload["public_run"]["formal_active_pointer_ref"]
        or payload["formal_activation_receipt_ref"]
        != payload["public_run"]["formal_activation_receipt_ref"]
        or payload["snapshot_id"] != f"canary-public-{payload['session_id']}"
    ):
        raise ArtifactContractError("canary public snapshot binding mismatch")
    return result


def validate_default_eligibility_intent(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DefaultEligibilityIntentArtifact:
    result = _common(
        payload,
        DefaultEligibilityIntentArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, DefaultEligibilityIntentArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created_at = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    if created_at < cutoff:
        raise ArtifactContractError("default eligibility intent predates its run cutoff")
    explicit = [
        _ref(
            payload["formal_active_pointer_ref"],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v4.formal-active-pointer.v1",
            label="formal_active_pointer_ref",
        ),
        _ref(
            payload["selector_bootstrap_receipt_ref"],
            strategy_id=result.strategy_id,
            version_prefix="myquant.research-runtime.",
            label="selector_bootstrap_receipt_ref",
        ),
        _ref(
            payload["rollback_drill_receipt_ref"],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v4.rollback-drill-receipt.v1",
            label="rollback_drill_receipt_ref",
        ),
    ]
    public = _refs(
        payload["public_surface_receipt_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="public_surface_receipt_refs",
        expected_version=("myquant.v17.v4." "public-surface-compatibility-receipt.v1"),
    )
    validation = _refs(
        payload["validation_receipt_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="validation_receipt_refs",
        expected_version="myquant.v17.v4.validation-receipt.v1",
    )
    evidence = _refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=None,
        label="evidence_refs",
    )
    _require_evidence_contains(
        evidence,
        [*explicit, *public, *validation],
        label="default eligibility intent evidence",
    )
    return result


def validate_default_eligibility_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DefaultEligibilityReceiptArtifact:
    result = _common(
        payload,
        DefaultEligibilityReceiptArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, DefaultEligibilityReceiptArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    intent_ref = _ref(
        payload["intent_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.default-eligibility-intent.v1",
        label="intent_ref",
    )
    pointer_ref = _ref(
        payload["pointer_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.default-eligible-pointer.v1",
        label="pointer_ref",
    )
    evidence = _refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="evidence_refs",
    )
    _require_evidence_contains(
        evidence,
        [intent_ref, pointer_ref],
        label="default eligibility completion evidence",
    )
    if pointer_ref["byte_sha256"] != payload["proposed_pointer_sha256"]:
        raise ArtifactContractError("default eligibility pointer differs from proposed bytes")
    _cas(payload, success=True, label="default eligibility completion")
    return result


def validate_default_eligible_pointer(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DefaultEligiblePointerArtifact:
    result = _common(
        payload,
        DefaultEligiblePointerArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, DefaultEligiblePointerArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["intent_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.default-eligibility-intent.v1",
        label="intent_ref",
    )
    if payload["updated_at"] < cutoff:
        raise ArtifactContractError("default eligibility pointer predates its cutoff")
    return result


def validate_canary_transition_intent(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> CanaryTransitionIntentArtifact:
    result = _common(
        payload,
        CanaryTransitionIntentArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, CanaryTransitionIntentArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created_at = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    if created_at < cutoff:
        raise ArtifactContractError("canary intent predates its cutoff")
    session_window = payload["session_window"]
    if session_window["start_session"] > session_window["end_session"]:
        raise ArtifactContractError("canary session window is inverted")
    transition = payload["transition"]
    expected_states = {
        "START": ("DEFAULT_ELIGIBLE", "CANARY"),
        "COMPLETE": ("CANARY", "CANARY"),
        "FAIL": ("CANARY", "DEFAULT_ELIGIBLE"),
    }
    if (
        payload["from_state"],
        payload["to_state"],
    ) != expected_states[transition]:
        raise ArtifactContractError("canary intent transition state mismatch")
    explicit = [
        _ref(
            payload["eligibility_pointer_ref"],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v4.default-eligible-pointer.v1",
            label="eligibility_pointer_ref",
        ),
        _ref(
            payload["historical_canary_policy_ref"],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v4.historical-canary-policy.v1",
            label="historical_canary_policy_ref",
        ),
        _ref(
            payload["v15_protocol_target_ref"],
            strategy_id=result.strategy_id,
            version_prefix="myquant.research-runtime.",
            label="v15_protocol_target_ref",
        ),
        _ref(
            payload["v15_active_run_pointer_ref"],
            strategy_id=result.strategy_id,
            version_prefix="myquant.research-runtime.",
            label="v15_active_run_pointer_ref",
        ),
    ]
    evidence = _refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=None,
        label="evidence_refs",
    )
    completion_fields = {
        "comparison_refs",
        "completed_sessions",
        "side_effect_counters",
        "threshold_results",
    }
    if transition == "START":
        if completion_fields & set(payload):
            raise ArtifactContractError("canary START cannot carry completion-only fields")
        if len(payload["paired_run_ids"]) != 1:
            raise ArtifactContractError("canary START must bind exactly the first paired-run ID")
    else:
        if not completion_fields.issubset(payload):
            raise ArtifactContractError(
                "completed or failed canary receipt lacks completion fields"
            )
        comparisons = _refs(
            payload["comparison_refs"],
            strategy_id=result.strategy_id,
            cutoff=None,
            label="comparison_refs",
            expected_version="myquant.v17.v4.dual-run-comparison.v1",
            path_ordered=False,
        )
        explicit.extend(comparisons)
        if len(payload["paired_run_ids"]) != 5:
            raise ArtifactContractError("operational canary requires five paired runs")
        completed_sessions = payload["completed_sessions"]
        if (
            completed_sessions != sorted(completed_sessions)
            or completed_sessions[0] != payload["session_window"]["start_session"]
            or completed_sessions[-1] != payload["session_window"]["end_session"]
        ):
            raise ArtifactContractError("operational canary sessions must match the fixed window")
        threshold_ids = [row["threshold_id"] for row in payload["threshold_results"]]
        if threshold_ids != sorted(threshold_ids) or len(set(threshold_ids)) != len(threshold_ids):
            raise ArtifactContractError("canary threshold results must be ID ordered")
        if transition == "COMPLETE":
            if any(row["status"] != "PASS" for row in payload["threshold_results"]) or any(
                value != 0 for value in payload["side_effect_counters"].values()
            ):
                raise ArtifactContractError(
                    "CANARY_COMPLETED requires all thresholds and counters to pass"
                )
    _require_evidence_contains(evidence, explicit, label="canary evidence")
    return result


def validate_canary_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> CanaryReceiptArtifact:
    result = _common(
        payload,
        CanaryReceiptArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, CanaryReceiptArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    expected_states = {
        "CANARY_STARTED": ("DEFAULT_ELIGIBLE", "CANARY"),
        "CANARY_COMPLETED": ("CANARY", "CANARY"),
        "CANARY_FAILED": ("CANARY", "DEFAULT_ELIGIBLE"),
    }
    if (
        payload["from_state"],
        payload["to_state"],
    ) != expected_states[payload["status"]]:
        raise ArtifactContractError("canary completion state/status mismatch")
    intent_ref = _ref(
        payload["intent_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.canary-transition-intent.v1",
        label="intent_ref",
    )
    pointer_ref = _ref(
        payload["pointer_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.canary-pointer.v1",
        label="pointer_ref",
    )
    evidence = _refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="evidence_refs",
    )
    _require_evidence_contains(
        evidence,
        [intent_ref, pointer_ref],
        label="canary completion evidence",
    )
    if pointer_ref["byte_sha256"] != payload["proposed_pointer_sha256"]:
        raise ArtifactContractError("canary pointer differs from proposed bytes")
    _cas(payload, success=True, label="canary completion")
    return result


def validate_canary_pointer(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> CanaryPointerArtifact:
    result = _common(
        payload,
        CanaryPointerArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, CanaryPointerArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["intent_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.canary-transition-intent.v1",
        label="intent_ref",
    )
    if payload["updated_at"] < cutoff:
        raise ArtifactContractError("canary pointer predates its cutoff")
    return result


def validate_dual_run_comparison(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DualRunComparisonArtifact:
    result = _common(
        payload,
        DualRunComparisonArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, DualRunComparisonArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["v15_run_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="v15_run_ref",
    )
    _ref(
        payload["v4_run_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        version_prefix="myquant.v17.v4.",
        label="v4_run_ref",
    )
    exact_input_names = (
        "benchmark",
        "canonical_calendar",
        "holdings_snapshot",
        "market_bars",
    )
    mismatched: list[dict[str, Any]] = []
    for name, pair in payload["comparison_inputs"].items():
        left = _ref(
            pair["v15_ref"],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            label=f"comparison_inputs.{name}.v15_ref",
        )
        right = _ref(
            pair["v4_ref"],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            label=f"comparison_inputs.{name}.v4_ref",
        )
        if name in exact_input_names and left["byte_sha256"] != right["byte_sha256"]:
            mismatched.extend((left, right))
    differing = _refs(
        payload["differing_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="differing_refs",
    )
    if payload["classification"] == "COMPARABLE":
        if mismatched or differing:
            raise ArtifactContractError("COMPARABLE dual run must have exact lower-level bytes")
    elif not mismatched or not differing:
        raise ArtifactContractError("NON_COMPARABLE dual run must bind exact differing refs")
    return result


def validate_historical_canary_policy(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> HistoricalCanaryPolicyArtifact:
    result = _common(
        payload,
        HistoricalCanaryPolicyArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, HistoricalCanaryPolicyArtifact)
    created = require_utc_timestamp(payload["created_at"], label="created_at")
    pairs = _refs(
        payload["pair_refs"],
        strategy_id=result.strategy_id,
        cutoff=created,
        label="pair_refs",
        expected_version="myquant.v17.v4.dual-run-comparison.v1",
        path_ordered=False,
    )
    if len(pairs) != 60:
        raise ArtifactContractError("historical canary policy requires 60 pairs")
    for field, value in payload["minimum_bands"].items():
        observed = _decimal(value, label=f"minimum_bands.{field}")
        if not Decimal("0") <= observed <= Decimal("1"):
            raise ArtifactContractError(f"minimum_bands.{field} must be within [0,1]")
    for field, value in payload["maximum_bands"].items():
        if _decimal(value, label=f"maximum_bands.{field}") < 0:
            raise ArtifactContractError(f"maximum_bands.{field} must be nonnegative")
    return result


_PIT_ROLES: Final = (
    "benchmark_total_return",
    "cn_open_day_calendar",
    "corporate_actions",
    "market_bars",
    "official_delisting_cash",
    "pit_fundamentals",
    "universe_membership",
)


def validate_pit_generation_catalog(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PitGenerationCatalogArtifact:
    result = _common(
        payload,
        PitGenerationCatalogArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, PitGenerationCatalogArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if payload["history_start"] > payload["decision_session"]:
        raise ArtifactContractError("PIT catalog session range is inverted")
    for name, version_kind in (
        ("dataset_refs", "dataset"),
        ("expected_key_inventory_refs", "expected-keys"),
    ):
        values = payload[name]
        if type(values) is not dict or set(values) != set(_PIT_ROLES):
            raise ArtifactContractError(f"{name} role inventory mismatch")
        for role in _PIT_ROLES:
            _ref(
                values[role],
                strategy_id=result.strategy_id,
                cutoff=cutoff,
                expected_version=(f"myquant.v17.v4.{version_kind}.{role}.v1"),
                label=f"{name}.{role}",
            )
    summaries = payload["dataset_summaries"]
    if tuple(row["role"] for row in summaries) != _PIT_ROLES:
        raise ArtifactContractError("PIT dataset summary role order mismatch")
    admission_payload = {
        "history_start": payload["history_start"],
        "decision_session": payload["decision_session"],
        "decision_cutoff": cutoff,
        "datasets": summaries,
    }
    if (
        hashlib.sha256(canonical_bytes(admission_payload)).hexdigest()
        != payload["admission_closure_sha256"]
    ):
        raise ArtifactContractError("PIT admission closure SHA mismatch")
    source_payload = {
        "admission_closure_sha256": payload["admission_closure_sha256"],
        "dataset_refs": payload["dataset_refs"],
        "expected_key_inventory_refs": payload["expected_key_inventory_refs"],
    }
    if (
        hashlib.sha256(canonical_bytes(source_payload)).hexdigest()
        != payload["source_closure_sha256"]
    ):
        raise ArtifactContractError("PIT source closure SHA mismatch")
    return result


def validate_pit_catalog_pointer(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PitCatalogPointerArtifact:
    result = _common(
        payload,
        PitCatalogPointerArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, PitCatalogPointerArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["catalog_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.pit-generation-catalog.v1",
        label="catalog_ref",
    )
    if payload["updated_at"] < cutoff:
        raise ArtifactContractError("PIT catalog pointer predates its cutoff")
    return result


def validate_preselect_locator(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PreselectLocatorArtifact:
    if set(payload) != {
        "authority",
        "cutoff",
        "locator_id",
        "origin",
        "pit_catalog_ref",
        "protocol_version",
        "semantic_sha256",
        "strategy_id",
        "version",
    }:
        raise ArtifactContractError("preselect locator native shape mismatch")
    result = _common(
        payload,
        PreselectLocatorArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, PreselectLocatorArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["pit_catalog_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.pit-generation-catalog.v1",
        label="pit_catalog_ref",
    )
    try:
        origin = date.fromisoformat(payload["origin"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("preselect origin is invalid") from exc
    if origin != payload["origin"] or origin > cutoff[:10]:
        raise ArtifactContractError("preselect origin is after its cutoff")
    return result


def validate_initial_pool_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> InitialPoolOutputArtifact:
    if set(payload) != {
        "authority",
        "cutoff",
        "ordered_pool",
        "origin",
        "output_id",
        "preselect_locator_ref",
        "protocol_version",
        "semantic_sha256",
        "strategy_id",
        "version",
    }:
        raise ArtifactContractError("initial pool native shape mismatch")
    result = _common(
        payload,
        InitialPoolOutputArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, InitialPoolOutputArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["preselect_locator_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.preselect-locator.v1",
        label="preselect_locator_ref",
    )
    pool = payload["ordered_pool"]
    try:
        origin = date.fromisoformat(payload["origin"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("initial pool origin is invalid") from exc
    if (
        type(pool) is not list
        or not 24 <= len(pool) <= 500
        or any(
            type(symbol) is not str or _CN_SYMBOL_RE.fullmatch(symbol) is None for symbol in pool
        )
        or len(pool) != len(set(pool))
        or origin != payload["origin"]
        or origin > cutoff[:10]
    ):
        raise ArtifactContractError("initial pool native payload mismatch")
    return result


def validate_branch_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> BranchOutputArtifact:
    if set(payload) != {
        "authority",
        "branch_kind",
        "cutoff",
        "initial_pool_ref",
        "origin",
        "output_id",
        "protocol_version",
        "score_rows",
        "semantic_sha256",
        "strategy_id",
        "version",
    }:
        raise ArtifactContractError("branch output native shape mismatch")
    result = _common(
        payload,
        BranchOutputArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, BranchOutputArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["initial_pool_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.initial-pool-output.v1",
        label="initial_pool_ref",
    )
    rows = payload["score_rows"]
    if (
        payload["branch_kind"] not in {"FUNDAMENTAL", "QUANT"}
        or type(rows) is not list
        or not 24 <= len(rows) <= 500
        or any(type(row) is not dict or set(row) != {"score", "symbol"} for row in rows)
    ):
        raise ArtifactContractError("branch native score payload mismatch")
    symbols = [row["symbol"] for row in rows]
    try:
        origin = date.fromisoformat(payload["origin"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("branch origin is invalid") from exc
    if (
        any(
            type(symbol) is not str or _CN_SYMBOL_RE.fullmatch(symbol) is None for symbol in symbols
        )
        or len(symbols) != len(set(symbols))
        or origin != payload["origin"]
        or origin > cutoff[:10]
    ):
        raise ArtifactContractError("branch native score payload mismatch")
    for row in payload["score_rows"]:
        _decimal(row["score"], label=f"score_rows.{row['symbol']}")
    return result


def validate_research_quant_branch_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ResearchQuantBranchOutputArtifact:
    expected_fields = {
        "authority",
        "branch_kind",
        "canary_evidence_eligible",
        "cutoff",
        "factor_definition_sha256",
        "factor_mode",
        "factor_names",
        "factor_policy_sha256",
        "factor_rows",
        "formal_activation_eligible",
        "incubator_version",
        "initial_pool_ref",
        "market_slice_ref",
        "origin",
        "output_id",
        "protocol_version",
        "score_rows",
        "semantic_sha256",
        "shadow_only",
        "strategy_id",
        "version",
    }
    if set(payload) != expected_fields:
        raise ArtifactContractError(
            "research Quant branch native shape mismatch"
        )
    result = _common(
        payload,
        ResearchQuantBranchOutputArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, ResearchQuantBranchOutputArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["initial_pool_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.initial-pool-output.v1",
        label="initial_pool_ref",
    )
    _ref(
        payload["market_slice_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.dataset.quant-factor-input.v1",
        label="market_slice_ref",
    )
    factor_names = [
        "cn_fip_continuous_direction_12m",
        "cn_low_market_adjusted_tail_asymmetry_252d",
        "cn_low_total_skewness_20d",
    ]
    if (
        payload["branch_kind"] != "QUANT"
        or payload["factor_mode"] != "LITERATURE_INCUBATOR_RESEARCH"
        or payload["incubator_version"] != "v4-literature-incubator.v10"
        or payload["factor_names"] != factor_names
        or payload["shadow_only"] is not True
        or payload["formal_activation_eligible"] is not False
        or payload["canary_evidence_eligible"] is not False
    ):
        raise ArtifactContractError(
            "research Quant branch authority or factor identity mismatch"
        )
    factor_rows = payload["factor_rows"]
    score_rows = payload["score_rows"]
    if (
        type(factor_rows) is not list
        or type(score_rows) is not list
        or not 24 <= len(factor_rows) <= 500
        or len(score_rows) != len(factor_rows)
    ):
        raise ArtifactContractError(
            "research Quant branch row shape mismatch"
        )
    factor_symbols = [row.get("symbol") for row in factor_rows]
    score_symbols = [row.get("symbol") for row in score_rows]
    if (
        factor_symbols != score_symbols
        or len(factor_symbols) != len(set(factor_symbols))
        or any(
            type(symbol) is not str
            or _CN_SYMBOL_RE.fullmatch(symbol) is None
            for symbol in factor_symbols
        )
    ):
        raise ArtifactContractError(
            "research Quant branch symbol order mismatch"
        )
    for factor_row, score_row in zip(
        factor_rows,
        score_rows,
        strict=True,
    ):
        if (
            type(factor_row) is not dict
            or set(factor_row) != {"factor_values", "symbol"}
            or type(score_row) is not dict
            or set(score_row) != {"score", "symbol"}
        ):
            raise ArtifactContractError(
                "research Quant branch row shape mismatch"
            )
        values = factor_row["factor_values"]
        if (
            type(values) is not list
            or [value.get("factor_name") for value in values]
            != factor_names
            or any(
                type(value) is not dict
                or set(value) != {"factor_name", "value"}
                for value in values
            )
        ):
            raise ArtifactContractError(
                "research Quant branch factor order mismatch"
            )
        decimals = [
            _decimal(
                value["value"],
                label=(
                    f"factor_rows.{factor_row['symbol']}."
                    f"{value['factor_name']}"
                ),
            )
            for value in values
        ]
        observed = _decimal(
            score_row["score"],
            label=f"score_rows.{score_row['symbol']}",
        )
        expected_score = (
            sum(decimals, Decimal("0")) / Decimal("3")
        ).quantize(Decimal("0.0000000000000001"))
        if observed != expected_score:
            raise ArtifactContractError(
                "research Quant branch composite score mismatch"
            )
    try:
        origin = date.fromisoformat(payload["origin"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "research Quant branch origin is invalid"
        ) from exc
    if origin != payload["origin"] or origin > cutoff[:10]:
        raise ArtifactContractError(
            "research Quant branch origin is invalid"
        )
    require_sha256(
        payload["factor_policy_sha256"],
        label="factor_policy_sha256",
    )
    require_sha256(
        payload["factor_definition_sha256"],
        label="factor_definition_sha256",
    )
    return result


def validate_research_factor_shadow_assertion(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ResearchFactorShadowAssertionArtifact:
    result = _common(
        payload,
        ResearchFactorShadowAssertionArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, ResearchFactorShadowAssertionArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created = require_utc_timestamp(payload["created_at"], label="created_at")
    try:
        session = date.fromisoformat(payload["decision_session"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(
            "research-factor Shadow assertion decision_session is invalid"
        ) from exc
    if (
        created < cutoff
        or session != payload["decision_session"]
        or session != cutoff[:10]
        or payload["assertion_scope"]
        != "ONE_RUN_V17_V4_SHADOW_RESEARCH_TRIO"
        or payload["factor_evidence_mode"] != "RESEARCH_TRIO_SHADOW_ONLY"
        or payload["operator_asserted"] is not True
        or payload["factor_names"]
        != [
            "cn_fip_continuous_direction_12m",
            "cn_low_market_adjusted_tail_asymmetry_252d",
            "cn_low_total_skewness_20d",
        ]
    ):
        raise ArtifactContractError(
            "research-factor Shadow assertion binding mismatch"
        )
    require_opaque_id(payload["override_id"], label="override_id")
    require_opaque_id(payload["shadow_run_id"], label="shadow_run_id")
    require_sha256(
        payload["factor_policy_sha256"],
        label="factor_policy_sha256",
    )
    return result


def validate_shadow_run_v2(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ShadowRunArtifact:
    expected_fields = {
        "authority",
        "canary_evidence_eligible",
        "comparison_inputs",
        "created_at",
        "cutoff",
        "decision_session",
        "deep_bundle_ref",
        "factor_evidence_mode",
        "formal_activation_eligible",
        "fundamental_branch_ref",
        "fusion_top24_ref",
        "initial_pool_ref",
        "model_output_present",
        "protocol_version",
        "quant_branch_ref",
        "research_factor_shadow_assertion_ref",
        "research_quant_factor_names",
        "research_quant_factor_policy_sha256",
        "semantic_sha256",
        "shadow_only",
        "shadow_run_id",
        "source_locator_ref",
        "state",
        "strategy_id",
        "version",
    }
    if set(payload) != expected_fields:
        raise ArtifactContractError("shadow run native shape mismatch")
    result = _common(
        payload,
        ShadowRunArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, ShadowRunArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    try:
        session = date.fromisoformat(payload["decision_session"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("shadow run decision_session is invalid") from exc
    if (
        created < cutoff
        or session != payload["decision_session"]
        or session != cutoff[:10]
        or payload["factor_evidence_mode"] != "RESEARCH_TRIO_SHADOW_ONLY"
        or payload["research_quant_factor_names"]
        != [
            "cn_fip_continuous_direction_12m",
            "cn_low_market_adjusted_tail_asymmetry_252d",
            "cn_low_total_skewness_20d",
        ]
    ):
        raise ArtifactContractError("shadow run time or factor inventory mismatch")
    require_sha256(
        payload["research_quant_factor_policy_sha256"],
        label="research_quant_factor_policy_sha256",
    )
    for field, version in (
        (
            "research_factor_shadow_assertion_ref",
            "myquant.v17.v4.research-factor-shadow-assertion.v1",
        ),
        (
            "source_locator_ref",
            "myquant.v17.v4.preselect-locator.v1",
        ),
        (
            "initial_pool_ref",
            "myquant.v17.v4.initial-pool-output.v1",
        ),
        (
            "quant_branch_ref",
            "myquant.v17.v4.research-quant-branch-output.v1",
        ),
        (
            "fundamental_branch_ref",
            "myquant.v17.v4.branch-output.v1",
        ),
        (
            "fusion_top24_ref",
            "myquant.v17.v4.fusion-top24.v1",
        ),
        (
            "deep_bundle_ref",
            "myquant.v17.v4.deep-evidence-bundle.v2",
        ),
    ):
        _ref(
            payload[field],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version=version,
            label=field,
        )
    comparison = payload["comparison_inputs"]
    _ref(
        comparison["source_closure_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.pit-generation-catalog.v1",
        label="comparison_inputs.source_closure_ref",
    )
    _ref(
        comparison["holdings_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.holdings-snapshot.v1",
        label="comparison_inputs.holdings_ref",
    )
    for field in ("calendar_ref", "market_bars_ref"):
        _ref(
            comparison[field],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            label=f"comparison_inputs.{field}",
        )
    return result


def validate_total_return_labels(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> TotalReturnLabelsArtifact:
    if set(payload) != {
        "authority",
        "cutoff",
        "label_end_session",
        "label_id",
        "label_kind",
        "origin",
        "protocol_version",
        "rows",
        "semantic_sha256",
        "strategy_id",
        "version",
    }:
        raise ArtifactContractError("total-return label native shape mismatch")
    result = _common(
        payload,
        TotalReturnLabelsArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, TotalReturnLabelsArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    rows = payload["rows"]
    if (
        payload["label_kind"] not in {"LABEL_60", "LABEL_252"}
        or type(rows) is not list
        or not 24 <= len(rows) <= 500
        or any(
            type(row) is not dict
            or set(row)
            != {
                "delisted",
                "forward_return",
                "official_terminal_cash",
                "symbol",
            }
            or type(row.get("delisted")) is not bool
            or type(row.get("official_terminal_cash")) is not bool
            for row in rows
        )
    ):
        raise ArtifactContractError("total-return label native payload mismatch")
    symbols = [row["symbol"] for row in rows]
    try:
        origin = date.fromisoformat(payload["origin"]).isoformat()
        label_end = date.fromisoformat(payload["label_end_session"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("total-return label date is invalid") from exc
    if (
        any(
            type(symbol) is not str or _CN_SYMBOL_RE.fullmatch(symbol) is None for symbol in symbols
        )
        or len(symbols) != len(set(symbols))
        or origin != payload["origin"]
        or label_end != payload["label_end_session"]
        or origin >= label_end
        or label_end > cutoff[:10]
    ):
        raise ArtifactContractError("total-return label native payload mismatch")
    for row in payload["rows"]:
        _decimal(
            row["forward_return"],
            label=f"rows.{row['symbol']}.forward_return",
        )
        if row["delisted"] is True and row["official_terminal_cash"] is not True:
            raise ArtifactContractError("delisted label lacks official terminal cash")
    return result


def validate_fusion_top24(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FusionTop24Artifact:
    result = _common(
        payload,
        FusionTop24Artifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FusionTop24Artifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("fusion Top24 created_at precedes cutoff")
    _ref(
        payload["promotion_receipt_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=("myquant.v17.v4.fusion-promotion-receipt.v1"),
        label="promotion_receipt_ref",
    )
    rows = payload["rows"]
    ranks = [row["rank"] for row in rows]
    symbols = [row["symbol"] for row in rows]
    if ranks != list(range(1, 25)) or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("fusion Top24 rank or symbol inventory mismatch")
    total = Decimal("0")
    for index, row in enumerate(rows):
        _decimal(
            row["fused_score"],
            label=f"rows[{index}].fused_score",
        )
        target = _decimal(
            row["base_target"],
            label=f"rows[{index}].base_target",
        )
        if target < 0:
            raise ArtifactContractError("fusion Top24 base target is negative")
        total += target
    if total > Decimal("1"):
        raise ArtifactContractError("fusion Top24 base targets exceed one")
    return result


def validate_official_evidence(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> OfficialEvidenceArtifact:
    result = _common(
        payload,
        OfficialEvidenceArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, OfficialEvidenceArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    published = require_utc_timestamp(
        payload["published_at"],
        label="published_at",
    )
    available = require_utc_timestamp(
        payload["available_at"],
        label="available_at",
    )
    if published > available or available > cutoff:
        raise ArtifactContractError("official evidence availability exceeds cutoff")
    return result


def _validate_deep_source_refs(
    values: Any,
    *,
    strategy_id: str,
    cutoff: str,
    label: str,
) -> list[dict[str, Any]]:
    refs = _refs(
        values,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.official-evidence.v1",
        label=label,
    )
    if not refs:
        raise ArtifactContractError(f"{label} must contain official evidence")
    return refs


def validate_issuer_dossier(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> IssuerDossierArtifact:
    result = _common(
        payload,
        IssuerDossierArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, IssuerDossierArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("issuer dossier created_at precedes cutoff")
    try:
        as_of = date.fromisoformat(payload["as_of"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("issuer dossier as_of is invalid") from exc
    if as_of != payload["as_of"] or as_of > cutoff[:10]:
        raise ArtifactContractError("issuer dossier as_of exceeds cutoff")
    _validate_deep_source_refs(
        payload["official_evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="official_evidence_refs",
    )
    return result


def validate_event_scan(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> EventScanArtifact:
    result = _common(
        payload,
        EventScanArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, EventScanArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("event scan created_at precedes cutoff")
    try:
        as_of = date.fromisoformat(payload["as_of"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("event scan as_of is invalid") from exc
    if (
        as_of != payload["as_of"]
        or as_of > cutoff[:10]
        or payload["flags"] != sorted(payload["flags"])
    ):
        raise ArtifactContractError("event scan native payload mismatch")
    _validate_deep_source_refs(
        payload["official_evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="official_evidence_refs",
    )
    return result


def validate_deep_evidence_bundle(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DeepEvidenceBundleArtifact:
    result = _common(
        payload,
        DeepEvidenceBundleArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, DeepEvidenceBundleArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("Deep bundle created_at precedes cutoff")
    _ref(
        payload["fusion_top24_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.fusion-top24.v1",
        label="fusion_top24_ref",
    )
    rows = payload["rows"]
    symbols = [row["symbol"] for row in rows]
    if len(symbols) != 24 or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("Deep bundle must contain exactly one row per Top24 symbol")
    for index, row in enumerate(rows):
        target = _decimal(
            row["target_after_deep"],
            label=f"rows[{index}].target_after_deep",
        )
        if target < 0:
            raise ArtifactContractError("Deep target must be nonnegative")
        if row["status"] == "UNAVAILABLE":
            if (
                row["official_evidence_refs"]
                or row["issuer_dossier_ref"] is not None
                or row["event_scan_ref"] is not None
                or row["signal"] is not None
                or row["buy_veto"] is not True
                or not row["reason"]
                or target != 0
            ):
                raise ArtifactContractError("unavailable Deep row must remain a zero BUY veto")
            continue
        if (
            not row["official_evidence_refs"]
            or row["issuer_dossier_ref"] is None
            or row["event_scan_ref"] is None
            or row["signal"] is None
            or row["reason"]
        ):
            raise ArtifactContractError("complete Deep row evidence closure is incomplete")
        _validate_deep_source_refs(
            row["official_evidence_refs"],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            label=f"rows[{index}].official_evidence_refs",
        )
        for field, version in (
            ("issuer_dossier_ref", "myquant.v17.v4.issuer-dossier.v1"),
            ("event_scan_ref", "myquant.v17.v4.event-scan.v1"),
        ):
            _ref(
                row[field],
                strategy_id=result.strategy_id,
                cutoff=cutoff,
                expected_version=version,
                label=f"rows[{index}].{field}",
            )
        _decimal(row["signal"], label=f"rows[{index}].signal")
    return result


_DEEP_MODULE_ORDER: Final = (
    "financial_reconciliation",
    "business_model",
    "industry",
    "competition",
    "management",
    "valuation",
    "catalysts",
    "contrary_evidence",
    "falsification_conditions",
    "monitoring",
)


def _validate_v2_module_rows(
    modules: Any,
    *,
    evidence_ids: set[str] | None = None,
    evidence_refs: set[tuple[str, str]] | None = None,
    label: str,
) -> None:
    if type(modules) is not list or [row.get("module_id") for row in modules] != list(
        _DEEP_MODULE_ORDER
    ):
        raise ArtifactContractError(f"{label} must contain the fixed ten-module order")
    for index, row in enumerate(modules):
        score = _decimal(
            row["score"],
            label=f"{label}[{index}].score",
        )
        if score < -1 or score > 1:
            raise ArtifactContractError(f"{label}[{index}] score is outside [-1,1]")
        conclusion = row["conclusion"]
        if (
            (conclusion == "POSITIVE" and score <= 0)
            or (conclusion == "NEUTRAL" and score != 0)
            or (conclusion in {"NEGATIVE", "RED_FLAG"} and score >= 0)
        ):
            raise ArtifactContractError(f"{label}[{index}] conclusion and score disagree")
        if evidence_ids is not None and not set(row["evidence_ids"]).issubset(evidence_ids):
            raise ArtifactContractError(f"{label}[{index}] cites unknown evidence")
        if evidence_refs is not None:
            observed = {(ref["relative_path"], ref["byte_sha256"]) for ref in row["evidence_refs"]}
            if not observed.issubset(evidence_refs):
                raise ArtifactContractError(f"{label}[{index}] cites unbound evidence")


def validate_deep_assessment_manifest(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DeepAssessmentManifestArtifact:
    result = _common(
        payload,
        DeepAssessmentManifestArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, DeepAssessmentManifestArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("Deep assessment manifest predates cutoff")
    _ref(
        payload["fusion_top24_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.fusion-top24.v1",
        label="fusion_top24_ref",
    )
    symbols = [row["symbol"] for row in payload["rows"]]
    if len(symbols) != 24 or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("Deep assessment manifest must cover 24 unique symbols")
    for row in payload["rows"]:
        if row["blocker_codes"] != sorted(row["blocker_codes"]):
            raise ArtifactContractError("Deep blocker codes must be ASCII ordered")
        if row["event_flags"] != sorted(row["event_flags"]):
            raise ArtifactContractError("Deep event flags must be ASCII ordered")
        if row["status"] == "UNAVAILABLE":
            if (
                row["raw_documents"]
                or row["modules"]
                or row["event_flags"]
                or not row["blocker_codes"]
            ):
                raise ArtifactContractError("unavailable Deep assessment must be blocker-only")
            continue
        if row["blocker_codes"] or not row["raw_documents"] or len(row["modules"]) != 10:
            raise ArtifactContractError("complete Deep assessment closure is incomplete")
        evidence_ids = [document["evidence_id"] for document in row["raw_documents"]]
        if evidence_ids != sorted(evidence_ids) or len(evidence_ids) != len(set(evidence_ids)):
            raise ArtifactContractError("raw Deep evidence inventory is not canonical")
        for document in row["raw_documents"]:
            published = require_utc_timestamp(
                document["published_at"],
                label="published_at",
            )
            available = require_utc_timestamp(
                document["available_at"],
                label="available_at",
            )
            captured = require_utc_timestamp(
                document["captured_at"],
                label="captured_at",
            )
            raw_path = document["raw_relative_path"]
            if (
                published > available
                or available > cutoff
                or captured < available
                or not raw_path.startswith("data/private/v17_v4_sources/deep_raw/")
                or raw_path.startswith("/")
                or "\\" in raw_path
                or any(part in {"", ".", ".."} for part in raw_path.split("/"))
            ):
                raise ArtifactContractError("raw Deep evidence time or path closure is invalid")
        _validate_v2_module_rows(
            row["modules"],
            evidence_ids=set(evidence_ids),
            label=f"modules.{row['symbol']}",
        )
    return result


def validate_official_evidence_v2(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> OfficialEvidenceV2Artifact:
    result = _common(
        payload,
        OfficialEvidenceV2Artifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, OfficialEvidenceV2Artifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    published = require_utc_timestamp(
        payload["published_at"],
        label="published_at",
    )
    available = require_utc_timestamp(
        payload["available_at"],
        label="available_at",
    )
    captured = require_utc_timestamp(
        payload["captured_at"],
        label="captured_at",
    )
    _ref(
        payload["assessment_manifest_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=("myquant.v17.v4.deep-assessment-manifest.v1"),
        label="assessment_manifest_ref",
    )
    raw_path = payload["content_relative_path"]
    if (
        published > available
        or available > cutoff
        or captured < available
        or not raw_path.startswith("data/private/v17_v4_sources/deep_raw/")
        or "\\" in raw_path
        or any(part in {"", ".", ".."} for part in raw_path.split("/"))
    ):
        raise ArtifactContractError("official evidence v2 raw closure is invalid")
    return result


def validate_issuer_dossier_v2(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> IssuerDossierV2Artifact:
    result = _common(
        payload,
        IssuerDossierV2Artifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, IssuerDossierV2Artifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("issuer dossier v2 predates cutoff")
    try:
        as_of = date.fromisoformat(payload["as_of"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("issuer dossier v2 as_of is invalid") from exc
    if as_of != payload["as_of"] or as_of > cutoff[:10]:
        raise ArtifactContractError("issuer dossier v2 as_of exceeds cutoff")
    manifest_ref = _ref(
        payload["assessment_manifest_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=("myquant.v17.v4.deep-assessment-manifest.v1"),
        label="assessment_manifest_ref",
    )
    refs = _refs(
        payload["official_evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.official-evidence.v2",
        label="official_evidence_refs",
    )
    evidence_keys = {(ref["relative_path"], ref["byte_sha256"]) for ref in refs}
    _validate_v2_module_rows(
        payload["modules"],
        evidence_refs=evidence_keys,
        label="modules",
    )
    expected_summary = hashlib.sha256(canonical_bytes(payload["modules"])).hexdigest()
    if (
        payload["summary_sha256"] != expected_summary
        or payload["red_flags"] != sorted(payload["red_flags"])
        or manifest_ref["artifact_id"] != payload["assessment_manifest_ref"]["artifact_id"]
    ):
        raise ArtifactContractError("issuer dossier v2 native payload mismatch")
    return result


def validate_event_scan_v2(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> EventScanV2Artifact:
    result = _common(
        payload,
        EventScanV2Artifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, EventScanV2Artifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("event scan v2 predates cutoff")
    try:
        as_of = date.fromisoformat(payload["as_of"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("event scan v2 as_of is invalid") from exc
    if (
        as_of != payload["as_of"]
        or as_of > cutoff[:10]
        or payload["flags"] != sorted(payload["flags"])
    ):
        raise ArtifactContractError("event scan v2 native payload mismatch")
    _ref(
        payload["assessment_manifest_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=("myquant.v17.v4.deep-assessment-manifest.v1"),
        label="assessment_manifest_ref",
    )
    _refs(
        payload["official_evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.official-evidence.v2",
        label="official_evidence_refs",
    )
    return result


def validate_deep_evidence_bundle_v2(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DeepEvidenceBundleV2Artifact:
    result = _common(
        payload,
        DeepEvidenceBundleV2Artifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, DeepEvidenceBundleV2Artifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("Deep bundle v2 predates cutoff")
    _ref(
        payload["fusion_top24_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.fusion-top24.v1",
        label="fusion_top24_ref",
    )
    _ref(
        payload["assessment_manifest_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=("myquant.v17.v4.deep-assessment-manifest.v1"),
        label="assessment_manifest_ref",
    )
    symbols = [row["symbol"] for row in payload["rows"]]
    if len(symbols) != 24 or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("Deep bundle v2 must cover 24 unique symbols")
    for index, row in enumerate(payload["rows"]):
        target = _decimal(
            row["target_after_deep"],
            label=f"rows[{index}].target_after_deep",
        )
        if target < 0 or row["blocker_codes"] != sorted(row["blocker_codes"]):
            raise ArtifactContractError("Deep bundle v2 target or blockers are invalid")
        if row["status"] == "UNAVAILABLE":
            if (
                row["official_evidence_refs"]
                or row["issuer_dossier_ref"] is not None
                or row["event_scan_ref"] is not None
                or row["signal"] is not None
                or row["buy_veto"] is not True
                or not row["blocker_codes"]
                or target != 0
            ):
                raise ArtifactContractError("unavailable Deep v2 row must be zero BUY veto")
            continue
        if (
            row["blocker_codes"]
            or not row["official_evidence_refs"]
            or row["issuer_dossier_ref"] is None
            or row["event_scan_ref"] is None
            or row["signal"] is None
        ):
            raise ArtifactContractError("complete Deep v2 row closure is incomplete")
        _refs(
            row["official_evidence_refs"],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v4.official-evidence.v2",
            label=f"rows[{index}].official_evidence_refs",
        )
        for field, version in (
            ("issuer_dossier_ref", "myquant.v17.v4.issuer-dossier.v2"),
            ("event_scan_ref", "myquant.v17.v4.event-scan.v2"),
        ):
            _ref(
                row[field],
                strategy_id=result.strategy_id,
                cutoff=cutoff,
                expected_version=version,
                label=f"rows[{index}].{field}",
            )
        signal = _decimal(
            row["signal"],
            label=f"rows[{index}].signal",
        )
        if signal < -1 or signal > 1:
            raise ArtifactContractError("Deep bundle v2 signal is outside [-1,1]")
    return result


def validate_shadow_readiness(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ShadowReadinessArtifact:
    result = _common(
        payload,
        ShadowReadinessArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, ShadowReadinessArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    try:
        session = date.fromisoformat(payload["decision_session"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("shadow readiness decision_session is invalid") from exc
    if (
        created < cutoff
        or session != payload["decision_session"]
        or session != cutoff[:10]
        or payload["blocker_codes"] != sorted(payload["blocker_codes"])
    ):
        raise ArtifactContractError("shadow readiness time or blocker closure mismatch")
    return result


def validate_shadow_run(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ShadowRunArtifact:
    result = _common(
        payload,
        ShadowRunArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, ShadowRunArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    try:
        session = date.fromisoformat(payload["decision_session"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("shadow run decision_session is invalid") from exc
    if (
        created < cutoff
        or session != payload["decision_session"]
        or session != cutoff[:10]
        or payload["production_factor_names"] != sorted(payload["production_factor_names"])
        or payload["research_quant_factor_names"]
        != [
            "cn_fip_continuous_direction_12m",
            "cn_low_market_adjusted_tail_asymmetry_252d",
            "cn_low_total_skewness_20d",
        ]
    ):
        raise ArtifactContractError("shadow run time or factor inventory mismatch")
    require_sha256(
        payload["research_quant_factor_policy_sha256"],
        label="research_quant_factor_policy_sha256",
    )
    for field, version in (
        (
            "factor_control_active_set_ref",
            _FACTOR_ACTIVE_SET_SCHEMA,
        ),
        (
            "factor_control_receipt_ref",
            _FACTOR_CONTROL_RECEIPT_SCHEMA,
        ),
        (
            "source_locator_ref",
            "myquant.v17.v4.preselect-locator.v1",
        ),
        (
            "initial_pool_ref",
            "myquant.v17.v4.initial-pool-output.v1",
        ),
        (
            "quant_branch_ref",
            "myquant.v17.v4.research-quant-branch-output.v1",
        ),
        (
            "fundamental_branch_ref",
            "myquant.v17.v4.branch-output.v1",
        ),
        (
            "fusion_top24_ref",
            "myquant.v17.v4.fusion-top24.v1",
        ),
        (
            "deep_bundle_ref",
            "myquant.v17.v4.deep-evidence-bundle.v2",
        ),
    ):
        _ref(
            payload[field],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version=version,
            label=field,
        )
    comparison = payload["comparison_inputs"]
    _ref(
        comparison["source_closure_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.pit-generation-catalog.v1",
        label="comparison_inputs.source_closure_ref",
    )
    _ref(
        comparison["holdings_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.holdings-snapshot.v1",
        label="comparison_inputs.holdings_ref",
    )
    for field in ("calendar_ref", "market_bars_ref"):
        _ref(
            comparison[field],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            label=f"comparison_inputs.{field}",
        )
    return result


def validate_shadow_session_ref(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ShadowSessionRefArtifact:
    result = _common(
        payload,
        ShadowSessionRefArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, ShadowSessionRefArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    try:
        session = date.fromisoformat(payload["decision_session"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("shadow session decision_session is invalid") from exc
    if created < cutoff or session != payload["decision_session"] or session != cutoff[:10]:
        raise ArtifactContractError("shadow session time closure mismatch")
    if payload["version"] == "myquant.v17.v4.shadow-session-ref.v1":
        expected_run_version = "myquant.v17.v4.shadow-run.v1"
    elif payload["version"] == "myquant.v17.v4.shadow-session-ref.v2":
        if payload["factor_evidence_mode"] != "RESEARCH_TRIO_SHADOW_ONLY":
            raise ArtifactContractError("shadow session factor evidence mode mismatch")
        _ref(
            payload["research_factor_shadow_assertion_ref"],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version=(
                "myquant.v17.v4.research-factor-shadow-assertion.v1"
            ),
            label="research_factor_shadow_assertion_ref",
        )
        expected_run_version = SHADOW_RUN_RESEARCH_VERSION
    else:
        raise ArtifactContractError("unsupported shadow session version")
    _ref(
        payload["shadow_run_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=expected_run_version,
        label="shadow_run_ref",
    )
    return result


def validate_holdings_snapshot(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> HoldingsSnapshotArtifact:
    result = _common(
        payload,
        HoldingsSnapshotArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, HoldingsSnapshotArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(payload["created_at"], label="created_at") < cutoff
        or require_utc_timestamp(payload["available_at"], label="available_at") > cutoff
    ):
        raise ArtifactContractError("holdings snapshot time closure mismatch")
    try:
        as_of = date.fromisoformat(payload["as_of_session"]).isoformat()
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError("holdings snapshot as_of_session is invalid") from exc
    if as_of != payload["as_of_session"] or as_of > cutoff[:10]:
        raise ArtifactContractError("holdings snapshot session exceeds cutoff")
    positions = payload["positions"]
    symbols = [row["symbol"] for row in positions]
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("holdings positions must be unique and symbol ordered")
    nav = _decimal(payload["nav"], label="nav")
    cash = _decimal(payload["cash"], label="cash")
    position_total = Decimal("0")
    for index, row in enumerate(positions):
        market_value = _decimal(
            row["market_value"],
            label=f"positions[{index}].market_value",
        )
        if market_value <= 0:
            raise ArtifactContractError("holding market value must be positive")
        position_total += market_value
    if (
        nav <= 0
        or cash < 0
        or cash + position_total != nav
        or payload["declared_all_cash"] != (not positions and cash == nav)
    ):
        raise ArtifactContractError("holdings snapshot NAV reconciliation mismatch")
    return result


def validate_portfolio_risk_policy(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PortfolioRiskPolicyArtifact:
    result = _common(
        payload,
        PortfolioRiskPolicyArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, PortfolioRiskPolicyArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created_at = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    effective_from = require_utc_timestamp(
        payload["effective_from"],
        label="effective_from",
    )
    expires_at = require_utc_timestamp(
        payload["expires_at"],
        label="expires_at",
    )
    if created_at < cutoff or not effective_from <= cutoff < expires_at:
        raise ArtifactContractError("risk policy is not effective and unexpired at cutoff")
    values = {
        field: _decimal(payload[field], label=field)
        for field in (
            "cash_floor",
            "cluster_cap",
            "gross_cap",
            "industry_cap",
            "single_name_cap",
            "turnover_cap",
        )
    }
    if (
        any(value < 0 or value > 1 for value in values.values())
        or any(
            values[field] <= 0
            for field in (
                "cluster_cap",
                "gross_cap",
                "industry_cap",
                "single_name_cap",
                "turnover_cap",
            )
        )
        or values["gross_cap"] + values["cash_floor"] > 1
    ):
        raise ArtifactContractError("risk policy limits are incoherent")
    return result


def validate_pretrade_permissions(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PretradePermissionsArtifact:
    result = _common(
        payload,
        PretradePermissionsArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, PretradePermissionsArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if require_utc_timestamp(payload["created_at"], label="created_at") < cutoff:
        raise ArtifactContractError("pretrade permissions created_at precedes cutoff")
    _ref(
        payload["canonical_calendar_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=("myquant.v17.v4.dataset.cn_open_day_calendar.v1"),
        label="canonical_calendar_ref",
    )
    _ref(
        payload["pit_catalog_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=("myquant.v17.v4.pit-generation-catalog.v1"),
        label="pit_catalog_ref",
    )
    _ref(
        payload["holdings_snapshot_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.holdings-snapshot.v1",
        label="holdings_snapshot_ref",
    )
    _ref(
        payload["risk_policy_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.portfolio-risk-policy.v1",
        label="risk_policy_ref",
    )
    rows = payload["payload"]
    symbols = [row["symbol"] for row in rows]
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("pretrade permissions must be unique and symbol ordered")
    for index, row in enumerate(rows):
        current = _decimal(
            row["current_target"],
            label=f"payload[{index}].current_target",
        )
        if (
            current < 0
            or row["held"] != (current > 0)
            or (row["lane"] == "REVIEW_ONLY_HOLDING" and not row["held"])
        ):
            raise ArtifactContractError("pretrade permission holding truth table mismatch")
    return result


def validate_portfolio_overlay(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PortfolioOverlayArtifact:
    result = _common(
        payload,
        PortfolioOverlayArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, PortfolioOverlayArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if require_utc_timestamp(payload["created_at"], label="created_at") < cutoff:
        raise ArtifactContractError("portfolio overlay created_at precedes cutoff")
    _ref(
        payload["baseline_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        version_prefix="myquant.v17.v4.",
        label="baseline_ref",
    )
    for field, version in (
        ("permissions_ref", "myquant.v17.v4.pretrade-permissions.v1"),
        ("risk_policy_ref", "myquant.v17.v4.portfolio-risk-policy.v1"),
    ):
        _ref(
            payload[field],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version=version,
            label=field,
        )
    evidence = _refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="evidence_refs",
    )
    if not evidence:
        raise ArtifactContractError("APPLIED portfolio overlay requires evidence")
    rows = payload["target_weights"]
    symbols = [row["symbol"] for row in rows]
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("overlay targets must be unique and symbol ordered")
    input_gross = _decimal(payload["input_gross"], label="input_gross")
    output_gross = _decimal(payload["output_gross"], label="output_gross")
    released = _decimal(payload["released_to_cash"], label="released_to_cash")
    observed = sum(
        (
            _decimal(row["target"], label=f"target_weights[{index}].target")
            for index, row in enumerate(rows)
        ),
        Decimal("0"),
    )
    if (
        input_gross < 0
        or output_gross < 0
        or output_gross > input_gross
        or output_gross != observed
        or released != input_gross - output_gross
    ):
        raise ArtifactContractError("portfolio overlay gross reconciliation mismatch")
    return result


def validate_regime_evidence(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> RegimeEvidenceArtifact:
    result = _common(
        payload,
        RegimeEvidenceArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, RegimeEvidenceArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(payload["created_at"], label="created_at") < cutoff
        or require_utc_timestamp(payload["available_at"], label="available_at") > cutoff
    ):
        raise ArtifactContractError("regime evidence time closure mismatch")
    multiplier = _decimal(
        payload["gross_multiplier"],
        label="gross_multiplier",
    )
    if multiplier < 0 or multiplier > 1:
        raise ArtifactContractError("regime gross multiplier must be in [0, 1]")
    return result


def validate_portfolio_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PortfolioOutputArtifact:
    result = _common(
        payload,
        PortfolioOutputArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, PortfolioOutputArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if require_utc_timestamp(payload["created_at"], label="created_at") < cutoff:
        raise ArtifactContractError("portfolio output created_at precedes cutoff")
    for field, version in (
        ("deep_bundle_ref", "myquant.v17.v4.deep-evidence-bundle.v1"),
        ("fusion_top24_ref", "myquant.v17.v4.fusion-top24.v1"),
        ("holdings_snapshot_ref", "myquant.v17.v4.holdings-snapshot.v1"),
        ("macro_overlay_ref", "myquant.v17.v4.portfolio-overlay.v1"),
        ("markov_overlay_ref", "myquant.v17.v4.portfolio-overlay.v1"),
        ("permissions_ref", "myquant.v17.v4.pretrade-permissions.v1"),
        ("risk_policy_ref", "myquant.v17.v4.portfolio-risk-policy.v1"),
    ):
        _ref(
            payload[field],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version=version,
            label=field,
        )
    selected = payload["selection_pool_symbols"]
    if len(selected) != 24 or len(selected) != len(set(selected)):
        raise ArtifactContractError("portfolio selection pool must be exact Top24")
    rows = payload["targets"]
    symbols = [row["symbol"] for row in rows]
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("portfolio targets must be unique and symbol ordered")
    if not set(selected).issubset(symbols):
        raise ArtifactContractError("portfolio targets omit a Top24 symbol")
    gross = sum(
        (
            _decimal(row["final_target"], label=f"targets[{index}].final_target")
            for index, row in enumerate(rows)
        ),
        Decimal("0"),
    )
    declared_gross = _decimal(payload["gross_weight"], label="gross_weight")
    cash = _decimal(payload["cash_weight"], label="cash_weight")
    if gross != declared_gross or gross < 0 or cash < 0 or gross + cash != 1:
        raise ArtifactContractError("portfolio output gross/cash reconciliation mismatch")
    return result


_BOOTSTRAP_MATRIX_SHA256: Final = "8e4467cf152ca8de71c94ed1a20715a18ba8eefa19428217541e0baa17df9458"


def _next_calendar_month(value: str) -> str:
    year, month = (int(part) for part in value.split("-"))
    if month == 12:
        return f"{year + 1:04d}-01"
    return f"{year:04d}-{month + 1:02d}"


def _consecutive_month_end_dates(
    values: Sequence[str],
    *,
    label: str,
) -> None:
    if list(values) != sorted(values) or len(values) != len(set(values)):
        raise ArtifactContractError(f"{label} must be unique and ascending")
    for value in values:
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except ValueError as exc:
            raise ArtifactContractError(f"{label} contains a noncanonical date") from exc
    months = [value[:7] for value in values]
    for previous, current in zip(months, months[1:]):
        if _next_calendar_month(previous) != current:
            raise ArtifactContractError(f"{label} skips a calendar month")


def _bootstrap(value: Any) -> None:
    if value != {
        "block_length_months": 12,
        "generator": "PCG64",
        "matrix_sha256": _BOOTSTRAP_MATRIX_SHA256,
        "replicates": 10_000,
        "seed": 170_317,
    }:
        raise ArtifactContractError("calibration bootstrap identity mismatch")


def validate_calibration_origin_inventory(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> CalibrationOriginInventoryArtifact:
    result = _common(
        payload,
        CalibrationOriginInventoryArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, CalibrationOriginInventoryArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    created = require_utc_timestamp(
        payload["created_at"],
        label="created_at",
    )
    if created < cutoff:
        raise ArtifactContractError("calibration inventory created_at precedes cutoff")
    rows = payload["origins"]
    if len(rows) != payload["input_origin_count"]:
        raise ArtifactContractError("calibration origin inventory count mismatch")
    origins = [row["origin"] for row in rows]
    _consecutive_month_end_dates(origins, label="calibration origins")
    closure = payload["closure_origins"]
    _consecutive_month_end_dates(
        closure,
        label="calibration closure origins",
    )
    if closure != origins[-120:]:
        raise ArtifactContractError("calibration closure must be the last 120 origins")
    for index, row in enumerate(rows):
        _factor_control_ref(
            row["factor_active_set_ref"],
            label=f"origins[{index}].factor_active_set_ref",
        )
        require_sha256(
            row["factor_set_sha256"],
            label=f"origins[{index}].factor_set_sha256",
        )
        require_sha256(
            row["origin_semantic_sha256"],
            label=f"origins[{index}].origin_semantic_sha256",
        )
        require_sha256(
            row["source_closure_sha256"],
            label=f"origins[{index}].source_closure_sha256",
        )
    return result


def validate_calibration_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
    artifact_loader: Callable[[Mapping[str, str]], bytes] | None = None,
) -> CalibrationReceiptArtifact:
    result = _common(
        payload,
        CalibrationReceiptArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, CalibrationReceiptArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("calibration receipt created_at precedes cutoff")
    expected_span = {
        "QUANT_TIMING": 1260,
        "FUNDAMENTAL_FORWARD": 2520,
    }[payload["calibration_kind"]]
    if (
        payload["accepted"] is not True
        or payload["status"] != "ACCEPTED"
        or payload["minimum_open_session_span"] != expected_span
        or payload["input_origin_count"] < payload["closure_origin_count"]
    ):
        raise ArtifactContractError("calibration receipt gate result mismatch")
    _bootstrap(payload["bootstrap"])
    inventory_ref = _ref(
        payload["origin_inventory_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=("myquant.v17.v4.calibration-origin-inventory.v1"),
        label="origin_inventory_ref",
    )
    inventory_document = _read_exact_ref(
        inventory_ref,
        artifact_loader=artifact_loader,
        expected_version=("myquant.v17.v4.calibration-origin-inventory.v1"),
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="origin_inventory_ref",
    )
    inventory = validate_calibration_origin_inventory(
        inventory_document,
    )
    if (
        inventory_document["cutoff"] != payload["cutoff"]
        or inventory_document["run_id"] != payload["run_id"]
        or inventory_document["closure_origin_count"] != payload["closure_origin_count"]
        or inventory_document["input_origin_count"] != payload["input_origin_count"]
    ):
        raise ArtifactContractError("calibration receipt inventory binding mismatch")
    if inventory.strategy_id != result.strategy_id:
        raise ArtifactContractError("calibration receipt inventory strategy mismatch")
    return result


def validate_fusion_promotion_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
    artifact_loader: Callable[[Mapping[str, str]], bytes] | None = None,
) -> FusionPromotionReceiptArtifact:
    result = _common(
        payload,
        FusionPromotionReceiptArtifact,
        formal_research_publication=False,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FusionPromotionReceiptArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if (
        require_utc_timestamp(
            payload["created_at"],
            label="created_at",
        )
        < cutoff
    ):
        raise ArtifactContractError("fusion receipt created_at precedes cutoff")
    _bootstrap(payload["bootstrap"])
    inventory = _ref(
        payload["origin_inventory_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version=("myquant.v17.v4.calibration-origin-inventory.v1"),
        label="origin_inventory_ref",
    )
    quant = _ref(
        payload["quant_calibration_receipt_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.calibration-receipt.v1",
        label="quant_calibration_receipt_ref",
    )
    fundamental = _ref(
        payload["fundamental_calibration_receipt_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.calibration-receipt.v1",
        label="fundamental_calibration_receipt_ref",
    )
    if (
        len(
            {
                inventory["byte_sha256"],
                quant["byte_sha256"],
                fundamental["byte_sha256"],
            }
        )
        != 3
    ):
        raise ArtifactContractError("fusion receipt contains colliding source refs")
    inventory_document = _read_exact_ref(
        inventory,
        artifact_loader=artifact_loader,
        expected_version=("myquant.v17.v4.calibration-origin-inventory.v1"),
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="origin_inventory_ref",
    )
    validate_calibration_origin_inventory(inventory_document)
    quant_document = _read_exact_ref(
        quant,
        artifact_loader=artifact_loader,
        expected_version="myquant.v17.v4.calibration-receipt.v1",
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="quant_calibration_receipt_ref",
    )
    fundamental_document = _read_exact_ref(
        fundamental,
        artifact_loader=artifact_loader,
        expected_version="myquant.v17.v4.calibration-receipt.v1",
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="fundamental_calibration_receipt_ref",
    )
    validate_calibration_receipt(
        quant_document,
        artifact_loader=artifact_loader,
    )
    validate_calibration_receipt(
        fundamental_document,
        artifact_loader=artifact_loader,
    )
    if (
        inventory_document["cutoff"] != payload["cutoff"]
        or inventory_document["run_id"] != payload["run_id"]
        or quant_document["calibration_kind"] != "QUANT_TIMING"
        or fundamental_document["calibration_kind"] != "FUNDAMENTAL_FORWARD"
        or quant_document["origin_inventory_ref"] != inventory
        or fundamental_document["origin_inventory_ref"] != inventory
        or quant_document["bootstrap"] != payload["bootstrap"]
        or fundamental_document["bootstrap"] != payload["bootstrap"]
        or quant_document["run_id"] != payload["run_id"]
        or fundamental_document["run_id"] != payload["run_id"]
        or quant_document["closure_origin_count"] != inventory_document["closure_origin_count"]
        or fundamental_document["closure_origin_count"]
        != inventory_document["closure_origin_count"]
        or quant_document["input_origin_count"] != inventory_document["input_origin_count"]
        or fundamental_document["input_origin_count"] != inventory_document["input_origin_count"]
    ):
        raise ArtifactContractError("fusion receipt calibration binding mismatch")
    folds = payload["folds"]
    if [row["fold_index"] for row in folds] != [1, 2, 3, 4, 5]:
        raise ArtifactContractError("fusion fold inventory mismatch")
    stitched: list[str] = []
    for row in folds:
        training = row["training_origins"]
        oos = row["oos_origins"]
        _consecutive_month_end_dates(
            training,
            label=f"fold_{row['fold_index']}.training_origins",
        )
        _consecutive_month_end_dates(
            oos,
            label=f"fold_{row['fold_index']}.oos_origins",
        )
        if training[-1] >= oos[0]:
            raise ArtifactContractError("fusion fold training is not before OOS")
        weight = _decimal(
            row["selected_quant_weight"],
            label="selected_quant_weight",
        )
        if weight not in {Decimal(value) / Decimal(100) for value in range(25, 76, 5)}:
            raise ArtifactContractError("fusion fold weight is outside the frozen grid")
        stitched.extend(oos)
    active = payload["active_refit_origins"]
    _consecutive_month_end_dates(
        active,
        label="active_refit_origins",
    )
    if stitched != active:
        raise ArtifactContractError("fusion outer OOS origins differ from active refit origins")
    accepted = payload["accepted"]
    expected_status = "PROMOTED" if accepted else "CALIBRATION_CLOSURE_BLOCKED"
    if payload["status"] != expected_status:
        raise ArtifactContractError("fusion promotion status/accepted mismatch")
    hit_lower = _decimal(
        payload["oos_hit60_lower_95"],
        label="oos_hit60_lower_95",
    )
    q25_lower = _decimal(
        payload["oos_q25_252_lower_95"],
        label="oos_q25_252_lower_95",
    )
    blockers = payload["blockers"]
    expected_blockers: list[str] = []
    if hit_lower <= Decimal("0.50"):
        expected_blockers.append("oos_hit60_lower_95_not_above_0.50")
    if q25_lower <= 0:
        expected_blockers.append("oos_q25_252_lower_95_not_above_zero")
    if blockers != expected_blockers or accepted == bool(expected_blockers):
        raise ArtifactContractError("fusion promotion threshold closure mismatch")
    return result


_VALIDATORS: Final[Mapping[str, Callable[..., ValidatedArtifact]]] = {
    "myquant.v17.v4.branch-output.v1": validate_branch_output,
    "myquant.v17.v4.calibration-origin-inventory.v1": (validate_calibration_origin_inventory),
    "myquant.v17.v4.calibration-receipt.v1": (validate_calibration_receipt),
    "myquant.v17.v4.canary-pointer.v1": validate_canary_pointer,
    "myquant.v17.v4.canary-public-snapshot.v1": (validate_canary_public_snapshot),
    "myquant.v17.v4.canary-receipt.v1": validate_canary_receipt,
    "myquant.v17.v4.canary-transition-intent.v1": (validate_canary_transition_intent),
    "myquant.v17.v4.default-eligibility-receipt.v1": (validate_default_eligibility_receipt),
    "myquant.v17.v4.default-eligibility-intent.v1": (validate_default_eligibility_intent),
    "myquant.v17.v4.default-eligible-pointer.v1": validate_default_eligible_pointer,
    "myquant.v17.v4.deep-assessment-manifest.v1": (validate_deep_assessment_manifest),
    "myquant.v17.v4.deep-assessment-manifest.v2": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.deep-evidence-bundle.v1": (validate_deep_evidence_bundle),
    "myquant.v17.v4.deep-evidence-bundle.v2": (validate_deep_evidence_bundle_v2),
    "myquant.v17.v4.deep-evidence-bundle.v3": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.dual-run-comparison.v1": validate_dual_run_comparison,
    "myquant.v17.v4.event-scan.v1": validate_event_scan,
    "myquant.v17.v4.event-scan.v2": validate_event_scan_v2,
    "myquant.v17.v4.event-scan.v3": validate_research_shadow_artifact,
    "myquant.v17.v4.formal-activation-receipt.v1": (validate_formal_activation_receipt),
    "myquant.v17.v4.formal-activation-intent.v1": (validate_formal_activation_intent),
    "myquant.v17.v4.formal-activation-rejection.v1": (validate_formal_activation_rejection),
    "myquant.v17.v4.formal-active-pointer.v1": validate_formal_active_pointer,
    "myquant.v17.v4.formal-output.v1": validate_formal_output,
    "myquant.v17.v4.fusion-promotion-receipt.v1": (validate_fusion_promotion_receipt),
    "myquant.v17.v4.fusion-top24.v1": validate_fusion_top24,
    "myquant.v17.v4.fusion-top24.v2": validate_research_shadow_artifact,
    "myquant.v17.v4.historical-canary-policy.v1": (validate_historical_canary_policy),
    "myquant.v17.v4.holdings-snapshot.v1": (validate_holdings_snapshot),
    "myquant.v17.v4.initial-pool-output.v1": (validate_initial_pool_output),
    "myquant.v17.v4.issuer-dossier.v1": validate_issuer_dossier,
    "myquant.v17.v4.issuer-dossier.v2": validate_issuer_dossier_v2,
    "myquant.v17.v4.issuer-dossier.v3": validate_research_shadow_artifact,
    "myquant.v17.v4.official-evidence.v1": (validate_official_evidence),
    "myquant.v17.v4.official-evidence.v2": (validate_official_evidence_v2),
    "myquant.v17.v4.official-evidence.v3": validate_research_shadow_artifact,
    "myquant.v17.v4.portfolio-output.v1": validate_portfolio_output,
    "myquant.v17.v4.portfolio-overlay.v1": validate_portfolio_overlay,
    "myquant.v17.v4.portfolio-risk-policy.v1": (validate_portfolio_risk_policy),
    "myquant.v17.v4.pretrade-permissions.v1": (validate_pretrade_permissions),
    "myquant.v17.v4.regime-evidence.v1": validate_regime_evidence,
    "myquant.v17.v4.research-factor-shadow-assertion.v1": (
        validate_research_factor_shadow_assertion
    ),
    "myquant.v17.v4.research-factor-shadow-assertion.v2": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.research-factor-input-bundle.v1": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.research-fundamental-branch-output.v2": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.research-initial-pool-output.v2": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.research-quant-branch-output.v1": (
        validate_research_quant_branch_output
    ),
    "myquant.v17.v4.research-quant-branch-output.v2": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.research-source-locator.v2": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.research-shadow-factor-set.v1": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.research-shadow-factor-set-pointer.v1": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.shadow-fusion-matured-label.v1": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.shadow-fusion-observation.v1": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.shadow-fusion-policy.v1": (
        validate_research_shadow_artifact
    ),
    "myquant.v17.v4.shadow-readiness.v1": (validate_shadow_readiness),
    "myquant.v17.v4.shadow-readiness.v2": validate_research_shadow_artifact,
    "myquant.v17.v4.shadow-run.v1": validate_shadow_run,
    "myquant.v17.v4.shadow-run.v2": validate_shadow_run_v2,
    "myquant.v17.v4.shadow-run.v3": validate_research_shadow_artifact,
    "myquant.v17.v4.shadow-session-ref.v1": (validate_shadow_session_ref),
    "myquant.v17.v4.shadow-session-ref.v2": (validate_shadow_session_ref),
    "myquant.v17.v4.shadow-session-ref.v3": validate_research_shadow_artifact,
    "myquant.v17.v4.rollback-drill-receipt.v1": (validate_rollback_drill_receipt),
    "myquant.v17.v4.pit-catalog-pointer.v1": validate_pit_catalog_pointer,
    "myquant.v17.v4.pit-generation-catalog.v1": (validate_pit_generation_catalog),
    "myquant.v17.v4.preselect-locator.v1": (validate_preselect_locator),
    "myquant.v17.v4.public-surface-compatibility-receipt.v1": (
        validate_public_surface_compatibility_receipt
    ),
    "myquant.v17.v4.public-run-dto.v1": validate_public_run_dto,
    "myquant.v17.v4.total-return-labels.v1": (validate_total_return_labels),
    "myquant.v17.v4.validation-receipt.v1": (validate_validation_receipt),
}


def validate_typed_artifact(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
    artifact_loader: Callable[[Mapping[str, str]], bytes] | None = None,
) -> ValidatedArtifact:
    version = payload.get("version")
    validator = _VALIDATORS.get(version)
    if validator is None:
        raise ArtifactContractError(f"unsupported v4 artifact version: {version!r}")
    if version in {
        "myquant.v17.v4.calibration-receipt.v1",
        "myquant.v17.v4.fusion-promotion-receipt.v1",
    }:
        return validator(
            payload,
            schema_checked=schema_checked,
            artifact_loader=artifact_loader,
        )
    return validator(payload, schema_checked=schema_checked)


__all__ = [
    "ArtifactContractError",
    "BranchOutputArtifact",
    "CalibrationOriginInventoryArtifact",
    "CalibrationReceiptArtifact",
    "CanaryPointerArtifact",
    "CanaryPublicSnapshotArtifact",
    "CanaryReceiptArtifact",
    "CanaryTransitionIntentArtifact",
    "DefaultEligibilityReceiptArtifact",
    "DefaultEligibilityIntentArtifact",
    "DefaultEligiblePointerArtifact",
    "DeepAssessmentManifestArtifact",
    "DeepEvidenceBundleArtifact",
    "DeepEvidenceBundleV2Artifact",
    "DualRunComparisonArtifact",
    "EventScanArtifact",
    "EventScanV2Artifact",
    "FormalActivationReceiptArtifact",
    "FormalActivationIntentArtifact",
    "FormalActivationRejectionArtifact",
    "FormalActivePointerArtifact",
    "FormalOutputArtifact",
    "FusionPromotionReceiptArtifact",
    "FusionTop24Artifact",
    "HistoricalCanaryPolicyArtifact",
    "HoldingsSnapshotArtifact",
    "InitialPoolOutputArtifact",
    "IssuerDossierArtifact",
    "IssuerDossierV2Artifact",
    "OfficialEvidenceArtifact",
    "OfficialEvidenceV2Artifact",
    "PitCatalogPointerArtifact",
    "PitGenerationCatalogArtifact",
    "PortfolioOutputArtifact",
    "PortfolioOverlayArtifact",
    "PortfolioRiskPolicyArtifact",
    "PretradePermissionsArtifact",
    "RegimeEvidenceArtifact",
    "ResearchFactorShadowAssertionArtifact",
    "ResearchQuantBranchOutputArtifact",
    "ShadowReadinessArtifact",
    "ShadowRunArtifact",
    "ShadowSessionRefArtifact",
    "RollbackDrillReceiptArtifact",
    "PreselectLocatorArtifact",
    "PublicSurfaceCompatibilityReceiptArtifact",
    "PublicRunDTOArtifact",
    "TotalReturnLabelsArtifact",
    "ValidatedArtifact",
    "ValidationReceiptArtifact",
    "validate_canary_pointer",
    "validate_canary_public_snapshot",
    "validate_canary_receipt",
    "validate_canary_transition_intent",
    "validate_calibration_origin_inventory",
    "validate_calibration_receipt",
    "validate_branch_output",
    "validate_default_eligibility_receipt",
    "validate_default_eligibility_intent",
    "validate_default_eligible_pointer",
    "validate_deep_assessment_manifest",
    "validate_deep_evidence_bundle",
    "validate_deep_evidence_bundle_v2",
    "validate_dual_run_comparison",
    "validate_event_scan",
    "validate_event_scan_v2",
    "validate_formal_activation_receipt",
    "validate_formal_activation_intent",
    "validate_formal_activation_rejection",
    "validate_formal_active_pointer",
    "validate_formal_output",
    "validate_fusion_promotion_receipt",
    "validate_fusion_top24",
    "validate_historical_canary_policy",
    "validate_holdings_snapshot",
    "validate_initial_pool_output",
    "validate_issuer_dossier",
    "validate_issuer_dossier_v2",
    "validate_official_evidence",
    "validate_official_evidence_v2",
    "validate_pit_catalog_pointer",
    "validate_portfolio_output",
    "validate_portfolio_overlay",
    "validate_portfolio_risk_policy",
    "validate_pretrade_permissions",
    "validate_regime_evidence",
    "validate_research_factor_shadow_assertion",
    "validate_research_quant_branch_output",
    "validate_shadow_readiness",
    "validate_shadow_run",
    "validate_shadow_session_ref",
    "validate_rollback_drill_receipt",
    "validate_preselect_locator",
    "validate_public_surface_compatibility_receipt",
    "validate_public_run_dto",
    "validate_total_return_labels",
    "validate_pit_generation_catalog",
    "validate_typed_artifact",
    "validate_validation_receipt",
]

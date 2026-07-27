"""Pure cross-document validators for the V17 v4 contract scaffold."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Final

from .canonical import CanonicalContractError, validate_semantic_sha
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
class FormalActivePointerArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FormalOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DefaultEligibilityReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DefaultEligiblePointerArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class CanaryReceiptArtifact(ValidatedArtifact):
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
        raise ArtifactContractError(
            "v4 authority exceeds the artifact state authority ceiling"
        )


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


def _ref(
    value: Any,
    *,
    strategy_id: str,
    cutoff: str | None = None,
    expected_version: str | None = None,
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
    if expected_version is not None and version != expected_version:
        raise ArtifactContractError(f"{label} artifact version mismatch")
    if version_prefix is not None and not version.startswith(version_prefix):
        raise ArtifactContractError(f"{label} protocol identity mismatch")
    if version.startswith("myquant.v17.v3."):
        raise ArtifactContractError(f"{label} cannot relabel a v3 artifact as v4")
    path = value["relative_path"]
    if (
        type(path) is not str
        or path.startswith("/")
        or ".." in path.split("/")
        or "\\" in path
    ):
        raise ArtifactContractError(f"{label} path is unsafe")
    return dict(value)


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
    identities = [
        (row["relative_path"], row["byte_sha256"], row["artifact_id"])
        for row in result
    ]
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
        (row["artifact_version"], row["relative_path"], row["byte_sha256"])
        for row in evidence
    }
    required_keys = {
        (row["artifact_version"], row["relative_path"], row["byte_sha256"])
        for row in required
    }
    if not required_keys.issubset(evidence_keys):
        raise ArtifactContractError(f"{label} omits an explicit transition reference")


def validate_formal_activation_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FormalActivationReceiptArtifact:
    success = payload.get("status") == "FORMAL_ACTIVATED"
    result = _common(
        payload,
        FormalActivationReceiptArtifact,
        formal_research_publication=success,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FormalActivationReceiptArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    if payload["to_state"] != ("FORMAL_ACTIVE" if success else "V15_DEFAULT"):
        raise ArtifactContractError("formal transition state/status mismatch")
    expected = {
        "formal_output_ref": "myquant.v17.v4.formal-output.v1",
        "source_locator_ref": "myquant.v17.v4.",
        "quant_calibration_receipt_ref": "myquant.v17.v4.",
        "fundamental_calibration_receipt_ref": "myquant.v17.v4.",
        "fusion_promotion_receipt_ref": "myquant.v17.v4.",
        "deep_bundle_ref": "myquant.v17.v4.",
        "holdings_snapshot_ref": "myquant.v17.v4.",
        "risk_policy_ref": "myquant.v17.v4.",
        "macro_overlay_ref": "myquant.v17.v4.",
        "markov_overlay_ref": "myquant.v17.v4.",
        "factor_control_active_set_ref": "factor-governance-production-control.",
        "factor_control_activation_receipt_ref": (
            "factor-governance-production-control."
        ),
    }
    explicit = [
        _ref(
            payload[field],
            strategy_id=result.strategy_id,
            cutoff=cutoff,
            expected_version=(
                prefix
                if field == "formal_output_ref"
                else None
            ),
            version_prefix=(
                None
                if field == "formal_output_ref"
                else prefix
            ),
            label=field,
        )
        for field, prefix in expected.items()
    ]
    evidence = _refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        label="evidence_refs",
    )
    _require_evidence_contains(evidence, explicit, label="formal activation evidence")
    _cas(payload, success=success, label="formal activation")
    return result


def validate_formal_active_pointer(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FormalActivePointerArtifact:
    result = _common(
        payload,
        FormalActivePointerArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FormalActivePointerArtifact)
    cutoff = require_utc_timestamp(payload["cutoff"], label="cutoff")
    _ref(
        payload["receipt_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.formal-activation-receipt.v1",
        label="receipt_ref",
    )
    _ref(
        payload["formal_output_ref"],
        strategy_id=result.strategy_id,
        cutoff=cutoff,
        expected_version="myquant.v17.v4.formal-output.v1",
        label="formal_output_ref",
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
    success = payload["status"] == "DEFAULT_ELIGIBLE"
    if payload["to_state"] != ("DEFAULT_ELIGIBLE" if success else "FORMAL_ACTIVE"):
        raise ArtifactContractError("eligibility transition state/status mismatch")
    explicit = [
        _ref(
            payload["formal_active_pointer_ref"],
            strategy_id=result.strategy_id,
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
            version_prefix="myquant.research-runtime.",
            label="rollback_drill_receipt_ref",
        ),
    ]
    public = _refs(
        payload["public_surface_receipt_refs"],
        strategy_id=result.strategy_id,
        cutoff=None,
        label="public_surface_receipt_refs",
    )
    validation = _refs(
        payload["validation_receipt_refs"],
        strategy_id=result.strategy_id,
        cutoff=None,
        label="validation_receipt_refs",
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
        label="default eligibility evidence",
    )
    _cas(payload, success=success, label="default eligibility")
    return result


def validate_default_eligible_pointer(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DefaultEligiblePointerArtifact:
    result = _common(
        payload,
        DefaultEligiblePointerArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, DefaultEligiblePointerArtifact)
    _ref(
        payload["formal_active_pointer_ref"],
        strategy_id=result.strategy_id,
        expected_version="myquant.v17.v4.formal-active-pointer.v1",
        label="formal_active_pointer_ref",
    )
    _ref(
        payload["eligibility_receipt_ref"],
        strategy_id=result.strategy_id,
        expected_version="myquant.v17.v4.default-eligibility-receipt.v1",
        label="eligibility_receipt_ref",
    )
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
    status = payload["status"]
    expected_states = {
        "CANARY_STARTED": ("DEFAULT_ELIGIBLE", "CANARY"),
        "CANARY_COMPLETED": ("CANARY", "CANARY"),
        "CANARY_FAILED": ("CANARY", "DEFAULT_ELIGIBLE"),
    }
    if (payload["from_state"], payload["to_state"]) != expected_states[status]:
        raise ArtifactContractError("canary transition state/status mismatch")
    explicit = [
        _ref(
            payload["eligibility_pointer_ref"],
            strategy_id=result.strategy_id,
            expected_version="myquant.v17.v4.default-eligible-pointer.v1",
            label="eligibility_pointer_ref",
        ),
        _ref(
            payload["historical_canary_policy_ref"],
            strategy_id=result.strategy_id,
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
    if status == "CANARY_STARTED":
        if completion_fields & set(payload):
            raise ArtifactContractError(
                "CANARY_STARTED cannot carry completion-only fields"
            )
        if len(payload["paired_run_ids"]) != 1:
            raise ArtifactContractError(
                "CANARY_STARTED must bind exactly the first paired-run ID"
            )
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
        if status == "CANARY_COMPLETED":
            if any(
                row["status"] != "PASS" for row in payload["threshold_results"]
            ) or any(value != 0 for value in payload["side_effect_counters"].values()):
                raise ArtifactContractError(
                    "CANARY_COMPLETED requires all thresholds and counters to pass"
                )
    _require_evidence_contains(evidence, explicit, label="canary evidence")
    _cas(payload, success=True, label="canary pointer")
    return result


def validate_canary_pointer(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> CanaryPointerArtifact:
    result = _common(
        payload,
        CanaryPointerArtifact,
        formal_research_publication=True,
        schema_checked=schema_checked,
    )
    assert isinstance(result, CanaryPointerArtifact)
    _ref(
        payload["eligibility_pointer_ref"],
        strategy_id=result.strategy_id,
        expected_version="myquant.v17.v4.default-eligible-pointer.v1",
        label="eligibility_pointer_ref",
    )
    _ref(
        payload["canary_receipt_ref"],
        strategy_id=result.strategy_id,
        expected_version="myquant.v17.v4.canary-receipt.v1",
        label="canary_receipt_ref",
    )
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
            raise ArtifactContractError(
                "COMPARABLE dual run must have exact lower-level bytes"
            )
    elif not mismatched or not differing:
        raise ArtifactContractError(
            "NON_COMPARABLE dual run must bind exact differing refs"
        )
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


_VALIDATORS: Final[Mapping[str, Callable[..., ValidatedArtifact]]] = {
    "myquant.v17.v4.canary-pointer.v1": validate_canary_pointer,
    "myquant.v17.v4.canary-receipt.v1": validate_canary_receipt,
    "myquant.v17.v4.default-eligibility-receipt.v1": (
        validate_default_eligibility_receipt
    ),
    "myquant.v17.v4.default-eligible-pointer.v1": validate_default_eligible_pointer,
    "myquant.v17.v4.dual-run-comparison.v1": validate_dual_run_comparison,
    "myquant.v17.v4.formal-activation-receipt.v1": (
        validate_formal_activation_receipt
    ),
    "myquant.v17.v4.formal-active-pointer.v1": validate_formal_active_pointer,
    "myquant.v17.v4.formal-output.v1": validate_formal_output,
    "myquant.v17.v4.historical-canary-policy.v1": (
        validate_historical_canary_policy
    ),
}


def validate_typed_artifact(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ValidatedArtifact:
    version = payload.get("version")
    validator = _VALIDATORS.get(version)
    if validator is None:
        raise ArtifactContractError(f"unsupported v4 artifact version: {version!r}")
    return validator(payload, schema_checked=schema_checked)


__all__ = [
    "ArtifactContractError",
    "CanaryPointerArtifact",
    "CanaryReceiptArtifact",
    "DefaultEligibilityReceiptArtifact",
    "DefaultEligiblePointerArtifact",
    "DualRunComparisonArtifact",
    "FormalActivationReceiptArtifact",
    "FormalActivePointerArtifact",
    "FormalOutputArtifact",
    "HistoricalCanaryPolicyArtifact",
    "ValidatedArtifact",
    "validate_canary_pointer",
    "validate_canary_receipt",
    "validate_default_eligibility_receipt",
    "validate_default_eligible_pointer",
    "validate_dual_run_comparison",
    "validate_formal_activation_receipt",
    "validate_formal_active_pointer",
    "validate_formal_output",
    "validate_historical_canary_policy",
    "validate_typed_artifact",
]

"""Typed, pure cross-document validators for protocol v3 artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
from types import MappingProxyType
from typing import Any, Final

from .canonical import CanonicalContractError, canonical_bytes, validate_semantic_sha
from .identities import (
    IdentityContractError,
    require_opaque_id,
    require_security_code,
    require_sha256,
    require_utc_cutoff,
)
from .namespace import NamespaceContractError, root_for_path
from .policy import (
    PolicyContractError,
    activation_statuses,
    state_machine,
    source_role_registries,
    role_requirement,
    terminal_class,
    validate_authority,
)
from .resources import PACKAGE_MANIFEST_SHA256, load_packaged_json

PROTOCOL_VERSION: Final = "myquant.v17.v3"
ACTIVATION_STATUSES: Final = activation_statuses()
ResolvedArtifactMap = Mapping[str, tuple[bytes, Mapping[str, Any]]]


class ArtifactContractError(ValueError):
    """Raised when cross-document v3 semantics fail closed."""

    exit_code = 2


@dataclass(frozen=True)
class ValidatedArtifact:
    version: str
    strategy_id: str
    cutoff: str
    semantic_sha256: str
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class SourceLocatorArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class SourceManifestArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class InitialPoolOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class BranchOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class CalibrationGateInputsArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FusionCalibrationReceiptArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FusionCalibrationInputsArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FusionPromotionReceiptArtifact(ValidatedArtifact):
    status: str


@dataclass(frozen=True)
class FusionOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DeepOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class DeepResearchInputsArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class QuantPreselectionInputsArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FactorGovernanceReadinessArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ProvisionalFactorBaselineArtifact(ValidatedArtifact):
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
class FormalResearchOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ShadowOutputArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ActivationReceiptArtifact(ValidatedArtifact):
    status: str


@dataclass(frozen=True)
class ActivationPointerArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ShadowLatestArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class FormalLatestArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class UnpublishedEvidenceArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class LedgerArtifact(ValidatedArtifact):
    pass


@dataclass(frozen=True)
class ActivationTransition:
    receipt: ActivationReceiptArtifact
    next_pointer: ActivationPointerArtifact | None


_TYPED_CLASS_BY_VERSION: Final = {
    "myquant.v17.v3.activation-pointer.v1": ActivationPointerArtifact,
    "myquant.v17.v3.branch-output.v1": BranchOutputArtifact,
    "myquant.v17.v3.calibration-gate-inputs.v1": CalibrationGateInputsArtifact,
    "myquant.v17.v3.deep-output.v1": DeepOutputArtifact,
    "myquant.v17.v3.deep-research-inputs.v1": DeepResearchInputsArtifact,
    "myquant.v17.v3.factor-governance-readiness.v1": (FactorGovernanceReadinessArtifact),
    "myquant.v17.v3.formal-latest.v1": FormalLatestArtifact,
    "myquant.v17.v3.formal-research-output.v1": FormalResearchOutputArtifact,
    "myquant.v17.v3.fusion-calibration-receipt.v1": FusionCalibrationReceiptArtifact,
    "myquant.v17.v3.fusion-calibration-inputs.v1": FusionCalibrationInputsArtifact,
    "myquant.v17.v3.fusion-output.v1": FusionOutputArtifact,
    "myquant.v17.v3.fusion-promotion-receipt.v1": FusionPromotionReceiptArtifact,
    "myquant.v17.v3.initial-pool-output.v1": InitialPoolOutputArtifact,
    "myquant.v17.v3.ledger.v1": LedgerArtifact,
    "myquant.v17.v3.portfolio-overlay.v1": PortfolioOverlayArtifact,
    "myquant.v17.v3.portfolio-output.v1": PortfolioOutputArtifact,
    "myquant.v17.v3.pretrade-permissions.v1": PretradePermissionsArtifact,
    "myquant.v17.v3.provisional-factor-baseline.v1": (ProvisionalFactorBaselineArtifact),
    "myquant.v17.v3.quant-preselection-inputs.v1": QuantPreselectionInputsArtifact,
    "myquant.v17.v3.shadow-latest.v1": ShadowLatestArtifact,
    "myquant.v17.v3.shadow-output.v1": ShadowOutputArtifact,
    "myquant.v17.v3.source-locator.v1": SourceLocatorArtifact,
    "myquant.v17.v3.source-manifest.v1": SourceManifestArtifact,
    "myquant.v17.v3.unpublished-evidence.v1": UnpublishedEvidenceArtifact,
}


def _schema_check(payload: Mapping[str, Any]) -> None:
    from .schema_validation import validate_schema_version

    validate_schema_version(payload, payload.get("version"))


def _common(
    payload: Mapping[str, Any],
    *,
    formal_authority: bool,
    schema_checked: bool,
) -> tuple[str, str, str, str, Mapping[str, Any]]:
    if type(payload) is not dict:
        raise ArtifactContractError("v3 artifact must be an object")
    if not schema_checked:
        _schema_check(payload)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ArtifactContractError("v3 artifact protocol mismatch")
    try:
        sealed = validate_semantic_sha(payload)
        version = require_opaque_id(
            payload.get("version"),
            label="artifact version",
        )
        strategy_id = require_opaque_id(payload.get("strategy_id"), label="strategy_id")
        cutoff = require_utc_cutoff(payload.get("cutoff"))
        digest = require_sha256(payload.get("semantic_sha256"), label="semantic_sha256")
        validate_authority(
            payload.get("authority"),
            formal_research_publication_authority=formal_authority,
        )
    except (CanonicalContractError, IdentityContractError, PolicyContractError) as exc:
        raise ArtifactContractError(str(exc)) from exc
    return version, strategy_id, cutoff, digest, MappingProxyType(sealed)


def _typed(
    payload: Mapping[str, Any],
    artifact_class: type[ValidatedArtifact],
    *,
    formal_authority: bool = False,
    schema_checked: bool,
) -> ValidatedArtifact:
    version, strategy_id, cutoff, digest, sealed = _common(
        payload,
        formal_authority=formal_authority,
        schema_checked=schema_checked,
    )
    return artifact_class(version, strategy_id, cutoff, digest, sealed)


def _packaged_policy_sha256(resource_name: str) -> str:
    value = load_packaged_json(f"resources/{resource_name}.v1.json").get("semantic_sha256")
    try:
        return require_sha256(value, label=f"{resource_name} semantic_sha256")
    except IdentityContractError as exc:
        raise ArtifactContractError(str(exc)) from exc


def _packaged_factor_inventory(resource_name: str) -> tuple[dict[str, Any], ...]:
    rows = load_packaged_json(f"resources/{resource_name}.v1.json").get("factor_inventory")
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ArtifactContractError(f"{resource_name} factor inventory is invalid")
    return tuple(dict(row) for row in rows)


def _validate_factor_baseline_binding(
    reference: Any,
    mode: Any,
    *,
    strategy_id: str,
    cutoff: str,
    label: str,
) -> dict[str, Any]:
    expected_versions = {
        "FACTOR_V4_PRODUCTION": ("myquant.v17.v3.factor-governance-readiness.v1"),
        "PROVISIONAL_RESEARCH": ("myquant.v17.v3.provisional-factor-baseline.v1"),
    }
    if mode not in expected_versions:
        raise ArtifactContractError(f"{label} factor_baseline_mode is invalid")
    return _validate_ref(
        reference,
        strategy_id=strategy_id,
        cutoff=cutoff,
        expected_version=expected_versions[mode],
        label=f"{label}.factor_baseline_ref",
    )


def _validate_ref(
    value: Any,
    *,
    strategy_id: str,
    cutoff: str | None,
    expected_version: str | None = None,
    label: str = "artifact reference",
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ArtifactContractError(f"{label} must be an object")
    expected_keys = {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
    if set(value) != expected_keys:
        raise ArtifactContractError(f"{label} shape mismatch")
    try:
        require_opaque_id(value["artifact_id"], label=f"{label}.artifact_id")
        require_opaque_id(value["artifact_version"], label=f"{label}.artifact_version")
        require_sha256(value["byte_sha256"], label=f"{label}.byte_sha256")
        require_sha256(value["semantic_sha256"], label=f"{label}.semantic_sha256")
        require_utc_cutoff(value["cutoff"], label=f"{label}.cutoff")
        root_for_path(value["relative_path"])
    except (IdentityContractError, NamespaceContractError) as exc:
        raise ArtifactContractError(str(exc)) from exc
    if value["strategy_id"] != strategy_id or (cutoff is not None and value["cutoff"] != cutoff):
        raise ArtifactContractError(f"{label} crosses strategy or cutoff")
    if expected_version is not None and value["artifact_version"] != expected_version:
        raise ArtifactContractError(f"{label} artifact version mismatch")
    if not value["artifact_version"].startswith("myquant.v17.v3."):
        raise ArtifactContractError(f"{label} references a non-v3 artifact")
    return dict(value)


def _sorted_unique_refs(
    values: Any,
    *,
    strategy_id: str,
    cutoff: str,
    label: str,
) -> tuple[dict[str, Any], ...]:
    if type(values) is not list:
        raise ArtifactContractError(f"{label} must be an array")
    refs = tuple(
        _validate_ref(value, strategy_id=strategy_id, cutoff=cutoff, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    keys = tuple((value["relative_path"], value["byte_sha256"]) for value in refs)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ArtifactContractError(f"{label} must be uniquely sorted by path and byte SHA")
    return refs


def _resolve_exact_reference(
    reference: Mapping[str, Any],
    resolved_artifacts: ResolvedArtifactMap,
    *,
    label: str,
) -> ValidatedArtifact:
    path = reference["relative_path"]
    entry = resolved_artifacts.get(path)
    if entry is None:
        raise ArtifactContractError(f"{label} is dangling")
    raw, document = entry
    try:
        from .references import build_artifact_ref

        observed = build_artifact_ref(document, raw, path)
    except (RuntimeError, ValueError) as exc:
        raise ArtifactContractError(f"{label} failed exact resolution") from exc
    if observed != dict(reference):
        raise ArtifactContractError(f"{label} exact reference binding mismatch")
    return validate_typed_artifact(document)


def _ranked_candidates(
    values: Any,
    *,
    evidence: bool,
) -> None:
    if type(values) is not list:
        raise ArtifactContractError("candidates must be an array")
    keys: list[tuple[int, str]] = []
    symbols: set[str] = set()
    for index, row in enumerate(values):
        if type(row) is not dict:
            raise ArtifactContractError(f"candidate {index} must be an object")
        try:
            symbol = require_security_code(row.get("symbol"), label=f"candidate {index} symbol")
        except IdentityContractError as exc:
            raise ArtifactContractError(str(exc)) from exc
        rank = row.get("rank")
        if type(rank) is not int or rank < 1:
            raise ArtifactContractError(f"candidate {index} rank must be positive")
        if symbol in symbols:
            raise ArtifactContractError(f"duplicate candidate symbol: {symbol}")
        if evidence:
            refs = row.get("evidence_refs")
            if type(refs) is not list:
                raise ArtifactContractError(f"candidate {index} evidence_refs must be an array")
            ref_keys = [
                (
                    (ref.get("relative_path"), ref.get("byte_sha256"))
                    if type(ref) is dict
                    else (None, None)
                )
                for ref in refs
            ]
            if ref_keys != sorted(ref_keys) or len(ref_keys) != len(set(ref_keys)):
                raise ArtifactContractError(
                    f"candidate {index} evidence refs are not uniquely ordered"
                )
        symbols.add(symbol)
        keys.append((rank, symbol))
    if keys != sorted(keys) or [rank for rank, _ in keys] != list(range(1, len(keys) + 1)):
        raise ArtifactContractError("candidate ranks must be contiguous in canonical order")


def _decimal(value: Any, *, label: str) -> Decimal:
    if type(value) is not str or not value or value.strip() != value:
        raise ArtifactContractError(f"{label} must be a canonical decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ArtifactContractError(f"{label} must be a canonical decimal string") from exc
    if not result.is_finite():
        raise ArtifactContractError(f"{label} must be finite")
    if format(result, "f") != value and not (value == "0" and result == 0):
        raise ArtifactContractError(f"{label} is not fixed-point canonical")
    return result


def validate_source_locator(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> SourceLocatorArtifact:
    result = _typed(
        payload,
        SourceLocatorArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, SourceLocatorArtifact)
    _validate_ref(
        payload["source_manifest_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        expected_version="myquant.v17.v3.source-manifest.v1",
        label="source_manifest_ref",
    )
    predecessor = payload["preselection_locator_ref"]
    if predecessor is not None:
        ref = _validate_ref(
            predecessor,
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.source-locator.v1",
            label="preselection_locator_ref",
        )
        if ref["artifact_id"] == payload["locator_id"]:
            raise ArtifactContractError("source locator cannot reference itself")
    return result


def validate_source_manifest(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> SourceManifestArtifact:
    result = _typed(
        payload,
        SourceManifestArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, SourceManifestArtifact)
    sources = payload["sources"]
    roles: list[str] = []
    for index, binding in enumerate(sources):
        if type(binding) is not dict:
            raise ArtifactContractError(f"source binding {index} must be an object")
        role = binding.get("role")
        if type(role) is not str:
            raise ArtifactContractError(f"source binding {index} role is invalid")
        _validate_ref(
            binding.get("artifact_ref"),
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            label=f"source binding {index}",
        )
        roles.append(role)
    if roles != sorted(roles) or len(roles) != len(set(roles)):
        raise ArtifactContractError("source bindings must be uniquely ordered by role")
    raw_roles, derived_roles = source_role_registries()
    if payload["closure_kind"] == "RAW":
        raw_profile = payload.get("raw_profile")
        legacy_baseline = {
            "cn_open_day_calendar",
            "corporate_actions",
            "market_bars",
            "pit_fundamentals",
            "universe_membership",
        }
        historical_formal_baseline = {
            *legacy_baseline,
            "benchmark_total_return",
            "factor_governance_readiness",
            "official_delisting_cash",
        }
        shadow_current_baseline = {
            "cn_open_day_calendar",
            "factor_governance_readiness",
            "market_bars",
            "pit_fundamentals",
            "universe_membership",
        }
        required_raw = {
            None: legacy_baseline,
            "HISTORICAL_FORMAL": historical_formal_baseline,
            "SHADOW_CURRENT": shadow_current_baseline,
        }.get(raw_profile)
        if required_raw is None:
            raise ArtifactContractError("RAW source manifest raw_profile is invalid")
        if not required_raw.issubset(roles) or not set(roles).issubset(raw_roles):
            raise ArtifactContractError(
                "RAW source manifest does not satisfy its exact raw_profile baseline"
            )
        if raw_profile is None and "factor_governance_readiness" in roles:
            raise ArtifactContractError(
                "legacy historical RAW source manifest cannot carry shadow readiness"
            )
        if raw_profile == "SHADOW_CURRENT" and "holdings_snapshot" in roles:
            raise ArtifactContractError(
                "SHADOW_CURRENT RAW source manifest forbids private holdings"
            )
        expected_raw_versions = {
            "factor_governance_readiness": ("myquant.v17.v3.factor-governance-readiness.v1"),
            "official_delisting_cash": ("myquant.v17.v3.dataset.official-delisting-cash.v1"),
        }
        for binding in sources:
            expected = expected_raw_versions.get(binding["role"])
            if expected is not None and binding["artifact_ref"]["artifact_version"] != expected:
                raise ArtifactContractError(f"raw role {binding['role']} artifact version mismatch")
    else:
        parent = _validate_ref(
            payload["parent_raw_manifest_ref"],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.source-manifest.v1",
            label="parent_raw_manifest_ref",
        )
        if parent["artifact_id"] == payload["manifest_id"]:
            raise ArtifactContractError("derived manifest cannot reference itself as parent")
        try:
            requirement = role_requirement(payload["phase"])
        except PolicyContractError as exc:
            raise ArtifactContractError(str(exc)) from exc
        required_derived = set(requirement.required_roles) & set(derived_roles)
        allowed_derived = required_derived | (set(requirement.optional_roles) & set(derived_roles))
        if not required_derived.issubset(roles):
            raise ArtifactContractError("derived source manifest is missing required derived roles")
        if not set(roles).issubset(allowed_derived):
            raise ArtifactContractError("derived source manifest contains raw or forbidden roles")
        expected_versions = {
            "deep_research_inputs": "myquant.v17.v3.deep-research-inputs.v1",
            "fundamental_branch_output": "myquant.v17.v3.branch-output.v1",
            "fundamental_forward_calibration_inputs": ("myquant.v17.v3.calibration-gate-inputs.v1"),
            "fundamental_forward_calibration": ("myquant.v17.v3.fusion-calibration-receipt.v1"),
            "fusion_calibration": "myquant.v17.v3.fusion-calibration-inputs.v1",
            "fusion_promotion_receipt": ("myquant.v17.v3.fusion-promotion-receipt.v1"),
            "initial_pool_output": "myquant.v17.v3.initial-pool-output.v1",
            "macro_overlay": "myquant.v17.v3.portfolio-overlay.v1",
            "markov_overlay": "myquant.v17.v3.portfolio-overlay.v1",
            "permissions": "myquant.v17.v3.pretrade-permissions.v1",
            "provisional_factor_baseline": ("myquant.v17.v3.provisional-factor-baseline.v1"),
            "quant_branch_output": "myquant.v17.v3.branch-output.v1",
            "quant_preselection_inputs": ("myquant.v17.v3.quant-preselection-inputs.v1"),
            "quant_timing_calibration_inputs": ("myquant.v17.v3.calibration-gate-inputs.v1"),
            "quant_timing_calibration": ("myquant.v17.v3.fusion-calibration-receipt.v1"),
        }
        for binding in sources:
            role = binding["role"]
            if role in derived_roles:
                expected = expected_versions[role]
                if binding["artifact_ref"]["artifact_version"] != expected:
                    raise ArtifactContractError(f"derived role {role} artifact version mismatch")
    return result


def validate_initial_pool_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> InitialPoolOutputArtifact:
    result = _typed(
        payload,
        InitialPoolOutputArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, InitialPoolOutputArtifact)
    _validate_factor_baseline_binding(
        payload["factor_baseline_ref"],
        payload["factor_baseline_mode"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        label="initial-pool output",
    )
    if payload["policy_sha256"] != _packaged_policy_sha256("preselector_policy"):
        raise ArtifactContractError(
            "initial-pool policy_sha256 does not bind the packaged preselector policy"
        )
    ordered_domain = payload["ordered_domain"]
    selected_symbols = payload["selected_symbols"]
    if len(ordered_domain) != len(set(ordered_domain)):
        raise ArtifactContractError("initial pool ordered domain contains duplicates")
    if len(selected_symbols) > 500 or len(selected_symbols) != len(set(selected_symbols)):
        raise ArtifactContractError("initial pool must contain at most 500 unique symbols")
    try:
        for symbol in (*ordered_domain, *selected_symbols):
            require_security_code(symbol)
    except IdentityContractError as exc:
        raise ArtifactContractError(str(exc)) from exc
    _validate_ref(
        payload["source_locator_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        expected_version="myquant.v17.v3.source-locator.v1",
        label="source_locator_ref",
    )
    _validate_ref(
        payload["raw_source_manifest_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        label="raw_source_manifest_ref",
    )
    dispositions = payload["dispositions"]
    disposition_domain = [row["symbol"] for row in dispositions]
    if disposition_domain != ordered_domain:
        raise ArtifactContractError(
            "initial-pool dispositions must contain exactly one ordered-domain record"
        )
    ready_domain: list[str] = []
    selected_from_rows: list[str] = []
    for index, row in enumerate(dispositions):
        status = row["status"]
        reasons = row["reasons"]
        score = row["score"]
        selected = row["selected"]
        if status == "READY":
            if reasons or score is None:
                raise ArtifactContractError(
                    f"initial-pool READY disposition {index} has invalid reason/score"
                )
            _decimal(score, label=f"dispositions[{index}].score")
            ready_domain.append(row["symbol"])
            if selected:
                selected_from_rows.append(row["symbol"])
        else:
            if not reasons or score is not None or selected:
                raise ArtifactContractError(
                    f"initial-pool UNAVAILABLE disposition {index} is not fail closed"
                )
    if payload["ready_domain"] != ready_domain:
        raise ArtifactContractError("initial-pool ready_domain is not the READY subsequence")
    if set(payload["selected_symbols"]) != set(selected_from_rows):
        raise ArtifactContractError("initial-pool selected symbols disagree with dispositions")
    if payload["pool_count"] != len(selected_symbols):
        raise ArtifactContractError("initial-pool pool_count mismatch")
    expected_order_sha = hashlib.sha256(canonical_bytes(selected_symbols)).hexdigest()
    if payload["pool_symbol_order_sha256"] != expected_order_sha:
        raise ArtifactContractError("initial-pool symbol-order SHA mismatch")
    coverage_ids = [row["factor_id"] for row in payload["factor_coverage"]]
    if coverage_ids != sorted(coverage_ids) or len(coverage_ids) != len(set(coverage_ids)):
        raise ArtifactContractError("factor coverage rows must be uniquely sorted")
    for index, row in enumerate(payload["factor_coverage"]):
        coverage = _decimal(row["coverage"], label=f"factor_coverage[{index}].coverage")
        if not Decimal("0") <= coverage <= Decimal("1"):
            raise ArtifactContractError("factor coverage must be within [0,1]")
    if payload["status"] == "READY":
        if payload["blockers"]:
            raise ArtifactContractError("READY initial pool must not carry blockers")
    elif selected_symbols or not payload["blockers"]:
        raise ArtifactContractError(
            "UNAVAILABLE initial pool must have blockers and no selected symbols"
        )
    return result


def validate_branch_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> BranchOutputArtifact:
    result = _typed(payload, BranchOutputArtifact, schema_checked=schema_checked)
    assert isinstance(result, BranchOutputArtifact)
    policy_resource = {
        "quant": "quant_branch_policy",
        "fundamental": "fundamental_branch_policy",
    }[payload["branch"]]
    if payload["policy_sha256"] != _packaged_policy_sha256(policy_resource):
        raise ArtifactContractError(
            f"{payload['branch']} branch policy_sha256 does not bind its packaged policy"
        )
    _validate_ref(
        payload["source_locator_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        expected_version="myquant.v17.v3.source-locator.v1",
        label="source_locator_ref",
    )
    _validate_ref(
        payload["initial_pool_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        expected_version="myquant.v17.v3.initial-pool-output.v1",
        label="initial_pool_ref",
    )
    try:
        require_sha256(payload["policy_sha256"], label="policy_sha256")
        require_sha256(
            payload["initial_pool_symbol_order_sha256"],
            label="initial_pool_symbol_order_sha256",
        )
    except IdentityContractError as exc:
        raise ArtifactContractError(str(exc)) from exc
    ordered_domain = payload["ordered_domain"]
    if len(ordered_domain) > 500:
        raise ArtifactContractError("branch ordered domain exceeds initial-pool Top500")
    if payload["initial_pool_count"] != len(ordered_domain):
        raise ArtifactContractError("branch initial_pool_count does not match ordered domain")
    expected_order_sha = hashlib.sha256(canonical_bytes(ordered_domain)).hexdigest()
    if payload["initial_pool_symbol_order_sha256"] != expected_order_sha:
        raise ArtifactContractError("branch pool symbol-order SHA mismatch")
    record_domain = [row["symbol"] for row in payload["records"]]
    if record_domain != ordered_domain:
        raise ArtifactContractError("branch records must contain exactly one ordered-domain record")
    for index, row in enumerate(payload["records"]):
        if row["status"] == "READY":
            if row["reason"] is not None or row["score"] is None:
                raise ArtifactContractError(f"branch READY record {index} has invalid reason/score")
            _decimal(row["score"], label=f"records[{index}].score")
        elif row["reason"] is None or not row["reason"] or row["score"] is not None:
            raise ArtifactContractError(
                f"branch UNAVAILABLE record {index} must carry reason and no score"
            )
    return result


def validate_branch_same_pool_binding(
    payload: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any] | None = None,
    expected_source_locator_ref: Mapping[str, Any] | None = None,
    expected_initial_pool_ref: Mapping[str, Any] | None = None,
    expected_initial_pool_count: int | None = None,
    expected_initial_pool_symbol_order_sha256: str | None = None,
    expected_policy_sha256: str | None = None,
) -> BranchOutputArtifact:
    """Bind a branch to the exact admitted locator and initial-pool closure."""

    result = validate_branch_output(payload)
    if expected_bindings is None:
        if (
            expected_source_locator_ref is None
            or expected_initial_pool_ref is None
            or expected_initial_pool_count is None
            or expected_initial_pool_symbol_order_sha256 is None
            or expected_policy_sha256 is None
        ):
            raise ArtifactContractError("complete expected branch bindings are required")
        locator = _validate_ref(
            expected_source_locator_ref,
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.source-locator.v1",
            label="expected_source_locator_ref",
        )
        pool = _validate_ref(
            expected_initial_pool_ref,
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.initial-pool-output.v1",
            label="expected_initial_pool_ref",
        )
        expected_bindings = {
            "initial_pool_count": expected_initial_pool_count,
            "initial_pool_ref": dict(pool),
            "initial_pool_symbol_order_sha256": (expected_initial_pool_symbol_order_sha256),
            "policy_sha256": expected_policy_sha256,
            "source_locator_ref": dict(locator),
        }
    expected = dict(expected_bindings)
    observed = {
        "initial_pool_count": payload["initial_pool_count"],
        "initial_pool_ref": payload["initial_pool_ref"],
        "initial_pool_symbol_order_sha256": payload["initial_pool_symbol_order_sha256"],
        "policy_sha256": payload["policy_sha256"],
        "source_locator_ref": payload["source_locator_ref"],
    }
    if observed != expected:
        raise ArtifactContractError("branch same-pool bindings do not exactly match")
    return result


def validate_staged_analysis_lineage(
    *,
    analyze_locator: Mapping[str, Any],
    derived_manifest: Mapping[str, Any],
    initial_pool: Mapping[str, Any],
    quant_branch: Mapping[str, Any],
    fundamental_branch: Mapping[str, Any],
) -> tuple[
    SourceLocatorArtifact,
    SourceManifestArtifact,
    InitialPoolOutputArtifact,
    BranchOutputArtifact,
    BranchOutputArtifact,
]:
    """Validate the cycle-free PRESELECT -> ANALYZE exact-reference lineage."""

    locator = validate_source_locator(analyze_locator)
    manifest = validate_source_manifest(derived_manifest)
    pool = validate_initial_pool_output(initial_pool)
    quant = validate_branch_output(quant_branch)
    fundamental = validate_branch_output(fundamental_branch)
    if derived_manifest["closure_kind"] != "DERIVED_CLOSURE":
        raise ArtifactContractError("analysis lineage requires a derived manifest")
    preselection_ref = analyze_locator["preselection_locator_ref"]
    if preselection_ref is None:
        raise ArtifactContractError("analysis locator must bind a PRESELECT locator")
    if pool.payload["source_locator_ref"] != preselection_ref:
        raise ArtifactContractError("initial pool PRESELECT locator binding drift")
    if pool.payload["raw_source_manifest_ref"] != derived_manifest["parent_raw_manifest_ref"]:
        raise ArtifactContractError("initial pool raw manifest lineage drift")
    expected_pool = {
        "artifact_id": pool.payload["output_id"],
        "artifact_version": pool.version,
        "cutoff": pool.cutoff,
        "semantic_sha256": pool.semantic_sha256,
        "strategy_id": pool.strategy_id,
    }
    for label, branch in (("quant", quant), ("fundamental", fundamental)):
        if branch.payload["branch"] != label:
            raise ArtifactContractError(f"{label} branch role mismatch")
        if branch.payload["source_locator_ref"] != preselection_ref:
            raise ArtifactContractError(f"{label} branch PRESELECT locator drift")
        observed_pool = branch.payload["initial_pool_ref"]
        if any(observed_pool[key] != value for key, value in expected_pool.items()):
            raise ArtifactContractError(f"{label} branch initial-pool binding drift")
        if branch.payload["ordered_domain"] != pool.payload["selected_symbols"]:
            raise ArtifactContractError(f"{label} branch pool domain drift")
    by_role = {row["role"]: row["artifact_ref"] for row in derived_manifest["sources"]}
    expected_documents = {
        "initial_pool_output": pool,
        "quant_branch_output": quant,
        "fundamental_branch_output": fundamental,
    }
    for role, artifact in expected_documents.items():
        ref = by_role.get(role)
        if ref is None:
            raise ArtifactContractError(f"derived manifest is missing {role}")
        identity_field = "output_id" if "output_id" in artifact.payload else "run_id"
        if (
            ref["artifact_id"] != artifact.payload[identity_field]
            or ref["semantic_sha256"] != artifact.semantic_sha256
            or ref["artifact_version"] != artifact.version
        ):
            raise ArtifactContractError(f"derived manifest {role} binding drift")
    manifest_ref = analyze_locator["source_manifest_ref"]
    if (
        manifest_ref["artifact_id"] != manifest.payload["manifest_id"]
        or manifest_ref["semantic_sha256"] != manifest.semantic_sha256
    ):
        raise ArtifactContractError("analysis locator derived manifest binding drift")
    return locator, manifest, pool, quant, fundamental


def validate_calibration_gate_inputs(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> CalibrationGateInputsArtifact:
    result = _typed(
        payload,
        CalibrationGateInputsArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, CalibrationGateInputsArtifact)
    expected_role = {
        "QUANT_TIMING": "quant_timing_calibration_inputs",
        "FUNDAMENTAL_FORWARD": "fundamental_forward_calibration_inputs",
    }[payload["calibration_kind"]]
    if payload["role"] != expected_role:
        raise ArtifactContractError("calibration gate input role/kind binding mismatch")
    if not (payload["observation_start_at"] <= payload["observation_end_at"] <= result.cutoff):
        raise ArtifactContractError("calibration gate observation window exceeds cutoff")
    return result


def validate_fusion_calibration_receipt(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FusionCalibrationReceiptArtifact:
    result = _typed(
        payload,
        FusionCalibrationReceiptArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FusionCalibrationReceiptArtifact)
    if payload["observation_end_at"] > result.cutoff:
        raise ArtifactContractError("calibration observation_end_at is after the exact cutoff")
    _sorted_unique_refs(
        payload["evidence_refs"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        label="evidence_refs",
    )
    return result


def validate_fusion_promotion_receipt(
    payload: Mapping[str, Any],
    *,
    resolved_artifacts: ResolvedArtifactMap | None = None,
    schema_checked: bool = False,
) -> FusionPromotionReceiptArtifact:
    status = payload.get("status")
    if status not in {"PROMOTED", "PROMOTION_REJECTED"}:
        raise ArtifactContractError("fusion promotion status is invalid")
    version, strategy_id, cutoff, digest, sealed = _common(
        payload,
        formal_authority=False,
        schema_checked=schema_checked,
    )
    if payload["observation_end_at"] > cutoff:
        raise ArtifactContractError("promotion observation_end_at is after cutoff")
    refs = tuple(
        _validate_ref(
            ref,
            strategy_id=strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v3.fusion-calibration-receipt.v1",
            label=f"calibration_receipt_refs[{index}]",
        )
        for index, ref in enumerate(payload["calibration_receipt_refs"])
    )
    ref_keys = tuple((ref["relative_path"], ref["byte_sha256"]) for ref in refs)
    if len(refs) != 3 or len(ref_keys) != len(set(ref_keys)):
        raise ArtifactContractError("fusion promotion requires three unique calibration receipts")
    evidence_refs = _sorted_unique_refs(
        payload["evidence_refs"],
        strategy_id=strategy_id,
        cutoff=cutoff,
        label="evidence_refs",
    )
    evidence_versions = tuple(ref["artifact_version"] for ref in evidence_refs)
    expected_evidence_counts = {
        "myquant.v17.v3.branch-output.v1": 2,
        "myquant.v17.v3.fusion-calibration-inputs.v1": 1,
        "myquant.v17.v3.initial-pool-output.v1": 1,
    }
    if len(evidence_refs) != 4 or any(
        evidence_versions.count(version) != count
        for version, count in expected_evidence_counts.items()
    ):
        raise ArtifactContractError(
            "fusion promotion evidence must bind one initial pool, both branches, "
            "and one calibration-input artifact"
        )
    expected_hashes = {
        "contract_package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "preselector_policy_sha256": load_packaged_json("resources/preselector_policy.v1.json")[
            "semantic_sha256"
        ],
        "quant_branch_policy_sha256": load_packaged_json("resources/quant_branch_policy.v1.json")[
            "semantic_sha256"
        ],
        "fundamental_branch_policy_sha256": load_packaged_json(
            "resources/fundamental_branch_policy.v1.json"
        )["semantic_sha256"],
        "fusion_policy_sha256": load_packaged_json("resources/fusion_policy.v1.json")[
            "semantic_sha256"
        ],
    }
    for field, expected_sha256 in expected_hashes.items():
        if payload[field] != expected_sha256:
            raise ArtifactContractError(f"fusion promotion {field} binding mismatch")
    active = payload["active_refit_origins"]
    outer = payload["outer_oos_origins"]
    folds = payload["fold_inventory"]
    if active != sorted(active) or outer != sorted(outer):
        raise ArtifactContractError("promotion origin windows must be date ordered")
    if [row["fold_index"] for row in folds] != [1, 2, 3, 4, 5]:
        raise ArtifactContractError("promotion fold inventory must be ordered 1..5")
    stitched: list[str] = []
    for index, row in enumerate(folds):
        if row["training_origins"] != sorted(row["training_origins"]):
            raise ArtifactContractError(
                f"promotion fold {index + 1} training origins are not ordered"
            )
        if row["oos_origins"] != sorted(row["oos_origins"]):
            raise ArtifactContractError(f"promotion fold {index + 1} OOS origins are not ordered")
        stitched.extend(row["oos_origins"])
    if stitched != outer:
        raise ArtifactContractError("promotion outer OOS origins do not match stitched folds")
    for field in (
        "oos_mean_hit60",
        "oos_mean_q25_252",
        "oos_p5_hit60",
        "oos_p5_q25_252",
    ):
        _decimal(payload[field], label=field)
    passes = _decimal(payload["oos_p5_hit60"], label="oos_p5_hit60") > Decimal("0.50") and _decimal(
        payload["oos_p5_q25_252"], label="oos_p5_q25_252"
    ) > Decimal("0")
    if status == "PROMOTED":
        if not passes:
            raise ArtifactContractError(
                "PROMOTED receipt does not satisfy deterministic lower bounds"
            )
        _decimal(
            payload["active_formal_research_weight"],
            label="active_formal_research_weight",
        )
    else:
        _decimal(
            payload["evaluated_quant_weight"],
            label="evaluated_quant_weight",
        )
        reasons = payload["rejection_reasons"]
        if reasons != sorted(reasons) or len(reasons) != len(set(reasons)):
            raise ArtifactContractError("promotion rejection reasons must be unique and sorted")
    result = FusionPromotionReceiptArtifact(
        version,
        strategy_id,
        cutoff,
        digest,
        sealed,
        status,
    )
    if resolved_artifacts is not None:
        _validate_fusion_promotion_evidence(result, resolved_artifacts)
    return result


def _validate_fusion_promotion_evidence(
    promotion: FusionPromotionReceiptArtifact,
    resolved_artifacts: ResolvedArtifactMap,
) -> None:
    resolved: list[tuple[dict[str, Any], ValidatedArtifact]] = []
    for index, ref in enumerate(promotion.payload["evidence_refs"]):
        resolved.append(
            (
                dict(ref),
                _resolve_exact_reference(
                    ref,
                    resolved_artifacts,
                    label=f"promotion evidence_refs[{index}]",
                ),
            )
        )
    pools = [
        (ref, artifact)
        for ref, artifact in resolved
        if isinstance(artifact, InitialPoolOutputArtifact)
    ]
    branches = [
        (ref, artifact) for ref, artifact in resolved if isinstance(artifact, BranchOutputArtifact)
    ]
    calibration_inputs = [
        artifact
        for _, artifact in resolved
        if isinstance(artifact, FusionCalibrationInputsArtifact)
    ]
    if len(pools) != 1 or len(branches) != 2 or len(calibration_inputs) != 1:
        raise ArtifactContractError("resolved promotion evidence type inventory mismatch")
    branch_by_kind = {artifact.payload["branch"]: (ref, artifact) for ref, artifact in branches}
    if set(branch_by_kind) != {"quant", "fundamental"}:
        raise ArtifactContractError("resolved promotion evidence must contain both branch kinds")
    pool_ref, pool = pools[0]
    for branch_name, (_, branch) in branch_by_kind.items():
        if branch.payload["initial_pool_ref"] != pool_ref:
            raise ArtifactContractError(
                f"promotion {branch_name} branch initial-pool evidence drift"
            )
        if branch.payload["ordered_domain"] != pool.payload["selected_symbols"]:
            raise ArtifactContractError(f"promotion {branch_name} branch domain evidence drift")


def validate_factor_governance_readiness(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FactorGovernanceReadinessArtifact:
    result = _typed(
        payload,
        FactorGovernanceReadinessArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FactorGovernanceReadinessArtifact)
    if payload["available_at"] > result.cutoff:
        raise ArtifactContractError("factor governance readiness available_at is after cutoff")
    try:
        source_as_of = date.fromisoformat(payload["source_as_of"])
        cutoff_date = date.fromisoformat(result.cutoff[:10])
    except ValueError as exc:
        raise ArtifactContractError("factor governance readiness source_as_of is invalid") from exc
    age_days = (cutoff_date - source_as_of).days
    if age_days < 0 or age_days > 8:
        raise ArtifactContractError("factor governance readiness freshness exceeds 8 calendar days")
    blockers = payload["blockers"]
    if blockers != sorted(blockers) or len(blockers) != len(set(blockers)):
        raise ArtifactContractError("factor governance readiness blockers must be uniquely sorted")
    healthy = payload["healthy_factor_count"]
    production = payload["production_factor_count"]
    families = payload["production_family_count"]
    if healthy > production:
        raise ArtifactContractError("healthy factor count cannot exceed production factor count")
    ready_conditions = (
        healthy >= 5 and production >= 5 and families >= 3 and payload["activation_receipt_valid"]
    )
    if payload["readiness_status"] == "FACTOR_V4_READY":
        if not payload["factor_governance_ready"] or not ready_conditions or blockers:
            raise ArtifactContractError(
                "FACTOR_V4_READY status/count/activation consistency failed"
            )
    elif payload["factor_governance_ready"] or ready_conditions or not blockers:
        raise ArtifactContractError("FACTOR_V4_NOT_READY status/count/blocker consistency failed")
    return result


def validate_provisional_factor_baseline(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ProvisionalFactorBaselineArtifact:
    result = _typed(
        payload,
        ProvisionalFactorBaselineArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, ProvisionalFactorBaselineArtifact)
    _validate_ref(
        payload["factor_governance_readiness_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        expected_version="myquant.v17.v3.factor-governance-readiness.v1",
        label="factor_governance_readiness_ref",
    )
    policy = load_packaged_json("resources/provisional_factor_baseline_policy.v1.json")
    if payload["policy_sha256"] != policy["semantic_sha256"]:
        raise ArtifactContractError("provisional factor baseline policy_sha256 binding mismatch")
    if payload["preselector_factors"] != policy["preselector_factors"]:
        raise ArtifactContractError("provisional preselector factor inventory equality failed")
    if payload["quant_factors"] != policy["quant_factors"]:
        raise ArtifactContractError("provisional Quant factor inventory equality failed")
    for rows, label in (
        (payload["preselector_factors"], "preselector"),
        (payload["quant_factors"], "Quant"),
    ):
        factor_ids = [row["factor_id"] for row in rows]
        if factor_ids != sorted(factor_ids) or len(factor_ids) != len(set(factor_ids)):
            raise ArtifactContractError(
                f"provisional {label} factor inventory is not uniquely sorted"
            )
    for axis in (
        "definition_sha256",
        "factor_id",
        "family_id",
        "lineage_id",
    ):
        left = {row[axis] for row in payload["preselector_factors"]}
        right = {row[axis] for row in payload["quant_factors"]}
        if left & right:
            raise ArtifactContractError(f"provisional factor inventories overlap on {axis}")
    total_weight = sum(
        (
            _decimal(row["weight"], label="provisional preselector weight")
            for row in payload["preselector_factors"]
        ),
        Decimal("0"),
    )
    if total_weight != Decimal("1"):
        raise ArtifactContractError(
            "provisional preselector factor weights must sum exactly to one"
        )
    return result


def validate_quant_preselection_inputs(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> QuantPreselectionInputsArtifact:
    result = _typed(
        payload,
        QuantPreselectionInputsArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, QuantPreselectionInputsArtifact)
    _validate_factor_baseline_binding(
        payload["factor_baseline_ref"],
        payload["factor_baseline_mode"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        label="quant-preselection inputs",
    )
    body = payload["payload"]
    if body["policy_sha256"] != _packaged_policy_sha256("preselector_policy"):
        raise ArtifactContractError(
            "quant-preselection policy_sha256 does not bind the packaged policy"
        )
    factor_names = [row["name"] for row in body["factor_contract"]]
    branch_names = [row["name"] for row in body["quant_branch_inventory"]]
    symbols = [row["symbol"] for row in body["observations"]]
    if factor_names != sorted(factor_names) or len(factor_names) != len(set(factor_names)):
        raise ArtifactContractError("factor contract must be uniquely sorted by name")
    if branch_names != sorted(branch_names) or len(branch_names) != len(set(branch_names)):
        raise ArtifactContractError("Quant branch inventory must be uniquely sorted by name")
    expected_preselector = _packaged_factor_inventory("preselector_policy")
    expected_quant = _packaged_factor_inventory("quant_branch_policy")
    if len(body["factor_contract"]) != len(expected_preselector):
        raise ArtifactContractError(
            "preselection factor contract does not equal packaged inventory"
        )
    by_preselector_name = {row["name"]: row for row in body["factor_contract"]}
    for expected in expected_preselector:
        observed = by_preselector_name.get(expected["factor_id"])
        if observed is None or any(
            observed[field] != value
            for field, value in {
                "definition_hash": expected["definition_sha256"],
                "family": expected["family_id"],
                "lineage": expected["lineage_id"],
                "lookback": expected["lookback_open_days"],
                "weight": expected["weight"],
            }.items()
        ):
            raise ArtifactContractError(
                "preselection factor contract does not equal packaged inventory"
            )
        if observed["warmup"] < observed["lookback"]:
            raise ArtifactContractError(
                "preselection factor warmup cannot be shorter than lookback"
            )
        coverage = _decimal(
            observed["minimum_coverage"],
            label=f"{observed['name']} minimum_coverage",
        )
        if not Decimal("0") <= coverage <= Decimal("1"):
            raise ArtifactContractError("preselection minimum_coverage must be within [0,1]")
    normalized_quant = [
        {
            "definition_hash": row["definition_sha256"],
            "family": row["family_id"],
            "lineage": row["lineage_id"],
            "name": row["factor_id"],
        }
        for row in expected_quant
    ]
    if body["quant_branch_inventory"] != normalized_quant:
        raise ArtifactContractError("Quant branch inventory does not equal packaged policy")
    if len(symbols) != len(set(symbols)):
        raise ArtifactContractError("preselection observations contain duplicate symbols")
    for index, row in enumerate(body["observations"]):
        value_ids = [value["factor_id"] for value in row["factor_values"]]
        if value_ids != sorted(value_ids) or len(value_ids) != len(set(value_ids)):
            raise ArtifactContractError(
                f"observation {index} factor values must be uniquely sorted"
            )
        expected_factor_ids = set(factor_names)
        if not set(value_ids).issubset(expected_factor_ids) or (
            row["data_ready"] and set(value_ids) != expected_factor_ids
        ):
            raise ArtifactContractError(
                f"observation {index} factor values disagree with packaged inventory"
            )
        for value in row["factor_values"]:
            _decimal(value["value"], label=f"observations[{index}].factor value")
    for axis in ("definition_hash", "family", "lineage"):
        left = {row[axis] for row in body["factor_contract"]}
        right = {row[axis] for row in body["quant_branch_inventory"]}
        if left & right:
            raise ArtifactContractError(
                f"preselection and Quant branch inventories overlap on {axis}"
            )
    return result


def validate_deep_research_inputs(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DeepResearchInputsArtifact:
    result = _typed(payload, DeepResearchInputsArtifact, schema_checked=schema_checked)
    assert isinstance(result, DeepResearchInputsArtifact)
    symbols = [row["symbol"] for row in payload["payload"]]
    if len(symbols) != len(set(symbols)):
        raise ArtifactContractError("deep research inputs contain duplicate symbols")
    for index, row in enumerate(payload["payload"]):
        if row["lane"] == "REVIEW_ONLY_HOLDING" and not row["held"]:
            raise ArtifactContractError("review-only deep input must be a held symbol")
        if not row["available"] and (
            row["signal"] is not None or row["evidence_refs"] or not row["veto_buy"]
        ):
            raise ArtifactContractError(
                "unavailable deep input must have null signal, empty evidence, and veto_buy"
            )
        for field in ("base_target", "current_target"):
            if _decimal(row[field], label=f"deep inputs[{index}].{field}") < 0:
                raise ArtifactContractError("deep target must be nonnegative")
        if row["signal"] is not None:
            _decimal(row["signal"], label=f"deep inputs[{index}].signal")
        _sorted_unique_refs(
            row["evidence_refs"],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            label=f"deep inputs[{index}].evidence_refs",
        )
    return result


def validate_pretrade_permissions(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PretradePermissionsArtifact:
    result = _typed(payload, PretradePermissionsArtifact, schema_checked=schema_checked)
    assert isinstance(result, PretradePermissionsArtifact)
    symbols = [row["symbol"] for row in payload["payload"]]
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("pretrade permissions must be uniquely sorted")
    _validate_ref(
        payload["canonical_calendar_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        label="canonical_calendar_ref",
    )
    holdings_ref = payload["holdings_snapshot_ref"]
    as_of = payload["holdings_snapshot_as_of_session"]
    age = payload["holdings_snapshot_age_sessions"]
    basis = payload["portfolio_basis"]
    if basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS":
        if holdings_ref is not None or as_of is not None or age is not None:
            raise ArtifactContractError(
                "MODEL_ONLY_NO_PRIVATE_HOLDINGS permissions must not bind private holdings"
            )
    else:
        if holdings_ref is None or as_of is None or age is None:
            raise ArtifactContractError(
                "HOLDINGS_AWARE permissions require complete holdings freshness"
            )
        _validate_ref(
            holdings_ref,
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            label="holdings_snapshot_ref",
        )
        decision = payload["decision_session"]
        if as_of > decision or (as_of == decision) != (age == 0):
            raise ArtifactContractError("holdings snapshot session freshness is inconsistent")
    for index, row in enumerate(payload["payload"]):
        current = _decimal(
            row["current_target"],
            label=f"permissions[{index}].current_target",
        )
        if current < 0:
            raise ArtifactContractError("permission current target must be nonnegative")
        if row["lane"] == "REVIEW_ONLY_HOLDING" and (not row["held"] or row["can_buy"]):
            raise ArtifactContractError(
                "review-only holding must be held and cannot receive buy permission"
            )
        if basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS" and (
            row["lane"] != "SELECTION_POOL" or row["held"] or current != Decimal("0")
        ):
            raise ArtifactContractError(
                "MODEL_ONLY_NO_PRIVATE_HOLDINGS permissions cannot expose holding state"
            )
    return result


def validate_portfolio_overlay(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PortfolioOverlayArtifact:
    result = _typed(payload, PortfolioOverlayArtifact, schema_checked=schema_checked)
    assert isinstance(result, PortfolioOverlayArtifact)
    targets = payload["payload"]["target_weights"]
    symbols = [row["symbol"] for row in targets]
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("overlay targets must be uniquely sorted")
    for index, row in enumerate(targets):
        if _decimal(row["target"], label=f"target_weights[{index}].target") < 0:
            raise ArtifactContractError("overlay target must be nonnegative")
    return result


def _validate_overlay_stages(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy_id: str,
    cutoff: str,
) -> None:
    if [row["stage"] for row in rows] != ["MACRO", "MARKOV"]:
        raise ArtifactContractError("overlay stages must be fixed ordered MACRO then MARKOV")
    for index, row in enumerate(rows):
        if row["status"] == "APPLIED":
            if row["overlay_ref"] is None:
                raise ArtifactContractError(
                    f"overlay stage {index} APPLIED status requires an overlay_ref"
                )
            _validate_ref(
                row["overlay_ref"],
                strategy_id=strategy_id,
                cutoff=cutoff,
                expected_version="myquant.v17.v3.portfolio-overlay.v1",
                label=f"overlay_stages[{index}].overlay_ref",
            )
        elif row["overlay_ref"] is not None:
            raise ArtifactContractError(
                f"overlay stage {index} UNAVAILABLE_NO_OP must have null overlay_ref"
            )


def validate_portfolio_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> PortfolioOutputArtifact:
    result = _typed(payload, PortfolioOutputArtifact, schema_checked=schema_checked)
    assert isinstance(result, PortfolioOutputArtifact)
    _validate_factor_baseline_binding(
        payload["factor_baseline_ref"],
        payload["factor_baseline_mode"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        label="portfolio output",
    )
    if payload["allocation_policy_sha256"] != _packaged_policy_sha256(
        "portfolio_allocation_policy"
    ):
        raise ArtifactContractError("portfolio allocation_policy_sha256 binding mismatch")
    _validate_overlay_stages(
        payload["overlay_stages"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
    )
    for field, version in (
        ("fusion_output_ref", "myquant.v17.v3.fusion-output.v1"),
        ("deep_output_ref", "myquant.v17.v3.deep-output.v1"),
    ):
        _validate_ref(
            payload[field],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version=version,
            label=field,
        )
    holdings_ref = payload["holdings_snapshot_ref"]
    basis = payload["portfolio_basis"]
    if basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS":
        if holdings_ref is not None:
            raise ArtifactContractError(
                "MODEL_ONLY_NO_PRIVATE_HOLDINGS portfolio must not bind private holdings"
            )
    else:
        if holdings_ref is None:
            raise ArtifactContractError("HOLDINGS_AWARE portfolio requires holdings_snapshot_ref")
        _validate_ref(
            holdings_ref,
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            label="holdings_snapshot_ref",
        )
    _validate_ref(
        payload["permissions_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        expected_version="myquant.v17.v3.pretrade-permissions.v1",
        label="permissions_ref",
    )
    selection = payload["selection_pool_symbols"]
    review = payload["review_only_holdings"]
    if len(selection) != len(set(selection)):
        raise ArtifactContractError("selection pool contains duplicates")
    if review != sorted(review) or len(review) != len(set(review)):
        raise ArtifactContractError("review-only holdings must be uniquely sorted")
    if set(selection) & set(review):
        raise ArtifactContractError("review-only holdings cannot enter the organic selection pool")
    if basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS" and review:
        raise ArtifactContractError(
            "MODEL_ONLY_NO_PRIVATE_HOLDINGS portfolio cannot contain review-only holdings"
        )
    targets = payload["targets"]
    symbols = [row["symbol"] for row in targets]
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("portfolio targets must be uniquely sorted")
    if set(symbols) != set(selection) | set(review):
        raise ArtifactContractError("portfolio target domain does not match both lanes")
    total = Decimal("0")
    for index, row in enumerate(targets):
        current = _decimal(
            row["current_target"],
            label=f"targets[{index}].current_target",
        )
        final = _decimal(
            row["final_target"],
            label=f"targets[{index}].final_target",
        )
        if current < 0 or final < 0:
            raise ArtifactContractError("portfolio targets must be nonnegative")
        expected_lane = "REVIEW_ONLY_HOLDING" if row["symbol"] in set(review) else "SELECTION_POOL"
        if row["lane"] != expected_lane:
            raise ArtifactContractError("portfolio target lane mismatch")
        if expected_lane == "REVIEW_ONLY_HOLDING" and final > current:
            raise ArtifactContractError(
                "review-only holding cannot receive a positive target delta"
            )
        if basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS" and (
            row["lane"] != "SELECTION_POOL" or current != Decimal("0")
        ):
            raise ArtifactContractError(
                "MODEL_ONLY_NO_PRIVATE_HOLDINGS portfolio cannot expose holding state"
            )
        total += final
    gross = _decimal(payload["gross_weight"], label="gross_weight")
    cash = _decimal(payload["cash_weight"], label="cash_weight")
    if gross != total or gross + cash != Decimal("1"):
        raise ArtifactContractError("portfolio weights must reconcile exactly to one")
    if payload["status"] == "COMPLETE":
        if payload["blockers"] or not targets:
            raise ArtifactContractError("complete portfolio must have targets and no blockers")
        if basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS":
            if (
                len(targets) != 24
                or any(
                    _decimal(row["final_target"], label="model-only final_target")
                    not in {Decimal("0"), Decimal("0.03")}
                    for row in targets
                )
                or gross > Decimal("0.72")
            ):
                raise ArtifactContractError(
                    "model-only complete portfolio must preserve the exact Top24 "
                    "domain and allocate only eligible names at 0.03"
                )
    elif not payload["blockers"] or targets or gross != 0 or cash != 1:
        raise ArtifactContractError("incomplete portfolio must fail closed to cash with blockers")
    return result


def validate_fusion_calibration_inputs(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FusionCalibrationInputsArtifact:
    result = _typed(
        payload,
        FusionCalibrationInputsArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, FusionCalibrationInputsArtifact)
    body = payload["payload"]
    sessions = body["canonical_sessions"]
    origins = body["scheduled_origins"]
    months = body["months"]
    if sessions != sorted(sessions) or origins != sorted(origins):
        raise ArtifactContractError("calibration sessions and origins must be ordered")
    month_origins = [row["origin"] for row in months]
    if month_origins != origins:
        raise ArtifactContractError("calibration month origins do not match schedule")
    if body["active_cutoff"] not in sessions:
        raise ArtifactContractError("active cutoff is not a canonical session")
    for index, row in enumerate(months):
        origin = row["origin"]
        for field in ("quant_branch_ref", "fundamental_branch_ref"):
            ref = _validate_ref(
                row[field],
                strategy_id=result.strategy_id,
                cutoff=None,
                expected_version="myquant.v17.v3.branch-output.v1",
                label=f"months[{index}].{field}",
            )
            if ref["cutoff"][:10] != origin:
                raise ArtifactContractError(f"months[{index}].{field} cutoff does not match origin")
        if row["quant_branch_ref"] == row["fundamental_branch_ref"]:
            raise ArtifactContractError("calibration branch references must be distinct")
        if len(row["ordered_pool"]) != len(set(row["ordered_pool"])):
            raise ArtifactContractError("calibration ordered pool contains duplicates")
        for field in ("forward_return_60", "forward_return_252"):
            rows = row[field]
            symbols = [item["symbol"] for item in rows]
            if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
                raise ArtifactContractError(f"months[{index}].{field} must be uniquely sorted")
            for item in rows:
                _decimal(item["value"], label=f"months[{index}].{field}")
    return result


def validate_fusion_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FusionOutputArtifact:
    result = _typed(payload, FusionOutputArtifact, schema_checked=schema_checked)
    assert isinstance(result, FusionOutputArtifact)
    for field in ("quant_branch_ref", "fundamental_branch_ref"):
        _validate_ref(
            payload[field],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.branch-output.v1",
            label=field,
        )
    refs = payload["calibration_receipt_refs"]
    for index, ref in enumerate(refs):
        _validate_ref(
            ref,
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.fusion-calibration-receipt.v1",
            label=f"calibration_receipt_refs[{index}]",
        )
    ref_ids = [ref["artifact_id"] for ref in refs]
    if len(ref_ids) != len(set(ref_ids)):
        raise ArtifactContractError("fusion calibration receipt references must be unique")
    quant_weight = _decimal(payload["quant_weight"], label="quant_weight")
    fundamental_weight = _decimal(
        payload["fundamental_weight"],
        label="fundamental_weight",
    )
    if quant_weight + fundamental_weight != Decimal("1"):
        raise ArtifactContractError("fusion weights must sum exactly to one")
    promotion_ref = payload["promotion_receipt_ref"]
    if payload["calibration_label"] == "UNCALIBRATED_50_50":
        if refs or promotion_ref is not None:
            raise ArtifactContractError(
                "uncalibrated fusion cannot bind calibration or promotion receipts"
            )
        if quant_weight != Decimal("0.50") or fundamental_weight != Decimal("0.50"):
            raise ArtifactContractError("uncalibrated fusion must use exact 0.50/0.50")
    else:
        if len(refs) != 3 or promotion_ref is None:
            raise ArtifactContractError(
                "calibrated fusion requires three receipts and promotion receipt"
            )
        _validate_ref(
            promotion_ref,
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.fusion-promotion-receipt.v1",
            label="promotion_receipt_ref",
        )
    ordered_domain = payload["ordered_domain"]
    dispositions = payload["dispositions"]
    if [row["symbol"] for row in dispositions] != ordered_domain:
        raise ArtifactContractError(
            "fusion dispositions must contain exactly one ordered-domain record"
        )
    ready_domain: list[str] = []
    selected_from_rows: list[str] = []
    for index, row in enumerate(dispositions):
        if row["status"] == "READY":
            if (
                row["reason"] is not None
                or row["quant_percentile"] is None
                or row["fundamental_percentile"] is None
                or row["fusion_score"] is None
            ):
                raise ArtifactContractError(
                    f"fusion READY disposition {index} has incomplete scores"
                )
            for field in (
                "quant_percentile",
                "fundamental_percentile",
                "fusion_score",
            ):
                value = _decimal(row[field], label=f"dispositions[{index}].{field}")
                if not Decimal("0") <= value <= Decimal("1"):
                    raise ArtifactContractError("fusion score must be within [0,1]")
            ready_domain.append(row["symbol"])
            if row["selected"]:
                selected_from_rows.append(row["symbol"])
        elif (
            not row["reason"]
            or row["quant_percentile"] is not None
            or row["fundamental_percentile"] is not None
            or row["fusion_score"] is not None
            or row["selected"]
        ):
            raise ArtifactContractError(
                f"fusion UNAVAILABLE disposition {index} is not fail closed"
            )
    if payload["common_ready_domain"] != ready_domain:
        raise ArtifactContractError("fusion common_ready_domain is not READY subsequence")
    if set(payload["selected_symbols"]) != set(selected_from_rows):
        raise ArtifactContractError("fusion selected symbols disagree with dispositions")
    if len(payload["selected_symbols"]) > 24:
        raise ArtifactContractError("fusion output exceeds Top24")
    if payload["status"] == "READY":
        if payload["blockers"] or len(payload["selected_symbols"]) != 24:
            raise ArtifactContractError("READY fusion has blockers or incomplete Top24")
    elif not payload["blockers"] or len(payload["selected_symbols"]) >= 24:
        raise ArtifactContractError(
            "UNAVAILABLE fusion must carry blockers and cannot claim Top24 completion"
        )
    return result


def validate_deep_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> DeepOutputArtifact:
    result = _typed(payload, DeepOutputArtifact, schema_checked=schema_checked)
    assert isinstance(result, DeepOutputArtifact)
    _validate_ref(
        payload["fusion_output_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        expected_version="myquant.v17.v3.fusion-output.v1",
        label="fusion_output_ref",
    )
    symbols: list[str] = []
    for index, row in enumerate(payload["results"]):
        try:
            symbol = require_security_code(row["symbol"])
        except IdentityContractError as exc:
            raise ArtifactContractError(str(exc)) from exc
        symbols.append(symbol)
        decimals: dict[str, Decimal] = {}
        for field in (
            "base_target",
            "current_target",
            "penalty",
            "raw_adjusted_target",
            "target",
        ):
            decimals[field] = _decimal(
                row[field],
                label=f"results[{index}].{field}",
            )
        if row["signal"] is not None:
            _decimal(row["signal"], label=f"results[{index}].signal")
        if any(decimals[field] < 0 for field in decimals):
            raise ArtifactContractError("deep targets and penalty must be nonnegative")
        if row["lane"] == "REVIEW_ONLY_HOLDING" and decimals["target"] > decimals["current_target"]:
            raise ArtifactContractError(
                "review-only holding cannot receive a positive deep target delta"
            )
        _sorted_unique_refs(
            row["evidence_refs"],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            label=f"results[{index}].evidence_refs",
        )
    if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
        raise ArtifactContractError("deep results must be uniquely sorted by symbol")
    return result


def _validate_terminal_output(
    payload: Mapping[str, Any],
    artifact_class: type[ValidatedArtifact],
    *,
    expected_class: str,
    schema_checked: bool,
) -> ValidatedArtifact:
    result = _typed(payload, artifact_class, schema_checked=schema_checked)
    state = payload["terminal_state"]
    observed_class = terminal_class(state)
    if observed_class not in {expected_class, "HARD_STOP"}:
        raise ArtifactContractError(f"{state} is not a {expected_class} or hard-stop terminal")
    refs = _sorted_unique_refs(
        payload["artifact_refs"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        label="artifact_refs",
    )
    factor_baseline_ref = _validate_factor_baseline_binding(
        payload["factor_baseline_ref"],
        payload["factor_baseline_mode"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        label=f"{expected_class.lower()} terminal",
    )
    if factor_baseline_ref not in refs:
        raise ArtifactContractError("factor baseline reference must be present in artifact_refs")
    expected_portfolio = {
        "FORMAL_PORTFOLIO_INFEASIBLE": "INFEASIBLE",
        "FORMAL_RANK_COMPLETE_NO_PORTFOLIO": "NOT_REQUESTED",
        "FORMAL_RESEARCH_COMPLETE": "COMPLETE",
        "SHADOW_COMPLETE": "COMPLETE",
        "SHADOW_PORTFOLIO_INFEASIBLE": "INFEASIBLE",
        "SHADOW_RANK_COMPLETE_NO_PORTFOLIO": "NOT_REQUESTED",
        "HARD_STOP_CONTRACT_VIOLATION": "NOT_REQUESTED",
        "HARD_STOP_INVALID_EVIDENCE": "NOT_REQUESTED",
        "HARD_STOP_INVALID_SOURCE": "NOT_REQUESTED",
        "HARD_STOP_SNAPSHOT_DRIFT": "NOT_REQUESTED",
    }
    if payload["portfolio_status"] != expected_portfolio[state]:
        raise ArtifactContractError("terminal state and portfolio status disagree")
    if payload["portfolio_status"] == "NOT_REQUESTED":
        if payload["portfolio_basis"] is not None:
            raise ArtifactContractError(
                "rank-only or hard-stop terminal must have null portfolio_basis"
            )
    elif payload["portfolio_basis"] is None:
        raise ArtifactContractError("portfolio terminal must carry portfolio_basis")
    analyze_ref = _validate_ref(
        payload["analyze_locator_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        expected_version="myquant.v17.v3.source-locator.v1",
        label="analyze_locator_ref",
    )
    portfolio_ref = payload["portfolio_output_ref"]
    if payload["portfolio_status"] in {"COMPLETE", "INFEASIBLE"}:
        if portfolio_ref is None:
            raise ArtifactContractError("portfolio terminal must bind a typed portfolio output")
        normalized_portfolio_ref = _validate_ref(
            portfolio_ref,
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.portfolio-output.v1",
            label="portfolio_output_ref",
        )
        if normalized_portfolio_ref not in refs:
            raise ArtifactContractError(
                "portfolio output reference must be present in artifact_refs"
            )
    elif portfolio_ref is not None:
        raise ArtifactContractError(
            "rank-only or hard-stop terminal cannot bind a portfolio output"
        )
    if analyze_ref not in refs:
        raise ArtifactContractError("analyze locator reference must be present in artifact_refs")
    return result


def validate_formal_research_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FormalResearchOutputArtifact:
    result = _validate_terminal_output(
        payload,
        FormalResearchOutputArtifact,
        expected_class="FORMAL",
        schema_checked=schema_checked,
    )
    assert isinstance(result, FormalResearchOutputArtifact)
    if payload["factor_baseline_mode"] != "FACTOR_V4_PRODUCTION":
        raise ArtifactContractError("formal output requires FACTOR_V4_PRODUCTION baseline")
    return result


def validate_shadow_output(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ShadowOutputArtifact:
    result = _validate_terminal_output(
        payload,
        ShadowOutputArtifact,
        expected_class="SHADOW",
        schema_checked=schema_checked,
    )
    assert isinstance(result, ShadowOutputArtifact)
    if payload["allocation_policy_sha256"] != _packaged_policy_sha256(
        "portfolio_allocation_policy"
    ):
        raise ArtifactContractError("shadow allocation_policy_sha256 binding mismatch")
    _validate_overlay_stages(
        payload["overlay_stages"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
    )
    return result


def validate_activation_receipt(
    payload: Mapping[str, Any],
    *,
    formal_output: Mapping[str, Any] | FormalResearchOutputArtifact | None = None,
    promotion_receipt: Mapping[str, Any] | FusionPromotionReceiptArtifact | None = None,
    resolved_artifacts: ResolvedArtifactMap | None = None,
    schema_checked: bool = False,
) -> ActivationReceiptArtifact:
    status = payload.get("status")
    if status not in ACTIVATION_STATUSES:
        raise ArtifactContractError("activation receipt status is invalid")
    version, strategy_id, cutoff, digest, sealed = _common(
        payload,
        formal_authority=status == "ACTIVE",
        schema_checked=schema_checked,
    )
    if status == "ACTIVE":
        formal_ref = _validate_ref(
            payload["formal_output_ref"],
            strategy_id=strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v3.formal-research-output.v1",
            label="formal_output_ref",
        )
        promotion_ref = _validate_ref(
            payload["promotion_receipt_ref"],
            strategy_id=strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v3.fusion-promotion-receipt.v1",
            label="promotion_receipt_ref",
        )
        if formal_output is not None:
            formal = (
                formal_output
                if isinstance(formal_output, FormalResearchOutputArtifact)
                else validate_formal_research_output(formal_output)
            )
            if terminal_class(formal.payload["terminal_state"]) != "FORMAL":
                raise ArtifactContractError("ACTIVE receipt cannot bind a hard-stop output")
            if (
                formal_ref["artifact_id"] != formal.payload["output_id"]
                or formal_ref["semantic_sha256"] != formal.semantic_sha256
            ):
                raise ArtifactContractError("ACTIVE formal output binding mismatch")
        if promotion_receipt is not None:
            promotion = (
                promotion_receipt
                if isinstance(promotion_receipt, FusionPromotionReceiptArtifact)
                else validate_fusion_promotion_receipt(promotion_receipt)
            )
            if promotion.status != "PROMOTED":
                raise ArtifactContractError("ACTIVE requires an accepted promotion receipt")
            if (
                promotion_ref["artifact_id"] != promotion.payload["promotion_id"]
                or promotion_ref["semantic_sha256"] != promotion.semantic_sha256
            ):
                raise ArtifactContractError("ACTIVE promotion receipt binding mismatch")
        if resolved_artifacts is not None:
            resolved_formal = _resolve_exact_reference(
                formal_ref,
                resolved_artifacts,
                label="formal_output_ref",
            )
            resolved_promotion = _resolve_exact_reference(
                promotion_ref,
                resolved_artifacts,
                label="promotion_receipt_ref",
            )
            if not isinstance(resolved_formal, FormalResearchOutputArtifact):
                raise ArtifactContractError("resolved formal output has the wrong type")
            if (
                not isinstance(resolved_promotion, FusionPromotionReceiptArtifact)
                or resolved_promotion.status != "PROMOTED"
            ):
                raise ArtifactContractError("resolved promotion receipt is not accepted")
            for index, ref in enumerate(resolved_formal.payload["artifact_refs"]):
                _resolve_exact_reference(
                    ref,
                    resolved_artifacts,
                    label=f"formal artifact_refs[{index}]",
                )
            calibration_kinds: list[str] = []
            for index, ref in enumerate(resolved_promotion.payload["calibration_receipt_refs"]):
                resolved_calibration = _resolve_exact_reference(
                    ref,
                    resolved_artifacts,
                    label=f"promotion calibration_receipt_refs[{index}]",
                )
                if (
                    not isinstance(
                        resolved_calibration,
                        FusionCalibrationReceiptArtifact,
                    )
                    or not resolved_calibration.payload["accepted"]
                ):
                    raise ArtifactContractError(
                        "ACTIVE promotion calibration receipt is not accepted"
                    )
                calibration_kinds.append(resolved_calibration.payload["calibration_kind"])
            if sorted(calibration_kinds) != [
                "FUNDAMENTAL_FORWARD",
                "FUSION_PROMOTION",
                "QUANT_TIMING",
            ]:
                raise ArtifactContractError("ACTIVE promotion calibration kinds are incomplete")
            _validate_fusion_promotion_evidence(
                resolved_promotion,
                resolved_artifacts,
            )
    elif status == "REVOKED":
        _validate_ref(
            payload["predecessor_active_receipt_ref"],
            strategy_id=strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v3.activation-receipt.v1",
            label="predecessor_active_receipt_ref",
        )
    else:
        promotion_ref = _validate_ref(
            payload["promotion_receipt_ref"],
            strategy_id=strategy_id,
            cutoff=cutoff,
            expected_version="myquant.v17.v3.fusion-promotion-receipt.v1",
            label="promotion_receipt_ref",
        )
        reasons = payload["rejection_reasons"]
        if reasons != sorted(reasons) or len(reasons) != len(set(reasons)):
            raise ArtifactContractError("activation rejection reasons must be unique and sorted")
        if promotion_receipt is not None:
            promotion = (
                promotion_receipt
                if isinstance(promotion_receipt, FusionPromotionReceiptArtifact)
                else validate_fusion_promotion_receipt(promotion_receipt)
            )
            if promotion.status != "PROMOTION_REJECTED":
                raise ArtifactContractError(
                    "ACTIVATION_REJECTED requires a rejected promotion receipt"
                )
            if (
                promotion_ref["artifact_id"] != promotion.payload["promotion_id"]
                or promotion_ref["semantic_sha256"] != promotion.semantic_sha256
            ):
                raise ArtifactContractError("activation rejection promotion binding mismatch")
        if resolved_artifacts is not None:
            resolved_promotion = _resolve_exact_reference(
                promotion_ref,
                resolved_artifacts,
                label="promotion_receipt_ref",
            )
            if (
                not isinstance(resolved_promotion, FusionPromotionReceiptArtifact)
                or resolved_promotion.status != "PROMOTION_REJECTED"
            ):
                raise ArtifactContractError(
                    "resolved activation rejection promotion is not rejected"
                )
    return ActivationReceiptArtifact(
        version,
        strategy_id,
        cutoff,
        digest,
        sealed,
        status,
    )


def validate_activation_pointer(
    payload: Mapping[str, Any],
    *,
    active_receipt: Mapping[str, Any] | ActivationReceiptArtifact | None = None,
    revocation_receipt: Mapping[str, Any] | ActivationReceiptArtifact | None = None,
    schema_checked: bool = False,
) -> ActivationPointerArtifact:
    result = _typed(
        payload,
        ActivationPointerArtifact,
        formal_authority=payload.get("status") == "ACTIVE",
        schema_checked=schema_checked,
    )
    assert isinstance(result, ActivationPointerArtifact)
    if payload["status"] == "ACTIVE":
        receipt_ref = _validate_ref(
            payload["active_receipt_ref"],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.activation-receipt.v1",
            label="active_receipt_ref",
        )
        formal_ref = _validate_ref(
            payload["formal_output_ref"],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.formal-research-output.v1",
            label="formal_output_ref",
        )
    else:
        revocation_ref = _validate_ref(
            payload["revocation_receipt_ref"],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.activation-receipt.v1",
            label="revocation_receipt_ref",
        )
        predecessor_ref = _validate_ref(
            payload["predecessor_active_receipt_ref"],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            expected_version="myquant.v17.v3.activation-receipt.v1",
            label="predecessor_active_receipt_ref",
        )
        if revocation_receipt is not None:
            revoked = (
                revocation_receipt
                if isinstance(revocation_receipt, ActivationReceiptArtifact)
                else validate_activation_receipt(revocation_receipt)
            )
            if revoked.status != "REVOKED":
                raise ArtifactContractError("revoked pointer must bind a REVOKED receipt")
            if (
                revocation_ref["artifact_id"] != revoked.payload["receipt_id"]
                or revocation_ref["semantic_sha256"] != revoked.semantic_sha256
                or predecessor_ref != revoked.payload["predecessor_active_receipt_ref"]
            ):
                raise ArtifactContractError("revoked pointer receipt binding mismatch")
        return result
    if active_receipt is not None:
        receipt = (
            active_receipt
            if isinstance(active_receipt, ActivationReceiptArtifact)
            else validate_activation_receipt(active_receipt)
        )
        if receipt.status != "ACTIVE":
            raise ArtifactContractError("activation pointer must bind an ACTIVE receipt")
        if (receipt.strategy_id, receipt.cutoff) != (result.strategy_id, result.cutoff):
            raise ArtifactContractError("activation pointer crosses strategy or cutoff")
        if (
            receipt_ref["artifact_id"] != receipt.payload["receipt_id"]
            or receipt_ref["semantic_sha256"] != receipt.semantic_sha256
        ):
            raise ArtifactContractError("activation pointer receipt binding mismatch")
        if formal_ref != receipt.payload["formal_output_ref"]:
            raise ArtifactContractError("activation pointer formal output binding mismatch")
    return result


def validate_activation_transition(
    receipt_payload: Mapping[str, Any],
    *,
    predecessor_active: Mapping[str, Any] | ActivationReceiptArtifact | None = None,
    current_pointer: Mapping[str, Any] | ActivationPointerArtifact | None = None,
    proposed_pointer: Mapping[str, Any] | None = None,
    history: Sequence[Mapping[str, Any] | ActivationReceiptArtifact] = (),
) -> ActivationTransition:
    receipt = validate_activation_receipt(receipt_payload)
    prior = tuple(
        (
            value
            if isinstance(value, ActivationReceiptArtifact)
            else validate_activation_receipt(value)
        )
        for value in history
    )
    same_binding_revoked = any(
        item.status == "REVOKED"
        and (item.strategy_id, item.cutoff) == (receipt.strategy_id, receipt.cutoff)
        for item in prior
    )
    if receipt.status == "ACTIVE":
        if predecessor_active is not None:
            raise ArtifactContractError("ACTIVE receipt must not bind a predecessor")
        if current_pointer is not None:
            raise ArtifactContractError("ACTIVE receipt requires an absent activation pointer")
        if same_binding_revoked:
            raise ArtifactContractError("a revoked strategy+cutoff cannot be reactivated")
        if proposed_pointer is None:
            raise ArtifactContractError("ACTIVE receipt requires an exact proposed pointer")
        pointer = validate_activation_pointer(proposed_pointer, active_receipt=receipt)
        return ActivationTransition(receipt, pointer)
    if receipt.status == "ACTIVATION_REJECTED":
        if predecessor_active is not None:
            raise ArtifactContractError("ACTIVATION_REJECTED must not bind a predecessor")
        if current_pointer is not None or proposed_pointer is not None:
            raise ArtifactContractError(
                "ACTIVATION_REJECTED must preserve an absent activation pointer"
            )
        return ActivationTransition(receipt, None)

    if predecessor_active is None:
        raise ArtifactContractError("REVOKED must bind its predecessor ACTIVE receipt")
    predecessor = (
        predecessor_active
        if isinstance(predecessor_active, ActivationReceiptArtifact)
        else validate_activation_receipt(predecessor_active)
    )
    if predecessor.status != "ACTIVE":
        raise ArtifactContractError("REVOKED predecessor is not ACTIVE")
    if (predecessor.strategy_id, predecessor.cutoff) != (
        receipt.strategy_id,
        receipt.cutoff,
    ):
        raise ArtifactContractError("REVOKED predecessor crosses strategy or cutoff")
    predecessor_ref = receipt.payload["predecessor_active_receipt_ref"]
    if (
        predecessor_ref["artifact_id"] != predecessor.payload["receipt_id"]
        or predecessor_ref["semantic_sha256"] != predecessor.semantic_sha256
    ):
        raise ArtifactContractError("REVOKED predecessor binding mismatch")
    if current_pointer is None:
        raise ArtifactContractError("REVOKED requires the predecessor ACTIVE pointer")
    pointer = (
        current_pointer
        if isinstance(current_pointer, ActivationPointerArtifact)
        else validate_activation_pointer(current_pointer, active_receipt=predecessor)
    )
    if (pointer.strategy_id, pointer.cutoff) != (receipt.strategy_id, receipt.cutoff):
        raise ArtifactContractError("REVOKED pointer crosses strategy or cutoff")
    if proposed_pointer is None:
        raise ArtifactContractError("REVOKED requires an exact revoked pointer")
    next_pointer = validate_activation_pointer(
        proposed_pointer,
        revocation_receipt=receipt,
    )
    if next_pointer.payload["status"] != "REVOKED":
        raise ArtifactContractError("REVOKED transition must propose a revoked pointer")
    if next_pointer.payload["pointer_id"] != pointer.payload["pointer_id"]:
        raise ArtifactContractError("revoked pointer must preserve pointer identity")
    return ActivationTransition(receipt, next_pointer)


def validate_shadow_latest(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> ShadowLatestArtifact:
    result = _typed(payload, ShadowLatestArtifact, schema_checked=schema_checked)
    assert isinstance(result, ShadowLatestArtifact)
    _validate_ref(
        payload["shadow_output_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        expected_version="myquant.v17.v3.shadow-output.v1",
        label="shadow_output_ref",
    )
    return result


def validate_formal_latest(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> FormalLatestArtifact:
    result = _typed(
        payload,
        FormalLatestArtifact,
        formal_authority=payload.get("status") == "ACTIVE",
        schema_checked=schema_checked,
    )
    assert isinstance(result, FormalLatestArtifact)
    if payload["status"] == "ACTIVE":
        for field, version in (
            ("activation_pointer_ref", "myquant.v17.v3.activation-pointer.v1"),
            ("active_receipt_ref", "myquant.v17.v3.activation-receipt.v1"),
            ("formal_output_ref", "myquant.v17.v3.formal-research-output.v1"),
        ):
            _validate_ref(
                payload[field],
                strategy_id=result.strategy_id,
                cutoff=result.cutoff,
                expected_version=version,
                label=field,
            )
    else:
        for field, version in (
            ("revoked_pointer_ref", "myquant.v17.v3.activation-pointer.v1"),
            ("revocation_receipt_ref", "myquant.v17.v3.activation-receipt.v1"),
            (
                "historical_formal_output_ref",
                "myquant.v17.v3.formal-research-output.v1",
            ),
        ):
            _validate_ref(
                payload[field],
                strategy_id=result.strategy_id,
                cutoff=result.cutoff,
                expected_version=version,
                label=field,
            )
    return result


def validate_unpublished_evidence(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> UnpublishedEvidenceArtifact:
    result = _typed(
        payload,
        UnpublishedEvidenceArtifact,
        schema_checked=schema_checked,
    )
    assert isinstance(result, UnpublishedEvidenceArtifact)
    _validate_ref(
        payload["artifact_ref"],
        strategy_id=result.strategy_id,
        cutoff=result.cutoff,
        label="artifact_ref",
    )
    reasons = payload["failure_reasons"]
    if reasons != sorted(reasons) or len(reasons) != len(set(reasons)):
        raise ArtifactContractError("unpublished evidence reasons must be unique and sorted")
    return result


def validate_ledger(
    payload: Mapping[str, Any],
    *,
    schema_checked: bool = False,
) -> LedgerArtifact:
    result = _typed(payload, LedgerArtifact, schema_checked=schema_checked)
    assert isinstance(result, LedgerArtifact)
    machine = state_machine()
    states = set(machine["states"])
    normal_transitions = {(row["from"], row["to"]) for row in machine["transitions"]}
    event_ids: set[str] = set()
    previous_at: str | None = None
    for index, event in enumerate(payload["events"]):
        event_id = event["event_id"]
        if event_id in event_ids:
            raise ArtifactContractError(f"duplicate ledger event ID: {event_id}")
        if event["from_state"] not in states or event["to_state"] not in states:
            raise ArtifactContractError(f"ledger event {index} references an unknown state")
        transition = (event["from_state"], event["to_state"])
        if (
            transition not in normal_transitions
            and terminal_class(event["to_state"]) != "HARD_STOP"
        ):
            raise ArtifactContractError(f"ledger event {index} is not an allowed transition")
        if previous_at is not None and event["at"] < previous_at:
            raise ArtifactContractError("ledger event timestamps are not append ordered")
        event_ids.add(event_id)
        previous_at = event["at"]
    if payload["events"] and payload["events"][-1]["to_state"] != payload["state"]:
        raise ArtifactContractError("ledger state does not match the final event")
    artifact_keys: list[tuple[str, str]] = []
    for index, row in enumerate(payload["artifacts"]):
        ref = _validate_ref(
            row["artifact_ref"],
            strategy_id=result.strategy_id,
            cutoff=result.cutoff,
            label=f"artifacts[{index}].artifact_ref",
        )
        artifact_keys.append((ref["relative_path"], ref["byte_sha256"]))
    if artifact_keys != sorted(artifact_keys) or len(artifact_keys) != len(set(artifact_keys)):
        raise ArtifactContractError("ledger artifacts must be uniquely sorted")
    return result


_VALIDATOR_BY_VERSION: Final[Mapping[str, Callable[..., ValidatedArtifact]]] = {
    "myquant.v17.v3.activation-pointer.v1": validate_activation_pointer,
    "myquant.v17.v3.activation-receipt.v1": validate_activation_receipt,
    "myquant.v17.v3.branch-output.v1": validate_branch_output,
    "myquant.v17.v3.calibration-gate-inputs.v1": (validate_calibration_gate_inputs),
    "myquant.v17.v3.deep-output.v1": validate_deep_output,
    "myquant.v17.v3.deep-research-inputs.v1": validate_deep_research_inputs,
    "myquant.v17.v3.factor-governance-readiness.v1": (validate_factor_governance_readiness),
    "myquant.v17.v3.formal-latest.v1": validate_formal_latest,
    "myquant.v17.v3.formal-research-output.v1": validate_formal_research_output,
    "myquant.v17.v3.fusion-calibration-inputs.v1": (validate_fusion_calibration_inputs),
    "myquant.v17.v3.fusion-calibration-receipt.v1": validate_fusion_calibration_receipt,
    "myquant.v17.v3.fusion-output.v1": validate_fusion_output,
    "myquant.v17.v3.fusion-promotion-receipt.v1": (validate_fusion_promotion_receipt),
    "myquant.v17.v3.initial-pool-output.v1": validate_initial_pool_output,
    "myquant.v17.v3.ledger.v1": validate_ledger,
    "myquant.v17.v3.portfolio-overlay.v1": validate_portfolio_overlay,
    "myquant.v17.v3.portfolio-output.v1": validate_portfolio_output,
    "myquant.v17.v3.pretrade-permissions.v1": validate_pretrade_permissions,
    "myquant.v17.v3.provisional-factor-baseline.v1": (validate_provisional_factor_baseline),
    "myquant.v17.v3.quant-preselection-inputs.v1": (validate_quant_preselection_inputs),
    "myquant.v17.v3.shadow-latest.v1": validate_shadow_latest,
    "myquant.v17.v3.shadow-output.v1": validate_shadow_output,
    "myquant.v17.v3.source-locator.v1": validate_source_locator,
    "myquant.v17.v3.source-manifest.v1": validate_source_manifest,
    "myquant.v17.v3.unpublished-evidence.v1": validate_unpublished_evidence,
}


def validate_typed_artifact(payload: Mapping[str, Any]) -> ValidatedArtifact:
    version = payload.get("version")
    validator = _VALIDATOR_BY_VERSION.get(version)
    if validator is None:
        raise ArtifactContractError(f"unsupported typed v3 artifact version: {version!r}")
    return validator(payload, schema_checked=True)


__all__ = [
    "ACTIVATION_STATUSES",
    "ActivationPointerArtifact",
    "ActivationReceiptArtifact",
    "ActivationTransition",
    "ArtifactContractError",
    "BranchOutputArtifact",
    "CalibrationGateInputsArtifact",
    "DeepOutputArtifact",
    "DeepResearchInputsArtifact",
    "FactorGovernanceReadinessArtifact",
    "FormalLatestArtifact",
    "FormalResearchOutputArtifact",
    "FusionCalibrationInputsArtifact",
    "FusionCalibrationReceiptArtifact",
    "FusionPromotionReceiptArtifact",
    "FusionOutputArtifact",
    "InitialPoolOutputArtifact",
    "LedgerArtifact",
    "PortfolioOverlayArtifact",
    "PortfolioOutputArtifact",
    "PretradePermissionsArtifact",
    "ProvisionalFactorBaselineArtifact",
    "QuantPreselectionInputsArtifact",
    "ShadowLatestArtifact",
    "ShadowOutputArtifact",
    "SourceLocatorArtifact",
    "SourceManifestArtifact",
    "UnpublishedEvidenceArtifact",
    "ValidatedArtifact",
    "validate_activation_pointer",
    "validate_activation_receipt",
    "validate_activation_transition",
    "validate_branch_output",
    "validate_branch_same_pool_binding",
    "validate_calibration_gate_inputs",
    "validate_deep_output",
    "validate_deep_research_inputs",
    "validate_factor_governance_readiness",
    "validate_formal_latest",
    "validate_formal_research_output",
    "validate_fusion_calibration_inputs",
    "validate_fusion_calibration_receipt",
    "validate_fusion_output",
    "validate_fusion_promotion_receipt",
    "validate_initial_pool_output",
    "validate_ledger",
    "validate_portfolio_overlay",
    "validate_portfolio_output",
    "validate_pretrade_permissions",
    "validate_provisional_factor_baseline",
    "validate_quant_preselection_inputs",
    "validate_shadow_latest",
    "validate_shadow_output",
    "validate_source_locator",
    "validate_source_manifest",
    "validate_staged_analysis_lineage",
    "validate_typed_artifact",
    "validate_unpublished_evidence",
]

"""Crash-safe FORMAL_ACTIVE publication for V17 v4 research outputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Final, Mapping, NoReturn, Sequence

from quant_investor.factors.production_control_v1 import (
    ACTIVE_SET_SCHEMA_VERSION,
    CONTROL_RECEIPT_SCHEMA_VERSION,
    canonical_file_bytes as factor_file_bytes,
    validate_active_set_pointer,
    validate_authorization_receipt,
    validate_control_receipt,
    validate_pre_activation_eligibility,
    validate_production_control_transaction,
    validate_production_registry,
)
from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
    verify_package,
    verify_runtime_build,
)
from quant_investor.v17_v4_contract.canonical import (
    load_canonical_resource,
    validate_semantic_sha,
)
from quant_investor.v17_v4_contract.identities import require_opaque_id
from quant_investor.v17_v4_contract.resources import (
    PACKAGE_MANIFEST_PATH,
    PACKAGE_MANIFEST_SHA256,
    RUNTIME_BUILD_MANIFEST_PATH,
    load_package_manifest,
    load_packaged_json,
    read_packaged_asset,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)
from quant_investor.v17_v4_runtime.deep_control import (
    revalidate_deep_evidence_bundle,
)
from quant_investor.v17_v4_runtime.portfolio_control import (
    revalidate_production_portfolio,
)
from quant_investor.v17_v4_runtime.source_storage import (
    EMPTY_SHA256,
    FORMAL_RESEARCH_ROOT,
    ExactReferenceReader,
    GovernedStore,
    SourceCASMismatch,
    SourceNotFoundError,
    SourceStorageError,
    SourceStorageSecurityError,
    canonical_governed_path,
)

INTENT_VERSION: Final = "myquant.v17.v4.formal-activation-intent.v1"
POINTER_VERSION: Final = "myquant.v17.v4.formal-active-pointer.v1"
COMPLETION_VERSION: Final = "myquant.v17.v4.formal-activation-receipt.v1"
REJECTION_VERSION: Final = "myquant.v17.v4.formal-activation-rejection.v1"
FORMAL_OUTPUT_VERSION: Final = "myquant.v17.v4.formal-output.v1"
NO_AUTHORITY: Final = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
PUBLICATION_AUTHORITY: Final = {
    **NO_AUTHORITY,
    "formal_research_publication": True,
}
_PACKAGE_REF_PATH: Final = (
    "quant_investor/v17_v4_contract/" + PACKAGE_MANIFEST_PATH
)
_RUNTIME_REF_PATH: Final = (
    "quant_investor/v17_v4_contract/" + RUNTIME_BUILD_MANIFEST_PATH
)
_CONTRACT_ROOT: Final = Path(__file__).resolve().parents[1] / "v17_v4_contract"


class FormalActivationError(RuntimeError):
    """Raised when formal publication cannot prove its full closure."""


class FormalActivationCrash(RuntimeError):
    """Deterministic fault-injection boundary for recovery tests."""


@dataclass(frozen=True)
class FormalActivationResult:
    status: str
    intent_ref: Mapping[str, str]
    pointer_ref: Mapping[str, str]
    completion_ref: Mapping[str, str]
    recovered: bool


@dataclass(frozen=True)
class FormalState:
    status: str
    intent: Mapping[str, Any] | None
    pointer: Mapping[str, Any] | None
    completion: Mapping[str, Any] | None


def _blocked(reason: str) -> NoReturn:
    raise FormalActivationError(f"V17_V4_FORMAL_ACTIVATION_BLOCKED:{reason}")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _ordered_refs(
    references: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (dict(reference) for reference in references),
        key=lambda row: (
            row["relative_path"],
            row["byte_sha256"],
            row["artifact_id"],
        ),
    )


def _manifest_bytes_document(
    relative_path: str,
) -> tuple[bytes, dict[str, Any]]:
    if relative_path == PACKAGE_MANIFEST_PATH:
        document = load_package_manifest()
        raw = (_CONTRACT_ROOT / relative_path).read_bytes()
        if _sha(raw) != PACKAGE_MANIFEST_SHA256:
            _blocked("PACKAGE_MANIFEST_READBACK")
        return raw, document
    return (
        read_packaged_asset(relative_path),
        load_packaged_json(relative_path),
    )


def artifact_ref(
    document: Mapping[str, Any],
    *,
    relative_path: str,
    cutoff: str | None = None,
) -> dict[str, str]:
    version = str(document.get("version", ""))
    identity_field = artifact_identity_field(version)
    raw = canonical_resource_bytes(document)
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": version,
        "byte_sha256": _sha(raw),
        "cutoff": cutoff or str(document["cutoff"]),
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def factor_artifact_ref(
    document: Mapping[str, Any],
    *,
    relative_path: str,
    strategy_id: str,
    cutoff: str,
) -> dict[str, str]:
    schema = str(document["schema_version"])
    identity = document.get("active_set_id") or document.get("receipt_id")
    if type(identity) is not str:
        _blocked("FACTOR_ARTIFACT_IDENTITY")
    raw = factor_file_bytes(document)
    return {
        "artifact_id": identity,
        "artifact_version": schema,
        "byte_sha256": _sha(raw),
        "cutoff": cutoff,
        "relative_path": relative_path,
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": require_opaque_id(
            strategy_id,
            label="strategy_id",
        ),
    }


def manifest_refs(
    *,
    strategy_id: str,
    cutoff: str,
) -> tuple[dict[str, str], dict[str, str]]:
    strategy = require_opaque_id(strategy_id, label="strategy_id")
    verify_package()
    verify_runtime_build()
    result: list[dict[str, str]] = []
    for path, reference_path in (
        (PACKAGE_MANIFEST_PATH, _PACKAGE_REF_PATH),
        (RUNTIME_BUILD_MANIFEST_PATH, _RUNTIME_REF_PATH),
    ):
        raw, document = _manifest_bytes_document(path)
        version = str(document["version"])
        result.append(
            {
                "artifact_id": version,
                "artifact_version": version,
                "byte_sha256": _sha(raw),
                "cutoff": cutoff,
                "relative_path": reference_path,
                "semantic_sha256": str(document["semantic_sha256"]),
                "strategy_id": strategy,
            }
        )
    return result[0], result[1]


def build_formal_output(
    *,
    output_id: str,
    strategy_id: str,
    cutoff: str,
    evidence_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    document = seal_semantic(
        {
            "authority": dict(PUBLICATION_AUTHORITY),
            "cutoff": cutoff,
            "evidence_refs": _ordered_refs(evidence_refs),
            "output_id": require_opaque_id(output_id, label="output_id"),
            "protocol_version": PROTOCOL_VERSION,
            "strategy_id": require_opaque_id(
                strategy_id,
                label="strategy_id",
            ),
            "terminal_state": "PUBLISHED_RESEARCH_ONLY",
            "version": FORMAL_OUTPUT_VERSION,
        }
    )
    validate_artifact(document)
    return document


def build_activation_intent(
    *,
    intent_id: str,
    strategy_id: str,
    cutoff: str,
    created_at: str,
    expected_pointer_sha256: str,
    formal_output_ref: Mapping[str, Any],
    source_locator_ref: Mapping[str, Any],
    quant_calibration_receipt_ref: Mapping[str, Any],
    fundamental_calibration_receipt_ref: Mapping[str, Any],
    fusion_promotion_receipt_ref: Mapping[str, Any],
    factor_control_active_set_ref: Mapping[str, Any],
    factor_control_activation_receipt_ref: Mapping[str, Any],
    deep_bundle_ref: Mapping[str, Any],
    portfolio_output_ref: Mapping[str, Any],
    holdings_snapshot_ref: Mapping[str, Any],
    risk_policy_ref: Mapping[str, Any],
    macro_overlay_ref: Mapping[str, Any],
    markov_overlay_ref: Mapping[str, Any],
    package_manifest_ref: Mapping[str, Any],
    runtime_manifest_ref: Mapping[str, Any],
) -> dict[str, Any]:
    refs = {
        "deep_bundle_ref": dict(deep_bundle_ref),
        "factor_control_activation_receipt_ref": dict(
            factor_control_activation_receipt_ref
        ),
        "factor_control_active_set_ref": dict(
            factor_control_active_set_ref
        ),
        "formal_output_ref": dict(formal_output_ref),
        "fundamental_calibration_receipt_ref": dict(
            fundamental_calibration_receipt_ref
        ),
        "fusion_promotion_receipt_ref": dict(
            fusion_promotion_receipt_ref
        ),
        "holdings_snapshot_ref": dict(holdings_snapshot_ref),
        "macro_overlay_ref": dict(macro_overlay_ref),
        "markov_overlay_ref": dict(markov_overlay_ref),
        "package_manifest_ref": dict(package_manifest_ref),
        "portfolio_output_ref": dict(portfolio_output_ref),
        "quant_calibration_receipt_ref": dict(
            quant_calibration_receipt_ref
        ),
        "risk_policy_ref": dict(risk_policy_ref),
        "runtime_manifest_ref": dict(runtime_manifest_ref),
        "source_locator_ref": dict(source_locator_ref),
    }
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "created_at": created_at,
            "cutoff": cutoff,
            **refs,
            "evidence_refs": _ordered_refs(list(refs.values())),
            "expected_pointer_sha256": expected_pointer_sha256,
            "from_state": (
                "V15_DEFAULT"
                if expected_pointer_sha256 == EMPTY_SHA256
                else "FORMAL_ACTIVE"
            ),
            "intent_id": require_opaque_id(intent_id, label="intent_id"),
            "protocol_version": PROTOCOL_VERSION,
            "strategy_id": require_opaque_id(
                strategy_id,
                label="strategy_id",
            ),
            "version": INTENT_VERSION,
        }
    )
    validate_artifact(document)
    return document


class _FormalActivationWriter(GovernedStore):
    """Write only the five formal-activation path shapes."""

    def _canonical_path(
        self,
        value: str | PurePosixPath,
    ) -> PurePosixPath:
        path = canonical_governed_path(value)
        parts = path.parts
        prefix = (
            "results",
            "v17_v4_formal_research",
            "strategies",
        )
        if parts[:3] != prefix or len(parts) < 5:
            raise SourceStorageSecurityError(
                "path is outside formal activation strategy roots"
            )
        try:
            require_opaque_id(parts[3], label="strategy_id")
        except ValueError as exc:
            raise SourceStorageSecurityError(
                "formal activation strategy path is invalid"
            ) from exc
        suffix = parts[4:]
        if suffix in {(".active.lock",), ("_active.json",)}:
            return path
        if (
            len(suffix) == 2
            and suffix[0]
            in {
                "completion_receipts",
                "intents",
                "rejection_receipts",
            }
            and suffix[1].endswith(".json")
        ):
            try:
                require_opaque_id(
                    suffix[1][:-5],
                    label="activation object id",
                )
            except ValueError as exc:
                raise SourceStorageSecurityError(
                    "formal activation object path is invalid"
                ) from exc
            return path
        raise SourceStorageSecurityError(
            "path is outside formal activation writer whitelist"
        )


class FormalActivationService:
    """Own the narrow formal activation paths and exact transition."""

    def __init__(self, workspace_root: str | Path) -> None:
        self._writer = _FormalActivationWriter(workspace_root)
        self._reader = ExactReferenceReader(workspace_root)

    @staticmethod
    def _strategy_root(strategy_id: str) -> PurePosixPath:
        strategy = require_opaque_id(strategy_id, label="strategy_id")
        return FORMAL_RESEARCH_ROOT / "strategies" / strategy

    def _paths(
        self,
        strategy_id: str,
        intent_id: str,
    ) -> dict[str, PurePosixPath]:
        intent = require_opaque_id(intent_id, label="intent_id")
        root = self._strategy_root(strategy_id)
        return {
            "completion": root / "completion_receipts" / f"{intent}.json",
            "intent": root / "intents" / f"{intent}.json",
            "lock": root / ".active.lock",
            "pointer": root / "_active.json",
            "rejection": root / "rejection_receipts" / f"{intent}.json",
        }

    def artifact_loader(self, reference: Mapping[str, str]) -> bytes:
        return self._reader.read(
            reference["relative_path"],
            reference["byte_sha256"],
        )

    def _load_v4(
        self,
        reference: Mapping[str, Any],
        *,
        expected_version: str,
    ) -> dict[str, Any]:
        raw = self.artifact_loader(reference)
        try:
            document = load_canonical_resource(raw, label=expected_version)
            if type(document) is not dict:
                _blocked("ARTIFACT_ROOT")
            validated = validate_artifact(
                document,
                artifact_loader=self.artifact_loader,
            )
        except (TypeError, ValueError) as exc:
            raise FormalActivationError(
                "V17_V4_FORMAL_ACTIVATION_BLOCKED:ARTIFACT_VALIDATION"
            ) from exc
        if (
            document.get("version") != expected_version
            or document.get("semantic_sha256")
            != reference["semantic_sha256"]
            or document.get("strategy_id") != reference["strategy_id"]
            or document.get(artifact_identity_field(expected_version))
            != reference["artifact_id"]
            or validated.payload != document
        ):
            _blocked("ARTIFACT_REFERENCE_MISMATCH")
        return document

    def _read_factor_common(
        self,
        reference: Mapping[str, Any],
        *,
        expected_schema: str,
    ) -> dict[str, Any]:
        raw = self.artifact_loader(reference)
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormalActivationError(
                "V17_V4_FORMAL_ACTIVATION_BLOCKED:FACTOR_JSON"
            ) from exc
        if (
            type(document) is not dict
            or factor_file_bytes(document) != raw
            or document.get("schema_version") != expected_schema
            or document.get("semantic_sha256")
            != reference["semantic_sha256"]
            or reference["artifact_version"] != expected_schema
        ):
            _blocked("FACTOR_COMMON_REFERENCE_MISMATCH")
        identity = document.get("active_set_id") or document.get(
            "receipt_id"
        )
        if identity != reference["artifact_id"]:
            _blocked("FACTOR_COMMON_IDENTITY_MISMATCH")
        return document

    def _read_factor_native(
        self,
        reference: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = self._reader.read(
            str(reference["relative_path"]),
            str(reference["byte_sha256"]),
        )
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormalActivationError(
                "V17_V4_FORMAL_ACTIVATION_BLOCKED:FACTOR_NATIVE_JSON"
            ) from exc
        if (
            type(document) is not dict
            or factor_file_bytes(document) != raw
            or document.get("semantic_sha256")
            != reference["semantic_sha256"]
            or document.get("schema_version")
            != reference["artifact_schema"]
        ):
            _blocked("FACTOR_NATIVE_REFERENCE_MISMATCH")
        return document

    def _revalidate_factor_closure(
        self,
        intent: Mapping[str, Any],
    ) -> None:
        active = self._read_factor_common(
            intent["factor_control_active_set_ref"],
            expected_schema=ACTIVE_SET_SCHEMA_VERSION,
        )
        receipt = self._read_factor_common(
            intent["factor_control_activation_receipt_ref"],
            expected_schema=CONTROL_RECEIPT_SCHEMA_VERSION,
        )
        validate_active_set_pointer(active)
        transaction = self._read_factor_native(receipt["transaction_ref"])
        registry = self._read_factor_native(
            transaction["proposed_registry_ref"]
        )
        eligibility = self._read_factor_native(
            transaction["v4_pre_activation_eligibility_ref"]
        )
        authorization = self._read_factor_native(
            transaction["production_control_authorization_receipt_ref"]
        )
        validate_production_registry(registry)
        validate_pre_activation_eligibility(
            eligibility,
            registry=registry,
        )
        validate_authorization_receipt(
            authorization,
            observed_at=transaction["activated_at"],
        )
        normalized = validate_production_control_transaction(
            transaction,
            registry=registry,
            pre_activation_eligibility=eligibility,
            authorization_receipt=authorization,
        )
        validate_control_receipt(receipt, transaction=normalized)
        expected_active_native_ref = {
            "artifact_schema": (
                intent["factor_control_active_set_ref"][
                    "artifact_version"
                ]
            ),
            "byte_sha256": (
                intent["factor_control_active_set_ref"]["byte_sha256"]
            ),
            "relative_path": (
                intent["factor_control_active_set_ref"]["relative_path"]
            ),
            "schema_version": (
                "factor-governance-production-control.artifact-ref.v1"
            ),
            "semantic_sha256": (
                intent["factor_control_active_set_ref"][
                    "semantic_sha256"
                ]
            ),
        }
        if (
            normalized["proposed_active_set"] != active
            or receipt["active_set_ref"] != expected_active_native_ref
            or receipt["active_set_readback_sha256"]
            != intent["factor_control_active_set_ref"]["byte_sha256"]
            or receipt["transaction_id"] != active["transaction_id"]
        ):
            _blocked("FACTOR_CLOSURE_MISMATCH")

    def _revalidate_manifests(
        self,
        intent: Mapping[str, Any],
    ) -> None:
        package_verified = verify_package()
        runtime_verified = verify_runtime_build()
        package_raw, package = _manifest_bytes_document(
            PACKAGE_MANIFEST_PATH
        )
        runtime_raw, runtime = _manifest_bytes_document(
            RUNTIME_BUILD_MANIFEST_PATH
        )
        for reference, path, raw, document, version in (
            (
                intent["package_manifest_ref"],
                _PACKAGE_REF_PATH,
                package_raw,
                package,
                "myquant.v17.v4.package-manifest.v1",
            ),
            (
                intent["runtime_manifest_ref"],
                _RUNTIME_REF_PATH,
                runtime_raw,
                runtime,
                "myquant.v17.v4.runtime-build-manifest.v1",
            ),
        ):
            if (
                reference["relative_path"] != path
                or reference["artifact_id"] != version
                or reference["artifact_version"] != version
                or reference["byte_sha256"] != _sha(raw)
                or reference["semantic_sha256"]
                != document["semantic_sha256"]
            ):
                _blocked("MANIFEST_REFERENCE_MISMATCH")
        if (
            package_verified[PACKAGE_MANIFEST_PATH]
            != PACKAGE_MANIFEST_SHA256
            or not runtime_verified
        ):
            _blocked("MANIFEST_READBACK")

    def _revalidate_closure(
        self,
        intent: Mapping[str, Any],
    ) -> None:
        validate_artifact(intent)
        formal = self._load_v4(
            intent["formal_output_ref"],
            expected_version=FORMAL_OUTPUT_VERSION,
        )
        locator = self._load_v4(
            intent["source_locator_ref"],
            expected_version="myquant.v17.v4.preselect-locator.v1",
        )
        self._load_v4(
            locator["pit_catalog_ref"],
            expected_version="myquant.v17.v4.pit-generation-catalog.v1",
        )
        quant = self._load_v4(
            intent["quant_calibration_receipt_ref"],
            expected_version="myquant.v17.v4.calibration-receipt.v1",
        )
        fundamental = self._load_v4(
            intent["fundamental_calibration_receipt_ref"],
            expected_version="myquant.v17.v4.calibration-receipt.v1",
        )
        promotion = self._load_v4(
            intent["fusion_promotion_receipt_ref"],
            expected_version=(
                "myquant.v17.v4.fusion-promotion-receipt.v1"
            ),
        )
        deep, _ = revalidate_deep_evidence_bundle(
            intent["deep_bundle_ref"],
            artifact_loader=self.artifact_loader,
        )
        portfolio = revalidate_production_portfolio(
            intent["portfolio_output_ref"],
            artifact_loader=self.artifact_loader,
        )
        if (
            quant["calibration_kind"] != "QUANT_TIMING"
            or fundamental["calibration_kind"] != "FUNDAMENTAL_FORWARD"
            or quant["accepted"] is not True
            or fundamental["accepted"] is not True
            or promotion["accepted"] is not True
            or promotion["status"] != "PROMOTED"
            or promotion["quant_calibration_receipt_ref"]
            != intent["quant_calibration_receipt_ref"]
            or promotion["fundamental_calibration_receipt_ref"]
            != intent["fundamental_calibration_receipt_ref"]
            or portfolio["deep_bundle_ref"] != intent["deep_bundle_ref"]
            or portfolio["holdings_snapshot_ref"]
            != intent["holdings_snapshot_ref"]
            or portfolio["risk_policy_ref"] != intent["risk_policy_ref"]
            or portfolio["macro_overlay_ref"] != intent["macro_overlay_ref"]
            or portfolio["markov_overlay_ref"] != intent["markov_overlay_ref"]
            or deep["bundle_id"]
            != intent["deep_bundle_ref"]["artifact_id"]
            or intent["portfolio_output_ref"] not in formal["evidence_refs"]
        ):
            _blocked("FORMAL_CLOSURE_MISMATCH")
        self._revalidate_factor_closure(intent)
        self._revalidate_manifests(intent)

    def _pointer_document(
        self,
        intent: Mapping[str, Any],
        intent_ref: Mapping[str, str],
    ) -> dict[str, Any]:
        return seal_semantic(
            {
                "authority": dict(NO_AUTHORITY),
                "cutoff": intent["cutoff"],
                "intent_ref": dict(intent_ref),
                "pointer_id": f"formal-pointer-{intent['intent_id']}",
                "protocol_version": PROTOCOL_VERSION,
                "state": "PENDING_COMPLETION",
                "strategy_id": intent["strategy_id"],
                "updated_at": intent["created_at"],
                "version": POINTER_VERSION,
            }
        )

    def _completion_document(
        self,
        *,
        intent: Mapping[str, Any],
        intent_ref: Mapping[str, str],
        pointer_ref: Mapping[str, str],
    ) -> dict[str, Any]:
        proposed = pointer_ref["byte_sha256"]
        return seal_semantic(
            {
                "authority": dict(PUBLICATION_AUTHORITY),
                "cutoff": intent["cutoff"],
                "evidence_refs": _ordered_refs(
                    [intent_ref, pointer_ref]
                ),
                "expected_pointer_sha256": (
                    intent["expected_pointer_sha256"]
                ),
                "from_state": intent["from_state"],
                "intent_ref": dict(intent_ref),
                "observed_pointer_sha256": (
                    intent["expected_pointer_sha256"]
                ),
                "pointer_ref": dict(pointer_ref),
                "post_readback_sha256": proposed,
                "proposed_pointer_sha256": proposed,
                "protocol_version": PROTOCOL_VERSION,
                "receipt_id": intent["intent_id"],
                "recorded_at": intent["created_at"],
                "status": "FORMAL_ACTIVATED",
                "strategy_id": intent["strategy_id"],
                "to_state": "FORMAL_ACTIVE",
                "version": COMPLETION_VERSION,
            }
        )

    def _rejection_document(
        self,
        *,
        intent: Mapping[str, Any],
        observed_pointer_sha256: str,
        reason: str,
    ) -> dict[str, Any]:
        return seal_semantic(
            {
                "attempted_evidence_refs": list(intent["evidence_refs"]),
                "authority": dict(NO_AUTHORITY),
                "expected_pointer_sha256": (
                    intent["expected_pointer_sha256"]
                ),
                "from_state": intent["from_state"],
                "observed_pointer_sha256": observed_pointer_sha256,
                "protocol_version": PROTOCOL_VERSION,
                "receipt_id": intent["intent_id"],
                "recorded_at": intent["created_at"],
                "rejection_reasons": [reason],
                "status": "FORMAL_ACTIVATION_REJECTED",
                "strategy_id": intent["strategy_id"],
                "to_state": intent["from_state"],
                "version": REJECTION_VERSION,
            }
        )

    def _write_rejection(
        self,
        *,
        intent: Mapping[str, Any],
        path: PurePosixPath,
        observed_pointer_sha256: str,
        reason: str,
    ) -> None:
        rejection = self._rejection_document(
            intent=intent,
            observed_pointer_sha256=observed_pointer_sha256,
            reason=reason,
        )
        validate_artifact(rejection)
        self._writer.write_exact_once(
            path,
            canonical_resource_bytes(rejection),
        )

    def activate(
        self,
        intent: Mapping[str, Any],
        *,
        crash_after: str | None = None,
    ) -> FormalActivationResult:
        validated = validate_artifact(intent)
        if validated.version != INTENT_VERSION:
            _blocked("INTENT_VERSION")
        normalized = dict(validated.payload)
        paths = self._paths(
            normalized["strategy_id"],
            normalized["intent_id"],
        )
        with self._writer.locked(paths["lock"]):
            current = self._writer.read_optional(paths["pointer"])
            observed = (
                EMPTY_SHA256 if current is None else current.byte_sha256
            )
            if self._writer.read_optional(paths["rejection"]) is not None:
                _blocked("INTENT_PREVIOUSLY_REJECTED")
            try:
                self._revalidate_closure(normalized)
            except (
                FormalActivationError,
                OSError,
                SourceStorageError,
                TypeError,
                ValueError,
            ):
                if observed == normalized["expected_pointer_sha256"]:
                    self._write_rejection(
                        intent=normalized,
                        path=paths["rejection"],
                        observed_pointer_sha256=observed,
                        reason="CLOSURE_REVALIDATION_FAILED",
                    )
                raise
            intent_raw = canonical_resource_bytes(normalized)
            self._writer.write_exact_once(paths["intent"], intent_raw)
            intent_ref = artifact_ref(
                normalized,
                relative_path=str(paths["intent"]),
            )
            pointer = self._pointer_document(normalized, intent_ref)
            validate_artifact(pointer)
            pointer_raw = canonical_resource_bytes(pointer)
            proposed = _sha(pointer_raw)
            if crash_after == "intent":
                raise FormalActivationCrash("crash after intent")
            recovered = False
            if observed == normalized["expected_pointer_sha256"]:
                try:
                    self._writer.replace_cas(
                        paths["pointer"],
                        normalized["expected_pointer_sha256"],
                        pointer_raw,
                    )
                except SourceCASMismatch as exc:
                    self._write_rejection(
                        intent=normalized,
                        path=paths["rejection"],
                        observed_pointer_sha256=exc.observed_sha256,
                        reason="POINTER_CAS_CONFLICT",
                    )
                    raise FormalActivationError(
                        "V17_V4_FORMAL_ACTIVATION_BLOCKED:"
                        "POINTER_CAS_CONFLICT"
                    ) from exc
            elif observed == proposed and current is not None:
                if current.data != pointer_raw:
                    _blocked("POINTER_HASH_COLLISION")
                recovered = True
            else:
                _blocked("POINTER_THIRD_STATE")
            if crash_after == "cas":
                raise FormalActivationCrash("crash after pointer CAS")
            readback = self._writer.read(paths["pointer"], proposed)
            if readback != pointer_raw:
                _blocked("POINTER_READBACK")
            if crash_after == "readback":
                raise FormalActivationCrash("crash after pointer readback")
            pointer_ref = artifact_ref(
                pointer,
                relative_path=str(paths["pointer"]),
            )
            completion = self._completion_document(
                intent=normalized,
                intent_ref=intent_ref,
                pointer_ref=pointer_ref,
            )
            validate_artifact(completion)
            completion_raw = canonical_resource_bytes(completion)
            self._writer.write_exact_once(
                paths["completion"],
                completion_raw,
            )
            if crash_after == "completion":
                raise FormalActivationCrash("crash after completion")
            state = self.resolve(normalized["strategy_id"])
            if state.status != "FORMAL_ACTIVE":
                _blocked("COMPLETION_READBACK")
            completion_ref = artifact_ref(
                completion,
                relative_path=str(paths["completion"]),
            )
            return FormalActivationResult(
                "FORMAL_ACTIVE",
                intent_ref,
                pointer_ref,
                completion_ref,
                recovered,
            )

    def resolve(self, strategy_id: str) -> FormalState:
        root = self._strategy_root(strategy_id)
        pointer_path = root / "_active.json"
        stored = self._writer.read_optional(pointer_path)
        if stored is None:
            return FormalState("V15_DEFAULT", None, None, None)
        try:
            pointer_value = load_canonical_resource(
                stored.data,
                label="formal active pointer",
            )
            if type(pointer_value) is not dict:
                _blocked("POINTER_ROOT")
            pointer_artifact = validate_artifact(pointer_value)
            if pointer_artifact.version != POINTER_VERSION:
                _blocked("POINTER_VERSION")
            pointer = dict(pointer_artifact.payload)
            intent = self._load_v4(
                pointer["intent_ref"],
                expected_version=INTENT_VERSION,
            )
        except (SourceNotFoundError, TypeError, ValueError) as exc:
            raise FormalActivationError(
                "V17_V4_FORMAL_ACTIVATION_BLOCKED:POINTER_CHAIN"
            ) from exc
        paths = self._paths(strategy_id, intent["intent_id"])
        if (
            str(paths["pointer"]) != stored.relative_path
            or pointer["intent_ref"]["relative_path"] != str(paths["intent"])
            or pointer["cutoff"] != intent["cutoff"]
            or pointer["strategy_id"] != intent["strategy_id"]
        ):
            _blocked("POINTER_INTENT_PATH_BINDING")
        completion_stored = self._writer.read_optional(paths["completion"])
        if completion_stored is None:
            return FormalState(
                "PENDING_COMPLETION",
                intent,
                pointer,
                None,
            )
        try:
            completion_value = load_canonical_resource(
                completion_stored.data,
                label="formal completion receipt",
            )
            if type(completion_value) is not dict:
                _blocked("COMPLETION_ROOT")
            completion_artifact = validate_artifact(completion_value)
            if completion_artifact.version != COMPLETION_VERSION:
                _blocked("COMPLETION_VERSION")
            completion = dict(completion_artifact.payload)
        except (TypeError, ValueError) as exc:
            raise FormalActivationError(
                "V17_V4_FORMAL_ACTIVATION_BLOCKED:COMPLETION_CHAIN"
            ) from exc
        intent_ref = artifact_ref(
            intent,
            relative_path=str(paths["intent"]),
        )
        pointer_ref = artifact_ref(
            pointer,
            relative_path=str(paths["pointer"]),
        )
        if (
            completion["receipt_id"] != intent["intent_id"]
            or completion["intent_ref"] != intent_ref
            or completion["pointer_ref"] != pointer_ref
            or completion["proposed_pointer_sha256"]
            != stored.byte_sha256
            or completion["post_readback_sha256"]
            != stored.byte_sha256
            or completion["expected_pointer_sha256"]
            != intent["expected_pointer_sha256"]
            or completion["observed_pointer_sha256"]
            != intent["expected_pointer_sha256"]
        ):
            _blocked("COMPLETION_POINTER_BINDING")
        self._revalidate_closure(intent)
        return FormalState("FORMAL_ACTIVE", intent, pointer, completion)


__all__ = [
    "COMPLETION_VERSION",
    "FORMAL_OUTPUT_VERSION",
    "FormalActivationCrash",
    "FormalActivationError",
    "FormalActivationResult",
    "FormalActivationService",
    "FormalState",
    "INTENT_VERSION",
    "POINTER_VERSION",
    "REJECTION_VERSION",
    "artifact_ref",
    "build_activation_intent",
    "build_formal_output",
    "factor_artifact_ref",
    "manifest_refs",
]

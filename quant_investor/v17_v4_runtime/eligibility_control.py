"""Crash-safe DEFAULT_ELIGIBLE closure for V17 v4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Sequence

from quant_investor.research_runtime_control.canonical import (
    BOOTSTRAP_RECEIPT_VERSION,
    SELECTOR_VERSION,
    artifact_reference as control_artifact_reference,
    decode_reference as decode_control_reference,
)
from quant_investor.research_runtime_control.control import SELECTOR_PATH
from quant_investor.research_runtime_control.storage import (
    ControlStorageError,
    ControlStore,
)
from quant_investor.v17_v4_contract import (
    PROTOCOL_VERSION,
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import (
    load_canonical_resource,
)
from quant_investor.v17_v4_contract.identities import (
    require_opaque_id,
    require_utc_timestamp,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)
from quant_investor.v17_v4_runtime.formal_activation import (
    artifact_ref,
)
from quant_investor.v17_v4_runtime.public_surfaces import (
    build_public_surface_compatibility_receipts,
    resolve_public_run,
)
from quant_investor.v17_v4_runtime.source_storage import (
    EMPTY_SHA256,
    FORMAL_RESEARCH_ROOT,
    ExactReferenceReader,
    GovernedStore,
    SourceStorageError,
    SourceStorageSecurityError,
    canonical_governed_path,
)

INTENT_VERSION: Final = "myquant.v17.v4.default-eligibility-intent.v1"
POINTER_VERSION: Final = "myquant.v17.v4.default-eligible-pointer.v1"
COMPLETION_VERSION: Final = (
    "myquant.v17.v4.default-eligibility-receipt.v1"
)
PUBLIC_RECEIPT_VERSION: Final = (
    "myquant.v17.v4.public-surface-compatibility-receipt.v1"
)
VALIDATION_RECEIPT_VERSION: Final = (
    "myquant.v17.v4.validation-receipt.v1"
)
ROLLBACK_DRILL_VERSION: Final = (
    "myquant.v17.v4.rollback-drill-receipt.v1"
)
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
VALIDATION_KINDS: Final = frozenset(
    {
        "PACKAGE_SCHEMA_MANIFEST",
        "SECRET_SCAN",
        "SIDE_EFFECT_SCAN",
        "V15_REGRESSION",
        "V4_FULL_TESTS",
    }
)


class EligibilityError(RuntimeError):
    """Raised when DEFAULT_ELIGIBLE cannot prove its complete closure."""


class EligibilityCrash(RuntimeError):
    """Deterministic fault injection for eligibility recovery tests."""


@dataclass(frozen=True)
class EligibilityResult:
    status: str
    intent_ref: Mapping[str, str]
    pointer_ref: Mapping[str, str]
    completion_ref: Mapping[str, str]
    recovered: bool


@dataclass(frozen=True)
class EligibilityState:
    status: str
    intent: Mapping[str, Any] | None
    pointer: Mapping[str, Any] | None
    completion: Mapping[str, Any] | None


def _blocked(reason: str) -> None:
    raise EligibilityError(f"V17_V4_ELIGIBILITY_BLOCKED:{reason}")


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


def build_validation_receipt(
    *,
    receipt_id: str,
    strategy_id: str,
    cutoff: str,
    recorded_at: str,
    validation_kind: str,
    command_id: str,
    command_sha256: str,
    result_sha256: str,
    passed_count: int,
    skipped_count: int = 0,
) -> dict[str, Any]:
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "blocker_count": 0,
            "command_id": require_opaque_id(
                command_id,
                label="command_id",
            ),
            "command_sha256": command_sha256,
            "cutoff": cutoff,
            "exit_code": 0,
            "failed_count": 0,
            "passed_count": passed_count,
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": require_opaque_id(
                receipt_id,
                label="receipt_id",
            ),
            "recorded_at": recorded_at,
            "result_sha256": result_sha256,
            "skipped_count": skipped_count,
            "status": "PASSED",
            "strategy_id": require_opaque_id(
                strategy_id,
                label="strategy_id",
            ),
            "validation_kind": validation_kind,
            "version": VALIDATION_RECEIPT_VERSION,
        }
    )
    validate_artifact(document)
    return document


def build_rollback_drill_receipt(
    *,
    receipt_id: str,
    strategy_id: str,
    cutoff: str,
    recorded_at: str,
    isolated_control_root_digest: str,
    bootstrap_receipt_sha256: str,
    cutover_receipt_sha256: str,
    rollback_receipt_sha256: str,
    final_selector_sha256: str,
) -> dict[str, Any]:
    document = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "bootstrap_receipt_sha256": bootstrap_receipt_sha256,
            "cutoff": cutoff,
            "cutover_receipt_sha256": cutover_receipt_sha256,
            "failure_injection_recovered": True,
            "final_protocol_id": "v15",
            "final_selector_sha256": final_selector_sha256,
            "isolated_control_root_digest": isolated_control_root_digest,
            "production_selector_writes": False,
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": require_opaque_id(
                receipt_id,
                label="receipt_id",
            ),
            "recorded_at": recorded_at,
            "rollback_receipt_sha256": rollback_receipt_sha256,
            "scenarios": [
                "BOOTSTRAP",
                "CUTOVER",
                "CRASH_AFTER_CAS_RECOVERY",
                "ROLLBACK",
            ],
            "status": "PASSED",
            "strategy_id": require_opaque_id(
                strategy_id,
                label="strategy_id",
            ),
            "version": ROLLBACK_DRILL_VERSION,
        }
    )
    validate_artifact(document)
    return document


class _EligibilityWriter(GovernedStore):
    """Write only eligibility evidence, intent, pointer, and completion."""

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
        if parts[:3] != prefix or len(parts) < 6:
            raise SourceStorageSecurityError(
                "path is outside eligibility strategy roots"
            )
        try:
            require_opaque_id(parts[3], label="strategy_id")
        except ValueError as exc:
            raise SourceStorageSecurityError(
                "eligibility strategy identity is invalid"
            ) from exc
        if parts[4] != "eligibility":
            raise SourceStorageSecurityError(
                "path is outside the eligibility namespace"
            )
        suffix = parts[5:]
        if suffix in {(".active.lock",), ("_active.json",)}:
            return path
        if (
            len(suffix) == 2
            and suffix[0]
            in {
                "completion_receipts",
                "intents",
                "public_surface_receipts",
                "rollback_drills",
                "validation_receipts",
            }
            and suffix[1].endswith(".json")
        ):
            try:
                require_opaque_id(
                    suffix[1][:-5],
                    label="eligibility object id",
                )
            except ValueError as exc:
                raise SourceStorageSecurityError(
                    "eligibility object identity is invalid"
                ) from exc
            return path
        raise SourceStorageSecurityError(
            "path is outside the eligibility writer whitelist"
        )


class EligibilityService:
    """Prove and publish DEFAULT_ELIGIBLE without selector authority."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        repo_root: str | Path,
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._repo_root = Path(repo_root)
        self._writer = _EligibilityWriter(workspace_root)
        self._reader = ExactReferenceReader(workspace_root)
        self._control_reader = ControlStore(workspace_root)

    @staticmethod
    def _root(strategy_id: str) -> PurePosixPath:
        strategy = require_opaque_id(strategy_id, label="strategy_id")
        return (
            FORMAL_RESEARCH_ROOT
            / "strategies"
            / strategy
            / "eligibility"
        )

    def _paths(
        self,
        strategy_id: str,
        intent_id: str,
    ) -> dict[str, PurePosixPath]:
        intent = require_opaque_id(intent_id, label="intent_id")
        root = self._root(strategy_id)
        return {
            "completion": (
                root / "completion_receipts" / f"{intent}.json"
            ),
            "intent": root / "intents" / f"{intent}.json",
            "lock": root / ".active.lock",
            "pointer": root / "_active.json",
        }

    def _load_v4(
        self,
        reference: Mapping[str, Any],
        *,
        expected_version: str,
    ) -> dict[str, Any]:
        try:
            raw = self._reader.read(
                str(reference["relative_path"]),
                str(reference["byte_sha256"]),
            )
            document = load_canonical_resource(
                raw,
                label=expected_version,
            )
            if type(document) is not dict:
                _blocked("ARTIFACT_ROOT")
            validated = validate_artifact(document)
            identity = artifact_identity_field(expected_version)
        except (SourceStorageError, TypeError, ValueError) as exc:
            raise EligibilityError(
                "V17_V4_ELIGIBILITY_BLOCKED:ARTIFACT_VALIDATION"
            ) from exc
        if (
            validated.version != expected_version
            or document.get(identity) != reference.get("artifact_id")
            or document.get("semantic_sha256")
            != reference.get("semantic_sha256")
            or document.get("strategy_id")
            != reference.get("strategy_id")
            or _sha(raw) != reference.get("byte_sha256")
        ):
            _blocked("ARTIFACT_REFERENCE_MISMATCH")
        return dict(document)

    def _validate_bootstrap(
        self,
        reference: Mapping[str, Any],
        *,
        strategy_id: str,
    ) -> None:
        try:
            raw = self._control_reader.read(
                str(reference["relative_path"]),
                str(reference["byte_sha256"]),
            )
            receipt = decode_control_reference(
                raw,
                expected_version=BOOTSTRAP_RECEIPT_VERSION,
            )
            computed_ref = control_artifact_reference(
                relative_path=str(reference["relative_path"]),
                document=receipt,
                raw=raw,
                strategy_id=str(reference["strategy_id"]),
                cutoff=str(reference["cutoff"]),
            )
            selector_raw = self._control_reader.read(SELECTOR_PATH)
            selector = decode_control_reference(
                selector_raw,
                expected_version=SELECTOR_VERSION,
            )
        except (ControlStorageError, OSError, TypeError, ValueError) as exc:
            raise EligibilityError(
                "V17_V4_ELIGIBILITY_BLOCKED:SELECTOR_BOOTSTRAP"
            ) from exc
        if (
            reference.get("artifact_version")
            != BOOTSTRAP_RECEIPT_VERSION
            or computed_ref != dict(reference)
            or reference.get("strategy_id") != strategy_id
            or receipt.get("outcome")
            not in {"BOOTSTRAP_RECOVERED", "BOOTSTRAP_SUCCEEDED"}
            or receipt.get("post_readback_sha256")
            != _sha(selector_raw)
            or selector.get("status") != "V15_DEFAULT"
            or selector.get("protocol_target_ref")
            != receipt.get("proposed_protocol_target_ref")
        ):
            _blocked("SELECTOR_BOOTSTRAP_NOT_LIVE")

    def _revalidate_intent(
        self,
        intent: Mapping[str, Any],
    ) -> None:
        validate_artifact(intent)
        public_documents = [
            self._load_v4(
                reference,
                expected_version=PUBLIC_RECEIPT_VERSION,
            )
            for reference in intent["public_surface_receipt_refs"]
        ]
        if {row["surface"] for row in public_documents} != {
            "CLI",
            "DASHBOARD",
            "SCHEDULE",
            "WEB",
        }:
            _blocked("PUBLIC_SURFACE_INVENTORY")
        live_public = build_public_surface_compatibility_receipts(
            self._repo_root,
            self._workspace_root,
            strategy_id=str(intent["strategy_id"]),
            created_at=str(intent["created_at"]),
        )
        if tuple(public_documents) != live_public:
            _blocked("PUBLIC_SURFACE_READBACK")
        validations = [
            self._load_v4(
                reference,
                expected_version=VALIDATION_RECEIPT_VERSION,
            )
            for reference in intent["validation_receipt_refs"]
        ]
        if (
            {row["validation_kind"] for row in validations}
            != VALIDATION_KINDS
            or any(row["status"] != "PASSED" for row in validations)
        ):
            _blocked("VALIDATION_INVENTORY")
        rollback = self._load_v4(
            intent["rollback_drill_receipt_ref"],
            expected_version=ROLLBACK_DRILL_VERSION,
        )
        if rollback["status"] != "PASSED":
            _blocked("ROLLBACK_DRILL")
        self._validate_bootstrap(
            intent["selector_bootstrap_receipt_ref"],
            strategy_id=str(intent["strategy_id"]),
        )
        public = resolve_public_run(
            self._workspace_root,
            strategy_id=str(intent["strategy_id"]),
            surface="CLI",
        )
        if (
            public["formal_active_pointer_ref"]
            != intent["formal_active_pointer_ref"]
            or public["cutoff"] != intent["cutoff"]
        ):
            _blocked("FORMAL_POINTER_NOT_LIVE")

    def _store_evidence(
        self,
        strategy_id: str,
        folder: str,
        document: Mapping[str, Any],
    ) -> dict[str, str]:
        root = self._root(strategy_id)
        identity = str(
            document[
                artifact_identity_field(str(document["version"]))
            ]
        )
        path = root / folder / f"{identity}.json"
        self._writer.write_exact_once(
            path,
            canonical_resource_bytes(document),
        )
        return artifact_ref(document, relative_path=str(path))

    def qualify(
        self,
        *,
        intent_id: str,
        strategy_id: str,
        created_at: str,
        expected_pointer_sha256: str,
        selector_bootstrap_receipt_ref: Mapping[str, Any],
        public_surface_receipts: Sequence[Mapping[str, Any]],
        validation_receipts: Sequence[Mapping[str, Any]],
        rollback_drill_receipt: Mapping[str, Any],
        crash_after: str | None = None,
    ) -> EligibilityResult:
        strategy = require_opaque_id(strategy_id, label="strategy_id")
        require_utc_timestamp(created_at, label="created_at")
        public_run = resolve_public_run(
            self._workspace_root,
            strategy_id=strategy,
            surface="CLI",
        )
        cutoff = str(public_run["cutoff"])
        if created_at < cutoff:
            _blocked("CREATED_AT")
        if len(public_surface_receipts) != 4:
            _blocked("PUBLIC_SURFACE_INVENTORY")
        if len(validation_receipts) != 5:
            _blocked("VALIDATION_INVENTORY")
        for document in public_surface_receipts:
            validated = validate_artifact(document)
            if (
                validated.version != PUBLIC_RECEIPT_VERSION
                or document.get("strategy_id") != strategy
                or document.get("cutoff") != cutoff
            ):
                _blocked("PUBLIC_SURFACE_RECEIPT")
        for document in validation_receipts:
            validated = validate_artifact(document)
            if (
                validated.version != VALIDATION_RECEIPT_VERSION
                or document.get("strategy_id") != strategy
                or document.get("cutoff") != cutoff
            ):
                _blocked("VALIDATION_RECEIPT")
        rollback_validated = validate_artifact(
            rollback_drill_receipt
        )
        if (
            rollback_validated.version != ROLLBACK_DRILL_VERSION
            or rollback_drill_receipt.get("strategy_id") != strategy
            or rollback_drill_receipt.get("cutoff") != cutoff
        ):
            _blocked("ROLLBACK_DRILL_RECEIPT")
        public_refs = [
            self._store_evidence(
                strategy,
                "public_surface_receipts",
                document,
            )
            for document in public_surface_receipts
        ]
        validation_refs = [
            self._store_evidence(
                strategy,
                "validation_receipts",
                document,
            )
            for document in validation_receipts
        ]
        rollback_ref = self._store_evidence(
            strategy,
            "rollback_drills",
            rollback_drill_receipt,
        )
        explicit = [
            dict(public_run["formal_active_pointer_ref"]),
            dict(selector_bootstrap_receipt_ref),
            rollback_ref,
            *public_refs,
            *validation_refs,
        ]
        intent = seal_semantic(
            {
                "authority": dict(NO_AUTHORITY),
                "created_at": created_at,
                "cutoff": cutoff,
                "evidence_refs": _ordered_refs(explicit),
                "expected_pointer_sha256": expected_pointer_sha256,
                "formal_active_pointer_ref": dict(
                    public_run["formal_active_pointer_ref"]
                ),
                "from_state": "FORMAL_ACTIVE",
                "intent_id": require_opaque_id(
                    intent_id,
                    label="intent_id",
                ),
                "protocol_version": PROTOCOL_VERSION,
                "public_surface_receipt_refs": _ordered_refs(
                    public_refs
                ),
                "rollback_drill_receipt_ref": rollback_ref,
                "selector_bootstrap_receipt_ref": dict(
                    selector_bootstrap_receipt_ref
                ),
                "strategy_id": strategy,
                "to_state": "DEFAULT_ELIGIBLE",
                "validation_receipt_refs": _ordered_refs(
                    validation_refs
                ),
                "version": INTENT_VERSION,
            }
        )
        validate_artifact(intent)
        paths = self._paths(strategy, intent_id)
        with self._writer.locked(paths["lock"]):
            current = self._writer.read_optional(paths["pointer"])
            observed = (
                EMPTY_SHA256
                if current is None
                else current.byte_sha256
            )
            self._revalidate_intent(intent)
            intent_raw = canonical_resource_bytes(intent)
            self._writer.write_exact_once(paths["intent"], intent_raw)
            intent_ref = artifact_ref(
                intent,
                relative_path=str(paths["intent"]),
            )
            pointer = seal_semantic(
                {
                    "authority": dict(NO_AUTHORITY),
                    "cutoff": cutoff,
                    "intent_ref": intent_ref,
                    "pointer_id": f"eligibility-pointer-{intent_id}",
                    "protocol_version": PROTOCOL_VERSION,
                    "state": "PENDING_COMPLETION",
                    "strategy_id": strategy,
                    "updated_at": created_at,
                    "version": POINTER_VERSION,
                }
            )
            validate_artifact(pointer)
            pointer_raw = canonical_resource_bytes(pointer)
            proposed = _sha(pointer_raw)
            if crash_after == "intent":
                raise EligibilityCrash("crash after eligibility intent")
            recovered = False
            if observed == expected_pointer_sha256:
                self._writer.replace_cas(
                    paths["pointer"],
                    expected_pointer_sha256,
                    pointer_raw,
                )
            elif observed == proposed and current is not None:
                if current.data != pointer_raw:
                    _blocked("POINTER_HASH_COLLISION")
                recovered = True
            else:
                _blocked("POINTER_THIRD_STATE")
            if crash_after == "cas":
                raise EligibilityCrash("crash after eligibility CAS")
            if (
                self._writer.read(paths["pointer"], proposed)
                != pointer_raw
            ):
                _blocked("POINTER_READBACK")
            if crash_after == "readback":
                raise EligibilityCrash(
                    "crash after eligibility readback"
                )
            pointer_ref = artifact_ref(
                pointer,
                relative_path=str(paths["pointer"]),
            )
            completion = seal_semantic(
                {
                    "authority": dict(PUBLICATION_AUTHORITY),
                    "cutoff": cutoff,
                    "evidence_refs": _ordered_refs(
                        [intent_ref, pointer_ref]
                    ),
                    "expected_pointer_sha256": (
                        expected_pointer_sha256
                    ),
                    "from_state": "FORMAL_ACTIVE",
                    "intent_ref": intent_ref,
                    "observed_pointer_sha256": (
                        expected_pointer_sha256
                    ),
                    "pointer_ref": pointer_ref,
                    "post_readback_sha256": proposed,
                    "proposed_pointer_sha256": proposed,
                    "protocol_version": PROTOCOL_VERSION,
                    "receipt_id": intent_id,
                    "recorded_at": created_at,
                    "status": "DEFAULT_ELIGIBLE",
                    "strategy_id": strategy,
                    "to_state": "DEFAULT_ELIGIBLE",
                    "version": COMPLETION_VERSION,
                }
            )
            validate_artifact(completion)
            self._writer.write_exact_once(
                paths["completion"],
                canonical_resource_bytes(completion),
            )
            if crash_after == "completion":
                raise EligibilityCrash(
                    "crash after eligibility completion"
                )
            state = self.resolve(strategy)
            if state.status != "DEFAULT_ELIGIBLE":
                _blocked("COMPLETION_READBACK")
            completion_ref = artifact_ref(
                completion,
                relative_path=str(paths["completion"]),
            )
            return EligibilityResult(
                "DEFAULT_ELIGIBLE",
                intent_ref,
                pointer_ref,
                completion_ref,
                recovered,
            )

    def resolve(self, strategy_id: str) -> EligibilityState:
        strategy = require_opaque_id(strategy_id, label="strategy_id")
        root = self._root(strategy)
        stored = self._writer.read_optional(root / "_active.json")
        if stored is None:
            return EligibilityState("FORMAL_ACTIVE", None, None, None)
        try:
            pointer = load_canonical_resource(
                stored.data,
                label="default eligibility pointer",
            )
            if type(pointer) is not dict:
                _blocked("POINTER_ROOT")
            validate_artifact(pointer)
            if pointer.get("strategy_id") != strategy:
                _blocked("POINTER_STRATEGY")
            intent = self._load_v4(
                pointer["intent_ref"],
                expected_version=INTENT_VERSION,
            )
        except (TypeError, ValueError) as exc:
            raise EligibilityError(
                "V17_V4_ELIGIBILITY_BLOCKED:POINTER_CHAIN"
            ) from exc
        paths = self._paths(strategy, str(intent["intent_id"]))
        completion_stored = self._writer.read_optional(
            paths["completion"]
        )
        if completion_stored is None:
            return EligibilityState(
                "PENDING_COMPLETION",
                intent,
                pointer,
                None,
            )
        completion = load_canonical_resource(
            completion_stored.data,
            label="default eligibility completion",
        )
        if type(completion) is not dict:
            _blocked("COMPLETION_ROOT")
        validate_artifact(completion)
        intent_ref = artifact_ref(
            intent,
            relative_path=str(paths["intent"]),
        )
        pointer_ref = artifact_ref(
            pointer,
            relative_path=str(paths["pointer"]),
        )
        if (
            pointer["intent_ref"] != intent_ref
            or completion["intent_ref"] != intent_ref
            or completion["pointer_ref"] != pointer_ref
            or completion["post_readback_sha256"]
            != stored.byte_sha256
            or completion["proposed_pointer_sha256"]
            != stored.byte_sha256
            or completion["expected_pointer_sha256"]
            != intent["expected_pointer_sha256"]
            or completion["observed_pointer_sha256"]
            != intent["expected_pointer_sha256"]
        ):
            _blocked("COMPLETION_POINTER_BINDING")
        self._revalidate_intent(intent)
        return EligibilityState(
            "DEFAULT_ELIGIBLE",
            intent,
            pointer,
            completion,
        )


__all__ = [
    "COMPLETION_VERSION",
    "EligibilityCrash",
    "EligibilityError",
    "EligibilityResult",
    "EligibilityService",
    "EligibilityState",
    "INTENT_VERSION",
    "POINTER_VERSION",
    "ROLLBACK_DRILL_VERSION",
    "VALIDATION_KINDS",
    "VALIDATION_RECEIPT_VERSION",
    "build_rollback_drill_receipt",
    "build_validation_receipt",
]

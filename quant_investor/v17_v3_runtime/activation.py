"""Typed formal-research activation, revocation, and current-result reads."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
from typing import Any, Iterator, Mapping

from quant_investor.v17_v3_contract.identities import (
    IdentityContractError,
    require_path_id,
    require_sha256,
    require_utc_cutoff,
)
from quant_investor.v17_v3_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    load_canonical_resource,
)
from quant_investor.v17_v3_contract.namespace import derive_namespace_path
from quant_investor.v17_v3_contract.resources import (
    PackageResourceError,
    load_packaged_json,
    package_resource_session,
    verify_runtime_build,
)

from .artifacts import (
    RuntimeArtifact,
    load_typed_artifact,
    replace_typed_cas,
    runtime_artifact,
    seal_typed_artifact,
    write_typed_exact_once,
)
from .authority import PROTOCOL_VERSION, authority_envelope
from .redaction import assert_public_envelope_safe
from .storage import (
    EMPTY_SHA256,
    FORMAL_RESULTS_ROOT,
    SecureStore,
    StorageError,
)

ACTIVE = "ACTIVE"
REVOKED = "REVOKED"
ACTIVATION_REJECTED = "ACTIVATION_REJECTED"
IMMUTABLE_UNPUBLISHED_EVIDENCE = "IMMUTABLE_UNPUBLISHED_EVIDENCE"
NO_CURRENT_ACTIVE_FORMAL_RESULT = "NO_CURRENT_ACTIVE_FORMAL_RESULT"
FORMAL_RESULT_PUBLISHED = "FORMAL_RESULT_PUBLISHED"

FORMAL_OUTPUT_VERSION = "myquant.v17.v3.formal-research-output.v1"
PROMOTION_VERSION = "myquant.v17.v3.fusion-promotion-receipt.v1"
RECEIPT_VERSION = "myquant.v17.v3.activation-receipt.v1"
POINTER_VERSION = "myquant.v17.v3.activation-pointer.v1"
LATEST_VERSION = "myquant.v17.v3.formal-latest.v1"
UNPUBLISHED_VERSION = "myquant.v17.v3.unpublished-evidence.v1"


class ActivationError(ValueError):
    """A formal-research lifecycle operation failed closed."""

    exit_code = 2


@dataclass(frozen=True)
class PublicationOutcome:
    status: str
    strategy_id: str
    cutoff: str
    receipt_sha256: str | None = None
    core_sha256: str | None = None
    latest_sha256: str | None = None
    write_count: int = 0
    idempotent: bool = False

    def to_public_wire(self) -> dict[str, Any]:
        payload = {
            "version": f"{PROTOCOL_VERSION}.publication-outcome.v1",
            "status": self.status,
            "strategy_id": self.strategy_id,
            "cutoff": self.cutoff,
            "receipt_sha256": self.receipt_sha256,
            "core_sha256": self.core_sha256,
            "latest_sha256": self.latest_sha256,
            "write_count": self.write_count,
            "idempotent": self.idempotent,
            **authority_envelope(
                formal_research_active=self.status in {ACTIVE, FORMAL_RESULT_PUBLISHED}
            ),
        }
        assert_public_envelope_safe(payload)
        return payload


@dataclass(frozen=True)
class CurrentFormalResult:
    status: str
    strategy_id: str
    cutoff: str | None
    active_receipt_sha256: str | None
    core_sha256: str | None
    core_bytes: bytes | None

    def to_public_wire(self) -> dict[str, Any]:
        payload = {
            "version": f"{PROTOCOL_VERSION}.current-formal-result.v1",
            "status": self.status,
            "strategy_id": self.strategy_id,
            "cutoff": self.cutoff,
            "active_receipt_sha256": self.active_receipt_sha256,
            "core_sha256": self.core_sha256,
            "result_available": self.core_bytes is not None,
            **authority_envelope(formal_research_active=self.status == ACTIVE),
        }
        assert_public_envelope_safe(payload)
        return payload


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strategy(value: Any) -> str:
    try:
        return require_path_id(value, label="strategy_id")
    except IdentityContractError as exc:
        raise ActivationError(str(exc)) from exc


def _cutoff(value: Any, *, label: str = "cutoff") -> str:
    try:
        return require_utc_cutoff(value, label=label)
    except IdentityContractError as exc:
        raise ActivationError(str(exc)) from exc


def _expected_sha(value: Any, *, label: str) -> str:
    try:
        return require_sha256(value, label=label)
    except IdentityContractError as exc:
        raise ActivationError(str(exc)) from exc


def _cutoff_key(value: str) -> str:
    return _cutoff(value).replace("-", "").replace(":", "").replace("T", "t").replace("Z", "z")


@dataclass
class ActivationPublisher:
    """Serialize formal mutation by formal-latest then strategy+cutoff lock."""

    store: SecureStore

    def __post_init__(self) -> None:
        if not isinstance(self.store, SecureStore):
            raise TypeError("store must be SecureStore")

    @staticmethod
    def _strategy_root(strategy_id: str) -> PurePosixPath:
        return FORMAL_RESULTS_ROOT / "strategies" / strategy_id

    def _latest_path(self, strategy_id: str) -> PurePosixPath:
        return derive_namespace_path("FORMAL_LATEST", strategy_id=strategy_id)

    def _latest_lock(self, strategy_id: str) -> PurePosixPath:
        return self._strategy_root(strategy_id) / ".formal-latest.lock"

    def _cutoff_root(self, strategy_id: str, cutoff: str) -> PurePosixPath:
        return self._strategy_root(strategy_id) / "activations" / _cutoff_key(cutoff)

    def _activation_lock(self, strategy_id: str, cutoff: str) -> PurePosixPath:
        return self._cutoff_root(strategy_id, cutoff) / ".activation.lock"

    def _activation_pointer(self, strategy_id: str, cutoff: str) -> PurePosixPath:
        return derive_namespace_path(
            "FORMAL_ACTIVATION_POINTER",
            strategy_id=strategy_id,
            cutoff_id=_cutoff_key(cutoff),
        )

    def _receipt_path(
        self,
        strategy_id: str,
        cutoff: str,
        status: str,
    ) -> PurePosixPath:
        return derive_namespace_path(
            "FORMAL_ACTIVATION_RECEIPT",
            strategy_id=strategy_id,
            cutoff_id=_cutoff_key(cutoff),
            status=status.casefold(),
        )

    def _unpublished_path(
        self,
        strategy_id: str,
        digest: str,
    ) -> PurePosixPath:
        return derive_namespace_path(
            "FORMAL_UNPUBLISHED_EVIDENCE",
            strategy_id=strategy_id,
            byte_sha256=digest,
        )

    def _receipt_artifact(
        self,
        *,
        strategy_id: str,
        cutoff: str,
        document: Mapping[str, Any],
    ) -> RuntimeArtifact:
        return runtime_artifact(
            relative_path=self._receipt_path(
                strategy_id,
                cutoff,
                str(document["status"]),
            ),
            document=document,
        )

    @contextmanager
    def _formal_transaction(
        self,
        strategy_id: str,
        cutoff: str,
    ) -> Iterator[None]:
        with self.store.locked(self._latest_lock(strategy_id)):
            with self.store.locked(self._activation_lock(strategy_id, cutoff)):
                yield

    def _read_optional(
        self,
        path: PurePosixPath,
        *,
        label: str,
        expected_version: str,
    ) -> tuple[bytes, dict[str, Any]] | None:
        observed = self.store.read_optional(path)
        if observed is None:
            return None
        return (
            observed.data,
            load_typed_artifact(
                observed.data,
                label=label,
                expected_version=expected_version,
            ),
        )

    def _latest_optional(
        self,
        strategy_id: str,
    ) -> tuple[bytes, dict[str, Any]] | None:
        return self._read_optional(
            self._latest_path(strategy_id),
            label="formal latest",
            expected_version=LATEST_VERSION,
        )

    def _pointer_optional(
        self,
        strategy_id: str,
        cutoff: str,
    ) -> tuple[bytes, dict[str, Any]] | None:
        return self._read_optional(
            self._activation_pointer(strategy_id, cutoff),
            label="activation pointer",
            expected_version=POINTER_VERSION,
        )

    def _input_artifact(
        self,
        *,
        raw: bytes,
        path: str | PurePosixPath,
        expected_sha256: str,
        expected_version: str,
        strategy_id: str,
        cutoff: str,
        label: str,
    ) -> RuntimeArtifact:
        expected = _expected_sha(expected_sha256, label=f"{label} SHA-256")
        relative_path = self.store.relative_from_path(str(path))
        stored = self.store.read(relative_path, expected)
        if type(raw) is not bytes or stored != raw:
            raise ActivationError(f"{label} exact bytes mismatch")
        document = load_typed_artifact(
            raw,
            label=label,
            expected_version=expected_version,
        )
        if document.get("strategy_id") != strategy_id or document.get("cutoff") != cutoff:
            raise ActivationError(f"{label} strategy/cutoff mismatch")
        artifact = runtime_artifact(
            relative_path=relative_path,
            document=document,
        )
        if artifact.byte_sha256 != expected:
            raise ActivationError(f"{label} exact SHA-256 mismatch")
        return artifact

    def _read_ref(
        self,
        reference: Any,
        *,
        expected_version: str,
        label: str,
    ) -> RuntimeArtifact:
        if not isinstance(reference, Mapping):
            raise ActivationError(f"{label} is not an artifact reference")
        path = reference.get("relative_path")
        digest = reference.get("byte_sha256")
        if type(path) is not str or type(digest) is not str:
            raise ActivationError(f"{label} reference is incomplete")
        raw = self.store.read(path, digest)
        document = load_typed_artifact(
            raw,
            label=label,
            expected_version=expected_version,
        )
        artifact = runtime_artifact(relative_path=path, document=document)
        if artifact.reference != dict(reference):
            raise ActivationError(f"{label} exact reference mismatch")
        return artifact

    def _read_ref_cached(
        self,
        reference: Any,
        *,
        expected_version: str,
        label: str,
        resolved: dict[tuple[str, str], RuntimeArtifact],
    ) -> RuntimeArtifact:
        if not isinstance(reference, Mapping):
            raise ActivationError(f"{label} is not an artifact reference")
        path = reference.get("relative_path")
        digest = reference.get("byte_sha256")
        if type(path) is not str or type(digest) is not str:
            raise ActivationError(f"{label} reference is incomplete")
        key = (path, digest)
        artifact = resolved.get(key)
        if artifact is None:
            artifact = self._read_ref(
                reference,
                expected_version=expected_version,
                label=label,
            )
            resolved[key] = artifact
        elif (
            artifact.reference != dict(reference)
            or artifact.document.get("version") != expected_version
        ):
            raise ActivationError(f"{label} cached exact reference mismatch")
        return artifact

    def _validate_calibration_history(
        self,
        calibration_input: RuntimeArtifact,
        *,
        resolved: dict[tuple[str, str], RuntimeArtifact],
    ) -> None:
        payload = calibration_input.document.get("payload")
        if not isinstance(payload, Mapping):
            raise ActivationError("fusion calibration input payload is invalid")
        months = payload.get("months")
        if not isinstance(months, list) or not months:
            raise ActivationError("fusion calibration month inventory is empty")
        for index, month in enumerate(months):
            if not isinstance(month, Mapping):
                raise ActivationError("fusion calibration month is invalid")
            ordered_pool = month.get("ordered_pool")
            origin = month.get("origin")
            if not isinstance(ordered_pool, list) or type(origin) is not str:
                raise ActivationError("fusion calibration month binding is invalid")
            branches: dict[str, RuntimeArtifact] = {}
            for field, expected_branch in (
                ("quant_branch_ref", "quant"),
                ("fundamental_branch_ref", "fundamental"),
            ):
                branch = self._read_ref_cached(
                    month.get(field),
                    expected_version="myquant.v17.v3.branch-output.v1",
                    label=f"calibration month {index} {expected_branch} branch",
                    resolved=resolved,
                )
                if (
                    branch.document.get("branch") != expected_branch
                    or branch.document.get("strategy_id")
                    != calibration_input.document.get("strategy_id")
                    or str(branch.document.get("cutoff", ""))[:10] != origin
                    or branch.document.get("ordered_domain") != ordered_pool
                ):
                    raise ActivationError("historical calibration branch binding mismatch")
                branches[expected_branch] = branch

            quant = branches["quant"].document
            fundamental = branches["fundamental"].document
            if quant.get("initial_pool_ref") != fundamental.get("initial_pool_ref") or quant.get(
                "source_locator_ref"
            ) != fundamental.get("source_locator_ref"):
                raise ActivationError(
                    "historical calibration branches do not bind the same exact pool/locator"
                )
            pool = self._read_ref_cached(
                quant.get("initial_pool_ref"),
                expected_version="myquant.v17.v3.initial-pool-output.v1",
                label=f"calibration month {index} initial pool",
                resolved=resolved,
            )
            locator = self._read_ref_cached(
                quant.get("source_locator_ref"),
                expected_version="myquant.v17.v3.source-locator.v1",
                label=f"calibration month {index} preselection locator",
                resolved=resolved,
            )
            expected_order_sha256 = hashlib.sha256(canonical_bytes(ordered_pool)).hexdigest()
            if (
                pool.document.get("strategy_id") != calibration_input.document.get("strategy_id")
                or str(pool.document.get("cutoff", ""))[:10] != origin
                or pool.document.get("selected_symbols") != ordered_pool
                or pool.document.get("pool_count") != len(ordered_pool)
                or pool.document.get("pool_symbol_order_sha256") != expected_order_sha256
                or pool.document.get("source_locator_ref") != locator.reference
                or quant.get("initial_pool_count") != len(ordered_pool)
                or fundamental.get("initial_pool_count") != len(ordered_pool)
                or quant.get("initial_pool_symbol_order_sha256") != expected_order_sha256
                or fundamental.get("initial_pool_symbol_order_sha256") != expected_order_sha256
            ):
                raise ActivationError("historical calibration pool/branch exact binding mismatch")
            manifest = self._read_ref_cached(
                locator.document.get("source_manifest_ref"),
                expected_version="myquant.v17.v3.source-manifest.v1",
                label=f"calibration month {index} PRESELECT manifest",
                resolved=resolved,
            )
            if (
                manifest.document.get("phase") != "PRESELECT"
                or manifest.document.get("closure_kind") != "DERIVED_CLOSURE"
            ):
                raise ActivationError("historical calibration branch locator is not PRESELECT")
            raw_manifest = self._read_ref_cached(
                manifest.document.get("parent_raw_manifest_ref"),
                expected_version="myquant.v17.v3.source-manifest.v1",
                label=f"calibration month {index} raw source manifest",
                resolved=resolved,
            )
            if (
                raw_manifest.document.get("closure_kind") != "RAW"
                or pool.document.get("raw_source_manifest_ref") != raw_manifest.reference
            ):
                raise ActivationError("historical calibration pool raw source binding mismatch")

    def _resolve_receipt_evidence(
        self,
        reference: Any,
        *,
        label: str,
        resolved: dict[tuple[str, str], RuntimeArtifact],
        validated_history: set[tuple[str, str]],
    ) -> tuple[RuntimeArtifact, str | None]:
        if not isinstance(reference, Mapping) or type(reference.get("artifact_version")) is not str:
            raise ActivationError(f"{label} reference is invalid")
        version = str(reference["artifact_version"])
        artifact = self._read_ref_cached(
            reference,
            expected_version=version,
            label=label,
            resolved=resolved,
        )
        if version == "myquant.v17.v3.fusion-calibration-inputs.v1":
            key = (str(artifact.relative_path), artifact.byte_sha256)
            if key not in validated_history:
                self._validate_calibration_history(
                    artifact,
                    resolved=resolved,
                )
                validated_history.add(key)
            return artifact, None
        elif version == "myquant.v17.v3.source-locator.v1":
            from .sources import admit_source_locator

            admission = admit_source_locator(
                self.store,
                locator_path=str(artifact.relative_path),
                expected_locator_sha256=artifact.byte_sha256,
            )
            phase = admission.documents["source_manifest"].get("phase")
            if type(phase) is not str:
                raise ActivationError(f"{label} locator phase is invalid")
            return artifact, phase
        return artifact, None

    def _rejection_documents(
        self,
        *,
        strategy_id: str,
        cutoff: str,
        event_at: str,
        promotion: RuntimeArtifact,
    ) -> RuntimeArtifact:
        reasons = promotion.document.get("rejection_reasons")
        if not isinstance(reasons, list) or not reasons:
            raise ActivationError("rejected promotion has no ordered reasons")
        receipt = seal_typed_artifact(
            {
                "version": RECEIPT_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "receipt_id": f"rejected-{promotion.byte_sha256[:24]}",
                "strategy_id": strategy_id,
                "cutoff": cutoff,
                "status": ACTIVATION_REJECTED,
                "rejected_at": event_at,
                "rejection_reasons": sorted(set(reasons)),
                "promotion_receipt_ref": promotion.reference,
                "authority": authority_envelope(),
            }
        )
        return self._receipt_artifact(
            strategy_id=strategy_id,
            cutoff=cutoff,
            document=receipt,
        )

    @package_resource_session()
    def _validate_promotion_closure(
        self,
        promotion: RuntimeArtifact,
    ) -> dict[tuple[str, str], RuntimeArtifact]:
        try:
            verify_runtime_build()
        except PackageResourceError as exc:
            raise ActivationError("v3 runtime build identity mismatch") from exc
        resolved: dict[tuple[str, str], RuntimeArtifact] = {}
        validated_history: set[tuple[str, str]] = set()
        receipts: list[RuntimeArtifact] = []
        for index, reference in enumerate(promotion.document.get("calibration_receipt_refs", ())):
            receipts.append(
                self._read_ref_cached(
                    reference,
                    expected_version=("myquant.v17.v3.fusion-calibration-receipt.v1"),
                    label=f"promotion calibration receipt {index}",
                    resolved=resolved,
                )
            )
        kinds = [receipt.document.get("calibration_kind") for receipt in receipts]
        if kinds != [
            "QUANT_TIMING",
            "FUNDAMENTAL_FORWARD",
            "FUSION_PROMOTION",
        ]:
            raise ActivationError("promotion calibration receipt order is invalid")
        all_accepted = all(receipt.document.get("accepted") is True for receipt in receipts)
        if (promotion.document.get("status") == "PROMOTED" and not all_accepted) or (
            promotion.document.get("status") == "PROMOTION_REJECTED" and all_accepted
        ):
            raise ActivationError("promotion decision disagrees with calibration receipts")
        for receipt_index, receipt in enumerate(receipts):
            evidence_refs = receipt.document.get("evidence_refs")
            if not isinstance(evidence_refs, list) or not evidence_refs:
                raise ActivationError("calibration receipt evidence closure is empty")
            evidence_versions: list[str] = []
            evidence_phases: list[str] = []
            for evidence_index, reference in enumerate(evidence_refs):
                evidence, phase = self._resolve_receipt_evidence(
                    reference,
                    label=(f"calibration receipt {receipt_index} evidence " f"{evidence_index}"),
                    resolved=resolved,
                    validated_history=validated_history,
                )
                evidence_versions.append(str(evidence.document["version"]))
                if phase is not None:
                    evidence_phases.append(phase)
            expected_phase = {
                "QUANT_TIMING": "QUANT_TIMING_CALIBRATION",
                "FUNDAMENTAL_FORWARD": "FUNDAMENTAL_FORWARD_CALIBRATION",
                "FUSION_PROMOTION": "FUSION_PROMOTION",
            }[str(receipt.document["calibration_kind"])]
            if evidence_phases != [expected_phase]:
                raise ActivationError("calibration receipt does not bind its exact phase locator")
            fusion_input_count = evidence_versions.count(
                "myquant.v17.v3.fusion-calibration-inputs.v1"
            )
            if (
                expected_phase == "FUSION_PROMOTION"
                and (fusion_input_count != 1 or len(evidence_versions) != 2)
            ) or (
                expected_phase != "FUSION_PROMOTION"
                and (fusion_input_count != 0 or len(evidence_versions) != 1)
            ):
                raise ActivationError("calibration receipt evidence profile is invalid")
        for index, reference in enumerate(promotion.document.get("evidence_refs", ())):
            self._resolve_receipt_evidence(
                reference,
                label=f"promotion evidence {index}",
                resolved=resolved,
                validated_history=validated_history,
            )
        return resolved

    def _validate_activation_closure(
        self,
        *,
        strategy_id: str,
        cutoff: str,
        promotion: RuntimeArtifact,
        formal_output: RuntimeArtifact,
        resolved: dict[tuple[str, str], RuntimeArtifact],
    ) -> None:
        """Resolve every authority-bearing input reference before any write."""

        resolved_by_version: dict[str, RuntimeArtifact] = {}
        resolved_artifacts: list[RuntimeArtifact] = []
        for index, reference in enumerate(formal_output.document.get("artifact_refs", ())):
            if (
                not isinstance(reference, Mapping)
                or type(reference.get("artifact_version")) is not str
            ):
                raise ActivationError("formal artifact reference is invalid")
            artifact = self._read_ref_cached(
                reference,
                expected_version=str(reference["artifact_version"]),
                label=f"formal artifact {index}",
                resolved=resolved,
            )
            if (
                artifact.document.get("strategy_id") != strategy_id
                or artifact.document.get("cutoff") != cutoff
            ):
                raise ActivationError("formal artifact strategy/cutoff drift")
            resolved_by_version[str(artifact.document["version"])] = artifact
            resolved_artifacts.append(artifact)

        locator = resolved_by_version.get("myquant.v17.v3.source-locator.v1")
        fusion = resolved_by_version.get("myquant.v17.v3.fusion-output.v1")
        portfolio = resolved_by_version.get("myquant.v17.v3.portfolio-output.v1")
        if locator is None or fusion is None:
            raise ActivationError("formal output closure lacks analyze locator or fusion output")
        if formal_output.document.get("factor_baseline_mode") != "FACTOR_V4_PRODUCTION":
            raise ActivationError("activation_rejects_provisional_factor_baseline")
        if formal_output.document.get("portfolio_basis") == "MODEL_ONLY_NO_PRIVATE_HOLDINGS":
            raise ActivationError("activation_rejects_model_only_portfolio")
        readiness = self._read_ref_cached(
            formal_output.document.get("factor_baseline_ref"),
            expected_version=("myquant.v17.v3.factor-governance-readiness.v1"),
            label="formal factor-v4 readiness",
            resolved=resolved,
        )
        if (
            readiness.document.get("strategy_id") != strategy_id
            or readiness.document.get("cutoff") != cutoff
            or readiness.document.get("readiness_status") != "FACTOR_V4_READY"
            or readiness.document.get("factor_governance_ready") is not True
            or readiness.document.get("activation_receipt_valid") is not True
            or not isinstance(
                readiness.document.get("production_factor_count"),
                int,
            )
            or readiness.document["production_factor_count"] < 5
            or not isinstance(
                readiness.document.get("production_family_count"),
                int,
            )
            or readiness.document["production_family_count"] < 3
        ):
            raise ActivationError("formal factor-v4 readiness gate is not satisfied")
        from .sources import admit_source_locator

        admission = admit_source_locator(
            self.store,
            locator_path=str(locator.relative_path),
            expected_locator_sha256=locator.byte_sha256,
        )
        if (
            fusion.document.get("promotion_receipt_ref") != promotion.reference
            or fusion.document.get("calibration_receipt_refs")
            != promotion.document.get("calibration_receipt_refs")
            or admission.reference_for_role("fusion_promotion_receipt") != promotion.reference
        ):
            raise ActivationError("formal fusion output does not bind the exact promotion")
        evidence_artifacts = [
            artifact
            for artifact in resolved_artifacts
            if artifact.document["version"]
            in {
                "myquant.v17.v3.initial-pool-output.v1",
                "myquant.v17.v3.branch-output.v1",
                "myquant.v17.v3.fusion-calibration-inputs.v1",
            }
        ]
        evidence_refs = sorted(
            (artifact.reference for artifact in evidence_artifacts),
            key=lambda reference: (
                reference["relative_path"],
                reference["byte_sha256"],
            ),
        )
        if (
            len(evidence_artifacts) != 4
            or sum(
                artifact.document["version"] == "myquant.v17.v3.branch-output.v1"
                for artifact in evidence_artifacts
            )
            != 2
            or promotion.document.get("evidence_refs") != evidence_refs
        ):
            raise ActivationError("promotion evidence does not exactly cover analyzed inputs")
        pool = next(
            artifact
            for artifact in evidence_artifacts
            if artifact.document["version"] == "myquant.v17.v3.initial-pool-output.v1"
        )
        branches = [
            artifact
            for artifact in evidence_artifacts
            if artifact.document["version"] == "myquant.v17.v3.branch-output.v1"
        ]
        branch_policies = {
            artifact.document["branch"]: artifact.document["policy_sha256"] for artifact in branches
        }
        if (
            pool.document["policy_sha256"] != promotion.document["preselector_policy_sha256"]
            or branch_policies.get("quant") != promotion.document["quant_branch_policy_sha256"]
            or branch_policies.get("fundamental")
            != promotion.document["fundamental_branch_policy_sha256"]
        ):
            raise ActivationError("promotion policy evidence binding mismatch")
        if portfolio is not None:
            if portfolio.document.get("factor_baseline_mode") == "PROVISIONAL_RESEARCH":
                raise ActivationError("activation_rejects_provisional_factor_baseline")
            if portfolio.document.get("portfolio_basis") == "MODEL_ONLY_NO_PRIVATE_HOLDINGS":
                raise ActivationError("activation_rejects_model_only_portfolio")
            allocation_policy = load_packaged_json("resources/portfolio_allocation_policy.v1.json")
            if (
                portfolio.document.get("factor_baseline_mode")
                != formal_output.document.get("factor_baseline_mode")
                or portfolio.document.get("factor_baseline_ref") != readiness.reference
                or portfolio.document.get("portfolio_basis")
                != formal_output.document.get("portfolio_basis")
                or portfolio.document.get("allocation_policy_sha256")
                != allocation_policy.get("semantic_sha256")
            ):
                raise ActivationError("formal portfolio baseline/policy binding drift")
        state = formal_output.document.get("terminal_state")
        if state == "FORMAL_RESEARCH_COMPLETE":
            if (
                portfolio is None
                or portfolio.document.get("status") != "COMPLETE"
                or formal_output.document.get("portfolio_output_ref") != portfolio.reference
            ):
                raise ActivationError("formal COMPLETE lacks complete portfolio")
            holdings_ref = portfolio.document.get("holdings_snapshot_ref")
            if not isinstance(holdings_ref, Mapping) or dict(
                admission.reference_for_role("holdings_snapshot")
            ) != dict(holdings_ref):
                raise ActivationError("formal holdings snapshot binding drift")
        elif state == "FORMAL_PORTFOLIO_INFEASIBLE":
            if (
                portfolio is None
                or portfolio.document.get("status") != "INFEASIBLE"
                or formal_output.document.get("portfolio_output_ref") != portfolio.reference
            ):
                raise ActivationError("formal INFEASIBLE lacks its typed portfolio")
        elif (
            portfolio is not None or formal_output.document.get("portfolio_output_ref") is not None
        ):
            raise ActivationError("rank-only formal output carries a portfolio artifact")

    def _active_documents(
        self,
        *,
        strategy_id: str,
        cutoff: str,
        event_at: str,
        promotion: RuntimeArtifact,
        formal_output: RuntimeArtifact,
    ) -> tuple[RuntimeArtifact, RuntimeArtifact, RuntimeArtifact]:
        receipt = self._receipt_artifact(
            strategy_id=strategy_id,
            cutoff=cutoff,
            document=seal_typed_artifact(
                {
                    "version": RECEIPT_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "receipt_id": (
                        f"active-{formal_output.byte_sha256[:12]}-" f"{promotion.byte_sha256[:12]}"
                    ),
                    "strategy_id": strategy_id,
                    "cutoff": cutoff,
                    "status": ACTIVE,
                    "activated_at": event_at,
                    "formal_output_ref": formal_output.reference,
                    "promotion_receipt_ref": promotion.reference,
                    "authority": authority_envelope(formal_research_active=True),
                }
            ),
        )
        pointer = runtime_artifact(
            relative_path=self._activation_pointer(strategy_id, cutoff),
            document=seal_typed_artifact(
                {
                    "version": POINTER_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "pointer_id": f"activation-{_cutoff_key(cutoff)}",
                    "strategy_id": strategy_id,
                    "cutoff": cutoff,
                    "status": ACTIVE,
                    "updated_at": event_at,
                    "active_receipt_ref": receipt.reference,
                    "formal_output_ref": formal_output.reference,
                    "authority": authority_envelope(formal_research_active=True),
                }
            ),
        )
        latest = runtime_artifact(
            relative_path=self._latest_path(strategy_id),
            document=seal_typed_artifact(
                {
                    "version": LATEST_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "latest_id": f"formal-latest-{strategy_id}",
                    "strategy_id": strategy_id,
                    "cutoff": cutoff,
                    "status": ACTIVE,
                    "updated_at": event_at,
                    "activation_pointer_ref": pointer.reference,
                    "active_receipt_ref": receipt.reference,
                    "formal_output_ref": formal_output.reference,
                    "authority": authority_envelope(formal_research_active=True),
                }
            ),
        )
        return receipt, pointer, latest

    def _write_unpublished(
        self,
        *,
        strategy_id: str,
        cutoff: str,
        event_at: str,
        formal_output: RuntimeArtifact,
    ) -> tuple[str, int]:
        evidence = seal_typed_artifact(
            {
                "version": UNPUBLISHED_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "evidence_id": f"unpublished-{formal_output.byte_sha256[:24]}",
                "strategy_id": strategy_id,
                "cutoff": cutoff,
                "created_at": event_at,
                "terminal_state": IMMUTABLE_UNPUBLISHED_EVIDENCE,
                "attempted_action": "ACTIVATE_FORMAL_RESEARCH",
                "failure_reasons": ["latest_cas_mismatch"],
                "artifact_ref": formal_output.reference,
                "authority": authority_envelope(),
            }
        )
        evidence_raw = canonical_resource_bytes(evidence)
        raw = runtime_artifact(
            relative_path=self._unpublished_path(
                strategy_id,
                _sha(evidence_raw),
            ),
            document=evidence,
        )
        result = write_typed_exact_once(self.store, raw)
        return formal_output.byte_sha256, int(result.created)

    def activate(
        self,
        *,
        strategy_id: str,
        cutoff: str,
        promotion_receipt_bytes: bytes,
        promotion_receipt_path: str | PurePosixPath,
        expected_promotion_receipt_sha256: str,
        formal_output_bytes: bytes | None = None,
        formal_output_path: str | PurePosixPath | None = None,
        expected_formal_output_sha256: str | None = None,
        event_at: str | None = None,
        expected_latest_sha256: str | None = None,
    ) -> PublicationOutcome:
        """Activate only a typed promotion/output pair, or record typed rejection."""

        strategy_id = _strategy(strategy_id)
        cutoff = _cutoff(cutoff)
        event_at = _cutoff(cutoff if event_at is None else event_at, label="event_at")
        promotion = self._input_artifact(
            raw=promotion_receipt_bytes,
            path=promotion_receipt_path,
            expected_sha256=expected_promotion_receipt_sha256,
            expected_version=PROMOTION_VERSION,
            strategy_id=strategy_id,
            cutoff=cutoff,
            label="fusion promotion receipt",
        )
        is_promoted = (
            promotion.document.get("status") == "PROMOTED"
            and promotion.document.get("accepted") is True
        )
        is_rejected = (
            promotion.document.get("status") == "PROMOTION_REJECTED"
            and promotion.document.get("accepted") is False
        )
        if not is_promoted and not is_rejected:
            raise ActivationError("promotion receipt lifecycle status is invalid")
        resolved = self._validate_promotion_closure(promotion)

        formal_output: RuntimeArtifact | None = None
        active_documents: (
            tuple[
                RuntimeArtifact,
                RuntimeArtifact,
                RuntimeArtifact,
            ]
            | None
        ) = None
        rejection: RuntimeArtifact | None = None
        if is_promoted:
            if (
                formal_output_bytes is None
                or formal_output_path is None
                or expected_formal_output_sha256 is None
            ):
                raise ActivationError("PROMOTED activation requires exact formal output")
            preflight = load_canonical_resource(
                formal_output_bytes,
                label="formal research output preflight",
            )
            if isinstance(preflight, Mapping):
                if preflight.get("factor_baseline_mode") == "PROVISIONAL_RESEARCH":
                    raise ActivationError("activation_rejects_provisional_factor_baseline")
                if preflight.get("portfolio_basis") == "MODEL_ONLY_NO_PRIVATE_HOLDINGS":
                    raise ActivationError("activation_rejects_model_only_portfolio")
            formal_output = self._input_artifact(
                raw=formal_output_bytes,
                path=formal_output_path,
                expected_sha256=expected_formal_output_sha256,
                expected_version=FORMAL_OUTPUT_VERSION,
                strategy_id=strategy_id,
                cutoff=cutoff,
                label="formal research output",
            )
            if formal_output.document.get("terminal_state") not in {
                "FORMAL_RESEARCH_COMPLETE",
                "FORMAL_RANK_COMPLETE_NO_PORTFOLIO",
                "FORMAL_PORTFOLIO_INFEASIBLE",
            }:
                raise ActivationError("hard-stop formal output cannot be activated")
            if formal_output.document.get("authority") != authority_envelope():
                raise ActivationError("pre-activation formal output must have no authority")
            self._validate_activation_closure(
                strategy_id=strategy_id,
                cutoff=cutoff,
                promotion=promotion,
                formal_output=formal_output,
                resolved=resolved,
            )
            active_documents = self._active_documents(
                strategy_id=strategy_id,
                cutoff=cutoff,
                event_at=event_at,
                promotion=promotion,
                formal_output=formal_output,
            )
        else:
            rejection = self._rejection_documents(
                strategy_id=strategy_id,
                cutoff=cutoff,
                event_at=event_at,
                promotion=promotion,
            )

        expected_latest: str | None = None
        if expected_latest_sha256 is not None:
            expected_latest = (
                EMPTY_SHA256
                if expected_latest_sha256 == EMPTY_SHA256
                else _expected_sha(
                    expected_latest_sha256,
                    label="formal latest SHA-256",
                )
            )

        with self._formal_transaction(strategy_id, cutoff):
            latest = self._latest_optional(strategy_id)
            pointer = self._pointer_optional(strategy_id, cutoff)
            rejected = self._read_optional(
                self._receipt_path(
                    strategy_id,
                    cutoff,
                    ACTIVATION_REJECTED,
                ),
                label="ACTIVATION_REJECTED receipt",
                expected_version=RECEIPT_VERSION,
            )
            observed_latest = EMPTY_SHA256 if latest is None else _sha(latest[0])
            if latest is not None and str(latest[1]["cutoff"]) > cutoff:
                return PublicationOutcome(
                    "OLDER_CUTOFF_REJECTED",
                    strategy_id,
                    cutoff,
                    latest_sha256=observed_latest,
                    write_count=0,
                )
            if rejected is not None or (pointer is not None and pointer[1]["status"] == REVOKED):
                return PublicationOutcome(
                    "SAME_CUTOFF_TERMINAL_CONFLICT",
                    strategy_id,
                    cutoff,
                    latest_sha256=observed_latest,
                    write_count=0,
                )
            if pointer is not None:
                if not is_promoted or active_documents is None or formal_output is None:
                    return PublicationOutcome(
                        "SAME_CUTOFF_CORE_CONFLICT",
                        strategy_id,
                        cutoff,
                        write_count=0,
                    )
                receipt, expected_pointer, expected_latest_artifact = active_documents
                if pointer[0] != expected_pointer.raw:
                    return PublicationOutcome(
                        "SAME_CUTOFF_CORE_CONFLICT",
                        strategy_id,
                        cutoff,
                        write_count=0,
                    )
                if latest is None or latest[0] != expected_latest_artifact.raw:
                    latest_result = replace_typed_cas(
                        self.store,
                        expected_latest_artifact,
                        expected_sha256=observed_latest,
                    )
                    return PublicationOutcome(
                        ACTIVE,
                        strategy_id,
                        cutoff,
                        receipt_sha256=receipt.byte_sha256,
                        core_sha256=formal_output.byte_sha256,
                        latest_sha256=latest_result.byte_sha256,
                        write_count=int(latest_result.created or latest_result.replaced),
                        idempotent=True,
                    )
                return PublicationOutcome(
                    ACTIVE,
                    strategy_id,
                    cutoff,
                    receipt_sha256=receipt.byte_sha256,
                    core_sha256=formal_output.byte_sha256,
                    latest_sha256=observed_latest,
                    write_count=0,
                    idempotent=True,
                )
            if expected_latest is not None and expected_latest != observed_latest:
                if formal_output is None:
                    return PublicationOutcome(
                        "CAS_MISMATCH",
                        strategy_id,
                        cutoff,
                        latest_sha256=observed_latest,
                        write_count=0,
                    )
                core_sha, writes = self._write_unpublished(
                    strategy_id=strategy_id,
                    cutoff=cutoff,
                    event_at=event_at,
                    formal_output=formal_output,
                )
                return PublicationOutcome(
                    IMMUTABLE_UNPUBLISHED_EVIDENCE,
                    strategy_id,
                    cutoff,
                    core_sha256=core_sha,
                    latest_sha256=observed_latest,
                    write_count=writes,
                )
            if is_rejected:
                assert rejection is not None
                result = write_typed_exact_once(self.store, rejection)
                return PublicationOutcome(
                    ACTIVATION_REJECTED,
                    strategy_id,
                    cutoff,
                    receipt_sha256=rejection.byte_sha256,
                    latest_sha256=observed_latest,
                    write_count=int(result.created),
                    idempotent=not result.created,
                )

            assert active_documents is not None and formal_output is not None
            receipt, active_pointer, active_latest = active_documents
            receipt_result = write_typed_exact_once(self.store, receipt)
            pointer_result = replace_typed_cas(
                self.store,
                active_pointer,
                expected_sha256=EMPTY_SHA256,
            )
            latest_result = replace_typed_cas(
                self.store,
                active_latest,
                expected_sha256=observed_latest,
            )
            return PublicationOutcome(
                ACTIVE,
                strategy_id,
                cutoff,
                receipt_sha256=receipt.byte_sha256,
                core_sha256=formal_output.byte_sha256,
                latest_sha256=latest_result.byte_sha256,
                write_count=sum(
                    int(result.created or result.replaced)
                    for result in (
                        receipt_result,
                        pointer_result,
                        latest_result,
                    )
                ),
            )

    def publish_formal_result(
        self,
        *,
        strategy_id: str,
        cutoff: str,
        expected_active_receipt_sha256: str,
        deterministic_core_bytes: bytes,
        expected_latest_sha256: str | None = None,
    ) -> PublicationOutcome:
        """Permit only byte-identical replay of the currently ACTIVE output."""

        del expected_latest_sha256
        current = self.current_active(strategy_id)
        cutoff = _cutoff(cutoff)
        expected = _expected_sha(
            expected_active_receipt_sha256,
            label="ACTIVE receipt SHA-256",
        )
        requested = _sha(deterministic_core_bytes)
        if (
            current.status != ACTIVE
            or current.cutoff != cutoff
            or current.active_receipt_sha256 != expected
        ):
            return PublicationOutcome(
                NO_CURRENT_ACTIVE_FORMAL_RESULT,
                _strategy(strategy_id),
                cutoff,
                write_count=0,
            )
        if current.core_sha256 != requested or current.core_bytes != deterministic_core_bytes:
            return PublicationOutcome(
                "SAME_CUTOFF_CORE_CONFLICT",
                _strategy(strategy_id),
                cutoff,
                receipt_sha256=expected,
                core_sha256=requested,
                write_count=0,
            )
        return PublicationOutcome(
            FORMAL_RESULT_PUBLISHED,
            _strategy(strategy_id),
            cutoff,
            receipt_sha256=expected,
            core_sha256=requested,
            write_count=0,
            idempotent=True,
        )

    def _revoked_documents(
        self,
        *,
        strategy_id: str,
        cutoff: str,
        event_at: str,
        reason: str,
        active_pointer: RuntimeArtifact,
        active_receipt: RuntimeArtifact,
        formal_output: RuntimeArtifact,
    ) -> tuple[RuntimeArtifact, RuntimeArtifact, RuntimeArtifact]:
        revoked_receipt = self._receipt_artifact(
            strategy_id=strategy_id,
            cutoff=cutoff,
            document=seal_typed_artifact(
                {
                    "version": RECEIPT_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "receipt_id": f"revoked-{active_receipt.byte_sha256[:24]}",
                    "strategy_id": strategy_id,
                    "cutoff": cutoff,
                    "status": REVOKED,
                    "revoked_at": event_at,
                    "reason": reason,
                    "predecessor_active_receipt_ref": active_receipt.reference,
                    "authority": authority_envelope(),
                }
            ),
        )
        revoked_pointer = runtime_artifact(
            relative_path=self._activation_pointer(strategy_id, cutoff),
            document=seal_typed_artifact(
                {
                    "version": POINTER_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "pointer_id": f"activation-{_cutoff_key(cutoff)}",
                    "strategy_id": strategy_id,
                    "cutoff": cutoff,
                    "status": REVOKED,
                    "updated_at": event_at,
                    "predecessor_active_receipt_ref": active_receipt.reference,
                    "revocation_receipt_ref": revoked_receipt.reference,
                    "authority": authority_envelope(),
                }
            ),
        )
        tombstone = runtime_artifact(
            relative_path=self._latest_path(strategy_id),
            document=seal_typed_artifact(
                {
                    "version": LATEST_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "latest_id": f"formal-latest-{strategy_id}",
                    "strategy_id": strategy_id,
                    "cutoff": cutoff,
                    "status": REVOKED,
                    "updated_at": event_at,
                    "revoked_pointer_ref": revoked_pointer.reference,
                    "revocation_receipt_ref": revoked_receipt.reference,
                    "historical_formal_output_ref": formal_output.reference,
                    "authority": authority_envelope(),
                }
            ),
        )
        return revoked_receipt, revoked_pointer, tombstone

    def revoke(
        self,
        *,
        strategy_id: str,
        cutoff: str,
        expected_active_receipt_sha256: str,
        reason: str,
        event_at: str | None = None,
    ) -> PublicationOutcome:
        """CAS ACTIVE pointer to REVOKED, then CAS latest to a tombstone."""

        strategy_id = _strategy(strategy_id)
        cutoff = _cutoff(cutoff)
        event_at = _cutoff(cutoff if event_at is None else event_at, label="event_at")
        expected_receipt = _expected_sha(
            expected_active_receipt_sha256,
            label="ACTIVE receipt SHA-256",
        )
        if type(reason) is not str or not reason or reason.strip() != reason:
            raise ActivationError("revocation reason must be canonical nonempty text")
        with self._formal_transaction(strategy_id, cutoff):
            latest = self._latest_optional(strategy_id)
            pointer_pair = self._pointer_optional(strategy_id, cutoff)
            if pointer_pair is None:
                return PublicationOutcome(
                    NO_CURRENT_ACTIVE_FORMAL_RESULT,
                    strategy_id,
                    cutoff,
                    write_count=0,
                )
            pointer_raw, pointer_document = pointer_pair
            pointer = runtime_artifact(
                relative_path=self._activation_pointer(strategy_id, cutoff),
                document=pointer_document,
            )
            if pointer_document["status"] == REVOKED:
                active_receipt = self._read_ref(
                    pointer_document["predecessor_active_receipt_ref"],
                    expected_version=RECEIPT_VERSION,
                    label="historical ACTIVE receipt",
                )
                formal_output = self._read_ref(
                    active_receipt.document["formal_output_ref"],
                    expected_version=FORMAL_OUTPUT_VERSION,
                    label="historical formal output",
                )
                revoked_receipt = self._read_ref(
                    pointer_document["revocation_receipt_ref"],
                    expected_version=RECEIPT_VERSION,
                    label="REVOKED receipt",
                )
                _, expected_pointer, tombstone = self._revoked_documents(
                    strategy_id=strategy_id,
                    cutoff=cutoff,
                    event_at=event_at,
                    reason=str(revoked_receipt.document["reason"]),
                    active_pointer=pointer,
                    active_receipt=active_receipt,
                    formal_output=formal_output,
                )
                if pointer.raw != expected_pointer.raw:
                    raise ActivationError("REVOKED pointer binding drift")
                writes = 0
                latest_sha = None if latest is None else _sha(latest[0])
                if latest is None:
                    changed = replace_typed_cas(
                        self.store,
                        tombstone,
                        expected_sha256=EMPTY_SHA256,
                    )
                    latest_sha = changed.byte_sha256
                    writes += int(changed.created)
                elif str(latest[1]["cutoff"]) <= cutoff and latest[0] != tombstone.raw:
                    changed = replace_typed_cas(
                        self.store,
                        tombstone,
                        expected_sha256=_sha(latest[0]),
                    )
                    latest_sha = changed.byte_sha256
                    writes += int(changed.replaced)
                return PublicationOutcome(
                    REVOKED,
                    strategy_id,
                    cutoff,
                    receipt_sha256=revoked_receipt.byte_sha256,
                    core_sha256=formal_output.byte_sha256,
                    latest_sha256=latest_sha,
                    write_count=writes,
                    idempotent=True,
                )
            receipt_ref = pointer_document["active_receipt_ref"]
            if receipt_ref["byte_sha256"] != expected_receipt:
                return PublicationOutcome(
                    NO_CURRENT_ACTIVE_FORMAL_RESULT,
                    strategy_id,
                    cutoff,
                    write_count=0,
                )
            active_receipt = self._read_ref(
                receipt_ref,
                expected_version=RECEIPT_VERSION,
                label="ACTIVE receipt",
            )
            formal_output = self._read_ref(
                pointer_document["formal_output_ref"],
                expected_version=FORMAL_OUTPUT_VERSION,
                label="ACTIVE formal output",
            )
            revoked_receipt, revoked_pointer, tombstone = self._revoked_documents(
                strategy_id=strategy_id,
                cutoff=cutoff,
                event_at=event_at,
                reason=reason,
                active_pointer=pointer,
                active_receipt=active_receipt,
                formal_output=formal_output,
            )
            receipt_result = write_typed_exact_once(self.store, revoked_receipt)
            pointer_result = replace_typed_cas(
                self.store,
                revoked_pointer,
                expected_sha256=_sha(pointer_raw),
            )
            latest_sha = None if latest is None else _sha(latest[0])
            latest_writes = 0
            if latest is None:
                latest_result = replace_typed_cas(
                    self.store,
                    tombstone,
                    expected_sha256=EMPTY_SHA256,
                )
                latest_sha = latest_result.byte_sha256
                latest_writes = int(latest_result.created)
            elif str(latest[1]["cutoff"]) <= cutoff:
                latest_result = replace_typed_cas(
                    self.store,
                    tombstone,
                    expected_sha256=_sha(latest[0]),
                )
                latest_sha = latest_result.byte_sha256
                latest_writes = int(latest_result.created or latest_result.replaced)
            return PublicationOutcome(
                REVOKED,
                strategy_id,
                cutoff,
                receipt_sha256=revoked_receipt.byte_sha256,
                core_sha256=formal_output.byte_sha256,
                latest_sha256=latest_sha,
                write_count=(
                    int(receipt_result.created) + int(pointer_result.replaced) + latest_writes
                ),
            )

    def _current_chain(
        self,
        strategy_id: str,
    ) -> tuple[RuntimeArtifact, RuntimeArtifact, RuntimeArtifact] | None:
        latest_pair = self._latest_optional(strategy_id)
        if latest_pair is None or latest_pair[1]["status"] != ACTIVE:
            return None
        latest_raw, latest_document = latest_pair
        try:
            pointer = self._read_ref(
                latest_document["activation_pointer_ref"],
                expected_version=POINTER_VERSION,
                label="current activation pointer",
            )
            if pointer.document["status"] != ACTIVE:
                return None
            receipt = self._read_ref(
                pointer.document["active_receipt_ref"],
                expected_version=RECEIPT_VERSION,
                label="current ACTIVE receipt",
            )
            if receipt.document["status"] != ACTIVE:
                return None
            formal_output = self._read_ref(
                pointer.document["formal_output_ref"],
                expected_version=FORMAL_OUTPUT_VERSION,
                label="current formal output",
            )
            promotion = self._read_ref(
                receipt.document["promotion_receipt_ref"],
                expected_version=PROMOTION_VERSION,
                label="current promotion receipt",
            )
            latest_artifact = runtime_artifact(
                relative_path=self._latest_path(strategy_id),
                document=latest_document,
            )
            resolved = self._validate_promotion_closure(promotion)
            self._validate_activation_closure(
                strategy_id=strategy_id,
                cutoff=str(formal_output.document["cutoff"]),
                promotion=promotion,
                formal_output=formal_output,
                resolved=resolved,
            )
            if (
                latest_artifact.raw != latest_raw
                or latest_document["activation_pointer_ref"] != pointer.reference
                or latest_document["active_receipt_ref"] != receipt.reference
                or latest_document["formal_output_ref"] != formal_output.reference
                or receipt.document["formal_output_ref"] != formal_output.reference
                or promotion.document.get("status") != "PROMOTED"
                or promotion.document.get("accepted") is not True
            ):
                return None
        except (KeyError, StorageError, RuntimeError, TypeError, ValueError):
            return None
        return receipt, pointer, formal_output

    def current_active(self, strategy_id: str) -> CurrentFormalResult:
        """Revalidate latest -> pointer -> receipt -> promotion/output closure."""

        strategy_id = _strategy(strategy_id)
        chain = self._current_chain(strategy_id)
        if chain is None:
            return CurrentFormalResult(
                NO_CURRENT_ACTIVE_FORMAL_RESULT,
                strategy_id,
                None,
                None,
                None,
                None,
            )
        receipt, _, formal_output = chain
        return CurrentFormalResult(
            ACTIVE,
            strategy_id,
            str(formal_output.document["cutoff"]),
            receipt.byte_sha256,
            formal_output.byte_sha256,
            formal_output.raw,
        )

    def active_receipt_for_cutoff(
        self,
        *,
        strategy_id: str,
        cutoff: str,
    ) -> str | None:
        current = self.current_active(strategy_id)
        if current.status != ACTIVE or current.cutoff != _cutoff(cutoff):
            return None
        return current.active_receipt_sha256

    def read_historical_result(
        self,
        *,
        strategy_id: str,
        core_sha256: str,
    ) -> bytes:
        """Read a revoked current result through its exact typed tombstone."""

        strategy_id = _strategy(strategy_id)
        digest = _expected_sha(core_sha256, label="formal output SHA-256")
        latest = self._latest_optional(strategy_id)
        if latest is None or latest[1]["status"] != REVOKED:
            raise ActivationError("historical formal result is not indexed")
        formal_output = self._read_ref(
            latest[1]["historical_formal_output_ref"],
            expected_version=FORMAL_OUTPUT_VERSION,
            label="historical formal output",
        )
        if (
            formal_output.byte_sha256 != digest
            or formal_output.document["strategy_id"] != strategy_id
        ):
            raise ActivationError("historical formal result identity mismatch")
        return formal_output.raw


__all__ = [
    "ACTIVE",
    "ACTIVATION_REJECTED",
    "ActivationError",
    "ActivationPublisher",
    "CurrentFormalResult",
    "FORMAL_RESULT_PUBLISHED",
    "IMMUTABLE_UNPUBLISHED_EVIDENCE",
    "NO_CURRENT_ACTIVE_FORMAL_RESULT",
    "PublicationOutcome",
    "REVOKED",
]

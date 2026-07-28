"""Neutral bootstrap, cutover, recovery, and rollback control plane."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .canonical import (
    ACTIVE_POINTER_VERSION,
    BOOTSTRAP_RECEIPT_VERSION,
    CUTOVER_RECEIPT_VERSION,
    INTENT_VERSION,
    ROLLBACK_RECEIPT_VERSION,
    RUN_VERSION,
    SELECTOR_VERSION,
    TARGET_VERSION,
    V4_FORMAL_ACTIVE_POINTER_VERSION,
    V4_REGISTERED_REFERENCE_VERSIONS,
    artifact_reference,
    authority_ceiling,
    decode,
    decode_reference,
    encode,
    intent_identity,
    seal,
    sha256,
)
from .storage import (
    CONTROL_ROOT,
    ControlCASMismatch,
    ControlNotFoundError,
    ControlStorageError,
    ControlStorageSecurityError,
    ControlStore,
    EMPTY_SHA256,
)

SELECTOR_PATH = CONTROL_ROOT / "default_protocol_selector.v1.json"
SELECTOR_LOCK = CONTROL_ROOT / ".default-protocol-selector.lock"


class RuntimeControlError(ValueError):
    """A route transition failed closed."""


class RuntimeControlThirdState(RuntimeControlError):
    """Recovery observed neither the exact old nor exact proposed selector."""


class SimulatedTransitionCrash(RuntimeError):
    """Test-only interruption after selector CAS and before receipt."""


@dataclass(frozen=True)
class ControlArtifact:
    relative_path: str
    document: Mapping[str, Any]
    raw: bytes
    byte_sha256: str
    strategy_id: str
    cutoff: str

    @property
    def reference(self) -> dict[str, str]:
        return artifact_reference(
            relative_path=self.relative_path,
            document=self.document,
            raw=self.raw,
            strategy_id=self.strategy_id,
            cutoff=self.cutoff,
        )


@dataclass(frozen=True)
class TransitionOutcome:
    outcome: str
    selector_sha256: str
    receipt: ControlArtifact
    selector_changed: bool


def _artifact(
    path: str | PurePosixPath,
    payload: Mapping[str, Any],
    *,
    strategy_id: str | None = None,
    cutoff: str | None = None,
) -> ControlArtifact:
    document = seal(payload)
    raw = encode(document)
    resolved_strategy = strategy_id or document.get("strategy_id")
    resolved_cutoff = cutoff or document.get("cutoff")
    if type(resolved_strategy) is not str or type(resolved_cutoff) is not str:
        raise RuntimeControlError(
            "artifact reference strategy and cutoff are required"
        )
    return ControlArtifact(
        str(path),
        document,
        raw,
        sha256(raw),
        resolved_strategy,
        resolved_cutoff,
    )


def _sorted_refs(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(value) for value in values]
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("relative_path")),
            str(row.get("byte_sha256")),
        ),
    )


class V15ActiveRunPublisher:
    """The only V15 writer: one per-strategy active-run pointer."""

    def __init__(self, store: ControlStore) -> None:
        if not isinstance(store, ControlStore):
            raise TypeError("store must be ControlStore")
        self._store = store

    @staticmethod
    def pointer_path(strategy_id: str) -> PurePosixPath:
        _validate_id(strategy_id, label="strategy_id")
        return CONTROL_ROOT / "active_runs" / "v15" / f"{strategy_id}.json"

    @staticmethod
    def lock_path(strategy_id: str) -> PurePosixPath:
        _validate_id(strategy_id, label="strategy_id")
        return (
            CONTROL_ROOT
            / "active_runs"
            / "v15"
            / ".locks"
            / f"{strategy_id}.lock"
        )

    def publish(
        self,
        *,
        strategy_id: str,
        run_ref: Mapping[str, Any],
        cutoff: str,
        updated_at: str,
        expected_pointer_sha256: str,
    ) -> ControlArtifact:
        run = _read_reference(
            self._store,
            run_ref,
            expected_version=RUN_VERSION,
        )
        if (
            run.document.get("protocol_id") != "v15"
            or run.document.get("strategy_id") != strategy_id
            or run.document.get("cutoff") != cutoff
            or run.document.get("status") != "HEALTHY"
        ):
            raise RuntimeControlError("V15 run scope or health mismatch")
        pointer = _artifact(
            self.pointer_path(strategy_id),
            {
                "version": ACTIVE_POINTER_VERSION,
                "protocol_id": "v15",
                "strategy_id": strategy_id,
                "run_ref": dict(run_ref),
                "cutoff": cutoff,
                "updated_at": updated_at,
            },
        )
        with self._store.locked(self.lock_path(strategy_id)):
            self._store.replace_cas(
                pointer.relative_path,
                expected_pointer_sha256,
                pointer.raw,
            )
            if self._store.read(
                pointer.relative_path,
                pointer.byte_sha256,
            ) != pointer.raw:
                raise RuntimeControlError("V15 pointer readback mismatch")
        return pointer


class ResearchRuntimeControl:
    """Selector-transition service with immutable intent and receipt evidence."""

    def __init__(self, workspace_root: str) -> None:
        self.store = ControlStore(workspace_root)
        self.store.initialize()
        self.v15_publisher = V15ActiveRunPublisher(self.store)
        self.lock_trace: list[str] = []

    def install_protocol_target(
        self,
        *,
        protocol_id: str,
        strategy_id: str,
        cutoff: str,
    ) -> ControlArtifact:
        _validate_id(strategy_id, label="strategy_id")
        slug = {
            "v15": "v15",
            "myquant.v17.v4": "myquant-v17-v4",
        }.get(protocol_id)
        if slug is None:
            raise RuntimeControlError("protocol_id is closed")
        template = {
            "v15": (
                "results/research_runtime_control/active_runs/"
                "v15/{strategy_id}.json"
            ),
            "myquant.v17.v4": (
                "results/v17_v4_formal_research/strategies/"
                "{strategy_id}/_active.json"
            ),
        }[protocol_id]
        document = seal(
            {
                "version": TARGET_VERSION,
                "target_id": f"{slug}-{strategy_id}",
                "protocol_id": protocol_id,
                "strategy_scope": strategy_id,
                "active_run_pointer_template": template,
                "authority_ceiling": authority_ceiling(),
            }
        )
        raw = encode(document)
        artifact = ControlArtifact(
            str(
                CONTROL_ROOT
                / "protocol_targets"
                / slug
                / f"{sha256(raw)}.json"
            ),
            document,
            raw,
            sha256(raw),
            strategy_id,
            cutoff,
        )
        self.store.write_exact_once(artifact.relative_path, artifact.raw)
        return artifact

    def create_v15_run(
        self,
        *,
        run_id: str,
        strategy_id: str,
        cutoff: str,
        evidence_refs: Sequence[Mapping[str, Any]],
    ) -> ControlArtifact:
        _validate_id(run_id, label="run_id")
        _validate_id(strategy_id, label="strategy_id")
        artifact = _artifact(
            CONTROL_ROOT / "v15_runs" / run_id / "run.json",
            {
                "version": RUN_VERSION,
                "run_id": run_id,
                "protocol_id": "v15",
                "strategy_id": strategy_id,
                "cutoff": cutoff,
                "status": "HEALTHY",
                "evidence_refs": _sorted_refs(evidence_refs),
                "authority_ceiling": authority_ceiling(),
            },
        )
        self.store.write_exact_once(artifact.relative_path, artifact.raw)
        return artifact

    def bootstrap_v15(
        self,
        *,
        strategy_id: str,
        run_id: str,
        cutoff: str,
        recorded_at: str,
        intent_id: str,
        receipt_id: str,
        evidence_refs: Sequence[Mapping[str, Any]] = (),
        expected_active_pointer_sha256: str = EMPTY_SHA256,
        expected_selector_sha256: str = EMPTY_SHA256,
        crash_before_cas: bool = False,
        crash_after_cas: bool = False,
    ) -> TransitionOutcome:
        target = self.install_protocol_target(
            protocol_id="v15",
            strategy_id=strategy_id,
            cutoff=cutoff,
        )
        run = self.create_v15_run(
            run_id=run_id,
            strategy_id=strategy_id,
            cutoff=cutoff,
            evidence_refs=evidence_refs,
        )
        active = self.v15_publisher.publish(
            strategy_id=strategy_id,
            run_ref=run.reference,
            cutoff=cutoff,
            updated_at=recorded_at,
            expected_pointer_sha256=expected_active_pointer_sha256,
        )
        intent = self._create_intent(
            transition="BOOTSTRAP",
            intent_id=intent_id,
            created_at=recorded_at,
            expected_selector_sha256=expected_selector_sha256,
            expected_protocol_target_ref=None,
            proposed_protocol_target_ref=target.reference,
            expected_target_active_pointer_sha256=active.byte_sha256,
            expected_target_run_ref=run.reference,
            required_evidence_refs=evidence_refs,
        )
        return self._execute(
            intent=intent,
            receipt_id=receipt_id,
            recorded_at=recorded_at,
            crash_before_cas=crash_before_cas,
            crash_after_cas=crash_after_cas,
            recovery=False,
        )

    def cutover(
        self,
        *,
        strategy_id: str,
        v4_protocol_target_ref: Mapping[str, Any],
        expected_v4_active_pointer_sha256: str,
        expected_v4_run_ref: Mapping[str, Any],
        expected_selector_sha256: str,
        recorded_at: str,
        intent_id: str,
        receipt_id: str,
        required_evidence_refs: Sequence[Mapping[str, Any]],
        crash_before_cas: bool = False,
        crash_after_cas: bool = False,
        before_lock_hook: Callable[[], None] | None = None,
    ) -> TransitionOutcome:
        selector = self.current_selector(
            expected_sha256=expected_selector_sha256
        )
        if selector.document.get("status") not in {
            "V15_DEFAULT",
            "ROLLED_BACK_TO_V15",
        }:
            raise RuntimeControlError("cutover requires V15 selector")
        target = _read_reference(
            self.store,
            v4_protocol_target_ref,
            expected_version=TARGET_VERSION,
        )
        if (
            target.document.get("protocol_id") != "myquant.v17.v4"
            or target.document.get("strategy_scope") != strategy_id
        ):
            raise RuntimeControlError("cutover v4 target scope mismatch")
        intent = self._create_intent(
            transition="CUTOVER",
            intent_id=intent_id,
            created_at=recorded_at,
            expected_selector_sha256=expected_selector_sha256,
            expected_protocol_target_ref=selector.document[
                "protocol_target_ref"
            ],
            proposed_protocol_target_ref=v4_protocol_target_ref,
            expected_target_active_pointer_sha256=(
                expected_v4_active_pointer_sha256
            ),
            expected_target_run_ref=expected_v4_run_ref,
            required_evidence_refs=required_evidence_refs,
        )
        if before_lock_hook is not None:
            before_lock_hook()
        return self._execute(
            intent=intent,
            receipt_id=receipt_id,
            recorded_at=recorded_at,
            crash_before_cas=crash_before_cas,
            crash_after_cas=crash_after_cas,
            recovery=False,
        )

    def rollback_to_v15(
        self,
        *,
        strategy_id: str,
        v15_protocol_target_ref: Mapping[str, Any],
        expected_v15_active_pointer_sha256: str,
        expected_v15_run_ref: Mapping[str, Any],
        expected_selector_sha256: str,
        successful_cutover_receipt_ref: Mapping[str, Any],
        recorded_at: str,
        intent_id: str,
        receipt_id: str,
        required_evidence_refs: Sequence[Mapping[str, Any]] = (),
        crash_before_cas: bool = False,
        crash_after_cas: bool = False,
    ) -> TransitionOutcome:
        selector = self.current_selector(
            expected_sha256=expected_selector_sha256
        )
        if selector.document.get("status") != "RESEARCH_DEFAULT_ACTIVE":
            raise RuntimeControlError("rollback requires active v4 selector")
        cutover_receipt = _read_reference(
            self.store,
            successful_cutover_receipt_ref,
            expected_version=CUTOVER_RECEIPT_VERSION,
        )
        if cutover_receipt.document.get("outcome") not in {
            "CUTOVER_SUCCEEDED",
            "CUTOVER_RECOVERED",
        }:
            raise RuntimeControlError(
                "rollback requires successful cutover receipt"
            )
        target = _read_reference(
            self.store,
            v15_protocol_target_ref,
            expected_version=TARGET_VERSION,
        )
        if (
            target.document.get("protocol_id") != "v15"
            or target.document.get("strategy_scope") != strategy_id
        ):
            raise RuntimeControlError("rollback V15 target scope mismatch")
        intent = self._create_intent(
            transition="ROLLBACK",
            intent_id=intent_id,
            created_at=recorded_at,
            expected_selector_sha256=expected_selector_sha256,
            expected_protocol_target_ref=selector.document[
                "protocol_target_ref"
            ],
            proposed_protocol_target_ref=v15_protocol_target_ref,
            expected_target_active_pointer_sha256=(
                expected_v15_active_pointer_sha256
            ),
            expected_target_run_ref=expected_v15_run_ref,
            required_evidence_refs=(
                *required_evidence_refs,
                successful_cutover_receipt_ref,
            ),
        )
        return self._execute(
            intent=intent,
            receipt_id=receipt_id,
            recorded_at=recorded_at,
            crash_before_cas=crash_before_cas,
            crash_after_cas=crash_after_cas,
            recovery=False,
        )

    def recover(
        self,
        *,
        transition: str,
        intent_id: str,
        receipt_id: str,
        recorded_at: str,
    ) -> TransitionOutcome:
        _validate_id(intent_id, label="intent_id")
        if transition not in {"BOOTSTRAP", "CUTOVER", "ROLLBACK"}:
            raise RuntimeControlError("recovery transition is closed")
        path = _intent_path(transition, intent_id)
        raw = self.store.read(path)
        document = decode(raw, expected_version=INTENT_VERSION)
        if document["transition"] != transition:
            raise RuntimeControlError("recovery transition mismatch")
        run_ref = document["expected_target_run_ref"]
        intent = ControlArtifact(
            str(path),
            document,
            raw,
            sha256(raw),
            str(run_ref["strategy_id"]),
            str(run_ref["cutoff"]),
        )
        return self._execute(
            intent=intent,
            receipt_id=receipt_id,
            recorded_at=recorded_at,
            crash_before_cas=False,
            crash_after_cas=False,
            recovery=True,
        )

    def current_selector(
        self,
        *,
        expected_sha256: str | None = None,
    ) -> ControlArtifact:
        raw = self.store.read(SELECTOR_PATH, expected_sha256)
        document = decode(raw, expected_version=SELECTOR_VERSION)
        return ControlArtifact(
            str(SELECTOR_PATH),
            document,
            raw,
            sha256(raw),
            str(document["protocol_target_ref"]["strategy_id"]),
            str(document["protocol_target_ref"]["cutoff"]),
        )

    def resolve_current(
        self,
        *,
        strategy_id: str,
    ) -> tuple[ControlArtifact, ControlArtifact, ControlArtifact, ControlArtifact]:
        selector = self.current_selector()
        target = _read_reference(
            self.store,
            selector.document["protocol_target_ref"],
            expected_version=TARGET_VERSION,
        )
        pointer_path = _render_active_path(target.document, strategy_id)
        pointer_raw = self.store.read(pointer_path)
        pointer_document = _decode_active_pointer(
            pointer_raw,
            protocol_id=str(target.document["protocol_id"]),
        )
        pointer = ControlArtifact(
            pointer_path,
            pointer_document,
            pointer_raw,
            sha256(pointer_raw),
            str(pointer_document["strategy_id"]),
            str(pointer_document["cutoff"]),
        )
        run_ref = _pointer_run_ref(pointer.document, store=self.store)
        run = (
            _read_v4_typed_reference(
                self.store,
                run_ref,
                expected_version="myquant.v17.v4.formal-output.v1",
            )
            if target.document.get("protocol_id") == "myquant.v17.v4"
            else _read_reference(self.store, run_ref)
        )
        _validate_pointer_run(
            store=self.store,
            target=target,
            pointer=pointer,
            run=run,
            strategy_id=strategy_id,
        )
        return selector, target, pointer, run

    def _create_intent(
        self,
        *,
        transition: str,
        intent_id: str,
        created_at: str,
        expected_selector_sha256: str,
        expected_protocol_target_ref: Mapping[str, Any] | None,
        proposed_protocol_target_ref: Mapping[str, Any],
        expected_target_active_pointer_sha256: str,
        expected_target_run_ref: Mapping[str, Any],
        required_evidence_refs: Sequence[Mapping[str, Any]],
    ) -> ControlArtifact:
        _validate_id(intent_id, label="intent_id")
        intent_path = _intent_path(transition, intent_id)
        selector = _selector_artifact(
            transition=transition,
            protocol_target_ref=proposed_protocol_target_ref,
            transition_intent_ref=intent_identity(
                intent_id=intent_id,
                relative_path=str(intent_path),
            ),
            updated_at=created_at,
        )
        intent = _artifact(
            intent_path,
            {
                "version": INTENT_VERSION,
                "intent_id": intent_id,
                "transition": transition,
                "created_at": created_at,
                "expected_selector_sha256": expected_selector_sha256,
                "expected_protocol_target_ref": (
                    None
                    if expected_protocol_target_ref is None
                    else dict(expected_protocol_target_ref)
                ),
                "proposed_protocol_target_ref": dict(
                    proposed_protocol_target_ref
                ),
                "expected_target_active_pointer_sha256": (
                    expected_target_active_pointer_sha256
                ),
                "expected_target_run_ref": dict(expected_target_run_ref),
                "proposed_selector_bytes_sha256": selector.byte_sha256,
                "required_evidence_refs": _sorted_refs(
                    required_evidence_refs
                ),
            },
            strategy_id=str(expected_target_run_ref["strategy_id"]),
            cutoff=str(expected_target_run_ref["cutoff"]),
        )
        self.store.write_exact_once(intent.relative_path, intent.raw)
        return intent

    def _execute(
        self,
        *,
        intent: ControlArtifact,
        receipt_id: str,
        recorded_at: str,
        crash_before_cas: bool,
        crash_after_cas: bool,
        recovery: bool,
    ) -> TransitionOutcome:
        transition = str(intent.document["transition"])
        proposed_target = _read_reference(
            self.store,
            intent.document["proposed_protocol_target_ref"],
            expected_version=TARGET_VERSION,
        )
        expected_target_ref = intent.document[
            "expected_protocol_target_ref"
        ]
        if expected_target_ref is not None:
            _read_reference(
                self.store,
                expected_target_ref,
                expected_version=TARGET_VERSION,
            )
        expected_run_version = (
            RUN_VERSION
            if proposed_target.document["protocol_id"] == "v15"
            else None
        )
        run_ref = intent.document["expected_target_run_ref"]
        run = (
            _read_v4_typed_reference(
                self.store,
                run_ref,
                expected_version="myquant.v17.v4.formal-output.v1",
            )
            if proposed_target.document["protocol_id"] == "myquant.v17.v4"
            else _read_reference(
                self.store,
                run_ref,
                expected_version=expected_run_version,
            )
        )
        run_strategy = str(
            intent.document["expected_target_run_ref"]["strategy_id"]
        )
        run_cutoff = str(
            intent.document["expected_target_run_ref"]["cutoff"]
        )
        for reference in (
            intent.document["proposed_protocol_target_ref"],
            *(
                ()
                if expected_target_ref is None
                else (expected_target_ref,)
            ),
            *intent.document["required_evidence_refs"],
        ):
            _validate_reference_scope(
                reference,
                strategy_id=run_strategy,
                cutoff=run_cutoff,
            )
        strategy_id = str(proposed_target.document["strategy_scope"])
        pointer_path = _render_active_path(
            proposed_target.document,
            strategy_id,
        )
        pointer_lock, existing_only = _target_lock(
            proposed_target.document,
            strategy_id,
        )
        self.lock_trace.clear()
        with self.store.locked(
            pointer_lock,
            existing_only=existing_only,
        ):
            self.lock_trace.append(str(pointer_lock))
            with self.store.locked(SELECTOR_LOCK):
                self.lock_trace.append(str(SELECTOR_LOCK))
                pointer_raw = self.store.read(pointer_path)
                pointer_document = _decode_active_pointer(
                    pointer_raw,
                    protocol_id=str(
                        proposed_target.document["protocol_id"]
                    ),
                )
                pointer = ControlArtifact(
                    pointer_path,
                    pointer_document,
                    pointer_raw,
                    sha256(pointer_raw),
                    str(pointer_document["strategy_id"]),
                    str(pointer_document["cutoff"]),
                )
                observed_pointer_sha = pointer.byte_sha256
                observed_selector = self.store.read_optional(SELECTOR_PATH)
                observed_selector_sha = (
                    EMPTY_SHA256
                    if observed_selector is None
                    else observed_selector.byte_sha256
                )
                proposed_selector = _selector_artifact(
                    transition=transition,
                    protocol_target_ref=proposed_target.reference,
                    transition_intent_ref=intent_identity(
                        intent_id=str(intent.document["intent_id"]),
                        relative_path=intent.relative_path,
                    ),
                    updated_at=str(intent.document["created_at"]),
                )
                if (
                    proposed_selector.byte_sha256
                    != intent.document["proposed_selector_bytes_sha256"]
                ):
                    raise RuntimeControlError(
                        "intent proposed selector bytes drift"
                    )
                expected_selector_sha = str(
                    intent.document["expected_selector_sha256"]
                )
                if recovery:
                    if proposed_target.document["protocol_id"] == "myquant.v17.v4":
                        if (
                            observed_pointer_sha
                            != intent.document[
                                "expected_target_active_pointer_sha256"
                            ]
                            or _pointer_run_ref(
                                pointer.document,
                                store=self.store,
                            )
                            != intent.document["expected_target_run_ref"]
                        ):
                            raise RuntimeControlError(
                                "v4 recovery target evidence drifted"
                            )
                        _validate_pointer_run(
                            store=self.store,
                            target=proposed_target,
                            pointer=pointer,
                            run=run,
                            strategy_id=strategy_id,
                        )
                        _validate_v4_cutover_evidence(
                            store=self.store,
                            intent=intent,
                            expected_target=(
                                None
                                if expected_target_ref is None
                                else _read_reference(
                                    self.store,
                                    expected_target_ref,
                                    expected_version=TARGET_VERSION,
                                )
                            ),
                            formal_pointer=pointer,
                            formal_output=run,
                        )
                    if (
                        observed_selector_sha
                        == proposed_selector.byte_sha256
                    ):
                        return self._write_receipt(
                            intent=intent,
                            receipt_id=receipt_id,
                            recorded_at=recorded_at,
                            observed_prevalue_sha256=(
                                observed_selector_sha
                            ),
                            observed_target_active_pointer_sha256=(
                                observed_pointer_sha
                            ),
                            post_readback_sha256=observed_selector_sha,
                            outcome=f"{transition}_RECOVERED",
                            selector_changed=False,
                        )
                    if observed_selector_sha == expected_selector_sha:
                        return self._write_receipt(
                            intent=intent,
                            receipt_id=receipt_id,
                            recorded_at=recorded_at,
                            observed_prevalue_sha256=(
                                observed_selector_sha
                            ),
                            observed_target_active_pointer_sha256=(
                                observed_pointer_sha
                            ),
                            post_readback_sha256=observed_selector_sha,
                            outcome=f"{transition}_ABORTED",
                            selector_changed=False,
                        )
                    raise RuntimeControlThirdState(
                        "selector is neither exact old nor exact proposed"
                    )
                if (
                    observed_pointer_sha
                    != intent.document[
                        "expected_target_active_pointer_sha256"
                    ]
                    or _pointer_run_ref(
                        pointer.document,
                        store=self.store,
                    )
                    != intent.document["expected_target_run_ref"]
                ):
                    return self._write_receipt(
                        intent=intent,
                        receipt_id=receipt_id,
                        recorded_at=recorded_at,
                        observed_prevalue_sha256=observed_selector_sha,
                        observed_target_active_pointer_sha256=(
                            observed_pointer_sha
                        ),
                        post_readback_sha256=observed_selector_sha,
                        outcome=f"{transition}_CAS_BLOCKED",
                        selector_changed=False,
                    )
                _validate_pointer_run(
                    store=self.store,
                    target=proposed_target,
                    pointer=pointer,
                    run=run,
                    strategy_id=strategy_id,
                )
                if proposed_target.document["protocol_id"] == "myquant.v17.v4":
                    _validate_v4_cutover_evidence(
                        store=self.store,
                        intent=intent,
                        expected_target=(
                            None
                            if expected_target_ref is None
                            else _read_reference(
                                self.store,
                                expected_target_ref,
                                expected_version=TARGET_VERSION,
                            )
                        ),
                        formal_pointer=pointer,
                        formal_output=run,
                    )
                else:
                    for evidence_ref in intent.document[
                        "required_evidence_refs"
                    ]:
                        _read_reference(self.store, evidence_ref)
                if observed_selector_sha == proposed_selector.byte_sha256:
                    return self._write_receipt(
                        intent=intent,
                        receipt_id=receipt_id,
                        recorded_at=recorded_at,
                        observed_prevalue_sha256=observed_selector_sha,
                        observed_target_active_pointer_sha256=(
                            observed_pointer_sha
                        ),
                        post_readback_sha256=observed_selector_sha,
                        outcome=f"{transition}_RECOVERED",
                        selector_changed=False,
                    )
                if observed_selector_sha != expected_selector_sha:
                    raise RuntimeControlThirdState(
                        "selector is neither exact old nor exact proposed"
                    )
                if observed_selector is not None:
                    old_document = decode(
                        observed_selector.data,
                        expected_version=SELECTOR_VERSION,
                    )
                    if (
                        old_document.get("protocol_target_ref")
                        != expected_target_ref
                    ):
                        raise RuntimeControlThirdState(
                            "selector target does not match intent prevalue"
                        )
                elif expected_target_ref is not None:
                    raise RuntimeControlThirdState(
                        "missing selector cannot bind expected target"
                    )
                if crash_before_cas:
                    raise SimulatedTransitionCrash(
                        "simulated crash before selector CAS"
                    )
                try:
                    self.store.replace_cas(
                        SELECTOR_PATH,
                        expected_selector_sha,
                        proposed_selector.raw,
                    )
                except ControlCASMismatch as exc:
                    raise RuntimeControlThirdState(
                        "selector changed during CAS"
                    ) from exc
                if crash_after_cas:
                    raise SimulatedTransitionCrash(
                        "simulated crash after selector CAS"
                    )
                readback = self.store.read(
                    SELECTOR_PATH,
                    proposed_selector.byte_sha256,
                )
                if readback != proposed_selector.raw:
                    raise RuntimeControlError(
                        "selector exact readback mismatch"
                    )
                return self._write_receipt(
                    intent=intent,
                    receipt_id=receipt_id,
                    recorded_at=recorded_at,
                    observed_prevalue_sha256=observed_selector_sha,
                    observed_target_active_pointer_sha256=(
                        observed_pointer_sha
                    ),
                    post_readback_sha256=proposed_selector.byte_sha256,
                    outcome=f"{transition}_SUCCEEDED",
                    selector_changed=True,
                )

    def _write_receipt(
        self,
        *,
        intent: ControlArtifact,
        receipt_id: str,
        recorded_at: str,
        observed_prevalue_sha256: str,
        observed_target_active_pointer_sha256: str,
        post_readback_sha256: str,
        outcome: str,
        selector_changed: bool,
    ) -> TransitionOutcome:
        _validate_id(receipt_id, label="receipt_id")
        transition = str(intent.document["transition"])
        version = {
            "BOOTSTRAP": BOOTSTRAP_RECEIPT_VERSION,
            "CUTOVER": CUTOVER_RECEIPT_VERSION,
            "ROLLBACK": ROLLBACK_RECEIPT_VERSION,
        }[transition]
        receipt = _artifact(
            _receipt_path(transition, receipt_id),
            {
                "version": version,
                "receipt_id": receipt_id,
                "intent_id": intent.document["intent_id"],
                "transition": transition,
                "recorded_at": recorded_at,
                "expected_selector_sha256": intent.document[
                    "expected_selector_sha256"
                ],
                "expected_protocol_target_ref": intent.document[
                    "expected_protocol_target_ref"
                ],
                "proposed_protocol_target_ref": intent.document[
                    "proposed_protocol_target_ref"
                ],
                "expected_target_active_pointer_sha256": intent.document[
                    "expected_target_active_pointer_sha256"
                ],
                "expected_target_run_ref": intent.document[
                    "expected_target_run_ref"
                ],
                "proposed_selector_bytes_sha256": intent.document[
                    "proposed_selector_bytes_sha256"
                ],
                "required_evidence_refs": intent.document[
                    "required_evidence_refs"
                ],
                "observed_prevalue_sha256": observed_prevalue_sha256,
                "observed_target_active_pointer_sha256": (
                    observed_target_active_pointer_sha256
                ),
                "post_readback_sha256": post_readback_sha256,
                "outcome": outcome,
            },
            strategy_id=str(
                intent.document["expected_target_run_ref"]["strategy_id"]
            ),
            cutoff=str(
                intent.document["expected_target_run_ref"]["cutoff"]
            ),
        )
        self.store.write_exact_once(receipt.relative_path, receipt.raw)
        return TransitionOutcome(
            outcome,
            post_readback_sha256,
            receipt,
            selector_changed,
        )


def _validate_id(value: str, *, label: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 128
        or not value[0].isalnum()
        or not value.isascii()
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in value
        )
    ):
        raise RuntimeControlError(f"{label} is invalid")


def _intent_path(transition: str, intent_id: str) -> PurePosixPath:
    folder = {
        "BOOTSTRAP": "bootstrap_intents",
        "CUTOVER": "cutover_intents",
        "ROLLBACK": "rollback_intents",
    }[transition]
    return CONTROL_ROOT / folder / f"{intent_id}.json"


def _receipt_path(transition: str, receipt_id: str) -> PurePosixPath:
    folder = {
        "BOOTSTRAP": "bootstrap_receipts",
        "CUTOVER": "cutover_receipts",
        "ROLLBACK": "rollback_receipts",
    }[transition]
    return CONTROL_ROOT / folder / f"{receipt_id}.json"


def _selector_artifact(
    *,
    transition: str,
    protocol_target_ref: Mapping[str, Any],
    transition_intent_ref: Mapping[str, Any],
    updated_at: str,
) -> ControlArtifact:
    status = {
        "BOOTSTRAP": "V15_DEFAULT",
        "CUTOVER": "RESEARCH_DEFAULT_ACTIVE",
        "ROLLBACK": "ROLLED_BACK_TO_V15",
    }[transition]
    return _artifact(
        SELECTOR_PATH,
        {
            "version": SELECTOR_VERSION,
            "selector_id": "default-protocol-selector",
            "status": status,
            "protocol_target_ref": dict(protocol_target_ref),
            "transition_intent_ref": dict(transition_intent_ref),
            "updated_at": updated_at,
        },
        strategy_id=str(protocol_target_ref["strategy_id"]),
        cutoff=str(protocol_target_ref["cutoff"]),
    )


def _read_reference(
    store: ControlStore,
    reference: Mapping[str, Any],
    *,
    expected_version: str | None = None,
) -> ControlArtifact:
    if not isinstance(reference, Mapping):
        raise RuntimeControlError("artifact reference must be an object")
    path = reference.get("relative_path")
    digest = reference.get("byte_sha256")
    if type(path) is not str or type(digest) is not str:
        raise RuntimeControlError("artifact reference is incomplete")
    raw = store.read(path, digest)
    document = decode_reference(
        raw,
        expected_version=expected_version,
    )
    artifact = ControlArtifact(
        path,
        document,
        raw,
        sha256(raw),
        str(reference["strategy_id"]),
        str(reference["cutoff"]),
    )
    if artifact.reference != dict(reference):
        raise RuntimeControlError("artifact exact reference mismatch")
    return artifact


def _read_v4_typed_reference(
    store: ControlStore,
    reference: Mapping[str, Any],
    *,
    expected_version: str,
) -> ControlArtifact:
    if expected_version not in V4_REGISTERED_REFERENCE_VERSIONS:
        raise RuntimeControlError("v4 evidence version is not registered")
    try:
        from quant_investor.v17_v4_contract import load_canonical_artifact

        artifact = _read_reference(
            store,
            reference,
            expected_version=expected_version,
        )
        validated = load_canonical_artifact(
            artifact.raw,
            expected_version=expected_version,
            label="v4 selector evidence",
        )
    except (ImportError, ValueError, ControlStorageError) as exc:
        raise RuntimeControlError("v4 selector evidence is invalid") from exc
    if dict(validated.payload) != dict(artifact.document):
        raise RuntimeControlError("v4 selector evidence validation drift")
    return artifact


def _evidence_by_version(
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    required_versions = {
        V4_FORMAL_ACTIVE_POINTER_VERSION,
        "myquant.v17.v4.default-eligible-pointer.v1",
        "myquant.v17.v4.canary-pointer.v1",
    }
    matched: dict[str, Mapping[str, Any]] = {}
    for reference in references:
        version = reference.get("artifact_version")
        if version not in required_versions:
            continue
        if version in matched:
            raise RuntimeControlError("duplicate v4 cutover evidence version")
        matched[str(version)] = reference
    if set(matched) != required_versions:
        raise RuntimeControlError(
            "v4 cutover requires formal, eligibility, and canary pointers"
        )
    return matched


def _read_v4_completion_chain(
    *,
    store: ControlStore,
    pointer: ControlArtifact,
    intent_version: str,
    completion_version: str,
    completion_path: str,
    expected_status: str,
) -> tuple[ControlArtifact, ControlArtifact]:
    if pointer.document.get("state") != "PENDING_COMPLETION":
        raise RuntimeControlError("v4 pointer state is invalid")
    intent = _read_v4_typed_reference(
        store,
        pointer.document["intent_ref"],
        expected_version=intent_version,
    )
    stored = store.read_optional(completion_path)
    if stored is None:
        raise RuntimeControlError("v4 pointer is pending completion")
    try:
        from quant_investor.v17_v4_contract import load_canonical_artifact

        document = decode_reference(
            stored.data,
            expected_version=completion_version,
        )
        validated = load_canonical_artifact(
            stored.data,
            expected_version=completion_version,
            label="v4 pointer completion receipt",
        )
    except (ImportError, ValueError, ControlStorageError) as exc:
        raise RuntimeControlError(
            "v4 pointer completion receipt is invalid"
        ) from exc
    completion = ControlArtifact(
        completion_path,
        document,
        stored.data,
        stored.byte_sha256,
        pointer.strategy_id,
        pointer.cutoff,
    )
    if (
        dict(validated.payload) != dict(completion.document)
        or completion.document.get("status") != expected_status
        or completion.document.get("intent_ref") != intent.reference
        or completion.document.get("pointer_ref") != pointer.reference
        or completion.document.get("post_readback_sha256")
        != pointer.byte_sha256
        or completion.document.get("proposed_pointer_sha256")
        != pointer.byte_sha256
        or completion.document.get("expected_pointer_sha256")
        != intent.document.get("expected_pointer_sha256")
        or completion.document.get("observed_pointer_sha256")
        != intent.document.get("expected_pointer_sha256")
        or completion.document.get("from_state")
        != intent.document.get("from_state")
        or completion.document.get("to_state")
        != intent.document.get("to_state")
    ):
        raise RuntimeControlError(
            "v4 pointer completion binding is invalid"
        )
    return intent, completion


def _validate_v4_cutover_evidence(
    *,
    store: ControlStore,
    intent: ControlArtifact,
    expected_target: ControlArtifact | None,
    formal_pointer: ControlArtifact,
    formal_output: ControlArtifact,
) -> None:
    if expected_target is None or expected_target.document.get("protocol_id") != "v15":
        raise RuntimeControlError("v4 cutover requires the exact V15 target")
    evidence = _evidence_by_version(
        intent.document["required_evidence_refs"]
    )
    bound_formal_pointer = _read_v4_typed_reference(
        store,
        evidence[V4_FORMAL_ACTIVE_POINTER_VERSION],
        expected_version=V4_FORMAL_ACTIVE_POINTER_VERSION,
    )
    if bound_formal_pointer.reference != formal_pointer.reference:
        raise RuntimeControlError("v4 formal pointer evidence is not live")
    activation_intent = _read_v4_typed_reference(
        store,
        formal_pointer.document["intent_ref"],
        expected_version="myquant.v17.v4.formal-activation-intent.v1",
    )
    completion_path = (
        "results/v17_v4_formal_research/strategies/"
        f"{formal_pointer.strategy_id}/completion_receipts/"
        f"{activation_intent.document['intent_id']}.json"
    )
    completion_stored = store.read_optional(completion_path)
    if completion_stored is None:
        raise RuntimeControlError(
            "v4 formal pointer is pending completion"
        )
    try:
        from quant_investor.v17_v4_contract import (
            load_canonical_artifact,
        )

        completion_document = decode_reference(
            completion_stored.data,
            expected_version=(
                "myquant.v17.v4.formal-activation-receipt.v1"
            ),
        )
        validated_completion = load_canonical_artifact(
            completion_stored.data,
            expected_version=(
                "myquant.v17.v4.formal-activation-receipt.v1"
            ),
            label="v4 formal completion receipt",
        )
    except (ImportError, ValueError, ControlStorageError) as exc:
        raise RuntimeControlError(
            "v4 formal completion receipt is invalid"
        ) from exc
    activation_receipt = ControlArtifact(
        completion_path,
        completion_document,
        completion_stored.data,
        completion_stored.byte_sha256,
        formal_pointer.strategy_id,
        formal_pointer.cutoff,
    )
    if (
        dict(validated_completion.payload)
        != dict(activation_receipt.document)
        or activation_receipt.document.get("status")
        != "FORMAL_ACTIVATED"
        or activation_receipt.document.get("to_state") != "FORMAL_ACTIVE"
        or activation_receipt.document.get("intent_ref")
        != activation_intent.reference
        or activation_receipt.document.get("pointer_ref")
        != formal_pointer.reference
        or activation_receipt.document.get("post_readback_sha256")
        != formal_pointer.byte_sha256
        or activation_receipt.document.get("proposed_pointer_sha256")
        != formal_pointer.byte_sha256
        or activation_intent.document.get("formal_output_ref")
        != formal_output.reference
    ):
        raise RuntimeControlError(
            "v4 formal activation does not bind the immutable output"
        )
    eligible_pointer = _read_v4_typed_reference(
        store,
        evidence["myquant.v17.v4.default-eligible-pointer.v1"],
        expected_version="myquant.v17.v4.default-eligible-pointer.v1",
    )
    eligibility_intent_id = str(
        eligible_pointer.document["intent_ref"]["artifact_id"]
    )
    eligibility_intent, eligibility_receipt = (
        _read_v4_completion_chain(
            store=store,
            pointer=eligible_pointer,
            intent_version=(
                "myquant.v17.v4.default-eligibility-intent.v1"
            ),
            completion_version=(
                "myquant.v17.v4.default-eligibility-receipt.v1"
            ),
            completion_path=(
                "results/v17_v4_formal_research/strategies/"
                f"{eligible_pointer.strategy_id}/eligibility/"
                "completion_receipts/"
                f"{eligibility_intent_id}.json"
            ),
            expected_status="DEFAULT_ELIGIBLE",
        )
    )
    if (
        eligibility_intent.document.get("intent_id")
        != eligibility_intent_id
        or eligibility_intent.document.get("formal_active_pointer_ref")
        != formal_pointer.reference
        or eligibility_receipt.document.get("to_state") != "DEFAULT_ELIGIBLE"
    ):
        raise RuntimeControlError("v4 eligibility evidence is not live")
    canary_pointer = _read_v4_typed_reference(
        store,
        evidence["myquant.v17.v4.canary-pointer.v1"],
        expected_version="myquant.v17.v4.canary-pointer.v1",
    )
    canary_intent_id = str(
        canary_pointer.document["intent_ref"]["artifact_id"]
    )
    canary_intent, canary_receipt = _read_v4_completion_chain(
        store=store,
        pointer=canary_pointer,
        intent_version="myquant.v17.v4.canary-transition-intent.v1",
        completion_version="myquant.v17.v4.canary-receipt.v1",
        completion_path=(
            "results/v17_v4_canary/strategies/"
            f"{canary_pointer.strategy_id}/transitions/"
            f"completion_receipts/{canary_intent_id}.json"
        ),
        expected_status="CANARY_COMPLETED",
    )
    paired_run_ids = canary_intent.document.get("paired_run_ids")
    if (
        canary_intent.document.get("intent_id") != canary_intent_id
        or canary_intent.document.get("transition") != "COMPLETE"
        or canary_intent.document.get("eligibility_pointer_ref")
        != eligible_pointer.reference
        or type(paired_run_ids) is not list
        or len(paired_run_ids) != 5
        or canary_intent.document.get("v15_protocol_target_ref")
        != expected_target.reference
        or len(canary_intent.document.get("comparison_refs", ())) != 5
        or len(canary_intent.document.get("completed_sessions", ())) != 5
        or any(
            row.get("status") != "PASS"
            for row in canary_intent.document.get(
                "threshold_results",
                (),
            )
        )
        or any(
            value != 0
            for value in canary_intent.document.get(
                "side_effect_counters",
                {},
            ).values()
        )
        or canary_receipt.document.get("to_state") != "CANARY"
    ):
        raise RuntimeControlError("v4 completed-canary evidence is not live")


def _validate_reference_scope(
    reference: Mapping[str, Any],
    *,
    strategy_id: str,
    cutoff: str,
) -> None:
    if (
        reference.get("strategy_id") != strategy_id
        or type(reference.get("cutoff")) is not str
        or str(reference["cutoff"]) > cutoff
    ):
        raise RuntimeControlError(
            "artifact reference strategy or cutoff mismatch"
        )


def _render_active_path(
    target: Mapping[str, Any],
    strategy_id: str,
) -> str:
    template = target.get("active_run_pointer_template")
    if type(template) is not str or template.count("{strategy_id}") != 1:
        raise RuntimeControlError("active-run pointer template is invalid")
    path = template.replace("{strategy_id}", strategy_id)
    if "{" in path or "}" in path:
        raise RuntimeControlError("active-run pointer template is unresolved")
    return path


def _target_lock(
    target: Mapping[str, Any],
    strategy_id: str,
) -> tuple[str, bool]:
    protocol_id = target.get("protocol_id")
    if protocol_id == "v15":
        return str(V15ActiveRunPublisher.lock_path(strategy_id)), False
    if protocol_id == "myquant.v17.v4":
        return (
            "results/v17_v4_formal_research/strategies/"
            f"{strategy_id}/.active.lock",
            True,
        )
    raise RuntimeControlError("target protocol is invalid")


def _validate_pointer_run(
    *,
    store: ControlStore,
    target: ControlArtifact,
    pointer: ControlArtifact,
    run: ControlArtifact,
    strategy_id: str,
) -> None:
    protocol_id = target.document.get("protocol_id")
    run_ref = _pointer_run_ref(pointer.document, store=store)
    if protocol_id == "myquant.v17.v4":
        if (
            pointer.document.get("protocol_version") != protocol_id
            or pointer.document.get("state") != "PENDING_COMPLETION"
            or pointer.document.get("strategy_id") != strategy_id
            or pointer.document.get("cutoff") != run_ref.get("cutoff")
            or run_ref != run.reference
            or run.reference.get("strategy_id") != strategy_id
            or run.reference.get("cutoff") != pointer.document.get("cutoff")
        ):
            raise RuntimeControlError(
                "v4 formal pointer or immutable output binding mismatch"
            )
        return
    if (
        pointer.document.get("protocol_id")
        != protocol_id
        or pointer.document.get("strategy_id") != strategy_id
        or run_ref != run.reference
        or pointer.document.get("cutoff") != run.document.get("cutoff")
        or run.document.get("protocol_id")
        != target.document.get("protocol_id")
        or run.document.get("strategy_id") != strategy_id
        or run.document.get("status") != "HEALTHY"
    ):
        raise RuntimeControlError(
            "target active pointer or immutable run binding mismatch"
        )


def _decode_active_pointer(
    raw: bytes,
    *,
    protocol_id: str,
) -> dict[str, Any]:
    expected = {
        "v15": ACTIVE_POINTER_VERSION,
        "myquant.v17.v4": V4_FORMAL_ACTIVE_POINTER_VERSION,
    }.get(protocol_id)
    if expected is None:
        raise RuntimeControlError("active-pointer protocol is closed")
    return decode_reference(raw, expected_version=expected)


def _pointer_run_ref(
    pointer: Mapping[str, Any],
    *,
    store: ControlStore,
) -> Mapping[str, Any]:
    version = pointer.get("version")
    if version == ACTIVE_POINTER_VERSION:
        reference = pointer.get("run_ref")
    elif version == V4_FORMAL_ACTIVE_POINTER_VERSION:
        intent_ref = pointer.get("intent_ref")
        if not isinstance(intent_ref, Mapping):
            raise RuntimeControlError(
                "v4 active pointer intent reference is invalid"
            )
        intent = _read_v4_typed_reference(
            store,
            intent_ref,
            expected_version=(
                "myquant.v17.v4.formal-activation-intent.v1"
            ),
        )
        reference = intent.document.get("formal_output_ref")
    else:
        raise RuntimeControlError("active pointer version is closed")
    if not isinstance(reference, Mapping):
        raise RuntimeControlError("active pointer run reference is invalid")
    return reference


__all__ = [
    "ControlArtifact",
    "ResearchRuntimeControl",
    "RuntimeControlError",
    "RuntimeControlThirdState",
    "SELECTOR_LOCK",
    "SELECTOR_PATH",
    "SimulatedTransitionCrash",
    "TransitionOutcome",
    "V15ActiveRunPublisher",
]

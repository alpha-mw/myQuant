"""Generation-bound, read-only stable Mainline facade."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.intelligence._common import (
    artifact_ref,
    business_identity,
    canonical_value,
    identifier,
    require_artifact_ref,
    timestamp,
    validate_artifact_ref,
    validate_stable_artifact,
)
from quant_investor.intelligence.runtime import validate_readiness
from quant_investor.system import (
    SystemStore,
    validate_custody_record,
    validate_source_verification_snapshot,
    validate_validation_completion,
    validate_validation_intent,
    validate_validation_prepared,
)

from .candidate import validate_mainline_candidate
from .errors import (
    MAINLINE_ARGUMENTS_INVALID,
    MAINLINE_BLOCKED,
    MAINLINE_UNINITIALIZED,
    MainlineError,
)
from .readiness import validate_mainline_readiness

_POINTER_FIELDS = frozenset(
    {
        "activated_at",
        "generation_id",
        "manifest_sha256",
        "os_actor",
        "previous_pointer_sha256",
    }
)
_READINESS_KINDS = frozenset({"intelligence_readiness", "system.readiness"})
_ACTIVE_ADMISSION_ROUTES = frozenset({"BOOTSTRAP_EXCEPTION", "PROSPECTIVE_ADMISSION"})
_VALIDATION_RESOLUTION_FIELDS = frozenset(
    {
        "completion_sha256",
        "contextual_result",
        "contextual_result_ref",
        "custody_record",
        "outcome",
        "source_verification_snapshot",
        "validation_attestation",
        "validation_attestation_ref",
        "validation_completion",
        "validation_intent",
        "validation_prepared",
        "validation_request",
        "validation_request_ref",
    }
)


def _blocker_rows(value: Any, *, fallback: str) -> list[str]:
    if type(value) is not list or any(type(row) is not str or not row for row in value):
        return [fallback]
    if len(value) != len(set(value)):
        return [fallback]
    return list(value)


def _status(
    *,
    status: str,
    mainline_state: str,
    investment_state: str,
    blockers: list[str],
    active_generation_id: str | None,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "active_generation_id": active_generation_id,
        "blockers": sorted(set(blockers), key=lambda item: item.encode("utf-8")),
        "investment_state": investment_state,
        "mainline_state": mainline_state,
        "result": None if result is None else dict(result),
        "status": status,
    }


def _bound_generation_id(value: Mapping[str, Any]) -> str | None:
    generation_id = value.get("generation_id")
    manifest_sha256 = value.get("manifest_sha256")
    pointer = value.get("pointer")
    if (
        type(generation_id) is not str
        or len(generation_id) != 64
        or any(character not in "0123456789abcdef" for character in generation_id)
        or type(manifest_sha256) is not str
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
        or type(pointer) is not dict
        or set(pointer) != set(_POINTER_FIELDS)
        or pointer.get("generation_id") != generation_id
        or pointer.get("manifest_sha256") != manifest_sha256
    ):
        return None
    return generation_id


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _validated_factor_sources(
    value: Mapping[str, Any],
    *,
    manifest_payload: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> list[dict[str, str]]:
    source_ref_values = value.get("factor_source_object_refs")
    source_objects = value.get("factor_source_objects")
    if type(source_ref_values) is not list or type(source_objects) is not list:
        raise ValueError("protected validation source closure is invalid")
    source_refs = [
        validate_artifact_ref(row, label=f"factor_source_object_refs[{index}]")
        for index, row in enumerate(source_ref_values)
    ]
    resolved_source_refs = [
        artifact_ref(validate_stable_artifact(row, expected_kind="system.source_object"))
        for row in source_objects
    ]
    if (
        not source_refs
        or source_refs != resolved_source_refs
        or source_refs != manifest_payload.get("factor_source_object_refs")
        or source_refs != snapshot.get("source_object_refs")
    ):
        raise ValueError("protected validation source binding is invalid")
    return source_refs


def _validated_factor_roots(
    value: Mapping[str, Any],
    *,
    manifest_payload: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], list[dict[str, str]]]:
    policy = validate_stable_artifact(value.get("factor_policy"))
    active_set = validate_stable_artifact(value.get("factor_active_set"))
    evidence_value = value.get("factor_evidence")
    if type(evidence_value) is not list or not evidence_value:
        raise ValueError("Factor evidence closure is invalid")
    evidence = [validate_stable_artifact(row) for row in evidence_value]
    policy_ref = artifact_ref(policy)
    active_set_ref = artifact_ref(active_set)
    evidence_refs = [artifact_ref(row) for row in evidence]
    if (
        not policy_ref["kind"].startswith("factor.")
        or not active_set_ref["kind"].startswith("factor.")
        or policy_ref != manifest_payload.get("factor_policy_ref")
        or active_set_ref != manifest_payload.get("factor_active_set_ref")
        or evidence_refs != manifest_payload.get("factor_evidence_refs")
    ):
        raise ValueError("Factor manifest root binding is invalid")
    return policy_ref, active_set_ref, evidence_refs


def _validate_factor_status_authorization(
    factor_status: Mapping[str, Any],
    *,
    readiness_payload: Mapping[str, Any],
    active_set_ref: Mapping[str, Any],
    receipt_ref: Mapping[str, Any],
    context_ref: Mapping[str, Any],
    attestation_ref: Mapping[str, Any],
) -> None:
    status_payload = factor_status.get("payload")
    status_active = status_payload.get("active") if type(status_payload) is dict else None
    if type(status_active) is not dict:
        raise ValueError("Factor status active projection is invalid")
    status_bindings = (
        (status_active.get("factor_set_ref"), active_set_ref),
        (status_active.get("validation_receipt_ref"), receipt_ref),
        (status_active.get("contextual_result_ref"), context_ref),
        (status_active.get("validation_attestation_ref"), attestation_ref),
    )
    if any(
        validate_artifact_ref(observed, label="factor_status.active.ref") != expected
        for observed, expected in status_bindings
    ):
        raise ValueError("Factor status validation binding is invalid")
    status_blockers = status_payload.get("blockers")
    readiness_blockers = readiness_payload.get("blockers")
    if (
        status_active.get("state") != "ACTIVE"
        or status_payload.get("readiness") != readiness_payload.get("factor_state")
        or status_active.get("admission_route") != readiness_payload.get("admission_route")
        or status_active.get("producer_identity") != readiness_payload.get("producer_identity")
        or type(status_blockers) is not list
        or type(readiness_blockers) is not list
        or not set(status_blockers) <= set(readiness_blockers)
        or status_payload.get("activation_mutation_authorized") is not False
    ):
        raise ValueError("Factor status/readiness projection is invalid")


def _validate_factor_authorization(
    value: Mapping[str, Any],
    *,
    manifest_payload: Mapping[str, Any],
    factor_status: Mapping[str, Any],
    readiness_payload: Mapping[str, Any],
) -> None:
    try:
        resolution_value = value.get("factor_validation_resolution")
        if (
            type(resolution_value) is not dict
            or set(resolution_value) != set(_VALIDATION_RESOLUTION_FIELDS)
            or resolution_value.get("outcome") != "VALIDATED"
        ):
            raise ValueError("protected validation projection is invalid")
        resolution = dict(resolution_value)

        request = validate_stable_artifact(
            resolution["validation_request"],
            expected_kind="system.validation_run_request",
        )
        request_ref = artifact_ref(request)
        context = validate_stable_artifact(
            resolution["contextual_result"],
            expected_kind="factor.contextual_validation_result",
        )
        context_ref = artifact_ref(context)
        attestation = validate_stable_artifact(
            resolution["validation_attestation"],
            expected_kind="system.validation_attestation",
        )
        attestation_ref = artifact_ref(attestation)
        receipt = validate_stable_artifact(
            value.get("factor_validation_receipt"),
            expected_kind="factor.validation_receipt",
        )
        receipt_ref = artifact_ref(receipt)
        policy_ref, active_set_ref, evidence_refs = _validated_factor_roots(
            value,
            manifest_payload=manifest_payload,
        )

        intent = validate_validation_intent(resolution["validation_intent"])
        prepared = validate_validation_prepared(resolution["validation_prepared"])
        custody = validate_custody_record(resolution["custody_record"])
        snapshot = validate_source_verification_snapshot(resolution["source_verification_snapshot"])
        completion = validate_validation_completion(resolution["validation_completion"])

        reference_bindings = (
            (resolution["validation_request_ref"], request_ref),
            (resolution["contextual_result_ref"], context_ref),
            (resolution["validation_attestation_ref"], attestation_ref),
            (value.get("factor_validation_receipt_ref"), receipt_ref),
            (value.get("factor_validation_attestation_ref"), attestation_ref),
            (manifest_payload.get("factor_validation_attestation_ref"), attestation_ref),
        )
        if any(
            validate_artifact_ref(observed, label="factor_validation_ref") != expected
            for observed, expected in reference_bindings
        ):
            raise ValueError("protected validation ref binding is invalid")

        if (
            value.get("factor_validation_attestation") != attestation
            or value.get("factor_contextual_result") != context
            or value.get("factor_validation_completion") != completion
            or value.get("factor_source_verification_snapshot") != snapshot
        ):
            raise ValueError("protected validation direct projection is invalid")

        source_refs = _validated_factor_sources(
            value,
            manifest_payload=manifest_payload,
            snapshot=snapshot,
        )
        _validate_factor_status_authorization(
            factor_status,
            readiness_payload=readiness_payload,
            active_set_ref=active_set_ref,
            receipt_ref=receipt_ref,
            context_ref=context_ref,
            attestation_ref=attestation_ref,
        )

        request_payload = request["payload"]
        receipt_payload = receipt["payload"]
        context_payload = context["payload"]
        attestation_payload = attestation["payload"]
        if (
            receipt_payload.get("validated") is not True
            or receipt_payload.get("authority") != "NON_AUTHORIZING"
            or receipt_payload.get("policy_ref") != policy_ref
            or receipt_payload.get("active_set_ref") != active_set_ref
            or receipt_payload.get("evidence_refs") != evidence_refs
            or context_payload.get("validated") is not True
            or context_payload.get("blockers") != []
            or context_payload.get("authority") != "NON_AUTHORIZING"
            or context_payload.get("intrinsic_receipt_ref") != receipt_ref
            or context_payload.get("policy_ref") != policy_ref
            or context_payload.get("active_set_ref") != active_set_ref
            or context_payload.get("evidence_refs") != evidence_refs
            or attestation_payload.get("outcome") != "VALIDATED"
            or attestation_payload.get("authority") != "NON_AUTHORIZING"
            or attestation_payload.get("validation_request_ref") != request_ref
            or attestation_payload.get("contextual_result_ref") != context_ref
            or attestation_payload.get("intrinsic_receipt_ref") != receipt_ref
            or attestation_payload.get("policy_ref") != policy_ref
            or attestation_payload.get("active_set_ref") != active_set_ref
            or attestation_payload.get("evidence_refs") != evidence_refs
            or attestation_payload.get("source_object_refs") != source_refs
            or attestation_payload.get("validation_lane") != context_payload.get("lane")
            or request_payload.get("intrinsic_receipt_ref") != receipt_ref
            or request_payload.get("release_manifest_ref")
            != manifest_payload.get("release_manifest_ref")
            or request_payload.get("factor_validator_manifest_ref")
            != context_payload.get("factor_validator_manifest_ref")
            or request_payload.get("candidate_state_ref")
            != context_payload.get("composite_state_ref")
        ):
            raise ValueError("protected validation artifact closure is invalid")

        intent_sha = _canonical_sha256(intent)
        prepared_sha = _canonical_sha256(prepared)
        custody_sha = _canonical_sha256(custody)
        snapshot_sha = _canonical_sha256(snapshot)
        completion_sha = _canonical_sha256(completion)
        if (
            intent.get("validation_request_ref") != request_ref
            or prepared.get("validation_request_ref") != request_ref
            or prepared.get("contextual_result_ref") != context_ref
            or prepared.get("validation_attestation_ref") != attestation_ref
            or custody.get("validation_request_ref") != request_ref
            or custody.get("contextual_result_ref") != context_ref
            or custody.get("attestation_ref") != attestation_ref
            or snapshot.get("validation_attestation_ref") != attestation_ref
            or completion.get("validation_request_ref") != request_ref
            or completion.get("contextual_result_ref") != context_ref
            or completion.get("validation_attestation_ref") != attestation_ref
            or prepared.get("intent_sha256") != intent_sha
            or completion.get("intent_sha256") != intent_sha
            or completion.get("prepared_sha256") != prepared_sha
            or completion.get("custody_record_sha256") != custody_sha
            or completion.get("source_verification_snapshot_sha256") != snapshot_sha
            or resolution.get("completion_sha256") != completion_sha
            or attestation_payload.get("validation_intent_sha256") != intent_sha
        ):
            raise ValueError("protected validation completion closure is invalid")
    except MainlineError:
        raise
    except Exception as exc:
        raise MainlineError(
            MAINLINE_BLOCKED,
            blockers=["FACTOR_VALIDATION_CLOSURE_INVALID"],
        ) from exc


def _resolved_active(  # noqa: C901 - complete generation/ref verification boundary
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        if type(value) is not dict or value.get("verified") is not True:
            raise MainlineError(MAINLINE_BLOCKED, blockers=["GENERATION_NOT_VERIFIED"])
        if (
            value.get("generation_state") == "OPERATIONAL"
            and value.get("deployed_release_verified") is not True
        ):
            raise MainlineError(
                MAINLINE_BLOCKED,
                blockers=["DEPLOYED_RELEASE_NOT_VERIFIED"],
            )
        generation_id = value.get("generation_id")
        if type(generation_id) is not str or len(generation_id) != 64:
            raise MainlineError(MAINLINE_BLOCKED, blockers=["GENERATION_ID_INVALID"])
        manifest = validate_stable_artifact(
            value.get("manifest"), expected_kind="system.generation_manifest"
        )
        if manifest.get("semantic_sha256") != generation_id:
            raise MainlineError(MAINLINE_BLOCKED, blockers=["GENERATION_BINDING_INVALID"])
        manifest_ref = artifact_ref(manifest)
        manifest_sha256 = value.get("manifest_sha256")
        if manifest_ref["byte_sha256"] != manifest_sha256:
            raise MainlineError(MAINLINE_BLOCKED, blockers=["GENERATION_MANIFEST_BYTES_INVALID"])
        pointer = value.get("pointer")
        if type(pointer) is not dict or set(pointer) != set(_POINTER_FIELDS):
            raise MainlineError(MAINLINE_BLOCKED, blockers=["ACTIVE_POINTER_INVALID"])
        if (
            pointer.get("generation_id") != generation_id
            or pointer.get("manifest_sha256") != manifest_sha256
        ):
            raise MainlineError(MAINLINE_BLOCKED, blockers=["ACTIVE_POINTER_BINDING_INVALID"])
        payload = manifest.get("payload")
        if type(payload) is not dict:
            raise MainlineError(MAINLINE_BLOCKED, blockers=["GENERATION_MANIFEST_INVALID"])
        readiness = validate_stable_artifact(value.get("readiness"))
        if readiness.get("kind") not in _READINESS_KINDS:
            raise MainlineError(MAINLINE_BLOCKED, blockers=["READINESS_KIND_INVALID"])
        require_artifact_ref(
            payload.get("readiness_matrix_ref"),
            readiness,
            label="manifest.readiness_matrix_ref",
        )
        mainline = value.get("mainline")
        mainline_ref = payload.get("mainline_ref")
        if mainline is None:
            if mainline_ref is not None:
                raise MainlineError(MAINLINE_BLOCKED, blockers=["MAINLINE_BINDING_MISSING"])
            candidate = None
        else:
            candidate = validate_mainline_candidate(mainline)
            require_artifact_ref(mainline_ref, candidate, label="manifest.mainline_ref")
            research = value.get("research")
            if type(research) is not list:
                raise MainlineError(
                    MAINLINE_BLOCKED, blockers=["MAINLINE_RESEARCH_CLOSURE_INVALID"]
                )
            research_refs = [artifact_ref(validate_stable_artifact(row)) for row in research]
            if research_refs != payload.get("research_refs"):
                raise MainlineError(
                    MAINLINE_BLOCKED, blockers=["MAINLINE_RESEARCH_CLOSURE_INVALID"]
                )
            if len(research_refs) != len(
                {(row["kind"], row["artifact_id"]) for row in research_refs}
            ):
                raise MainlineError(
                    MAINLINE_BLOCKED, blockers=["MAINLINE_RESEARCH_CLOSURE_INVALID"]
                )
            candidate_payload = candidate["payload"]
            required_refs = {
                tuple(sorted(candidate_payload[field].items()))
                for field in (
                    "decision_ref",
                    "evidence_bundle_ref",
                    "portfolio_ref",
                )
            }
            resolved_refs = {tuple(sorted(row.items())) for row in research_refs}
            if not required_refs.issubset(resolved_refs):
                raise MainlineError(
                    MAINLINE_BLOCKED, blockers=["MAINLINE_RESEARCH_CLOSURE_INVALID"]
                )
        readiness_payload = readiness["payload"]
        factor_ref_value = readiness_payload.get("factor_status_ref")
        mainline_readiness_ref = readiness_payload.get("mainline_candidate_ref")
        if candidate is None:
            if mainline_readiness_ref is not None:
                raise MainlineError(MAINLINE_BLOCKED, blockers=["READINESS_MAINLINE_REF_INVALID"])
            if readiness["kind"] == "intelligence_readiness":
                readiness = validate_readiness(readiness)
        elif mainline_readiness_ref != artifact_ref(candidate):
            raise MainlineError(MAINLINE_BLOCKED, blockers=["READINESS_MAINLINE_REF_MISMATCH"])
        elif readiness["kind"] == "intelligence_readiness":
            readiness = validate_mainline_readiness(readiness)
        else:
            raise MainlineError(MAINLINE_BLOCKED, blockers=["READINESS_KIND_INVALID"])
        factor_status = value.get("factor_status")
        resolved_factor_ref = value.get("factor_status_ref")
        validated_factor_status = None
        if factor_ref_value is None:
            if factor_status is not None or resolved_factor_ref is not None:
                raise MainlineError(MAINLINE_BLOCKED, blockers=["FACTOR_STATUS_REF_MISSING"])
        else:
            from quant_investor.factors.governance import validate_factor_status

            validated_factor_status = validate_factor_status(factor_status)
            factor_ref = validate_artifact_ref(factor_ref_value, label="factor_status_ref")
            if (
                artifact_ref(validated_factor_status) != factor_ref
                or resolved_factor_ref != factor_ref
            ):
                raise MainlineError(MAINLINE_BLOCKED, blockers=["FACTOR_STATUS_REF_MISMATCH"])
        if value.get("generation_state") == "OPERATIONAL":
            if validated_factor_status is None:
                raise MainlineError(
                    MAINLINE_BLOCKED,
                    blockers=["FACTOR_VALIDATION_CLOSURE_INVALID"],
                )
            _validate_factor_authorization(
                value,
                manifest_payload=payload,
                factor_status=validated_factor_status,
                readiness_payload=readiness_payload,
            )
        return {
            **value,
            "mainline": candidate,
            "manifest": manifest,
            "readiness": readiness,
        }
    except MainlineError:
        raise
    except Exception as exc:
        raise MainlineError(MAINLINE_BLOCKED, blockers=["GENERATION_BINDING_INVALID"]) from exc


def mainline_status(
    active: Mapping[str, Any] | None,
    *,
    strategy_id: str,
) -> dict[str, Any]:
    """Derive fail-closed public state from a verified resolved generation."""

    try:
        strategy = identifier(strategy_id, label="strategy_id")
    except Exception:
        return _status(
            status="BLOCKED",
            mainline_state="BLOCKED",
            investment_state="BLOCKED",
            blockers=[MAINLINE_ARGUMENTS_INVALID],
            active_generation_id=None,
        )
    if active is None:
        return _status(
            status="BLOCKED",
            mainline_state="UNINITIALIZED",
            investment_state="BLOCKED",
            blockers=["ACTIVE_GENERATION_ABSENT"],
            active_generation_id=None,
        )
    try:
        resolved = _resolved_active(active)
    except MainlineError as exc:
        return _status(
            status="BLOCKED",
            mainline_state="BLOCKED",
            investment_state="BLOCKED",
            blockers=list(exc.blockers),
            active_generation_id=_bound_generation_id(active),
        )
    readiness = resolved["readiness"]["payload"]
    blockers = _blocker_rows(readiness.get("blockers"), fallback="READINESS_BLOCKERS_INVALID")
    generation_state = resolved.get("generation_state")
    factor_state = readiness.get("factor_state")
    admission_route = readiness.get("admission_route")
    mainline_state = readiness.get("mainline_state")
    investment_state = readiness.get("investment_state")
    candidate = resolved.get("mainline")
    if generation_state != "OPERATIONAL":
        return _status(
            status="BLOCKED",
            mainline_state="BLOCKED",
            investment_state="BLOCKED",
            blockers=blockers or ["GENERATION_NOT_OPERATIONAL"],
            active_generation_id=resolved["generation_id"],
        )
    if factor_state == "READY" and (mainline_state == "UNINITIALIZED" or candidate is None):
        return _status(
            status="BLOCKED",
            mainline_state="UNINITIALIZED",
            investment_state="BLOCKED",
            blockers=blockers or ["MAINLINE_CANDIDATE_ABSENT"],
            active_generation_id=resolved["generation_id"],
        )
    if (
        factor_state != "READY"
        or admission_route not in _ACTIVE_ADMISSION_ROUTES
        or mainline_state != "READY"
        or blockers
        or investment_state != "PAPER_CANDIDATE"
    ):
        return _status(
            status="BLOCKED",
            mainline_state="BLOCKED",
            investment_state="BLOCKED",
            blockers=blockers or ["READINESS_NOT_ACTIVE"],
            active_generation_id=resolved["generation_id"],
        )
    candidate_payload = candidate["payload"]
    if (
        candidate_payload.get("strategy_id") != strategy
        or candidate_payload.get("status") != "CANDIDATE_READY"
        or candidate_payload.get("investment_state") != investment_state
    ):
        return _status(
            status="BLOCKED",
            mainline_state="BLOCKED",
            investment_state="BLOCKED",
            blockers=["MAINLINE_CANDIDATE_MISMATCH"],
            active_generation_id=resolved["generation_id"],
        )
    return _status(
        status="ACTIVE",
        mainline_state="ACTIVE",
        investment_state=investment_state,
        blockers=[],
        active_generation_id=resolved["generation_id"],
        result=candidate_payload["result"],
    )


def _public_run(active: Mapping[str, Any], *, strategy_id: str) -> dict[str, Any]:
    state = mainline_status(active, strategy_id=strategy_id)
    if state["status"] != "ACTIVE":
        code = (
            MAINLINE_ARGUMENTS_INVALID
            if MAINLINE_ARGUMENTS_INVALID in state["blockers"]
            else (
                MAINLINE_UNINITIALIZED
                if state["mainline_state"] == "UNINITIALIZED"
                else MAINLINE_BLOCKED
            )
        )
        raise MainlineError(code, blockers=state["blockers"], public_state=state)
    resolved = _resolved_active(active)
    candidate = resolved["mainline"]
    readiness = resolved["readiness"]
    strategy = identifier(strategy_id, label="strategy_id")
    generation_id = resolved["generation_id"]
    created_at = timestamp(resolved["manifest"].get("created_at"), label="manifest.created_at")
    run_id = business_identity(
        kind="public_run",
        identity_inputs={"generation_id": generation_id, "strategy_id": strategy},
    )
    payload = {
        "candidate_ref": artifact_ref(candidate),
        "active_generation_id": generation_id,
        "investment_state": candidate["payload"]["investment_state"],
        "readiness_ref": artifact_ref(readiness),
        "result": dict(candidate["payload"]["result"]),
        "run_id": run_id,
        "status": "ACTIVE",
        "strategy_id": strategy,
    }
    canonical_value(payload)
    return validate_stable_artifact(
        seal_artifact("public_run", payload, created_at=created_at),
        expected_kind="public_run",
    )


class MainlineStore:
    """Read-only public facade over the unified SystemStore."""

    def __init__(
        self,
        workspace_root: str | Any,
        *,
        system_store: SystemStore | None = None,
    ) -> None:
        self._system = system_store or SystemStore(workspace_root)
        self.workspace_root = self._system.workspace_root

    def read_active(self) -> dict[str, Any] | None:
        return self._system.read_active()

    def status(self, *, strategy_id: str) -> dict[str, Any]:
        try:
            active = self.read_active()
        except Exception:
            return _status(
                status="BLOCKED",
                mainline_state="BLOCKED",
                investment_state="BLOCKED",
                blockers=["SYSTEM_ACTIVE_READ_BLOCKED"],
                active_generation_id=None,
            )
        return mainline_status(active, strategy_id=strategy_id)

    def read_public_run(self, *, strategy_id: str) -> dict[str, Any]:
        try:
            active = self.read_active()
        except Exception as exc:
            raise MainlineError(MAINLINE_BLOCKED, blockers=["SYSTEM_ACTIVE_READ_BLOCKED"]) from exc
        if active is None:
            state = _status(
                status="BLOCKED",
                mainline_state="UNINITIALIZED",
                investment_state="BLOCKED",
                blockers=["ACTIVE_GENERATION_ABSENT"],
                active_generation_id=None,
            )
            raise MainlineError(
                MAINLINE_UNINITIALIZED,
                blockers=state["blockers"],
                public_state=state,
            )
        return _public_run(active, strategy_id=strategy_id)


def read_public_run(workspace_root: str | Any, *, strategy_id: str) -> dict[str, Any]:
    """Read one official generation-bound result; never build or activate it."""

    try:
        identifier(strategy_id, label="strategy_id")
    except Exception as exc:
        raise MainlineError(
            MAINLINE_ARGUMENTS_INVALID, blockers=[MAINLINE_ARGUMENTS_INVALID]
        ) from exc
    return MainlineStore(workspace_root).read_public_run(strategy_id=strategy_id)


__all__ = ["MainlineStore", "mainline_status", "read_public_run"]

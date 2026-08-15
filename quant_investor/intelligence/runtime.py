"""Stable inactive research lifecycle for Intelligence.

These callables replace the secondary versioned CLI surface.  They consume
caller-supplied, already available data and never call a Provider, model,
broker, order, trade, or activation path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from quant_investor.contracts import seal_artifact

from ._common import (
    IntelligenceError,
    artifact_payload,
    artifact_ref,
    artifact_identity,
    build_artifact,
    business_identity,
    canonical_value,
    identifier,
    require_no_future,
    timestamp,
    validate_artifact_ref,
    validate_stable_artifact,
)

RESEARCH_STAGE_STATUSES: Final = frozenset({"COMPLETE", "BLOCKED", "NOT_RUN"})
_ACTIVE_ADMISSION_ROUTES: Final = frozenset({"BOOTSTRAP_EXCEPTION", "PROSPECTIVE_ADMISSION"})


def _stages(values: Sequence[Any]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceError("stages must be a sequence")
    stages = [identifier(value, label=f"stages[{index}]") for index, value in enumerate(values)]
    if not stages or len(stages) != len(set(stages)):
        raise IntelligenceError("stages must be nonempty and unique")
    return stages


def _refs(values: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, str]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IntelligenceError(f"{label} must be a sequence")
    rows = [
        validate_artifact_ref(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    ]
    identities = [(row["kind"], row["artifact_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise IntelligenceError(f"{label} is duplicated")
    return sorted(
        rows,
        key=lambda row: (row["kind"].encode("ascii"), row["artifact_id"].encode("ascii")),
    )


def forward(
    request: Mapping[str, Any],
    *,
    created_at: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Validate and seal one inactive forward-research request."""

    required = {"as_of", "input_refs", "stages", "strategy_id"}
    if type(request) is not dict or set(request) != required:
        raise IntelligenceError("forward request shape is invalid")
    cutoff = timestamp(request["as_of"], label="request.as_of")
    instant = timestamp(created_at or cutoff, label="created_at")
    if instant < cutoff:
        raise IntelligenceError("forward request cannot predate its research cutoff")
    strategy = identifier(request["strategy_id"], label="strategy_id")
    stages = _stages(request["stages"])
    input_refs = _refs(request["input_refs"], label="input_refs")
    return build_artifact(
        kind="research_request",
        identity_field="request_id",
        identity=request_id
        or business_identity(
            kind="research_request",
            identity_inputs={"as_of": cutoff, "strategy_id": strategy},
        ),
        created_at=instant,
        fields={
            "as_of": cutoff,
            "input_refs": input_refs,
            "stages": stages,
            "status": "ACCEPTED_INACTIVE",
            "strategy_id": strategy,
        },
    )


def evaluate(
    request: Mapping[str, Any] | bytes,
    *,
    stage_results: Mapping[str, Mapping[str, Any]],
    evaluated_at: str,
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate precomputed stage results; callbacks and external calls are forbidden."""

    request_artifact, request_payload = artifact_payload(request, expected_kind="research_request")
    instant = timestamp(evaluated_at, label="evaluated_at")
    require_no_future(request_artifact, as_of=instant, label="research_request")
    stages = request_payload.get("stages")
    if type(stage_results) is not dict or set(stage_results) != set(stages):
        raise IntelligenceError("stage_results must exactly close the requested stages")
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for stage in stages:
        value = stage_results[stage]
        if type(value) is not dict or set(value) != {
            "blocker_codes",
            "output_refs",
            "status",
        }:
            raise IntelligenceError(f"stage_results.{stage} shape is invalid")
        status = value["status"]
        if status not in RESEARCH_STAGE_STATUSES:
            raise IntelligenceError(f"stage_results.{stage}.status is invalid")
        codes = value["blocker_codes"]
        if isinstance(codes, (str, bytes)) or not isinstance(codes, Sequence):
            raise IntelligenceError(f"stage_results.{stage}.blocker_codes must be a sequence")
        blocker_codes = sorted(
            {identifier(code, label=f"stage_results.{stage}.blocker_codes") for code in codes},
            key=lambda item: item.encode("ascii"),
        )
        if len(blocker_codes) != len(value["blocker_codes"]):
            raise IntelligenceError(f"stage_results.{stage}.blocker_codes is duplicated")
        output_refs = _refs(value["output_refs"], label=f"stage_results.{stage}.output_refs")
        if status == "COMPLETE" and blocker_codes:
            raise IntelligenceError("complete research stage cannot retain blockers")
        if status != "COMPLETE" and not blocker_codes:
            blocker_codes = [f"{stage.upper()}_{status}"]
        blockers.extend(blocker_codes)
        rows.append(
            {
                "blocker_codes": blocker_codes,
                "output_refs": output_refs,
                "stage": stage,
                "status": status,
            }
        )
    status = "COMPLETE" if not blockers else "BLOCKED"
    return build_artifact(
        kind="research_evaluation",
        identity_field="evaluation_id",
        identity=evaluation_id
        or business_identity(
            kind="research_evaluation",
            identity_inputs={"request_id": request_artifact["artifact_id"]},
        ),
        created_at=instant,
        fields={
            "blocker_codes": sorted(set(blockers), key=lambda item: item.encode("ascii")),
            "evaluated_at": instant,
            "request_ref": artifact_ref(request_artifact),
            "stage_rows": rows,
            "status": status,
            "strategy_id": request_payload["strategy_id"],
        },
    )


def compile_evidence(
    evaluation: Mapping[str, Any] | bytes,
    *,
    evidence: Sequence[Mapping[str, Any] | bytes],
    compiled_at: str,
    bundle_id: str | None = None,
) -> dict[str, Any]:
    """Compile exact prepositioned evidence artifacts into one inactive bundle."""

    evaluation_artifact, evaluation_payload = artifact_payload(
        evaluation, expected_kind="research_evaluation"
    )
    instant = timestamp(compiled_at, label="compiled_at")
    require_no_future(evaluation_artifact, as_of=instant, label="research_evaluation")
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise IntelligenceError("evidence must be a sequence")
    references: list[dict[str, str]] = []
    for index, value in enumerate(evidence):
        artifact = validate_stable_artifact(value)
        require_no_future(artifact, as_of=instant, label=f"evidence[{index}]")
        reference = artifact_ref(artifact)
        if reference["kind"] in {
            "intelligence_readiness",
            "mainline_candidate",
            "public_run",
            "system.generation_manifest",
            "system.migration.complete",
            "system.migration.receipt",
            "system.readiness",
        }:
            raise IntelligenceError("evidence bundle cannot depend on activation state")
        references.append(reference)
    identities = [(row["kind"], row["artifact_id"]) for row in references]
    if len(identities) != len(set(identities)):
        raise IntelligenceError("evidence artifact closure is duplicated")
    references.sort(
        key=lambda row: (row["kind"].encode("ascii"), row["artifact_id"].encode("ascii"))
    )
    evaluation_blockers = evaluation_payload.get("blocker_codes")
    if (
        type(evaluation_blockers) is not list
        or any(
            identifier(code, label="evaluation.blocker_codes") != code
            for code in evaluation_blockers
        )
        or len(evaluation_blockers) != len(set(evaluation_blockers))
    ):
        raise IntelligenceError("research evaluation blockers are invalid")
    blockers = list(evaluation_blockers)
    if not references:
        blockers.append("EVIDENCE_CLOSURE_EMPTY")
    status = "READY" if not blockers else "BLOCKED"
    return build_artifact(
        kind="evidence_bundle",
        identity_field="bundle_id",
        identity=bundle_id
        or business_identity(
            kind="evidence_bundle",
            identity_inputs={"evaluation_id": evaluation_artifact["artifact_id"]},
        ),
        created_at=instant,
        fields={
            "blocker_codes": sorted(set(blockers), key=lambda item: item.encode("ascii")),
            "compiled_at": instant,
            "evaluation_ref": artifact_ref(evaluation_artifact),
            "evidence_refs": references,
            "status": status,
            "strategy_id": evaluation_payload["strategy_id"],
        },
    )


def _factor_payload(
    artifact: Mapping[str, Any] | bytes | None,
) -> tuple[str, str, str | None, list[str], dict[str, str] | None]:
    if artifact is None:
        return "BLOCKED", "NONE", None, ["FACTOR_STATUS_UNAVAILABLE"], None
    try:
        from quant_investor.factors.governance import validate_factor_status

        normalized = validate_factor_status(artifact)
    except Exception as exc:
        raise IntelligenceError("factor status contract is invalid") from exc
    payload = normalized.get("payload")
    if type(payload) is not dict:
        raise IntelligenceError("factor status payload is invalid")
    readiness = payload.get("readiness")
    blockers = payload.get("blockers")
    if (
        readiness not in {"READY", "BLOCKED"}
        or type(blockers) is not list
        or any(type(code) is not str or not code for code in blockers)
        or len(blockers) != len(set(blockers))
    ):
        raise IntelligenceError("factor status readiness is invalid")
    active = payload.get("active")
    route = "NONE"
    factor_producer = None
    if readiness == "READY":
        if type(active) is not dict:
            raise IntelligenceError("ready Factor status has no active set")
        route = active.get("admission_route")
        if route not in {"BOOTSTRAP_EXCEPTION", "PROSPECTIVE_ADMISSION"}:
            raise IntelligenceError("active Factor admission route is invalid")
        factor_producer = identifier(
            active.get("producer_identity"), label="factor producer_identity"
        )
    return (
        str(readiness),
        str(route),
        factor_producer,
        list(blockers),
        artifact_ref(normalized),
    )


def _validate_readiness_factor_projection(payload: Mapping[str, Any]) -> tuple[str, str]:
    producer = identifier(payload.get("producer_identity"), label="producer_identity")
    factor_state = payload.get("factor_state")
    route = payload.get("admission_route")
    factor_status_ref = payload.get("factor_status_ref")
    if factor_state not in {"READY", "BLOCKED"}:
        raise IntelligenceError("readiness Factor state is invalid")
    if factor_status_ref is not None:
        factor_status_ref = validate_artifact_ref(
            factor_status_ref,
            label="factor_status_ref",
        )
        if factor_status_ref["kind"] != "factor.status":
            raise IntelligenceError("readiness Factor status kind is invalid")
    if factor_state == "READY":
        if route not in _ACTIVE_ADMISSION_ROUTES or factor_status_ref is None:
            raise IntelligenceError("ready readiness Factor binding is invalid")
    elif route != "NONE":
        raise IntelligenceError("blocked readiness admission route is invalid")
    return factor_state, producer


def _validate_readiness_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers = payload.get("blockers")
    if type(blockers) is not list:
        raise IntelligenceError("readiness blockers must be a list")
    normalized_blockers = [
        identifier(code, label=f"blockers[{index}]") for index, code in enumerate(blockers)
    ]
    if normalized_blockers != sorted(
        set(normalized_blockers), key=lambda item: item.encode("ascii")
    ):
        raise IntelligenceError("readiness blockers must be sorted and unique")
    return normalized_blockers


def validate_readiness(
    artifact: Mapping[str, Any] | bytes,
) -> dict[str, Any]:
    """Validate one Intelligence-owned, Mainline-absent readiness artifact."""

    normalized, payload = artifact_payload(
        artifact,
        expected_kind="intelligence_readiness",
    )
    artifact_identity(payload.get("readiness_id"), label="readiness_id")
    factor_state, producer = _validate_readiness_factor_projection(payload)
    normalized_blockers = _validate_readiness_blockers(payload)
    if "MAINLINE_CANDIDATE_ABSENT" not in normalized_blockers:
        raise IntelligenceError("Intelligence readiness must retain Mainline absence")
    if (
        payload.get("mainline_candidate_ref") is not None
        or payload.get("mainline_state") != "UNINITIALIZED"
        or payload.get("investment_state") != "BLOCKED"
    ):
        raise IntelligenceError("Intelligence readiness contains Mainline state")
    if factor_state == "READY" and producer not in {
        "NOT_CLAIMED",
        "PROSPECTIVE_GOVERNANCE",
    }:
        raise IntelligenceError("ready readiness producer is invalid")
    return normalized


def assess_readiness(
    *,
    producer_identity: str,
    assessed_at: str,
    factor_status: Mapping[str, Any] | bytes | None,
    source_blockers: Sequence[str] = (),
    readiness_id: str | None = None,
) -> dict[str, Any]:
    """Build Mainline-absent Intelligence readiness from Factor/source state."""

    producer = identifier(producer_identity, label="producer_identity")
    instant = timestamp(assessed_at, label="assessed_at")
    factor_state, route, factor_producer, blockers, factor_status_ref = _factor_payload(
        factor_status
    )
    if factor_state == "READY" and producer != factor_producer:
        raise IntelligenceError("readiness producer does not match active Factor status")
    if isinstance(source_blockers, (str, bytes)) or not isinstance(source_blockers, Sequence):
        raise IntelligenceError("source_blockers must be a sequence")
    preserved_source_blockers = [
        identifier(code, label=f"source_blockers[{index}]")
        for index, code in enumerate(source_blockers)
    ]
    if len(preserved_source_blockers) != len(set(preserved_source_blockers)):
        raise IntelligenceError("source_blockers must be nonempty and unique")
    blockers.extend(preserved_source_blockers)
    blockers.append("MAINLINE_CANDIDATE_ABSENT")
    blockers = sorted(set(blockers), key=lambda item: item.encode("ascii"))
    identity = readiness_id or business_identity(
        kind="intelligence_readiness",
        identity_inputs={"assessed_at": instant, "producer_identity": producer},
    )
    payload = {
        "admission_route": route,
        "blockers": blockers,
        "factor_state": factor_state,
        "factor_status_ref": factor_status_ref,
        "investment_state": "BLOCKED",
        "mainline_candidate_ref": None,
        "mainline_state": "UNINITIALIZED",
        "producer_identity": producer,
        "readiness_id": artifact_identity(identity, label="readiness_id"),
    }
    canonical_value(payload)
    return validate_readiness(seal_artifact("intelligence_readiness", payload, created_at=instant))


def inspect(
    artifact: Mapping[str, Any] | bytes | None = None,
    *,
    inspected_at: str,
    inspection_id: str | None = None,
) -> dict[str, Any]:
    """Return a no-write inspection artifact for one stable artifact or empty state."""

    instant = timestamp(inspected_at, label="inspected_at")
    target_ref = None
    status = "UNINITIALIZED"
    blockers = ["ARTIFACT_UNAVAILABLE"]
    target_kind = None
    if artifact is not None:
        target = validate_stable_artifact(artifact)
        require_no_future(target, as_of=instant, label="inspected_artifact")
        target_ref = artifact_ref(target)
        target_kind = target["kind"]
        status = "VALID"
        blockers = []
        payload = target.get("payload", {})
        if type(payload) is dict and (
            payload.get("run_state") not in {None, "INACTIVE"}
            or payload.get("production") not in {None, False}
        ):
            status = "BLOCKED"
            blockers = ["ARTIFACT_AUTHORITY_INVALID"]
    return build_artifact(
        kind="intelligence_inspection",
        identity_field="inspection_id",
        identity=inspection_id
        or business_identity(
            kind="intelligence_inspection",
            identity_inputs={
                "inspected_at": instant,
                "target_artifact_id": None if target_ref is None else target_ref["artifact_id"],
            },
        ),
        created_at=instant,
        fields={
            "blocker_codes": blockers,
            "inspected_at": instant,
            "status": status,
            "target_kind": target_kind,
            "target_ref": target_ref,
        },
    )


__all__ = [
    "IntelligenceError",
    "assess_readiness",
    "compile_evidence",
    "evaluate",
    "forward",
    "inspect",
    "validate_readiness",
]

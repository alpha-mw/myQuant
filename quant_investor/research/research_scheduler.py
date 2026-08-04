"""Sanitized daily research loop over explicit provisional-forward requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Final

from quant_investor.v17_v4_contract.canonical import (
    canonical_resource_bytes,
    load_canonical_resource,
)
from quant_investor.v17_v4_contract.schema_validation import validate_artifact
from quant_investor.v17_v4_runtime.provisional_forward import (
    REQUEST_VERSION,
    ProvisionalForwardError,
    _stable_workspace_read,
    run_provisional_forward,
    validate_provisional_request,
)
from quant_investor.v17_v4_runtime.source_storage import GovernedStore

from ._artifacts import (
    NO_AUTHORITY,
    PROTOCOL_VERSION,
    artifact_ref,
    seal,
    sorted_refs,
)
from .research_experiment_registry import build_experiment_registry
from .research_memory import build_memory_entry

RECEIPT_VERSION: Final = "myquant.v17.v4.daily-research-receipt.v1"


class ResearchLoopError(RuntimeError):
    """Fail-closed daily-loop error with preserved upstream artifacts."""

    exit_code = 2

    def __init__(
        self,
        code: str,
        *,
        preserved_artifact_refs: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(code)
        self.code = code
        self.preserved_artifact_refs = tuple(dict(row) for row in preserved_artifact_refs)


def _request_ref(request: Mapping[str, Any], *, relative_path: str) -> dict[str, str]:
    return {
        "artifact_id": str(request["request_id"]),
        "artifact_version": REQUEST_VERSION,
        "byte_sha256": hashlib.sha256(canonical_resource_bytes(request)).hexdigest(),
        "cutoff": str(request["cutoff"]),
        "relative_path": relative_path,
        "semantic_sha256": str(request["semantic_sha256"]),
        "strategy_id": str(request["strategy_id"]),
    }


def _write(
    store: GovernedStore,
    *,
    path: PurePosixPath,
    artifact: Mapping[str, Any],
    identity_field: str,
) -> dict[str, str]:
    raw = canonical_resource_bytes(artifact)
    try:
        store.write_exact_once(path, raw)
        readback = load_canonical_resource(
            store.read(path, hashlib.sha256(raw).hexdigest()),
            label="daily research artifact",
        )
        validate_artifact(readback)
    except Exception as exc:
        raise ResearchLoopError("RESEARCH_DAILY_EXACT_ONCE_OR_READBACK_FAILURE") from exc
    if readback != artifact:
        raise ResearchLoopError("RESEARCH_DAILY_READBACK_REPLAY_MISMATCH")
    return artifact_ref(artifact, identity_field=identity_field, relative_path=path)


def _receipt(
    *,
    request: Mapping[str, Any],
    request_ref: Mapping[str, Any],
    run_state: str,
    forward_manifest_ref: Mapping[str, Any] | None,
    memory_ref: Mapping[str, Any] | None,
    experiment_registry_ref: Mapping[str, Any] | None,
    preserved_artifact_refs: Sequence[Mapping[str, Any]],
    blocker_codes: Sequence[str],
) -> dict[str, Any]:
    document = {
        "authority": dict(NO_AUTHORITY),
        "blocker_codes": sorted(set(blocker_codes), key=lambda value: value.encode("ascii")),
        "created_at": str(request["created_at"]),
        "cutoff": str(request["cutoff"]),
        "decision_session": str(request["decision_session"]),
        "experiment_registry_ref": (
            None if experiment_registry_ref is None else dict(experiment_registry_ref)
        ),
        "factor_governance_write": False,
        "forward_manifest_ref": (
            None if forward_manifest_ref is None else dict(forward_manifest_ref)
        ),
        "historical_backfill_eligible": False,
        "memory_ref": None if memory_ref is None else dict(memory_ref),
        "preserved_artifact_refs": sorted_refs(list(preserved_artifact_refs)),
        "production_governance_eligible": False,
        "protocol_version": PROTOCOL_VERSION,
        "provider_calls": False,
        "request_ref": dict(request_ref),
        "research_only": True,
        "run_id": str(request["request_id"]),
        "run_state": run_state,
        "strategy_id": str(request["strategy_id"]),
        "version": RECEIPT_VERSION,
    }
    return seal(document, identity_field="receipt_id")


def run_daily_research_loop(
    workspace_root: str,
    *,
    request_path: str,
    request_sha256: str,
) -> dict[str, Any]:
    """Run the research-only daily loop without discovery or provider access."""

    root = Path(workspace_root)
    if not root.is_absolute():
        raise ResearchLoopError("RESEARCH_WORKSPACE_ROOT_INVALID")
    try:
        request_raw = _stable_workspace_read(root, request_path)
        if hashlib.sha256(request_raw).hexdigest() != request_sha256:
            raise ResearchLoopError("RESEARCH_REQUEST_SHA_MISMATCH")
        request = validate_provisional_request(
            load_canonical_resource(request_raw, label="daily research request")
        )
    except ResearchLoopError:
        raise
    except Exception as exc:
        raise ResearchLoopError("RESEARCH_REQUEST_INVALID") from exc

    run_id = str(request["request_id"])
    strategy_id = str(request["strategy_id"])
    for value in (run_id, strategy_id):
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in value):
            raise ResearchLoopError("RESEARCH_OUTPUT_IDENTITY_PATH_UNSAFE")
    run_root = PurePosixPath("results/v17_v4_shadow/daily_research", strategy_id, run_id)
    store = GovernedStore(root)
    request_reference = _request_ref(request, relative_path=request_path)
    forward_ref: dict[str, Any] | None = None
    preserved: list[dict[str, Any]] = []
    blockers: list[str] = []
    state = "RUN_SUCCESS"
    try:
        forward_result = run_provisional_forward(
            workspace_root,
            request_path=request_path,
            request_sha256=request_sha256,
        )
        forward_ref = dict(forward_result["artifact_manifest_ref"])
        preserved.append(forward_ref)
    except ProvisionalForwardError as exc:
        preserved.extend(dict(row) for row in exc.preserved_artifact_refs)
        blockers.append(exc.code)
        state = "RUN_PARTIAL" if preserved else "RUN_BLOCKED"

    try:
        memory = build_memory_entry(
            run_id=run_id,
            strategy_id=strategy_id,
            decision_session=str(request["decision_session"]),
            cutoff=str(request["cutoff"]),
            created_at=str(request["created_at"]),
            source_refs=preserved,
            run_state=state,
            limitation_codes=blockers,
        )
    except Exception as exc:
        raise ResearchLoopError(
            "RESEARCH_DAILY_ARTIFACT_CONSTRUCTION_FAILED",
            preserved_artifact_refs=preserved,
        ) from exc
    memory_path = run_root / f"memory-{memory['entry_id']}.json"
    memory_ref = _write(
        store,
        path=memory_path,
        artifact=memory,
        identity_field="entry_id",
    )
    experiment_ref: dict[str, str] | None = None
    if forward_ref is not None:
        try:
            experiment = build_experiment_registry(
                experiment_id=run_id,
                run_id=run_id,
                strategy_id=strategy_id,
                decision_session=str(request["decision_session"]),
                cutoff=str(request["cutoff"]),
                created_at=str(request["created_at"]),
                forward_manifest_ref=forward_ref,
            )
        except Exception as exc:
            raise ResearchLoopError(
                "RESEARCH_DAILY_ARTIFACT_CONSTRUCTION_FAILED",
                preserved_artifact_refs=preserved,
            ) from exc
        experiment_path = run_root / f"experiment-{experiment['registry_id']}.json"
        experiment_ref = _write(
            store,
            path=experiment_path,
            artifact=experiment,
            identity_field="registry_id",
        )
    try:
        receipt = _receipt(
            request=request,
            request_ref=request_reference,
            run_state=state,
            forward_manifest_ref=forward_ref,
            memory_ref=memory_ref,
            experiment_registry_ref=experiment_ref,
            preserved_artifact_refs=preserved,
            blocker_codes=blockers,
        )
    except Exception as exc:
        raise ResearchLoopError(
            "RESEARCH_DAILY_ARTIFACT_CONSTRUCTION_FAILED",
            preserved_artifact_refs=preserved,
        ) from exc
    receipt_path = run_root / f"receipt-{receipt['receipt_id']}.json"
    receipt_ref = _write(
        store,
        path=receipt_path,
        artifact=receipt,
        identity_field="receipt_id",
    )
    return {
        "authority": dict(NO_AUTHORITY),
        "blocker_codes": list(receipt["blocker_codes"]),
        "broker": False,
        "default_protocol_state": "V15_DEFAULT",
        "execution": False,
        "factor_governance_write": False,
        "global_activation_state": "INACTIVE",
        "order": False,
        "production_governance_eligible": False,
        "provider_calls": False,
        "receipt_ref": receipt_ref,
        "research_only": True,
        "research_runtime_default": False,
        "run_state": state,
        "status": (
            "COMPLETE"
            if state == "RUN_SUCCESS"
            else "PARTIAL" if state == "RUN_PARTIAL" else "BLOCKED"
        ),
        "trade": False,
    }


__all__ = ["RECEIPT_VERSION", "ResearchLoopError", "run_daily_research_loop"]

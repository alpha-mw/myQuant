"""Minimal fail-closed suspended generation helpers."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import os
from typing import Any

from quant_investor.contracts import canonical_json_bytes, seal_artifact

from .errors import SystemContractError
from .release import installed_code_manifest_sha256
from .storage import EMPTY_POINTER_SHA256
from .store import SystemStore


def _store(value: SystemStore | str | os.PathLike[str]) -> SystemStore:
    return value if isinstance(value, SystemStore) else SystemStore(value)


def _sha(domain: str, **inputs: Any) -> str:
    return hashlib.sha256(
        canonical_json_bytes({"domain": domain, "identity_inputs": inputs})
    ).hexdigest()


def _blockers(values: Iterable[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise SystemContractError("suspension blockers must be an iterable")
    rows = list(values)
    if (
        not rows
        or any(type(row) is not str or not row or row != row.strip() for row in rows)
        or len(rows) != len(set(rows))
    ):
        raise SystemContractError("suspension blockers are invalid")
    return sorted(rows)


def build_suspended_generation(
    store_or_workspace_root: SystemStore | str | os.PathLike[str],
    *,
    blockers: Iterable[str],
    created_at: str,
    code_sha256: str | None = None,
    wheel_sha256: str | None = None,
    code_manifest_sha256: str | None = None,
    skill_tree_sha256: str | None = None,
    automation_semantic_sha256: str | None = None,
    producer_identity: str = "SYSTEM",
) -> dict[str, Any]:
    """Build, but do not activate, a complete minimal suspended generation."""

    store = _store(store_or_workspace_root)
    blocker_rows = _blockers(blockers)
    code_sha = code_sha256 or _sha("system.suspended.code", blockers=blocker_rows)
    code_manifest_sha = code_manifest_sha256 or installed_code_manifest_sha256()
    wheel_sha = wheel_sha256 or _sha(
        "system.suspended.wheel",
        code_manifest_sha256=code_manifest_sha,
    )
    skill_sha = skill_tree_sha256 or _sha("system.suspended.skill_tree", blockers=blocker_rows)
    automation_sha = automation_semantic_sha256 or _sha(
        "system.suspended.automation", blockers=blocker_rows
    )

    release_id = _sha(
        "system.suspended.release",
        code_sha256=code_sha,
        wheel_sha256=wheel_sha,
        code_manifest_sha256=code_manifest_sha,
        created_at=created_at,
    )
    release = seal_artifact(
        "system.release",
        {
            "release_id": release_id,
            "state": "SYSTEM_SUSPENDED",
            "code_sha256": code_sha,
            "wheel_sha256": wheel_sha,
            "code_manifest_sha256": code_manifest_sha,
        },
        created_at=created_at,
    )
    release_ref = store.put_object(release)

    readiness_id = _sha(
        "system.suspended.readiness",
        blockers=blocker_rows,
        producer_identity=producer_identity,
        created_at=created_at,
    )
    readiness = seal_artifact(
        "system.readiness",
        {
            "readiness_id": readiness_id,
            "factor_state": "SUSPENDED",
            "factor_status_ref": None,
            "admission_route": "SUSPENDED",
            "producer_identity": producer_identity,
            "mainline_state": "SUSPENDED",
            "mainline_candidate_ref": None,
            "investment_state": "SUSPENDED",
            "blockers": blocker_rows,
        },
        created_at=created_at,
    )
    readiness_ref = store.put_object(readiness)

    return store.assemble_generation(
        generation_state="SYSTEM_SUSPENDED",
        release_manifest_ref=release_ref,
        source_refs=[],
        factor_source_object_refs=[],
        factor_policy_ref=None,
        factor_evidence_refs=[],
        factor_active_set_ref=None,
        factor_validation_attestation_ref=None,
        mainline_ref=None,
        research_refs=[],
        migration_receipt_ref=None,
        migration_marker_ref=None,
        skill_tree_sha256=skill_sha,
        automation_semantic_sha256=automation_sha,
        readiness_matrix_ref=readiness_ref,
        emergency_controller_sha256=None,
        created_at=created_at,
    )


def suspend_system(
    store_or_workspace_root: SystemStore | str | os.PathLike[str],
    *,
    blockers: Iterable[str],
    created_at: str,
    expected_pointer_sha256: str,
    os_actor: str | None = None,
    code_sha256: str | None = None,
    wheel_sha256: str | None = None,
    code_manifest_sha256: str | None = None,
    skill_tree_sha256: str | None = None,
    automation_semantic_sha256: str | None = None,
    producer_identity: str = "SYSTEM",
) -> dict[str, Any]:
    """Build and CAS-activate a minimal suspended generation."""

    if expected_pointer_sha256 == EMPTY_POINTER_SHA256:
        raise SystemContractError("suspension cannot perform initial activation")

    store = _store(store_or_workspace_root)
    generation = build_suspended_generation(
        store,
        blockers=blockers,
        created_at=created_at,
        code_sha256=code_sha256,
        wheel_sha256=wheel_sha256,
        code_manifest_sha256=code_manifest_sha256,
        skill_tree_sha256=skill_tree_sha256,
        automation_semantic_sha256=automation_semantic_sha256,
        producer_identity=producer_identity,
    )
    return store.activate_generation(
        generation["generation_id"],
        expected_pointer_sha256=expected_pointer_sha256,
        activated_at=created_at,
        os_actor=os_actor,
        deployed_release_ref=generation["manifest"]["payload"]["release_manifest_ref"],
    )


__all__ = ["build_suspended_generation", "suspend_system"]

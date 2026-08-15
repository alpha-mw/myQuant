from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.migration import (
    build_concurrent_task_handoff,
    build_final_cutover_authorization,
    build_legacy_source_disposition,
    publish_authority_artifact,
    validate_concurrent_task_handoff,
    validate_final_cutover_authorization,
)
from quant_investor.system import SystemPreconditionError
from quant_investor.system.store import object_ref_for_artifact


BASE = "2026-08-16T00:00:00Z"


def _readbacks(commit: str, tree: str) -> list[dict[str, object]]:
    return [
        {
            "commit": commit,
            "tree": tree,
            "status_porcelain_sha256": "0" * 64,
            "path_inventory_sha256": "1" * 64,
            "observed_at": "2026-08-16T00:00:00Z",
        },
        {
            "commit": commit,
            "tree": tree,
            "status_porcelain_sha256": "0" * 64,
            "path_inventory_sha256": "1" * 64,
            "observed_at": "2026-08-16T00:00:01Z",
        },
    ]


def _handoff() -> dict[str, object]:
    return build_concurrent_task_handoff(
        handoff_id="v17-daily-data-handoff",
        task_name="A股 V17 日度数据更新（仅数据维护）",
        thread_id="01a00138-7152-7722-8dbd-8c9bd184273d",
        accepted_baseline_commit="a" * 40,
        task_commit="b" * 40,
        task_tree="c" * 40,
        path_rows=[
            {
                "path": "data/example.json",
                "status": "PRESENT",
                "mode": "100644",
                "size": 12,
                "git_blob_oid": "d" * 40,
                "byte_sha256": "e" * 64,
            }
        ],
        focused_test_rows=[
            {
                "command": "pytest tests/unit/test_data.py -q",
                "exit_code": 0,
                "stdout_sha256": "f" * 64,
                "status": "PASS",
            }
        ],
        readback_rows=_readbacks("b" * 40, "c" * 40),
        writer_ended=True,
        main_clean=True,
        created_at=BASE,
    )


def _disposition() -> dict[str, object]:
    return build_legacy_source_disposition(
        disposition_id="v17-source-to-stable",
        source_commit="b" * 40,
        rows=[
            {
                "source_path": "quant_investor/v17_v4_runtime/example.py",
                "source_blob_oid": "1" * 40,
                "classification": "PORTED_TO_STABLE",
                "stable_target_path": "quant_investor/market/example.py",
                "stable_target_blob_oid": "2" * 40,
                "behavior_test_selector": "test_stable_example",
                "reason": "reachable market behavior is owned by the stable module",
            }
        ],
        created_at=BASE,
    )


def test_concurrent_handoff_requires_clean_stable_double_read(tmp_path: Path) -> None:
    artifact = _handoff()
    assert validate_concurrent_task_handoff(artifact) == artifact

    target = publish_authority_artifact(tmp_path / "authority", artifact)
    assert target.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.stat().st_nlink == 1
    assert publish_authority_artifact(tmp_path / "authority", artifact) == target

    payload = dict(artifact["payload"])
    with pytest.raises(SystemPreconditionError, match="writer/main"):
        build_concurrent_task_handoff(
            handoff_id=payload["handoff_id"],
            task_name=payload["task_name"],
            thread_id=payload["thread_id"],
            accepted_baseline_commit=payload["accepted_baseline_commit"],
            task_commit=payload["task_commit"],
            task_tree=payload["task_tree"],
            path_rows=payload["path_rows"],
            focused_test_rows=payload["focused_test_rows"],
            readback_rows=payload["readback_rows"],
            writer_ended=False,
            main_clean=True,
            created_at=BASE,
        )


def test_legacy_disposition_blocks_unresolved_rows() -> None:
    row = dict(_disposition()["payload"]["rows"][0])
    row["classification"] = "BLOCKED_UNRESOLVED"
    with pytest.raises(SystemPreconditionError, match="BLOCKED_UNRESOLVED"):
        build_legacy_source_disposition(
            disposition_id="blocked",
            source_commit="b" * 40,
            rows=[row],
            created_at=BASE,
        )


def test_final_cutover_authorization_is_machine_derived_from_passed_gates() -> None:
    handoff_ref = object_ref_for_artifact(_handoff())
    disposition_ref = object_ref_for_artifact(_disposition())
    authorization = build_final_cutover_authorization(
        final_authorization_id="unified-final-cutover",
        accepted_baseline_commit="a" * 40,
        historical_integration_commit="3" * 40,
        historical_dirty_evidence_ref=handoff_ref,
        concurrent_task_handoff_ref=handoff_ref,
        legacy_disposition_ref=disposition_ref,
        final_integration_commit="4" * 40,
        final_integration_tree="5" * 40,
        ancestry_rows=[
            {"ancestor": "a" * 40, "descendant": "4" * 40, "proved": True},
            {"ancestor": "b" * 40, "descendant": "4" * 40, "proved": True},
        ],
        final_worktree_inventory_sha256="6" * 64,
        clean_checkout_readback_rows=_readbacks("4" * 40, "5" * 40),
        user_authorization_basis="explicit current-task production authorization",
        preflight_rows=[
            {"gate_id": "calendar", "status": "PASS", "evidence_sha256": "7" * 64},
            {"gate_id": "tests", "status": "PASS", "evidence_sha256": "8" * 64},
        ],
        created_at=BASE,
    )

    assert authorization["payload"]["final_build_authorized"] is True
    assert authorization["payload"]["cas_authorized"] is True
    assert validate_final_cutover_authorization(authorization) == authorization

    with pytest.raises(SystemPreconditionError, match="preflight"):
        build_final_cutover_authorization(
            final_authorization_id="blocked",
            accepted_baseline_commit="a" * 40,
            historical_integration_commit="3" * 40,
            historical_dirty_evidence_ref=handoff_ref,
            concurrent_task_handoff_ref=handoff_ref,
            legacy_disposition_ref=disposition_ref,
            final_integration_commit="4" * 40,
            final_integration_tree="5" * 40,
            ancestry_rows=[
                {"ancestor": "a" * 40, "descendant": "4" * 40, "proved": True}
            ],
            final_worktree_inventory_sha256="6" * 64,
            clean_checkout_readback_rows=_readbacks("4" * 40, "5" * 40),
            user_authorization_basis="explicit current-task production authorization",
            preflight_rows=[
                {"gate_id": "calendar", "status": "BLOCKED", "evidence_sha256": "7" * 64}
            ],
            created_at=BASE,
        )

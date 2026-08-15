from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any

from quant_investor.contracts import canonical_json_bytes
from quant_investor.migration.authority import (
    REQUIRED_FINAL_PREFLIGHT_GATES,
    build_concurrent_task_handoff,
    build_cutover_gate_evidence,
    build_final_cutover_authorization,
    build_legacy_source_disposition,
)
from quant_investor.migration.custody import build_authority_archive_plan
from quant_investor.migration.migration import (
    build_initial_active_pointer,
    build_pre_cas_migration_receipt,
)
from quant_investor.system.activation import build_activation_authorization
from quant_investor.system.store import SystemStore

from test_unified_migration_custody import _inventory, _workspace


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _test_final_authorization(
    store: SystemStore,
    release_ref: dict[str, str],
    *,
    created_at: str,
) -> dict[str, Any]:
    root = store.workspace_root
    if not (root / ".git").exists():
        _git(root, "init", "-q")
        _git(root, "config", "user.name", "Unified Test")
        _git(root, "config", "user.email", "unified-test@example.invalid")
        (root / ".git" / "info" / "exclude").write_text("*\n", encoding="utf-8")
        anchor = root / ".authority-anchor"
        anchor.write_bytes(b"unified-authority-anchor\n")
        _git(root, "add", "-f", ".authority-anchor")
        _git(root, "commit", "-q", "-m", "authority test anchor")
    commit = _git(root, "rev-parse", "HEAD^{commit}").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    ls_tree = _git(root, "ls-tree", "HEAD", "--", ".authority-anchor").decode().strip()
    mode, _kind, blob_and_path = ls_tree.split(" ", 2)
    blob, path = blob_and_path.split("\t", 1)
    anchor_raw = _git(root, "cat-file", "blob", blob)
    handoff = build_concurrent_task_handoff(
        handoff_id="test-clean-handoff",
        task_name="A股 V17 日度数据更新（仅数据维护）",
        thread_id="01a00138-7152-7722-8dbd-8c9bd184273d",
        accepted_baseline_commit=commit,
        task_commit=commit,
        task_tree=tree,
        path_rows=[
            {
                "path": path,
                "status": "PRESENT",
                "mode": mode,
                "size": len(anchor_raw),
                "git_blob_oid": blob,
                "byte_sha256": hashlib.sha256(anchor_raw).hexdigest(),
            }
        ],
        focused_test_rows=[
            {
                "command": "pytest test authority fixture",
                "exit_code": 0,
                "stdout_sha256": hashlib.sha256(b"passed").hexdigest(),
                "status": "PASS",
            }
        ],
        readback_rows=[
            {
                "commit": commit,
                "tree": tree,
                "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
                "path_inventory_sha256": "0" * 64,
                "observed_at": created_at,
            },
            {
                "commit": commit,
                "tree": tree,
                "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
                "path_inventory_sha256": "0" * 64,
                "observed_at": "2026-08-14T00:00:01Z",
            },
        ],
        writer_ended=True,
        main_clean=True,
        created_at=created_at,
    )
    disposition = build_legacy_source_disposition(
        disposition_id="test-legacy-custody",
        source_commit=commit,
        rows=[
            {
                "source_path": path,
                "source_blob_oid": blob,
                "classification": "LEGACY_CUSTODY_ONLY",
                "stable_target_path": "",
                "stable_target_blob_oid": "",
                "behavior_test_selector": "test_active_v17_zero_reachability",
                "reason": "test anchor is custody-only and unreachable",
            }
        ],
        created_at=created_at,
    )
    handoff_ref = store.put_object(handoff)
    disposition_ref = store.put_object(disposition)
    inventory_rows = [{"path": path, "mode": mode, "git_blob_oid": blob}]
    inventory_sha = hashlib.sha256(canonical_json_bytes(inventory_rows)).hexdigest()
    gate_rows: list[dict[str, Any]] = []
    for gate_id in sorted(REQUIRED_FINAL_PREFLIGHT_GATES):
        evidence = build_cutover_gate_evidence(
            gate_id=gate_id,
            final_commit=commit,
            final_tree=tree,
            command=f"test-gate {gate_id}",
            exit_code=0,
            stdout_sha256=hashlib.sha256(gate_id.encode()).hexdigest(),
            subject_ref=release_ref,
            observed_at=created_at,
        )
        gate_rows.append({"gate_id": gate_id, "evidence_ref": store.put_object(evidence)})
    readbacks = [
        {
            "commit": commit,
            "tree": tree,
            "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
            "path_inventory_sha256": inventory_sha,
            "observed_at": created_at,
        },
        {
            "commit": commit,
            "tree": tree,
            "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
            "path_inventory_sha256": inventory_sha,
            "observed_at": "2026-08-14T00:00:01Z",
        },
    ]
    final_authorization = build_final_cutover_authorization(
        final_authorization_id="test-final-cutover",
        accepted_baseline_commit=commit,
        historical_integration_commit=commit,
        historical_dirty_evidence_ref=release_ref,
        concurrent_task_handoff_ref=handoff_ref,
        legacy_disposition_ref=disposition_ref,
        deployed_release_ref=release_ref,
        release_commit=commit,
        release_tree=tree,
        final_integration_commit=commit,
        final_integration_tree=tree,
        ancestry_rows=[{"ancestor": commit, "descendant": commit, "proved": True}],
        excluded_commit_rows=[],
        final_worktree_inventory_sha256=inventory_sha,
        clean_checkout_readback_rows=readbacks,
        user_authorization_basis="explicit test-only activation authorization",
        preflight_rows=gate_rows,
        created_at=created_at,
    )
    store.put_object(final_authorization)
    return final_authorization


def prepare_migration_context(
    store: SystemStore,
    *,
    cutover_id: str = "test-initial-activation",
    created_at: str = "2026-08-14T00:00:00Z",
) -> dict[str, Any]:
    root = store.workspace_root
    rules_path = _workspace(root)
    inventory = _inventory(root)
    archive_plan = build_authority_archive_plan(
        root,
        inventory,
        cutover_id=cutover_id,
        created_at=created_at,
        rules_path=rules_path,
    )
    return {
        "archive_plan": archive_plan,
        "cutover_id": cutover_id,
        "inventory": inventory,
        "rules_path": rules_path,
    }


def prepare_initial_activation(
    store: SystemStore,
    generation: dict[str, Any],
    release_ref: dict[str, str],
    *,
    cutover_id: str = "test-initial-activation",
    prepared_at: str = "2026-08-14T00:00:00Z",
    activated_at: str = "2026-08-14T00:00:01Z",
    migration_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = store.workspace_root
    context = migration_context or prepare_migration_context(
        store,
        cutover_id=cutover_id,
        created_at=prepared_at,
    )
    if context["cutover_id"] != cutover_id:
        raise AssertionError("test migration context cutover mismatch")
    rules_path = context["rules_path"]
    inventory = context["inventory"]
    archive_plan = context["archive_plan"]
    pointer = build_initial_active_pointer(
        generation["manifest"],
        activated_at=activated_at,
        os_actor=f"uid:{os.geteuid()}",
    )
    receipt = build_pre_cas_migration_receipt(
        root,
        inventory,
        archive_plan,
        pointer,
        generation["manifest"],
        cutover_id=cutover_id,
        created_at=prepared_at,
        rules_path=rules_path,
    )
    final_authorization = _test_final_authorization(store, release_ref, created_at=prepared_at)
    authorization = build_activation_authorization(
        final_cutover_authorization=final_authorization,
        migration_receipt=receipt,
        target_active_pointer=pointer,
        target_generation_manifest=generation["manifest"],
        deployed_release_ref=release_ref,
        prepared_at=prepared_at,
        actor_uid=os.geteuid(),
    )
    return {
        "activation_authorization_raw": canonical_json_bytes(authorization),
        "final_cutover_authorization_raw": canonical_json_bytes(final_authorization),
        "deployed_release_ref": release_ref,
        "migration_receipt_raw": canonical_json_bytes(receipt),
        "target_active_pointer_raw": canonical_json_bytes(pointer),
    }


def activate_initial(
    store: SystemStore,
    generation: dict[str, Any],
    release_ref: dict[str, str],
    **kwargs: Any,
) -> dict[str, Any]:
    prepared = prepare_initial_activation(store, generation, release_ref, **kwargs)
    return store.activate_initial_generation(**prepared)

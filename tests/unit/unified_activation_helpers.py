from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from quant_investor.contracts import canonical_json_bytes, contract_catalog_sha256
from quant_investor.migration.authority import (
    _GATE_SPECS,
    _seal_cutover_gate_evidence,
    REQUIRED_FINAL_PREFLIGHT_GATES,
    build_concurrent_task_handoff,
    build_final_cutover_authorization,
    build_legacy_source_disposition,
)
from quant_investor.migration.custody import build_authority_archive_plan
from quant_investor.migration.migration import (
    build_initial_active_pointer,
    build_pre_cas_migration_receipt,
)
from quant_investor.system.activation import build_activation_authorization
from quant_investor.system.release_install import build_release_install_evidence
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
        runner = root / "quant_investor/migration/authority.py"
        runner.parent.mkdir(parents=True, exist_ok=True)
        runner.write_bytes(
            Path(
                __import__("quant_investor.migration.authority", fromlist=["x"]).__file__
            ).read_bytes()
        )
        assembler = root / "quant_investor/factors/governance/production.py"
        assembler.parent.mkdir(parents=True, exist_ok=True)
        assembler.write_bytes(
            Path(
                __import__("quant_investor.factors.governance.production", fromlist=["x"]).__file__
            ).read_bytes()
        )
        _git(
            root,
            "add",
            "-f",
            ".authority-anchor",
            runner.relative_to(root).as_posix(),
            assembler.relative_to(root).as_posix(),
        )
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
    runner_ls_tree = (
        _git(root, "ls-tree", "HEAD", "--", "quant_investor/migration/authority.py")
        .decode()
        .strip()
    )
    runner_mode, _runner_kind, runner_blob_and_path = runner_ls_tree.split(" ", 2)
    runner_blob, runner_path = runner_blob_and_path.split("\t", 1)
    assembler_ls_tree = (
        _git(
            root,
            "ls-tree",
            "HEAD",
            "--",
            "quant_investor/factors/governance/production.py",
        )
        .decode()
        .strip()
    )
    assembler_mode, _assembler_kind, assembler_blob_and_path = assembler_ls_tree.split(" ", 2)
    assembler_blob, assembler_path = assembler_blob_and_path.split("\t", 1)
    inventory_rows = sorted(
        [
            {"path": path, "mode": mode, "git_blob_oid": blob},
            {
                "path": runner_path,
                "mode": runner_mode,
                "git_blob_oid": runner_blob,
            },
            {
                "path": assembler_path,
                "mode": assembler_mode,
                "git_blob_oid": assembler_blob,
            },
        ],
        key=lambda row: row["path"],
    )
    inventory_sha = hashlib.sha256(canonical_json_bytes(inventory_rows)).hexdigest()
    gate_evidence: list[dict[str, Any]] = []
    runner_raw = _git(root, "show", f"{commit}:quant_investor/migration/authority.py")
    executable = Path(sys.executable).resolve()
    executable_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    release = store.get_object(release_ref)
    anchor_path = (root / path).resolve(strict=True)
    install_evidence = build_release_install_evidence(
        final_commit=commit,
        final_tree=tree,
        code_tree_sha256_value=release["payload"]["code_sha256"],
        git_code_manifest_sha256_value=release["payload"]["code_manifest_sha256"],
        release_ref=release_ref,
        source_archive={
            "path": str(anchor_path),
            "byte_sha256": hashlib.sha256(anchor_raw).hexdigest(),
            "size": len(anchor_raw),
        },
        wheel={
            "path": str(anchor_path),
            "byte_sha256": release["payload"]["wheel_sha256"],
            "size": len(anchor_raw),
        },
        install_root=str(root),
        python_executable=str(executable),
        python_executable_sha256=executable_sha,
        import_origin=str(
            Path(__import__("quant_investor", fromlist=["x"]).__file__).resolve(strict=True)
        ),
        installed_code_manifest_sha256=release["payload"]["code_manifest_sha256"],
        contract_catalog_sha256_value=contract_catalog_sha256(),
        lockfile_sha256=hashlib.sha256(anchor_raw).hexdigest(),
        created_at=created_at,
    )
    install_evidence_ref = store.put_object(install_evidence)
    for gate_id in sorted(REQUIRED_FINAL_PREFLIGHT_GATES):
        if gate_id == "release_install_origin":
            subject_ref = install_evidence_ref
            gate_stdin = canonical_json_bytes(
                {
                    "release_install_evidence": install_evidence,
                    "deployed_release": release,
                }
            )
            gate_stdout = canonical_json_bytes(
                {
                    "state": "PASS",
                    "release_ref": release_ref,
                    "source_archive_sha256": install_evidence["payload"]["source_archive"][
                        "byte_sha256"
                    ],
                    "wheel_sha256": release["payload"]["wheel_sha256"],
                    "code_tree_sha256": release["payload"]["code_sha256"],
                    "installed_code_manifest_sha256": release["payload"]["code_manifest_sha256"],
                    "contract_catalog_sha256": install_evidence["payload"][
                        "contract_catalog_sha256"
                    ],
                    "import_origin": install_evidence["payload"]["import_origin"],
                }
            )
        elif gate_id == "clean_detached_clone":
            subject_ref = release_ref
            gate_stdin = canonical_json_bytes({"final_commit": commit, "final_tree": tree})
            gate_stdout = canonical_json_bytes(
                {
                    "state": "PASS",
                    "repository_root": str(root),
                    "commit": commit,
                    "tree": tree,
                    "status_sha256": hashlib.sha256(b"").hexdigest(),
                    "detached": True,
                }
            )
        else:
            subject_ref = release_ref
            gate_stdin = b""
            gate_stdout = b""
        evidence = _seal_cutover_gate_evidence(
            gate_id=gate_id,
            final_commit=commit,
            final_tree=tree,
            runner_code_sha256=hashlib.sha256(runner_raw).hexdigest(),
            environment_sha256=hashlib.sha256(b"unit-fixture-environment").hexdigest(),
            batch_results=[
                {
                    "argv": list(argv),
                    "exit_code": 0,
                    "stdout_base64": base64.b64encode(gate_stdout).decode("ascii"),
                    "stdout_sha256": hashlib.sha256(gate_stdout).hexdigest(),
                    "stderr_base64": "",
                    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                    "executable_path": str(executable),
                    "executable_sha256": executable_sha,
                    "stdin_sha256": hashlib.sha256(gate_stdin).hexdigest(),
                }
                for argv in _GATE_SPECS[gate_id]
            ],
            subject_ref=subject_ref,
            started_at=created_at,
            finished_at=created_at,
        )
        store.put_object(evidence)
        gate_evidence.append(evidence)
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
        preflight_evidence=gate_evidence,
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

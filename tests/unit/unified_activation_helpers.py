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
    _seal_final_cutover_authorization,
    _seal_cutover_gate_evidence,
    REQUIRED_FINAL_PREFLIGHT_GATES,
    build_final_cutover_authorization,
    build_legacy_source_disposition,
    build_main_checkout_adoption,
)
from quant_investor.migration.custody import build_authority_archive_plan
from quant_investor.migration.migration import (
    build_initial_active_pointer,
    build_pre_cas_migration_receipt,
)
from quant_investor.system.activation import build_activation_authorization
from quant_investor.system.bootstrap_receipt import build_production_bootstrap_receipt
from quant_investor.system.release_install import build_release_install_evidence
from quant_investor.system.store import SystemStore, object_ref_for_artifact

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
    production_generation_manifest: dict[str, Any],
    production_bootstrap_receipt: dict[str, Any],
    created_at: str,
) -> dict[str, Any]:
    root = store.workspace_root
    repository_exists = (root / ".git").exists()
    real_checkout = repository_exists and bool(
        _git(root, "ls-tree", "HEAD", "--", "pyproject.toml").strip()
    )
    if not repository_exists:
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
        _git(root, "commit", "-q", "-m", "authority test baseline")
        for index in range(22):
            adopted = root / "adopted" / f"path-{index:02d}.txt"
            adopted.parent.mkdir(parents=True, exist_ok=True)
            adopted.write_bytes(f"adopted-{index:02d}\n".encode("ascii"))
        _git(root, "add", "-f", "adopted")
        _git(root, "commit", "-q", "-m", "prospective adoption fixture")
    commit = _git(root, "rev-parse", "HEAD^{commit}").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    if real_checkout:
        adoption_commit = ""
        for candidate_raw in _git(root, "rev-list", "--first-parent", commit).splitlines():
            candidate = candidate_raw.decode("ascii")
            try:
                delta = _git(
                    root,
                    "diff-tree",
                    "--no-commit-id",
                    "--name-status",
                    "-r",
                    "--no-renames",
                    f"{candidate}^",
                    candidate,
                ).decode("utf-8")
            except subprocess.CalledProcessError:
                continue
            rows = [line.split("\t", 1) for line in delta.splitlines() if line]
            if len(rows) == 22 and all(status in {"A", "M"} for status, _path in rows):
                adoption_commit = candidate
                break
        if not adoption_commit:
            raise AssertionError("test checkout has no exact 22-path adoption commit")
    else:
        adoption_commit = commit
    adoption_tree = _git(root, "rev-parse", f"{adoption_commit}^{{tree}}").decode().strip()
    baseline_commit = _git(root, "rev-parse", f"{adoption_commit}^").decode().strip()
    baseline_tree = _git(root, "rev-parse", f"{baseline_commit}^{{tree}}").decode().strip()
    anchor_relative = "pyproject.toml" if real_checkout else ".authority-anchor"
    ls_tree = _git(root, "ls-tree", "HEAD", "--", anchor_relative).decode().strip()
    mode, _kind, blob_and_path = ls_tree.split(" ", 2)
    blob, path = blob_and_path.split("\t", 1)
    anchor_raw = _git(root, "cat-file", "blob", blob)
    if real_checkout:
        source_ls_tree = (
            _git(root, "ls-tree", baseline_commit, "--", anchor_relative).decode().strip()
        )
        _source_mode, _source_kind, source_blob_and_path = source_ls_tree.split(" ", 2)
        source_blob, source_path = source_blob_and_path.split("\t", 1)
        disposition_rows = [
            {
                "source_path": source_path,
                "source_blob_oid": source_blob,
                "classification": "PORTED_TO_STABLE",
                "stable_target_path": path,
                "stable_target_blob_oid": blob,
                "behavior_test_selector": "test_two_root_release_identity_closure",
                "reason": "test release metadata remains bound to the stable final tree",
            }
        ]
    else:
        disposition_rows = [
            {
                "source_path": path,
                "source_blob_oid": blob,
                "classification": "LEGACY_CUSTODY_ONLY",
                "stable_target_path": "",
                "stable_target_blob_oid": "",
                "behavior_test_selector": "test_active_v17_zero_reachability",
                "reason": "test anchor is custody-only and unreachable",
            }
        ]
    disposition = build_legacy_source_disposition(
        disposition_id="test-legacy-custody",
        source_commit=baseline_commit,
        rows=disposition_rows,
        created_at=created_at,
    )
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

    def inventory_for(revision: str) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for raw_entry in _git(root, "ls-tree", "-rz", "--full-tree", revision).split(b"\0"):
            if not raw_entry:
                continue
            header, path_raw = raw_entry.split(b"\t", 1)
            entry_mode, entry_kind, entry_oid = header.split(b" ", 2)
            if entry_kind != b"blob":
                continue
            result.append(
                {
                    "path": path_raw.decode("utf-8"),
                    "mode": entry_mode.decode("ascii"),
                    "git_blob_oid": entry_oid.decode("ascii"),
                }
            )
        return result

    inventory_rows = inventory_for(commit)
    inventory_sha = hashlib.sha256(canonical_json_bytes(inventory_rows)).hexdigest()
    adoption_inventory_sha = hashlib.sha256(
        canonical_json_bytes(inventory_for(adoption_commit))
    ).hexdigest()
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
    adoption_readbacks = [
        {
            "commit": adoption_commit,
            "tree": adoption_tree,
            "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
            "path_inventory_sha256": adoption_inventory_sha,
            "observed_at": created_at,
        },
        {
            "commit": adoption_commit,
            "tree": adoption_tree,
            "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
            "path_inventory_sha256": adoption_inventory_sha,
            "observed_at": "2026-08-14T00:00:01Z",
        },
    ]
    delta = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "--no-renames",
        baseline_commit,
        adoption_commit,
    ).decode("utf-8")
    delta_rows = sorted(line.split("\t", 1) for line in delta.splitlines() if line)
    assert len(delta_rows) == 22
    adopted_rows: list[dict[str, Any]] = []
    adoption_disposition_rows: list[dict[str, Any]] = []
    for index, (status_code, adopted_path) in enumerate(delta_rows):
        adopted_ls_tree = (
            _git(root, "ls-tree", adoption_commit, "--", adopted_path).decode().strip()
        )
        adopted_mode, _adopted_kind, adopted_blob_and_path = adopted_ls_tree.split(" ", 2)
        adopted_blob, observed_path = adopted_blob_and_path.split("\t", 1)
        adopted_raw = _git(root, "cat-file", "blob", adopted_blob)
        adopted_rows.append(
            {
                "path": observed_path,
                "status": {"A": "ADDED", "M": "MODIFIED"}[status_code],
                "mode": adopted_mode,
                "size": len(adopted_raw),
                "git_blob_oid": adopted_blob,
                "byte_sha256": hashlib.sha256(adopted_raw).hexdigest(),
            }
        )
        final_ls_tree = _git(root, "ls-tree", commit, "--", observed_path).decode().strip()
        _final_mode, _final_kind, final_blob_and_path = final_ls_tree.split(" ", 2)
        final_blob, final_path = final_blob_and_path.split("\t", 1)
        exact = final_blob == adopted_blob and final_path == observed_path
        adoption_disposition_rows.append(
            {
                "path": observed_path,
                "partition": "TASK_ORIGIN" if index < 17 else "ORPHAN",
                "decision": "EXACT_PRESERVED" if exact else "REPAIRED_IN_FINAL_INTEGRATION",
                "target_path": final_path,
                "target_blob_oid": final_blob,
                "behavior_test_selector": f"test_adopted_path_{index:02d}",
                "reason": (
                    "test adoption path is exactly preserved in the frozen tree"
                    if exact
                    else "test adoption path is explicitly repaired in the frozen tree"
                ),
            }
        )
    gate_refs = sorted(
        (
            {
                "gate_id": evidence["payload"]["gate_id"],
                "evidence_ref": store.put_object(evidence),
            }
            for evidence in gate_evidence
        ),
        key=lambda row: row["gate_id"],
    )
    adoption = build_main_checkout_adoption(
        adoption_id="test-main-checkout-adoption",
        task_name="A股 V17 日度数据更新（仅数据维护）",
        thread_id="01a00138-7152-7722-8dbd-8c9bd184273d",
        accepted_baseline_commit=baseline_commit,
        accepted_baseline_tree=baseline_tree,
        adoption_commit=adoption_commit,
        adoption_tree=adoption_tree,
        adoption_parent=baseline_commit,
        path_rows=adopted_rows,
        task_origin_paths=[row["path"] for row in adopted_rows[:17]],
        orphan_paths=[row["path"] for row in adopted_rows[17:]],
        disposition_rows=adoption_disposition_rows,
        focused_test_rows=[
            {
                "command": "pytest test authority fixture",
                "exit_code": 0,
                "stdout_sha256": hashlib.sha256(b"passed").hexdigest(),
                "status": "PASS",
            }
        ],
        full_gate_refs=gate_refs,
        source_task_completion={
            "status": "COMPLETED_WITHOUT_COMMIT",
            "latest_turn_id": "01a0086f-f4f5-7bd2-873e-4e874369c021",
            "completed_at": created_at,
            "final_message_sha256": hashlib.sha256(b"test task final").hexdigest(),
        },
        readback_rows=adoption_readbacks,
        user_authorization_basis="explicit test prospective-adoption authorization",
        writer_ended=True,
        main_clean=True,
        created_at=created_at,
    )
    adoption_ref = store.put_object(adoption)
    common = {
        "final_authorization_id": "test-final-cutover",
        "accepted_baseline_commit": baseline_commit,
        "historical_integration_commit": baseline_commit,
        "historical_dirty_evidence_ref": release_ref,
        "concurrent_task_handoff_ref": None,
        "main_checkout_adoption_ref": adoption_ref,
        "legacy_disposition_ref": disposition_ref,
        "deployed_release_ref": release_ref,
        "release_commit": commit,
        "release_tree": tree,
        "final_integration_commit": commit,
        "final_integration_tree": tree,
        "ancestry_rows": sorted(
            [
                {"ancestor": ancestor, "descendant": commit, "proved": True}
                for ancestor in sorted({baseline_commit, adoption_commit, commit})
            ],
            key=lambda row: (row["ancestor"], row["descendant"]),
        ),
        "excluded_commit_rows": [],
        "final_worktree_inventory_sha256": inventory_sha,
        "clean_checkout_readback_rows": readbacks,
        "user_authorization_basis": "explicit test-only activation authorization",
        "created_at": created_at,
    }
    production_receipt_ref = store.put_object(production_bootstrap_receipt)
    if production_generation_manifest["payload"]["research_refs"] == [production_receipt_ref]:
        final_authorization = build_final_cutover_authorization(
            **common,
            system_store=store,
            production_generation_id=production_generation_manifest["semantic_sha256"],
            preflight_evidence=gate_evidence,
        )
    else:
        receipt_payload = production_bootstrap_receipt["payload"]
        final_authorization = _seal_final_cutover_authorization(
            **common,
            production_generation_manifest_ref=object_ref_for_artifact(
                production_generation_manifest
            ),
            production_bootstrap_receipt_ref=production_receipt_ref,
            calendar_authority_policy_ref=receipt_payload["calendar_authority_policy_ref"],
            calendar_compilation_ref=receipt_payload["calendar_compilation_ref"],
            calendar_capability_ref=receipt_payload["calendar_capability_ref"],
            calendar_capture_execution_ref=receipt_payload["calendar_capture_execution_ref"],
            calendar_authorization_basis=receipt_payload["calendar_authorization_basis"],
            calendar_source_limitations=receipt_payload["calendar_source_limitations"],
            preflight_rows=sorted(
                [
                    {
                        "gate_id": evidence["payload"]["gate_id"],
                        "evidence_ref": object_ref_for_artifact(evidence),
                    }
                    for evidence in gate_evidence
                ],
                key=lambda row: row["gate_id"],
            ),
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
    calendar_binding_receipt_ref: dict[str, str] | None = None,
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
    research_refs = generation["manifest"]["payload"]["research_refs"]
    if calendar_binding_receipt_ref is not None or research_refs:
        production_receipt_ref = calendar_binding_receipt_ref or research_refs[0]
        production_receipt_document = store.get_object(production_receipt_ref)
    else:
        source_refs = generation["manifest"]["payload"]["factor_source_object_refs"]
        basis = {
            "authority_route": "EXCHANGE_OFFICIAL",
            "policy_ref": source_refs[0],
            "compilation_ref": source_refs[1],
            "capability_ref": None,
            "capture_execution_ref": None,
            "source_limitations": [],
        }
        production_receipt_document = build_production_bootstrap_receipt(
            bootstrap_operator_request_ref=source_refs[0],
            source_root_id="test-isolated-pointer-protocol",
            input_source_rows=[
                {
                    "field": "calendar_authority_policy_file_ref",
                    "ordinal": 0,
                    "input_file_ref": {
                        "relative_path": "test/calendar-policy.json",
                        "byte_sha256": source_refs[0]["byte_sha256"],
                    },
                    "source_object_ref": source_refs[0],
                }
            ],
            deployed_release_ref=release_ref,
            calendar_authority_policy_ref=source_refs[0],
            calendar_compilation_ref=source_refs[1],
            calendar_capability_ref=None,
            calendar_capture_execution_ref=None,
            calendar_authorization_basis=basis,
            calendar_source_limitations=[],
            release_code_manifest_sha256=store.get_object(release_ref)["payload"][
                "code_manifest_sha256"
            ],
            generation_created_at=generation["manifest"]["created_at"],
            expected_assembly_id=generation["manifest"]["payload"]["assembly_id"],
            generation_intent_sha256="1" * 64,
            source_refs=generation["manifest"]["payload"]["source_refs"],
            factor_source_object_refs=source_refs,
            factor_policy_ref=generation["manifest"]["payload"]["factor_policy_ref"],
            factor_evidence_refs=generation["manifest"]["payload"]["factor_evidence_refs"],
            factor_active_set_ref=generation["manifest"]["payload"]["factor_active_set_ref"],
            factor_validation_attestation_ref=generation["manifest"]["payload"][
                "factor_validation_attestation_ref"
            ],
            readiness_matrix_ref=generation["manifest"]["payload"]["readiness_matrix_ref"],
            emergency_controller_sha256=generation["manifest"]["payload"][
                "emergency_controller_sha256"
            ],
            skill_tree_sha256=generation["manifest"]["payload"]["skill_tree_sha256"],
            automation_semantic_sha256=generation["manifest"]["payload"][
                "automation_semantic_sha256"
            ],
            source_blockers=[
                "FUNDAMENTAL_HISTORY_MIXED",
                "FUNDAMENTAL_HISTORY_NOT_HOMOGENEOUS",
                "FUNDAMENTAL_LEGACY_DIRECT_READER_PROVENANCE_LIMITED",
            ],
            fundamental_machine_states={
                "mixed": True,
                "legacy_direct_reader_provenance": "limited",
                "binding_aware_research_ready": True,
                "homogeneous_history_ready": False,
            },
            signal_statistics=[
                {
                    "factor_id": "pv_low_dollar_volume_5d",
                    "finite_count": 2,
                    "distinct_finite_count": 2,
                },
                {
                    "factor_id": "pv_blend_volstab19x2_mom90_amihud5_w80",
                    "finite_count": 2,
                    "distinct_finite_count": 2,
                },
            ],
            assembler_code_sha256="2" * 64,
            created_at=prepared_at,
        )
    final_authorization = _test_final_authorization(
        store,
        release_ref,
        production_generation_manifest=generation["manifest"],
        production_bootstrap_receipt=production_receipt_document,
        created_at=prepared_at,
    )
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

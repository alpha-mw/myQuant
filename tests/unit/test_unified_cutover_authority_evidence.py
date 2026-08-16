from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.migration import (
    REQUIRED_FINAL_PREFLIGHT_GATES,
    build_concurrent_task_handoff,
    build_final_cutover_authorization,
    build_legacy_source_disposition,
    build_main_checkout_adoption,
    publish_authority_artifact,
    run_cutover_gate,
    validate_concurrent_task_handoff,
    validate_final_cutover_authorization,
    validate_main_checkout_adoption,
    validate_main_checkout_adoption_closure,
    validate_cutover_gate_evidence,
)
from quant_investor.migration.authority import (
    _GATE_SPECS,
    _seal_cutover_gate_evidence,
)
from quant_investor.system import SystemContractError, SystemPreconditionError
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


def _adoption(gate_evidence: list[dict[str, object]]) -> dict[str, object]:
    rows = [
        {
            "path": f"adopted/path-{index:02d}.txt",
            "status": "ADDED",
            "mode": "100644",
            "size": index + 1,
            "git_blob_oid": f"{index + 10:040x}",
            "byte_sha256": f"{index + 100:064x}",
        }
        for index in range(22)
    ]
    dispositions = [
        {
            "path": row["path"],
            "partition": "TASK_ORIGIN" if index < 17 else "ORPHAN",
            "decision": "EXACT_PRESERVED",
            "target_path": row["path"],
            "target_blob_oid": row["git_blob_oid"],
            "behavior_test_selector": f"test_adopted_{index:02d}",
            "reason": "the exact test path is retained",
        }
        for index, row in enumerate(rows)
    ]
    gate_refs = sorted(
        [
            {
                "gate_id": evidence["payload"]["gate_id"],
                "evidence_ref": object_ref_for_artifact(evidence),
            }
            for evidence in gate_evidence
        ],
        key=lambda row: row["gate_id"],
    )
    return build_main_checkout_adoption(
        adoption_id="main-checkout-adoption",
        task_name="A股 V17 日度数据更新（仅数据维护）",
        thread_id="01a00138-7152-7722-8dbd-8c9bd184273d",
        accepted_baseline_commit="a" * 40,
        accepted_baseline_tree="9" * 40,
        adoption_commit="b" * 40,
        adoption_tree="c" * 40,
        adoption_parent="a" * 40,
        path_rows=rows,
        task_origin_paths=[row["path"] for row in rows[:17]],
        orphan_paths=[row["path"] for row in rows[17:]],
        disposition_rows=dispositions,
        focused_test_rows=[
            {
                "command": "pytest tests/unit/test_adoption.py -q",
                "exit_code": 0,
                "stdout_sha256": "f" * 64,
                "status": "PASS",
            }
        ],
        full_gate_refs=gate_refs,
        source_task_completion={
            "status": "COMPLETED_WITHOUT_COMMIT",
            "latest_turn_id": "01a0086f-f4f5-7bd2-873e-4e874369c021",
            "completed_at": BASE,
            "final_message_sha256": "e" * 64,
        },
        readback_rows=_readbacks("b" * 40, "c" * 40),
        user_authorization_basis="explicit repository-owner prospective adoption",
        writer_ended=True,
        main_clean=True,
        created_at=BASE,
    )


def _gate_evidence(commit: str, tree: str, subject_ref: dict[str, str]) -> list[dict[str, object]]:
    executable = Path(sys.executable).resolve()
    executable_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    return [
        _seal_cutover_gate_evidence(
            gate_id=gate_id,
            final_commit=commit,
            final_tree=tree,
            runner_code_sha256="7" * 64,
            environment_sha256="8" * 64,
            batch_results=[
                {
                    "argv": list(argv),
                    "exit_code": 0,
                    "stdout_base64": "",
                    "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                    "stderr_base64": "",
                    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                    "executable_path": str(executable),
                    "executable_sha256": executable_sha,
                    "stdin_sha256": hashlib.sha256(
                        canonical_json_bytes({"final_commit": commit, "final_tree": tree})
                        if gate_id == "clean_detached_clone"
                        else b""
                    ).hexdigest(),
                }
                for argv in _GATE_SPECS[gate_id]
            ],
            subject_ref=subject_ref,
            started_at=BASE,
            finished_at=BASE,
        )
        for gate_id in sorted(REQUIRED_FINAL_PREFLIGHT_GATES)
    ]


def _real_adoption(tmp_path: Path) -> tuple[Path, dict[str, object], list[dict[str, object]]]:
    root = tmp_path / "adoption-repository"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Adoption Test"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "adoption@example.invalid"],
        check=True,
    )
    (root / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "baseline.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "baseline"], check=True)
    baseline = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    baseline_tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    for index in range(22):
        path = root / "adopted" / f"path-{index:02d}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"adopted-{index:02d}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "adopted"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "adoption"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    rows: list[dict[str, object]] = []
    dispositions: list[dict[str, object]] = []
    inventory_rows: list[dict[str, str]] = []
    raw_inventory = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-rz", "--full-tree", commit],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    for entry in raw_inventory.split(b"\0"):
        if not entry:
            continue
        header, path_raw = entry.split(b"\t", 1)
        mode, kind, oid = header.split(b" ", 2)
        if kind != b"blob":
            continue
        inventory_rows.append(
            {
                "path": path_raw.decode(),
                "mode": mode.decode(),
                "git_blob_oid": oid.decode(),
            }
        )
    for index in range(22):
        path = f"adopted/path-{index:02d}.txt"
        ls_tree = subprocess.run(
            ["git", "-C", str(root), "ls-tree", commit, "--", path],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        mode, _kind, blob_and_path = ls_tree.split(" ", 2)
        oid, observed_path = blob_and_path.split("\t", 1)
        raw = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", oid],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        rows.append(
            {
                "path": observed_path,
                "status": "ADDED",
                "mode": mode,
                "size": len(raw),
                "git_blob_oid": oid,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        dispositions.append(
            {
                "path": observed_path,
                "partition": "TASK_ORIGIN" if index < 17 else "ORPHAN",
                "decision": "EXACT_PRESERVED",
                "target_path": observed_path,
                "target_blob_oid": oid,
                "behavior_test_selector": f"test_real_adoption_{index:02d}",
                "reason": "the immutable Git blob is retained exactly",
            }
        )
    subject_ref = object_ref_for_artifact(_handoff())
    gate_evidence = _gate_evidence(commit, tree, subject_ref)
    gate_refs = sorted(
        [
            {
                "gate_id": evidence["payload"]["gate_id"],
                "evidence_ref": object_ref_for_artifact(evidence),
            }
            for evidence in gate_evidence
        ],
        key=lambda row: row["gate_id"],
    )
    inventory_sha = hashlib.sha256(canonical_json_bytes(inventory_rows)).hexdigest()
    readbacks = [
        {
            "commit": commit,
            "tree": tree,
            "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
            "path_inventory_sha256": inventory_sha,
            "observed_at": BASE,
        },
        {
            "commit": commit,
            "tree": tree,
            "status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
            "path_inventory_sha256": inventory_sha,
            "observed_at": "2026-08-16T00:00:01Z",
        },
    ]
    adoption = build_main_checkout_adoption(
        adoption_id="real-main-checkout-adoption",
        task_name="A股 V17 日度数据更新（仅数据维护）",
        thread_id="01a00138-7152-7722-8dbd-8c9bd184273d",
        accepted_baseline_commit=baseline,
        accepted_baseline_tree=baseline_tree,
        adoption_commit=commit,
        adoption_tree=tree,
        adoption_parent=baseline,
        path_rows=rows,
        task_origin_paths=[row["path"] for row in rows[:17]],
        orphan_paths=[row["path"] for row in rows[17:]],
        disposition_rows=dispositions,
        focused_test_rows=[
            {
                "command": "pytest adoption",
                "exit_code": 0,
                "stdout_sha256": "f" * 64,
                "status": "PASS",
            }
        ],
        full_gate_refs=gate_refs,
        source_task_completion={
            "status": "COMPLETED_WITHOUT_COMMIT",
            "latest_turn_id": "01a0086f-f4f5-7bd2-873e-4e874369c021",
            "completed_at": BASE,
            "final_message_sha256": "e" * 64,
        },
        readback_rows=readbacks,
        user_authorization_basis="explicit repository-owner prospective adoption",
        writer_ended=True,
        main_clean=True,
        created_at=BASE,
    )
    return root, adoption, gate_refs


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
    gate_evidence = _gate_evidence("4" * 40, "5" * 40, handoff_ref)
    adoption = _adoption(gate_evidence)
    assert validate_main_checkout_adoption(adoption) == adoption
    adoption_ref = object_ref_for_artifact(adoption)
    authorization = build_final_cutover_authorization(
        final_authorization_id="unified-final-cutover",
        accepted_baseline_commit="a" * 40,
        historical_integration_commit="3" * 40,
        historical_dirty_evidence_ref=handoff_ref,
        concurrent_task_handoff_ref=None,
        main_checkout_adoption_ref=adoption_ref,
        legacy_disposition_ref=disposition_ref,
        deployed_release_ref=handoff_ref,
        release_commit="4" * 40,
        release_tree="5" * 40,
        final_integration_commit="4" * 40,
        final_integration_tree="5" * 40,
        ancestry_rows=[
            {"ancestor": "a" * 40, "descendant": "4" * 40, "proved": True},
            {"ancestor": "b" * 40, "descendant": "4" * 40, "proved": True},
        ],
        excluded_commit_rows=[],
        final_worktree_inventory_sha256="6" * 64,
        clean_checkout_readback_rows=_readbacks("4" * 40, "5" * 40),
        user_authorization_basis="explicit current-task production authorization",
        preflight_evidence=gate_evidence,
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
            concurrent_task_handoff_ref=None,
            main_checkout_adoption_ref=adoption_ref,
            legacy_disposition_ref=disposition_ref,
            deployed_release_ref=handoff_ref,
            release_commit="4" * 40,
            release_tree="5" * 40,
            final_integration_commit="4" * 40,
            final_integration_tree="5" * 40,
            ancestry_rows=[{"ancestor": "a" * 40, "descendant": "4" * 40, "proved": True}],
            excluded_commit_rows=[],
            final_worktree_inventory_sha256="6" * 64,
            clean_checkout_readback_rows=_readbacks("4" * 40, "5" * 40),
            user_authorization_basis="explicit current-task production authorization",
            preflight_evidence=[],
            created_at=BASE,
        )


def test_main_checkout_adoption_deeply_replays_exact_git_delta(tmp_path: Path) -> None:
    root, adoption, gate_refs = _real_adoption(tmp_path)
    assert validate_main_checkout_adoption_closure(
        adoption,
        repository_root=root,
        final_commit=adoption["payload"]["adoption_commit"],
        final_preflight_rows=gate_refs,
    ) == adoption


@pytest.mark.parametrize(
    "mutation",
    ["commit", "parent", "tree", "status", "mode", "blob", "size", "sha"],
)
def test_main_checkout_adoption_git_identity_tamper_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, adoption, gate_refs = _real_adoption(tmp_path)
    payload = dict(adoption["payload"])
    if mutation == "commit":
        payload["adoption_commit"] = "f" * 40
    elif mutation == "parent":
        payload["adoption_parent"] = "f" * 40
    elif mutation == "tree":
        payload["adoption_tree"] = "f" * 40
    else:
        path_rows = [dict(row) for row in payload["path_rows"]]
        if mutation == "status":
            path_rows[0]["status"] = "MODIFIED"
        elif mutation == "mode":
            path_rows[0]["mode"] = "100755"
        elif mutation == "blob":
            path_rows[0]["git_blob_oid"] = "f" * 40
        elif mutation == "size":
            path_rows[0]["size"] = int(path_rows[0]["size"]) + 1
        else:
            path_rows[0]["byte_sha256"] = "f" * 64
        payload["path_rows"] = path_rows
    tampered = seal_artifact(
        "system.main_checkout_adoption",
        payload,
        created_at=adoption["created_at"],
    )
    with pytest.raises((SystemContractError, SystemPreconditionError)):
        validate_main_checkout_adoption_closure(
            tampered,
            repository_root=root,
            final_commit=adoption["payload"]["adoption_commit"],
            final_preflight_rows=gate_refs,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "reordered"])
def test_main_checkout_adoption_path_set_tamper_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, adoption, gate_refs = _real_adoption(tmp_path)
    payload = dict(adoption["payload"])
    path_rows = [dict(row) for row in payload["path_rows"]]
    if mutation == "missing":
        path_rows.pop()
    elif mutation == "extra":
        extra = dict(path_rows[-1])
        extra["path"] = "adopted/path-99.txt"
        path_rows.append(extra)
    elif mutation == "duplicate":
        path_rows[-1] = dict(path_rows[-2])
    else:
        path_rows[0], path_rows[1] = path_rows[1], path_rows[0]
    payload["path_rows"] = path_rows
    tampered = seal_artifact(
        "system.main_checkout_adoption",
        payload,
        created_at=adoption["created_at"],
    )
    with pytest.raises((SystemContractError, SystemPreconditionError)):
        validate_main_checkout_adoption_closure(
            tampered,
            repository_root=root,
            final_commit=adoption["payload"]["adoption_commit"],
            final_preflight_rows=gate_refs,
        )


def test_arbitrary_command_exit_zero_cannot_be_sealed_as_a_cutover_gate() -> None:
    subject_ref = object_ref_for_artifact(_handoff())
    row = _gate_evidence("4" * 40, "5" * 40, subject_ref)[0]["payload"]["batch_results"][0]
    forged = dict(row)
    forged["argv"] = ["sh", "-c", "exit 0"]
    with pytest.raises(SystemContractError, match="fixed runner"):
        _seal_cutover_gate_evidence(
            gate_id=sorted(REQUIRED_FINAL_PREFLIGHT_GATES)[0],
            final_commit="4" * 40,
            final_tree="5" * 40,
            runner_code_sha256="7" * 64,
            environment_sha256="8" * 64,
            batch_results=[forged],
            subject_ref=subject_ref,
            started_at=BASE,
            finished_at=BASE,
        )


def test_system_owned_gate_runner_executes_only_fixed_argv(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Gate Test"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "gate@example.invalid"],
        check=True,
    )
    source_root = Path(__file__).resolve().parents[2]
    tracked_package_paths = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "-z", "quant_investor"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.split(b"\0")
    for relative_raw in tracked_package_paths:
        if not relative_raw:
            continue
        relative = relative_raw.decode("utf-8")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / relative).read_bytes())
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "gate runner"], check=True)
    commit = (
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD^{commit}"],
            check=True,
            stdout=subprocess.PIPE,
        )
        .stdout.decode()
        .strip()
    )
    tree = (
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
            check=True,
            stdout=subprocess.PIPE,
        )
        .stdout.decode()
        .strip()
    )

    attached = run_cutover_gate(
        repository_root=root,
        gate_id="clean_detached_clone",
        final_commit=commit,
        final_tree=tree,
        subject_ref=object_ref_for_artifact(_handoff()),
    )
    assert validate_cutover_gate_evidence(attached)["payload"]["state"] == "FAIL"

    subprocess.run(["git", "-C", str(root), "checkout", "--detach", "-q"], check=True)
    evidence = run_cutover_gate(
        repository_root=root,
        gate_id="clean_detached_clone",
        final_commit=commit,
        final_tree=tree,
        subject_ref=object_ref_for_artifact(_handoff()),
    )
    validated = validate_cutover_gate_evidence(evidence)
    assert validated["payload"]["state"] == "PASS", base64.b64decode(
        validated["payload"]["batch_results"][0]["stderr_base64"]
    ).decode("utf-8", errors="replace")
    assert validated["payload"]["batch_results"][0]["argv"] == [
        "FROZEN_PYTHON",
        "-m",
        "quant_investor.system.release_install",
        "verify-detached-checkout",
    ]

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from quant_investor.factors import governance_candidate_preregistration_bundle_v4_3 as subject
from quant_investor.factors import governance_candidate_preregistration_v4_3 as prereg
from quant_investor.factors import governance_private_bundle_io as private_io
from scripts import build_factor_v4_3_candidate_preregistration as cli


@pytest.fixture(scope="session")
def publication_inputs() -> Any:
    return cli._collect_publication_inputs(
        repository_root=cli.PROJECT_ROOT,
        protected_specs=cli.PROTECTED_BINDING_SPECS,
    )


@pytest.fixture(scope="session")
def artifacts(publication_inputs: Any) -> dict[str, dict[str, Any]]:
    source = prereg.build_aquant_source_set_receipt_v4_3(
        aquant_git_objects=publication_inputs.aquant_git_objects
    )
    operator = prereg.build_operator_semantics_v4_3()
    comparison = subject.build_comparison_catalog_receipt_v4_3()
    selection = prereg.build_selection_spec_v4_3(
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        preregistered_at="2026-07-19T12:00:00+08:00",
    )
    code = subject.build_code_binding_set_v4_3(
        repository_root=cli.PROJECT_ROOT,
        code_bindings=publication_inputs.code_bindings,
    )
    v4_2_contract_lock = subject.build_v4_2_contract_lock_v4_3()
    return subject.build_candidate_preregistration_bundle_artifacts_v4_3(
        aquant_source_set_receipt=source,
        operator_semantics=operator,
        comparison_catalog_receipt=comparison,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=publication_inputs.strict_source_binding,
        code_binding_set=code,
        v4_2_contract_lock=v4_2_contract_lock,
    )


def _complete(
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for filename in subject.INPUT_FILENAMES_V4_3:
        raw = prereg.canonical_file_bytes_v4_3(artifacts[filename])
        bindings.append(
            {
                "filename": filename,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "mode": 0o600,
                "uid": os.getuid(),
                "nlink": 1,
            }
        )
    report = subject._build_readback_report(
        run_id=subject.CYCLE_ID_V4_3,
        artifacts=artifacts,
        artifact_bindings=bindings,
    )
    return {**copy.deepcopy(artifacts), subject.READBACK_REPORT_FILENAME_V4_3: report}


def _portable_private_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    def rename_exclusive(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        try:
            os.stat(
                destination_name,
                dir_fd=destination_directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(destination_name)
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=source_directory_fd,
            dst_dir_fd=destination_directory_fd,
        )

    monkeypatch.setattr(private_io, "_require_exclusive_rename_support", lambda: None)
    monkeypatch.setattr(private_io, "_renameatx_np_exclusive", rename_exclusive)


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path.joinpath(*subject.ROOT_SUFFIX_V4_3)
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


def _copy_v4_2_locked_tree(tmp_path: Path) -> Path:
    repository_root = tmp_path / "repository"
    for spec in subject.V4_2_LOCKED_FILES_V4_3:
        source = subject.REPOSITORY_ROOT_V4_3 / spec["relative_path"]
        destination = repository_root / spec["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o644)
    return repository_root


@pytest.mark.parametrize(
    "locked_index",
    range(len(subject.V4_2_LOCKED_FILES_V4_3)),
    ids=[row["lock_role"] for row in subject.V4_2_LOCKED_FILES_V4_3],
)
def test_each_v4_2_locked_file_byte_drift_is_rejected_at_runtime(
    locked_index: int,
    tmp_path: Path,
) -> None:
    repository_root = _copy_v4_2_locked_tree(tmp_path)
    accepted = subject.build_v4_2_contract_lock_v4_3(
        repository_root=repository_root
    )
    assert accepted["locked_file_count"] == 6

    spec = subject.V4_2_LOCKED_FILES_V4_3[locked_index]
    target = repository_root / spec["relative_path"]
    target.write_bytes(target.read_bytes() + b"\n# runtime drift\n")
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="byte SHA mismatch",
    ):
        subject.build_v4_2_contract_lock_v4_3(
            repository_root=repository_root
        )


@pytest.mark.parametrize(
    ("tamper", "error_pattern"),
    (
        ("symlink", "hard-link contract"),
        ("non_regular", "hard-link contract"),
        ("wrong_mode", "mode mismatch"),
        ("hard_link", "hard-link contract"),
        ("wrong_hash", "byte SHA mismatch"),
    ),
)
def test_v4_2_contract_lock_rejects_unsafe_runtime_file_identity(
    tamper: str,
    error_pattern: str,
    tmp_path: Path,
) -> None:
    repository_root = _copy_v4_2_locked_tree(tmp_path)
    target = repository_root / subject.V4_2_LOCKED_FILES_V4_3[1]["relative_path"]
    if tamper == "symlink":
        replacement = (
            repository_root
            / subject.V4_2_LOCKED_FILES_V4_3[0]["relative_path"]
        )
        target.unlink()
        target.symlink_to(replacement)
    elif tamper == "non_regular":
        target.unlink()
        target.mkdir()
    elif tamper == "wrong_mode":
        target.chmod(0o600)
    elif tamper == "hard_link":
        os.link(target, repository_root / "unexpected-hard-link")
    else:
        target.write_bytes(b"wrong bytes")

    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match=error_pattern,
    ):
        subject.build_v4_2_contract_lock_v4_3(
            repository_root=repository_root
        )


def test_v4_2_contract_lock_rejects_wrong_owner_at_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = _copy_v4_2_locked_tree(tmp_path)
    actual_uid = os.getuid()
    monkeypatch.setattr(subject, "_assert_owned_nofollow_chain", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject.os, "getuid", lambda: actual_uid + 1)
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="owner/regular/non-symlink hard-link contract",
    ):
        subject.build_v4_2_contract_lock_v4_3(
            repository_root=repository_root
        )


def test_fixed_inventory_schemas_and_dual_sha_dag(
    artifacts: dict[str, dict[str, Any]],
) -> None:
    assert tuple(artifacts) == subject.INPUT_FILENAMES_V4_3
    assert len(artifacts) == 13
    assert subject.candidate_preregistration_bundle_contract_v4_3().canonical_filenames == (
        *subject.INPUT_FILENAMES_V4_3,
        subject.READBACK_REPORT_FILENAME_V4_3,
    )
    assert subject.CYCLE_ID_V4_3 == "cn_full_a_v4_3_20260717_20260717T172132Z"

    selection = artifacts[subject.CANDIDATE_SELECTION_SPEC_FILENAME_V4_3]
    assert [row["name"] for row in selection["predecessor_bindings"]] == [
        "aquant_source_set_receipt",
        "operator_semantics",
    ]
    future = artifacts[subject.FUTURE_SOURCE_ENVELOPE_FILENAME_V4_3]
    assert [row["name"] for row in future["predecessor_bindings"]] == [
        "selection_spec",
        "strict_source_binding",
        "code_binding_set",
    ]
    root = artifacts[subject.CYCLE_ROOT_FILENAME_V4_3]
    assert artifacts[subject.CODE_BINDING_SET_FILENAME_V4_3]["path_count"] == 10
    assert [row["name"] for row in root["ordered_predecessor_bindings"]] == [
        "selection_spec",
        "strict_source_binding",
        "code_binding_set",
        "future_source_envelope",
        "v4_2_contract_lock",
    ]
    contract_lock = root["v4_2_contract_lock"]
    assert contract_lock["locked_file_count"] == 6
    assert [
        {
            "order": row["order"],
            "lock_role": row["lock_role"],
            "relative_path": row["relative_path"],
            "expected_byte_sha256": row["expected_byte_sha256"],
        }
        for row in contract_lock["ordered_locked_files"]
    ] == list(subject.V4_2_LOCKED_FILES_V4_3)
    assert all(
        row["byte_sha256"] == row["expected_byte_sha256"]
        and row["mode"] == 0o644
        and row["uid"] == os.getuid()
        and row["nlink"] == 1
        for row in contract_lock["ordered_locked_files"]
    )
    assert root["ordered_predecessor_bindings"][-1] == (
        prereg.build_artifact_binding_v4_3(
            name="v4_2_contract_lock",
            artifact=contract_lock,
        )
    )
    collision = artifacts[
        subject.DEFINITION_IDENTITY_COLLISION_AUDIT_FILENAME_V4_3
    ]
    assert [row["name"] for row in collision["predecessor_bindings"]] == [
        "aquant_source_set_receipt",
        "comparison_catalog_receipt",
        "selection_spec",
    ]
    precommit = artifacts[subject.PRECOMMITTED_STATE_FILENAME_V4_3]
    assert precommit["state"] == "PRECOMMITTED"
    assert precommit["cycle_root_sha256"] == root["artifact_semantic_sha256"]
    assert precommit["source_chain_node_sha256"] == (
        prereg.build_precommit_source_chain_sha256_v4_3(future, collision)
    )
    source_node = artifacts[subject.DISCOVERY_SOURCE_NODE_FILENAME_V4_3]
    assert [row["name"] for row in source_node["predecessor_bindings"]] == [
        "precommitted_state",
        "selection_spec",
        "aquant_source_set_receipt",
    ]
    discovery = artifacts[subject.DISCOVERY_STATE_FILENAME_V4_3]
    assert discovery["state"] == "DISCOVERY"
    assert set(discovery["predecessor"]) == {
        "kind",
        "byte_sha256",
        "semantic_sha256",
    }
    orchestration = artifacts[
        subject.PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_3
    ]
    assert len(orchestration["graph_bindings"]) == 12
    assert {row["name"] for row in orchestration["graph_bindings"]} >= {
        "cycle_root",
        "precommitted_state",
        "discovery_state",
        "prereg_discovery_source_node",
    }
    for artifact in (
        selection,
        future,
        root,
        collision,
        source_node,
        orchestration,
    ):
        for row in artifact.get("predecessor_bindings", artifact.get("graph_bindings", [])):
            assert set(row) == {"name", "byte_sha256", "semantic_sha256"}


def test_readback_report_is_exact_precommit_intent_and_binds_all_inputs(
    artifacts: dict[str, dict[str, Any]],
) -> None:
    complete = _complete(artifacts)
    normalized = subject.validate_candidate_preregistration_bundle_artifacts_v4_3(
        complete
    )
    report = normalized[subject.READBACK_REPORT_FILENAME_V4_3]
    assert report["publication_phase"] == "PRECOMMIT_INTENT_ONLY"
    assert report["exclusive_rename_completed"] is False
    assert report["durability_commit_verified"] is False
    assert report["publication_authority"] is False
    assert [row["filename"] for row in report["artifact_bindings"]] == list(
        subject.INPUT_FILENAMES_V4_3
    )
    for row in report["artifact_bindings"]:
        assert set(row) == {
            "filename",
            "byte_sha256",
            "semantic_sha256",
            "size_bytes",
            "mode",
            "uid",
            "nlink",
        }
    root = artifacts[subject.CYCLE_ROOT_FILENAME_V4_3]
    root_readback_binding = next(
        row
        for row in report["artifact_bindings"]
        if row["filename"] == subject.CYCLE_ROOT_FILENAME_V4_3
    )
    assert root_readback_binding["semantic_sha256"] == root[
        "artifact_semantic_sha256"
    ]
    assert subject.validate_v4_2_contract_lock_v4_3(
        root["v4_2_contract_lock"]
    )["ordered_locked_files"] == root["v4_2_contract_lock"][
        "ordered_locked_files"
    ]

    tampered = copy.deepcopy(complete)
    tampered_report = tampered[subject.READBACK_REPORT_FILENAME_V4_3]
    tampered_report["artifact_bindings"][0]["semantic_sha256"] = "0" * 64
    tampered_report["artifact_semantic_sha256"] = prereg.semantic_sha256_v4_3(
        {
            key: value
            for key, value in tampered_report.items()
            if key != "artifact_semantic_sha256"
        }
    )
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="byte/semantic binding mismatch",
    ):
        subject.validate_candidate_preregistration_bundle_artifacts_v4_3(tampered)


def test_historical_readback_rebuild_uses_embedded_v4_2_lock_without_live_scan(
    artifacts: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = _complete(artifacts)

    def forbidden(**_kwargs: Any) -> Any:
        raise AssertionError("historical readback must not reopen current v4.2 files")

    monkeypatch.setattr(
        subject,
        "_observe_v4_2_contract_lock_snapshot",
        forbidden,
    )
    normalized = subject.validate_candidate_preregistration_bundle_artifacts_v4_3(
        complete
    )
    assert normalized[subject.CYCLE_ROOT_FILENAME_V4_3][
        "v4_2_contract_lock"
    ] == artifacts[subject.CYCLE_ROOT_FILENAME_V4_3]["v4_2_contract_lock"]


@pytest.mark.parametrize(
    ("tamper", "error_pattern"),
    (
        ("embedded_row", "SHA/mode mismatch"),
        ("dual_sha_binding", "contract lock binding mismatch"),
    ),
)
def test_cycle_root_rejects_v4_2_contract_lock_row_or_dual_sha_tamper(
    tamper: str,
    error_pattern: str,
    artifacts: dict[str, dict[str, Any]],
) -> None:
    root = copy.deepcopy(artifacts[subject.CYCLE_ROOT_FILENAME_V4_3])
    if tamper == "embedded_row":
        root["v4_2_contract_lock"]["ordered_locked_files"][0][
            "byte_sha256"
        ] = "0" * 64
    else:
        root["ordered_predecessor_bindings"][-1]["semantic_sha256"] = "0" * 64
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match=error_pattern,
    ):
        subject.validate_cycle_root_v4_3(root)


def test_strict_source_and_comparison_identity_tamper_fail_closed(
    publication_inputs: Any,
) -> None:
    strict = copy.deepcopy(publication_inputs.strict_source_binding)
    strict["ordered_source_file_bindings"][0]["byte_sha256"] = "0" * 64
    strict["artifact_semantic_sha256"] = prereg.semantic_sha256_v4_3(
        {
            key: value
            for key, value in strict.items()
            if key != "artifact_semantic_sha256"
        }
    )
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="descriptor mismatch",
    ):
        subject.validate_strict_full_a_source_binding_v4_3(strict)

    source_raw = subject.V4_2_IDENTITY_SOURCE_PATH_V4_3.read_bytes()
    changed = source_raw.replace(
        b"8e486283e2c36a4ecdfcd4059811afb4e42e75f53a6575f972ee17f2665a826f",
        b"0" * 64,
        1,
    )
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="range constants mismatch|definition hashes mismatch",
    ):
        subject._v4_2_identity_inventory(changed)
    duplicate = source_raw + b"\nEXPECTED_CANDIDATES = ()\n"
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="exactly once",
    ):
        subject._v4_2_identity_inventory(duplicate)


def test_owner_nlink1_descriptors_reject_relative_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="absolute",
    ):
        subject._absolute_path("relative.json", "test")

    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="hard-link contract",
    ):
        subject._stable_file(symlink, boundary=tmp_path, label="symlink")

    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="hard-link contract",
    ):
        subject._stable_file(target, boundary=tmp_path, label="hardlink")


def test_immutable_inventory_preserves_legal_hard_link_counts(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    first = root / "a.parquet"
    first.write_bytes(b"member")
    second = root / "b.parquet"
    os.link(first, second)

    snapshot = subject._stable_tree(root, boundary=tmp_path, label="hardlink tree")

    assert len(snapshot.inventory) == 2
    assert {row["hard_link_count"] for row in snapshot.inventory} == {2}
    expected = hashlib.sha256(
        prereg.canonical_file_bytes_v4_3(list(snapshot.inventory))
    ).hexdigest()
    assert snapshot.summary["inventory_semantic_sha256"] == expected


def test_code_and_protected_descriptor_drift_is_rejected(tmp_path: Path) -> None:
    def descriptor(path: Path) -> dict[str, Any]:
        metadata = path.stat()
        raw = path.read_bytes()
        return {
            "absolute_path": str(path),
            "byte_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "mode": metadata.st_mode & 0o7777,
            "uid": metadata.st_uid,
            "nlink": metadata.st_nlink,
        }

    code_rows: list[dict[str, Any]] = []
    for index, relative in enumerate(subject.CODE_BINDING_PATHS_V4_3):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"code-{index}".encode())
        code_rows.append({"relative_path": relative, **descriptor(path)})
    code = subject.build_code_binding_set_v4_3(
        repository_root=tmp_path,
        code_bindings=code_rows,
    )
    (tmp_path / subject.CODE_BINDING_PATHS_V4_3[0]).write_bytes(b"drift")
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="descriptor mismatch",
    ):
        subject.revalidate_code_binding_set_v4_3(
            repository_root=tmp_path,
            code_bindings=code_rows,
            value=code,
        )

    protected_rows: list[dict[str, Any]] = []
    for name, relative in zip(
        subject.PROTECTED_BINDING_NAMES_V4_3,
        subject.PROTECTED_BINDING_RELATIVE_PATHS_V4_3,
        strict=True,
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        protected_rows.append({"name": name, **descriptor(path)})
    subject.validate_protected_bindings_v4_3(
        repository_root=tmp_path,
        protected_bindings=protected_rows,
    )
    (tmp_path / subject.PROTECTED_BINDING_RELATIVE_PATHS_V4_3[-1]).write_bytes(
        b"changed"
    )
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="descriptor mismatch",
    ):
        subject.validate_protected_bindings_v4_3(
            repository_root=tmp_path,
            protected_bindings=protected_rows,
        )


def test_private_publication_committed_truth_exact_readback_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifacts: dict[str, dict[str, Any]],
) -> None:
    _portable_private_publication(monkeypatch)
    root = _private_root(tmp_path)
    published = private_io.publish_private_bundle(
        private_root=root,
        run_id=subject.CYCLE_ID_V4_3,
        artifacts=artifacts,
        contract=subject.candidate_preregistration_bundle_contract_v4_3(),
        revalidate_inputs=lambda: None,
    )
    report = published["readback_report"]
    assert report["publication_phase"] == "PRECOMMIT_INTENT_ONLY"
    assert report["exclusive_rename_completed"] is False
    bundle = Path(published["bundle_path"])
    assert len(list(bundle.iterdir())) == 14
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in bundle.iterdir())
    assert all(path.stat().st_nlink == 1 for path in bundle.iterdir())

    report_descriptor = published["artifact_descriptors"][
        subject.READBACK_REPORT_FILENAME_V4_3
    ]
    reread = subject.readback_candidate_preregistration_bundle_v4_3(
        bundle_path=bundle,
        expected_readback_report_byte_sha256=report_descriptor["byte_sha256"],
        expected_readback_report_semantic_sha256=report[
            "artifact_semantic_sha256"
        ],
    )
    assert reread["accepted"] is True
    assert reread["expected_hashes_verified"] is True
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_3Error,
        match="expected readback report byte SHA mismatch",
    ):
        subject.readback_candidate_preregistration_bundle_v4_3(
            bundle_path=bundle,
            expected_readback_report_byte_sha256="0" * 64,
            expected_readback_report_semantic_sha256=report[
                "artifact_semantic_sha256"
            ],
        )
    with pytest.raises(private_io.FactorGovernancePrivateBundleIOError, match="already exists"):
        private_io.publish_private_bundle(
            private_root=root,
            run_id=subject.CYCLE_ID_V4_3,
            artifacts=artifacts,
            contract=subject.candidate_preregistration_bundle_contract_v4_3(),
            revalidate_inputs=lambda: None,
        )


def test_publication_failure_before_rename_leaves_no_canonical_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifacts: dict[str, dict[str, Any]],
) -> None:
    _portable_private_publication(monkeypatch)
    root = _private_root(tmp_path)

    def fail(point: str) -> None:
        if point == "commit:rename:before":
            raise RuntimeError("stop before canonical rename")

    with pytest.raises(private_io.FactorGovernancePrivateBundleIOError):
        private_io.publish_private_bundle(
            private_root=root,
            run_id=subject.CYCLE_ID_V4_3,
            artifacts=artifacts,
            contract=subject.candidate_preregistration_bundle_contract_v4_3(),
            revalidate_inputs=lambda: None,
            _test_fault_hook=fail,
        )
    assert not (root / subject.CYCLE_ID_V4_3).exists()

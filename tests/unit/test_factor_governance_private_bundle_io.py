from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import threading
from typing import Any

import pytest

from quant_investor.factors import governance_private_bundle_io as subject


ROOT_SUFFIX = (
    "reports",
    "factor_governance",
    "private",
    "formal_catalog_v4_1",
)
INPUT_FILENAMES = (
    "alpha.v4_1.json",
    "beta.v4_1.json",
)
REPORT_FILENAME = "formal_catalog_materialization_readback.v4_1.json"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _validate_artifact(filename: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("artifact must be an object")
    if filename == REPORT_FILENAME:
        expected = {
            "schema_version",
            "filename",
            "run_id",
            "artifact_bindings",
        }
        if set(value) != expected:
            raise ValueError("readback report fields mismatch")
        if value.get("schema_version") != "test-private-readback.v1":
            raise ValueError("readback report schema mismatch")
        if value.get("filename") != REPORT_FILENAME:
            raise ValueError("readback report filename mismatch")
        if not isinstance(value.get("run_id"), str):
            raise ValueError("readback run id mismatch")
        bindings = value.get("artifact_bindings")
        if not isinstance(bindings, list):
            raise ValueError("readback bindings mismatch")
    else:
        expected = {"schema_version", "filename", "cycle_id", "value"}
        if set(value) != expected:
            raise ValueError("input artifact fields mismatch")
        if value.get("schema_version") != "test-private-artifact.v1":
            raise ValueError("input artifact schema mismatch")
        if value.get("filename") != filename:
            raise ValueError("input artifact filename mismatch")
        if value.get("cycle_id") != "cycle-test":
            raise ValueError("input artifact cycle mismatch")
        if type(value.get("value")) is not int:
            raise ValueError("input artifact value mismatch")
    return dict(value)


def _build_report(
    *,
    run_id: str,
    artifacts: dict[str, dict[str, Any]],
    artifact_bindings: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    assert tuple(artifacts) == INPUT_FILENAMES
    assert tuple(item["filename"] for item in artifact_bindings) == INPUT_FILENAMES
    return {
        "schema_version": "test-private-readback.v1",
        "filename": REPORT_FILENAME,
        "run_id": run_id,
        "artifact_bindings": [dict(item) for item in artifact_bindings],
    }


def _validate_complete(
    values: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(values) != {*INPUT_FILENAMES, REPORT_FILENAME}:
        raise ValueError("complete inventory mismatch")
    report = values[REPORT_FILENAME]
    bindings = report["artifact_bindings"]
    if [item["filename"] for item in bindings] != list(INPUT_FILENAMES):
        raise ValueError("complete binding membership mismatch")
    binding_by_name = {item["filename"]: item for item in bindings}
    for filename in INPUT_FILENAMES:
        expected_sha = hashlib.sha256(_canonical(values[filename])).hexdigest()
        if binding_by_name[filename]["byte_sha256"] != expected_sha:
            raise ValueError("complete byte binding mismatch")
    return {filename: dict(value) for filename, value in values.items()}


def _contract() -> subject.PrivateBundleContract:
    return subject.PrivateBundleContract(
        root_suffix=ROOT_SUFFIX,
        input_filenames=INPUT_FILENAMES,
        readback_report_filename=REPORT_FILENAME,
        canonicalize=_canonical,
        validate_artifact=_validate_artifact,
        validate_complete=_validate_complete,
        build_readback_report=_build_report,
    )


def _artifacts(*, offset: int = 0) -> dict[str, dict[str, Any]]:
    return {
        filename: {
            "schema_version": "test-private-artifact.v1",
            "filename": filename,
            "cycle_id": "cycle-test",
            "value": index + offset,
        }
        for index, filename in enumerate(INPUT_FILENAMES, start=1)
    }


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path.joinpath(*ROOT_SUFFIX)
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


_portable_rename_lock = threading.Lock()
_real_exclusive_rename = subject._renameatx_np_exclusive


@pytest.fixture(autouse=True)
def _portable_exclusive_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    def rename_exclusive(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        # Test-only emulation.  Production deliberately has no fallback.
        with _portable_rename_lock:
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

    monkeypatch.setattr(subject, "_require_exclusive_rename_support", lambda: None)
    monkeypatch.setattr(subject, "_renameatx_np_exclusive", rename_exclusive)


def _publish(
    root: Path,
    run_id: str = "formal-run",
    *,
    artifacts: dict[str, dict[str, Any]] | None = None,
    revalidate_inputs: Any = None,
    fault_hook: Any = None,
    race_hook: Any = None,
) -> dict[str, Any]:
    return subject.publish_private_bundle(
        private_root=root,
        run_id=run_id,
        artifacts=_artifacts() if artifacts is None else artifacts,
        contract=_contract(),
        revalidate_inputs=(
            (lambda: None) if revalidate_inputs is None else revalidate_inputs
        ),
        _test_fault_hook=fault_hook,
        _test_race_hook=race_hook,
    )


def _quarantined(root: Path, run_id: str) -> list[Path]:
    quarantine = root / subject.QUARANTINE_DIRECTORY
    if not quarantine.exists():
        return []
    return [path for path in quarantine.iterdir() if path.name.startswith(run_id)]


def test_module_is_active_factor_only_and_has_no_project_imports() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    project_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("quant_investor")
    ]
    assert project_imports == []


def test_publish_and_readback_are_exact_private_and_return_values(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    callback_count = 0

    def revalidate() -> None:
        nonlocal callback_count
        callback_count += 1

    result = _publish(root, revalidate_inputs=revalidate)

    assert callback_count == 1
    assert result["accepted"] is True
    assert set(result["artifacts"]) == {*INPUT_FILENAMES, REPORT_FILENAME}
    assert result["readback_report"] == result["artifacts"][REPORT_FILENAME]
    bundle = root / "formal-run"
    assert result["bundle_path"] == str(bundle)
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o700
    assert sorted(path.name for path in bundle.iterdir()) == sorted(
        (*INPUT_FILENAMES, REPORT_FILENAME)
    )
    for path in bundle.iterdir():
        metadata = path.stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1
        descriptor = result["artifact_descriptors"][path.name]
        assert descriptor["absolute_path"] == str(path)
        assert descriptor["byte_sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    reread = subject.readback_private_bundle(bundle, contract=_contract())
    assert reread == result


def test_publication_rejects_missing_extra_and_unsafe_paths(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    missing = _artifacts()
    missing.pop(INPUT_FILENAMES[0])
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="input artifact set mismatch",
    ):
        _publish(root, "missing", artifacts=missing)

    extra = _artifacts()
    extra["unexpected.v1.json"] = {}
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="input artifact set mismatch",
    ):
        _publish(root, "extra", artifacts=extra)

    for run_id in ("", ".hidden", "../escape", "a/b", "/absolute", "a..b"):
        with pytest.raises(
            subject.FactorGovernancePrivateBundleIOError,
            match="safe path segment",
        ):
            _publish(root, run_id)

    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="absolute normalized path",
    ):
        subject.publish_private_bundle(
            private_root="relative/root",
            run_id="relative-root",
            artifacts=_artifacts(),
            contract=_contract(),
            revalidate_inputs=lambda: None,
        )


def test_mutating_complete_validator_cannot_launder_staged_bytes(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)

    def mutate_nested_binding(
        values: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        values[REPORT_FILENAME]["artifact_bindings"][0]["byte_sha256"] = "0" * 64
        return values

    contract = subject.PrivateBundleContract(
        root_suffix=ROOT_SUFFIX,
        input_filenames=INPUT_FILENAMES,
        readback_report_filename=REPORT_FILENAME,
        canonicalize=_canonical,
        validate_artifact=_validate_artifact,
        validate_complete=mutate_nested_binding,
        build_readback_report=_build_report,
    )
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="cross-validator changed canonical",
    ):
        subject.publish_private_bundle(
            private_root=root,
            run_id="mutating-validator",
            artifacts=_artifacts(),
            contract=contract,
            revalidate_inputs=lambda: None,
        )
    assert not (root / "mutating-validator").exists()


def test_duplicate_run_id_is_no_clobber(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    first = _publish(root, "same-run")
    before = {
        name: (root / "same-run" / name).read_bytes()
        for name in (*INPUT_FILENAMES, REPORT_FILENAME)
    }

    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="already exists",
    ):
        _publish(root, "same-run", artifacts=_artifacts(offset=100))

    assert first["accepted"] is True
    assert before == {
        name: (root / "same-run" / name).read_bytes()
        for name in (*INPUT_FILENAMES, REPORT_FILENAME)
    }


def test_concurrent_writers_yield_exactly_one_success(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    results: list[dict[str, Any]] = []
    errors: list[Exception] = []
    result_lock = threading.Lock()

    def worker(offset: int) -> None:
        try:
            result = _publish(
                root,
                "concurrent-run",
                artifacts=_artifacts(offset=offset),
            )
        except Exception as exc:  # noqa: BLE001 - asserted below
            with result_lock:
                errors.append(exc)
        else:
            with result_lock:
                results.append(result)

    threads = [threading.Thread(target=worker, args=(offset,)) for offset in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 1
    assert results[0]["accepted"] is True
    assert len(errors) == 3
    assert all(
        isinstance(error, subject.FactorGovernancePrivateBundleIOError)
        and "already exists" in str(error)
        for error in errors
    )
    assert subject.readback_private_bundle(
        root / "concurrent-run",
        contract=_contract(),
    )["accepted"] is True


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin primitive")
def test_real_renameatx_np_exclusive_never_clobbers(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(FileExistsError):
            _real_exclusive_rename(
                parent_fd,
                source.name,
                parent_fd,
                destination.name,
            )
        assert source.is_dir()
        assert sentinel.read_text(encoding="utf-8") == "preserve"

        destination.rename(tmp_path / "occupied")
        _real_exclusive_rename(
            parent_fd,
            source.name,
            parent_fd,
            "committed",
        )
        assert not source.exists()
        assert (tmp_path / "committed").is_dir()
    finally:
        os.close(parent_fd)


def test_reader_never_observes_hidden_partial_bundle(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    staged = threading.Event()
    release = threading.Event()
    results: list[dict[str, Any]] = []
    errors: list[Exception] = []

    def revalidate() -> None:
        staged.set()
        assert release.wait(timeout=10)

    def writer() -> None:
        try:
            results.append(
                _publish(root, "reader-race", revalidate_inputs=revalidate)
            )
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    assert staged.wait(timeout=10)
    assert not (root / "reader-race").exists()
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="canonical private bundle is missing",
    ):
        subject.readback_private_bundle(
            root / "reader-race",
            contract=_contract(),
        )
    assert any(path.name.startswith(".reader-race.staging.") for path in root.iterdir())
    release.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0]["accepted"] is True


@pytest.mark.parametrize(
    "fault_point",
    (
        "staging-created:root-fsync:before",
        f"write:{INPUT_FILENAMES[0]}:chunk-0:after",
        f"file-fsync:{INPUT_FILENAMES[0]}:before",
        "staging-input:directory-fsync:before",
        f"staging-input:pass-1:{INPUT_FILENAMES[0]}:before",
        f"file-fsync:{REPORT_FILENAME}:before",
        "staging-complete:directory-fsync:before",
        f"staging-complete:pass-2:{REPORT_FILENAME}:after",
        "precommit:root-fsync:before",
        "commit:rename:before",
    ),
)
def test_precommit_faults_leave_no_final_and_quarantine_owned_staging(
    tmp_path: Path,
    fault_point: str,
) -> None:
    root = _private_root(tmp_path)
    run_id = "precommit-fault"

    def inject(point: str) -> None:
        if point == fault_point:
            raise OSError("injected")

    with pytest.raises(subject.FactorGovernancePrivateBundleIOError):
        _publish(root, run_id, fault_hook=inject)

    assert not (root / run_id).exists()
    assert _quarantined(root, run_id)


@pytest.mark.parametrize(
    "fault_point",
    (
        "commit:rename:after",
        "commit:root-fsync:before",
        f"canonical-readback:pass-1:{INPUT_FILENAMES[0]}:before",
        f"canonical-readback:pass-2:{REPORT_FILENAME}:after",
    ),
)
def test_postcommit_faults_remove_final_to_identity_bound_quarantine(
    tmp_path: Path,
    fault_point: str,
) -> None:
    root = _private_root(tmp_path)
    run_id = "postcommit-fault"

    def inject(point: str) -> None:
        if point == fault_point:
            raise OSError("injected")

    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="POSTCOMMIT_RECOVERED_FAIL_CLOSED",
    ):
        _publish(root, run_id, fault_hook=inject)

    assert not (root / run_id).exists()
    quarantined = _quarantined(root, run_id)
    assert quarantined
    assert all(path.is_dir() for path in quarantined)


def test_partial_write_and_file_fsync_faults_never_create_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    real_write = subject.os.write
    write_calls = 0

    def partial_then_fail(descriptor: int, value: Any) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            length = max(1, len(value) // 2)
            return real_write(descriptor, value[:length])
        if write_calls == 2:
            raise OSError("injected partial-write failure")
        return real_write(descriptor, value)

    monkeypatch.setattr(subject.os, "write", partial_then_fail)
    with pytest.raises(subject.FactorGovernancePrivateBundleIOError):
        _publish(root, "partial-write")
    assert not (root / "partial-write").exists()
    assert _quarantined(root, "partial-write")

    monkeypatch.setattr(subject.os, "write", real_write)
    real_fsync = subject.os.fsync
    failed = False

    def fail_first_nonempty_file_fsync(descriptor: int) -> None:
        nonlocal failed
        metadata = os.fstat(descriptor)
        if not failed and stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0:
            failed = True
            raise OSError("injected file fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(subject.os, "fsync", fail_first_nonempty_file_fsync)
    with pytest.raises(subject.FactorGovernancePrivateBundleIOError):
        _publish(root, "file-fsync")
    assert failed is True
    assert not (root / "file-fsync").exists()
    assert _quarantined(root, "file-fsync")


@pytest.mark.parametrize("error_type", (OSError, FileExistsError))
def test_rename_that_commits_then_errors_is_quarantined_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    root = _private_root(tmp_path)
    real_rename = subject._renameatx_np_exclusive
    injected = False

    def commit_then_error(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal injected
        real_rename(
            source_directory_fd,
            source_name,
            destination_directory_fd,
            destination_name,
        )
        if destination_name == "uncertain-rename" and not injected:
            injected = True
            raise error_type("rename returned error after commit")

    monkeypatch.setattr(subject, "_renameatx_np_exclusive", commit_then_error)
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="POSTCOMMIT_RECOVERED_FAIL_CLOSED",
    ):
        _publish(root, "uncertain-rename")

    assert injected is True
    assert not (root / "uncertain-rename").exists()
    assert _quarantined(root, "uncertain-rename")


def test_postcommit_identity_swap_preserves_unrelated_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    real_rename = subject._renameatx_np_exclusive
    replacement: Path | None = None

    def swap_after_commit(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal replacement
        real_rename(
            source_directory_fd,
            source_name,
            destination_directory_fd,
            destination_name,
        )
        if destination_name == "postcommit-swap":
            target = root / destination_name
            target.rename(root / f".{destination_name}.original")
            target.mkdir(mode=0o700)
            sentinel = target / "replacement-owned"
            sentinel.write_text("preserve", encoding="utf-8")
            sentinel.chmod(0o600)
            replacement = root / source_name

    monkeypatch.setattr(subject, "_renameatx_np_exclusive", swap_after_commit)
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="staging source identity changed during exclusive commit",
    ):
        _publish(root, "postcommit-swap")

    assert not (root / "postcommit-swap").exists()
    assert replacement is not None
    assert (replacement / "replacement-owned").read_text(encoding="utf-8") == (
        "preserve"
    )


def test_unrecoverable_root_fsync_fault_leaves_only_complete_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    target = root / "ambiguous-root-fsync"
    real_fsync = subject.os.fsync

    def fail_root_after_commit(descriptor: int) -> None:
        opened = os.fstat(descriptor)
        root_metadata = root.stat()
        if (
            target.exists()
            and opened.st_dev == root_metadata.st_dev
            and opened.st_ino == root_metadata.st_ino
        ):
            raise OSError("injected persistent root fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(subject.os, "fsync", fail_root_after_commit)
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="AMBIGUOUS_DURABILITY_FAIL_CLOSED",
    ):
        _publish(root, target.name)

    assert target.is_dir()
    monkeypatch.setattr(subject.os, "fsync", real_fsync)
    assert subject.readback_private_bundle(
        target,
        contract=_contract(),
    )["accepted"] is True


def test_target_race_is_no_clobber_and_preserves_racer(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    target = root / "target-race"

    def race() -> None:
        target.mkdir(mode=0o700)
        sentinel = target / "racer-owned"
        sentinel.write_text("preserve", encoding="utf-8")
        sentinel.chmod(0o600)

    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="appeared during exclusive commit",
    ):
        _publish(root, "target-race", race_hook=race)

    assert (target / "racer-owned").read_text(encoding="utf-8") == "preserve"


def test_staging_swap_does_not_quarantine_unrelated_replacement(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    replacement: Path | None = None

    def race() -> None:
        nonlocal replacement
        staging = next(
            path
            for path in root.iterdir()
            if path.name.startswith(".staging-swap.staging.")
        )
        staging.rename(root / f"{staging.name}.original")
        staging.mkdir(mode=0o700)
        sentinel = staging / "replacement-owned"
        sentinel.write_text("preserve", encoding="utf-8")
        sentinel.chmod(0o600)
        replacement = staging

    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="staging directory path identity changed",
    ):
        _publish(root, "staging-swap", race_hook=race)

    assert not (root / "staging-swap").exists()
    assert replacement is not None
    assert (replacement / "replacement-owned").read_text(encoding="utf-8") == (
        "preserve"
    )


def test_late_staging_swap_is_rolled_back_without_exposing_final(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    replacement: Path | None = None

    def late_swap(point: str) -> None:
        nonlocal replacement
        if point != "commit:rename:before":
            return
        staging = next(
            path
            for path in root.iterdir()
            if path.name.startswith(".late-stage-swap.staging.")
        )
        staging.rename(root / f"{staging.name}.original")
        staging.mkdir(mode=0o700)
        sentinel = staging / "replacement-owned"
        sentinel.write_text("preserve", encoding="utf-8")
        sentinel.chmod(0o600)
        replacement = staging

    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="staging source identity changed during exclusive commit",
    ):
        _publish(root, "late-stage-swap", fault_hook=late_swap)

    assert not (root / "late-stage-swap").exists()
    assert replacement is not None
    assert (replacement / "replacement-owned").read_text(encoding="utf-8") == (
        "preserve"
    )


def test_late_swap_rollback_failure_identity_quarantines_canonical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    real_rename = subject._renameatx_np_exclusive

    def occupy_source_after_unexpected_commit(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        real_rename(
            source_directory_fd,
            source_name,
            destination_directory_fd,
            destination_name,
        )
        if destination_name == "rollback-block":
            blocker = root / source_name
            blocker.mkdir(mode=0o700)
            sentinel = blocker / "rollback-blocker-owned"
            sentinel.write_text("preserve", encoding="utf-8")
            sentinel.chmod(0o600)

    monkeypatch.setattr(
        subject,
        "_renameatx_np_exclusive",
        occupy_source_after_unexpected_commit,
    )

    def late_swap(point: str) -> None:
        if point != "commit:rename:before":
            return
        staging = next(
            path
            for path in root.iterdir()
            if path.name.startswith(".rollback-block.staging.")
        )
        staging.rename(root / f"{staging.name}.original")
        staging.mkdir(mode=0o700)
        sentinel = staging / "replacement-owned"
        sentinel.write_text("preserve", encoding="utf-8")
        sentinel.chmod(0o600)

    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="POSTCOMMIT_RECOVERED_FAIL_CLOSED",
    ):
        _publish(root, "rollback-block", fault_hook=late_swap)

    assert not (root / "rollback-block").exists()
    quarantined = _quarantined(root, "rollback-block")
    assert any((path / "replacement-owned").exists() for path in quarantined)
    blocker = next(
        path
        for path in root.iterdir()
        if path.name.startswith(".rollback-block.staging.")
        and (path / "rollback-blocker-owned").exists()
    )
    assert (blocker / "rollback-blocker-owned").read_text(encoding="utf-8") == (
        "preserve"
    )


def test_readback_rejects_unknown_missing_and_noncanonical_inventory(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    bundle = Path(_publish(root, "inventory")["bundle_path"])
    extra = bundle / "unexpected.json"
    extra.write_bytes(b"{}\n")
    extra.chmod(0o600)
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="artifact set mismatch",
    ):
        subject.readback_private_bundle(bundle, contract=_contract())

    extra.unlink()
    target = bundle / INPUT_FILENAMES[0]
    original = json.loads(target.read_bytes())
    target.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
    target.chmod(0o600)
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="not exact canonical file bytes",
    ):
        subject.readback_private_bundle(bundle, contract=_contract())

    target.unlink()
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="artifact set mismatch",
    ):
        subject.readback_private_bundle(bundle, contract=_contract())


@pytest.mark.parametrize("mutation", ("mode", "hardlink", "symlink", "fifo"))
def test_readback_rejects_unsafe_file_objects(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _private_root(tmp_path)
    bundle = Path(_publish(root, f"unsafe-{mutation}")["bundle_path"])
    target = bundle / INPUT_FILENAMES[0]
    if mutation == "mode":
        target.chmod(0o640)
        match = "mode must be 0600"
    elif mutation == "hardlink":
        os.link(target, tmp_path / "external-hardlink")
        match = "hard-link count must be one"
    elif mutation == "symlink":
        raw = target.read_bytes()
        outside = tmp_path / "outside.json"
        outside.write_bytes(raw)
        outside.chmod(0o600)
        target.unlink()
        target.symlink_to(outside)
        match = "regular non-symlink file"
    else:
        target.unlink()
        os.mkfifo(target, 0o600)
        match = "regular non-symlink file"

    with pytest.raises(subject.FactorGovernancePrivateBundleIOError, match=match):
        subject.readback_private_bundle(bundle, contract=_contract())


def test_readback_rejects_directory_and_owner_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    bundle = Path(_publish(root, "directory-drift")["bundle_path"])
    bundle.chmod(0o750)
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="mode must be 0700",
    ):
        subject.readback_private_bundle(bundle, contract=_contract())

    bundle.chmod(0o700)
    real_uid = os.getuid()
    monkeypatch.setattr(subject.os, "getuid", lambda: real_uid + 1)
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="owner mismatch",
    ):
        subject.readback_private_bundle(bundle, contract=_contract())


def test_readback_rejects_metadata_drift_between_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    bundle = Path(_publish(root, "stable-drift")["bundle_path"])
    real_read = subject._read_private_file
    calls = 0

    def mutate_after_first_pass(
        directory_fd: int,
        filename: str,
        *,
        max_bytes: int,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal calls
        result = real_read(directory_fd, filename, max_bytes=max_bytes)
        calls += 1
        if calls == len((*INPUT_FILENAMES, REPORT_FILENAME)):
            target = bundle / INPUT_FILENAMES[0]
            current = target.stat()
            os.utime(
                target,
                ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
            )
        return result

    monkeypatch.setattr(subject, "_read_private_file", mutate_after_first_pass)
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="identity changed across readback passes",
    ):
        subject.readback_private_bundle(bundle, contract=_contract())


def test_readback_rejects_bundle_path_swap_after_complete_validation(
    tmp_path: Path,
) -> None:
    root = _private_root(tmp_path)
    bundle = Path(_publish(root, "bundle-path-swap")["bundle_path"])
    replacement: Path | None = None

    def validate_then_swap(
        values: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        nonlocal replacement
        normalized = _validate_complete(values)
        bundle.rename(root / ".bundle-path-swap.original")
        bundle.mkdir(mode=0o700)
        sentinel = bundle / "replacement-owned"
        sentinel.write_text("preserve", encoding="utf-8")
        sentinel.chmod(0o600)
        replacement = bundle
        return normalized

    contract = subject.PrivateBundleContract(
        root_suffix=ROOT_SUFFIX,
        input_filenames=INPUT_FILENAMES,
        readback_report_filename=REPORT_FILENAME,
        canonicalize=_canonical,
        validate_artifact=_validate_artifact,
        validate_complete=validate_then_swap,
        build_readback_report=_build_report,
    )
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="path identity changed",
    ):
        subject.readback_private_bundle(bundle, contract=contract)

    assert replacement is not None
    assert (replacement / "replacement-owned").read_text(encoding="utf-8") == (
        "preserve"
    )


def test_root_symlink_and_contract_suffix_escape_are_rejected(tmp_path: Path) -> None:
    real_root = _private_root(tmp_path / "real")
    linked_parent = tmp_path / "linked"
    linked_parent.mkdir()
    linked_root = linked_parent.joinpath(*ROOT_SUFFIX)
    linked_root.parent.mkdir(parents=True)
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(subject.FactorGovernancePrivateBundleIOError):
        _publish(linked_root, "symlink-root")

    bad_contract = subject.PrivateBundleContract(
        root_suffix=("reports", "not-factor-private", "private", "lane"),
        input_filenames=INPUT_FILENAMES,
        readback_report_filename=REPORT_FILENAME,
        canonicalize=_canonical,
        validate_artifact=_validate_artifact,
        validate_complete=_validate_complete,
        build_readback_report=_build_report,
    )
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="Factor private reports",
    ):
        subject.publish_private_bundle(
            private_root=real_root,
            run_id="bad-contract",
            artifacts=_artifacts(),
            contract=bad_contract,
            revalidate_inputs=lambda: None,
        )


def test_read_private_canonical_json_is_anchored_stable_and_exact(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "explicit-inputs"
    parent.mkdir()
    value = {"schema_version": "private-input.v1", "value": 7}
    path = parent / "base_catalog.v4.json"
    raw = subject.canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)

    result = subject.read_private_canonical_json(
        path,
        hashlib.sha256(raw).hexdigest(),
        lambda parsed: dict(parsed),
    )
    assert result["value"] == value
    assert result["descriptor"] == {
        "absolute_path": str(path),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "mode": 0o600,
        "uid": os.getuid(),
        "nlink": 1,
    }

    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)
    drift_raw = path.read_bytes()
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="not exact canonical file bytes",
    ):
        subject.read_private_canonical_json(
            path,
            hashlib.sha256(drift_raw).hexdigest(),
            lambda parsed: dict(parsed),
        )


def test_read_private_canonical_json_rejects_symlink_ancestor_and_hardlink(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-inputs"
    real_parent.mkdir()
    value = {"schema_version": "private-input.v1", "value": 9}
    path = real_parent / "base_ontology.v4.json"
    raw = subject.canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)
    linked_parent = tmp_path / "linked-inputs"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(subject.FactorGovernancePrivateBundleIOError):
        subject.read_private_canonical_json(
            linked_parent / path.name,
            hashlib.sha256(raw).hexdigest(),
            lambda parsed: dict(parsed),
        )

    os.link(path, tmp_path / "external-link")
    with pytest.raises(
        subject.FactorGovernancePrivateBundleIOError,
        match="hard-link count must be one",
    ):
        subject.read_private_canonical_json(
            path,
            hashlib.sha256(raw).hexdigest(),
            lambda parsed: dict(parsed),
        )

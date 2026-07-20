from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import threading
from typing import Any

import pytest

from quant_investor.factors import governance_discovery_readback_v4_1 as subject


def _private_root(tmp_path: Path) -> Path:
    root = (
        tmp_path
        / "reports"
        / "factor_governance"
        / "private"
        / "v4_1_cycle"
    )
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


def _artifacts() -> dict[str, dict[str, Any]]:
    return {
        filename: {
            "schema_version": "test-only.v1",
            "filename": filename,
            "cycle_id": "cycle_20260717",
        }
        for filename in subject.INPUT_ARTIFACT_FILENAMES
    }


def _base_ontology() -> dict[str, Any]:
    return {
        "schema_version": "test-only.base-ontology.v1",
        "primitives": [{"primitive_id": "close_return", "family": "return"}],
        "semantic_sha256": "base-ontology",
    }


def _base_catalog() -> dict[str, Any]:
    return {
        "schema_version": "test-only.base-catalog.v1",
        "ontology_sha256": "base-ontology",
        "candidates": [{"name": "base-factor"}],
        "semantic_sha256": "base-catalog",
    }


def _install_fake_core(monkeypatch: pytest.MonkeyPatch) -> None:
    def canonical_file_bytes(value: Any) -> bytes:
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

    def semantic_sha256(value: Any) -> str:
        return hashlib.sha256(canonical_file_bytes(value)[:-1]).hexdigest()

    def validate(filename: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("filename") != filename:
            raise ValueError("fake artifact filename mismatch")
        return dict(value)

    def build_report(
        *,
        cycle_id: str,
        run_id: str,
        artifact_bindings: list[dict[str, Any]],
        side_effects: dict[str, bool],
    ) -> dict[str, Any]:
        assert [item["filename"] for item in artifact_bindings] == sorted(
            item["filename"] for item in artifact_bindings
        )
        return {
            "schema_version": "test-only.readback.v1",
            "filename": subject.DISCOVERY_READBACK_REPORT_FILENAME,
            "cycle_id": cycle_id,
            "run_id": run_id,
            "artifact_bindings": artifact_bindings,
            "side_effects": side_effects,
        }

    def validate_bundle(
        values: dict[str, dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        if set(values) != set(subject.CANONICAL_ARTIFACT_FILENAMES):
            raise ValueError("fake bundle set mismatch")
        cycle_ids = {value.get("cycle_id") for value in values.values()}
        if cycle_ids != {"cycle_20260717"}:
            raise ValueError("fake cross-artifact cycle mismatch")
        base_ontology = kwargs.get("base_ontology")
        base_catalog = kwargs.get("base_catalog")
        if (base_ontology is None) != (base_catalog is None):
            raise ValueError("fake base artifacts must be supplied together")
        if base_ontology is not None:
            if base_ontology != _base_ontology():
                raise ValueError("fake base ontology substitution")
            if base_catalog != _base_catalog():
                raise ValueError("fake base catalog substitution")
        return {filename: dict(value) for filename, value in values.items()}

    rename_lock = threading.Lock()

    def fake_exclusive_rename(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        # Portable test emulation only.  Production has no such fallback.
        with rename_lock:
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

    monkeypatch.setattr(
        subject._core,
        "CANONICAL_ARTIFACT_FILENAMES",
        subject.CANONICAL_ARTIFACT_FILENAMES,
    )
    monkeypatch.setattr(subject._core, "canonical_file_bytes", canonical_file_bytes)
    monkeypatch.setattr(subject._core, "semantic_sha256", semantic_sha256)
    monkeypatch.setattr(
        subject._core,
        "validate_discovery_artifact_v4_1",
        validate,
        raising=False,
    )
    monkeypatch.setattr(
        subject._core,
        "build_discovery_readback_report_v4_1",
        build_report,
        raising=False,
    )
    monkeypatch.setattr(
        subject._core,
        "validate_discovery_bundle_v4_1",
        validate_bundle,
        raising=False,
    )
    monkeypatch.setattr(subject, "_require_exclusive_rename_support", lambda: None)
    monkeypatch.setattr(subject, "_renameatx_np_exclusive", fake_exclusive_rename)
    monkeypatch.setattr(
        subject,
        "_authoritative_semantic_sha256",
        lambda _filename, value: semantic_sha256(value),
    )


@pytest.fixture
def publication_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _install_fake_core(monkeypatch)
    return _private_root(tmp_path)


def _publish(root: Path, run_id: str = "discovery_run") -> dict[str, Any]:
    return subject.publish_discovery_bundle_v4_1(
        private_root=root,
        run_id=run_id,
        artifacts=_artifacts(),
        revalidate_inputs=lambda: None,
    )


def test_publish_and_live_readback_are_exact_and_owner_private(
    publication_environment: Path,
) -> None:
    callback_count = 0

    def revalidate() -> None:
        nonlocal callback_count
        callback_count += 1

    result = subject.publish_discovery_bundle_v4_1(
        private_root=publication_environment,
        run_id="discovery_run",
        artifacts=_artifacts(),
        revalidate_inputs=revalidate,
    )

    assert callback_count == 1
    assert result["accepted"] is True
    assert result["readiness"] == "EXPLORATORY_DISCOVERY"
    assert result["qualification"] is False
    assert result["side_effects"] == subject.FIXED_SIDE_EFFECTS
    bundle = Path(result["bundle_path"])
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o700
    assert sorted(path.name for path in bundle.iterdir()) == sorted(
        subject.CANONICAL_ARTIFACT_FILENAMES
    )
    for artifact in bundle.iterdir():
        metadata = artifact.stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1

    reread = subject.readback_discovery_bundle_v4_1(bundle)
    assert reread["accepted"] is True
    assert reread["readback"] == result["readback"]
    assert set(reread) == {
        "accepted",
        "readiness",
        "qualification",
        "formal_admission_authority",
        "production_apply_enabled",
        "bundle_path",
        "artifact_descriptors",
        "readback",
        "side_effects",
    }


def test_values_readback_returns_stable_deep_copies_and_existing_metadata(
    publication_environment: Path,
) -> None:
    bundle = Path(_publish(publication_environment, "values")["bundle_path"])
    base_ontology = _base_ontology()
    base_catalog = _base_catalog()

    first = subject.readback_discovery_bundle_values_v4_1(
        bundle,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
    )
    ordinary = subject.readback_discovery_bundle_v4_1(bundle)

    assert {key: value for key, value in first.items() if key != "values"} == ordinary
    assert set(first["values"]) == set(subject.CANONICAL_ARTIFACT_FILENAMES)
    first["values"][subject.DISCOVERY_CATALOG_FILENAME]["cycle_id"] = "mutated"
    first["values"][subject.DISCOVERY_CATALOG_FILENAME]["nested"] = {"x": [1]}

    second = subject.readback_discovery_bundle_values_v4_1(
        bundle,
        base_ontology=base_ontology,
        base_catalog=base_catalog,
    )
    assert second["values"][subject.DISCOVERY_CATALOG_FILENAME]["cycle_id"] == (
        "cycle_20260717"
    )
    assert "nested" not in second["values"][subject.DISCOVERY_CATALOG_FILENAME]
    assert base_ontology == _base_ontology()
    assert base_catalog == _base_catalog()


@pytest.mark.parametrize("base_kind", ["ontology", "catalog"])
def test_values_readback_rejects_base_substitution(
    publication_environment: Path,
    base_kind: str,
) -> None:
    bundle = Path(_publish(publication_environment, f"base_{base_kind}")["bundle_path"])
    base_ontology = _base_ontology()
    base_catalog = _base_catalog()
    if base_kind == "ontology":
        base_ontology["semantic_sha256"] = "substituted"
    else:
        base_catalog["semantic_sha256"] = "substituted"

    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match=f"fake base {base_kind} substitution",
    ):
        subject.readback_discovery_bundle_values_v4_1(
            bundle,
            base_ontology=base_ontology,
            base_catalog=base_catalog,
        )


def test_publication_rejects_missing_or_extra_input_artifacts(
    publication_environment: Path,
) -> None:
    missing = _artifacts()
    missing.pop(subject.SOURCE_IDEA_AUDIT_FILENAME)
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="input artifact set mismatch",
    ):
        subject.publish_discovery_bundle_v4_1(
            private_root=publication_environment,
            run_id="missing",
            artifacts=missing,
            revalidate_inputs=lambda: None,
        )

    extra = _artifacts()
    extra["unexpected.json"] = {}
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="input artifact set mismatch",
    ):
        subject.publish_discovery_bundle_v4_1(
            private_root=publication_environment,
            run_id="extra",
            artifacts=extra,
            revalidate_inputs=lambda: None,
        )


def test_cross_artifact_substitution_is_rejected_before_canonical_commit(
    publication_environment: Path,
) -> None:
    artifacts = _artifacts()
    artifacts[subject.DISCOVERY_CATALOG_FILENAME]["cycle_id"] = "other_cycle"
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="cross-artifact discovery bundle validation failed",
    ):
        subject.publish_discovery_bundle_v4_1(
            private_root=publication_environment,
            run_id="cross_substitution",
            artifacts=artifacts,
            revalidate_inputs=lambda: None,
        )
    assert not (publication_environment / "cross_substitution").exists()


def test_readback_rejects_missing_and_extra_canonical_artifacts(
    publication_environment: Path,
) -> None:
    bundle = Path(_publish(publication_environment)["bundle_path"])
    extra = bundle / "unexpected.json"
    extra.write_text("{}\n", encoding="utf-8")
    extra.chmod(0o600)
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="artifact set mismatch",
    ):
        subject.readback_discovery_bundle_v4_1(bundle)

    extra.unlink()
    (bundle / subject.SOURCE_IDEA_AUDIT_FILENAME).unlink()
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="artifact set mismatch",
    ):
        subject.readback_discovery_bundle_v4_1(bundle)


def test_readback_rejects_same_bytes_metadata_drift_between_passes(
    publication_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = Path(_publish(publication_environment, "stable_drift")["bundle_path"])
    real_read = subject._read_private_file
    calls = 0

    def mutate_after_first_pass(
        directory_fd: int,
        filename: str,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal calls
        result = real_read(directory_fd, filename)
        calls += 1
        if calls == len(subject.CANONICAL_ARTIFACT_FILENAMES):
            target = bundle / subject.AQUANT_SOURCE_RECEIPT_FILENAME
            current = target.stat()
            os.utime(
                target,
                ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
            )
        return result

    monkeypatch.setattr(subject, "_read_private_file", mutate_after_first_pass)
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="identity changed across readback passes",
    ):
        subject.readback_discovery_bundle_v4_1(bundle)


def test_preexisting_target_blocks_without_touching_it(
    publication_environment: Path,
) -> None:
    target = publication_environment / "already_there"
    target.mkdir(mode=0o700)
    sentinel = target / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    sentinel.chmod(0o600)

    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="already exists",
    ):
        _publish(publication_environment, "already_there")
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_injected_race_after_final_absent_check_preserves_racer_target(
    publication_environment: Path,
) -> None:
    target = publication_environment / "raced"
    callback_ran = False

    def revalidate() -> None:
        nonlocal callback_ran
        callback_ran = True

    def race() -> None:
        assert callback_ran is True
        target.mkdir(mode=0o700)
        sentinel = target / "racer-owned"
        sentinel.write_text("untouched", encoding="utf-8")
        sentinel.chmod(0o600)

    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="appeared during exclusive commit",
    ):
        subject.publish_discovery_bundle_v4_1(
            private_root=publication_environment,
            run_id="raced",
            artifacts=_artifacts(),
            revalidate_inputs=revalidate,
            race_hook=race,
        )
    assert (target / "racer-owned").read_text(encoding="utf-8") == "untouched"


def test_staging_path_swap_is_rejected_without_quarantining_replacement(
    publication_environment: Path,
) -> None:
    replacement: Path | None = None

    def race() -> None:
        nonlocal replacement
        staging = next(
            path
            for path in publication_environment.iterdir()
            if path.name.startswith(".stage_swap.staging.")
        )
        original = publication_environment / f"{staging.name}.original"
        staging.rename(original)
        staging.mkdir(mode=0o700)
        sentinel = staging / "replacement-owned"
        sentinel.write_text("preserve", encoding="utf-8")
        sentinel.chmod(0o600)
        replacement = staging

    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="staging directory path identity changed",
    ):
        subject.publish_discovery_bundle_v4_1(
            private_root=publication_environment,
            run_id="stage_swap",
            artifacts=_artifacts(),
            revalidate_inputs=lambda: None,
            race_hook=race,
        )
    assert not (publication_environment / "stage_swap").exists()
    assert replacement is not None
    assert (replacement / "replacement-owned").read_text(encoding="utf-8") == (
        "preserve"
    )


def test_two_publishers_yield_exactly_one_accepted_bundle(
    publication_environment: Path,
) -> None:
    results: list[dict[str, Any]] = []
    errors: list[Exception] = []
    result_lock = threading.Lock()

    def worker() -> None:
        try:
            value = _publish(publication_environment, "same_run")
        except Exception as exc:  # noqa: BLE001 - asserted below
            with result_lock:
                errors.append(exc)
        else:
            with result_lock:
                results.append(value)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 1
    assert results[0]["accepted"] is True
    assert len(errors) == 1
    assert isinstance(
        errors[0], subject.FactorGovernanceDiscoveryReadbackV4_1Error
    )
    assert "already exists" in str(errors[0])
    assert subject.readback_discovery_bundle_v4_1(
        publication_environment / "same_run"
    )["accepted"] is True


def test_input_revalidation_failure_leaves_no_canonical_bundle(
    publication_environment: Path,
) -> None:
    def reject() -> None:
        raise ValueError("source changed")

    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="input revalidation failed",
    ):
        subject.publish_discovery_bundle_v4_1(
            private_root=publication_environment,
            run_id="revalidation_failure",
            artifacts=_artifacts(),
            revalidate_inputs=reject,
        )
    assert not (publication_environment / "revalidation_failure").exists()
    assert any(
        "revalidation_failure.staging-failed" in child.name
        for child in (publication_environment / subject.QUARANTINE_DIRECTORY).iterdir()
    )


def test_write_failure_leaves_no_canonical_bundle_and_quarantines_staging(
    publication_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = subject.os.write
    calls = 0

    def fail_first_write(descriptor: int, value: Any) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected write failure")
        return real_write(descriptor, value)

    monkeypatch.setattr(subject.os, "write", fail_first_write)
    with pytest.raises(subject.FactorGovernanceDiscoveryReadbackV4_1Error):
        _publish(publication_environment, "write_failure")
    assert not (publication_environment / "write_failure").exists()
    assert any(
        "write_failure.staging-failed" in child.name
        for child in (publication_environment / subject.QUARANTINE_DIRECTORY).iterdir()
    )


def test_postcommit_readback_failure_moves_bundle_to_quarantine(
    publication_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_readback(_path: Any) -> dict[str, Any]:
        raise ValueError("injected live readback failure")

    monkeypatch.setattr(subject, "readback_discovery_bundle_v4_1", fail_readback)
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="POSTCOMMIT_RECOVERED_FAIL_CLOSED",
    ) as caught:
        _publish(publication_environment, "postcommit_failure")
    assert caught.value.accepted is False
    assert not (publication_environment / "postcommit_failure").exists()
    assert any(
        "postcommit_failure.postcommit" in child.name
        for child in (publication_environment / subject.QUARANTINE_DIRECTORY).iterdir()
    )


def test_postcommit_recovery_failure_is_ambiguous_and_never_accepted(
    publication_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fsync = subject.os.fsync
    canonical_seen = False

    def fail_root_fsync_after_commit(descriptor: int) -> None:
        nonlocal canonical_seen
        if (publication_environment / "ambiguous").exists():
            canonical_seen = True
        opened = os.fstat(descriptor)
        root_value = publication_environment.stat()
        if (
            canonical_seen
            and opened.st_dev == root_value.st_dev
            and opened.st_ino == root_value.st_ino
        ):
            raise OSError("injected parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(subject.os, "fsync", fail_root_fsync_after_commit)
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="AMBIGUOUS_DURABILITY_FAIL_CLOSED",
    ) as caught:
        _publish(publication_environment, "ambiguous")
    assert caught.value.accepted is False


def test_postcommit_identity_swap_does_not_quarantine_unrelated_replacement(
    publication_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rename = subject._renameatx_np_exclusive
    injected = False

    def replace_after_commit(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal injected
        original_rename(
            source_directory_fd,
            source_name,
            destination_directory_fd,
            destination_name,
        )
        if not injected and destination_name == "identity_swap":
            injected = True
            os.rename(
                destination_name,
                ".identity_swap.original",
                src_dir_fd=destination_directory_fd,
                dst_dir_fd=destination_directory_fd,
            )
            os.mkdir(destination_name, 0o700, dir_fd=destination_directory_fd)
            replacement_fd = os.open(
                destination_name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                dir_fd=destination_directory_fd,
            )
            try:
                sentinel_fd = os.open(
                    "replacement-owned",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=replacement_fd,
                )
                try:
                    os.write(sentinel_fd, b"preserve")
                finally:
                    os.close(sentinel_fd)
            finally:
                os.close(replacement_fd)

    monkeypatch.setattr(subject, "_renameatx_np_exclusive", replace_after_commit)
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="AMBIGUOUS_DURABILITY_FAIL_CLOSED",
    ) as caught:
        _publish(publication_environment, "identity_swap")
    assert caught.value.accepted is False
    replacement = publication_environment / "identity_swap"
    assert (replacement / "replacement-owned").read_bytes() == b"preserve"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "mode"])
def test_unsafe_existing_lock_is_rejected(
    publication_environment: Path,
    tmp_path: Path,
    kind: str,
) -> None:
    lock = publication_environment / subject.LOCK_FILENAME
    source = tmp_path / f"lock-source-{kind}"
    source.write_bytes(b"")
    source.chmod(0o600)
    if kind == "symlink":
        lock.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, lock)
    else:
        lock.write_bytes(b"")
        lock.chmod(0o644)

    with pytest.raises(subject.FactorGovernanceDiscoveryReadbackV4_1Error):
        _publish(publication_environment, f"unsafe_lock_{kind}")
    assert not (publication_environment / f"unsafe_lock_{kind}").exists()


def test_private_root_must_be_real_owner_mode_0700(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_core(monkeypatch)
    root = _private_root(tmp_path)
    root.chmod(0o755)
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="mode must be 0700",
    ):
        _publish(root, "bad_root_mode")

    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    root.rmdir()
    root.symlink_to(actual, target_is_directory=True)
    with pytest.raises(subject.FactorGovernanceDiscoveryReadbackV4_1Error):
        _publish(root, "root_symlink")


def test_unsafe_quarantine_is_rejected_without_canonical_acceptance(
    publication_environment: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "quarantine-target"
    target.mkdir(mode=0o700)
    (publication_environment / subject.QUARANTINE_DIRECTORY).symlink_to(
        target,
        target_is_directory=True,
    )

    def reject() -> None:
        raise ValueError("force staging recovery")

    with pytest.raises(subject.FactorGovernanceDiscoveryReadbackV4_1Error):
        subject.publish_discovery_bundle_v4_1(
            private_root=publication_environment,
            run_id="unsafe_quarantine",
            artifacts=_artifacts(),
            revalidate_inputs=reject,
        )
    assert not (publication_environment / "unsafe_quarantine").exists()


@pytest.mark.parametrize("drift", ["mode", "hardlink", "symlink"])
def test_readback_rejects_artifact_security_drift(
    publication_environment: Path,
    tmp_path: Path,
    drift: str,
) -> None:
    bundle = Path(_publish(publication_environment, f"drift_{drift}")["bundle_path"])
    artifact = bundle / subject.SOURCE_IDEA_AUDIT_FILENAME
    if drift == "mode":
        artifact.chmod(0o644)
    elif drift == "hardlink":
        os.link(artifact, tmp_path / "second-link")
    else:
        artifact.unlink()
        replacement = tmp_path / "replacement"
        replacement.write_text("{}\n", encoding="utf-8")
        replacement.chmod(0o600)
        artifact.symlink_to(replacement)

    with pytest.raises(subject.FactorGovernanceDiscoveryReadbackV4_1Error):
        subject.readback_discovery_bundle_v4_1(bundle)


def test_non_darwin_rejects_before_creating_lock_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    monkeypatch.setattr(subject.sys, "platform", "linux")
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="requires Darwin",
    ):
        subject.publish_discovery_bundle_v4_1(
            private_root=root,
            run_id="linux_rejected",
            artifacts=_artifacts(),
            revalidate_inputs=lambda: None,
        )
    assert list(root.iterdir()) == []


def test_binding_uses_and_independently_checks_authoritative_self_hash() -> None:
    semantic_payload = {
        "schema_version": "test-only.v1",
        "cycle_id": "cycle_20260717",
    }
    authoritative = subject._core.semantic_sha256(semantic_payload)
    value = {**semantic_payload, "receipt_semantic_sha256": authoritative}
    assert subject._authoritative_semantic_sha256(
        subject.AQUANT_SOURCE_RECEIPT_FILENAME,
        value,
    ) == authoritative
    assert authoritative != subject._core.semantic_sha256(value)

    forged = {**value, "cycle_id": "substituted"}
    with pytest.raises(
        subject.FactorGovernanceDiscoveryReadbackV4_1Error,
        match="does not seal",
    ):
        subject._authoritative_semantic_sha256(
            subject.AQUANT_SOURCE_RECEIPT_FILENAME,
            forged,
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin syscall contract")
def test_real_renameatx_np_exclusive_never_clobbers(tmp_path: Path) -> None:
    parent = tmp_path / "renameatx"
    parent.mkdir(mode=0o700)
    (parent / "source").mkdir(mode=0o700)
    (parent / "destination").mkdir(mode=0o700)
    sentinel = parent / "destination" / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    descriptor = os.open(
        parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        subject._require_exclusive_rename_support()
        with pytest.raises(FileExistsError):
            subject._renameatx_np_exclusive(
                descriptor,
                "source",
                descriptor,
                "destination",
            )
        assert (parent / "source").is_dir()
        assert sentinel.read_text(encoding="utf-8") == "preserve"

        subject._renameatx_np_exclusive(
            descriptor,
            "source",
            descriptor,
            "committed",
        )
        assert not (parent / "source").exists()
        assert (parent / "committed").is_dir()
    finally:
        os.close(descriptor)

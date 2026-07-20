from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import stat
import threading
from typing import Any

import pytest

from quant_investor.factors import (
    governance_prior_diagnostic_nomination_bundle_v4_3 as subject,
)
from quant_investor.factors import (
    governance_prior_diagnostic_nomination_v4_3 as diagnostic,
)
from quant_investor.factors import governance_private_bundle_io as private_io
from tests.unit.test_factor_governance_prior_diagnostic_nomination_v4_3 import (
    _nomination,
    _runtime_binding,
)


_portable_rename_lock = threading.Lock()


@lru_cache(maxsize=1)
def _cached_bundle_artifacts() -> dict[str, dict[str, Any]]:
    runtime = _runtime_binding()
    nomination = _nomination(runtime["artifact_semantic_sha256"])
    return subject.validate_prior_diagnostic_nomination_bundle_inputs_v4_3(
        {
            subject.PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3: runtime,
            subject.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3: nomination,
        }
    )


def _bundle_artifacts(_tmp_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return a fresh, real-core-validated two-input graph for integrations."""

    return copy.deepcopy(_cached_bundle_artifacts())


def _portable_private_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a locked test-only RENAME_EXCL emulation off Darwin."""

    def rename_exclusive(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
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

    monkeypatch.setattr(private_io, "_require_exclusive_rename_support", lambda: None)
    monkeypatch.setattr(private_io, "_renameatx_np_exclusive", rename_exclusive)


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path.joinpath(
        *subject.ROOT_SUFFIX_V4_3_PRIOR_DIAGNOSTIC_NOMINATION
    )
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


def _file_hashes(bundle_path: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle_path.iterdir()
    }


def _reseal_report(report: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(report)
    value.pop("artifact_semantic_sha256", None)
    value["artifact_semantic_sha256"] = diagnostic.semantic_sha256_v4_3(value)
    return value


def test_contract_is_exact_ordered_two_input_one_report_and_pure() -> None:
    contract = subject.prior_diagnostic_nomination_bundle_contract_v4_3()
    assert contract.root_suffix == (
        "reports",
        "factor_governance",
        "private",
        "v4_3_prior_diagnostic_nomination",
    )
    assert contract.input_filenames == (
        "prior_diagnostic_runtime_binding.v4_3.json",
        "prior_diagnostic_nomination.v4_3.json",
    )
    assert contract.readback_report_filename == (
        "prior_diagnostic_nomination_readback.v4_3.json"
    )
    assert contract.canonical_filenames == (
        "prior_diagnostic_runtime_binding.v4_3.json",
        "prior_diagnostic_nomination.v4_3.json",
        "prior_diagnostic_nomination_readback.v4_3.json",
    )

    reversed_inputs = dict(reversed(tuple(_bundle_artifacts().items())))
    normalized = subject.validate_prior_diagnostic_nomination_bundle_inputs_v4_3(
        reversed_inputs
    )
    assert tuple(normalized) == subject.INPUT_FILENAMES_V4_3
    assert normalized[
        subject.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3
    ]["runtime_binding_semantic_sha256"] == normalized[
        subject.PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3
    ]["artifact_semantic_sha256"]

    missing = _bundle_artifacts()
    missing.pop(subject.PRIOR_DIAGNOSTIC_RUNTIME_BINDING_FILENAME_V4_3)
    with pytest.raises(
        subject.FactorGovernancePriorDiagnosticNominationBundleV4_3Error,
        match="input inventory mismatch",
    ):
        subject.validate_prior_diagnostic_nomination_bundle_inputs_v4_3(missing)

    extra = _bundle_artifacts()
    extra["unexpected.v4_3.json"] = {}
    with pytest.raises(
        subject.FactorGovernancePriorDiagnosticNominationBundleV4_3Error,
        match="input inventory mismatch",
    ):
        subject.validate_prior_diagnostic_nomination_bundle_inputs_v4_3(extra)


def test_publish_and_explicit_historical_readback_are_exact_owner_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _portable_private_publication(monkeypatch)
    root = _private_root(tmp_path)
    artifacts = _bundle_artifacts()
    revalidation_count = 0

    def revalidate() -> None:
        nonlocal revalidation_count
        revalidation_count += 1
        assert (
            subject.validate_prior_diagnostic_nomination_bundle_inputs_v4_3(
                artifacts
            )
            == artifacts
        )

    result = subject.publish_prior_diagnostic_nomination_bundle_v4_3(
        private_root=root,
        artifacts=artifacts,
        revalidate_inputs=revalidate,
    )

    assert revalidation_count == 1
    assert result["accepted"] is True
    assert isinstance(result["artifact_descriptors"], dict)
    assert tuple(result["artifacts"]) == (
        *subject.INPUT_FILENAMES_V4_3,
        subject.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3,
    )
    bundle_path = Path(result["bundle_path"])
    assert bundle_path == root / diagnostic.RUN_ID
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(bundle_path.stat().st_mode) == 0o700
    assert set(result["artifact_descriptors"]) == {
        *subject.INPUT_FILENAMES_V4_3,
        subject.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3,
    }
    for filename, descriptor in result["artifact_descriptors"].items():
        path = bundle_path / filename
        metadata = path.stat()
        assert descriptor["absolute_path"] == str(path)
        assert descriptor["byte_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert descriptor["size_bytes"] == metadata.st_size
        assert descriptor["mode"] == stat.S_IMODE(metadata.st_mode) == 0o600
        assert descriptor["uid"] == metadata.st_uid == os.getuid()
        assert descriptor["nlink"] == metadata.st_nlink == 1

    report = result["readback_report"]
    assert report == result["artifacts"][
        subject.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3
    ]
    assert [row["filename"] for row in report["artifact_bindings"]] == list(
        subject.INPUT_FILENAMES_V4_3
    )
    for row in report["artifact_bindings"]:
        filename = row["filename"]
        assert row["byte_sha256"] == result["artifact_descriptors"][filename][
            "byte_sha256"
        ]
        assert row["semantic_sha256"] == artifacts[filename][
            "artifact_semantic_sha256"
        ]
    assert all(value is False for value in report["authority"].values())
    assert all(value is False for value in report["side_effects"].values())
    assert report["commit_success_claimed"] is False
    assert report["no_clobber_success_claimed"] is False
    assert report["fsync_success_claimed"] is False
    assert report["durability_success_claimed"] is False

    historical = subject.readback_prior_diagnostic_nomination_bundle_v4_3(
        bundle_path
    )
    assert historical == result


def test_complete_validation_rejects_report_tamper_and_crosslink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _portable_private_publication(monkeypatch)
    root = _private_root(tmp_path)
    published = subject.publish_prior_diagnostic_nomination_bundle_v4_3(
        private_root=root,
        artifacts=_bundle_artifacts(),
        revalidate_inputs=lambda: None,
    )
    complete = copy.deepcopy(published["artifacts"])
    report_name = subject.PRIOR_DIAGNOSTIC_NOMINATION_READBACK_FILENAME_V4_3

    self_tamper = copy.deepcopy(complete)
    self_tamper[report_name]["artifact_semantic_sha256"] = "0" * 64
    with pytest.raises(
        subject.FactorGovernancePriorDiagnosticNominationBundleV4_3Error,
        match="artifact_semantic_sha256 mismatch",
    ):
        subject.validate_prior_diagnostic_nomination_bundle_artifacts_v4_3(
            self_tamper
        )

    binding_tamper = copy.deepcopy(complete)
    binding_tamper[report_name]["artifact_bindings"][0]["size_bytes"] += 1
    binding_tamper[report_name] = _reseal_report(binding_tamper[report_name])
    with pytest.raises(
        subject.FactorGovernancePriorDiagnosticNominationBundleV4_3Error,
        match="byte binding mismatch",
    ):
        subject.validate_prior_diagnostic_nomination_bundle_artifacts_v4_3(
            binding_tamper
        )

    run_crosslink = copy.deepcopy(complete)
    run_crosslink[report_name]["run_id"] = "different-safe-run"
    run_crosslink[report_name]["intended_destination"][
        "directory_name"
    ] = "different-safe-run"
    run_crosslink[report_name] = _reseal_report(run_crosslink[report_name])
    with pytest.raises(
        subject.FactorGovernancePriorDiagnosticNominationBundleV4_3Error,
        match="run_id crosslink mismatch",
    ):
        subject.validate_prior_diagnostic_nomination_bundle_artifacts_v4_3(
            run_crosslink
        )


def test_duplicate_publish_is_no_clobber_and_preserves_first_three_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _portable_private_publication(monkeypatch)
    root = _private_root(tmp_path)
    artifacts = _bundle_artifacts()
    first = subject.publish_prior_diagnostic_nomination_bundle_v4_3(
        private_root=root,
        artifacts=artifacts,
        revalidate_inputs=lambda: None,
    )
    bundle_path = Path(first["bundle_path"])
    before = _file_hashes(bundle_path)
    assert len(before) == 3

    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="already exists",
    ):
        subject.publish_prior_diagnostic_nomination_bundle_v4_3(
            private_root=root,
            artifacts=artifacts,
            revalidate_inputs=lambda: None,
        )

    assert _file_hashes(bundle_path) == before
    assert sorted(path.name for path in bundle_path.iterdir()) == sorted(before)


def test_concurrent_publishers_yield_exactly_one_complete_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _portable_private_publication(monkeypatch)
    root = _private_root(tmp_path)
    successes: list[dict[str, Any]] = []
    failures: list[Exception] = []
    result_lock = threading.Lock()

    def worker() -> None:
        try:
            result = subject.publish_prior_diagnostic_nomination_bundle_v4_3(
                private_root=root,
                artifacts=_bundle_artifacts(),
                revalidate_inputs=lambda: None,
            )
        except Exception as exc:  # noqa: BLE001 - exact type asserted below
            with result_lock:
                failures.append(exc)
        else:
            with result_lock:
                successes.append(result)

    threads = [threading.Thread(target=worker) for _index in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1
    assert len(failures) == 2
    assert all(
        isinstance(error, private_io.FactorGovernancePrivateBundleIOError)
        and "already exists" in str(error)
        for error in failures
    )
    assert subject.readback_prior_diagnostic_nomination_bundle_v4_3(
        root / diagnostic.RUN_ID
    )["accepted"] is True


def test_fault_and_destination_race_fail_closed_without_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _portable_private_publication(monkeypatch)
    fault_root = _private_root(tmp_path / "fault")

    def fault(point: str) -> None:
        if point == "commit:rename:before":
            raise OSError("injected precommit fault")

    with pytest.raises(private_io.FactorGovernancePrivateBundleIOError):
        subject.publish_prior_diagnostic_nomination_bundle_v4_3(
            private_root=fault_root,
            artifacts=_bundle_artifacts(),
            revalidate_inputs=lambda: None,
            _test_fault_hook=fault,
        )
    assert not (fault_root / diagnostic.RUN_ID).exists()
    quarantine = fault_root / private_io.QUARANTINE_DIRECTORY
    assert quarantine.is_dir()
    assert any(path.name.startswith(diagnostic.RUN_ID) for path in quarantine.iterdir())

    race_root = _private_root(tmp_path / "race")
    destination = race_root / diagnostic.RUN_ID

    def race() -> None:
        destination.mkdir(mode=0o700)
        sentinel = destination / "sentinel"
        sentinel.write_bytes(b"preserve")
        sentinel.chmod(0o600)

    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="appeared during exclusive commit",
    ):
        subject.publish_prior_diagnostic_nomination_bundle_v4_3(
            private_root=race_root,
            artifacts=_bundle_artifacts(),
            revalidate_inputs=lambda: None,
            _test_race_hook=race,
        )
    assert (destination / "sentinel").read_bytes() == b"preserve"


def test_wrong_root_suffix_and_nomination_runtime_substitution_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _portable_private_publication(monkeypatch)
    wrong_root = tmp_path / "reports" / "factor_governance" / "private" / "wrong"
    wrong_root.mkdir(parents=True)
    wrong_root.chmod(0o700)
    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="private suffix",
    ):
        subject.publish_prior_diagnostic_nomination_bundle_v4_3(
            private_root=wrong_root,
            artifacts=_bundle_artifacts(),
            revalidate_inputs=lambda: None,
        )

    substituted = _bundle_artifacts()
    nomination_name = subject.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3
    substituted[nomination_name]["runtime_binding_semantic_sha256"] = "f" * 64
    value = substituted[nomination_name]
    value.pop("artifact_semantic_sha256")
    value["artifact_semantic_sha256"] = diagnostic.semantic_sha256_v4_3(value)
    with pytest.raises(
        diagnostic.FactorGovernancePriorDiagnosticNominationV4_3Error,
        match="crosslink mismatch",
    ):
        subject.validate_prior_diagnostic_nomination_bundle_inputs_v4_3(substituted)

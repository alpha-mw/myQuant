from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import build_factor_v4_2_candidate_preregistration as subject


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path.joinpath(*subject.bundle_v4_2.ROOT_SUFFIX_V4_2)
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


def _stable(path: Path, content: bytes = b"stable\n") -> subject.StableFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return subject._stable_file(path, label=path.name)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        snapshot_id="20260718T172132Z",
        analysis_start="2026-07-01",
        cutoff="2026-07-18",
    )


def _entry(control: subject.StableFile) -> subject.PublicationInputs:
    return subject.PublicationInputs(
        cycle_id="cn_full_a_v4_2_20260718_20260718T172132Z",
        artifacts={"synthetic.v4_2.json": {"protocol_version": "v4"}},
        protected_controls={"registry": control},
        source_binding_semantic_sha256="1" * 64,
        code_binding_set_semantic_sha256="2" * 64,
    )


def _readback_result(root: Path) -> dict[str, Any]:
    report_name = subject.bundle_v4_2.READBACK_REPORT_FILENAME_V4_2
    bundle_path = root / "cn_full_a_v4_2_20260718_20260718T172132Z"
    names = (*subject.bundle_v4_2.INPUT_FILENAMES_V4_2, report_name)
    return {
        "accepted": True,
        "bundle_path": str(bundle_path),
        "artifact_descriptors": {
            name: {
                "absolute_path": str(bundle_path / name),
                "byte_sha256": "3" * 64 if name == report_name else "5" * 64,
                "size_bytes": 1,
                "mode": 0o600,
                "uid": os.getuid(),
                "nlink": 1,
            }
            for name in names
        },
        "readback_report": {"artifact_semantic_sha256": "4" * 64},
        "artifacts": {
            subject.bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2: {
                "synthetic": True
            }
        },
    }


def test_public_cli_has_no_root_identity_or_side_effect_override() -> None:
    parser = subject.build_parser()
    help_text = parser.format_help()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    help_text += "".join(item.format_help() for item in subparsers.choices.values())
    for token in subject._FORBIDDEN_ARGUMENT_TOKENS:
        assert token not in help_text
    assert str(subject.PRODUCTION_PRIVATE_ROOT) == (
        "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
        "v4_2_candidate_preregistration"
    )


def test_default_protected_controls_are_the_exact_fixed_five() -> None:
    assert subject.PROTECTED_CONTROL_PATHS == (
        (
            "registry",
            subject.PROJECT_ROOT
            / "quant_investor"
            / "factor_registry"
            / "mined_factors.json",
        ),
        (
            "latest_pointer",
            subject.PROJECT_ROOT / "data" / "parquet" / "cn" / "_latest.json",
        ),
        (
            "catalog",
            subject.PROJECT_ROOT / "data" / "parquet" / "cn" / "_catalog.json",
        ),
        (
            "fundamental_latest",
            subject.PROJECT_ROOT
            / "data"
            / "parquet"
            / "cn"
            / "_fundamental_latest.json",
        ),
        (
            "latest_manifest",
            subject.PROJECT_ROOT
            / "data"
            / "parquet"
            / "cn"
            / "latest_manifest.json",
        ),
    )
    assert len({path for _name, path in subject.PROTECTED_CONTROL_PATHS}) == 5
    assert all(path.is_absolute() for _name, path in subject.PROTECTED_CONTROL_PATHS)


@pytest.mark.parametrize(
    ("snapshot_id", "analysis_start", "cutoff", "match"),
    [
        ("20260717T235959Z", "2026-07-01", "2026-07-17", "later"),
        ("20269999T000000Z", "2026-07-01", "2026-99-99", "invalid"),
        ("20260717T235959Z", "2026-07-01", "2026-07-18", "must not be before"),
        ("20260718T256199Z", "2026-07-01", "2026-07-18", "invalid"),
        ("20260718T000000Z", "2026-07-19", "2026-07-18", "must not"),
    ],
)
def test_cycle_identity_rejects_stale_impossible_or_inconsistent_dates(
    snapshot_id: str,
    analysis_start: str,
    cutoff: str,
    match: str,
) -> None:
    with pytest.raises(subject.FactorV4_2CandidatePreregistrationRunnerError, match=match):
        subject._validate_cycle_identity(
            snapshot_id=snapshot_id,
            analysis_start=analysis_start,
            cutoff=cutoff,
        )


def test_cycle_identity_is_deterministic() -> None:
    assert subject._validate_cycle_identity(
        snapshot_id="20260718T172132Z",
        analysis_start="2021-06-25",
        cutoff="2026-07-18",
    ) == "cn_full_a_v4_2_20260718_20260718T172132Z"
    assert subject._validate_cycle_identity(
        snapshot_id="20260719T001500Z",
        analysis_start="2021-06-25",
        cutoff="2026-07-18",
    ) == "cn_full_a_v4_2_20260718_20260719T001500Z"


def test_stale_cutoff_is_rejected_before_platform_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        snapshot_id="20260717T172132Z",
        analysis_start="2021-06-25",
        cutoff="2026-07-17",
    )
    root = tmp_path.joinpath(*subject.bundle_v4_2.ROOT_SUFFIX_V4_2)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("stale cutoff must reject before any publication boundary")

    monkeypatch.setattr(subject, "_collect_publication_inputs", forbidden)
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "publish_candidate_preregistration_bundle_v4_2",
        forbidden,
    )
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "readback_candidate_preregistration_bundle_files_v4_2",
        forbidden,
    )

    with pytest.raises(subject.FactorV4_2CandidatePreregistrationRunnerError, match="later"):
        subject.run_publish(
            args,
            private_root=root,
            exclusive_rename_probe=forbidden,
        )
    assert not root.exists()


def test_private_root_rejects_wrong_mode_and_existing_cycle(tmp_path: Path) -> None:
    root = _private_root(tmp_path)
    cycle_id = "cn_full_a_v4_2_20260718_20260718T172132Z"
    root.chmod(0o755)
    with pytest.raises(subject.FactorV4_2CandidatePreregistrationRunnerError, match="0700"):
        subject._validate_private_root_preflight(root, cycle_id=cycle_id)
    root.chmod(0o700)
    (root / cycle_id).mkdir()
    with pytest.raises(subject.FactorV4_2CandidatePreregistrationRunnerError, match="already exists"):
        subject._validate_private_root_preflight(root, cycle_id=cycle_id)


def test_code_expectation_inventory_is_exact_and_drift_fails(tmp_path: Path) -> None:
    expected: list[str] = []
    snapshots: dict[str, subject.StableFile] = {}
    for index, relative in enumerate(subject.bundle_v4_2.CODE_BINDING_PATHS_V4_2):
        path = tmp_path / relative
        observed = _stable(path, f"code-{index}\n".encode())
        expected.append(f"{relative}={observed.byte_sha256}")
        snapshots[relative] = observed
    parsed = subject._parse_code_expectations(expected)
    assert tuple(parsed) == subject.bundle_v4_2.CODE_BINDING_PATHS_V4_2
    (tmp_path / subject.bundle_v4_2.CODE_BINDING_PATHS_V4_2[0]).write_bytes(
        b"changed\n"
    )
    with pytest.raises(subject.FactorV4_2CandidatePreregistrationRunnerError):
        subject._assert_snapshots_unchanged(snapshots, label="code binding")


def test_protected_control_precommit_drift_fails(tmp_path: Path) -> None:
    paths = tuple(
        (name, tmp_path / f"{name}.json")
        for name in ("registry", "latest", "catalog", "fundamental", "manifest")
    )
    for _name, path in paths:
        _stable(path)
    snapshots = subject._snapshot_protected_controls(paths)
    paths[2][1].write_bytes(b"drift!\n")
    with pytest.raises(subject.FactorV4_2CandidatePreregistrationRunnerError):
        subject._assert_snapshots_unchanged(snapshots, label="protected control")


def test_postcommit_control_drift_is_diagnostic_only(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    before = _stable(path)
    path.write_bytes(b"changed after commit\n")
    result = subject._postcommit_control_diagnostics({"registry": before})
    assert result["status"] == "DRIFT_DIAGNOSTIC_ONLY"
    assert result["rows"][0]["unchanged"] is False


def test_locked_source_artifact_drift_fails_before_fake_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    control = _stable(tmp_path / "controls" / "registry.json")
    entry = _entry(control)
    changed = subject.PublicationInputs(
        **{
            **entry.__dict__,
            "source_binding_semantic_sha256": "9" * 64,
        }
    )
    values = iter((entry, changed))
    monkeypatch.setattr(subject, "_collect_publication_inputs", lambda *a, **k: next(values))

    def fake_publish(**kwargs: Any) -> dict[str, Any]:
        kwargs["revalidate_inputs"]()
        raise AssertionError("commit must not be reached")

    monkeypatch.setattr(
        subject.bundle_v4_2,
        "publish_candidate_preregistration_bundle_v4_2",
        fake_publish,
    )
    with pytest.raises(subject.FactorV4_2CandidatePreregistrationRunnerError, match="changed"):
        subject.run_publish(
            _args(),
            private_root=root,
            exclusive_rename_probe=lambda: None,
        )


def test_successful_commit_survives_postcommit_control_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    control_path = tmp_path / "controls" / "registry.json"
    entry = _entry(_stable(control_path))
    monkeypatch.setattr(subject, "_collect_publication_inputs", lambda *a, **k: entry)

    def fake_publish(**kwargs: Any) -> dict[str, Any]:
        kwargs["revalidate_inputs"]()
        control_path.write_bytes(b"postcommit drift\n")
        return {"accepted": True, "bundle_path": _readback_result(root)["bundle_path"]}

    monkeypatch.setattr(
        subject.bundle_v4_2,
        "publish_candidate_preregistration_bundle_v4_2",
        fake_publish,
    )
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "readback_candidate_preregistration_bundle_files_v4_2",
        lambda _path: _readback_result(root),
    )
    result = subject.run_publish(
        _args(),
        private_root=root,
        exclusive_rename_probe=lambda: None,
    )
    assert result["accepted"] is True
    assert result["publisher_return_accepted"] is True
    assert result["independent_reopen_accepted"] is True
    assert result["protected_controls"]["status"] == "DRIFT_DIAGNOSTIC_ONLY"
    assert result["immutable_source_postcommit"]["status"] == (
        "DRIFT_DIAGNOSTIC_ONLY"
    )


def test_run_publish_and_readback_use_actual_v42_bundle_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.test_factor_governance_candidate_preregistration_bundle_v4_2 import (
        _bundle_artifacts,
        _portable_private_publication,
    )

    _portable_private_publication(monkeypatch)
    artifacts = _bundle_artifacts(tmp_path / "graph")
    cycle_root = artifacts[subject.bundle_v4_2.CYCLE_ROOT_FILENAME_V4_2]
    strict_source = artifacts[
        subject.bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
    ]
    code_binding = artifacts[subject.bundle_v4_2.CODE_BINDING_SET_FILENAME_V4_2]
    controls = {
        name: _stable(
            tmp_path / "controls" / f"{name}.json",
            f"{name}-stable\n".encode(),
        )
        for name in (
            "registry",
            "latest_pointer",
            "catalog",
            "fundamental_latest",
            "latest_manifest",
        )
    }
    assert len(controls) == 5
    entry = subject.PublicationInputs(
        cycle_id=cycle_root["cycle_id"],
        artifacts=artifacts,
        protected_controls=controls,
        source_binding_semantic_sha256=strict_source["artifact_semantic_sha256"],
        code_binding_set_semantic_sha256=code_binding["artifact_semantic_sha256"],
    )
    collection_count = 0

    def collect(*_args: Any, **_kwargs: Any) -> subject.PublicationInputs:
        nonlocal collection_count
        collection_count += 1
        return entry

    real_publish = subject.bundle_v4_2.publish_candidate_preregistration_bundle_v4_2
    real_readback = (
        subject.bundle_v4_2.readback_candidate_preregistration_bundle_files_v4_2
    )
    publish_count = 0
    readback_count = 0

    def counted_publish(**kwargs: Any) -> dict[str, Any]:
        nonlocal publish_count
        publish_count += 1
        return real_publish(**kwargs)

    def counted_readback(path: str | os.PathLike[str]) -> dict[str, Any]:
        nonlocal readback_count
        readback_count += 1
        return real_readback(path)

    monkeypatch.setattr(subject, "_collect_publication_inputs", collect)
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "publish_candidate_preregistration_bundle_v4_2",
        counted_publish,
    )
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "readback_candidate_preregistration_bundle_files_v4_2",
        counted_readback,
    )
    root = _private_root(tmp_path / "runner")
    args = argparse.Namespace(
        snapshot_id="20260720T010203Z",
        analysis_start="2026-07-17",
        cutoff="2026-07-18",
    )
    summary = subject.run_publish(
        args,
        private_root=root,
        exclusive_rename_probe=lambda: None,
    )
    assert collection_count == 2
    assert publish_count == 1
    assert readback_count == 1
    assert summary["accepted"] is True
    assert summary["authority"] == subject.prereg_v4_2.AUTHORITY_FLAGS
    assert summary["side_effects"] == subject.prereg_v4_2.SIDE_EFFECT_FLAGS

    bundle = Path(summary["bundle_path"])
    actual = counted_readback(bundle)
    descriptors = actual["artifact_descriptors"]
    assert isinstance(descriptors, dict)
    assert len(descriptors) == 15
    bundle_files = tuple(bundle.iterdir())
    assert len(bundle_files) == 15
    assert {path.name for path in bundle_files} == set(descriptors)
    report_name = subject.bundle_v4_2.READBACK_REPORT_FILENAME_V4_2
    report_descriptor = descriptors[report_name]
    assert summary["readback_report_path"] == report_descriptor["absolute_path"]
    assert summary["readback_report_byte_sha256"] == report_descriptor["byte_sha256"]
    assert (
        summary["readback_report_semantic_sha256"]
        == actual["readback_report"]["artifact_semantic_sha256"]
    )
    assert root.stat().st_mode & 0o777 == 0o700
    assert bundle.stat().st_mode & 0o777 == 0o700
    for path in bundle_files:
        metadata = path.stat()
        assert path.is_file()
        assert metadata.st_uid == os.getuid()
        assert metadata.st_mode & 0o777 == 0o600
        assert metadata.st_nlink == 1

    historical = subject.run_readback(
        argparse.Namespace(
            bundle_path=str(bundle),
            expected_readback_report_byte_sha256=summary[
                "readback_report_byte_sha256"
            ],
            expected_readback_report_semantic_sha256=summary[
                "readback_report_semantic_sha256"
            ],
        )
    )
    assert historical["accepted"] is True
    assert historical["authority"] == subject.prereg_v4_2.AUTHORITY_FLAGS
    assert historical["side_effects"] == subject.prereg_v4_2.SIDE_EFFECT_FLAGS

    before_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.iterdir()
    }
    before_root_inventory = sorted(path.name for path in root.iterdir())
    calls_before_retry = (collection_count, publish_count, readback_count)
    with pytest.raises(
        subject.FactorV4_2CandidatePreregistrationRunnerError,
        match="already exists",
    ):
        subject.run_publish(
            args,
            private_root=root,
            exclusive_rename_probe=lambda: None,
        )
    assert (collection_count, publish_count, readback_count) == calls_before_retry
    assert sorted(path.name for path in root.iterdir()) == before_root_inventory
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.iterdir()
    } == before_hashes


def test_descriptor_validator_rejects_legacy_list_shape(tmp_path: Path) -> None:
    bundle_path = (
        _private_root(tmp_path)
        / "cn_full_a_v4_2_20260718_20260718T172132Z"
    )
    with pytest.raises(
        subject.FactorV4_2CandidatePreregistrationRunnerError,
        match="filename-keyed mapping",
    ):
        subject._validated_artifact_descriptors(
            [{"filename": "legacy-row-shape"}],
            bundle_path=bundle_path,
        )


def test_same_cycle_concurrent_fake_commit_has_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    entry = _entry(_stable(tmp_path / "controls" / "registry.json"))
    monkeypatch.setattr(subject, "_collect_publication_inputs", lambda *a, **k: entry)
    from threading import Barrier

    barrier = Barrier(2)

    def fake_publish(**kwargs: Any) -> dict[str, Any]:
        kwargs["revalidate_inputs"]()
        barrier.wait(timeout=5)
        destination = root / entry.cycle_id
        os.mkdir(destination, 0o700)
        return {"accepted": True, "bundle_path": str(destination)}

    monkeypatch.setattr(
        subject.bundle_v4_2,
        "publish_candidate_preregistration_bundle_v4_2",
        fake_publish,
    )
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "readback_candidate_preregistration_bundle_files_v4_2",
        lambda _path: _readback_result(root),
    )

    def attempt() -> bool:
        try:
            return bool(
                subject.run_publish(
                    _args(),
                    private_root=root,
                    exclusive_rename_probe=lambda: None,
                )["accepted"]
            )
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _value: attempt(), range(2)))
    assert sorted(outcomes) == [False, True]


def test_readback_mode_binds_report_hashes_and_calls_immutable_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    bundle_path = root / "cn_full_a_v4_2_20260718_20260718T172132Z"
    result = _readback_result(root)
    result["artifacts"] = {
        subject.bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2: {
            "immutable_reopen_descriptor": {}
        }
    }
    immutable = {
        "accepted": True,
        "current_pointer_read": False,
        "current_components_read": False,
        "serving_tree_read": False,
    }
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "readback_candidate_preregistration_bundle_files_v4_2",
        lambda _path: result,
    )
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "revalidate_recorded_immutable_source_v4_2",
        lambda _source: immutable,
    )
    args = argparse.Namespace(
        bundle_path=str(bundle_path),
        expected_readback_report_byte_sha256="3" * 64,
        expected_readback_report_semantic_sha256="4" * 64,
    )
    observed = subject.run_readback(args)
    assert observed["accepted"] is True
    assert observed["current_latest_pointer_read"] is False
    assert observed["current_components_read"] is False
    assert observed["current_protected_controls_read"] is False


def test_runner_has_no_forbidden_provider_or_execution_imports() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(
        {"requests", "httpx", "tushare", "yfinance", "ccxt", "ib_insync"}
    )

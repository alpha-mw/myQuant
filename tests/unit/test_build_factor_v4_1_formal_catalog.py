from __future__ import annotations

import argparse
import builtins
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from quant_investor.factors import governance_formal_catalog_adapter_v4_1 as adapter
from quant_investor.factors import governance_formal_catalog_bundle_v4_1 as bundle
from quant_investor.factors import (
    governance_formal_catalog_materialization_v4_1 as materializer,
)
from scripts import build_factor_v4_1_formal_catalog as runner


EXPECTED_CODE_SUFFIXES = (
    "quant_investor/factors/governance_formal_catalog_materialization_v4_1.py",
    "quant_investor/factors/governance_formal_catalog_adapter_v4_1.py",
    "quant_investor/factors/governance_private_bundle_io.py",
    "quant_investor/factors/governance_formal_catalog_bundle_v4_1.py",
    "quant_investor/factors/governance_discovery_v4_1.py",
    "quant_investor/factors/governance_discovery_readback_v4_1.py",
    "quant_investor/factors/governance_screening_v4.py",
    "quant_investor/factors/governance_cycle_state_v4_1.py",
    "quant_investor/factors/governance_source_v4_1.py",
    "quant_investor/factors/governance_source_readback_v4_1.py",
    "scripts/build_factor_v4_1_formal_catalog.py",
)
EXPECTED_PROTECTED_SUFFIXES = (
    "quant_investor/factor_registry/mined_factors.json",
    "data/parquet/cn/_latest.json",
    "data/parquet/cn/_catalog.json",
    "data/parquet/cn/_fundamental_latest.json",
    "data/parquet/cn/latest_manifest.json",
)
SYNTHETIC_PROTECTED_BINDINGS = {
    f"/synthetic/myQuant/{suffix}": hashlib.sha256(
        suffix.encode("utf-8")
    ).hexdigest()
    for suffix in EXPECTED_PROTECTED_SUFFIXES
}


def _bound_inputs() -> runner.BoundInputs:
    return runner.BoundInputs(
        base_ontology={"kind": "base_ontology"},
        base_catalog={"kind": "base_catalog"},
        discovery_values={"discovery.json": {"kind": "discovery"}},
        discovery_bundle_path="/tmp/formal-catalog-discovery",
        discovery_artifact_descriptors={},
        source_bindings=[{"binding_id": "source", "byte_sha256": "1" * 64}],
        code_bindings=[
            {
                "absolute_path": "/tmp/formal-code.py",
                "raw_sha256": "2" * 64,
                "size_bytes": 1,
            }
        ],
        protected_bindings=copy.deepcopy(SYNTHETIC_PROTECTED_BINDINGS),
    )


def _synthetic_artifacts(marker: str = "stable") -> dict[str, dict[str, Any]]:
    artifacts = {
        filename: {"filename": filename, "marker": marker}
        for filename in materializer.FORMAL_CATALOG_MATERIALIZATION_FILENAMES
    }
    artifacts[adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME] = {
        "filename": adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME,
        "marker": marker,
    }
    return artifacts


def _run_args(private_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        private_root=str(private_root),
        run_id="formal-catalog-unit",
        expected_discovery_readback_report_sha256="4" * 64,
        base_ontology_path="/tmp/base-ontology.v4.json",
        base_catalog_path="/tmp/base-catalog.v4.json",
    )


def test_code_and_protected_inventories_are_exact_and_fail_on_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(
        str(path.relative_to(runner.PROJECT_ROOT))
        for path in runner.REQUIRED_CODE_PATHS
    ) == EXPECTED_CODE_SUFFIXES
    assert len(materializer.REQUIRED_CODE_BINDING_SUFFIXES) == len(
        EXPECTED_CODE_SUFFIXES
    )
    assert set(materializer.REQUIRED_CODE_BINDING_SUFFIXES) == set(
        EXPECTED_CODE_SUFFIXES
    )
    assert tuple(
        str(path.relative_to(runner.PROJECT_ROOT))
        for path in runner.REQUIRED_PROTECTED_PATHS
    ) == EXPECTED_PROTECTED_SUFFIXES

    all_paths = (*runner.REQUIRED_CODE_PATHS, *runner.REQUIRED_PROTECTED_PATHS)
    hashes = {
        path: hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        for path in all_paths
    }
    seen: list[Path] = []

    def stable_read(path: Path, expected_sha256: str) -> bytes:
        assert expected_sha256 == hashes[path]
        seen.append(path)
        return path.name.encode("utf-8")

    monkeypatch.setattr(runner, "_stable_read_bound_file", stable_read)
    args = argparse.Namespace(
        code_binding=[
            f"{path}={hashes[path]}" for path in runner.REQUIRED_CODE_PATHS
        ],
        protected_binding=[
            f"{path}={hashes[path]}" for path in runner.REQUIRED_PROTECTED_PATHS
        ],
    )
    code_rows = runner._bind_code(args)
    protected = runner._bind_protected(args)
    assert {row["absolute_path"] for row in code_rows} == {
        str(path) for path in runner.REQUIRED_CODE_PATHS
    }
    assert protected == {
        str(path): hashes[path]
        for path in sorted(runner.REQUIRED_PROTECTED_PATHS, key=str)
    }
    assert set(seen) == set(all_paths)

    valid = args.code_binding
    with pytest.raises(
        runner.FactorV4_1FormalCatalogRunnerError,
        match="inventory mismatch",
    ):
        runner._parse_expected_bindings(
            valid[:-1],
            expected_paths=runner.REQUIRED_CODE_PATHS,
            label="code_binding",
        )
    with pytest.raises(
        runner.FactorV4_1FormalCatalogRunnerError,
        match="unexpected or duplicated",
    ):
        runner._parse_expected_bindings(
            [*valid, valid[0]],
            expected_paths=runner.REQUIRED_CODE_PATHS,
            label="code_binding",
        )
    with pytest.raises(
        runner.FactorV4_1FormalCatalogRunnerError,
        match="unexpected or duplicated",
    ):
        runner._parse_expected_bindings(
            [*valid[:-1], f"/tmp/unexpected-formal-code.py={'5' * 64}"],
            expected_paths=runner.REQUIRED_CODE_PATHS,
            label="code_binding",
        )


def test_stable_bound_file_read_rejects_absolute_parent_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "bound"
    parent.mkdir()
    path = parent / "control.json"
    original_bytes = b"original-bound-control"
    path.write_bytes(original_bytes)
    expected_sha256 = hashlib.sha256(original_bytes).hexdigest()
    real_read_once = runner._read_once
    read_count = 0

    def swap_parent_after_second_read(
        parent_fd: int,
        filename: str,
    ) -> tuple[bytes, tuple[int, ...]]:
        nonlocal read_count
        result = real_read_once(parent_fd, filename)
        read_count += 1
        if read_count == 2:
            parent.rename(tmp_path / "detached-original-parent")
            parent.mkdir()
            path.write_bytes(b"different-live-control")
        return result

    monkeypatch.setattr(runner, "_read_once", swap_parent_after_second_read)
    with pytest.raises(
        runner.FactorV4_1FormalCatalogRunnerError,
        match="absolute bound-file parent identity changed",
    ):
        runner._stable_read_bound_file(path, expected_sha256)


def test_build_artifacts_is_strictly_two_phase_and_adapter_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialization_calls: list[dict[str, Any] | None] = []
    adapter_inputs: list[dict[str, Any]] = []
    adapter_validation = {"adapter": "classification-only"}
    core_filenames = materializer.FORMAL_CATALOG_MATERIALIZATION_FILENAMES[:-1]
    manifest_filename = (
        materializer.FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME
    )

    def build_materialization(
        *, adapter_validation: dict[str, Any] | None, **_kwargs: Any
    ) -> dict[str, dict[str, Any]]:
        materialization_calls.append(copy.deepcopy(adapter_validation))
        values = {
            filename: {"filename": filename, "stable_core": True}
            for filename in core_filenames
        }
        values[manifest_filename] = {
            "filename": manifest_filename,
            "adapter_bound": adapter_validation is not None,
        }
        return values

    def build_adapter(**kwargs: Any) -> dict[str, Any]:
        adapter_inputs.append(copy.deepcopy(kwargs))
        return copy.deepcopy(adapter_validation)

    monkeypatch.setattr(
        runner.materialization,
        "build_formal_catalog_materialization_v4_1",
        build_materialization,
    )
    monkeypatch.setattr(
        runner.materialization,
        "validate_formal_catalog_materialization_v4_1",
        lambda value, **_kwargs: copy.deepcopy(value),
    )
    monkeypatch.setattr(
        runner.adapter,
        "build_formal_catalog_adapter_validation_v4_1",
        build_adapter,
    )
    monkeypatch.setattr(
        runner.adapter,
        "validate_formal_catalog_adapter_validation_v4_1",
        lambda value, **_kwargs: copy.deepcopy(value),
    )

    artifacts = runner._build_artifacts(_bound_inputs())

    assert materialization_calls == [None, adapter_validation]
    assert len(adapter_inputs) == 1
    assert adapter_inputs[0]["ontology"]["filename"] == (
        materializer.FORMAL_ONTOLOGY_FILENAME
    )
    assert adapter_inputs[0]["catalog"]["filename"] == (
        materializer.FORMAL_CATALOG_FILENAME
    )
    assert adapter_inputs[0]["mapping_proof"]["filename"] == (
        materializer.PRIMITIVE_MAPPING_PROOF_FILENAME
    )
    assert tuple(artifacts) == bundle.FORMAL_CATALOG_INPUT_FILENAMES
    assert artifacts[manifest_filename]["adapter_bound"] is True
    assert artifacts[adapter.FORMAL_CATALOG_ADAPTER_VALIDATION_FILENAME] == (
        adapter_validation
    )


def test_two_phase_build_rejects_adapter_induced_core_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_filenames = materializer.FORMAL_CATALOG_MATERIALIZATION_FILENAMES[:-1]
    manifest_filename = (
        materializer.FORMAL_CATALOG_MATERIALIZATION_MANIFEST_FILENAME
    )

    def build_materialization(
        *, adapter_validation: dict[str, Any] | None, **_kwargs: Any
    ) -> dict[str, dict[str, Any]]:
        values = {
            filename: {
                "filename": filename,
                "phase": "draft" if adapter_validation is None else "final",
            }
            for filename in core_filenames
        }
        values[manifest_filename] = {"filename": manifest_filename}
        return values

    monkeypatch.setattr(
        runner.materialization,
        "build_formal_catalog_materialization_v4_1",
        build_materialization,
    )
    monkeypatch.setattr(
        runner.adapter,
        "build_formal_catalog_adapter_validation_v4_1",
        lambda **_kwargs: {"adapter": "bound"},
    )

    with pytest.raises(
        runner.FactorV4_1FormalCatalogRunnerError,
        match="adapter binding changed a core materialization artifact",
    ):
        runner._build_artifacts(_bound_inputs())


def test_precommit_rebuild_drift_fails_before_publication_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bound = _bound_inputs()
    builds = iter((_synthetic_artifacts("initial"), _synthetic_artifacts("drift")))
    callback_reached = False
    contract_inputs: dict[str, Any] = {}

    monkeypatch.setattr(runner, "_bind_inputs", lambda _args: bound)
    monkeypatch.setattr(
        runner,
        "_build_artifacts",
        lambda _bound: copy.deepcopy(next(builds)),
    )
    def build_contract(**kwargs: Any) -> object:
        contract_inputs.update(copy.deepcopy(kwargs))
        return object()

    monkeypatch.setattr(
        runner.bundle,
        "build_formal_catalog_bundle_contract_v4_1",
        build_contract,
    )

    def publish_private_bundle(**kwargs: Any) -> dict[str, Any]:
        nonlocal callback_reached
        callback_reached = True
        kwargs["revalidate_inputs"]()
        raise AssertionError("drifted inputs must never reach a commit")

    monkeypatch.setattr(
        runner.private_io,
        "publish_private_bundle",
        publish_private_bundle,
    )

    with pytest.raises(
        runner.FactorV4_1FormalCatalogRunnerError,
        match="rebound formal artifacts differ before commit",
    ):
        runner.run(_run_args(tmp_path))
    assert callback_reached is True
    assert contract_inputs["protected_bindings"] == SYNTHETIC_PROTECTED_BINDINGS


def test_postpublication_protected_drift_reports_immutable_path_without_rollback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bound = _bound_inputs()
    artifacts = _synthetic_artifacts()
    bundle_path = "/tmp/immutable-formal-catalog-unit"
    after_protected = copy.deepcopy(SYNTHETIC_PROTECTED_BINDINGS)
    changed_path = sorted(after_protected)[0]
    after_protected[changed_path] = "f" * 64
    mutation_attempts: list[str] = []

    monkeypatch.setattr(runner, "_bind_inputs", lambda _args: bound)
    monkeypatch.setattr(
        runner,
        "_build_artifacts",
        lambda _bound: copy.deepcopy(artifacts),
    )
    monkeypatch.setattr(
        runner.bundle,
        "build_formal_catalog_bundle_contract_v4_1",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        runner,
        "_bind_protected",
        lambda _args: copy.deepcopy(after_protected),
    )

    def publish_private_bundle(**kwargs: Any) -> dict[str, Any]:
        kwargs["revalidate_inputs"]()
        return {
            "accepted": True,
            "bundle_path": bundle_path,
        }

    monkeypatch.setattr(
        runner.private_io,
        "publish_private_bundle",
        publish_private_bundle,
    )

    def mutation_as_error(*_args: Any, **_kwargs: Any) -> None:
        mutation_attempts.append("attempted")
        raise AssertionError("an accepted immutable bundle must not be rolled back")

    for name in ("unlink", "remove", "rmdir", "rename", "replace"):
        monkeypatch.setattr(runner.os, name, mutation_as_error)
    monkeypatch.setattr(
        runner.private_io,
        "_renameatx_np_exclusive",
        mutation_as_error,
    )
    monkeypatch.setattr(
        runner.private_io,
        "_quarantine_directory",
        mutation_as_error,
    )

    with pytest.raises(runner.FactorV4_1FormalCatalogRunnerError) as exc_info:
        runner.run(_run_args(tmp_path))

    message = str(exc_info.value)
    assert bundle_path in message
    assert "persisted build-and-precommit bindings" in message
    assert "not postcommit stability" in message
    assert mutation_attempts == []


def test_runner_transaction_revalidates_and_returns_report_only_without_forbidden_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bound = _bound_inputs()
    artifacts = _synthetic_artifacts()
    args = _run_args(tmp_path)
    bind_calls = 0
    build_calls = 0
    postpublish_protected_reads = 0
    contract_inputs: dict[str, Any] = {}

    def bind_inputs(_args: Any) -> runner.BoundInputs:
        nonlocal bind_calls
        bind_calls += 1
        return bound

    def build_artifacts(_bound: runner.BoundInputs) -> dict[str, dict[str, Any]]:
        nonlocal build_calls
        build_calls += 1
        return copy.deepcopy(artifacts)

    def build_contract(**kwargs: Any) -> object:
        contract_inputs.update(copy.deepcopy(kwargs))
        return object()

    def bind_protected(_args: Any) -> dict[str, str]:
        nonlocal postpublish_protected_reads
        postpublish_protected_reads += 1
        return copy.deepcopy(bound.protected_bindings)

    monkeypatch.setattr(runner, "_bind_inputs", bind_inputs)
    monkeypatch.setattr(runner, "_build_artifacts", build_artifacts)
    monkeypatch.setattr(
        runner.bundle,
        "build_formal_catalog_bundle_contract_v4_1",
        build_contract,
    )
    monkeypatch.setattr(runner, "_bind_protected", bind_protected)

    report = {
        "readiness": bundle.READINESS,
        "lifecycle_state": bundle.LIFECYCLE_STATE,
        "qualification": False,
        "formal_admission_authority": False,
        "production_apply_enabled": False,
        "source_authenticity_recomputed_by_materializer": True,
        "adapter_source_authenticity_recomputed": False,
        "protected_bindings": [
            {
                "absolute_path": path,
                "byte_sha256": SYNTHETIC_PROTECTED_BINDINGS[path],
            }
            for path in sorted(SYNTHETIC_PROTECTED_BINDINGS)
        ],
        "protected_bindings_semantic_sha256": "7" * 64,
        "protected_controls_bound_at_build_and_precommit": True,
        "postcommit_protected_stability_part_of_bundle_acceptance": False,
        "protected_stability_scope": (
            "build_and_precommit_only_external_controls_are_not_locked"
        ),
        "report_semantic_sha256": "6" * 64,
        "source_accounting": {"source_candidate_count": 100},
        "catalog_accounting": {"candidate_count": 267},
        "ontology_accounting": {"primitive_count": 18},
        "measurement_status": {"statistics": "not_run"},
        "blockers": ["statistics_not_run"],
        "side_effects": {
            "market_data_access_performed": False,
            "live_provider_called": False,
            "statistics_performed": False,
            "registry_write_performed": False,
            "broker_called": False,
            "trade_created": False,
        },
    }

    def publish_private_bundle(**kwargs: Any) -> dict[str, Any]:
        kwargs["revalidate_inputs"]()
        return {
            "accepted": True,
            "bundle_path": "/tmp/formal-catalog-unit",
            "readback_report": report,
            "artifact_descriptors": {
                filename: {"filename": filename}
                for filename in bundle.FORMAL_CATALOG_BUNDLE_FILENAMES
            },
        }

    monkeypatch.setattr(
        runner.private_io,
        "publish_private_bundle",
        publish_private_bundle,
    )

    forbidden_prefixes = (
        "pandas",
        "pyarrow",
        "yfinance",
        "statistics",
        "quant_investor.data",
        "quant_investor.factor_registry",
        "quant_investor.providers",
        "quant_investor.execution",
    )
    real_import = builtins.__import__

    def import_as_error(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith(forbidden_prefixes):
            raise AssertionError(f"forbidden runtime import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    def call_as_error(frame: Any, event: str, _arg: Any) -> None:
        if event != "call":
            return
        module_name = str(frame.f_globals.get("__name__", ""))
        if module_name.startswith(forbidden_prefixes):
            raise AssertionError(f"forbidden runtime call: {module_name}")

    monkeypatch.setattr(builtins, "__import__", import_as_error)
    previous_profile = sys.getprofile()
    sys.setprofile(call_as_error)
    try:
        result = runner.run(args)
    finally:
        sys.setprofile(previous_profile)

    assert result["accepted"] is True
    assert result["qualification"] is False
    assert result["formal_admission_authority"] is False
    assert result["production_apply_enabled"] is False
    assert result["v4_replay_path"] is None
    assert result["v4_replay_sha256"] is None
    assert result["transaction_plan_path"] is None
    assert result["transaction_plan_sha256"] is None
    assert result["research_head_created"] is False
    assert result["protected_bindings_before"] == SYNTHETIC_PROTECTED_BINDINGS
    assert result["protected_bindings_after"] == SYNTHETIC_PROTECTED_BINDINGS
    assert contract_inputs["protected_bindings"] == SYNTHETIC_PROTECTED_BINDINGS
    assert bind_calls == 2
    assert build_calls == 2
    assert postpublish_protected_reads == 1
    assert report["protected_controls_bound_at_build_and_precommit"] is True
    assert (
        report["postcommit_protected_stability_part_of_bundle_acceptance"]
        is False
    )
    assert all(value is False for value in report["side_effects"].values())

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from scripts import build_factor_v4_4_candidate_preregistration as subject


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path.joinpath(*subject.bundle_v4_4.ROOT_SUFFIX_V4_4)
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


def _stable(path: Path, content: bytes = b"stable\n") -> subject.StableFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return subject._stable_file(path, label=path.name)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        snapshot_id="20260720T172132Z",
        analysis_start="2026-07-01",
        cutoff="2026-07-20",
        publication_at="2026-07-20T18:00:00+08:00",
    )


def _entry(tmp_path: Path) -> subject.PublicationInputs:
    controls = {
        name: _stable(tmp_path / "controls" / f"{name}.json", name.encode())
        for name, _path in subject.PROTECTED_CONTROL_PATHS
    }
    code = {
        relative: _stable(
            tmp_path / "code" / relative,
            f"code:{index}".encode(),
        )
        for index, relative in enumerate(subject.prereg_v4_4.CODE_BINDING_PATHS_V4_4)
    }
    return subject.PublicationInputs(
        cycle_id="cn_full_a_v4_4_20260720_20260720T172132Z",
        artifacts={"synthetic.v4_4.json": {"protocol_version": "v4"}},
        raw_input_bindings=(
            {"filename": "synthetic.v4_4.json", "byte_sha256": "1" * 64, "size_bytes": 1},
        ),
        collected_raw_bytes={"synthetic.v4_4.json": b"x"},
        protected_controls=controls,
        code_bindings=code,
        source_binding_semantic_sha256="2" * 64,
        code_binding_set_semantic_sha256="3" * 64,
        diagnostic_current_mutable_sources_read=False,
    )


def _fake_readback_result(root: Path) -> dict[str, Any]:
    cycle = "cn_full_a_v4_4_20260720_20260720T172132Z"
    bundle_path = root / cycle
    report_name = subject.bundle_v4_4.READBACK_REPORT_FILENAME_V4_4
    names = (*subject.bundle_v4_4.INPUT_FILENAMES_V4_4, report_name)
    strict_name = (
        subject.prereg_v4_4.V4_2_PREDECESSOR_PREFIX
        + subject.bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
    )
    artifacts: dict[str, dict[str, Any]] = {
        name: {"artifact_semantic_sha256": "5" * 64} for name in names[:-1]
    }
    artifacts[strict_name] = {"synthetic_immutable_source": True}
    artifacts[subject.bundle_v4_4.CYCLE_ROOT_FILENAME_V4_4] = {"cycle_id": cycle}
    return {
        "accepted": True,
        "bundle_path": str(bundle_path),
        "artifact_descriptors": {
            name: {
                "absolute_path": str(bundle_path / name),
                "byte_sha256": "6" * 64 if name == report_name else "7" * 64,
                "size_bytes": 1,
                "mode": 0o600,
                "uid": os.getuid(),
                "nlink": 1,
            }
            for name in names
        },
        "readback_report": {"artifact_semantic_sha256": "8" * 64},
        "artifacts": artifacts,
    }


def _future_v4_2_artifacts(tmp_path: Path) -> dict[str, dict[str, Any]]:
    """Build a real v4.2 graph whose snapshot/cutoff are both 2026-07-20."""

    from quant_investor.factors.governance_source_readback_v4_1 import (
        BoundCutoffInputsV4_1,
    )
    from tests.unit import (
        test_factor_governance_candidate_preregistration_bundle_v4_2 as v42_fixture,
    )

    bound, pointer_raw, components_raw = v42_fixture._bound_inputs(
        tmp_path / "source-inputs"
    )

    def replace_date(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace_date(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_date(item) for item in value]
        if value == "2026-07-18":
            return "2026-07-20"
        if value == "20260718":
            return "20260720"
        return value

    pointer_value = replace_date(json.loads(pointer_raw))
    future_pointer_raw = v42_fixture._raw_json(pointer_value)
    binding = copy.deepcopy(bound.binding)
    binding["cutoff_date"] = "2026-07-20"
    binding["latest_pointer"]["size_bytes"] = len(future_pointer_raw)
    binding["latest_pointer"]["sha256"] = hashlib.sha256(
        future_pointer_raw
    ).hexdigest()
    calendar = binding["calendar"]
    calendar["cutoff_date"] = "2026-07-20"
    calendar["open_sessions"] = ["2026-07-17", "2026-07-20"]
    calendar_base = {
        key: item for key, item in calendar.items() if key != "semantic_sha256"
    }
    calendar["semantic_sha256"] = hashlib.sha256(
        subject.prereg_v4_2.canonical_json_bytes_v4_2(calendar_base)
    ).hexdigest()
    future_bound = BoundCutoffInputsV4_1(
        binding=binding,
        calendar_sessions=("2026-07-17", "2026-07-20"),
        component_symbols=bound.component_symbols,
        pit_records=bound.pit_records,
        bound_table_symbol_row_counts=bound.bound_table_symbol_row_counts,
    )
    source = subject.bundle_v4_2.build_strict_full_a_source_binding_v4_2(
        bound_inputs=future_bound,
        latest_pointer_raw=future_pointer_raw,
        components_raw=components_raw,
    )
    aquant = subject.prereg_v4_2.build_aquant_receipt_v4_2()
    myquant = subject.prereg_v4_2.build_myquant_receipt_v4_2()
    operators = subject.prereg_v4_2.build_operator_semantics_v4_2()
    comparison = subject.prereg_v4_2.build_comparison_catalog_receipt_v4_2(
        catalog_id="synthetic-comparison-v4",
        catalog_byte_sha256="1" * 64,
        catalog_semantic_sha256="2" * 64,
        primitive_count=1,
        definition_identity_inventory=[
            {"name": "legacy_synthetic", "definition_identity_sha256": "3" * 64}
        ],
    )
    selection = subject.prereg_v4_2.build_selection_spec_v4_2(
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    code = subject.bundle_v4_2.build_code_binding_set_v4_2(
        repository_root=v42_fixture._code_root(tmp_path / "v4-2-code")
    )
    return subject.bundle_v4_2.build_candidate_preregistration_bundle_artifacts_v4_2(
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=source,
        code_binding_set=code,
    )


def _valid_publication_entry(tmp_path: Path) -> subject.PublicationInputs:
    unprefixed = _future_v4_2_artifacts(tmp_path / "v4-2")
    predecessor = {
        subject.prereg_v4_4.V4_2_PREDECESSOR_PREFIX + filename: value
        for filename, value in unprefixed.items()
    }
    predecessor_raw = {
        subject.prereg_v4_4.V4_2_PREDECESSOR_PREFIX + filename: (
            subject.prereg_v4_2.canonical_file_bytes_v4_2(value)
        )
        for filename, value in unprefixed.items()
    }
    diagnostic, diagnostic_raw = subject._collect_prior_diagnostic_graph(
        bundle_path=subject.FIXED_DIAGNOSTIC_BUNDLE_PATH
    )
    code_snapshots = {
        relative: _stable(
            tmp_path / "v4-4-code" / relative,
            f"v4.4-code:{index}".encode(),
        )
        for index, relative in enumerate(
            subject.prereg_v4_4.CODE_BINDING_PATHS_V4_4,
            start=1,
        )
    }
    code = subject._build_code_binding_set(code_snapshots)
    collected_raw = {**predecessor_raw, **diagnostic_raw}
    artifacts = subject.bundle_v4_4.build_candidate_preregistration_bundle_artifacts_v4_4(
        v4_2_predecessor_artifacts=predecessor,
        prior_diagnostic_artifacts=diagnostic,
        code_binding_set=code,
        publication_at="2026-07-20T18:00:00+08:00",
        collected_raw_bytes=collected_raw,
    )
    raw_bindings = subject._raw_input_bindings(
        artifacts=artifacts,
        collected_raw_bytes=collected_raw,
    )
    normalized = subject.bundle_v4_4.validate_candidate_preregistration_bundle_inputs_v4_4(
        artifacts,
        raw_input_bindings=raw_bindings,
        collected_raw_bytes=collected_raw,
    )
    controls = {
        name: _stable(tmp_path / "controls" / f"{name}.json", name.encode())
        for name, _path in subject.PROTECTED_CONTROL_PATHS
    }
    strict_name = (
        subject.prereg_v4_4.V4_2_PREDECESSOR_PREFIX
        + subject.bundle_v4_2.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2
    )
    root = normalized[subject.bundle_v4_4.CYCLE_ROOT_FILENAME_V4_4]
    return subject.PublicationInputs(
        cycle_id=root["cycle_id"],
        artifacts=normalized,
        raw_input_bindings=raw_bindings,
        collected_raw_bytes=collected_raw,
        protected_controls=controls,
        code_bindings=code_snapshots,
        source_binding_semantic_sha256=normalized[strict_name][
            "artifact_semantic_sha256"
        ],
        code_binding_set_semantic_sha256=normalized[
            subject.bundle_v4_4.CODE_BINDING_SET_FILENAME_V4_4
        ]["artifact_semantic_sha256"],
        diagnostic_current_mutable_sources_read=False,
    )


def test_public_cli_has_no_identity_candidate_or_side_effect_override() -> None:
    parser = subject.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    help_text = parser.format_help() + "".join(
        command.format_help() for command in subparsers.choices.values()
    )
    for token in subject._FORBIDDEN_ARGUMENT_TOKENS:
        assert token not in help_text
    assert str(subject.PRODUCTION_PRIVATE_ROOT) == (
        "/Users/maxwell/mySpace/myQuant/reports/factor_governance/private/"
        "v4_4_candidate_preregistration"
    )


def test_code_and_protected_control_inventories_are_exact() -> None:
    assert subject.prereg_v4_4.CODE_BINDING_PATHS_V4_4 == (
        "scripts/build_factor_v4_4_candidate_preregistration.py",
        "quant_investor/factors/governance_candidate_preregistration_v4_4.py",
        "quant_investor/factors/governance_candidate_preregistration_bundle_v4_4.py",
        "quant_investor/factors/governance_candidate_preregistration_v4_2.py",
        "quant_investor/factors/governance_candidate_preregistration_bundle_v4_2.py",
        "scripts/build_factor_v4_2_candidate_preregistration.py",
        "quant_investor/factors/governance_prior_diagnostic_nomination_v4_3.py",
        "quant_investor/factors/governance_prior_diagnostic_nomination_bundle_v4_3.py",
        "quant_investor/factors/governance_cycle_state_v4_1.py",
        "quant_investor/factors/governance_private_bundle_io.py",
        "quant_investor/factors/governance_source_readback_v4_1.py",
        "quant_investor/factors/governance_screening_v4.py",
        "quant_investor/codex_review/storage.py",
        "quant_investor/market/pit_universe.py",
        "quant_investor/factors/governance_source_v4_1.py",
    )
    assert tuple(name for name, _path in subject.PROTECTED_CONTROL_PATHS) == (
        "registry",
        "latest_pointer",
        "catalog",
        "fundamental_latest",
        "latest_manifest",
    )
    assert len({path for _name, path in subject.PROTECTED_CONTROL_PATHS}) == 5


@pytest.mark.parametrize(
    ("snapshot_id", "analysis_start", "cutoff", "match"),
    [
        ("20260719T235959Z", "2026-07-01", "2026-07-19", "strictly later"),
        ("20260720T235959Z", "2026-07-21", "2026-07-20", "must not"),
        ("20260721T000000Z", "2026-07-01", "2026-07-20", "exactly equal"),
        ("20260720T256199Z", "2026-07-01", "2026-07-20", "real UTC"),
        ("20269999T000000Z", "2026-07-01", "2026-99-99", "invalid"),
    ],
)
def test_cycle_identity_rejects_stale_or_inconsistent_dates(
    snapshot_id: str, analysis_start: str, cutoff: str, match: str
) -> None:
    with pytest.raises(subject.FactorV4_4CandidatePreregistrationRunnerError, match=match):
        subject._validate_cycle_identity(
            snapshot_id=snapshot_id,
            analysis_start=analysis_start,
            cutoff=cutoff,
        )


def test_cycle_identity_is_deterministic() -> None:
    assert subject._validate_cycle_identity(
        snapshot_id="20260720T172132Z",
        analysis_start="2021-06-25",
        cutoff="2026-07-20",
    ) == "cn_full_a_v4_4_20260720_20260720T172132Z"


def test_stale_cutoff_rejects_before_probe_root_or_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = argparse.Namespace(
        snapshot_id="20260719T172132Z",
        analysis_start="2021-06-25",
        cutoff="2026-07-19",
        publication_at="2026-07-19T18:00:00+08:00",
    )
    root = tmp_path.joinpath(*subject.bundle_v4_4.ROOT_SUFFIX_V4_4)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("stale cutoff crossed the publication boundary")

    monkeypatch.setattr(subject, "_collect_publication_inputs", forbidden)
    monkeypatch.setattr(
        subject.bundle_v4_4,
        "publish_candidate_preregistration_bundle_v4_4",
        forbidden,
    )
    with pytest.raises(subject.FactorV4_4CandidatePreregistrationRunnerError, match="strictly later"):
        subject.run_publish(
            args,
            private_root=root,
            exclusive_rename_probe=forbidden,
        )
    assert not root.exists()


def test_private_root_is_fixed_private_and_duplicate_rejects_before_collect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private_root(tmp_path)
    cycle = "cn_full_a_v4_4_20260720_20260720T172132Z"
    root.chmod(0o755)
    with pytest.raises(subject.FactorV4_4CandidatePreregistrationRunnerError, match="0700"):
        subject._validate_private_root_preflight(root, cycle_id=cycle)
    root.chmod(0o700)
    destination = root / cycle
    destination.mkdir(mode=0o700)
    sentinel = destination / "sentinel.json"
    sentinel.write_bytes(b"preserve")
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("duplicate publish must reject before collection")

    monkeypatch.setattr(subject, "_collect_publication_inputs", forbidden)
    with pytest.raises(subject.FactorV4_4CandidatePreregistrationRunnerError, match="already exists"):
        subject.run_publish(
            _args(),
            private_root=root,
            exclusive_rename_probe=lambda: None,
        )
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before


def test_git_object_reads_scrub_all_inherited_git_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout=b"bound\n", stderr=b"")

    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_INDEX_FILE",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
    ):
        monkeypatch.setenv(key, f"/tmp/injected-{key.lower()}")
    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    assert subject._run_git(tmp_path, ["rev-parse", "HEAD"]) == b"bound\n"
    environment = captured["environment"]
    assert {key for key in environment if key.startswith("GIT_")} == {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
    }
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
    assert captured["command"] == ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]


def test_expected_hash_inventories_are_ordered_exact_and_reject_extra() -> None:
    code_values = [
        f"{name}={index:064x}"
        for index, name in enumerate(subject.prereg_v4_4.CODE_BINDING_PATHS_V4_4, start=1)
    ]
    parsed = subject._parse_named_hashes(
        code_values,
        expected_names=subject.prereg_v4_4.CODE_BINDING_PATHS_V4_4,
        label="code",
    )
    assert tuple(parsed) == subject.prereg_v4_4.CODE_BINDING_PATHS_V4_4
    with pytest.raises(subject.FactorV4_4CandidatePreregistrationRunnerError, match="inventory"):
        subject._parse_named_hashes(
            [*code_values, f"extra.py={'f' * 64}"],
            expected_names=subject.prereg_v4_4.CODE_BINDING_PATHS_V4_4,
            label="code",
        )


def test_raw_input_bindings_preserve_exact_fourteen_plus_three_bytes() -> None:
    artifacts = {
        filename: {"filename": filename, "protocol_version": "v4"}
        for filename in subject.bundle_v4_4.INPUT_FILENAMES_V4_4
    }
    raw_names = (
        *subject.prereg_v4_4.V4_2_PREDECESSOR_FILENAMES,
        *subject.prereg_v4_4.PRIOR_DIAGNOSTIC_FILENAMES,
    )
    raw = {
        filename: f"original:{index}:{filename}".encode()
        for index, filename in enumerate(raw_names)
    }
    bindings = subject._raw_input_bindings(
        artifacts=artifacts,
        collected_raw_bytes=raw,
    )
    assert len(bindings) == 26
    by_name = {row["filename"]: row for row in bindings}
    for filename in raw_names:
        assert by_name[filename] == {
            "filename": filename,
            "byte_sha256": hashlib.sha256(raw[filename]).hexdigest(),
            "size_bytes": len(raw[filename]),
        }


def test_locked_double_collection_drift_rejects_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private_root(tmp_path)
    first = _entry(tmp_path / "first")
    drifted = subject.PublicationInputs(
        **{
            **first.__dict__,
            "collected_raw_bytes": {"synthetic.v4_4.json": b"drift"},
        }
    )
    values = iter((first, drifted))
    collection_count = 0

    def collect(*_args: Any, **_kwargs: Any) -> subject.PublicationInputs:
        nonlocal collection_count
        collection_count += 1
        return next(values)

    def fake_publish(**kwargs: Any) -> dict[str, Any]:
        kwargs["revalidate_inputs"]()
        raise AssertionError("commit must not be reached")

    monkeypatch.setattr(subject, "_collect_publication_inputs", collect)
    monkeypatch.setattr(
        subject.bundle_v4_4,
        "publish_candidate_preregistration_bundle_v4_4",
        fake_publish,
    )
    with pytest.raises(subject.FactorV4_4CandidatePreregistrationRunnerError, match="changed"):
        subject.run_publish(
            _args(),
            private_root=root,
            exclusive_rename_probe=lambda: None,
        )
    assert collection_count == 2
    assert not (root / first.cycle_id).exists()


def test_publish_summary_uses_exact_descriptor_mapping_and_false_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private_root(tmp_path)
    entry = _entry(tmp_path / "entry")
    readback = _fake_readback_result(root)
    collection_count = 0

    def collect(*_args: Any, **_kwargs: Any) -> subject.PublicationInputs:
        nonlocal collection_count
        collection_count += 1
        return entry

    def fake_publish(**kwargs: Any) -> dict[str, Any]:
        kwargs["revalidate_inputs"]()
        return {"accepted": True, "bundle_path": readback["bundle_path"]}

    monkeypatch.setattr(subject, "_collect_publication_inputs", collect)
    monkeypatch.setattr(
        subject.bundle_v4_4,
        "publish_candidate_preregistration_bundle_v4_4",
        fake_publish,
    )
    monkeypatch.setattr(
        subject.bundle_v4_4,
        "readback_candidate_preregistration_bundle_files_v4_4",
        lambda _path: readback,
    )
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "revalidate_recorded_immutable_source_v4_2",
        lambda _source: {
            "accepted": True,
            "current_pointer_read": False,
            "current_components_read": False,
            "serving_tree_read": False,
        },
    )
    result = subject.run_publish(
        _args(),
        private_root=root,
        exclusive_rename_probe=lambda: None,
    )
    report_name = subject.bundle_v4_4.READBACK_REPORT_FILENAME_V4_4
    assert collection_count == 2
    assert result["readback_report_path"] == readback["artifact_descriptors"][report_name][
        "absolute_path"
    ]
    assert result["readback_report_byte_sha256"] == "6" * 64
    assert result["readback_report_semantic_sha256"] == "8" * 64
    assert result["authority"] == subject.prereg_v4_4.AUTHORITY_FLAGS
    assert result["side_effects"] == subject.prereg_v4_4.SIDE_EFFECT_FLAGS
    assert result["external_maintenance_serialization_claimed"] is False


def test_real_temp_publish_has_exact_27_private_files_raw_bytes_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_factor_governance_candidate_preregistration_bundle_v4_2 import (
        _portable_private_publication,
    )

    _portable_private_publication(monkeypatch)
    entry = _valid_publication_entry(tmp_path / "entry")
    root = _private_root(tmp_path / "published")
    args = argparse.Namespace(
        snapshot_id="20260720T010203Z",
        analysis_start="2026-07-17",
        cutoff="2026-07-20",
        publication_at="2026-07-20T18:00:00+08:00",
    )
    collection_count = 0

    def collect(*_args: Any, **_kwargs: Any) -> subject.PublicationInputs:
        nonlocal collection_count
        collection_count += 1
        return entry

    monkeypatch.setattr(subject, "_collect_publication_inputs", collect)
    summary = subject.run_publish(
        args,
        private_root=root,
        exclusive_rename_probe=lambda: None,
    )
    assert collection_count == 2
    assert summary["accepted"] is True
    assert summary["authority"] == subject.prereg_v4_4.AUTHORITY_FLAGS
    assert summary["side_effects"] == subject.prereg_v4_4.SIDE_EFFECT_FLAGS
    assert summary["diagnostic_current_mutable_sources_read"] is False

    bundle_path = Path(summary["bundle_path"])
    files = tuple(bundle_path.iterdir())
    assert len(files) == 27
    assert {path.name for path in files} == {
        *subject.bundle_v4_4.INPUT_FILENAMES_V4_4,
        subject.bundle_v4_4.READBACK_REPORT_FILENAME_V4_4,
    }
    assert root.stat().st_mode & 0o777 == 0o700
    assert bundle_path.stat().st_mode & 0o777 == 0o700
    for path in files:
        metadata = path.stat()
        assert path.is_file()
        assert metadata.st_mode & 0o777 == 0o600
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1
    for filename, original_raw in entry.collected_raw_bytes.items():
        assert (bundle_path / filename).read_bytes() == original_raw

    independent = subject.bundle_v4_4.readback_candidate_preregistration_bundle_files_v4_4(
        bundle_path
    )
    descriptors = independent["artifact_descriptors"]
    assert isinstance(descriptors, dict)
    assert len(descriptors) == 27
    report_name = subject.bundle_v4_4.READBACK_REPORT_FILENAME_V4_4
    assert summary["readback_report_path"] == descriptors[report_name]["absolute_path"]
    assert summary["readback_report_byte_sha256"] == descriptors[report_name][
        "byte_sha256"
    ]
    assert summary["readback_report_semantic_sha256"] == independent[
        "readback_report"
    ]["artifact_semantic_sha256"]

    historical = subject.run_readback(
        argparse.Namespace(
            bundle_path=str(bundle_path),
            expected_readback_report_byte_sha256=summary[
                "readback_report_byte_sha256"
            ],
            expected_readback_report_semantic_sha256=summary[
                "readback_report_semantic_sha256"
            ],
        )
    )
    assert historical["accepted"] is True
    assert historical["current_latest_pointer_read"] is False
    assert historical["current_diagnostic_sources_read"] is False
    assert historical["authority"] == subject.prereg_v4_4.AUTHORITY_FLAGS
    assert historical["side_effects"] == subject.prereg_v4_4.SIDE_EFFECT_FLAGS

    before_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    before_inventory = sorted(path.name for path in root.iterdir())
    calls_before_retry = collection_count

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("duplicate cycle must reject before collection")

    monkeypatch.setattr(subject, "_collect_publication_inputs", forbidden)
    with pytest.raises(subject.FactorV4_4CandidatePreregistrationRunnerError, match="already exists"):
        subject.run_publish(
            args,
            private_root=root,
            exclusive_rename_probe=lambda: None,
        )
    assert collection_count == calls_before_retry
    assert sorted(path.name for path in root.iterdir()) == before_inventory
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle_path.iterdir()
    } == before_hashes


def test_real_temp_publish_fault_before_rename_leaves_no_final_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_factor_governance_candidate_preregistration_bundle_v4_2 import (
        _portable_private_publication,
    )

    _portable_private_publication(monkeypatch)
    entry = _valid_publication_entry(tmp_path / "entry")
    root = _private_root(tmp_path / "fault")
    monkeypatch.setattr(
        subject,
        "_collect_publication_inputs",
        lambda *_args, **_kwargs: entry,
    )

    def fault(point: str) -> None:
        if point == "commit:rename:before":
            raise OSError("injected pre-rename fault")

    with pytest.raises(Exception, match="injected pre-rename fault"):
        subject.run_publish(
            argparse.Namespace(
                snapshot_id="20260720T010203Z",
                analysis_start="2026-07-17",
                cutoff="2026-07-20",
                publication_at="2026-07-20T18:00:00+08:00",
            ),
            private_root=root,
            exclusive_rename_probe=lambda: None,
            _test_fault_hook=fault,
        )
    assert not (root / entry.cycle_id).exists()
    assert not any(".staging" in path.name for path in root.iterdir())


def test_real_temp_publish_race_never_clobbers_competing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.test_factor_governance_candidate_preregistration_bundle_v4_2 import (
        _portable_private_publication,
    )

    _portable_private_publication(monkeypatch)
    entry = _valid_publication_entry(tmp_path / "entry")
    root = _private_root(tmp_path / "race")
    monkeypatch.setattr(
        subject,
        "_collect_publication_inputs",
        lambda *_args, **_kwargs: entry,
    )
    sentinel_raw = b"competing publisher owns this destination"

    def race() -> None:
        destination = root / entry.cycle_id
        destination.mkdir(mode=0o700)
        (destination / "sentinel").write_bytes(sentinel_raw)

    with pytest.raises(Exception):
        subject.run_publish(
            argparse.Namespace(
                snapshot_id="20260720T010203Z",
                analysis_start="2026-07-17",
                cutoff="2026-07-20",
                publication_at="2026-07-20T18:00:00+08:00",
            ),
            private_root=root,
            exclusive_rename_probe=lambda: None,
            _test_race_hook=race,
        )
    destination = root / entry.cycle_id
    assert (destination / "sentinel").read_bytes() == sentinel_raw
    assert tuple(path.name for path in destination.iterdir()) == ("sentinel",)
    assert not any(".staging" in path.name for path in root.iterdir())


def test_historical_readback_uses_only_embedded_immutable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private_root(tmp_path)
    result = _fake_readback_result(root)
    bundle_path = Path(result["bundle_path"])
    immutable = {
        "accepted": True,
        "current_pointer_read": False,
        "current_components_read": False,
        "serving_tree_read": False,
    }
    monkeypatch.setattr(
        subject.bundle_v4_4,
        "readback_candidate_preregistration_bundle_files_v4_4",
        lambda _path: result,
    )
    monkeypatch.setattr(
        subject.bundle_v4_2,
        "revalidate_recorded_immutable_source_v4_2",
        lambda _source: immutable,
    )
    observed = subject.run_readback(
        argparse.Namespace(
            bundle_path=str(bundle_path),
            expected_readback_report_byte_sha256="6" * 64,
            expected_readback_report_semantic_sha256="8" * 64,
        )
    )
    assert observed["accepted"] is True
    assert observed["current_latest_pointer_read"] is False
    assert observed["current_protected_controls_read"] is False
    assert observed["current_diagnostic_sources_read"] is False
    assert observed["authority"] == subject.prereg_v4_4.AUTHORITY_FLAGS


def test_runner_imports_no_network_provider_or_execution_client() -> None:
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

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any

import pytest

from quant_investor.factors import (
    governance_candidate_preregistration_bundle_v4_2 as v42_bundle,
)
from quant_investor.factors import governance_candidate_preregistration_v4_2 as v42
from quant_investor.factors import (
    governance_candidate_preregistration_bundle_v4_4 as subject,
)
from quant_investor.factors import governance_candidate_preregistration_v4_4 as core
from quant_investor.factors import (
    governance_prior_diagnostic_nomination_bundle_v4_3 as diagnostic_bundle,
)
from quant_investor.factors import governance_private_bundle_io as private_io
from quant_investor.factors.governance_source_readback_v4_1 import (
    BoundCutoffInputsV4_1,
    INPUT_BINDING_SCHEMA_VERSION,
    SOURCE_USE_PROHIBITED,
)
from tests.unit.test_factor_governance_candidate_preregistration_bundle_v4_2 import (
    _code_root,
)
from tests.unit.test_factor_governance_prior_diagnostic_nomination_bundle_v4_3 import (
    _bundle_artifacts as _diagnostic_inputs,
)


_rename_lock = threading.Lock()


def _raw_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _record(path: Path, raw: bytes) -> dict[str, Any]:
    return {"absolute_path": str(path), "size_bytes": len(raw), "sha256": _sha(raw)}


def _bound_inputs(
    tmp_path: Path,
    *,
    cutoff: str = "2026-07-20",
) -> tuple[BoundCutoffInputsV4_1, bytes, bytes]:
    compact = cutoff.replace("-", "")
    snapshot_id = f"{compact}T010203Z"
    symbols = ("000001.SZ", "600000.SH")
    components_raw = _raw_json(
        {"full_a": list(symbols), "stats": {"full_a": len(symbols)}}
    )
    manifest_raw = b"snapshot-manifest"
    pit_manifest_raw = b"pit-generation-manifest"
    pit_membership_raw = b"pit-membership"
    pointer_path = tmp_path / "data" / "parquet" / "cn" / "_latest.json"
    snapshot_root = pointer_path.parent / "_snapshots" / snapshot_id
    manifest_path = pointer_path.parent / "_snapshots" / f"{snapshot_id}.json"
    components_path = tmp_path / "components.json"
    pit_root = pointer_path.parent / "reference" / "_generations" / f"pit-{compact}-test"
    pit_manifest_path = pit_root / "manifest.json"
    pit_membership_path = pit_root / "membership.parquet"
    table_root = snapshot_root / "table" / "bars"
    serving_root = snapshot_root / "serving" / "bars"
    scope_sha = _sha("\n".join(symbols).encode())
    pointer_raw = _raw_json(
        {
            "snapshot_id": snapshot_id,
            "status": "OK",
            "blockers": [],
            "latest_trade_date": compact,
            "latest_available_trade_date": compact,
            "latest_complete_trade_date": compact,
            "manifest_path": str(manifest_path),
            "table_root": str(table_root),
            "derived_serving_root": str(serving_root),
            "coverage": {
                "coverage_schema_version": "cn-full-a-coverage.v4",
                "complete": True,
                "coverage_ratio": 1.0,
                "categories_checked": ["full_a"],
                "expected_scope_count": len(symbols),
                "coverage_complete_count": len(symbols),
                "expected_scope_sha256": scope_sha,
                "coverage_trade_date": compact,
                "latest_available_trade_date": compact,
                "latest_complete_trade_date": compact,
                "blocking_incomplete_count": 0,
                "classification_sets_disjoint": True,
                "true_missing_symbols": [],
                "pit_generation_id": f"pit-{compact}-test",
                "pit_generation_manifest_path": str(pit_manifest_path),
                "pit_generation_manifest_sha256": _sha(pit_manifest_raw),
                "pit_membership_path": str(pit_membership_path),
                "pit_membership_sha256": _sha(pit_membership_raw),
            },
        }
    )
    table_raw = b"table-bytes"
    for path, raw in (
        (manifest_path, manifest_raw),
        (pit_manifest_path, pit_manifest_raw),
        (pit_membership_path, pit_membership_raw),
        (table_root / "year=2026" / "month=07" / "part.parquet", table_raw),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    table_rows = [
        {
            "relative_path": "year=2026/month=07/part.parquet",
            "size_bytes": len(table_raw),
            "sha256": _sha(table_raw),
            "hard_link_count": 1,
            "dataset_member": True,
        }
    ]
    analysis_start = "2026-07-18"
    sessions = [analysis_start, cutoff]
    calendar_base = {
        "analysis_start": analysis_start,
        "cutoff_date": cutoff,
        "open_session_count": len(sessions),
        "open_sessions": sessions,
    }
    binding = {
        "schema_version": INPUT_BINDING_SCHEMA_VERSION,
        "market": "CN",
        "snapshot_id": snapshot_id,
        "cutoff_date": cutoff,
        "latest_pointer": _record(pointer_path, pointer_raw),
        "snapshot_manifest": _record(manifest_path, manifest_raw),
        "components": {
            **_record(components_path, components_raw),
            "universe": "full_a",
            "count": len(symbols),
            "newline_set_sha256": scope_sha,
        },
        "pit_generation": {
            "generation_id": f"pit-{compact}-test",
            "manifest": _record(pit_manifest_path, pit_manifest_raw),
            "membership": _record(pit_membership_path, pit_membership_raw),
            "row_count": 2,
            "historical_alias_table_evidence": [],
        },
        "table": {
            "absolute_root": str(table_root),
            "regular_file_count": 1,
            "parquet_file_count": 1,
            "inventory_sha256": hashlib.sha256(
                v42.canonical_json_bytes_v4_2(table_rows)
            ).hexdigest(),
            "parquet_inventory": table_rows,
            "bound_symbol_inventory": {
                "symbol_count": len(symbols),
                "symbols_newline_sha256": scope_sha,
                "noncanonical_symbol_count": 0,
            },
        },
        "calendar": {
            **calendar_base,
            "semantic_sha256": hashlib.sha256(
                v42.canonical_json_bytes_v4_2(calendar_base)
            ).hexdigest(),
        },
        "eligibility_boundary": {
            "component_source": str(components_path),
            "pit_source": str(pit_membership_path),
            "bar_source": str(table_root),
            "serving_inventory": {
                "absolute_root": str(serving_root),
                "symbol_count": 3,
                "use": SOURCE_USE_PROHIBITED,
                "was_scanned": False,
            },
        },
        "readiness": "EXPLORATORY_INPUT_BOUND",
        "side_effects": {
            "registry": False,
            "wal": False,
            "budget": False,
            "apply": False,
            "broker": False,
            "order": False,
            "trade": False,
            "network": False,
        },
    }
    return (
        BoundCutoffInputsV4_1(
            binding=binding,
            calendar_sessions=tuple(sessions),
            component_symbols=symbols,
            pit_records=({"symbol": symbols[0]}, {"symbol": symbols[1]}),
            bound_table_symbol_row_counts=((symbols[0], 2), (symbols[1], 2)),
        ),
        pointer_raw,
        components_raw,
    )


def _v42_prefixed_graph(
    tmp_path: Path, *, cutoff: str = "2026-07-20"
) -> dict[str, dict[str, Any]]:
    bound, pointer_raw, components_raw = _bound_inputs(tmp_path / "source", cutoff=cutoff)
    source = v42_bundle.build_strict_full_a_source_binding_v4_2(
        bound_inputs=bound,
        latest_pointer_raw=pointer_raw,
        components_raw=components_raw,
    )
    aquant = v42.build_aquant_receipt_v4_2()
    myquant = v42.build_myquant_receipt_v4_2()
    operators = v42.build_operator_semantics_v4_2()
    comparison = v42.build_comparison_catalog_receipt_v4_2(
        catalog_id="synthetic-comparison-v4",
        catalog_byte_sha256="1" * 64,
        catalog_semantic_sha256="2" * 64,
        primitive_count=1,
        definition_identity_inventory=[
            {"name": "legacy_synthetic", "definition_identity_sha256": "3" * 64}
        ],
    )
    selection = v42.build_selection_spec_v4_2(
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    code = v42_bundle.build_code_binding_set_v4_2(
        repository_root=_code_root(tmp_path / "code")
    )
    graph = v42_bundle.build_candidate_preregistration_bundle_artifacts_v4_2(
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=source,
        code_binding_set=code,
    )
    return {
        core.V4_2_PREDECESSOR_PREFIX + filename: value
        for filename, value in graph.items()
    }


def _diagnostic_graph(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    inputs = _diagnostic_inputs()
    bindings = []
    for filename in diagnostic_bundle.INPUT_FILENAMES_V4_3:
        raw = core.canonical_file_bytes_v4_4(inputs[filename])
        bindings.append(
            {
                "filename": filename,
                "byte_sha256": _sha(raw),
                "size_bytes": len(raw),
                "mode": 0o600,
                "uid": os.getuid(),
                "nlink": 1,
            }
        )
    report = getattr(diagnostic_bundle, "_build_readback_report")(
        run_id=inputs[diagnostic_bundle.PRIOR_DIAGNOSTIC_NOMINATION_FILENAME_V4_3][
            "run_id"
        ],
        artifacts=inputs,
        artifact_bindings=bindings,
    )
    complete = diagnostic_bundle.validate_prior_diagnostic_nomination_bundle_artifacts_v4_3(
        {**inputs, diagnostic_bundle.READBACK_REPORT_FILENAME_V4_3: report}
    )
    expected = tuple(
        core.build_artifact_binding_v4_4(filename=filename, artifact=complete[filename])
        for filename in core.PRIOR_DIAGNOSTIC_FILENAMES
    )
    monkeypatch.setattr(core, "EXPECTED_PRIOR_DIAGNOSTIC_BINDINGS", expected)
    return complete


def _code_binding() -> dict[str, Any]:
    return core.build_code_binding_set_v4_4(
        ordered_bindings=[
            {
                "order": index,
                "relative_path": relative,
                "byte_sha256": hashlib.sha256(relative.encode()).hexdigest(),
                "size_bytes": len(relative.encode()),
            }
            for index, relative in enumerate(core.CODE_BINDING_PATHS_V4_4, start=1)
        ]
    )


def _artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cutoff: str = "2026-07-20",
) -> dict[str, dict[str, Any]]:
    return subject.build_candidate_preregistration_bundle_artifacts_v4_4(
        v4_2_predecessor_artifacts=_v42_prefixed_graph(
            tmp_path / "v42", cutoff=cutoff
        ),
        prior_diagnostic_artifacts=_diagnostic_graph(monkeypatch),
        code_binding_set=_code_binding(),
        publication_at=f"{cutoff}T18:00:00+08:00",
    )


def _raw_collected(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, bytes]:
    return {
        filename: core.canonical_file_bytes_v4_4(artifacts[filename])
        for filename in subject.COLLECTED_RAW_FILENAMES_V4_4
    }


def _portable_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    def rename_exclusive(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        with _rename_lock:
            try:
                os.stat(destination_name, dir_fd=destination_directory_fd, follow_symlinks=False)
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
    root = tmp_path.joinpath(*subject.ROOT_SUFFIX_V4_4)
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


def _file_hashes(path: Path) -> dict[str, str]:
    return {item.name: _sha(item.read_bytes()) for item in path.iterdir()}


def _assert_no_staging_residue(root: Path, cycle_id: str) -> None:
    assert not any(
        item.name.startswith(f".{cycle_id}.staging.") for item in root.iterdir()
    )


def _base_readback_bindings(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    bindings = []
    for filename in subject.INPUT_FILENAMES_V4_4:
        raw = core.canonical_file_bytes_v4_4(artifacts[filename])
        bindings.append(
            {
                "filename": filename,
                "byte_sha256": _sha(raw),
                "size_bytes": len(raw),
                "mode": 0o600,
                "uid": os.getuid(),
                "nlink": 1,
            }
        )
    return bindings


def _readback_report(
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return getattr(subject, "_build_readback_report")(
        run_id=artifacts[subject.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"],
        artifacts=artifacts,
        artifact_bindings=_base_readback_bindings(artifacts),
    )


def test_exact_26_plus_one_inventory_and_crosslinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    assert tuple(artifacts) == subject.INPUT_FILENAMES_V4_4
    assert len(artifacts) == 26
    contract = subject.candidate_preregistration_bundle_contract_v4_4()
    assert contract.input_filenames == subject.INPUT_FILENAMES_V4_4
    assert contract.canonical_filenames[-1] == subject.READBACK_REPORT_FILENAME_V4_4
    assert len(contract.canonical_filenames) == 27
    assert subject.validate_candidate_preregistration_bundle_inputs_v4_4(
        artifacts,
        raw_input_bindings=subject.raw_input_bindings_v4_4(artifacts),
        collected_raw_bytes=_raw_collected(artifacts),
    ) == artifacts


def test_standalone_readback_rejects_nonhex_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    original = _readback_report(artifacts)
    cases = (
        (
            ("artifact_bindings", 0, "byte_sha256"),
            "z" * 64,
            r"readback binding\[0\] byte SHA must be lowercase SHA-256",
        ),
        (
            ("artifact_bindings", 0, "semantic_sha256"),
            "A" * 64,
            r"readback binding\[0\] semantic SHA must be lowercase SHA-256",
        ),
        (
            ("cycle_root_sha256",),
            "z" * 64,
            "readback cycle root SHA must be lowercase SHA-256",
        ),
    )
    for path, replacement, reason in cases:
        report = copy.deepcopy(original)
        target: Any = report
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = replacement
        report.pop("artifact_semantic_sha256")
        report["artifact_semantic_sha256"] = core.semantic_sha256_v4_4(report)
        with pytest.raises(
            subject.FactorGovernanceCandidatePreregistrationBundleV4_4Error,
            match=reason,
        ):
            subject.validate_candidate_preregistration_readback_v4_4(report)

    bad_self_hash = copy.deepcopy(original)
    bad_self_hash["artifact_semantic_sha256"] = "z" * 64
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_4Error,
        match="artifact semantic SHA must be lowercase SHA-256",
    ):
        subject.validate_candidate_preregistration_readback_v4_4(bad_self_hash)


def test_readback_builder_rejects_nonhex_artifact_semantic_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    changed = copy.deepcopy(artifacts)
    filename = subject.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4
    changed[filename]["artifact_semantic_sha256"] = "z" * 64
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_4Error,
        match=f"{filename} artifact semantic SHA must be lowercase SHA-256",
    ):
        getattr(subject, "_build_readback_report")(
            run_id=changed[subject.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"],
            artifacts=changed,
            artifact_bindings=_base_readback_bindings(changed),
        )


def test_standalone_readback_rejects_each_binding_metadata_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    original = _readback_report(artifacts)
    cases = (
        ("size_bytes", 0, "owner 0600/nlink1 with exact hashes"),
        ("mode", 0o640, "owner 0600/nlink1 with exact hashes"),
        ("uid", os.getuid() + 1, "owner 0600/nlink1 with exact hashes"),
        ("nlink", 2, "owner 0600/nlink1 with exact hashes"),
    )
    for field, replacement, reason in cases:
        report = copy.deepcopy(original)
        report["artifact_bindings"][0][field] = replacement
        report.pop("artifact_semantic_sha256")
        report["artifact_semantic_sha256"] = core.semantic_sha256_v4_4(report)
        with pytest.raises(
            subject.FactorGovernanceCandidatePreregistrationBundleV4_4Error,
            match=reason,
        ):
            subject.validate_candidate_preregistration_readback_v4_4(report)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("reorder", "bundle input inventory/order mismatch"),
        ("missing", "bundle input inventory/order mismatch"),
        ("extra", "bundle input inventory/order mismatch"),
        ("tamper", "cross-artifact graph mismatch"),
    ),
)
def test_inventory_and_graph_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: str,
) -> None:
    artifacts = _artifacts(tmp_path / mutation, monkeypatch)
    if mutation == "reorder":
        changed = dict(reversed(tuple(artifacts.items())))
    elif mutation == "missing":
        changed = copy.deepcopy(artifacts)
        changed.pop(subject.CYCLE_ROOT_FILENAME_V4_4)
    elif mutation == "extra":
        changed = copy.deepcopy(artifacts)
        changed["extra.v4_4.json"] = {}
    else:
        changed = copy.deepcopy(artifacts)
        changed[subject.EXPANDED_CANDIDATE_SELECTION_FILENAME_V4_4]["candidates"][0][
            "initial_weight"
        ] = 1
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_4Error,
        match=reason,
    ):
        subject.validate_candidate_preregistration_bundle_inputs_v4_4(changed)


def test_original_raw_bytes_and_descriptor_tamper_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path, monkeypatch)
    bindings = subject.raw_input_bindings_v4_4(artifacts)
    raw = _raw_collected(artifacts)
    full_raw = {
        filename: core.canonical_file_bytes_v4_4(artifacts[filename])
        for filename in subject.INPUT_FILENAMES_V4_4
    }
    assert subject.validate_candidate_preregistration_bundle_inputs_v4_4(
        artifacts,
        raw_input_bindings=bindings,
        collected_raw_bytes=full_raw,
    ) == artifacts

    bad_bindings = copy.deepcopy(bindings)
    bad_bindings[0]["byte_sha256"] = "0" * 64
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_4Error,
        match="raw input binding mismatch",
    ):
        subject.validate_candidate_preregistration_bundle_inputs_v4_4(
            artifacts, raw_input_bindings=bad_bindings, collected_raw_bytes=raw
        )
    bad_raw = dict(raw)
    bad_raw[subject.COLLECTED_RAW_FILENAMES_V4_4[0]] += b" "
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_4Error,
        match="collected raw bytes are not exact canonical bytes",
    ):
        subject.validate_candidate_preregistration_bundle_inputs_v4_4(
            artifacts, raw_input_bindings=bindings, collected_raw_bytes=bad_raw
        )

    binding_inventory_cases = (
        list(reversed(copy.deepcopy(bindings))),
        copy.deepcopy(bindings[:-1]),
        [*copy.deepcopy(bindings), copy.deepcopy(bindings[-1])],
    )
    for changed_bindings in binding_inventory_cases:
        with pytest.raises(
            subject.FactorGovernanceCandidatePreregistrationBundleV4_4Error,
            match="raw_input_bindings exact inventory mismatch|raw input binding mismatch",
        ):
            subject.validate_candidate_preregistration_bundle_inputs_v4_4(
                artifacts,
                raw_input_bindings=changed_bindings,
                collected_raw_bytes=raw,
            )

    raw_inventory_cases = []
    reversed_raw = dict(reversed(tuple(raw.items())))
    raw_inventory_cases.append(reversed_raw)
    missing_raw = dict(raw)
    missing_raw.pop(subject.COLLECTED_RAW_FILENAMES_V4_4[-1])
    raw_inventory_cases.append(missing_raw)
    extra_raw = dict(raw)
    extra_raw["extra.v4_4.json"] = b"{}\n"
    raw_inventory_cases.append(extra_raw)
    for changed_raw in raw_inventory_cases:
        with pytest.raises(
            subject.FactorGovernanceCandidatePreregistrationBundleV4_4Error,
            match="collected_raw_bytes inventory/order mismatch",
        ):
            subject.validate_candidate_preregistration_bundle_inputs_v4_4(
                artifacts,
                raw_input_bindings=bindings,
                collected_raw_bytes=changed_raw,
            )


def test_real_temp_publish_readback_permissions_no_clobber_and_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _portable_publication(monkeypatch)
    artifacts = _artifacts(tmp_path / "inputs", monkeypatch)
    root = _private_root(tmp_path / "published")
    bindings = subject.raw_input_bindings_v4_4(artifacts)
    raw = _raw_collected(artifacts)
    revalidation_calls: list[str] = []

    def revalidate() -> None:
        revalidation_calls.append("called")

    result = subject.publish_candidate_preregistration_bundle_v4_4(
        private_root=root,
        artifacts=artifacts,
        raw_input_bindings=bindings,
        collected_raw_bytes=raw,
        revalidate_inputs=revalidate,
    )
    assert revalidation_calls == ["called"]
    bundle_path = Path(result["bundle_path"])
    assert len(tuple(bundle_path.iterdir())) == 27
    assert (bundle_path.stat().st_mode & 0o777) == 0o700
    for path in bundle_path.iterdir():
        metadata = path.stat()
        assert (metadata.st_mode & 0o777) == 0o600
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1
    reopened = subject.readback_candidate_preregistration_bundle_v4_4(bundle_path)
    assert reopened["accepted"] is True
    assert reopened["readback_report"]["authority"] == core.AUTHORITY_FLAGS
    report_bindings = reopened["readback_report"]["artifact_bindings"]
    assert [row["filename"] for row in report_bindings] == list(
        subject.INPUT_FILENAMES_V4_4
    )
    for filename, row in zip(
        subject.INPUT_FILENAMES_V4_4, report_bindings, strict=True
    ):
        raw_bytes = core.canonical_file_bytes_v4_4(artifacts[filename])
        semantic_sha = artifacts[filename].get(
            "artifact_semantic_sha256",
            artifacts[filename].get("state_semantic_sha256"),
        )
        assert row == {
            "filename": filename,
            "byte_sha256": _sha(raw_bytes),
            "semantic_sha256": semantic_sha,
            "size_bytes": len(raw_bytes),
            "mode": 0o600,
            "uid": os.getuid(),
            "nlink": 1,
        }
    first_hashes = _file_hashes(bundle_path)
    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="canonical private bundle already exists",
    ):
        subject.publish_candidate_preregistration_bundle_v4_4(
            private_root=root,
            artifacts=artifacts,
            raw_input_bindings=bindings,
            collected_raw_bytes=raw,
            revalidate_inputs=revalidate,
        )
    assert revalidation_calls == ["called"]
    assert _file_hashes(bundle_path) == first_hashes
    assert {item.name for item in root.iterdir()} == {
        private_io.LOCK_FILENAME,
        artifacts[subject.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"],
    }
    _assert_no_staging_residue(
        root, artifacts[subject.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"]
    )

    fault_root = _private_root(tmp_path / "fault")

    def fault(point: str) -> None:
        if point == "precommit:root-fsync:before":
            raise RuntimeError("injected fault")

    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="injected test fault at precommit:root-fsync:before",
    ):
        subject.publish_candidate_preregistration_bundle_v4_4(
            private_root=fault_root,
            artifacts=artifacts,
            raw_input_bindings=bindings,
            collected_raw_bytes=raw,
            revalidate_inputs=lambda: None,
            _test_fault_hook=fault,
        )
    cycle_id = artifacts[subject.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"]
    assert not (fault_root / cycle_id).exists()
    _assert_no_staging_residue(fault_root, cycle_id)
    assert {item.name for item in fault_root.iterdir()} == {
        private_io.LOCK_FILENAME,
        private_io.QUARANTINE_DIRECTORY,
    }


def test_locked_revalidation_failure_rejects_without_canonical_or_staging_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _portable_publication(monkeypatch)
    artifacts = _artifacts(tmp_path / "inputs", monkeypatch)
    root = _private_root(tmp_path / "revalidation")
    bindings = subject.raw_input_bindings_v4_4(artifacts)
    raw = _raw_collected(artifacts)

    def reject_stale_inputs() -> None:
        raise RuntimeError("stale bound inputs")

    with pytest.raises(
        private_io.FactorGovernancePrivateBundleIOError,
        match="input revalidation failed: stale bound inputs",
    ):
        subject.publish_candidate_preregistration_bundle_v4_4(
            private_root=root,
            artifacts=artifacts,
            raw_input_bindings=bindings,
            collected_raw_bytes=raw,
            revalidate_inputs=reject_stale_inputs,
        )
    cycle_id = artifacts[subject.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"]
    assert not (root / cycle_id).exists()
    _assert_no_staging_residue(root, cycle_id)
    assert {item.name for item in root.iterdir()} == {
        private_io.LOCK_FILENAME,
        private_io.QUARANTINE_DIRECTORY,
    }


def test_concurrent_publishers_produce_one_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _portable_publication(monkeypatch)
    artifacts = _artifacts(tmp_path / "inputs", monkeypatch)
    root = _private_root(tmp_path / "concurrent")
    bindings = subject.raw_input_bindings_v4_4(artifacts)
    raw = _raw_collected(artifacts)
    outcomes: list[str] = []

    def worker() -> None:
        try:
            subject.publish_candidate_preregistration_bundle_v4_4(
                private_root=root,
                artifacts=artifacts,
                raw_input_bindings=bindings,
                collected_raw_bytes=raw,
                revalidate_inputs=lambda: None,
            )
        except private_io.FactorGovernancePrivateBundleIOError as exc:
            assert "canonical private bundle already exists" in str(exc)
            outcomes.append("rejected")
        else:
            outcomes.append("accepted")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert sorted(outcomes) == ["accepted", "rejected"]
    bundle = root / artifacts[subject.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"]
    assert len(tuple(bundle.iterdir())) == 27
    assert subject.readback_candidate_preregistration_bundle_v4_4(bundle)[
        "accepted"
    ] is True
    _assert_no_staging_residue(
        root, artifacts[subject.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"]
    )
    assert {item.name for item in root.iterdir()} == {
        private_io.LOCK_FILENAME,
        artifacts[subject.CYCLE_ROOT_FILENAME_V4_4]["cycle_id"],
    }

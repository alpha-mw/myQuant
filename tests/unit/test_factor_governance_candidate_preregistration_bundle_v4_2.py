from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from quant_investor.factors.governance_source_readback_v4_1 import (
    BoundCutoffInputsV4_1,
    INPUT_BINDING_SCHEMA_VERSION,
    SOURCE_USE_PROHIBITED,
)
from quant_investor.factors import (
    governance_candidate_preregistration_bundle_v4_2 as subject,
)
from quant_investor.factors import governance_candidate_preregistration_v4_2 as prereg
from quant_investor.factors import governance_private_bundle_io as private_io


def _raw_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _record(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "absolute_path": str(path),
        "size_bytes": len(raw),
        "sha256": _sha(raw),
    }


def _bound_inputs(tmp_path: Path) -> tuple[BoundCutoffInputsV4_1, bytes, bytes]:
    snapshot_id = "20260720T010203Z"
    cutoff = "2026-07-18"
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
    pit_root = (
        pointer_path.parent
        / "reference"
        / "_generations"
        / "pit-20260720-test"
    )
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
            "latest_trade_date": "20260718",
            "latest_available_trade_date": "20260718",
            "latest_complete_trade_date": "20260718",
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
                "coverage_trade_date": "20260718",
                "latest_available_trade_date": "20260718",
                "latest_complete_trade_date": "20260718",
                "blocking_incomplete_count": 0,
                "classification_sets_disjoint": True,
                "true_missing_symbols": [],
                "pit_generation_id": "pit-20260720-test",
                "pit_generation_manifest_path": str(pit_manifest_path),
                "pit_generation_manifest_sha256": _sha(pit_manifest_raw),
                "pit_membership_path": str(pit_membership_path),
                "pit_membership_sha256": _sha(pit_membership_raw),
            },
        }
    )
    for path, raw in (
        (manifest_path, manifest_raw),
        (pit_manifest_path, pit_manifest_raw),
        (pit_membership_path, pit_membership_raw),
        (table_root / "year=2026" / "month=07" / "part.parquet", b"table-bytes"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    table_rows = [
        {
            "relative_path": "year=2026/month=07/part.parquet",
            "size_bytes": 11,
            "sha256": _sha(b"table-bytes"),
            "hard_link_count": 1,
            "dataset_member": True,
        }
    ]
    table_sha = hashlib.sha256(
        prereg.canonical_json_bytes_v4_2(table_rows)
    ).hexdigest()
    sessions = ["2026-07-17", cutoff]
    calendar_base = {
        "analysis_start": sessions[0],
        "cutoff_date": cutoff,
        "open_session_count": len(sessions),
        "open_sessions": sessions,
    }
    calendar = {
        **calendar_base,
        "semantic_sha256": hashlib.sha256(
            prereg.canonical_json_bytes_v4_2(calendar_base)
        ).hexdigest(),
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
            "generation_id": "pit-20260720-test",
            "manifest": _record(pit_manifest_path, pit_manifest_raw),
            "membership": _record(pit_membership_path, pit_membership_raw),
            "row_count": 2,
            "historical_alias_table_evidence": [],
        },
        "table": {
            "absolute_root": str(table_root),
            "regular_file_count": 1,
            "parquet_file_count": 1,
            "inventory_sha256": table_sha,
            "parquet_inventory": table_rows,
            "bound_symbol_inventory": {
                "symbol_count": len(symbols),
                "symbols_newline_sha256": scope_sha,
                "noncanonical_symbol_count": 0,
            },
        },
        "calendar": calendar,
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


def _source(tmp_path: Path) -> dict[str, Any]:
    bound, pointer, components = _bound_inputs(tmp_path)
    return subject.build_strict_full_a_source_binding_v4_2(
        bound_inputs=bound,
        latest_pointer_raw=pointer,
        components_raw=components,
    )


def test_strict_source_embeds_raw_evidence_and_historical_reopen_only(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)

    assert source["backend_binding_schema_version"] == INPUT_BINDING_SCHEMA_VERSION
    assert source["historical_validation_reads_current_pointer"] is False
    assert source["historical_validation_reads_current_components"] is False
    assert set(source["immutable_reopen_descriptor"]) == {
        "snapshot_manifest",
        "pit_generation_manifest",
        "pit_membership",
        "table_inventory",
    }
    assert source["serving_historical_semantics"] == {
        "derived_serving_root": source["backend_binding"]["eligibility_boundary"][
            "serving_inventory"
        ]["absolute_root"],
        "symbol_count": 3,
        "was_scanned": False,
    }
    assert source["components_raw_evidence"]["normalized_symbols"] == source[
        "component_symbols"
    ]
    assert source["components_raw_evidence"]["symbol_count"] == 2
    assert source["components_raw_evidence"]["full_a_scope_sha256"] == source[
        "full_a_scope_sha256"
    ]
    assert subject.validate_historical_strict_full_a_source_binding_v4_2(source) == source


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("latest_pointer_raw_evidence", "byte_sha256"), "0" * 64, "byte_sha256 mismatch"),
        (
            ("components_raw_evidence", "raw_base64"),
            base64.b64encode(b"{}").decode(),
            "size_bytes mismatch",
        ),
        (("component_symbols",), ["600000.SH", "000001.SZ"], "component_symbols"),
        (("serving_historical_semantics", "was_scanned"), True, "serving historical"),
    ],
)
def test_strict_source_rejects_raw_component_and_serving_tamper(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    source = copy.deepcopy(_source(tmp_path))
    target: Any = source
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_2Error,
        match=message,
    ):
        subject.validate_strict_full_a_source_binding_v4_2(source)


def test_strict_source_rejects_cross_artifact_raw_substitution(tmp_path: Path) -> None:
    first = _source(tmp_path / "first")
    second = _source(tmp_path / "second")
    substituted = copy.deepcopy(first)
    substituted["latest_pointer_raw_evidence"] = second["latest_pointer_raw_evidence"]
    # Raw bytes are deliberately identical, but the absolute backend identities
    # differ.  Substitution of the whole source wrapper must still be detected
    # by its semantic binding in the enclosing graph; changing one backend leaf
    # here also breaks this artifact's self seal.
    substituted["backend_binding"]["latest_pointer"] = second["backend_binding"][
        "latest_pointer"
    ]
    with pytest.raises(subject.FactorGovernanceCandidatePreregistrationBundleV4_2Error):
        subject.validate_strict_full_a_source_binding_v4_2(substituted)


def _code_root(tmp_path: Path) -> Path:
    for index, relative in enumerate(subject.CODE_BINDING_PATHS_V4_2, start=1):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{index}\n".encode())
    return tmp_path


def test_code_binding_set_uses_exact_order_and_revalidates_stable_bytes(
    tmp_path: Path,
) -> None:
    root = _code_root(tmp_path)
    artifact = subject.build_code_binding_set_v4_2(repository_root=root)

    assert [row["relative_path"] for row in artifact["ordered_bindings"]] == list(
        subject.CODE_BINDING_PATHS_V4_2
    )
    assert subject.revalidate_code_binding_set_v4_2(
        repository_root=root, value=artifact
    ) == artifact

    (root / subject.CODE_BINDING_PATHS_V4_2[0]).write_bytes(b"drift\n")
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_2Error,
        match="drifted",
    ):
        subject.revalidate_code_binding_set_v4_2(
            repository_root=root, value=artifact
        )


def test_cycle_id_is_exact_deterministic_identity() -> None:
    assert (
        subject.deterministic_cycle_id_v4_2(
            cutoff="2026-07-18", snapshot_id="20260720T010203Z"
        )
        == "cn_full_a_v4_2_20260718_20260720T010203Z"
    )
    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_2Error,
        match="real canonical UTC timestamp",
    ):
        subject.deterministic_cycle_id_v4_2(
            cutoff="2026-07-18", snapshot_id="20269999T010203Z"
        )


def test_contract_inventory_and_root_are_fixed() -> None:
    contract = subject.candidate_preregistration_bundle_contract_v4_2()
    assert contract.root_suffix == (
        "reports",
        "factor_governance",
        "private",
        "v4_2_candidate_preregistration",
    )
    assert contract.input_filenames == subject.INPUT_FILENAMES_V4_2
    assert len(contract.input_filenames) == 14
    assert contract.readback_report_filename == subject.READBACK_REPORT_FILENAME_V4_2


def _bundle_artifacts(tmp_path: Path) -> dict[str, dict[str, Any]]:
    aquant = prereg.build_aquant_receipt_v4_2()
    myquant = prereg.build_myquant_receipt_v4_2()
    operators = prereg.build_operator_semantics_v4_2()
    comparison = prereg.build_comparison_catalog_receipt_v4_2(
        catalog_id="synthetic-comparison-v4",
        catalog_byte_sha256="1" * 64,
        catalog_semantic_sha256="2" * 64,
        primitive_count=1,
        definition_identity_inventory=[
            {"name": "legacy_synthetic", "definition_identity_sha256": "3" * 64}
        ],
    )
    selection = prereg.build_selection_spec_v4_2(
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
    )
    source = _source(tmp_path / "source-inputs")
    code = subject.build_code_binding_set_v4_2(
        repository_root=_code_root(tmp_path / "code-root")
    )
    return subject.build_candidate_preregistration_bundle_artifacts_v4_2(
        aquant_receipt=aquant,
        myquant_receipt=myquant,
        operator_semantics=operators,
        comparison_catalog_receipt=comparison,
        candidate_selection_spec=selection,
        strict_full_a_source_binding=source,
        code_binding_set=code,
    )


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
    root = tmp_path.joinpath(*subject.ROOT_SUFFIX_V4_2)
    root.mkdir(parents=True)
    root.chmod(0o700)
    return root


def test_full_graph_is_deterministic_and_pure_validator_is_bound(
    tmp_path: Path,
) -> None:
    artifacts = _bundle_artifacts(tmp_path)

    assert tuple(artifacts) == subject.INPUT_FILENAMES_V4_2
    assert artifacts[subject.PRECOMMITTED_STATE_FILENAME_V4_2]["state"] == "PRECOMMITTED"
    assert artifacts[subject.DISCOVERY_STATE_FILENAME_V4_2]["state"] == "DISCOVERY"
    orchestration = artifacts[
        subject.PREREG_DISCOVERY_ORCHESTRATION_FILENAME_V4_2
    ]
    assert orchestration["persisted_state_sequence"] == ["PRECOMMITTED", "DISCOVERY"]
    assert orchestration["precommitted_state_role"] == "INTRA_BUNDLE_LINEAGE_ONLY"
    assert orchestration["discovery_state_role"] == "FINAL_CURRENT"
    assert orchestration["external_state_pointer_mutation"] is False


def test_publish_and_historical_readback_are_exact_private_precommit_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _portable_private_publication(monkeypatch)
    artifacts = _bundle_artifacts(tmp_path / "inputs")
    root = _private_root(tmp_path)

    result = subject.publish_candidate_preregistration_bundle_v4_2(
        private_root=root,
        artifacts=artifacts,
        revalidate_inputs=lambda: None,
    )

    assert result["accepted"] is True
    report = result["readback_report"]
    assert report["publication_evidence_scope"] == "PRECOMMIT_INTENT_ONLY"
    assert report["required_commit_primitive"] == "renameatx_np(RENAME_EXCL)"
    assert report["commit_success_claimed"] is False
    assert report["no_clobber_success_claimed"] is False
    assert report["fsync_success_claimed"] is False
    assert report["durability_success_claimed"] is False
    assert report["state_contract"]["sole_final_current_state"] == "DISCOVERY"
    assert report["state_contract"]["external_pointer_mutation"] is False
    bundle = Path(result["bundle_path"])
    assert bundle.name == artifacts[subject.CYCLE_ROOT_FILENAME_V4_2]["cycle_id"]
    assert sorted(path.name for path in bundle.iterdir()) == sorted(
        (*subject.INPUT_FILENAMES_V4_2, subject.READBACK_REPORT_FILENAME_V4_2)
    )
    for path in bundle.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.stat().st_nlink == 1

    # A current pointer may later move.  Historical readback is intentionally
    # anchored only to the embedded raw bytes and immutable reopen descriptor.
    source = artifacts[subject.STRICT_FULL_A_SOURCE_BINDING_FILENAME_V4_2]
    live_pointer = Path(source["backend_binding"]["latest_pointer"]["absolute_path"])
    live_pointer.parent.mkdir(parents=True, exist_ok=True)
    live_pointer.write_bytes(b'{"snapshot_id":"later"}')
    reread = subject.readback_candidate_preregistration_bundle_v4_2(bundle)
    assert reread["artifacts"] == result["artifacts"]
    assert reread["readback_report"] == result["readback_report"]
    immutable = reread["immutable_source_readback"]
    assert immutable["accepted"] is True
    assert immutable["validation_scope"] == "RECORDED_IMMUTABLE_REOPEN_ONLY"
    assert immutable["current_pointer_read"] is False
    assert immutable["current_components_read"] is False
    assert immutable["serving_tree_read"] is False


def test_historical_immutable_reopen_fails_on_recorded_table_drift(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    assert subject.revalidate_recorded_immutable_source_v4_2(source)["accepted"] is True
    table_root = Path(
        source["immutable_reopen_descriptor"]["table_inventory"]["absolute_root"]
    )
    (table_root / "year=2026" / "month=07" / "part.parquet").write_bytes(
        b"changed-table"
    )

    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_2Error,
        match="table inventory mismatch",
    ):
        subject.revalidate_recorded_immutable_source_v4_2(source)


@pytest.mark.parametrize(
    "descriptor_key",
    ["snapshot_manifest", "pit_generation_manifest", "pit_membership"],
)
def test_historical_immutable_reopen_fails_on_recorded_file_drift(
    tmp_path: Path, descriptor_key: str
) -> None:
    source = _source(tmp_path)
    record = source["immutable_reopen_descriptor"][descriptor_key]
    Path(record["absolute_path"]).write_bytes(b"immutable-drift")

    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_2Error,
        match="recorded size/SHA mismatch",
    ):
        subject.revalidate_recorded_immutable_source_v4_2(source)


def test_full_graph_rejects_cross_cycle_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _portable_private_publication(monkeypatch)
    first = _bundle_artifacts(tmp_path / "first")
    second = _bundle_artifacts(tmp_path / "second")
    root = _private_root(tmp_path / "published")
    published = subject.publish_candidate_preregistration_bundle_v4_2(
        private_root=root,
        artifacts=first,
        revalidate_inputs=lambda: None,
    )
    complete = copy.deepcopy(published["artifacts"])
    complete[subject.CYCLE_ROOT_FILENAME_V4_2] = second[
        subject.CYCLE_ROOT_FILENAME_V4_2
    ]

    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_2Error,
        match="cross-artifact graph mismatch|cycle identity mismatch",
    ):
        subject.validate_candidate_preregistration_bundle_artifacts_v4_2(complete)


def test_complete_validator_rejects_resealed_report_size_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _portable_private_publication(monkeypatch)
    artifacts = _bundle_artifacts(tmp_path / "inputs")
    root = _private_root(tmp_path)
    published = subject.publish_candidate_preregistration_bundle_v4_2(
        private_root=root,
        artifacts=artifacts,
        revalidate_inputs=lambda: None,
    )
    complete = copy.deepcopy(published["artifacts"])
    report = complete[subject.READBACK_REPORT_FILENAME_V4_2]
    report["artifact_bindings"][0]["size_bytes"] += 1
    report["artifact_semantic_sha256"] = prereg.semantic_sha256_v4_2(
        {
            key: value
            for key, value in report.items()
            if key != "artifact_semantic_sha256"
        }
    )

    with pytest.raises(
        subject.FactorGovernanceCandidatePreregistrationBundleV4_2Error,
        match="readback byte binding mismatch",
    ):
        subject.validate_candidate_preregistration_bundle_artifacts_v4_2(complete)


def test_publish_no_clobber_is_deterministic_cycle_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _portable_private_publication(monkeypatch)
    artifacts = _bundle_artifacts(tmp_path / "inputs")
    root = _private_root(tmp_path)
    subject.publish_candidate_preregistration_bundle_v4_2(
        private_root=root,
        artifacts=artifacts,
        revalidate_inputs=lambda: None,
    )

    with pytest.raises(private_io.FactorGovernancePrivateBundleIOError, match="already exists"):
        subject.publish_candidate_preregistration_bundle_v4_2(
            private_root=root,
            artifacts=artifacts,
            revalidate_inputs=lambda: None,
        )

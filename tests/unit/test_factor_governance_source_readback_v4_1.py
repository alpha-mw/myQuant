from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_investor.factors.governance_source_readback_v4_1 as readback
from quant_investor.codex_review.storage import canonical_json_bytes
from quant_investor.factors.governance_cycle_state_v4_1 import (
    build_genesis_cycle_state_v4_1,
)
from quant_investor.factors.governance_source_v4_1 import (
    build_design_source_node_v4_1,
)
from quant_investor.market.pit_universe import PITUniverseRecord, PITUniverseStore


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(paths: dict[str, object], key: str) -> Path:
    value = paths[key]
    assert isinstance(value, Path)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _fixture(tmp_path: Path) -> dict[str, object]:
    market_root = tmp_path / "data" / "parquet" / "cn"
    snapshot_id = "20260717T172132Z"
    table_root = market_root / "_snapshots" / snapshot_id / "table" / "bars"
    serving_root = market_root / "_snapshots" / snapshot_id / "serving" / "bars"
    serving_root.mkdir(parents=True)
    table_part = table_root / "year=2024" / "month=01" / "part.parquet"
    table_part.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
                "trade_date": ["20260716", "20260716", "20260717"],
                "close": [1.0, 2.0, 1.1],
            }
        ),
        table_part,
    )

    components_path = tmp_path / "data" / "cn_universe" / "components.json"
    symbols = ["000001.SZ", "000002.SZ"]
    _write_json(components_path, {"full_a": symbols, "stats": {"full_a": 2}})
    scope_sha = hashlib.sha256("\n".join(symbols).encode()).hexdigest()

    generation_id = "pit-fixture-v41"
    pit_root = market_root / "reference" / "_generations" / generation_id
    pit_path = pit_root / "stock_basic_membership.parquet"
    pit_path.parent.mkdir(parents=True)
    rows = [
        {
            "schema_version": "cn_pit_universe.v1",
            "symbol": symbol,
            "name": symbol,
            "area": "",
            "industry": "fixture",
            "board_market": "main",
            "source_list_status": "L",
            "list_date": "20200101",
            "delist_date": "",
            "effective_from": "20200101",
            "effective_to": "",
            "observed_at": "2024-01-03T00:00:00Z",
            "source": "fixture",
            "source_run_id": "fixture-run",
            "raw_payload_hash": hashlib.sha256(symbol.encode()).hexdigest()[:16],
            "membership_quality": "ok",
        }
        for symbol in symbols
    ]
    pq.write_table(pa.Table.from_pylist(rows), pit_path)
    normalized_records = [PITUniverseRecord.from_dict(row) for row in rows]
    pit_manifest_path = pit_root / "manifest.json"
    pit_manifest = {
        "schema_version": "cn_pit_universe_manifest.v1",
        "membership_schema_version": "cn_pit_universe.v1",
        "generation_id": generation_id,
        "canonical_path": str(pit_path),
        "canonical_sha256": _sha(pit_path),
        "row_count": 2,
        "raw_row_count": 2,
        "membership_quality_counts": {"ok": 2},
        "records_sha256": PITUniverseStore._records_sha256(normalized_records),
        "status_counts": {"L": 2},
    }
    _write_json(pit_manifest_path, pit_manifest)

    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "coverage_ratio": 1.0,
        "coverage_complete_count": 2,
        "expected_scope_count": 2,
        "categories_checked": ["full_a"],
        "coverage_trade_date": "20260717",
        "latest_available_trade_date": "20260717",
        "latest_complete_trade_date": "20260717",
        "blocking_incomplete_count": 0,
        "classification_sets_disjoint": True,
        "true_missing_symbols": [],
        "expected_scope_sha256": scope_sha,
        "pit_generation_id": generation_id,
        "pit_generation_manifest_path": str(pit_manifest_path),
        "pit_generation_manifest_sha256": _sha(pit_manifest_path),
        "pit_membership_path": str(pit_path),
        "pit_membership_sha256": _sha(pit_path),
    }
    manifest_path = market_root / "_snapshots" / f"{snapshot_id}.json"
    manifest = {
        "snapshot_id": snapshot_id,
        "market": "CN",
        "status": "OK",
        "readback_validated": True,
        "blockers": [],
        "manifest_path": str(manifest_path),
        "table_root": str(table_root),
        "derived_serving_root": str(serving_root),
        "latest_available_trade_date": "20260717",
        "latest_complete_trade_date": "20260717",
        "symbol_count": 5728,
        "coverage": coverage,
    }
    _write_json(manifest_path, manifest)
    pointer_path = market_root / "_latest.json"
    pointer = {
        "snapshot_id": snapshot_id,
        "status": "OK",
        "blockers": [],
        "manifest_path": str(manifest_path),
        "table_root": str(table_root),
        "derived_serving_root": str(serving_root),
        "latest_available_trade_date": "20260717",
        "latest_complete_trade_date": "20260717",
        "coverage": coverage,
    }
    _write_json(pointer_path, pointer)
    paths: dict[str, object] = {
        "pointer": pointer_path,
        "manifest": manifest_path,
        "components": components_path,
        "pit_manifest": pit_manifest_path,
        "pit": pit_path,
        "table_root": table_root,
        "table_part": table_part,
        "scope_sha": scope_sha,
        "snapshot_id": snapshot_id,
    }
    return paths


def _bind(paths: dict[str, object], **overrides: object) -> readback.BoundCutoffInputsV4_1:
    kwargs: dict[str, object] = {
        "latest_pointer_path": paths["pointer"],
        "expected_latest_pointer_sha256": _sha(paths["pointer"]),  # type: ignore[arg-type]
        "snapshot_manifest_path": paths["manifest"],
        "expected_snapshot_manifest_sha256": _sha(paths["manifest"]),  # type: ignore[arg-type]
        "components_path": paths["components"],
        "expected_components_sha256": _sha(paths["components"]),  # type: ignore[arg-type]
        "expected_full_a_semantic_sha256": paths["scope_sha"],
        "pit_generation_manifest_path": paths["pit_manifest"],
        "expected_pit_generation_manifest_sha256": _sha(paths["pit_manifest"]),  # type: ignore[arg-type]
        "pit_membership_path": paths["pit"],
        "expected_pit_membership_sha256": _sha(paths["pit"]),  # type: ignore[arg-type]
        "table_root": paths["table_root"],
        "snapshot_id": paths["snapshot_id"],
        "analysis_start": "2026-07-16",
        "cutoff_date": "2026-07-17",
        "expected_full_a_count": 2,
        "expected_serving_inventory_count": 5728,
    }
    kwargs.update(overrides)
    return readback.bind_explicit_cutoff_inputs_v4_1(**kwargs)  # type: ignore[arg-type]


def test_explicit_binding_is_strict_and_serving_is_diagnostic_only(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    bound = _bind(paths)

    assert bound.calendar_sessions == ("2026-07-16", "2026-07-17")
    assert bound.component_symbols == ("000001.SZ", "000002.SZ")
    assert len(bound.pit_records) == 2
    assert bound.bound_table_symbol_row_counts == (
        ("000001.SZ", 2),
        ("000002.SZ", 1),
    )
    assert bound.binding["table"]["parquet_file_count"] == 1
    serving = bound.binding["eligibility_boundary"]["serving_inventory"]
    assert serving == {
        "absolute_root": str(
            _path(paths, "pointer").parent
            / "_snapshots"
            / str(paths["snapshot_id"])
            / "serving"
            / "bars"
        ),
        "symbol_count": 5728,
        "use": readback.SOURCE_USE_PROHIBITED,
        "was_scanned": False,
    }
    assert bound.binding["side_effects"] == {
        "registry": False,
        "wal": False,
        "budget": False,
        "apply": False,
        "broker": False,
        "order": False,
        "trade": False,
        "network": False,
    }


def test_binding_rejects_relative_path_and_sha_tamper(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(
        readback.FactorGovernanceSourceReadbackV4_1Error,
        match="explicit absolute path",
    ):
        _bind(paths, latest_pointer_path=Path("data/parquet/cn/_latest.json"))
    with pytest.raises(
        readback.FactorGovernanceSourceReadbackV4_1Error,
        match="latest pointer SHA-256 mismatch",
    ):
        _bind(paths, expected_latest_pointer_sha256="0" * 64)


def test_binding_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pointer = _path(paths, "pointer")
    pointer.write_bytes(b'{"snapshot_id":"a","snapshot_id":"b"}\n')
    with pytest.raises(
        readback.FactorGovernanceSourceReadbackV4_1Error,
        match="duplicate JSON object key",
    ):
        _bind(paths)


def test_binding_rejects_symlinked_input(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    original = _path(paths, "components")
    linked = original.with_name("linked-components.json")
    linked.symlink_to(original)
    with pytest.raises(
        readback.FactorGovernanceSourceReadbackV4_1Error,
        match="regular non-symlink",
    ):
        _bind(
            paths,
            components_path=linked,
            expected_components_sha256=_sha(original),
        )


def test_binding_rejects_table_tree_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    original = readback._extract_calendar_and_symbol_counts

    def mutate_after_read(
        *args: object, **kwargs: object
    ) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        with _path(paths, "table_part").open("ab") as handle:
            handle.write(b"drift")
        return result

    monkeypatch.setattr(
        readback, "_extract_calendar_and_symbol_counts", mutate_after_read
    )
    with pytest.raises(
        readback.FactorGovernanceSourceReadbackV4_1Error,
        match="changed during calendar readback",
    ):
        _bind(paths)


def test_table_inventory_hashes_hidden_non_dataset_bytes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    hidden = _path(paths, "table_root") / ".part.parquet.tmp-interrupted"
    hidden.write_bytes(b"immutable interrupted-write evidence")

    bound = _bind(paths)

    assert bound.binding["table"]["regular_file_count"] == 2
    assert bound.binding["table"]["parquet_file_count"] == 1
    inventory = {
        row["relative_path"]: row
        for row in bound.binding["table"]["parquet_inventory"]
    }
    assert inventory[hidden.name]["dataset_member"] is False
    assert inventory[hidden.name]["sha256"] == _sha(hidden)


def test_authoritative_pit_binding_rejects_column_order_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    pit_path = _path(paths, "pit")
    table = pq.read_table(pit_path)
    pq.write_table(table.select(list(reversed(table.column_names))), pit_path)
    manifest_path = _path(paths, "pit_manifest")
    manifest = json.loads(manifest_path.read_text())
    manifest["canonical_sha256"] = _sha(pit_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        readback.FactorGovernanceSourceReadbackV4_1Error,
        match="pit_generation_canonical_schema_invalid",
    ):
        readback._pit_generation_records(
            pit_generation_manifest_path=manifest_path,
            expected_pit_generation_manifest_sha256=_sha(manifest_path),
        )


def test_precommitted_publication_is_private_read_back_and_cas_exact_once(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    bound = _bind(paths)
    private_root = tmp_path / "private"
    source_binding_sha = "a" * 64
    design = build_design_source_node_v4_1(
        cycle_id="cycle-v41",
        pit_records=list(bound.pit_records),
        component_symbols=list(bound.component_symbols),
        calendar_sessions=list(bound.calendar_sessions),
        market_binding_sha256=readback.binding_semantic_sha256_v4_1(
            bound.binding
        ),
        source_binding_sha256=source_binding_sha,
        expected_component_count=2,
    )
    source_node = readback.build_cutoff_source_node_v4_1(
        cycle_id="cycle-v41",
        input_binding=bound.binding,
        design_source=design,
        source_binding_sha256=source_binding_sha,
    )
    cycle_root_sha = readback.cycle_root_semantic_sha256_v4_1(
        cycle_id="cycle-v41",
        input_binding=bound.binding,
        design_source=design,
    )
    state = build_genesis_cycle_state_v4_1(
        cycle_id="cycle-v41",
        cycle_root_sha256=cycle_root_sha,
        source_chain_node_sha256=source_node["semantic_sha256"],
    )
    result = readback.publish_precommitted_cutoff_source_v4_1(
        private_root=private_root,
        run_id="run-v41",
        cycle_id="cycle-v41",
        input_binding=bound.binding,
        design_source=design,
        source_chain_node=source_node,
        precommitted_cycle_state=state,
        pit_records=bound.pit_records,
        expected_component_count=2,
        expected_source_binding_sha256=source_binding_sha,
        expected_state_sha256="empty",
    )

    run_dir = private_root / "run-v41"
    assert result["readiness"] == "EXPLORATORY_PRECOMMITTED"
    assert stat.S_IMODE(private_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    for descriptor in [
        *result["artifacts"].values(),
        result["readback_report"],
    ]:
        artifact = Path(descriptor["absolute_path"])
        metadata = os.lstat(artifact)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert metadata.st_uid == os.getuid()
        assert metadata.st_nlink == 1
        assert _sha(artifact) == descriptor["sha256"]
        assert artifact.read_bytes() == canonical_json_bytes(
            json.loads(artifact.read_text())
        )

    with pytest.raises(
        readback.FactorGovernanceSourceReadbackV4_1Error,
        match="state CAS mismatch",
    ):
        readback.publish_precommitted_cutoff_source_v4_1(
            private_root=private_root,
            run_id="run-v41",
            cycle_id="cycle-v41",
            input_binding=bound.binding,
            design_source=design,
            source_chain_node=source_node,
            precommitted_cycle_state=state,
            pit_records=bound.pit_records,
            expected_component_count=2,
            expected_source_binding_sha256=source_binding_sha,
            expected_state_sha256="empty",
        )


def test_blocked_publication_never_creates_source_or_state(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    bound = _bind(paths)
    result = readback.publish_blocked_cutoff_readback_v4_1(
        private_root=tmp_path / "private",
        run_id="blocked-v41",
        cycle_id="cycle-v41",
        input_binding=bound.binding,
        blocker_code="PIT_SOURCE_REJECTED",
        blocker_detail="invalid interval overlaps the bound calendar",
    )

    run_dir = tmp_path / "private" / "blocked-v41"
    assert result["readiness"] == "BLOCKED_FAIL_CLOSED"
    assert result["design_source"] is None
    assert result["source_chain_node"] is None
    assert result["precommitted_cycle_state"] is None
    assert not (run_dir / readback.DESIGN_SOURCE_FILENAME).exists()
    assert not (run_dir / readback.SOURCE_CHAIN_NODE_FILENAME).exists()
    assert not (run_dir / readback.PRECOMMITTED_STATE_FILENAME).exists()
    blocker_path = Path(result["blocker_report"]["absolute_path"])
    assert stat.S_IMODE(blocker_path.stat().st_mode) == 0o600


def test_publication_rejects_unvalidated_cross_artifact_bundle(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    bound = _bind(paths)
    private_root = tmp_path / "private-invalid"
    with pytest.raises(
        readback.FactorGovernanceSourceReadbackV4_1Error,
        match="design source validation failed",
    ):
        readback.publish_precommitted_cutoff_source_v4_1(
            private_root=private_root,
            run_id="invalid-v41",
            cycle_id="cycle-v41",
            input_binding=bound.binding,
            design_source={"schema_version": "invented"},
            source_chain_node={"schema_version": "invented"},
            precommitted_cycle_state={"schema_version": "invented"},
            pit_records=bound.pit_records,
            expected_component_count=2,
            expected_source_binding_sha256="a" * 64,
        )
    assert not private_root.exists()


def test_cutoff_node_reports_historical_alias_without_remap(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    bound = _bind(paths)
    alias = {
        "schema_version": "cn_pit_universe.v1",
        "symbol": "T600018.SH",
        "source_list_status": "D",
        "effective_from": "20000719",
        "effective_to": "20061020",
        "list_date": "20000719",
        "delist_date": "20061020",
        "membership_quality": "ok",
    }
    pit_records = [*bound.pit_records, alias]
    binding = copy.deepcopy(bound.binding)
    binding["pit_generation"]["row_count"] = 3
    binding["pit_generation"]["historical_alias_table_evidence"] = [
        {"symbol": "T600018.SH", "table_row_count": 0}
    ]
    source_binding_sha = "a" * 64
    design = build_design_source_node_v4_1(
        cycle_id="cycle-v41",
        pit_records=pit_records,
        component_symbols=list(bound.component_symbols),
        calendar_sessions=list(bound.calendar_sessions),
        market_binding_sha256=readback.binding_semantic_sha256_v4_1(binding),
        source_binding_sha256=source_binding_sha,
        expected_component_count=2,
    )

    node = readback.build_cutoff_source_node_v4_1(
        cycle_id="cycle-v41",
        input_binding=binding,
        design_source=design,
        source_binding_sha256=source_binding_sha,
    )

    assert node["out_of_bound_calendar_nonparticipating"]["records"] == [
        {
            "symbol": "T600018.SH",
            "source_list_status": "D",
            "effective_from": "2000-07-19",
            "effective_to": "2006-10-20",
            "active_bound_session_count": 0,
            "table_row_count": 0,
        }
    ]

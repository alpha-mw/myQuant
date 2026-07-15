from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market import macro_mart as macro_mart_module

from quant_investor.market.branch_readiness import (
    STATUS_BLOCK,
    assess_macro_readiness,
    load_macro_record,
)
from quant_investor.market.macro_mart import (
    MacroMartPromotionError,
    read_macro_mart,
    run_cn_macro_maintenance,
)
from quant_investor.market.market_data_store import MarketDataStore
from tests.helpers.macro_fixture import bind_macro_generation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(*, source: str = "tushare_primary") -> dict[str, object]:
    return {
        "trade_date": "2024-05-10",
        "macro_score": 0.2,
        "liquidity_score": 0.4,
        "volatility_percentile": 45.0,
        "policy_signal": "neutral",
        "source": source,
        "source_priority": "tushare_primary",
        "pit_status": "market_point_in_time",
        "fetched_at": "2024-05-10T08:00:00+00:00",
    }


def _bind_catalog_generation(
    market_root: Path,
    *,
    row: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    _, table, manifest, _ = bind_macro_generation(
        market_root / "macro_daily",
        generation_id="g1",
        row=row or _row(),
    )
    return table, manifest


def _rewrite_generation_manifest(
    market_root: Path,
    **updates: object,
) -> None:
    manifest_path = (
        market_root
        / "macro_daily"
        / "_generations"
        / "g1"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(updates)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    catalog_path = market_root / "_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["tables"]["macro_daily"][
        "generation_manifest_sha256"
    ] = _sha256(manifest_path)
    catalog_path.write_text(
        json.dumps(catalog, sort_keys=True),
        encoding="utf-8",
    )


def test_offline_compatibility_input_is_staged_and_cannot_claim_tushare(
    tmp_path: Path,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    root = market_root / "macro_daily"
    result = run_cn_macro_maintenance(
        indicators=_row(),
        as_of="2024-05-10",
        data_root=root,
        raw_snapshot_root=tmp_path / "raw",
        run_id="offline",
    )

    manifest = result["manifest"]
    assert result["status"] == "staged"
    assert result["promoted"] is False
    assert manifest["production_eligible"] is False
    assert manifest["applied"] is False
    assert manifest["source"] == "manual_offline_snapshot"
    assert manifest["source_priority"] == "manual_offline_snapshot"
    assert not (market_root / "_catalog.json").exists()
    assert not (root / "latest_manifest.json").exists()

    record, loaded = load_macro_record(as_of="2024-05-10", root=root)
    assert record == {}
    assert loaded == {"read_error": "macro_catalog_missing"}


def test_pointerless_or_subpointer_macro_layout_is_never_runtime_canonical(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parquet" / "cn" / "macro_daily"
    root.mkdir(parents=True)
    pd.DataFrame([_row()]).to_parquet(root / "part.parquet", index=False)
    (root / "latest_manifest.json").write_text(
        json.dumps({"generation_id": "legacy", "table_path": "part.parquet"}),
        encoding="utf-8",
    )

    with pytest.raises(MacroMartPromotionError, match="macro_catalog_missing"):
        read_macro_mart(data_root=root)
    record, manifest = load_macro_record(as_of="2024-05-10", root=root)
    assert record == {}
    assert manifest == {"read_error": "macro_catalog_missing"}


def test_catalog_rejects_data_root_with_ancestor_symlink(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    market_root = real_parent / "parquet" / "cn"
    _bind_catalog_generation(market_root)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        MacroMartPromotionError,
        match="macro_root_symlink_rejected",
    ):
        read_macro_mart(
            data_root=alias / "parquet" / "cn" / "macro_daily"
        )


def test_catalog_and_runtime_bind_the_same_hash_verified_generation(
    tmp_path: Path,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    table, manifest_path = _bind_catalog_generation(market_root)

    frame, manifest = read_macro_mart(data_root=market_root / "macro_daily")
    record, loaded_manifest = load_macro_record(
        as_of="2024-05-10",
        root=market_root / "macro_daily",
    )

    assert frame.iloc[0]["macro_score"] == pytest.approx(0.2)
    assert record["macro_score"] == pytest.approx(0.2)
    assert manifest["generation_id"] == "g1"
    assert loaded_manifest["generation_id"] == "g1"
    assert manifest["resolved_table_path"] == str(table.resolve())
    assert manifest["resolved_generation_manifest"] == str(manifest_path.resolve())


def test_catalog_rejects_manifest_table_path_with_hidden_parent(
    tmp_path: Path,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    _bind_catalog_generation(market_root)
    _rewrite_generation_manifest(
        market_root,
        table_path="hidden/part.parquet",
    )

    with pytest.raises(
        MacroMartPromotionError,
        match="macro_generation_manifest_table_path_mismatch",
    ):
        read_macro_mart(data_root=market_root / "macro_daily")


@pytest.mark.parametrize(
    ("row_updates", "blocker"),
    [
        ({"unexpected": "field"}, "macro_catalog_table_contract_invalid"),
        ({"macro_score": 1.01}, "macro_score_out_of_range"),
        ({"liquidity_score": -1.01}, "macro_liquidity_score_out_of_range"),
        (
            {"volatility_percentile": 100.01},
            "macro_volatility_percentile_out_of_range",
        ),
        ({"policy_signal": ""}, "macro_policy_signal_empty"),
        ({"source": "nbs_official"}, "macro_source_lineage_mismatch"),
        ({"source_priority": ""}, "macro_source_lineage_mismatch"),
        ({"pit_status": ""}, "macro_pit_lineage_mismatch"),
        ({"fetched_at": "not-a-time"}, "macro_fetched_at_invalid"),
    ],
)
def test_catalog_rejects_invalid_canonical_frame_contract(
    tmp_path: Path,
    row_updates: dict[str, object],
    blocker: str,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    _bind_catalog_generation(
        market_root,
        row={**_row(), **row_updates},
    )

    with pytest.raises(MacroMartPromotionError, match=blocker):
        read_macro_mart(data_root=market_root / "macro_daily")


def test_catalog_rejects_manifest_as_of_not_bound_to_latest_row(
    tmp_path: Path,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    _bind_catalog_generation(market_root)
    _rewrite_generation_manifest(market_root, as_of="2024-05-09")

    with pytest.raises(
        MacroMartPromotionError,
        match="macro_generation_manifest_as_of_mismatch",
    ):
        read_macro_mart(data_root=market_root / "macro_daily")


def test_catalog_and_readiness_reject_forged_source_priority_mapping(
    tmp_path: Path,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    forged = {
        **_row(source="manual_offline_snapshot"),
        "source_priority": "tushare_primary",
    }
    _bind_catalog_generation(market_root, row=forged)
    _rewrite_generation_manifest(
        market_root,
        source="manual_offline_snapshot",
        source_priority="tushare_primary",
    )

    with pytest.raises(
        MacroMartPromotionError,
        match="macro_source_priority_mismatch",
    ):
        read_macro_mart(data_root=market_root / "macro_daily")

    readiness = assess_macro_readiness(
        macro_record=forged,
        manifest={
            "source": "manual_offline_snapshot",
            "source_priority": "tushare_primary",
            "provider_status": "verified_provider_snapshot",
            "production_eligible": True,
            "generation_id": "g1",
        },
        as_of="2024-05-10",
    )
    assert readiness.status == STATUS_BLOCK
    assert "macro_source_priority_mismatch" in readiness.blockers


def test_catalog_generation_tamper_is_fail_closed(tmp_path: Path) -> None:
    market_root = tmp_path / "parquet" / "cn"
    table, _ = _bind_catalog_generation(market_root)
    pd.DataFrame([dict(_row(), macro_score=-0.8)]).to_parquet(table, index=False)

    with pytest.raises(MacroMartPromotionError, match="macro_catalog_table_hash_mismatch"):
        read_macro_mart(data_root=market_root / "macro_daily")
    record, manifest = load_macro_record(
        as_of="2024-05-10",
        root=market_root / "macro_daily",
    )
    assert record == {}
    assert manifest == {
        "read_error": "macro_catalog_table_hash_mismatch"
    }


@pytest.mark.parametrize("generation_id", [".", ".."])
def test_catalog_rejects_dot_generation_ids(
    tmp_path: Path,
    generation_id: str,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    _bind_catalog_generation(market_root)
    catalog_path = market_root / "_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["tables"]["macro_daily"]["generation_id"] = generation_id
    catalog_path.write_text(json.dumps(catalog, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        MacroMartPromotionError,
        match="macro_catalog_generation_invalid",
    ):
        read_macro_mart(data_root=market_root / "macro_daily")


@pytest.mark.parametrize(
    ("member", "blocker"),
    [
        ("table", "macro_catalog_table_path_invalid"),
        ("manifest", "macro_catalog_manifest_path_invalid"),
    ],
)
def test_catalog_rejects_final_component_symlinks(
    tmp_path: Path,
    member: str,
    blocker: str,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    table, manifest = _bind_catalog_generation(market_root)
    path = table if member == "table" else manifest
    target = path.with_name(f"real-{path.name}")
    path.rename(target)
    path.symlink_to(target.name)

    with pytest.raises(MacroMartPromotionError, match=blocker):
        read_macro_mart(data_root=market_root / "macro_daily")


def test_catalog_rejects_intermediate_component_symlink(tmp_path: Path) -> None:
    market_root = tmp_path / "parquet" / "cn"
    _bind_catalog_generation(market_root)
    generation = market_root / "macro_daily" / "_generations" / "g1"
    target = generation.with_name("real-g1")
    generation.rename(target)
    generation.symlink_to(target.name, target_is_directory=True)

    with pytest.raises(
        MacroMartPromotionError,
        match="macro_catalog_table_path_invalid",
    ):
        read_macro_mart(data_root=market_root / "macro_daily")


def test_catalog_detects_table_replacement_during_verified_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    table, _ = _bind_catalog_generation(market_root)
    table_inode = table.stat().st_ino
    original_read = os.read
    replaced = False

    def replace_after_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        payload = original_read(descriptor, size)
        if payload and not replaced and os.fstat(descriptor).st_ino == table_inode:
            replacement = table.with_name("replacement.parquet")
            pd.DataFrame([dict(_row(), macro_score=-0.8)]).to_parquet(
                replacement,
                index=False,
            )
            os.replace(replacement, table)
            replaced = True
        return payload

    monkeypatch.setattr(macro_mart_module.os, "read", replace_after_read)
    with pytest.raises(
        MacroMartPromotionError,
        match="macro_catalog_table_changed_during_read",
    ):
        read_macro_mart(data_root=market_root / "macro_daily")
    assert replaced is True


def test_catalog_detects_in_place_table_mutation_during_verified_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_root = tmp_path / "parquet" / "cn"
    table, _ = _bind_catalog_generation(market_root)
    table_inode = table.stat().st_ino
    original_read = os.read
    mutated = False

    def mutate_after_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        payload = original_read(descriptor, size)
        if payload and not mutated and os.fstat(descriptor).st_ino == table_inode:
            with table.open("ab") as handle:
                handle.write(b"concurrent-mutation")
                handle.flush()
                os.fsync(handle.fileno())
            mutated = True
        return payload

    monkeypatch.setattr(macro_mart_module.os, "read", mutate_after_read)
    with pytest.raises(
        MacroMartPromotionError,
        match="macro_catalog_table_changed_during_read",
    ):
        read_macro_mart(data_root=market_root / "macro_daily")
    assert mutated is True


def test_storage_validate_and_readiness_verify_the_same_macro_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    market_root = data_root / "parquet" / "cn"
    table, _ = _bind_catalog_generation(market_root)
    store = MarketDataStore(market="CN", data_root=data_root)
    monkeypatch.setattr(
        store.reader,
        "clean_snapshot_gate",
        lambda refresh=True: {
            "healthy": True,
            "blockers": [],
            "snapshot_id": "fixture",
            "latest_complete_trade_date": "20240510",
            "latest_trade_date": "20240510",
            "latest_pointer_path": "fixture",
            "table_root": "fixture",
            "serving_root": "fixture",
            "manifest_path": "fixture",
            "mode_policy": "strict",
        },
    )
    monkeypatch.setattr(store.reader, "_load_latest_payload", lambda refresh=True: {})

    passed = store.validate_latest()
    assert passed["status"] == "passed"
    assert passed["macro_generation"]["generation_id"] == "g1"
    assert passed["macro_generation"]["resolved_table_path"] == str(
        table.resolve()
    )

    pd.DataFrame([dict(_row(), macro_score=-0.8)]).to_parquet(table, index=False)
    failed = store.validate_latest()
    assert failed["status"] == "failed"
    assert "macro_catalog_table_hash_mismatch" in failed["blockers"]


def test_storage_validate_keeps_legacy_macro_catalog_as_nonblocking_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    market_root = data_root / "parquet" / "cn"
    market_root.mkdir(parents=True)
    (market_root / "_catalog.json").write_text(
        json.dumps(
            {
                "schema_version": "myquant-cn-clean-catalog.v1",
                "required_tables": ["macro_daily"],
                "tables": {
                    "macro_daily": {
                        "logical_table": "macro_daily",
                        "path": "data/parquet/cn/macro_daily/part.parquet",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    store = MarketDataStore(market="CN", data_root=data_root)
    monkeypatch.setattr(
        store.reader,
        "clean_snapshot_gate",
        lambda refresh=True: {
            "healthy": True,
            "blockers": [],
            "snapshot_id": "fixture",
            "latest_complete_trade_date": "20240510",
            "latest_trade_date": "20240510",
            "latest_pointer_path": "fixture",
            "table_root": "fixture",
            "serving_root": "fixture",
            "manifest_path": "fixture",
            "mode_policy": "strict",
        },
    )
    monkeypatch.setattr(store.reader, "_load_latest_payload", lambda refresh=True: {})
    monkeypatch.setattr(
        macro_mart_module,
        "read_macro_mart",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy catalog must not be claimed as a v14 generation")
        ),
    )

    result = store.validate_latest()

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["macro_generation"] == {
        "status": "legacy_catalog_entry_not_v14_generation",
        "catalog_schema_version": "myquant-cn-clean-catalog.v1",
        "production_eligible": False,
        "branch_readiness": "blocked",
        "blockers": ["macro_v14_generation_unavailable"],
    }


def test_macro_readiness_does_not_trust_row_reported_source_priority() -> None:
    record = _row(source="offline_input")
    readiness = assess_macro_readiness(
        macro_record=record,
        manifest={
            "source": "manual_offline_snapshot",
            "source_priority": "manual_offline_snapshot",
            "provider_status": "offline_input",
            "production_eligible": False,
        },
        as_of="2024-05-10",
    )

    assert readiness.status == STATUS_BLOCK
    assert "macro_not_tushare_primary" in readiness.blockers
    assert "macro_generation_not_production_eligible" in readiness.blockers


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("pit_status", "unknown", "macro_pit_status_invalid"),
        ("fetched_at", "", "macro_fetched_at_missing_or_invalid"),
        ("trade_date", "2024-05-09", "macro_trade_date_as_of_mismatch"),
    ],
)
def test_macro_readiness_validates_pit_and_freshness(
    field: str,
    value: str,
    blocker: str,
) -> None:
    record = dict(_row(), **{field: value})
    readiness = assess_macro_readiness(
        macro_record=record,
        manifest={
            "source": "tushare_primary",
            "source_priority": "tushare_primary",
            "provider_status": "verified_provider_snapshot",
            "production_eligible": True,
            "generation_id": "g1",
        },
        as_of="2024-05-10",
    )

    assert readiness.status == STATUS_BLOCK
    assert blocker in readiness.blockers

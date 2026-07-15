from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import quant_investor.market.macro_mart as macro_mart
from quant_investor.market.macro_mart import (
    MacroMartPromotionError,
    read_macro_mart,
    run_cn_macro_maintenance,
    write_macro_mart,
)


def _row(trade_date: str = "2024-05-10") -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "macro_score": 0.2,
        "liquidity_score": 0.4,
        "volatility_percentile": 45.0,
        "policy_signal": "neutral",
        "source": "official_fixture",
        "source_priority": "official_primary",
    }


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_catalog_generation(root, *, generation_id: str = "canonical-good"):
    generation = root / "_generations" / generation_id
    generation.mkdir(parents=True)
    table = generation / "part.parquet"
    canonical_row = {
        **_row(),
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "pit_status": "market_point_in_time",
        "fetched_at": "2024-05-10T08:00:00+00:00",
    }
    pd.DataFrame([canonical_row]).to_parquet(table, index=False)
    manifest_path = generation / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "cn-macro-mart.v14",
                "generation_id": generation_id,
                "table": "macro_daily",
                "table_path": table.name,
                "parquet_sha256": _digest(table),
                "source": "tushare_primary",
                "source_priority": "tushare_primary",
                "provider_status": "verified_provider_snapshot",
                "pit_status": "market_point_in_time",
                "as_of": "2024-05-10",
                "production_eligible": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    catalog_path = root.parent / "_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "strict-parquet-catalog.v1",
                "required_tables": ["macro_daily"],
                "tables": {
                    "macro_daily": {
                        "path": str(table.relative_to(root.parent)),
                        "generation_manifest": str(
                            manifest_path.relative_to(root.parent)
                        ),
                        "generation_id": generation_id,
                        "parquet_sha256": _digest(table),
                        "generation_manifest_sha256": _digest(manifest_path),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return catalog_path, table, manifest_path


def test_empty_and_unimplemented_live_paths_preserve_last_good(tmp_path):
    root = tmp_path / "parquet" / "cn" / "macro_daily"
    raw = tmp_path / "raw"
    catalog, table, manifest = _bind_catalog_generation(root)
    before = tuple(_digest(path) for path in (catalog, table, manifest))

    result = run_cn_macro_maintenance(
        allow_live=True,
        as_of="2024-05-11",
        data_root=root,
        raw_snapshot_root=raw,
    )

    assert result["status"] == "blocked"
    assert result["promoted"] is False
    assert tuple(_digest(path) for path in (catalog, table, manifest)) == before
    frame, loaded = read_macro_mart(data_root=root)
    assert loaded["generation_id"] == "canonical-good"
    assert frame.iloc[0]["trade_date"] == "2024-05-10"


def test_invalid_and_older_candidate_cannot_advance_catalog(tmp_path):
    root = tmp_path / "parquet" / "cn" / "macro_daily"
    raw = tmp_path / "raw"
    catalog, table, manifest = _bind_catalog_generation(root)
    before = tuple(_digest(path) for path in (catalog, table, manifest))

    with pytest.raises(
        MacroMartPromotionError,
        match="macro_required_fields_missing",
    ):
        write_macro_mart(
            {"trade_date": "2024-05-11", "macro_score": 0.1},
            data_root=root,
            raw_snapshot_root=raw,
            run_id="invalid",
        )
    older = write_macro_mart(
        _row("2024-05-09"),
        data_root=root,
        raw_snapshot_root=raw,
        run_id="older",
    )
    assert older["production_eligible"] is False
    assert older["applied"] is False
    assert tuple(_digest(path) for path in (catalog, table, manifest)) == before
    frame, loaded = read_macro_mart(data_root=root)
    assert loaded["generation_id"] == "canonical-good"
    assert frame.iloc[0]["trade_date"] == "2024-05-10"


def test_candidate_manifest_failure_keeps_catalog_last_good(tmp_path, monkeypatch):
    root = tmp_path / "parquet" / "cn" / "macro_daily"
    raw = tmp_path / "raw"
    catalog, table, manifest = _bind_catalog_generation(root)
    before = tuple(_digest(path) for path in (catalog, table, manifest))
    original = macro_mart._atomic_write_bytes

    def _fail_candidate_manifest(path, payload, **kwargs):
        if path.name == "manifest.json":
            raise OSError("simulated_candidate_manifest_failure")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(
        macro_mart,
        "_atomic_write_bytes",
        _fail_candidate_manifest,
    )
    with pytest.raises(OSError, match="simulated_candidate_manifest_failure"):
        write_macro_mart(
            _row("2024-05-11"),
            data_root=root,
            raw_snapshot_root=raw,
            run_id="candidate",
        )

    assert tuple(_digest(path) for path in (catalog, table, manifest)) == before
    frame, loaded = read_macro_mart(data_root=root)
    assert loaded["generation_id"] == "canonical-good"
    assert frame.iloc[0]["trade_date"] == "2024-05-10"


def test_unsafe_paths_and_run_ids_are_rejected(tmp_path):
    with pytest.raises(MacroMartPromotionError, match="macro_run_id_unsafe"):
        write_macro_mart(
            _row(),
            data_root=tmp_path / "macro",
            raw_snapshot_root=tmp_path / "raw",
            run_id="../escape",
        )
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(
        MacroMartPromotionError,
        match="macro_root_symlink_rejected",
    ):
        write_macro_mart(
            _row(),
            data_root=link,
            raw_snapshot_root=tmp_path / "raw",
            run_id="safe",
        )


def test_same_as_of_candidates_are_isolated_from_catalog(tmp_path):
    root = tmp_path / "parquet" / "cn" / "macro_daily"
    raw = tmp_path / "raw"
    catalog, table, manifest = _bind_catalog_generation(root)
    before = tuple(_digest(path) for path in (catalog, table, manifest))
    first = write_macro_mart(
        _row(),
        data_root=root,
        raw_snapshot_root=raw,
        run_id="first",
    )
    second = write_macro_mart(
        _row(),
        data_root=root,
        raw_snapshot_root=raw,
        run_id="same",
    )
    conflicting = write_macro_mart(
        dict(_row(), macro_score=-0.4),
        data_root=root,
        raw_snapshot_root=raw,
        run_id="conflict",
    )
    assert {
        first["generation_id"],
        second["generation_id"],
        conflicting["generation_id"],
    } == {"first", "same", "conflict"}
    assert all(
        candidate["production_eligible"] is False
        and candidate["applied"] is False
        for candidate in (first, second, conflicting)
    )
    assert tuple(_digest(path) for path in (catalog, table, manifest)) == before
    with pytest.raises(
        MacroMartPromotionError,
        match="macro_candidate_generation_exists",
    ):
        write_macro_mart(
            _row(),
            data_root=root,
            raw_snapshot_root=raw,
            run_id="first",
        )
    _, loaded = read_macro_mart(data_root=root)
    assert loaded["generation_id"] == "canonical-good"


def test_blank_policy_signal_is_rejected(tmp_path):
    row = dict(_row(), policy_signal="   ")
    with pytest.raises(MacroMartPromotionError, match="macro_policy_signal_empty"):
        write_macro_mart(
            row,
            data_root=tmp_path / "macro",
            raw_snapshot_root=tmp_path / "raw",
            run_id="blank-policy",
        )


def test_future_trade_date_cannot_poison_requested_as_of(tmp_path):
    root = tmp_path / "parquet" / "cn" / "macro_daily"
    raw = tmp_path / "raw"
    catalog, table, manifest = _bind_catalog_generation(root)
    before = tuple(_digest(path) for path in (catalog, table, manifest))

    with pytest.raises(
        MacroMartPromotionError,
        match="macro_trade_date_as_of_mismatch",
    ):
        write_macro_mart(
            _row("2099-01-01"),
            as_of="2024-05-11",
            data_root=root,
            raw_snapshot_root=raw,
            run_id="future-poison",
        )
    assert tuple(_digest(path) for path in (catalog, table, manifest)) == before
    _, loaded = read_macro_mart(data_root=root)
    assert loaded["generation_id"] == "canonical-good"

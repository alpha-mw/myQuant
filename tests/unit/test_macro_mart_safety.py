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


def test_empty_and_unimplemented_live_paths_preserve_last_good(tmp_path):
    root = tmp_path / "macro"
    raw = tmp_path / "raw"
    manifest = write_macro_mart(_row(), data_root=root, raw_snapshot_root=raw, run_id="good")
    pointer = root / "latest_manifest.json"
    table = root / manifest["table_path"]
    before = (_digest(pointer), _digest(table), pointer.read_bytes())

    result = run_cn_macro_maintenance(
        allow_live=True,
        as_of="2024-05-11",
        data_root=root,
        raw_snapshot_root=raw,
    )

    assert result["status"] == "blocked"
    assert result["promoted"] is False
    assert (_digest(pointer), _digest(table), pointer.read_bytes()) == before


def test_invalid_and_older_candidate_cannot_advance_pointer(tmp_path):
    root = tmp_path / "macro"
    raw = tmp_path / "raw"
    write_macro_mart(_row("2024-05-10"), data_root=root, raw_snapshot_root=raw, run_id="good")
    pointer = root / "latest_manifest.json"
    before = pointer.read_bytes()

    with pytest.raises(MacroMartPromotionError, match="required_fields_missing"):
        write_macro_mart(
            {"trade_date": "2024-05-11", "macro_score": 0.1},
            data_root=root,
            raw_snapshot_root=raw,
            run_id="invalid",
        )
    with pytest.raises(MacroMartPromotionError, match="older_generation"):
        write_macro_mart(
            _row("2024-05-09"),
            data_root=root,
            raw_snapshot_root=raw,
            run_id="older",
        )
    assert pointer.read_bytes() == before


def test_failure_before_pointer_swap_keeps_readable_last_good(tmp_path, monkeypatch):
    root = tmp_path / "macro"
    raw = tmp_path / "raw"
    first = write_macro_mart(_row(), data_root=root, raw_snapshot_root=raw, run_id="good")
    pointer = root / "latest_manifest.json"
    before = pointer.read_bytes()
    original = macro_mart._atomic_write_bytes

    def _fail_pointer(path, payload, **kwargs):
        if path.name == "latest_manifest.json":
            raise OSError("simulated_pointer_failure")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(macro_mart, "_atomic_write_bytes", _fail_pointer)
    with pytest.raises(OSError, match="simulated_pointer_failure"):
        write_macro_mart(
            _row("2024-05-11"),
            data_root=root,
            raw_snapshot_root=raw,
            run_id="candidate",
        )

    assert pointer.read_bytes() == before
    frame, manifest = read_macro_mart(data_root=root)
    assert manifest["generation_id"] == first["generation_id"]
    assert frame.iloc[0]["trade_date"] == "2024-05-10"


def test_unsafe_paths_and_run_ids_are_rejected(tmp_path):
    with pytest.raises(MacroMartPromotionError, match="run_id_unsafe"):
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
    with pytest.raises(MacroMartPromotionError, match="root_symlink"):
        write_macro_mart(
            _row(),
            data_root=link,
            raw_snapshot_root=tmp_path / "raw",
            run_id="safe",
        )


def test_corrupt_v2_pointer_never_falls_back_to_legacy_table(tmp_path):
    root = tmp_path / "macro"
    raw = tmp_path / "raw"
    write_macro_mart(_row(), data_root=root, raw_snapshot_root=raw, run_id="good")
    pd.DataFrame([_row()]).to_parquet(root / "part.parquet", index=False)
    (root / "latest_manifest.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(MacroMartPromotionError, match="pointer_invalid"):
        read_macro_mart(data_root=root)

    from quant_investor.market.branch_readiness import load_macro_record

    record, manifest = load_macro_record(as_of="20240510", root=root)
    assert record == {}
    assert manifest == {}


def test_bad_hash_and_symlink_pointer_are_fail_closed(tmp_path):
    from quant_investor.market.branch_readiness import load_macro_record

    for case in ("bad_hash", "symlink"):
        root = tmp_path / case / "macro"
        raw = tmp_path / case / "raw"
        write_macro_mart(_row(), data_root=root, raw_snapshot_root=raw, run_id="good")
        pd.DataFrame([_row()]).to_parquet(root / "part.parquet", index=False)
        pointer = root / "latest_manifest.json"
        if case == "bad_hash":
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            payload["parquet_sha256"] = "0" * 64
            pointer.write_text(json.dumps(payload), encoding="utf-8")
            expected = "hash_mismatch"
        else:
            external = tmp_path / case / "external.json"
            external.write_bytes(pointer.read_bytes())
            pointer.unlink()
            pointer.symlink_to(external)
            expected = "pointer_invalid"

        with pytest.raises(MacroMartPromotionError, match=expected):
            read_macro_mart(data_root=root)
        record, _ = load_macro_record(as_of="20240510", root=root)
        assert record == {}


def test_same_as_of_is_idempotent_only_for_identical_payload(tmp_path):
    root = tmp_path / "macro"
    raw = tmp_path / "raw"
    first = write_macro_mart(_row(), data_root=root, raw_snapshot_root=raw, run_id="first")
    pointer = root / "latest_manifest.json"
    before = pointer.read_bytes()

    second = write_macro_mart(_row(), data_root=root, raw_snapshot_root=raw, run_id="same")
    assert second["idempotent"] is True
    assert pointer.read_bytes() == before

    conflicting = dict(_row(), macro_score=-0.4)
    with pytest.raises(MacroMartPromotionError, match="same_as_of_conflict"):
        write_macro_mart(conflicting, data_root=root, raw_snapshot_root=raw, run_id="conflict")
    assert pointer.read_bytes() == before
    assert json.loads(pointer.read_text(encoding="utf-8"))["generation_id"] == first["generation_id"]


def test_blank_policy_signal_is_rejected(tmp_path):
    row = dict(_row(), policy_signal="   ")
    with pytest.raises(MacroMartPromotionError, match="policy_signal_empty"):
        write_macro_mart(
            row,
            data_root=tmp_path / "macro",
            raw_snapshot_root=tmp_path / "raw",
            run_id="blank-policy",
        )


def test_future_trade_date_cannot_poison_requested_as_of(tmp_path):
    root = tmp_path / "macro"
    raw = tmp_path / "raw"
    write_macro_mart(
        _row("2024-05-10"),
        as_of="2024-05-10",
        data_root=root,
        raw_snapshot_root=raw,
        run_id="good",
    )
    pointer = root / "latest_manifest.json"
    before = pointer.read_bytes()

    with pytest.raises(MacroMartPromotionError, match="trade_date_as_of_mismatch"):
        write_macro_mart(
            _row("2099-01-01"),
            as_of="2024-05-11",
            data_root=root,
            raw_snapshot_root=raw,
            run_id="future-poison",
        )
    assert pointer.read_bytes() == before


def test_hashless_v2_like_pointer_cannot_use_legacy_fallback(tmp_path):
    from quant_investor.market.branch_readiness import load_macro_record

    root = tmp_path / "macro"
    root.mkdir()
    pd.DataFrame([_row()]).to_parquet(root / "part.parquet", index=False)
    (root / "latest_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "cn-macro-mart.v2",
                "generation_id": "incomplete",
                "table_path": "part.parquet",
            }
        ),
        encoding="utf-8",
    )

    record, _ = load_macro_record(as_of="20240510", root=root)

    assert record == {}

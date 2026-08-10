from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile

import pytest

from quant_investor.strategy_records.store import (
    StrategyRecordConflict,
    StrategyRecordStoreError,
    load_registered_catalog,
)
from scripts import manage_cn_strategy_records as manager


def _args(**values: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "record_root": "",
        "record_dir": [],
        "active_record_id": None,
        "previous_record_id": None,
        "dashboard_projection_json": None,
        "generation_id": None,
        "published_at": "2026-08-10T00:00:00Z",
        "expected_current_id": None,
        "expected_previous_id": None,
        "expected_inventory_sha": None,
        "project_root": None,
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def _bootstrap(root: Path) -> dict[str, object]:
    run = root / "20260809_1000"
    run.mkdir(parents=True)
    (run / "holdings.json").write_text('{"cash":1}\n', encoding="utf-8")
    return manager.command_bootstrap(
        _args(
            record_root=str(root),
            generation_id="g1",
            record_dir=[run.name],
        )
    )


def _pointer_sha(root: Path) -> str:
    return hashlib.sha256(
        (root / "_record_store" / "current.v1.json").read_bytes()
    ).hexdigest()


def test_inventory_reports_unregistered_legacy_and_registered_orphans(
    tmp_path: Path,
) -> None:
    (tmp_path / "legacy").mkdir()
    before = manager.command_inventory(_args(record_root=str(tmp_path)))
    assert before["registered"] is False
    assert before["orphan_record_dirs"] == ["legacy"]
    _bootstrap(tmp_path)
    after = manager.command_inventory(_args(record_root=str(tmp_path)))
    assert after["registered"] is True
    assert after["orphan_record_dirs"] == ["legacy"]
    assert after["orphans_preserved"] is True


def test_bootstrap_live_expectations_fail_before_registration(
    tmp_path: Path,
) -> None:
    (tmp_path / "20260809_1000").mkdir()
    (tmp_path / "20260810_1000").mkdir()
    projection_path = tmp_path.parent / "projection.json"
    projection_path.write_text(
        json.dumps(
            {
                "valid_records": [
                    {"record": "20260809_1000"},
                    {"record": "20260810_1000"},
                ],
                "historical_records": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(StrategyRecordStoreError, match="active record"):
        manager.command_bootstrap_live(
            _args(
                record_root=str(tmp_path),
                record_dir=[],
                dashboard_projection_json=str(projection_path),
                expected_current_id="other",
                generation_id="g1",
            )
        )
    assert not (tmp_path / "_record_store").exists()


def test_stage_seal_exact_once_and_collision(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    stage_args = _args(record_root=str(tmp_path), record_id="20260810_1000")
    stage = Path(manager.command_stage_init(stage_args)["staging_dir"])
    (stage / "payload.json").write_text('{"value":1}\n', encoding="utf-8")
    sealed = manager.command_seal_publish(
        _args(
            record_root=str(tmp_path),
            record_id="20260810_1000",
            expected_pointer_sha=_pointer_sha(tmp_path),
            generation_id="g2",
            published_at="2026-08-10T00:01:00Z",
        )
    )
    target = tmp_path / "20260810_1000"
    assert target.is_dir()
    assert not stage.exists()
    idempotent = manager.command_seal_publish(
        _args(
            record_root=str(tmp_path),
            record_id="20260810_1000",
            expected_pointer_sha=sealed["pointer_sha256"],
            generation_id="unused",
        )
    )
    assert idempotent["idempotent"] is True
    stage.mkdir()
    (stage / "payload.json").write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(StrategyRecordConflict, match="different bytes"):
        manager.command_seal_publish(
            _args(
                record_root=str(tmp_path),
                record_id="20260810_1000",
                expected_pointer_sha=sealed["pointer_sha256"],
                generation_id="g3",
            )
        )


def test_stage_rejects_noncanonical_new_record_id(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    with pytest.raises(
        StrategyRecordStoreError,
        match="new record_id must use YYYYMMDD_HHMM",
    ):
        manager.command_stage_init(
            _args(record_root=str(tmp_path), record_id="research-output")
        )


def test_new_record_budgets_reject_file_count_and_total_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "record"
    record.mkdir()
    (record / "a").write_bytes(b"1234")
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_FILE_BYTES", 3)
    with pytest.raises(StrategyRecordStoreError, match="file exceeds"):
        manager.build_inventory(record, enforce_new_record_budget=True)
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_FILE_BYTES", 10)
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_TOTAL_BYTES", 3)
    with pytest.raises(StrategyRecordStoreError, match="total byte"):
        manager.build_inventory(record, enforce_new_record_budget=True)
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_TOTAL_BYTES", 10)
    monkeypatch.setattr(manager, "NEW_RECORD_MAX_FILES", 0)
    with pytest.raises(StrategyRecordStoreError, match="file-count"):
        manager.build_inventory(record, enforce_new_record_budget=True)


def test_inventory_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    record = tmp_path / "record"
    record.mkdir()
    source = record / "source"
    source.write_bytes(b"x")
    os.symlink("source", record / "link")
    with pytest.raises(StrategyRecordStoreError, match="symlink"):
        manager.build_inventory(record, enforce_new_record_budget=False)
    (record / "link").unlink()
    os.link(source, record / "hard")
    with pytest.raises(StrategyRecordStoreError, match="hard links"):
        manager.build_inventory(record, enforce_new_record_budget=False)


def test_no_action_receipt_references_checkpoint_without_copying_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bootstrap(tmp_path)
    result = manager.command_no_action(
        _args(
            record_root=str(tmp_path),
            receipt_id="no-action-1",
            reason="readiness blocked",
            expected_pointer_sha=_pointer_sha(tmp_path),
            generation_id="g2",
            published_at="2026-08-10T00:01:00Z",
        )
    )
    receipt = result["catalog"]["receipts"][-1]
    assert receipt["active_checkpoint"] == result["pointer"]["active_closure"]
    assert receipt["payload_copied"] is False
    assert "inventory" not in receipt["active_checkpoint"]
    monkeypatch.setattr(manager, "NO_ACTION_RECEIPT_MAX_BYTES", 200)
    with pytest.raises(StrategyRecordStoreError, match="receipt exceeds"):
        manager.command_no_action(
            _args(
                record_root=str(tmp_path),
                receipt_id="no-action-2",
                reason="x" * 500,
                expected_pointer_sha=result["pointer_sha256"],
                generation_id="g3",
            )
        )


def test_restore_rejects_malicious_archive_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    restore = tmp_path / "restore"
    restore.mkdir()
    with pytest.raises(StrategyRecordStoreError, match="canonical relative"):
        manager._restore_tar(archive_path, restore)
    assert not (tmp_path / "escape").exists()

    symlink_archive = tmp_path / "symlink.tar"
    with tarfile.open(symlink_archive, "w") as archive:
        info = tarfile.TarInfo("record/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(StrategyRecordStoreError, match="member type"):
        manager._restore_tar(symlink_archive, restore)


@pytest.mark.skipif(
    shutil.which("zstd") is None, reason="zstd executable unavailable"
)
def test_archive_rehearsal_is_copy_only_and_full_restore_verified(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    root.mkdir()
    _bootstrap(root)
    source = root / "20260809_1000" / "holdings.json"
    before = source.read_bytes()
    output = tmp_path / "archive"
    result = manager.command_archive_rehearsal(
        _args(record_root=str(root), before="2027", output_root=str(output))
    )
    assert Path(result["archive_path"]).is_file()
    assert source.read_bytes() == before
    assert result["source_records_preserved"] is True
    assert result["moved"] is False
    assert result["deleted"] is False
    assert load_registered_catalog(root) is not None


def test_main_inventory_json_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "legacy").mkdir()
    assert (
        manager.main(["inventory", "--record-root", str(tmp_path), "--json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["registered"] is False


def test_quarantine_move_is_resume_safe_and_rollback_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "records"
    root.mkdir()
    rows = []
    for record_id in ("20260601_1000", "20260602_1000"):
        record_dir = root / record_id
        record_dir.mkdir()
        (record_dir / "payload").write_text(record_id, encoding="utf-8")
        rows.append(
            {
                "record_id": record_id,
                "relative_path": record_id,
                **manager.build_inventory(
                    record_dir, enforce_new_record_budget=False
                ),
            }
        )
    plan = {
        "transaction_id": "tx-resume",
        "record_root": str(root),
        "quarantine_root": str(tmp_path / "quarantine/tx-resume/records"),
        "record_ids": [row["record_id"] for row in rows],
        "source_catalog": {"records": rows},
    }
    real_rename = os.rename
    calls = 0

    def crash_on_second(source: object, target: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated crash")
        real_rename(source, target)

    monkeypatch.setattr(manager.os, "rename", crash_on_second)
    with pytest.raises(OSError, match="simulated crash"):
        manager._move_records(plan, direction="cutover")
    monkeypatch.setattr(manager.os, "rename", real_rename)
    assert manager._move_records(plan, direction="cutover") == 1
    assert all(not (root / row["record_id"]).exists() for row in rows)
    assert manager._move_records(plan, direction="rollback") == 2
    assert all((root / row["record_id"]).is_dir() for row in rows)


def test_quarantine_refuses_dual_or_lost_locations(tmp_path: Path) -> None:
    root = tmp_path / "records"
    quarantine = tmp_path / "quarantine"
    source = root / "20260601_1000"
    target = quarantine / "20260601_1000"
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    row = {
        "record_id": source.name,
        "relative_path": source.name,
        **manager.build_inventory(source, enforce_new_record_budget=False),
    }
    with pytest.raises(StrategyRecordStoreError, match="dual"):
        manager._verify_location(root, quarantine, row)
    source.rmdir()
    target.rmdir()
    with pytest.raises(StrategyRecordStoreError, match="lost"):
        manager._verify_location(root, quarantine, row)


def test_archive_candidate_rejects_unregistered_strict_record(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "results/strategy_records/CN/aggressive_tech_manufacturing"
    )
    root.mkdir(parents=True)
    _bootstrap(root)
    (root / "20260810_1948").mkdir()
    args = _args(
        record_root=str(root),
        project_root=str(tmp_path),
        expected_pointer_sha=_pointer_sha(root),
        archive_manifest=[],
    )
    with pytest.raises(
        StrategyRecordStoreError,
        match=(
            "unregistered strict record directory blocks archive candidate: "
            "20260810_1948"
        ),
    ):
        manager.command_archive_candidate(args)

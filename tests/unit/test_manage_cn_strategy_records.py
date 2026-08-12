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
    CATALOG_SCHEMA_V2,
    StrategyRecordConflict,
    StrategyRecordStoreError,
    bootstrap_catalog,
    canonical_json_bytes,
    content_sha256,
    load_registered_catalog,
)
from scripts import manage_cn_strategy_records as manager
from scripts.cn_dashboard_common import DashboardInputError


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


def test_seal_publish_normalizes_relative_record_root_for_dashboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "records"
    _bootstrap(root)
    target = root / "20260810_1000"
    target.mkdir()
    (target / "payload.json").write_text('{"value":1}\n', encoding="utf-8")
    observed: list[Path] = []

    def scan_valid(record_root: Path, project_root: Path) -> tuple[list, list, str]:
        observed.append(record_root)
        return (
            [
                {"record": "20260809_1000", "source_refs": []},
                {"record": "20260810_1000", "source_refs": []},
            ],
            [],
            "20260810_1000",
        )

    def scan_historical(**values: object) -> tuple[list, list]:
        observed.append(values["record_root"])
        return ([{"record": "20260809_1000", "source_refs": []}], [])

    monkeypatch.setattr(
        "scripts.cn_dashboard_common.scan_valid_records", scan_valid
    )
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.scan_historical_performance_records",
        scan_historical,
    )
    monkeypatch.setattr(manager, "_attach_dashboard_closure", lambda *a, **k: None)
    relative_root = Path(os.path.relpath(root, Path.cwd()))
    result = manager.command_seal_publish(
        _args(
            record_root=str(relative_root),
            record_id="20260810_1000",
            expected_pointer_sha=_pointer_sha(root),
            project_root=str(tmp_path),
            generation_id="g-relative",
        )
    )
    assert result["pointer"]["active_record_id"] == "20260810_1000"
    assert observed == [root.resolve(), root.resolve()]


def test_seal_publish_atomically_adopts_existing_source_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "records"
    _bootstrap(root)
    for record_id in ("20260810_1943", "20260810_1948"):
        target = root / record_id
        target.mkdir()
        (target / "payload.json").write_text(
            '{"value":1}\n', encoding="utf-8"
        )

    valid = [
        {
            "record": "20260809_1000",
            "source_record": None,
            "source_refs": [],
        },
        {
            "record": "20260810_1943",
            "source_record": "20260809_1000",
            "source_refs": [],
        },
        {
            "record": "20260810_1948",
            "source_record": "20260810_1943",
            "source_refs": [],
        },
    ]
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.scan_valid_records",
        lambda *args: (valid, [], "20260810_1948"),
    )
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.scan_historical_performance_records",
        lambda **kwargs: ([], []),
    )
    monkeypatch.setattr(
        manager, "_attach_dashboard_closure", lambda *args, **kwargs: None
    )
    result = manager.command_seal_publish(
        _args(
            record_root=str(root),
            record_id="20260810_1948",
            expected_pointer_sha=_pointer_sha(root),
            project_root=str(tmp_path),
            generation_id="g-adopt-chain",
        )
    )
    assert result["pointer"]["active_record_id"] == "20260810_1948"
    assert result["pointer"]["previous_record_id"] == "20260810_1943"
    by_id = {row["record_id"]: row for row in result["catalog"]["records"]}
    assert {"20260810_1943", "20260810_1948"} <= set(by_id)


def test_dashboard_projection_source_refs_deduplicate_and_conflict() -> None:
    projection = {
        "valid_records": [
            {
                "record": "20260810_1948",
                "source_refs": [
                    {"path": "b", "sha256": "2" * 64},
                    {"path": "a", "sha256": "1" * 64},
                    {"path": "a", "sha256": "1" * 64},
                ],
            }
        ],
        "historical_records": [],
    }
    manager._normalize_dashboard_projection_source_refs(projection)
    assert projection["valid_records"][0]["source_refs"] == [
        {"path": "a", "sha256": "1" * 64},
        {"path": "b", "sha256": "2" * 64},
    ]
    projection["valid_records"][0]["source_refs"].append(
        {"path": "a", "sha256": "3" * 64}
    )
    with pytest.raises(StrategyRecordStoreError, match="refs conflict"):
        manager._normalize_dashboard_projection_source_refs(projection)


def test_archive_aware_seal_publish_extends_catalog_without_hot_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "records"
    target = root / "20260810_2000"
    target.mkdir(parents=True)
    (target / "payload").write_text("new", encoding="utf-8")
    pointer = {
        "active_record_id": "20260810_1948",
        "previous_record_id": "20260810_1943",
    }
    catalog = {
        "schema_id": "myquant.strategy_record_catalog.v2",
        "records": [
            {
                "record_id": "20260810_1948",
                "relative_path": "20260810_1948",
                "state": "ONLINE",
                "storage_state": "ONLINE",
            },
            {
                "record_id": "20260630_1614",
                "relative_path": "20260630_1614",
                "state": "ARCHIVED",
                "storage_state": "ARCHIVED",
            },
        ],
        "dashboard_projection": {
            "valid_records": [
                {
                    "record": "20260810_1948",
                    "source_record": "20260810_1943",
                    "source_refs": [],
                }
            ],
            "rejected": [],
            "latest_seen": "20260810_1948",
            "historical_records": [
                {"record": "20260630_1614", "source_refs": []}
            ],
            "historical_rejected": [],
        },
    }
    monkeypatch.setattr(
        manager, "load_registered_catalog", lambda value: (pointer, catalog)
    )
    monkeypatch.setattr(manager, "_pointer_sha", lambda value: "a" * 64)
    new_valid = {
        "record": "20260810_2000",
        "source_record": "20260810_1948",
        "source_refs": [{"path": "new", "sha256": "b" * 64}],
    }
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.validate_record",
        lambda *args: new_valid,
    )
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.validate_historical_record",
        lambda **kwargs: (_ for _ in ()).throw(
            DashboardInputError("historical_official_valuation_incomplete")
        ),
    )
    monkeypatch.setattr(
        "scripts.cn_dashboard_common.validate_historical_performance_sequence",
        lambda rows: None,
    )
    monkeypatch.setattr(manager, "_attach_dashboard_closure", lambda *a, **k: None)
    registry_calls: list[dict[str, object]] = []

    def build_registry(**kwargs: object) -> tuple[dict, dict]:
        registry_calls.append(kwargs)
        return ({"schema_id": "registry"}, {"path": "registry", "sha256": "c" * 64})

    monkeypatch.setattr(manager, "_build_candidate_history_registry", build_registry)
    published: dict[str, object] = {}

    def publish(*args: object, **kwargs: object) -> dict:
        published.update(kwargs)
        return {"pointer": {"active_record_id": "20260810_2000"}}

    monkeypatch.setattr(manager, "publish_catalog", publish)
    result = manager.command_seal_publish(
        _args(
            record_root=str(root),
            record_id="20260810_2000",
            expected_pointer_sha="a" * 64,
            project_root=str(tmp_path),
            generation_id="g-archive-aware",
        )
    )
    assert result["pointer"]["active_record_id"] == "20260810_2000"
    assert published["catalog_schema"] == manager.CATALOG_SCHEMA_V2
    assert published["inherit_history_registry"] is False
    assert published["previous_record_id"] == "20260810_1948"
    projection = published["dashboard_projection"]
    assert [row["record"] for row in projection["historical_records"]] == [
        "20260630_1614"
    ]
    assert projection["historical_rejected"] == [
        "20260810_2000:historical_official_valuation_incomplete"
    ]
    assert registry_calls[0]["generation_id"] == "g-archive-aware"


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


def test_consecutive_no_actions_preserve_history_state_binding(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    root = project / (
        "results/strategy_records/CN/aggressive_tech_manufacturing"
    )
    record_ids = ("20260809_1000", "20260810_1000")
    for record_id in record_ids:
        (root / record_id).mkdir(parents=True)
    empty_inventory_sha = hashlib.sha256(b"[]\n").hexdigest()
    records = [
        {
            "record_id": record_id,
            "relative_path": record_id,
            "state": "ONLINE",
            "storage_state": "ONLINE",
            "sealed_at": "2026-08-10T00:00:00Z",
            "inventory": [],
            "inventory_sha256": empty_inventory_sha,
            "file_count": 0,
            "total_bytes": 0,
        }
        for record_id in record_ids
    ]
    projection = {
        "valid_records": [],
        "rejected": [],
        "latest_seen": None,
        "historical_records": [],
        "historical_rejected": [],
    }
    registry = {
        "schema_version": "cn_aggressive_dashboard_history_integrity.v2",
        "market": "CN",
        "strategy_label": "aggressive_tech_manufacturing",
        "generated_at": "2026-08-10T00:00:00Z",
        "authority": "DASHBOARD_POST_HOC_INTEGRITY_DECLARATION",
        "intended_generation_id": "g-history",
        "dashboard_projection_sha256": hashlib.sha256(
            canonical_json_bytes(projection)
        ).hexdigest(),
        "record_count": 0,
        "records": [],
    }
    registry["content_sha256"] = content_sha256(registry)
    registry_path = project / "results/history-integrity/g-history.v2.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_bytes(canonical_json_bytes(registry))
    registry_ref = {
        "path": registry_path.relative_to(project).as_posix(),
        "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    }
    initial = bootstrap_catalog(
        root,
        records=records,
        dashboard_projection=projection,
        active_record_id=record_ids[1],
        previous_record_id=record_ids[0],
        generation_id="g-history",
        published_at="2026-08-10T00:00:00Z",
        catalog_schema=CATALOG_SCHEMA_V2,
        history_registry=registry,
        history_registry_ref=registry_ref,
    )
    registry_bytes = registry_path.read_bytes()
    first = manager.command_no_action(
        _args(
            record_root=str(root),
            receipt_id="no-action-1",
            reason="metadata only",
            expected_pointer_sha=initial["pointer_sha256"],
            generation_id="g-noop-1",
            published_at="2026-08-10T00:01:00Z",
        )
    )
    second = manager.command_no_action(
        _args(
            record_root=str(root),
            receipt_id="no-action-2",
            reason="metadata only",
            expected_pointer_sha=first["pointer_sha256"],
            generation_id="g-noop-2",
            published_at="2026-08-10T00:02:00Z",
        )
    )

    assert first["pointer"]["generation_id"] == "g-noop-1"
    assert second["pointer"]["generation_id"] == "g-noop-2"
    assert second["catalog"]["generation_id"] == "g-noop-2"
    for key in (
        "active_record_id",
        "previous_record_id",
        "active_closure",
    ):
        assert second["pointer"][key] == initial["pointer"][key]
    for key in (
        "records",
        "dashboard_projection",
        "history_registry",
        "history_registry_ref",
    ):
        assert first["catalog"][key] == initial["catalog"][key]
        assert second["catalog"][key] == initial["catalog"][key]
    assert second["catalog"]["history_registry"][
        "intended_generation_id"
    ] == "g-history"
    assert registry_path.read_bytes() == registry_bytes
    assert [row["receipt_id"] for row in second["catalog"]["receipts"]] == [
        "no-action-1",
        "no-action-2",
    ]


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

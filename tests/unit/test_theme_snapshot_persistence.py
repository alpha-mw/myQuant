from __future__ import annotations

from pathlib import Path

from quant_investor.market.dag.theme_context import persist_theme_rotation_snapshot


def test_persist_theme_rotation_snapshot_disabled(tmp_path):
    status = persist_theme_rotation_snapshot(
        theme_rotation={"status": "success"},
        enabled=False,
        root_dir=tmp_path,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
    )

    assert status["enabled"] is False
    assert status["status"] == "disabled"
    assert status["path"] == ""
    assert status["error"] == ""
    assert status["diagnostic_notes"] == ["theme_snapshot_disabled"]
    assert list(tmp_path.rglob("*.json")) == []


def test_persist_theme_rotation_snapshot_success(tmp_path):
    status = persist_theme_rotation_snapshot(
        theme_rotation={"status": "success", "metadata": {"run_id": "scan-001"}},
        enabled=True,
        root_dir=tmp_path,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        run_id="hash-001",
    )

    assert status["enabled"] is True
    assert status["status"] == "success"
    assert status["error"] == ""
    assert Path(status["path"]).exists()


def test_persist_theme_rotation_snapshot_skips_disabled_theme_rotation(tmp_path):
    status = persist_theme_rotation_snapshot(
        theme_rotation={"status": "disabled"},
        enabled=True,
        root_dir=tmp_path,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        save_disabled=False,
    )

    assert status["enabled"] is True
    assert status["status"] == "skipped"
    assert status["path"] == ""
    assert "theme_rotation_disabled_not_saved" in status["diagnostic_notes"]
    assert list(tmp_path.rglob("*.json")) == []


def test_persist_theme_rotation_snapshot_can_save_disabled_when_configured(tmp_path):
    status = persist_theme_rotation_snapshot(
        theme_rotation={"status": "disabled"},
        enabled=True,
        root_dir=tmp_path,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        save_disabled=True,
    )

    assert status["status"] == "success"
    assert Path(status["path"]).exists()


def test_persist_theme_rotation_snapshot_error_safe(tmp_path, monkeypatch):
    def _raise(self, theme_rotation, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "quant_investor.market.dag.theme_context.ThemeSnapshotStore.save",
        _raise,
    )

    status = persist_theme_rotation_snapshot(
        theme_rotation={"status": "success"},
        enabled=True,
        root_dir=tmp_path,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
    )

    assert status["enabled"] is True
    assert status["status"] == "error"
    assert status["path"] == ""
    assert status["error"] == "boom"
    assert any(note.startswith("theme_snapshot_error: boom") for note in status["diagnostic_notes"])

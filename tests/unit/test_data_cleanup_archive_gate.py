"""Archive-backed data cleanup gate contract tests."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from scripts.data_cleanup_archive_gate import (
    build_archive_gate_report,
    main as archive_gate_main,
    write_archive_gate_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_source_root(tmp_path: Path) -> Path:
    source_root = tmp_path / "reports" / "storage" / "csv_quarantine"
    (source_root / "daily").mkdir(parents=True)
    (source_root / "daily" / "000001.SZ.csv").write_text(
        "symbol,trade_date\n000001.SZ,20260630\n",
        encoding="utf-8",
    )
    (source_root / "daily" / "000002.SZ.csv").write_text(
        "symbol,trade_date\n000002.SZ,20260630\n",
        encoding="utf-8",
    )
    return source_root


def _write_archive_and_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    source_root = _write_source_root(tmp_path)
    archive_path = tmp_path / "backup" / "csv_quarantine.tar.gz"
    archive_path.parent.mkdir()
    with tarfile.open(archive_path, mode="w:gz") as archive:
        archive.add(
            source_root,
            arcname="reports/storage/csv_quarantine",
        )
    members: list[str]
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [member.name for member in archive.getmembers()]
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    dirs = sorted(path for path in source_root.rglob("*") if path.is_dir())
    manifest: dict[str, object] = {
        "schema_version": "myquant.csv_quarantine_restore_manifest.v1",
        "generated_at": "2026-07-06T15:00:00Z",
        "source_root": "reports/storage/csv_quarantine",
        "source_file_count": len(files),
        "source_directory_count": len(dirs),
        "source_size_bytes": sum(path.stat().st_size for path in files),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256(archive_path),
        "archive_member_count": len(members),
        "archive_source_prefix": "reports/storage/csv_quarantine/",
    }
    return archive_path, manifest


def test_archive_gate_allows_delete_when_archive_and_references_are_clean(tmp_path):
    _archive_path, manifest = _write_archive_and_manifest(tmp_path)

    report = build_archive_gate_report(manifest, repo_root=tmp_path)

    assert report["schema_version"] == "myquant.data_cleanup_archive_gate.v1"
    assert report["delete_candidate_count"] == 1
    assert report["summary"]["gate_status"] == "delete_allowed"
    assert report["summary"]["delete_allowed"] is True
    result = report["result"]
    assert result["delete_allowed"] is True
    assert result["failed_checks"] == []
    assert "archive_sha256_matches_manifest" in result["passed_checks"]
    assert result["runtime_references"] == {}
    assert result["strategy_references"] == {}


def test_archive_gate_blocks_runtime_and_strategy_references(tmp_path):
    _archive_path, manifest = _write_archive_and_manifest(tmp_path)
    referenced = "reports/storage/csv_quarantine/daily/000001.SZ.csv"
    latest = tmp_path / "data" / "parquet" / "cn" / "_latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(
        json.dumps({"manifest_path": "data/parquet/cn/manifest.json"}),
        encoding="utf-8",
    )
    runtime_manifest = tmp_path / "data" / "parquet" / "cn" / "manifest.json"
    runtime_manifest.write_text(
        json.dumps({"restore_source": referenced}),
        encoding="utf-8",
    )
    strategy_note = (
        tmp_path
        / "results"
        / "strategy_records"
        / "CN"
        / "note.md"
    )
    strategy_note.parent.mkdir(parents=True)
    strategy_note.write_text(f"manual reference: {referenced}\n", encoding="utf-8")

    report = build_archive_gate_report(manifest, repo_root=tmp_path)
    result = report["result"]

    assert result["delete_allowed"] is False
    assert report["summary"]["gate_status"] == "blocked"
    assert "runtime_reference_check" in result["failed_checks"]
    assert "strategy_record_reference_check" in result["failed_checks"]
    assert result["runtime_references"][referenced] == [
        "data/parquet/cn/manifest.json"
    ]
    assert result["strategy_references"][referenced] == [
        "results/strategy_records/CN/note.md"
    ]


def test_archive_gate_blocks_hash_mismatch(tmp_path):
    _archive_path, manifest = _write_archive_and_manifest(tmp_path)
    manifest["archive_sha256"] = "0" * 64

    report = build_archive_gate_report(manifest, repo_root=tmp_path)
    result = report["result"]

    assert result["delete_allowed"] is False
    assert "archive_sha256_matches_manifest" in result["failed_checks"]
    assert "archive_sha256_mismatch" in result["blockers"]


def test_archive_gate_never_allows_retirement_evidence_root_deletion(tmp_path):
    _archive_path, manifest = _write_archive_and_manifest(tmp_path)
    manifest["source_root"] = "reports/daily"

    report = build_archive_gate_report(manifest, repo_root=tmp_path)
    result = report["result"]

    assert result["delete_allowed"] is False
    assert "retirement_evidence_protection_check" in result["failed_checks"]
    assert "source_is_protected_retirement_evidence" in result["blockers"]


def test_archive_gate_never_allows_parent_of_retirement_evidence(tmp_path):
    _archive_path, manifest = _write_archive_and_manifest(tmp_path)
    manifest["source_root"] = "reports"

    result = build_archive_gate_report(manifest, repo_root=tmp_path)["result"]

    assert result["delete_allowed"] is False
    assert "source_is_protected_retirement_evidence" in result["blockers"]


def test_archive_gate_canonicalizes_protected_source_root(tmp_path):
    _archive_path, manifest = _write_archive_and_manifest(tmp_path)
    protected = tmp_path / "reports" / "daily"
    protected.mkdir()
    (protected / "history.md").write_text("history", encoding="utf-8")
    manifest["source_root"] = "reports/tmp/../daily"

    result = build_archive_gate_report(manifest, repo_root=tmp_path)["result"]

    assert result["delete_allowed"] is False
    assert "source_is_protected_retirement_evidence" in result["blockers"]


def test_archive_gate_writes_reports_and_cli_output(tmp_path, capsys):
    _archive_path, manifest = _write_archive_and_manifest(tmp_path)
    manifest_path = tmp_path / "restore_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "reports" / "archive_gate"

    written = write_archive_gate_report(
        manifest_path,
        root=tmp_path,
        output_dir=output_dir,
    )

    payload = json.loads(
        (output_dir / "data_cleanup_archive_gate.json").read_text()
    )
    assert written["json"] == str(output_dir / "data_cleanup_archive_gate.json")
    assert payload["summary"]["delete_allowed"] is True
    markdown = (output_dir / "data_cleanup_archive_gate.md").read_text()
    assert markdown.startswith("# Data Cleanup Archive Gate")

    exit_code = archive_gate_main(
        [
            "--root",
            str(tmp_path),
            "--manifest-json",
            str(manifest_path),
            "--output-dir",
            str(output_dir / "cli"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "data cleanup archive gate mode: dry-run" in stdout
    assert "delete allowed: True" in stdout

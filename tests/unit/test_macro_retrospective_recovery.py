from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.macro.retrospective_recovery import (
    MacroRetrospectiveRecoveryError,
    build_retrospective_market_projections,
)


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _fixture(tmp_path: Path):
    attempt = tmp_path / "attempt"
    scope = attempt / "scope.json"
    scope_sha = _write(scope, {"full_a": ["000001.SZ", "000002.SZ", "000003.SZ"]})
    sessions = []
    for target in ("20260818", "20260819", "20260820"):
        evidence = {
            "schema_version": "cn-daily-pit-classification-evidence.v1",
            "target_trade_date": target,
            "pit_binding": {},
            "reason_sets": {},
        }
        _write(attempt / f"pit-classification-evidence-{target}.json", evidence)
        sessions.append(
            {
                "trade_date": target,
                "classification": {
                    "status": "PASSED",
                    "classification_sets_disjoint": True,
                    "expected_scope_count": 3,
                    "coverage_complete_count": 3,
                    "observed_bar_count": 2,
                    "counts": {"observed": 2},
                    "symbols": {
                        "observed": ["000001.SZ", "000002.SZ"],
                        "suspended": ["000003.SZ"],
                        "inactive": [],
                        "delisted": [],
                        "prelisting": [],
                        "non_trading": [],
                    },
                },
            }
        )
    capture_path = attempt / "market_capture/manifest.json"
    capture = {
        "target_trade_dates": ["20260818", "20260819", "20260820"],
        "sessions": sessions,
    }
    capture_sha = _write(capture_path, capture)
    source_path = tmp_path / "data/parquet/cn/_snapshots/source.json"
    source = {
        "snapshot_id": "20260820T180000Z",
        "market": "CN",
        "status": "OK",
        "readback_validated": True,
        "blockers": [],
        "manifest_path": str(source_path),
        "table_root": str(tmp_path / "data/parquet/cn/_snapshots/source/table/bars"),
        "coverage": {
            "coverage_schema_version": "cn-full-a-coverage.v4",
            "expected_scope_sha256": scope_sha,
            "pit_generation_id": "pit-test",
            "pit_generation_manifest_path": "pit.json",
            "pit_generation_manifest_sha256": "1" * 64,
            "pit_membership_path": "membership.parquet",
            "pit_membership_sha256": "2" * 64,
            "categories_checked": ["full_a"],
        },
        "metadata": {"capture_manifest_sha256": capture_sha},
    }
    source_sha = _write(source_path, source)
    return attempt, source_path, source_sha, capture_path, capture_sha


def test_retrospective_projection_is_idempotent_and_non_authorizing(tmp_path: Path) -> None:
    attempt, source_path, source_sha, capture_path, capture_sha = _fixture(tmp_path)
    reconstructed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    kwargs = {
        "source_snapshot_manifest_path": source_path,
        "expected_source_snapshot_sha256": source_sha,
        "capture_manifest_path": capture_path,
        "expected_capture_manifest_sha256": capture_sha,
        "attempt_root": attempt,
        "reconstructed_at": reconstructed_at,
        "output_root": tmp_path / "candidates",
    }
    first = build_retrospective_market_projections(**kwargs)
    second = build_retrospective_market_projections(**kwargs)
    assert first == second
    assert first["classification"] == "RETROSPECTIVE_RECONSTRUCTION"
    assert first["canonical_pointer_write"] is False
    assert first["market_pointer_write"] is False
    assert first["pit_pointer_write"] is False
    assert [row["target_trade_date"] for row in first["projections"]] == [
        "20260818",
        "20260819",
        "20260820",
    ]
    for row in first["projections"]:
        projection = json.loads(Path(row["path"]).read_text())
        assert projection["latest_complete_trade_date"] == row["target_trade_date"]
        assert (
            projection["retrospective_reconstruction"]["classification"]
            == "RETROSPECTIVE_RECONSTRUCTION"
        )


def test_retrospective_projection_rejects_source_sha_mismatch(tmp_path: Path) -> None:
    attempt, source_path, _source_sha, capture_path, capture_sha = _fixture(tmp_path)
    with pytest.raises(MacroRetrospectiveRecoveryError, match="source_snapshot_sha_mismatch"):
        build_retrospective_market_projections(
            source_snapshot_manifest_path=source_path,
            expected_source_snapshot_sha256="f" * 64,
            capture_manifest_path=capture_path,
            expected_capture_manifest_sha256=capture_sha,
            attempt_root=attempt,
            reconstructed_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            output_root=tmp_path / "candidates",
        )


def test_retrospective_projection_rejects_backdated_reconstruction(tmp_path: Path) -> None:
    attempt, source_path, source_sha, capture_path, capture_sha = _fixture(tmp_path)
    with pytest.raises(MacroRetrospectiveRecoveryError, match="not_current"):
        build_retrospective_market_projections(
            source_snapshot_manifest_path=source_path,
            expected_source_snapshot_sha256=source_sha,
            capture_manifest_path=capture_path,
            expected_capture_manifest_sha256=capture_sha,
            attempt_root=attempt,
            reconstructed_at="2026-08-18T00:00:00+00:00",
            output_root=tmp_path / "candidates",
        )

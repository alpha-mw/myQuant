from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_investor.macro import maintenance
from quant_investor.macro.release_calendar import ReleaseCalendarIdentity


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    generation = tmp_path / "release-parent"
    raw = generation / "raw"
    raw.mkdir(parents=True)
    (generation / "plan.json").write_text("{}", encoding="utf-8")
    (generation / "market_open_days.json").write_text(
        json.dumps({"schema_version": "market-open-days.v1", "market": "CN", "open_dates": ["20260805"]}),
        encoding="utf-8",
    )
    (generation / "capture_manifest.json").write_text(
        json.dumps({
            "captured_at": "2026-08-04T17:00:00+00:00",
            "issuer_coverage": [
                {"issuer": "nbs_official", "through": "2026-08-04T17:00:00+00:00", "source_ids": ["old-nbs"]},
                {"issuer": "pbc_official", "through": "2026-08-04T17:00:00+00:00", "source_ids": ["old-pbc"]},
            ],
            "sources": [], "events": [], "resolutions": [],
        }),
        encoding="utf-8",
    )
    inputs = []
    for name in ("snapshot.json", "coverage.json", "scope.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        inputs.append((path, _sha(path)))
    identity = ReleaseCalendarIdentity(
        pointer_path=str(tmp_path / "release" / "_latest.json"),
        pointer_sha256="a" * 64,
        generation_id="parent", generation_path=str(generation),
        manifest_sha256="b" * 64, semantic_sha256="c" * 64,
        parent_generation_id="", parent_pointer_sha256="",
        parent_manifest_sha256="", parent_semantic_sha256="",
    )
    release = SimpleNamespace(identity=identity)
    return release, inputs


def test_module_never_imports_retired_macro_mart() -> None:
    source = Path(maintenance.__file__).read_text(encoding="utf-8")
    assert "quant_investor.market.macro_mart" not in source


def test_dry_run_is_zero_write_and_commit_requires_live(tmp_path: Path, monkeypatch) -> None:
    release, inputs = _fixture(tmp_path)
    monkeypatch.setattr(maintenance, "load_release_calendar", lambda **_: release)
    monkeypatch.setattr(maintenance, "observation_pointer_sha256", lambda _: "d" * 64)
    kwargs = dict(
        market="CN", target_date="20260805",
        snapshot_manifest_path=inputs[0][0], expected_snapshot_manifest_sha256=inputs[0][1],
        coverage_manifest_path=inputs[1][0], expected_coverage_manifest_sha256=inputs[1][1],
        scope_artifact_path=inputs[2][0], expected_scope_artifact_sha256=inputs[2][1],
        release_root=tmp_path, expected_release_pointer_sha256="a" * 64,
        observations_root=tmp_path, expected_observations_pointer_sha256="d" * 64,
        release_run_id="release-child", observations_run_id="observations-child",
    )
    assert maintenance.run_cn_macro_maintenance(**kwargs)["status"] == "DRY_RUN_OK"
    with pytest.raises(maintenance.MacroMaintenanceError, match="allow_live_required"):
        maintenance.run_cn_macro_maintenance(**kwargs, commit=True)


def test_live_capture_is_hash_bound_before_both_publishers(tmp_path: Path, monkeypatch) -> None:
    release, inputs = _fixture(tmp_path)
    monkeypatch.setattr(maintenance, "load_release_calendar", lambda **_: release)
    monkeypatch.setattr(maintenance, "observation_pointer_sha256", lambda _: "d" * 64)
    captured = {}

    def fake_release_publish(**kwargs):
        payload = json.loads(Path(kwargs["capture_manifest_path"]).read_text(encoding="utf-8"))
        captured["payload"] = payload
        child = tmp_path / "release-child"
        child.mkdir()
        (child / "market_open_days.json").write_text(
            json.dumps({"schema_version": "market-open-days.v1", "market": "CN", "open_dates": ["20260805"]}),
            encoding="utf-8",
        )
        identity = ReleaseCalendarIdentity(**{**release.identity.__dict__, "generation_path": str(child), "generation_id": "release-child"})
        evidence = SimpleNamespace(open_dates=("20260805",), market_open_days_sha256=_sha(child / "market_open_days.json"))
        return SimpleNamespace(identity=identity, evidence=evidence)

    monkeypatch.setattr(maintenance, "publish_release_calendar", fake_release_publish)
    monkeypatch.setattr(maintenance, "publish_local_market_breadth_roll", lambda **_: {"status": "OK"})
    clocks = {"nbs_official": "2026-08-05T17:00:01+00:00", "pbc_official": "2026-08-05T17:00:02+00:00"}
    result = maintenance.run_cn_macro_maintenance(
        market="CN", target_date="20260805",
        snapshot_manifest_path=inputs[0][0], expected_snapshot_manifest_sha256=inputs[0][1],
        coverage_manifest_path=inputs[1][0], expected_coverage_manifest_sha256=inputs[1][1],
        scope_artifact_path=inputs[2][0], expected_scope_artifact_sha256=inputs[2][1],
        release_root=tmp_path, expected_release_pointer_sha256="a" * 64,
        observations_root=tmp_path, expected_observations_pointer_sha256="d" * 64,
        release_run_id="release-child", observations_run_id="observations-child",
        allow_live=True, commit=True,
        fetcher=lambda _url, issuer: (f"{issuer}-response".encode(), clocks[issuer]),
    )
    assert result["status"] == "OK"
    assert result["cutoff_at"] == clocks["pbc_official"]
    assert {row["artifact_kind"] for row in captured["payload"]["sources"]} == {
        "coverage_response", "coverage_receipt"
    }

"""The default maintenance producer hands native immutable inputs to research."""

from functools import partial
import hashlib
import json
from types import SimpleNamespace

import pytest
from quant_investor.cli import unified
from quant_investor.market import fundamental_generation as native
from quant_investor.market.daily_fundamental_source import retain_fundamental_source
from quant_investor.market.daily_maintenance import _fundamental_health, DailyMaintenanceError
from test_fundamental_generation_promotion import _publish_verified_primary


def context(workspace, mode="execute"):
    attempt = workspace / "results/operations/maintenance/attempt"
    attempt.mkdir(parents=True, mode=0o700)
    return SimpleNamespace(workspace_root=workspace, attempt_root=attempt, mode=mode)


def test_native_health_handoff_survives_current_pointer_removal(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("live network attempted")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    root = tmp_path / "data/parquet/cn"
    _publish_verified_primary(root)
    pointer = root / native.FUNDAMENTAL_POINTER_FILENAME
    original = pointer.read_bytes()
    ctx = context(tmp_path)
    result = _fundamental_health(ctx)
    assert result["status"] == "READY" and result["write_performed"] is False
    ref = result["evidence"]["research_source_ref"]
    raw = (tmp_path / ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    descriptor = json.loads(raw)
    assert (tmp_path / descriptor["pointer"]["path"]).read_bytes() == original
    assert pointer.read_bytes() == original
    pointer.unlink()
    frame, source = unified._daily_fundamental_source(
        {"fundamental_source": descriptor},
        partial(unified._daily_source_file, tmp_path),
        workspace=tmp_path,
    )
    assert not frame.empty
    assert source["available_at"] == "2024-05-11T00:00:00Z"
    assert result["evidence"]["research_evidence_write_performed"] is True


def test_health_shadow_has_no_source_writes(tmp_path):
    _publish_verified_primary(tmp_path / "data/parquet/cn")
    ctx = context(tmp_path, "shadow")
    result = _fundamental_health(ctx)
    assert result["status"] == "READY"
    assert "research_source_ref" not in result["evidence"]
    assert list(ctx.attempt_root.iterdir()) == []


def test_changed_binding_is_rejected_before_writing(tmp_path):
    _publish_verified_primary(tmp_path / "data/parquet/cn")
    ctx = context(tmp_path)
    with pytest.raises(DailyMaintenanceError, match="BINDING_CHANGED"):
        retain_fundamental_source(ctx, expected_binding={})
    assert list(ctx.attempt_root.iterdir()) == []


def test_pointer_race_does_not_publish_a_source(tmp_path, monkeypatch):
    root = tmp_path / "data/parquet/cn"
    _publish_verified_primary(root)
    expected = native.load_fundamental_binding(root)
    ctx = context(tmp_path)
    original = native._stable_file_bytes
    reads = 0

    def changed(path):
        nonlocal reads
        raw, signature = original(path)
        if path.name == native.FUNDAMENTAL_POINTER_FILENAME:
            reads += 1
            if reads > 1:
                return raw + b"\n", signature
        return raw, signature

    monkeypatch.setattr(native, "_stable_file_bytes", changed)
    with pytest.raises(DailyMaintenanceError, match="POINTER_CHANGED"):
        retain_fundamental_source(ctx, expected_binding=expected)
    assert list(ctx.attempt_root.iterdir()) == []


def test_source_failure_does_not_claim_published_or_change_health(tmp_path, monkeypatch):
    _publish_verified_primary(tmp_path / "data/parquet/cn")
    ctx = context(tmp_path)

    def fail(*args, **kwargs):
        raise DailyMaintenanceError("FUNDAMENTAL_SOURCE_TIME_UNAVAILABLE")

    monkeypatch.setattr(
        "quant_investor.market.daily_fundamental_source.retain_fundamental_source", fail
    )
    result = _fundamental_health(ctx)
    assert result["status"] == "READY"
    assert "research_source_ref" not in result["evidence"]
    assert result["evidence"]["research_source_blockers"]
    assert result["evidence"]["research_evidence_write_status"] == "UNCONFIRMED"
    assert result["evidence"]["research_source_error_code"] == "FUNDAMENTAL_SOURCE_TIME_UNAVAILABLE"


def test_native_stage_source_materializes_and_replays_without_current_pointer(tmp_path):
    """Native producer/consumer boundary; unrelated DAG core uses explicit fixtures."""
    from quant_investor.market.daily_maintenance import (
        MaintenanceContext,
        _run_component,
        _stage_record,
    )
    from quant_investor.operations.research_materialization import publish_research_input
    from test_daily_evidence_research_materialization import context as controls

    root = tmp_path / "data/parquet/cn"
    _publish_verified_primary(root)
    journal, recovered, put = controls(tmp_path)
    core = put("maintenance/attempts/one/core-completion.json", {"fixture": "core"})
    attempt = tmp_path / "maintenance/attempts/one"
    ctx = MaintenanceContext(
        workspace_root=tmp_path,
        run_root=attempt.parent,
        attempt_root=attempt,
        target_date="20260904",
        attempt_slot="one",
        mode="execute",
        close_session_receipt={},
        close_session_receipt_path=attempt / "unused.json",
        close_session_receipt_sha256="a" * 64,
    )
    result = _run_component(
        stage="FUNDAMENTAL", callback=_fundamental_health, context=ctx, prior_results=[]
    )
    assert result["status"] == "READY"
    recorded = _stage_record(attempt, "FUNDAMENTAL", result)
    stage_ref = {
        "path": str((attempt / "stage-FUNDAMENTAL.json").relative_to(tmp_path)),
        "sha256": recorded["sha256"],
    }
    stage = json.loads((tmp_path / stage_ref["path"]).read_bytes())
    recovered["recipe"]["research_sources"]["fundamental"] = {
        "mode": "MAINTENANCE_STAGE",
        "source_ref": None,
    }
    recovered["handoff"]["recipe_ref"] = put("recipe.json", recovered["recipe"])
    recovered["handoff"]["maintenance_core_ref"] = core
    recovered["handoff_ref"] = put(recovered["handoff_ref"]["path"], recovered["handoff"])
    auxiliary = {
        "stages": {"fundamental": {"state": "RECORDED", "ref": stage_ref, "document": stage}}
    }
    (root / native.FUNDAMENTAL_POINTER_FILENAME).unlink()
    with journal.locked():
        published = publish_research_input(
            journal=journal, recovered=recovered, auxiliary=auxiliary
        )
    request = json.loads((tmp_path / published["research_request_ref"]["path"]).read_bytes())
    assert published["auxiliary_stage_refs"]["fundamental"] == stage_ref
    frame, _ = unified._daily_fundamental_source(
        request["company_evidence"],
        partial(unified._daily_source_file, tmp_path),
        workspace=tmp_path,
    )
    assert not frame.empty
    replayed = publish_research_input(
        journal=journal, recovered=recovered, auxiliary=auxiliary, verify_only=True
    )
    assert replayed["research_request_ref"] == published["research_request_ref"]

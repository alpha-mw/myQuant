"""Positive native Macro readiness using shared typed Market/PIT, synthetic time."""

import pytest

from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
from _native_shared_macro_fixture import build_shared_macro


@pytest.mark.parametrize("count", [100, 3000])
def test_shared_market_pit_native_macro_ready(tmp_path, count):
    fixture = NativeFactorInputs(tmp_path / "inputs", count=count)
    for offset in range(4):
        args = fixture.day(offset)
        day = args["as_of"]
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        reader, pit = strict_market_from_factor_inputs(
            tmp_path,
            args,
            macro_ready_layout=True,
            pit_observed_at=iso + "T00:00:00Z",
            simulated_available_at=iso + "T07:30:00Z",
        )
    result = build_shared_macro(tmp_path, "2026-08-27")
    assert result["verification"]["target_date"] == "20260827"
    assert (
        result["closure"]["frozen_pointers"]["pit"]["snapshot"]["generation_id"]
        == pit["generation_id"]
    )
    assert (
        result["closure"]["frozen_pointers"]["market"]["snapshot"]["snapshot_id"]
        == reader.snapshot()["snapshot_id"]
    )
    assert result["real_time_oos_eligible"] is False


@pytest.mark.parametrize("cohort", [100, 3000])
def test_five_native_macro_successors_keep_parent_history(tmp_path, cohort):
    from quant_investor.macro.readiness_closure import validate_macro_readiness_closure
    from quant_investor.macro.store import load_observations

    fixture = NativeFactorInputs(tmp_path / "inputs", count=cohort, extra_future_sessions=3)
    closures = []
    previous_generation = None
    for offset in range(8):
        args = fixture.day(offset)
        day = args["as_of"]
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        strict_market_from_factor_inputs(
            tmp_path,
            args,
            macro_ready_layout=True,
            pit_observed_at=iso + "T00:00:00Z",
            simulated_available_at=iso + "T07:30:00Z",
        )
        if offset < 3:
            continue
        result = build_shared_macro(tmp_path, iso)
        assert result["verification"]["target_date"] == day
        rows, pointer = load_observations(tmp_path / "data/parquet/cn/macro_observations")
        assert len(rows) == 39
        if closures:
            assert pointer["generation_manifest"]["parent_generation_id"] == previous_generation
        previous_generation = pointer["generation_id"]
        closures.append(result["closure"])
    assert len(closures) == 5
    for closure in closures:
        assert validate_macro_readiness_closure(workspace_root=tmp_path, closure=closure) == closure


def test_native_macro_archives_bound_veto_after_commit(tmp_path):
    from datetime import datetime, timezone
    from quant_investor.market.daily_maintenance import _write_veto
    from quant_investor.macro.readiness_closure import VETO_PATH

    fixture = NativeFactorInputs(tmp_path / "inputs", count=100)
    for offset in range(4):
        args = fixture.day(offset)
        day = args["as_of"]
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        strict_market_from_factor_inputs(
            tmp_path,
            args,
            macro_ready_layout=True,
            pit_observed_at=iso + "T00:00:00Z",
            simulated_available_at=iso + "T07:30:00Z",
        )
    veto = tmp_path / VETO_PATH
    veto.parent.mkdir(parents=True, mode=0o700)
    _, digest = _write_veto(
        veto.parent,
        {
            "schema_version": "cn-daily-maintenance-macro-write-veto.v1",
            "created_at": "2026-08-27T13:00:00Z",
            "target_date": "20260827",
            "blockers": ["MACRO_RELEASE_COMPONENT_NOT_REGISTERED"],
        },
        filename=veto.name,
    )
    result = build_shared_macro(tmp_path, "2026-08-27")
    lifecycle = result["closure"]["veto_lifecycle"]
    assert lifecycle["state"] == "CLEARED" and lifecycle["original_veto_sha256"] == digest
    assert not veto.exists()
    assert (tmp_path / lifecycle["archive_ref"]["path"]).is_file()
    assert datetime.fromisoformat(result["closure"]["available_at"]) >= datetime(
        2026, 8, 27, 13, 10, tzinfo=timezone.utc
    )

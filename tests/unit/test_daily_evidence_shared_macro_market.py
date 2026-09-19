"""Native Macro local observation reads the same strict large Market/PIT fixture."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
from quant_investor.macro.local_market_observations import compile_local_market_breadth_observation


def test_native_macro_reads_shared_3000_market_without_timestamp_backfill(tmp_path):
    fixture = NativeFactorInputs(tmp_path / "inputs")
    args = fixture.day(3)
    reader, pit = strict_market_from_factor_inputs(
        tmp_path, args, pit_observed_at="2026-08-27T00:00:00Z", macro_ready_layout=True
    )
    snapshot = reader.snapshot()
    assert snapshot["healthy"]
    manifest = Path(snapshot["manifest_path"])
    scope = tmp_path / "data/cn_universe/cn_index_components.json"

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    before = (sha(manifest), sha(scope))
    observation, evidence = compile_local_market_breadth_observation(
        snapshot_manifest_path=manifest,
        expected_snapshot_manifest_sha256=sha(manifest),
        coverage_manifest_path=manifest,
        expected_coverage_manifest_sha256=sha(manifest),
        target_trade_date="20260827",
        scope_artifact_path=scope,
        expected_scope_artifact_sha256=sha(scope),
        as_of=datetime.now(timezone.utc),
    )
    assert observation.value == 100.0
    assert before == (sha(manifest), sha(scope))
    assert observation.period_end == "2026-08-27"


def test_explicit_synthetic_availability_is_marked_and_cannot_touch_unmarked_data(tmp_path):
    import json
    import pytest

    fixture = NativeFactorInputs(tmp_path / "inputs", count=100)
    args = fixture.day(3)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "data").mkdir()
    with pytest.raises(ValueError, match="unmarked"):
        strict_market_from_factor_inputs(
            existing, args, macro_ready_layout=True, simulated_available_at="2026-08-27T07:30:00Z"
        )
    fresh = tmp_path / "simulation"
    fresh.mkdir()
    reader, _ = strict_market_from_factor_inputs(
        fresh,
        args,
        macro_ready_layout=True,
        pit_observed_at="2026-08-27T00:00:00Z",
        simulated_available_at="2026-08-27T07:30:00Z",
    )
    manifest = Path(reader.snapshot()["manifest_path"])
    scope = fresh / "data/cn_universe/cn_index_components.json"

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    observation, _ = compile_local_market_breadth_observation(
        snapshot_manifest_path=manifest,
        expected_snapshot_manifest_sha256=sha(manifest),
        coverage_manifest_path=manifest,
        expected_coverage_manifest_sha256=sha(manifest),
        target_trade_date="20260827",
        scope_artifact_path=scope,
        expected_scope_artifact_sha256=sha(scope),
        as_of="2026-08-27T13:30:00Z",
    )
    assert observation.value == 100.0
    provenance = json.loads(manifest.read_bytes())["metadata"]["synthetic_time"]
    assert provenance["synthetic"] and provenance["real_time_oos_eligible"] is False
    assert datetime.fromisoformat(provenance["wall_clock_created_at"]) > datetime(
        2026, 8, 27, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError, match="already exists"):
        strict_market_from_factor_inputs(
            fresh, args, macro_ready_layout=True, simulated_available_at="2026-08-27T07:30:00Z"
        )

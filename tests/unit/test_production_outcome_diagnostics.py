"""Behavioral checks for non-authorizing raw-close observation diagnostics."""

from datetime import datetime, timezone

from quant_investor.factors.production_outcomes import classify_seal, evaluate_cross_section


def test_raw_close_returns_preserve_missing_original_denominator():
    signals = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}
    prices = {"A": (10, 11), "B": (10, 12), "C": (10, None), "D": (0, 12), "E": (10, 15)}
    result = evaluate_cross_section(signals, prices)
    assert result["denominator"] == 5
    assert result["joint_count"] == 3
    assert len(result["securities"]) == 5
    assert result["securities"]["C"]["reason"] == "END_CLOSE_MISSING"
    assert result["securities"]["D"]["reason"] == "ORIGIN_CLOSE_INVALID"
    assert abs(float(result["ic"]["value"]) - 1) < 1e-12
    assert result["economic_return"]["state"] == "UNAVAILABLE"
    assert result["executable_return"]["state"] == "UNAVAILABLE"
    assert result["cost_result"]["state"] == "UNAVAILABLE"


def test_constant_signal_has_coverage_but_unavailable_correlation():
    result = evaluate_cross_section({"A": "1", "B": "1"}, {"A": (10, 11), "B": (10, 12)})
    assert result["joint_count"] == 2
    assert result["rank_ic"]["state"] == "UNAVAILABLE"


def test_post_close_and_after_outcome_seals_never_become_original_close_prospective():
    late = classify_seal(
        signal_date="20260820",
        seal_time_upper_bound="2026-08-20T12:00:00Z",
        registered_at="2026-08-25T06:00:00Z",
    )
    after = classify_seal(
        signal_date="20260820",
        seal_time_upper_bound="2026-08-25T05:00:00Z",
        registered_at="2026-08-25T06:00:00Z",
    )
    assert late["prospective_eligible"] is False
    assert after["prospective_eligible"] is False
    assert late["registration_time"] == "2026-08-25T06:00:00Z"


def test_recovered_registration_uses_original_provable_seal_time():
    result = classify_seal(
        signal_date="20260820",
        seal_time_upper_bound="2026-08-20T06:00:00Z",
        registered_at="2026-08-25T06:00:00Z",
    )
    assert result["cohort"] == "ORIGINAL_CLOSE_COHORT"
    assert result["prospective_eligible"] is True
    assert result["registration_time"] == "2026-08-25T06:00:00Z"


def test_unknown_seal_time_is_not_inferred_from_registration_or_mtime():
    result = classify_seal(
        signal_date="20260820", seal_time_upper_bound=None, registered_at="2026-08-20T06:00:00Z"
    )
    assert result["cohort"] == "SEAL_TIME_UNPROVEN"
    assert result["prospective_eligible"] is False


def _settlement_fixture(
    tmp_path, monkeypatch, *, registered_at="2026-08-20T13:00:00Z", seal_time="2026-08-20T12:00:00Z"
):
    from quant_investor.factors import production_outcomes as module
    from quant_investor.factors.production_authority import FactorProductionStore
    from quant_investor.factors.production_observation import (
        build_factor_production_observation,
        _observation_path,
    )
    from quant_investor.contracts import canonical_json_bytes
    from test_unified_factor_production_observation import _inputs

    inputs = _inputs()
    inputs["factor_rows"] = [dict(inputs["factor_rows"][0], symbol_count=2)]
    factor_id = inputs["factor_rows"][0]["factor_id"]
    inputs.update(
        {
            "signal_values": {factor_id: {"A": (1.0).hex(), "B": (2.0).hex()}},
            "active_factor_rows": [],
            "factor_implementation_refs": [],
            "factor_policy_ref": {"sha256": "f" * 64},
            "source_generation_ref": {},
            "seal_time_upper_bound": seal_time,
        }
    )
    store = FactorProductionStore(tmp_path)
    observation = build_factor_production_observation(
        inputs=inputs, factor_row=inputs["factor_rows"][0], registered_at=registered_at
    )
    path = _observation_path("20260820", "LOW")
    original = canonical_json_bytes(observation)
    store.write_exact_once(path, original)
    calendar = {
        "target_trade_date": "20260825",
        "calendar_start_date": "20260820",
        "calendar_end_date": "20260825",
        "ordered_open_dates": ["20260820", "20260821", "20260824", "20260825"],
    }
    market = {
        "prices": {"A": {"20260820": "10", "20260821": "11"}, "B": {"20260820": "10"}},
        "market_date": "20260821",
        "source_family": "CN_CANONICAL_RAW_CLOSE",
        "manifest_ref": {},
        "file_refs": [],
        "pointer_bytes": "{}",
        "pointer_sha256": "c" * 64,
    }
    monkeypatch.setattr(FactorProductionStore, "read_observation_history", lambda self: [inputs])
    monkeypatch.setattr(module, "close_calendar", lambda *args: calendar)
    monkeypatch.setattr(module, "signal_calendar", lambda *args: ["20260820"])
    monkeypatch.setattr(module, "load_market_slice", lambda *args, **kwargs: market)
    args = {
        "workspace_root": str(tmp_path),
        "calendar_receipt": str(tmp_path / "fixture-calendar"),
        "expected_calendar_sha256": "d" * 64,
    }
    return module, store, path, original, market, args


def test_one_day_settles_independently_of_sixty_days_and_missing_symbols(tmp_path, monkeypatch):
    module, store, path, original, market, args = _settlement_fixture(tmp_path, monkeypatch)
    result = module.settle_production_observations(**args)
    assert result["horizon_states"]["1"] == {"EVALUATED": 1}
    assert result["horizon_states"]["60"] == {"WAITING": 1}
    one = next(row for row in result["outcomes"] if row["horizon"] == 1)
    assert one["metrics"]["denominator"] == 2
    assert one["metrics"]["joint_count"] == 1
    assert one["metrics"]["ic"]["state"] == "UNAVAILABLE"
    assert store.read(path).data == original
    assert result["provider_calls"] is False
    assert result["paper_comparison"]["state"] == "UNAVAILABLE"
    assert module.settle_production_observations(**args)["new_evaluation_count"] == 0


def test_mature_but_missing_endpoint_is_data_pending_not_waiting(tmp_path, monkeypatch):
    module, store, path, original, market, args = _settlement_fixture(tmp_path, monkeypatch)
    market["market_date"] = "20260820"
    result = module.settle_production_observations(**args)
    assert result["horizon_states"]["1"] == {"DATA_PENDING": 1}
    assert result["new_evaluation_count"] == 0


def test_processing_index_loss_rebuilds_from_evidence_without_double_count(tmp_path, monkeypatch):
    module, store, path, original, market, args = _settlement_fixture(tmp_path, monkeypatch)
    first = module.settle_production_observations(**args)
    assert first["new_evaluation_count"] == 1
    (tmp_path / "results/factors/outcome-processing.json").unlink()
    second = module.settle_production_observations(**args)
    assert second["new_evaluation_count"] == 0
    assert [r.get("evaluation_ref") for r in first["outcomes"]] == [
        r.get("evaluation_ref") for r in second["outcomes"]
    ]


def test_calendar_disagreement_is_invalid_and_does_not_rewrite_observation(tmp_path, monkeypatch):
    module, store, path, original, market, args = _settlement_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "signal_calendar", lambda *args: ["20260819", "20260821"])
    result = module.settle_production_observations(**args)
    assert result["errors"][0]["state"] == "INVALID"
    assert result["new_evaluation_count"] == 0
    assert store.read(path).data == original


def test_source_preserves_exact_pretty_printed_market_pointer_bytes(tmp_path):
    import json, base64
    import pyarrow as pa
    import pyarrow.parquet as pq
    from quant_investor.factors.production_authority import FactorProductionStore
    from quant_investor.factors.production_outcome_sources import load_market_slice, persist_source

    data = tmp_path / "data/parquet/cn"
    snapshot_id = "20260821T130000Z"
    table = data / "_snapshots" / snapshot_id / "table/bars"
    part = table / "year=2026/month=08/part.parquet"
    part.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"ts_code": "000001.SZ", "trade_date": "20260820", "close": 10.0},
                {"ts_code": "000001.SZ", "trade_date": "20260821", "close": 11.0},
            ]
        ),
        part,
    )
    manifest = data / "_snapshots" / (snapshot_id + ".json")
    manifest.write_text(
        json.dumps({"snapshot_id": snapshot_id, "market": "CN", "table_root": str(table)})
    )
    pointer = {
        "snapshot_id": snapshot_id,
        "status": "OK",
        "manifest_path": str(manifest),
        "table_root": str(table),
        "derived_serving_root": str(data / "_snapshots" / snapshot_id / "serving/bars"),
        "latest_complete_trade_date": "20260821",
        "latest_trade_date": "20260821",
    }
    raw = json.dumps(pointer, indent=2).encode()
    (data / "_latest.json").write_bytes(raw)
    source = load_market_slice(tmp_path, symbols=["000001.SZ"], start="20260820", end="20260821")
    assert base64.b64decode(source["pointer_bytes_base64"]) == raw
    assert source["prices"]["000001.SZ"]["20260821"] == "11"
    assert persist_source(FactorProductionStore(tmp_path), source)["sha256"]


def test_daily_report_does_not_call_invalid_horizons_completed(tmp_path):
    from quant_investor.market.daily_factor_loop import DailyFactorLoop

    loop = object.__new__(DailyFactorLoop)
    loop.workspace = tmp_path
    loop.run_root = tmp_path
    loop.installation = {}
    loop.started_at = "2026-09-06T00:00:00Z"
    loop.stages = {
        "historical_settlement": {
            "as_of": "20260904",
            "horizon_states": {"1": {"INVALID": 1}},
            "errors": [],
        }
    }
    result = loop.report(maintenance=None)
    assert result["status"] == "PARTIAL"
    assert result["blockers"] == ["historical_settlement"]


def test_actual_float_hex_generation_signals_produce_numeric_ic():
    from quant_investor.factors.production_outcomes import decode_generation_signals

    encoded = {"A": (-14.0349).hex(), "B": (-13.75).hex(), "C": (-12.125).hex()}
    decoded = decode_generation_signals(encoded)
    assert {k: v.hex() for k, v in decoded.items()} == encoded
    result = evaluate_cross_section(decoded, {"A": (10, 11), "B": (10, 12), "C": (10, 13)})
    assert result["signal_count"] == 3
    assert result["joint_count"] == 3
    assert result["rank_ic"]["state"] == "AVAILABLE"
    assert float(result["rank_ic"]["value"]) == 1
    assert result["missing_reasons"] == {}


def test_noncanonical_generation_signal_is_invalid_before_evaluation():
    import pytest
    from quant_investor.factors.production_outcomes import decode_generation_signals

    with pytest.raises(Exception, match="PRODUCTION_SIGNAL_ENCODING_INVALID"):
        decode_generation_signals({"A": "1.25"})

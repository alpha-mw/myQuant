"""Prepare real native synthetic sources before any target core or request exists."""

import hashlib
import json
from _native_daily_factor_fixture import NativeFactorInputs, strict_market_from_factor_inputs
from _native_initial_sources_fixture import prepare_initial_sources


def test_pre_core_broad_sources_do_not_depend_on_selected_pool(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("initial source preparation attempted network")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    workspace = tmp_path / "factor-workspace"
    fixture = NativeFactorInputs(workspace / "synthetic-inputs", extra_future_sessions=3)
    for offset in range(4):
        args = fixture.day(offset, extra_history=9)
        day = args["as_of"]
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        strict_market_from_factor_inputs(
            workspace,
            args,
            macro_ready_layout=True,
            pit_observed_at=iso + "T00:00:00Z",
            simulated_available_at=iso + "T07:30:00Z",
        )
    sources = prepare_initial_sources(tmp_path, fixture.symbols)
    assert set(sources) == {"industry_source_ref", "exposure_rows_ref", "fundamental", "macro"}
    assert not (workspace / "results/operations/daily_production/CN/20260827").exists()
    assert not (workspace / "initial-execute/request.json").exists()

    def read(ref):
        raw = (workspace / ref["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
        return json.loads(raw)

    industry = read(sources["industry_source_ref"])
    for ref in industry["membership_partitions"]:
        read(ref)
    from functools import partial
    from quant_investor.cli.unified import _daily_industry_projection, _daily_source_document

    projected = _daily_industry_projection(
        {"as_of": "2026-08-27T13:30:00Z", "industry_source": industry},
        sorted(fixture.symbols),
        partial(_daily_source_document, str(workspace)),
    )
    assert projected["payload"]["blocker_codes"] == []
    exposures = read(sources["exposure_rows_ref"])
    assert sorted(row["company_code"] for row in exposures) == sorted(fixture.symbols)
    fundamental = read(sources["fundamental"]["source_ref"])
    assert set(fundamental) == {"available_at", "daily_parquet", "pointer"}
    assert sources["fundamental"]["mode"] == sources["macro"]["mode"] == "PINNED"
    macro = read(sources["macro"]["source_ref"])
    assert macro["classification"] == "CANONICAL_MACRO_READY"
    read(macro["source"])
    receipt = json.loads((tmp_path / "initial-pre-core-sources.json").read_text())
    assert receipt["company_count"] == len(fixture.symbols)
    assert receipt["synthetic"] is True

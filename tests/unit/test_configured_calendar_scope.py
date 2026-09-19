"""The configured fixture routes exact current/future horizons without network."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest
from _native_daily_calendar_fixture import configured_future_calendar_scope
from quant_investor.market import tushare_calendar_authority as native
from quant_investor.market._calendar_fixture_capability import require_fixture_capability
from quant_investor.operations.daily_contract import ContractError


def test_configured_scope_routes_and_pins_future_wire_bytes(tmp_path, monkeypatch):
    workspace = tmp_path / "factor-workspace"
    workspace.mkdir()
    source = tmp_path / "repository/tests/unit/test_tushare_calendar_authority.py"
    source.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("test_tushare_calendar_authority.py"), source)
    (tmp_path / "release-input.json").write_text('{"fixture":true}')
    monkeypatch.setattr("socket.socket.connect", lambda *a: pytest.fail("fixture used network"))
    original = native.OfficialTushareHttpsClient
    with configured_future_calendar_scope(root=tmp_path, trade_date="20260827"):
        cap = require_fixture_capability(workspace=workspace, trade_date="20260827")
        manifest = json.loads(cap.manifest_raw)
        client = native.OfficialTushareHttpsClient()
        for exchange in ("SSE", "SZSE", "BSE"):
            current = client.request(
                api_name="trade_cal", params={"exchange": exchange, "end_date": "20260827"}
            )
            future = client.request(
                api_name="trade_cal", params={"exchange": exchange, "end_date": "20260917"}
            )
            assert (
                hashlib.sha256(future.raw_body).hexdigest()
                == manifest["provider_response_sha256"][exchange]
            )
            if exchange != "BSE":
                assert current.rows[-1][1] == "20260827"
                assert future.rows[-1][1] == "20260917"
                assert current.raw_body != future.raw_body
        with pytest.raises(ContractError, match="PROVENANCE_UNAVAILABLE"):
            client.request(api_name="trade_cal", params={"exchange": "SSE", "end_date": "20260918"})
    assert native.OfficialTushareHttpsClient is original
    assert not list(workspace.iterdir())


def test_future_clock_does_not_backdate_baseline_factor_capture(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from quant_investor.market import _calendar_fixture_capability as capability
    from quant_investor.market import future_calendar_producer as producer

    workspace = tmp_path / "factor-workspace"
    workspace.mkdir()
    source = tmp_path / "repository/tests/unit/test_tushare_calendar_authority.py"
    source.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("test_tushare_calendar_authority.py"), source)
    (tmp_path / "release-input.json").write_text('{"fixture":true}')
    original = native.datetime
    logical = datetime(2026, 8, 27, 13, 20, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return logical.astimezone(tz) if tz else logical.replace(tzinfo=None)

    monkeypatch.setattr(capability, "datetime", Clock)
    calls = []

    def acquire(**kw):
        calls.append(kw)
        return native.datetime.now(timezone.utc)

    monkeypatch.setattr(producer, "capture_next_session_calendar", acquire)
    with configured_future_calendar_scope(root=tmp_path, trade_date="20260827"):
        assert native.datetime is original
        assert producer.capture_next_session_calendar(exact_argument="forwarded") == logical
        assert native.datetime is original
    assert calls == [{"exact_argument": "forwarded"}]
    assert native.datetime is original

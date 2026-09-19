"""Offline recorder tests; injected sessions/route seam are not live provenance."""

from contextlib import contextmanager
import hashlib
import json
import ssl
from types import SimpleNamespace

import pytest

from quant_investor.market import _calendar_production_transport as recorder
from quant_investor.market import _calendar_fixture_capability as fixture
from quant_investor.market import tushare_calendar_authority as native
from quant_investor.market import tushare_transport as transport
from quant_investor.system.errors import SystemSecurityError


@contextmanager
def _test_session(tmp_path, monkeypatch):
    # Explicit controlled seam: tests never publish a production proof/receipt.
    monkeypatch.setattr(recorder, "_guard_route", lambda: None)
    value = recorder._Session(tmp_path, "20260902", {}, {}, {})
    token = recorder._ACTIVE.set(value)
    try:
        yield value
    finally:
        recorder._ACTIVE.reset(token)


def finish_docs():
    ticket = recorder._begin_response("DOCUMENTATION")
    recorder._complete_response(ticket, raw=b"docs", tls_context=ssl.create_default_context())


def request_args(exchange="SSE"):
    return dict(
        api_name="trade_cal",
        params={"start_date": "20231201", "end_date": "20260923", "exchange": exchange},
        expected_fields=native.EXPECTED_FIELDS,
    )


def test_inactive_hooks_do_not_collect_or_change_other_api_calls():
    assert recorder._begin_response("PROVIDER") is None
    recorder._complete_response(None, raw=b"x", tls_context=None)
    with recorder._provider_request_scope(api_name="daily", params={}, expected_fields=[]):
        assert recorder._ACTIVE.get() is None


@pytest.mark.parametrize(
    "replacement", ["connection", "tls", "request", "fetch", "documentation", "clock", "fixture"]
)
def test_production_guard_refuses_substitution_before_install_or_io(
    tmp_path, monkeypatch, replacement
):
    if replacement == "connection":
        monkeypatch.setattr(transport, "_HTTPS_CONNECTION", object())
    elif replacement == "tls":
        monkeypatch.setattr(transport, "_CREATE_DEFAULT_CONTEXT", object())
    elif replacement == "request":
        monkeypatch.setattr(transport.OfficialTushareHttpsClient, "request", object())
    elif replacement == "fetch":
        monkeypatch.setattr(transport.OfficialTushareHttpsClient, "_fetch_raw", object())
    elif replacement == "documentation":
        monkeypatch.setattr(native, "_official_documentation_fetch", object())
    elif replacement == "clock":
        monkeypatch.setattr(native, "datetime", object())
    token = fixture._ACTIVE.set(object()) if replacement == "fixture" else None
    try:
        with pytest.raises(SystemSecurityError, match="TRANSPORT_ROUTE_UNAVAILABLE"):
            with recorder._production_transport_scope(
                workspace=tmp_path,
                day="20260902",
                install_raw=b"",
                install_ref={},
                repository=tmp_path,
            ):
                pytest.fail("scope admitted substitution")
        assert list(tmp_path.iterdir()) == []
        assert recorder._ACTIVE.get() is None
    finally:
        if token is not None:
            fixture._ACTIVE.reset(token)


def test_four_ordered_response_events_are_detached_and_secret_free(tmp_path, monkeypatch):
    with _test_session(tmp_path, monkeypatch) as session:
        finish_docs()
        for exchange in ("SSE", "SZSE", "BSE"):
            with recorder._provider_request_scope(**request_args(exchange)):
                ticket = recorder._begin_response("PROVIDER")
                recorder._complete_response(
                    ticket, raw=exchange.encode(), tls_context=ssl.create_default_context()
                )
        events = recorder._retained_session_events(session)
        assert [x["ordinal"] for x in events] == [0, 1, 2, 3]
        assert events[1]["response_sha256"] == hashlib.sha256(b"SSE").hexdigest()
        assert events[1]["response_bytes"] == 3
        assert {x["http_status"] for x in events} == {200}
        assert "token" not in json.dumps(events).lower()
        events[0]["host"] = "changed"
        assert session.events[0]["host"] == "tushare.pro"
    with pytest.raises(SystemSecurityError, match="EVENT_SET_INVALID"):
        recorder._retained_session_events(session)


@pytest.mark.parametrize(
    "case", ["wrong_exchange", "secret", "incomplete", "failed", "tls", "extra"]
)
def test_invalid_or_incomplete_scope_cannot_supply_evidence(tmp_path, monkeypatch, case):
    with _test_session(tmp_path, monkeypatch) as session:
        finish_docs()
        if case == "incomplete":
            with pytest.raises(SystemSecurityError, match="EVENT_SET_INVALID"):
                recorder._retained_session_events(session)
            return
        args = request_args()
        if case == "wrong_exchange":
            args["params"]["exchange"] = "BSE"
        elif case == "secret":
            args["params"]["token"] = "must-not-be-retained"
        if case in {"wrong_exchange", "secret"}:
            with pytest.raises(SystemSecurityError, match="EVENT_SET_INVALID"):
                with recorder._provider_request_scope(**args):
                    pytest.fail("invalid request admitted")
        elif case == "failed":
            with pytest.raises(OSError):
                with recorder._provider_request_scope(**args):
                    recorder._begin_response("PROVIDER")
                    raise OSError("test network failure")
        elif case == "tls":
            with pytest.raises(SystemSecurityError, match="BINDING_INVALID"):
                with recorder._provider_request_scope(**args):
                    ticket = recorder._begin_response("PROVIDER")
                    recorder._complete_response(
                        ticket,
                        raw=b"x",
                        tls_context=SimpleNamespace(
                            verify_mode=ssl.CERT_NONE, check_hostname=False
                        ),
                    )
        else:
            for exchange in ("SSE", "SZSE", "BSE"):
                with recorder._provider_request_scope(**request_args(exchange)):
                    ticket = recorder._begin_response("PROVIDER")
                    recorder._complete_response(
                        ticket, raw=b"x", tls_context=ssl.create_default_context()
                    )
            with pytest.raises(SystemSecurityError, match="EVENT_SET_INVALID"):
                with recorder._provider_request_scope(**args):
                    pytest.fail("extra request admitted")
        with pytest.raises(SystemSecurityError, match="EVENT_SET_INVALID"):
            recorder._retained_session_events(session)
        assert "must-not-be-retained" not in json.dumps(session.events)
        assert recorder._PENDING.get() is None


def test_real_fetch_hook_records_only_successful_bytes(tmp_path, monkeypatch):
    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return SimpleNamespace(status=200, read=lambda count: b"wire-bytes")

        def close(self):
            pass

    with _test_session(tmp_path, monkeypatch) as session:
        finish_docs()
        monkeypatch.setattr(transport, "_HTTPS_CONNECTION", Connection)
        with recorder._provider_request_scope(**request_args()):
            assert (
                transport.OfficialTushareHttpsClient()._fetch_raw(b"unused-test-body")
                == b"wire-bytes"
            )
        assert session.events[-1]["response_sha256"] == hashlib.sha256(b"wire-bytes").hexdigest()


def test_integrity_error_cannot_be_downgraded_by_native_failure_wrapper(tmp_path, monkeypatch):
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.system import release, release_install
    from quant_investor.system.errors import SystemPreconditionError

    # Explicit installed-verifier unit seam; no network or receipt publication.
    manifest = {
        "files": [{"path": p, "byte_sha256": "a" * 64} for p in sorted(recorder._MODULE_PATHS)]
    }
    monkeypatch.setattr(release, "installed_code_manifest", lambda: manifest)
    monkeypatch.setattr(
        release_install,
        "verify_running_release_install_input",
        lambda *a, **k: {
            "state": "PASS",
            "release_ref": {"kind": "system.release", "byte_sha256": "b" * 64},
            "installed_code_manifest_sha256": hashlib.sha256(
                canonical_json_bytes(manifest)
            ).hexdigest(),
        },
    )
    with pytest.raises(SystemSecurityError, match="TRANSPORT_EVENT_SET_INVALID"):
        with recorder._production_transport_scope(
            workspace=tmp_path,
            day="20260902",
            install_raw=b"unit-only",
            install_ref={"path": "input.json", "sha256": hashlib.sha256(b"unit-only").hexdigest()},
            repository=tmp_path,
        ):
            try:
                recorder._begin_response("PROVIDER")  # Missing docs/staged request.
            except SystemSecurityError:
                raise SystemPreconditionError("simulated native immutable failure wrapper")
    assert recorder._ACTIVE.get() is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", [302, 500])
def test_failed_http_response_never_records_success(tmp_path, monkeypatch, status):
    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return SimpleNamespace(
                status=status, read=lambda count: pytest.fail("body must not be read")
            )

        def close(self):
            pass

    with _test_session(tmp_path, monkeypatch) as session:
        finish_docs()
        monkeypatch.setattr(transport, "_HTTPS_CONNECTION", Connection)
        with pytest.raises(transport.TushareHttpsError):
            with recorder._provider_request_scope(**request_args()):
                transport.OfficialTushareHttpsClient()._fetch_raw(b"unit-only")
        assert len(session.events) == 1
        assert session.failed is True
        with pytest.raises(SystemSecurityError, match="EVENT_SET_INVALID"):
            recorder._retained_session_events(session)

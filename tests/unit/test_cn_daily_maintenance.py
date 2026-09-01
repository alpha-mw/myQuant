from __future__ import annotations

from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from quant_investor.cli import main as cli_main
from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError
from quant_investor.contracts import canonical_json_bytes
from quant_investor.market.close_session_authority import (
    CloseSessionAuthorityError,
    CloseSessionAuthorityResult,
    acquire_close_session_authority,
)
from quant_investor.market import daily_maintenance
from quant_investor.market.daily_maintenance import (
    DailyMaintenanceError,
    MaintenanceComponents,
    clear_cn_daily_write_veto,
    recover_transient_cn_daily_write_veto,
    resolve_attempt_slot,
    run_cn_daily_maintenance,
)
from quant_investor.market.credential_preflight import (
    CredentialPreflightError,
    read_project_env_token,
    write_credential_preflight,
)
from quant_investor.market.tushare_transport import (
    TushareHttpsError,
    TushareTransportDiagnostic,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def _assert_shared_canonical(path: Path) -> dict:
    raw = path.read_bytes()
    parsed = json.loads(raw)
    assert raw == canonical_json_bytes(parsed)
    assert not raw.endswith(b"\n")
    return parsed


def _legacy_maintenance_bytes(value: dict) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


class _CalendarClient:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.calls = []

    def request(self, **kwargs):
        self.calls.append(kwargs)
        raw = b'{"provider":"exact"}'
        return SimpleNamespace(
            api_name="trade_cal",
            request_id="request-id",
            reported_count=len(self.rows),
            has_more=False,
            fields=("exchange", "cal_date", "is_open", "pretrade_date"),
            rows=self.rows,
            raw_body=raw,
            provider_reported_count=len(self.rows),
            item_count=len(self.rows),
        )


def _calendar_rows(start: str = "20260719", end: str = "20260819"):
    first = datetime.strptime(start, "%Y%m%d").date()
    last = datetime.strptime(end, "%Y%m%d").date()
    previous = first - timedelta(days=1)
    while previous.weekday() >= 5:
        previous -= timedelta(days=1)
    rows = []
    current = first
    while current <= last:
        is_open = current.weekday() < 5
        rows.append(
            (
                "SSE",
                current.strftime("%Y%m%d"),
                int(is_open),
                previous.strftime("%Y%m%d"),
            )
        )
        if is_open:
            previous = current
        current += timedelta(days=1)
    return rows


def _close_result(target="20260819"):
    raw = b'{"calendar":"sealed"}'
    return CloseSessionAuthorityResult(
        receipt={
            "schema_version": "cn-close-session-receipt.v1",
            "status": "TARGET_AUTHORIZED",
            "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
            "target_trade_date": target,
        },
        raw_response_bytes=raw,
    )


def _ready(context):
    return {
        "status": "READY",
        "write_performed": context.mode == "execute",
        "blockers": [],
        "evidence": {"target_date": context.target_date},
    }


def _health(_context):
    return {
        "status": "READY",
        "write_performed": False,
        "blockers": [],
        "evidence": {"binding_aware_research_ready": True},
    }


def _components(stage_callback=_ready):
    return MaintenanceComponents(
        pit=stage_callback,
        market=stage_callback,
        history=stage_callback,
        fundamental=_health,
        macro_release=stage_callback,
        system_status=lambda _context: {"capabilities": {"investment": "READY"}},
    )


def _transport_error(
    failure_class: str,
    *,
    phase: str = "REQUEST_SEND",
    elapsed_ms: int = 15,
) -> TushareHttpsError:
    return TushareHttpsError(
        "TUSHARE_TRANSPORT_ERROR",
        transport_diagnostic=TushareTransportDiagnostic(
            failure_class=failure_class,
            failure_phase=phase,
            elapsed_ms=elapsed_ms,
        ),
    )


def test_close_authority_uses_one_exact_raw_sse_partition():
    client = _CalendarClient(_calendar_rows())

    result = acquire_close_session_authority(
        now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI), client=client
    )

    assert result.receipt["target_trade_date"] == "20260819"
    assert result.receipt["calendar_date_count"] == 32
    assert result.receipt["ordered_open_dates"][-3:] == [
        "20260817",
        "20260818",
        "20260819",
    ]
    assert (
        result.receipt["raw_response_sha256"]
        == hashlib.sha256(result.raw_response_bytes).hexdigest()
    )
    assert client.calls == [
        {
            "api_name": "trade_cal",
            "params": {
                "exchange": "SSE",
                "start_date": "20260719",
                "end_date": "20260819",
            },
            "expected_fields": (
                "exchange",
                "cal_date",
                "is_open",
                "pretrade_date",
            ),
        }
    ]


def test_close_authority_rejects_incomplete_requested_date_coverage():
    rows = _calendar_rows()
    del rows[5]
    client = _CalendarClient(rows)

    with pytest.raises(
        CloseSessionAuthorityError,
        match="CLOSE_CALENDAR_DATE_COVERAGE_INCOMPLETE",
    ):
        acquire_close_session_authority(
            now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
            client=client,
        )


def test_close_authority_rejects_broken_pretrade_chain():
    rows = _calendar_rows()
    exchange, cal_date, is_open, _pretrade = rows[-1]
    rows[-1] = (exchange, cal_date, is_open, "20260817")
    client = _CalendarClient(rows)

    with pytest.raises(
        CloseSessionAuthorityError,
        match="CLOSE_CALENDAR_PRETRADE_CHAIN_INVALID",
    ):
        acquire_close_session_authority(
            now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
            client=client,
        )


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(16, 20, "1620"), (17, 59, "1720"), (19, 0, "1820"), (22, 0, "2020")],
)
def test_auto_slot_buckets_are_exact(hour, minute, expected):
    assert (
        resolve_attempt_slot(
            now=datetime(2026, 8, 19, hour, minute, tzinfo=SHANGHAI),
            requested="auto",
        )
        == expected
    )


def test_auto_slot_rejects_pre_window():
    with pytest.raises(DailyMaintenanceError, match="OUTSIDE_ATTEMPT_WINDOW"):
        resolve_attempt_slot(
            now=datetime(2026, 8, 19, 16, 19, tzinfo=SHANGHAI),
            requested="auto",
        )


def test_shadow_seals_immutable_receipts_and_rejects_no_writes(tmp_path):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)
    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="shadow",
        attempt_slot="1620",
        components=_components(
            stage_callback=lambda context: {
                **_ready(context),
                "write_performed": False,
            }
        ),
        now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
        close_authority=lambda **_kwargs: _close_result(),
    )

    assert result["status"] == "SHADOW_COMPLETE"
    assert result["canonical_unchanged"] is True
    assert result["canonical_write_count"] == 0
    assert result["usable_for_investment_research"] is True
    receipt_path = Path(result["attempt_receipt_ref"]["path"])
    state_path = Path(result["state_ref"]["path"])
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        == result["attempt_receipt_ref"]["sha256"]
    )
    _assert_shared_canonical(receipt_path)
    _assert_shared_canonical(state_path)
    _assert_shared_canonical(Path(result["close_session_receipt_ref"]["path"]))


def test_retry_is_quiet_early_and_sla_failure_in_final_slot(tmp_path):
    def retry(_context):
        return {
            "status": "RETRY_PENDING",
            "write_performed": False,
            "blockers": ["PROVIDER_NOT_READY"],
            "evidence": {},
        }

    for slot, expected in (("1620", "RETRY_PENDING"), ("2020", "SAME_DAY_SLA_MISSED")):
        run_root = tmp_path / slot
        run_root.mkdir(mode=0o700)
        result = run_cn_daily_maintenance(
            workspace_root=tmp_path,
            run_root=run_root,
            mode="execute",
            attempt_slot=slot,
            components=_components(stage_callback=retry),
            now=datetime(2026, 8, 19, 20, 20, tzinfo=SHANGHAI),
            close_authority=lambda **_kwargs: _close_result(),
        )
        assert result["status"] == expected
        assert not (run_root / "WRITE_VETO.json").exists()


def test_close_authority_retries_one_eligible_transport_failure_then_runs_dag_once(
    tmp_path,
    monkeypatch,
):
    run_root = tmp_path / "data/private/cn_daily_maintenance"
    run_root.mkdir(parents=True, mode=0o700)
    close_calls = []
    stage_calls = []
    sleeps = []
    monotonic = iter((1.0, 1.010, 2.0, 2.020))
    monkeypatch.setattr(daily_maintenance, "_MONOTONIC", lambda: next(monotonic))
    monkeypatch.setattr(daily_maintenance, "_SLEEP", sleeps.append)

    def close(**_kwargs):
        close_calls.append(len(close_calls) + 1)
        if len(close_calls) == 1:
            raise _transport_error("TIMEOUT", phase="RESPONSE_HEADERS", elapsed_ms=15_000)
        return _close_result()

    def stage(context):
        stage_calls.append(context.target_date)
        return _ready(context)

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="2020",
        components=_components(stage_callback=stage),
        now=datetime(2026, 8, 19, 20, 20, tzinfo=SHANGHAI),
        close_authority=close,
    )

    assert result["status"] == "COMPLETE"
    assert close_calls == [1, 2]
    assert sleeps == [0.25]
    assert stage_calls == ["20260819"] * 4
    assert result["transport_retry"] == {
        "schema_version": "cn-close-authority-transport-retry.v1",
        "attempt_count": 2,
        "retry_eligible": True,
        "retry_performed": True,
        "retry_delay_ms": 250,
        "terminal_outcome": "SUCCESS",
        "attempts": [
            {
                "attempt": 1,
                "outcome": "TRANSPORT_FAILURE",
                "failure_class": "TIMEOUT",
                "failure_phase": "RESPONSE_HEADERS",
                "elapsed_ms": 15_000,
            },
            {"attempt": 2, "outcome": "SUCCESS", "elapsed_ms": 20},
        ],
    }
    receipt_path = Path(result["attempt_receipt_ref"]["path"])
    receipt = _assert_shared_canonical(receipt_path)
    state = _assert_shared_canonical(Path(result["state_ref"]["path"]))
    assert receipt["transport_retry"] == result["transport_retry"]
    assert "transport_retry" not in state


@pytest.mark.parametrize(
    ("slot", "expected_status"),
    [("1620", "RETRY_PENDING"), ("2020", "SAME_DAY_SLA_MISSED")],
)
def test_close_authority_stops_after_two_eligible_transport_failures(
    tmp_path,
    monkeypatch,
    slot,
    expected_status,
):
    run_root = tmp_path / slot
    run_root.mkdir(mode=0o700)
    close_calls = []
    sleeps = []
    monotonic = iter((1.0, 1.001, 2.0, 2.001))
    monkeypatch.setattr(daily_maintenance, "_MONOTONIC", lambda: next(monotonic))
    monkeypatch.setattr(daily_maintenance, "_SLEEP", sleeps.append)

    def close(**_kwargs):
        close_calls.append(True)
        raise _transport_error("DNS", elapsed_ms=len(close_calls) * 10)

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot=slot,
        components=_components(stage_callback=lambda _context: pytest.fail("stage ran")),
        now=datetime(2026, 8, 19, 20, 20, tzinfo=SHANGHAI),
        close_authority=close,
    )

    assert result["status"] == expected_status
    assert result["blockers"] == ["TUSHARE_TRANSPORT_ERROR"]
    assert result["stage_results"] == []
    assert result["canonical_unchanged"] is True
    assert len(close_calls) == 2
    assert sleeps == [0.25]
    assert result["transport_retry"]["attempt_count"] == 2
    assert result["transport_retry"]["terminal_outcome"] == "FAILURE"
    assert [item["elapsed_ms"] for item in result["transport_retry"]["attempts"]] == [
        10,
        20,
    ]
    assert not (run_root / "WRITE_VETO.json").exists()
    state = _assert_shared_canonical(Path(result["state_ref"]["path"]))
    assert "transport_retry" not in state


@pytest.mark.parametrize("failure_class", ["READ", "CONNECTION_RESET", "UNKNOWN"])
def test_noneligible_transport_failure_records_diagnostic_without_same_slot_retry(
    tmp_path,
    monkeypatch,
    failure_class,
):
    run_root = tmp_path / failure_class
    run_root.mkdir(mode=0o700)
    close_calls = []
    sleeps = []
    monotonic = iter((1.0, 1.001))
    monkeypatch.setattr(daily_maintenance, "_MONOTONIC", lambda: next(monotonic))
    monkeypatch.setattr(daily_maintenance, "_SLEEP", sleeps.append)

    def close(**_kwargs):
        close_calls.append(True)
        raise _transport_error(failure_class, phase="RESPONSE_BODY")

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="1620",
        components=_components(stage_callback=lambda _context: pytest.fail("stage ran")),
        now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
        close_authority=close,
    )

    assert len(close_calls) == 1
    assert sleeps == []
    assert result["status"] == "RETRY_PENDING"
    assert result["transport_retry"]["attempt_count"] == 1
    assert result["transport_retry"]["retry_eligible"] is False
    assert result["transport_retry"]["retry_performed"] is False
    assert result["transport_retry"]["retry_delay_ms"] == 0
    assert "transport_retry" not in _assert_shared_canonical(Path(result["state_ref"]["path"]))


@pytest.mark.parametrize(
    ("second_error", "expected_status", "expected_blocker", "second_outcome"),
    [
        (
            TushareHttpsError("TUSHARE_API_ERROR"),
            "RETRY_PENDING",
            "TUSHARE_API_ERROR",
            "CONTROLLED_FAILURE",
        ),
        (
            TushareHttpsError("TUSHARE_RESPONSE_INVALID"),
            "BLOCKED",
            "TUSHARE_RESPONSE_INVALID",
            "CONTROLLED_FAILURE",
        ),
        (
            CloseSessionAuthorityError("CLOSE_CALENDAR_ROW_INVALID"),
            "BLOCKED",
            "CLOSE_CALENDAR_ROW_INVALID",
            "CONTROLLED_FAILURE",
        ),
        (
            RuntimeError("SECOND_ATTEMPT_SECRET"),
            "BLOCKED",
            "CLOSE_AUTHORITY_EXCEPTION",
            "EXCEPTION",
        ),
    ],
)
def test_retry_uses_second_attempt_as_terminal_authority_without_third_call(
    tmp_path,
    monkeypatch,
    second_error,
    expected_status,
    expected_blocker,
    second_outcome,
):
    run_root = tmp_path / expected_blocker
    run_root.mkdir(mode=0o700)
    close_calls = []
    sleeps = []
    monotonic = iter((1.0, 1.001, 2.0, 2.005))
    monkeypatch.setattr(daily_maintenance, "_MONOTONIC", lambda: next(monotonic))
    monkeypatch.setattr(daily_maintenance, "_SLEEP", sleeps.append)

    def close(**_kwargs):
        close_calls.append(True)
        if len(close_calls) == 1:
            raise _transport_error("CONNECT")
        raise second_error

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="1620",
        components=_components(stage_callback=lambda _context: pytest.fail("stage ran")),
        now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
        close_authority=close,
    )

    assert len(close_calls) == 2
    assert sleeps == [0.25]
    assert result["status"] == expected_status
    assert result["blockers"] == [expected_blocker]
    assert result["transport_retry"]["terminal_outcome"] == "FAILURE"
    assert result["transport_retry"]["attempts"][1]["outcome"] == second_outcome
    receipt_raw = Path(result["attempt_receipt_ref"]["path"]).read_bytes()
    assert b"SECOND_ATTEMPT_SECRET" not in receipt_raw
    assert "transport_retry" not in json.loads(Path(result["state_ref"]["path"]).read_bytes())
    assert (run_root / "WRITE_VETO.json").exists() is (expected_status == "BLOCKED")


@pytest.mark.parametrize(
    ("code", "expected_status", "expect_veto"),
    [
        ("TUSHARE_API_ERROR", "RETRY_PENDING", False),
        ("TUSHARE_HTTP_STATUS_ERROR", "RETRY_PENDING", False),
        ("TUSHARE_REDIRECT_BLOCKED", "BLOCKED", True),
        ("TUSHARE_RESPONSE_TOO_LARGE", "BLOCKED", True),
        ("TUSHARE_RESPONSE_INVALID", "BLOCKED", True),
    ],
)
def test_nontransport_provider_failures_never_get_same_slot_retry(
    tmp_path,
    monkeypatch,
    code,
    expected_status,
    expect_veto,
):
    run_root = tmp_path / code
    run_root.mkdir(mode=0o700)
    close_calls = []
    sleeps = []
    monotonic = iter((1.0, 1.001))
    monkeypatch.setattr(daily_maintenance, "_MONOTONIC", lambda: next(monotonic))
    monkeypatch.setattr(daily_maintenance, "_SLEEP", sleeps.append)

    def close(**_kwargs):
        close_calls.append(True)
        raise TushareHttpsError(code)

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="1620",
        components=_components(stage_callback=lambda _context: pytest.fail("stage ran")),
        now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
        close_authority=close,
    )

    assert len(close_calls) == 1
    assert sleeps == []
    assert result["status"] == expected_status
    assert result["blockers"] == [code]
    assert "transport_retry" not in result
    assert (run_root / "WRITE_VETO.json").exists() is expect_veto


def test_new_transport_retry_attempt_does_not_rewrite_historical_receipt(
    tmp_path,
    monkeypatch,
):
    run_root = tmp_path / "data/private/cn_daily_maintenance"
    historical = run_root / "attempts/historical/attempt.json"
    historical.parent.mkdir(parents=True, mode=0o700)
    run_root.chmod(0o700)
    (run_root / "attempts").chmod(0o700)
    historical_raw = b'{"immutable":"historical"}'
    historical.write_bytes(historical_raw)
    historical.chmod(0o600)
    monotonic = iter((1.0, 1.001, 2.0, 2.001))
    monkeypatch.setattr(daily_maintenance, "_MONOTONIC", lambda: next(monotonic))
    monkeypatch.setattr(daily_maintenance, "_SLEEP", lambda _delay: None)
    calls = []

    def close(**_kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise _transport_error("DNS")
        return _close_result()

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="shadow",
        attempt_slot="1620",
        components=_components(
            stage_callback=lambda context: {**_ready(context), "write_performed": False}
        ),
        now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
        close_authority=close,
    )

    assert result["transport_retry"]["terminal_outcome"] == "SUCCESS"
    assert historical.read_bytes() == historical_raw


@pytest.mark.parametrize(
    "code",
    [
        "CLOSE_CALENDAR_DATE_COVERAGE_INCOMPLETE",
        "CLOSE_SESSION_TARGET_NOT_TODAY",
        "CLOSE_CALENDAR_EMPTY",
    ],
)
def test_incomplete_close_authority_is_retryable_without_write_veto(tmp_path, code):
    def unavailable(**_kwargs):
        raise CloseSessionAuthorityError(code)

    for slot, expected in (
        ("1620", "RETRY_PENDING"),
        ("2020", "SAME_DAY_SLA_MISSED"),
    ):
        run_root = tmp_path / code / slot
        run_root.mkdir(parents=True, mode=0o700)
        result = run_cn_daily_maintenance(
            workspace_root=tmp_path,
            run_root=run_root,
            mode="execute",
            attempt_slot=slot,
            components=_components(),
            now=datetime(2026, 8, 19, 20, 20, tzinfo=SHANGHAI),
            close_authority=unavailable,
        )
        assert result["status"] == expected
        assert result["blockers"] == [code]
        assert not (run_root / "WRITE_VETO.json").exists()


def test_malformed_close_chain_remains_blocking_and_sets_veto(tmp_path):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)

    def malformed(**_kwargs):
        raise CloseSessionAuthorityError("CLOSE_CALENDAR_PRETRADE_CHAIN_INVALID")

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="1620",
        components=_components(),
        now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
        close_authority=malformed,
    )

    assert result["status"] == "BLOCKED"
    assert (run_root / "WRITE_VETO.json").is_file()


def test_all_no_action_components_produce_no_action(tmp_path):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)

    def no_action(_context):
        return {
            "status": "NO_ACTION",
            "write_performed": False,
            "blockers": [],
            "evidence": {"already_current": True},
        }

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="1720",
        components=MaintenanceComponents(
            pit=no_action,
            market=no_action,
            history=no_action,
            fundamental=no_action,
            macro_release=no_action,
        ),
        now=datetime(2026, 8, 19, 17, 20, tzinfo=SHANGHAI),
        close_authority=lambda **_kwargs: _close_result(),
    )

    assert result["status"] == "NO_ACTION"
    assert result["canonical_unchanged"] is True
    assert result["usable_for_investment_research"] is False


def test_fundamental_and_macro_are_independent_post_history_branches(tmp_path):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)
    macro_called = []

    def fundamental_blocked(_context):
        return {
            "status": "BLOCKED",
            "write_performed": False,
            "blockers": ["FUNDAMENTAL_BINDING_NOT_READY"],
            "evidence": {},
        }

    def macro_ready(_context):
        macro_called.append(True)
        return {
            "status": "READY",
            "write_performed": False,
            "blockers": [],
            "evidence": {"prepared": True},
        }

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="shadow",
        attempt_slot="1720",
        components=MaintenanceComponents(
            pit=lambda context: {**_ready(context), "write_performed": False},
            market=lambda context: {**_ready(context), "write_performed": False},
            history=lambda context: {**_ready(context), "write_performed": False},
            fundamental=fundamental_blocked,
            macro_release=macro_ready,
        ),
        now=datetime(2026, 8, 19, 17, 20, tzinfo=SHANGHAI),
        close_authority=lambda **_kwargs: _close_result(),
    )

    assert macro_called == [True]
    assert result["same_day_status"] == "SHADOW_COMPLETE"
    assert result["fundamental_integrity_status"] == "BLOCKED"
    assert result["fundamental_refresh_status"] == "HEALTH_ONLY"
    assert result["maintenance_status"] == "BLOCKED"


def test_naive_attempt_time_is_rejected_before_any_io(tmp_path):
    with pytest.raises(DailyMaintenanceError, match="ATTEMPT_TIME_INVALID"):
        run_cn_daily_maintenance(
            workspace_root=tmp_path,
            run_root=tmp_path / "not-created",
            mode="shadow",
            attempt_slot="1620",
            now=datetime(2026, 8, 19, 16, 20),
        )
    assert not (tmp_path / "not-created").exists()


def test_blocked_execute_sets_exact_veto_and_clear_archives_it(tmp_path):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="1620",
        components=MaintenanceComponents(),
        now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
        close_authority=lambda **_kwargs: _close_result(),
    )

    assert result["status"] == "BLOCKED"
    veto_ref = result["write_veto_ref"]
    veto_path = Path(veto_ref["path"])
    assert hashlib.sha256(veto_path.read_bytes()).hexdigest() == veto_ref["sha256"]
    _assert_shared_canonical(veto_path)
    blocked_again = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="1720",
        components=_components(),
        now=datetime(2026, 8, 19, 17, 20, tzinfo=SHANGHAI),
        close_authority=lambda **_kwargs: pytest.fail("veto must precede provider call"),
    )
    assert blocked_again["status"] == "WRITE_VETO_ACTIVE"

    cleared = clear_cn_daily_write_veto(
        run_root=run_root,
        expected_veto_sha256=veto_ref["sha256"],
        reason="reviewed contract repair",
    )
    assert cleared["status"] == "CLEARED"
    assert not veto_path.exists()
    assert Path(cleared["archived_veto_ref"]["path"]).is_file()
    _assert_shared_canonical(Path(cleared["clear_receipt_ref"]["path"]))


def test_macro_veto_has_registered_exact_sha_clear_path(tmp_path):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)
    veto_path = run_root / "MACRO_WRITE_VETO.json"
    raw = b'{"schema_version":"cn-daily-maintenance-macro-write-veto.v1"}\n'
    veto_path.write_bytes(raw)
    veto_path.chmod(0o600)
    digest = hashlib.sha256(raw).hexdigest()

    cleared = clear_cn_daily_write_veto(
        run_root=run_root,
        expected_veto_sha256=digest,
        reason="reviewed macro recovery",
        lane="macro",
    )

    assert cleared["status"] == "CLEARED"
    assert cleared["lane"] == "macro"
    assert not veto_path.exists()
    assert Path(cleared["archived_veto_ref"]["path"]).is_file()


def _token_missing_veto(tmp_path):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)

    def missing(**_kwargs):
        raise CloseSessionAuthorityError("TUSHARE_TOKEN_MISSING")

    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="1820",
        components=_components(),
        now=datetime(2026, 8, 24, 18, 20, tzinfo=SHANGHAI),
        close_authority=missing,
    )
    assert result["status"] == "BLOCKED"
    return run_root, result


def _ready_preflight(run_root):
    result = write_credential_preflight(
        run_root=run_root,
        attempt_slot="2020",
        receipt_id="slot-2020-test",
        access_state="READY",
        checked_at="2026-08-24T12:20:00Z",
    )
    raw = Path(result["receipt_path"]).read_bytes()
    assert b"token" in raw
    assert b"secret" not in raw
    assert b'"credential_source":"PROJECT_ENV"' in raw
    assert b'"env_file":".env"' in raw
    assert b'"env_key":"TUSHARE_TOKEN"' in raw
    assert b"MACOS_KEYCHAIN" not in raw
    assert result["token_material_recorded"] is False
    return result


def test_project_env_token_reader_accepts_one_owner_only_token(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OTHER=value\nTUSHARE_TOKEN=test-token-1234567890\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)

    assert read_project_env_token(env_file) == "test-token-1234567890"


@pytest.mark.parametrize(
    "content",
    [
        "OTHER=value\n",
        "TUSHARE_TOKEN=short\n",
        "TUSHARE_TOKEN=test-token-1234567890\nTUSHARE_TOKEN=duplicate-token-123456\n",
    ],
)
def test_project_env_token_reader_rejects_missing_invalid_or_duplicate(tmp_path, content):
    env_file = tmp_path / ".env"
    env_file.write_text(content, encoding="utf-8")
    env_file.chmod(0o600)

    with pytest.raises(CredentialPreflightError, match="CREDENTIAL_ENV_TOKEN_INVALID"):
        read_project_env_token(env_file)


def test_project_env_token_reader_rejects_unsafe_mode_and_symlink(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TUSHARE_TOKEN=test-token-1234567890\n", encoding="utf-8")
    env_file.chmod(0o644)
    with pytest.raises(CredentialPreflightError, match="CREDENTIAL_ENV_UNSAFE"):
        read_project_env_token(env_file)

    env_file.chmod(0o600)
    link = tmp_path / "linked.env"
    link.symlink_to(env_file)
    with pytest.raises(CredentialPreflightError, match="CREDENTIAL_ENV_UNAVAILABLE"):
        read_project_env_token(link)


def test_token_veto_recovers_once_with_exact_zero_write_closure(tmp_path, monkeypatch):
    run_root, blocked = _token_missing_veto(tmp_path)
    preflight = _ready_preflight(run_root)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-only-not-recorded")

    recovered = recover_transient_cn_daily_write_veto(
        workspace_root=tmp_path,
        run_root=run_root,
        expected_veto_sha256=blocked["write_veto_ref"]["sha256"],
        credential_preflight_receipt=preflight["receipt_path"],
        expected_credential_preflight_sha256=preflight["receipt_sha256"],
    )
    replay = recover_transient_cn_daily_write_veto(
        workspace_root=tmp_path,
        run_root=run_root,
        expected_veto_sha256=blocked["write_veto_ref"]["sha256"],
        credential_preflight_receipt=preflight["receipt_path"],
        expected_credential_preflight_sha256=preflight["receipt_sha256"],
    )

    assert recovered["status"] == "RECOVERED"
    assert recovered["recovered"] is True
    assert replay["status"] == "NO_ACTION"
    assert replay["recovered"] is False
    assert not (run_root / "WRITE_VETO.json").exists()
    assert Path(recovered["original_veto_ref"]["path"]).is_file()
    receipt_raw = Path(recovered["recovery_receipt_ref"]["path"]).read_bytes()
    assert b"test-only-not-recorded" not in receipt_raw
    _assert_shared_canonical(Path(recovered["recovery_receipt_ref"]["path"]))


def test_legacy_token_veto_and_attempt_recover_without_broadening_reader(tmp_path, monkeypatch):
    run_root, blocked = _token_missing_veto(tmp_path)
    veto_path = Path(blocked["write_veto_ref"]["path"])
    veto = json.loads(veto_path.read_bytes())
    veto_path.write_bytes(_legacy_maintenance_bytes(veto))
    legacy_veto_sha = hashlib.sha256(veto_path.read_bytes()).hexdigest()

    attempt_path = Path(blocked["attempt_receipt_ref"]["path"])
    attempt = json.loads(attempt_path.read_bytes())
    attempt["write_veto_ref"]["sha256"] = legacy_veto_sha
    attempt_path.write_bytes(_legacy_maintenance_bytes(attempt))
    legacy_attempt_raw = attempt_path.read_bytes()

    with pytest.raises(CommandError, match="REQUEST_NOT_CANONICAL"):
        read_exact_request(
            tmp_path,
            attempt_path.relative_to(tmp_path).as_posix(),
            hashlib.sha256(legacy_attempt_raw).hexdigest(),
        )

    preflight = _ready_preflight(run_root)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-only-not-recorded")
    recovered = recover_transient_cn_daily_write_veto(
        workspace_root=tmp_path,
        run_root=run_root,
        expected_veto_sha256=legacy_veto_sha,
        credential_preflight_receipt=preflight["receipt_path"],
        expected_credential_preflight_sha256=preflight["receipt_sha256"],
    )
    assert recovered["status"] == "RECOVERED"
    assert attempt_path.read_bytes() == legacy_attempt_raw
    recovery_path = Path(recovered["recovery_receipt_ref"]["path"])
    recovery = _assert_shared_canonical(recovery_path)

    recovery_path.write_bytes(_legacy_maintenance_bytes(recovery))
    replay = recover_transient_cn_daily_write_veto(
        workspace_root=tmp_path,
        run_root=run_root,
        expected_veto_sha256=legacy_veto_sha,
        credential_preflight_receipt=preflight["receipt_path"],
        expected_credential_preflight_sha256=preflight["receipt_sha256"],
    )
    assert replay["schema_version"] == "cn-daily-maintenance-transient-veto-recovery.v1"
    assert replay["status"] == "NO_ACTION"
    assert replay["recovered"] is False
    assert (
        replay["recovery_receipt_ref"]["sha256"]
        == hashlib.sha256(recovery_path.read_bytes()).hexdigest()
    )


def test_token_veto_recovery_blocks_pointer_drift(tmp_path, monkeypatch):
    run_root, blocked = _token_missing_veto(tmp_path)
    preflight = _ready_preflight(run_root)
    pointer = tmp_path / "results/factors/_active.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("{}\n", encoding="ascii")
    pointer.chmod(0o600)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-only-not-recorded")

    with pytest.raises(DailyMaintenanceError, match="RECOVER_WRITE_VETO_POINTER_DRIFT"):
        recover_transient_cn_daily_write_veto(
            workspace_root=tmp_path,
            run_root=run_root,
            expected_veto_sha256=blocked["write_veto_ref"]["sha256"],
            credential_preflight_receipt=preflight["receipt_path"],
            expected_credential_preflight_sha256=preflight["receipt_sha256"],
        )
    assert (run_root / "WRITE_VETO.json").is_file()


def test_noncredential_veto_cannot_auto_recover(tmp_path, monkeypatch):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)

    def invalid(**_kwargs):
        raise CloseSessionAuthorityError("CLOSE_CALENDAR_PRETRADE_CHAIN_INVALID")

    blocked = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="execute",
        attempt_slot="1820",
        components=_components(),
        now=datetime(2026, 8, 24, 18, 20, tzinfo=SHANGHAI),
        close_authority=invalid,
    )
    preflight = _ready_preflight(run_root)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-only-not-recorded")

    with pytest.raises(DailyMaintenanceError, match="RECOVER_WRITE_VETO_NOT_TRANSIENT"):
        recover_transient_cn_daily_write_veto(
            workspace_root=tmp_path,
            run_root=run_root,
            expected_veto_sha256=blocked["write_veto_ref"]["sha256"],
            credential_preflight_receipt=preflight["receipt_path"],
            expected_credential_preflight_sha256=preflight["receipt_sha256"],
        )


def test_nonblocking_lock_truth_table(tmp_path):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)
    lock_path = run_root / ".daily-maintenance.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        early = run_cn_daily_maintenance(
            workspace_root=tmp_path,
            run_root=run_root,
            mode="execute",
            attempt_slot="1620",
            components=_components(),
            now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
        )
        final = run_cn_daily_maintenance(
            workspace_root=tmp_path,
            run_root=run_root,
            mode="execute",
            attempt_slot="2020",
            components=_components(),
            now=datetime(2026, 8, 19, 20, 20, tzinfo=SHANGHAI),
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert early["status"] == "ALREADY_RUNNING"
    assert final["status"] == "SAME_DAY_SLA_MISSED"


def test_cli_registers_daily_commands():
    parser = cli_main._build_parser()
    daily = parser.parse_args(
        [
            "market",
            "daily-maintain",
            "--market",
            "CN",
            "--run-root",
            "/private/tmp/cn-daily",
            "--mode",
            "shadow",
        ]
    )
    clear = parser.parse_args(
        [
            "market",
            "clear-write-veto",
            "--market",
            "CN",
            "--run-root",
            "/private/tmp/cn-daily",
            "--expected-veto-sha256",
            "0" * 64,
            "--reason",
            "reviewed",
        ]
    )
    assert daily.market_command == "daily-maintain"
    assert clear.market_command == "clear-write-veto"


def test_attempt_receipt_is_canonical_json(tmp_path):
    run_root = tmp_path / "private-runs"
    run_root.mkdir(mode=0o700)
    result = run_cn_daily_maintenance(
        workspace_root=tmp_path,
        run_root=run_root,
        mode="shadow",
        attempt_slot="1620",
        components=_components(
            stage_callback=lambda context: {
                **_ready(context),
                "write_performed": False,
            }
        ),
        now=datetime(2026, 8, 19, 16, 20, tzinfo=SHANGHAI),
        close_authority=lambda **_kwargs: _close_result(),
    )
    raw = Path(result["attempt_receipt_ref"]["path"]).read_bytes()
    parsed = json.loads(raw)
    assert raw == canonical_json_bytes(parsed)
    assert not raw.endswith(b"\n")
    relative = Path(result["attempt_receipt_ref"]["path"]).relative_to(tmp_path)
    loaded_raw, loaded = read_exact_request(
        tmp_path,
        relative.as_posix(),
        result["attempt_receipt_ref"]["sha256"],
    )
    assert loaded_raw == raw
    assert loaded == parsed

    newline_path = tmp_path / "attempt-with-terminal-lf.json"
    newline_path.write_bytes(raw + b"\n")
    with pytest.raises(CommandError, match="REQUEST_NOT_CANONICAL"):
        read_exact_request(
            tmp_path,
            newline_path.relative_to(tmp_path).as_posix(),
            hashlib.sha256(newline_path.read_bytes()).hexdigest(),
        )

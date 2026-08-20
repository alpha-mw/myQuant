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
from quant_investor.market.close_session_authority import (
    CloseSessionAuthorityError,
    CloseSessionAuthorityResult,
    acquire_close_session_authority,
)
from quant_investor.market.daily_maintenance import (
    DailyMaintenanceError,
    MaintenanceComponents,
    clear_cn_daily_write_veto,
    resolve_attempt_slot,
    run_cn_daily_maintenance,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


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
    assert raw == (
        json.dumps(
            parsed, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode("ascii")

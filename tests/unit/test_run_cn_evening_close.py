"""Explicit maintenance-attempt override for the evening close (catch-up runs)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_cn_evening_close",
    Path(__file__).resolve().parents[2] / "scripts/operations/run_cn_evening_close.py",
)
evening = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evening)


def _write(path: Path, value: dict) -> str:
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _attempt(tmp_path: Path, **overrides) -> tuple[Path, str, str]:
    calendar = tmp_path / "close-session-receipt.json"
    calendar_sha = _write(calendar, {"target_trade_date": "20260930"})
    body = {
        "target_date": "20260930",
        "core_blockers": [],
        "close_session_receipt_ref": {"path": str(calendar), "sha256": calendar_sha},
        **overrides,
    }
    attempt = tmp_path / "attempt.json"
    return attempt, _write(attempt, body), calendar_sha


def test_explicit_attempt_binds_exact_receipt_for_trade_date(tmp_path):
    attempt, sha, calendar_sha = _attempt(tmp_path)

    result = evening.explicit_maintenance_attempt("2026-09-30", str(attempt), sha)

    assert result == (attempt, sha, tmp_path / "close-session-receipt.json", calendar_sha)


@pytest.mark.parametrize(
    "overrides",
    [{"target_date": "20261002"}, {"core_blockers": ["MARKET_CONTRACT_BLOCKED"]}],
)
def test_explicit_attempt_refuses_other_day_or_blocked_core(tmp_path, overrides):
    attempt, sha, _ = _attempt(tmp_path, **overrides)

    with pytest.raises(evening.Blocked, match="MAINTENANCE_ATTEMPT_NOT_USABLE:20260930"):
        evening.explicit_maintenance_attempt("2026-09-30", str(attempt), sha)


def test_explicit_attempt_refuses_sha_or_calendar_drift(tmp_path):
    attempt, sha, _ = _attempt(tmp_path)

    with pytest.raises(evening.Blocked, match="MAINTENANCE_ATTEMPT_NOT_USABLE"):
        evening.explicit_maintenance_attempt("2026-09-30", str(attempt), "0" * 64)
    (tmp_path / "close-session-receipt.json").write_text("{}")
    with pytest.raises(evening.Blocked, match="CALENDAR_RECEIPT_SHA_MISMATCH"):
        evening.explicit_maintenance_attempt("2026-09-30", str(attempt), sha)


def _calendar_proof(tmp_path: Path, monkeypatch, *, state_trade_date: str = "20260930") -> str:
    """Seal state -> publication -> proof for a 20260930 session whose next open is 20261008."""
    workspace = tmp_path / "workspace"
    maintenance = tmp_path / "maintenance"
    workspace.mkdir()
    maintenance.mkdir()
    proof_sha = _write(
        workspace / "proof.json",
        {"eod_trade_date": "20260930", "next_open_session": "20261008"},
    )
    publication_sha = _write(
        workspace / "publication.json",
        {"proof_ref": {"path": "proof.json", "sha256": proof_sha}},
    )
    _write(
        maintenance / "factor-loop-state.json",
        {
            "trade_date": state_trade_date,
            "next_session_calendar_proof_ref": {
                "path": "publication.json",
                "sha256": publication_sha,
            },
        },
    )
    monkeypatch.setattr(evening, "WORKSPACE", workspace)
    monkeypatch.setattr(evening, "MAINTENANCE_ROOT", maintenance)
    return proof_sha


def test_holiday_between_sealed_sessions_is_a_proven_non_trading_day(tmp_path, monkeypatch):
    proof_sha = _calendar_proof(tmp_path, monkeypatch)

    assert evening.non_trading_day("2026-10-02") == {
        "status": "NO_ACTION",
        "reason": "NON_TRADING_DAY",
        "last_session": "20260930",
        "next_open_session": "20261008",
        "calendar_proof_sha256": proof_sha,
    }


@pytest.mark.parametrize("day", ["2026-09-30", "2026-10-08", "2026-10-09"])
def test_session_days_and_days_past_the_proof_are_not_claimed_closed(tmp_path, monkeypatch, day):
    _calendar_proof(tmp_path, monkeypatch)

    assert evening.non_trading_day(day) is None


def test_stale_or_missing_factor_state_proves_nothing(tmp_path, monkeypatch):
    _calendar_proof(tmp_path, monkeypatch, state_trade_date="20260929")
    assert evening.non_trading_day("2026-10-02") is None

    (evening.MAINTENANCE_ROOT / "factor-loop-state.json").unlink()
    assert evening.non_trading_day("2026-10-02") is None


def test_tampered_calendar_proof_blocks_instead_of_skipping(tmp_path, monkeypatch):
    _calendar_proof(tmp_path, monkeypatch)
    (evening.WORKSPACE / "proof.json").write_text(
        json.dumps({"eod_trade_date": "20260930", "next_open_session": "20261231"})
    )

    with pytest.raises(evening.Blocked, match="CALENDAR_PROOF_SHA_MISMATCH:proof.json"):
        evening.non_trading_day("2026-10-02")


def test_holiday_is_quiet_only_after_the_prior_session_is_closed(tmp_path, monkeypatch):
    records = tmp_path / "records"
    workspace = tmp_path / "workspace"
    (records / "_event_store").mkdir(parents=True)
    (workspace / "data/parquet/cn/benchmarks").mkdir(parents=True)
    monkeypatch.setattr(evening, "RECORD_ROOT", records)
    monkeypatch.setattr(evening, "WORKSPACE", workspace)
    events = records / "_event_store/current.v1.json"
    benchmark = workspace / "data/parquet/cn/benchmarks/_latest.json"

    _write(events, {"trade_dates": ["2026-09-29", "2026-09-30"]})
    _write(benchmark, {"end_date": "2026-09-30"})
    evening.prior_session_closed("20260930")

    _write(benchmark, {"end_date": "2026-09-29"})
    with pytest.raises(evening.Blocked, match="PRIOR_SESSION_NOT_CLOSED:20260930"):
        evening.prior_session_closed("20260930")

    _write(benchmark, {"end_date": "2026-09-30"})
    _write(events, {"trade_dates": ["2026-09-29"]})
    with pytest.raises(evening.Blocked, match="PRIOR_SESSION_NOT_CLOSED:20260930"):
        evening.prior_session_closed("20260930")


@pytest.mark.parametrize(
    ("status", "execute", "expected_exit", "alerted"),
    [
        ("BLOCKED", True, 2, True),
        ("BLOCKED", False, 2, False),
        ("COMPLETED", True, 0, False),
        ("NO_ACTION", True, 0, False),
    ],
)
def test_only_a_blocked_scheduled_close_alerts_the_owner(
    tmp_path, monkeypatch, capsys, status, execute, expected_exit, alerted
):
    alerts = []
    monkeypatch.setattr(evening, "RECEIPT_ROOT", tmp_path / "receipts")
    monkeypatch.setattr(evening, "alert", alerts.append)
    receipt = {
        "status": status,
        "trade_date": "2026-09-30",
        "execute": execute,
        "blocker": "MAINTENANCE_NOT_COMPLETE:20260930" if status == "BLOCKED" else None,
    }

    assert evening.finish(receipt, "20260930") == expected_exit
    assert json.loads(capsys.readouterr().out)["status"] == status
    assert (alerts == [receipt]) is alerted


def _closed_stores(tmp_path: Path, monkeypatch, *, benchmark_end: str) -> None:
    records = tmp_path / "records"
    (records / "_event_store").mkdir(parents=True)
    (evening.WORKSPACE / "data/parquet/cn/benchmarks").mkdir(parents=True)
    monkeypatch.setattr(evening, "RECORD_ROOT", records)
    monkeypatch.setattr(evening, "RECEIPT_ROOT", tmp_path / "receipts")
    _write(records / "_event_store/current.v1.json", {"trade_dates": ["2026-09-30"]})
    _write(
        evening.WORKSPACE / "data/parquet/cn/benchmarks/_latest.json",
        {"end_date": benchmark_end},
    )


def test_scheduled_close_on_a_proven_holiday_is_no_action_without_alert(
    tmp_path, monkeypatch, capsys
):
    _calendar_proof(tmp_path, monkeypatch)
    _closed_stores(tmp_path, monkeypatch, benchmark_end="2026-09-30")
    alerts = []
    monkeypatch.setattr(evening, "alert", alerts.append)
    monkeypatch.setattr("sys.argv", ["evening", "--trade-date", "2026-10-02", "--execute"])

    assert evening.main() == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "NO_ACTION" and printed["blocker"] is None
    receipt = json.loads(Path(printed["receipt"]).read_text())
    assert [item["step"] for item in receipt["steps"]] == ["calendar"]
    assert receipt["steps"][0]["result"]["next_open_session"] == "20261008"
    assert alerts == []


def test_holiday_with_an_unclosed_prior_session_still_blocks_and_alerts(
    tmp_path, monkeypatch, capsys
):
    _calendar_proof(tmp_path, monkeypatch)
    _closed_stores(tmp_path, monkeypatch, benchmark_end="2026-09-29")
    alerts = []
    monkeypatch.setattr(evening, "alert", alerts.append)
    monkeypatch.setattr("sys.argv", ["evening", "--trade-date", "2026-10-02", "--execute"])

    assert evening.main() == 2

    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "BLOCKED"
    assert printed["blocker"] == "PRIOR_SESSION_NOT_CLOSED:20260930"
    assert len(alerts) == 1

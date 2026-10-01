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

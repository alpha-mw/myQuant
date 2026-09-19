"""Historical task date reuses native budgets while attempt clocks remain current."""

from datetime import datetime, timezone
import json

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.market.daily_maintenance import DailyMaintenanceError, _write_once
from quant_investor.market.maintenance_journal import DailyOperationJournal
from test_daily_evidence_slot_claim import inventory


def reopen(root, now, day=None, **changes):
    args = dict(now=now, slot="2020", mode="execute", _historical_trade_date=day)
    args.update(changes)
    return DailyOperationJournal(root / "maintenance", root, **args)


def test_cross_day_reopening_shares_original_attempt_and_request_budgets(tmp_path):
    (tmp_path / "maintenance").mkdir(mode=0o700)
    original = reopen(tmp_path, datetime(2026, 9, 4, 13, tzinfo=timezone.utc))
    claim_raw = (original.path / "claim.json").read_bytes()
    calls = []

    def failure(**kwargs):
        calls.append(kwargs["now"])
        raise TimeoutError("test-only transport failure")

    for n, stamp in enumerate(["2026-09-07T13:00:00+00:00", "2026-09-08T13:00:00+00:00"], 1):
        now = datetime.fromisoformat(stamp)
        journal = reopen(tmp_path, now, "20260904")
        assert journal.now is now
        assert journal.path == original.path and journal.claim_ref == original.claim_ref
        assert (journal.path / "claim.json").read_bytes() == claim_raw
        attempt = journal.root / "attempts" / f"attempt-{n}"
        attempt.mkdir(parents=True, mode=0o700)
        started = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_once(
            attempt / "started.json",
            canonical_json_bytes({"state": "STARTED", "started_at": started}),
        )
        journal.bind(attempt)
        with pytest.raises(TimeoutError):
            journal.acquire(failure, now=now)
        assert json.loads((attempt / "started.json").read_bytes())["started_at"] == started
    before = inventory(tmp_path)
    third = reopen(tmp_path, datetime(2026, 9, 9, 13, tzinfo=timezone.utc), "20260904")
    with pytest.raises(DailyMaintenanceError, match="ATTEMPT_BUDGET_EXHAUSTED"):
        third.bind(third.root / "attempts" / "third")
    with pytest.raises(DailyMaintenanceError, match="CLOSE_REQUEST_BUDGET_EXHAUSTED"):
        third.acquire(failure, now=third.now)
    assert len(calls) == 2 and len(third.attempts()) == 2
    assert inventory(tmp_path) == before
    assert sorted(p.name for p in (third.root / "logical_tasks").iterdir()) == [
        "20260904-2020-execute"
    ]


@pytest.mark.parametrize(
    "day,now,extra",
    [
        ("2026094", "2026-09-09T13:00:00+00:00", {}),
        (False, "2026-09-09T13:00:00+00:00", {}),
        ("20260230", "2026-09-09T13:00:00+00:00", {}),
        ("20260909", "2026-09-09T13:00:00+00:00", {}),
        ("20260910", "2026-09-09T13:00:00+00:00", {}),
        ("20260904", "2026-09-09T13:00:00", {}),
        ("20260904", "2026-09-09T13:00:00+00:00", {"mode": "shadow"}),
        ("20260904", "2026-09-09T13:00:00+00:00", {"slot": "2100"}),
    ],
)
def test_invalid_historical_identity_writes_nothing(tmp_path, day, now, extra):
    (tmp_path / "maintenance").mkdir(mode=0o700)
    before = inventory(tmp_path)
    with pytest.raises(DailyMaintenanceError, match="HISTORICAL_TASK_IDENTITY_INVALID"):
        reopen(tmp_path, datetime.fromisoformat(now), day, **extra)
    assert inventory(tmp_path) == before


def test_shanghai_date_and_distinct_target_identity(tmp_path):
    (tmp_path / "maintenance").mkdir(mode=0o700)
    # UTC is still Sep 8, but the actual Shanghai runtime is already Sep 9.
    now = datetime(2026, 9, 8, 17, tzinfo=timezone.utc)
    first = reopen(tmp_path, now, "20260908")
    second = reopen(tmp_path, now, "20260907")
    assert first.now is second.now is now
    assert first.path != second.path and first.claim_ref != second.claim_ref
    assert first.key == "20260908-2020-execute"


def test_historical_reopen_cannot_adopt_installation_drift(tmp_path):
    (tmp_path / "maintenance").mkdir(mode=0o700)
    first = reopen(tmp_path, datetime(2026, 9, 4, 13, tzinfo=timezone.utc))
    path = first.path / "claim.json"
    value = json.loads(path.read_bytes())
    value["installation"]["python"] = "different-installation"
    path.write_bytes(canonical_json_bytes(value))
    before = inventory(tmp_path)
    with pytest.raises(DailyMaintenanceError, match="INSTALL_OR_POLICY_DRIFT"):
        reopen(tmp_path, datetime(2026, 9, 9, 13, tzinfo=timezone.utc), "20260904")
    assert inventory(tmp_path) == before

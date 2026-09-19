"""Real native/day lock absence proof with explicitly synthetic EOD admission."""

from datetime import datetime, timezone
import json

import pytest
from _public_catchup_fixture import put
from test_automatic_catchup_storage import prepared
from test_daily_evidence_public_catchup import snapshot
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.automatic_catchup_resolution import read_automatic_resolution
from quant_investor.operations.automatic_catchup_closure import (
    RUN_ROOT,
    expire_unstarted,
    prime_day_locks,
)
from quant_investor.operations.automatic_catchup_contract import run_path
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.journal_storage import JournalStorage
from quant_investor.system.errors import SystemStorageError

EXPIRED = datetime(2026, 8, 29, 14, tzinfo=timezone.utc)


def active(root, monkeypatch):
    storage, ref, resolution, pending = prepared(root, monkeypatch)
    put(root, RUN_ROOT + "/.daily-maintenance.lock", b"")
    with storage.locked():
        storage.write(ref["path"], canonical_json_bytes(resolution))
        prime_day_locks(storage, resolution)
        storage.set_pending(pending)
    context = read_automatic_resolution(
        workspace=str(root),
        resolution_ref=ref,
        release_install_ref=resolution["release_install_ref"],
        synthetic=True,
    )
    return storage, ref, context


def test_expired_unstarted_closure_recovers_after_write_before_idle(tmp_path, monkeypatch):
    storage, ref, context = active(tmp_path, monkeypatch)
    old_resolution = (tmp_path / ref["path"]).read_bytes()
    with storage.locked():
        with monkeypatch.context() as patch:
            patch.setattr(
                storage, "set_pending", lambda value: (_ for _ in ()).throw(OSError("before idle"))
            )
            with pytest.raises(ContractError, match="MAINTENANCE_LOCK_UNSAFE") as failed:
                expire_unstarted(storage, context, resolution_ref=ref, synthetic=True, now=EXPIRED)
            assert isinstance(failed.value.__cause__, OSError)
        assert storage.pending()["state"] == "ACTIVE"
        path = run_path(context["resolution"]["auto_request_ref"], "closure.v1.json")
        original = (tmp_path / path).read_bytes()
        record = json.loads(original)
        assert record["state"] == "EXPIRED_UNSTARTED"
        assert record["unresolved_trade_dates"] == ["20260827", "20260828"]
        expire_unstarted(
            storage,
            context,
            resolution_ref=ref,
            synthetic=True,
            now=datetime(2026, 8, 30, 14, tzinfo=timezone.utc),
        )
        assert storage.pending()["state"] == "IDLE" and (tmp_path / path).read_bytes() == original
        assert (tmp_path / ref["path"]).read_bytes() == old_resolution


@pytest.mark.parametrize(
    "fault", ["claim", "orphan", "dag_start", "unknown", "missing_native_lock"]
)
def test_any_start_or_uncertain_absence_keeps_active(tmp_path, monkeypatch, fault):
    storage, ref, context = active(tmp_path, monkeypatch)
    if fault == "claim":
        put(
            tmp_path,
            RUN_ROOT + "/logical_tasks/20260828-2020-execute/claim.json",
            {"started": True},
        )
    elif fault == "orphan":
        put(tmp_path, RUN_ROOT + "/attempts/orphan/started.json", {"state": "STARTED"})
    elif fault in {"dag_start", "unknown"}:
        put(
            tmp_path,
            "results/operations/daily_production/CN/20260828/" + fault + ".json",
            {"evidence": True},
        )
    else:
        (tmp_path / RUN_ROOT / ".daily-maintenance.lock").unlink()
    with storage.locked():
        before = snapshot(tmp_path)
        with pytest.raises((ContractError, SystemStorageError, OSError)):
            expire_unstarted(storage, context, resolution_ref=ref, synthetic=True, now=EXPIRED)
        assert storage.pending()["state"] == "ACTIVE" and snapshot(tmp_path) == before


def test_busy_day_and_unexpired_scope_cannot_close(tmp_path, monkeypatch):
    storage, ref, context = active(tmp_path, monkeypatch)
    with storage.locked():
        with pytest.raises(ContractError, match="AUTO_PENDING_REQUEST_CONFLICT"):
            expire_unstarted(
                storage,
                context,
                resolution_ref=ref,
                synthetic=True,
                now=datetime(2026, 8, 28, 14, tzinfo=timezone.utc),
            )
        journal = JournalStorage(str(tmp_path))
        with journal.lock("results/operations/daily_production/CN/20260828/.lock"):
            before = snapshot(tmp_path)
            with pytest.raises(ContractError, match="AUTO_DAY_BUSY"):
                expire_unstarted(storage, context, resolution_ref=ref, synthetic=True, now=EXPIRED)
            assert snapshot(tmp_path) == before and storage.pending()["state"] == "ACTIVE"


@pytest.mark.parametrize("historical", [False, True])
def test_native_attempt_actual_target_is_checked_across_claim_dates(
    tmp_path, monkeypatch, historical
):
    from quant_investor.market.maintenance_journal import DailyOperationJournal
    from quant_investor.market.historical_session import (
        FILENAME,
        CALENDAR_FILENAME,
        RAW_FILENAME,
        build_historical_session,
    )
    from test_daily_evidence_requested_session import capture

    storage, ref, context = active(tmp_path, monkeypatch)
    root = tmp_path / RUN_ROOT
    journal = DailyOperationJournal(
        root,
        tmp_path,
        now=EXPIRED,
        slot="2020",
        mode="execute",
        _historical_trade_date="20260826" if historical else None,
    )
    attempt = root / "attempts/prior-native-attempt"
    put(tmp_path, str((attempt / "started.json").relative_to(tmp_path)), {"state": "STARTED"})
    native = capture("2026-08-29T21:00:00+08:00")
    raw_path = str((attempt / RAW_FILENAME).relative_to(tmp_path))
    calendar_path = str((attempt / CALENDAR_FILENAME).relative_to(tmp_path))
    put(tmp_path, raw_path, native.raw_response_bytes)
    put(tmp_path, calendar_path, {**native.receipt, "raw_response_path": str(tmp_path / raw_path)})
    if historical:
        proof = build_historical_session(
            requested_trade_date="20260826",
            previous_trade_date="20260825",
            calendar_bytes=(tmp_path / calendar_path).read_bytes(),
            raw=native.raw_response_bytes,
        )
        put(tmp_path, str((attempt / FILENAME).relative_to(tmp_path)), proof)
    journal.bind(attempt)
    with storage.locked():
        if historical:
            # A proven historical Aug26 attempt is not an Aug28 start merely
            # because its captured Calendar's latest authorized close is Aug28.
            expire_unstarted(storage, context, resolution_ref=ref, synthetic=True, now=EXPIRED)
            assert storage.pending()["state"] == "IDLE"
        else:
            before = snapshot(tmp_path)
            with pytest.raises(ContractError, match="AUTO_EXPIRY_NATIVE_TARGET_PRESENT"):
                expire_unstarted(storage, context, resolution_ref=ref, synthetic=True, now=EXPIRED)
            assert storage.pending()["state"] == "ACTIVE" and snapshot(tmp_path) == before

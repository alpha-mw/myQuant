"""Actual native core sealing and full Factor closure validation for historical dates."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.market import daily_maintenance as daily
from quant_investor.market import historical_session as history
from quant_investor.market.maintenance_journal import DailyOperationJournal
from quant_investor.factors.production_rollover import validate_daily_maintenance_receipt
from quant_investor.factors.governance import FactorGovernanceError
from test_daily_evidence_requested_session import capture
from test_unified_factor_production_rollover import _maintenance_attempt, _write


def context(tmp_path, *, historical=True):
    workspace, receipt_path, _ = _maintenance_attempt(tmp_path)
    receipt = json.loads(receipt_path.read_bytes())
    attempt = receipt_path.parent
    if historical:
        value = capture("2026-08-21T20:20:00+08:00")
        raw_path = attempt / history.RAW_FILENAME
        _write(raw_path, value.raw_response_bytes)
        calendar = {**value.receipt, "raw_response_path": str(raw_path)}
        calendar_path = attempt / history.CALENDAR_FILENAME
        _write(calendar_path, calendar)
        proof = history.build_historical_session(
            requested_trade_date="20260820",
            previous_trade_date="20260819",
            calendar_bytes=calendar_path.read_bytes(),
            raw=value.raw_response_bytes,
        )
        proof_path = attempt / history.FILENAME
        proof_ref = {"path": str(proof_path), "sha256": _write(proof_path, proof)}
    else:
        calendar_path = attempt / history.CALENDAR_FILENAME
        calendar = json.loads(calendar_path.read_bytes())
        proof_ref = None
    _write(
        attempt / "started.json",
        {
            "state": "STARTED",
            "mode": "execute",
            "started_at": "2026-08-21T12:00:00Z" if historical else "2026-08-20T12:00:00Z",
        },
    )
    claim = attempt / "claim.json"
    claim_ref = {"path": str(claim), "sha256": _write(claim, {"test_only": "claim"})}
    rows = receipt["stage_results"][:3]
    for row in rows:
        daily._stage_record(attempt, row["stage"], row)
    ctx = daily.MaintenanceContext(
        workspace_root=workspace,
        run_root=attempt.parent.parent,
        attempt_root=attempt,
        target_date="20260820",
        attempt_slot="2020",
        mode="execute",
        close_session_receipt=calendar,
        close_session_receipt_path=calendar_path,
        close_session_receipt_sha256=hashlib.sha256(calendar_path.read_bytes()).hexdigest(),
        historical_session_ref=proof_ref,
    )
    return ctx, rows, claim_ref


def validate(ctx, ref):
    return validate_daily_maintenance_receipt(
        workspace_root=ctx.workspace_root,
        receipt_path=ref["path"],
        expected_receipt_sha256=ref["sha256"],
    )


def test_historical_seal_passes_full_native_factor_closure(tmp_path):
    ctx, rows, claim = context(tmp_path)
    ref = daily._seal_core_checkpoint(ctx, rows, logical_claim_ref=claim)
    result = validate(ctx, ref)
    assert result["target_date"] == "20260820"
    assert result["historical_session_ref"] == ctx.historical_session_ref
    assert result["historical_session"]["authorized_close_trade_date"] == "20260821"
    assert result["historical_session"]["evidence_classification"] == "RETROSPECTIVE_RECOMPUTE"
    assert result["historical_session"]["prospective"] is False
    assert result["historical_session"]["execution_authorized"] is False


@pytest.mark.parametrize("fault", ["sha", "path", "target", "time", "raw"])
def test_invalid_historical_source_cannot_seal_core(tmp_path, fault):
    ctx, rows, claim = context(tmp_path)
    if fault == "sha":
        ctx = replace(
            ctx, historical_session_ref={**ctx.historical_session_ref, "sha256": "0" * 64}
        )
    elif fault == "path":
        ctx = replace(
            ctx,
            historical_session_ref={
                **ctx.historical_session_ref,
                "path": str(ctx.attempt_root.parent / history.FILENAME),
            },
        )
    elif fault == "target":
        ctx = replace(ctx, target_date="20260819")
    elif fault == "time":
        _write(
            ctx.attempt_root / "started.json",
            {"state": "STARTED", "mode": "execute", "started_at": "2200-08-21T12:30:00Z"},
        )
    else:
        _write(ctx.attempt_root / history.RAW_FILENAME, b"{}")
    with pytest.raises(daily.DailyMaintenanceError):
        daily._seal_core_checkpoint(ctx, rows, logical_claim_ref=claim)
    assert not (ctx.attempt_root / "core-completion.json").exists()


@pytest.mark.parametrize("fault", ["missing", "schema", "extra", "source", "started", "v1_null"])
def test_factor_reader_rejects_historical_drift(tmp_path, fault):
    ctx, rows, claim = context(tmp_path)
    ref = daily._seal_core_checkpoint(ctx, rows, logical_claim_ref=claim)
    path = Path(ref["path"])
    value = json.loads(path.read_bytes())
    if fault == "missing":
        value.pop("historical_session_ref")
    elif fault == "schema":
        value["schema_version"] = "cn-daily-maintenance-core.v1"
    elif fault == "extra":
        value["prospective"] = True
    elif fault == "source":
        _write(ctx.attempt_root / history.RAW_FILENAME, b"{}")
    elif fault == "started":
        started = {"state": "STARTED", "mode": "execute", "started_at": "2200-08-21T12:30:00Z"}
        value["started_ref"]["sha256"] = _write(ctx.attempt_root / "started.json", started)
    else:
        value["schema_version"] = "cn-daily-maintenance-core.v1"
        value["historical_session_ref"] = None
    ref["sha256"] = _write(path, value)
    with pytest.raises(FactorGovernanceError):
        validate(ctx, ref)


def test_ordinary_core_bytes_match_pre_historical_sealer(tmp_path, monkeypatch):
    ctx, rows, claim = context(tmp_path, historical=False)
    monkeypatch.setattr(daily, "operation_provider_summary", lambda: {"test_only": "fixed"})
    raw = (Path(__file__).parent / "fixtures/core_checkpoint_v1_pre_historical.txt").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "f2081bba38218fb20394b9f5655e8d9c587741742ffc7e2048e24322faac7f84"
    )
    namespace = dict(vars(daily))
    exec(compile(raw, "pre-historical-core-oracle", "exec"), namespace)
    previous = namespace["_seal_core_checkpoint"](ctx, rows, logical_claim_ref=claim)
    path = Path(previous["path"])
    expected = path.read_bytes()
    path.unlink()  # Remove only this test's oracle output before exercising the new writer.
    current = daily._seal_core_checkpoint(ctx, rows, logical_claim_ref=claim)
    assert current == previous and path.read_bytes() == expected
    assert "historical_session" not in validate(ctx, current)


@pytest.mark.parametrize("reader", [False, True])
def test_started_bytes_changed_during_validation_are_rejected(tmp_path, monkeypatch, reader):
    ctx, rows, claim = context(tmp_path)
    ref = daily._seal_core_checkpoint(ctx, rows, logical_claim_ref=claim) if reader else None
    original = history.read_historical_fileset

    def mutate(**kwargs):
        result = original(**kwargs)
        path = ctx.attempt_root / "started.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(history, "read_historical_fileset", mutate)
    if reader:
        with pytest.raises(FactorGovernanceError, match="historical core evidence differs"):
            validate(ctx, ref)
    else:
        with pytest.raises(daily.DailyMaintenanceError, match="HISTORICAL_CORE_INVALID"):
            daily._seal_core_checkpoint(ctx, rows, logical_claim_ref=claim)
        assert not (ctx.attempt_root / "core-completion.json").exists()


def test_native_retry_reuses_calendar_and_seals_after_both_events(tmp_path):
    ctx, rows, _ = context(tmp_path)
    old_files = {p: p.read_bytes() for p in ctx.attempt_root.iterdir() if p.is_file()}
    journal = DailyOperationJournal(
        ctx.run_root,
        ctx.workspace_root,
        now=datetime(2026, 8, 21, 12, tzinfo=timezone.utc),
        slot="2020",
        mode="execute",
        _historical_trade_date="20260820",
    )
    journal.bind(ctx.attempt_root)
    first_claim = journal.claim_ref
    journal = DailyOperationJournal(
        ctx.run_root,
        ctx.workspace_root,
        now=datetime(2026, 8, 22, 13, tzinfo=timezone.utc),
        slot="2020",
        mode="execute",
        _historical_trade_date="20260820",
    )
    assert journal.claim_ref == first_claim
    retry = ctx.run_root / "attempts" / "retry"
    retry.mkdir(mode=0o700)
    _write(
        retry / "started.json",
        {"state": "STARTED", "mode": "execute", "started_at": "2026-08-22T13:00:00Z"},
    )
    journal.bind(retry)

    def forbidden(**kwargs):
        pytest.fail("retained Calendar must not be fetched again")

    retained = journal.acquire(forbidden, now=journal.now)
    assert not list(journal.path.glob("close-request-*.json"))
    assert len(journal.attempts()) == 2
    with pytest.raises(daily.DailyMaintenanceError, match="LOGICAL_TASK_ATTEMPT_BUDGET_EXHAUSTED"):
        journal.bind(ctx.run_root / "attempts" / "third")
    raw_path = retry / history.RAW_FILENAME
    _write(raw_path, retained.raw_response_bytes)
    close = {**retained.receipt, "raw_response_path": str(raw_path)}
    close_path = retry / history.CALENDAR_FILENAME
    close_sha = _write(close_path, close)
    proof = history.build_historical_session(
        requested_trade_date="20260820",
        previous_trade_date="20260819",
        calendar_bytes=close_path.read_bytes(),
        raw=retained.raw_response_bytes,
    )
    proof_path = retry / history.FILENAME
    proof_ref = {"path": str(proof_path), "sha256": _write(proof_path, proof)}
    for row in rows:
        daily._stage_record(retry, row["stage"], row)
    retry_ctx = replace(
        ctx,
        attempt_root=retry,
        close_session_receipt=close,
        close_session_receipt_path=close_path,
        close_session_receipt_sha256=close_sha,
        historical_session_ref=proof_ref,
    )
    before = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ref = daily._seal_core_checkpoint(retry_ctx, rows, logical_claim_ref=journal.claim_ref)
    after = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = validate(retry_ctx, ref)
    assert before <= result["sealed_at"] <= after
    assert result["historical_session"]["observed_at"] == "2026-08-21T12:20:00Z"
    assert (
        result["historical_session"]["observed_at"] < "2026-08-22T13:00:00Z" <= result["sealed_at"]
    )
    assert old_files == {p: p.read_bytes() for p in old_files}
    assert result["historical_session"]["prospective"] is False


@pytest.mark.parametrize(
    "stamp",
    [None, "invalid", "2200-01-01T00:00:00Z", "2026-08-21T12:10:00Z", "2026-08-21T11:59:59Z"],
)
def test_reader_rejects_missing_or_invalid_seal_time(tmp_path, stamp):
    ctx, rows, claim = context(tmp_path)
    ref = daily._seal_core_checkpoint(ctx, rows, logical_claim_ref=claim)
    path = Path(ref["path"])
    value = json.loads(path.read_bytes())
    if stamp is None:
        value.pop("sealed_at")
    else:
        value["sealed_at"] = stamp
    ref["sha256"] = _write(path, value)
    with pytest.raises(FactorGovernanceError):
        validate(ctx, ref)


def test_writer_rejects_calendar_observation_after_local_seal_clock(tmp_path, monkeypatch):
    ctx, rows, claim = context(tmp_path)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 21, 12, 10, tzinfo=timezone.utc)

    monkeypatch.setattr(daily, "datetime", Clock)
    with pytest.raises(daily.DailyMaintenanceError, match="HISTORICAL_CORE_INVALID"):
        daily._seal_core_checkpoint(ctx, rows, logical_claim_ref=claim)
    assert not (ctx.attempt_root / "core-completion.json").exists()

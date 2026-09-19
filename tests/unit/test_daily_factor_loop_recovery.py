from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Event

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.market import daily_maintenance as maintenance
from quant_investor.market.close_session_authority import CloseSessionAuthorityResult
from quant_investor.market.maintenance_journal import DailyOperationJournal
from quant_investor.factors.production_rollover import validate_daily_maintenance_receipt
from quant_investor.factors.production_authority import FactorProductionStore
from quant_investor.factors.production_outcomes import (
    _publish_outcome,
    _head,
    _sha,
    LABEL_POLICY,
    evaluate_cross_section,
)
from quant_investor.factors.production_outcome_sources import persist_source
from test_unified_factor_production_rollover import _maintenance_attempt
from test_unified_factor_production_observation import _inputs
from quant_investor.factors.production_observation import build_factor_production_observation


def _components(tmp_path):
    workspace, receipt_path, _ = _maintenance_attempt(tmp_path)
    (workspace / "data/private/cn_daily_maintenance").chmod(0o700)
    for directory in (workspace / "data/private/cn_daily_maintenance").rglob("*"):
        if directory.is_dir():
            directory.chmod(0o700)
    receipt = json.loads(receipt_path.read_bytes())
    close = json.loads(Path(receipt["close_session_receipt_ref"]["path"]).read_bytes())
    raw = Path(close["raw_response_path"]).read_bytes()
    stages = {r["stage"]: r for r in receipt["stage_results"]}
    callbacks = {stage: lambda _context, row=row: row for stage, row in stages.items()}
    return workspace, receipt, close, raw, callbacks


def test_core_checkpoint_is_consumable_while_macro_is_still_running(tmp_path):
    workspace, receipt, close, raw, callbacks = _components(tmp_path)
    arrived, release = Event(), Event()
    checkpoints = []

    def factor(ref):
        validated = validate_daily_maintenance_receipt(
            workspace_root=workspace,
            receipt_path=ref["path"],
            expected_receipt_sha256=ref["sha256"],
        )
        checkpoints.append(validated)
        return {"status": "FACTOR_TEST_PASSED", "provider_calls": False}

    def macro(_context):
        arrived.set()
        assert release.wait(10)
        return {
            "stage": "MACRO_RELEASE",
            "status": "BLOCKED",
            "write_performed": False,
            "blockers": ["MACRO_TEST_VETO"],
            "evidence": {},
        }

    components = maintenance.MaintenanceComponents(
        pit=callbacks["PIT"],
        market=callbacks["MARKET"],
        history=callbacks["HISTORY"],
        fundamental=callbacks["FUNDAMENTAL"],
        macro_release=macro,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            maintenance.run_cn_daily_maintenance,
            workspace_root=workspace,
            run_root=workspace / "data/private/cn_daily_maintenance",
            mode="execute",
            attempt_slot="2020",
            now=datetime(2026, 8, 20, 13, tzinfo=timezone.utc),
            components=components,
            close_authority=lambda **_: CloseSessionAuthorityResult(close, raw),
            core_completed=factor,
        )
        try:
            assert arrived.wait(10)
            assert checkpoints[0]["status"] == "CORE_COMPLETE"
            assert checkpoints[0]["upstream_maintenance_status"] == "IN_PROGRESS"
            assert not Path(checkpoints[0]["receipt_path"]).with_name("attempt.json").exists()
        finally:
            release.set()
        result = future.result(timeout=10)
    assert result["status"] == "PARTIAL"
    assert result["factor_loop"]["status"] == "FACTOR_TEST_PASSED"
    assert maintenance.cli_exit_required(result) is True
    assert Path(result["core_completion_ref"]["path"]).read_bytes()


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "shadow"),
        ("target_date", "20260819"),
        ("scope", "SYSTEM_INPUTS"),
        ("other_authority", "SYSTEM"),
    ],
)
def test_core_checkpoint_rejects_mode_session_scope_and_authority_drift(tmp_path, field, value):
    workspace, receipt, close, raw, callbacks = _components(tmp_path)
    result = maintenance.run_cn_daily_maintenance(
        workspace_root=workspace,
        run_root=workspace / "data/private/cn_daily_maintenance",
        mode="execute",
        attempt_slot="2020",
        now=datetime(2026, 8, 20, 13, tzinfo=timezone.utc),
        components=maintenance.MaintenanceComponents(
            pit=callbacks["PIT"],
            market=callbacks["MARKET"],
            history=callbacks["HISTORY"],
            fundamental=callbacks["FUNDAMENTAL"],
            macro_release=callbacks["MACRO_RELEASE"],
        ),
        close_authority=lambda **_: CloseSessionAuthorityResult(close, raw),
    )
    ref = result["core_completion_ref"]
    p = Path(ref["path"])
    body = json.loads(p.read_bytes())
    body[field] = value
    p.write_bytes(canonical_json_bytes(body))
    with pytest.raises(Exception):
        validate_daily_maintenance_receipt(
            workspace_root=workspace,
            receipt_path=p,
            expected_receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        )


def test_slot_budget_survives_new_attempt_and_install_drift_fails(tmp_path):
    root = tmp_path / "run"
    root.mkdir(mode=0o700)
    journal = DailyOperationJournal(
        root,
        tmp_path,
        now=datetime(2026, 8, 20, 13, tzinfo=timezone.utc),
        slot="2020",
        mode="execute",
    )
    calls = []

    def failure(**_):
        calls.append(1)
        raise TimeoutError()

    for _ in range(2):
        with pytest.raises(TimeoutError):
            journal.acquire(failure, now=datetime.now(timezone.utc))
    successor = DailyOperationJournal(
        root,
        tmp_path,
        now=datetime(2026, 8, 20, 14, tzinfo=timezone.utc),
        slot="2020",
        mode="execute",
    )
    with pytest.raises(maintenance.DailyMaintenanceError, match="BUDGET_EXHAUSTED"):
        successor.acquire(failure, now=datetime.now(timezone.utc))
    assert len(calls) == 2
    claim = journal.path / "claim.json"
    body = json.loads(claim.read_bytes())
    body["installation"]["python"] = "changed"
    claim.write_bytes(canonical_json_bytes(body))
    with pytest.raises(maintenance.DailyMaintenanceError, match="INSTALL_OR_POLICY_DRIFT"):
        DailyOperationJournal(
            root,
            tmp_path,
            now=datetime(2026, 8, 20, 14, tzinfo=timezone.utc),
            slot="2020",
            mode="execute",
        )


def test_interrupted_component_preserves_started_and_requires_journal_reconciliation(tmp_path):
    root = tmp_path / "run"
    root.mkdir(mode=0o700)
    journal = DailyOperationJournal(
        root,
        tmp_path,
        now=datetime(2026, 8, 20, 13, tzinfo=timezone.utc),
        slot="2020",
        mode="execute",
    )
    attempt = root / "attempts" / "one"
    attempt.mkdir(parents=True, mode=0o700)
    started = canonical_json_bytes({"state": "STARTED"})
    maintenance._write_once(attempt / "started.json", started)
    journal.bind(attempt)
    maintenance._write_once(
        attempt / "start-MARKET.json", canonical_json_bytes({"state": "STAGE_STARTED"})
    )
    result = journal.recover_or_replay()
    assert result["status"] == "IN_DOUBT"
    assert (attempt / "started.json").read_bytes() == started
    assert (attempt / "recovery.json").exists()


def _outcome_body(tmp_path):
    tmp_path.chmod(0o700)
    store = FactorProductionStore(tmp_path)
    inputs = _inputs()
    obs = build_factor_production_observation(
        inputs=inputs, factor_row=inputs["factor_rows"][0], registered_at="2026-08-20T13:00:00Z"
    )
    ref = persist_source(store, obs)
    classification = persist_source(store, {"cohort": "POST_CLOSE_LATE"})
    source = persist_source(store, {"price": 10})
    body = {
        "series_id": "a" * 64,
        "state": "EVALUATED",
        "authority": "NON_AUTHORIZING",
        "other_authority": "NONE",
        "observation_ref": ref,
        "horizon": 1,
        "label_policy": LABEL_POLICY,
        "origin_session": "20260820",
        "end_session": "20260821",
        "signal_evidence": {"factor_id": inputs["factor_rows"][0]["factor_id"]},
        "classification_ref": classification,
        "outcome_source_ref": source,
        "source_slice_sha256": _sha({"price": 10}),
        "evaluation_implementation": {"sha256": "b" * 64},
        "diagnostics": evaluate_cross_section({"A": "1", "B": "2"}, {"A": (10, 11), "B": (10, 12)}),
    }
    return store, body


def test_concurrent_outcome_writers_and_source_revision_have_one_head(tmp_path):
    store, body = _outcome_body(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _publish_outcome(store, dict(body)), range(2)))
    assert sum(created for _, created in results) == 1
    first = results[0][0]
    assert results[1][0] == first
    changed = {
        **body,
        "source_slice_sha256": _sha({"price": 11}),
        "outcome_source_ref": persist_source(store, {"price": 11}),
    }
    second, created = _publish_outcome(store, changed)
    assert created and second != first
    assert _head(store, body["series_id"]) == second
    assert json.loads(store.read(second["path"]).data)["payload"]["supersedes_ref"] == first
    assert _publish_outcome(store, changed) == (second, False)


def test_outcome_after_write_crash_recovers_without_duplicate(tmp_path, monkeypatch):
    store, body = _outcome_body(tmp_path)
    original = store.write_exact_once
    tripped = []

    def crash(path, raw):
        result = original(path, raw)
        if "outcomes/" in str(path) and not tripped:
            tripped.append(1)
            raise RuntimeError("injected after outcome publication before index")
        return result

    monkeypatch.setattr(store, "write_exact_once", crash)
    with pytest.raises(RuntimeError):
        _publish_outcome(store, body)
    ref, created = _publish_outcome(store, body)
    assert created is False
    assert _head(store, body["series_id"]) == ref


@pytest.mark.parametrize("surface", ["core", "final"])
@pytest.mark.parametrize("source", ["calendar", "PIT", "MARKET", "HISTORY"])
def test_both_receipt_paths_reject_exact_source_tamper(tmp_path, surface, source):
    workspace, receipt, close, raw, callbacks = _components(tmp_path)
    result = maintenance.run_cn_daily_maintenance(
        workspace_root=workspace,
        run_root=workspace / "data/private/cn_daily_maintenance",
        mode="execute",
        attempt_slot="2020",
        now=datetime(2026, 8, 20, 13, tzinfo=timezone.utc),
        components=maintenance.MaintenanceComponents(
            pit=callbacks["PIT"],
            market=callbacks["MARKET"],
            history=callbacks["HISTORY"],
            fundamental=callbacks["FUNDAMENTAL"],
            macro_release=callbacks["MACRO_RELEASE"],
        ),
        close_authority=lambda **_: CloseSessionAuthorityResult(close, raw),
    )
    rows = {r["stage"]: r for r in receipt["stage_results"]}
    targets = {
        "calendar": Path(result["attempt_receipt_ref"]["path"]).with_name("close-session.raw.json"),
        "PIT": Path(rows["PIT"]["evidence"]["pit_binding"]["canonical_path"]),
        "MARKET": Path(rows["MARKET"]["evidence"]["snapshot_manifest_path"]),
        "HISTORY": Path(rows["HISTORY"]["evidence"]["audit_path"]),
    }
    targets[source].write_bytes(b'{"tampered":true}')
    selected = result["core_completion_ref" if surface == "core" else "attempt_receipt_ref"]
    with pytest.raises(Exception):
        validate_daily_maintenance_receipt(
            workspace_root=workspace,
            receipt_path=selected["path"],
            expected_receipt_sha256=selected["sha256"],
        )


@pytest.mark.parametrize(
    "recovery,capture_parent_exists", [(False, True), (True, True), (True, False)]
)
def test_new_signal_failure_does_not_skip_historical_settlement(
    tmp_path, monkeypatch, recovery, capture_parent_exists
):
    from types import SimpleNamespace
    from quant_investor.market import daily_factor_loop as module
    from quant_investor.cli import unified

    loop = object.__new__(module.DailyFactorLoop)
    loop.workspace = tmp_path
    loop.run_root = tmp_path
    loop.stages = {}
    capture_parent = tmp_path / "calendar-captures"
    if capture_parent_exists:
        capture_parent.mkdir(mode=0o700)
    loop.context = {
        "release_commit": "a" * 40,
        "calendar_capture_parent": str(capture_parent),
        "release_install_input_ref": {"path": "release.json", "sha256": "b" * 64},
        "release_repository_root": str(tmp_path),
    }
    loop.store = SimpleNamespace(read=lambda path: SimpleNamespace(byte_sha256="c" * 64))
    monkeypatch.setattr(
        module,
        "validate_daily_maintenance_receipt",
        lambda **kwargs: {
            "target_date": "20260820",
            "close_session_receipt_ref": {"path": "close.json", "sha256": "d" * 64},
        },
    )
    monkeypatch.setattr(
        module,
        "register_factor_production_observations",
        lambda *args, **kwargs: {"status": "NO_ACTION"},
    )

    captures = []

    def failed(**kwargs):
        captures.append(kwargs)
        assert kwargs["cutoff_date"] == "2026-08-20"
        raise ValueError("injected new signal source failure")

    monkeypatch.setattr(unified, "system_calendar_capture", failed)
    settled = []
    loop._settle = lambda ref: settled.append(ref) or {"new_evaluation_count": 1}
    loop._save_state = lambda state: None
    loop.report = lambda **kwargs: loop.stages
    callback = loop._replay_core_completed if recovery else loop.core_completed
    result = callback({"path": "core.json", "sha256": "e" * 64})
    assert result["factor_rollover"]["status"] == "BLOCKED"
    assert len(captures) == (0 if recovery else 1)
    if recovery and capture_parent_exists:
        assert result["factor_rollover"]["blocker"] == "CORE_RECOVERY_CALENDAR_CAPTURE_MISSING"
    if not capture_parent_exists:
        assert not capture_parent.exists()
    assert loop._require_existing_calendar_capture is False
    assert len(settled) == 1
    assert result["outcome_settlement"]["new_evaluation_count"] == 1


@pytest.mark.parametrize("previous", [False, True])
def test_existing_capture_replay_flag_restored_after_exception(monkeypatch, previous):
    from quant_investor.market.daily_factor_loop import DailyFactorLoop

    loop = object.__new__(DailyFactorLoop)
    loop._require_existing_calendar_capture = previous

    def fail(checkpoint):
        assert loop._require_existing_calendar_capture is True
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(loop, "core_completed", fail)
    with pytest.raises(RuntimeError, match="controlled failure"):
        loop._replay_core_completed({})
    assert loop._require_existing_calendar_capture is previous


def test_outcome_revision_fork_is_rejected(tmp_path):
    from quant_investor.contracts import seal_artifact
    from quant_investor.factors.production_outcomes import KIND, ROOT

    store, body = _outcome_body(tmp_path)
    first, _ = _publish_outcome(store, body)
    payload = json.loads(store.read(first["path"]).data)["payload"]
    fork = {k: v for k, v in payload.items() if k != "outcome_id"}
    fork["evaluated_at"] = "2026-08-22T00:00:00Z"
    identity = "production-outcome-" + _sha(fork)
    artifact = seal_artifact(
        KIND, {"outcome_id": identity, **fork}, created_at=fork["evaluated_at"]
    )
    store.write_exact_once(
        ROOT / body["series_id"] / (identity + ".json"), canonical_json_bytes(artifact)
    )
    with pytest.raises(Exception, match="FORK_OR_CYCLE"):
        _head(store, body["series_id"])


def test_launcher_keeps_terminal_receipt_for_credential_or_config_failure(tmp_path):
    receipt_id = "slot-2020-20260905T120000Z-12345"
    start = maintenance.write_launcher_record(
        run_root=str(tmp_path), receipt_id=receipt_id, phase="STARTED"
    )
    directory = Path(start["launcher_receipt_ref"]["path"]).parent
    (directory / "recovery.stderr.log").write_bytes(b"CONFIGURATION_BLOCKED")
    (directory / "recovery.stderr.log").chmod(0o600)
    (directory / "recovery.stdout.json").write_bytes(b"")
    (directory / "recovery.stdout.json").chmod(0o600)
    end = maintenance.write_launcher_record(
        run_root=str(tmp_path), receipt_id=receipt_id, phase="ENDED", exit_code=3
    )
    payload = json.loads(Path(end["launcher_receipt_ref"]["path"]).read_bytes())
    assert payload["process_exit_code"] == 3
    assert len(payload["output_refs"]) == 3
    assert Path(start["launcher_receipt_ref"]["path"]).exists()


def test_auxiliary_restated_request_keeps_bounded_backoff(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from quant_investor.market import fundamental_mart

    calls = []
    delays = []

    def provider(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("synthetic transient transport failure")
        return pd.DataFrame()

    monkeypatch.setattr(fundamental_mart.time, "sleep", lambda seconds: delays.append(seconds))
    rows, count, _ = fundamental_mart._fetch_restated_rows(
        provider,
        symbol="000001.SZ",
        table="balancesheet",
        table_start_text="20260101",
        end_text="20260904",
        primary=pd.DataFrame(),
        limiter=SimpleNamespace(wait=lambda: None),
        attempt_limit=2,
        initial_backoff=0.25,
        maximum_backoff=1,
    )
    assert rows.empty and count == 2
    assert delays == [0.25]

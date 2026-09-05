from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_investor.market import scope_transition as st
from quant_investor.market.pit_universe import (
    PITUniverseStore,
    publish_pit_universe_capture,
    validate_pit_universe_capture,
    acquire_pit_universe_capture,
)
from tests.unit.test_pit_refresh_capture import _CountingProvider, _acquire


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(st.encoded(value))
    path.chmod(0o600)
    return st.reference(path)


@pytest.fixture
def transition(tmp_path):
    root = tmp_path.resolve()
    scope = root / "data/cn_universe/cn_index_components.json"
    old = put(scope, {"full_a": ["000001.SZ", "000002.SZ"]})
    oldcopy = put(root / "inputs/old.json", {"full_a": ["000001.SZ", "000002.SZ"]})
    store = PITUniverseStore(root_dir=root / "data/parquet/cn/reference")
    first = acquire_pit_universe_capture(
        _CountingProvider(extra_listed=True),
        capture_root=root / "first/capture",
        observed_at="2026-08-19T09:19:00Z",
        source_run_id="unit-first",
    )
    publish_pit_universe_capture(
        first["capture_receipt_path"],
        first["capture_receipt_sha256"],
        store=store,
        canonical_scope_path=scope,
        expected_scope_sha256=old["sha256"],
    )
    new = put(root / "inputs/new.json", {"full_a": ["000001.SZ", "000003.SZ"]})
    capture = _acquire(root / "second", _CountingProvider(extra_listed=True))
    raw = put(root / "inputs/calendar.json", {"fixture": True})
    close = put(
        root / "inputs/close.json",
        {
            "status": "TARGET_AUTHORIZED",
            "target_trade_date": "20260819",
            "endpoint_url": "https://api.tushare.pro/",
            "raw_response_path": raw["path"],
            "raw_response_sha256": raw["sha256"],
        },
    )
    run = root / "data/private/cn_daily_maintenance"
    veto = put(
        run / "WRITE_VETO.json",
        {
            "schema_version": "cn-daily-maintenance-write-veto.v1",
            "target_date": "20260819",
            "attempt_slot": "2020",
            "attempt_root": str(run / "attempts/source"),
            "blockers": ["PIT_CONTRACT_BLOCKED", "UPSTREAM_STAGE_NOT_READY"],
        },
    )
    failed = put(
        run / "attempts/source/attempt.json",
        {
            "mode": "execute",
            "target_date": "20260819",
            "attempt_slot": "2020",
            "close_session_receipt_ref": close,
            "blockers": ["PIT_CONTRACT_BLOCKED", "UPSTREAM_STAGE_NOT_READY"],
            "canonical_write_count": 0,
            "canonical_unchanged": True,
            "write_veto_ref": veto,
            "stage_results": [
                {
                    "write_performed": False,
                    "evidence": {"error_code": "pit_frozen_scope_predecessor_binding_changed"},
                }
            ],
        },
    )
    run.chmod(0o700)
    (run / "attempts").chmod(0o700)
    market = put(root / "data/parquet/cn/_latest.json", {"fixture": "old-market"})
    fund = put(root / "data/parquet/cn/_fundamental_latest.json", {"fixture": "fund"})
    checkpoint = put(root / "data/staging/checkpoint/latest.json", {"revision": 1})
    for p in [
        run / ".daily-maintenance.lock",
        root / "data/parquet/cn/.market_writer.lock",
        Path(checkpoint["path"]).parent / ".checkpoint.lock",
    ]:
        p.touch(mode=0o600)
    q = {
        "schema_version": st.SCHEMA,
        "operation_id": "unit-scope",
        "workspace_root": str(root),
        "effective_date": "20260819",
        "canonical_scope_path": str(scope),
        "old_scope_ref": oldcopy,
        "new_scope_ref": new,
        "pit_capture_ref": {
            "path": capture["capture_receipt_path"],
            "sha256": capture["capture_receipt_sha256"],
        },
        "close_receipt_ref": close,
        "pit_pointer_ref": st.reference(store.manifest_path),
        "market_pointer_ref": market,
        "fundamental_pointer_ref": fund,
        "checkpoint_ref": checkpoint,
        "veto_ref": veto,
        "failed_attempt_ref": failed,
        "added": ["000003.SZ"],
        "removed": ["000002.SZ"],
        "authority": "OWNER_AUTHORIZED_DATA_SCOPE_ONLY",
    }
    request = root / "inputs/request.json"
    request_ref = put(request, q)
    return root, store, q, request, request_ref["sha256"]


def validate(fixture, with_request=True):
    root, store, q, request, sha = fixture
    return validate_pit_universe_capture(
        q["pit_capture_ref"]["path"],
        q["pit_capture_ref"]["sha256"],
        store=store,
        canonical_scope_path=q["new_scope_ref"]["path"],
        expected_scope_sha256=q["new_scope_ref"]["sha256"],
        expected_parent_pointer_sha256=q["pit_pointer_ref"]["sha256"],
        **(
            {"scope_transition_request": request, "expected_scope_transition_sha256": sha}
            if with_request
            else {}
        ),
    )


def test_explicit_capture_admission_preserves_historical_records(transition):
    result = validate(transition)
    assert result["scope_expansion_pending_count"] == 0
    assert result["row_count"] == 3
    assert result["full_a_scope_count"] == 2
    assert result["dynamic_whole_market_complete"] is True
    assert result["scope_transition_ref"]["sha256"] == transition[-1]
    assert st.reference(transition[1].manifest_path) == transition[2]["pit_pointer_ref"]


def test_default_capture_does_not_admit_changed_scope(transition):
    with pytest.raises(RuntimeError, match="predecessor_binding_changed"):
        validate(transition, False)


def test_request_alone_cannot_publish_outside_locked_transition(transition):
    root, store, q, request, sha = transition
    with pytest.raises(RuntimeError, match="REQUIRES_LOCKED_OPERATION"):
        publish_pit_universe_capture(
            q["pit_capture_ref"]["path"],
            q["pit_capture_ref"]["sha256"],
            store=store,
            canonical_scope_path=q["new_scope_ref"]["path"],
            expected_scope_sha256=q["new_scope_ref"]["sha256"],
            expected_parent_pointer_sha256=q["pit_pointer_ref"]["sha256"],
            scope_transition_request=request,
            expected_scope_transition_sha256=sha,
        )
    assert st.reference(store.manifest_path) == q["pit_pointer_ref"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("added", []),
        ("removed", []),
        ("effective_date", "20260818"),
        ("authority", "UNCONFIRMED"),
        ("canonical_scope_path", "/private/tmp/other.json"),
    ],
)
def test_invalid_request_never_writes(transition, field, value):
    root, store, q, request, sha = transition
    q[field] = value
    updated = put(request, q)
    with pytest.raises(RuntimeError):
        st.load_request(request, updated["sha256"])
    assert st.reference(store.manifest_path) == q["pit_pointer_ref"]


def test_request_modes_and_sha_fail_closed(transition):
    root, store, q, request, sha = transition
    request.chmod(0o644)
    with pytest.raises(RuntimeError, match="OWNER_ONLY"):
        st.load_request(request, sha)
    request.chmod(0o600)
    with pytest.raises(RuntimeError, match="SHA_MISMATCH"):
        st.load_request(request, "0" * 64)


def test_marker_blocks_market_pit_and_fundamental_readers(transition):
    from quant_investor.market.market_data_reader import MarketDataReader
    from quant_investor.market.fundamental_mart import build_canonical_scope_evidence

    root, store, q, request, sha = transition
    put(st.marker_path(root), {"request_sha256": sha})
    with pytest.raises(RuntimeError, match="SCOPE_TRANSITION_IN_PROGRESS"):
        store.load_generation_binding()
    with pytest.raises(RuntimeError, match="SCOPE_TRANSITION_IN_PROGRESS"):
        MarketDataReader(data_root=root / "data")._load_latest_payload()
    with pytest.raises(RuntimeError, match="SCOPE_TRANSITION_IN_PROGRESS"):
        build_canonical_scope_evidence(
            ["000001.SZ"],
            canonical_path=q["canonical_scope_path"],
            market_pointer_path=q["market_pointer_ref"]["path"],
            membership_path="unused",
            as_of="20260819",
        )
    with st._owned(sha):
        assert store.load_generation_binding()["generation_id"]


def test_marker_blocks_low_level_market_and_pit_writers(transition):
    from quant_investor.market.market_data_store import MarketDataStore

    root, store, q, request, sha = transition
    put(st.marker_path(root), {"request_sha256": sha})
    market = MarketDataStore(market="CN", data_root=root / "data")
    for lock in (store._writer_lock, market._market_writer_lock):
        with pytest.raises(RuntimeError, match="SCOPE_TRANSITION_IN_PROGRESS"):
            with lock():
                pytest.fail("ordinary writer passed marker")
        with st._owned(sha), lock():
            pass


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_date", "20260818"),
        ("attempt_slot", "1620"),
        ("schema_version", "unknown"),
        ("blockers", ["OTHER"]),
    ],
)
def test_wrong_source_veto_semantics_rejected_even_with_matching_hashes(transition, field, value):
    root, store, q, request, sha = transition
    v = json.loads(st.read_ref(q["veto_ref"]))
    v[field] = value
    q["veto_ref"] = put(Path(q["veto_ref"]["path"]), v)
    failure = json.loads(st.read_ref(q["failed_attempt_ref"]))
    failure["write_veto_ref"] = q["veto_ref"]
    q["failed_attempt_ref"] = put(Path(q["failed_attempt_ref"]["path"]), failure)
    updated = put(request, q)
    with pytest.raises(RuntimeError, match="SOURCE_FAILURE_INVALID"):
        st.load_request(request, updated["sha256"])


def install_components(monkeypatch, fixture):
    from quant_investor.market import daily_components, scope_transition_history

    root, store, q, request, sha = fixture

    def pit(context):
        head = store.load_generation_binding()
        if head["manifest"]["source_bindings"].get("scope_transition"):
            return {"status": "NO_ACTION", "write_performed": False, "blockers": [], "evidence": {}}
        publish_pit_universe_capture(
            q["pit_capture_ref"]["path"],
            q["pit_capture_ref"]["sha256"],
            store=store,
            canonical_scope_path=q["canonical_scope_path"],
            expected_scope_sha256=q["new_scope_ref"]["sha256"],
            expected_parent_pointer_sha256=q["pit_pointer_ref"]["sha256"],
            scope_transition_request=request,
            expected_scope_transition_sha256=sha,
        )
        return {"status": "READY", "write_performed": True, "blockers": [], "evidence": {}}

    def market(context):
        put(Path(q["market_pointer_ref"]["path"]), {"fixture": "new-market"})
        return {"status": "READY", "write_performed": True, "blockers": [], "evidence": {}}

    good = lambda context: {
        "status": "READY",
        "write_performed": False,
        "blockers": [],
        "evidence": {},
    }
    monkeypatch.setattr(
        daily_components,
        "build_default_components",
        lambda **_: SimpleNamespace(pit=pit, market=market, history=good),
    )
    monkeypatch.setattr(
        scope_transition_history,
        "repair_and_verify_added_history",
        lambda *a, **k: {"repaired_row_count": 0},
    )
    monkeypatch.setattr(
        st, "_verify_closure", lambda q: {"market": st.reference(q["market_pointer_ref"]["path"])}
    )


def execute(fixture):
    root, store, q, request, sha = fixture
    return st.run_scope_transition(
        workspace_root=root,
        run_root=root / "data/private/cn_daily_maintenance",
        mode="execute",
        attempt_slot="2020",
        request_path=request,
        request_sha256=sha,
    )


def test_concurrent_maintenance_lock_prevents_scope_write(transition):
    import fcntl

    root, store, q, request, sha = transition
    lock = root / "data/private/cn_daily_maintenance/.daily-maintenance.lock"
    before = Path(q["canonical_scope_path"]).read_bytes()
    with lock.open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="ALREADY_RUNNING"):
            execute(transition)
    assert Path(q["canonical_scope_path"]).read_bytes() == before
    assert not st.marker_path(root).exists()


def test_pointer_drift_prevents_scope_write(transition):
    root, store, q, request, sha = transition
    before = Path(q["canonical_scope_path"]).read_bytes()
    Path(q["market_pointer_ref"]["path"]).write_text('{"drift":true}')
    with pytest.raises(RuntimeError, match="SHA_MISMATCH"):
        execute(transition)
    assert Path(q["canonical_scope_path"]).read_bytes() == before


def test_pit_cas_rejects_stale_parent_after_validation(transition):
    root, store, q, request, sha = transition
    validate(transition)
    put(store.manifest_path, {"drift": True})
    with pytest.raises(RuntimeError):
        validate(transition)


def test_retry_after_pre_cas_failure_rechecks_market_before_scope_write(monkeypatch, transition):
    install_components(monkeypatch, transition)
    root, store, q, request, sha = transition
    original = st._write_new

    def fail(path, raw):
        if path.name == "scope-replaced.json":
            raise RuntimeError("before PIT CAS")
        return original(path, raw)

    monkeypatch.setattr(st, "_write_new", fail)
    assert execute(transition)["status"] == "BLOCKED"
    Path(q["market_pointer_ref"]["path"]).write_text('{"drift":true}')
    monkeypatch.setattr(st, "_write_new", original)
    with pytest.raises(RuntimeError, match="SHA_MISMATCH"):
        execute(transition)
    assert Path(q["canonical_scope_path"]).read_bytes() == st.read_ref(q["old_scope_ref"])


def test_vanished_veto_without_archive_never_becomes_ready(monkeypatch, transition):
    install_components(monkeypatch, transition)
    root, store, q, request, sha = transition
    original = st._recover_exact_veto

    def fail(*a, **kw):
        raise RuntimeError("before archive")

    monkeypatch.setattr(st, "_recover_exact_veto", fail)
    assert execute(transition)["status"] == "BLOCKED"
    Path(q["veto_ref"]["path"]).unlink()
    monkeypatch.setattr(st, "_recover_exact_veto", original)
    with pytest.raises(RuntimeError):
        execute(transition)
    assert st.marker_path(root).exists()
    op = root / "data/private/cn_daily_maintenance/scope_transitions" / sha
    assert not (op / "readiness.json").exists()


def test_archive_before_clear_receipt_crash_recovers_from_exact_intent(monkeypatch, transition):
    from quant_investor.market import daily_maintenance

    install_components(monkeypatch, transition)
    original = daily_maintenance._write_once
    fired = []

    def fail(path, raw):
        if path.name.endswith(".clear.json") and not fired:
            fired.append(True)
            raise RuntimeError("after archive before clear receipt")
        return original(path, raw)

    monkeypatch.setattr(daily_maintenance, "_write_once", fail)
    assert execute(transition)["status"] == "BLOCKED"
    recovered = execute(transition)
    assert recovered["claude_input_ready"] is True
    root, store, q, request, sha = transition
    op = root / "data/private/cn_daily_maintenance/scope_transitions" / sha
    recovery = json.loads((op / "veto-recovery.json").read_bytes())
    assert recovery["clear_receipt_ref"] is None
    assert st.read_ref(recovery["archived_veto_ref"])


def test_final_closure_uses_real_market_reference_shape(monkeypatch, transition):
    from quant_investor.market import market_data_store

    root, store, q, request, sha = transition
    put(st.marker_path(root), {"request_sha256": sha})
    Path(q["canonical_scope_path"]).write_bytes(st.read_ref(q["new_scope_ref"]))
    monkeypatch.setattr(
        market_data_store,
        "MarketDataStore",
        lambda **kw: SimpleNamespace(validate_latest=lambda: {"status": "passed"}),
    )
    with st._owned(sha):
        result = publish_pit_universe_capture(
            q["pit_capture_ref"]["path"],
            q["pit_capture_ref"]["sha256"],
            store=store,
            canonical_scope_path=q["canonical_scope_path"],
            expected_scope_sha256=q["new_scope_ref"]["sha256"],
            expected_parent_pointer_sha256=q["pit_pointer_ref"]["sha256"],
            scope_transition_request=request,
            expected_scope_transition_sha256=sha,
        )
        manifest = put(root / "data/parquet/cn/market-manifest.json", {"snapshot": "fixture"})
        put(
            Path(q["market_pointer_ref"]["path"]),
            {
                "manifest_path": manifest["path"],
                "latest_complete_trade_date": q["effective_date"],
                "coverage": {
                    "expected_scope_count": 2,
                    "expected_scope_sha256": st.digest(b"000001.SZ\n000003.SZ"),
                    "pit_membership_sha256": result["canonical_sha256"],
                    "pit_generation_manifest_sha256": result["generation_manifest_sha256"],
                },
            },
        )
        closure = st._verify_closure(q)
        assert closure["market_manifest_ref"] == manifest
        assert closure["scope_count"] == 2


@pytest.mark.parametrize(
    "crash_name",
    [
        "scope-replaced.json",
        "pit-committed.json",
        "market-committed.json",
        "terminal.json",
        "veto-recovery.json",
    ],
)
def test_crash_boundaries_recover_without_rewriting_predecessor(
    monkeypatch, transition, crash_name
):
    install_components(monkeypatch, transition)
    root, store, q, request, sha = transition
    original = st._write_new
    fired = []

    def fail_once(path, raw):
        if path.name == crash_name and not fired:
            fired.append(True)
            raise RuntimeError("simulated crash")
        return original(path, raw)

    monkeypatch.setattr(st, "_write_new", fail_once)
    first = execute(transition)
    assert first["status"] == "BLOCKED"
    if crash_name == "scope-replaced.json":
        assert Path(q["canonical_scope_path"]).read_bytes() == st.read_ref(q["old_scope_ref"])
    second = execute(transition)
    assert second["status"] in {"VERIFIED", "NO_ACTION"}, second
    assert not st.marker_path(root).exists()
    assert not Path(q["veto_ref"]["path"]).exists()
    assert st.read_ref(q["old_scope_ref"])
    assert execute(transition)["status"] == "NO_ACTION"


@pytest.mark.parametrize("tamper", ["schema", "status", "extra_key", "clear_receipt"])
def test_veto_recovery_or_clear_receipt_tamper_blocks_replay(monkeypatch, transition, tamper):
    install_components(monkeypatch, transition)
    assert execute(transition)["claude_input_ready"] is True
    root, store, q, request, sha = transition
    record = (
        root / "data/private/cn_daily_maintenance/scope_transitions" / sha / "veto-recovery.json"
    )
    value = json.loads(record.read_bytes())
    if tamper == "clear_receipt":
        Path(value["clear_receipt_ref"]["path"]).write_text("{}")
    else:
        if tamper == "schema":
            value["schema_version"] = "wrong"
        elif tamper == "status":
            value["status"] = "BLOCKED"
        else:
            value["extra"] = True
        record.write_bytes(st.encoded(value))
    with pytest.raises(RuntimeError):
        execute(transition)

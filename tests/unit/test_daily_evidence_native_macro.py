"""Native Macro generation postchecks over synthetic provider/local fixtures.

No generation tree, release loader, observation loader or transaction postcheck
is mocked. One explicitly labelled synthetic transaction-clock case verifies the
positive decision-time path; the real-clock case retains late-evidence rejection.
This is not yet the target-day Macro readiness/DAG proof.
"""

import pytest

from tests.unit.test_macro_release_calendar import _publish_initial
from tests.unit.test_macro_production_observation_bundle import _inputs, _publish
from quant_investor.macro.maintenance_transaction import _postcheck
from quant_investor.macro.store import load_observations
from quant_investor.macro.release_calendar import release_calendar_pointer_sha256


def test_real_macro_generations_pass_native_transaction_postcheck(tmp_path):
    calendar, evidence = _publish_initial(tmp_path / "release-inputs")
    observation_root = tmp_path / "macro-inputs"
    receipt = _publish(observation_root, _inputs(observation_root))
    rows, pointer = load_observations(observation_root / "observations")
    assert len(rows) == 39
    _postcheck(
        {
            "canonical_root": str(calendar.canonical_root),
            "new_pointer_sha256": release_calendar_pointer_sha256(
                canonical_root=calendar.canonical_root
            ),
        },
        {
            "canonical_root": str(observation_root / "observations"),
            "new_pointer_sha256": pointer["pointer_sha256"],
            "generation_id": pointer["generation_id"],
        },
    )
    assert receipt is not None and evidence is not None


@pytest.mark.parametrize("logical_clock", [False, True])
def test_real_macro_transaction_and_readiness_preserve_actual_availability(tmp_path, logical_clock):
    import hashlib
    import json
    from datetime import datetime, timezone, timedelta
    from quant_investor.macro import maintenance_transaction as transaction
    from quant_investor.macro import readiness_closure as readiness
    import pytest

    calendar, _ = _publish_initial(tmp_path / "release-inputs")
    source = tmp_path / "macro-inputs"
    initial = _publish(source, _inputs(source))
    from tests.unit.test_macro_production_observation_bundle import (
        _next_daily_fixture,
        _daily_kwargs,
    )
    from quant_investor.macro.production_observation_bundle import (
        publish_local_market_breadth_update,
    )

    daily = _next_daily_fixture(tmp_path / "local-next")
    updated = publish_local_market_breadth_update(
        **_daily_kwargs(daily),
        as_of="2026-07-16T04:30:00+00:00",
        canonical_observations_root=source / "observations",
        run_id="native-local-update",
        expected_pointer_sha256=initial["pointer_sha256"],
    )

    from tests.unit.test_macro_production_observation_bundle import _roll_fixture, _open_days
    from quant_investor.macro.production_observation_bundle import publish_local_market_breadth_roll

    open_path, open_sha, open_dates = _open_days(tmp_path / "open-days.json")
    rolled_fixture = _roll_fixture(tmp_path / "local-roll")
    rolled = publish_local_market_breadth_roll(
        **_daily_kwargs(rolled_fixture),
        target_as_of="20260716",
        decision_cutoff_at="2026-07-16T07:30:00+00:00",
        pinned_open_dates=open_dates,
        market_open_days_path=open_path,
        expected_market_open_days_sha256=open_sha,
        canonical_observations_root=source / "observations",
        run_id="native-local-roll",
        expected_pointer_sha256=updated["pointer_sha256"],
    )

    from tests.unit.test_macro_production_observation_bundle import _next_roll_fixture

    next_fixture = _next_roll_fixture(tmp_path / "local-next-roll")
    publish_local_market_breadth_roll(
        **_daily_kwargs(next_fixture),
        target_as_of="20260717",
        decision_cutoff_at="2026-07-17T07:30:00+00:00",
        pinned_open_dates=open_dates,
        market_open_days_path=open_path,
        expected_market_open_days_sha256=open_sha,
        canonical_observations_root=source / "observations",
        run_id="native-next-roll",
        expected_pointer_sha256=rolled["pointer_sha256"],
    )

    # Synthetic pointer authorities are explicit test inputs; the Macro generations,
    # two-pointer transaction, seven journal entries and loaders are native.
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        path.write_bytes(raw)
        path.chmod(0o600)
        return hashlib.sha256(raw).hexdigest()

    market = tmp_path / readiness.MARKET_POINTER
    pit = tmp_path / readiness.PIT_POINTER
    market_sha = put(
        market, {"snapshot_id": "synthetic-market", "latest_complete_trade_date": "20260717"}
    )
    pit_sha = put(pit, {"generation_id": "pit-20260717-synthetic"})
    release_root = tmp_path / readiness.RELEASE_POINTER.parent
    observations_root = tmp_path / readiness.OBSERVATIONS_POINTER.parent
    release_root.mkdir(parents=True, mode=0o700)
    observations_root.mkdir(parents=True, mode=0o700)
    for canonical in (release_root, observations_root):
        (canonical / "_generations").mkdir(mode=0o700)
    identity = "native-macro-fixture"
    base = tmp_path / readiness.TRANSACTION_ROOT / identity
    prepared = base / "_prepared" / identity / "prepared"
    prepared.mkdir(parents=True, mode=0o700)
    sealed = transaction.seal_prepared_macro_transaction(
        prepared_root=prepared,
        release_candidate_root=calendar.canonical_root,
        observations_candidate_root=source / "observations",
        release_canonical_root=release_root,
        observations_canonical_root=observations_root,
        expected_release_pointer_sha256=transaction.EMPTY_POINTER_SHA256,
        expected_observations_pointer_sha256=transaction.EMPTY_POINTER_SHA256,
        market_pointer_path=market,
        expected_market_pointer_sha256=market_sha,
        pit_pointer_path=pit,
        expected_pit_pointer_sha256=pit_sha,
        authority_mode="canonical",
        target_date="20260717",
    )
    (base / "journals").mkdir(mode=0o700)
    started = datetime.now(timezone.utc)
    (tmp_path / "SYNTHETIC-TIMING.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "logical_clock_simulated": logical_clock,
                "wall_clock_observed_at": started.isoformat(),
                "real_time_oos_eligible": False,
            }
        )
    )
    from unittest.mock import patch

    class LogicalTransactionClock(datetime):
        tick = 0

        @classmethod
        def now(cls, tz=None):
            cls.tick += 1
            instant = datetime(2026, 7, 17, 23, tzinfo=timezone.utc) + timedelta(seconds=cls.tick)
            return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

    # Only the test transaction clock is simulated. No stored timestamp is edited.
    # All native generation, CAS, journal and readiness validators remain unchanged.
    clock = LogicalTransactionClock if logical_clock else datetime
    with patch.object(transaction, "datetime", clock):
        transaction.commit_prepared_macro_transaction(
            prepared_path=sealed["prepared_path"],
            expected_prepared_sha256=sealed["prepared_sha256"],
            journal_root=base / "journals",
            journal_run_id=identity,
            market_pointer_path=market,
            expected_market_pointer_sha256=market_sha,
            pit_pointer_path=pit,
            expected_pit_pointer_sha256=pit_sha,
        )
    terminal = base / "journals" / identity / "0007-terminal.json"
    value = readiness.build_macro_readiness_closure(
        workspace_root=tmp_path,
        terminal_path=str(terminal.relative_to(tmp_path)),
        terminal_sha256=hashlib.sha256(terminal.read_bytes()).hexdigest(),
    )
    assert value["status"] == "READY"
    assert len(value["journal_refs"]) == 7
    assert (
        readiness.validate_macro_readiness_closure(workspace_root=tmp_path, closure=value) == value
    )
    if logical_clock:
        assert datetime.fromisoformat(value["available_at"]) < started
        verified = readiness.verify_current_macro_readiness_closure(
            workspace_root=tmp_path,
            closure=value,
            expected_target_date="20260717",
            decision_as_of="2026-07-17T23:59:00Z",
        )
        assert verified["target_date"] == "20260717"
        assert verified["available_at"] == value["available_at"]
    else:
        assert datetime.fromisoformat(value["available_at"]) >= started
    with pytest.raises(readiness.MacroReadinessClosureError, match="NOT_AVAILABLE_AT_DECISION"):
        readiness.verify_current_macro_readiness_closure(
            workspace_root=tmp_path,
            closure=value,
            expected_target_date="20260717",
            decision_as_of=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
        )
    with pytest.raises(readiness.MacroReadinessClosureError):
        readiness.verify_current_macro_readiness_closure(
            workspace_root=tmp_path,
            closure=value,
            expected_target_date="20260717",
            decision_as_of="2026-07-17T13:30:00Z",
        )

"""Automatic -> Part A execution with native Calendar/real storage; EOD/producers controlled."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json

import pytest
from _public_catchup_fixture import put
from test_automatic_catchup_resolution import fixture, NOW
from test_daily_evidence_public_catchup import snapshot
from quant_investor.operations import automatic_catchup_resolution as resolver
from quant_investor.operations import automatic_catchup_closure as closure
from quant_investor.operations.automatic_catchup_contract import run_path, AutomaticCatchupError
from quant_investor.operations.automatic_catchup_storage import AutomaticRunStorage, PENDING
from scripts import daily_automatic_catchup as auto
from scripts import daily_materialization, daily_catchup, daily_production


def execution_fixture(root, monkeypatch):
    request, ref, seal, calls, serving = fixture(root, monkeypatch)
    request["action"] = "CATCH_UP"
    ref = put(root, ref["path"], request)
    clock = [NOW]

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz) if tz is not None else clock[0].replace(tzinfo=None)

    for module in (resolver, closure, auto):
        monkeypatch.setattr(module, "datetime", FixedDatetime)
    original = daily_materialization.execute_daily_recipe

    def execute(**kwargs):
        result = original(**kwargs)
        ref = result["completion_ref"]
        day = ref["path"].split("/")[-2]
        recorded = daily_catchup.inspect_recorded_completion(trade_date=day)["recorded_completion"]
        recorded["native_validation_completed_at"] = clock[0].strftime("%Y-%m-%dT%H:%M:%SZ")
        return {**result, "completion_ref": put(root, ref["path"], recorded)}

    monkeypatch.setattr(daily_materialization, "execute_daily_recipe", execute)
    return request, ref, clock, calls, serving


def run(root, request, ref):
    return daily_production.dispatch_daily_request(
        workspace=str(root),
        request_ref=ref,
        release_install_ref=request["release_install_ref"],
        synthetic=True,
    )


def test_completed_eod_releases_lease_despite_failed_current_serving_and_repeat_is_readonly(
    tmp_path, monkeypatch
):
    request, ref, _, calls, serving = execution_fixture(tmp_path, monkeypatch)
    result = run(tmp_path, request, ref)
    assert result["status"] == "PARTIAL" and result["result"]["business_state"] == "INCOMPLETE"
    assert calls == [("20260827", True, False), ("20260828", False, True)]
    assert serving == ["20260828"]
    pending = AutomaticRunStorage(str(tmp_path)).pending()
    assert pending["state"] == "IDLE"
    before = snapshot(tmp_path)
    again = run(tmp_path, request, ref)
    assert again["status"] == "PARTIAL" and again["resolution_ref"] == result["resolution_ref"]
    assert snapshot(tmp_path) == before and len(calls) == 2
    assert serving == ["20260828", "20260828"]
    resolution = json.loads((tmp_path / result["resolution_ref"]["path"]).read_bytes())
    from quant_investor.operations.daily_contract import ContractError

    with pytest.raises(ContractError, match="AUTO_OWNED_REQUEST_INTERNAL_ONLY"):
        daily_production.dispatch_daily_request(
            workspace=str(tmp_path),
            request_ref=resolution["derived_request_ref"],
            release_install_ref=request["release_install_ref"],
            synthetic=True,
        )


@pytest.mark.parametrize("after", [1, 2, 3])
def test_resolution_and_each_derived_file_crash_recovers_exact_selection(
    tmp_path, monkeypatch, after
):
    request, ref, _, calls, _ = execution_fixture(tmp_path, monkeypatch)
    original = AutomaticRunStorage.write
    writes = [0]

    def crash(storage, path, raw, **kwargs):
        value = original(storage, path, raw, **kwargs)
        writes[0] += 1
        if writes[0] == after:
            raise OSError("after immutable write")
        return value

    with monkeypatch.context() as patch:
        patch.setattr(AutomaticRunStorage, "write", crash)
        with pytest.raises(OSError, match="after immutable write"):
            run(tmp_path, request, ref)
    assert calls == []
    saved = (tmp_path / run_path(ref, "resolution.v1.json")).read_bytes()
    from quant_investor.operations.dashboard_serving_contract import PREFIX, HEAD_JSON

    put(tmp_path, f"{PREFIX}/{HEAD_JSON}", b"moving later head must not be selected")
    result = run(tmp_path, request, ref)
    assert result["status"] == "PARTIAL" and len(calls) == 2
    assert (tmp_path / run_path(ref, "resolution.v1.json")).read_bytes() == saved


def test_active_unfinished_request_cannot_be_stolen_and_expired_unstarted_allows_new_history(
    tmp_path, monkeypatch
):
    request, ref, clock, calls, serving = execution_fixture(tmp_path, monkeypatch)
    put(tmp_path, closure.RUN_ROOT + "/.daily-maintenance.lock", b"")

    @contextmanager
    def crash(*args, **kwargs):
        raise OSError("after lease activation before callbacks")
        yield

    with monkeypatch.context() as patch:
        patch.setattr(auto, "automatic_execution", crash)
        with pytest.raises(OSError, match="after lease activation"):
            run(tmp_path, request, ref)
    original_pending = (tmp_path / PENDING).read_bytes()
    copied = put(tmp_path, "copied-auto.json", request)
    with pytest.raises(AutomaticCatchupError, match="AUTO_PENDING_REQUEST_CONFLICT") as conflict:
        run(tmp_path, request, copied)
    assert conflict.value.fields["pending_request_ref"] == ref
    assert (tmp_path / PENDING).read_bytes() == original_pending and calls == []
    clock[0] = datetime(2026, 8, 29, 14, tzinfo=timezone.utc)
    from test_daily_evidence_requested_session import capture

    captured = capture("2026-08-29T21:00:00+08:00")
    raw_ref = put(tmp_path, "next-calendar.raw", captured.raw_response_bytes)
    calendar_ref = put(tmp_path, "next-calendar.json", captured.receipt)
    newer = {**request, "calendar_ref": calendar_ref, "raw_calendar_ref": raw_ref}
    next_ref = put(tmp_path, "next-auto.json", newer)
    result = run(tmp_path, newer, next_ref)
    assert result["status"] == "SUCCEEDED" and serving == []
    assert calls == [("20260827", True, False), ("20260828", True, False)]
    old_closure = json.loads((tmp_path / run_path(ref, "closure.v1.json")).read_bytes())
    assert old_closure["state"] == "EXPIRED_UNSTARTED"
    before = snapshot(tmp_path)
    repeated = run(tmp_path, newer, next_ref)
    assert repeated["status"] == "NO_ACTION" and snapshot(tmp_path) == before
    with pytest.raises(AutomaticCatchupError, match="AUTO_RESOLUTION_EXPIRED_UNSTARTED"):
        run(tmp_path, request, ref)
    assert len(calls) == 2


def test_missing_inputs_report_exact_dates_before_committing_run(tmp_path, monkeypatch):
    request, ref, _, calls, _ = execution_fixture(tmp_path, monkeypatch)
    collection = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
    del collection["recipes"]["20260828"]
    request["recipe_ref"] = put(tmp_path, request["recipe_ref"]["path"], collection)
    ref = put(tmp_path, ref["path"], request)
    with pytest.raises(AutomaticCatchupError, match="AUTO_INPUTS_MISSING") as missing:
        run(tmp_path, request, ref)
    assert missing.value.fields == {"missing_input_dates": ["20260828"]}
    assert not (tmp_path / run_path(ref, "resolution.v1.json")).exists() and calls == []


def test_plan_remains_readonly_while_automatic_lock_is_busy(tmp_path, monkeypatch):
    request, ref, _, calls, _ = execution_fixture(tmp_path, monkeypatch)
    request["action"] = "PLAN"
    ref = put(tmp_path, "plan-auto.json", request)
    storage = AutomaticRunStorage(str(tmp_path))
    with storage.locked():
        before = snapshot(tmp_path)
        names = set(tmp_path.rglob("*"))
        result = run(tmp_path, request, ref)
        assert (
            result["status"] == "PLANNED" and result["result"] is result["resolution_ref"] is None
        )
        assert snapshot(tmp_path) == before and set(tmp_path.rglob("*")) == names and calls == []

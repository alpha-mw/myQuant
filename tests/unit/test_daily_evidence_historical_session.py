"""Historical Calendar edges use real native replay, without wall-clock backdating."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from quant_investor.market import requested_session as sessions
from quant_investor.market.close_session_authority import CloseSessionAuthorityError
from quant_investor.market.tushare_transport import OfficialTushareHttpsClient
from quant_investor.operations import catchup
from test_daily_evidence_catchup import fixture
from test_daily_evidence_requested_session import capture


@pytest.mark.parametrize(
    "now,previous,day,authorized",
    [
        ("2026-09-08T20:20:00+08:00", "20260903", "20260904", "20260908"),
        ("2026-09-08T10:00:00+08:00", "20260904", "20260907", "20260907"),
        ("2026-09-06T00:30:00+08:00", "20260903", "20260904", "20260904"),
    ],
)
def test_historical_edge_preserves_real_observation_and_never_authorizes(
    now, previous, day, authorized, monkeypatch
):
    value = capture(now)
    before = deepcopy(value.receipt), bytes(value.raw_response_bytes)

    def forbidden(*args, **kwargs):
        raise AssertionError("real provider must not be called")

    monkeypatch.setattr(OfficialTushareHttpsClient, "request", forbidden)
    result = sessions.classify_catchup_session(
        requested_trade_date=day,
        previous_trade_date=previous,
        receipt=value.receipt,
        raw=value.raw_response_bytes,
    )
    assert result == {
        "requested_trade_date": day,
        "previous_trade_date": previous,
        "authorized_close_trade_date": authorized,
        "observed_at": value.receipt["captured_at"],
        "classification": "HISTORICAL_CLOSED",
        "evidence_classification": "RETROSPECTIVE_RECOMPUTE",
        "prospective": False,
        "execution_authorized": False,
    }
    assert before == (value.receipt, value.raw_response_bytes)
    # Same evidence remains inadmissible to the ordinary current-session route.
    with pytest.raises(CloseSessionAuthorityError, match="OBSERVATION_DATE_MISMATCH"):
        sessions.classify_requested_session(
            requested_trade_date=day, receipt=value.receipt, raw=value.raw_response_bytes
        )


@pytest.mark.parametrize(
    "previous,day,code",
    [
        ("20260903", "2026094", "DATE_INVALID"),
        (None, "20260904", "DATE_INVALID"),
        ("20260903", "20260230", "DATE_INVALID"),
        ("20260904", "20260904", "DATE_ORDER_INVALID"),
        ("20260907", "20260904", "DATE_ORDER_INVALID"),
        ("20200101", "20260904", "COVERAGE_INCOMPLETE"),
        ("20260908", "20260909", "COVERAGE_INCOMPLETE"),
        ("20260905", "20260907", "PREVIOUS_NOT_OPEN"),
        ("20260904", "20260905", "REQUESTED_NOT_OPEN"),
        ("20260903", "20260907", "NOT_ADJACENT"),
        ("20260907", "20260908", "NOT_HISTORICAL"),
    ],
)
def test_invalid_historical_edges_fail_closed(previous, day, code):
    value = capture("2026-09-08T20:20:00+08:00")
    with pytest.raises(CloseSessionAuthorityError, match="CATCHUP_SESSION_" + code):
        sessions.classify_catchup_session(
            requested_trade_date=day,
            previous_trade_date=previous,
            receipt=value.receipt,
            raw=value.raw_response_bytes,
        )


def test_today_before_close_has_no_historical_authorization():
    value = capture("2026-09-08T10:00:00+08:00")
    with pytest.raises(CloseSessionAuthorityError, match="CATCHUP_SESSION_NOT_AUTHORIZED"):
        sessions.classify_catchup_session(
            requested_trade_date="20260908",
            previous_trade_date="20260907",
            receipt=value.receipt,
            raw=value.raw_response_bytes,
        )


@pytest.mark.parametrize("fault", ["raw", "receipt"])
def test_changed_calendar_is_controlled_replay_failure(fault):
    value = capture("2026-09-08T20:20:00+08:00")
    raw = value.raw_response_bytes
    if fault == "raw":
        raw += b" "
    else:
        value.receipt["target_trade_date"] = "20260904"
    with pytest.raises(CloseSessionAuthorityError, match="CATCHUP_SESSION_REPLAY_INVALID") as error:
        sessions.classify_catchup_session(
            requested_trade_date="20260904",
            previous_trade_date="20260903",
            receipt=value.receipt,
            raw=raw,
        )
    assert error.value.__cause__ is not None


def test_unexpected_replay_failure_is_not_hidden(monkeypatch):
    failure = RuntimeError("unexpected native failure")

    def fail(*args):
        raise failure

    monkeypatch.setattr(sessions, "replay_close_session_authority", fail)
    with pytest.raises(RuntimeError) as error:
        sessions.classify_catchup_session(
            requested_trade_date="20260904", previous_trade_date="20260903", receipt={}, raw=b""
        )
    assert error.value is failure


@pytest.mark.parametrize("observed_day,edge_count", [(29, 3), (28, 2)])
def test_existing_planner_checks_each_historical_edge(
    tmp_path, monkeypatch, observed_day, edge_count
):
    args = fixture(tmp_path, now=datetime(2026, 8, observed_day, 13, tzinfo=timezone.utc))
    checked = []
    native = catchup.classify_catchup_session

    def inspect(**kwargs):
        result = native(**kwargs)
        checked.append((result["previous_trade_date"], result["requested_trade_date"]))
        assert result["prospective"] is False and result["execution_authorized"] is False
        return result

    monkeypatch.setattr(catchup, "classify_catchup_session", inspect)
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    result = catchup.plan_catchup(**args, day_input_refs={})
    assert (
        checked
        == [
            ("20260825", "20260826"),
            ("20260826", "20260827"),
            ("20260827", "20260828"),
        ][:edge_count]
    )
    assert result["status"] == "BLOCKED"
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}

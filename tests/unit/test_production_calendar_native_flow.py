"""Native capture -> observed transport receipt -> v2 proof with explicit offline seams."""

from copy import deepcopy

import pytest

from quant_investor.market import future_calendar_producer as producer
from quant_investor.market.next_session_proof import read_next_session_proof
from quant_investor.market.next_session_failure import read_next_session_failure
from quant_investor.market import _calendar_production_transport as recorder
from _native_production_calendar_fixture import install_seam, offline_https
from test_production_future_calendar import configured, retained
from _verify_five_native_days import inventory


def setup(tmp_path, monkeypatch):
    calls = offline_https(monkeypatch)
    ref = install_seam(tmp_path, monkeypatch)
    context, state = configured(), retained()
    context["release_install_input_ref"] = ref
    context["release_repository_root"] = str(tmp_path)
    state["trade_date"] = "20260915"
    saves = []
    return (
        dict(
            workspace=tmp_path,
            context=context,
            context_sha="c" * 64,
            state=state,
            save_state=lambda value: saves.append(deepcopy(value)),
        ),
        saves,
        calls,
    )


def test_native_transport_capture_publication_and_readonly_bound_recovery(tmp_path, monkeypatch):
    args, saves, calls = setup(tmp_path, monkeypatch)
    state = producer.bind_future_calendar(**args)
    assert calls == ["DOCUMENTATION", "SSE", "SZSE", "BSE"]
    assert [s["phase"] for s in saves] == [
        "CORE_READY",
        "CAPTURE_SEALED",
        "TRANSPORT_BOUND",
        "FUTURE_BOUND",
    ]
    proof = read_next_session_proof(
        workspace=str(tmp_path),
        eod_trade_date=state["trade_date"],
        publication_ref=state["next_session_calendar_proof_ref"],
    )
    assert proof["proof"]["next_open_session"] == "20260916"
    assert proof["live_eligible"] is True and proof["consumer_admission"] is False
    before = inventory(tmp_path)
    monkeypatch.setattr(
        recorder,
        "_production_transport_scope",
        lambda **kw: pytest.fail("recovery activated recorder"),
    )
    args["save_state"] = lambda value: pytest.fail("bound recovery wrote")
    assert producer.bind_future_calendar(**args, recovery=True) == state
    assert inventory(tmp_path) == before
    assert len(calls) == 4


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("CAPTURE_SEALED", "PROVENANCE_UNAVAILABLE"),
        ("TRANSPORT_BOUND", "proof"),
        ("FUTURE_BOUND", "proof"),
    ],
)
def test_native_crash_boundaries_no_second_capture(tmp_path, monkeypatch, phase, expected):
    args, saves, calls = setup(tmp_path, monkeypatch)

    def crash(value):
        saves.append(deepcopy(value))
        if value["phase"] == phase:
            raise RuntimeError("native crash boundary")

    args["save_state"] = crash
    with pytest.raises(RuntimeError, match="native crash boundary"):
        producer.bind_future_calendar(**args)
    args["state"] = saves[-1]
    args["save_state"] = lambda value: None
    monkeypatch.setattr(
        recorder,
        "_production_transport_scope",
        lambda **kw: pytest.fail("recovery activated recorder"),
    )
    state = producer.bind_future_calendar(**args, recovery=True)
    assert calls == ["DOCUMENTATION", "SSE", "SZSE", "BSE"]
    if expected == "proof":
        result = read_next_session_proof(
            workspace=str(tmp_path),
            eod_trade_date=state["trade_date"],
            publication_ref=state["next_session_calendar_proof_ref"],
        )
        assert result["live_eligible"] is True
    else:
        result = read_next_session_failure(
            workspace=str(tmp_path),
            eod_trade_date=state["trade_date"],
            failure_ref=state["next_session_calendar_failure_ref"],
        )
        assert result["failure"]["failure_code"] == expected

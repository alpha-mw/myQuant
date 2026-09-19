"""Real capture/proof custody flow; installation closure alone is a unit-test seam."""

import hashlib
from copy import deepcopy

import pytest

from quant_investor.market import next_session_acquisition as acquisition
from quant_investor.market import future_calendar_producer as producer
from quant_investor.market._calendar_fixture_capability import _offline_calendar_fixture
from quant_investor.market.next_session_proof import read_next_session_proof
from test_tushare_calendar_authority import _patch_capture_release_closure, _docs, _provider_raw
from test_future_calendar_context import context, state
from _verify_five_native_days import inventory


def test_native_capture_proof_and_retained_recovery(tmp_path, monkeypatch):
    raw = _patch_capture_release_closure(tmp_path, monkeypatch)
    sha = hashlib.sha256(raw).hexdigest()
    (tmp_path / "release.json").write_bytes(raw)
    (tmp_path / "release.json").chmod(0o600)
    monkeypatch.setattr(
        acquisition, "verify_running_release_install_input", lambda *a, **k: {"state": "PASS"}
    )
    configured = context()
    configured["release_install_input_ref"] = {"path": "release.json", "sha256": sha}
    configured["release_repository_root"] = str(tmp_path)
    retained = state()
    retained["trade_date"] = "20250610"
    saved = []
    with _offline_calendar_fixture(
        workspace=tmp_path,
        trade_date="20250610",
        install_sha=sha,
        fixture_source_sha=hashlib.sha256(b"test_tushare_calendar_authority fixture").hexdigest(),
        documentation_raw=_docs(),
        provider_raw={k: _provider_raw(k) for k in ("SSE", "SZSE", "BSE")},
    ):
        args = dict(
            workspace=tmp_path,
            context=configured,
            context_sha="c" * 64,
            state=retained,
            save_state=lambda value: saved.append(deepcopy(value)),
        )
        result = producer.bind_future_calendar(**args)
        assert [row["phase"] for row in saved] == ["CORE_READY", "CAPTURE_BOUND", "FUTURE_BOUND"]
        proof = read_next_session_proof(
            workspace=str(tmp_path),
            eod_trade_date="20250610",
            publication_ref=result["next_session_calendar_proof_ref"],
        )
        assert proof["synthetic"] is True and proof["live_eligible"] is False
        assert proof["proof"]["next_open_session"] == "20250611"
        before = inventory(tmp_path)
        monkeypatch.setattr(
            producer, "capture_next_session_calendar", lambda **kw: pytest.fail("recovery acquired")
        )
        args["save_state"] = lambda value: pytest.fail("bound recovery wrote state")
        assert producer.bind_future_calendar(**args, recovery=True) == result
        assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "point,expected",
    [
        ("after_capture", "PROVENANCE_UNAVAILABLE"),
        ("after_evidence", "PROOF"),
        ("after_proof", "PROOF"),
        ("after_state", "PROOF"),
    ],
)
def test_native_crash_recovery_never_recaptures(tmp_path, monkeypatch, point, expected):
    from quant_investor.market.next_session_failure import read_next_session_failure

    raw = _patch_capture_release_closure(tmp_path, monkeypatch)
    sha = hashlib.sha256(raw).hexdigest()
    (tmp_path / "release.json").write_bytes(raw)
    (tmp_path / "release.json").chmod(0o600)
    monkeypatch.setattr(
        acquisition, "verify_running_release_install_input", lambda *a, **k: {"state": "PASS"}
    )
    configured = context()
    configured["release_install_input_ref"] = {"path": "release.json", "sha256": sha}
    configured["release_repository_root"] = str(tmp_path)
    retained = state()
    retained["trade_date"] = "20250610"
    durable = []

    def save(value):
        if (point == "after_evidence" and value["phase"] == "CAPTURE_BOUND") or (
            point == "after_proof" and value["phase"] == "FUTURE_BOUND"
        ):
            raise RuntimeError("injected crash")
        durable.append(deepcopy(value))
        if point == "after_state" and value["phase"] == "FUTURE_BOUND":
            raise RuntimeError("injected crash")

    with _offline_calendar_fixture(
        workspace=tmp_path,
        trade_date="20250610",
        install_sha=sha,
        fixture_source_sha="b" * 64,
        documentation_raw=_docs(),
        provider_raw={k: _provider_raw(k) for k in ("SSE", "SZSE", "BSE")},
    ):
        with monkeypatch.context() as crash:
            if point == "after_capture":

                def stop(**kw):
                    raise RuntimeError("injected crash")

                crash.setattr(producer, "publish_fixture_evidence", stop)
            with pytest.raises(RuntimeError, match="injected crash"):
                producer.bind_future_calendar(
                    workspace=tmp_path,
                    context=configured,
                    context_sha="c" * 64,
                    state=retained,
                    save_state=save,
                )
        before_capture = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.raw")}
        monkeypatch.setattr(
            producer,
            "capture_next_session_calendar",
            lambda **kw: pytest.fail("recovery provider call"),
        )
        result = producer.bind_future_calendar(
            workspace=tmp_path,
            context=configured,
            context_sha="c" * 64,
            state=deepcopy(durable[-1]),
            save_state=lambda v: durable.append(deepcopy(v)),
            recovery=True,
        )
        if expected == "PROOF":
            replay = read_next_session_proof(
                workspace=str(tmp_path),
                eod_trade_date="20250610",
                publication_ref=result["next_session_calendar_proof_ref"],
            )
            assert replay["synthetic"] is True
        else:
            failure = read_next_session_failure(
                workspace=str(tmp_path),
                eod_trade_date="20250610",
                failure_ref=result["next_session_calendar_failure_ref"],
            )
            assert failure["failure"]["failure_code"] == expected
        assert before_capture == {
            p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.raw")
        }


@pytest.mark.parametrize("sealed", [False, True])
def test_actual_selected_calendar_or_eod_cannot_be_repaired(tmp_path, monkeypatch, sealed):
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.operations.daily_journal import DailyJournal
    from quant_investor.operations.daily_contract import ContractError

    journal = DailyJournal(str(tmp_path), "20250610")
    if sealed:
        journal.storage.write(str(journal.root / "completion.v1.json"), b"{}")
    else:
        journal.storage.write(
            str(journal.root / "nodes/calendar/binding.json"),
            canonical_json_bytes({"request_key": "a" * 64}),
        )
    configured = context()
    configured["release_install_input_ref"] = {"path": "release.json", "sha256": "a" * 64}
    retained = state()
    retained["trade_date"] = "20250610"
    before = inventory(tmp_path)
    monkeypatch.setattr(
        producer,
        "capture_next_session_calendar",
        lambda **kw: pytest.fail("selected calendar recaptured"),
    )
    with _offline_calendar_fixture(
        workspace=tmp_path,
        trade_date="20250610",
        install_sha="a" * 64,
        fixture_source_sha="b" * 64,
        documentation_raw=_docs(),
        provider_raw={k: _provider_raw(k) for k in ("SSE", "SZSE", "BSE")},
    ):
        with pytest.raises(ContractError, match="ALREADY_SELECTED"):
            producer.bind_future_calendar(
                workspace=tmp_path,
                context=configured,
                context_sha="c" * 64,
                state=retained,
                save_state=lambda value: pytest.fail("selected calendar state written"),
                recovery=True,
            )
    assert inventory(tmp_path) == before

"""Consumer flow tests with native quote parsing; full EOD replay is separately exercised."""

from pathlib import Path
import sys
import hashlib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from scripts import daily_morning_consumer as consumer
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import EOD_NODE_IDS, ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from test_unified_morning_strategy import _quote_capture


def setup(tmp_path, monkeypatch):
    def put(path, value):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else canonical_json_bytes(value)
        p.write_bytes(raw)
        p.chmod(0o600)
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}

    ledger_path = "results/strategy_records/CN/aggressive_tech_manufacturing/record/ledger.parquet"
    p = tmp_path / ledger_path
    p.parent.mkdir(parents=True)
    pd.DataFrame([{"symbol": "002463.SZ", "shares": 100}]).to_parquet(p, index=False)
    ledger_ref = put(ledger_path, p.read_bytes())
    future = {"path": "future.json", "sha256": "a" * 64}
    calendar = put(
        "calendar-terminal.json", {"output_refs": {"next_session_calendar_proof": future}}
    )
    store = put("store-terminal.json", {"output_refs": {"ledger": ledger_ref}})
    decision = put(
        "decision-terminal.json",
        {"state": "SUCCEEDED", "finished_at": "2026-08-26T13:00:00Z"},
    )
    completion_ref = {
        "path": "results/operations/daily_production/CN/20260826/completion.v1.json",
        "sha256": "b" * 64,
    }
    completion = {
        "node_terminal_refs": {"calendar": calendar, "store": store, "decision": decision},
        "synthetic": True,
        "native_validation_completed_at": "2026-08-26T14:00:00Z",
    }
    monkeypatch.setattr(
        consumer, "inspect_recorded_completion", lambda **kw: {"recorded_completion": completion}
    )
    monkeypatch.setattr(
        consumer,
        "read_next_session_proof",
        lambda **kw: {
            "proof": {
                "next_open_session": "20260827",
                "schema_version": "cn-next-session-calendar-proof.v1",
            },
            "proof_sealed_at": "2026-08-26T13:00:00Z",
            "live_eligible": False,
            "synthetic": True,
            "projection": [{"date": d, "sse_is_open": 1} for d in ["20260826", "20260827"]],
        },
    )
    calls = []

    def replay(**kw):
        calls.append(kw)
        return {
            "native_replay_validated": True,
            "validated_nodes": sorted(EOD_NODE_IDS),
            "completion_ref": completion_ref,
            "trade_date": "20260826",
            "decision": {"status": "COMPLETE"},
        }

    monkeypatch.setattr(consumer, "replay_native_completion", replay)
    quote_path, quote_sha = _quote_capture(tmp_path)
    import json

    quote = json.loads((tmp_path / quote_path).read_text())
    policy_ref = put(
        "policy.json",
        {
            "schema_version": "morning-quote-policy.v1",
            "strategy_id": "aggressive_tech_manufacturing",
            "market": "CN",
            "effective_from": "20260801",
            "effective_through": "20260831",
            "revoked_at": None,
            "additional_symbols": [],
            "authority": FALSE_AUTHORITY,
        },
    )
    request = {
        "schema_version": "morning-strategy-request.v2",
        "action": "REPLAY",
        "run_date": "20260827",
        "previous_completion_ref": completion_ref,
        "quote_capture_ref": {"path": quote_path, "sha256": quote_sha},
        "quote_raw_ref": {k: quote["raw_ref"][k] for k in ["path", "sha256"]},
        "owner_policy_ref": policy_ref,
        "output_ref": None,
    }
    return request, calls


def inventory(root):
    return {
        str(p): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else None)
        for p in root.rglob("*")
    }


def test_replay_consumes_bound_inputs_and_writes_nothing(tmp_path, monkeypatch):
    request, calls = setup(tmp_path, monkeypatch)
    before = inventory(tmp_path)
    result = consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    assert result["command_status"] == "REPLAY_VERIFIED"
    assert result["admission"] == "RESEARCH_ONLY" and result["synthetic"] is True
    assert result["expected_symbols"] == ["002463.SZ"]
    assert len(calls) == 1
    assert inventory(tmp_path) == before


def test_synthetic_replay_cannot_pass_live_preflight(tmp_path, monkeypatch):
    request, _ = setup(tmp_path, monkeypatch)
    request["action"] = "PREFLIGHT"
    before = inventory(tmp_path)
    with pytest.raises(ContractError, match="LIVE_PROVENANCE_REQUIRED"):
        consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    assert inventory(tmp_path) == before


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "scope",
        "late",
        "synthetic",
        "ledger_ref",
        "missing_ref",
        "date",
        "rollover",
        "regressed",
        "calendar_eligibility",
        "calendar_version",
    ],
)
def test_live_preflight_requires_native_contemporaneous_ledger_without_writes(
    tmp_path, monkeypatch, fault
):
    from datetime import datetime, timezone

    request, _ = setup(tmp_path, monkeypatch)
    request["action"] = "PREFLIGHT"
    recorded = consumer.inspect_recorded_completion()["recorded_completion"]
    recorded["synthetic"] = False
    recorded["prospective_admission_state"] = "LEDGER_ELIGIBLE"
    recorded["prospective_ledger_ref"] = {"path": "ledger.json", "sha256": "a" * 64}
    original_future = consumer.read_next_session_proof
    monkeypatch.setattr(
        consumer,
        "read_next_session_proof",
        lambda **kw: {
            **original_future(**kw),
            "synthetic": False,
            "live_eligible": fault != "calendar_eligibility",
            "proof": {
                **original_future(**kw)["proof"],
                "schema_version": (
                    "cn-next-session-calendar-proof.v1"
                    if fault == "calendar_version"
                    else "cn-next-session-calendar-proof.v2"
                ),
            },
        },
    )
    original_replay = consumer.replay_native_completion
    proof = {
        "validation_scope": "NATIVE_LEDGER_DERIVATION",
        "recomputed": False,
        "classification": "CONTEMPORANEOUS",
        "prospective": True,
        "synthetic": False,
        "ledger_ref": recorded["prospective_ledger_ref"],
    }
    if fault == "scope":
        proof["validation_scope"] = "RECORDED_ONLY"
    elif fault == "late":
        proof["classification"] = "LATE_REGISTERED"
    elif fault == "synthetic":
        proof["synthetic"] = True
    elif fault == "ledger_ref":
        proof["ledger_ref"] = {"path": "other.json", "sha256": "b" * 64}
    elif fault == "missing_ref":
        proof["ledger_ref"] = None
        recorded["prospective_ledger_ref"] = None
    monkeypatch.setattr(
        consumer,
        "replay_native_completion",
        lambda **kw: {**original_replay(**kw), "synthetic": False, "ledger": proof},
    )

    class Clock(datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            crossed = fault == "rollover" and cls.calls > 1
            hour = 2 if fault == "regressed" and cls.calls > 1 else 3
            return datetime(
                2026, 8, 28 if fault == "date" or crossed else 27, hour, tzinfo=timezone.utc
            )

    monkeypatch.setattr(consumer, "datetime", Clock)
    before = inventory(tmp_path)
    if fault:
        with pytest.raises(ContractError, match="MORNING_|REF_SCHEMA_INVALID"):
            consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    else:
        result = consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
        assert result["command_status"] == "PREFLIGHT_COMPLETE"
        assert result["admission"] == "LIVE_RESEARCH_CONSUMER"
        assert result["prospective_admission_state"] == "NOT_CLAIMED"
        assert "receipt_ref" not in result
    assert inventory(tmp_path) == before


def test_future_calendar_changed_during_read_is_rejected(tmp_path, monkeypatch):
    request, _ = setup(tmp_path, monkeypatch)
    original = consumer.read_next_session_proof
    calls = []

    def changed(**kw):
        value = original(**kw)
        calls.append(kw)
        if len(calls) > 1:
            value = {**value, "proof_sealed_at": "2026-08-26T13:01:00Z"}
        return value

    monkeypatch.setattr(consumer, "read_next_session_proof", changed)
    before = inventory(tmp_path)
    with pytest.raises(ContractError, match="CALENDAR_PROOF_CHANGED_DURING_READ"):
        consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    assert inventory(tmp_path) == before

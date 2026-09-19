"""Historical receipt -> real input reader/quote parser; EOD and Calendar replay controlled."""

import hashlib
import importlib
import json
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.morning_receipt import receipt_path
from quant_investor.intelligence.morning import classify_sina_quote_timing
from test_daily_evidence_morning_consumer import setup, consumer, inventory

history = importlib.import_module("scripts.daily_morning_history")


@pytest.mark.parametrize(
    "fault", [None, "symbols", "report", "raw", "synthetic", "late", "future", "replay"]
)
def test_historical_reader_reconstructs_real_quote_and_owner_inputs(tmp_path, monkeypatch, fault):
    request, calls = setup(tmp_path, monkeypatch)
    recorded = consumer.inspect_recorded_completion()["recorded_completion"]
    recorded.update(
        synthetic=False,
        prospective_admission_state="LEDGER_ELIGIBLE",
        prospective_ledger_ref={"path": "ledger.json", "sha256": "c" * 64},
    )
    original_future = consumer.read_next_session_proof
    monkeypatch.setattr(
        consumer,
        "read_next_session_proof",
        lambda **kw: {
            **original_future(**kw),
            "synthetic": fault == "synthetic",
            "live_eligible": True,
            "proof": {
                **original_future(**kw)["proof"],
                "schema_version": "cn-next-session-calendar-proof.v2",
            },
        },
    )
    original_replay = consumer.replay_native_completion
    monkeypatch.setattr(
        consumer,
        "replay_native_completion",
        lambda **kw: {
            **original_replay(**kw),
            "synthetic": False,
            "ledger": {
                "validation_scope": "NATIVE_LEDGER_DERIVATION",
                "recomputed": False,
                "classification": "LATE_REGISTERED" if fault == "late" else "CONTEMPORANEOUS",
                "prospective": True,
                "synthetic": False,
                "ledger_ref": recorded["prospective_ledger_ref"],
            },
        },
    )

    def put(path, value):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        parent = p.parent
        while parent != tmp_path:
            parent.chmod(0o700)
            parent = parent.parent
        raw = value if type(value) is bytes else canonical_json_bytes(value)
        p.write_bytes(raw)
        p.chmod(0o600)
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}

    quote = json.loads((tmp_path / request["quote_capture_ref"]["path"]).read_bytes())
    timing = classify_sina_quote_timing(quote["request_time"], run_date=request["run_date"])
    declarations = {
        "research_only": "true",
        "broker": "false",
        "live_order": "false",
        "actual_holdings_mutation": "false",
        **timing,
    }
    report = "# Research\n" + "\n".join(f"{k}={v}" for k, v in declarations.items())
    request.update(
        action="SEAL",
        output_ref=put(
            "results/operations/morning_strategy/CN/20260827/0945-strategy.v2.md", report.encode()
        ),
    )
    request_ref = put("seal-request.json", request)
    receipt = {
        "schema_version": "morning-strategy-run.v2",
        "run_date": "20260827",
        "previous_trade_date": "20260826",
        "request_ref": request_ref,
        **{
            key: request[key]
            for key in (
                "previous_completion_ref",
                "quote_capture_ref",
                "quote_raw_ref",
                "owner_policy_ref",
                "output_ref",
            )
        },
        "validated_at": "2026-08-27T03:00:00Z",
        "status": "COMPLETE",
        "admission": "LIVE_RESEARCH_CONSUMER",
        "synthetic": False,
        "prospective_admission_state": "NOT_CLAIMED",
        "expected_symbols": ["002463.SZ"],
        "quote_timing": timing,
        "authority": dict(FALSE_AUTHORITY),
    }
    if fault == "symbols":
        receipt["expected_symbols"] = ["000001.SZ"]
    elif fault == "future":
        receipt["validated_at"] = "2999-08-27T03:00:00Z"
    elif fault == "replay":
        receipt["schema_version"] = "morning-strategy-replay.v2"
    reference = put(receipt_path("20260827"), receipt)
    if fault in {"report", "raw"}:
        path = request["output_ref" if fault == "report" else "quote_raw_ref"]["path"]
        (tmp_path / path).write_bytes(b"changed")

    def forbidden(*args, **kwargs):
        pytest.fail("historical reader reached live consumer or publisher")

    monkeypatch.setattr(consumer, "prepare_morning_consumer", forbidden)
    sealing = importlib.import_module("scripts.daily_morning_seal")
    monkeypatch.setattr(sealing, "_prepare_seal_context", forbidden)
    monkeypatch.setattr(sealing.MorningReceiptStorage, "write", forbidden)
    before = inventory(tmp_path)
    if fault:
        with pytest.raises((ContractError, ValueError)):
            history.read_historical_morning_receipt(workspace=str(tmp_path), receipt_ref=reference)
    else:
        result = history.read_historical_morning_receipt(
            workspace=str(tmp_path), receipt_ref=reference
        )
        assert result["command_status"] == "RECEIPT_VERIFIED"
        assert result["receipt"] == receipt
        assert len(calls) == 1
    assert inventory(tmp_path) == before

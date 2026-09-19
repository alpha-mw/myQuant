"""Morning-only publication with controlled native preflight, no live-admission claim."""

import hashlib
import importlib
import sys
from pathlib import Path
from datetime import datetime, timezone
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.morning_receipt import receipt_path
from test_daily_evidence_morning_receipt import receipt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
module = importlib.import_module("scripts.daily_morning_seal")


def fixture(root, monkeypatch, fault=None):
    base = receipt()

    def put(path, value):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        parent = p.parent
        while parent != root:
            parent.chmod(0o700)
            parent = parent.parent
        raw = value if type(value) is bytes else canonical_json_bytes(value)
        p.write_bytes(raw)
        p.chmod(0o600)
        return {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}

    request = {
        "schema_version": "morning-strategy-request.v2",
        "action": "SEAL",
        "run_date": "20260827",
    }
    for field, path, value in [
        (
            "previous_completion_ref",
            base["previous_completion_ref"]["path"],
            {"native_validation_completed_at": "2026-08-26T14:00:00Z"},
        ),
        ("quote_capture_ref", "capture.json", {"response_time": "2026-08-27T01:45:01Z"}),
        ("quote_raw_ref", "raw.json", {}),
        ("owner_policy_ref", "policy.json", {}),
    ]:
        request[field] = put(path, value)
    declarations = {
        "research_only": "true",
        "broker": "false",
        "live_order": "false",
        "actual_holdings_mutation": "false",
        **base["quote_timing"],
    }
    report = "# Morning research\n" + "\n".join(f"{k}={v}" for k, v in declarations.items())
    if fault == "report":
        report += "\nbroker=true"
    request["output_ref"] = put(base["output_ref"]["path"], report.encode())
    reference = put("request.json", request)
    prepared = {
        **request,
        "command_status": "PREFLIGHT_COMPLETE",
        "admission": "LIVE_RESEARCH_CONSUMER",
        "synthetic": False,
        "prospective_admission_state": "NOT_CLAIMED",
        "validated_at": "2026-08-27T01:50:00Z",
        "previous_trade_date": "20260826",
        "expected_symbols": base["expected_symbols"],
        "quote_timing": base["quote_timing"],
    }
    calls = []

    def preflight(**kwargs):
        assert kwargs["request"]["action"] == "PREFLIGHT"
        assert kwargs["request"]["output_ref"] is None
        calls.append(kwargs)
        if fault == "native":
            raise ContractError("NATIVE_REPLAY_REJECTED")
        if fault == "synthetic":
            prepared["synthetic"] = True
        if fault == "changed":
            (root / "raw.json").write_bytes(b"changed")
        return prepared

    monkeypatch.setattr(module, "prepare_morning_consumer", preflight)
    history = importlib.import_module("scripts.daily_morning_history")

    def read_evidence(**kwargs):
        # Native semantic replay is the controlled boundary in this fixture.
        calls.append(kwargs)

        def recheck():
            for key in (
                "previous_completion_ref",
                "quote_capture_ref",
                "quote_raw_ref",
                "owner_policy_ref",
            ):
                selected = kwargs["request"][key]
                raw = (root / selected["path"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != selected["sha256"]:
                    raise ContractError("MORNING_HISTORY_SOURCE_SHA_MISMATCH")

        recheck()
        ledger_ref = {"path": "ledger.json", "sha256": "c" * 64}
        return {
            "values": kwargs["request"],
            "recorded": {
                "native_validation_completed_at": "2026-08-26T14:00:00Z",
                "prospective_admission_state": "LEDGER_ELIGIBLE",
                "prospective_ledger_ref": ledger_ref,
            },
            "future_calendar": {
                "synthetic": False,
                "live_eligible": True,
                "proof": {"schema_version": "cn-next-session-calendar-proof.v2"},
                "proof_sealed_at": "2026-08-26T13:00:00Z",
            },
            "native": {
                "synthetic": False,
                "ledger": {
                    "validation_scope": "NATIVE_LEDGER_DERIVATION",
                    "recomputed": False,
                    "classification": "CONTEMPORANEOUS",
                    "prospective": True,
                    "synthetic": False,
                    "ledger_ref": ledger_ref,
                },
            },
            "quote": {
                "request_time": "2026-08-27T01:45:00Z",
                "response_time": "2026-08-27T01:45:01Z",
            },
            "symbols": base["expected_symbols"],
            "synthetic": False,
            "decision_completed_at": "2026-08-26T13:00:00Z",
            "recheck": recheck,
        }

    monkeypatch.setattr(history, "_read_morning_evidence", read_evidence)

    class Clock(datetime):
        minute = 55

        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 27, 1, cls.minute, tzinfo=timezone.utc)

    monkeypatch.setattr(module, "datetime", Clock)
    return reference, calls, Clock


def test_seal_writes_only_receipt_and_replay_preserves_original_time(tmp_path, monkeypatch):
    ref, calls, clock = fixture(tmp_path, monkeypatch)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    first = module.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    assert first["command_status"] == "PUBLISHED"
    after = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    assert set(after) - set(before) == {str(tmp_path / receipt_path("20260827"))}
    assert all(after[p] == v for p, v in before.items())
    clock.minute = 56
    again = module.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    assert again["command_status"] == "NO_ACTION"
    assert again["receipt_ref"] == first["receipt_ref"]
    assert again["receipt"]["validated_at"] == first["receipt"]["validated_at"]
    assert len(calls) == 2
    assert after == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("fault", ["report", "native", "synthetic", "changed"])
def test_seal_failure_never_publishes_success(tmp_path, monkeypatch, fault):
    ref, _, _ = fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(ContractError):
        module.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    assert not (tmp_path / receipt_path("20260827")).exists()


def test_live_receipt_readback_cannot_reach_publisher(tmp_path, monkeypatch):
    ref, calls, clock = fixture(tmp_path, monkeypatch)
    first = module.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("readback invoked receipt publisher")

    monkeypatch.setattr(module.MorningReceiptStorage, "write", forbidden)
    monkeypatch.setattr(module, "seal_morning_consumer", forbidden)
    monkeypatch.setattr(module, "_prepare_seal_context", forbidden)
    monkeypatch.setattr(module, "prepare_morning_consumer", forbidden)
    consumer = importlib.import_module("scripts.daily_morning_consumer")
    monkeypatch.setattr(consumer, "prepare_morning_consumer", forbidden)
    journal_io = importlib.import_module("quant_investor.operations.journal_storage")
    monkeypatch.setattr(journal_io.JournalStorage, "lock", forbidden)
    clock.minute = 56
    verified = module.read_morning_consumer_receipt(
        workspace=str(tmp_path), receipt_ref=first["receipt_ref"]
    )
    assert verified["command_status"] == "RECEIPT_VERIFIED"
    assert verified["receipt"] == first["receipt"]
    assert len(calls) == 2
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


def test_live_receipt_readback_revalidates_original_source_bytes(tmp_path, monkeypatch):
    ref, _, _ = fixture(tmp_path, monkeypatch)
    first = module.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    (tmp_path / "raw.json").write_bytes(b"changed")
    with pytest.raises(ContractError, match="SOURCE_SHA_MISMATCH"):
        module.read_morning_consumer_receipt(
            workspace=str(tmp_path), receipt_ref=first["receipt_ref"]
        )


def test_previous_day_receipt_readback_preserves_time_but_new_seal_rejects(tmp_path, monkeypatch):
    ref, _, _ = fixture(tmp_path, monkeypatch)
    first = module.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }

    class LaterClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 28, 1, 55, tzinfo=timezone.utc)

    monkeypatch.setattr(module, "datetime", LaterClock)
    with pytest.raises(ContractError, match="MORNING_SEAL_CLOCK_INVALID"):
        module.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    verified = module.read_morning_consumer_receipt(
        workspace=str(tmp_path), receipt_ref=first["receipt_ref"]
    )
    assert verified["receipt"] == first["receipt"]
    assert verified["command_status"] == "RECEIPT_VERIFIED"
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("field", ["completion", "decision", "request", "response"])
def test_historical_receipt_requires_each_original_custody_before_validation(
    tmp_path, monkeypatch, field
):
    ref, _, _ = fixture(tmp_path, monkeypatch)
    first = module.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    history = importlib.import_module("scripts.daily_morning_history")
    original = history._read_morning_evidence

    def later(**kwargs):
        evidence = original(**kwargs)
        stamp = "2026-08-27T02:00:00Z"
        if field == "completion":
            evidence["recorded"]["native_validation_completed_at"] = stamp
        elif field == "decision":
            evidence["decision_completed_at"] = stamp
        else:
            evidence["quote"][field + "_time"] = stamp
        return evidence

    monkeypatch.setattr(history, "_read_morning_evidence", later)
    with pytest.raises(ContractError, match="EVIDENCE_NOT_YET_AVAILABLE"):
        module.read_morning_consumer_receipt(
            workspace=str(tmp_path), receipt_ref=first["receipt_ref"]
        )

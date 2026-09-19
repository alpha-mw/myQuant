"""V3 receipt lifecycle with controlled full-EOD/live provenance; no real live-run claim."""

from datetime import datetime, timezone

import pytest

from test_morning_v3_consumer import setup
from test_morning_threshold_review import inventory
from _native_corporate_fixture import put
from scripts import daily_morning_consumer as consumer
from scripts import daily_morning_seal as seal
from scripts import daily_morning_history as history
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.morning_receipt import MorningReceiptStorage, receipt_path


def prepare(root, monkeypatch):
    request = setup(root, monkeypatch)
    # This deliberately supplies the live-provenance test seam needed for a receipt.
    # Native EOD admission and timing were already marked synthetic in the fixture.
    recorded = consumer.inspect_recorded_completion()["recorded_completion"]
    ledger = put(root, "fixtures/morning/prospective-seam.json", {"live_admission_test_seam": True})
    recorded.update(
        synthetic=False,
        native_validation_completed_at="2026-08-27T14:00:00Z",
        prospective_admission_state="LEDGER_ELIGIBLE",
        prospective_ledger_ref=ledger,
    )
    ref = put(root, request["previous_completion_ref"]["path"], recorded)
    request["previous_completion_ref"].update(ref)
    prior_future = consumer.read_next_session_proof
    monkeypatch.setattr(
        consumer,
        "read_next_session_proof",
        lambda **kw: {
            **prior_future(**kw),
            "synthetic": False,
            "live_eligible": True,
            "proof": {
                **prior_future(**kw)["proof"],
                "schema_version": "cn-next-session-calendar-proof.v2",
            },
        },
    )
    prior_replay = consumer.replay_native_completion
    monkeypatch.setattr(
        consumer,
        "replay_native_completion",
        lambda **kw: {
            **prior_replay(**kw),
            "synthetic": False,
            "ledger": {
                "validation_scope": "NATIVE_LEDGER_DERIVATION",
                "recomputed": False,
                "classification": "CONTEMPORANEOUS",
                "prospective": True,
                "synthetic": False,
                "ledger_ref": ledger,
            },
        },
    )

    class Clock(datetime):
        minute = 50

        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 28, 1, cls.minute, tzinfo=timezone.utc)

    monkeypatch.setattr(consumer, "datetime", Clock)
    monkeypatch.setattr(seal, "datetime", Clock)
    request["action"] = "PREFLIGHT"
    prepared = consumer.prepare_morning_consumer(workspace=str(root), request=request)
    report = root / "results/operations/morning_strategy/CN/20260828/0945-strategy.v3.md"
    report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent in (report.parent, report.parent.parent, report.parent.parent.parent):
        parent.chmod(0o700)
    report.write_bytes(prepared["report_markdown"].encode())
    report.chmod(0o600)
    import hashlib

    request.update(
        action="SEAL",
        output_ref={
            "path": str(report.relative_to(root)),
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
    )
    request_ref = put(root, "fixtures/morning/seal-request.json", request)
    return request, request_ref, Clock


def test_v3_seal_repeat_and_historical_rebuild_preserve_original_bytes(tmp_path, monkeypatch):
    request, ref, clock = prepare(tmp_path, monkeypatch)
    (tmp_path / request["threshold_policy_refs"]["trailing"]["path"]).chmod(0o644)
    result = seal.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    assert result["schema_version"] == "morning-strategy-seal-result.v3"
    assert result["receipt"]["schema_version"] == "morning-strategy-run.v3"
    assert result["receipt"]["review_summary_state"] == "COMPLETE_RESEARCH_REVIEW"
    assert result["receipt_ref"]["path"] == receipt_path("20260828", "v3")
    assert not (tmp_path / receipt_path("20260828")).exists()
    before = inventory(tmp_path)
    clock.minute = 55
    repeated = seal.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    assert repeated["command_status"] == "NO_ACTION"
    assert repeated["receipt_ref"] == result["receipt_ref"]
    assert repeated["receipt"]["validated_at"] == result["receipt"]["validated_at"]
    assert inventory(tmp_path) == before
    monkeypatch.setattr(
        consumer,
        "prepare_morning_consumer",
        lambda **kw: pytest.fail("historical read called live preflight"),
    )
    monkeypatch.setattr(
        seal, "_prepare_seal_context", lambda **kw: pytest.fail("historical read called publisher")
    )
    monkeypatch.setattr(
        MorningReceiptStorage, "write", lambda *a: pytest.fail("historical receipt write")
    )
    verified = history.read_historical_morning_receipt(
        workspace=str(tmp_path), receipt_ref=result["receipt_ref"]
    )
    assert (
        verified["receipt"] == result["receipt"]
        and verified["command_status"] == "RECEIPT_VERIFIED"
    )
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("fault", ["report", "review_hash", "policy_ref", "summary"])
def test_v3_history_cannot_accept_tampered_receipt_or_report(tmp_path, monkeypatch, fault):
    request, ref, _ = prepare(tmp_path, monkeypatch)
    result = seal.seal_morning_consumer(workspace=str(tmp_path), request_ref=ref)
    receipt = result["receipt"]
    if fault == "report":
        (tmp_path / request["output_ref"]["path"]).write_bytes(b"altered report")
        reference = result["receipt_ref"]
    else:
        if fault == "review_hash":
            receipt["threshold_review_sha256"] = "f" * 64
        elif fault == "summary":
            receipt["review_summary_state"] = "PARTIAL_RESEARCH_REVIEW"
        else:
            receipt["threshold_policy_refs"]["initial_stop"] = {
                "path": "fake.json",
                "sha256": "f" * 64,
            }
        reference = put(tmp_path, result["receipt_ref"]["path"], receipt)
    before = inventory(tmp_path)
    with pytest.raises((ContractError, ValueError, OSError)):
        history.read_historical_morning_receipt(workspace=str(tmp_path), receipt_ref=reference)
    assert inventory(tmp_path) == before


def test_public_v3_seal_dispatch_replays_exact_receipt(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from quant_investor.cli import morning_v2

    request, ref, _ = prepare(tmp_path, monkeypatch)
    install = put(
        tmp_path, "fixtures/morning/install-seam.json", {"synthetic_install_admission_seam": True}
    )

    @contextmanager
    def bridge(**kwargs):
        yield {
            "morning_seal": seal.seal_morning_consumer,
            "morning_receipt": history.read_historical_morning_receipt,
        }

    monkeypatch.setattr(morning_v2, "verified_native_context", bridge)
    result = morning_v2.run_morning_v2(
        workspace=str(tmp_path),
        request=request,
        request_ref=ref,
        release_repository_root=str(tmp_path),
        release_install_input_path=install["path"],
        expected_release_install_input_sha256=install["sha256"],
    )
    assert result["command_status"] == "PUBLISHED"
    assert result["receipt_ref"]["path"] == receipt_path("20260828", "v3")

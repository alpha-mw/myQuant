"""Calendar capability is optional; a recorded acquisition failure is not EOD failure."""

import pytest
from quant_investor.market.next_session_failure import publish_next_session_failure
from quant_investor.operations.core_pool import CoreContext
from quant_investor.operations.future_calendar_binding import future_calendar_outputs
from quant_investor.operations.daily_contract import ContractError


def test_failure_ref_binds_calendar_request_and_output_without_blocking(tmp_path):
    ref = publish_next_session_failure(
        workspace=str(tmp_path),
        eod_trade_date="20260901",
        phase="ACQUISITION",
        failure_code="ACQUISITION_FAILED",
    )
    source = {"path": "calendar.json", "sha256": "a" * 64}
    context = CoreContext(
        str(tmp_path), "20260901", "a" * 64, source, next_session_calendar_failure_ref=ref
    )
    context.pointer_ref = source
    snapshot = {
        "factor_generation": {
            "payload": {
                "calendar_compilation_ref": source,
                "calendar_capture_custody_attestation_ref": source,
            }
        }
    }
    context.snapshot = lambda: snapshot
    context._object = lambda value: value
    context.future_refs = context.future_outputs(snapshot)
    assert context.template("calendar")["input_refs"]["next_session_calendar_failure"] == ref
    assert context.core_outputs("calendar")["next_session_calendar_failure"] == ref
    assert "next_session_calendar_failure" not in context.template("factor")["input_refs"]


def test_conflicting_capabilities_and_late_failure_fail_closed(tmp_path):
    with pytest.raises(ContractError, match="PROOF_FAILURE_CONFLICT"):
        future_calendar_outputs(
            workspace=str(tmp_path), trade_date="20260901", proof_ref={}, failure_ref={}
        )
    ref = publish_next_session_failure(
        workspace=str(tmp_path),
        eod_trade_date="20260901",
        phase="ACQUISITION",
        failure_code="ACQUISITION_FAILED",
    )
    with pytest.raises(ContractError, match="AFTER_CALENDAR_TERMINAL"):
        future_calendar_outputs(
            workspace=str(tmp_path),
            trade_date="20260901",
            failure_ref=ref,
            finished_at="2020-01-01T00:00:00Z",
        )
    assert future_calendar_outputs(workspace=str(tmp_path), trade_date="20260901") == {}

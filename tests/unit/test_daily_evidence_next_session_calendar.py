"""Native Calendar capture validation, using synthetic provider wire examples."""

import pytest
from quant_investor.market.next_session_calendar import build_next_session_projection
from quant_investor.operations.daily_contract import ContractError
from quant_investor.system.errors import SystemContractError
from test_tushare_calendar_authority import _case, _raw_resolver


def kwargs():
    case = _case()
    return {
        "eod_trade_date": "20250610",
        "captures": {
            c["payload"]["exchange_id"]: c
            for c in case["captures"]
            if c["payload"]["exchange_id"] in {"SSE", "SZSE"}
        },
        "capability": case["capability"],
        "docs_raw": case["docs"],
        "raw_resolver": _raw_resolver(case),
    }


def test_native_next_session_projection_has_exact_twenty_two_rows():
    result = build_next_session_projection(**kwargs())
    assert result["observed_through_date"] == "20250701"
    assert result["next_open_session"] == "20250611"
    assert len(result["projection"]) == 22
    assert "consumer_admission" not in result


def test_capture_horizon_cannot_be_silently_shortened():
    values = kwargs()
    values["eod_trade_date"] = "20250611"
    with pytest.raises(ContractError, match="HORIZON_MISMATCH"):
        build_next_session_projection(**values)


def test_direct_exchange_set_cannot_be_substituted():
    values = kwargs()
    values["captures"].pop("SZSE")
    with pytest.raises(ContractError, match="EXCHANGE_SET_INVALID"):
        build_next_session_projection(**values)


def rewrite_native_capture(values, exchange, transform):
    """Rebuild a valid native artifact around deliberately different fixture rows."""
    import json
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.market.tushare_calendar_authority import (
        build_trusted_provider_calendar_capture,
    )
    from test_tushare_calendar_authority import _ref, CREATED_AT, CAPTURE_START, CUTOFF

    old = values["captures"][exchange]
    original_resolver = values["raw_resolver"]
    body = json.loads(original_resolver(old["payload"]["raw_file_ref"]))
    prior = ""
    for row in body["data"]["items"]:
        transform(row)
        row[3] = prior
        if row[2] == 1:
            prior = row[1]
    raw = canonical_json_bytes(body)
    ref = _ref(f"raw/changed-{exchange.lower()}.json", raw)
    values["captures"][exchange] = build_trusted_provider_calendar_capture(
        exchange_id=exchange,
        raw=raw,
        raw_file_ref=ref,
        capability=values["capability"],
        docs_raw=values["docs_raw"],
        captured_at=CREATED_AT,
        capture_start_date=CAPTURE_START.isoformat(),
        cutoff_date=CUTOFF.isoformat(),
        request_parameters_sanitized={
            "end_date": CUTOFF.strftime("%Y%m%d"),
            "exchange": exchange,
            "start_date": CAPTURE_START.strftime("%Y%m%d"),
        },
        response_headers={"content-type": "application/json"},
        created_at=CREATED_AT,
    )
    values["raw_resolver"] = lambda observed: (
        raw if observed == ref else original_resolver(observed)
    )


def test_disagreeing_valid_native_exchanges_cannot_choose_a_day():
    values = kwargs()

    def changed(row):
        if row[1] == "20250611":
            row[2] = 0

    rewrite_native_capture(values, "SSE", changed)
    with pytest.raises(ContractError, match="EXCHANGE_DISAGREEMENT"):
        build_next_session_projection(**values)


@pytest.mark.parametrize("close_future,error", [(True, "NO_LATER_OPEN"), (False, "EOD_NOT_OPEN")])
def test_no_future_open_or_closed_eod_is_not_inferred(close_future, error):
    values = kwargs()

    def closed(row):
        if (close_future and row[1] > "20250610") or (not close_future and row[1] == "20250610"):
            row[2] = 0

    for exchange in ("SSE", "SZSE"):
        rewrite_native_capture(values, exchange, closed)
    with pytest.raises(ContractError, match=error):
        build_next_session_projection(**values)


def test_raw_mutation_rejected_by_native_capture_validator():
    values = kwargs()
    original = values["raw_resolver"]
    values["raw_resolver"] = lambda ref: original(ref) + b" "
    with pytest.raises(SystemContractError, match="capture binding differs"):
        build_next_session_projection(**values)

"""Real Calendar replay plus read-only planning; no native execution claim."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import pytest
from quant_investor.market.close_session_authority import acquire_close_session_authority
from quant_investor.market.tushare_transport import replay_tushare_response_bytes
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.catchup import plan_catchup
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.native_input_contract import FIELDS


def day_input(day, *, v2=False):
    # Planner proves Calendar/date/shape only. Native input semantics remain the
    # separate execution preflight, and the plan grants no execution authority.
    value = dict.fromkeys(FIELDS)
    value.update(schema_version="cn-daily-native-inputs.v1", trade_date=day)
    if v2:
        value.update(
            schema_version="cn-daily-native-inputs.v2",
            next_session_calendar_proof_ref=None,
            next_session_calendar_failure_ref=None,
        )
    return value


def put(root, name, raw):
    path = root / name
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(root, *, now=None):
    class Client:
        def request(self, *, params, **kwargs):
            day = datetime.strptime(params["start_date"], "%Y%m%d")
            end = datetime.strptime(params["end_date"], "%Y%m%d")
            prior = day - timedelta(days=1)
            while prior.weekday() > 4:
                prior -= timedelta(days=1)
            previous = prior.strftime("%Y%m%d")
            rows = []
            while day <= end:
                opened = day.weekday() < 5
                rows.append(["SSE", day.strftime("%Y%m%d"), int(opened), previous])
                if opened:
                    previous = day.strftime("%Y%m%d")
                day += timedelta(days=1)
            raw = json.dumps(
                {
                    "code": 0,
                    "msg": "",
                    "detail": "",
                    "request_id": "synthetic",
                    "data": {
                        "fields": ["exchange", "cal_date", "is_open", "pretrade_date"],
                        "items": rows,
                        "count": 0,
                        "has_more": False,
                    },
                }
            ).encode()
            return replay_tushare_response_bytes(
                raw,
                api_name="trade_cal",
                expected_fields=["exchange", "cal_date", "is_open", "pretrade_date"],
            )

    result = acquire_close_session_authority(
        now=now or datetime(2026, 8, 29, 13, tzinfo=timezone.utc), client=Client()
    )
    return {
        "workspace": str(root),
        "calendar_ref": put(root, "calendar.json", canonical_json_bytes(result.receipt)),
        "raw_calendar_ref": put(root, "calendar.raw", result.raw_response_bytes),
        "previous_trade_date": "20260825",
        "target_trade_date": "20260828",
    }


def test_calendar_order_missing_inputs_and_zero_writes(tmp_path):
    args = fixture(tmp_path)
    inputs = {
        day: put(
            tmp_path,
            day + ".json",
            canonical_json_bytes(day_input(day)),
        )
        for day in ("20260826", "20260827", "20260828")
    }
    before = {str(p): p.read_bytes() for p in tmp_path.iterdir()}
    plan = plan_catchup(**args, day_input_refs=dict(reversed(list(inputs.items()))))
    assert plan["ordered_trade_dates"] == ["20260826", "20260827", "20260828"]
    assert list(plan["day_input_refs"]) == plan["ordered_trade_dates"]
    assert plan["status"] == "PLANNED" and plan["execution_authorized"] is False
    partial = plan_catchup(**args, day_input_refs={"20260828": inputs["20260828"]})
    assert partial["status"] == "BLOCKED" and partial["missing_input_dates"] == [
        "20260826",
        "20260827",
    ]
    assert before == {str(p): p.read_bytes() for p in tmp_path.iterdir()}
    closed = plan_catchup(**{**args, "target_trade_date": "20260829"}, day_input_refs={})
    assert closed["status"] == "NON_TRADING_DAY_NO_ACTION" and closed["ordered_trade_dates"] == []


def test_raw_calendar_and_day_binding_are_not_substituted(tmp_path):
    args = fixture(tmp_path)
    with pytest.raises(ContractError, match="SOURCE_SHA_MISMATCH"):
        plan_catchup(
            **{**args, "raw_calendar_ref": {**args["raw_calendar_ref"], "sha256": "0" * 64}},
            day_input_refs={},
        )
    wrong = put(
        tmp_path,
        "wrong.json",
        canonical_json_bytes(day_input("20260828")),
    )
    with pytest.raises(ContractError, match="DAY_INPUT_BINDING_INVALID"):
        plan_catchup(**args, day_input_refs={"20260826": wrong})
    with pytest.raises(ContractError, match="CALENDAR_RANGE_INCOMPLETE"):
        plan_catchup(**{**args, "previous_trade_date": "20200101"}, day_input_refs={})


def test_open_but_not_yet_closed_target_is_not_authorized(tmp_path):
    args = fixture(tmp_path, now=datetime(2026, 8, 28, 6, tzinfo=timezone.utc))
    with pytest.raises(ContractError, match="TARGET_NOT_AUTHORIZED"):
        plan_catchup(**args, day_input_refs={})


def test_v2_calendar_extension_is_preserved_and_unknown_shape_rejected(tmp_path):
    args = fixture(tmp_path)
    inputs = {}
    for day in ("20260826", "20260827", "20260828"):
        inputs[day] = put(tmp_path, day + ".json", canonical_json_bytes(day_input(day, v2=True)))
    plan = plan_catchup(**args, day_input_refs=inputs)
    assert plan["day_input_refs"] == inputs
    assert plan["execution_authorized"] is False
    bad = day_input("20260826", v2=True)
    bad["command"] = "arbitrary"
    inputs["20260826"] = put(tmp_path, "bad.json", canonical_json_bytes(bad))
    with pytest.raises(ContractError, match="SCHEMA_INVALID"):
        plan_catchup(**args, day_input_refs=inputs)

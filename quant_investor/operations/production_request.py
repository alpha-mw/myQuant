"""Action-specific daily-close request; produced refs are never initial EXECUTE inputs."""

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from .daily_contract import ContractError, GRAPH_SHA256, validate_ref
from .daily_journal import _validate_day
from .production_result import ACTIONS

SCHEMA = "cn-daily-production-request.v1"
HISTORICAL_SCHEMA = "cn-daily-production-request.v2"
FIELDS = frozenset(
    {
        "schema_version",
        "market",
        "strategy_id",
        "action",
        "target_trade_date",
        "graph_sha256",
        "release_install_ref",
        "recipe_ref",
        "maintenance_handoff_ref",
        "calendar_ref",
        "raw_calendar_ref",
        "previous_completion_ref",
        "day_input_refs",
    }
)


def validate_production_request(value: dict, *, release_install_ref: dict) -> dict:
    if type(value) is not dict or set(value) != FIELDS:
        raise ContractError("PRODUCTION_REQUEST_FIELDS_INVALID")
    if (
        type(value["schema_version"]) is not str
        or value["schema_version"] not in {SCHEMA, HISTORICAL_SCHEMA}
        or value["market"] != "CN"
        or value["strategy_id"] != "aggressive_tech_manufacturing"
        or value["graph_sha256"] != GRAPH_SHA256
    ):
        raise ContractError("PRODUCTION_REQUEST_SCOPE_INVALID")
    action = value["action"]
    if type(action) is not str or action not in ACTIONS:
        raise ContractError("PRODUCTION_REQUEST_ACTION_INVALID")
    if value["schema_version"] == HISTORICAL_SCHEMA and action != "EXECUTE":
        raise ContractError("PRODUCTION_REQUEST_HISTORICAL_EXECUTE_REQUIRED")
    _validate_day(value["target_trade_date"])
    validate_ref(value["release_install_ref"])
    validate_ref(release_install_ref)
    if value["release_install_ref"] != release_install_ref:
        raise ContractError("PRODUCTION_REQUEST_INSTALL_BINDING_MISMATCH")
    required = (
        {"recipe_ref"}
        if action in {"PLAN", "EXECUTE"}
        else (
            {"maintenance_handoff_ref"}
            if action == "RESUME"
            else {"calendar_ref", "raw_calendar_ref", "previous_completion_ref"}
        )
    )
    for key in {
        "recipe_ref",
        "maintenance_handoff_ref",
        "calendar_ref",
        "raw_calendar_ref",
        "previous_completion_ref",
    }:
        if key in required:
            validate_ref(value[key])
        elif action == "CATCH_UP" and key == "recipe_ref" and value[key] is not None:
            validate_ref(value[key])
        elif value[key] is not None:
            raise ContractError("PRODUCTION_REQUEST_PRODUCED_REF_FORBIDDEN")
    inputs = value["day_input_refs"]
    if type(inputs) is not dict or (action != "CATCH_UP" and inputs):
        raise ContractError("PRODUCTION_REQUEST_DAY_INPUTS_INVALID")
    for day, ref in inputs.items():
        _validate_day(day)
        if day > value["target_trade_date"]:
            raise ContractError("PRODUCTION_REQUEST_DAY_RANGE_INVALID")
        validate_ref(ref)
    return parse_canonical_json_bytes(canonical_json_bytes(value))

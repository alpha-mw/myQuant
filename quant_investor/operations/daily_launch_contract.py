"""Exact pre-credential launcher decision; no inspection grants write authority."""

from .automatic_catchup_contract import RESULT_SCHEMA
from .daily_contract import ContractError, validate_ref
from .daily_journal import _false_authority
from .production_result import validate_production_result
from .daily_journal import _validate_day

SCHEMA = "cn-daily-launch-inspection.v1"
BOOTSTRAP_SCHEMA = "cn-daily-launch-inspection.v2"
REGISTERED_SCHEMA = "cn-daily-launch-inspection.v3"
MODES = {"COMPLETE_READ_ONLY": 0, "LOCAL_REPAIR": 10, "PRODUCER_REQUIRED": 11}
FIELDS = {"schema_version", "request_ref", "release_install_ref", "mode", "result", "authority"}


def validate_launch_inspection(value, *, request_ref, release_install_ref, target_trade_date=None):
    if type(value) is dict and value.get("schema_version") == REGISTERED_SCHEMA:
        if set(value) != FIELDS | {"recovery_scope"} or type(value["recovery_scope"]) is not str:
            raise ContractError("AUTO_LAUNCH_RECOVERY_SCOPE_INVALID")
        scope = value["recovery_scope"]
        if (
            value["mode"] == "LOCAL_REPAIR"
            and scope not in {"NONE", "SERVING_ONLY", "COMMITTED_DAG_RECOVERY"}
        ) or (value["mode"] != "LOCAL_REPAIR" and scope != "NONE"):
            raise ContractError("AUTO_LAUNCH_RECOVERY_SCOPE_INVALID")
        validate_launch_inspection(
            {**{key: value[key] for key in FIELDS}, "schema_version": SCHEMA},
            request_ref=request_ref,
            release_install_ref=release_install_ref,
        )
        return value
    if type(value) is dict and value.get("schema_version") == BOOTSTRAP_SCHEMA:
        return _validate_bootstrap_inspection(
            value,
            request_ref=request_ref,
            release_install_ref=release_install_ref,
            target_trade_date=target_trade_date,
        )
    if (
        type(value) is not dict
        or set(value) != FIELDS
        or value["schema_version"] != SCHEMA
        or value["request_ref"] != validate_ref(request_ref)
        or value["release_install_ref"] != validate_ref(release_install_ref)
        or type(value["mode"]) is not str
        or value["mode"] not in MODES
        or not _false_authority(value["authority"])
    ):
        raise ContractError("AUTO_LAUNCH_INSPECTION_INVALID")
    result = value["result"]
    if value["mode"] == "COMPLETE_READ_ONLY":
        if (
            type(result) is not dict
            or result.get("schema_version") != RESULT_SCHEMA
            or result.get("action") != "CATCH_UP"
            or result.get("status") != "NO_ACTION"
            or result.get("auto_request_ref") != request_ref
            or type(result.get("result")) is not dict
            or result["result"].get("execution_state") != "NO_ACTION"
            or result["result"].get("business_state") != "COMPLETE"
        ):
            raise ContractError("AUTO_LAUNCH_COMPLETE_RESULT_INVALID")
    elif result is not None:
        raise ContractError("AUTO_LAUNCH_NONCOMPLETE_RESULT_FORBIDDEN")
    return value


def _validate_bootstrap_inspection(value, *, request_ref, release_install_ref, target_trade_date):
    if (
        set(value) != FIELDS | {"target_trade_date", "recovery_scope"}
        or value["request_ref"] != validate_ref(request_ref)
        or value["release_install_ref"] != validate_ref(release_install_ref)
        or value["target_trade_date"] != target_trade_date
        or not _false_authority(value["authority"])
        or type(value["mode"]) is not str
        or value["mode"] not in MODES
        or type(value["recovery_scope"]) is not str
    ):
        raise ContractError("BOOTSTRAP_LAUNCH_INSPECTION_INVALID")
    _validate_day(target_trade_date)
    mode, scope, result = value["mode"], value["recovery_scope"], value["result"]
    if mode == "LOCAL_REPAIR":
        if scope not in {"SERVING_ONLY", "COMMITTED_DAG_RECOVERY"} or result is not None:
            raise ContractError("BOOTSTRAP_LAUNCH_RECOVERY_SCOPE_INVALID")
    elif scope != "NONE":
        raise ContractError("BOOTSTRAP_LAUNCH_RECOVERY_SCOPE_INVALID")
    elif mode == "COMPLETE_READ_ONLY":
        if type(result) is not dict or result.get("execution_state") != "NO_ACTION":
            raise ContractError("BOOTSTRAP_LAUNCH_COMPLETE_RESULT_INVALID")
        business = result.get("business_state")
        if business not in {"COMPLETE", "NON_TRADING_DAY"}:
            raise ContractError("BOOTSTRAP_LAUNCH_COMPLETE_RESULT_INVALID")
        validate_production_result(
            result,
            action="EXECUTE",
            target_trade_date=target_trade_date,
            expected_dates=[] if business == "NON_TRADING_DAY" else [target_trade_date],
        )
    elif result is not None:
        raise ContractError("BOOTSTRAP_LAUNCH_NONCOMPLETE_RESULT_FORBIDDEN")
    return value

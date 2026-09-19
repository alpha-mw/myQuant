"""Exact configured-launcher source contracts; none grants execution authority."""

import hashlib

from quant_investor.contracts import canonical_json_bytes
from .daily_contract import ContractError, validate_ref
from .daily_journal import FALSE_AUTHORITY, _validate_day
from .daily_preparation_contract import preparation_root

PREFIX = "results/operations/daily_production/CN/source-configs"
LOCATOR_SCHEMA = "cn-daily-source-request.v1"
CALENDAR_SCHEMA = "cn-daily-source-calendar.v1"
PLAN_SCHEMA = "cn-daily-source-plan.v1"
PLAN_SCHEMA_V2 = "cn-daily-source-plan.v2"
RESULT_SCHEMA = "cn-daily-source-result.v1"
MODES = {
    "REQUEST_AVAILABLE": 0,
    "NON_TRADING_DAY": 0,
    "LOCAL_PREPARATION": 10,
    "ACQUISITION_REQUIRED": 11,
}
LOCATOR_FIELDS = {
    "schema_version",
    "state",
    "config_ref",
    "trade_date",
    "calendar_capture_ref",
    "calendar_ref",
    "raw_calendar_ref",
    "source_plan_ref",
    "preparation_commitment_ref",
    "request_ref",
    "previous_locator_sha256",
    "authority",
    "content_sha256",
}
PLAN_FIELDS = {
    "schema_version",
    "config_ref",
    "trade_date",
    "calendar_capture_ref",
    "store_pointer_ref",
    "event_pointer_ref",
    "benchmark_pointer_ref",
    "benchmark_start_date",
    "benchmark_end_date",
    "benchmark_required_dates",
    "benchmark_generation_id",
    "event_generation_id",
    "event_operation",
    "event_source_mode",
    "authority",
}
REGISTERED_PLAN_FIELDS = {
    "registered_event_declaration_ref",
    "registered_source_state",
    "registered_store_plan_ref",
    "decision_baseline_pointer_ref",
    "writer_pointer_ref",
}
RESULT_FIELDS = {
    "schema_version",
    "mode",
    "config_ref",
    "release_install_ref",
    "trade_date",
    "request_ref",
    "calendar_capture_ref",
    "preparation_commitment_ref",
    "authority",
}


class SourceSlotError(ContractError):
    def __init__(self, code, **fields):
        super().__init__(code)
        self.code, self.fields = code, fields


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def seal(value):
    body = {k: v for k, v in value.items() if k != "content_sha256"}
    return {**body, "content_sha256": digest(canonical_json_bytes(body))}


def reference(path, value):
    return {"path": path, "sha256": digest(canonical_json_bytes(value))}


def paths(config_ref, day, version=1):
    validate_ref(config_ref)
    _validate_day(day)
    if type(version) is not int or version not in (1, 2):
        raise SourceSlotError("SOURCE_VERSION_INVALID")
    root = preparation_root(config_ref, day)
    source = root + "/sources"
    return {
        "root": root,
        "capture": source + "/calendar-capture.v1.json",
        "calendar": source + "/calendar.json",
        "raw": source + "/calendar.raw.json",
        "plan": source + f"/plan.v{version}.json",
        "marker1": source + "/calendar-request-1.json",
        "marker2": source + "/calendar-request-2.json",
        "commitment": root + f"/commitment.v{version}.json",
        "request": root + "/request.json",
        "locator": f"{PREFIX}/{config_ref['sha256']}/request.v1.json",
    }


def validate_locator(value, *, config_ref):
    if (
        type(value) is not dict
        or set(value) != LOCATOR_FIELDS
        or value.get("schema_version") != LOCATOR_SCHEMA
        or seal(value) != value
        or value["config_ref"] != config_ref
        or value["authority"] != FALSE_AUTHORITY
    ):
        raise SourceSlotError("SOURCE_LOCATOR_INVALID")
    plan_path = validate_ref(value["source_plan_ref"])["path"]
    choices = {paths(config_ref, value["trade_date"], v)["plan"]: v for v in (1, 2)}
    if plan_path not in choices:
        raise SourceSlotError("SOURCE_LOCATOR_PATH_INVALID")
    selected = paths(config_ref, value["trade_date"], choices[plan_path])
    for name, key in (
        ("calendar_capture_ref", "capture"),
        ("calendar_ref", "calendar"),
        ("raw_calendar_ref", "raw"),
        ("source_plan_ref", "plan"),
    ):
        if validate_ref(value[name])["path"] != selected[key]:
            raise SourceSlotError("SOURCE_LOCATOR_PATH_INVALID")
    if value["state"] == "PREPARING_SOURCES":
        if value["request_ref"] is not None or value["preparation_commitment_ref"] is not None:
            raise SourceSlotError("SOURCE_LOCATOR_EARLY_REQUEST")
    elif value["state"] == "REQUEST_AVAILABLE":
        for name, key in (("request_ref", "request"), ("preparation_commitment_ref", "commitment")):
            if validate_ref(value[name])["path"] != selected[key]:
                raise SourceSlotError("SOURCE_LOCATOR_REQUEST_PATH_INVALID")
    else:
        raise SourceSlotError("SOURCE_LOCATOR_STATE_INVALID")
    if value["previous_locator_sha256"] is not None:
        validate_ref({"path": "history.json", "sha256": value["previous_locator_sha256"]})
    return value


def validate_transition(old, new):
    if old is None:
        if new["state"] != "PREPARING_SOURCES" or new["previous_locator_sha256"] is not None:
            raise SourceSlotError("SOURCE_LOCATOR_INITIAL_STATE_INVALID")
        return
    if old["config_ref"] != new["config_ref"]:
        raise SourceSlotError("SOURCE_LOCATOR_CONFIG_CHANGED")
    if old["trade_date"] == new["trade_date"]:
        stable = LOCATOR_FIELDS - {
            "state",
            "request_ref",
            "preparation_commitment_ref",
            "previous_locator_sha256",
            "content_sha256",
        }
        if (
            old["state"] != "PREPARING_SOURCES"
            or new["state"] != "REQUEST_AVAILABLE"
            or any(old[k] != new[k] for k in stable)
        ):
            raise SourceSlotError("SOURCE_LOCATOR_SAME_DAY_CHANGE")
    elif not (
        old["trade_date"] < new["trade_date"]
        and old["state"] == "REQUEST_AVAILABLE"
        and new["state"] == "PREPARING_SOURCES"
    ):
        raise SourceSlotError("SOURCE_LOCATOR_TRANSITION_INVALID")


def validate_result(value, *, config_ref, release_install_ref):
    if (
        type(value) is not dict
        or set(value) != RESULT_FIELDS
        or value["schema_version"] != RESULT_SCHEMA
        or value["config_ref"] != config_ref
        or value["release_install_ref"] != release_install_ref
        or value["authority"] != FALSE_AUTHORITY
        or value["mode"] not in MODES
    ):
        raise SourceSlotError("SOURCE_RESULT_INVALID")
    _validate_day(value["trade_date"])
    for key in ("request_ref", "calendar_capture_ref", "preparation_commitment_ref"):
        if value[key] is not None:
            validate_ref(value[key])
    if value["mode"] == "REQUEST_AVAILABLE" and value["request_ref"] is None:
        raise SourceSlotError("SOURCE_RESULT_REQUEST_REQUIRED")
    if (
        value["mode"] in {"NON_TRADING_DAY", "ACQUISITION_REQUIRED"}
        and value["request_ref"] is not None
    ):
        raise SourceSlotError("SOURCE_RESULT_REQUEST_FORBIDDEN")
    return value

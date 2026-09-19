"""Exact handoff versions and task-date interpretation; no IO or authority writes."""

from zoneinfo import ZoneInfo
from .daily_contract import ContractError, GRAPH_SHA256, utc_stamp, validate_ref
from .daily_journal import _false_authority, _validate_day

LEGACY_SCHEMA = "cn-daily-maintenance-handoff.v1"
SCHEMA = "cn-daily-maintenance-handoff.v2"
HISTORICAL_SCHEMA = "cn-daily-maintenance-handoff.v3"
AUTOMATIC_SCHEMA = "cn-daily-maintenance-handoff.v4"

HANDOFF_FIELDS = frozenset(
    {
        "schema_version",
        "trade_date",
        "graph_sha256",
        "request_ref",
        "recipe_ref",
        "release_ref",
        "release_install_ref",
        "logical_claim_ref",
        "maintenance_started_ref",
        "maintenance_core_ref",
        "calendar_ref",
        "raw_calendar_ref",
        "core_handoff_ref",
        "factor_pointer_ref",
        "market_pointer_ref",
        "market_snapshot_ref",
        "sealed_at",
        "authority",
    }
)
HANDOFF_V2_FIELDS = HANDOFF_FIELDS | {"prospective_policy_ref"}


HANDOFF_V3_FIELDS = HANDOFF_V2_FIELDS | {"historical_session_ref", "catchup_binding_ref"}
HANDOFF_V4_FIELDS = HANDOFF_V2_FIELDS | {"automatic_origin_ref"}


def validate_handoff_shape(value):
    versions = {
        LEGACY_SCHEMA: HANDOFF_FIELDS,
        SCHEMA: HANDOFF_V2_FIELDS,
        HISTORICAL_SCHEMA: HANDOFF_V3_FIELDS,
        AUTOMATIC_SCHEMA: HANDOFF_V4_FIELDS,
    }
    if (
        type(value) is not dict
        or type(value.get("schema_version")) is not str
        or value["schema_version"] not in versions
        or set(value) != versions[value["schema_version"]]
        or value["graph_sha256"] != GRAPH_SHA256
        or not _false_authority(value["authority"])
    ):
        raise ContractError("MAINTENANCE_HANDOFF_READBACK_SCHEMA_INVALID")
    _validate_day(value["trade_date"])
    utc_stamp(value["sealed_at"])
    if value["schema_version"] == AUTOMATIC_SCHEMA:
        validate_ref(value["automatic_origin_ref"])
    return value


def handoff_task_date(value, started):
    validate_handoff_shape(value)
    actual = (
        utc_stamp(started["started_at"]).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    )
    if value["schema_version"] == HISTORICAL_SCHEMA:
        if value["trade_date"] >= actual:
            raise ContractError("HISTORICAL_HANDOFF_TASK_DATE_INVALID")
        return value["trade_date"]
    return actual

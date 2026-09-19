"""V2 cutover is an immutable recommendation, never scheduler-state evidence."""

from datetime import datetime, timezone
from pathlib import PurePosixPath
from zoneinfo import ZoneInfo
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.errors import SystemSecurityError
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import _validate_day, _false_authority
from .journal_storage import JournalStorage

ROOT = PurePosixPath("results/operations/morning_strategy/CN")
REQUEST_SCHEMA = "morning-strategy-cutover-request.v2"
SCHEMA = "morning-strategy-cutover.v2"
REQUEST_FIELDS = {
    "schema_version",
    "target_date",
    "daily_completion_ref",
    "morning_receipts",
    "current_schedule_state",
}
STATES = {"EVENING_PRIMARY", "DUAL_RUN", "MORNING_PRIMARY"}
FIELDS = {
    "schema_version",
    "target_date",
    "request_ref",
    "daily_completion_ref",
    "morning_receipts",
    "release_install_ref",
    "current_schedule_state",
    "current_schedule_state_basis",
    "next_schedule_state",
    "schedule_action",
    "consecutive_morning_success_count",
    "validated_at",
    "status",
    "admission",
    "application_performed",
    "authority",
}


def cutover_path(day):
    _validate_day(day)
    return str(ROOT / day / "2020-cutover.v2.json")


def validate_cutover_request(value):
    if (
        type(value) is not dict
        or set(value) != REQUEST_FIELDS
        or value["schema_version"] != REQUEST_SCHEMA
    ):
        raise ContractError("MORNING_CUTOVER_REQUEST_INVALID")
    day = value["target_date"]
    _validate_day(day)
    if value["current_schedule_state"] not in STATES:
        raise ContractError("MORNING_CUTOVER_STATE_INVALID")
    ref = validate_ref(value["daily_completion_ref"])
    if ref["path"] != f"results/operations/daily_production/CN/{day}/completion.v1.json":
        raise ContractError("MORNING_CUTOVER_EOD_PATH_INVALID")
    refs = value["morning_receipts"]
    if type(refs) is not list or len(refs) > 2:
        raise ContractError("MORNING_CUTOVER_RECEIPTS_INVALID")
    days = []
    for ref in refs:
        checked = validate_ref(ref)
        date = PurePosixPath(checked["path"]).parent.name
        _validate_day(date)
        if (
            checked["path"]
            not in {
                str(ROOT / date / ("0945-run." + version + ".json")) for version in ("v2", "v3")
            }
            or date > day
        ):
            raise ContractError("MORNING_CUTOVER_RECEIPTS_INVALID")
        days.append(date)
    if days != sorted(set(days)):
        raise ContractError("MORNING_CUTOVER_RECEIPTS_INVALID")
    return value


def validate_cutover_receipt(value):
    if type(value) is not dict or set(value) != FIELDS or value["schema_version"] != SCHEMA:
        raise ContractError("MORNING_CUTOVER_RECEIPT_INVALID")
    validate_cutover_request(
        {
            **{k: value[k] for k in REQUEST_FIELDS if k != "schema_version"},
            "schema_version": REQUEST_SCHEMA,
        }
    )
    if (
        value["current_schedule_state_basis"] != "OWNER_DECLARATION"
        or value["status"] != "COMPLETE"
        or value["admission"] != "SCHEDULE_RECOMMENDATION_ONLY"
        or value["application_performed"] is not False
        or not _false_authority(value["authority"])
        or value["next_schedule_state"] not in STATES
        or type(value["consecutive_morning_success_count"]) is not int
        or value["consecutive_morning_success_count"] not in {0, 1, 2}
    ):
        raise ContractError("MORNING_CUTOVER_AUTHORITY_INVALID")
    from quant_investor.intelligence.morning import _schedule_transition

    refs = value["morning_receipts"]
    has_today = bool(refs and PurePosixPath(refs[-1]["path"]).parent.name == value["target_date"])
    count = value["consecutive_morning_success_count"]
    if count > len(refs) or (count > 0) != has_today:
        raise ContractError("MORNING_CUTOVER_COUNT_INVALID")
    expected = _schedule_transition(True, value["current_schedule_state"], count, has_today)
    if (value["next_schedule_state"], value["schedule_action"]) != expected:
        raise ContractError("MORNING_CUTOVER_TRANSITION_INVALID")
    for key in ("request_ref", "release_install_ref"):
        validate_ref(value[key])
    stamp = utc_stamp(value["validated_at"])
    if (
        stamp > datetime.now(timezone.utc)
        or stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") != value["target_date"]
    ):
        raise ContractError("MORNING_CUTOVER_TIME_INVALID")
    return value


class _CutoverIO(JournalStorage):
    @staticmethod
    def _path(value):
        validate_ref({"path": value, "sha256": "0" * 64})
        path = PurePosixPath(value)
        if (
            len(path.parts) != 6
            or path.parts[:4] != ROOT.parts
            or value != cutover_path(path.parts[4])
        ):
            raise SystemSecurityError("MORNING_CUTOVER_PATH_INVALID")
        return path

    @staticmethod
    def _governed_directory(path):
        return path == ROOT or ROOT in path.parents


class CutoverStorage:
    def __init__(self, workspace):
        self._io = _CutoverIO(workspace)

    def read(self, day):
        stored = self._io.read(cutover_path(day))
        if stored is not None:
            value = validate_cutover_receipt(parse_canonical_json_bytes(stored.data))
            if value["target_date"] != day:
                raise ContractError("MORNING_CUTOVER_DATE_INVALID")
        return stored

    def write(self, value):
        validate_cutover_receipt(value)
        return self._io.write(cutover_path(value["target_date"]), canonical_json_bytes(value))

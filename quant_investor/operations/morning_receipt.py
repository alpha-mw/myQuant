"""Exact Morning v2 receipt shape and fixed-path immutable storage.

Shape validation is not live admission. SEAL and cutover must also replay all
native consumer predicates from the receipt's original immutable input refs.
"""

from datetime import datetime, timezone
from pathlib import PurePosixPath
import re
from zoneinfo import ZoneInfo

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.intelligence.morning import classify_sina_quote_timing
from quant_investor.system.errors import SystemSecurityError
from .daily_contract import ContractError, utc_stamp, validate_ref
from .daily_journal import _validate_day, _false_authority
from .journal_storage import JournalStorage

ROOT = PurePosixPath("results/operations/morning_strategy/CN")
SCHEMA = "morning-strategy-run.v2"
SCHEMAS = {SCHEMA: "v2", "morning-strategy-run.v3": "v3"}
V3_FIELDS = {"threshold_policy_refs", "threshold_review_sha256", "review_summary_state"}


def receipt_version(value):
    version = SCHEMAS.get(value.get("schema_version")) if type(value) is dict else None
    if version is None:
        raise ContractError("MORNING_RECEIPT_SCHEMA_INVALID")
    return version


FIELDS = frozenset(
    {
        "schema_version",
        "run_date",
        "previous_trade_date",
        "request_ref",
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "owner_policy_ref",
        "output_ref",
        "validated_at",
        "status",
        "admission",
        "synthetic",
        "prospective_admission_state",
        "expected_symbols",
        "quote_timing",
        "authority",
    }
)


def receipt_path(day: str, version="v2") -> str:
    _validate_day(day)
    if version not in {"v2", "v3"}:
        raise ContractError("MORNING_RECEIPT_VERSION_INVALID")
    return str(ROOT / day / ("0945-run." + version + ".json"))


def validate_morning_receipt(value: dict) -> dict:
    version = receipt_version(value)
    if type(value) is not dict or set(value) != FIELDS | (V3_FIELDS if version == "v3" else set()):
        raise ContractError("MORNING_RECEIPT_FIELDS_INVALID")
    if (
        value["status"] != "COMPLETE"
        or value["admission"] != "LIVE_RESEARCH_CONSUMER"
        or value["synthetic"] is not False
        or value["prospective_admission_state"] != "NOT_CLAIMED"
        or not _false_authority(value["authority"])
    ):
        raise ContractError("MORNING_RECEIPT_ADMISSION_INVALID")
    day, previous = value["run_date"], value["previous_trade_date"]
    _validate_day(day)
    _validate_day(previous)
    if previous >= day:
        raise ContractError("MORNING_RECEIPT_DATE_INVALID")
    for key in (
        "request_ref",
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "owner_policy_ref",
        "output_ref",
    ):
        validate_ref(value[key])
    if value["previous_completion_ref"][
        "path"
    ] != f"results/operations/daily_production/CN/{previous}/completion.v1.json" or value[
        "output_ref"
    ][
        "path"
    ] != str(
        ROOT / day / ("0945-strategy." + version + ".md")
    ):
        raise ContractError("MORNING_RECEIPT_BINDING_PATH_INVALID")
    if version == "v3":
        refs = value["threshold_policy_refs"]
        if type(refs) is not dict or set(refs) != {"trailing", "initial_stop"}:
            raise ContractError("MORNING_RECEIPT_THRESHOLD_REFS_INVALID")
        for ref in refs.values():
            validate_ref(ref)
        validate_ref({"path": "review.json", "sha256": value["threshold_review_sha256"]})
        if value["review_summary_state"] not in {
            "COMPLETE_RESEARCH_REVIEW",
            "PARTIAL_RESEARCH_REVIEW",
        }:
            raise ContractError("MORNING_RECEIPT_REVIEW_STATE_INVALID")
    symbols = value["expected_symbols"]
    if (
        type(symbols) is not list
        or not symbols
        or any(type(s) is not str or not re.fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", s) for s in symbols)
        or symbols != sorted(set(symbols))
    ):
        raise ContractError("MORNING_RECEIPT_SYMBOLS_INVALID")
    timing = value["quote_timing"]
    if type(timing) is not dict or "actual_capture_time" not in timing:
        raise ContractError("MORNING_RECEIPT_TIMING_INVALID")
    try:
        captured = datetime.fromisoformat(timing["actual_capture_time"])
        if captured.utcoffset() is None:
            raise ValueError("timezone required")
    except (TypeError, ValueError) as exc:
        raise ContractError("MORNING_RECEIPT_TIMING_INVALID") from exc
    request_time = captured.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if timing != classify_sina_quote_timing(request_time, run_date=day):
        raise ContractError("MORNING_RECEIPT_TIMING_INVALID")
    stamp = utc_stamp(value["validated_at"])
    if (
        stamp > datetime.now(timezone.utc)
        or stamp < captured
        or stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d") != day
    ):
        raise ContractError("MORNING_RECEIPT_TIME_INVALID")
    return value


class _MorningIO(JournalStorage):
    @staticmethod
    def _path(value: str) -> PurePosixPath:
        validate_ref({"path": value, "sha256": "0" * 64})
        path = PurePosixPath(value)
        if len(path.parts) != 6 or path.parts[:4] != ROOT.parts:
            raise SystemSecurityError("MORNING_RECEIPT_PATH_INVALID")
        if value not in {receipt_path(path.parts[4], version) for version in ("v2", "v3")}:
            raise SystemSecurityError("MORNING_RECEIPT_PATH_INVALID")
        return path

    @staticmethod
    def _governed_directory(path: PurePosixPath) -> bool:
        return path == ROOT or ROOT in path.parents


class MorningReceiptStorage:
    """No report writer, lock file, projection replacement or arbitrary destination."""

    def __init__(self, workspace: str):
        self._storage = _MorningIO(workspace)

    def read(self, day: str, version="v2"):
        stored = self._storage.read(receipt_path(day, version))
        if stored is not None:
            value = validate_morning_receipt(parse_canonical_json_bytes(stored.data))
            if value["run_date"] != day or receipt_version(value) != version:
                raise ContractError("MORNING_RECEIPT_DATE_INVALID")
        return stored

    def write(self, value: dict):
        validated = validate_morning_receipt(value)
        return self._storage.write(
            receipt_path(validated["run_date"], receipt_version(validated)),
            canonical_json_bytes(validated),
        )

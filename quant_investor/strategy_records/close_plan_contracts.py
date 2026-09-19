"""Versioned native close plan identities; no source selection or write authority."""

import hashlib
import re
from pathlib import PurePosixPath
from datetime import date, datetime
from zoneinfo import ZoneInfo

from quant_investor.operations.daily_contract import validate_ref
from .store import StrategyRecordStoreError, canonical_json_bytes, content_sha256

PLAN_V1 = "myquant.cn_official_close_batch_plan.v1"
PLAN_V2 = "myquant.cn_official_close_batch_plan.v2"
RECEIPT_V2 = "myquant.strategy_daily_close_receipt.v2"
COMPLETION_V2 = "myquant.cn_official_close_batch_completion.v2"
IMPLEMENTATION_V2 = "registered-source.1"
PROFILE = "OWNER_DECLARED_BUYS_V1"
REGISTERED_RECEIPT_FIELDS = {
    "schema_id",
    "receipt_id",
    "transaction_id",
    "input_fingerprint",
    "trade_date",
    "record_id",
    "status",
    "effective_at",
    "payload_copied",
    "actual_holdings_mutation_authority",
    "cash_mutation_authority",
    "broker_order_trade_authority",
    "registered_event_declaration_ref",
    "source_profile",
    "writer_active_checkpoint_digest",
    "decision_baseline_pointer_ref",
    "content_sha256",
}
PROFILE_FIELDS = {
    "source_profile",
    "registered_event_declaration_ref",
    "decision_baseline_pointer_ref",
    "decision_baseline_catalog_ref",
    "decision_baseline_record_id",
    "source_valuation_date",
}
PLAN_FIELDS = {
    "schema_id",
    "batch_implementation_version",
    "transaction_id",
    "input_fingerprint",
    "transaction_planned_at",
    "effective_at",
    "source_active_record_id",
    "last_official_date",
    "requested_target",
    "missing_dates",
    "record_ids",
    "catalog_generation_id",
    "performance_generation_id",
    "event_generation_id",
    "benchmark_generation_id",
    "preimages",
    "publication_class",
    "all_or_nothing",
    "broker_order_trade_authority",
    "content_sha256",
} | PROFILE_FIELDS
PREIMAGE_FIELDS = {
    "store_pointer_sha256",
    "store_catalog_sha256",
    "performance_manifest_sha256",
    "market_pointer_sha256",
    "benchmark_pointer_sha256",
    "event_pointer_sha256",
    "calendar_receipt_sha256",
    "policy_sha256",
    "retrospective_sha256",
    "evidence_sha256",
}


def version(value):
    if value == PLAN_V1:
        return 1
    if value == PLAN_V2:
        return 2
    raise StrategyRecordStoreError("CLOSE_PLAN_VERSION_UNSUPPORTED")


def version_number(value):
    if type(value) is not int or value not in (1, 2):
        raise StrategyRecordStoreError("CLOSE_PLAN_VERSION_UNSUPPORTED")
    return value


def fingerprint_fields(value):
    fields = {
        k: value[k]
        for k in ("batch_implementation_version", "requested_target", "missing_dates", "preimages")
    }
    if value.get("source_profile") is not None:
        fields.update({k: value[k] for k in PROFILE_FIELDS})
    return fields


def validate_plan(value, *, path=None):
    if type(value) is not dict:
        raise StrategyRecordStoreError("CLOSE_PLAN_INVALID")
    selected = version(value.get("schema_id"))
    if path is not None and PurePosixPath(str(path)).name != f"plan.v{selected}.json":
        raise StrategyRecordStoreError("CLOSE_PLAN_PATH_VERSION_MISMATCH")
    if selected == 1:
        if PROFILE_FIELDS & set(value):
            raise StrategyRecordStoreError("CLOSE_PLAN_V1_REGISTERED_FIELDS_FORBIDDEN")
        return selected
    if set(value) != PLAN_FIELDS or value["content_sha256"] != content_sha256(value):
        raise StrategyRecordStoreError("CLOSE_PLAN_V2_FIELDS_INVALID")
    for key in (
        "source_profile",
        "batch_implementation_version",
        "source_active_record_id",
        "decision_baseline_record_id",
        "last_official_date",
        "requested_target",
        "source_valuation_date",
        "transaction_planned_at",
        "effective_at",
    ):
        if type(value[key]) is not str:
            raise StrategyRecordStoreError("CLOSE_PLAN_V2_FIELDS_INVALID")
    if (
        value["source_profile"] != PROFILE
        or value["batch_implementation_version"] != IMPLEMENTATION_V2
        or value["all_or_nothing"] is not True
        or value["broker_order_trade_authority"] is not False
        or value["publication_class"] != "BATCH_CATCH_UP_OFFICIAL_VALUATION"
        or value["missing_dates"] != [value["requested_target"]]
        or value["source_valuation_date"] != value["requested_target"]
        or type(value["record_ids"]) is not list
        or len(value["record_ids"]) != 1
        or value["last_official_date"] >= value["requested_target"]
        or value["source_active_record_id"] == value["decision_baseline_record_id"]
        or type(value["preimages"]) is not dict
        or set(value["preimages"]) != PREIMAGE_FIELDS
    ):
        raise StrategyRecordStoreError("CLOSE_PLAN_V2_PROFILE_INVALID")
    for key in (
        "registered_event_declaration_ref",
        "decision_baseline_pointer_ref",
        "decision_baseline_catalog_ref",
    ):
        validate_ref(value[key])
    preimages = value["preimages"]
    try:
        day = date.fromisoformat(value["requested_target"])
        previous = date.fromisoformat(value["last_official_date"])
        planned = datetime.strptime(value["effective_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=ZoneInfo("UTC")
        )
        prefix = planned.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        if (
            day.isoformat() != value["requested_target"]
            or previous >= day
            or value["transaction_planned_at"] != value["effective_at"]
            or planned.astimezone(ZoneInfo("Asia/Shanghai")).date() < day
            or type(value["record_ids"][0]) is not str
            or re.fullmatch(re.escape(prefix) + r"-b[0-9]{2}", value["record_ids"][0]) is None
        ):
            raise ValueError("invalid date or record")
    except (ValueError, TypeError) as exc:
        raise StrategyRecordStoreError("CLOSE_PLAN_V2_CLOCK_INVALID") from exc
    if type(preimages["evidence_sha256"]) is not dict:
        raise StrategyRecordStoreError("CLOSE_PLAN_V2_PREIMAGES_INVALID")
    if preimages["retrospective_sha256"] is not None or set(preimages["evidence_sha256"]) != {
        value["requested_target"]
    }:
        raise StrategyRecordStoreError("CLOSE_PLAN_V2_PREIMAGES_INVALID")
    for key in PREIMAGE_FIELDS - {"retrospective_sha256", "evidence_sha256"}:
        validate_ref({"path": "source", "sha256": preimages[key]})
    validate_ref(
        {"path": "evidence", "sha256": preimages["evidence_sha256"][value["requested_target"]]}
    )
    digest = hashlib.sha256(canonical_json_bytes(fingerprint_fields(value))).hexdigest()
    transaction = f"daily-close-{value['requested_target'].replace('-', '')}-{digest[:16]}"
    if value["input_fingerprint"] != digest or value["transaction_id"] != transaction:
        raise StrategyRecordStoreError("CLOSE_PLAN_V2_IDENTITY_INVALID")
    if re.fullmatch(r"daily-close-[0-9]{8}-[0-9a-f]{16}", transaction) is None:
        raise StrategyRecordStoreError("CLOSE_PLAN_V2_DATE_INVALID")
    return selected


def portfolio_identity(plan):
    selected = validate_plan(plan)
    if selected == 1:
        return (
            plan["preimages"]["store_pointer_sha256"],
            plan["preimages"]["store_catalog_sha256"],
            plan["source_active_record_id"],
        )
    return (
        plan["decision_baseline_pointer_ref"]["sha256"],
        plan["decision_baseline_catalog_ref"]["sha256"],
        plan["decision_baseline_record_id"],
    )


def validate_registered_receipt(value):
    """Classify exact receipt-v2 grammar; financial admission still needs native replay."""
    if (
        type(value) is not dict
        or set(value) != REGISTERED_RECEIPT_FIELDS
        or value.get("schema_id") != RECEIPT_V2
    ):
        raise StrategyRecordStoreError("REGISTERED_CLOSE_RECEIPT_FIELDS_INVALID")
    if (
        value["content_sha256"] != content_sha256(value)
        or value["source_profile"] != PROFILE
        or value["status"] != "OFFICIAL_CLOSE_PREPARED"
        or any(
            value[k] is not False
            for k in (
                "payload_copied",
                "actual_holdings_mutation_authority",
                "cash_mutation_authority",
                "broker_order_trade_authority",
            )
        )
    ):
        raise StrategyRecordStoreError("REGISTERED_CLOSE_RECEIPT_SCOPE_INVALID")
    for key in ("registered_event_declaration_ref", "decision_baseline_pointer_ref"):
        validate_ref(value[key])
    for key in ("input_fingerprint", "writer_active_checkpoint_digest"):
        validate_ref({"path": "receipt", "sha256": value[key]})
    try:
        day = date.fromisoformat(value["trade_date"])
        stamp = datetime.strptime(value["effective_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=ZoneInfo("UTC")
        )
        prefix = stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        if (
            day.isoformat() != value["trade_date"]
            or stamp.astimezone(ZoneInfo("Asia/Shanghai")).date() < day
            or re.fullmatch(re.escape(prefix) + r"-b[0-9]{2}", value["record_id"]) is None
        ):
            raise ValueError("receipt clock")
    except (ValueError, TypeError) as exc:
        raise StrategyRecordStoreError("REGISTERED_CLOSE_RECEIPT_CLOCK_INVALID") from exc
    declaration_sha = value["registered_event_declaration_ref"]["sha256"]
    if (
        value["transaction_id"]
        != f"daily-close-{day.strftime('%Y%m%d')}-{value['input_fingerprint'][:16]}"
        or value["receipt_id"] != f"daily-close/{day.isoformat()}/{declaration_sha[:16]}"
    ):
        raise StrategyRecordStoreError("REGISTERED_CLOSE_RECEIPT_IDENTITY_INVALID")
    return value

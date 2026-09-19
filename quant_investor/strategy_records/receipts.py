"""Native no-action receipt validation, independent of current-pointer selection."""

from collections.abc import Mapping
from zoneinfo import ZoneInfo

from .store import StrategyRecordStoreError, content_sha256, _publication_timestamp

NO_ACTION_SCHEMA = "myquant.strategy_record_no_action_receipt.v1"
NO_ACTION_FIELDS = frozenset(
    {
        "schema_id",
        "receipt_id",
        "created_at",
        "status",
        "reason",
        "active_record_id",
        "active_checkpoint",
        "payload_copied",
        "v17_mainline_authority",
        "broker_order_trade_authority",
        "content_sha256",
    }
)


def validate_no_action_receipt(
    receipt,
    *,
    receipt_id,
    expected_sha,
    record_id,
    checkpoint,
    trade_date=None,
):
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != NO_ACTION_FIELDS
        or receipt["schema_id"] != NO_ACTION_SCHEMA
    ):
        raise StrategyRecordStoreError("continuity receipt fields/schema mismatch")
    if content_sha256(receipt) != receipt["content_sha256"]:
        raise StrategyRecordStoreError("continuity receipt content hash mismatch")
    if receipt["receipt_id"] != receipt_id or receipt["content_sha256"] != expected_sha:
        raise StrategyRecordStoreError("continuity receipt SHA-256 mismatch")
    if (
        receipt["status"] != "NO_ACTION"
        or receipt["payload_copied"] is not False
        or receipt["v17_mainline_authority"] is not False
        or receipt["broker_order_trade_authority"] is not False
        or type(receipt["reason"]) is not str
        or not receipt["reason"].strip()
    ):
        raise StrategyRecordStoreError("continuity receipt authority/status is invalid")
    if (
        type(record_id) is not str
        or not isinstance(checkpoint, dict)
        or (receipt["active_record_id"] != record_id or receipt["active_checkpoint"] != checkpoint)
    ):
        raise StrategyRecordStoreError("continuity receipt active checkpoint mismatch")
    instant = _publication_timestamp(receipt["created_at"], label="continuity receipt created_at")
    if (
        trade_date is not None
        and instant.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat() != trade_date
    ):
        raise StrategyRecordStoreError("continuity receipt date mismatch")
    return dict(receipt)

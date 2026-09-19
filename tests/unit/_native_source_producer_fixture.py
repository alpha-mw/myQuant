"""Synthetic native source producer inputs; never real provider/account evidence."""

from argparse import Namespace
from datetime import datetime, timezone
import json

from _daily_preparation_fixture import put
from test_daily_evidence_requested_session import capture
from quant_investor.market.tushare_transport import replay_tushare_response_bytes

NOW = datetime(2026, 8, 25, 12, 20, tzinfo=timezone.utc)


def event_arguments(root, *, now=NOW, holidays=()):
    from _native_daily_store_fixture import NativeStoreFixture

    book = NativeStoreFixture(root)
    book.advance("2026-08-24")
    policy = json.loads((root / book.policy_path).read_bytes())
    policy.update(
        effective_from="2026-08-01T00:00:00Z",
        event_inbox={
            "pointer_path": str(book.root.relative_to(root)) + "/_event_store/current.v1.json",
            "owner_append_cutoff_local": "15:30:00",
            "timezone": "Asia/Shanghai",
            "sealed_empty_inventory_is_owner_authorized_closure": True,
            "late_event_behavior": "OFFICIAL_CLOSE_RESTATEMENT_REQUIRED",
        },
    )
    policy_ref = put(root, "fixtures/current-standing-policy.json", policy)
    authority = capture(now.isoformat(), holidays=holidays)
    raw_ref = put(root, "fixtures/current-calendar.raw.json", authority.raw_response_bytes)
    authority.receipt["raw_response_path"] = str(root / raw_ref["path"])
    cal_ref = put(root, "fixtures/current-calendar.json", authority.receipt)
    return book, Namespace(
        project_root=str(root),
        record_root=str(book.root),
        trade_date="2026-08-25",
        policy_path=policy_ref["path"],
        policy_sha256=policy_ref["sha256"],
        maintenance_receipt=None,
        maintenance_receipt_sha256=None,
        calendar_receipt=str(root / cal_ref["path"]),
        calendar_receipt_sha256=cal_ref["sha256"],
        raw_calendar=str(root / raw_ref["path"]),
        raw_calendar_sha256=raw_ref["sha256"],
        expected_event_pointer_sha256=book.event_pointer,
        generation_id="synthetic-current-day-20260825",
    )


class BenchmarkClient:
    """Actual transport envelope decoder, with only HTTPS responses replaced."""

    def __init__(self, days=("20260824", "20260825"), fault=None):
        self.days = days
        self.fault = fault
        self.calls = []

    def request(self, *, api_name, params, expected_fields):
        self.calls.append((api_name, dict(params), expected_fields))
        items = [
            [params["ts_code"], day, 1234.56 + index]
            for index, day in enumerate(self.days)
            if params["start_date"] <= day <= params["end_date"]
        ]
        if self.fault is not None:
            items = self.fault(items, params)
        wire = json.dumps(
            {
                "code": 0,
                "msg": "",
                "detail": "",
                "request_id": "synthetic-index",
                "data": {
                    "fields": list(expected_fields),
                    "items": items,
                    "has_more": False,
                    "count": 0,
                },
            }
        ).encode()
        return replay_tushare_response_bytes(
            wire, api_name=api_name, expected_fields=expected_fields, strict_decimal_decode=True
        )

from __future__ import annotations

from decimal import Decimal
import json
from typing import Any

import pytest

from quant_investor.intelligence_v2._core import canonical_bytes, content_ref
from quant_investor.intelligence_v2.sources.tushare import (
    TushareContractError,
    build_endpoint_execution_plan,
    build_tushare_capability_receipt,
    build_tushare_endpoint_policy,
    build_tushare_request_receipt,
    probe_tushare_capabilities,
    validate_tushare_capability_receipt,
    validate_tushare_execution_receipt,
    validate_tushare_request_receipt,
)
from quant_investor.v17_v4_runtime.tushare_https import (
    TushareHttpsError,
    TushareResponse,
)

NOW = "2026-08-09T08:00:00Z"


def plan(
    api_name: str = "daily_basic",
    *,
    permission_class: str = "POINTS",
    row_limit: int = 6000,
) -> dict[str, Any]:
    separate = permission_class == "SEPARATE"
    return build_endpoint_execution_plan(
        api_name=api_name,
        lane="FUNDAMENTAL" if not separate else "DIAGNOSTIC",
        permission_class=permission_class,
        official_document_url="https://tushare.pro/document/2?doc_id=32",
        official_document_id=f"tushare.{api_name}",
        document_observed_at=NOW,
        documented_min_points=2000,
        strict_decimal_decode=True,
        expected_fields=["ts_code", "trade_date", "close"],
        fixed_params={} if separate else {"trade_date": "20260807"},
        partition_dimensions=[] if separate else ["trade_date"],
        ordered_expected_partition_keyset=[] if separate else ["trade_date=20260807"],
        documented_row_limit=row_limit,
        max_attempts=1,
        retry_schedule=[0],
        empty_partition_rule="BASELINE_IDENTITY_EMPTY",
        completeness_proof="EXACT_PARTITION_AND_COUNT",
        limit_hit_action="BLOCK",
        planned_terminal_request_count=0 if separate else 1,
        planned_max_network_attempts=0 if separate else 1,
        created_at=NOW,
    )


class FakeClient:
    def __init__(self, response: TushareResponse | BaseException) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> TushareResponse:
        self.calls.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def response(
    rows: tuple[tuple[Any, ...], ...],
    *,
    reported_count: int | None = None,
    has_more: bool = False,
) -> TushareResponse:
    return TushareResponse(
        api_name="daily_basic",
        request_id="request-1",
        reported_count=len(rows) if reported_count is None else reported_count,
        has_more=has_more,
        fields=("ts_code", "trade_date", "close"),
        rows=rows,
    )


def test_available_probe_builds_replayable_receipt_chain() -> None:
    endpoint = plan()
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[endpoint])
    client = FakeClient(response((("000001.SZ", "20260807", Decimal("10.500000000001")),)))

    result = probe_tushare_capabilities(
        policy=policy,
        probed_at=NOW,
        client=client,
    )

    assert result["network_attempts"] == 1
    assert len(client.calls) == 1
    request_receipt = result["request_receipts"][0]
    capability = result["capability_receipts"][0]
    execution = result["execution_receipts"][0]
    assert capability["status"] == "AVAILABLE"
    assert capability["reported_count"] == capability["accepted_count"] == 1
    assert capability["transport_calls"] == 1
    assert (
        validate_tushare_request_receipt(
            request_receipt,
            plan=endpoint,
            sanitized_params={"trade_date": "20260807"},
        )
        == request_receipt
    )
    assert (
        validate_tushare_capability_receipt(
            capability,
            plan=endpoint,
        )
        == capability
    )
    assert (
        validate_tushare_execution_receipt(
            execution,
            policy=policy,
            plan=endpoint,
            capability_receipt=capability,
        )
        == execution
    )


@pytest.mark.parametrize(
    ("provider_response", "expected_status", "expected_blocker"),
    [
        (response(tuple()), "EMPTY", None),
        (
            response((("000001.SZ", "20260807", Decimal("1")),), has_more=True),
            "INCOMPLETE",
            "HAS_MORE",
        ),
        (
            response(
                (
                    ("000001.SZ", "20260807", Decimal("1")),
                    ("000001.SZ", "20260807", Decimal("1")),
                )
            ),
            "INCOMPLETE",
            "DUPLICATE_ROWS",
        ),
        (
            response((("000001.SZ", "20260807", Decimal("1")),), reported_count=2),
            "INCOMPLETE",
            "COUNT_MISMATCH",
        ),
        (
            TushareHttpsError("TUSHARE_API_ERROR"),
            "PROVIDER_ERROR",
            None,
        ),
        (
            TushareHttpsError("TUSHARE_RESPONSE_INVALID"),
            "SCHEMA_MISMATCH",
            None,
        ),
        (
            TushareHttpsError("TUSHARE_TRANSPORT_ERROR"),
            "TRANSPORT_ERROR",
            None,
        ),
        (object(), "SCHEMA_MISMATCH", None),
    ],
)
def test_capability_probe_exposes_all_terminal_failure_classes(
    provider_response: TushareResponse | BaseException,
    expected_status: str,
    expected_blocker: str | None,
) -> None:
    endpoint = plan()
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[endpoint])
    result = probe_tushare_capabilities(
        policy=policy,
        probed_at=NOW,
        client=FakeClient(provider_response),
    )
    receipt = result["capability_receipts"][0]
    assert receipt["status"] == expected_status
    if expected_blocker is not None:
        assert expected_blocker in receipt["blocker_codes"]


def test_row_limit_hit_is_incomplete() -> None:
    endpoint = plan(row_limit=2)
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[endpoint])
    result = probe_tushare_capabilities(
        policy=policy,
        probed_at=NOW,
        client=FakeClient(
            response(
                (
                    ("000001.SZ", "20260807", Decimal("1")),
                    ("000002.SZ", "20260807", Decimal("2")),
                )
            )
        ),
    )
    receipt = result["capability_receipts"][0]
    assert receipt["status"] == "INCOMPLETE"
    assert "ROW_LIMIT_HIT" in receipt["blocker_codes"]


def test_separate_endpoint_performs_zero_transport_calls() -> None:
    endpoint = plan("stk_mins", permission_class="SEPARATE")
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[endpoint])
    client = FakeClient(AssertionError("transport must not be called"))
    result = probe_tushare_capabilities(
        policy=policy,
        probed_at=NOW,
        client=client,
    )
    capability = result["capability_receipts"][0]
    assert capability["status"] == "NOT_PROBED"
    assert capability["transport_calls"] == 0
    assert result["network_attempts"] == 0
    assert client.calls == []


def test_probe_rejects_retry_authority_before_transport() -> None:
    endpoint = plan()
    retry_plan = build_endpoint_execution_plan(
        api_name=endpoint["api_name"],
        lane=endpoint["lane"],
        permission_class=endpoint["permission_class"],
        official_document_url=endpoint["official_document_url"],
        official_document_id=endpoint["official_document_id"],
        document_observed_at=endpoint["document_observed_at"],
        documented_min_points=endpoint["documented_min_points"],
        strict_decimal_decode=True,
        expected_fields=endpoint["expected_fields"],
        fixed_params=endpoint["fixed_params"],
        partition_dimensions=endpoint["partition_dimensions"],
        ordered_expected_partition_keyset=endpoint["ordered_expected_partition_keyset"],
        documented_row_limit=endpoint["documented_row_limit"],
        max_attempts=2,
        retry_schedule=[0, 1],
        empty_partition_rule=endpoint["empty_partition_rule"],
        completeness_proof=endpoint["completeness_proof"],
        limit_hit_action=endpoint["limit_hit_action"],
        planned_terminal_request_count=1,
        planned_max_network_attempts=2,
        created_at=endpoint["created_at"],
    )
    retry_policy = build_tushare_endpoint_policy(
        created_at=NOW,
        endpoint_plans=[retry_plan],
    )
    client = FakeClient(AssertionError("transport must not be called"))

    with pytest.raises(TushareContractError):
        probe_tushare_capabilities(
            policy=retry_policy,
            probed_at=NOW,
            client=client,
        )
    assert client.calls == []


def test_request_receipt_contains_no_params_or_token_and_binds_strict_mode() -> None:
    endpoint = plan()
    receipt = build_tushare_request_receipt(
        plan=endpoint,
        partition_key="trade_date=20260807",
        partition_ordinal=0,
        sanitized_params={"trade_date": "20260807"},
        requested_at=NOW,
    )
    rendered = canonical_bytes(receipt)
    assert b"token" not in rendered.lower()
    assert b'"fixed_params"' not in rendered
    assert b'"sanitized_params"' not in rendered
    assert receipt["partition_key"] == "trade_date=20260807"
    assert receipt["strict_decimal_decode"] is True
    assert len(receipt["transport_implementation_sha256"]) == 64
    assert receipt["plan_ref"] == content_ref(endpoint, identity_field="plan_id")


def test_capability_resealed_forgery_is_rejected() -> None:
    endpoint = plan()
    request = build_tushare_request_receipt(
        plan=endpoint,
        partition_key="trade_date=20260807",
        partition_ordinal=0,
        sanitized_params={"trade_date": "20260807"},
        requested_at=NOW,
    )
    capability = build_tushare_capability_receipt(
        plan=endpoint,
        status="AVAILABLE",
        transport_calls=1,
        reported_count=1,
        accepted_count=1,
        blocker_codes=[],
        request_ref=content_ref(request, identity_field="request_receipt_id"),
        response_projection_sha256="a" * 64,
        probed_at=NOW,
    )
    forged = json.loads(json.dumps(capability))
    forged["accepted_count"] = 2
    with pytest.raises(TushareContractError):
        validate_tushare_capability_receipt(forged, plan=endpoint)

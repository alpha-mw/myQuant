from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from quant_investor.market.tushare import (
    TushareContractError,
    build_endpoint_execution_plan,
    build_tushare_endpoint_policy,
    probe_tushare_capabilities,
    validate_tushare_capability_receipt,
    validate_tushare_execution_receipt,
    validate_tushare_request_receipt,
)
from quant_investor.market.tushare_transport import TushareHttpsError, TushareResponse

NOW = "2026-08-14T08:00:00Z"


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
        fixed_params={} if separate else {"trade_date": "20260813"},
        partition_dimensions=[] if separate else ["trade_date"],
        ordered_expected_partition_keyset=[] if separate else ["trade_date=20260813"],
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
    def __init__(self, response: TushareResponse | BaseException | object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> TushareResponse:
        self.calls.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response  # type: ignore[return-value]


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


def test_available_probe_builds_offline_replayable_receipt_chain() -> None:
    endpoint = plan()
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[endpoint])
    client = FakeClient(
        response((("000001.SZ", "20260813", Decimal("10.500000000001")),))
    )
    result = probe_tushare_capabilities(policy=policy, probed_at=NOW, client=client)

    request = result["request_receipts"][0]
    capability = result["capability_receipts"][0]
    execution = result["execution_receipts"][0]
    assert result["network_attempts"] == 1
    assert len(client.calls) == 1
    assert capability["status"] == "AVAILABLE"
    assert validate_tushare_request_receipt(
        request,
        plan=endpoint,
        sanitized_params={"trade_date": "20260813"},
    ) == request
    assert validate_tushare_capability_receipt(capability, plan=endpoint) == capability
    assert validate_tushare_execution_receipt(
        execution,
        policy=policy,
        plan=endpoint,
        capability_receipt=capability,
    ) == execution


@pytest.mark.parametrize(
    ("provider_response", "expected_status", "expected_blocker"),
    [
        (response(tuple()), "EMPTY", None),
        (response((("000001.SZ", "20260813", Decimal("1")),), has_more=True), "INCOMPLETE", "HAS_MORE"),
        (response((("000001.SZ", "20260813", Decimal("1")),), reported_count=2), "INCOMPLETE", "COUNT_MISMATCH"),
        (TushareHttpsError("TUSHARE_API_ERROR"), "PROVIDER_ERROR", None),
        (TushareHttpsError("TUSHARE_RESPONSE_INVALID"), "SCHEMA_MISMATCH", None),
        (TushareHttpsError("TUSHARE_TRANSPORT_ERROR"), "TRANSPORT_ERROR", None),
        (object(), "SCHEMA_MISMATCH", None),
    ],
)
def test_probe_terminal_classes_are_fail_closed(
    provider_response: TushareResponse | BaseException | object,
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


def test_separate_endpoint_never_calls_injected_transport() -> None:
    endpoint = plan("stk_mins", permission_class="SEPARATE")
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[endpoint])
    client = FakeClient(AssertionError("transport must not be called"))
    result = probe_tushare_capabilities(policy=policy, probed_at=NOW, client=client)
    assert result["network_attempts"] == 0
    assert result["capability_receipts"][0]["status"] == "NOT_PROBED"
    assert client.calls == []


def test_probe_rejects_retry_authority_before_transport() -> None:
    endpoint = plan()
    retry = build_endpoint_execution_plan(
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
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[retry])
    client = FakeClient(AssertionError("transport must not be called"))
    with pytest.raises(TushareContractError):
        probe_tushare_capabilities(policy=policy, probed_at=NOW, client=client)
    assert client.calls == []

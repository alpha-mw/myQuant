"""Bounded capability probing through the stable Tushare transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quant_investor.market.tushare_transport import (
    OfficialTushareHttpsClient,
    TushareHttpsError,
)

from ._core import content_ref, timestamp
from .contracts import (
    build_tushare_capability_receipt,
    build_tushare_execution_receipt,
    build_tushare_request_receipt,
    response_projection_sha256,
    validate_endpoint_execution_plan,
    validate_tushare_endpoint_policy,
)
from .models import TushareContractError, TushareRequestClient


def _provider_failure_status(error: TushareHttpsError) -> str:
    if error.code == "TUSHARE_API_ERROR":
        return "PROVIDER_ERROR"
    if error.code == "TUSHARE_RESPONSE_INVALID":
        return "SCHEMA_MISMATCH"
    return "TRANSPORT_ERROR"


def _classify_response(
    *,
    plan: Mapping[str, Any],
    response: Any,
) -> tuple[str, list[str], int, int, str | None]:
    if response.api_name != plan["api_name"] or tuple(response.fields) != tuple(
        plan["expected_fields"]
    ):
        return "SCHEMA_MISMATCH", [], 0, 0, None
    rows = tuple(response.rows)
    projection_sha = response_projection_sha256(rows)
    reported = response.reported_count
    accepted = len(rows)
    blockers: list[str] = []
    if reported != accepted:
        blockers.append("COUNT_MISMATCH")
    if len(rows) != len(set(rows)):
        blockers.append("DUPLICATE_ROWS")
    if response.has_more:
        blockers.append("HAS_MORE")
    if accepted >= plan["documented_row_limit"]:
        blockers.append("ROW_LIMIT_HIT")
    if blockers:
        return "INCOMPLETE", blockers, reported, accepted, projection_sha
    return (
        "EMPTY" if accepted == 0 else "AVAILABLE",
        [],
        reported,
        accepted,
        projection_sha,
    )


def probe_tushare_capabilities(
    *,
    policy: Mapping[str, Any],
    probed_at: str,
    client: TushareRequestClient | None = None,
) -> dict[str, Any]:
    """Probe each POINTS endpoint once and every SEPARATE endpoint zero times."""

    validated_policy = validate_tushare_endpoint_policy(policy)
    observed_at = timestamp(probed_at, label="probed_at")
    transport: TushareRequestClient = (
        OfficialTushareHttpsClient(strict_decimal_decode=True) if client is None else client
    )
    request_receipts: list[dict[str, Any]] = []
    capabilities: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    total_attempts = 0

    for raw_plan in validated_policy["endpoint_plans"]:
        plan = validate_endpoint_execution_plan(raw_plan)
        if plan["permission_class"] == "SEPARATE":
            capability = build_tushare_capability_receipt(
                plan=plan,
                status="NOT_PROBED",
                transport_calls=0,
                reported_count=0,
                accepted_count=0,
                blocker_codes=[],
                request_ref=None,
                response_projection_sha256=None,
                probed_at=observed_at,
            )
            execution = build_tushare_execution_receipt(
                policy=validated_policy,
                plan=plan,
                request_refs=[],
                capability_receipt=capability,
                network_attempts=0,
                completed_partition_keys=[],
                missing_partition_keys=[],
                failed_partition_keys=[],
                executed_at=observed_at,
            )
            capabilities.append(capability)
            executions.append(execution)
            continue

        keyset = plan["ordered_expected_partition_keyset"]
        if (
            len(keyset) != 1
            or plan["max_attempts"] != 1
            or plan["planned_max_network_attempts"] != 1
        ):
            raise TushareContractError(
                "capability probe requires exactly one sealed POINTS attempt"
            )
        request_receipt = build_tushare_request_receipt(
            plan=plan,
            partition_key=keyset[0],
            partition_ordinal=0,
            sanitized_params=plan["fixed_params"],
            requested_at=observed_at,
        )
        request_ref = content_ref(
            request_receipt,
            identity_field="request_receipt_id",
        )
        total_attempts += 1
        try:
            response = transport.request(
                api_name=plan["api_name"],
                params=plan["fixed_params"],
                expected_fields=plan["expected_fields"],
            )
        except TushareHttpsError as error:
            status = _provider_failure_status(error)
            blockers: list[str] = []
            reported = 0
            accepted = 0
            projection_sha = None
        except Exception:
            status = "TRANSPORT_ERROR"
            blockers = []
            reported = 0
            accepted = 0
            projection_sha = None
        else:
            try:
                status, blockers, reported, accepted, projection_sha = _classify_response(
                    plan=plan, response=response
                )
            except (AttributeError, TypeError, ValueError):
                status = "SCHEMA_MISMATCH"
                blockers = []
                reported = 0
                accepted = 0
                projection_sha = None
        capability = build_tushare_capability_receipt(
            plan=plan,
            status=status,
            transport_calls=1,
            reported_count=reported,
            accepted_count=accepted,
            blocker_codes=blockers,
            request_ref=request_ref,
            response_projection_sha256=projection_sha,
            probed_at=observed_at,
        )
        failed = [keyset[0]] if status not in {"AVAILABLE", "EMPTY"} else []
        completed = [keyset[0]] if not failed else []
        execution = build_tushare_execution_receipt(
            policy=validated_policy,
            plan=plan,
            request_refs=[request_ref],
            capability_receipt=capability,
            network_attempts=1,
            completed_partition_keys=completed,
            missing_partition_keys=[],
            failed_partition_keys=failed,
            executed_at=observed_at,
        )
        request_receipts.append(request_receipt)
        capabilities.append(capability)
        executions.append(execution)

    planned_max = sum(
        plan["planned_max_network_attempts"] for plan in validated_policy["endpoint_plans"]
    )
    if total_attempts > planned_max:
        raise TushareContractError("capability probe exceeded planned attempt bound")
    return {
        "capability_receipts": tuple(capabilities),
        "execution_receipts": tuple(executions),
        "network_attempts": total_attempts,
        "policy": validated_policy,
        "request_receipts": tuple(request_receipts),
    }


__all__ = ["probe_tushare_capabilities"]

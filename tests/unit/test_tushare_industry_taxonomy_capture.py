from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from quant_investor.intelligence_v2.sources.tushare import (
    TushareContractError,
    build_industry_membership_execution_plan,
    build_industry_membership_partition_capture,
    build_industry_taxonomy_capture,
    build_industry_taxonomy_execution_plan,
    capture_tushare_industry_taxonomy,
    capture_industry_membership_partition,
    validate_industry_taxonomy_capture,
    validate_industry_taxonomy_execution_plan,
    validate_industry_membership_execution_plan,
    validate_industry_membership_partition_capture,
)
from quant_investor.intelligence_v2.sources.tushare.industry_taxonomy import (
    INDEX_CLASSIFY_FIELDS,
    OFFICIAL_PARTITIONS,
)
from quant_investor.v17_v4_runtime.tushare_https import TushareResponse

NOW = "2026-08-11T07:30:00Z"


def _rows(level: str, count: int) -> list[dict[str, Any]]:
    result = []
    for index in range(count):
        if level == "L1":
            code = f"L1{index:03d}.SI"
            parent = "0"
        elif level == "L2":
            code = f"L2{index:03d}.SI"
            parent = f"L1I{index % 31:03d}"
        else:
            code = f"L3{index:03d}.SI"
            parent = f"L2I{index % 134:03d}"
        result.append(
            {
                "index_code": code,
                "industry_name": f"{level}-{index}",
                "parent_code": parent,
                "level": level,
                "industry_code": f"{level}I{index:03d}",
                "is_pub": "1",
                "src": "SW2021",
            }
        )
    return result


def _partition_rows() -> list[dict[str, Any]]:
    result = []
    for level, count in OFFICIAL_PARTITIONS:
        rows = _rows(level, count)
        tuples = tuple(tuple(row[field] for field in INDEX_CLASSIFY_FIELDS) for row in rows)
        from quant_investor.intelligence_v2.sources.tushare.contracts import (
            response_projection_sha256,
        )

        result.append(
            {
                "level": level,
                "reported_count": count,
                "response_projection_sha256": response_projection_sha256(tuples),
                "rows": rows,
            }
        )
    return result


def _membership_plan() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    taxonomy_plan = build_industry_taxonomy_execution_plan(
        document_observed_at=NOW,
        created_at=NOW,
    )
    taxonomy_capture = build_industry_taxonomy_capture(
        plan=taxonomy_plan,
        partition_rows=_partition_rows(),
        captured_at=NOW,
    )
    plan = build_industry_membership_execution_plan(
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        document_observed_at=NOW,
        created_at=NOW,
    )
    return taxonomy_plan, taxonomy_capture, plan


def _member_row(*, l3_code: str, flag: str) -> dict[str, Any]:
    return {
        "l1_code": "L1000.SI",
        "l1_name": "L1-0",
        "l2_code": "L2000.SI",
        "l2_name": "L2-0",
        "l3_code": l3_code,
        "l3_name": "L3-0",
        "ts_code": "000001.SZ",
        "name": "公司",
        "in_date": "20211213",
        "out_date": None,
        "is_new": flag,
    }


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> TushareResponse:
        self.calls.append(kwargs)
        level = kwargs["params"]["level"]
        count = dict(OFFICIAL_PARTITIONS)[level]
        rows = _rows(level, count)
        return TushareResponse(
            api_name="index_classify",
            request_id=f"request-{level}",
            reported_count=count,
            has_more=False,
            fields=INDEX_CLASSIFY_FIELDS,
            rows=tuple(tuple(row[field] for field in INDEX_CLASSIFY_FIELDS) for row in rows),
        )


class FakeMembershipClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def request(self, **kwargs: Any) -> TushareResponse:
        self.calls.append(kwargs)
        row = _member_row(
            l3_code=kwargs["params"]["l3_code"],
            flag=kwargs["params"]["is_new"],
        )
        values = tuple(row[field] for field in kwargs["expected_fields"])
        return TushareResponse(
            api_name="index_member_all",
            request_id="membership-request-1",
            reported_count=1,
            has_more=False,
            fields=tuple(kwargs["expected_fields"]),
            rows=(values,),
        )


def test_plan_and_capture_replay_with_exact_official_cardinality() -> None:
    plan = build_industry_taxonomy_execution_plan(
        document_observed_at=NOW,
        created_at=NOW,
    )
    assert validate_industry_taxonomy_execution_plan(plan) == plan
    capture = build_industry_taxonomy_capture(
        plan=plan,
        partition_rows=_partition_rows(),
        captured_at=NOW,
    )
    assert validate_industry_taxonomy_capture(capture, plan=plan) == capture
    assert [row["reported_count"] for row in capture["partition_rows"]] == [31, 134, 346]


def test_live_boundary_makes_exactly_three_no_fallback_calls() -> None:
    plan = build_industry_taxonomy_execution_plan(
        document_observed_at=NOW,
        created_at=NOW,
    )
    client = FakeClient()
    capture = capture_tushare_industry_taxonomy(plan=plan, captured_at=NOW, client=client)
    assert len(client.calls) == 3
    assert [call["params"]["level"] for call in client.calls] == ["L1", "L2", "L3"]
    assert validate_industry_taxonomy_capture(capture, plan=plan) == capture


def test_count_hierarchy_and_resealed_forgery_are_rejected() -> None:
    plan = build_industry_taxonomy_execution_plan(
        document_observed_at=NOW,
        created_at=NOW,
    )
    partitions = _partition_rows()
    partitions[0]["rows"].pop()
    partitions[0]["reported_count"] -= 1
    with pytest.raises(TushareContractError, match="cardinality"):
        build_industry_taxonomy_capture(
            plan=plan,
            partition_rows=partitions,
            captured_at=NOW,
        )

    capture = build_industry_taxonomy_capture(
        plan=plan,
        partition_rows=_partition_rows(),
        captured_at=NOW,
    )
    forged = deepcopy(capture)
    forged["partition_rows"][1]["rows"][0]["parent_code"] = "UNKNOWN.SI"
    with pytest.raises(TushareContractError):
        validate_industry_taxonomy_capture(forged, plan=plan)


def test_membership_plan_binds_all_l3_y_n_partitions() -> None:
    taxonomy_plan, capture, plan = _membership_plan()
    endpoint = plan["endpoint_plan"]
    assert len(plan["l3_keyset"]) == 346
    assert endpoint["planned_terminal_request_count"] == 692
    assert endpoint["planned_max_network_attempts"] == 692
    assert endpoint["ordered_expected_partition_keyset"][:2] == [
        "l3_code=L3000.SI|is_new=Y",
        "l3_code=L3000.SI|is_new=N",
    ]
    assert (
        validate_industry_membership_execution_plan(
            plan,
            taxonomy_plan=taxonomy_plan,
            taxonomy_capture=capture,
        )
        == plan
    )

    forged = deepcopy(plan)
    forged["l3_keyset"].pop()
    with pytest.raises(TushareContractError):
        validate_industry_membership_execution_plan(
            forged,
            taxonomy_plan=taxonomy_plan,
            taxonomy_capture=capture,
        )


def test_membership_partition_capture_is_one_request_and_replayable() -> None:
    taxonomy_plan, taxonomy_capture, plan = _membership_plan()
    client = FakeMembershipClient()
    partition = capture_industry_membership_partition(
        membership_plan=plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        partition_ordinal=0,
        captured_at=NOW,
        client=client,
    )
    assert len(client.calls) == 1
    assert client.calls[0]["params"] == {"l3_code": "L3000.SI", "is_new": "Y"}
    assert partition["status"] == "AVAILABLE"
    assert (
        validate_industry_membership_partition_capture(
            partition,
            membership_plan=plan,
            taxonomy_plan=taxonomy_plan,
            taxonomy_capture=taxonomy_capture,
        )
        == partition
    )


def test_membership_partition_scope_limit_and_resealed_forgery_are_rejected() -> None:
    taxonomy_plan, taxonomy_capture, plan = _membership_plan()
    row = _member_row(l3_code="L3000.SI", flag="Y")
    partition = build_industry_membership_partition_capture(
        membership_plan=plan,
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
        partition_key="l3_code=L3000.SI|is_new=Y",
        partition_ordinal=0,
        provider_request_id="request-1",
        reported_count=1,
        rows=[row],
        captured_at=NOW,
    )
    forged = deepcopy(partition)
    forged["rows"][0]["l3_code"] = "UNKNOWN.SI"
    with pytest.raises(TushareContractError):
        validate_industry_membership_partition_capture(
            forged,
            membership_plan=plan,
            taxonomy_plan=taxonomy_plan,
            taxonomy_capture=taxonomy_capture,
        )
    with pytest.raises(TushareContractError, match="row limit"):
        build_industry_membership_partition_capture(
            membership_plan=plan,
            taxonomy_plan=taxonomy_plan,
            taxonomy_capture=taxonomy_capture,
            partition_key="l3_code=L3000.SI|is_new=Y",
            partition_ordinal=0,
            provider_request_id="request-2",
            reported_count=2000,
            rows=[row] * 2000,
            captured_at=NOW,
        )

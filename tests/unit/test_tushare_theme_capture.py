from __future__ import annotations

from copy import deepcopy

import pytest

from quant_investor.intelligence_v2.sources.tushare import (
    TushareContractError,
    build_theme_provider_capture,
    build_theme_provider_execution_plan,
    capture_theme_partition,
    compile_tushare_theme_source,
    derive_tdx_fallback_company_keyset,
    project_theme_provider_capture,
    validate_theme_partition_capture,
    validate_theme_provider_capture,
    validate_theme_provider_execution_plan,
)
from quant_investor.v17_v4_runtime.tushare_https import TushareResponse

CREATED = "2026-08-11T08:00:00Z"
TRADE_DATE = "20260810"


def exact_ref(name: str) -> dict[str, str]:
    digest = __import__("hashlib").sha256(name.encode()).hexdigest()
    return {
        "artifact_id": digest,
        "artifact_version": "test.exact-ref.v1",
        "available_at": CREATED,
        "byte_sha256": digest,
        "cutoff": "2026-08-10T23:59:59Z",
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": digest,
    }


class FakeClient:
    def __init__(self, rows_by_api_and_company: dict[tuple[str, str], list[tuple]]) -> None:
        self.rows_by_api_and_company = rows_by_api_and_company
        self.calls: list[tuple[str, dict[str, object]]] = []

    def request(self, *, api_name: str, params: dict, expected_fields: tuple) -> TushareResponse:
        self.calls.append((api_name, dict(params)))
        company = str(params.get("con_code", "ALL"))
        rows = self.rows_by_api_and_company[(api_name, company)]
        return TushareResponse(
            api_name=api_name,
            request_id=f"request-{len(self.calls)}",
            reported_count=len(rows),
            has_more=False,
            fields=tuple(expected_fields),
            rows=tuple(rows),
        )


def capture_provider(provider: str, companies: list[str], client: FakeClient):
    plan = build_theme_provider_execution_plan(
        provider=provider,
        trade_date=TRADE_DATE,
        company_keyset=companies,
        document_observed_at=CREATED,
        created_at=CREATED,
    )
    partitions = [
        capture_theme_partition(
            plan=plan,
            partition_ordinal=ordinal,
            captured_at="2026-08-11T08:01:00Z",
            client=client,
        )
        for ordinal in range(len(companies) + 1)
    ]
    capture = build_theme_provider_capture(
        plan=plan,
        partition_documents=partitions,
        completed_at="2026-08-11T08:02:00Z",
    )
    validate_theme_provider_execution_plan(plan)
    validate_theme_provider_capture(capture, plan=plan, partition_documents=partitions)
    return plan, partitions, capture


def test_dc_capture_seals_complete_keyset_and_derives_only_unknown_code_fallback() -> None:
    companies = ["000001.SZ", "000002.SZ", "000003.SZ"]
    client = FakeClient(
        {
            ("dc_index", "ALL"): [("BK1001.DC", TRADE_DATE, "机器人", "概念板块", "1")],
            ("dc_member", "000001.SZ"): [(TRADE_DATE, "BK1001.DC", "000001.SZ", "平安银行")],
            ("dc_member", "000002.SZ"): [(TRADE_DATE, "BK9999.DC", "000002.SZ", "万科A")],
            ("dc_member", "000003.SZ"): [],
        }
    )
    plan, partitions, capture = capture_provider("TUSHARE_DC", companies, client)
    assert capture["status"] == "COMPLETE"
    assert len(client.calls) == 4
    assert derive_tdx_fallback_company_keyset(
        dc_plan=plan,
        dc_capture=capture,
        dc_partition_documents=partitions,
    ) == ["000002.SZ"]


def test_dc_to_tdx_projection_compiles_existing_i3_contract_without_mixing() -> None:
    companies = ["000001.SZ", "000002.SZ", "000003.SZ"]
    dc_client = FakeClient(
        {
            ("dc_index", "ALL"): [("BK1001.DC", TRADE_DATE, "机器人", "概念板块", "1")],
            ("dc_member", "000001.SZ"): [(TRADE_DATE, "BK1001.DC", "000001.SZ", "平安银行")],
            ("dc_member", "000002.SZ"): [(TRADE_DATE, "BK9999.DC", "000002.SZ", "万科A")],
            ("dc_member", "000003.SZ"): [],
        }
    )
    dc_plan, dc_partitions, dc_capture = capture_provider("TUSHARE_DC", companies, dc_client)
    fallback = derive_tdx_fallback_company_keyset(
        dc_plan=dc_plan,
        dc_capture=dc_capture,
        dc_partition_documents=dc_partitions,
    )
    tdx_client = FakeClient(
        {
            ("tdx_index", "ALL"): [("880001.TDX", TRADE_DATE, "航运概念", "概念板块", 20)],
            ("tdx_member", "000002.SZ"): [("880001.TDX", TRADE_DATE, "000002.SZ", "万科A")],
        }
    )
    tdx_plan, tdx_partitions, tdx_capture = capture_provider("TUSHARE_TDX", fallback, tdx_client)
    dc_projection = project_theme_provider_capture(
        plan=dc_plan,
        capture=dc_capture,
        partition_documents=dc_partitions,
    )
    tdx_projection = project_theme_provider_capture(
        plan=tdx_plan,
        capture=tdx_capture,
        partition_documents=tdx_partitions,
    )
    bundle = compile_tushare_theme_source(
        trade_date=TRADE_DATE,
        company_keyset=companies,
        dc_registry_rows=dc_projection["registry_rows"],
        dc_registry_source_ref=dc_projection["registry_source_ref"],
        dc_membership_captures=dc_projection["membership_captures"],
        tdx_registry_rows=tdx_projection["registry_rows"],
        tdx_registry_source_ref=tdx_projection["registry_source_ref"],
        tdx_membership_captures=tdx_projection["membership_captures"],
        scope_ref=exact_ref("scope"),
        owner_policy_ref=exact_ref("policy"),
        captured_at=tdx_projection["captured_at"],
        as_of=tdx_projection["captured_at"],
    )
    assert bundle["source_receipt"]["tdx_fallback_company_keyset"] == ["000002.SZ"]
    rows = {row["company_code"]: row for row in bundle["source_receipt"]["company_rows"]}
    assert rows["000001.SZ"]["theme_ids"] == ["TUSHARE_DC:BK1001.DC"]
    assert rows["000002.SZ"]["theme_ids"] == ["TUSHARE_TDX:880001.TDX"]
    assert rows["000003.SZ"]["provider"] == "TUSHARE_DC"
    assert rows["000003.SZ"]["theme_ids"] == []


def test_partition_replay_and_scope_tampering_fail_closed() -> None:
    client = FakeClient(
        {
            ("dc_index", "ALL"): [("BK1001.DC", TRADE_DATE, "机器人", "概念板块", "1")],
            ("dc_member", "000001.SZ"): [],
        }
    )
    plan, partitions, _ = capture_provider("TUSHARE_DC", ["000001.SZ"], client)
    forged = deepcopy(partitions[1])
    forged["partition_key"] = "member:000002.SZ:20260810"
    with pytest.raises(TushareContractError):
        validate_theme_partition_capture(forged, plan=plan)


def test_transport_failure_is_terminal_incomplete_and_enters_fallback() -> None:
    class FailingClient(FakeClient):
        def request(self, *, api_name: str, params: dict, expected_fields: tuple):
            if api_name == "dc_member":
                raise RuntimeError("secret provider text")
            return super().request(
                api_name=api_name,
                params=params,
                expected_fields=expected_fields,
            )

    client = FailingClient(
        {("dc_index", "ALL"): [("BK1001.DC", TRADE_DATE, "机器人", "概念板块", "1")]}
    )
    plan, partitions, capture = capture_provider("TUSHARE_DC", ["000001.SZ"], client)
    assert capture["status"] == "PARTIAL"
    assert partitions[1]["blocker_codes"] == ["TRANSPORT_ERROR"]
    assert "secret provider text" not in repr(partitions[1])
    assert derive_tdx_fallback_company_keyset(
        dc_plan=plan,
        dc_capture=capture,
        dc_partition_documents=partitions,
    ) == ["000001.SZ"]

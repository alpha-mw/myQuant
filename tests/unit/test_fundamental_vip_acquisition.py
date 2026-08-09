from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.intelligence_v2.sources.tushare import (
    build_endpoint_execution_plan,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    acquire_fundamental_vip_v4,
    build_fundamental_execution_closure_v4,
    build_fundamental_request_plan_v4,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.models import (
    SOURCE_ENDPOINTS,
)
from quant_investor.v17_v4_runtime.tushare_https import TushareResponse

NOW = "2026-08-09T08:00:00Z"
CUTOFF = "2026-08-07T23:59:59Z"

FIELDS = {
    "balancesheet": ["ts_code", "end_date", "total_assets"],
    "cashflow": ["ts_code", "end_date", "n_cashflow_act"],
    "daily_basic": ["ts_code", "trade_date", "total_mv"],
    "fina_indicator": ["ts_code", "end_date", "roe"],
    "forecast": ["ts_code", "end_date", "type"],
    "income": ["ts_code", "end_date", "n_income"],
}


def exact_ref(name: str) -> dict[str, str]:
    return {
        "artifact_id": name,
        "artifact_version": f"{name}.v1",
        "available_at": CUTOFF,
        "byte_sha256": "a" * 64,
        "cutoff": CUTOFF,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": "b" * 64,
    }


def sessions() -> list[str]:
    return [value.strftime("%Y%m%d") for value in pd.bdate_range("20210807", "20260807")]


def periods() -> list[str]:
    result: list[str] = []
    for value in pd.date_range("20190930", "20260630", freq="QE"):
        result.append(value.strftime("%Y%m%d"))
    return result


def endpoint_plans() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for table, endpoint in SOURCE_ENDPOINTS.items():
        dimension = "trade_date" if table == "daily_basic" else "period"
        values = sessions() if table == "daily_basic" else periods()
        keyset = [f"{dimension}={value}" for value in values]
        result[table] = build_endpoint_execution_plan(
            api_name=endpoint,
            lane="FUNDAMENTAL",
            permission_class="POINTS",
            official_document_url="https://tushare.pro/document/2?doc_id=1",
            official_document_id=f"tushare.{endpoint}",
            document_observed_at=NOW,
            documented_min_points=2000,
            strict_decimal_decode=True,
            expected_fields=FIELDS[table],
            fixed_params={},
            partition_dimensions=[dimension],
            ordered_expected_partition_keyset=keyset,
            documented_row_limit=6000,
            max_attempts=2,
            retry_schedule=[0, 0],
            empty_partition_rule="BASELINE_IDENTITY_EMPTY",
            completeness_proof="EXACT_PARTITION_AND_COUNT",
            limit_hit_action="BLOCK",
            planned_terminal_request_count=len(keyset),
            planned_max_network_attempts=len(keyset) * 2,
            created_at=NOW,
        )
    return result


def execution() -> dict[str, Any]:
    endpoints = endpoint_plans()
    plan = build_fundamental_request_plan_v4(
        as_of="20260807",
        pit_cutoff=CUTOFF,
        symbols=["000001.SZ", "600000.SH"],
        canonical_open_sessions=sessions(),
        market_scope_ref=exact_ref("market-scope"),
        market_calendar_ref=exact_ref("market-calendar"),
        baseline_provider_manifest_ref=exact_ref("baseline-provider"),
        baseline_network_attempts=12,
        baseline_empty_partition_keyset=[],
        endpoint_plans=endpoints,
        max_attempts_per_partition=2,
        implementation_sha256="c" * 64,
        created_at=NOW,
    )
    return build_fundamental_execution_closure_v4(
        plan=plan,
        endpoint_plans=endpoints,
        created_at=NOW,
    )


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareResponse:
        self.calls += 1
        table = next(table for table, endpoint in SOURCE_ENDPOINTS.items() if endpoint == api_name)
        partition = params["trade_date" if table == "daily_basic" else "period"]
        values: list[Any] = []
        for field in expected_fields:
            if field == "ts_code":
                values.append("000001.SZ")
            elif field in {"trade_date", "end_date"}:
                values.append(partition)
            elif field == "type":
                values.append("预增")
            else:
                values.append(Decimal("0.1"))
        return TushareResponse(
            api_name=api_name,
            request_id=f"request-{self.calls}",
            reported_count=1,
            has_more=False,
            fields=tuple(expected_fields),
            rows=(tuple(values),),
        )


def test_full_partition_executor_closes_exact_keyset_without_fallback() -> None:
    closure = execution()
    client = FakeClient()
    result = acquire_fundamental_vip_v4(
        execution_closure=closure,
        client=client,
        captured_at=NOW,
        sleeper=lambda _seconds: None,
    )

    assert result["status"] == "COMPLETE"
    assert result["network_attempts"] == closure["request_plan"]["planned_terminal_request_count"]
    assert client.calls == result["network_attempts"]
    assert len(result["physical_receipts"]) == result["network_attempts"]
    assert set(result["raw_tables"]) == set(SOURCE_ENDPOINTS)
    assert all(not frame.empty for frame in result["raw_tables"].values())


def test_scope_mismatch_blocks_one_partition_without_retry_or_fallback() -> None:
    closure = execution()

    class WrongScope(FakeClient):
        def request(self, **kwargs: Any) -> TushareResponse:
            response = super().request(**kwargs)
            if self.calls != 1:
                return response
            row = list(response.rows[0])
            row[0] = "999999.SZ"
            return TushareResponse(
                api_name=response.api_name,
                request_id=response.request_id,
                reported_count=1,
                has_more=False,
                fields=response.fields,
                rows=(tuple(row),),
            )

    client = WrongScope()
    result = acquire_fundamental_vip_v4(
        execution_closure=closure,
        client=client,
        captured_at=NOW,
        sleeper=lambda _seconds: None,
    )
    assert result["status"] == "BLOCKED"
    assert result["physical_receipts"][0]["blocker_codes"] == ["SCOPE_MISMATCH"]
    assert client.calls == closure["request_plan"]["planned_terminal_request_count"]


def test_endpoint_partition_policy_must_cover_full_schedule() -> None:
    closure = execution()
    closure["endpoint_plans"]["income"]["ordered_expected_partition_keyset"] = ["period=20260630"]
    client = FakeClient()
    try:
        acquire_fundamental_vip_v4(
            execution_closure=closure,
            client=client,
            captured_at=NOW,
            sleeper=lambda _seconds: None,
        )
    except Exception:
        pass
    else:
        raise AssertionError("forged endpoint closure was accepted")
    assert client.calls == 0


def test_private_checkpoint_resume_replays_without_network(tmp_path: Path) -> None:
    closure = execution()
    checkpoint = tmp_path / "vip-checkpoint"
    client = FakeClient()
    first = acquire_fundamental_vip_v4(
        execution_closure=closure,
        client=client,
        captured_at=NOW,
        checkpoint_root=checkpoint,
        sleeper=lambda _seconds: None,
    )
    first_calls = client.calls

    class NetworkForbidden(FakeClient):
        def request(self, **_kwargs: Any) -> TushareResponse:
            raise AssertionError("resume attempted a provider call")

    second = acquire_fundamental_vip_v4(
        execution_closure=closure,
        client=NetworkForbidden(),
        captured_at=NOW,
        checkpoint_root=checkpoint,
        sleeper=lambda _seconds: None,
    )
    assert first["network_attempts"] == second["network_attempts"] == first_calls
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in checkpoint.rglob("*.json"))
    assert checkpoint.stat().st_mode & 0o777 == 0o700
    assert (checkpoint / "partition_records").stat().st_mode & 0o777 == 0o700

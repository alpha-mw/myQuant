from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant_investor.intelligence_v2.sources.tushare import (
    build_endpoint_execution_plan,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    REQUIRED_EVIDENCE_PATHS,
    acquire_fundamental_vip_v4,
    build_fundamental_comparison_policy,
    build_fundamental_execution_closure_v4,
    build_fundamental_request_plan_v4,
    build_fundamental_shadow_bundle_v4,
    build_logical_coverages_from_shadow_v4,
    capture_provider_evidence_directory,
    materialize_fundamental_v4_staging_generation,
    write_fundamental_shadow_bundle_v4,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.models import (
    SOURCE_ENDPOINTS,
    SOURCE_TABLES,
    FundamentalV4ContractError,
)
from quant_investor.v17_v4_runtime.tushare_https import TushareResponse
from quant_investor.market.fundamental_mart import (
    _issue_live_tushare_v4_attestation,
    _provider_source_priority,
)

NOW = "2026-08-09T08:00:00Z"
CUTOFF = "2026-08-07T23:59:59Z"
SYMBOL = "000001.SZ"


def _exact_ref(name: str) -> dict[str, str]:
    return {
        "artifact_id": name,
        "artifact_version": f"{name}.v1",
        "available_at": CUTOFF,
        "byte_sha256": "a" * 64,
        "cutoff": CUTOFF,
        "relative_path": f"fixtures/{name}.json",
        "semantic_sha256": "b" * 64,
    }


def _sessions() -> list[str]:
    return [value.strftime("%Y%m%d") for value in pd.bdate_range("20210807", "20260807")]


def _periods() -> list[str]:
    return [value.strftime("%Y%m%d") for value in pd.date_range("20190930", "20260630", freq="QE")]


def _fields(table: str) -> list[str]:
    return [
        "ts_code",
        "trade_date" if table == "daily_basic" else "end_date",
        "value",
    ]


def _endpoint_plans() -> dict[str, dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    for table, endpoint in SOURCE_ENDPOINTS.items():
        dimension = "trade_date" if table == "daily_basic" else "period"
        values = _sessions() if table == "daily_basic" else _periods()
        plans[table] = build_endpoint_execution_plan(
            api_name=endpoint,
            lane="FUNDAMENTAL",
            permission_class="POINTS",
            official_document_url="https://tushare.pro/document/2?doc_id=1",
            official_document_id=f"tushare.{endpoint}",
            document_observed_at=NOW,
            documented_min_points=2000,
            strict_decimal_decode=True,
            expected_fields=_fields(table),
            fixed_params={},
            partition_dimensions=[dimension],
            ordered_expected_partition_keyset=[f"{dimension}={value}" for value in values],
            documented_row_limit=6000,
            max_attempts=1,
            retry_schedule=[0],
            empty_partition_rule="BASELINE_SAME_IDENTITY_ONLY",
            completeness_proof="EXACT_TERMINAL_KEYSET",
            limit_hit_action="BLOCK",
            planned_terminal_request_count=len(values),
            planned_max_network_attempts=len(values),
            created_at=NOW,
        )
    return plans


def _execution() -> dict[str, Any]:
    endpoints = _endpoint_plans()
    plan = build_fundamental_request_plan_v4(
        as_of="20260807",
        pit_cutoff=CUTOFF,
        symbols=[SYMBOL],
        canonical_open_sessions=_sessions(),
        market_scope_ref=_exact_ref("scope"),
        market_calendar_ref=_exact_ref("calendar"),
        baseline_provider_manifest_ref=_exact_ref("baseline"),
        baseline_network_attempts=12,
        baseline_empty_partition_keyset=[],
        endpoint_plans=endpoints,
        max_attempts_per_partition=1,
        implementation_sha256="c" * 64,
        created_at=NOW,
    )
    return build_fundamental_execution_closure_v4(
        plan=plan,
        endpoint_plans=endpoints,
        created_at=NOW,
    )


class _Client:
    def request(
        self,
        *,
        api_name: str,
        params: dict[str, Any],
        expected_fields: list[str],
    ) -> TushareResponse:
        date_value = str(params.get("trade_date") or params.get("period"))
        return TushareResponse(
            api_name=api_name,
            request_id=f"request-{api_name}-{date_value}",
            reported_count=1,
            has_more=False,
            fields=tuple(expected_fields),
            rows=((SYMBOL, date_value, Decimal("1.25")),),
        )


def _comparison_policy(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    table_policies: dict[str, dict[str, Any]] = {}
    for table in SOURCE_TABLES:
        date_column = "trade_date" if table == "daily_basic" else "end_date"
        table_policies[table] = {
            "canonical_key_columns": ["ts_code", date_column],
            "column_rows": [
                {"column": "ts_code", "kind": "TEXT"},
                {"column": date_column, "kind": "DATE"},
                {"column": "value", "kind": "DECIMAL"},
            ],
            "table": table,
            "winner_implementation_sha256": "d" * 64,
            "winner_order_columns": list(tables[table].columns),
            "winner_rule": "ASCII_CANONICAL_LAST",
        }
    return build_fundamental_comparison_policy(
        table_policies=table_policies,
        created_at=NOW,
    )


def test_shadow_bundle_is_complete_private_and_promotion_blocked_by_performance(
    tmp_path: Path,
) -> None:
    execution = _execution()
    acquired = acquire_fundamental_vip_v4(
        execution_closure=execution,
        client=_Client(),
        captured_at=NOW,
        sleeper=lambda _delay: None,
    )
    tables = acquired["raw_tables"]
    derived = {
        table: {"baseline_sha256": "e" * 64, "vip_sha256": "e" * 64}
        for table in ("coverage", "fundamental_daily", "fundamental_period", "quarantine")
    }
    bundle = build_fundamental_shadow_bundle_v4(
        execution_closure=execution,
        physical_receipts=acquired["physical_receipts"],
        logical_coverages=build_logical_coverages_from_shadow_v4(
            execution_closure=execution,
            physical_receipts=acquired["physical_receipts"],
            vip_tables=tables,
            assessed_at=NOW,
        ),
        baseline_tables=tables,
        vip_tables=tables,
        comparison_policy=_comparison_policy(tables),
        derived_fingerprints=derived,
        assembled_at=NOW,
    )
    assert bundle["status"] == "BLOCKED"
    assert bundle["provider_manifest"] is None
    assert bundle["reconciliation"]["blocker_codes"] == ["VIP_NETWORK_ATTEMPT_LIMIT_EXCEEDED"]
    assert set(bundle["payloads"]) == {
        *REQUIRED_EVIDENCE_PATHS,
        "fileset_manifest.json",
    }

    output = tmp_path / "provider_evidence"
    written = write_fundamental_shadow_bundle_v4(bundle=bundle, output_root=output)
    assert written["status"] == "BLOCKED"
    assert written["provider_manifest_sha256"] is None
    assert capture_provider_evidence_directory(output) == bundle["payloads"]
    assert oct(output.stat().st_mode & 0o777) == "0o700"
    assert all(
        oct(path.stat().st_mode & 0o777) == "0o600" for path in output.rglob("*") if path.is_file()
    )

    staging = tmp_path / "staging"
    with pytest.raises(FundamentalV4ContractError, match="blocked shadow bundle"):
        materialize_fundamental_v4_staging_generation(
            execution_closure=execution,
            bundle=bundle,
            vip_tables=tables,
            vip_derived_tables={},
            data_root=staging,
            raw_snapshot_root=tmp_path / "snapshots",
            reports_root=tmp_path / "reports",
            run_id="blocked-shadow",
        )
    assert not staging.exists()


def test_v4_live_attestation_is_internal_and_manifest_bound() -> None:
    tables = {table: pd.DataFrame() for table in SOURCE_TABLES}
    manifest = {
        "authoritative_full_rebuild": True,
        "performance_gate_passed": True,
        "raw_table_fingerprints": {table: "a" * 64 for table in SOURCE_TABLES},
        "schema_version": "cn-fundamental-provider-manifest.v4",
    }
    attestation = _issue_live_tushare_v4_attestation(
        "live_tushare_vip",
        manifest,
        tables,
    )
    assert (
        _provider_source_priority(
            "live_tushare_vip",
            manifest,
            tables,
            attestation,
        )
        == "tushare_primary"
    )
    changed = {**manifest, "performance_gate_passed": False}
    with pytest.raises(ValueError, match="internal live Tushare attestation"):
        _provider_source_priority(
            "live_tushare_vip",
            changed,
            tables,
            attestation,
        )

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.market.tushare import (
    build_endpoint_execution_plan,
    build_tushare_endpoint_policy,
    validate_tushare_schema_diagnostic_receipt,
)
from quant_investor.market.tushare._core import canonical_bytes
from quant_investor.market.tushare_transport import TushareSchemaDiagnostic
from scripts import diagnose_tushare_endpoint_schema as schema_script

NOW = "2026-08-14T12:30:00Z"
CANARY = "PRIVATE_BUSINESS_VALUE_CANARY"


def plan(*, api_name: str = "forecast_vip") -> dict:
    return build_endpoint_execution_plan(
        api_name=api_name,
        lane="FUNDAMENTAL",
        permission_class="POINTS",
        official_document_url="https://tushare.pro/document/2?doc_id=45",
        official_document_id=f"tushare.{api_name}",
        document_observed_at="2026-08-14T12:29:00Z",
        documented_min_points=5000,
        strict_decimal_decode=True,
        expected_fields=["ts_code", "ann_date", "end_date", "summary"],
        fixed_params={"period": "20260630"},
        partition_dimensions=["period"],
        ordered_expected_partition_keyset=["period=20260630"],
        documented_row_limit=3500,
        max_attempts=1,
        retry_schedule=[0],
        empty_partition_rule="BASELINE_IDENTITY_EMPTY",
        completeness_proof="EXACT_PARTITION_AND_COUNT",
        limit_hit_action="BLOCK",
        planned_terminal_request_count=1,
        planned_max_network_attempts=1,
        created_at=NOW,
    )


def policy_file(tmp_path: Path, *, plans: list[dict] | None = None) -> tuple[Path, str]:
    policy = build_tushare_endpoint_policy(
        created_at=NOW,
        endpoint_plans=[plan()] if plans is None else plans,
    )
    raw = canonical_bytes(policy)
    path = tmp_path / "policy.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def args(path: Path, digest: str, output: Path, *, live: bool) -> argparse.Namespace:
    return argparse.Namespace(
        allow_live=live,
        policy_path=str(path),
        policy_sha256=digest,
        output_root=str(output),
        diagnosed_at=NOW,
    )


class FakeSchemaClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def diagnose_schema(self, **kwargs: object) -> TushareSchemaDiagnostic:
        self.calls.append(dict(kwargs))
        return TushareSchemaDiagnostic(
            api_name="forecast_vip",
            status="OBSERVED",
            provider_code=0,
            request_id_sha256=hashlib.sha256(b"provider-request-secret").hexdigest(),
            response_body_sha256=hashlib.sha256(CANARY.encode()).hexdigest(),
            provider_reported_count=1,
            item_count=1,
            has_more=False,
            observed_fields=("ts_code", "ann_date", "end_date"),
            expected_fields_match=False,
            row_widths=(3,),
            cell_types=("TEXT",),
            text_cell_count=3,
            non_nfc_text_count=0,
            max_text_utf8_bytes=len(CANARY.encode()),
        )


def test_dry_run_never_constructs_or_calls_transport(tmp_path: Path) -> None:
    path, digest = policy_file(tmp_path)
    output = tmp_path / "dry-output"
    client = FakeSchemaClient()
    result = schema_script.run(args(path, digest, output, live=False), client=client)
    assert result["status"] == "DRY_RUN_VALIDATED"
    assert client.calls == []
    assert not output.exists()


def test_live_capture_uses_injected_client_and_writes_stable_receipt(
    tmp_path: Path,
) -> None:
    path, digest = policy_file(tmp_path)
    output = tmp_path / "live-output"
    client = FakeSchemaClient()
    result = schema_script.run(args(path, digest, output, live=True), client=client)

    assert result["status"] == "LIVE_DIAGNOSTIC_RECORDED"
    assert result["network_attempts"] == 1
    assert len(client.calls) == 1
    policy = json.loads((output / "policy.json").read_bytes())
    request = json.loads((output / "request_receipt.json").read_bytes())
    receipt = json.loads((output / "schema_diagnostic_receipt.json").read_bytes())
    assert receipt["kind"] == "market.tushare.schema_diagnostic_receipt"
    assert len(receipt["contract_sha256"]) == 64
    assert CANARY.encode() not in canonical_bytes(receipt)
    diagnostic = vars(FakeSchemaClient().diagnose_schema())
    assert validate_tushare_schema_diagnostic_receipt(
        receipt,
        plan=policy["endpoint_plans"][0],
        request_receipt=request,
        sanitized_params={"period": "20260630"},
        diagnostic=diagnostic,
    ) == receipt


def test_more_than_one_plan_is_rejected_before_transport(tmp_path: Path) -> None:
    path, digest = policy_file(
        tmp_path,
        plans=[plan(), plan(api_name="income_vip")],
    )
    client = FakeSchemaClient()
    with pytest.raises(schema_script.ProbeSafetyError, match="ONE_REQUEST"):
        schema_script.run(
            args(path, digest, tmp_path / "blocked", live=True),
            client=client,
        )
    assert client.calls == []

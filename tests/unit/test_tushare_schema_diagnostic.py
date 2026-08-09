from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from quant_investor.intelligence_v2._core import canonical_bytes
from quant_investor.intelligence_v2.sources.tushare import (
    TushareContractError,
    build_endpoint_execution_plan,
    build_tushare_endpoint_policy,
    build_tushare_request_receipt,
    build_tushare_schema_diagnostic_receipt,
    validate_tushare_schema_diagnostic_receipt,
)
from quant_investor.v17_v4_runtime import tushare_https
from quant_investor.v17_v4_runtime.tushare_https import (
    OfficialTushareHttpsClient,
    TushareHttpsError,
)
from scripts import diagnose_tushare_endpoint_schema as schema_script

TOKEN = "A" * 40
CANARY = "PRIVATE_BUSINESS_VALUE_CANARY"
NOW = "2026-08-09T12:30:00Z"
EXPECTED_FIELDS = ("ts_code", "ann_date", "end_date", "summary")
OBSERVED_FIELDS = ("ts_code", "ann_date", "end_date")


class Response:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self, amount: int) -> bytes:
        return self.body[:amount]


class Connection:
    body = b""
    calls = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def request(self, *args: Any, **kwargs: Any) -> None:
        type(self).calls += 1

    def getresponse(self) -> Response:
        return Response(self.body)

    def close(self) -> None:
        pass


def response_body() -> bytes:
    return json.dumps(
        {
            "code": 0,
            "data": {
                "count": 0,
                "fields": list(OBSERVED_FIELDS),
                "has_more": False,
                "items": [["000001.SZ", "20260715", CANARY]],
            },
            "detail": "",
            "msg": "",
            "request_id": "provider-request-secret",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def plan() -> dict[str, Any]:
    return build_endpoint_execution_plan(
        api_name="forecast_vip",
        lane="FUNDAMENTAL",
        permission_class="POINTS",
        official_document_url="https://tushare.pro/document/2?doc_id=45",
        official_document_id="tushare.doc.45",
        document_observed_at="2026-08-09T12:29:00Z",
        documented_min_points=5000,
        strict_decimal_decode=True,
        expected_fields=EXPECTED_FIELDS,
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


def request_receipt(endpoint_plan: dict[str, Any]) -> dict[str, Any]:
    return build_tushare_request_receipt(
        plan=endpoint_plan,
        partition_key="period=20260630",
        partition_ordinal=0,
        sanitized_params={"period": "20260630"},
        requested_at=NOW,
    )


@pytest.fixture(autouse=True)
def install_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    Connection.body = response_body()
    Connection.calls = 0
    monkeypatch.setenv("TUSHARE_TOKEN", TOKEN)
    monkeypatch.setattr(tushare_https, "_HTTPS_CONNECTION", Connection)
    monkeypatch.setattr(tushare_https, "_CREATE_DEFAULT_CONTEXT", object)


def test_transport_projects_schema_without_business_values() -> None:
    client = OfficialTushareHttpsClient(strict_decimal_decode=True)
    diagnostic = client.diagnose_schema(
        api_name="forecast_vip",
        params={"period": "20260630"},
        expected_fields=EXPECTED_FIELDS,
    )

    assert diagnostic.observed_fields == OBSERVED_FIELDS
    assert diagnostic.expected_fields_match is False
    assert diagnostic.provider_reported_count == 0
    assert diagnostic.item_count == 1
    assert diagnostic.row_widths == (3,)
    assert diagnostic.cell_types == ("TEXT",)
    assert diagnostic.request_id_sha256 == hashlib.sha256(b"provider-request-secret").hexdigest()
    diagnostic_bytes = json.dumps(
        asdict(diagnostic), default=list, separators=(",", ":"), sort_keys=True
    ).encode()
    assert CANARY.encode() not in diagnostic_bytes
    assert b"provider-request-secret" not in diagnostic_bytes
    assert Connection.calls == 1

    with pytest.raises(TushareHttpsError, match="TUSHARE_RESPONSE_INVALID"):
        client.request(
            api_name="forecast_vip",
            params={"period": "20260630"},
            expected_fields=EXPECTED_FIELDS,
        )


def test_schema_diagnostic_requires_explicit_strict_mode() -> None:
    with pytest.raises(TushareHttpsError, match="TUSHARE_CLIENT_CONFIG_INVALID"):
        OfficialTushareHttpsClient().diagnose_schema(
            api_name="forecast_vip",
            params={"period": "20260630"},
            expected_fields=EXPECTED_FIELDS,
        )
    assert Connection.calls == 0


def test_receipt_is_sealed_replayed_and_rejects_business_values() -> None:
    endpoint_plan = plan()
    request = request_receipt(endpoint_plan)
    diagnostic = asdict(
        OfficialTushareHttpsClient(strict_decimal_decode=True).diagnose_schema(
            api_name="forecast_vip",
            params={"period": "20260630"},
            expected_fields=EXPECTED_FIELDS,
        )
    )
    receipt = build_tushare_schema_diagnostic_receipt(
        plan=endpoint_plan,
        request_receipt=request,
        sanitized_params={"period": "20260630"},
        diagnostic=diagnostic,
        captured_at=NOW,
    )

    assert (
        validate_tushare_schema_diagnostic_receipt(
            receipt,
            plan=endpoint_plan,
            request_receipt=request,
            sanitized_params={"period": "20260630"},
            diagnostic=diagnostic,
        )
        == receipt
    )
    assert CANARY.encode() not in canonical_bytes(receipt)
    assert set(receipt) == {
        "api_name",
        "authority",
        "cell_types",
        "decision_protocol",
        "expected_fields_match",
        "frozen_v1_manifest_sha256",
        "has_more",
        "item_count",
        "observed_fields",
        "plan_ref",
        "production",
        "provider_code",
        "provider_reported_count",
        "request_id_sha256",
        "request_ref",
        "research_only",
        "response_body_sha256",
        "row_widths",
        "schema_diagnostic_receipt_id",
        "semantic_sha256",
        "status",
        "strict_decimal_decode",
        "timestamp",
        "version",
    }

    tampered = dict(receipt)
    tampered["observed_fields"] = list(EXPECTED_FIELDS)
    with pytest.raises(TushareContractError):
        validate_tushare_schema_diagnostic_receipt(
            tampered,
            plan=endpoint_plan,
            request_receipt=request,
            sanitized_params={"period": "20260630"},
            diagnostic=diagnostic,
        )

    forged = dict(diagnostic)
    forged["business_value"] = CANARY
    with pytest.raises(TushareContractError, match="shape"):
        build_tushare_schema_diagnostic_receipt(
            plan=endpoint_plan,
            request_receipt=request,
            sanitized_params={"period": "20260630"},
            diagnostic=forged,
            captured_at=NOW,
        )


def test_guarded_script_dry_run_and_single_live_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_plan = plan()
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[endpoint_plan])
    policy_path = tmp_path / "policy.json"
    policy_raw = canonical_bytes(policy)
    policy_path.write_bytes(policy_raw)
    output_root = tmp_path / "diagnostic-output"
    args = argparse.Namespace(
        allow_live=False,
        policy_path=str(policy_path),
        policy_sha256=hashlib.sha256(policy_raw).hexdigest(),
        output_root=str(output_root),
        diagnosed_at=NOW,
    )

    dry = schema_script.run(args)
    assert dry["status"] == "DRY_RUN_VALIDATED"
    assert not output_root.exists()
    assert Connection.calls == 0

    args.allow_live = True
    live = schema_script.run(args)
    assert live["status"] == "LIVE_DIAGNOSTIC_RECORDED"
    assert live["network_attempts"] == 1
    assert Connection.calls == 1
    assert oct(output_root.stat().st_mode & 0o777) == "0o700"
    for path in output_root.iterdir():
        assert oct(path.stat().st_mode & 0o777) == "0o600"
        assert CANARY.encode() not in path.read_bytes()


def test_guarded_script_rejects_more_than_one_plan(tmp_path: Path) -> None:
    endpoint_plan = plan()
    second = build_endpoint_execution_plan(
        **{
            **{
                key: value
                for key, value in endpoint_plan.items()
                if key
                not in {
                    "authority",
                    "decision_protocol",
                    "frozen_v1_manifest_sha256",
                    "plan_id",
                    "production",
                    "research_only",
                    "semantic_sha256",
                    "timestamp",
                    "version",
                }
            },
            "api_name": "income_vip",
        }
    )
    policy = build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[endpoint_plan, second])
    policy_raw = canonical_bytes(policy)
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(policy_raw)
    with pytest.raises(schema_script.ProbeSafetyError, match="ONE_REQUEST"):
        schema_script.run(
            argparse.Namespace(
                allow_live=False,
                policy_path=str(policy_path),
                policy_sha256=hashlib.sha256(policy_raw).hexdigest(),
                output_root=str(tmp_path / "out"),
                diagnosed_at=NOW,
            )
        )

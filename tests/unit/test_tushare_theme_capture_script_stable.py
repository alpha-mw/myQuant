from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import stat

import pytest

from quant_investor.market.tushare import (
    build_theme_provider_execution_plan,
    validate_theme_provider_capture,
    validate_theme_provider_execution_plan,
)
from quant_investor.market.tushare._core import canonical_bytes
from quant_investor.market.tushare_transport import TushareResponse
from tests.unit.tushare_response_fixtures import make_tushare_response
from scripts.capture_tushare_theme_provider import run
from scripts.probe_tushare_10000_capabilities import ProbeSafetyError


class Client:
    def __init__(self) -> None:
        self.calls = 0

    def request(self, *, api_name: str, params: dict, expected_fields: tuple):
        self.calls += 1
        company = params.get("con_code")
        if api_name == "dc_index":
            rows = [("BK1001.DC", "20260810", "机器人", "概念板块", "1")]
        elif company == "000001.SZ":
            rows = [("20260810", "BK1001.DC", company, "平安银行")]
        else:
            rows = []
        return make_tushare_response(
            api_name=api_name,
            request_id=f"request-{self.calls}",
            reported_count=len(rows),
            has_more=False,
            fields=tuple(expected_fields),
            rows=tuple(rows),
        )


def plan_file(tmp_path: Path) -> tuple[Path, str, dict]:
    plan = build_theme_provider_execution_plan(
        provider="TUSHARE_DC",
        trade_date="20260810",
        company_keyset=["000001.SZ", "000002.SZ"],
        document_observed_at="2026-08-11T08:00:00Z",
        created_at="2026-08-11T08:00:00Z",
    )
    path = tmp_path / "plan.json"
    raw = canonical_bytes(plan)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), plan


def args(path: Path, digest: str, output: Path, *, live: bool, resume: bool = False):
    return Namespace(
        allow_live=live,
        resume=resume,
        plan_path=str(path),
        plan_sha256=digest,
        output_root=str(output),
    )


def test_dry_run_has_zero_calls_and_does_not_create_output(tmp_path: Path) -> None:
    path, digest, _ = plan_file(tmp_path)
    output = tmp_path / "dry-output"
    client = Client()
    result = run(args(path, digest, output, live=False), client=client)
    assert result["status"] == "DRY_RUN_VALIDATED"
    assert result["planned_partitions"] == 3
    assert client.calls == 0
    assert not output.exists()


def test_live_capture_is_exact_and_resume_makes_zero_calls(tmp_path: Path) -> None:
    path, digest, plan = plan_file(tmp_path)
    output = tmp_path / "capture"
    client = Client()
    result = run(args(path, digest, output, live=True), client=client)
    assert result["status"] == "COMPLETE"
    assert client.calls == 3
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.rglob("*.json"))
    partitions = [
        json.loads((output / "partitions" / f"{ordinal:05d}.json").read_bytes())
        for ordinal in range(3)
    ]
    validate_theme_provider_capture(
        json.loads((output / "capture.json").read_bytes()),
        plan=validate_theme_provider_execution_plan(plan),
        partition_documents=partitions,
    )
    resumed = Client()
    resume_result = run(
        args(path, digest, output, live=True, resume=True),
        client=resumed,
    )
    assert resume_result["capture_id"] == result["capture_id"]
    assert resume_result["network_attempts"] == 0
    assert resumed.calls == 0


def test_registry_failure_stops_before_company_requests(tmp_path: Path) -> None:
    class BrokenRegistry(Client):
        def request(self, *, api_name: str, params: dict, expected_fields: tuple):
            self.calls += 1
            raise RuntimeError("provider unavailable")

    path, digest, _ = plan_file(tmp_path)
    client = BrokenRegistry()
    with pytest.raises(ProbeSafetyError, match="THEME_REGISTRY_INCOMPLETE"):
        run(args(path, digest, tmp_path / "blocked", live=True), client=client)
    assert client.calls == 1

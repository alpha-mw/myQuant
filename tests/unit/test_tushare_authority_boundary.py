from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import stat
import sys
from typing import Any

import pytest

from quant_investor.intelligence_v2._core import canonical_bytes
from quant_investor.intelligence_v2.sources.tushare import (
    TushareContractError,
    build_endpoint_execution_plan,
    build_tushare_endpoint_policy,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "probe_tushare_10000_capabilities.py"
SPEC = importlib.util.spec_from_file_location("probe_tushare_10000_capabilities", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PROBE_SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE_SCRIPT
SPEC.loader.exec_module(PROBE_SCRIPT)

NOW = "2026-08-09T08:00:00Z"


def policy() -> dict[str, Any]:
    plan = build_endpoint_execution_plan(
        api_name="daily_basic",
        lane="FUNDAMENTAL",
        permission_class="POINTS",
        official_document_url="https://tushare.pro/document/2?doc_id=32",
        official_document_id="tushare.doc.32",
        document_observed_at="2026-08-09T07:59:00Z",
        documented_min_points=2000,
        strict_decimal_decode=True,
        expected_fields=["ts_code", "trade_date"],
        fixed_params={"trade_date": "20260807"},
        partition_dimensions=["trade_date"],
        ordered_expected_partition_keyset=["trade_date=20260807"],
        documented_row_limit=6000,
        max_attempts=1,
        retry_schedule=[0],
        empty_partition_rule="BASELINE_IDENTITY_EMPTY",
        completeness_proof="EXACT_PARTITION_AND_COUNT",
        limit_hit_action="BLOCK",
        planned_terminal_request_count=1,
        planned_max_network_attempts=1,
        created_at=NOW,
    )
    return build_tushare_endpoint_policy(created_at=NOW, endpoint_plans=[plan])


def write_policy(root: Path) -> tuple[Path, str]:
    path = root / "policy.json"
    raw = canonical_bytes(policy())
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def args(
    policy_path: Path, policy_sha: str, output_root: Path, *, live: bool
) -> argparse.Namespace:
    return argparse.Namespace(
        allow_live=live,
        output_root=str(output_root),
        policy_path=str(policy_path),
        policy_sha256=policy_sha,
        probed_at=NOW,
    )


def test_offline_probe_reads_no_token_calls_no_transport_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path, policy_sha = write_policy(tmp_path)
    output_root = tmp_path / "offline-output"
    canary = "TUSHARE_TOKEN_CANARY_MUST_NOT_LEAK"
    monkeypatch.setenv("TUSHARE_TOKEN", canary)
    monkeypatch.setattr(
        PROBE_SCRIPT,
        "probe_tushare_capabilities",
        lambda **_: (_ for _ in ()).throw(AssertionError("transport called")),
    )

    result = PROBE_SCRIPT.run(args(policy_path, policy_sha, output_root, live=False))

    assert result["status"] == "DRY_RUN_VALIDATED"
    assert result["planned_max_network_attempts"] == 1
    assert not output_root.exists()
    assert canary.encode() not in canonical_bytes(result)


def test_injected_live_probe_writes_only_exact_private_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path, policy_sha = write_policy(tmp_path)
    output_root = tmp_path / "live-output"
    monkeypatch.setattr(
        PROBE_SCRIPT,
        "probe_tushare_capabilities",
        lambda **_: {
            "request_receipts": (),
            "capability_receipts": (),
            "execution_receipts": (),
            "network_attempts": 0,
        },
    )

    result = PROBE_SCRIPT.run(args(policy_path, policy_sha, output_root, live=True))

    assert result["status"] == "LIVE_PROBE_RECORDED"
    assert stat.S_IMODE(output_root.stat().st_mode) == 0o700
    assert {path.name for path in output_root.iterdir()} == {
        "capability_receipts.json",
        "execution_receipts.json",
        "policy.json",
        "request_receipts.json",
        "summary.json",
    }
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600 and path.stat().st_nlink == 1
        for path in output_root.iterdir()
    )


def test_output_root_rejects_existing_symlink_and_casefold_collision(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(PROBE_SCRIPT.ProbeSafetyError):
        PROBE_SCRIPT._validate_new_output_root(existing, create=False)

    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(PROBE_SCRIPT.ProbeSafetyError):
        PROBE_SCRIPT._validate_new_output_root(link, create=False)

    collision = tmp_path / "ProbeOutput"
    collision.mkdir()
    with pytest.raises(PROBE_SCRIPT.ProbeSafetyError):
        PROBE_SCRIPT._validate_new_output_root(tmp_path / "probeoutput", create=False)


def test_token_parameter_and_public_surface_registration_are_forbidden() -> None:
    with pytest.raises(TushareContractError):
        build_endpoint_execution_plan(
            api_name="daily_basic",
            lane="FUNDAMENTAL",
            permission_class="POINTS",
            official_document_url="https://tushare.pro/document/2?doc_id=32",
            official_document_id="tushare.doc.32",
            document_observed_at=NOW,
            documented_min_points=2000,
            strict_decimal_decode=True,
            expected_fields=["ts_code"],
            fixed_params={"token": "SECRET"},
            partition_dimensions=["trade_date"],
            ordered_expected_partition_keyset=["trade_date=20260807"],
            documented_row_limit=6000,
            max_attempts=1,
            retry_schedule=[0],
            empty_partition_rule="BASELINE_IDENTITY_EMPTY",
            completeness_proof="EXACT_PARTITION_AND_COUNT",
            limit_hit_action="BLOCK",
            planned_terminal_request_count=1,
            planned_max_network_attempts=1,
            created_at=NOW,
        )

    forbidden_import = "intelligence_v2.sources.tushare"
    assert forbidden_import not in (ROOT / "quant_investor" / "__init__.py").read_text()
    assert forbidden_import not in (ROOT / "quant_investor" / "cli" / "main.py").read_text()


def test_interrupt_is_not_reclassified_as_transport_failure() -> None:
    endpoint_policy = policy()

    class InterruptingClient:
        def request(self, **_: Any) -> Any:
            raise KeyboardInterrupt

    from quant_investor.intelligence_v2.sources.tushare import probe_tushare_capabilities

    with pytest.raises(KeyboardInterrupt):
        probe_tushare_capabilities(
            policy=endpoint_policy,
            probed_at=NOW,
            client=InterruptingClient(),
        )

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_tushare_vip_fundamental_shadow.py"


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "tushare_fundamental_shadow_stable",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path: Path, *, allow_live: bool) -> argparse.Namespace:
    return argparse.Namespace(
        allow_live=allow_live,
        as_of="20260807",
        baseline_provider_manifest_path=None,
        baseline_provider_manifest_sha256=None,
        captured_at="2026-08-14T00:30:00Z" if allow_live else None,
        checkpoint_root=str(tmp_path / "checkpoint") if allow_live else None,
        comparison_policy_path=None,
        comparison_policy_sha256=None,
        execution_closure_path=str(tmp_path / "execution.json"),
        execution_closure_sha256="a" * 64,
        official_plan_path=None,
        official_plan_sha256=None,
        probe_observations_path=None,
        probe_observations_sha256=None,
        required_free_bytes=1,
        requests_per_second=8.0,
    )


def execution() -> dict[str, Any]:
    return {
        "closure_id": "b" * 64,
        "contract_sha256": "c" * 64,
        "request_plan": {
            "as_of": "20260807",
            "planned_max_network_attempts": 200,
            "planned_terminal_request_count": 100,
        },
    }


def inputs() -> tuple[dict[str, Any], None, None, None]:
    return execution(), None, None, None


def test_dry_run_is_pure_and_reports_stable_closure_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    monkeypatch.setattr(module, "_inputs", lambda _args: inputs())
    monkeypatch.setattr(
        module,
        "load_env_file",
        lambda: (_ for _ in ()).throw(AssertionError("credential file read")),
    )
    monkeypatch.setattr(
        module,
        "OfficialTushareHttpsClient",
        lambda **_: (_ for _ in ()).throw(AssertionError("provider constructed")),
    )
    monkeypatch.setenv("TUSHARE_TOKEN", "SECRET_CANARY_MUST_NOT_BE_READ")

    result = module.run(args(tmp_path, allow_live=False))

    assert result == {
        "as_of": "20260807",
        "execution_closure_id": "b" * 64,
        "execution_contract_sha256": "c" * 64,
        "planned_max_network_attempts": 200,
        "planned_terminal_request_count": 100,
        "status": "DRY_RUN_VALIDATED",
    }


def test_injected_shadow_capture_stops_before_evidence_or_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    values = args(tmp_path, allow_live=True)
    monkeypatch.setattr(module, "_inputs", lambda _args: inputs())
    monkeypatch.setattr(module, "_disk_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "load_env_file",
        lambda: (_ for _ in ()).throw(AssertionError("credential file read")),
    )
    client = object()
    captured: dict[str, Any] = {}

    def acquire(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "network_attempts": 100,
            "physical_receipts": (),
            "raw_tables": {},
            "status": "COMPLETE",
        }

    monkeypatch.setattr(module, "acquire_fundamental_partitions", acquire)

    result = module.run(values, client=client)

    assert captured["execution_closure"] == execution()
    assert captured["client"]._client is client
    assert result["actual_network_attempts"] == 100
    assert result["status"] == "SHADOW_CAPTURED"
    assert "generation_id" not in result
    assert "staging_root" not in result
    assert "provider_manifest_sha256" not in result


def test_injected_official_capture_reconciles_without_constructing_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    values = args(tmp_path, allow_live=True)
    official_plan = {
        "as_of": "20260807",
        "local_max_response_items": 20_000,
        "partition_plan_id": "d" * 64,
        "planned_max_network_attempts": 240,
        "planned_terminal_request_count": 120,
    }
    comparison = {"frozen": "validated-in-memory-only"}
    baseline = {"frozen": "provider-manifest"}
    baseline_bytes = b"{}"
    monkeypatch.setattr(
        module,
        "_inputs",
        lambda _args: (execution(), comparison, baseline, baseline_bytes),
    )
    monkeypatch.setattr(
        module,
        "_official_inputs",
        lambda _args, **_kwargs: (official_plan, [{"probe": True}]),
    )
    monkeypatch.setattr(module, "_disk_preflight", lambda *_args, **_kwargs: None)
    baseline_tables = {"daily_basic": pd.DataFrame()}
    monkeypatch.setattr(module, "_baseline_tables", lambda *_args, **_kwargs: baseline_tables)
    monkeypatch.setattr(
        module,
        "load_env_file",
        lambda: (_ for _ in ()).throw(AssertionError("credential file read")),
    )
    client = object()
    captured: dict[str, Any] = {}

    def acquire(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "receipt_network_attempts": 120,
            "status": "COMPLETE",
            "transport_calls": 120,
        }

    monkeypatch.setattr(module, "acquire_official_fundamental_partitions", acquire)

    result = module.run(values, client=client)

    assert captured["client"]._client is client
    assert captured["comparison_policy"] is comparison
    assert captured["baseline_tables"] is baseline_tables
    assert result["official_partition_plan_id"] == "d" * 64
    assert result["actual_network_attempts"] == 120
    assert result["status"] == "OFFICIAL_SHADOW_VALIDATED"


def test_market_shadow_import_boundary_has_no_legacy_runtime_dependency() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    retired_modules = tuple(
        ".".join(("quant_investor", name))
        for name in (
            "".join(("intelligence", "_v2")),
            "".join(("v17", "_v4_runtime")),
            "".join(("v17", "_v4_contract")),
            "".join(("v", "17")),
        )
    )
    for forbidden in retired_modules:
        assert forbidden not in source


@pytest.mark.parametrize("value", [True, 0.0, float("nan"), 8.1])
def test_request_pacer_rejects_nonexact_or_unsafe_rates(value: object) -> None:
    module = load_script()
    with pytest.raises(module.ShadowSafetyError, match="REQUEST_RATE_INVALID"):
        module._PacedTushareClient(object(), requests_per_second=value)

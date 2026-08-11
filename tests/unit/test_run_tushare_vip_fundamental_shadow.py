from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest


def load_script() -> Any:
    path = Path("scripts/run_tushare_vip_fundamental_shadow.py").resolve()
    spec = importlib.util.spec_from_file_location("vip_shadow_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path: Path, *, allow_live: bool) -> argparse.Namespace:
    return argparse.Namespace(
        allow_live=allow_live,
        as_of="20260807",
        baseline_provider_manifest_path=str(tmp_path / "baseline.json"),
        baseline_provider_manifest_sha256="a" * 64,
        captured_at="2026-08-09T12:00:00Z" if allow_live else None,
        checkpoint_root=str(tmp_path / "checkpoint") if allow_live else None,
        comparison_policy_path=str(tmp_path / "comparison.json"),
        comparison_policy_sha256="b" * 64,
        evidence_root=str(tmp_path / "evidence") if allow_live else None,
        execution_closure_path=str(tmp_path / "execution.json"),
        execution_closure_sha256="c" * 64,
        membership_path=str(tmp_path / "membership.parquet") if allow_live else None,
        membership_sha256="d" * 64 if allow_live else None,
        official_plan_path=None,
        official_plan_sha256=None,
        probe_observations_path=None,
        probe_observations_sha256=None,
        required_free_bytes=1,
        requests_per_second=8.0,
        run_id="vip-shadow-20260807" if allow_live else None,
        staging_data_root=str(tmp_path / "staging-data") if allow_live else None,
        staging_raw_root=str(tmp_path / "staging-raw") if allow_live else None,
        staging_reports_root=str(tmp_path / "staging-reports") if allow_live else None,
    )


def inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    plan = {
        "as_of": "20260807",
        "baseline_network_attempts": 600,
        "baseline_provider_manifest_ref": {"byte_sha256": "e" * 64},
        "planned_max_network_attempts": 200,
        "planned_terminal_request_count": 100,
    }
    return {"request_plan": plan}, {"policy": "validated"}, {"manifest": True}, b"{}"


def test_dry_run_does_not_read_token_checkpoint_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    monkeypatch.setattr(module, "_inputs", lambda _args: inputs())
    monkeypatch.setattr(
        module,
        "verify_package",
        lambda: {"semantic_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        module,
        "OfficialTushareHttpsClient",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("provider client")),
    )
    monkeypatch.setattr(
        module,
        "load_env_file",
        lambda: (_ for _ in ()).throw(AssertionError("credential file")),
    )
    monkeypatch.setattr(
        module,
        "_baseline_tables",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("checkpoint")),
    )
    monkeypatch.setenv("TUSHARE_TOKEN", "SECRET_CANARY_MUST_NOT_BE_READ")

    result = module.run(args(tmp_path, allow_live=False))
    assert result == {
        "as_of": "20260807",
        "package_sha256": "f" * 64,
        "planned_max_network_attempts": 200,
        "planned_terminal_request_count": 100,
        "status": "DRY_RUN_VALIDATED",
    }


def test_live_shadow_composes_once_without_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    values = args(tmp_path, allow_live=True)
    (tmp_path / "membership.parquet").write_bytes(b"membership")
    monkeypatch.setattr(module, "_inputs", lambda _args: inputs())
    monkeypatch.setattr(
        module,
        "verify_package",
        lambda: {"semantic_sha256": "f" * 64},
    )
    monkeypatch.setattr(module, "_disk_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_env_file", lambda: {})
    tables = {"daily_basic": pd.DataFrame({"ts_code": ["000001.SZ"]})}
    monkeypatch.setattr(module, "_baseline_tables", lambda *_args, **_kwargs: tables)
    client = object()
    monkeypatch.setattr(module, "OfficialTushareHttpsClient", lambda **_kwargs: client)
    calls: list[str] = []

    def acquire(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["client"] is client
        calls.append("acquire")
        return {
            "network_attempts": 80,
            "physical_receipts": ("receipt",),
            "raw_tables": tables,
            "status": "COMPLETE",
        }

    monkeypatch.setattr(module, "acquire_fundamental_vip_v4", acquire)
    monkeypatch.setattr(
        module,
        "build_logical_coverages_from_shadow_v4",
        lambda **_kwargs: calls.append("coverage") or ("coverage",),
    )
    monkeypatch.setattr(
        module,
        "derive_fundamental_shadow_v4",
        lambda **_kwargs: calls.append("derive")
        or {
            "derived_fingerprints": {"fingerprints": True},
            "vip_derived_tables": {"derived": pd.DataFrame()},
        },
    )
    bundle = {"status": "PASSED"}
    monkeypatch.setattr(
        module,
        "build_fundamental_shadow_bundle_v4",
        lambda **_kwargs: calls.append("bundle") or bundle,
    )
    monkeypatch.setattr(
        module,
        "write_fundamental_shadow_bundle_v4",
        lambda **_kwargs: calls.append("evidence")
        or {
            "fileset_sha256": "1" * 64,
            "output_root": values.evidence_root,
        },
    )
    monkeypatch.setattr(
        module,
        "materialize_fundamental_v4_staging_generation",
        lambda **_kwargs: calls.append("staging")
        or {
            "generation_id": values.run_id,
            "provider_manifest_sha256": "2" * 64,
        },
    )

    result = module.run(values)
    assert calls == ["acquire", "coverage", "derive", "bundle", "evidence", "staging"]
    assert result["actual_network_attempts"] == 80
    assert result["status"] == "STAGING_READY"


def test_official_partition_mode_stops_after_exact_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    values = args(tmp_path, allow_live=True)
    values.official_plan_path = str(tmp_path / "official-plan.json")
    values.official_plan_sha256 = "1" * 64
    values.probe_observations_path = str(tmp_path / "probes.json")
    values.probe_observations_sha256 = "2" * 64
    official_plan = {
        "as_of": "20260807",
        "local_max_response_items": 20_000,
        "partition_plan_id": "3" * 64,
        "planned_max_network_attempts": 200,
        "planned_terminal_request_count": 100,
    }
    monkeypatch.setattr(module, "_inputs", lambda _args: inputs())
    monkeypatch.setattr(
        module,
        "_official_inputs",
        lambda _args, **_kwargs: (official_plan, [{"probe": True}]),
    )
    monkeypatch.setattr(module, "verify_package", lambda: {"semantic_sha256": "f" * 64})
    monkeypatch.setattr(module, "_disk_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_env_file", lambda: {})
    tables = {"daily_basic": pd.DataFrame({"ts_code": ["000001.SZ"]})}
    monkeypatch.setattr(module, "_baseline_tables", lambda *_args, **_kwargs: tables)
    client = object()
    client_kwargs: dict[str, Any] = {}

    def build_client(**kwargs: Any) -> object:
        client_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(module, "OfficialTushareHttpsClient", build_client)
    calls: list[str] = []

    def acquire(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["official_plan"] is official_plan
        assert kwargs["client"]._client is client
        calls.append("official-acquire")
        return {
            "receipt_network_attempts": 100,
            "status": "COMPLETE",
            "transport_calls": 100,
        }

    monkeypatch.setattr(
        module,
        "acquire_official_partition_fundamental_vip_v4",
        acquire,
    )
    monkeypatch.setattr(
        module,
        "build_logical_coverages_from_shadow_v4",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy continuation")),
    )

    result = module.run(values)

    assert client_kwargs == {
        "max_response_items": 20_000,
        "strict_decimal_decode": True,
    }
    assert calls == ["official-acquire"]
    assert result["actual_network_attempts"] == 100
    assert result["official_partition_plan_id"] == "3" * 64
    assert result["requests_per_second"] == 8.0
    assert result["status"] == "OFFICIAL_PARTITION_SHADOW_VALIDATED"


def test_official_partition_pacer_enforces_eight_requests_per_second() -> None:
    module = load_script()
    now = [10.0]
    sleeps: list[float] = []
    calls: list[int] = []

    class Client:
        def request(self, **_kwargs: Any) -> int:
            calls.append(len(calls))
            return len(calls)

    def clock() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = module._PacedTushareClient(
        Client(), requests_per_second=8.0, clock=clock, sleeper=sleeper
    )
    assert client.request() == 1
    assert client.request() == 2
    now[0] += 0.25
    assert client.request() == 3
    assert calls == [0, 1, 2]
    assert sleeps == [0.125]

    with pytest.raises(module.ShadowSafetyError, match="REQUEST_RATE_INVALID"):
        module._PacedTushareClient(Client(), requests_per_second=8.1)


def test_official_partition_dry_run_does_not_construct_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    values = args(tmp_path, allow_live=False)
    values.official_plan_path = str(tmp_path / "official-plan.json")
    values.official_plan_sha256 = "1" * 64
    values.probe_observations_path = str(tmp_path / "probes.json")
    values.probe_observations_sha256 = "2" * 64
    official_plan = {
        "as_of": "20260807",
        "partition_plan_id": "3" * 64,
        "planned_max_network_attempts": 200,
        "planned_terminal_request_count": 100,
    }
    monkeypatch.setattr(module, "_inputs", lambda _args: inputs())
    monkeypatch.setattr(
        module,
        "_official_inputs",
        lambda _args, **_kwargs: (official_plan, [{"probe": True}]),
    )
    monkeypatch.setattr(module, "verify_package", lambda: {"semantic_sha256": "f" * 64})
    monkeypatch.setattr(
        module,
        "OfficialTushareHttpsClient",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("provider client")),
    )
    monkeypatch.setattr(
        module,
        "load_env_file",
        lambda: (_ for _ in ()).throw(AssertionError("credential file")),
    )

    result = module.run(values)

    assert result["official_partition_plan_id"] == "3" * 64
    assert result["requests_per_second"] == 8.0
    assert result["status"] == "DRY_RUN_VALIDATED"


def test_incomplete_acquisition_preserves_checkpoint_and_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    values = args(tmp_path, allow_live=True)
    (tmp_path / "membership.parquet").write_bytes(b"membership")
    monkeypatch.setattr(module, "_inputs", lambda _args: inputs())
    monkeypatch.setattr(module, "verify_package", lambda: {"semantic_sha256": "f" * 64})
    monkeypatch.setattr(module, "_disk_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "load_env_file", lambda: {})
    monkeypatch.setattr(module, "_baseline_tables", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        module,
        "OfficialTushareHttpsClient",
        lambda **_kwargs: SimpleNamespace(request=None),
    )
    monkeypatch.setattr(
        module,
        "acquire_fundamental_vip_v4",
        lambda **_kwargs: {
            "network_attempts": 12,
            "physical_receipts": (),
            "raw_tables": {},
            "status": "BLOCKED",
        },
    )
    monkeypatch.setattr(
        module,
        "build_logical_coverages_from_shadow_v4",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("continued")),
    )

    result = module.run(values)
    assert result["actual_network_attempts"] == 12
    assert result["status"] == "ACQUISITION_BLOCKED"


def test_baseline_manifest_binds_actual_provider_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    raw = b"{}"
    plan = {
        "as_of": "20260807",
        "baseline_network_attempts": 601,
        "baseline_provider_manifest_ref": {"byte_sha256": hashlib.sha256(raw).hexdigest()},
        "symbols": ["000001.SZ"],
    }
    manifest = {
        "schema_version": "myquant-fundamental-provider-manifest.v3",
        "strict_pit_as_of": "20260807",
        "provider_calls_attempted": 600,
        "symbols_requested": 1,
    }
    monkeypatch.setattr(
        "quant_investor.market.fundamental_generation._capture_provider_checkpoint_v3",
        lambda _manifest: SimpleNamespace(tables={}),
    )
    with pytest.raises(module.ShadowSafetyError, match="BASELINE_MANIFEST_MISMATCH"):
        module._baseline_tables(manifest, plan=plan, manifest_bytes=raw)


def test_legacy_provider_manifest_uses_legacy_canonical_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    manifest = {
        "provider_calls_attempted": 33012,
        "schema_version": "myquant-fundamental-provider-manifest.v3",
        "symbol_table_outcomes": [{"status": "success"}],
    }
    raw = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    path = tmp_path / "provider-manifest.json"
    path.write_bytes(raw)
    monkeypatch.setattr(
        module,
        "canonical_bytes",
        lambda _value: (_ for _ in ()).throw(AssertionError("v2 artifact cap")),
    )

    loaded, loaded_raw = module._load_legacy_provider_manifest(
        path,
        hashlib.sha256(raw).hexdigest(),
    )

    assert loaded == manifest
    assert loaded_raw == raw

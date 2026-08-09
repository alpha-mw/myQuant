from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
        required_free_bytes=1,
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

#!/usr/bin/env python3
"""Validate or acquire Fundamental VIP v4 shadow data without promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Any

from quant_investor.intelligence_v2._core import canonical_bytes
from quant_investor.intelligence_v2.package import verify_package
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    acquire_official_partition_fundamental_vip_v4,
    acquire_fundamental_vip_v4,
    build_fundamental_shadow_bundle_v4,
    build_logical_coverages_from_shadow_v4,
    derive_fundamental_shadow_v4,
    materialize_fundamental_v4_staging_generation,
    validate_fundamental_comparison_policy,
    validate_fundamental_execution_closure_v4,
    validate_official_partition_execution_plan,
    write_fundamental_shadow_bundle_v4,
)
from quant_investor.env_loading import load_env_file
from quant_investor.market.fundamental_provider_contract import frame_fingerprint
from quant_investor.v17_v4_runtime.tushare_https import OfficialTushareHttpsClient


class ShadowSafetyError(RuntimeError):
    """Static, secret-free shadow orchestration failure."""


class _PacedTushareClient:
    """Keep official shadow calls within the validated v3 request rate."""

    def __init__(
        self,
        client: Any,
        *,
        requests_per_second: float,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        if not 0 < requests_per_second <= 8.0:
            raise ShadowSafetyError("VIP_SHADOW_REQUEST_RATE_INVALID")
        self._client = client
        self._minimum_interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleeper = sleeper
        self._last_started_at: float | None = None

    def request(self, **kwargs: Any) -> Any:
        now = self._clock()
        if self._last_started_at is not None:
            remaining = self._minimum_interval - (now - self._last_started_at)
            if remaining > 0:
                self._sleeper(remaining)
                now = self._clock()
        self._last_started_at = now
        return self._client.request(**kwargs)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ShadowSafetyError("VIP_SHADOW_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _read_exact_json(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ShadowSafetyError("VIP_SHADOW_INPUT_PATH_INVALID")
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ShadowSafetyError("VIP_SHADOW_INPUT_PATH_UNSAFE")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ShadowSafetyError("VIP_SHADOW_INPUT_SHA_MISMATCH")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ShadowSafetyError("VIP_SHADOW_NONFINITE_JSON")
            ),
        )
    except ShadowSafetyError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ShadowSafetyError("VIP_SHADOW_INPUT_INVALID") from exc
    if type(value) is not dict:
        raise ShadowSafetyError("VIP_SHADOW_INPUT_NOT_OBJECT")
    return value, raw


def _load_exact_json(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    value, raw = _read_exact_json(path, expected_sha256)
    if canonical_bytes(value) != raw:
        raise ShadowSafetyError("VIP_SHADOW_INPUT_NOT_CANONICAL")
    return value, raw


def _load_exact_array(path: Path, expected_sha256: str) -> list[Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ShadowSafetyError("VIP_SHADOW_INPUT_PATH_INVALID")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ShadowSafetyError("VIP_SHADOW_INPUT_SHA_MISMATCH")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ShadowSafetyError("VIP_SHADOW_NONFINITE_JSON")
            ),
        )
    except ShadowSafetyError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ShadowSafetyError("VIP_SHADOW_INPUT_INVALID") from exc
    if type(value) is not list or canonical_bytes(value) != raw:
        raise ShadowSafetyError("VIP_SHADOW_INPUT_NOT_CANONICAL")
    return value


def _load_legacy_provider_manifest(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Validate legacy v3 canonical bytes without applying the v2 8 MiB cap."""

    from quant_investor.market.fundamental_generation import _json_bytes

    value, raw = _read_exact_json(path, expected_sha256)
    if (
        value.get("schema_version") != "myquant-fundamental-provider-manifest.v3"
        or _json_bytes(value) != raw
    ):
        raise ShadowSafetyError("VIP_SHADOW_LEGACY_BASELINE_NOT_CANONICAL")
    return value, raw


def _new_absolute_path(value: str, *, label: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(character in value for character in "*?[]")
        or path.exists()
        or path.is_symlink()
    ):
        raise ShadowSafetyError(f"VIP_SHADOW_{label}_PATH_INVALID")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ShadowSafetyError(f"VIP_SHADOW_{label}_PARENT_INVALID")
    return path


def _checkpoint_path(value: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(character in value for character in "*?[]")
        or path.is_symlink()
    ):
        raise ShadowSafetyError("VIP_SHADOW_CHECKPOINT_PATH_INVALID")
    if path.exists():
        metadata = os.lstat(path)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
        ):
            raise ShadowSafetyError("VIP_SHADOW_CHECKPOINT_PATH_UNSAFE")
    elif not path.parent.is_dir() or path.parent.is_symlink():
        raise ShadowSafetyError("VIP_SHADOW_CHECKPOINT_PARENT_INVALID")
    return path


def _disk_preflight(paths: list[Path], *, required_free_bytes: int) -> None:
    if type(required_free_bytes) is not int or required_free_bytes < 1:
        raise ShadowSafetyError("VIP_SHADOW_DISK_REQUIREMENT_INVALID")
    for path in paths:
        if shutil.disk_usage(path.parent).free < required_free_bytes:
            raise ShadowSafetyError("VIP_SHADOW_DISK_SPACE_INSUFFICIENT")


def _baseline_tables(
    manifest: dict[str, Any],
    *,
    plan: dict[str, Any],
    manifest_bytes: bytes,
) -> dict[str, Any]:
    from quant_investor.market.fundamental_generation import (
        _capture_provider_checkpoint_v3,
    )

    manifest_ref = plan["baseline_provider_manifest_ref"]
    if manifest_ref["byte_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ShadowSafetyError("VIP_SHADOW_BASELINE_REF_MISMATCH")
    if (
        manifest.get("schema_version") != "myquant-fundamental-provider-manifest.v3"
        or manifest.get("strict_pit_as_of") != plan["as_of"]
        or manifest.get("provider_calls_attempted") != plan["baseline_network_attempts"]
        or manifest.get("symbols_requested") != len(plan["symbols"])
    ):
        raise ShadowSafetyError("VIP_SHADOW_BASELINE_MANIFEST_MISMATCH")
    captured = _capture_provider_checkpoint_v3(manifest)
    return captured.tables


def _inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict, bytes]:
    execution_raw, _raw = _load_exact_json(
        Path(args.execution_closure_path),
        args.execution_closure_sha256,
    )
    execution = validate_fundamental_execution_closure_v4(execution_raw)
    comparison_raw, _raw = _load_exact_json(
        Path(args.comparison_policy_path),
        args.comparison_policy_sha256,
    )
    comparison = validate_fundamental_comparison_policy(comparison_raw)
    baseline, baseline_bytes = _load_legacy_provider_manifest(
        Path(args.baseline_provider_manifest_path),
        args.baseline_provider_manifest_sha256,
    )
    if execution["request_plan"]["as_of"] != args.as_of:
        raise ShadowSafetyError("VIP_SHADOW_ASOF_MISMATCH")
    return execution, comparison, baseline, baseline_bytes


def _official_inputs(
    args: argparse.Namespace,
    *,
    execution: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    names = (
        "official_plan_path",
        "official_plan_sha256",
        "probe_observations_path",
        "probe_observations_sha256",
    )
    values = [getattr(args, name, None) for name in names]
    if not any(values):
        return None
    if not all(values):
        raise ShadowSafetyError("VIP_SHADOW_OFFICIAL_INPUT_INCOMPLETE")
    observations = _load_exact_array(
        Path(args.probe_observations_path),
        args.probe_observations_sha256,
    )
    if any(type(row) is not dict for row in observations):
        raise ShadowSafetyError("VIP_SHADOW_PROBE_OBSERVATIONS_INVALID")
    plan_raw, _raw = _load_exact_json(
        Path(args.official_plan_path),
        args.official_plan_sha256,
    )
    plan = validate_official_partition_execution_plan(
        plan_raw,
        source_execution_closure=execution,
        probe_observations=observations,
    )
    return plan, observations


def run(args: argparse.Namespace) -> dict[str, Any]:
    execution, comparison, baseline_manifest, baseline_bytes = _inputs(args)
    legacy_plan = execution["request_plan"]
    official = _official_inputs(args, execution=execution)
    plan = legacy_plan if official is None else official[0]
    package = verify_package()
    summary: dict[str, Any] = {
        "as_of": plan["as_of"],
        "package_sha256": package["semantic_sha256"],
        "planned_max_network_attempts": plan["planned_max_network_attempts"],
        "planned_terminal_request_count": plan["planned_terminal_request_count"],
        "status": "DRY_RUN_VALIDATED",
    }
    if official is not None:
        summary["official_partition_plan_id"] = plan["partition_plan_id"]
        summary["requests_per_second"] = args.requests_per_second
    if not args.allow_live:
        return summary
    required = (
        ("captured_at", "checkpoint_root")
        if official is not None
        else (
            "captured_at",
            "checkpoint_root",
            "evidence_root",
            "membership_path",
            "membership_sha256",
            "run_id",
            "staging_data_root",
            "staging_raw_root",
            "staging_reports_root",
        )
    )
    if any(not getattr(args, name) for name in required):
        raise ShadowSafetyError("VIP_SHADOW_LIVE_ARGUMENT_MISSING")
    load_env_file()
    checkpoint = _checkpoint_path(args.checkpoint_root)
    if official is not None:
        _disk_preflight([checkpoint], required_free_bytes=args.required_free_bytes)
        baseline_tables = _baseline_tables(
            baseline_manifest,
            plan=legacy_plan,
            manifest_bytes=baseline_bytes,
        )
        acquisition = acquire_official_partition_fundamental_vip_v4(
            official_plan=plan,
            source_execution_closure=execution,
            probe_observations=official[1],
            baseline_tables=baseline_tables,
            baseline_table_fingerprints={
                table: frame_fingerprint(frame) for table, frame in baseline_tables.items()
            },
            comparison_policy=comparison,
            client=_PacedTushareClient(
                OfficialTushareHttpsClient(
                    strict_decimal_decode=True,
                    max_response_items=plan["local_max_response_items"],
                ),
                requests_per_second=args.requests_per_second,
            ),
            captured_at=args.captured_at,
            checkpoint_root=checkpoint,
        )
        summary.update(
            {
                "actual_network_attempts": acquisition["transport_calls"],
                "receipt_network_attempts": acquisition["receipt_network_attempts"],
                "status": (
                    "OFFICIAL_PARTITION_SHADOW_VALIDATED"
                    if acquisition["status"] == "COMPLETE"
                    else acquisition["status"]
                ),
            }
        )
        return summary
    evidence = _new_absolute_path(args.evidence_root, label="EVIDENCE")
    staging_paths = [
        _new_absolute_path(args.staging_data_root, label="STAGING_DATA"),
        _new_absolute_path(args.staging_raw_root, label="STAGING_RAW"),
        _new_absolute_path(args.staging_reports_root, label="STAGING_REPORTS"),
    ]
    _disk_preflight(
        [checkpoint, evidence, *staging_paths],
        required_free_bytes=args.required_free_bytes,
    )
    baseline_tables = _baseline_tables(
        baseline_manifest,
        plan=plan,
        manifest_bytes=baseline_bytes,
    )
    acquisition = acquire_fundamental_vip_v4(
        execution_closure=execution,
        client=OfficialTushareHttpsClient(strict_decimal_decode=True),
        captured_at=args.captured_at,
        checkpoint_root=checkpoint,
    )
    summary["actual_network_attempts"] = acquisition["network_attempts"]
    if acquisition["status"] != "COMPLETE":
        summary["status"] = "ACQUISITION_BLOCKED"
        return summary
    logical = build_logical_coverages_from_shadow_v4(
        execution_closure=execution,
        physical_receipts=acquisition["physical_receipts"],
        vip_tables=acquisition["raw_tables"],
        assessed_at=args.captured_at,
    )
    derived = derive_fundamental_shadow_v4(
        execution_closure=execution,
        baseline_tables=baseline_tables,
        vip_tables=acquisition["raw_tables"],
        membership_path=args.membership_path,
        membership_sha256=args.membership_sha256,
        run_id=args.run_id,
        derivation_timestamp=args.captured_at,
    )
    bundle = build_fundamental_shadow_bundle_v4(
        execution_closure=execution,
        physical_receipts=acquisition["physical_receipts"],
        logical_coverages=logical,
        baseline_tables=baseline_tables,
        vip_tables=acquisition["raw_tables"],
        comparison_policy=comparison,
        derived_fingerprints=derived["derived_fingerprints"],
        assembled_at=args.captured_at,
    )
    written = write_fundamental_shadow_bundle_v4(bundle=bundle, output_root=evidence)
    summary.update(
        {
            "evidence_fileset_sha256": written["fileset_sha256"],
            "evidence_root": written["output_root"],
        }
    )
    if bundle["status"] != "PASSED":
        summary["status"] = "RECONCILIATION_BLOCKED"
        return summary
    staging = materialize_fundamental_v4_staging_generation(
        execution_closure=execution,
        bundle=bundle,
        vip_tables=acquisition["raw_tables"],
        vip_derived_tables=derived["vip_derived_tables"],
        data_root=staging_paths[0],
        raw_snapshot_root=staging_paths[1],
        reports_root=staging_paths[2],
        run_id=args.run_id,
    )
    summary.update(
        {
            "generation_id": staging["generation_id"],
            "provider_manifest_sha256": staging["provider_manifest_sha256"],
            "staging_root": str(staging_paths[0]),
            "status": "STAGING_READY",
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--execution-closure-path", required=True)
    parser.add_argument("--execution-closure-sha256", required=True)
    parser.add_argument("--official-plan-path")
    parser.add_argument("--official-plan-sha256")
    parser.add_argument("--probe-observations-path")
    parser.add_argument("--probe-observations-sha256")
    parser.add_argument("--comparison-policy-path", required=True)
    parser.add_argument("--comparison-policy-sha256", required=True)
    parser.add_argument("--baseline-provider-manifest-path", required=True)
    parser.add_argument("--baseline-provider-manifest-sha256", required=True)
    parser.add_argument("--captured-at")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--evidence-root")
    parser.add_argument("--membership-path")
    parser.add_argument("--membership-sha256")
    parser.add_argument("--required-free-bytes", type=int, default=1)
    parser.add_argument("--requests-per-second", type=float, default=8.0)
    parser.add_argument("--run-id")
    parser.add_argument("--staging-data-root")
    parser.add_argument("--staging-raw-root")
    parser.add_argument("--staging-reports-root")
    return parser.parse_args()


def main() -> int:
    try:
        result = run(parse_args())
    except Exception:
        print('{"status":"VIP_SHADOW_BLOCKED"}')
        return 2
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

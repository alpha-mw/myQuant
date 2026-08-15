#!/usr/bin/env python3
"""Validate or acquire inactive Fundamental Tushare shadow data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import time
from typing import Any

from quant_investor.env_loading import load_env_file
from quant_investor.market.fundamental_provider_contract import frame_fingerprint
from quant_investor.market.fundamental_provider_evidence import (
    canonical_provider_json_bytes,
    validate_fundamental_comparison_policy,
)
from quant_investor.market.tushare._core import canonical_bytes
from quant_investor.market.tushare.fundamental import (
    acquire_fundamental_partitions,
    acquire_official_fundamental_partitions,
    validate_fundamental_execution_closure,
    validate_official_partition_plan,
)
from quant_investor.market.tushare_transport import OfficialTushareHttpsClient


class ShadowSafetyError(RuntimeError):
    """Static, secret-free shadow orchestration failure."""

    code = "FUNDAMENTAL_SHADOW_BLOCKED"
    exit_code = 2
    public_fields: dict[str, Any] = {}


class _PacedTushareClient:
    """Keep official shadow calls within the validated request rate."""

    def __init__(
        self,
        client: Any,
        *,
        requests_per_second: float,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        if (
            type(requests_per_second) not in {int, float}
            or not math.isfinite(float(requests_per_second))
            or not 0 < requests_per_second <= 8.0
        ):
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
    if type(value) is not list or canonical_bytes(value) != raw:
        raise ShadowSafetyError("VIP_SHADOW_INPUT_NOT_CANONICAL")
    return value


def _load_frozen_provider_manifest(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    """Read a frozen pre-cutover manifest without re-emitting its tokens."""

    from quant_investor.market.fundamental_generation import _json_bytes

    value, raw = _read_exact_json(path, expected_sha256)
    if (
        value.get("schema_version") != "myquant-fundamental-provider-manifest.v3"
        or _json_bytes(value) != raw
    ):
        raise ShadowSafetyError("VIP_SHADOW_LEGACY_BASELINE_NOT_CANONICAL")
    return value, raw


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
            or path.resolve(strict=True) != path
        ):
            raise ShadowSafetyError("VIP_SHADOW_CHECKPOINT_PATH_UNSAFE")
    elif (
        not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
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


def _inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    bytes | None,
]:
    execution_raw, _raw = _load_exact_json(
        Path(args.execution_closure_path),
        args.execution_closure_sha256,
    )
    execution = validate_fundamental_execution_closure(execution_raw)
    comparison_path = getattr(args, "comparison_policy_path", None)
    comparison_sha256 = getattr(args, "comparison_policy_sha256", None)
    if bool(comparison_path) != bool(comparison_sha256):
        raise ShadowSafetyError("VIP_SHADOW_COMPARISON_INPUT_INCOMPLETE")
    comparison = None
    if comparison_path:
        comparison_raw, comparison_bytes = _read_exact_json(
            Path(comparison_path),
            comparison_sha256,
        )
        if canonical_provider_json_bytes(comparison_raw) != comparison_bytes:
            raise ShadowSafetyError("VIP_SHADOW_COMPARISON_INPUT_NOT_CANONICAL")
        comparison = validate_fundamental_comparison_policy(comparison_raw)
    baseline_path = getattr(args, "baseline_provider_manifest_path", None)
    baseline_sha256 = getattr(args, "baseline_provider_manifest_sha256", None)
    if bool(baseline_path) != bool(baseline_sha256):
        raise ShadowSafetyError("VIP_SHADOW_BASELINE_INPUT_INCOMPLETE")
    baseline = None
    baseline_bytes = None
    if baseline_path:
        baseline, baseline_bytes = _load_frozen_provider_manifest(
            Path(baseline_path),
            baseline_sha256,
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
    plan = validate_official_partition_plan(
        plan_raw,
        source_execution_closure=execution,
        probe_observations=observations,
    )
    return plan, observations


def run(
    args: argparse.Namespace,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    execution, comparison, baseline_manifest, baseline_bytes = _inputs(args)
    source_plan = execution["request_plan"]
    official = _official_inputs(args, execution=execution)
    plan = source_plan if official is None else official[0]
    summary: dict[str, Any] = {
        "as_of": plan["as_of"],
        "execution_closure_id": execution["closure_id"],
        "execution_contract_sha256": execution["contract_sha256"],
        "planned_max_network_attempts": plan["planned_max_network_attempts"],
        "planned_terminal_request_count": plan["planned_terminal_request_count"],
        "status": "DRY_RUN_VALIDATED",
    }
    if official is not None:
        summary["official_partition_plan_id"] = plan["partition_plan_id"]
        summary["requests_per_second"] = args.requests_per_second
    if not args.allow_live:
        return summary
    required = ("captured_at", "checkpoint_root")
    if any(not getattr(args, name) for name in required):
        raise ShadowSafetyError("VIP_SHADOW_LIVE_ARGUMENT_MISSING")
    checkpoint = _checkpoint_path(args.checkpoint_root)
    _disk_preflight([checkpoint], required_free_bytes=args.required_free_bytes)
    if client is None:
        load_env_file()
        client = OfficialTushareHttpsClient(
            strict_decimal_decode=True,
            max_response_items=(
                plan["local_max_response_items"] if official is not None else 100_000
            ),
        )
    paced_client = _PacedTushareClient(
        client,
        requests_per_second=args.requests_per_second,
    )
    if official is not None:
        if comparison is None or baseline_manifest is None or baseline_bytes is None:
            raise ShadowSafetyError("VIP_SHADOW_OFFICIAL_REPLAY_INPUT_MISSING")
        baseline_tables = _baseline_tables(
            baseline_manifest,
            plan=source_plan,
            manifest_bytes=baseline_bytes,
        )
        acquisition = acquire_official_fundamental_partitions(
            official_plan=plan,
            source_execution_closure=execution,
            probe_observations=official[1],
            baseline_tables=baseline_tables,
            baseline_table_fingerprints={
                table: frame_fingerprint(frame) for table, frame in baseline_tables.items()
            },
            comparison_policy=comparison,
            client=paced_client,
            captured_at=args.captured_at,
            checkpoint_root=checkpoint,
        )
        summary.update(
            {
                "actual_network_attempts": acquisition["transport_calls"],
                "receipt_network_attempts": acquisition["receipt_network_attempts"],
                "status": (
                    "OFFICIAL_SHADOW_VALIDATED"
                    if acquisition["status"] == "COMPLETE"
                    else acquisition["status"]
                ),
            }
        )
        return summary
    acquisition = acquire_fundamental_partitions(
        execution_closure=execution,
        client=paced_client,
        captured_at=args.captured_at,
        checkpoint_root=checkpoint,
    )
    summary["actual_network_attempts"] = acquisition["network_attempts"]
    if acquisition["status"] != "COMPLETE":
        summary["status"] = "ACQUISITION_BLOCKED"
        return summary
    summary["status"] = "SHADOW_CAPTURED"
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
    parser.add_argument("--comparison-policy-path")
    parser.add_argument("--comparison-policy-sha256")
    parser.add_argument("--baseline-provider-manifest-path")
    parser.add_argument("--baseline-provider-manifest-sha256")
    parser.add_argument("--captured-at")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--required-free-bytes", type=int, default=1)
    parser.add_argument("--requests-per-second", type=float, default=8.0)
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

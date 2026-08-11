#!/usr/bin/env python3
"""Capture one sealed DC or TDX Theme provider plan without concurrency or retry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

from quant_investor.intelligence_v2._core import canonical_bytes
from quant_investor.intelligence_v2.sources.tushare import (
    TushareContractError,
    build_theme_provider_capture,
    capture_theme_partition,
    validate_theme_provider_capture,
    validate_theme_provider_execution_plan,
)
from quant_investor.v17_v4_runtime.tushare_https import OfficialTushareHttpsClient
from scripts.probe_tushare_10000_capabilities import (
    ProbeSafetyError,
    _unique_object,
    _validate_new_output_root,
    _write_exact,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_exact(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ProbeSafetyError("THEME_PLAN_PATH_INVALID")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ProbeSafetyError("THEME_PLAN_SHA_MISMATCH")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProbeSafetyError("THEME_PLAN_INVALID") from exc
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise ProbeSafetyError("THEME_PLAN_NOT_CANONICAL")
    return validate_theme_provider_execution_plan(value)


def _validate_resume_root(path: Path, plan: dict[str, Any]) -> Path:
    observed = path.lstat()
    if (
        not path.is_absolute()
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise ProbeSafetyError("THEME_RESUME_ROOT_INVALID")
    plan_path = path / "plan.json"
    if plan_path.is_symlink() or plan_path.read_bytes() != canonical_bytes(plan):
        raise ProbeSafetyError("THEME_RESUME_PLAN_MISMATCH")
    partitions = path / "partitions"
    partition_stat = partitions.lstat()
    if (
        not stat.S_ISDIR(partition_stat.st_mode)
        or stat.S_ISLNK(partition_stat.st_mode)
        or stat.S_IMODE(partition_stat.st_mode) != 0o700
    ):
        raise ProbeSafetyError("THEME_RESUME_PARTITIONS_INVALID")
    return partitions


def _prepare_root(path: Path, *, resume: bool, plan: dict[str, Any]) -> Path:
    if resume:
        return _validate_resume_root(path, plan)
    _validate_new_output_root(path, create=True)
    partitions = path / "partitions"
    partitions.mkdir(mode=0o700)
    os.chmod(partitions, 0o700)
    _write_exact(path / "plan.json", plan)
    return partitions


def _load_partition(path: Path) -> dict[str, Any]:
    observed = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        raise ProbeSafetyError("THEME_PARTITION_PATH_INVALID")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProbeSafetyError("THEME_PARTITION_INVALID") from exc
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise ProbeSafetyError("THEME_PARTITION_NOT_CANONICAL")
    return value


def _finish_capture(
    *,
    output_root: Path,
    plan: dict[str, Any],
    partitions: list[dict[str, Any]],
) -> dict[str, Any]:
    capture_path = output_root / "capture.json"
    if not capture_path.exists():
        capture = build_theme_provider_capture(
            plan=plan,
            partition_documents=partitions,
            completed_at=_now(),
        )
        _write_exact(capture_path, capture)
        return capture
    raw = capture_path.read_bytes()
    try:
        capture_value = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProbeSafetyError("THEME_CAPTURE_INVALID") from exc
    capture = validate_theme_provider_capture(
        capture_value,
        plan=plan,
        partition_documents=partitions,
    )
    if canonical_bytes(capture) != raw:
        raise ProbeSafetyError("THEME_CAPTURE_NOT_CANONICAL")
    return capture


def run(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    plan = _load_exact(Path(args.plan_path), args.plan_sha256)
    output_root = Path(args.output_root)
    planned = 1 + len(plan["company_keyset"])
    if not args.allow_live:
        _validate_new_output_root(output_root, create=False)
        return {
            "network_attempts": 0,
            "plan_id": plan["plan_id"],
            "planned_partitions": planned,
            "provider": plan["provider"],
            "status": "DRY_RUN_VALIDATED",
        }
    partitions_root = _prepare_root(output_root, resume=bool(args.resume), plan=plan)
    transport = client or OfficialTushareHttpsClient(strict_decimal_decode=True)
    partitions = []
    network_attempts = 0
    for ordinal in range(planned):
        path = partitions_root / f"{ordinal:05d}.json"
        if path.exists():
            partition = _load_partition(path)
        else:
            partition = capture_theme_partition(
                plan=plan,
                partition_ordinal=ordinal,
                captured_at=_now(),
                client=transport,
            )
            network_attempts += 1
            _write_exact(path, partition)
        if partition["partition_ordinal"] != ordinal:
            raise ProbeSafetyError("THEME_RESUME_KEYSET_MISMATCH")
        partitions.append(partition)
        if ordinal == 0 and partition["status"] == "INCOMPLETE":
            raise ProbeSafetyError("THEME_REGISTRY_INCOMPLETE")
        if ordinal and ordinal % 100 == 0:
            print(f"theme progress {ordinal}/{planned - 1}", file=sys.stderr, flush=True)
    capture = _finish_capture(output_root=output_root, plan=plan, partitions=partitions)
    summary = {
        "capture_id": capture["capture_id"],
        "completed_partitions": planned,
        "incomplete_partitions": capture["incomplete_partition_count"],
        "network_attempts": network_attempts,
        "plan_id": plan["plan_id"],
        "provider": plan["provider"],
        "status": capture["status"],
    }
    if not (output_root / "summary.json").exists():
        _write_exact(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        summary = run(parse_args())
    except (ProbeSafetyError, TushareContractError) as exc:
        print(json.dumps({"blocker": str(exc), "status": "THEME_CAPTURE_BLOCKED"}))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

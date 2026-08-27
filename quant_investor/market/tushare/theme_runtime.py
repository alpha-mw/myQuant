"""Owner-controlled runtime for one sealed Tushare Theme provider capture."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from quant_investor.market.tushare_transport import OfficialTushareHttpsClient

from ._core import canonical_bytes
from .models import TushareContractError
from .theme_capture import (
    build_theme_provider_capture,
    capture_theme_partition,
    validate_theme_provider_capture,
    validate_theme_provider_execution_plan,
)


class ThemeCaptureSafetyError(ValueError):
    """Fail-closed local storage or invocation error."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ThemeCaptureSafetyError(f"{label}_INVALID") from exc
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise ThemeCaptureSafetyError(f"{label}_NOT_CANONICAL")
    return value


def write_exact(path: Path, value: dict[str, Any]) -> str:
    """Write canonical JSON exactly once or verify the existing exact bytes."""

    raw = canonical_bytes(value)
    if path.exists():
        observed = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
            or path.read_bytes() != raw
        ):
            raise ThemeCaptureSafetyError("THEME_EXISTING_FILE_CONFLICT")
        return hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ThemeCaptureSafetyError("THEME_FILE_WRITE_FAILED")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def load_exact_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ThemeCaptureSafetyError("THEME_PLAN_PATH_INVALID")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ThemeCaptureSafetyError("THEME_PLAN_SHA_MISMATCH")
    return validate_theme_provider_execution_plan(_load_json(raw, label="THEME_PLAN"))


def _validate_output_root(path: Path, *, create: bool) -> None:
    if not path.is_absolute() or path.is_symlink():
        raise ThemeCaptureSafetyError("THEME_OUTPUT_ROOT_INVALID")
    if path.exists():
        raise ThemeCaptureSafetyError("THEME_OUTPUT_ROOT_EXISTS")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ThemeCaptureSafetyError("THEME_OUTPUT_PARENT_INVALID")
    if create:
        path.mkdir(mode=0o700)
        path.chmod(0o700)


def _validate_resume_root(path: Path, plan: dict[str, Any]) -> Path:
    observed = path.lstat()
    if (
        not path.is_absolute()
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise ThemeCaptureSafetyError("THEME_RESUME_ROOT_INVALID")
    plan_path = path / "plan.json"
    if plan_path.is_symlink() or plan_path.read_bytes() != canonical_bytes(plan):
        raise ThemeCaptureSafetyError("THEME_RESUME_PLAN_MISMATCH")
    partitions = path / "partitions"
    partition_stat = partitions.lstat()
    if (
        not stat.S_ISDIR(partition_stat.st_mode)
        or stat.S_ISLNK(partition_stat.st_mode)
        or stat.S_IMODE(partition_stat.st_mode) != 0o700
    ):
        raise ThemeCaptureSafetyError("THEME_RESUME_PARTITIONS_INVALID")
    return partitions


def _prepare_root(path: Path, *, resume: bool, plan: dict[str, Any]) -> Path:
    if resume:
        return _validate_resume_root(path, plan)
    _validate_output_root(path, create=True)
    partitions = path / "partitions"
    partitions.mkdir(mode=0o700)
    partitions.chmod(0o700)
    write_exact(path / "plan.json", plan)
    return partitions


def _load_partition(path: Path) -> dict[str, Any]:
    observed = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        raise ThemeCaptureSafetyError("THEME_PARTITION_PATH_INVALID")
    return _load_json(path.read_bytes(), label="THEME_PARTITION")


def load_capture_root(
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Replay one complete output root into validated plan/capture/partitions."""

    plan = validate_theme_provider_execution_plan(
        _load_json((output_root / "plan.json").read_bytes(), label="THEME_PLAN")
    )
    planned = 1 + len(plan["company_keyset"])
    partitions = [
        _load_partition(output_root / "partitions" / f"{ordinal:05d}.json")
        for ordinal in range(planned)
    ]
    capture = validate_theme_provider_capture(
        _load_json((output_root / "capture.json").read_bytes(), label="THEME_CAPTURE"),
        plan=plan,
        partition_documents=partitions,
    )
    return plan, capture, partitions


def capture_theme_plan(
    *,
    plan_path: Path,
    plan_sha256: str,
    output_root: Path,
    allow_live: bool,
    resume: bool = False,
    client: Any | None = None,
    now: Callable[[], str] = _now,
) -> dict[str, Any]:
    """Capture one exact DC or TDX plan with no concurrency or retry."""

    plan = load_exact_plan(plan_path, plan_sha256)
    planned = 1 + len(plan["company_keyset"])
    if not allow_live:
        _validate_output_root(output_root, create=False)
        return {
            "network_attempts": 0,
            "plan_id": plan["plan_id"],
            "planned_partitions": planned,
            "provider": plan["provider"],
            "status": "DRY_RUN_VALIDATED",
        }
    partitions_root = _prepare_root(output_root, resume=resume, plan=plan)
    transport = client or OfficialTushareHttpsClient(strict_decimal_decode=True)
    partitions: list[dict[str, Any]] = []
    network_attempts = 0
    for ordinal in range(planned):
        path = partitions_root / f"{ordinal:05d}.json"
        if path.exists():
            partition = _load_partition(path)
        else:
            partition = capture_theme_partition(
                plan=plan,
                partition_ordinal=ordinal,
                captured_at=now(),
                client=transport,
            )
            network_attempts += 1
            write_exact(path, partition)
        if partition["partition_ordinal"] != ordinal:
            raise ThemeCaptureSafetyError("THEME_RESUME_KEYSET_MISMATCH")
        partitions.append(partition)
        if ordinal == 0 and partition["status"] == "INCOMPLETE":
            raise ThemeCaptureSafetyError("THEME_REGISTRY_INCOMPLETE")
    capture_path = output_root / "capture.json"
    if capture_path.exists():
        capture = validate_theme_provider_capture(
            _load_json(capture_path.read_bytes(), label="THEME_CAPTURE"),
            plan=plan,
            partition_documents=partitions,
        )
    else:
        capture = build_theme_provider_capture(
            plan=plan,
            partition_documents=partitions,
            completed_at=now(),
        )
        write_exact(capture_path, capture)
    summary = {
        "capture_id": capture["capture_id"],
        "completed_partitions": planned,
        "incomplete_partitions": capture["incomplete_partition_count"],
        "network_attempts": network_attempts,
        "plan_id": plan["plan_id"],
        "provider": plan["provider"],
        "status": capture["status"],
    }
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        existing = _load_json(summary_path.read_bytes(), label="THEME_SUMMARY")
        for field in (
            "capture_id",
            "completed_partitions",
            "incomplete_partitions",
            "plan_id",
            "provider",
            "status",
        ):
            if existing.get(field) != summary[field]:
                raise ThemeCaptureSafetyError("THEME_SUMMARY_BINDING_MISMATCH")
        return existing
    write_exact(summary_path, summary)
    return summary


__all__ = [
    "ThemeCaptureSafetyError",
    "capture_theme_plan",
    "load_capture_root",
    "load_exact_plan",
    "write_exact",
]

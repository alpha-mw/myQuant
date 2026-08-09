#!/usr/bin/env python3
"""Offline-first, exact-policy Tushare 10,000-point capability probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from quant_investor.intelligence_v2._core import canonical_bytes
from quant_investor.intelligence_v2.sources.tushare import (
    probe_tushare_capabilities,
    validate_tushare_endpoint_policy,
)


class ProbeSafetyError(RuntimeError):
    """A static, secret-free probe boundary failure."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProbeSafetyError("PROBE_POLICY_DUPLICATE_KEY")
        result[key] = value
    return result


def _load_policy(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ProbeSafetyError("PROBE_POLICY_PATH_INVALID")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ProbeSafetyError("PROBE_POLICY_SHA_MISMATCH")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ProbeSafetyError("PROBE_POLICY_NONFINITE")
            ),
        )
    except ProbeSafetyError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProbeSafetyError("PROBE_POLICY_INVALID") from exc
    policy = validate_tushare_endpoint_policy(value)
    if canonical_bytes(policy) != raw:
        raise ProbeSafetyError("PROBE_POLICY_NOT_CANONICAL")
    return policy


def _validate_new_output_root(path: Path, *, create: bool) -> None:
    if (
        not path.is_absolute()
        or any(character in str(path) for character in "*?[]")
        or path.exists()
        or path.is_symlink()
    ):
        raise ProbeSafetyError("PROBE_OUTPUT_ROOT_INVALID")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ProbeSafetyError("PROBE_OUTPUT_PARENT_INVALID")
    folded = path.name.casefold()
    if any(child.name.casefold() == folded for child in parent.iterdir()):
        raise ProbeSafetyError("PROBE_OUTPUT_CASEFOLD_COLLISION")
    current = parent
    while current != current.parent:
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ProbeSafetyError("PROBE_OUTPUT_PARENT_SYMLINK")
        current = current.parent
    if create:
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)
        if stat.S_IMODE(path.stat().st_mode) != 0o700:
            raise ProbeSafetyError("PROBE_OUTPUT_ROOT_MODE_INVALID")


def _write_exact(path: Path, document: Any) -> None:
    raw = canonical_bytes(document)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ProbeSafetyError("PROBE_WRITE_FAILED")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
        or path.read_bytes() != raw
    ):
        raise ProbeSafetyError("PROBE_WRITE_READBACK_FAILED")


def _summary(policy: dict[str, Any], *, live: bool) -> dict[str, Any]:
    plans = policy["endpoint_plans"]
    return {
        "lane": "TUSHARE_10000_CAPABILITY_PROBE",
        "live": live,
        "planned_max_network_attempts": sum(plan["planned_max_network_attempts"] for plan in plans),
        "points_endpoint_count": sum(plan["permission_class"] == "POINTS" for plan in plans),
        "policy_id": policy["policy_id"],
        "separate_endpoint_count": sum(plan["permission_class"] == "SEPARATE" for plan in plans),
        "status": "DRY_RUN_VALIDATED" if not live else "LIVE_PROBE_PENDING",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = _load_policy(Path(args.policy_path), args.policy_sha256)
    output_root = Path(args.output_root)
    _validate_new_output_root(output_root, create=False)
    summary = _summary(policy, live=bool(args.allow_live))
    if not args.allow_live:
        return summary
    _validate_new_output_root(output_root, create=True)
    result = probe_tushare_capabilities(
        policy=policy,
        probed_at=args.probed_at,
    )
    _write_exact(output_root / "policy.json", policy)
    _write_exact(
        output_root / "request_receipts.json",
        list(result["request_receipts"]),
    )
    _write_exact(
        output_root / "capability_receipts.json",
        list(result["capability_receipts"]),
    )
    _write_exact(
        output_root / "execution_receipts.json",
        list(result["execution_receipts"]),
    )
    summary.update(
        {
            "network_attempts": result["network_attempts"],
            "status": "LIVE_PROBE_RECORDED",
        }
    )
    _write_exact(output_root / "summary.json", summary)
    directory_fd = os.open(output_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--probed-at", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        summary = run(parse_args())
    except Exception:
        print(
            json.dumps(
                {"status": "PROBE_BLOCKED"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

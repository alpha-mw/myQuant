#!/usr/bin/env python3
"""Validate or promote an already sealed Fundamental VIP v4 staging generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from quant_investor.intelligence_v2._core import canonical_bytes
from quant_investor.intelligence_v2.package import verify_package
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4 import (
    validate_fundamental_execution_closure_v4,
)
from quant_investor.intelligence_v2.sources.tushare.fundamental_v4.upgrade import (
    run_staged_vip_promotion,
)
from quant_investor.market.fundamental_generation import (
    pointer_sha256,
    preflight_staged_fundamental_promotion,
)


class UpgradeSafetyError(RuntimeError):
    """Static, secret-free upgrade boundary failure."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UpgradeSafetyError("VIP_UPGRADE_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _load_exact_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise UpgradeSafetyError("VIP_UPGRADE_INPUT_PATH_INVALID")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise UpgradeSafetyError("VIP_UPGRADE_INPUT_SHA_MISMATCH")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                UpgradeSafetyError("VIP_UPGRADE_NONFINITE_JSON")
            ),
        )
    except UpgradeSafetyError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise UpgradeSafetyError("VIP_UPGRADE_INPUT_INVALID") from exc
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise UpgradeSafetyError("VIP_UPGRADE_INPUT_NOT_CANONICAL")
    return value


def _absolute_path(value: str, *, must_exist: bool) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(character in value for character in "*?[]")
        or path.is_symlink()
    ):
        raise UpgradeSafetyError("VIP_UPGRADE_PATH_INVALID")
    if must_exist and path.resolve(strict=True) != path:
        raise UpgradeSafetyError("VIP_UPGRADE_PATH_UNAVAILABLE")
    return path


def _journal_parent(path: Path) -> None:
    parent = path.parent
    if parent.exists():
        metadata = os.lstat(parent)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
        ):
            raise UpgradeSafetyError("VIP_UPGRADE_JOURNAL_PARENT_UNSAFE")
    else:
        parent.mkdir(mode=0o700)
    if path.exists() or path.is_symlink():
        raise UpgradeSafetyError("VIP_UPGRADE_ATTEMPT_ALREADY_EXISTS")


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    policy = validate_fundamental_execution_closure_v4(
        _load_exact_json(Path(args.policy_path), args.policy_sha256)
    )
    scope_path = Path(args.scope_path)
    _load_exact_json(scope_path, args.scope_sha256)
    plan = policy["request_plan"]
    if plan["as_of"] != args.as_of or plan["market_scope_ref"]["byte_sha256"] != args.scope_sha256:
        raise UpgradeSafetyError("VIP_UPGRADE_SCOPE_OR_ASOF_MISMATCH")
    staging = _absolute_path(args.staging_root, must_exist=bool(args.execute))
    canonical = _absolute_path(args.canonical_root, must_exist=True)
    if staging == canonical:
        raise UpgradeSafetyError("VIP_UPGRADE_ROOTS_COLLIDE")
    return policy, staging, canonical


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy, staging, canonical = _validate_inputs(args)
    summary: dict[str, Any] = {
        "as_of": args.as_of,
        "execute": bool(args.execute),
        "planned_max_network_attempts": policy["request_plan"]["planned_max_network_attempts"],
        "planned_terminal_request_count": policy["request_plan"]["planned_terminal_request_count"],
        "status": "DRY_RUN_VALIDATED",
    }
    if not args.execute:
        return summary
    if not args.allow_live or not args.expected_fundamental_pointer_sha256:
        raise UpgradeSafetyError("VIP_UPGRADE_EXECUTION_AUTHORITY_MISSING")
    observed = pointer_sha256(canonical)
    if observed != args.expected_fundamental_pointer_sha256:
        raise UpgradeSafetyError("VIP_UPGRADE_EXPECTED_POINTER_MISMATCH")
    preflight = preflight_staged_fundamental_promotion(
        staging_root=staging,
        canonical_root=canonical,
        expected_pointer_sha256=observed,
    )
    if preflight["provider_evidence"] is None:
        raise UpgradeSafetyError("VIP_UPGRADE_V4_EVIDENCE_MISSING")
    journal = _absolute_path(args.journal_root, must_exist=False)
    _journal_parent(journal)
    package = verify_package()
    authorized_arguments = {
        "allow_live": True,
        "as_of": args.as_of,
        "canonical_root": str(canonical),
        "execute": True,
        "expected_fundamental_pointer_sha256": observed,
        "policy_sha256": args.policy_sha256,
        "scope_sha256": args.scope_sha256,
        "staging_root": str(staging),
    }
    result = run_staged_vip_promotion(
        staging_root=staging,
        canonical_root=canonical,
        journal_root=journal,
        attempt_id=args.attempt_id,
        as_of=args.as_of,
        expected_pointer_sha256=observed,
        package_sha256=package["semantic_sha256"],
        authorized_arguments=authorized_arguments,
    )
    summary.update(result)
    summary["status"] = result["state"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--scope-path", required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--journal-root")
    parser.add_argument("--attempt-id")
    parser.add_argument("--expected-fundamental-pointer-sha256")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if args.execute and (not args.journal_root or not args.attempt_id):
            raise UpgradeSafetyError("VIP_UPGRADE_JOURNAL_IDENTITY_MISSING")
        result = run(args)
    except Exception:
        print('{"status":"VIP_UPGRADE_BLOCKED"}')
        return 2
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

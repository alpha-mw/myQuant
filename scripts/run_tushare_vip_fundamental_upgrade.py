#!/usr/bin/env python3
"""Validate an inactive Fundamental shadow result for unified activation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

from quant_investor.market.tushare._core import canonical_bytes
from quant_investor.market.tushare.fundamental import (
    validate_fundamental_execution_closure,
)


class UpgradeSafetyError(RuntimeError):
    """Static, secret-free upgrade boundary failure."""

    code = "FUNDAMENTAL_UPGRADE_BLOCKED"
    exit_code = 2
    public_fields: dict[str, Any] = {}


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
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise UpgradeSafetyError("VIP_UPGRADE_INPUT_PATH_UNSAFE")
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


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    execution = validate_fundamental_execution_closure(
        _load_exact_json(
            Path(args.execution_closure_path),
            args.execution_closure_sha256,
        )
    )
    scope_path = Path(args.scope_path)
    _load_exact_json(scope_path, args.scope_sha256)
    plan = execution["request_plan"]
    if plan["as_of"] != args.as_of or plan["market_scope_ref"]["byte_sha256"] != args.scope_sha256:
        raise UpgradeSafetyError("VIP_UPGRADE_SCOPE_OR_ASOF_MISMATCH")
    staging = _absolute_path(args.staging_root, must_exist=False)
    canonical = _absolute_path(args.canonical_root, must_exist=True)
    if staging == canonical:
        raise UpgradeSafetyError("VIP_UPGRADE_ROOTS_COLLIDE")
    return execution, staging, canonical


def run(args: argparse.Namespace) -> dict[str, Any]:
    execution, _staging, _canonical = _validate_inputs(args)
    plan = execution["request_plan"]
    summary: dict[str, Any] = {
        "as_of": args.as_of,
        "execution_closure_id": execution["closure_id"],
        "execution_contract_sha256": execution["contract_sha256"],
        "execute": bool(args.execute),
        "planned_max_network_attempts": plan["planned_max_network_attempts"],
        "planned_terminal_request_count": plan["planned_terminal_request_count"],
        "status": "DRY_RUN_VALIDATED",
    }
    if not args.execute:
        return summary
    raise UpgradeSafetyError("VIP_UPGRADE_REQUIRES_UNIFIED_ACTIVATION")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-closure-path", required=True)
    parser.add_argument("--execution-closure-sha256", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--scope-path", required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--canonical-root", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        result = run(args)
    except Exception:
        print('{"status":"VIP_UPGRADE_BLOCKED"}')
        return 2
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

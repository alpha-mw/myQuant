#!/usr/bin/env python3
"""Offline-first capture of the exact SW2021 industry taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from quant_investor.market.tushare import (
    TushareContractError,
    capture_tushare_industry_taxonomy,
    validate_industry_taxonomy_execution_plan,
)
from quant_investor.market.tushare._core import canonical_bytes
from quant_investor.market.tushare_transport import OfficialTushareHttpsClient
from scripts.probe_tushare_10000_capabilities import (
    ProbeSafetyError,
    _unique_object,
    _validate_new_output_root,
    _write_exact,
)


def _load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ProbeSafetyError("TAXONOMY_PLAN_PATH_INVALID")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ProbeSafetyError("TAXONOMY_PLAN_SHA_MISMATCH")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ProbeSafetyError("TAXONOMY_PLAN_NONFINITE")
            ),
        )
    except ProbeSafetyError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProbeSafetyError("TAXONOMY_PLAN_INVALID") from exc
    plan = validate_industry_taxonomy_execution_plan(value)
    if canonical_bytes(plan) != raw:
        raise ProbeSafetyError("TAXONOMY_PLAN_NOT_CANONICAL")
    return plan


def run(
    args: argparse.Namespace,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    plan = _load_plan(Path(args.plan_path), args.plan_sha256)
    output_root = Path(args.output_root)
    _validate_new_output_root(output_root, create=False)
    summary = {
        "capture_id": None,
        "lane": "TUSHARE_I2_INDUSTRY_TAXONOMY",
        "live": bool(args.allow_live),
        "network_attempts": 0,
        "plan_id": plan["plan_id"],
        "planned_max_network_attempts": plan["planned_max_network_attempts"],
        "status": "DRY_RUN_VALIDATED" if not args.allow_live else "LIVE_PENDING",
    }
    if not args.allow_live:
        return summary

    _validate_new_output_root(output_root, create=True)
    request_client = client or OfficialTushareHttpsClient(strict_decimal_decode=True)
    capture = capture_tushare_industry_taxonomy(
        plan=plan,
        captured_at=args.captured_at,
        client=request_client,
    )
    summary.update(
        {
            "capture_id": capture["capture_id"],
            "network_attempts": plan["planned_terminal_request_count"],
            "status": "LIVE_CAPTURE_RECORDED",
        }
    )
    _write_exact(output_root / "plan.json", plan)
    _write_exact(output_root / "capture.json", capture)
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
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--captured-at", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        summary = run(parse_args())
    except (ProbeSafetyError, TushareContractError) as exc:
        print(
            json.dumps(
                {"blocker": str(exc), "status": "TAXONOMY_CAPTURE_BLOCKED"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    except Exception:
        print(
            json.dumps(
                {"status": "TAXONOMY_CAPTURE_BLOCKED"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

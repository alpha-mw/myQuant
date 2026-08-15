#!/usr/bin/env python3
"""Offline-first, one-request sanitized Tushare schema diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

from quant_investor.market.tushare import (
    build_tushare_request_receipt,
    build_tushare_schema_diagnostic_receipt,
)
from quant_investor.market.tushare_transport import (
    OfficialTushareHttpsClient,
    TushareSchemaDiagnostic,
)
from scripts.probe_tushare_10000_capabilities import (
    ProbeSafetyError,
    _load_policy,
    _validate_new_output_root,
    _write_exact,
)


class SchemaDiagnosticClient(Protocol):
    def diagnose_schema(
        self,
        *,
        api_name: str,
        params: dict[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareSchemaDiagnostic: ...


def _single_plan(policy: dict[str, Any]) -> dict[str, Any]:
    plans = policy["endpoint_plans"]
    if (
        len(plans) != 1
        or plans[0]["permission_class"] != "POINTS"
        or plans[0]["strict_decimal_decode"] is not True
        or plans[0]["max_attempts"] != 1
        or plans[0]["planned_max_network_attempts"] != 1
        or plans[0]["planned_terminal_request_count"] != 1
        or len(plans[0]["ordered_expected_partition_keyset"]) != 1
    ):
        raise ProbeSafetyError("SCHEMA_DIAGNOSTIC_REQUIRES_ONE_REQUEST")
    return plans[0]


def run(
    args: argparse.Namespace,
    *,
    client: SchemaDiagnosticClient | None = None,
) -> dict[str, Any]:
    policy = _load_policy(Path(args.policy_path), args.policy_sha256)
    plan = _single_plan(policy)
    output_root = Path(args.output_root)
    _validate_new_output_root(output_root, create=False)
    summary = {
        "api_name": plan["api_name"],
        "lane": "TUSHARE_SANITIZED_SCHEMA_DIAGNOSTIC",
        "live": bool(args.allow_live),
        "planned_max_network_attempts": 1,
        "policy_id": policy["policy_id"],
        "status": "DRY_RUN_VALIDATED" if not args.allow_live else "LIVE_PENDING",
    }
    if not args.allow_live:
        return summary
    _validate_new_output_root(output_root, create=True)
    partition_key = plan["ordered_expected_partition_keyset"][0]
    request_receipt = build_tushare_request_receipt(
        plan=plan,
        partition_key=partition_key,
        partition_ordinal=0,
        sanitized_params=plan["fixed_params"],
        requested_at=args.diagnosed_at,
    )
    transport = OfficialTushareHttpsClient(strict_decimal_decode=True) if client is None else client
    diagnostic = asdict(
        transport.diagnose_schema(
            api_name=plan["api_name"],
            params=plan["fixed_params"],
            expected_fields=plan["expected_fields"],
        )
    )
    receipt = build_tushare_schema_diagnostic_receipt(
        plan=plan,
        request_receipt=request_receipt,
        sanitized_params=plan["fixed_params"],
        diagnostic=diagnostic,
        captured_at=args.diagnosed_at,
    )
    summary.update(
        {
            "expected_fields_match": receipt["expected_fields_match"],
            "network_attempts": 1,
            "schema_diagnostic_receipt_id": receipt["schema_diagnostic_receipt_id"],
            "status": "LIVE_DIAGNOSTIC_RECORDED",
        }
    )
    _write_exact(output_root / "policy.json", policy)
    _write_exact(output_root / "request_receipt.json", request_receipt)
    _write_exact(output_root / "schema_diagnostic_receipt.json", receipt)
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
    parser.add_argument("--diagnosed-at", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        summary = run(parse_args())
    except Exception:
        print(
            json.dumps(
                {"status": "SCHEMA_DIAGNOSTIC_BLOCKED"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

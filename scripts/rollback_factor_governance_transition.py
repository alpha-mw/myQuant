#!/usr/bin/env python3
"""Retired v2 rollback CLI; v3 has no authorized registry mutation yet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.governance_protocol_v3 import (  # noqa: E402
    FORWARD_PRODUCTION_APPLY_BLOCKER,
)


def _required_hash(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return text


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--inverse-wal", required=True)
    parser.add_argument("--mutation-budget-ledger", required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--expected-protocol-hash", required=True)
    parser.add_argument("--expected-current-registry-sha256", required=True)
    parser.add_argument("--expected-inverse-wal-sha256", required=True)
    parser.add_argument("--expected-transition-hash", required=True)
    parser.add_argument("--expected-mutation-plan-hash", required=True)
    parser.add_argument("--expected-evidence-hash", required=True)
    parser.add_argument(
        "--rollback-wal",
        default="",
        help="Required output WAL path only with --apply-rollback.",
    )
    parser.add_argument(
        "--apply-rollback",
        action="store_true",
        help="Apply the inverse patch. The default is a read-only dry run.",
    )
    args = parser.parse_args(argv)
    for field in (
        "expected_protocol_hash",
        "expected_current_registry_sha256",
        "expected_inverse_wal_sha256",
        "expected_transition_hash",
        "expected_mutation_plan_hash",
        "expected_evidence_hash",
    ):
        try:
            _required_hash(getattr(args, field), f"--{field.replace('_', '-')}")
        except ValueError as exc:
            parser.error(str(exc))
    if args.apply_rollback and not str(args.rollback_wal or "").strip():
        parser.error("--apply-rollback requires --rollback-wal")
    return args


def run_rollback(args: argparse.Namespace) -> dict[str, Any]:
    del args
    raise ValueError(
        "Legacy governance rollback is retired; "
        + FORWARD_PRODUCTION_APPLY_BLOCKER
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = run_rollback(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"factor_governance_rollback_blocked={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

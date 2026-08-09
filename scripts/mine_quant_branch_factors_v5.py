#!/usr/bin/env python3
"""Explicit v5 factor-lane helpers; no live mining or registry mutation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from quant_investor.factors.governance_v5 import (
    build_diagnostic_scan_receipt,
    canonical_bytes,
    strict_json_loads,
)
from quant_investor.factors.governance_v5.contracts import validate_preregistration


def load_preregistered_candidates(
    path: Path, *, expected_byte_sha256: str, policy: dict
) -> tuple[dict, ...]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_byte_sha256:
        raise ValueError("v5 preregistration byte SHA mismatch")
    payload = strict_json_loads(raw, label="v5 preregistration bundle")
    if type(payload) is not dict or set(payload) != {"policy", "preregistration"}:
        raise ValueError("v5 preregistration bundle shape is invalid")
    if payload["policy"] != policy:
        raise ValueError("v5 preregistration policy mismatch")
    document = validate_preregistration(payload["preregistration"], policy=policy)
    return tuple(row for row in document["candidates"] if row["role"] == "PRIMARY")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-candidate-id", action="append", default=[])
    parser.add_argument("--scanned-at", required=True)
    parser.add_argument("--implementation-sha256", required=True)
    args = parser.parse_args()
    receipt = build_diagnostic_scan_receipt(
        scanned_at=args.scanned_at,
        implementation_sha256=args.implementation_sha256,
        candidate_ids=args.diagnostic_candidate_id,
    )
    print((canonical_bytes(receipt) + b"\n").decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare, validate and report a non-authorizing Macro recovery candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_investor.macro.retrospective_recovery import (
    build_retrospective_market_projections,
)
from quant_investor.macro.local_market_observations import (
    compile_local_market_breadth_observation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-receipt", type=Path, required=True)
    parser.add_argument("--expected-attempt-receipt-sha256", required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-snapshot-sha256", required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--expected-capture-manifest-sha256", required=True)
    parser.add_argument("--scope-artifact", type=Path, required=True)
    parser.add_argument("--expected-scope-sha256", required=True)
    parser.add_argument("--reconstructed-at", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt_raw = args.attempt_receipt.read_bytes()
    if hashlib.sha256(receipt_raw).hexdigest() != args.expected_attempt_receipt_sha256:
        raise ValueError("attempt_receipt_sha_mismatch")
    receipt = json.loads(receipt_raw)
    if receipt.get("target_date") != "20260820" or receipt.get("mode") != "execute":
        raise ValueError("attempt_receipt_not_exact_20260820_execute")
    candidate = build_retrospective_market_projections(
        source_snapshot_manifest_path=args.source_snapshot_manifest,
        expected_source_snapshot_sha256=args.expected_source_snapshot_sha256,
        capture_manifest_path=args.capture_manifest,
        expected_capture_manifest_sha256=args.expected_capture_manifest_sha256,
        attempt_root=args.attempt_receipt.parent,
        reconstructed_at=args.reconstructed_at,
        output_root=args.output_root,
    )
    observations = []
    validation_at = datetime.fromtimestamp(
        max(Path(row["path"]).stat().st_mtime for row in candidate["projections"]),
        tz=timezone.utc,
    ) + timedelta(seconds=1)
    for row in candidate["projections"]:
        observation, evidence = compile_local_market_breadth_observation(
            snapshot_manifest_path=row["path"],
            expected_snapshot_manifest_sha256=row["sha256"],
            coverage_manifest_path=row["path"],
            expected_coverage_manifest_sha256=row["sha256"],
            target_trade_date=row["target_trade_date"],
            scope_artifact_path=args.scope_artifact,
            expected_scope_artifact_sha256=args.expected_scope_sha256,
            as_of=validation_at,
            clock=lambda: validation_at,
        )
        observations.append(
            {
                "target_trade_date": row["target_trade_date"],
                "observation_content_hash": observation.content_hash,
                "evidence_sha256": evidence["evidence_sha256"],
            }
        )
    print(
        json.dumps(
            {
                "status": "PREPARED_NON_AUTHORIZING",
                "candidate_id": candidate["candidate_id"],
                "manifest_path": candidate["manifest_path"],
                "manifest_sha256": candidate["manifest_sha256"],
                "observations": observations,
                "canonical_pointer_write": False,
                "market_pointer_write": False,
                "pit_pointer_write": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

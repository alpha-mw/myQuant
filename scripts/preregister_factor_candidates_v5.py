#!/usr/bin/env python3
"""Build a research-only Factor Governance v5 preregistration artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from quant_investor.factors.governance_v5 import (
    build_governance_policy,
    build_preregistration,
    canonical_bytes,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_catalog() -> list[dict[str, Any]]:
    definitions = (
        ("val_book_yield", "value", "PRIMARY", "cs_rank(1.0 / pb)", ("pb",)),
        ("size_small_cap", "size", "PRIMARY", "-cs_rank(total_mv)", ("total_mv",)),
        ("qual_roe", "quality", "PRIMARY", "cs_rank(fin_roe)", ("fin_roe",)),
        (
            "lev_low_debt",
            "leverage",
            "PRIMARY",
            "-cs_rank(fin_debt_to_assets)",
            ("fin_debt_to_assets",),
        ),
        (
            "eq_ocf_backing",
            "earnings_quality",
            "PRIMARY",
            "cs_rank(fin_ocf_to_profit)",
            ("fin_ocf_to_profit",),
        ),
        (
            "val_fcf_yield",
            "value",
            "ALTERNATE_FOR:val_book_yield",
            "cs_rank(fcf_to_price)",
            ("fcf_to_price",),
        ),
        (
            "size_float_cap",
            "size",
            "ALTERNATE_FOR:size_small_cap",
            "-cs_rank(circ_mv)",
            ("circ_mv",),
        ),
        (
            "qual_roa",
            "quality",
            "ALTERNATE_FOR:qual_roe",
            "cs_rank(fin_roa)",
            ("fin_roa",),
        ),
        (
            "eq_fcf_backing",
            "earnings_quality",
            "ALTERNATE_FOR:eq_ocf_backing",
            "cs_rank(fin_fcf_to_profit)",
            ("fin_fcf_to_profit",),
        ),
    )
    return [
        {
            "candidate_id": candidate_id,
            "expression": expression,
            "family": family,
            "implementation_sha256": _sha(f"aquant-expression-v5:{expression}"),
            "input_fields": list(inputs),
            "parameterization": "NONE",
            "role": role,
            "source_sha256": _sha(f"documented-prior:{candidate_id}"),
        }
        for candidate_id, family, role, expression, inputs in definitions
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--sealed-at", required=True)
    parser.add_argument("--evaluation-start-session", required=True)
    parser.add_argument("--evaluation-end-session", required=True)
    parser.add_argument("--label-available-at", required=True)
    parser.add_argument("--coverage-threshold", required=True)
    parser.add_argument("--label-horizon-sessions", required=True, type=int)
    parser.add_argument("--minimum-prospective-paths", required=True, type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    policy = build_governance_policy(
        created_at=args.created_at,
        coverage_threshold=args.coverage_threshold,
        label_horizon_sessions=args.label_horizon_sessions,
        minimum_prospective_paths=args.minimum_prospective_paths,
    )
    document = build_preregistration(
        policy=policy,
        sealed_at=args.sealed_at,
        evaluation_start_session=args.evaluation_start_session,
        evaluation_end_session=args.evaluation_end_session,
        label_available_at=args.label_available_at,
        candidates=candidate_catalog(),
    )
    payload = canonical_bytes({"policy": policy, "preregistration": document}) + b"\n"
    target = Path(args.out)
    if not target.is_absolute():
        raise SystemExit("--out must be an explicit absolute path")
    if not args.execute:
        print(payload.decode("utf-8"), end="")
        return 0
    if target.exists() or target.is_symlink():
        raise SystemExit("refusing to overwrite preregistration output")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    if target.read_bytes() != payload:
        raise SystemExit("preregistration readback mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

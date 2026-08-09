#!/usr/bin/env python3
"""Seal a pre-commitment record of factor candidates before any testing.

The deflated Sharpe ratio charges a result for the size of the search that
produced it, so the trial count has to be fixed before the data is touched. The
2026-06-04 mining run is the counter-example: 138 candidates, 70 of them
smoothing variants of one idea, and the two "qualified" winners came out at
DSR 0.712 and 0.639 against a 0.95 floor once the search size was accounted for.

So this file is written and hashed first, and the candidate set may not grow
afterwards. One candidate per family, chosen a priori from documented anomalies
rather than from anything observed in this panel. Alternates exist only to
substitute for a primary whose input coverage turns out to be inadequate; a
substitution replaces the primary and never adds a trial.

Governance target (quant_investor/factors/governance_protocol_v4.py):
    MIN_NEW_RISK_FACTOR_COUNT = 5, TARGET_PRODUCTION_FACTOR_COUNT = 10
    exact_five_requires_five_distinct_families - at exactly 5 factors, all five
    families must differ, so the five primaries below are one per family.
    MAX_FACTOR_ABS_WEIGHT = 0.20, MAX_FAMILY_ABS_WEIGHT = 0.35.

Every expression is evaluated by quant_investor.factors.aquant_expression, whose
whitelist now carries pe/pb/total_mv/circ_mv alongside the fin_* PIT metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "factor-candidate-preregistration.v1"

# `cs_rank` is ascending, so a factor is written to rank the names it prefers
# highest. Where the raw field is inverse to the hypothesis (a large market cap,
# a high debt ratio) the expression negates or inverts it rather than relying on
# a separate direction flag, so the sign is visible in the expression itself.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "val_book_yield",
        "family": "value",
        "role": "primary",
        "expression": "cs_rank(1.0 / pb)",
        "hypothesis": "High book-to-price earns a positive premium.",
        "rationale": (
            "The value premium is among the most replicated cross-sectional "
            "effects and has historically been strong in CN A-shares. B/P is "
            "preferred to E/P as the primary because pb coverage is 99% against "
            "92% for pe, and because pb stays defined for loss-making firms "
            "where pe is null and non-monotonic around zero."
        ),
        "inputs": ["pb"],
    },
    {
        "candidate_id": "size_small_cap",
        "family": "size",
        "role": "primary",
        "expression": "-cs_rank(total_mv)",
        "hypothesis": "Small total market cap earns a positive premium.",
        "rationale": (
            "The size effect has weakened in developed markets but remained "
            "economically large in CN A-shares. Negated because the hypothesis "
            "prefers small caps while cs_rank ascends with size."
        ),
        "inputs": ["total_mv"],
    },
    {
        "candidate_id": "qual_roe",
        "family": "quality",
        "role": "primary",
        "expression": "cs_rank(fin_roe)",
        "hypothesis": "High return on equity earns a positive premium.",
        "rationale": (
            "Profitability is the quality dimension with the most consistent "
            "out-of-sample support. ROE is used rather than ROA because it is "
            "the more widely followed metric in this market and so is the more "
            "honest a priori choice."
        ),
        "inputs": ["fin_roe"],
    },
    {
        "candidate_id": "lev_low_debt",
        "family": "leverage",
        "role": "primary",
        "expression": "-cs_rank(fin_debt_to_assets)",
        "hypothesis": "Low balance-sheet leverage earns a positive premium.",
        "rationale": (
            "Distress risk. This is the weakest of the five a priori and is "
            "included because governance requires five distinct families at a "
            "five-factor set, not because the prior evidence is strong. It is "
            "the candidate most likely to fail honestly, and that is an "
            "acceptable outcome to record."
        ),
        "inputs": ["fin_debt_to_assets"],
    },
    {
        "candidate_id": "eq_ocf_backing",
        "family": "earnings_quality",
        "role": "primary",
        "expression": "cs_rank(fin_ocf_to_profit)",
        "hypothesis": (
            "Earnings backed by operating cash flow earn a positive premium."
        ),
        "inputs": ["fin_ocf_to_profit"],
        "rationale": (
            "The accruals anomaly: reported profit unsupported by cash tends to "
            "reverse. Directly relevant in a market where earnings management is "
            "a live concern."
        ),
    },
    # Substitutes only. Each replaces its family's primary if that primary's
    # input coverage fails the admission gates; it never runs alongside it.
    {
        "candidate_id": "val_fcf_yield",
        "family": "value",
        "role": "alternate_for:val_book_yield",
        "expression": "cs_rank(fcf_to_price)",
        "hypothesis": "High free-cash-flow yield earns a positive premium.",
        "rationale": "Cash-based value, unaffected by book-value accounting policy.",
        "inputs": ["fcf_to_price"],
    },
    {
        "candidate_id": "size_float_cap",
        "family": "size",
        "role": "alternate_for:size_small_cap",
        "expression": "-cs_rank(circ_mv)",
        "hypothesis": "Small free-float market cap earns a positive premium.",
        "rationale": (
            "Free float rather than total cap, which matters in CN where large "
            "restricted state holdings distort total market capitalisation."
        ),
        "inputs": ["circ_mv"],
    },
    {
        "candidate_id": "qual_roa",
        "family": "quality",
        "role": "alternate_for:qual_roe",
        "expression": "cs_rank(fin_roa)",
        "hypothesis": "High return on assets earns a positive premium.",
        "rationale": "Profitability net of leverage, so it does not double-count the leverage family.",
        "inputs": ["fin_roa"],
    },
    {
        "candidate_id": "eq_fcf_backing",
        "family": "earnings_quality",
        "role": "alternate_for:eq_ocf_backing",
        "expression": "cs_rank(fin_fcf_to_profit)",
        "hypothesis": "Earnings backed by free cash flow earn a positive premium.",
        "rationale": "Stricter than OCF backing because it charges for capital expenditure.",
        "inputs": ["fin_fcf_to_profit"],
    },
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def build_record(*, as_of: str) -> dict[str, Any]:
    primaries = [row for row in CANDIDATES if row["role"] == "primary"]
    families = sorted({str(row["family"]) for row in primaries})
    if len(primaries) != len(families):
        raise ValueError(
            "exact_five_requires_five_distinct_families: each primary must own a distinct family"
        )

    body = {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of,
        "candidates": list(CANDIDATES),
        "primary_count": len(primaries),
        "families": families,
        "trial_accounting": {
            "declared_trial_count": len(primaries),
            "alternate_count": len(CANDIDATES) - len(primaries),
            "rule": (
                "An alternate substitutes for its family's primary on inadequate "
                "input coverage and never runs alongside it, so the trial count "
                "stays at the primary count regardless of substitutions."
            ),
        },
        "honest_test_contract": {
            "deflated_sharpe_floor": 0.95,
            "harvey_liu_zhu_t_hurdle": 3.0,
            "pbo_ceiling": 0.5,
            "min_month_end_rankic_count": 12,
            "min_nonoverlap_30d_cohort_count": 8,
            "fdr_method": "benjamini_hochberg_by_family",
            "rule": (
                "No candidate may be admitted by relaxing any of these. A failing "
                "candidate is recorded as failed."
            ),
        },
        "prohibitions": [
            "no candidate may be added after this record is sealed",
            "no parameter or smoothing variants may be generated from these expressions",
            "no expression may be edited in response to an observed result",
        ],
    }
    body["record_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "results" / "factor_governance" / "candidate_preregistration.json"
        ),
    )
    parser.add_argument("--execute", action="store_true", help="write the record")
    args = parser.parse_args()

    record = build_record(as_of=args.as_of)
    primaries = [row for row in record["candidates"] if row["role"] == "primary"]

    print(f"schema        : {record['schema_version']}")
    print(f"as_of         : {record['as_of']}")
    print(f"primaries     : {len(primaries)} across {len(record['families'])} families")
    print(f"alternates    : {record['trial_accounting']['alternate_count']}")
    print(f"declared trials: {record['trial_accounting']['declared_trial_count']}")
    print(f"record_sha256 : {record['record_sha256']}")
    print()
    for row in primaries:
        print(f"  {row['family']:<17} {row['candidate_id']:<18} {row['expression']}")

    out = Path(args.out)
    if not args.execute:
        print(f"\nDRY RUN - would write {out}. Re-run with --execute.")
        return 0
    if out.exists():
        print(f"\nREFUSING to overwrite an existing sealed record: {out}")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"\nsealed -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

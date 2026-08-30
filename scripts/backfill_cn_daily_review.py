#!/usr/bin/env python3
"""Seal one explicitly retrospective CN daily-review evidence artifact.

This command never creates a scheduled-run receipt, Strategy Record Store
continuity, a formal decision, an order, or a trade.  It reconstructs only what
can be established from caller-bound immutable evidence and preserves missing
contemporaneous inputs as blockers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_investor.strategy_records.performance import immutable_write  # noqa: E402
from quant_investor.strategy_records.store import (  # noqa: E402
    canonical_json_bytes,
    regular_file_sha256,
)
from scripts.export_cn_weekly_review_evidence import (  # noqa: E402
    _registered_cn_trade_dates,
)

SCHEMA_ID = "myquant.research.daily-review-retrospective.v1"
OUTPUT_ROOT = Path("results/operations/daily_reviews/CN")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRADE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class RetrospectiveReviewError(RuntimeError):
    """Stable fail-closed retrospective-review error."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_utc(value: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RetrospectiveReviewError("generated_at is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RetrospectiveReviewError("generated_at is not canonical UTC") from exc
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _exact_json(path: Path, expected_sha: str, *, label: str) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_sha) is None:
        raise RetrospectiveReviewError(f"{label} SHA is invalid")
    digest, _ = regular_file_sha256(path, label=label)
    if digest != expected_sha:
        raise RetrospectiveReviewError(f"{label} exact bytes mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrospectiveReviewError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RetrospectiveReviewError(f"{label} is not an object")
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    trade_date = args.trade_date
    if _TRADE_DATE.fullmatch(trade_date) is None:
        raise RetrospectiveReviewError("trade_date is invalid")
    generated_at = _parse_utc(args.generated_at)
    open_dates, calendar_refs = _registered_cn_trade_dates(
        PROJECT_ROOT,
        start_date=trade_date,
        end_date=trade_date,
    )
    if open_dates != [trade_date]:
        raise RetrospectiveReviewError("trade_date is not a registered CN session")

    catalog_path = PROJECT_ROOT / args.catalog_path
    maintenance_path = PROJECT_ROOT / args.maintenance_path
    try:
        catalog_path.relative_to(PROJECT_ROOT)
        maintenance_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise RetrospectiveReviewError("evidence path escapes workspace") from exc
    catalog = _exact_json(catalog_path, args.catalog_sha256, label="catalog")
    maintenance = _exact_json(
        maintenance_path,
        args.maintenance_sha256,
        label="maintenance receipt",
    )
    if catalog.get("schema_id") != "myquant.strategy_record_catalog.v3":
        raise RetrospectiveReviewError("catalog is not Store-v3")
    published_at = catalog.get("published_at")
    if not isinstance(published_at, str) or published_at[:10] > trade_date:
        raise RetrospectiveReviewError("catalog was not available by trade date")
    maintenance_target = maintenance.get("target_date")
    if not isinstance(maintenance_target, str) or re.fullmatch(r"\d{8}", maintenance_target) is None:
        raise RetrospectiveReviewError("maintenance target is invalid")

    target_compact = trade_date.replace("-", "")
    observation_paths = [
        f"results/factors/observations/{trade_date[:4]}/{trade_date[5:7]}/{trade_date[8:]}/{alias}.json"
        for alias in ("LOW", "W80")
    ]
    present_observations = [
        path for path in observation_paths if (PROJECT_ROOT / path).exists()
    ]
    if present_observations:
        raise RetrospectiveReviewError(
            "retrospective missing-observation declaration conflicts with disk"
        )

    blockers = [
        "SCHEDULED_DAILY_REVIEW_TASK_MISSING",
        "REGISTERED_DAILY_REVIEW_RECEIPT_MISSING",
        "FACTOR_LOW_OBSERVATION_MISSING",
        "FACTOR_W80_OBSERVATION_MISSING",
        "SAME_DATE_MAINTENANCE_RECEIPT_MISSING",
        "HOLDINGS_CONTINUITY_UNCONFIRMED",
        "FORMAL_DECISION_UNAVAILABLE",
    ]
    evidence_refs = [
        {
            "role": "PRIOR_STORE_CATALOG",
            "path": args.catalog_path,
            "sha256": args.catalog_sha256,
        },
        {
            "role": "OBSERVED_MAINTENANCE_RECEIPT",
            "path": args.maintenance_path,
            "sha256": args.maintenance_sha256,
            "target_date": maintenance_target,
        },
        *[{"role": "REGISTERED_CALENDAR", **ref} for ref in calendar_refs],
    ]
    artifact: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "review_id": f"retrospective-daily-review-{target_compact}-v1",
        "strategy_id": "cn-aggressive-tech-manufacturing",
        "trade_date": trade_date,
        "generated_at": generated_at,
        "mode": "RETROSPECTIVE_RECONSTRUCTION",
        "scheduled_execution_status": "MISSING",
        "review_status": "COMPLETED_RETROSPECTIVE",
        "evidence_quality": "PARTIAL",
        "research_state": "INSUFFICIENT_EVIDENCE",
        "formal_advisory_status": "FORMAL_ADVISORY_BLOCKED",
        "decision_log_status": "NOT_APPLICABLE",
        "actions": [],
        "blockers": blockers,
        "missing_expected_paths": observation_paths,
        "evidence_refs": evidence_refs,
        "authority": {
            "portfolio": False,
            "holdings": False,
            "decision_log": False,
            "broker": False,
            "order": False,
            "execution": False,
            "trade": False,
        },
    }
    artifact["content_sha256"] = _sha(canonical_json_bytes(artifact))
    output_path = PROJECT_ROOT / OUTPUT_ROOT / target_compact / "retrospective-review.v1.json"
    raw = canonical_json_bytes(artifact)
    if output_path.exists():
        if output_path.read_bytes() != raw:
            raise RetrospectiveReviewError("retrospective review already exists with different bytes")
        digest = _sha(raw)
        status = "NO_ACTION"
    else:
        digest = immutable_write(output_path, raw, max_bytes=128 * 1024)
        status = "SEALED"
    return {
        "status": status,
        "path": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "byte_sha256": digest,
        "content_sha256": artifact["content_sha256"],
        "research_state": artifact["research_state"],
        "formal_advisory_status": artifact["formal_advisory_status"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--catalog-path", required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--maintenance-path", required=True)
    parser.add_argument("--maintenance-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        result = build(build_parser().parse_args(argv))
    except (OSError, KeyError, RetrospectiveReviewError) as exc:
        print(json.dumps({"status": "BLOCKED", "blocker": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

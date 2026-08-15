#!/usr/bin/env python3
"""Build a four-section, redacted static site for GitHub Pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (
    PROJECT_ROOT
    / "portfolio_dashboard"
    / "private"
    / "generated"
    / "cn_aggressive_dashboard.v1.json"
)
MARKER = ".cn-dashboard-public-site"
PUBLIC_MODE = "window.CNPublicDashboard = true;\n"
REDACTED = "PUBLIC_REDACTED"
ZERO_SHA = "0" * 64
APPROVED_PUBLIC_MARKERS = (
    "public-section-brand",
    "public-section-performance",
    "public-section-monthly",
    "public-section-readout",
)
FORBIDDEN_PUBLIC_COPY = ("外部资金流", "入金", "出金", "资金事件")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sanitize_evidence(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in result:
        if key.endswith("_path"):
            result[key] = REDACTED
        elif key.endswith("_sha256"):
            result[key] = ZERO_SHA
        elif isinstance(result[key], bool):
            result[key] = False
        elif result[key] is not None:
            result[key] = REDACTED
    return result


def sanitize_bundle(source: dict[str, Any]) -> dict[str, Any]:
    """Remove values outside the approved public performance view."""

    bundle = json.loads(json.dumps(source))
    bundle["public_redacted"] = True
    bundle["public_redaction_policy"] = [
        "absolute_portfolio_values",
        "holdings_details",
        "absolute_funding_values",
        "source_paths_and_sha256",
        "internal_evidence_and_risk_detail",
    ]

    portfolio = bundle["portfolio"]
    for key in (
        "cash",
        "market_value",
        "total_value",
        "cash_weight",
        "gross_exposure",
        "portfolio_pnl",
        "current_unrealized_pnl",
        "latest_record_realized_pnl_from_rebalance",
        "performance_initial_capital",
        "excluded_external_flow",
        "adjusted_total_value",
        "cumulative_profit_excluding_external_flow",
    ):
        portfolio[key] = 0.0
    for point in portfolio["performance_points"]:
        point["total_value"] = 0.0
        point["excluded_external_flow"] = 0.0
        point["adjusted_total_value"] = 0.0
    if "current_valuation_status" in portfolio:
        portfolio["current_valuation_status"] = REDACTED

    # Empty arrays avoid leaking even the number of holdings or changes.
    # Public mode renders only the approved performance sections.
    bundle["positions"] = []
    bundle["changes"] = []

    for key in (
        "equity_hhi",
        "holding_count",
        "top1_equity_weight",
        "top3_equity_weight",
    ):
        bundle["concentration"][key] = 0
    bundle["concentration"]["thesis_status_counts"] = {}
    bundle["current_evidence"] = _sanitize_evidence(
        bundle["current_evidence"]
    )
    bundle["previous_evidence"] = _sanitize_evidence(
        bundle["previous_evidence"]
    )
    bundle["source_refs"] = []
    bundle["rejected_record_samples"] = []

    history = bundle["history"]
    history["baseline_manifest_path"] = REDACTED
    history["baseline_manifest_sha256"] = ZERO_SHA
    history["baseline_ledger_path"] = REDACTED
    history["baseline_ledger_sha256"] = ZERO_SHA
    history["net_external_flow"] = 0.0
    history["rejected_record_samples"] = []
    for event in history["funding_events"]:
        event["amount"] = 0.0
        event["total_value_before"] = 0.0
        event["total_value_after"] = 0.0
        event["evidence_path"] = REDACTED
        event["evidence_sha256"] = ZERO_SHA

    for benchmark in bundle["benchmarks"]:
        benchmark["source_path"] = REDACTED
        benchmark["source_sha256"] = ZERO_SHA

    bundle["risk_free"]["source_path"] = REDACTED
    bundle["risk_free"]["source_sha256"] = ZERO_SHA

    bundle["risks"] = []
    bundle["warnings"] = ["public_pages_redacted_snapshot"]
    bundle.pop("content_sha256", None)
    bundle["content_sha256"] = hashlib.sha256(
        _canonical_json(bundle)
    ).hexdigest()
    return bundle


def _prepare_destination(destination: Path) -> None:
    if destination in {Path("/"), Path.home()}:
        raise ValueError("unsafe_destination")
    if destination.exists():
        if not (destination / MARKER).is_file():
            raise ValueError("destination_missing_public_site_marker")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    (destination / MARKER).write_text("generated\n", encoding="utf-8")


def validate_public_template(public_html: str) -> None:
    """Fail closed before writing a public site outside the approved view."""

    if any(
        public_html.count(marker) != 1
        for marker in APPROVED_PUBLIC_MARKERS
    ):
        raise ValueError("public_template_section_contract_failed")
    if 'id="method"' in public_html or 'href="#method"' in public_html:
        raise ValueError("public_template_unapproved_method_section")
    if any(phrase in public_html for phrase in FORBIDDEN_PUBLIC_COPY):
        raise ValueError("public_template_forbidden_funding_copy")


def build_site(
    *, source_root: Path, bundle_path: Path, destination: Path
) -> dict[str, Any]:
    dashboard = source_root / "portfolio_dashboard"
    if destination == source_root or source_root in destination.parents:
        raise ValueError("destination_must_not_be_inside_source_root")
    validate_public_template(
        (dashboard / "public.html").read_text(encoding="utf-8")
    )
    bundle = sanitize_bundle(
        json.loads(bundle_path.read_text(encoding="utf-8"))
    )
    _prepare_destination(destination)
    (destination / "js").mkdir()
    (destination / "private" / "generated").mkdir(parents=True)

    copies = {
        "public.html": "index.html",
        "styles.css": "styles.css",
        "app.js": "app.js",
        "js/cn_aggressive_input.js": "js/cn_aggressive_input.js",
        "js/cn_aggressive_dashboard_contract_v1.js": (
            "js/cn_aggressive_dashboard_contract_v1.js"
        ),
        "js/cn_aggressive_dashboard_analysis_v1.js": (
            "js/cn_aggressive_dashboard_analysis_v1.js"
        ),
    }
    for source_name, target_name in copies.items():
        target = destination / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dashboard / source_name, target)

    (destination / "js" / "cn_aggressive_public_mode.js").write_text(
        PUBLIC_MODE, encoding="utf-8"
    )
    (destination / ".nojekyll").write_text("", encoding="utf-8")
    (destination / "404.html").write_text(
        '<meta http-equiv="refresh" content="0; url=./">\n',
        encoding="utf-8",
    )
    (destination / "robots.txt").write_text(
        "User-agent: *\nDisallow: /\n", encoding="utf-8"
    )
    json_bytes = _canonical_json(bundle)
    generated = destination / "private" / "generated"
    (generated / "cn_aggressive_dashboard.v1.json").write_bytes(
        json_bytes + b"\n"
    )
    (generated / "cn_aggressive_dashboard.v1.js").write_bytes(
        b"window.MyQuantCNAggressiveDashboard = " + json_bytes + b";\n"
    )
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = build_site(
        source_root=args.source_root.resolve(),
        bundle_path=args.bundle.resolve(),
        destination=args.destination.resolve(),
    )
    print(
        json.dumps(
            {
                "built": True,
                "destination": str(args.destination.resolve()),
                "content_sha256": bundle["content_sha256"],
                "public_sections": len(APPROVED_PUBLIC_MARKERS),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

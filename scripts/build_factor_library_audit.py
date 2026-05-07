#!/usr/bin/env python3
"""Build an offline Phase 9 production factor library audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.factors.library import (  # noqa: E402
    FactorLibraryPolicy,
    audit_factor_library,
    build_production_library_from_artifacts,
)
from quant_investor.factors.report import (  # noqa: E402
    build_factor_governance_dashboard_payload,
    render_factor_library_audit_markdown,
)
from quant_investor.factors.store import (  # noqa: E402
    FactorCorrelationContributionStore,
    FactorGovernanceStore,
    FactorLibraryAuditStore,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline production factor library audit from local artifacts."
    )
    parser.add_argument("--root-dir", default="data/factor_library")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument(
        "--require-incremental-review",
        dest="require_incremental_review",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-require-incremental-review",
        dest="require_incremental_review",
        action="store_false",
    )
    parser.add_argument("--allow-redundant-factors", action="store_true", default=False)
    parser.add_argument("--allow-negative-contribution", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir) if args.output_dir else root_dir / "audit"
    governance_store = FactorGovernanceStore(root_dir)
    incremental_store = FactorCorrelationContributionStore(root_dir / "incremental")
    audit_store = FactorLibraryAuditStore(output_dir)
    policy = FactorLibraryPolicy(
        require_incremental_review=bool(args.require_incremental_review),
        allow_redundant_factors=bool(args.allow_redundant_factors),
        allow_negative_contribution=bool(args.allow_negative_contribution),
    )

    definitions = governance_store.read_factor_definitions()
    decisions = governance_store.read_admission_decisions()
    validation_reports = governance_store.read_validation_reports()
    redundancy_reports = incremental_store.read_redundancy_reports()
    contribution_reports = incremental_store.read_contribution_reports()

    has_any_artifact = any(
        [
            definitions,
            decisions,
            validation_reports,
            redundancy_reports,
            contribution_reports,
        ]
    )
    library = None
    if has_any_artifact:
        library = build_production_library_from_artifacts(
            definitions=definitions,
            admission_decisions=decisions,
            validation_reports=validation_reports,
            generated_at=args.generated_at,
            policy=policy,
            metadata={"source": "scripts/build_factor_library_audit.py"},
        )
        if library.entries:
            governance_store.save_production_library(library)

    report = audit_factor_library(
        library=library,
        definitions=definitions,
        admission_decisions=decisions,
        validation_reports=validation_reports,
        redundancy_reports=redundancy_reports,
        contribution_reports=contribution_reports,
        policy=policy,
        as_of=args.as_of,
        generated_at=args.generated_at,
        metadata={
            "source": "scripts/build_factor_library_audit.py",
            "root_dir": str(root_dir),
            "output_dir": str(output_dir),
        },
    )
    markdown = render_factor_library_audit_markdown(report)
    dashboard = build_factor_governance_dashboard_payload(
        library=library,
        audit_report=report,
        metadata={"source": "scripts/build_factor_library_audit.py"},
    )
    audit_store.append_audit_report(report)
    markdown_path = audit_store.save_audit_markdown(markdown)
    dashboard_path = audit_store.save_dashboard_payload(dashboard)

    print(f"verdict: {report.verdict}")
    print(f"production_factor_count: {report.production_factor_count}")
    print(f"issue_count: {report.issue_count}")
    print(f"blocker_count: {report.blocker_count}")
    print(f"warning_count: {report.warning_count}")
    if library is not None and library.entries:
        print(f"production_library: {governance_store.production_library_path}")
    else:
        print("production_library: not_written")
    print(f"audit_reports: {audit_store.audit_reports_path}")
    print(f"audit_markdown: {markdown_path}")
    print(f"dashboard_payload: {dashboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

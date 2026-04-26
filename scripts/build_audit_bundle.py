#!/usr/bin/env python3
"""Build an offline staged-upgrade audit bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_investor.observability import (  # noqa: E402
    ObservabilityStore,
    build_audit_bundle,
    discover_phase_artifacts,
    render_audit_report_markdown,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an offline staged-upgrade audit bundle.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--outcome-ledger-dir", default=None)
    parser.add_argument("--calibration-v2-dir", default=None)
    parser.add_argument("--data-quality-dir", default=None)
    parser.add_argument("--risk-tensor-dir", default=None)
    parser.add_argument("--portfolio-optimizer-dir", default=None)
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--scripts-dir", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--market", default=None)
    parser.add_argument("--universe-key", default=None)
    parser.add_argument("--universe-hash", default=None)
    parser.add_argument("--architecture-version", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifact_refs = discover_phase_artifacts(
        outcome_ledger_dir=args.outcome_ledger_dir,
        calibration_v2_dir=args.calibration_v2_dir,
        data_quality_dir=args.data_quality_dir,
        risk_tensor_dir=args.risk_tensor_dir,
        portfolio_optimizer_dir=args.portfolio_optimizer_dir,
        docs_dir=args.docs_dir,
        scripts_dir=args.scripts_dir,
    )
    bundle = build_audit_bundle(
        run_id=args.run_id,
        artifact_refs=artifact_refs,
        as_of=args.as_of,
        market=args.market,
        universe_key=args.universe_key,
        universe_hash=args.universe_hash,
        architecture_version=args.architecture_version,
    )
    store = ObservabilityStore(args.output_dir)
    bundle_path = store.save_audit_bundle(bundle)
    report_path = store.save_audit_report(render_audit_report_markdown(bundle))
    dashboard_path = store.save_dashboard_payload(bundle.dashboard_payload)
    manifest_path = store.save_run_manifest(bundle.run_manifest)

    print(f"run_id: {bundle.run_manifest.run_id}")
    print(f"overall_status: {bundle.observability_summary.overall_status}")
    print(f"total_artifacts: {bundle.observability_summary.total_artifacts}")
    print(f"total_records: {bundle.observability_summary.total_records}")
    print(f"total_warnings: {bundle.observability_summary.total_warnings}")
    print(f"audit_bundle: {bundle_path}")
    print(f"audit_report: {report_path}")
    print(f"dashboard_payload: {dashboard_path}")
    print(f"run_manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
